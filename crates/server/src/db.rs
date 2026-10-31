//! SQLite state: episodes, their per-stage runs, and settings.

use std::path::Path;

use anyhow::Result;
use serde::Serialize;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions};
use sqlx::{Row, SqlitePool};

/// Every stage the queue runs, in order. `fetch` is server-side (yt-dlp or a
/// local file); the rest are [`crate::pipeline::Stage`]s.
pub const STAGES: [&str; 7] = [
    "fetch",
    "extract",
    "asr",
    "align",
    "cues",
    "translate",
    "render",
];

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS episodes (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  title       TEXT NOT NULL,
  source_kind TEXT NOT NULL,            -- url | file
  source      TEXT NOT NULL,            -- URL or absolute path
  series      TEXT,
  season      INTEGER,
  episode     INTEGER,
  translator  TEXT NOT NULL DEFAULT 'qwen9b',
  state       TEXT NOT NULL DEFAULT 'queued', -- queued | running | done | failed | cancelled
  error       TEXT,
  created_at  TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS stage_runs (
  episode_id  INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
  stage       TEXT NOT NULL,
  status      TEXT NOT NULL DEFAULT 'pending', -- pending | running | done | failed
  attempts    INTEGER NOT NULL DEFAULT 0,
  run_after   TEXT,
  stats       TEXT,
  error       TEXT,
  started_at  TEXT,
  finished_at TEXT,
  PRIMARY KEY (episode_id, stage)
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"#;

pub async fn open(path: &Path) -> Result<SqlitePool> {
    let options = SqliteConnectOptions::new()
        .filename(path)
        .create_if_missing(true)
        .journal_mode(SqliteJournalMode::Wal)
        .foreign_keys(true)
        .busy_timeout(std::time::Duration::from_secs(5));
    let pool = SqlitePoolOptions::new()
        .max_connections(4)
        .connect_with(options)
        .await?;
    sqlx::raw_sql(SCHEMA).execute(&pool).await?;
    // Anything "running" belonged to a previous process: run it again.
    sqlx::query("UPDATE stage_runs SET status = 'pending' WHERE status = 'running'")
        .execute(&pool)
        .await?;
    sqlx::query("UPDATE episodes SET state = 'queued' WHERE state = 'running'")
        .execute(&pool)
        .await?;
    Ok(pool)
}

#[derive(Debug, Clone, Serialize)]
pub struct Episode {
    pub id: i64,
    pub title: String,
    pub source_kind: String,
    pub source: String,
    pub series: Option<String>,
    pub season: Option<i64>,
    pub episode: Option<i64>,
    pub translator: String,
    pub state: String,
    pub error: Option<String>,
    pub created_at: String,
    pub stages: Vec<StageRun>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StageRun {
    pub stage: String,
    pub status: String,
    pub attempts: i64,
    pub error: Option<String>,
    pub stats: Option<serde_json::Value>,
}

pub struct NewEpisode {
    pub title: String,
    pub source_kind: String,
    pub source: String,
    pub series: Option<String>,
    pub season: Option<u32>,
    pub episode: Option<u32>,
    pub translator: String,
}

pub async fn insert_episode(pool: &SqlitePool, e: &NewEpisode) -> Result<i64> {
    let mut tx = pool.begin().await?;
    let id: i64 = sqlx::query(
        "INSERT INTO episodes (title, source_kind, source, series, season, episode, translator)
         VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING id",
    )
    .bind(&e.title)
    .bind(&e.source_kind)
    .bind(&e.source)
    .bind(&e.series)
    .bind(e.season)
    .bind(e.episode)
    .bind(&e.translator)
    .fetch_one(&mut *tx)
    .await?
    .get(0);
    for stage in STAGES {
        sqlx::query("INSERT INTO stage_runs (episode_id, stage) VALUES (?, ?)")
            .bind(id)
            .bind(stage)
            .execute(&mut *tx)
            .await?;
    }
    tx.commit().await?;
    Ok(id)
}

pub async fn episode(pool: &SqlitePool, id: i64) -> Result<Option<Episode>> {
    let row = sqlx::query("SELECT * FROM episodes WHERE id = ?")
        .bind(id)
        .fetch_optional(pool)
        .await?;
    let Some(row) = row else { return Ok(None) };
    let mut ep = episode_from_row(&row);
    ep.stages = stage_runs(pool, id).await?;
    Ok(Some(ep))
}

pub async fn list_episodes(pool: &SqlitePool, limit: i64) -> Result<Vec<Episode>> {
    let rows = sqlx::query("SELECT * FROM episodes ORDER BY id DESC LIMIT ?")
        .bind(limit)
        .fetch_all(pool)
        .await?;
    let mut out = Vec::with_capacity(rows.len());
    for row in &rows {
        let mut ep = episode_from_row(row);
        ep.stages = stage_runs(pool, ep.id).await?;
        out.push(ep);
    }
    Ok(out)
}

fn episode_from_row(row: &sqlx::sqlite::SqliteRow) -> Episode {
    Episode {
        id: row.get("id"),
        title: row.get("title"),
        source_kind: row.get("source_kind"),
        source: row.get("source"),
        series: row.get("series"),
        season: row.get("season"),
        episode: row.get("episode"),
        translator: row.get("translator"),
        state: row.get("state"),
        error: row.get("error"),
        created_at: row.get("created_at"),
        stages: Vec::new(),
    }
}

async fn stage_runs(pool: &SqlitePool, id: i64) -> Result<Vec<StageRun>> {
    let rows = sqlx::query(
        "SELECT stage, status, attempts, error, stats FROM stage_runs WHERE episode_id = ?",
    )
    .bind(id)
    .fetch_all(pool)
    .await?;
    let mut runs: Vec<StageRun> = rows
        .iter()
        .map(|r| StageRun {
            stage: r.get("stage"),
            status: r.get("status"),
            attempts: r.get("attempts"),
            error: r.get("error"),
            stats: r
                .get::<Option<String>, _>("stats")
                .and_then(|s| serde_json::from_str(&s).ok()),
        })
        .collect();
    runs.sort_by_key(|r| STAGES.iter().position(|s| *s == r.stage));
    Ok(runs)
}

/// The next stage each runnable episode needs: the first one not done, if it
/// is pending and due. Oldest episodes first.
pub async fn runnable(pool: &SqlitePool) -> Result<Vec<(i64, String)>> {
    let rows = sqlx::query(
        "SELECT e.id, s.stage, s.status, s.run_after <= datetime('now') OR s.run_after IS NULL AS due
         FROM episodes e JOIN stage_runs s ON s.episode_id = e.id
         WHERE e.state IN ('queued', 'running')
         ORDER BY e.id",
    )
    .fetch_all(pool)
    .await?;
    let mut by_episode: std::collections::BTreeMap<i64, Vec<(String, String, bool)>> =
        Default::default();
    for r in &rows {
        by_episode.entry(r.get("id")).or_default().push((
            r.get("stage"),
            r.get("status"),
            r.get::<bool, _>("due"),
        ));
    }
    let mut out = Vec::new();
    for (id, mut stages) in by_episode {
        stages.sort_by_key(|(s, _, _)| STAGES.iter().position(|x| x == s));
        if let Some((stage, status, due)) = stages.into_iter().find(|(_, st, _)| st != "done")
            && status == "pending"
            && due
        {
            out.push((id, stage));
        }
    }
    Ok(out)
}

pub async fn start_stage(pool: &SqlitePool, id: i64, stage: &str) -> Result<()> {
    sqlx::query(
        "UPDATE stage_runs SET status = 'running', started_at = datetime('now'), error = NULL
         WHERE episode_id = ? AND stage = ?",
    )
    .bind(id)
    .bind(stage)
    .execute(pool)
    .await?;
    set_state(pool, id, "running", None).await
}

pub async fn finish_stage(
    pool: &SqlitePool,
    id: i64,
    stage: &str,
    stats: &serde_json::Value,
) -> Result<()> {
    sqlx::query(
        "UPDATE stage_runs SET status = 'done', finished_at = datetime('now'), stats = ?
         WHERE episode_id = ? AND stage = ?",
    )
    .bind(stats.to_string())
    .bind(id)
    .bind(stage)
    .execute(pool)
    .await?;
    if stage == "render" {
        set_state(pool, id, "done", None).await?;
    }
    Ok(())
}

/// Records a failure. `retry_in_s = Some(n)` puts the stage back in the queue
/// after `n` seconds; `count = false` doesn't use up an attempt (GPU contention).
pub async fn fail_stage(
    pool: &SqlitePool,
    id: i64,
    stage: &str,
    error: &str,
    retry_in_s: Option<u32>,
    count: bool,
) -> Result<()> {
    let status = if retry_in_s.is_some() {
        "pending"
    } else {
        "failed"
    };
    sqlx::query(
        "UPDATE stage_runs SET status = ?, error = ?, attempts = attempts + ?, finished_at = datetime('now'),
           run_after = datetime('now', ?)
         WHERE episode_id = ? AND stage = ?",
    )
    .bind(status)
    .bind(error)
    .bind(i64::from(count))
    .bind(format!("+{} seconds", retry_in_s.unwrap_or(0)))
    .bind(id)
    .bind(stage)
    .execute(pool)
    .await?;
    if retry_in_s.is_none() {
        set_state(pool, id, "failed", Some(error)).await?;
    }
    Ok(())
}

pub async fn attempts(pool: &SqlitePool, id: i64, stage: &str) -> Result<i64> {
    Ok(
        sqlx::query("SELECT attempts FROM stage_runs WHERE episode_id = ? AND stage = ?")
            .bind(id)
            .bind(stage)
            .fetch_one(pool)
            .await?
            .get(0),
    )
}

/// Requeues a failed or cancelled episode from its first unfinished stage.
pub async fn retry(pool: &SqlitePool, id: i64) -> Result<()> {
    sqlx::query(
        "UPDATE stage_runs SET status = 'pending', attempts = 0, run_after = NULL
         WHERE episode_id = ? AND status IN ('failed', 'running')",
    )
    .bind(id)
    .execute(pool)
    .await?;
    set_state(pool, id, "queued", None).await
}

pub async fn set_state(pool: &SqlitePool, id: i64, state: &str, error: Option<&str>) -> Result<()> {
    sqlx::query(
        "UPDATE episodes SET state = ?, error = ?, updated_at = datetime('now') WHERE id = ?",
    )
    .bind(state)
    .bind(error)
    .bind(id)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn setting(pool: &SqlitePool, key: &str) -> Result<Option<String>> {
    Ok(sqlx::query("SELECT value FROM settings WHERE key = ?")
        .bind(key)
        .fetch_optional(pool)
        .await?
        .map(|r| r.get(0)))
}

pub async fn set_setting(pool: &SqlitePool, key: &str, value: &str) -> Result<()> {
    sqlx::query("INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value")
        .bind(key)
        .bind(value)
        .execute(pool)
        .await?;
    Ok(())
}
