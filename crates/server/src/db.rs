//! SQLite state: episodes, their per-stage runs, and settings.

use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};
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
-- Editable cues, imported after render. `ar_asr` and `en_mt` are what the
-- machines produced (never changed); `ar_text` / `en_text` are current.
CREATE TABLE IF NOT EXISTS cues (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  episode_id INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
  ord        INTEGER NOT NULL,
  start_ms   INTEGER NOT NULL,
  end_ms     INTEGER NOT NULL,
  ar_asr     TEXT NOT NULL,
  ar_text    TEXT NOT NULL,
  en_mt      TEXT NOT NULL,
  en_text    TEXT NOT NULL,
  uncertain  INTEGER NOT NULL DEFAULT 0,
  status     TEXT NOT NULL DEFAULT 'auto', -- auto | edited | verified | deleted
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE (episode_id, ord)
);
-- Every human change, for training data later.
CREATE TABLE IF NOT EXISTS cue_edits (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  cue_id     INTEGER NOT NULL REFERENCES cues(id) ON DELETE CASCADE,
  field      TEXT NOT NULL,
  old_value  TEXT,
  new_value  TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
-- Per-series terms the translator must use.
CREATE TABLE IF NOT EXISTS glossary (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  series     TEXT NOT NULL,
  ar         TEXT NOT NULL,
  en         TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE (series, ar)
);
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

#[derive(Debug, Clone, Serialize, sqlx::FromRow)]
pub struct CueRow {
    pub id: i64,
    pub ord: i64,
    pub start_ms: i64,
    pub end_ms: i64,
    pub ar_asr: String,
    pub ar_text: String,
    pub en_mt: String,
    pub en_text: String,
    pub uncertain: bool,
    pub status: String,
}

pub async fn cues(pool: &SqlitePool, episode_id: i64) -> Result<Vec<CueRow>> {
    Ok(sqlx::query_as(
        "SELECT id, ord, start_ms, end_ms, ar_asr, ar_text, en_mt, en_text, uncertain, status
         FROM cues WHERE episode_id = ? ORDER BY ord",
    )
    .bind(episode_id)
    .fetch_all(pool)
    .await?)
}

pub async fn cue(pool: &SqlitePool, id: i64) -> Result<Option<(i64, CueRow)>> {
    let row = sqlx::query("SELECT episode_id FROM cues WHERE id = ?")
        .bind(id)
        .fetch_optional(pool)
        .await?;
    let Some(row) = row else { return Ok(None) };
    let episode_id: i64 = row.get(0);
    let cue = sqlx::query_as(
        "SELECT id, ord, start_ms, end_ms, ar_asr, ar_text, en_mt, en_text, uncertain, status FROM cues WHERE id = ?",
    )
    .bind(id)
    .fetch_one(pool)
    .await?;
    Ok(Some((episode_id, cue)))
}

pub struct ImportedCue {
    pub start_ms: i64,
    pub end_ms: i64,
    pub ar_asr: String,
    pub ar_text: String,
    pub en: String,
    pub uncertain: bool,
}

/// Imports a fresh render. New episodes get every cue; on a retranslate,
/// cues the user edited or verified are kept and only `auto` ones change.
pub async fn import_cues(pool: &SqlitePool, episode_id: i64, cues: &[ImportedCue]) -> Result<()> {
    let mut tx = pool.begin().await?;
    for (ord, c) in cues.iter().enumerate() {
        sqlx::query(
            "INSERT INTO cues (episode_id, ord, start_ms, end_ms, ar_asr, ar_text, en_mt, en_text, uncertain)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT (episode_id, ord) DO UPDATE SET
               ar_text = CASE WHEN status = 'auto' THEN excluded.ar_text ELSE ar_text END,
               en_mt = excluded.en_mt,
               en_text = CASE WHEN status = 'auto' THEN excluded.en_text ELSE en_text END,
               uncertain = CASE WHEN status = 'auto' THEN excluded.uncertain ELSE uncertain END",
        )
        .bind(episode_id)
        .bind(ord as i64)
        .bind(c.start_ms)
        .bind(c.end_ms)
        .bind(&c.ar_asr)
        .bind(&c.ar_text)
        .bind(&c.en)
        .bind(&c.en)
        .bind(c.uncertain)
        .execute(&mut *tx)
        .await?;
    }
    tx.commit().await?;
    Ok(())
}

#[derive(Debug, Default, Deserialize)]
pub struct CuePatch {
    pub ar_text: Option<String>,
    pub en_text: Option<String>,
    pub start_ms: Option<i64>,
    pub end_ms: Option<i64>,
    /// "verified", "deleted" or "auto"; edits set "edited" by themselves.
    pub status: Option<String>,
}

/// Applies a user edit and logs each changed field.
pub async fn patch_cue(
    pool: &SqlitePool,
    id: i64,
    patch: &CuePatch,
) -> Result<Option<(i64, CueRow)>> {
    let Some((_, old)) = cue(pool, id).await? else {
        return Ok(None);
    };
    let mut tx = pool.begin().await?;
    let mut changed = false;
    let text_fields = [
        ("ar_text", &patch.ar_text, &old.ar_text),
        ("en_text", &patch.en_text, &old.en_text),
    ];
    for (field, new, old_value) in text_fields {
        if let Some(new) = new.as_ref().filter(|n| *n != old_value) {
            sqlx::query(update_sql(field))
                .bind(new)
                .bind(id)
                .execute(&mut *tx)
                .await?;
            log_edit(&mut tx, id, field, old_value, new).await?;
            changed = true;
        }
    }
    let time_fields = [
        ("start_ms", patch.start_ms, old.start_ms),
        ("end_ms", patch.end_ms, old.end_ms),
    ];
    for (field, new, old_value) in time_fields {
        if let Some(new) = new.filter(|n| *n != old_value) {
            sqlx::query(update_sql(field))
                .bind(new)
                .bind(id)
                .execute(&mut *tx)
                .await?;
            log_edit(&mut tx, id, field, &old_value.to_string(), &new.to_string()).await?;
            changed = true;
        }
    }
    let status = match patch.status.as_deref() {
        Some(s @ ("verified" | "deleted" | "auto")) => Some(s),
        Some(other) => anyhow::bail!("unknown cue status {other:?}"),
        None if changed && old.status == "auto" => Some("edited"),
        None => None,
    };
    if let Some(status) = status.filter(|s| *s != old.status) {
        sqlx::query("UPDATE cues SET status = ?, uncertain = 0 WHERE id = ?")
            .bind(status)
            .bind(id)
            .execute(&mut *tx)
            .await?;
        log_edit(&mut tx, id, "status", &old.status, status).await?;
    }
    sqlx::query("UPDATE cues SET updated_at = datetime('now') WHERE id = ?")
        .bind(id)
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;
    cue(pool, id).await
}

fn update_sql(field: &str) -> &'static str {
    match field {
        "ar_text" => "UPDATE cues SET ar_text = ? WHERE id = ?",
        "en_text" => "UPDATE cues SET en_text = ? WHERE id = ?",
        "start_ms" => "UPDATE cues SET start_ms = ? WHERE id = ?",
        "end_ms" => "UPDATE cues SET end_ms = ? WHERE id = ?",
        _ => unreachable!("not an editable cue field: {field}"),
    }
}

async fn log_edit(
    tx: &mut sqlx::SqliteConnection,
    cue_id: i64,
    field: &str,
    old: &str,
    new: &str,
) -> Result<()> {
    sqlx::query("INSERT INTO cue_edits (cue_id, field, old_value, new_value) VALUES (?, ?, ?, ?)")
        .bind(cue_id)
        .bind(field)
        .bind(old)
        .bind(new)
        .execute(tx)
        .await?;
    Ok(())
}

#[derive(Debug, Clone, Serialize, sqlx::FromRow)]
pub struct GlossaryEntry {
    pub id: i64,
    pub series: String,
    pub ar: String,
    pub en: String,
}

pub async fn glossary(pool: &SqlitePool, series: &str) -> Result<Vec<GlossaryEntry>> {
    Ok(
        sqlx::query_as("SELECT id, series, ar, en FROM glossary WHERE series = ? ORDER BY ar")
            .bind(series)
            .fetch_all(pool)
            .await?,
    )
}

pub async fn add_glossary(pool: &SqlitePool, series: &str, ar: &str, en: &str) -> Result<()> {
    sqlx::query(
        "INSERT INTO glossary (series, ar, en) VALUES (?, ?, ?)
         ON CONFLICT (series, ar) DO UPDATE SET en = excluded.en",
    )
    .bind(series)
    .bind(ar.trim())
    .bind(en.trim())
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn delete_glossary(pool: &SqlitePool, id: i64) -> Result<()> {
    sqlx::query("DELETE FROM glossary WHERE id = ?")
        .bind(id)
        .execute(pool)
        .await?;
    Ok(())
}

/// Lines the user fixed or verified in this series, newest first: the
/// translator's examples of the wanted style.
pub async fn approved_examples(
    pool: &SqlitePool,
    series: &str,
    limit: i64,
) -> Result<Vec<(String, String)>> {
    let rows = sqlx::query(
        "SELECT c.ar_text, c.en_text FROM cues c JOIN episodes e ON e.id = c.episode_id
         WHERE e.series = ? AND c.status IN ('edited', 'verified') AND c.en_text <> ''
           AND (c.status = 'verified' OR c.en_text <> c.en_mt)
         ORDER BY (c.en_text <> c.en_mt) DESC, c.updated_at DESC LIMIT ?",
    )
    .bind(series)
    .bind(limit)
    .fetch_all(pool)
    .await?;
    Ok(rows.iter().map(|r| (r.get(0), r.get(1))).collect())
}

/// Puts the translate and render stages back in the queue.
pub async fn requeue_translation(pool: &SqlitePool, id: i64) -> Result<()> {
    sqlx::query(
        "UPDATE stage_runs SET status = 'pending', attempts = 0, run_after = NULL, error = NULL
         WHERE episode_id = ? AND stage IN ('translate', 'render')",
    )
    .bind(id)
    .execute(pool)
    .await?;
    set_state(pool, id, "queued", None).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn imported(en: &str) -> ImportedCue {
        ImportedCue {
            start_ms: 0,
            end_ms: 1000,
            ar_asr: "شو".into(),
            ar_text: "شو".into(),
            en: en.into(),
            uncertain: false,
        }
    }

    #[tokio::test]
    async fn retranslate_keeps_user_edits_and_examples_come_from_them() {
        let dir = std::env::temp_dir().join(format!("shu2al-db-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pool = open(&dir.join("t.db")).await.unwrap();
        let id = insert_episode(
            &pool,
            &NewEpisode {
                title: "t".into(),
                source_kind: "file".into(),
                source: "/x".into(),
                series: Some("الندم".into()),
                season: None,
                episode: Some(1),
                translator: "qwen9b".into(),
            },
        )
        .await
        .unwrap();

        import_cues(&pool, id, &[imported("a1"), imported("b1"), imported("c1")])
            .await
            .unwrap();
        let cues = cues(&pool, id).await.unwrap();
        let patch = CuePatch {
            en_text: Some("What?".into()),
            ..Default::default()
        };
        let (_, edited) = patch_cue(&pool, cues[1].id, &patch).await.unwrap().unwrap();
        assert_eq!(edited.status, "edited");

        // A retranslate brings new machine output for every cue.
        import_cues(&pool, id, &[imported("a2"), imported("b2"), imported("c2")])
            .await
            .unwrap();
        let after = super::cues(&pool, id).await.unwrap();
        let en: Vec<&str> = after.iter().map(|c| c.en_text.as_str()).collect();
        assert_eq!(
            en,
            ["a2", "What?", "c2"],
            "edited cue kept, auto cues replaced"
        );
        assert_eq!(
            after[1].en_mt, "b2",
            "machine output still tracked for the edited cue"
        );

        let examples = approved_examples(&pool, "الندم", 10).await.unwrap();
        assert_eq!(examples, vec![("شو".to_owned(), "What?".to_owned())]);

        let logged: i64 = sqlx::query("SELECT COUNT(*) FROM cue_edits")
            .fetch_one(&pool)
            .await
            .unwrap()
            .get(0);
        assert_eq!(logged, 2, "en_text change + status change");
        std::fs::remove_dir_all(&dir).ok();
    }
}
