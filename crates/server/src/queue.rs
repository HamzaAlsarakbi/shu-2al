//! The job queue: runs each episode's stages in order, at most one GPU stage
//! at a time, and holds GPU work while something else (a game) uses the GPU.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::{Value, json};
use sqlx::SqlitePool;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::{Notify, broadcast};

use crate::db::{self, Episode};
use crate::pipeline::{self, Llm, Stage};
use crate::worker::{Launcher, StageError};

pub struct Config {
    /// Root for episode work directories (`<data>/episodes/<id>`).
    pub data: PathBuf,
    pub models: PathBuf,
    pub worker_image: String,
    pub llm_image: String,
    pub llm_port: u16,
    /// External VRAM use (MiB) above which GPU stages wait.
    pub gpu_busy_mb: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Live {
    pub stage: String,
    pub pct: f64,
    pub msg: String,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct GpuStatus {
    pub paused: bool,
    /// Why GPU stages are on hold right now, if they are.
    pub holding: Option<String>,
    pub vram_used_mb: Option<u64>,
}

pub struct Queue {
    pub pool: SqlitePool,
    pub cfg: Config,
    pub events: broadcast::Sender<String>,
    pub wake: Notify,
    live: Mutex<HashMap<i64, Live>>,
    running: Mutex<HashSet<(i64, &'static str)>>,
    gpu: Mutex<GpuStatus>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Lane {
    Gpu,
    Io,
    Cpu,
}

fn lane(stage: &str) -> Lane {
    match stage {
        "asr" | "align" | "translate" => Lane::Gpu,
        "fetch" | "extract" => Lane::Io,
        _ => Lane::Cpu,
    }
}

fn capacity(lane: Lane) -> usize {
    match lane {
        Lane::Gpu => 1,
        Lane::Io => 2,
        Lane::Cpu => 4,
    }
}

impl Queue {
    pub async fn new(pool: SqlitePool, cfg: Config) -> Result<Arc<Self>> {
        let paused = db::setting(&pool, "gpu_paused").await?.as_deref() == Some("true");
        Ok(Arc::new(Self {
            pool,
            cfg,
            events: broadcast::channel(256).0,
            wake: Notify::new(),
            live: Mutex::default(),
            running: Mutex::default(),
            gpu: Mutex::new(GpuStatus {
                paused,
                ..Default::default()
            }),
        }))
    }

    pub fn live(&self) -> HashMap<i64, Live> {
        self.live.lock().unwrap().clone()
    }

    pub fn gpu_status(&self) -> GpuStatus {
        self.gpu.lock().unwrap().clone()
    }

    pub async fn set_paused(&self, paused: bool) -> Result<()> {
        db::set_setting(
            &self.pool,
            "gpu_paused",
            if paused { "true" } else { "false" },
        )
        .await?;
        self.gpu.lock().unwrap().paused = paused;
        self.broadcast_gpu();
        self.wake.notify_one();
        Ok(())
    }

    /// Scheduler loop: wakes on new work, finished stages, or every 5 s.
    pub async fn run(self: Arc<Self>) {
        loop {
            if let Err(e) = self.clone().tick().await {
                eprintln!("queue: {e:#}");
            }
            let _ = tokio::time::timeout(Duration::from_secs(5), self.wake.notified()).await;
        }
    }

    async fn tick(self: Arc<Self>) -> Result<()> {
        self.refresh_gpu().await;
        for (id, stage) in db::runnable(&self.pool).await? {
            let Some(stage) = db::STAGES.iter().copied().find(|s| *s == stage) else {
                continue;
            };
            let lane = lane(stage);
            {
                let running = self.running.lock().unwrap();
                if running.iter().any(|(e, _)| *e == id) {
                    continue;
                }
                if running
                    .iter()
                    .filter(|(_, s)| self::lane(s) == lane)
                    .count()
                    >= capacity(lane)
                {
                    continue;
                }
            }
            if lane == Lane::Gpu && self.gpu_status().holding.is_some() {
                continue;
            }
            self.running.lock().unwrap().insert((id, stage));
            let queue = self.clone();
            tokio::spawn(async move {
                queue.clone().run_one(id, stage).await;
                queue.running.lock().unwrap().remove(&(id, stage));
                queue.live.lock().unwrap().remove(&id);
                queue.wake.notify_one();
            });
        }
        Ok(())
    }

    /// Updates the GPU hold: paused by the user, or VRAM in use by something
    /// other than our own GPU stage (e.g. a game).
    async fn refresh_gpu(&self) {
        let ours = self
            .running
            .lock()
            .unwrap()
            .iter()
            .any(|(_, s)| lane(s) == Lane::Gpu);
        let used = vram_used_mb().await;
        let mut gpu = self.gpu.lock().unwrap();
        let before = gpu.clone();
        gpu.vram_used_mb = used;
        gpu.holding = if gpu.paused {
            Some("paused".into())
        } else if !ours && used.is_some_and(|mb| mb > self.cfg.gpu_busy_mb) {
            Some(format!(
                "GPU in use by another app ({} MiB)",
                used.unwrap_or(0)
            ))
        } else {
            None
        };
        let changed = *gpu != before;
        drop(gpu);
        if changed {
            self.broadcast_gpu();
        }
    }

    fn broadcast_gpu(&self) {
        let _ = self
            .events
            .send(json!({ "type": "gpu", "gpu": self.gpu_status() }).to_string());
    }

    pub async fn broadcast_episode(&self, id: i64) {
        if let Ok(Some(ep)) = db::episode(&self.pool, id).await {
            let live = self.live.lock().unwrap().get(&id).cloned();
            let _ = self
                .events
                .send(json!({ "type": "episode", "episode": ep, "live": live }).to_string());
        }
    }

    async fn run_one(self: Arc<Self>, id: i64, stage: &'static str) {
        if let Err(e) = db::start_stage(&self.pool, id, stage).await {
            eprintln!("queue: starting {stage} for #{id}: {e:#}");
            return;
        }
        self.live.lock().unwrap().insert(
            id,
            Live {
                stage: stage.into(),
                pct: 0.0,
                msg: String::new(),
            },
        );
        self.broadcast_episode(id).await;

        let outcome = match db::episode(&self.pool, id).await {
            Ok(Some(ep)) => self.execute(&ep, stage).await,
            Ok(None) => Err(anyhow::anyhow!("episode #{id} disappeared")),
            Err(e) => Err(e),
        };
        let result = match outcome {
            Ok(stats) => db::finish_stage(&self.pool, id, stage, &stats).await,
            Err(e) => {
                let message = format!("{e:#}");
                eprintln!("queue: #{id} {stage}: {message}");
                if e.downcast_ref::<StageError>()
                    .is_some_and(|s| s.code == "gpu_contention")
                {
                    // Something grabbed the GPU mid-run; wait for it, don't burn an attempt.
                    db::fail_stage(&self.pool, id, stage, &message, Some(120), false).await
                } else {
                    let attempts = db::attempts(&self.pool, id, stage).await.unwrap_or(0) + 1;
                    let retry = (attempts < max_attempts(stage)).then_some(30);
                    db::fail_stage(&self.pool, id, stage, &message, retry, true).await
                }
            }
        };
        if let Err(e) = result {
            eprintln!("queue: recording {stage} for #{id}: {e:#}");
        }
        self.live.lock().unwrap().remove(&id);
        self.broadcast_episode(id).await;
    }

    async fn execute(self: &Arc<Self>, ep: &Episode, stage: &'static str) -> Result<Value> {
        let work = self.cfg.data.join("episodes").join(ep.id.to_string());
        tokio::fs::create_dir_all(&work).await?;
        let queue = self.clone();
        let id = ep.id;
        let last = Mutex::new(Instant::now() - Duration::from_secs(10));
        let progress = move |pct: f64, msg: &str| {
            queue.live.lock().unwrap().insert(
                id,
                Live {
                    stage: stage.into(),
                    pct,
                    msg: msg.to_owned(),
                },
            );
            let mut last = last.lock().unwrap();
            if last.elapsed() >= Duration::from_millis(700) {
                *last = Instant::now();
                let live = queue.live.lock().unwrap().get(&id).cloned();
                let _ = queue
                    .events
                    .send(json!({ "type": "progress", "id": id, "live": live }).to_string());
            }
        };

        if stage == "fetch" {
            return fetch(ep, &work, &progress).await;
        }
        let stage = Stage::from_name(stage).context("unknown stage")?;
        // A brief for the series, if one exists at <data>/briefs/<series>.md.
        let brief = match &ep.series {
            Some(series) => {
                tokio::fs::read_to_string(self.cfg.data.join("briefs").join(format!("{series}.md")))
                    .await
                    .ok()
            }
            None => None,
        };
        let opts = pipeline::Options {
            brief,
            media: media_path(ep, &work)?,
            name: file_name(ep),
            launcher: Launcher::Docker {
                image: self.cfg.worker_image.clone(),
                data: work.clone(),
                models: self.cfg.models.clone(),
            },
            work,
            models: self.cfg.models.clone(),
            llm: match ep.translator.as_str() {
                "moe" => Llm::Docker {
                    image: self.cfg.llm_image.clone(),
                    gguf: "llm/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf".into(),
                    port: self.cfg.llm_port,
                    ctx: 16384,
                    extra: vec!["--cpu-moe".into()],
                },
                _ => Llm::Docker {
                    image: self.cfg.llm_image.clone(),
                    gguf: "llm/Qwen3.5-9B-Q4_K_M.gguf".into(),
                    port: self.cfg.llm_port,
                    ctx: 16384,
                    extra: vec![],
                },
            },
        };
        pipeline::run_stage(stage, &opts, &progress).await
    }

    /// Path of the rendered subtitle file for `lang` ("ar" | "en"), if any.
    pub fn subtitle_path(&self, ep: &Episode, lang: &str) -> Option<PathBuf> {
        let work = self.cfg.data.join("episodes").join(ep.id.to_string());
        let paths = pipeline::Paths::new(&work, &file_name(ep));
        match lang {
            "ar" => Some(paths.out_ar),
            "en" => Some(paths.out_en),
            _ => None,
        }
    }
}

fn max_attempts(stage: &str) -> i64 {
    if stage == "fetch" { 4 } else { 2 }
}

/// "Series S01E05" when we know it, else the video title; safe as a file name.
pub fn file_name(ep: &Episode) -> String {
    let base = match (&ep.series, ep.season, ep.episode) {
        (Some(s), Some(season), Some(n)) => format!("{s} S{season:02}E{n:02}"),
        (Some(s), None, Some(n)) => format!("{s} E{n:02}"),
        _ => ep.title.clone(),
    };
    base.chars()
        .map(|c| {
            if matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') {
                ' '
            } else {
                c
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn media_path(ep: &Episode, work: &Path) -> Result<PathBuf> {
    if ep.source_kind == "file" {
        return Ok(PathBuf::from(&ep.source));
    }
    std::fs::read_dir(work)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| {
            p.file_stem().is_some_and(|s| s == "source")
                && p.extension().is_some_and(|x| x != "part" && x != "ytdl")
        })
        .context("no downloaded source in the work directory; redo fetch")
}

/// Downloads a URL with yt-dlp (480p video for the editor's proxy, best audio),
/// or checks that a local file exists.
async fn fetch(ep: &Episode, work: &Path, progress: &(dyn Fn(f64, &str) + Sync)) -> Result<Value> {
    if ep.source_kind == "file" {
        if !Path::new(&ep.source).is_file() {
            bail!("{} does not exist", ep.source);
        }
        return Ok(json!({ "path": ep.source }));
    }
    if let Ok(existing) = media_path(ep, work) {
        return Ok(json!({ "path": existing, "cached": true }));
    }
    let mut child = tokio::process::Command::new("yt-dlp")
        .args([
            "--newline",
            "--no-warnings",
            "--no-playlist",
            "-f",
            "bv*[height<=480]+ba/b[height<=480]/b",
        ])
        .args(["--merge-output-format", "mkv", "-o"])
        .arg(work.join("source.%(ext)s"))
        .arg(&ep.source)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .context("starting yt-dlp")?;
    let mut lines = BufReader::new(child.stdout.take().expect("piped")).lines();
    while let Some(line) = lines.next_line().await? {
        // "[download]  42.3% of  120.00MiB at  5.21MiB/s ETA 00:15"
        if let Some(rest) = line.strip_prefix("[download]")
            && let Some(pct) = rest
                .split_whitespace()
                .next()
                .and_then(|t| t.strip_suffix('%'))
            && let Ok(pct) = pct.parse::<f64>()
        {
            progress(pct, rest.trim());
        }
    }
    let out = child.wait_with_output().await?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let reason = err
            .lines()
            .rev()
            .find(|l| l.contains("ERROR"))
            .unwrap_or("yt-dlp failed");
        if reason.contains("DRM") {
            bail!("DRM-protected content is not supported");
        }
        bail!("{reason}");
    }
    let path = media_path(ep, work)?;
    Ok(json!({ "path": path }))
}

/// Total VRAM in use per nvidia-smi, or `None` when there's no NVIDIA GPU.
async fn vram_used_mb() -> Option<u64> {
    let out = tokio::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .await
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()
}
