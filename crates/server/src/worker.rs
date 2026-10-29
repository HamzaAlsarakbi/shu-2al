//! Runs one Python worker stage and reads its JSON Lines reply.
//!
//! Protocol v1 (see `worker/shu2al_worker/protocol.py`): we write a request
//! envelope to `request.json`, start one process per stage, and read
//! `progress` events followed by exactly one `result` or `error` from stdout.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

/// How to start the worker process.
#[derive(Debug, Clone)]
pub enum Launcher {
    /// `docker run` the worker image with the GPU, mounting `data` at the same
    /// path inside the container so request paths work on both sides.
    Docker {
        image: String,
        data: PathBuf,
        models: PathBuf,
    },
    /// `python -m shu2al_worker` from a local venv (dev / mock).
    Local { python: PathBuf },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Event {
    Progress {
        pct: f64,
        #[serde(default)]
        msg: String,
    },
    Result(Value),
    Error {
        code: String,
        message: String,
    },
}

/// A stage that reported an error. `code` is machine-readable, e.g.
/// `gpu_contention` (requeue later) or `bad_input`.
#[derive(Debug, thiserror::Error)]
#[error("worker stage failed ({code}): {message}")]
pub struct StageError {
    pub code: String,
    pub message: String,
}

impl Launcher {
    /// Runs `stage` and returns its result object (`outputs`, `stats`).
    pub async fn run(
        &self,
        stage: &str,
        inputs: Value,
        output_dir: &Path,
        params: Value,
        on_progress: impl Fn(f64, &str),
    ) -> Result<Value> {
        tokio::fs::create_dir_all(output_dir).await?;
        let request_path = output_dir.join(format!("{stage}.request.json"));
        let request = json!({
            "protocol": 1,
            "stage": stage,
            "run_id": format!("{stage}-{}", std::process::id()),
            "inputs": inputs,
            "output_dir": output_dir,
            "params": params,
        });
        tokio::fs::write(&request_path, serde_json::to_vec_pretty(&request)?).await?;

        let mut cmd = match self {
            Self::Docker {
                image,
                data,
                models,
            } => {
                let mut cmd = Command::new("docker");
                cmd.args(["run", "--rm", "--gpus", "all", "-e", "HF_TOKEN", "-v"])
                    .arg(format!("{0}:{0}", data.display()))
                    .arg("-v")
                    .arg(format!("{}:/models", models.display()))
                    .arg(image)
                    .arg(&request_path);
                cmd
            }
            Self::Local { python } => {
                let mut cmd = Command::new(python);
                cmd.args(["-m", "shu2al_worker"]).arg(&request_path);
                cmd
            }
        };
        let mut child = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .context("starting the worker")?;

        // Keep the tail of stderr for error reports; the worker logs there.
        let stderr = child.stderr.take().expect("piped");
        let stderr_tail = tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            let mut tail = std::collections::VecDeque::with_capacity(20);
            while let Ok(Some(line)) = lines.next_line().await {
                if tail.len() == 20 {
                    tail.pop_front();
                }
                tail.push_back(line);
            }
            tail.into_iter().collect::<Vec<_>>().join("\n")
        });

        let mut outcome: Option<Result<Value>> = None;
        let mut lines = BufReader::new(child.stdout.take().expect("piped")).lines();
        while let Some(line) = lines.next_line().await? {
            match serde_json::from_str::<Event>(&line) {
                Ok(Event::Progress { pct, msg }) => on_progress(pct, &msg),
                Ok(Event::Result(value)) => outcome = Some(Ok(value)),
                Ok(Event::Error { code, message }) => {
                    outcome = Some(Err(StageError { code, message }.into()))
                }
                Err(_) => {} // stray print from a library; ignore
            }
        }
        let status = child.wait().await?;
        let tail = stderr_tail.await.unwrap_or_default();
        match outcome {
            Some(result) => result,
            None => bail!("worker exited ({status}) without a result; stderr:\n{tail}"),
        }
    }
}
