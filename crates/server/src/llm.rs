//! Local LLM via llama-server's OpenAI-compatible API.
//!
//! A [`Lease`] starts llama-server in Docker on the GPU for the duration of the
//! translate stage and always removes the container afterwards, so the GPU is
//! free again for ASR jobs (or games) as soon as translation ends.

use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

const CONTAINER: &str = "shu2al-llm";

pub struct Lease {
    pub base_url: String,
    container: Option<String>,
}

impl Lease {
    /// Uses an already-running OpenAI-compatible server.
    pub fn external(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_owned(),
            container: None,
        }
    }

    /// Starts llama-server with `gguf` (a path relative to `models`) and waits
    /// until it reports healthy.
    pub async fn start(image: &str, models: &Path, gguf: &str, port: u16) -> Result<Self> {
        let _ = docker(&["rm", "-f", CONTAINER]).await;
        docker(&[
            "run",
            "-d",
            "--rm",
            "--name",
            CONTAINER,
            "--gpus",
            "all",
            "-p",
            &format!("127.0.0.1:{port}:8080"),
            "-v",
            &format!("{}:/models:ro", models.display()),
            image,
            "-m",
            &format!("/models/{gguf}"),
            "-ngl",
            "99",
            "-c",
            "16384",
            "-fa",
            "on",
            "--jinja",
        ])
        .await
        .context("starting llama-server")?;
        let lease = Self {
            base_url: format!("http://127.0.0.1:{port}"),
            container: Some(CONTAINER.to_owned()),
        };
        let health = format!("{}/health", lease.base_url);
        for _ in 0..120 {
            if reqwest::get(&health)
                .await
                .is_ok_and(|r| r.status().is_success())
            {
                return Ok(lease);
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        lease.stop().await;
        bail!("llama-server did not become healthy within 2 minutes")
    }

    pub async fn stop(&self) {
        if let Some(name) = &self.container {
            let _ = docker(&["rm", "-f", name]).await;
        }
    }

    /// One chat completion whose content must match `schema`. Thinking is off.
    pub async fn chat_json(
        &self,
        http: &reqwest::Client,
        system: &str,
        user: &str,
        schema: &Value,
        max_tokens: u32,
    ) -> Result<(Value, Usage)> {
        let body = json!({
            "model": "local",
            "temperature": 0.3,
            "max_tokens": max_tokens,
            // Small models loop inside constrained JSON strings; DRY penalizes
            // repeating sequences without hurting normal text.
            "dry_multiplier": 0.8,
            "dry_base": 1.75,
            "dry_allowed_length": 3,
            "chat_template_kwargs": { "enable_thinking": false },
            "response_format": {
                "type": "json_schema",
                "json_schema": { "name": "subtitles", "strict": true, "schema": schema },
            },
            "messages": [
                { "role": "system", "content": system },
                { "role": "user", "content": user },
            ],
        });
        let resp: Value = http
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let content = resp["choices"][0]["message"]["content"]
            .as_str()
            .context("no message content in completion")?;
        let usage = Usage {
            prompt_tokens: resp["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
            completion_tokens: resp["usage"]["completion_tokens"].as_u64().unwrap_or(0),
        };
        Ok((
            serde_json::from_str(content).context("completion is not valid JSON")?,
            usage,
        ))
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, rhs: Self) {
        self.prompt_tokens += rhs.prompt_tokens;
        self.completion_tokens += rhs.completion_tokens;
    }
}

async fn docker(args: &[&str]) -> Result<()> {
    let out = tokio::process::Command::new("docker")
        .args(args)
        .output()
        .await?;
    if !out.status.success() {
        bail!(
            "docker {}: {}",
            args.first().unwrap_or(&""),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(())
}
