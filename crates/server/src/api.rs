//! HTTP API for the dashboard, plus the built web app as static files.

use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_stream::StreamExt as _;
use tokio_stream::wrappers::BroadcastStream;
use tower_http::services::{ServeDir, ServeFile};

use crate::db::{self, NewEpisode};
use crate::queue::Queue;
use crate::titles;

type AppState = Arc<Queue>;

pub fn router(queue: AppState, web: PathBuf) -> Router {
    let spa = ServeDir::new(&web).fallback(ServeFile::new(web.join("index.html")));
    Router::new()
        .route("/api/episodes", get(list).post(create))
        .route("/api/files", post(create_from_file))
        .route("/api/episodes/{id}/retry", post(retry))
        .route("/api/episodes/{id}/cancel", post(cancel))
        .route("/api/episodes/{id}/subs/{lang}", get(subtitles))
        .route("/api/preview", post(preview))
        .route("/api/gpu", get(gpu).post(set_gpu))
        .route("/api/events", get(events))
        .fallback_service(spa)
        .with_state(queue)
}

struct ApiError(anyhow::Error);

impl<E: Into<anyhow::Error>> From<E> for ApiError {
    fn from(e: E) -> Self {
        Self(e.into())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": format!("{:#}", self.0) })),
        )
            .into_response()
    }
}

type ApiResult<T> = Result<T, ApiError>;

async fn list(State(q): State<AppState>) -> ApiResult<Json<Value>> {
    let episodes = db::list_episodes(&q.pool, 200).await?;
    Ok(Json(
        json!({ "episodes": episodes, "live": q.live(), "gpu": q.gpu_status() }),
    ))
}

#[derive(Deserialize)]
struct PreviewRequest {
    /// A video/playlist URL, or words to search YouTube for.
    input: String,
}

/// Lists what a URL, playlist or search would queue, with parsed episode numbers.
async fn preview(Json(req): Json<PreviewRequest>) -> ApiResult<Json<Value>> {
    let input = req.input.trim();
    let target = if input.starts_with("http://") || input.starts_with("https://") {
        input.to_owned()
    } else {
        // Strip bidi marks that copy-paste from web pages often carries.
        let query: String = input.chars().filter(|c| !matches!(c, '\u{200E}' | '\u{200F}' | '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}')).collect();
        format!("ytsearch20:{query}")
    };
    let out = tokio::process::Command::new("yt-dlp")
        .args(["--flat-playlist", "-J", "--no-warnings"])
        .arg(&target)
        .output()
        .await?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let reason = err
            .lines()
            .rev()
            .find(|l| l.contains("ERROR"))
            .unwrap_or("yt-dlp failed");
        return Err(anyhow::anyhow!("{reason}").into());
    }
    let info: Value = serde_json::from_slice(&out.stdout)?;
    let entries: Vec<Value> = match info["entries"].as_array() {
        Some(list) => list.clone(),
        None => vec![info.clone()],
    };
    let items: Vec<Value> = entries
        .iter()
        .filter_map(|e| {
            let title = e["title"].as_str()?.to_owned();
            let url = e["webpage_url"]
                .as_str()
                .or(e["url"].as_str())
                .map(str::to_owned)
                .or_else(|| {
                    e["id"]
                        .as_str()
                        .map(|id| format!("https://www.youtube.com/watch?v={id}"))
                })?;
            Some(json!({
                "url": url,
                "title": title,
                "duration": e["duration"].as_f64(),
                "channel": e["channel"].as_str().or(e["uploader"].as_str()),
                "parsed": titles::parse(&title),
            }))
        })
        .collect();
    Ok(Json(json!({ "title": info["title"], "items": items })))
}

#[derive(Deserialize)]
struct CreateRequest {
    items: Vec<NewItem>,
    #[serde(default = "default_translator")]
    translator: String,
}

#[derive(Deserialize)]
struct NewItem {
    url: String,
    title: String,
    series: Option<String>,
    season: Option<u32>,
    episode: Option<u32>,
}

fn default_translator() -> String {
    "qwen9b".into()
}

async fn create(
    State(q): State<AppState>,
    Json(req): Json<CreateRequest>,
) -> ApiResult<Json<Value>> {
    check_translator(&req.translator)?;
    let mut ids = Vec::new();
    for item in req.items {
        let parsed = titles::parse(&item.title);
        ids.push(
            db::insert_episode(
                &q.pool,
                &NewEpisode {
                    title: item.title,
                    source_kind: "url".into(),
                    source: item.url,
                    series: item.series.or(parsed.series),
                    season: item.season.or(parsed.season),
                    episode: item.episode.or(parsed.episode),
                    translator: req.translator.clone(),
                },
            )
            .await?,
        );
    }
    for id in &ids {
        q.broadcast_episode(*id).await;
    }
    q.wake.notify_one();
    Ok(Json(json!({ "ids": ids })))
}

#[derive(Deserialize)]
struct FileRequest {
    /// Absolute path on the server, e.g. a file in your import folder.
    path: String,
    title: Option<String>,
    series: Option<String>,
    season: Option<u32>,
    episode: Option<u32>,
    #[serde(default = "default_translator")]
    translator: String,
}

async fn create_from_file(
    State(q): State<AppState>,
    Json(req): Json<FileRequest>,
) -> ApiResult<Json<Value>> {
    check_translator(&req.translator)?;
    let path = std::path::Path::new(&req.path);
    if !path.is_absolute() || !path.is_file() {
        return Err(anyhow::anyhow!("{} is not a file on the server", req.path).into());
    }
    let title = req
        .title
        .or_else(|| path.file_stem().and_then(|s| s.to_str()).map(str::to_owned))
        .unwrap_or_default();
    let parsed = titles::parse(&title);
    let id = db::insert_episode(
        &q.pool,
        &NewEpisode {
            title,
            source_kind: "file".into(),
            source: req.path,
            series: req.series.or(parsed.series),
            season: req.season.or(parsed.season),
            episode: req.episode.or(parsed.episode),
            translator: req.translator,
        },
    )
    .await?;
    q.broadcast_episode(id).await;
    q.wake.notify_one();
    Ok(Json(json!({ "ids": [id] })))
}

fn check_translator(t: &str) -> ApiResult<()> {
    if !matches!(t, "qwen9b" | "moe") {
        return Err(anyhow::anyhow!("unknown translator {t:?} (qwen9b | moe)").into());
    }
    Ok(())
}

async fn retry(State(q): State<AppState>, Path(id): Path<i64>) -> ApiResult<StatusCode> {
    db::retry(&q.pool, id).await?;
    q.broadcast_episode(id).await;
    q.wake.notify_one();
    Ok(StatusCode::NO_CONTENT)
}

/// Stops scheduling further stages; a stage already running finishes first.
async fn cancel(State(q): State<AppState>, Path(id): Path<i64>) -> ApiResult<StatusCode> {
    db::set_state(&q.pool, id, "cancelled", None).await?;
    q.broadcast_episode(id).await;
    Ok(StatusCode::NO_CONTENT)
}

async fn subtitles(
    State(q): State<AppState>,
    Path((id, lang)): Path<(i64, String)>,
) -> ApiResult<Response> {
    let ep = db::episode(&q.pool, id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("no episode #{id}"))?;
    let path = q
        .subtitle_path(&ep, &lang)
        .ok_or_else(|| anyhow::anyhow!("unknown language {lang}"))?;
    let body = tokio::fs::read(&path)
        .await
        .map_err(|_| anyhow::anyhow!("subtitles not ready yet"))?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("subtitles.srt")
        .to_owned();
    // RFC 5987 so Arabic file names survive.
    let encoded: String = name
        .bytes()
        .map(|b| {
            if b.is_ascii_alphanumeric() || b".-_".contains(&b) {
                (b as char).to_string()
            } else {
                format!("%{b:02X}")
            }
        })
        .collect();
    Ok((
        [
            (
                header::CONTENT_TYPE,
                "application/x-subrip; charset=utf-8".to_owned(),
            ),
            (
                header::CONTENT_DISPOSITION,
                format!("attachment; filename*=UTF-8''{encoded}"),
            ),
        ],
        body,
    )
        .into_response())
}

async fn gpu(State(q): State<AppState>) -> Json<Value> {
    Json(json!(q.gpu_status()))
}

#[derive(Deserialize)]
struct GpuRequest {
    paused: bool,
}

async fn set_gpu(State(q): State<AppState>, Json(req): Json<GpuRequest>) -> ApiResult<Json<Value>> {
    q.set_paused(req.paused).await?;
    Ok(Json(json!(q.gpu_status())))
}

/// Server-sent events: episode snapshots, progress ticks and GPU status.
async fn events(
    State(q): State<AppState>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let stream = BroadcastStream::new(q.events.subscribe())
        .filter_map(|msg| msg.ok())
        .map(|data| Ok(Event::default().data(data)));
    Sse::new(stream).keep_alive(KeepAlive::default())
}
