//! Editor support: cues in the database, publishing SRTs from them, the
//! translator's learning context, and the browser media (proxy + peaks).

use std::path::Path;
use std::process::Stdio;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use sqlx::SqlitePool;
use subs::{Cue, Timestamp, srt};
use tokio::io::AsyncReadExt;

use crate::db::{self, ImportedCue};
use crate::pipeline::{self, Paths};
use crate::translate::Translated;

/// After a render: loads the cues and translation into the database (keeping
/// the user's edited/verified cues), then writes the SRTs from the database.
pub async fn import_and_publish(pool: &SqlitePool, episode_id: i64, paths: &Paths) -> Result<()> {
    let cues = srt::parse(&tokio::fs::read_to_string(&paths.cues).await?)?;
    let translation: Value =
        serde_json::from_str(&tokio::fs::read_to_string(&paths.translation).await?)?;
    let lines = translation["lines"]
        .as_array()
        .context("translation.json has no lines")?;
    if lines.len() != cues.len() {
        bail!(
            "translation has {} lines for {} cues",
            lines.len(),
            cues.len()
        );
    }
    let imported: Vec<ImportedCue> = cues
        .iter()
        .zip(lines)
        .map(|(c, l)| {
            let ar_asr = c.text.replace('\n', " ");
            let ar = l["ar"].as_str().unwrap_or_default().trim();
            ImportedCue {
                start_ms: c.start.0 as i64,
                end_ms: c.end.0 as i64,
                ar_text: if ar.is_empty() {
                    ar_asr.clone()
                } else {
                    ar.to_owned()
                },
                ar_asr,
                en: l["en"].as_str().unwrap_or_default().trim().to_owned(),
                uncertain: l["uncertain"].as_bool().unwrap_or(false),
            }
        })
        .collect();
    db::import_cues(pool, episode_id, &imported).await?;
    publish(pool, episode_id, paths).await.map(|_| ())
}

/// Writes `.ar.srt` / `.en.srt` from the database cues (deleted cues skipped).
pub async fn publish(pool: &SqlitePool, episode_id: i64, paths: &Paths) -> Result<(usize, usize)> {
    let rows = db::cues(pool, episode_id).await?;
    let (cues, lines): (Vec<Cue>, Vec<Translated>) = rows
        .iter()
        .filter(|r| r.status != "deleted")
        .map(|r| {
            (
                Cue::new(
                    Timestamp(r.start_ms.max(0) as u64),
                    Timestamp(r.end_ms.max(0) as u64),
                    "",
                ),
                Translated {
                    ar: r.ar_text.clone(),
                    en: r.en_text.clone(),
                    ..Default::default()
                },
            )
        })
        .unzip();
    let (ar, en) = pipeline::render(&cues, &lines);
    tokio::fs::create_dir_all(paths.out_ar.parent().expect("has parent")).await?;
    tokio::fs::write(&paths.out_ar, srt::write(&ar)).await?;
    tokio::fs::write(&paths.out_en, srt::write(&en)).await?;
    Ok((ar.len(), en.len()))
}

/// What the translator should learn from for this series: the user's
/// glossary and the lines they fixed or verified.
pub async fn learning_context(pool: &SqlitePool, series: &str) -> Result<String> {
    let mut out = String::new();
    let glossary = db::glossary(pool, series).await?;
    if !glossary.is_empty() {
        out.push_str("\n## Glossary (always use these translations)\n");
        for g in &glossary {
            out.push_str(&format!("- {} → {}\n", g.ar, g.en));
        }
    }
    let examples = db::approved_examples(pool, series, 12).await?;
    if !examples.is_empty() {
        out.push_str(
            "\n## Lines the viewer corrected or approved (match this style and these choices)\n",
        );
        for (ar, en) in &examples {
            out.push_str(&format!("- {ar} → {en}\n"));
        }
    }
    Ok(out)
}

/// Browser media for the editor, if it's ready.
pub fn media_status(work: &Path) -> Value {
    json!({
        "proxy": work.join("proxy.mp4").is_file(),
        "peaks": work.join("peaks.json").is_file(),
    })
}

/// Builds `proxy.mp4` (480p H.264/AAC, 2 s keyframes for snappy seeking) and
/// `peaks.json` (waveform, 50 peaks per second) in the work directory.
pub async fn prepare_media(work: &Path, media: &Path) -> Result<()> {
    let audio = work.join("audio.16k.flac");
    if !work.join("peaks.json").is_file() {
        let peaks = peaks(&audio, 50).await?;
        tokio::fs::write(work.join("peaks.json"), serde_json::to_vec(&peaks)?).await?;
    }
    if !work.join("proxy.mp4").is_file() {
        let tmp = work.join("proxy.mp4.tmp");
        let status = tokio::process::Command::new("ffmpeg")
            .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i"])
            .arg(media)
            .args([
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-vf",
                "scale=-2:480,format=yuv420p",
            ])
            .args(["-c:v", "libx264", "-preset", "veryfast", "-crf", "27"])
            .args(["-force_key_frames", "expr:gte(t,n_forced*2)"])
            .args([
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                "-ac",
                "2",
                "-movflags",
                "+faststart",
                "-f",
                "mp4",
            ])
            .arg(&tmp)
            .status()
            .await?;
        if !status.success() {
            bail!(
                "ffmpeg could not build the editor video from {}",
                media.display()
            );
        }
        tokio::fs::rename(&tmp, work.join("proxy.mp4")).await?;
    }
    Ok(())
}

/// Max-amplitude peaks (0–100) at `rate` per second, decoded with ffmpeg.
async fn peaks(audio: &Path, rate: u32) -> Result<Value> {
    let mut child = tokio::process::Command::new("ffmpeg")
        .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(audio)
        .args(["-f", "s16le", "-ac", "1", "-ar", "16000", "-"])
        .stdout(Stdio::piped())
        .spawn()?;
    let mut pcm = Vec::new();
    child
        .stdout
        .take()
        .expect("piped")
        .read_to_end(&mut pcm)
        .await?;
    if !child.wait().await?.success() {
        bail!("ffmpeg could not decode {}", audio.display());
    }
    let window = (16_000 / rate) as usize * 2;
    let peaks: Vec<u8> = pcm
        .chunks(window)
        .map(|chunk| {
            let max = chunk
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| i16::from_le_bytes(*b).unsigned_abs())
                .max()
                .unwrap_or(0);
            (u32::from(max) * 100 / 32_768) as u8
        })
        .collect();
    let duration = pcm.len() as f64 / 2.0 / 16_000.0;
    Ok(json!({ "rate": rate, "duration": duration, "peaks": peaks }))
}
