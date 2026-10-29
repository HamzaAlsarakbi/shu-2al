//! `shu2al run`: the hard-coded workflow for one episode, start to finish.
//!
//! extract → asr → align → cues → translate → render. Every stage writes its
//! output into the work directory and is skipped next time if that output
//! exists, unless `--from` asks to redo it and everything after.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::ValueEnum;
use serde_json::{Value, json};
use subs::{Cue, filter, segment, srt, transcript::Transcript};

use crate::llm::Lease;
use crate::translate::{self, Translated};
use crate::worker::Launcher;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Stage {
    Extract,
    Asr,
    Align,
    Cues,
    Translate,
    Render,
}

pub enum Llm {
    /// Start llama-server in Docker for the translate stage.
    Docker {
        image: String,
        gguf: String,
        port: u16,
    },
    /// Use a server that's already running.
    External(String),
}

pub struct Options {
    pub media: PathBuf,
    pub work: PathBuf,
    pub name: String,
    pub from: Option<Stage>,
    pub launcher: Launcher,
    pub models: PathBuf,
    pub llm: Llm,
}

struct Paths {
    audio: PathBuf,
    asr_dir: PathBuf,
    transcript: PathBuf,
    aligned: PathBuf,
    cues: PathBuf,
    translation: PathBuf,
    out_ar: PathBuf,
    out_en: PathBuf,
}

impl Paths {
    fn new(work: &Path, name: &str) -> Self {
        let asr_dir = work.join("asr");
        Self {
            audio: work.join("audio.16k.flac"),
            transcript: asr_dir.join("transcript.json"),
            aligned: asr_dir.join("transcript.aligned.json"),
            asr_dir,
            cues: work.join("cues.ar.srt"),
            translation: work.join("translation.json"),
            out_ar: work.join("out").join(format!("{name}.ar.srt")),
            out_en: work.join("out").join(format!("{name}.en.srt")),
        }
    }
}

pub async fn run(opts: Options) -> Result<()> {
    tokio::fs::create_dir_all(&opts.work).await?;
    let p = Paths::new(&opts.work, &opts.name);
    let redo =
        |stage: Stage, output: &Path| opts.from.is_some_and(|f| f <= stage) || !output.exists();
    // Stats accumulate across runs, so a rerun of one stage keeps the others.
    let stats_path = opts.work.join("run.json");
    let mut stats: serde_json::Map<String, Value> = std::fs::read_to_string(&stats_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    if redo(Stage::Extract, &p.audio) {
        let t = Instant::now();
        extract_audio(&opts.media, &p.audio).await?;
        stats.insert("extract".into(), json!({ "elapsed_s": secs(t) }));
        done("extract", t);
    }

    if redo(Stage::Asr, &p.transcript) {
        let t = Instant::now();
        let result = opts
            .launcher
            .run(
                "asr",
                json!({ "audio": p.audio }),
                &p.asr_dir,
                json!({}),
                progress("asr"),
            )
            .await?;
        stats.insert("asr".into(), with_elapsed(&result["stats"], t));
        done("asr", t);
    }

    if redo(Stage::Align, &p.aligned) {
        let t = Instant::now();
        let inputs = json!({ "audio": p.audio, "transcript": p.transcript });
        let result = opts
            .launcher
            .run("align", inputs, &p.asr_dir, json!({}), progress("align"))
            .await?;
        stats.insert("align".into(), with_elapsed(&result["stats"], t));
        done("align", t);
    }

    if redo(Stage::Cues, &p.cues) {
        let t = Instant::now();
        let transcript: Transcript =
            serde_json::from_str(&tokio::fs::read_to_string(&p.aligned).await?)
                .context("parsing aligned transcript")?;
        let cues = segment::cues_from_transcript(&transcript, &segment::Rules::default());
        let (kept, dropped) = filter::apply(cues, &filter::Rules::default());
        tokio::fs::write(&p.cues, srt::write(&kept)).await?;
        stats.insert(
            "cues".into(),
            json!({ "cues": kept.len(), "dropped": dropped.len() }),
        );
        done("cues", t);
    }
    let cues = srt::parse(&tokio::fs::read_to_string(&p.cues).await?)?;

    if redo(Stage::Translate, &p.translation) {
        let t = Instant::now();
        let lease = match &opts.llm {
            Llm::External(url) => Lease::external(url),
            Llm::Docker { image, gguf, port } => {
                Lease::start(image, &opts.models, gguf, *port).await?
            }
        };
        let outcome = translate::translate(&lease, &cues, &Default::default(), |n, total| {
            eprint!("\r  translate {n}/{total} cues   ");
            let _ = std::io::stderr().flush();
        })
        .await;
        lease.stop().await;
        let (lines, usage) = outcome?;
        let elapsed = t.elapsed().as_secs_f64();
        let json_lines: Vec<Value> = lines
            .iter()
            .map(|l| json!({ "ar": l.ar, "en": l.en, "uncertain": l.uncertain, "overcorrected": l.overcorrected }))
            .collect();
        tokio::fs::write(
            &p.translation,
            serde_json::to_vec_pretty(&json!({ "lines": json_lines }))?,
        )
        .await?;
        stats.insert(
            "translate".into(),
            json!({
                "elapsed_s": round1(elapsed),
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "tokens_per_s": round1(usage.completion_tokens as f64 / elapsed),
                "uncertain": lines.iter().filter(|l| l.uncertain).count(),
                "overcorrected": lines.iter().filter(|l| l.overcorrected).count(),
            }),
        );
        eprintln!();
        done("translate", t);
    }

    let translation: Value =
        serde_json::from_str(&tokio::fs::read_to_string(&p.translation).await?)?;
    let lines: Vec<Translated> = translation["lines"]
        .as_array()
        .context("translation.json has no lines")?
        .iter()
        .map(|l| Translated {
            ar: l["ar"].as_str().unwrap_or_default().to_owned(),
            en: l["en"].as_str().unwrap_or_default().to_owned(),
            uncertain: l["uncertain"].as_bool().unwrap_or(false),
            overcorrected: l["overcorrected"].as_bool().unwrap_or(false),
        })
        .collect();
    if lines.len() != cues.len() {
        bail!(
            "translation has {} lines for {} cues; rerun with --from translate",
            lines.len(),
            cues.len()
        );
    }

    let (ar, en) = render(&cues, &lines);
    tokio::fs::create_dir_all(p.out_ar.parent().expect("has parent")).await?;
    tokio::fs::write(&p.out_ar, srt::write(&ar)).await?;
    tokio::fs::write(&p.out_en, srt::write(&en)).await?;
    eprintln!("  wrote {} and {}", p.out_ar.display(), p.out_en.display());

    tokio::fs::write(
        &stats_path,
        serde_json::to_vec_pretty(&Value::Object(stats))?,
    )
    .await?;
    eprintln!("  stats: {}", stats_path.display());
    Ok(())
}

/// Builds the two tracks on the shared timing. Cues the translator emptied
/// (non-speech) are dropped from both. Arabic lines start with an RLM so
/// trailing punctuation renders on the correct side in LTR-default players.
fn render(cues: &[Cue], lines: &[Translated]) -> (Vec<Cue>, Vec<Cue>) {
    let (mut ar, mut en) = (Vec::new(), Vec::new());
    for (cue, line) in cues.iter().zip(lines) {
        if line.ar.trim().is_empty() && line.en.trim().is_empty() {
            continue;
        }
        let ar_text = if line.ar.trim().is_empty() {
            cue.text.replace('\n', " ")
        } else {
            line.ar.clone()
        };
        let ar_text = segment::break_lines(&ar_text, 42)
            .lines()
            .map(|l| format!("\u{200F}{l}"))
            .collect::<Vec<_>>()
            .join("\n");
        ar.push(Cue::new(cue.start, cue.end, ar_text));
        en.push(Cue::new(
            cue.start,
            cue.end,
            segment::break_lines(&line.en, 42),
        ));
    }
    (ar, en)
}

async fn extract_audio(media: &Path, audio: &Path) -> Result<()> {
    let status = tokio::process::Command::new("ffmpeg")
        .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(media)
        .args([
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "flac",
            "-sample_fmt",
            "s16",
        ])
        .arg(audio)
        .status()
        .await
        .context("running ffmpeg")?;
    if !status.success() {
        bail!("ffmpeg failed to extract audio from {}", media.display());
    }
    Ok(())
}

fn progress(stage: &'static str) -> impl Fn(f64, &str) {
    move |pct, msg| {
        eprint!("\r  {stage:<9} {pct:>5.1}%  {msg:<40}");
        let _ = std::io::stderr().flush();
    }
}

fn done(stage: &str, t: Instant) {
    eprintln!(
        "\r  {stage:<9} done in {:.1}s{:<40}",
        t.elapsed().as_secs_f64(),
        ""
    );
}

fn secs(t: Instant) -> f64 {
    round1(t.elapsed().as_secs_f64())
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

fn with_elapsed(stats: &Value, t: Instant) -> Value {
    let mut v = stats.clone();
    if let Some(obj) = v.as_object_mut() {
        obj.insert("wall_s".into(), json!(secs(t)));
    }
    v
}
