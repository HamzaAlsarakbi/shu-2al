//! The hard-coded workflow for one episode:
//! extract → asr → align → cues → translate → render.
//!
//! Every stage reads and writes files in the episode's work directory, so the
//! CLI (`shu2al run`) and the server's queue run the same code: the CLI loops
//! over [`Stage::ALL`], the queue schedules one [`run_stage`] at a time per
//! lane (GPU stages strictly one at a time).

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, ValueEnum)]
pub enum Stage {
    Extract,
    Asr,
    Align,
    Cues,
    Translate,
    Render,
}

impl Stage {
    pub const ALL: [Stage; 6] = [
        Stage::Extract,
        Stage::Asr,
        Stage::Align,
        Stage::Cues,
        Stage::Translate,
        Stage::Render,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Stage::Extract => "extract",
            Stage::Asr => "asr",
            Stage::Align => "align",
            Stage::Cues => "cues",
            Stage::Translate => "translate",
            Stage::Render => "render",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|s| s.name() == name)
    }
}

pub enum Llm {
    /// A seq2seq translation model run by the worker's `mt` stage (e.g. the
    /// fine-tuned NLLB). `model` is a path inside the models mount.
    Seq2Seq { model: String },
    /// Start llama-server in Docker for the translate stage.
    Docker {
        image: String,
        gguf: String,
        port: u16,
        /// Context size; large-KV models (Jais-3) need 8192 on 8 GB.
        ctx: u32,
        /// Extra llama-server flags, e.g. `--cpu-moe` for MoE expert offload.
        extra: Vec<String>,
    },
    /// Use a server that's already running.
    External(String),
}

pub struct Options {
    /// Source media for `extract`.
    pub media: PathBuf,
    pub work: PathBuf,
    pub name: String,
    pub launcher: Launcher,
    pub models: PathBuf,
    pub llm: Llm,
    /// Show brief for the translator (see `translate::Settings::brief`).
    pub brief: Option<String>,
}

pub struct Paths {
    pub audio: PathBuf,
    pub asr_dir: PathBuf,
    pub transcript: PathBuf,
    pub aligned: PathBuf,
    pub cues: PathBuf,
    pub translation: PathBuf,
    pub out_ar: PathBuf,
    pub out_en: PathBuf,
}

impl Paths {
    pub fn new(work: &Path, name: &str) -> Self {
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

    /// The file a stage produces; its existence means the stage is done.
    pub fn output(&self, stage: Stage) -> &Path {
        match stage {
            Stage::Extract => &self.audio,
            Stage::Asr => &self.transcript,
            Stage::Align => &self.aligned,
            Stage::Cues => &self.cues,
            Stage::Translate => &self.translation,
            Stage::Render => &self.out_en,
        }
    }
}

/// `shu2al run`: every stage in order, skipping finished ones unless `from`
/// asks to redo a stage and everything after it.
pub async fn run(opts: Options, from: Option<Stage>) -> Result<()> {
    tokio::fs::create_dir_all(&opts.work).await?;
    let p = Paths::new(&opts.work, &opts.name);
    let stats_path = opts.work.join("run.json");
    let mut stats: serde_json::Map<String, Value> = std::fs::read_to_string(&stats_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    for stage in Stage::ALL {
        if from.is_none_or(|f| stage < f) && p.output(stage).exists() {
            continue;
        }
        let t = Instant::now();
        let name = stage.name();
        let progress = move |pct: f64, msg: &str| {
            eprint!("\r  {name:<9} {pct:>5.1}%  {msg:<40}");
            let _ = std::io::stderr().flush();
        };
        let result = run_stage(stage, &opts, &progress).await?;
        eprintln!(
            "\r  {name:<9} done in {:.1}s{:<40}",
            t.elapsed().as_secs_f64(),
            ""
        );
        stats.insert(name.into(), result);
    }
    eprintln!("  wrote {} and {}", p.out_ar.display(), p.out_en.display());
    tokio::fs::write(
        &stats_path,
        serde_json::to_vec_pretty(&Value::Object(stats))?,
    )
    .await?;
    Ok(())
}

/// Runs exactly one stage and returns its stats. Inputs are the outputs of the
/// earlier stages, already on disk.
pub async fn run_stage(
    stage: Stage,
    opts: &Options,
    progress: &(dyn Fn(f64, &str) + Sync),
) -> Result<Value> {
    tokio::fs::create_dir_all(&opts.work).await?;
    let p = Paths::new(&opts.work, &opts.name);
    let t = Instant::now();
    match stage {
        Stage::Extract => {
            extract_audio(&opts.media, &p.audio).await?;
            Ok(json!({ "elapsed_s": secs(t) }))
        }
        Stage::Asr => {
            let result = opts
                .launcher
                .run(
                    "asr",
                    json!({ "audio": p.audio }),
                    &p.asr_dir,
                    json!({}),
                    progress,
                )
                .await?;
            Ok(with_elapsed(&result["stats"], t))
        }
        Stage::Align => {
            let inputs = json!({ "audio": p.audio, "transcript": p.transcript });
            let result = opts
                .launcher
                .run("align", inputs, &p.asr_dir, json!({}), progress)
                .await?;
            Ok(with_elapsed(&result["stats"], t))
        }
        Stage::Cues => {
            let transcript: Transcript =
                serde_json::from_str(&tokio::fs::read_to_string(&p.aligned).await?)
                    .context("parsing aligned transcript")?;
            let cues = segment::cues_from_transcript(&transcript, &segment::Rules::default());
            let (kept, dropped) = filter::apply(cues, &filter::Rules::default());
            tokio::fs::write(&p.cues, srt::write(&kept)).await?;
            Ok(json!({ "cues": kept.len(), "dropped": dropped.len() }))
        }
        Stage::Translate => {
            let cues = srt::parse(&tokio::fs::read_to_string(&p.cues).await?)?;
            let (lines, report) = match &opts.llm {
                Llm::Seq2Seq { model } => translate_seq2seq(opts, &cues, model, progress).await?,
                Llm::External(_) | Llm::Docker { .. } => {
                    let lease = match &opts.llm {
                        Llm::Docker {
                            image,
                            gguf,
                            port,
                            ctx,
                            extra,
                        } => {
                            progress(0.0, "starting llama-server");
                            Lease::start(image, &opts.models, gguf, *port, *ctx, extra).await?
                        }
                        Llm::External(url) => Lease::external(url),
                        Llm::Seq2Seq { .. } => unreachable!(),
                    };
                    let settings = translate::Settings {
                        brief: opts.brief.clone(),
                        ..Default::default()
                    };
                    let outcome = translate::translate(&lease, &cues, &settings, |n, total| {
                        progress(
                            100.0 * n as f64 / total.max(1) as f64,
                            &format!("{n}/{total} cues"),
                        )
                    })
                    .await;
                    lease.stop().await;
                    outcome?
                }
            };
            let json_lines: Vec<Value> = lines
                .iter()
                .map(|l| json!({ "ar": l.ar, "en": l.en, "uncertain": l.uncertain, "overcorrected": l.overcorrected }))
                .collect();
            tokio::fs::write(
                &p.translation,
                serde_json::to_vec_pretty(&json!({ "lines": json_lines }))?,
            )
            .await?;
            let elapsed = t.elapsed().as_secs_f64();
            let usage = report.usage;
            Ok(json!({
                "elapsed_s": round1(elapsed),
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "tokens_per_s": round1(usage.completion_tokens as f64 / elapsed),
                "uncertain": lines.iter().filter(|l| l.uncertain).count(),
                "overcorrected": lines.iter().filter(|l| l.overcorrected).count(),
                "slipped": report.slipped,
                "shortened": report.shortened,
            }))
        }
        Stage::Render => {
            let cues = srt::parse(&tokio::fs::read_to_string(&p.cues).await?)?;
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
                    "translation has {} lines for {} cues; redo the translate stage",
                    lines.len(),
                    cues.len()
                );
            }
            let (ar, en) = render(&cues, &lines);
            let too_long = en
                .iter()
                .filter(|c| c.text.lines().any(|l| l.chars().count() > 42))
                .count();
            tokio::fs::create_dir_all(p.out_ar.parent().expect("has parent")).await?;
            tokio::fs::write(&p.out_ar, srt::write(&ar)).await?;
            tokio::fs::write(&p.out_en, srt::write(&en)).await?;
            Ok(json!({ "ar_cues": ar.len(), "en_cues": en.len(), "en_too_long": too_long }))
        }
    }
}

/// Translates cue by cue with the worker's `mt` stage. The Arabic track keeps
/// the ASR text as-is (a seq2seq model can't correct or normalize it).
async fn translate_seq2seq(
    opts: &Options,
    cues: &[Cue],
    model: &str,
    progress: &(dyn Fn(f64, &str) + Sync),
) -> Result<(Vec<Translated>, translate::Report)> {
    let dir = opts.work.join("mt");
    tokio::fs::create_dir_all(&dir).await?;
    let lines_path = dir.join("lines.json");
    let lines: Vec<String> = cues.iter().map(|c| c.text.replace('\n', " ")).collect();
    tokio::fs::write(&lines_path, serde_json::to_vec(&lines)?).await?;
    let result = opts
        .launcher
        .run(
            "mt",
            json!({ "lines": lines_path }),
            &dir,
            json!({ "model": model }),
            progress,
        )
        .await?;
    let mt_path = result["outputs"]["mt"]
        .as_str()
        .context("mt stage returned no output")?;
    let mt: Value = serde_json::from_str(&tokio::fs::read_to_string(mt_path).await?)?;
    let en = mt["en"].as_array().context("mt.json has no en list")?;
    if en.len() != cues.len() {
        bail!("mt returned {} lines for {} cues", en.len(), cues.len());
    }
    let out = cues
        .iter()
        .zip(en)
        .map(|(cue, en)| Translated {
            ar: cue.text.replace('\n', " "),
            en: en.as_str().unwrap_or_default().to_owned(),
            ..Default::default()
        })
        .collect();
    Ok((out, translate::Report::default()))
}

/// Builds the two tracks on the shared timing. Each track drops the cues it
/// has no text for (non-speech). Arabic lines start with an RLM so trailing
/// punctuation renders on the correct side in LTR-default players.
pub fn render(cues: &[Cue], lines: &[Translated]) -> (Vec<Cue>, Vec<Cue>) {
    let (mut ar, mut en) = (Vec::new(), Vec::new());
    for (cue, line) in cues.iter().zip(lines) {
        let ar_text = clean(&line.ar);
        let en_text = clean(&line.en);
        if !ar_text.is_empty() {
            let rtl = segment::break_lines(&ar_text, 42)
                .lines()
                .map(|l| format!("\u{200F}{l}"))
                .collect::<Vec<_>>()
                .join("\n");
            ar.push(Cue::new(cue.start, cue.end, rtl));
        }
        if !en_text.is_empty() {
            en.push(Cue::new(
                cue.start,
                cue.end,
                segment::break_lines(&en_text, 42),
            ));
        }
    }
    (ar, en)
}

/// Strips lyric and music markers some ASR models emit, and extra spaces.
fn clean(text: &str) -> String {
    text.replace(['#', '♪', '♫'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
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
