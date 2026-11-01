//! `shu2al`: the shu-2al server (from M2) and subtitle command-line tools.

use std::path::{Path, PathBuf};

mod api;
mod db;
mod llm;
mod pipeline;
mod queue;
mod titles;
mod translate;
mod worker;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use subs::{Cue, filter};

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Subtitle file tools (format picked by .srt / .vtt extension).
    #[command(subcommand)]
    Subs(SubsCommand),
    /// Subtitle one episode end to end: writes <work>/out/<name>.{ar,en}.srt.
    Run(Box<RunArgs>),
    /// Run the dashboard server and job queue.
    Serve(ServeArgs),
}

#[derive(clap::Args)]
struct ServeArgs {
    /// Address to listen on. Keep it on localhost/LAN/Tailscale; there is no auth.
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: String,
    /// Data directory (database and episode work dirs). Default: ~/shu2al-data/library
    #[arg(long)]
    data: Option<PathBuf>,
    /// Model cache. Default: ~/shu2al-models
    #[arg(long)]
    models: Option<PathBuf>,
    /// Built web app to serve.
    #[arg(long, default_value = "web/dist")]
    web: PathBuf,
    #[arg(long, default_value = "shu2al-worker:dev")]
    worker_image: String,
    #[arg(long, default_value = "ghcr.io/ggml-org/llama.cpp:server-cuda")]
    llm_image: String,
    #[arg(long, default_value_t = 8081)]
    llm_port: u16,
    /// Hold GPU stages while other apps use more than this much VRAM (MiB).
    #[arg(long, default_value_t = 1200)]
    gpu_busy_mb: u64,
}

#[derive(clap::Args)]
struct RunArgs {
    /// Video or audio file.
    media: PathBuf,
    /// Work directory for intermediate files and output.
    #[arg(long)]
    work: PathBuf,
    /// Output base name (default: the media file name).
    #[arg(long)]
    name: Option<String>,
    /// Redo this stage and every later one, even if outputs exist.
    #[arg(long, value_enum)]
    from: Option<pipeline::Stage>,
    #[arg(long, value_enum, default_value = "docker")]
    worker: WorkerKind,
    #[arg(long, default_value = "shu2al-worker:dev")]
    worker_image: String,
    /// Python for `--worker local`.
    #[arg(long, default_value = "python3")]
    python: PathBuf,
    /// Model cache (Hugging Face cache and GGUF files). Default: ~/shu2al-models.
    #[arg(long)]
    models: Option<PathBuf>,
    /// Use an already-running OpenAI-compatible server instead of starting one.
    #[arg(long)]
    llm_url: Option<String>,
    /// Translate with a seq2seq model (e.g. fine-tuned NLLB) instead of the LLM.
    /// Path inside the models mount, such as /models/mt/nllb-levantine.
    #[arg(long, conflicts_with = "llm_url")]
    mt_model: Option<String>,
    #[arg(long, default_value = "ghcr.io/ggml-org/llama.cpp:server-cuda")]
    llm_image: String,
    /// GGUF path relative to the models directory.
    #[arg(long, default_value = "llm/Qwen3.5-9B-Q4_K_M.gguf")]
    llm_model: String,
    #[arg(long, default_value_t = 8081)]
    llm_port: u16,
    /// Show brief (synopsis + character names) for the translator, as a text file.
    #[arg(long)]
    brief: Option<PathBuf>,
    /// Extra llama-server flag (repeatable), e.g. --llm-arg=--cpu-moe
    #[arg(long = "llm-arg", allow_hyphen_values = true)]
    llm_args: Vec<String>,
}

#[derive(Clone, Copy, ValueEnum)]
enum WorkerKind {
    Docker,
    Local,
}

#[derive(Subcommand)]
enum SubsCommand {
    /// Convert between SRT and WebVTT.
    Convert { input: PathBuf, output: PathBuf },
    /// Shift every cue by a number of milliseconds (negative = earlier).
    Shift {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, allow_negative_numbers = true)]
        ms: i64,
    },
    /// Drop empty cues, known hallucination phrases and decoding loops.
    Clean { input: PathBuf, output: PathBuf },
    /// Build filtered cues from a worker transcript.json.
    Cues {
        transcript: PathBuf,
        output: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Secrets such as HF_TOKEN live in a gitignored .env next to the repo.
    let _ = dotenvy::dotenv();
    match Cli::parse().command {
        Command::Subs(cmd) => run_subs(cmd),
        Command::Run(args) => run_episode(*args).await,
        Command::Serve(args) => serve(args).await,
    }
}

fn run_subs(cmd: SubsCommand) -> Result<()> {
    match cmd {
        SubsCommand::Convert { input, output } => write(&output, &read(&input)?),
        SubsCommand::Shift { input, output, ms } => {
            let mut cues = read(&input)?;
            subs::shift_all(&mut cues, ms);
            write(&output, &cues)
        }
        SubsCommand::Clean { input, output } => {
            let (kept, dropped) = filter::apply(read(&input)?, &filter::Rules::default());
            for (cue, reason) in &dropped {
                eprintln!(
                    "dropped {} {reason:?}: {}",
                    cue.start,
                    cue.text.replace('\n', " / ")
                );
            }
            eprintln!("kept {}, dropped {}", kept.len(), dropped.len());
            write(&output, &kept)
        }
        SubsCommand::Cues { transcript, output } => {
            let json = std::fs::read_to_string(&transcript)
                .with_context(|| format!("reading {}", transcript.display()))?;
            let transcript: subs::transcript::Transcript =
                serde_json::from_str(&json).context("parsing transcript")?;
            let cues = subs::segment::cues_from_transcript(&transcript, &Default::default());
            let (kept, dropped) = filter::apply(cues, &filter::Rules::default());
            eprintln!(
                "{} cues ({} dropped as hallucinations)",
                kept.len(),
                dropped.len()
            );
            write(&output, &kept)
        }
    }
}

enum Format {
    Srt,
    Vtt,
}

fn format_of(path: &Path) -> Result<Format> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    match ext.as_deref() {
        Some("srt") => Ok(Format::Srt),
        Some("vtt") => Ok(Format::Vtt),
        _ => bail!("{}: expected a .srt or .vtt file", path.display()),
    }
}

fn read(path: &Path) -> Result<Vec<Cue>> {
    let format = format_of(path)?;
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    match format {
        Format::Srt => subs::srt::parse(&text),
        Format::Vtt => subs::vtt::parse(&text),
    }
    .with_context(|| format!("parsing {}", path.display()))
}

fn write(path: &Path, cues: &[Cue]) -> Result<()> {
    let text = match format_of(path)? {
        Format::Srt => subs::srt::write(cues),
        Format::Vtt => subs::vtt::write(cues),
    };
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

async fn run_episode(args: RunArgs) -> Result<()> {
    let absolute = |p: &Path| -> Result<PathBuf> {
        std::fs::create_dir_all(p).ok();
        p.canonicalize()
            .with_context(|| format!("resolving {}", p.display()))
    };
    let work = absolute(&args.work)?;
    let models = match args.models {
        Some(m) => m,
        None => {
            PathBuf::from(std::env::var("HOME").context("HOME is not set")?).join("shu2al-models")
        }
    };
    let models = absolute(&models)?;
    let name = match args.name {
        Some(n) => n,
        None => args
            .media
            .file_stem()
            .and_then(|s| s.to_str())
            .context("media path has no file name")?
            .to_owned(),
    };
    let launcher = match args.worker {
        WorkerKind::Docker => worker::Launcher::Docker {
            image: args.worker_image,
            data: work.clone(),
            models: models.clone(),
        },
        WorkerKind::Local => worker::Launcher::Local {
            python: args.python,
        },
    };
    let llm = match (args.llm_url, args.mt_model) {
        (_, Some(model)) => pipeline::Llm::Seq2Seq { model },
        (Some(url), None) => pipeline::Llm::External(url),
        (None, None) => pipeline::Llm::Docker {
            image: args.llm_image,
            gguf: args.llm_model,
            port: args.llm_port,
            extra: args.llm_args,
        },
    };
    let brief = match &args.brief {
        Some(path) => Some(
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?,
        ),
        None => None,
    };
    pipeline::run(
        pipeline::Options {
            brief,
            media: args.media,
            work,
            name,
            launcher,
            models,
            llm,
        },
        args.from,
    )
    .await
}

fn home_dir(sub: &str) -> Result<PathBuf> {
    Ok(PathBuf::from(std::env::var("HOME").context("HOME is not set")?).join(sub))
}

async fn serve(args: ServeArgs) -> Result<()> {
    let data = match args.data {
        Some(d) => d,
        None => home_dir("shu2al-data/library")?,
    };
    std::fs::create_dir_all(&data)?;
    let data = data.canonicalize()?;
    let models = match args.models {
        Some(m) => m,
        None => home_dir("shu2al-models")?,
    };
    let pool = db::open(&data.join("shu2al.db")).await?;
    let queue = queue::Queue::new(
        pool,
        queue::Config {
            data,
            models: models.canonicalize().unwrap_or(models),
            worker_image: args.worker_image,
            llm_image: args.llm_image,
            llm_port: args.llm_port,
            gpu_busy_mb: args.gpu_busy_mb,
        },
    )
    .await?;
    tokio::spawn(queue.clone().run());
    let app = api::router(queue, args.web);
    let listener = tokio::net::TcpListener::bind(&args.bind).await?;
    eprintln!("shu2al listening on http://{}", args.bind);
    axum::serve(listener, app).await?;
    Ok(())
}
