//! `shu2al`: the shu-2al server (from M2) and subtitle command-line tools.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
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

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Subs(cmd) => run_subs(cmd),
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
