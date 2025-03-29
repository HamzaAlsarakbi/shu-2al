use std::env;

use srt::srt::SRT;

mod srt;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <input_file> <output_file>", args[0]);
        return;
    }
    // Initialize the subscriber
    tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(std::io::stdout)
        .init();

    let input_file = &args[1];
    let output_file = &args[2];

    let mut srt = SRT::new(input_file);
    match srt.read_file() {
        Ok(_) => tracing::debug!("File read successfully!"),
        Err(e) => {
            tracing::error!("Error reading file: {}", e);
            return;
        }
    }

    match srt.write_file(output_file) {
        Ok(_) => tracing::info!("File written successfully!"),
        Err(e) => tracing::error!("Error writing file: {}", e),
    }
}
