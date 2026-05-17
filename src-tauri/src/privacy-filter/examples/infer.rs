//! Example: run privacy filter inference on sample text.
//!
//! ```bash
//! cargo run --example infer --release -- --model-dir /path/to/privacy-filter
//! ```

use std::path::PathBuf;
use clap::Parser;
use privacy_filter_rs::backend::{B, Device};

#[derive(Parser)]
struct Args {
    #[arg(short = 'm', long)]
    model_dir: PathBuf,

    #[arg(short = 't', long, default_value = "0")]
    threads: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let n = privacy_filter_rs::init_threads(Some(args.threads));
    eprintln!("Using {n} threads");

    let device = <Device as Default>::default();

    eprintln!("Loading model...");
    let engine = privacy_filter_rs::PrivacyFilterInference::<B>::load(
        &args.model_dir,
        device,
    )?;

    let samples = [
        "My name is Alice Smith and I live at 123 Main Street, Springfield.",
        "You can reach me at alice.smith@example.com or call 555-0123.",
        "My account number is 4532-1234-5678-9012 and my password is hunter2.",
        "Born on January 15, 1990, Alice visited https://secret-site.com/login.",
        "The weather is nice today and the stock market went up.",
    ];

    for text in &samples {
        println!("\n--- Input: {text}");
        let spans = engine.predict(text)?;
        if spans.is_empty() {
            println!("  No PII detected.");
        } else {
            for span in &spans {
                println!(
                    "  [{:>15}] {:<30} (score: {:.4}, chars {}..{})",
                    span.entity_group, span.word, span.score, span.start, span.end
                );
            }
        }
    }

    Ok(())
}
