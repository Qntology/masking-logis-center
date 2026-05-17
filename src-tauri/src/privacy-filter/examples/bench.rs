//! Benchmark: time Rust inference across sample texts.
//!
//! ```bash
//! cargo run --example bench --release -- -m data
//! ```

use std::path::PathBuf;
use std::time::Instant;
use clap::Parser;
use privacy_filter_rs::backend::{B, Device};
use privacy_filter_rs::PrivacyFilterInference;

#[derive(Parser)]
struct Args {
    #[arg(short = 'm', long)]
    model_dir: PathBuf,

    #[arg(short = 't', long, default_value = "0")]
    threads: usize,

    /// Number of warmup iterations per sample
    #[arg(long, default_value = "1")]
    warmup: usize,

    /// Number of timed iterations per sample
    #[arg(long, default_value = "5")]
    iters: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let n = privacy_filter_rs::init_threads(Some(args.threads));
    eprintln!("Using {n} threads\n");

    let device = <Device as Default>::default();

    eprintln!("Loading model...");
    let t0 = Instant::now();
    let engine = PrivacyFilterInference::<B>::load(&args.model_dir, device)?;
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("Model loaded in {load_ms:.0} ms\n");

    let samples = [
        "My name is Alice Smith",
        "You can reach me at alice.smith@example.com or call 555-0123.",
        "My account number is 4532-1234-5678-9012 and my password is hunter2.",
        "Born on January 15, 1990, Alice visited https://secret-site.com/login.",
        "The weather is nice today and the stock market went up.",
        "My name is Harry Potter and my email is harry.potter@hogwarts.edu.",
    ];

    println!(
        "{:<65} {:>6} {:>8} {:>8} {:>8}",
        "Text", "Tokens", "Entities", "Avg ms", "Min ms"
    );
    println!("{}", "-".repeat(100));

    let mut total_tokens = 0usize;
    let mut total_avg = 0.0f64;
    let mut total_min = 0.0f64;

    for text in &samples {
        // Warmup
        for _ in 0..args.warmup {
            let _ = engine.predict(text)?;
        }

        // Timed runs
        let mut times = Vec::with_capacity(args.iters);
        let mut last_spans = Vec::new();
        let mut last_tokens = 0;
        for _ in 0..args.iters {
            let t0 = Instant::now();
            let spans = engine.predict(text)?;
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            times.push(elapsed);

            // Get token count from logits call (only need once)
            if last_tokens == 0 {
                let (ids, _) = engine.predict_logits(text)?;
                last_tokens = ids.len();
            }
            last_spans = spans;
        }

        let avg_ms: f64 = times.iter().sum::<f64>() / times.len() as f64;
        let min_ms: f64 = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let n_entities = last_spans.len();

        let display_text = if text.len() > 60 {
            format!("{}...", &text[..60])
        } else {
            text.to_string()
        };

        println!(
            "{:<65} {:>6} {:>8} {:>8.1} {:>8.1}",
            display_text, last_tokens, n_entities, avg_ms, min_ms
        );

        total_tokens += last_tokens;
        total_avg += avg_ms;
        total_min += min_ms;
    }

    println!("{}", "-".repeat(100));
    println!(
        "{:<65} {:>6} {:>8} {:>8.1} {:>8.1}",
        "TOTAL", total_tokens, "", total_avg, total_min
    );
    println!(
        "\nThroughput (avg): {:.0} tokens/sec",
        total_tokens as f64 / (total_avg / 1000.0)
    );
    println!(
        "Throughput (min): {:.0} tokens/sec",
        total_tokens as f64 / (total_min / 1000.0)
    );

    Ok(())
}
