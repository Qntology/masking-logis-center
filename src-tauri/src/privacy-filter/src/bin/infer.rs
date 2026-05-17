use std::path::PathBuf;

use clap::Parser;
use privacy_filter_rs::backend::{B, Device};

#[derive(Parser, Debug)]
#[command(name = "privacy-filter")]
#[command(about = "OpenAI Privacy Filter — PII detection in Rust")]
struct Args {
    /// Path to the model directory (containing config.json, model.safetensors, tokenizer.json)
    #[arg(short = 'm', long)]
    model_dir: PathBuf,

    /// Input text to classify. If omitted, reads from stdin.
    #[arg()]
    text: Option<String>,

    /// Number of threads (0 = all CPUs)
    #[arg(short = 't', long, default_value = "0")]
    threads: usize,

    /// Output format: "spans" (default), "labels", "logits"
    #[arg(short = 'f', long, default_value = "spans")]
    format: String,

    /// Viterbi operating point name
    #[arg(long, default_value = "default")]
    operating_point: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Thread pool
    let n = privacy_filter_rs::init_threads(Some(args.threads));
    eprintln!("Using {n} threads");

    // Device
    let device = <Device as Default>::default();

    // Load model
    let engine = privacy_filter_rs::PrivacyFilterInference::<B>::load(
        &args.model_dir,
        device,
    )?;

    // Get input text
    let text = if let Some(t) = args.text {
        t
    } else {
        let mut buf = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
        buf.trim_end().to_string()
    };

    if text.is_empty() {
        eprintln!("No input text provided.");
        return Ok(());
    }

    match args.format.as_str() {
        "spans" => {
            let spans = engine.predict(&text)?;
            // Output as JSON array
            print!("[");
            for (i, span) in spans.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!(
                    "{{\"entity_group\": \"{}\", \"score\": {:.6}, \"word\": {}, \"start\": {}, \"end\": {}}}",
                    span.entity_group,
                    span.score,
                    serde_json::to_string(&span.word)?,
                    span.start,
                    span.end,
                );
            }
            println!("]");
        }
        "labels" => {
            let labels = engine.predict_argmax(&text)?;
            for label in &labels {
                println!("{label}");
            }
        }
        "logits" => {
            let (ids, logits) = engine.predict_logits(&text)?;
            let num_labels = 33;
            println!("token_id\tlogits");
            for (t, &id) in ids.iter().enumerate() {
                let offset = t * num_labels;
                let logit_str: Vec<String> = logits[offset..offset + num_labels]
                    .iter()
                    .map(|v| format!("{v:.4}"))
                    .collect();
                println!("{id}\t{}", logit_str.join("\t"));
            }
        }
        other => {
            anyhow::bail!("Unknown format: {other}. Use 'spans', 'labels', or 'logits'.");
        }
    }

    Ok(())
}
