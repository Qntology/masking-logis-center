/// Top-level inference API.
///
/// Pipeline: text → tokenize → model forward → Viterbi decode → span extraction

use std::path::Path;
use std::time::Instant;

use burn::prelude::*;

use crate::config::{ModelConfig, ViterbiConfig};
use crate::model::privacy_filter::PrivacyFilterModel;
use crate::viterbi::{self, PrivacySpan};
use crate::weights;

/// The main inference engine.
pub struct PrivacyFilterInference<B: Backend> {
    pub model: PrivacyFilterModel<B>,
    pub tokenizer: tokenizers::Tokenizer,
    pub viterbi_config: ViterbiConfig,
    pub device: B::Device,
}

impl<B: Backend> PrivacyFilterInference<B> {
    /// Load the model, tokenizer, and Viterbi config from a model directory.
    ///
    /// The directory should contain:
    ///   - config.json
    ///   - model.safetensors
    ///   - tokenizer.json
    ///   - viterbi_calibration.json (optional)
    pub fn load(model_dir: &Path, device: B::Device) -> anyhow::Result<Self> {
        let config_path = model_dir.join("config.json");
        let weights_path = model_dir.join("model.safetensors");
        let tokenizer_path = model_dir.join("tokenizer.json");
        let viterbi_path = model_dir.join("viterbi_calibration.json");

        // Load config
        let config = ModelConfig::from_file(&config_path)?;
        eprintln!("Model config: {} layers, {} hidden, {} experts (top-{})",
            config.num_hidden_layers, config.hidden_size,
            config.num_local_experts, config.num_experts_per_tok);

        // Load tokenizer
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        eprintln!("Tokenizer loaded ({} tokens)", tokenizer.get_vocab_size(false));

        // Load Viterbi config (optional, defaults to all-zero biases)
        let viterbi_config = if viterbi_path.exists() {
            ViterbiConfig::from_file(&viterbi_path, "default")
                .unwrap_or_default()
        } else {
            ViterbiConfig::default()
        };

        // Load model weights
        let t0 = Instant::now();
        let model = weights::load_model(
            &config,
            weights_path.to_str().unwrap(),
            &device,
        )?;
        eprintln!("Weights loaded in {:.1}s", t0.elapsed().as_secs_f64());

        Ok(Self {
            model,
            tokenizer,
            viterbi_config,
            device,
        })
    }

    /// Run inference on a text string.
    ///
    /// Returns detected privacy spans with entity type, confidence, and text.
    pub fn predict(&self, text: &str) -> anyhow::Result<Vec<PrivacySpan>> {
        let t0 = Instant::now();

        // 1. Tokenize
        let encoding = self.tokenizer.encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let input_ids = encoding.get_ids();
        let tokens: Vec<String> = encoding.get_tokens().iter().map(|s| s.to_string()).collect();
        let offsets: Vec<(usize, usize)> = encoding.get_offsets().to_vec();
        let seq_len = input_ids.len();

        eprintln!("Tokenized: {} tokens in {:.1}ms",
            seq_len, t0.elapsed().as_secs_f64() * 1000.0);

        // 2. Model forward pass
        let t1 = Instant::now();
        let logits = self.model.forward(input_ids, &self.device);
        // [1, seq_len, num_labels]

        let logits_data: Vec<f32> = logits.to_data().convert::<f32>().to_vec::<f32>().unwrap();
        eprintln!("Forward pass: {:.1}ms", t1.elapsed().as_secs_f64() * 1000.0);

        // 3. Viterbi decode
        let t2 = Instant::now();
        let label_path = viterbi::viterbi_decode(&logits_data, seq_len, &self.viterbi_config);
        eprintln!("Viterbi decode: {:.1}ms", t2.elapsed().as_secs_f64() * 1000.0);

        // 4. Extract spans
        let spans = viterbi::extract_spans(&label_path, &logits_data, &tokens, &offsets, text);

        eprintln!("Total: {:.1}ms, {} spans detected",
            t0.elapsed().as_secs_f64() * 1000.0, spans.len());

        Ok(spans)
    }

    /// Run inference and return raw logits (no Viterbi decoding).
    ///
    /// Returns logits as Vec<f32> of shape [seq_len, num_labels].
    pub fn predict_logits(&self, text: &str) -> anyhow::Result<(Vec<u32>, Vec<f32>)> {
        let encoding = self.tokenizer.encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();

        let logits = self.model.forward(&input_ids, &self.device);
        let logits_data: Vec<f32> = logits.to_data().convert::<f32>().to_vec::<f32>().unwrap();

        Ok((input_ids, logits_data))
    }

    /// Run inference and return per-token argmax labels (no Viterbi).
    pub fn predict_argmax(&self, text: &str) -> anyhow::Result<Vec<String>> {
        let (_, logits_data) = self.predict_logits(text)?;
        let labels = crate::config::build_label_list();
        let num_labels = labels.len();
        let seq_len = logits_data.len() / num_labels;

        let mut result = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let offset = t * num_labels;
            let mut best_idx = 0;
            let mut best_val = f32::NEG_INFINITY;
            for l in 0..num_labels {
                if logits_data[offset + l] > best_val {
                    best_val = logits_data[offset + l];
                    best_idx = l;
                }
            }
            result.push(labels[best_idx].clone());
        }

        Ok(result)
    }
}
