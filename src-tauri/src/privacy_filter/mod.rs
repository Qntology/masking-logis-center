pub mod config;
pub mod viterbi;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear_no_bias, VarBuilder, Embedding};
use anyhow::Result;
use std::path::Path;
use tokenizers::Tokenizer;
use self::config::{ModelConfig, ViterbiConfig};
use self::viterbi::PrivacySpan;

pub struct PrivacyFilterModel {
    embed_tokens: Embedding,
    layers: Vec<TransformerLayer>,
    norm: RmsNorm,
    score_weight: Tensor,
    score_bias: Tensor,
    tokenizer: Tokenizer,
    config: ModelConfig,
    viterbi_config: ViterbiConfig,
    device: Device,
}

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.powf(2.0)?.mean_keepdim(candle_core::D::Minus1)?;
        let x_normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let x_normed = x_normed.to_dtype(x_dtype)?;
        Ok(x_normed.broadcast_mul(&self.weight)?)
    }
}

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new_yarn(dim: usize, max_seq_len: usize, theta: f64, factor: f64, beta_fast: f64, beta_slow: f64, original_len: usize, _truncate: bool, device: &Device) -> Result<Self> {
        // Full Yarn RoPE implementation
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / (theta.powf(i as f64 / dim as f64) as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?.to_dtype(DType::F32)?.unsqueeze(1)?;
        
        // Yarn scaling (simplified for now but using the config values)
        let freqs = t.matmul(&inv_freq.unsqueeze(0)?)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        Ok(Self { cos, sin })
    }

    fn forward(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_b, _h, seq_len, _d) = q.dims4()?;
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        
        let cos = Tensor::cat(&[&cos, &cos], candle_core::D::Minus1)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = Tensor::cat(&[&sin, &sin], candle_core::D::Minus1)?.unsqueeze(0)?.unsqueeze(0)?;
        
        let apply_rotary = |x: &Tensor| -> Result<Tensor> {
            let last_dim = x.dim(candle_core::D::Minus1)?;
            let half = last_dim / 2;
            let x1 = x.narrow(candle_core::D::Minus1, 0, half)?;
            let x2 = x.narrow(candle_core::D::Minus1, half, half)?;
            let rotated = Tensor::cat(&[&x2.neg()?, &x1], candle_core::D::Minus1)?;
            Ok((x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?)
        };

        Ok((apply_rotary(q)?, apply_rotary(k)?))
    }
}

struct SparseMoE {
    router: candle_nn::Linear,
    gate_up_proj: Tensor,
    gate_up_proj_bias: Tensor,
    down_proj: Tensor,
    down_proj_bias: Tensor,
    num_experts: usize,
    num_experts_per_tok: usize,
    intermediate_size: usize,
}

impl SparseMoE {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let router = candle_nn::linear(cfg.hidden_size, cfg.num_local_experts, vb.pp("router"))?;
        let gate_up_proj = vb.get((cfg.num_local_experts, cfg.hidden_size, 2 * cfg.intermediate_size), "experts.gate_up_proj")?;
        let gate_up_proj_bias = vb.get((cfg.num_local_experts, 2 * cfg.intermediate_size), "experts.gate_up_proj_bias")?;
        let down_proj = vb.get((cfg.num_local_experts, cfg.intermediate_size, cfg.hidden_size), "experts.down_proj")?;
        let down_proj_bias = vb.get((cfg.num_local_experts, cfg.hidden_size), "experts.down_proj_bias")?;

        Ok(Self {
            router,
            gate_up_proj,
            gate_up_proj_bias,
            down_proj,
            down_proj_bias,
            num_experts: cfg.num_local_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            intermediate_size: cfg.intermediate_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        use candle_transformers::models::deepseek2::TopKLastDimOp;

        let (b, s, h) = x.dims3()?;
        let x_flat = x.reshape((b * s, h))?;
        let router_logits = self.router.forward(&x_flat)?;
        
        let top_k = router_logits.topk(self.num_experts_per_tok)?;
        let values = top_k.values;
        let indices = top_k.indices;
        
        let routing_weights = candle_nn::ops::softmax(&values, candle_core::D::Minus1)?;
        
        let mut final_out = Tensor::zeros((b * s, h), x.dtype(), x.device())?;
        
        let indices_vec: Vec<u32> = indices.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?;
        let weights_vec: Vec<f32> = routing_weights.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        
        for i in 0..(b * s) {
            for k in 0..self.num_experts_per_tok {
                let expert_idx = indices_vec[i * self.num_experts_per_tok + k] as usize;
                let weight = weights_vec[i * self.num_experts_per_tok + k] / (self.num_experts_per_tok as f32);
                
                let token_input = x_flat.get(i)?.unsqueeze(0)?;
                
                let gu_w = self.gate_up_proj.get(expert_idx)?;
                let gu_b = self.gate_up_proj_bias.get(expert_idx)?;
                let dp_w = self.down_proj.get(expert_idx)?;
                let dp_b = self.down_proj_bias.get(expert_idx)?;

                let gate_up = token_input.matmul(&gu_w)?.broadcast_add(&gu_b)?;
                let gate = gate_up.narrow(candle_core::D::Minus1, 0, self.intermediate_size)?.minimum(7.0)?;
                let up = gate_up.narrow(candle_core::D::Minus1, self.intermediate_size, self.intermediate_size)?.clamp(-7.0, 7.0)?;
                let glu = (gate.clone() * candle_nn::ops::sigmoid(&(gate * 1.702)?))?;
                let out = ((up + 1.0)? * glu)?;
                
                let expert_out = out.matmul(&dp_w)?.broadcast_add(&dp_b)?;
                let weighted_out = (expert_out.get(0)? * weight as f64)?;
                final_out = final_out.pad_with_zeros(0, i, (b*s) - 1 - i)?.broadcast_add(&weighted_out.unsqueeze(0)?)?.narrow(0, i, 1)?.pad_with_zeros(0, i, (b*s) - 1 - i)?;
            }
        }
        
        Ok(final_out.reshape((b, s, h))?)
    }
}

struct TransformerLayer {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    sinks: Tensor,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    moe: SparseMoE,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scaling: f64,
}

impl TransformerLayer {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let q_proj = linear_no_bias(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim, vb.pp("self_attn.q_proj"))?;
        let k_proj = linear_no_bias(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, vb.pp("self_attn.k_proj"))?;
        let v_proj = linear_no_bias(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, vb.pp("self_attn.v_proj"))?;
        let o_proj = linear_no_bias(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, vb.pp("self_attn.o_proj"))?;
        let sinks = vb.get(cfg.num_attention_heads, "self_attn.sinks")?;
        
        let input_layernorm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;
        let moe = SparseMoE::new(cfg, vb.pp("mlp"))?;
        
        Ok(Self {
            q_proj, k_proj, v_proj, o_proj,
            sinks,
            input_layernorm, post_attention_layernorm,
            moe,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            scaling: (cfg.head_dim as f64).powf(-0.25),
        })
    }

    fn forward(&self, x: &Tensor, rope: &RotaryEmbedding, mask: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let residual = x;
        let x_norm = self.input_layernorm.forward(x)?;
        
        let q = self.q_proj.forward(&x_norm)?.reshape((b, s, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = self.k_proj.forward(&x_norm)?.reshape((b, s, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = self.v_proj.forward(&x_norm)?.reshape((b, s, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        
        let (q, k) = rope.forward(&q, &k)?;
        let q = (q * self.scaling)?;
        let k = (k * self.scaling)?;
        
        let n_rep = self.num_heads / self.num_kv_heads;
        let k = if n_rep > 1 { k.unsqueeze(2)?.expand((b, self.num_kv_heads, n_rep, s, self.head_dim))?.flatten(1, 2)? } else { k };
        let v = if n_rep > 1 { v.unsqueeze(2)?.expand((b, self.num_kv_heads, n_rep, s, self.head_dim))?.flatten(1, 2)? } else { v };

        let attn_weights = q.matmul(&k.transpose(2, 3)?)?;
        let attn_weights = attn_weights.broadcast_add(mask)?;
        
        let sinks = self.sinks.reshape((1, self.num_heads, 1, 1))?.expand((b, self.num_heads, s, 1))?;
        let combined = Tensor::cat(&[&attn_weights, &sinks], 3)?;
        let probs = candle_nn::ops::softmax(&combined, 3)?;
        let scores = probs.narrow(3, 0, s)?.contiguous()?;
        
        let attn_out = scores.matmul(&v.contiguous()?)?.transpose(1, 2)?.reshape((b, s, ()))?;
        let attn_out = self.o_proj.forward(&attn_out)?;
        
        let x = (residual + attn_out)?;
        let residual = &x;
        let x_norm = self.post_attention_layernorm.forward(&x)?;
        let moe_out = self.moe.forward(&x_norm)?;
        
        (residual + moe_out).map_err(Into::into)
    }
}

fn create_sliding_window_mask(seq_len: usize, window_size: usize, device: &Device) -> Result<Tensor> {
    let mut mask_data = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if (i as isize - j as isize).abs() > window_size as isize {
                mask_data[i * seq_len + j] = -1e9;
            }
        }
    }
    Ok(Tensor::from_vec(mask_data, (1, 1, seq_len, seq_len), device)?)
}

impl PrivacyFilterModel {
    pub fn load(model_dir: &Path, device: &Device) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let weights_path = model_dir.join("model.safetensors");
        let tokenizer_path = model_dir.join("tokenizer.json");
        let viterbi_path = model_dir.join("viterbi_calibration.json");

        let config = ModelConfig::from_file(&config_path)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)? };

        let embed_tokens = candle_nn::embedding(config.vocab_size, config.hidden_size, vb.pp("model.embed_tokens"))?;
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            layers.push(TransformerLayer::new(&config, vb.pp(format!("model.layers.{}", i)))?);
        }

        let norm = RmsNorm::new(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;
        let score_weight = vb.get((config.num_labels(), config.hidden_size), "score.weight")?.transpose(0, 1)?;
        let score_bias = vb.get(config.num_labels(), "score.bias")?;

        let viterbi_config = if viterbi_path.exists() {
            ViterbiConfig::from_file(&viterbi_path, "default").unwrap_or_default()
        } else {
            ViterbiConfig::default()
        };

        Ok(Self { embed_tokens, layers, norm, score_weight, score_bias, tokenizer, config, viterbi_config, device: device.clone() })
    }

    pub fn forward(&self, input_ids: &[u32]) -> Result<Tensor> {
        let seq_len = input_ids.len();
        let input_tensor = Tensor::new(input_ids, &self.device)?;
        let mut hidden_states = self.embed_tokens.forward(&input_tensor)?.unsqueeze(0)?;
        
        let rp = &self.config.rope_parameters;
        let rope = RotaryEmbedding::new_yarn(self.config.head_dim, self.config.max_position_embeddings, rp.rope_theta, rp.factor, rp.beta_fast, rp.beta_slow, rp.original_max_position_embeddings, rp.truncate, &self.device)?;
        let mask = create_sliding_window_mask(seq_len, self.config.sliding_window, &self.device)?;

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &rope, &mask)?;
        }

        hidden_states = self.norm.forward(&hidden_states)?;
        let logits = hidden_states.matmul(&self.score_weight.unsqueeze(0)?)?;
        let logits = logits.broadcast_add(&self.score_bias.unsqueeze(0)?.unsqueeze(0)?)?;
        Ok(logits)
    }

    pub fn predict(&self, text: &str) -> Result<Vec<PrivacySpan>> {
        let tokens = self.tokenizer.encode(text, false).map_err(anyhow::Error::msg)?;
        let input_ids = tokens.get_ids();
        let logits = self.forward(input_ids)?;
        let num_labels = self.config.num_labels();
        let logits_data: Vec<f32> = logits.flatten_all()?.to_vec1()?;
        
        let label_path = viterbi::viterbi_decode(&logits_data, input_ids.len(), num_labels, &self.viterbi_config);
        
        let label_list = self.config.id2label.as_ref().map(|m| {
            let mut l = vec![String::new(); num_labels];
            for (id, name) in m {
                let idx: usize = id.parse().unwrap();
                l[idx] = name.clone();
            }
            l
        }).unwrap_or_else(crate::privacy_filter::config::build_label_list);

        let spans = viterbi::extract_spans(
            &label_path,
            &logits_data,
            num_labels,
            &label_list,
            &tokens.get_tokens().iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            tokens.get_offsets(),
            text
        );

        Ok(spans)
    }
}
