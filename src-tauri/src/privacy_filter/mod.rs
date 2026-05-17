pub mod config;
pub mod viterbi;

use candle_core::{DType, Device, Module, Tensor, IndexOp};
use candle_nn::{linear, VarBuilder, Embedding};
use anyhow::Result;
use std::path::Path;
use tokenizers::Tokenizer;
use self::config::{ModelConfig, ViterbiConfig};
use self::viterbi::PrivacySpan;
use candle_transformers::models::deepseek2::TopKLastDimOp;

const ALPHA: f32 = 1.702;
const LIMIT: f32 = 7.0;

pub struct PrivacyFilterModel {
    embed_tokens: Embedding,
    layers: Vec<TransformerLayer>,
    norm: RmsNorm,
    score_weight: Tensor,
    score_bias: Tensor,
    pub tokenizer: Tokenizer,
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

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.powf(2.0)?.mean_keepdim(candle_core::D::Minus1)?;
        let x_normed = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)
    }
}

struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new_yarn(cfg: &config::RopeParameters, head_dim: usize, max_seq_len: usize, device: &Device) -> Result<Self> {
        let dim = head_dim;
        let half_dim = dim / 2;
        let theta = cfg.rope_theta;
        let factor = cfg.factor;

        let attention_scaling = if factor <= 1.0 { 1.0 } else { 0.1 * factor.ln() + 1.0 };

        let mut inv_freq_extrapolation = vec![0f32; half_dim];
        let mut inv_freq_interpolation = vec![0f32; half_dim];
        for i in 0..half_dim {
            let freq = theta.powf(2.0 * i as f64 / dim as f64);
            inv_freq_extrapolation[i] = 1.0 / freq as f32;
            inv_freq_interpolation[i] = 1.0 / (factor * freq) as f32;
        }

        let find_correction_dim = |num_rotations: f64| -> f64 {
            (dim as f64 * (cfg.original_max_position_embeddings as f64 / (num_rotations * 2.0 * std::f64::consts::PI)).ln()) / (2.0 * theta.ln())
        };

        let low_raw = find_correction_dim(cfg.beta_fast);
        let high_raw = find_correction_dim(cfg.beta_slow);

        let (low, high) = if cfg.truncate {
            (low_raw.floor(), high_raw.ceil())
        } else {
            (low_raw, high_raw)
        };
        let low = low.max(0.0).min((dim - 1) as f64);
        let high = high.max(0.0).min((dim - 1) as f64);

        let mut inv_freq = vec![0f32; half_dim];
        for i in 0..half_dim {
            let ramp = if (high - low).abs() < 1e-9 { 1.0 } else { ((i as f64 - low) / (high - low)).clamp(0.0, 1.0) as f32 };
            inv_freq[i] = inv_freq_interpolation[i] * ramp + inv_freq_extrapolation[i] * (1.0 - ramp);
        }

        let mut cos_data = vec![0f32; max_seq_len * half_dim];
        let mut sin_data = vec![0f32; max_seq_len * half_dim];
        let scale = attention_scaling as f32;
        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let angle = pos as f32 * inv_freq[i];
                cos_data[pos * half_dim + i] = angle.cos() * scale;
                sin_data[pos * half_dim + i] = angle.sin() * scale;
            }
        }

        let cos = Tensor::from_vec(cos_data, (max_seq_len, half_dim), device)?;
        let sin = Tensor::from_vec(sin_data, (max_seq_len, half_dim), device)?;
        Ok(Self { cos, sin })
    }

    fn forward(&self, x: &Tensor, seq_len: usize) -> candle_core::Result<Tensor> {
        let (b, h, s, d) = x.dims4()?;
        let cos = self.cos.narrow(0, 0, s)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = self.sin.narrow(0, 0, s)?.unsqueeze(0)?.unsqueeze(0)?;

        let x_pairs = x.reshape((b, h, s, d / 2, 2))?;
        let x0 = x_pairs.narrow(candle_core::D::Minus1, 0, 1)?.reshape((b, h, s, d / 2))?;
        let x1 = x_pairs.narrow(candle_core::D::Minus1, 1, 1)?.reshape((b, h, s, d / 2))?;

        let r0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
        let r1 = (x1.broadcast_mul(&cos)? + x0.broadcast_mul(&sin)?)?;

        let interleaved = Tensor::cat(&[r0.unsqueeze(candle_core::D::Minus1)?, r1.unsqueeze(candle_core::D::Minus1)?], candle_core::D::Minus1)?;
        interleaved.reshape((b, h, s, d))
    }
}

struct SparseMoE {
    router_weight: Tensor,
    router_bias: Tensor,
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
        let router_weight = vb.get((cfg.num_local_experts, cfg.hidden_size), "router.weight")?.transpose(0, 1)?;
        let router_bias = vb.get(cfg.num_local_experts, "router.bias")?;
        let gate_up_proj = vb.get((cfg.num_local_experts, cfg.hidden_size, 2 * cfg.intermediate_size), "experts.gate_up_proj")?;
        let gate_up_proj_bias = vb.get((cfg.num_local_experts, 2 * cfg.intermediate_size), "experts.gate_up_proj_bias")?;
        let down_proj = vb.get((cfg.num_local_experts, cfg.intermediate_size, cfg.hidden_size), "experts.down_proj")?;
        let down_proj_bias = vb.get((cfg.num_local_experts, cfg.hidden_size), "experts.down_proj_bias")?;

        Ok(Self {
            router_weight, router_bias, gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias,
            num_experts: cfg.num_local_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            intermediate_size: cfg.intermediate_size,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, s, h) = x.dims3()?;
        let x_flat = x.reshape((b * s, h))?;
        let router_logits = (x_flat.matmul(&self.router_weight)?.broadcast_add(&self.router_bias))?;
        
        let top_k = router_logits.topk(self.num_experts_per_tok)?;
        let routing_weights = candle_nn::ops::softmax(&top_k.values, candle_core::D::Minus1)?;
        
        let indices_vec = top_k.indices.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u32>()?;
        let weights_vec = routing_weights.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        
        let mut final_out = Tensor::zeros((b * s, h), x.dtype(), x.device())?;
        
        let mut expert_mask = vec![Vec::new(); self.num_experts];
        for i in 0..(b * s) {
            for k in 0..self.num_experts_per_tok {
                let expert_idx = indices_vec[i * self.num_experts_per_tok + k] as usize;
                let weight = weights_vec[i * self.num_experts_per_tok + k] as f64;
                expert_mask[expert_idx].push((i, weight));
            }
        }

        for (eidx, tokens) in expert_mask.into_iter().enumerate() {
            if tokens.is_empty() { continue; }
            
            let n = tokens.len();
            let mut expert_input_vec = Vec::with_capacity(n);
            for &(tidx, _) in &tokens {
                expert_input_vec.push(x_flat.get(tidx)?);
            }
            let expert_input = Tensor::stack(&expert_input_vec, 0)?;

            let gu_w = self.gate_up_proj.get(eidx)?;
            let gu_b = self.gate_up_proj_bias.get(eidx)?;
            let gate_up = (expert_input.matmul(&gu_w)?.broadcast_add(&gu_b))?;
            
            let gate = gate_up.narrow(candle_core::D::Minus1, 0, self.intermediate_size)?.minimum(LIMIT as f64)?;
            let up = gate_up.narrow(candle_core::D::Minus1, self.intermediate_size, self.intermediate_size)?.clamp(-(LIMIT as f64), LIMIT as f64)?;
            let glu = (gate.clone() * candle_nn::ops::sigmoid(&(gate * ALPHA as f64)?)?)?;
            let expert_out_mid = (up.affine(1.0, 1.0)?.mul(&glu))?;
            
            let dp_w = self.down_proj.get(eidx)?;
            let dp_b = self.down_proj_bias.get(eidx)?;
            let expert_output = (expert_out_mid.matmul(&dp_w)?.broadcast_add(&dp_b))?;

            for (i, &(tidx, weight)) in tokens.iter().enumerate() {
                let weighted = (expert_output.get(i)?.affine(weight, 0.0))?;
                let current = final_out.get(tidx)?;
                final_out = final_out.slice_assign(&[tidx..tidx+1, 0..h], &(current + weighted)?.unsqueeze(0)?)?;
            }
        }
        
        final_out.affine(self.num_experts_per_tok as f64, 0.0)?.reshape((b, s, h))
    }
}

pub struct UnifiedKvCache {
    pub k: Tensor,
    pub v: Tensor,
    pub current_pos: usize,
}

impl UnifiedKvCache {
    pub fn new(num_layers: usize, num_kv_heads: usize, head_dim: usize, max_seq_len: usize, device: &Device) -> candle_core::Result<Self> {
        let k = Tensor::zeros((num_layers, 1, num_kv_heads, max_seq_len, head_dim), DType::F32, device)?;
        let v = Tensor::zeros((num_layers, 1, num_kv_heads, max_seq_len, head_dim), DType::F32, device)?;
        Ok(Self { k, v, current_pos: 0 })
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
    scaling: f32,
    layer_idx: usize,
}

impl TransformerLayer {
    fn new(cfg: &ModelConfig, vb: VarBuilder, layer_idx: usize) -> Result<Self> {
        let q_proj = linear(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim, vb.pp("self_attn.q_proj"))?;
        let k_proj = linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, vb.pp("self_attn.k_proj"))?;
        let v_proj = linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, vb.pp("self_attn.v_proj"))?;
        let o_proj = linear(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, vb.pp("self_attn.o_proj"))?;
        let sinks = vb.get(cfg.num_attention_heads, "self_attn.sinks")?;
        let input_layernorm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;
        let moe = SparseMoE::new(cfg, vb.pp("mlp"))?;
        
        Ok(Self {
            q_proj, k_proj, v_proj, o_proj, sinks, input_layernorm, post_attention_layernorm, moe,
            num_heads: cfg.num_attention_heads, num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim, scaling: (cfg.head_dim as f32).powf(-0.25),
            layer_idx,
        })
    }

    fn forward(&self, x: &Tensor, rope: &RotaryEmbedding, mask: &Tensor, cache: &mut Option<&mut UnifiedKvCache>) -> candle_core::Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let x_norm = self.input_layernorm.forward(x)?;
        
        let q = self.q_proj.forward(&x_norm)?.reshape((b, s, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let mut k = self.k_proj.forward(&x_norm)?.reshape((b, s, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let mut v = self.v_proj.forward(&x_norm)?.reshape((b, s, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        
        let q = rope.forward(&q, s)?;
        k = rope.forward(&k, s)?;

        if let Some(cache) = cache {
            let pos = cache.current_pos;
            cache.k.slice_assign(&[self.layer_idx..self.layer_idx+1, 0..b, 0..self.num_kv_heads, pos..pos+s, 0..self.head_dim], &k)?;
            cache.v.slice_assign(&[self.layer_idx..self.layer_idx+1, 0..b, 0..self.num_kv_heads, pos..pos+s, 0..self.head_dim], &v)?;
            k = cache.k.get(self.layer_idx)?.narrow(2, 0, pos + s)?;
            v = cache.v.get(self.layer_idx)?.narrow(2, 0, pos + s)?;
        }

        let q = q.affine(self.scaling as f64, 0.0)?;
        let k = k.affine(self.scaling as f64, 0.0)?;
        
        let n_rep = self.num_heads / self.num_kv_heads;
        let k = if n_rep > 1 { crate::privacy_filter::repeat_kv(&k, n_rep)? } else { k };
        let v = if n_rep > 1 { crate::privacy_filter::repeat_kv(&v, n_rep)? } else { v };

        let attn_weights = (q.matmul(&k.transpose(2, 3)?)?.broadcast_add(mask))?;
        let sinks = self.sinks.reshape((1, self.num_heads, 1, 1))?.expand((b, self.num_heads, s, 1))?;
        let combined = Tensor::cat(&[attn_weights, sinks], 3)?;
        
        let max_vals = combined.max_keepdim(3)?;
        let combined = combined.broadcast_sub(&max_vals)?;
        
        let probs = candle_nn::ops::softmax(&combined, 3)?;
        let scores = probs.narrow(3, 0, s)?.contiguous()?;
        
        let attn_out = scores.matmul(&v.contiguous()?)?.transpose(1, 2)?.reshape((b, s, ()))?;
        let x_attn = (x + self.o_proj.forward(&attn_out)?)?;
        let x_norm = self.post_attention_layernorm.forward(&x_attn)?;
        x_attn + self.moe.forward(&x_norm)?
    }
}

pub fn repeat_kv(x: &Tensor, num_repeats: usize) -> candle_core::Result<Tensor> {
    if num_repeats == 1 { return Ok(x.clone()); }
    let (b, n_kv, s, d) = x.dims4()?;
    x.unsqueeze(2)?.expand((b, n_kv, num_repeats, s, d))?.flatten(1, 2)
}

pub fn create_sliding_window_mask(seq_len: usize, window_size: usize, device: &Device) -> candle_core::Result<Tensor> {
    let mut mask_data = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if (i as isize - j as isize).abs() > window_size as isize {
                mask_data[i * seq_len + j] = -1e9;
            }
        }
    }
    Tensor::from_vec(mask_data, (1, 1, seq_len, seq_len), device)
}

impl PrivacyFilterModel {
    pub fn load(model_dir: &Path, device: &Device) -> Result<Self> {
        let config = ModelConfig::from_file(&model_dir.join("config.json"))?;
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_dir.join("model.safetensors")], DType::F32, device)? };

        let embed_tokens = candle_nn::embedding(config.vocab_size, config.hidden_size, vb.pp("model.embed_tokens"))?;
        let layers = (0..config.num_hidden_layers).map(|i| TransformerLayer::new(&config, vb.pp(format!("model.layers.{}", i)), i)).collect::<Result<Vec<_>>>()?;
        let norm = RmsNorm::new(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;
        let score_weight = vb.get((config.num_labels(), config.hidden_size), "score.weight")?.transpose(0, 1)?;
        let score_bias = vb.get(config.num_labels(), "score.bias")?;
        let viterbi_config = ViterbiConfig::from_file(&model_dir.join("viterbi_calibration.json"), "default").unwrap_or_default();

        Ok(Self { embed_tokens, layers, norm, score_weight, score_bias, tokenizer, config, viterbi_config, device: device.clone() })
    }

    pub fn forward(&self, input_ids: &[u32], cache: &mut Option<&mut UnifiedKvCache>) -> candle_core::Result<Tensor> {
        let seq_len = input_ids.len();
        let mut x = self.embed_tokens.forward(&Tensor::new(input_ids, &self.device)?)?.unsqueeze(0)?;
        
        let rope = RotaryEmbedding::new_yarn(&self.config.rope_parameters, self.config.head_dim, self.config.max_position_embeddings, &self.device).map_err(candle_core::Error::msg)?;
        let mask = create_sliding_window_mask(seq_len, self.config.sliding_window, &self.device)?;

        for layer in &self.layers {
            x = layer.forward(&x, &rope, &mask, cache)?;
        }

        let x = self.norm.forward(&x)?;
        x.matmul(&self.score_weight.unsqueeze(0)?)?.broadcast_add(&self.score_bias.unsqueeze(0)?.unsqueeze(0)?)
    }

    pub fn predict(&self, text: &str) -> Result<Vec<PrivacySpan>> {
        let tokens = self.tokenizer.encode(text, false).map_err(anyhow::Error::msg)?;
        let input_ids = tokens.get_ids();
        let s = input_ids.len();
        let logits = self.forward(input_ids, &mut None).map_err(anyhow::Error::msg)?;
        let logits_data = logits.flatten_all()?.to_vec1::<f32>().map_err(anyhow::Error::msg)?;
        
        let num_labels = self.config.num_labels();
        let label_list = self.config.id2label.as_ref().map(|m| {
            let mut l = vec![String::new(); num_labels];
            for (id, name) in m { l[id.parse::<usize>().unwrap()] = name.clone(); }
            l
        }).unwrap_or_else(crate::privacy_filter::config::build_label_list);

        Ok(viterbi::extract_spans(&viterbi::viterbi_decode(&logits_data, s, num_labels, &self.viterbi_config), &logits_data, num_labels, &label_list, &tokens.get_tokens().iter().map(|s| s.to_string()).collect::<Vec<_>>(), tokens.get_offsets(), text))
    }
}
