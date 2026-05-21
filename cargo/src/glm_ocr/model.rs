//! GLM-OCR Model Implementation

use anyhow::Result;
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{
    Activation, Embedding, LayerNorm, Linear, Module, VarBuilder,
    embedding, layer_norm, linear, linear_no_bias,
};

use crate::{
    models::common::{InferenceModel, modules::GateUpDownMLP},
    glm_ocr::config::{GlmOcrConfig, GlmOcrTextConfig, GlmOcrVisionConfig},
    position_embed::rope::{apply_rotary_pos_emb_vision, glm_ocr_apply_rotary_pos_emb},
    utils::tensor_utils::{prepare_causal_attention_mask, repeat_kv},
};

// 🚀 BF16/F16 환경에서 분산(Variance) 계산 시 오버플로우로 인한 NaN 발생을 원천 차단하기 위해
// candle_nn::RmsNorm 래퍼 대신 가중치를 직접 보유하고 F32로 안전하게 캐스팅하여 연산하는 구조로 변경합니다.
pub struct GlmOcrRMSNorm {
    weight: Tensor,
    eps: f64,
}

impl GlmOcrRMSNorm {
    pub fn new(vb: VarBuilder, hidden_size: usize, eps: f64) -> Result<Self> {
        let weight = vb.get(hidden_size, "weight")?;
        Ok(Self { weight, eps })
    }
    pub fn new_on_device(vb: VarBuilder, hidden_size: usize, eps: f64, device: &Device) -> Result<Self> {
        let weight = vb.get(hidden_size, "weight")?.to_device(device)?;
        Ok(Self { weight, eps })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x_dtype = xs.dtype();
        let x_f32 = xs.to_dtype(DType::F32)?;
        let variance = x_f32.powf(2.0)?.mean_keepdim(candle_core::D::Minus1)?;
        let x_normed = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        Ok(x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)?)
    }
    pub fn extra_repr(&self) -> String {
        "GlmOcrRMSNorm".to_string()
    }
}

pub struct GlmOcrVisionMlp(GateUpDownMLP);

impl GlmOcrVisionMlp {
    pub fn new(vb: VarBuilder, config: &GlmOcrVisionConfig) -> Result<Self> {
        let mlp = GateUpDownMLP::new(
            vb,
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            config.attention_bias,
            Some("gate_proj"),
            Some("up_proj"),
            Some("down_proj"),
        )?;
        Ok(Self(mlp))
    }

    pub fn forward(&self, hidden_state: &Tensor) -> Result<Tensor> {
        Ok(self.0.forward(hidden_state)?)
    }
}

fn eager_attention_forward(
    query_states: &Tensor,
    key_states: &Tensor,
    value_states: &Tensor,
    num_key_value_groups: Option<usize>,
    attention_mask: Option<&Tensor>,
    scaling: f64,
    dropout: f64,
) -> Result<(Tensor, Tensor)> {
    let key_states = match num_key_value_groups {
        Some(g) => repeat_kv(key_states.clone(), g)?.contiguous()?,
        None => key_states.clone(),
    };
    let value_states = match num_key_value_groups {
        Some(g) => repeat_kv(value_states.clone(), g)?.contiguous()?,
        None => value_states.clone(),
    };
    let query_states = query_states.contiguous()?;
    let key_states = key_states.contiguous()?;
    let value_states = value_states.contiguous()?;

    let output = {
        #[cfg(feature = "flash-attn")]
        {
            // Flash attention: causal iff attention_mask is present.
            // Explicit contiguous() ensures proper memory layout for flash_attn kernel
            let q = query_states.transpose(1, 2)?.contiguous()?;
            let k = key_states.transpose(1, 2)?.contiguous()?;
            let v = value_states.transpose(1, 2)?.contiguous()?;
            candle_flash_attn::flash_attn(&q, &k, &v, scaling as f32, attention_mask.is_some())?
            // flash_attn returns [batch, q_len, heads, head_dim] — already in final layout
        }
        #[cfg(not(feature = "flash-attn"))]
        {
            // Chunked Q-attention: process Q in blocks so the attention matrix
            // [batch, heads, CHUNK, k_len] stays bounded regardless of q_len.
            // Peak memory per chunk: CHUNK × k_len × heads × 4 bytes (f32 softmax).
            // Mathematically equivalent to full attention.
            // CHUNK_SIZE=512 is empirically optimal for most hardware (CPU/GPU balance)
            const CHUNK_SIZE: usize = 512;
            let q_len = query_states.dim(2)?;
            let k_t = key_states.transpose(D::Minus2, D::Minus1)?.contiguous()?;

            let raw = if q_len > CHUNK_SIZE {
                let mut chunks: Vec<Tensor> =
                    Vec::with_capacity((q_len + CHUNK_SIZE - 1) / CHUNK_SIZE);
                let mut start = 0;
                while start < q_len {
                    let len = CHUNK_SIZE.min(q_len - start);
                    let q_chunk = query_states.narrow(2, start, len)?;
                    let attn = (q_chunk.matmul(&k_t)? * scaling)?;
                    let attn = match attention_mask {
                        None => attn,
                        Some(mask) => attn
                            .broadcast_add(&mask.narrow(2, start, len)?.to_dtype(attn.dtype())?)?,
                    };
                    // Softmax computation: Optimize dtype conversions for CPU (which uses F32)
                    let attn = if query_states.dtype() == DType::F32 {
                        candle_nn::ops::softmax_last_dim(&attn)?
                    } else {
                        candle_nn::ops::softmax_last_dim(&attn.to_dtype(DType::F32)?)?
                            .to_dtype(query_states.dtype())?
                    };
                    // Apply dropout uniformly across chunked and non-chunked paths for consistency
                    let attn = candle_nn::ops::dropout(&attn, dropout as f32)?;
                    chunks.push(attn.matmul(&value_states)?);
                    start += len;
                }
                Tensor::cat(&chunks, 2)? // [batch, heads, q_len, head_dim]
            } else {
                let attn = (query_states.matmul(&k_t)? * scaling)?;
                let attn = match attention_mask {
                    None => attn,
                    Some(mask) => attn.broadcast_add(&mask.to_dtype(attn.dtype())?)?,
                };
                // Softmax computation: Same optimization as chunked path
                let attn = if query_states.dtype() == DType::F32 {
                    candle_nn::ops::softmax_last_dim(&attn)?
                } else {
                    candle_nn::ops::softmax_last_dim(&attn.to_dtype(DType::F32)?)?
                        .to_dtype(query_states.dtype())?
                };
                // Apply dropout uniformly (now consistent across both paths)
                let attn = candle_nn::ops::dropout(&attn, dropout as f32)?;
                attn.matmul(&value_states)?
            };
            // [batch, heads, q_len, head_dim] -> [batch, q_len, heads, head_dim]
            raw.transpose(1, 2)?.contiguous()?
        }
    };

    // output layout: [batch, q_len, heads, head_dim]
    let placeholder = Tensor::zeros((0,), query_states.dtype(), query_states.device())?;
    Ok((output, placeholder))
}

pub struct GlmOcrTextAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f64,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl GlmOcrTextAttention {
    pub fn new(
        vb: VarBuilder,
        config: &GlmOcrTextConfig,
        _layer_idx: Option<usize>,
    ) -> Result<Self> {
        let head_dim = config.head_dim.unwrap_or_else(|| {
            // Integer division, panics if num_attention_heads is 0 (like Python)
            config.hidden_size / config.num_attention_heads
        });
        let num_kv_groups = config.num_attention_heads / config.num_key_value_heads;

        let scaling = 1.0 / (head_dim as f64).sqrt();

        let q_proj = linear_no_bias(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_no_bias(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_no_bias(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_no_bias(
            config.num_attention_heads * head_dim,
            config.hidden_size,
            vb.pp("o_proj"),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            num_kv_groups,
            head_dim,
            scaling,
            kv_cache: None,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (bs, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((bs, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((bs, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((bs, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (cos, sin) = position_embeddings;
        let (query_states, key_states) =
            glm_ocr_apply_rotary_pos_emb(&query_states, &key_states, cos, sin)?;

        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        // Python: dropout=0.0 if not self.training else self.attention_dropout
        // Rust is inference-only, so always 0.0
        let (attn_output, attn_weights) = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_kv_groups),
            attention_mask,
            self.scaling,
            0.0,
        )?;

        let attn_output = attn_output.reshape((bs, q_len, ()))?;
        Ok((self.o_proj.forward(&attn_output)?, attn_weights))
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

pub struct GlmOcrVisionRotaryEmbedding {
    inv_freq: Tensor,
}

impl GlmOcrVisionRotaryEmbedding {
    pub fn new(dim: usize, theta: f32, device: &candle_core::Device, dtype: DType) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (dim / 2,), device)?.to_dtype(dtype)?;
        Ok(Self { inv_freq })
    }

    pub fn forward(&self, seqlen: usize) -> Result<Tensor> {
        // Python: freqs = torch.outer(seq, self.inv_freq) -> (seqlen, dim/4)
        let target_dtype = self.inv_freq.dtype();
        let seq = Tensor::arange(0f32, seqlen as f32, self.inv_freq.device())?;
        // 🚀 CPU BF16 matmul 에러 원천 차단
        let freqs = seq.unsqueeze(1)?.matmul(&self.inv_freq.to_dtype(DType::F32)?.unsqueeze(0)?)?;
        Ok(freqs.to_dtype(target_dtype)?)
    }

    pub fn rot_pos_emb(
        &self,
        grid_thw: &[(usize, usize, usize)],
        spatial_merge_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        let sms = spatial_merge_size;
        let mut all_hpos: Vec<u32> = Vec::new();
        let mut all_wpos: Vec<u32> = Vec::new();
        let mut max_grid_size: usize = 0;

        for &(t, h, w) in grid_thw {
            max_grid_size = max_grid_size.max(h).max(w);

            for _ in 0..t {
                for hi in 0..h {
                    for wi in 0..w {
                        // Apply spatial merge rearrangement
                        let _hb = hi / sms;
                        let _si = hi % sms;
                        let _wb = wi / sms;
                        let _sj = wi % sms;

                        // After permute(0,2,1,3): position = (hb, wb, si, sj)
                        // Flatten: idx = hb * w_blocks * sms * sms + wb * sms * sms + si * sms + sj
                        // But we just need the h and w positions for rotary embedding
                        all_hpos.push(hi as u32);
                        all_wpos.push(wi as u32);
                    }
                }
            }
        }

        let total_seq = all_hpos.len();
        let freqs_full = self.forward(max_grid_size)?; // (max_grid_size, dim/4)

        let h_indices = Tensor::from_vec(all_hpos, (total_seq,), self.inv_freq.device())?;
        let w_indices = Tensor::from_vec(all_wpos, (total_seq,), self.inv_freq.device())?;
        let h_freqs = freqs_full.index_select(&h_indices, 0)?; // (total_seq, dim/4)
        let w_freqs = freqs_full.index_select(&w_indices, 0)?; // (total_seq, dim/4)

        // Concatenate h and w freqs: (total_seq, dim/2)
        let rotary_pos_emb = Tensor::cat(&[&h_freqs, &w_freqs], 1)?;

        let emb = Tensor::cat(&[&rotary_pos_emb, &rotary_pos_emb], 1)?;
        let cos = emb.cos()?;
        let sin = emb.sin()?;

        Ok((cos, sin))
    }
}

pub struct GlmOcrTextMLP {
    gate_up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl GlmOcrTextMLP {
    pub fn new(vb: VarBuilder, config: &GlmOcrTextConfig) -> Result<Self> {
        let gate_up_proj = linear_no_bias(
            config.hidden_size,
            2 * config.intermediate_size,
            vb.pp("gate_up_proj"),
        )?;
        let down_proj = linear_no_bias(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("down_proj"),
        )?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            act_fn: config.hidden_act,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let up_states = self.gate_up_proj.forward(xs)?;
        let dim = up_states.dims().len() - 1;
        let chunks = up_states.chunk(2, dim)?;
        let gate = &chunks[0];
        let up = &chunks[1];
        let up_states = up.broadcast_mul(&self.act_fn.forward(gate)?)?;
        Ok(self.down_proj.forward(&up_states)?)
    }
}

pub struct GlmOcrTextDecoderLayer {
    self_attn: GlmOcrTextAttention,
    mlp: GlmOcrTextMLP,
    input_layernorm: GlmOcrRMSNorm,
    post_attention_layernorm: GlmOcrRMSNorm,
    post_self_attn_layernorm: GlmOcrRMSNorm,
    post_mlp_layernorm: GlmOcrRMSNorm,
}

impl GlmOcrTextDecoderLayer {
    pub fn new(vb: VarBuilder, config: &GlmOcrTextConfig, layer_idx: usize) -> Result<Self> {
        let self_attn = GlmOcrTextAttention::new(vb.pp("self_attn"), config, Some(layer_idx))?;
        let mlp = GlmOcrTextMLP::new(vb.pp("mlp"), config)?;
        let input_layernorm = GlmOcrRMSNorm::new(vb.pp("input_layernorm"), config.hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm = GlmOcrRMSNorm::new(vb.pp("post_attention_layernorm"), config.hidden_size, config.rms_norm_eps)?;
        let post_self_attn_layernorm = GlmOcrRMSNorm::new(vb.pp("post_self_attn_layernorm"), config.hidden_size, config.rms_norm_eps)?;
        let post_mlp_layernorm = GlmOcrRMSNorm::new(vb.pp("post_mlp_layernorm"), config.hidden_size, config.rms_norm_eps)?;
        Ok(Self { self_attn, mlp, input_layernorm, post_attention_layernorm, post_self_attn_layernorm, post_mlp_layernorm })
    }

    pub fn new_skeleton(config: &GlmOcrTextConfig, device: &Device) -> Result<Self> {
        let dummy = Tensor::zeros((1, 1), DType::F32, device)?;
        let dummy_1d = Tensor::zeros((1,), DType::F32, device)?;
        let head_dim = config.head_dim.unwrap_or(config.hidden_size / config.num_attention_heads);
        let scaling = 1.0 / (head_dim as f64).sqrt();
        let self_attn = GlmOcrTextAttention {
            q_proj: Linear::new(dummy.clone(), None),
            k_proj: Linear::new(dummy.clone(), None),
            v_proj: Linear::new(dummy.clone(), None),
            o_proj: Linear::new(dummy.clone(), None),
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            num_kv_groups: config.num_attention_heads / config.num_key_value_heads,
            head_dim, scaling, kv_cache: None,
        };
        let mlp = GlmOcrTextMLP {
            gate_up_proj: Linear::new(dummy.clone(), None),
            down_proj: Linear::new(dummy.clone(), None),
            act_fn: config.hidden_act,
        };
        let input_layernorm = GlmOcrRMSNorm { weight: dummy_1d.clone(), eps: config.rms_norm_eps };
        let post_attention_layernorm = GlmOcrRMSNorm { weight: dummy_1d.clone(), eps: config.rms_norm_eps };
        let post_self_attn_layernorm = GlmOcrRMSNorm { weight: dummy_1d.clone(), eps: config.rms_norm_eps };
        let post_mlp_layernorm = GlmOcrRMSNorm { weight: dummy_1d.clone(), eps: config.rms_norm_eps };
        Ok(Self { self_attn, mlp, input_layernorm, post_attention_layernorm, post_self_attn_layernorm, post_mlp_layernorm })
    }

    pub fn clear_weights(&mut self) {
        let dummy = Tensor::zeros((1, 1), DType::F32, &Device::Cpu).unwrap();
        let dummy_1d = Tensor::zeros((1,), DType::F32, &Device::Cpu).unwrap();
        self.self_attn.q_proj = Linear::new(dummy.clone(), None);
        self.self_attn.k_proj = Linear::new(dummy.clone(), None);
        self.self_attn.v_proj = Linear::new(dummy.clone(), None);
        self.self_attn.o_proj = Linear::new(dummy.clone(), None);
        self.mlp.gate_up_proj = Linear::new(dummy.clone(), None);
        self.mlp.down_proj = Linear::new(dummy.clone(), None);
        self.input_layernorm.weight = dummy_1d.clone();
        self.post_attention_layernorm.weight = dummy_1d.clone();
        self.post_self_attn_layernorm.weight = dummy_1d.clone();
        self.post_mlp_layernorm.weight = dummy_1d.clone();
    }

    pub fn is_cleared(&self) -> bool {
        self.self_attn.q_proj.weight().elem_count() <= 1
    }

    pub fn load_weights_inplace<R: std::io::Read + std::io::Seek>(&mut self, ct: &candle_core::quantized::gguf_file::Content, reader: &mut R, prefix: &str, device: &Device, dtype: DType) -> Result<()> {
        // 클로저 충돌을 피하기 위해 텐서 로드 유틸리티 함수를 인라인으로 구현하거나 헬퍼 메서드화 합니다.
        let load_lin = |r: &mut R, name: &str| -> Result<Linear> {
            let w = ct.tensor(r, &format!("{}.weight", name), device)?;
            let w = w.dequantize_f16(device).or_else(|_| w.dequantize(device))?.to_dtype(dtype)?;
            Ok(Linear::new(w, None))
        };

        let load_norm = |r: &mut R, name: &str| -> Result<GlmOcrRMSNorm> {
            let w = ct.tensor(r, &format!("{}.weight", name), device)?;
            let w = w.dequantize_f16(device).or_else(|_| w.dequantize(device))?.to_dtype(dtype)?;
            Ok(GlmOcrRMSNorm { weight: w, eps: 1e-5 })
        };

        self.self_attn.q_proj = load_lin(reader, &format!("{}attn_q", prefix))?;
        self.self_attn.k_proj = load_lin(reader, &format!("{}attn_k", prefix))?;
        self.self_attn.v_proj = load_lin(reader, &format!("{}attn_v", prefix))?;
        self.self_attn.o_proj = load_lin(reader, &format!("{}attn_output", prefix))?;

        let gate_name = format!("{}ffn_gate.weight", prefix);
        let up_name = format!("{}ffn_up.weight", prefix);
        
        // 🚀 ffn_gate 텐서가 독립적으로 존재하는지 확인하고, 없다면 ffn_up에 병합되어 있다고 간주합니다.
        let gate_up = if ct.tensor_infos.contains_key(&gate_name) {
            let t_gate = ct.tensor(reader, &gate_name, device)?;
            let gate = t_gate.dequantize_f16(device).or_else(|_| t_gate.dequantize(device))?.to_dtype(dtype)?;
            let t_up = ct.tensor(reader, &up_name, device)?;
            let up = t_up.dequantize_f16(device).or_else(|_| t_up.dequantize(device))?.to_dtype(dtype)?;
            Tensor::cat(&[&gate, &up], 0)?
        } else {
            let t_up = ct.tensor(reader, &up_name, device)?;
            t_up.dequantize_f16(device).or_else(|_| t_up.dequantize(device))?.to_dtype(dtype)?
        };
        
        self.mlp.gate_up_proj = Linear::new(gate_up, None);
        self.mlp.down_proj = load_lin(reader, &format!("{}ffn_down", prefix))?;

        self.input_layernorm = load_norm(reader, &format!("{}attn_norm", prefix))?;
        self.post_attention_layernorm = load_norm(reader, &format!("{}ffn_norm", prefix))?;
        self.post_self_attn_layernorm = load_norm(reader, &format!("{}post_attention_norm", prefix))?;
        self.post_mlp_layernorm = load_norm(reader, &format!("{}post_ffw_norm", prefix))?;
        Ok(())
    }

    pub fn forward(&mut self, xs: &Tensor, position_embeddings: (&Tensor, &Tensor), attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.input_layernorm.forward(xs)?;
        let (xs, _attn_weights) = self.self_attn.forward(&xs, position_embeddings, attention_mask)?;
        let xs = self.post_self_attn_layernorm.forward(&xs)?;
        let xs = residual.add(&xs)?;

        let residual = xs.clone();
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = self.post_mlp_layernorm.forward(&xs)?;
        Ok(xs.add(&residual)?)
    }

    pub fn clear_kv_cache(&mut self) { self.self_attn.clear_kv_cache(); }
}

pub struct GlmOcrVisionAttention {
    num_heads: usize,
    head_dim: usize,
    scaling: f64,
    qkv: Linear,
    proj: Linear,
    q_norm: GlmOcrRMSNorm,
    k_norm: GlmOcrRMSNorm,
}

impl GlmOcrVisionAttention {
    pub fn new(vb: VarBuilder, config: &GlmOcrVisionConfig) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_heads;
        let scaling = 1.0 / (head_dim as f64).sqrt();
        let qkv = linear(config.hidden_size, config.hidden_size * 3, vb.pp("qkv"))?;
        let proj = linear(config.hidden_size, config.hidden_size, vb.pp("proj"))?;
        let q_norm = GlmOcrRMSNorm::new(vb.pp("q_norm"), head_dim, config.rms_norm_eps)?;
        let k_norm = GlmOcrRMSNorm::new(vb.pp("k_norm"), head_dim, config.rms_norm_eps)?;

        Ok(Self { num_heads: config.num_heads, head_dim, scaling, qkv, proj, q_norm, k_norm })
    }

    pub fn forward_with_params(&self, xs: &Tensor, _cu_seqlens: &Tensor, _rotary_pos_emb: Option<&Tensor>, position_embeddings: Option<(&Tensor, &Tensor)>) -> Result<Tensor> {
        let (seq_len, _) = xs.dims2()?;
        let qkv = self.qkv.forward(xs)?;
        let qkv = qkv.reshape((seq_len, 3, self.num_heads, self.head_dim))?.permute((1, 0, 2, 3))?;

        let q = qkv.i(0)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let (cos, sin) = position_embeddings.unwrap();
        let (q, k) = apply_rotary_pos_emb_vision(&q, &k, cos, sin)?;

        let q = q.transpose(0, 1)?.unsqueeze(0)?;
        let k = k.transpose(0, 1)?.unsqueeze(0)?;
        let v = v.transpose(0, 1)?.unsqueeze(0)?;

        let (attn_output, _attn_weights) = eager_attention_forward(&q, &k, &v, Some(1), None, self.scaling, 0.0)?;
        let attn_output = attn_output.reshape((seq_len, ()))?;
        Ok(self.proj.forward(&attn_output)?)
    }
}

pub struct GlmOcrVisionBlock {
    norm1: GlmOcrRMSNorm,
    norm2: GlmOcrRMSNorm,
    attn: GlmOcrVisionAttention,
    mlp: GlmOcrVisionMlp,
}

impl GlmOcrVisionBlock {
    pub fn new(vb: VarBuilder, config: &GlmOcrVisionConfig) -> Result<Self> {
        let norm1 = GlmOcrRMSNorm::new(vb.pp("norm1"), config.hidden_size, config.rms_norm_eps)?;
        let attn = GlmOcrVisionAttention::new(vb.pp("attn"), config)?;
        let norm2 = GlmOcrRMSNorm::new(vb.pp("norm2"), config.hidden_size, config.rms_norm_eps)?;
        let mlp = GlmOcrVisionMlp::new(vb.pp("mlp"), config)?;
        Ok(Self { norm1, norm2, attn, mlp })
    }

    pub fn new_skeleton(config: &GlmOcrVisionConfig, device: &Device) -> Result<Self> {
        let dummy = Tensor::zeros((1, 1), DType::F32, device)?;
        let dummy_1d = Tensor::zeros((1,), DType::F32, device)?;
        let head_dim = config.hidden_size / config.num_heads;
        let scaling = 1.0 / (head_dim as f64).sqrt();
        let attn = GlmOcrVisionAttention {
            num_heads: config.num_heads, head_dim, scaling,
            qkv: Linear::new(dummy.clone(), None), proj: Linear::new(dummy.clone(), None),
            q_norm: GlmOcrRMSNorm { weight: dummy_1d.clone(), eps: config.rms_norm_eps },
            k_norm: GlmOcrRMSNorm { weight: dummy_1d.clone(), eps: config.rms_norm_eps },
        };
        let mlp = GlmOcrVisionMlp(GateUpDownMLP::new_dummy(device)); 
        Ok(Self {
            norm1: GlmOcrRMSNorm { weight: dummy_1d.clone(), eps: config.rms_norm_eps },
            norm2: GlmOcrRMSNorm { weight: dummy_1d.clone(), eps: config.rms_norm_eps },
            attn, mlp,
        })
    }
    
    pub fn clear_weights(&mut self) {
        let dummy = Tensor::zeros((1, 1), DType::F32, &Device::Cpu).unwrap();
        let dummy_1d = Tensor::zeros((1,), DType::F32, &Device::Cpu).unwrap();
        self.attn.qkv = Linear::new(dummy.clone(), None);
        self.attn.proj = Linear::new(dummy.clone(), None);
        self.attn.q_norm.weight = dummy_1d.clone();
        self.attn.k_norm.weight = dummy_1d.clone();
        self.mlp.0.clear_weights();
        self.norm1.weight = dummy_1d.clone();
        self.norm2.weight = dummy_1d.clone();
    }

    pub fn is_cleared(&self) -> bool { self.attn.qkv.weight().elem_count() <= 1 }

    pub fn load_weights_inplace<R: std::io::Read + std::io::Seek>(&mut self, ct: &candle_core::quantized::gguf_file::Content, reader: &mut R, prefix: &str, device: &Device, dtype: DType) -> Result<()> {
        // 클로저 충돌을 피하기 위해 헬퍼 클로저 대신 직접 로드 로직을 구현합니다.
        let load_lin_b = |r: &mut R, name: &str| -> Result<Linear> {
            let w_t = ct.tensor(r, &format!("{}.weight", name), device)?;
            let w = w_t.dequantize_f16(device).or_else(|_| w_t.dequantize(device))?.to_dtype(dtype)?;
            let b = if let Ok(b_t) = ct.tensor(r, &format!("{}.bias", name), device) {
                Some(b_t.dequantize_f16(device).or_else(|_| b_t.dequantize(device))?.to_dtype(dtype)?)
            } else { None };
            Ok(Linear::new(w, b))
        };

        let load_norm = |r: &mut R, name: &str| -> Result<GlmOcrRMSNorm> {
            let w_t = ct.tensor(r, &format!("{}.weight", name), device)?;
            let w = w_t.dequantize_f16(device).or_else(|_| w_t.dequantize(device))?.to_dtype(dtype)?;
            Ok(GlmOcrRMSNorm { weight: w, eps: 1e-5 })
        };

        self.attn.qkv = load_lin_b(reader, &format!("{}attn_qkv", prefix))?;
        self.attn.proj = load_lin_b(reader, &format!("{}attn_out", prefix))?;
        self.attn.q_norm = load_norm(reader, &format!("{}attn_q_norm", prefix))?;
        self.attn.k_norm = load_norm(reader, &format!("{}attn_k_norm", prefix))?;
        
        // MLP 가중치 로드 (reader 가변 참조 전달)
        self.mlp.0.load_weights_inplace(ct, reader, prefix, device, dtype, true)?;
        
        self.norm1 = load_norm(reader, &format!("{}ln1", prefix))?;
        self.norm2 = load_norm(reader, &format!("{}ln2", prefix))?;
        Ok(())
    }

    pub fn forward(&self, xs: &Tensor, cu_seqlens: &Tensor, rotary_pos_emb: Option<&Tensor>, position_embeddings: Option<(&Tensor, &Tensor)>) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.norm1.forward(xs)?;
        let xs = self.attn.forward_with_params(&xs, cu_seqlens, rotary_pos_emb, position_embeddings)?;
        let xs = residual.add(&xs)?;

        let residual = xs.clone();
        let xs = self.norm2.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        Ok(xs.add(&residual)?)
    }
}

pub struct GlmOcrVisionPatchMerger {
    proj: Linear,
    post_projection_norm: LayerNorm,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl GlmOcrVisionPatchMerger {
    pub fn new(vb: VarBuilder, config: &GlmOcrVisionConfig) -> Result<Self> {
        let proj = linear_no_bias(config.out_hidden_size, config.out_hidden_size, vb.pp("proj"))?;
        let post_projection_norm = layer_norm(config.out_hidden_size, config.rms_norm_eps, vb.pp("post_projection_norm"))?;
        let context_dim = config.out_hidden_size * config.in_channels;
        let gate_proj = linear_no_bias(config.out_hidden_size, context_dim, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(config.out_hidden_size, context_dim, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(context_dim, config.out_hidden_size, vb.pp("down_proj"))?;

        Ok(Self { proj, post_projection_norm, gate_proj, up_proj, down_proj, act_fn: config.hidden_act })
    }

    pub fn forward(&self, hidden_state: &Tensor) -> Result<Tensor> {
        let target_dtype = self.post_projection_norm.weight().dtype();
        
        let hs_f32 = hidden_state.to_dtype(DType::F32)?;
        let w_proj = self.proj.weight().to_dtype(DType::F32)?;
        let b_proj = match self.proj.bias() { Some(b) => Some(b.to_dtype(DType::F32)?), None => None };
        let proj_f32 = Linear::new(w_proj, b_proj);
        let mut hs = proj_f32.forward(&hs_f32)?.to_dtype(target_dtype)?;
        
        hs = self.post_projection_norm.forward(&hs)?;
        hs = hs.gelu()?.to_dtype(DType::F32)?; // 🚀 CPU BF16 matmul 에러 원천 차단

        let w_gate = self.gate_proj.weight().to_dtype(DType::F32)?;
        let b_gate = match self.gate_proj.bias() { Some(b) => Some(b.to_dtype(DType::F32)?), None => None };
        let gate_f32 = Linear::new(w_gate, b_gate);
        
        let w_up = self.up_proj.weight().to_dtype(DType::F32)?;
        let b_up = match self.up_proj.bias() { Some(b) => Some(b.to_dtype(DType::F32)?), None => None };
        let up_f32 = Linear::new(w_up, b_up);

        let gate = gate_f32.forward(&hs)?;
        let gate = self.act_fn.forward(&gate)?;
        let up = up_f32.forward(&hs)?;
        let result = gate.broadcast_mul(&up)?;

        let w_down = self.down_proj.weight().to_dtype(DType::F32)?;
        let b_down = match self.down_proj.bias() { Some(b) => Some(b.to_dtype(DType::F32)?), None => None };
        let down_f32 = Linear::new(w_down, b_down);

        Ok(down_f32.forward(&result)?.to_dtype(target_dtype)?)
    }
}

pub struct GlmOcrVisionPatchEmbed {
    patch_size: usize,
    temporal_patch_size: usize,
    in_channels: usize,
    _embed_dim: usize,
    proj: Linear,
}

impl GlmOcrVisionPatchEmbed {
    pub fn new(vb: VarBuilder, config: &GlmOcrVisionConfig) -> Result<Self> {
        let patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size;
        let weight = vb.get((config.hidden_size, config.in_channels, config.temporal_patch_size, config.patch_size, config.patch_size), "proj.weight")?.reshape((config.hidden_size, patch_dim))?;
        let bias = vb.get(config.hidden_size, "proj.bias").ok();
        Ok(Self { patch_size: config.patch_size, temporal_patch_size: config.temporal_patch_size, in_channels: config.in_channels, _embed_dim: config.hidden_size, proj: candle_nn::Linear::new(weight, bias) })
    }

    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let rank = pixel_values.rank();
        let target_dtype = self.proj.weight().dtype();
        let pixel_values_f32 = pixel_values.to_dtype(DType::F32)?; // 🚀 CPU BF16 matmul 에러 원천 차단
        
        // 가중치도 F32로 임시 변환
        let w_f32 = self.proj.weight().to_dtype(DType::F32)?;
        let b_f32 = match self.proj.bias() { Some(b) => Some(b.to_dtype(DType::F32)?), None => None };
        let proj_f32 = Linear::new(w_f32, b_f32);
        
        if rank == 2 {
            Ok(proj_f32.forward(&pixel_values_f32)?.to_dtype(target_dtype)?)
        } else {
            let (batch, _c, h, w) = pixel_values_f32.dims4()?;
            let patches_h = h / self.patch_size;
            let patches_w = w / self.patch_size;
            let num_patches = patches_h * patches_w;
            let pv = pixel_values_f32.reshape((batch, patches_h, self.patch_size, patches_w, self.patch_size, self.in_channels))?.permute((0, 1, 3, 5, 2, 4))?;
            let pv = pv.reshape((batch * num_patches, self.in_channels * self.patch_size * self.patch_size))?.unsqueeze(1)?;
            let ones_shape: Vec<usize> = vec![1, self.temporal_patch_size];
            let pv = pv.broadcast_mul(&Tensor::ones(ones_shape, pv.dtype(), pv.device())?)?;
            let pv = pv.reshape((batch * num_patches, self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size))?;
            Ok(proj_f32.forward(&pv)?.to_dtype(target_dtype)?)
        }
    }
}

pub struct GlmOcrVisionModel {
    patch_embed: GlmOcrVisionPatchEmbed,
    rotary_pos_emb: GlmOcrVisionRotaryEmbedding,
    blocks: Vec<GlmOcrVisionBlock>,
    merger: GlmOcrVisionPatchMerger,
    downsample: candle_nn::Conv2d, // 🚀 잃어버렸던 원본 Conv2d 레이어 복구
    post_layernorm: GlmOcrRMSNorm,
    config: GlmOcrVisionConfig,
    pub file: Option<std::fs::File>,
    pub ct: Option<std::sync::Arc<candle_core::quantized::gguf_file::Content>>,
    pub dtype: DType,
}

impl GlmOcrVisionModel {
    pub fn new_with_file(
        vb: VarBuilder, 
        config: &GlmOcrVisionConfig,
        file: Option<std::fs::File>,
        ct: Option<std::sync::Arc<candle_core::quantized::gguf_file::Content>>,
    ) -> Result<Self> {
        let patch_embed = GlmOcrVisionPatchEmbed::new(vb.pp("patch_embed"), config)?;
        let head_dim = config.hidden_size / config.num_heads;
        let rotary_pos_emb = GlmOcrVisionRotaryEmbedding::new(head_dim / 2, config.rope_theta, vb.device(), vb.dtype())?;
        
        let mut blocks = Vec::new();
        for i in 0..config.depth {
            let block = if file.is_some() {
                let mut b = GlmOcrVisionBlock::new_skeleton(config, vb.device())?;
                b.clear_weights();
                b
            } else {
                GlmOcrVisionBlock::new(vb.pp("blocks").pp(i), config)?
            };
            blocks.push(block);
        }

        let merger = GlmOcrVisionPatchMerger::new(vb.pp("merger"), config)?;
        
        // 🚀 원본 GLM-OCR의 공간 병합을 담당하는 Conv2d 초기화 복구
        let downsample = candle_nn::conv2d(
            config.hidden_size,
            config.out_hidden_size,
            config.spatial_merge_size,
            candle_nn::Conv2dConfig {
                stride: config.spatial_merge_size,
                ..Default::default()
            },
            vb.pp("downsample"),
        )?;

        let post_layernorm = GlmOcrRMSNorm::new(vb.pp("post_layernorm"), config.hidden_size, config.rms_norm_eps)?;

        Ok(Self { patch_embed, rotary_pos_emb, blocks, merger, downsample, post_layernorm, config: config.clone(), file, ct, dtype: vb.dtype() })
    }

    pub fn forward(&mut self, pixel_values: &Tensor, grid_thw: &Tensor) -> Result<Tensor> {
        let cpu = &Device::Cpu;
        let gpu = pixel_values.device(); // 원래 CUDA 장치 저장

        // VRAM 피크 차단을 위해 입력 이미지를 CPU로 내려서 전처리
        let pixel_values_cpu = pixel_values.to_device(cpu)?;
        let mut hidden_states = self.patch_embed.forward(&pixel_values_cpu)?;

        let grid_thw_cpu = grid_thw.to_device(cpu)?;
        let grid_thw_parsed = if grid_thw_cpu.dims().len() == 1 {
            let t = grid_thw_cpu.i(0)?.to_dtype(DType::F32)?.to_scalar::<f32>()? as usize;
            let h = grid_thw_cpu.i(1)?.to_dtype(DType::F32)?.to_scalar::<f32>()? as usize;
            let w = grid_thw_cpu.i(2)?.to_dtype(DType::F32)?.to_scalar::<f32>()? as usize;
            vec![(t, h, w)]
        } else {
            let grid_thw_f32 = grid_thw_cpu.to_dtype(DType::F32)?;
            let n = grid_thw_f32.dim(0)?;
            let mut result = Vec::new();
            for i in 0..n {
                let row = grid_thw_f32.i(i)?;
                result.push((row.i(0)?.to_scalar::<f32>()? as usize, row.i(1)?.to_scalar::<f32>()? as usize, row.i(2)?.to_scalar::<f32>()? as usize));
            }
            result
        };

        let (cos, sin) = self.rotary_pos_emb.rot_pos_emb(&grid_thw_parsed, self.config.spatial_merge_size)?;
        let rotary_pos_emb = Tensor::cat(&[&cos, &sin], D::Minus1)?;

        let mut cu_seqlens_values: Vec<i32> = vec![0];
        let mut cumsum: i32 = 0;
        for (t, h, w) in &grid_thw_parsed {
            let spatial_patches = (h * w) as i32;
            for _ in 0..*t { cumsum += spatial_patches; cu_seqlens_values.push(cumsum); }
        }
        let cu_seqlens = Tensor::from_slice(&cu_seqlens_values, &[cu_seqlens_values.len()], gpu)?;

        // === VRAM 전송 (블록 연산 전) ===
        hidden_states = hidden_states.to_device(gpu)?;
        let rotary_pos_emb_gpu = rotary_pos_emb.to_device(gpu)?;
        let cos_gpu = cos.to_device(gpu)?;
        let sin_gpu = sin.to_device(gpu)?;
        let position_embeddings_gpu = (&cos_gpu, &sin_gpu);

        // ★ [SSD 오프로딩] 비전 블록 핑퐁 로직: 연산 직전에만 VRAM으로 로드
        for (i, block) in self.blocks.iter_mut().enumerate() {
            if block.is_cleared() {
                if let (Some(f), Some(ct)) = (self.file.as_mut(), self.ct.as_ref()) {
                    let prefix = format!("v.blk.{}.", i);
                    block.load_weights_inplace(ct, f, &prefix, hidden_states.device(), self.dtype)?;
                }
            }

            hidden_states = block.forward(&hidden_states, &cu_seqlens, Some(&rotary_pos_emb_gpu), Some(position_embeddings_gpu))?;

            // 🚀 비전 인코더는 단 1회만 실행되므로, 연산이 끝난 블록은 즉시 VRAM에서 해제하여 
            // 텍스트 생성 단계(KV Cache 등)를 위한 GPU 메모리를 최대한 확보합니다.
            if self.file.is_some() {
                block.clear_weights(); 
            }
        }

        // === RAM 회수 (후처리) ===
        hidden_states = hidden_states.to_device(cpu)?;

        let hidden_states = self.post_layernorm.forward(&hidden_states)?;
        let sms = self.config.spatial_merge_size;
        let hidden_dim = hidden_states.dim(hidden_states.dims().len() - 1)?;
        
        let total_patches = hidden_states.dim(0)?; 
        let merged_patches = total_patches / (sms * sms); 
        
        let hidden_states = hidden_states.reshape((merged_patches, sms, sms, hidden_dim))?;
        // 🚀 [Conv2d 입력 규격 맞춤] [batch, height, width, channels] -> [batch, channels, height, width]
        let hidden_states = hidden_states.permute((0, 3, 1, 2))?;
        
        // 🚀 CPU 연산 호환성을 위해 F32로 안전하게 Conv2d 처리 (원본 모델의 시각 압축 로직 복구 완료)
        let target_dtype = hidden_states.dtype();
        let hidden_states_f32 = hidden_states.to_dtype(DType::F32)?;
        
        let w_ds = self.downsample.weight().to_dtype(DType::F32)?;
        let b_ds = match self.downsample.bias() { Some(b) => Some(b.to_dtype(DType::F32)?), None => None };
        let downsample_cfg = candle_nn::Conv2dConfig { stride: sms, ..Default::default() };
        let downsample_f32 = candle_nn::Conv2d::new(w_ds, b_ds, downsample_cfg);
        
        let hidden_states = downsample_f32.forward(&hidden_states_f32)?.to_dtype(target_dtype)?;
        let hidden_states = hidden_states.reshape((merged_patches, self.config.out_hidden_size))?; 
        
        Ok(self.merger.forward(&hidden_states)?.unsqueeze(0)?)
    }
}

pub struct GlmOcrTextRotaryEmbedding {
    inv_freq: Tensor,
    mrope_section: Vec<usize>,
}

impl GlmOcrTextRotaryEmbedding {
    pub fn new(config: &GlmOcrTextConfig, device: &candle_core::Device, dtype: DType) -> Result<Self> {
        let rope_theta = config.rope_parameters.rope_theta;
        let head_dim = config.head_dim.unwrap_or_else(|| config.hidden_size / config.num_attention_heads);
        let dim = (head_dim as f32 * config.rope_parameters.partial_rotary_factor) as usize;
        let inv_freq: Vec<f32> = (0..dim).step_by(2).map(|i| 1.0 / (rope_theta as f64).powf(i as f64 / dim as f64) as f32).collect();
        let inv_freq = Tensor::from_slice(&inv_freq, (1, inv_freq.len()), device)?.to_dtype(dtype)?;
        Ok(Self { inv_freq, mrope_section: config.rope_parameters.mrope_section.clone() })
    }

    fn apply_mrope(&self, freqs: &Tensor) -> Result<Tensor> {
        let section = &self.mrope_section;
        let mut chunks = Vec::new();
        let mut offset = 0;
        for &s in section.iter() {
            chunks.push(freqs.narrow(D::Minus1, offset, s)?);
            offset += s;
        }
        let mut result_parts = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            result_parts.push(chunk.i(i % 3)?);
        }
        Ok(Tensor::cat(&result_parts, D::Minus1)?)
    }

    pub fn forward_with_position_ids(&self, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_, bs, _seq_len) = position_ids.dims3()?;
        let inv_freq_len = self.inv_freq.dim(1)?;
        let inv_freq = self.inv_freq.to_device(position_ids.device())?.unsqueeze(0)?.unsqueeze(D::Minus1)?.broadcast_as((3, bs, inv_freq_len, 1))?.to_dtype(DType::F32)?.contiguous()?;
        let pos_expanded = position_ids.unsqueeze(D::Minus2)?.to_dtype(DType::F32)?.contiguous()?;
        let freqs = inv_freq.matmul(&pos_expanded)?.transpose(2, 3)?;
        let freqs = self.apply_mrope(&freqs)?; 
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?.contiguous()?;
        Ok((emb.cos()?.to_dtype(self.inv_freq.dtype())?, emb.sin()?.to_dtype(self.inv_freq.dtype())?))
    }

    pub fn forward(&self, seq_len: usize, seqlen_offset: usize, device: &candle_core::Device) -> Result<(Tensor, Tensor)> {
        let positions = Tensor::arange(seqlen_offset as f32, (seqlen_offset + seq_len) as f32, device)?.to_dtype(self.inv_freq.dtype())?;
        let positions_3d = positions.unsqueeze(0)?.unsqueeze(0)?.expand((3, 1, seq_len))?; 
        let inv_freq = self.inv_freq.to_device(device)?.unsqueeze(0)?.unsqueeze(D::Minus1)?.broadcast_as((3, 1, self.inv_freq.dim(1)?, 1))?.to_dtype(DType::F32)?.contiguous()?;
        let positions_expanded = positions_3d.unsqueeze(D::Minus2)?.to_dtype(DType::F32)?.contiguous()?;
        let freqs = inv_freq.matmul(&positions_expanded)?.transpose(2, 3)?;
        let freqs = self.apply_mrope(&freqs)?; 
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?.contiguous()?;
        Ok((emb.cos()?.to_dtype(self.inv_freq.dtype())?, emb.sin()?.to_dtype(self.inv_freq.dtype())?))
    }
}

pub struct GlmOcrTextModel {
    embed_tokens: Embedding,
    layers: Vec<GlmOcrTextDecoderLayer>,
    norm: GlmOcrRMSNorm,
    lm_head: Linear,
    rotary_emb: GlmOcrTextRotaryEmbedding,
    spatial_merge_size: usize,
    next_mrope_pos: usize,
    prefill_seq_len: usize,
    pub file: Option<std::fs::File>,
    pub ct: Option<std::sync::Arc<candle_core::quantized::gguf_file::Content>>,
    pub dtype: DType,
}

impl GlmOcrTextModel {
    pub fn new_with_file(
        vb: VarBuilder,
        config: GlmOcrTextConfig,
        spatial_merge_size: usize,
        device: &Device,
        file: Option<std::fs::File>,
        ct: Option<std::sync::Arc<candle_core::quantized::gguf_file::Content>>,
    ) -> Result<Self> {
        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer = if file.is_some() {
                let mut l = GlmOcrTextDecoderLayer::new_skeleton(&config, vb.device())?;
                l.clear_weights();
                l
            } else {
                GlmOcrTextDecoderLayer::new(vb.pp("layers").pp(i), &config, i)?
            };
            layers.push(layer);
        }

        // 🚀 [VRAM 상주 최적화] norm과 lm_head를 CPU가 아닌 GPU로 강제 업로드하여 VRAM에 상주하게 만듭니다.
        let norm = GlmOcrRMSNorm::new_on_device(vb.pp("norm"), config.hidden_size, config.rms_norm_eps, device)?;
        
        // 🚀 원본 DType(BF16) 그대로 GPU에 완전히 캐싱되므로 전송 병목이 0(Zero)가 되며 연산 속도가 극대화됩니다.
        let w_lm = vb.root().pp("lm_head").get((config.vocab_size, config.hidden_size), "weight")?.to_device(device)?;
        let lm_head = Linear::new(w_lm, None);
        
        let rotary_emb = GlmOcrTextRotaryEmbedding::new(&config, vb.device(), vb.dtype())?;

        Ok(Self {
            embed_tokens, layers, norm, lm_head, rotary_emb, spatial_merge_size, next_mrope_pos: 0, prefill_seq_len: 0, file, ct, dtype: vb.dtype()
        })
    }

    fn compute_mrope_position_ids(&mut self, image_mask: &Tensor, grid_thw: &Tensor, seq_len: usize, device: &candle_core::Device) -> Result<Tensor> {
        let t_dim = grid_thw.i(0)?.to_dtype(DType::F32)?.to_scalar::<f32>()? as usize;
        let h_dim = grid_thw.i(1)?.to_dtype(DType::F32)?.to_scalar::<f32>()? as usize;
        let w_dim = grid_thw.i(2)?.to_dtype(DType::F32)?.to_scalar::<f32>()? as usize;
        let llm_grid_t = t_dim;
        let llm_grid_h = h_dim / self.spatial_merge_size;
        let llm_grid_w = w_dim / self.spatial_merge_size;
        let _num_image_tokens = llm_grid_t * llm_grid_h * llm_grid_w;
        let mask_vec = image_mask.squeeze(0)?.to_dtype(DType::U8)?.to_vec1::<u8>()?;

        let mut t_ids: Vec<i64> = Vec::with_capacity(seq_len);
        let mut h_ids: Vec<i64> = Vec::with_capacity(seq_len);
        let mut w_ids: Vec<i64> = Vec::with_capacity(seq_len);

        let mut st_idx: i64 = 0; 
        let mut i = 0usize;

        while i < seq_len {
            let is_img = mask_vec[i] == 1;
            let start = i;
            while i < seq_len && (mask_vec[i] == 1) == is_img { i += 1; }
            let run_len = i - start;

            if is_img {
                for ti in 0..llm_grid_t {
                    for hi in 0..llm_grid_h {
                        for wi in 0..llm_grid_w {
                            t_ids.push(ti as i64 + st_idx);
                            h_ids.push(hi as i64 + st_idx);
                            w_ids.push(wi as i64 + st_idx);
                        }
                    }
                }
                let max_offset = (llm_grid_t as i64 - 1).max(llm_grid_h as i64 - 1).max(llm_grid_w as i64 - 1);
                st_idx += max_offset + 1;
            } else {
                for j in 0..run_len {
                    let pos = st_idx + j as i64;
                    t_ids.push(pos); h_ids.push(pos); w_ids.push(pos);
                }
                st_idx += run_len as i64;
            }
        }

        self.next_mrope_pos = st_idx as usize;
        self.prefill_seq_len = seq_len;

        let t_t = Tensor::from_vec(t_ids, (1, seq_len), device)?;
        let h_t = Tensor::from_vec(h_ids, (1, seq_len), device)?;
        let w_t = Tensor::from_vec(w_ids, (1, seq_len), device)?;
        Ok(Tensor::stack(&[&t_t, &h_t, &w_t], 0)?) 
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        image_features: Option<&Tensor>,
        image_mask: Option<&Tensor>,
        image_grid_thw: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (bs, seq_len) = input_ids.dims2()?;
        
        // 🚀 [Fix] 임베딩 텐서는 CPU에 상주하므로 input_ids를 CPU로 내려서 index_select를 수행합니다.
        let input_ids_cpu = input_ids.to_device(&Device::Cpu)?;
        let mut inputs_embeds = self.embed_tokens.forward(&input_ids_cpu)?;

        if let (Some(img_feats), Some(img_mask)) = (image_features, image_mask) {
            let img_mask_bool = img_mask.squeeze(0)?.to_dtype(DType::U8)?.to_vec1::<u8>()?;
            let image_indices: Vec<usize> = img_mask_bool.iter().enumerate().filter(|&(_, &v)| v == 1).map(|(i, _)| i).collect();
            let num_features = img_feats.dim(1)?;
            let num_to_replace = image_indices.len().min(num_features);

            let embeds_flat = inputs_embeds.squeeze(0)?; 
            let mut embeds_vec: Vec<Tensor> = Vec::new();
            let mut pos = 0;
            for (feat_idx, &img_pos) in image_indices.iter().take(num_to_replace).enumerate() {
                if img_pos > pos { embeds_vec.push(embeds_flat.narrow(0, pos, img_pos - pos)?); }
                embeds_vec.push(img_feats.i((0, feat_idx, ..))?.unsqueeze(0)?);
                pos = img_pos + 1;
            }
            if pos < seq_len { embeds_vec.push(embeds_flat.narrow(0, pos, seq_len - pos)?); }
            let refs: Vec<&Tensor> = embeds_vec.iter().collect();
            inputs_embeds = Tensor::cat(&refs, 0)?.unsqueeze(0)?;
        }

        // 🚀 [Fix] 이미지 임베딩과 결합이 끝난 뒤, 연산 레이어 통과를 위해 다시 GPU 장치로 올립니다.
        inputs_embeds = inputs_embeds.to_device(input_ids.device())?;

        let attention_mask = if seq_len > 1 { Some(prepare_causal_attention_mask(bs, seq_len, seqlen_offset, input_ids.device())?) } else { None };

        let (cos, sin) = if seqlen_offset == 0 {
            if let (Some(mask), Some(thw)) = (image_mask, image_grid_thw) {
                let pos_ids = self.compute_mrope_position_ids(mask, thw, seq_len, input_ids.device())?;
                self.prefill_seq_len = seq_len;
                self.rotary_emb.forward_with_position_ids(&pos_ids)?
            } else {
                self.next_mrope_pos = seq_len;
                self.prefill_seq_len = seq_len;
                self.rotary_emb.forward(seq_len, 0, input_ids.device())?
            }
        } else {
            let decode_pos = self.next_mrope_pos + (seqlen_offset - self.prefill_seq_len);
            self.rotary_emb.forward(1, decode_pos, input_ids.device())?
        };

        let mut hidden_states = inputs_embeds;
        
        // ★ [SSD 오프로딩] 텍스트 레이어 핑퐁 로직: 연산 직전에만 VRAM으로 로드
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if layer.is_cleared() {
                if let (Some(f), Some(ct)) = (self.file.as_mut(), self.ct.as_ref()) {
                    let prefix = format!("blk.{}.", i);
                    // hidden_states.device()를 통해 CUDA 장치로 가중치 주입
                    layer.load_weights_inplace(ct, f, &prefix, hidden_states.device(), self.dtype)?;
                }
            }

            hidden_states = layer.forward(&hidden_states, (&cos, &sin), attention_mask.as_ref())?;

            // 🚀 가중치를 VRAM에 계속 유지하기 위해 해제 로직을 비활성화합니다.
            // if self.file.is_some() {
            //     layer.clear_weights(); // 연산 종료 즉시 VRAM 해제
            // }
        }

        // 🚀 [고속 추론 최적화] 전체 시퀀스를 CPU로 내리면 PCIe 대역폭 병목이 극심하게 발생합니다.
        // 다음 토큰 예측에 필요한 '마지막 토큰'만 GPU에서 선제적으로 추출한 뒤 CPU로 전송하여 속도를 극대화합니다.
        // 🚀 [VRAM 상주 최적화] norm과 lm_head가 GPU에 상주하므로 CPU 통신 없이 마지막 토큰만 잘라서 100% GPU 연산을 수행합니다.
        let last_hidden = hidden_states.narrow(1, seq_len - 1, 1)?;
        let last_normed = self.norm.forward(&last_hidden)?;
        
        // 🚀 GPU(VRAM) 안에서 네이티브 DType으로 즉시 MatMul 처리되므로 속도가 수십 배 상승합니다.
        let logits = self.lm_head.forward(&last_normed)?;
        
        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() { layer.clear_kv_cache(); }
    }
}

pub struct GlmOcrModel {
    vision_encoder: GlmOcrVisionModel,
    language_model: GlmOcrTextModel,
    stop_token_ids: Vec<u32>,
}

impl GlmOcrModel {
    pub fn new_with_file(
        vb: VarBuilder, 
        config: GlmOcrConfig, 
        eos_ids: Vec<u32>,
        device: &Device,
        file_text: Option<std::fs::File>,
        ct_text: Option<std::sync::Arc<candle_core::quantized::gguf_file::Content>>,
        file_vision: Option<std::fs::File>,
        ct_vision: Option<std::sync::Arc<candle_core::quantized::gguf_file::Content>>,
    ) -> Result<Self> {
        let vision_encoder = GlmOcrVisionModel::new_with_file(
            vb.pp("model").pp("visual"), 
            &config.vision_config,
            file_vision,
            ct_vision
        )?;
        let language_model = GlmOcrTextModel::new_with_file(
            vb.pp("model").pp("language_model"),
            config.text_config,
            config.vision_config.spatial_merge_size,
            device,
            file_text,
            ct_text
        )?;

        Ok(Self { vision_encoder, language_model, stop_token_ids: eos_ids })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        image_grid_thw: Option<&Tensor>,
        image_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let image_features = if let Some(pixels) = pixel_values {
            let grid_thw = if let Some(grid) = image_grid_thw {
                grid.clone()
            } else {
                Tensor::new(&[ 1u32, (pixels.dim(0)? / 44) as u32, (pixels.dim(1)? / 44) as u32 ], input_ids.device())?
            };
            Some(self.vision_encoder.forward(pixels, &grid_thw)?)
        } else { None };

        self.language_model.forward(input_ids, image_features.as_ref(), image_mask, image_grid_thw, seqlen_offset)
    }

    pub fn clear_kv_cache(&mut self) { self.language_model.clear_kv_cache(); }
}

impl InferenceModel for GlmOcrModel {
    fn forward_initial(&mut self, input_ids: &Tensor, seqlen_offset: usize, data: crate::models::common::MultiModalData) -> Result<Tensor> {
        if data.data_vec.len() != 3 { return Err(anyhow::anyhow!("GlmOcr process data error, must have pixel_values, image_grid_thw, image_mask")); }
        self.forward(input_ids, Some(&data.data_vec[0]), Some(&data.data_vec[1]), Some(&data.data_vec[2]), seqlen_offset)
    }
    fn forward_step(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        self.forward(input_ids, None, None, None, seqlen_offset)
    }
    fn clear_cache(&mut self) { self.clear_kv_cache(); }
    fn stop_token_ids(&self) -> Vec<u32> { self.stop_token_ids.clone() }
}