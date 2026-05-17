/// Load pretrained weights from a safetensors file.
///
/// Weight key reference (HuggingFace format):
///
///   model.embed_tokens.weight                         [200064, 640]  bf16
///   model.layers.{i}.input_layernorm.weight            [640]         bf16
///   model.layers.{i}.self_attn.q_proj.weight           [896, 640]    bf16
///   model.layers.{i}.self_attn.q_proj.bias             [896]         bf16
///   model.layers.{i}.self_attn.k_proj.weight           [128, 640]    bf16
///   model.layers.{i}.self_attn.k_proj.bias             [128]         bf16
///   model.layers.{i}.self_attn.v_proj.weight           [128, 640]    bf16
///   model.layers.{i}.self_attn.v_proj.bias             [128]         bf16
///   model.layers.{i}.self_attn.o_proj.weight           [640, 896]    bf16
///   model.layers.{i}.self_attn.o_proj.bias             [640]         bf16
///   model.layers.{i}.self_attn.sinks                   [14]          f32
///   model.layers.{i}.post_attention_layernorm.weight   [640]         bf16
///   model.layers.{i}.mlp.router.weight                 [128, 640]    bf16
///   model.layers.{i}.mlp.router.bias                   [128]         bf16
///   model.layers.{i}.mlp.experts.gate_up_proj          [128, 640, 1280] bf16
///   model.layers.{i}.mlp.experts.gate_up_proj_bias     [128, 1280]   bf16
///   model.layers.{i}.mlp.experts.down_proj             [128, 640, 640]  bf16
///   model.layers.{i}.mlp.experts.down_proj_bias        [128, 640]    bf16
///   model.norm.weight                                  [640]         bf16
///   score.weight                                       [33, 640]     bf16
///   score.bias                                         [33]          bf16

use std::collections::HashMap;
use burn::prelude::*;
use burn::module::{Param, ParamId};
use half::bf16;
use safetensors::SafeTensors;

use crate::config::ModelConfig;
use crate::model::privacy_filter::PrivacyFilterModel;

// ── Raw tensor map ────────────────────────────────────────────────────────────

pub struct WeightMap {
    tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl WeightMap {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let st = SafeTensors::deserialize(&bytes)?;
        let mut tensors = HashMap::with_capacity(st.len());

        for (raw_key, view) in st.tensors() {
            let key = raw_key.to_string();
            let shape: Vec<usize> = view.shape().to_vec();
            let data = view.data();

            let f32s: Vec<f32> = match view.dtype() {
                safetensors::Dtype::BF16 => data
                    .chunks_exact(2)
                    .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::F32 => data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect(),
                other => anyhow::bail!("unsupported dtype {:?} for key {key}", other),
            };

            tensors.insert(key, (f32s, shape));
        }

        Ok(Self { tensors })
    }

    /// Take a tensor by key, removing it from the map.
    pub fn take<B: Backend, const N: usize>(
        &mut self,
        key: &str,
        device: &B::Device,
    ) -> anyhow::Result<Tensor<B, N>> {
        let (data, shape) = self.tensors.remove(key)
            .ok_or_else(|| anyhow::anyhow!("weight key not found: {key}"))?;

        if shape.len() != N {
            anyhow::bail!("rank mismatch for {key}: expected {N}, got {}", shape.len());
        }

        Ok(Tensor::<B, N>::from_data(
            TensorData::new(data, shape),
            device,
        ))
    }

    pub fn print_keys(&self) {
        let mut keys: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        keys.sort();
        for k in keys {
            let (_, s) = &self.tensors[k];
            println!("  {k:70}  {s:?}");
        }
    }
}

// ── Weight assignment helpers ────────────────────────────────────────────────

/// Assign weight (transpose [out, in] → [in, out]) + bias into nn::Linear.
fn set_linear_wb<B: Backend>(
    linear: &mut burn::nn::Linear<B>,
    w: Tensor<B, 2>,
    b: Tensor<B, 1>,
) {
    linear.weight = Param::initialized(ParamId::new(), w.transpose());
    linear.bias = Some(Param::initialized(ParamId::new(), b));
}

/// Assign weight only (transpose [out, in] → [in, out]) into nn::Linear.
#[allow(dead_code)]
fn set_linear_w<B: Backend>(
    linear: &mut burn::nn::Linear<B>,
    w: Tensor<B, 2>,
) {
    linear.weight = Param::initialized(ParamId::new(), w.transpose());
}

// ── Model loader ─────────────────────────────────────────────────────────────

pub fn load_model<B: Backend>(
    config: &ModelConfig,
    weights_path: &str,
    device: &B::Device,
) -> anyhow::Result<PrivacyFilterModel<B>> {
    eprintln!("Loading weights from {weights_path} ...");
    let mut wm = WeightMap::from_file(weights_path)?;
    eprintln!("Loaded {} tensors from safetensors file.", wm.tensors.len());

    let mut model = PrivacyFilterModel::new(config, device);

    // 1. Token embedding
    let emb_w: Tensor<B, 2> = wm.take("model.embed_tokens.weight", device)?;
    model.embed_tokens = Param::initialized(ParamId::new(), emb_w);

    // 2. Transformer layers
    for (i, layer) in model.layers.iter_mut().enumerate() {
        let p = format!("model.layers.{i}");

        // Input layernorm
        let ln_w: Tensor<B, 1> = wm.take(&format!("{p}.input_layernorm.weight"), device)?;
        layer.input_layernorm.weight = Param::initialized(ParamId::new(), ln_w);

        // Self-attention projections
        set_linear_wb(
            &mut layer.self_attn.q_proj,
            wm.take(&format!("{p}.self_attn.q_proj.weight"), device)?,
            wm.take(&format!("{p}.self_attn.q_proj.bias"), device)?,
        );
        set_linear_wb(
            &mut layer.self_attn.k_proj,
            wm.take(&format!("{p}.self_attn.k_proj.weight"), device)?,
            wm.take(&format!("{p}.self_attn.k_proj.bias"), device)?,
        );
        set_linear_wb(
            &mut layer.self_attn.v_proj,
            wm.take(&format!("{p}.self_attn.v_proj.weight"), device)?,
            wm.take(&format!("{p}.self_attn.v_proj.bias"), device)?,
        );
        set_linear_wb(
            &mut layer.self_attn.o_proj,
            wm.take(&format!("{p}.self_attn.o_proj.weight"), device)?,
            wm.take(&format!("{p}.self_attn.o_proj.bias"), device)?,
        );

        // Attention sinks
        let sinks: Tensor<B, 1> = wm.take(&format!("{p}.self_attn.sinks"), device)?;
        layer.self_attn.sinks = Param::initialized(ParamId::new(), sinks);

        // Post-attention layernorm
        let pln_w: Tensor<B, 1> = wm.take(&format!("{p}.post_attention_layernorm.weight"), device)?;
        layer.post_attention_layernorm.weight = Param::initialized(ParamId::new(), pln_w);

        // MoE router
        // Router weight in safetensors is [num_experts, hidden_size] (F.linear format)
        // For our manual matmul (hidden @ weight), we need [hidden_size, num_experts]
        let router_w: Tensor<B, 2> = wm.take(&format!("{p}.mlp.router.weight"), device)?;
        layer.mlp.router_weight = Param::initialized(ParamId::new(), router_w.transpose());
        let router_b: Tensor<B, 1> = wm.take(&format!("{p}.mlp.router.bias"), device)?;
        layer.mlp.router_bias = Param::initialized(ParamId::new(), router_b);

        // Expert weights (stored as [num_experts, in_dim, out_dim] for direct matmul)
        let gu: Tensor<B, 3> = wm.take(&format!("{p}.mlp.experts.gate_up_proj"), device)?;
        layer.mlp.gate_up_proj = Param::initialized(ParamId::new(), gu);
        let gu_b: Tensor<B, 2> = wm.take(&format!("{p}.mlp.experts.gate_up_proj_bias"), device)?;
        layer.mlp.gate_up_proj_bias = Param::initialized(ParamId::new(), gu_b);
        let dp: Tensor<B, 3> = wm.take(&format!("{p}.mlp.experts.down_proj"), device)?;
        layer.mlp.down_proj = Param::initialized(ParamId::new(), dp);
        let dp_b: Tensor<B, 2> = wm.take(&format!("{p}.mlp.experts.down_proj_bias"), device)?;
        layer.mlp.down_proj_bias = Param::initialized(ParamId::new(), dp_b);

        // Cache expert weights as CPU vecs for fast per-expert slicing
        layer.mlp.cache_weights();
    }

    // 3. Final norm
    let norm_w: Tensor<B, 1> = wm.take("model.norm.weight", device)?;
    model.norm.weight = Param::initialized(ParamId::new(), norm_w);

    // 4. Classification head
    // score.weight: [num_labels, hidden_size] -> transpose to [hidden_size, num_labels]
    let score_w: Tensor<B, 2> = wm.take("score.weight", device)?;
    model.score_weight = Param::initialized(ParamId::new(), score_w.transpose());
    let score_b: Tensor<B, 1> = wm.take("score.bias", device)?;
    model.score_bias = Param::initialized(ParamId::new(), score_b);

    // Check for remaining keys (should be empty if we loaded everything)
    if !wm.tensors.is_empty() {
        eprintln!("Warning: {} unused weight keys remain:", wm.tensors.len());
        wm.print_keys();
    }

    eprintln!("Model loaded successfully.");
    Ok(model)
}
