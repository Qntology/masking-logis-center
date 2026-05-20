use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_core::quantized::gguf_file;
use std::path::Path;
use std::fs::File;

use crate::glm_ocr::config::GlmOcrConfig;
use crate::glm_ocr::model::GlmOcrModel;

pub fn load_glm_ocr_gguf<P: AsRef<Path>>(
    model_path: P,
    mmproj_path: P,
    config: &GlmOcrConfig,
    device: &Device,
) -> Result<GlmOcrModel> {
    let mut vision_file = File::open(mmproj_path)?;
    let vision_content = gguf_file::Content::read(&mut vision_file)?;
    
    let mut text_file = File::open(model_path)?;
    let text_content = gguf_file::Content::read(&mut text_file)?;

    let mut tensors = std::collections::HashMap::new();

    // 1. Extract and map Vision Tensors
    for name in vision_content.tensor_names() {
        let t = vision_content.tensor(&mut vision_file, &name, device)?.dequantize(device)?;
        let mapped_name = map_vision_name(&name);
        tensors.insert(mapped_name, t);
    }

    // 2. Extract and map Text Tensors
    // Collect all text tensors first to handle combined layers
    let mut raw_text_tensors = std::collections::HashMap::new();
    for name in text_content.tensor_names() {
        let t = text_content.tensor(&mut text_file, &name, device)?.dequantize(device)?;
        raw_text_tensors.insert(name, t);
    }

    // Map and combine text tensors
    map_text_tensors(raw_text_tensors, &mut tensors, config.text_config.num_hidden_layers)?;

    // Handle tied weights for lm_head if missing
    if !tensors.contains_key("lm_head.weight") {
        if let Some(emb) = tensors.get("model.language_model.embed_tokens.weight") {
            tensors.insert("lm_head.weight".to_string(), emb.clone());
        }
    }

    let vb = candle_nn::VarBuilder::from_tensors(tensors, DType::F32, device);
    
    let model = GlmOcrModel::new(vb, config.clone(), config.text_config.eos_token_id.clone())?;
    
    Ok(model)
}

fn map_vision_name(name: &str) -> String {
    let n = name.replace("mm.", "");
    // Common mapping for clip-like vision encoders in GGUF
    let mapped = n.replace("visual.", "model.visual.")
                  .replace("patch_embed.proj", "patch_embed.proj")
                  .replace("blocks.", "blocks.")
                  .replace("norm1", "norm1")
                  .replace("norm2", "norm2")
                  .replace("attn.qkv", "attn.qkv")
                  .replace("attn.proj", "attn.proj")
                  .replace("mlp.fc1", "mlp.gate_proj") // Placeholder mapping, check GLM architecture
                  .replace("mlp.fc2", "mlp.down_proj");
    
    if !mapped.starts_with("model.visual") {
        format!("model.visual.{}", mapped)
    } else {
        mapped
    }
}

fn map_text_tensors(
    mut raw: std::collections::HashMap<String, Tensor>,
    out: &mut std::collections::HashMap<String, Tensor>,
    num_layers: usize,
) -> Result<()> {
    if let Some(t) = raw.remove("token_embd.weight") {
        out.insert("model.language_model.embed_tokens.weight".to_string(), t);
    }
    if let Some(t) = raw.remove("output_norm.weight") {
        out.insert("model.language_model.norm.weight".to_string(), t);
    }
    if let Some(t) = raw.remove("output.weight") {
        out.insert("lm_head.weight".to_string(), t);
    }

    for i in 0..num_layers {
        let prefix = format!("blk.{}.", i);
        let out_prefix = format!("model.language_model.layers.{}.", i);

        // Layernorms
        if let Some(t) = raw.remove(&format!("{}attn_norm.weight", prefix)) {
            out.insert(format!("{}input_layernorm.weight", out_prefix), t);
        }
        if let Some(t) = raw.remove(&format!("{}ffn_norm.weight", prefix)) {
            out.insert(format!("{}post_attention_layernorm.weight", out_prefix), t);
        }
        
        // Attention
        if let Some(t) = raw.remove(&format!("{}attn_q.weight", prefix)) {
            out.insert(format!("{}self_attn.q_proj.weight", out_prefix), t);
        }
        if let Some(t) = raw.remove(&format!("{}attn_k.weight", prefix)) {
            out.insert(format!("{}self_attn.k_proj.weight", out_prefix), t);
        }
        if let Some(t) = raw.remove(&format!("{}attn_v.weight", prefix)) {
            out.insert(format!("{}self_attn.v_proj.weight", out_prefix), t);
        }
        if let Some(t) = raw.remove(&format!("{}attn_output.weight", prefix)) {
            out.insert(format!("{}self_attn.o_proj.weight", out_prefix), t);
        }

        // MLP: Combined gate and up
        let gate = raw.remove(&format!("{}ffn_gate.weight", prefix));
        let up = raw.remove(&format!("{}ffn_up.weight", prefix));
        if let (Some(g), Some(u)) = (gate, up) {
            let combined = Tensor::cat(&[&g, &u], 0)?;
            out.insert(format!("{}mlp.gate_up_proj.weight", out_prefix), combined);
        }
        
        if let Some(t) = raw.remove(&format!("{}ffn_down.weight", prefix)) {
            out.insert(format!("{}mlp.down_proj.weight", out_prefix), t);
        }

        // GLM specific: some GGUF might have additional layernorms
        // In model.rs: post_self_attn_layernorm, post_mlp_layernorm
        // If they are missing in GGUF, we might need to use identity or find alternative names
    }

    Ok(())
}
