use anyhow::Result; // 미사용 anyhow 제거
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
    _device: &Device,
) -> Result<GlmOcrModel> {
    // 🚀 실행 환경에 맞춰 목표 DType을 결정합니다. CUDA는 BF16을 사용해야 충돌이 나지 않습니다.
    let target_dtype = if _device.is_cuda() { DType::BF16 } else { DType::F32 };

    let mut vision_file = File::open(mmproj_path)?;
    let vision_content = std::sync::Arc::new(gguf_file::Content::read(&mut vision_file)?);
    
    let mut text_file = File::open(model_path)?;
    let text_content = std::sync::Arc::new(gguf_file::Content::read(&mut text_file)?);

    let mut tensors = std::collections::HashMap::new();

    // 1. 레이어를 제외한 Vision 텐서를 CPU로 로드하되, DType을 통일합니다 (VRAM 피크 방어)
    for name in vision_content.tensor_infos.keys() {
        if !name.contains(".blk.") {
            let t_raw = vision_content.tensor(&mut vision_file, name, &Device::Cpu)?.dequantize(&Device::Cpu)?;
            let t = t_raw.to_dtype(target_dtype)?; // 🚀 DType 일치
            let mapped_name = map_vision_name(name);
            
            if mapped_name == "model.visual.patch_embed.proj.weight" && t.rank() == 4 {
                let temporal = config.vision_config.temporal_patch_size;
                let mut patch_t = t.clone();
                patch_t = patch_t.unsqueeze(2)?.broadcast_as((
                    patch_t.dim(0)?, patch_t.dim(1)?, temporal, patch_t.dim(2)?, patch_t.dim(3)?,
                ))?;
                patch_t = (patch_t.to_dtype(DType::F32)? / (temporal as f64))?.to_dtype(target_dtype)?;
                tensors.insert(mapped_name, patch_t);
            } else if mapped_name == "model.visual.merger.proj.weight" {
                if t.rank() == 4 {
                    let out_c = t.dim(0)?;
                    let flattened_in = t.dim(1)? * t.dim(2)? * t.dim(3)?;
                    let reshaped_t = t.reshape((out_c, flattened_in))?;
                    tensors.insert(mapped_name, reshaped_t);
                } else if t.rank() == 2 && t.dim(1)? == 4096 {
                    tensors.insert(mapped_name, t);
                } else {
                    tensors.insert(mapped_name, t);
                }
            } else {
                tensors.insert(mapped_name, t);
            }
        }
    }

    // 2. 레이어를 제외한 Text 텐서를 CPU로 로드
    let mut raw_text_tensors = std::collections::HashMap::new();
    for name in text_content.tensor_infos.keys() {
        if !name.contains(".blk.") {
            let t_raw = text_content.tensor(&mut text_file, name, &Device::Cpu)?.dequantize(&Device::Cpu)?;
            let t = t_raw.to_dtype(target_dtype)?; // 🚀 DType 일치
            raw_text_tensors.insert(name.clone(), t);
        }
    }

    map_text_tensors(raw_text_tensors, &mut tensors, 0)?;

    if !tensors.contains_key("lm_head.weight") {
        if let Some(emb) = tensors.get("model.language_model.embed_tokens.weight") {
            tensors.insert("lm_head.weight".to_string(), emb.clone());
        }
    }

    let mut mapped_keys: Vec<String> = tensors.keys().cloned().collect();
    mapped_keys.sort();

    // 🌟 VarBuilder 생성 시 DType::F32 하드코딩을 제거하여 BF16 DType 충돌 에러를 원천 차단합니다.
    let vb = candle_nn::VarBuilder::from_tensors(tensors, target_dtype, &Device::Cpu);
    
    // 파일 핸들을 쥐여주며 JIT 전용 생성자로 호출
    let model = match GlmOcrModel::new_with_file(
        vb, 
        config.clone(), 
        config.text_config.eos_token_id.clone(),
        Some(text_file),
        Some(text_content.clone()), // .clone() 추가
        Some(vision_file),
        Some(vision_content.clone()) // .clone() 추가
    ) {
        Ok(m) => m,
        Err(e) => {
            println!("\n[GGUF Loader] GlmOcrModel 텐서 매핑 또는 초기화 실패: {:?}", e);
            println!("============================================================");
            println!("[DEBUG] 매핑이 완료된 최종 텐서 목록:");
            for k in &mapped_keys {
                println!("  - {}", k);
            }
            println!("============================================================");
            println!("[DEBUG] 원본 Vision GGUF 텐서 목록:");
            let mut v_keys: Vec<&String> = vision_content.tensor_infos.keys().collect();
            v_keys.sort();
            for k in &v_keys {
                println!("  - {}", k);
            }
            println!("============================================================");
            println!("[DEBUG] 원본 Text GGUF 텐서 목록:");
            let mut t_keys: Vec<&String> = text_content.tensor_infos.keys().collect();
            t_keys.sort();
            for k in &t_keys {
                println!("  - {}", k);
            }
            println!("============================================================");
            return Err(e);
        }
    };
    
    Ok(model)
}

fn map_vision_name(name: &str) -> String {
    let mut n = name.to_string();
    if n.starts_with("v.") { n = n.replacen("v.", "", 1); }
    else if n.starts_with("mm.") { n = n.replacen("mm.", "merger.", 1); }
    else if n.starts_with("vision.") { n = n.replacen("vision.", "", 1); }
    else if n.starts_with("model.visual.") { n = n.replacen("model.visual.", "", 1); }

    n = n.replace("patch_embd.weight.1", "downsample.weight")
         .replace("patch_embd.bias.1", "downsample.bias")
         .replace("patch_embd.weight", "patch_embed.proj.weight")
         .replace("patch_embd.bias", "patch_embed.proj.bias")
         .replace("patch_embed.weight", "patch_embed.proj.weight")
         .replace("patch_embed.bias", "patch_embed.proj.bias")
         .replace("blk.", "blocks.")
         .replace("attn_qkv", "attn.qkv")
         .replace("attn_out", "attn.proj")
         .replace("attn_q_norm", "attn.q_norm")
         .replace("attn_k_norm", "attn.k_norm")
         .replace("ffn_down", "mlp.down_proj")
         .replace("ffn_up", "mlp.up_proj")
         .replace("ffn_gate", "mlp.gate_proj")
         .replace("ln1", "norm1")
         .replace("ln2", "norm2")
         .replace("post_ln", "post_layernorm")
         .replace("merger.patch_merger", "merger.proj")
         .replace("merger.post_norm", "merger.post_projection_norm")
         .replace("merger.gate", "merger.gate_proj")
         .replace("merger.up", "merger.up_proj")
         .replace("merger.down", "merger.down_proj");

    if n == "merger.model.fc.weight" {
        n = "downsample.weight".to_string();
    } else if n == "merger.model.fc.bias" {
        n = "downsample.bias".to_string();
    }

    format!("model.visual.{}", n)
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
