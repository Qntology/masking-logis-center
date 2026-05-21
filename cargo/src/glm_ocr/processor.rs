use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use super::config::GlmOcrPreprocessorConfig;
use crate::tokenizer::TokenizerModel;
use crate::utils::img_utils::get_image;
use crate::utils::video_utils::video_smart_resize;

/// GLM-OCR Processor for image and text preprocessing.
///
/// Matches Python's Glm46VImageProcessor behavior:
/// - Uses smart_resize to compute target dimensions
/// - Outputs flattened patches format [num_patches, patch_dim]
pub struct GlmOcrProcessor {
    image_mean: Vec<f32>,
    image_std: Vec<f32>,
    shortest_edge: usize, // min_pixels in Python
    longest_edge: usize,  // max_pixels in Python
    patch_size: usize,
    merge_size: usize,
    temporal_patch_size: usize,
    device: Device,
    dtype: DType,
}

pub struct ProcessedImage {
    pub pixel_values: Tensor, // Shape: [num_patches, patch_dim]
    pub grid_h: usize,
    pub grid_w: usize,
}

pub struct ProcessedInput {
    pub input_ids: Tensor,
    pub pixel_values: Tensor, // Shape: [num_patches, patch_dim]
    pub image_mask: Tensor,
    pub grid_thw: Tensor,
}

impl GlmOcrProcessor {
    pub fn new(path: &str, device: &Device, dtype: DType) -> Result<Self> {
        assert!(
            std::path::Path::new(path).exists(),
            "model path file not exists"
        );
        let config_path = format!("{}/preprocessor_config.json", path);
        assert!(
            std::path::Path::new(&config_path).exists(),
            "preprocessor_config.json not exists in model path"
        );
        let process_cfg: GlmOcrPreprocessorConfig =
            serde_json::from_slice(&std::fs::read(config_path)?)?;

        Ok(Self {
            image_mean: process_cfg.image_mean.clone(),
            image_std: process_cfg.image_std.clone(),
            shortest_edge: process_cfg.size.shortest_edge,
            longest_edge: process_cfg.size.longest_edge,
            patch_size: process_cfg.patch_size,
            merge_size: process_cfg.merge_size,
            temporal_patch_size: process_cfg.temporal_patch_size, // Fixed for images
            device: device.clone(),
            dtype,
        })
    }

    /// Process image for vision encoder.
    ///
    /// Matches Python's Glm46VImageProcessor._preprocess():
    /// 1. Resize using smart_resize
    /// 2. Normalize
    /// 3. Reshape into flattened patches [num_patches, patch_dim]
    ///
    /// Output format: [grid_t * grid_h * grid_w, channels * temporal_patch_size * patch_size * patch_size]
    /// For images: grid_t = 1, so [grid_h * grid_w, 3 * 2 * 14 * 14] = [num_patches, 1176]
    pub fn process_image(&self, image_path: &str) -> Result<ProcessedImage> {
        let mut img = get_image(image_path)?;
        
        // 🚀 가로가 1024를 초과할 경우 가로를 1024로 고정하고, 세로는 원본 비율에 맞춰 자동으로 조정합니다.
        if img.width() > 1024 {
            let new_width = 1024;
            let new_height = (img.height() as f32 * (1024.0 / img.width() as f32)).round() as u32;
            img = img.resize_exact(new_width, new_height, image::imageops::FilterType::Lanczos3);
        }

        let orig_w = img.width();
        let orig_h = img.height();
        // Use smart_resize to compute target dimensions
        let (target_h, target_w) = video_smart_resize(
            self.temporal_patch_size as u32,
            orig_h,
            orig_w,
            self.temporal_patch_size as u32,
            (self.patch_size * self.merge_size) as u32,
            self.shortest_edge as u32,
            self.longest_edge as u32,
            None,
        )?;

        println!("[System] 이미지 처리: 원본 {}x{} -> 리사이즈 {}x{}", orig_w, orig_h, target_w, target_h);

        // Resize image
        let img = img.resize_exact(target_w, target_h, image::imageops::FilterType::Lanczos3);

        let target_h = target_h as usize;
        let target_w = target_w as usize;
        // Convert to RGB and normalize
        let img = img.to_rgb8();
        let pixels: Vec<f32> = img
            .pixels()
            .flat_map(|p| {
                vec![
                    p[0] as f32 / 255.0,
                    p[1] as f32 / 255.0,
                    p[2] as f32 / 255.0,
                ]
            })
            .collect();

        let tensor = Tensor::from_vec(pixels, (target_h, target_w, 3), &self.device)?;

        let mean = Tensor::new(self.image_mean.clone(), &self.device)?.reshape((1, 1, 3))?;
        let std = Tensor::new(self.image_std.clone(), &self.device)?.reshape((1, 1, 3))?;
        let tensor = tensor.broadcast_sub(&mean)?.broadcast_div(&std)?;

        // Now reshape into flattened patches like Python
        let grid_h = target_h / self.patch_size;
        let grid_w = target_w / self.patch_size;
        let channels = 3;
        let tensor =
            tensor.reshape((grid_h, self.patch_size, grid_w, self.patch_size, channels))?;
        let tensor = tensor.permute((0, 2, 4, 1, 3))?;
        let num_patches = grid_h * grid_w;
        let tensor = tensor.reshape((num_patches, channels, self.patch_size, self.patch_size))?;
        let tensor = tensor.unsqueeze(2)?;
        let tensor = tensor.repeat((1, 1, self.temporal_patch_size, 1, 1))?;
        let patch_dim = channels * self.temporal_patch_size * self.patch_size * self.patch_size;
        let tensor = tensor.reshape((num_patches, patch_dim))?;

        let tensor = tensor.to_dtype(self.dtype)?;

        Ok(ProcessedImage {
            pixel_values: tensor,
            grid_h,
            grid_w,
        })
    }

    /// Process image and text for multimodal input
    pub fn process_info(
        &self,
        image_path: &str,
        prompt: &str,
        tokenizer: &TokenizerModel,
        image_token_id: u32,
        image_start_token_id: u32,
        image_end_token_id: u32,
        _patch_size: usize,
        _temporal_patch_size: usize,
        spatial_merge_size: usize,
    ) -> Result<ProcessedInput> {
        let processed_image = self.process_image(image_path)?;
        let pixel_values = processed_image.pixel_values;
        let grid_h = processed_image.grid_h;
        let grid_w = processed_image.grid_w;

        // After spatial merge, each spatial_merge_size x spatial_merge_size block becomes 1 token
        let merged_h = grid_h / spatial_merge_size;
        let merged_w = grid_w / spatial_merge_size;
        let num_image_tokens = merged_h * merged_w;

        // 🚀 하드코딩된 구형 토큰 ID 대신, 토크나이저에서 현재 GLM-4 모델의 토큰 ID를 동적으로 조회합니다.
        // 특정 토큰이 없을 경우 가장 안전한 기본값(1513xx 대역)을 사용하여 모델 붕괴를 방지합니다.
        let gmask_id = tokenizer.token_to_id("[gMASK]").unwrap_or(151331);
        let sop_id = tokenizer.token_to_id("<sop>").unwrap_or(151333); // 올바른 GLM-4 <sop> 토큰 ID
        let user_id = tokenizer.token_to_id("<|user|>").unwrap_or(151336);
        let assistant_id = tokenizer.token_to_id("<|assistant|>").unwrap_or(151337);
        
        // 줄바꿈 토큰을 정확하게 인코딩하여 컨텍스트 분리를 명확히 합니다.
        let nl_id = tokenizer.text_encode_vec("\n".to_string(), false).unwrap_or(vec![198]).into_iter().last().unwrap_or(198);

        // Header 구성을 GLM-OCR 표준 규격으로 재조합합니다.
        let mut input_ids_vec = vec![gmask_id, sop_id, user_id, nl_id, image_start_token_id];

        for _ in 0..num_image_tokens {
            input_ids_vec.push(image_token_id); // <|image|>
        }

        input_ids_vec.push(image_end_token_id); // <|end_of_image|>
        input_ids_vec.push(nl_id); // 이미지 토큰 직후 줄바꿈 추가 (표준 규격)

        // Text prompt (without special tokens - they're already added)
        let text_ids = tokenizer.text_encode_vec(prompt.to_string(), false)?;
        input_ids_vec.extend(text_ids);

        // Generation prompt: <|assistant|>
        input_ids_vec.push(assistant_id); // <|assistant|>
        // <|assistant|> 뒤의 줄바꿈 제거 (생성 방해 요인 제거)

        let input_ids = Tensor::from_vec(
            input_ids_vec.clone(),
            (1, input_ids_vec.len()),
            &self.device,
        )?;

        // Create image mask (1s at image token positions, 0 elsewhere)
        // Image tokens start after header (4 tokens) + start token (1 token) = index 5
        let mut image_mask_vec = vec![0u32; input_ids_vec.len()];
        let image_start_idx = 5; // After [gMASK, sop, user, newline, begin_image]
        for i in 0..num_image_tokens {
            image_mask_vec[image_start_idx + i] = 1;
        }
        let image_mask = Tensor::from_vec(image_mask_vec, (1, input_ids_vec.len()), &self.device)?;

        // Compute grid_thw for RoPE
        let grid_thw =
            Tensor::from_vec(vec![1u32, grid_h as u32, grid_w as u32], (3,), &self.device)?;

        Ok(ProcessedInput {
            input_ids,
            pixel_values,
            image_mask,
            grid_thw,
        })
    }
}
