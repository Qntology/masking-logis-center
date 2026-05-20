//! GLM-OCR Inference and Generation
use crate::{
    models::common::{
        MultiModalData,
        generate::{GenerationContext, generate_generic, generate_stream_generic},
    },
    params::chat::{ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse},
};
use anyhow::{Result, anyhow};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use rocket::futures::Stream;

use crate::{
    // chat_template::ChatTemplate,
    models::{
        GenerateModel,
        glm_ocr::{
            config::{GlmOcrConfig, GlmOcrGenerationConfig},
            model::GlmOcrModel,
            processor::GlmOcrProcessor,
        },
    },
    tokenizer::TokenizerModel,
    utils::{
        extract_user_text, find_type_files, get_device, get_dtype, img_utils::extract_image_url,
    },
};

pub struct GlmOcrGenerateModel {
    // chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    processor: GlmOcrProcessor,
    model: GlmOcrModel,
    device: Device,
    model_name: String,
    image_token_id: u32,
    image_start_token_id: u32,
    image_end_token_id: u32,
    patch_size: usize,
    temporal_patch_size: usize,
    spatial_merge_size: usize,
}

impl GlmOcrGenerateModel {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        // let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: GlmOcrConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let cfg_dtype = cfg.text_config.dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let processor = GlmOcrProcessor::new(path, &device, dtype)?;
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let generation_config_path = path.to_string() + "/generation_config.json";
        let generation_config: GlmOcrGenerationConfig =
            serde_json::from_slice(&std::fs::read(generation_config_path)?)?;
        let model = GlmOcrModel::new(vb, cfg.clone(), generation_config.eos_token_id.clone())?;
        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("ZhipuAI/GLM-OCR")
            .to_string();
        Ok(Self {
            // chat_template,
            tokenizer,
            processor,
            model,
            device,
            model_name,
            image_token_id: cfg.image_token_id,
            image_start_token_id: cfg.image_start_token_id,
            image_end_token_id: cfg.image_end_token_id,
            patch_size: cfg.vision_config.patch_size,
            temporal_patch_size: cfg.vision_config.temporal_patch_size,
            spatial_merge_size: cfg.vision_config.spatial_merge_size,
        })
    }
}

impl GenerateModel for GlmOcrGenerateModel {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let seed = mes.seed.unwrap_or(34562) as u64;
        // Extract image path and prompt from messages
        let image_urls = extract_image_url(&mes);
        let image_path = image_urls
            .first()
            .ok_or_else(|| anyhow!("No image provided"))?;

        // Get prompt text from messages
        let mut prompt = extract_user_text(&mes)?;
        if prompt.chars().count() == 0 {
            prompt = "Extract all text from this image.".to_string()
        }

        let processed = self.processor.process_info(
            image_path,
            &prompt,
            &self.tokenizer,
            self.image_token_id,
            self.image_start_token_id,
            self.image_end_token_id,
            self.patch_size,
            self.temporal_patch_size,
            self.spatial_merge_size,
        )?;

        let input_ids = processed.input_ids;
        let sample_len = mes.max_tokens.unwrap_or(1024);
        let mut ctx = GenerationContext::new(
            mes.temperature,
            mes.top_p,
            mes.top_k,
            mes.repeat_penalty,
            mes.repeat_last_n,
            seed,
            input_ids.dim(1)?,
            sample_len,
            self.device.clone(),
        );

        let data_vec = vec![
            processed.pixel_values.into(),
            processed.grid_thw.into(),
            processed.image_mask.into(),
        ];
        let data = MultiModalData::new(data_vec);
        generate_generic(
            &mut self.model,
            &self.tokenizer,
            input_ids,
            data,
            &mut ctx,
            &self.model_name,
        )
    }

    fn generate_stream(
        &mut self,
        mes: ChatCompletionParameters,
    ) -> Result<
        Box<
            dyn Stream<Item = Result<ChatCompletionChunkResponse, anyhow::Error>>
                + Send
                + Unpin
                + '_,
        >,
    > {
        let seed = mes.seed.unwrap_or(34562) as u64;
        // Extract image path and prompt from messages
        let image_urls = extract_image_url(&mes);
        let image_path = image_urls
            .first()
            .ok_or_else(|| anyhow!("No image provided"))?;

        // Get prompt text from messages
        let mut prompt = extract_user_text(&mes)?;
        if prompt.chars().count() == 0 {
            prompt = "Extract all text from this image.".to_string()
        }

        let processed = self.processor.process_info(
            image_path,
            &prompt,
            &self.tokenizer,
            self.image_token_id,
            self.image_start_token_id,
            self.image_end_token_id,
            self.patch_size,
            self.temporal_patch_size,
            self.spatial_merge_size,
        )?;

        let input_ids = processed.input_ids;
        let sample_len = mes.max_tokens.unwrap_or(1024);
        let data_vec = vec![
            processed.pixel_values.into(),
            processed.grid_thw.into(),
            processed.image_mask.into(),
        ];
        let data = MultiModalData::new(data_vec);
        let stream = generate_stream_generic(
            &mut self.model,
            &self.tokenizer,
            input_ids,
            data,
            mes.temperature,
            mes.top_p,
            None,
            mes.repeat_penalty,
            mes.repeat_last_n,
            seed,
            sample_len,
            false,
            &self.device,
            &self.model_name,
        )?;
        Ok(Box::new(Box::pin(stream)))
    }
}
