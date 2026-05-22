use crate::models::common::MultiModalData;
use crate::models::common::generate::{
    GenerationContext, generate_generic, generate_stream_generic,
};
use crate::params::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use rocket::futures::Stream;

use crate::models::qwen3::config::{Qwen3Config, Qwen3GenerationConfig};
use crate::models::qwen3::model::Qwen3Model;
use crate::utils::{find_type_files, get_device, get_dtype};
use crate::{chat_template::ChatTemplate, models::GenerateModel, tokenizer::TokenizerModel};

pub struct Qwen3GenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    qwen3: Qwen3Model,
    device: Device,
    generation_config: Qwen3GenerationConfig,
    model_name: String,
}

impl<'a> Qwen3GenerateModel<'a> {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: Qwen3Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = &get_device(device);
        let cfg_dtype = cfg.torch_dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, device)? };
        let generation_config_path = path.to_string() + "/generation_config.json";
        let generation_config: Qwen3GenerationConfig =
            serde_json::from_slice(&std::fs::read(generation_config_path)?)?;
        let qwen3 = Qwen3Model::new(&cfg, vb, generation_config.eos_token_id.clone())?;

        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("qwen3")
            .to_string();
        Ok(Qwen3GenerateModel {
            chat_template,
            tokenizer,
            qwen3,
            device: device.clone(),
            generation_config,
            model_name,
        })
    }
}

impl<'a> GenerateModel for Qwen3GenerateModel<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let temperature = mes
            .temperature
            .unwrap_or(self.generation_config.temperature);
        let top_p = mes.top_p.unwrap_or(self.generation_config.top_p);
        let top_k = self.generation_config.top_k;
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input_ids = self.tokenizer.text_encode(mes_render, &self.device)?;
        let sample_len = mes.max_tokens.unwrap_or(2048);
        let mut ctx = GenerationContext::new(
            temperature.into(),
            top_p.into(),
            top_k.into(),
            mes.repeat_penalty,
            mes.repeat_last_n,
            seed,
            input_ids.dim(1)?,
            sample_len,
            self.device.clone(),
        );

        let data = MultiModalData::new(vec![]);
        generate_generic(
            &mut self.qwen3,
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
        let temperature = mes
            .temperature
            .unwrap_or(self.generation_config.temperature);
        let top_p = mes.top_p.unwrap_or(self.generation_config.top_p);
        let top_k = self.generation_config.top_k;
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let in_reasoning = mes_render.ends_with("<think>\n");
        let input_ids = self.tokenizer.text_encode(mes_render, &self.device)?;
        let data = MultiModalData::new(vec![]);
        let sample_len = mes.max_tokens.unwrap_or(512);
        let stream = generate_stream_generic(
            &mut self.qwen3,
            &self.tokenizer,
            input_ids,
            data,
            temperature.into(),
            top_p.into(),
            top_k.into(),
            mes.repeat_penalty,
            mes.repeat_last_n,
            seed,
            sample_len,
            in_reasoning,
            &self.device,
            &self.model_name,
        )?;
        Ok(Box::new(Box::pin(stream)))
    }
}
