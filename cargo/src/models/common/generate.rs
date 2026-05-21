use anyhow::Result;
use candle_core::{Device, Tensor};
use crate::models::common::{InferenceModel, MultiModalData};
use crate::tokenizer::TokenizerModel;
use crate::params::chat::{ChatCompletionResponse, ChatCompletionChunkResponse, Choice, ChatMessage}; // Message 제거
use candle_core::IndexOp; // IndexOp 추가
use futures::stream::Stream;
use std::pin::Pin;
use candle_transformers::generation::LogitsProcessor; // 샘플링을 위한 LogitsProcessor 추가

pub struct GenerationContext {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
    pub input_len: usize,
    pub max_tokens: usize,
    pub device: Device,
}

impl GenerationContext {
    pub fn new(
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
        repeat_penalty: Option<f32>,
        repeat_last_n: Option<usize>,
        seed: u64,
        input_len: usize,
        max_tokens: usize,
        device: Device,
    ) -> Self {
        Self {
            temperature,
            top_p,
            top_k,
            repeat_penalty: repeat_penalty.unwrap_or(1.0),
            repeat_last_n: repeat_last_n.unwrap_or(64),
            seed,
            input_len,
            max_tokens,
            device,
        }
    }
}

pub fn generate_generic(
    model: &mut dyn InferenceModel,
    tokenizer: &TokenizerModel,
    input_ids: Tensor,
    data: MultiModalData,
    ctx: &mut GenerationContext,
    _model_name: &str,
) -> Result<ChatCompletionResponse> {
    // [Fix] 매 생성 시작 시 반드시 이전 상태(KV Cache)를 초기화하여 아이템 간 간섭(병합 현상)을 방지합니다.
    model.clear_cache();
    
    let mut tokens = Vec::new();
    let mut current_input_ids = input_ids;
    let mut seqlen_offset = 0;

    // 전달된 Temperature와 Top-P 파라미터를 기반으로 LogitsProcessor 초기화
    let temp = if let Some(t) = ctx.temperature {
        if t < 1e-7 { None } else { Some(t) }
    } else {
        None
    };
    let mut logits_processor = LogitsProcessor::new(ctx.seed, temp, ctx.top_p);

    let logits = model.forward_initial(&current_input_ids, seqlen_offset, data)?;
    seqlen_offset += current_input_ids.dim(1)?;
    
    // LogitsProcessor를 통한 샘플링 수행
    let last_logits = logits.i((0, logits.dim(1)? - 1))?;
    let mut next_token = logits_processor.sample(&last_logits)?;
    tokens.push(next_token);

    for _ in 1..ctx.max_tokens {
        if model.stop_token_ids().contains(&next_token) {
            break;
        }
        current_input_ids = Tensor::new(&[next_token], &ctx.device)?.unsqueeze(0)?;
        let logits = model.forward_step(&current_input_ids, seqlen_offset)?;
        seqlen_offset += 1;
        
        let next_logits = logits.i((0, 0))?;
        next_token = logits_processor.sample(&next_logits)?;
        tokens.push(next_token);
    }

    let text = tokenizer.decode(&tokens, true)?;
    Ok(ChatCompletionResponse {
        id: "chat-id".to_string(),
        object: "chat.completion".to_string(),
        created: 0,
        model: "glm-ocr".to_string(),
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: text,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Default::default(),
    })
}

pub fn generate_stream_generic(
    model: &mut dyn InferenceModel,
    _tokenizer: &TokenizerModel,
    _input_ids: Tensor,
    _data: MultiModalData,
    _temperature: Option<f64>,
    _top_p: Option<f64>,
    _top_k: Option<usize>,
    _repeat_penalty: Option<f32>,
    _repeat_last_n: Option<usize>,
    _seed: u64,
    _max_tokens: usize,
    _is_json: bool,
    _device: &Device,
    _model_name: &str,
) -> Result<Pin<Box<dyn Stream<Item = Result<ChatCompletionChunkResponse>> + Send>>> {
    // [Fix] 스트리밍 생성 시에도 캐시 초기화를 보장합니다.
    model.clear_cache();
    // Stub for now
    Err(anyhow::anyhow!("Streaming not implemented yet"))
}