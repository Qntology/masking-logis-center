use anyhow::Result;
use tokenizers::Tokenizer;
use std::path::Path;

pub struct TokenizerModel {
    tokenizer: Tokenizer,
}

impl TokenizerModel {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer }
    }

    pub fn init<P: AsRef<Path>>(path: P) -> Result<Self> {
        let target_path = path.as_ref().join("tokenizer.json");
        
        // 🚀 정확히 어떤 경로에서 파일을 찾으려 하는지 시스템 콘솔에 로그를 남깁니다.
        println!("[System] Tokenizer 로드 시도 경로: {}", target_path.display());
        
        let tokenizer = Tokenizer::from_file(&target_path).map_err(|e| {
            // 🚀 에러 발생 시 지정된 파일을 찾을 수 없다는 메시지와 함께 절대 경로를 출력합니다.
            anyhow::anyhow!("Tokenizer 파일 로드 실패 (경로: {}): {}", target_path.display(), e)
        })?;
        
        Ok(Self { tokenizer })
    }

    pub fn text_encode_vec(&self, text: String, add_special_tokens: bool) -> Result<Vec<u32>> {
        let tokens = self.tokenizer.encode(text, add_special_tokens).map_err(anyhow::Error::msg)?;
        Ok(tokens.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer.decode(ids, skip_special_tokens).map_err(anyhow::Error::msg)
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    pub fn token_decode(&self, ids: Vec<u32>) -> Result<String> { 
        self.decode(&ids, true) 
    }
    
    pub fn text_encode(&self, text: String, device: &candle_core::Device) -> Result<candle_core::Tensor> {
        let vec = self.text_encode_vec(text, false)?;
        let len = vec.len();
        candle_core::Tensor::from_vec(vec, (1, len), device).map_err(anyhow::Error::msg)
    }
}
