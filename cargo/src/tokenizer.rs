use anyhow::Result;
use tokenizers::Tokenizer;
use std::path::Path;

pub struct TokenizerModel {
    tokenizer: Tokenizer,
}

impl TokenizerModel {
    pub fn init<P: AsRef<Path>>(path: P) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path.as_ref().join("tokenizer.json")).map_err(anyhow::Error::msg)?;
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
}
