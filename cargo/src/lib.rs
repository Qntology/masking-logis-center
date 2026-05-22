pub mod db;
pub mod embedding;
pub mod model;
pub mod harness;
pub mod queue;
pub mod domain;
pub mod categorizer;
pub mod assistant;

pub mod utils;
pub mod models;
pub mod tokenizer;
pub mod position_embed;
pub mod openai_types;

pub mod chat_template {
    use anyhow::Result;
    pub struct ChatTemplate;
    impl ChatTemplate {
        pub fn init(_path: &str) -> Result<Self> { Ok(Self) }
        pub fn apply_chat_template(&self, _mes: &crate::params::chat::ChatCompletionParameters) -> Result<String> { Ok(String::new()) }
    }
}

pub mod scheduler {
    use once_cell::sync::OnceCell;
    use tokio::sync::mpsc;
    pub static PROGRESS_TX: OnceCell<mpsc::UnboundedSender<serde_json::Value>> = OnceCell::new();
}

pub static CURRENT_UI_CATEGORY: std::sync::RwLock<String> = std::sync::RwLock::new(String::new());

