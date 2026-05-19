use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub parts: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GeminiConfig {
    pub model: String,
    pub api_key: String,
}
