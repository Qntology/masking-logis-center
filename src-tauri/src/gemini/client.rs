use reqwest::Client;
use serde_json::json;
use futures::StreamExt;
use super::types::ChatMessage;
use super::auth::get_auth_token;

pub struct GeminiClient {
    client: Client,
    model: String,
}

impl GeminiClient {
    pub fn new(model: String) -> Self {
        Self {
            client: Client::new(),
            model,
        }
    }

    async fn get_auth_header(&self) -> Result<String, Box<dyn std::error::Error>> {
        // 실제 구현에서는 여기서 get_auth_token()으로 토큰을 얻고 필요시 갱신합니다.
        Ok(get_auth_token()?)
    }

    pub async fn stream_message<F>(&self, messages: Vec<ChatMessage>, mut on_chunk: F) -> Result<(), Box<dyn std::error::Error>> 
    where F: FnMut(String) {
        let token = self.get_auth_header().await?;
        let url = format!("https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent?alt=sse", self.model);
        
        let payload = json!({
            "contents": messages,
        });
let mut stream = self.client.post(url)
    .bearer_auth(token)
    .json(&payload)
    .send()
    .await?
    .bytes_stream();

while let Some(item) = stream.next().await {
    let chunk = item?;
    let text = String::from_utf8_lossy(&chunk);
    for line in text.lines() {
        if line.starts_with("data: ") {
            let json_str = &line[6..];
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str) {
                if let Some(text_part) = v["candidates"][0]["content"]["parts"][0]["text"].as_str() {
                    on_chunk(text_part.to_string());
                }
            }
        }
    }
}

        Ok(())
    }
}
