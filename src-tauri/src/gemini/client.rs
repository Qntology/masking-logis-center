use reqwest::Client;
use serde_json::json;
use futures::StreamExt;
use super::types::ChatMessage;
use super::auth::get_auth_token;

#[derive(Clone)]
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
        // 토큰 발급 및 만료 시 자동 갱신을 수행합니다.
        Ok(get_auth_token().await?)
    }

    async fn get_project_id(&self, token: &str) -> anyhow::Result<String> {
        let url = "https://cloudcode-pa.googleapis.com/v1internal:loadCodeAssist";
        let payload = json!({"metadata": {}});
        let res = self.client.post(url)
            .bearer_auth(token)
            .json(&payload)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!(e))?;

        if !res.status().is_success() {
            let err = res.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Project discovery failed: {}", err));
        }

        let v: serde_json::Value = res.json().await.map_err(|e| anyhow::anyhow!(e))?;
        if let Some(project) = v["cloudaicompanionProject"].as_str() {
            Ok(project.to_string())
        } else {
            Err(anyhow::anyhow!("No cloudaicompanionProject found in response"))
        }
    }

    pub async fn generate_content(&self, prompt: &str) -> anyhow::Result<String> {
        let token = self.get_auth_header().await.map_err(|e| anyhow::anyhow!(e.to_string()))?;
        
        let is_oauth = token.starts_with("ya29.") || token.starts_with("ya29_");
        
        let url = if is_oauth {
            "https://cloudcode-pa.googleapis.com/v1internal:generateContent".to_string()
        } else {
            format!("https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent", self.model)
        };
        
        let payload = if is_oauth {
            let project_id = self.get_project_id(&token).await?;
            json!({
                "model": self.model,
                "project": project_id,
                "request": {
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }]
                }
            })
        } else {
            json!({
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            })
        };

        let request = self.client.post(url).json(&payload);
        let request = if is_oauth {
            request.bearer_auth(&token)
        } else {
            request.header("x-goog-api-key", &token)
        };

        let response = request.send().await.map_err(|e| anyhow::anyhow!(e))?;

        let v: serde_json::Value = response.json().await.map_err(|e| anyhow::anyhow!(e))?;
        
        let text = if is_oauth {
            v["response"]["candidates"][0]["content"]["parts"][0]["text"].as_str()
        } else {
            v["candidates"][0]["content"]["parts"][0]["text"].as_str()
        }.ok_or_else(|| anyhow::anyhow!("Failed to extract text from Gemini response"))?;

        Ok(text.to_string())
    }

    pub async fn stream_message<F>(&self, messages: Vec<ChatMessage>, mut on_chunk: F) -> Result<(), Box<dyn std::error::Error>> 
    where F: FnMut(String) {
        let token = self.get_auth_header().await?;
        
        let is_oauth = token.starts_with("ya29.") || token.starts_with("ya29_");
        
        let url = if is_oauth {
            "https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse".to_string()
        } else {
            format!("https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent?alt=sse", self.model)
        };
        
        let payload = if is_oauth {
            let project_id = self.get_project_id(&token).await.map_err(|e| e.to_string())?;
            json!({
                "model": self.model,
                "project": project_id,
                "request": {
                    "contents": messages
                }
            })
        } else {
            json!({
                "contents": messages,
            })
        };

        let request = self.client.post(url).json(&payload);
        let request = if is_oauth {
            request.bearer_auth(&token)
        } else {
            request.header("x-goog-api-key", &token)
        };

        let response = request.send().await?;

        // HTTP 요청이 실패한 경우 스트림을 열지 않고 즉시 에러 반환
        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("API Error: {}", error_text).into());
        }

        let mut stream = response.bytes_stream();

        while let Some(item) = stream.next().await {
            let chunk = item?;
            let text = String::from_utf8_lossy(&chunk);
            for line in text.lines() {
                if line.starts_with("data: ") {
                    let json_str = &line[6..];
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str) {
                        let candidates = if is_oauth {
                            &v["response"]["candidates"]
                        } else {
                            &v["candidates"]
                        };
                        
                        if let Some(text_part) = candidates[0]["content"]["parts"][0]["text"].as_str() {
                            on_chunk(text_part.to_string());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
