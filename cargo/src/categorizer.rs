pub struct Categorizer;

impl Categorizer {
    pub fn new(_client: ()) -> Self {
        Self
    }

    pub async fn preprocess_web(&self, _yaml_content: &str) -> Result<serde_json::Value, anyhow::Error> {
        Ok(serde_json::json!({"domain": "DRAFT"}))
    }

    pub async fn preprocess_image(&self, _mime_type: &str, _base64_data: &str) -> Result<serde_json::Value, anyhow::Error> {
        Ok(serde_json::json!({
            "domain": "DRAFT",
            "ocr_full_text": "[System] Gemini 서비스 비활성화됨"
        }))
    }
}
