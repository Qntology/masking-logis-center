use crate::gemini::client::GeminiClient;

pub struct Categorizer {
    client: GeminiClient,
}

impl Categorizer {
    pub fn new(client: GeminiClient) -> Self {
        Self { client }
    }

    // 1. 웹페이지 전처리 파이프라인 (YAML -> Domain JSON)
    pub async fn preprocess_web(&self, yaml_content: &str) -> Result<serde_json::Value, anyhow::Error> {
        let prompt = format!(
            "[TASK]\n\
             You are a classification parser.\n\
             Analyze the provided YAML content and classify it into exactly one of the following domains: COMMERCE, LOGISTICS, TRADE, DRAFT.\n\
             You MUST respond with a valid JSON object containing ONLY the \"domain\" key. Do not include markdown formatting.\n\n\
             [INPUT]\n\
             {}",
            yaml_content
        );

        let response = self.client.generate_content(&prompt, None).await?;
        let clean_json = response.trim().trim_matches(|c| c == '`' || c == '\n');
        let parsed: serde_json::Value = serde_json::from_str(clean_json).unwrap_or_else(|_| serde_json::json!({"domain": "DRAFT"}));
        
        Ok(parsed)
    }

    // 2. 이미지 전처리 파이프라인 (Image -> Domain & OCR JSON)
    pub async fn preprocess_image(&self, mime_type: &str, base64_data: &str) -> Result<serde_json::Value, anyhow::Error> {
        let prompt = "[SYSTEM]\n\
             You are an OCR and classification parser.\n\
             Extract all readable text from the provided image exactly as it appears, maintaining line breaks.\n\
             Then, classify the image into one of the allowed domains specified in the [INPUT] block.\n\
             You MUST respond with a valid JSON object containing ONLY \"domain\" and \"ocr_full_text\" keys. Do not include markdown formatting.\n\n\
             [INPUT]\n\
             Allowed Domains: COMMERCE, LOGISTICS, TRADE, DRAFT";

        let response = self.client.generate_content_with_image(prompt, mime_type, base64_data).await?;
        let clean_json = response.trim().trim_matches(|c| c == '`' || c == '\n');
        let parsed: serde_json::Value = serde_json::from_str(clean_json).unwrap_or_else(|_| serde_json::json!({
            "domain": "DRAFT",
            "ocr_full_text": "Failed to parse OCR data."
        }));
        
        Ok(parsed)
    }
}
