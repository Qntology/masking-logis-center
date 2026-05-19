use crate::domain::Domain;
use crate::gemini::client::GeminiClient; // Assuming a GeminiClient exists

pub struct Categorizer {
    client: GeminiClient,
}

impl Categorizer {
    pub fn new(client: GeminiClient) -> Self {
        Self { client }
    }

    pub async fn classify_text(&self, text: &str) -> Result<Domain, anyhow::Error> {
        let prompt = format!(
            "Classify the following text into one of these domains: COMMERCE, LOGISTICS, TRADE.\n\
             Return ONLY the domain name.\n\
             Text: {}",
            text
        );

        let response = self.client.generate_content(&prompt).await?;
        let domain_str = response.trim();
        
        Domain::from_str(domain_str)
            .ok_or_else(|| anyhow::anyhow!("Failed to classify domain: {}", domain_str))
    }
}
