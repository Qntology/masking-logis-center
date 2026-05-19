use crate::db::{search_context, CommerceRecord};
use crate::categorizer::Categorizer;
use crate::privacy_filter::masking::{PrivacyManager, PrivacySession};
use crate::gemini::client::GeminiClient;
use anyhow::Result;

pub struct Assistant {
    categorizer: Categorizer,
    privacy_manager: PrivacyManager,
    gemini_client: GeminiClient,
}

impl Assistant {
    pub fn new(gemini_client: GeminiClient, privacy_model_dir: &str) -> Result<Self> {
        let categorizer = Categorizer::new(gemini_client.clone());
        let privacy_manager = PrivacyManager::new(privacy_model_dir)?;
        Ok(Self {
            categorizer,
            privacy_manager,
            gemini_client,
        })
    }

    pub async fn answer_question(&self, query: &str) -> Result<String> {
        // 1. 도메인 분류 (Intent Routing)
        let domain = self.categorizer.classify_text(query).await?;
        println!("[Assistant] Detected Domain: {:?}", domain);

        // 2. 필터링된 검색 (LanceDB Hybrid Search)
        let records = search_context(query, Some(domain.as_str())).await?;
        if records.is_empty() {
            return Ok("관련된 정보를 찾을 수 없습니다.".to_string());
        }

        // 3. 세션 기반 마스킹 (Reversible Masking)
        let mut privacy_session = PrivacySession::new();
        let record_texts: Vec<String> = records.iter().map(|r| r.context.clone()).collect();
        
        let mut masked_records = Vec::new();
        for (idx, text) in record_texts.into_iter().enumerate() {
            let spans = self.privacy_manager.model.predict(&text)?;
            let mut masked_text = text.clone();
            let mut sorted_spans = spans;
            sorted_spans.sort_by(|a, b| b.start.cmp(&a.start));

            for span in sorted_spans {
                let label = span.label.to_uppercase();
                if matches!(label.as_str(), "B-CITY" | "I-CITY" | "E-CITY" | "S-CITY" | 
                                           "B-COUNTY" | "I-COUNTY" | "E-COUNTY" | "S-COUNTY" |
                                           "B-STATE" | "I-STATE" | "E-STATE" | "S-STATE") {
                    continue; 
                }
                let placeholder = privacy_session.get_or_create_placeholder(&span.text, &label, idx);
                masked_text.replace_range(span.start..span.end, &placeholder);
            }
            masked_records.push(masked_text);
        }

        // 4. LLM 컨텍스트 구성 및 질의
        let context_str = masked_records.join("\n---\n");
        let prompt = format!(
            "다음은 [{:?}] 도메인의 검색 결과입니다:\n\n{}\n\n사용자 질문: {}\n\n위의 정보를 바탕으로 답변해주세요. 답변 시 마스킹된 ID(예: [RECORD_0][NAME_1])를 그대로 사용하세요.",
            domain, context_str, query
        );

        let masked_answer = self.gemini_client.generate_content(&prompt).await?;

        // 5. 마스킹 복구 (Unmasking)
        let final_answer = privacy_session.unmask_text(&masked_answer);

        Ok(final_answer)
    }
}
