use crate::db::{search_context};
use crate::privacy_filter::masking::{PrivacyManager, PrivacySession};
use anyhow::Result;
use candle_core::Device;

pub struct Assistant {
    privacy_manager: PrivacyManager,
}

impl Assistant {
    pub fn new(_client: (), privacy_model_dir: &str) -> Result<Self> {
        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        let privacy_manager = PrivacyManager::new(privacy_model_dir, &device)?;
        Ok(Self {
            privacy_manager,
        })
    }

    pub async fn answer_question(&self, query: &str) -> Result<String> {
        // 1. 도메인 분류 (Intent Routing) 제거 - 외부 config.json 로드 후 바로 검색
        let config_path = std::path::PathBuf::from("data/config.json");
        let loaded_domain = if let Ok(content) = std::fs::read_to_string(config_path) {
            let v: serde_json::Value = serde_json::from_str(&content).unwrap_or_default();
            let tab = v["default_tab"].as_str().unwrap_or("COMMERCE").to_uppercase();
            // DRAFT 탭일 경우 검색 도메인은 기본값인 COMMERCE로 설정
            if tab == "DRAFT" { "COMMERCE".to_string() } else { tab }
        } else {
            "COMMERCE".to_string()
        };
        
        println!("[Assistant] Bypassing Domain Classification. Searching directly in LanceDB (Default: {}).", loaded_domain);
        let domain_str = loaded_domain.as_str();

        // 2. 필터링된 검색 (LanceDB Hybrid Search)
        let records = search_context(query, Some(domain_str)).await?;
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
                let label = span.entity_group.to_uppercase();
                if matches!(label.as_str(), "CITY" | "COUNTY" | "STATE") {
                    continue; 
                }
                let placeholder = privacy_session.get_or_create_placeholder(&span.word, &label, idx);
                masked_text.replace_range(span.start..span.end, &placeholder);
            }
            masked_records.push(masked_text);
        }

        // 4. LLM 컨텍스트 구성 및 질의 (Gemini 제거로 인한 스텁 답변)
        let _context_str = masked_records.join("\n---\n");
        let _system_instruction = format!(
            "너는 데이터베이스 전문가 에이전트야. 다음 제공된 [{}] 도메인 데이터를 절대적인 사실로 삼아 사용자의 질문에 답변해.\n\n[도메인 데이터 시작]\n{}\n[도메인 데이터 끝]\n\n답변 시 마스킹된 ID(예: [RECORD_0][NAME_1])를 원본 그대로 사용해.",
            domain_str, _context_str
        );

        // Gemini 모듈 삭제로 인한 임시 반환
        Ok("[System] Gemini 서비스가 비활성화되었습니다. 로컬 모델로의 전환이 필요합니다.".to_string())
    }
}
