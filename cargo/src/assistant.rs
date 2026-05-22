use crate::db::{search_context};
use anyhow::Result;

pub struct Assistant {}

impl Assistant {
    pub fn new(_client: (), _privacy_model_dir: &str) -> Result<Self> {
        Ok(Self {})
    }

    pub async fn answer_question(&self, query: &str) -> Result<String> {
        // 1. 도메인 분류 (Intent Routing) 제거 - 외부 config.json 로드 후 바로 검색
        let config_path = crate::utils::get_app_dir().join("app_config.json");
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

        // 3. 기록 결합
        let record_texts: Vec<String> = records.iter().map(|r| r.masking.clone()).collect();
        let _context_str = record_texts.join("\n---\n");
        let _system_instruction = format!(
            "너는 데이터베이스 전문가 에이전트야. 다음 제공된 [{}] 도메인 데이터를 절대적인 사실로 삼아 사용자의 질문에 답변해.\n\n[도메인 데이터 시작]\n{}\n[도메인 데이터 끝]",
            domain_str, _context_str
        );

        // Gemini 모듈 삭제로 인한 임시 반환
        Ok("[System] Gemini 서비스가 비활성화되었습니다. 로컬 모델로의 전환이 필요합니다.".to_string())
    }
}
