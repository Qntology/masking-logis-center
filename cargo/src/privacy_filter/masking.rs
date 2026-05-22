use anyhow::Result;
use std::path::Path;
use std::collections::HashMap;
use std::fs;
use lazy_static::lazy_static;
use crate::privacy_filter::PrivacyFilterModel;

lazy_static! {
    static ref ADJECTIVES: Vec<String> = {
        // 🚀 파일이 없더라도 앱이 죽지 않도록 예외 처리 폴백을 추가하고, AppData 경로를 참조합니다.
        let adj_path = crate::utils::get_app_dir().join("adjectives.txt");
        let content = fs::read_to_string(&adj_path).unwrap_or_else(|_| "awesome\nfast\namazing\ngood\nbeautiful".to_string());
        
        let words: Vec<String> = content.lines()
            .map(|s| s.split_whitespace().last().unwrap_or("").to_string())
            .filter(|s| !s.is_empty())
            .collect();
            
        if words.is_empty() {
            panic!("[에러] adjectives.txt 파일이 비어있습니다!");
        }
        words
    };
    
    static ref NOUNS: Vec<String> = {
        // 🚀 파일이 없더라도 앱이 죽지 않도록 예외 처리 폴백을 추가하고, AppData 경로를 참조합니다.
        let noun_path = crate::utils::get_app_dir().join("nouns.txt");
        let content = fs::read_to_string(&noun_path).unwrap_or_else(|_| "apple\ncar\ncat\nairplane\nspaceship".to_string());
        
        let words: Vec<String> = content.lines()
            .map(|s| s.split_whitespace().last().unwrap_or("").to_string())
            .filter(|s| !s.is_empty())
            .collect();
            
        if words.is_empty() {
            panic!("[에러] nouns.txt 파일이 비어있습니다!");
        }
        words
    };
}

pub struct PrivacyManager {
    pub model: PrivacyFilterModel,
}

pub struct PrivacySession {
    entity_map: HashMap<String, String>, // Original text -> Placeholder
    reverse_map: HashMap<String, String>, // Placeholder -> Original text
    counter_map: HashMap<String, usize>, // Label -> Count
}

impl PrivacySession {
    pub fn new() -> Self {
        Self {
            entity_map: HashMap::new(),
            reverse_map: HashMap::new(),
            counter_map: HashMap::new(),
        }
    }

    pub fn get_or_create_placeholder(&mut self, text: &str, label: &str, record_idx: usize) -> String {
        // 🚀 [정규화] 앞뒤 공백을 제거하여 모델이 미세하게 다르게 인식한 동일 단어를 하나로 묶습니다.
        let normalized_text = text.trim();
        
        let mut clean_label = if label.contains('-') {
            label.split('-').nth(1).unwrap_or(label).to_lowercase()
        } else {
            label.to_lowercase()
        };

        // 🚀 [주소 마스킹 통일] 상세 주소(Street, BuildingNumber, ZipCode)를 'address' 태그 하나로 묶어줍니다.
        if matches!(clean_label.as_str(), "street" | "buildingnumber" | "zipcode") {
            clean_label = "address".to_string();
        }

        // 🚀 [재사용 로직] 이미 세션 맵에 등록된 단어라면 즉시 기존 플레이스홀더를 반환합니다.
        // 이를 통해 drag.context 내의 반복되는 주소가 모두 동일한 태그로 통일됩니다.
        if let Some(placeholder) = self.entity_map.get(normalized_text) {
            return placeholder.clone();
        }

        let count = self.counter_map.entry(clean_label.clone()).or_insert(0);
        let current_idx = *count;
        *count += 1;
        
        // 대용량 외부 파일(adjectives, nouns) 참조
        let adj = &ADJECTIVES[current_idx % ADJECTIVES.len()];
        let noun = &NOUNS[(current_idx / ADJECTIVES.len()) % NOUNS.len()];
        
        // 고유 태그 생성
        let placeholder = format!("[RECORD_{}][{}:{}-{}]", record_idx, clean_label, adj, noun);
        
        // 🚀 정규화된 텍스트를 키로 사용하여 맵에 저장합니다.
        self.entity_map.insert(normalized_text.to_string(), placeholder.clone());
        self.reverse_map.insert(placeholder.clone(), normalized_text.to_string());
        placeholder
    }

    pub fn unmask_text(&self, text: &str) -> String {
        let mut result = text.to_string();
        let mut placeholders: Vec<_> = self.reverse_map.keys().collect();
        placeholders.sort_by(|a, b| b.len().cmp(&a.len()));

        for placeholder in placeholders {
            if let Some(original) = self.reverse_map.get(placeholder) {
                result = result.replace(placeholder, original);
            }
        }
        result
    }
}

impl PrivacyManager {
    pub fn new(model_dir: &str, device: &candle_core::Device) -> Result<Self> {
        let model = PrivacyFilterModel::load(Path::new(model_dir), device)?;
        Ok(Self { model })
    }

    // 🚀 [구조 개선] session과 record_idx를 외부에서 주입받아, 여러 문서를 개별적으로 처리할 때도 일관성을 유지하게 합니다.
    pub fn mask_text_with_session(&self, text: &str, session: &mut PrivacySession, record_idx: usize) -> Result<String> {
        println!("[PrivacyManager] 텍스트 길이 {} 바이트, predict 호출 진입", text.len());
        
        if text.trim().is_empty() {
            println!("[PrivacyManager] 텍스트가 비어있어 마스킹을 건너뜁니다.");
            return Ok(text.to_string());
        }

        let spans = match self.model.predict(text) {
            Ok(s) => {
                println!("[PrivacyManager] predict 성공, {} 개의 식별된 Span 찾음", s.len());
                s
            },
            Err(e) => {
                println!("[PrivacyManager] predict 내부 모델 추론 중 에러 발생: {:?}", e);
                return Err(e);
            }
        };
        
        let mut masked_text = text.to_string();
        let mut sorted_spans = spans;
        sorted_spans.sort_by(|a, b| b.start.cmp(&a.start));

        for span in sorted_spans {
            let label = span.entity_group.to_uppercase();
            
            // 🚀 [지능형 필터링 고도화] 
            // 1. 신뢰도 대폭 상향: 0.85 미만은 버려서 상품명(클립, 파우치 등)이나 일반 명사의 오인식을 원천 차단합니다.
            // 2. 공백 제거 후 길이 검사: " 지", " 준" 처럼 토큰 앞뒤에 공백이 포함된 1글자 조사/단어의 피해를 막습니다.
            if span.score < 0.85 || span.word.trim().chars().count() <= 1 {
                continue;
            }

            // 3. 비핵심 정보 제외: 문맥 보존을 위해 금액(AMOUNT)은 마스킹에서 통과시킵니다.
            // 🚀 시/도/군/구(CITY, COUNTY, STATE)는 맥락 파악을 위해 남겨두고, 상세 주소(STREET, ZIPCODE, BUILDINGNUMBER)는 마스킹되도록 통과 목록에서 제거합니다.
            if matches!(label.as_str(), "CITY" | "COUNTY" | "STATE" | "AMOUNT") {
                continue; 
            }

            let placeholder = session.get_or_create_placeholder(&span.word, &label, record_idx);
            
            if span.start < masked_text.len() && span.end <= masked_text.len() {
                masked_text.replace_range(span.start..span.end, &placeholder);
            }
        }
        Ok(masked_text)
    }

    pub fn mask_records(&self, texts: Vec<String>) -> Result<Vec<String>> {
        let mut session = PrivacySession::new();
        let mut results = Vec::new();
        for (idx, text) in texts.into_iter().enumerate() {
            results.push(self.mask_text_with_session(&text, &mut session, idx).unwrap_or(text));
        }
        Ok(results)
    }

    pub fn mask_text(&self, text: &str) -> Result<String> {
        let mut session = PrivacySession::new();
        self.mask_text_with_session(text, &mut session, 0)
    }
}
