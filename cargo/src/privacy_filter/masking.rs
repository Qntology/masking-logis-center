use anyhow::Result;
use std::path::Path;
use std::collections::HashMap;
use std::fs;
use lazy_static::lazy_static;
use crate::privacy_filter::PrivacyFilterModel;

lazy_static! {
    static ref ADJECTIVES: Vec<String> = {
        // 🚀 파일이 없으면 서버 실행 시 즉각 에러를 발생시킵니다.
        let content = fs::read_to_string("adjectives.txt").expect("[에러] adjectives.txt 파일이 실행 경로에 없습니다!");
        
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
        // 🚀 파일이 없으면 서버 실행 시 즉각 에러를 발생시킵니다.
        let content = fs::read_to_string("nouns.txt").expect("[에러] nouns.txt 파일이 실행 경로에 없습니다!");
        
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
        
        let clean_label = if label.contains('-') {
            label.split('-').nth(1).unwrap_or(label)
        } else {
            label
        };

        // 🚀 [재사용 로직] 이미 세션 맵에 등록된 단어라면 즉시 기존 플레이스홀더를 반환합니다.
        // 이를 통해 drag.context 내의 반복되는 주소가 모두 동일한 태그로 통일됩니다.
        if let Some(placeholder) = self.entity_map.get(normalized_text) {
            return placeholder.clone();
        }

        let count = self.counter_map.entry(clean_label.to_string()).or_insert(0);
        let current_idx = *count;
        *count += 1;
        
        // 대용량 외부 파일(adjectives, nouns) 참조
        let adj = &ADJECTIVES[current_idx % ADJECTIVES.len()];
        let noun = &NOUNS[(current_idx / ADJECTIVES.len()) % NOUNS.len()];
        
        // 고유 태그 생성
        let placeholder = format!("[RECORD_{}][{}:{}-{}]", record_idx, clean_label.to_lowercase(), adj, noun);
        
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

    pub fn mask_records(&self, texts: Vec<String>) -> Result<Vec<String>> {
        let mut session = PrivacySession::new();
        let mut results = Vec::new();

        for (idx, text) in texts.into_iter().enumerate() {
            println!("[PrivacyManager] 텍스트 길이 {} 바이트, predict 호출 진입", text.len());
            
            // 텍스트가 비어있을 경우 GPU 연산을 수행하지 않고 원본을 바로 반환하도록 예외 처리합니다.
            if text.trim().is_empty() {
                println!("[PrivacyManager] 텍스트가 비어있어 마스킹을 건너뜁니다.");
                results.push(text);
                continue;
            }

            let spans = match self.model.predict(&text) {
                Ok(s) => {
                    println!("[PrivacyManager] predict 성공, {} 개의 식별된 Span 찾음", s.len());
                    s
                },
                Err(e) => {
                    println!("[PrivacyManager] predict 내부 모델 추론 중 에러 발생: {:?}", e);
                    return Err(e);
                }
            };
            let mut masked_text = text.clone();
            
            let mut sorted_spans = spans;
            // 🚀 [인덱스 밀림 해결] 시작 위치(start)를 기준으로 내림차순 정렬하여 뒤에서부터 치환합니다.
            // 문자열 길이가 변하더라도 앞쪽 단어들의 상대적 인덱스가 보존되어 반복되는 주소도 정확히 마스킹됩니다.
            sorted_spans.sort_by(|a, b| b.start.cmp(&a.start));

            for span in sorted_spans {
                let label = span.entity_group.to_uppercase();
                
                // 🚀 [지능형 필터링] 
                // 1. 점수 기반: 모델의 확신도가 0.5 미만인 경우 무지성 마스킹으로 간주하고 무시합니다.
                // 2. 길이 기반: 1글자 이하의 단어(조사 등)가 잘못 식별된 경우 컨텍스트 보존을 위해 제외합니다.
                // 3. 지리 정보: 이미 제외 중인 CITY, STATE 외에 과도하게 잡히는 주소 성분들을 추가 차단합니다.
                if span.score < 0.5 || span.word.chars().count() <= 1 {
                    continue;
                }

                if matches!(label.as_str(), "CITY" | "COUNTY" | "STATE" | "STREET") {
                    continue; 
                }

                let placeholder = session.get_or_create_placeholder(&span.word, &label, idx);
                
                // 🚀 안전한 범위 내에서만 치환을 수행하도록 체크 로직을 보강합니다.
                if span.start < masked_text.len() && span.end <= masked_text.len() {
                    masked_text.replace_range(span.start..span.end, &placeholder);
                }
            }
            results.push(masked_text);
        }

        Ok(results)
    }

    pub fn mask_text(&self, text: &str) -> Result<String> {
        let results = self.mask_records(vec![text.to_string()])?;
        Ok(results[0].clone())
    }
}
