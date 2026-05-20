use anyhow::Result;
use std::path::Path;
use std::collections::HashMap;
use crate::privacy_filter::PrivacyFilterModel;

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
        let clean_label = if label.contains('-') {
            label.split('-').nth(1).unwrap_or(label)
        } else {
            label
        };

        if let Some(placeholder) = self.entity_map.get(text) {
            placeholder.clone()
        } else {
            let count = self.counter_map.entry(clean_label.to_string()).or_insert(0);
            *count += 1;
            let placeholder = format!("[RECORD_{}][{}_{}]", record_idx, clean_label, count);
            self.entity_map.insert(text.to_string(), placeholder.clone());
            self.reverse_map.insert(placeholder.clone(), text.to_string());
            placeholder
        }
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
            sorted_spans.sort_by(|a, b| b.start.cmp(&a.start));

            for span in sorted_spans {
                let label = span.entity_group.to_uppercase();
                
                if matches!(label.as_str(), "CITY" | "COUNTY" | "STATE") {
                    continue; 
                }

                let placeholder = session.get_or_create_placeholder(&span.word, &label, idx);
                masked_text.replace_range(span.start..span.end, &placeholder);
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
