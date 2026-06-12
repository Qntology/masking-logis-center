use anyhow::Result;
use std::path::Path;
use std::collections::HashMap;
use std::fs;
use once_cell::sync::Lazy;
use super::PrivacyFilterModel;
use super::viterbi::PrivacySpan;

static ADJECTIVES: Lazy<Vec<String>> = Lazy::new(|| {
    let adj_path = crate::utils::get_app_dir().join("adjectives.txt");
    let content = fs::read_to_string(&adj_path).unwrap_or_else(|_| "awesome\nfast\namazing\ngood\nbeautiful".to_string());
    content.lines()
        .map(|s| s.split_whitespace().last().unwrap_or("").to_string())
        .filter(|s| !s.is_empty())
        .collect()
});

static NOUNS: Lazy<Vec<String>> = Lazy::new(|| {
    let noun_path = crate::utils::get_app_dir().join("nouns.txt");
    let content = fs::read_to_string(&noun_path).unwrap_or_else(|_| "apple\ncar\ncat\nairplane\nspaceship".to_string());
    content.lines()
        .map(|s| s.split_whitespace().last().unwrap_or("").to_string())
        .filter(|s| !s.is_empty())
        .collect()
});

fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot_product / (norm_a * norm_b) }
}

pub struct PrivacyManager {
    pub model: PrivacyFilterModel,
}

pub struct PrivacySession {
    entity_map: HashMap<String, String>, 
    reverse_map: HashMap<String, String>, 
    counter_map: HashMap<String, usize>, 
}

impl PrivacySession {
    pub fn new() -> Self {
        Self {
            entity_map: HashMap::new(),
            reverse_map: HashMap::new(),
            counter_map: HashMap::new(),
        }
    }

    // 🌟 [추가] 프론트엔드(UI)의 Alt+Hover 원문 엿보기 기능을 위한 매칭 결과 추출 함수
    pub fn get_matches(&self) -> Vec<serde_json::Value> {
        let mut matches = Vec::new();
        for (placeholder, original) in &self.reverse_map {
            let parts: Vec<&str> = placeholder.split("][").collect();
            if parts.len() >= 2 {
                let core = parts[1].trim_end_matches(']');
                let inner_parts: Vec<&str> = core.split(':').collect();
                if inner_parts.len() >= 2 {
                    matches.push(serde_json::json!({
                        "name": inner_parts[0].to_uppercase(),
                        "value": original,
                        "mnemonic": inner_parts[1..].join(":")
                    }));
                }
            }
        }
        matches
    }

    pub fn get_or_create_placeholder(&mut self, text: &str, label: &str, record_idx: usize) -> String {
        let normalized_text = text.trim();
        let mut clean_label = if label.contains('-') {
            label.split('-').nth(1).unwrap_or(label).to_lowercase()
        } else {
            label.to_lowercase()
        };

        // 🚀 [PII54] 라벨 통합 및 그룹화
        clean_label = match clean_label.as_str() {
            "firstname" | "lastname" | "middlename" | "accountname" | "username" | "private_person" | "name" => "name".to_string(),
            "phone" | "private_phone" => "phone".to_string(),
            "email" | "private_email" => "email".to_string(),
            "address" | "private_address" | "street" | "city" | "state" | "zipcode" | "county" | "buildingnumber" | "secondaryaddress" => "address".to_string(),
            "date" | "private_date" | "dateofbirth" | "time" => "date".to_string(),
            "amount" | "quantity" | "currency" | "currencycode" | "currencyname" | "currencysymbol" => "amount".to_string(),
            "bankaccount" | "iban" | "creditcard" | "bic" | "cvv" | "pin" | "bitcoinaddress" | "ethereumaddress" | "litecoinaddress" => "finance".to_string(),
            "password" | "ssn" | "passport" | "driverlicense" | "id" => "id".to_string(),
            "jobtitle" | "occupation" | "jobdepartment" | "organization" => "org".to_string(),
            _ => clean_label,
        };

        if let Some(placeholder) = self.entity_map.get(normalized_text) {
            return placeholder.clone();
        }

        let count = self.counter_map.entry(clean_label.clone()).or_insert(0);
        let current_idx = *count;
        *count += 1;
        
        let adj = &ADJECTIVES[current_idx % ADJECTIVES.len()];
        let noun = &NOUNS[(current_idx / ADJECTIVES.len()) % NOUNS.len()];
        let placeholder = format!("[RECORD_{}][{}:{}-{}]", record_idx, clean_label, adj, noun);
        
        self.entity_map.insert(normalized_text.to_string(), placeholder.clone());
        self.reverse_map.insert(placeholder.clone(), normalized_text.to_string());
        placeholder
    }
}

impl PrivacyManager {
    pub fn new(model_dir: &str, device: &candle_core::Device) -> Result<Self> {
        let model = PrivacyFilterModel::load(Path::new(model_dir), device)?;
        Ok(Self { model })
    }

    pub fn mask_text_with_session(
        &self, 
        text: &str, 
        session: &mut PrivacySession, 
        record_idx: usize,
        // 🚀 [Semantic Pipeline] 임베딩 모델과 사전 계산된 Bias Map을 수용합니다.
        em: &crate::models::embedding::EmbeddingModel,
        bias_map: &HashMap<String, (Vec<f32>, Vec<f32>)>,
        // 🌟 [Dynamic Logit Biasing] 가산점 맵 수용
        target_boosts: &HashMap<&str, f32>,
    ) -> Result<String> {
        if text.trim().is_empty() { return Ok(text.to_string()); }
        let normalized_input = text.chars().collect::<String>();

        // 🚀 [Semantic Guidance] 단어별 유사도 맵을 생성합니다.
        let mut semantic_scores: HashMap<String, Vec<f32>> = HashMap::new();
        
        // 문장을 단어 단위로 분리 (공백 기준)
        let words: Vec<&str> = normalized_input.split_whitespace().collect();
        let mut word_vectors = Vec::new();
        for word in &words {
            if let Ok(vec) = em.embed(word) {
                word_vectors.push(vec);
            } else {
                word_vectors.push(vec![0.0; 384]); // Fallback
            }
        }

        // bias.json의 각 카테고리에 대해 prejudice 패널티를 적용한 유사도를 미리 계산합니다.
        for (cat_name, (bias_vec, _prej_vec)) in bias_map {
            let mut scores = Vec::new();
            for word_vec in &word_vectors {
                let b_score = cosine_similarity_f32(word_vec, bias_vec);
                // 🚀 [Co-occurrence Fix] 단어/문장 내 복합 의미 충돌을 막기 위해 prejudice 페널티를 제거합니다.
                let score = b_score;
                scores.push(score);
            }
            semantic_scores.insert(cat_name.clone(), scores);
        }

        let mut spans = match self.model.predict_with_context(
            &normalized_input, 
            &semantic_scores,
            target_boosts
        ) {
            Ok(s) => s,
            Err(e) => return Err(e),
        };

        spans.retain(|s| s.score >= 0.2);
        spans.sort_by(|a, b| a.start.cmp(&b.start).then(b.end.cmp(&a.end)));
        let mut unique_spans = Vec::new();
        let mut last_end = 0;
        for span in spans {
            if span.start >= last_end {
                last_end = span.end;
                unique_spans.push(span);
            }
        }

        let mut result = normalized_input.to_string();
        unique_spans.sort_by(|a, b| b.start.cmp(&a.start));

        // 🚀 [False Positive Defense] 웹 UI 네비게이션 단어들을 벡터 유사도 기반으로 식별하여 마스킹에서 제외합니다.
        for span in unique_spans {
            let mut is_ui_nav = false;
            if let Some((nav_bias, _nav_prej)) = bias_map.get("ui_navigation") {
                if let Ok(word_vec) = em.embed(&span.word) {
                    let b_score = cosine_similarity_f32(&word_vec, nav_bias);
                    // 🚀 [Co-occurrence Fix] UI 네비게이션 판별 시에도 prejudice 페널티를 제거하여 순수 유사도만 봅니다.
                    let mut score = b_score;
                    
                    score *= 1.05; // 🚀 ui_navigation 가산점 동기화
                    
                    // 🚀 [Embedding Space Fix] 기본 유사도가 높으므로 확실한 UI(0.80 이상)일 때만 마스킹을 취소합니다.
                    if score >= 0.80 {
                        is_ui_nav = true;
                    }
                }
            }

            if is_ui_nav {
                continue;
            }

            let clean_label = if span.entity_group.contains('-') {
                span.entity_group.split('-').nth(1).unwrap_or(&span.entity_group).to_lowercase()
            } else {
                span.entity_group.to_lowercase()
            };

            // 🚀 [PII54] 카테고리별 정교한 임계값 적용
            let threshold = match clean_label.as_str() {
                "street" | "buildingnumber" | "zipcode" | "secondaryaddress" => 0.25,
                "city" | "state" | "county" => 0.85, // 지명 오탐 방지를 위해 높게 유지
                "sex" | "gender" => 0.95, // SEX 카테고리 오탐 방지를 위해 매우 높게 유지
                "organization" | "jobtitle" | "occupation" => 0.70,
                "name" | "firstname" | "lastname" => 0.35, // 🚀 [Sensitivity] 이름은 줄 단위 처리 시 점수가 낮을 수 있으므로 문턱을 낮춥니다.
                _ => 0.45, 
            };

            if span.score < threshold {
                continue;
            }

            let placeholder = session.get_or_create_placeholder(&span.word, &span.entity_group, record_idx);
            result.replace_range(span.start..span.end, &placeholder);
        }

        Ok(result)
    }

    pub fn mask_records(&self, texts: Vec<String>, em: &crate::models::embedding::EmbeddingModel, bias_map: &HashMap<String, (Vec<f32>, Vec<f32>)>) -> Result<Vec<String>> {
        let mut session = PrivacySession::new();
        let mut results = Vec::new();
        let empty_boosts = HashMap::new(); // 🌟 임시 빈 값 추가
        for (idx, text) in texts.into_iter().enumerate() {
            results.push(self.mask_text_with_session(&text, &mut session, idx, em, bias_map, &empty_boosts).unwrap_or(text));
        }
        Ok(results)
    }

    pub fn mask_text(&self, text: &str, em: &crate::models::embedding::EmbeddingModel, bias_map: &HashMap<String, (Vec<f32>, Vec<f32>)>) -> Result<String> {
        let mut session = PrivacySession::new();
        let empty_boosts = HashMap::new(); // 🌟 임시 빈 값 추가
        self.mask_text_with_session(text, &mut session, 0, em, bias_map, &empty_boosts)
    }
}
