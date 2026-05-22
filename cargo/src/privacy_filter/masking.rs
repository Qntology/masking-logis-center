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
}

impl PrivacySession {
    pub fn new() -> Self {
        Self {
            entity_map: HashMap::new(),
            reverse_map: HashMap::new(),
        }
    }

    pub fn get_or_create_placeholder(&mut self, text: &str, label: &str, record_idx: usize) -> String {
        let normalized_text = text.trim();
        
        let clean_label = if label.contains('-') {
            label.split('-').nth(1).unwrap_or(label).to_lowercase()
        } else {
            label.to_lowercase()
        };

        if let Some(placeholder) = self.entity_map.get(normalized_text) {
            println!("[PrivacySession] [Step 1. 기억 확인] 기존 마스킹 값 재사용: '{}' -> '{}'", normalized_text, placeholder);
            return placeholder.clone();
        }

        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        
        hasher.update(clean_label.as_bytes()); 
        hasher.update(normalized_text.as_bytes());
        let result = hasher.finalize();
        
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&result[0..8]);
        let hash_val = u64::from_le_bytes(bytes);
        
        // 🚀 [해시 비트 분리] 64비트 해시값을 쪼개어 형용사와 명사에 각각 다른 값을 부여하여 충돌을 100% 방지합니다.
        let adj_idx = (hash_val as usize) % ADJECTIVES.len();
        let noun_idx = ((hash_val >> 16) as usize) % NOUNS.len();
        
        let adj = &ADJECTIVES[adj_idx];
        let noun = &NOUNS[noun_idx];
        
        let placeholder = format!("[RECORD_{}][{}:{}-{}]", record_idx, clean_label, adj, noun);
        
        println!("[PrivacySession] [Step 1. 신규 기억] 새로운 마스킹 값 생성 및 메모리 저장: '{}' -> '{}'", normalized_text, placeholder);
        
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

        let mut spans = match self.model.predict(text) {
            Ok(s) => {
                println!("[PrivacyManager] predict 성공, {} 개의 식별된 Span 찾음", s.len());
                s
            },
            Err(e) => {
                println!("[PrivacyManager] predict 내부 모델 추론 중 에러 발생: {:?}", e);
                return Err(e);
            }
        };

        // 🚀 [스마트 병합 로직] 오직 AI가 찾아낸 스팬들을 기반으로 병합합니다. 정규식이나 무리한 확장은 사용하지 않습니다.
        spans.sort_by(|a, b| a.start.cmp(&b.start));
        let mut merged_spans: Vec<crate::privacy_filter::viterbi::PrivacySpan> = Vec::new();

        for span in spans {
            // [Pass 1] 임계값을 낮춰서 최대한 많이 후보 수집 (0.3)
            if span.score < 0.3 || span.word.trim().is_empty() {
                continue;
            }

            let label = span.entity_group.to_uppercase();

            if label.contains("AMOUNT") {
                continue;
            }

            let mut merged = false;
            if let Some(last) = merged_spans.last_mut() {
                let last_label = last.entity_group.to_uppercase();
                let last_is_phone = last_label.contains("PHONE");
                let curr_is_phone_fragment = label.contains("PHONE") || label.contains("AGE") || label.contains("ID") || span.word.chars().all(|c| c.is_numeric() || c == '-' || c == '+' || c == '.' || c.is_whitespace());

                // 🚀 [완벽 병합] 간격 허용치를 유지하며 연락처, 이름 등을 파편화 없이 거대하게 이어붙입니다.
                let last_is_name = last_label.contains("FIRSTNAME") || last_label.contains("LASTNAME") || last_label.contains("PERSON") || last_label.contains("NAME");
                let curr_is_name = label.contains("FIRSTNAME") || label.contains("LASTNAME") || label.contains("PERSON") || label.contains("NAME");
                
                // 이름은 5바이트(대략 공백 1~2개 수준) 이내로 가까울 때만 하나로 합침
                let is_name_merge = last_is_name && curr_is_name && span.start <= last.end + 5;
                
                // 🚀 연락처와 이메일은 파편화를 막되, 다른 문맥을 집어삼키지 않도록 간격 허용치를 5바이트(공백/하이픈)로 좁힙니다.
                let is_phone_merge = span.start <= last.end + 5 && last_is_phone && curr_is_phone_fragment;
                let is_email_merge = span.start <= last.end + 5 && label == "EMAIL" && last_label == "EMAIL";
                
                // 주소를 포함하여 그 외 동일 라벨은 5바이트 이내일 때만 합침 (서로 다른 주소 타입 병합 방지)
                let is_same_label_merge = span.start <= last.end + 5 && label == last_label && !is_name_merge && !is_phone_merge;

                let is_normal_merge = is_phone_merge || is_email_merge || is_same_label_merge;

                if is_name_merge || is_normal_merge {
                    last.end = last.end.max(span.end);
                    last.word = text[last.start..last.end].to_string();
                    
                    if is_phone_merge {
                        last.entity_group = "PHONE".to_string();
                    } else if is_name_merge {
                        last.entity_group = "PERSON".to_string(); // 병합된 이름은 단일 FIRSTNAME과 구분하기 위해 PERSON으로 통일
                    }
                    merged = true;
                }
            }
            
            if !merged {
                merged_spans.push(span);
            }
        }

        // 🚀 [Pass 2] 단일 줄(Single Line) 기반 맥락 재검증 전략
        // 후보가 속한 줄의 시작부터 끝까지 전체를 모델에 제공하여 문맥 흐름을 파악하게 합니다.
        let mut verified_spans = Vec::new();
        
        if !merged_spans.is_empty() {
            let mut pass2_bytes = vec![b' '; text.len()];
            let text_bytes = text.as_bytes();
            
            // 줄바꿈 위치를 미리 파악하여 줄 단위 경계를 설정합니다.
            let mut line_boundaries = Vec::new();
            let mut start = 0;
            for (i, &b) in text_bytes.iter().enumerate() {
                if b == b'\n' {
                    line_boundaries.push(start..i);
                    start = i + 1;
                }
            }
            line_boundaries.push(start..text_bytes.len());

            for candidate in &merged_spans {
                // 후보의 시작/끝 포지션이 포함된 줄(Line) 전체 영역을 찾습니다.
                let mut line_range = 0..text_bytes.len();
                for range in &line_boundaries {
                    if candidate.start >= range.start && candidate.start <= range.end {
                        line_range = range.clone();
                        break;
                    }
                }

                // 해당 줄 전체를 pass2_bytes에 복사하여 모델이 '단일 줄' 맥락을 온전히 읽게 합니다.
                let src_slice = &text_bytes[line_range.start..line_range.end];
                pass2_bytes[line_range.start..line_range.end].copy_from_slice(src_slice);
            }

            let pass2_text = String::from_utf8(pass2_bytes).unwrap();
            println!("[PrivacyManager] [Pass 2] 단일 줄 맥락 기반 재검증 시작 (줄 단위 문맥 보존)");

            let verify_spans = match self.model.predict(&pass2_text) {
                Ok(s) => s,
                Err(_) => {
                    println!("[PrivacyManager] [Pass 2] 추론 실패, Pass 1 결과를 fallback으로 사용합니다.");
                    merged_spans.clone()
                }
            };

            // 신규: Pass 1 후보와 매칭된 스팬을 기록하기 위한 배열
            let mut matched_v_spans = vec![false; verify_spans.len()];

            for mut candidate in merged_spans {
                let mut max_verify_score = 0.0;
                let mut best_v_span = None;

                for (v_idx, v_span) in verify_spans.iter().enumerate() {
                    // 바이트 오프셋이 원본과 동일하므로 그대로 비교 (교집합 확인)
                    if v_span.start < candidate.end && v_span.end > candidate.start {
                        matched_v_spans[v_idx] = true; // 매칭됨 기록
                        if v_span.score > max_verify_score {
                            max_verify_score = v_span.score;
                            best_v_span = Some(v_span.clone());
                        }
                    }
                }

                if let Some(v_span) = best_v_span {
                    // 🚀 [교집합 로직] Pass 1과 Pass 2 모두에서 발견된 경우 신뢰도가 매우 높으므로 검증 기준 대폭 완화
                    let final_label = v_span.entity_group.clone();
                    let upper_label = final_label.to_uppercase();
                    
                    // 두 번 모두 감지되었으므로, 경계를 Pass 2의 더 정확한 결과와 병합하여 최대한 넓게(안전하게) 잡음
                    let new_start = candidate.start.min(v_span.start);
                    let new_end = candidate.end.max(v_span.end);
                    let new_word = text[new_start..new_end].to_string();
                    
                    let char_count = new_word.chars().filter(|c| !c.is_whitespace()).count();
                    let mut final_is_valid = true;

                    if (upper_label.contains("PERSON") || upper_label.contains("NAME")) && char_count < 2 {
                        final_is_valid = false;
                    } else if upper_label.contains("PERSON") || upper_label.contains("NAME") {
                        if char_count == 2 && max_verify_score < 0.35 {
                            final_is_valid = false;
                        } else if max_verify_score < 0.25 {
                            final_is_valid = false;
                        }
                    } else if upper_label.contains("AGE") {
                        if max_verify_score < 0.4 {
                            final_is_valid = false;
                        }
                    } else if upper_label.contains("ADDRESS") || upper_label.contains("CITY") || upper_label.contains("STREET") || upper_label.contains("COUNTY") {
                        if max_verify_score < 0.25 {
                            final_is_valid = false;
                        }
                    } else {
                        if max_verify_score < 0.25 {
                            final_is_valid = false;
                        }
                    }

                    if final_is_valid {
                        candidate.start = new_start;
                        candidate.end = new_end;
                        candidate.word = new_word;
                        candidate.entity_group = final_label;
                        
                        println!("[PrivacyManager] ✅ [Pass 2 교집합 통과]: '{}' | 라벨: {} | Score: {}", candidate.word, candidate.entity_group, max_verify_score);
                        verified_spans.push(candidate);
                    } else {
                        println!("[PrivacyManager] 🗑️ [Pass 2 교집합 점수 미달]: '{}' | Score: {}", new_word, max_verify_score);
                    }
                } else {
                    println!("[PrivacyManager] 🗑️ [Pass 2 교집합 탈락 (Pass 2 미감지)]: '{}'", candidate.word);
                }
            }

            // 🚀 [차집합 로직] Pass 1에서 누락되었으나 Pass 2의 주변 문맥 검사 중 새롭게 단독 발견된 엔티티
            for (v_idx, mut v_span) in verify_spans.into_iter().enumerate() {
                if !matched_v_spans[v_idx] {
                    let label = v_span.entity_group.to_uppercase();
                    let char_count = v_span.word.chars().filter(|c| !c.is_whitespace()).count();

                    // 교집합이 아닌 단독 발견이므로, 환각(Hallucination) 방지를 위해 기준을 엄격하게(0.6 이상) 적용
                    if v_span.score >= 0.6 && !v_span.word.trim().is_empty() && !label.contains("AMOUNT") {
                        if (label.contains("PERSON") || label.contains("NAME")) && char_count < 2 {
                            continue;
                        }
                        
                        // 주소 파편화 보정 제거 (개별 주소 속성 라벨 유지)
                        v_span.entity_group = label;

                        println!("[PrivacyManager] ➕ [Pass 2 단독 감지] 신규 엔티티 추가: '{}' | 라벨: {} | Score: {}", v_span.word, v_span.entity_group, v_span.score);
                        verified_spans.push(v_span);
                    }
                }
            }
        }

        // 1. [단일 줄 내 동일 타입 병합 및 확장] 
        // 2차 검증까지 마친 verified_spans를 다시 한번 줄 단위 맥락으로 정제합니다.
        verified_spans.sort_by(|a, b| a.start.cmp(&b.start));
        let mut final_spans: Vec<crate::privacy_filter::viterbi::PrivacySpan> = Vec::new();

        for span in verified_spans {
            let mut merged = false;
            if let Some(last) = final_spans.last_mut() {
                // 같은 줄에 있는지 확인 (간격 기준은 줄의 시작과 끝)
                let is_same_line = text.as_bytes()[last.end..span.start].iter().all(|&b| b != b'\n');
                
                let last_label_upper = last.entity_group.to_uppercase();
                let span_label_upper = span.entity_group.to_uppercase();
                
                let last_is_name = last_label_upper.contains("FIRSTNAME") || last_label_upper.contains("LASTNAME") || last_label_upper.contains("PERSON") || last_label_upper.contains("NAME");
                let span_is_name = span_label_upper.contains("FIRSTNAME") || span_label_upper.contains("LASTNAME") || span_label_upper.contains("PERSON") || span_label_upper.contains("NAME");

                let is_same_type = last_label_upper == span_label_upper;

                // 🚀 같은 줄에 있고 (같은 타입이거나 둘 다 이름 관련 라벨)일 때만 병합
                if is_same_line && (is_same_type || (last_is_name && span_is_name)) {
                    last.end = span.end;
                    last.word = text[last.start..last.end].to_string();
                    
                    if last_is_name && span_is_name {
                        last.entity_group = "PERSON".to_string(); // 합쳐진 이름은 PERSON으로 승격
                    }
                    merged = true;
                }
            }
            if !merged {
                final_spans.push(span);
            }
        }

        // 🚀 [단독 카테고리 파편 무효화 로직]
        // 각 줄(Line)에서 동일한 Category에 속하는 엔티티가 단 1개만 존재할 경우, 유효한 문맥이 부족한 오탐으로 간주하여 무효화합니다.
        let get_category = |label: &str| -> &'static str {
            let l = label.to_uppercase();
            if ["FIRSTNAME", "MIDDLENAME", "LASTNAME", "PREFIX", "AGE", "GENDER", "SEX", "EYECOLOR", "HEIGHT", "USERNAME", "OCCUPATION", "JOBTITLE", "JOBDEPARTMENT", "ORGANIZATION", "USERAGENT", "PERSON", "NAME"].contains(&l.as_str()) { return "Identity"; }
            if ["EMAIL", "PHONE", "URL"].contains(&l.as_str()) { return "Contact"; }
            if ["STREET", "BUILDINGNUMBER", "SECONDARYADDRESS", "CITY", "COUNTY", "STATE", "ZIPCODE", "GPSCOORDINATES", "ORDINALDIRECTION", "ADDRESS"].contains(&l.as_str()) { return "Address"; }
            if ["DATE", "DATEOFBIRTH", "TIME"].contains(&l.as_str()) { return "Dates & time"; }
            if ["SSN"].contains(&l.as_str()) { return "Government IDs"; }
            if ["ACCOUNTNAME", "BANKACCOUNT", "IBAN", "BIC", "CREDITCARD", "CREDITCARDISSUER", "CVV", "PIN", "MASKEDNUMBER", "AMOUNT", "CURRENCY", "CURRENCYCODE", "CURRENCYNAME", "CURRENCYSYMBOL"].contains(&l.as_str()) { return "Financial"; }
            if ["BITCOINADDRESS", "ETHEREUMADDRESS", "LITECOINADDRESS"].contains(&l.as_str()) { return "Crypto"; }
            if ["VIN", "VRM"].contains(&l.as_str()) { return "Vehicle"; }
            if ["IPADDRESS", "MACADDRESS", "IMEI"].contains(&l.as_str()) { return "Digital"; }
            if ["PASSWORD"].contains(&l.as_str()) { return "Auth"; }
            "Unknown"
        };

        let mut block_category_counts: HashMap<(usize, &'static str), usize> = HashMap::new();
        
        for span in &final_spans {
            let cat = get_category(&span.entity_group);
            // 🚀 [핵심 수정] 줄바꿈(\n)뿐만 아니라 파이프(|)와 탭(\t) 기호도 블록 구분자로 사용하여,
            // 물리적으로 같은 줄에 있더라도 논리적으로 동떨어진 독립된 파편(예: 사업자번호 끝자리 ZIPCODE 오탐)을 완벽히 격리합니다.
            let block_idx = text[..span.start].chars().filter(|&c| c == '\n' || c == '|' || c == '\t').count();
            
            let label_upper = span.entity_group.to_uppercase();
            // 🚀 [핵심 밸런스] ZIPCODE, FIRSTNAME, CITY 등 조각난 파편은 가중치 1을 부여하여 단독 존재 시 삭제되게 합니다.
            // 반면, 단독으로 존재해도 문맥상 완전한 PII(이메일, 연락처, 카드번호 등) 및 이미 하나로 병합된 값(PERSON 등)은 가중치 2를 부여하여 억울한 삭제를 방지합니다.
            let fragment_labels = [
                "FIRSTNAME", "MIDDLENAME", "LASTNAME", "PREFIX", "AGE", "GENDER", "SEX", "EYECOLOR", "HEIGHT", 
                "OCCUPATION", "JOBTITLE", "JOBDEPARTMENT", 
                "STREET", "BUILDINGNUMBER", "SECONDARYADDRESS", "CITY", "COUNTY", "STATE", "ZIPCODE", "GPSCOORDINATES", "ORDINALDIRECTION",
                "AMOUNT", "CURRENCY", "CURRENCYCODE", "CURRENCYNAME", "CURRENCYSYMBOL", "DATE", "TIME", "DATEOFBIRTH"
            ];
            
            let weight = if fragment_labels.contains(&label_upper.as_str()) { 1 } else { 2 };
            *block_category_counts.entry((block_idx, cat)).or_insert(0) += weight;
        }

        final_spans.retain(|span| {
            let cat = get_category(&span.entity_group);
            if cat == "Unknown" { return true; }
            
            let block_idx = text[..span.start].chars().filter(|&c| c == '\n' || c == '|' || c == '\t').count();
            if let Some(&count) = block_category_counts.get(&(block_idx, cat)) {
                // 해당 블록에 속한 같은 카테고리의 총합이 1이라면 (논리적으로 단일 파편 오탐)
                if count == 1 {
                    println!("[PrivacyManager] 🗑️ [단일 카테고리 파편 필터링됨]: '{}' | 라벨: {} | 카테고리: {}", span.word, span.entity_group, cat);
                    return false;
                }
            }
            true
        });

        // 2. [역순 치환 전략] 
        // 원본 인덱스 파손을 막기 위해 텍스트 뒷부분부터 치환을 시작합니다.
        let mut masked_text = text.to_string();
        final_spans.sort_by(|a, b| b.start.cmp(&a.start));

        for span in final_spans {
            let label = span.entity_group.to_uppercase();
            
            // 치환 전에만 플레이스홀더를 생성하여 세션 메모리에 저장
            let placeholder = session.get_or_create_placeholder(&span.word, &label, record_idx);
            
            println!("[PrivacyManager] 🔄 최종 역순 치환: '{}' ({}~{}) -> {}", span.word, span.start, span.end, placeholder);
            
            if span.start < masked_text.len() && span.end <= text.len() {
                // 역순으로 진행하므로 앞쪽 인덱스 span.start는 여전히 원본과 일치함이 보장됨
                masked_text.replace_range(span.start..span.end, &placeholder);
            }
        }

        println!("[PrivacyManager] ✅ 모든 줄 단위 병합 및 역순 치환 완료.");
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
