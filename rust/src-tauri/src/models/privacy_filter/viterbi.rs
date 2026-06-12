/// Constrained Viterbi decoder for BIOES span labelling.
/// Enforces valid BIOES transitions and applies a Two-Pass Anchor-based Expansion & Shrinkage system.

use super::config::ViterbiConfig;

const NEG_INF: f64 = -1e30;

fn label_tag(label: usize) -> char {
    if label == 0 { 'O' }
    else {
        match (label - 1) % 4 {
            0 => 'B', 1 => 'I', 2 => 'E', 3 => 'S',
            _ => unreachable!(),
        }
    }
}

/// Unified categories based on label string content to avoid category tangling.
fn get_unified_cat_from_label(label_idx: usize, label_list: &[String]) -> Option<&'static str> {
    if label_idx == 0 { return None; }
    let label = &label_list[label_idx];
    if label.contains("NAME") || label.contains("PERSON") || label.contains("FIRSTNAME") || label.contains("LASTNAME") || label.contains("MIDDLENAME") { Some("NAME") }
    else if label.contains("STREET") || label.contains("BUILDINGNUMBER") || label.contains("CITY") || label.contains("COUNTY") || label.contains("STATE") || label.contains("ZIPCODE") || label.contains("SECONDARYADDRESS") || label.contains("ADDRESS") { Some("ADDRESS") }
    else if label.contains("AMOUNT") || label.contains("QUANTITY") || label.contains("CURRENCY") { Some("AMOUNT") }
    else if label.contains("PHONE") { Some("PHONE") }
    else if label.contains("EMAIL") { Some("EMAIL") }
    else if label.contains("BANKACCOUNT") || label.contains("IBAN") || label.contains("CREDITCARD") || label.contains("BIC") { Some("FINANCE") }
    else if label.contains("PASSWORD") || label.contains("PIN") || label.contains("SSN") || label.contains("USER") { Some("ID") }
    else if label.contains("DATE") || label.contains("TIME") { Some("DATE") }
    else if label.contains("ORGANIZATION") || label.contains("JOB") || label.contains("COMPANY") { Some("ORG") }
    else { Some("OTHER") }
}

fn is_valid_transition(prev: usize, curr: usize, label_list: &[String]) -> bool {
    let prev_tag = label_tag(prev);
    let curr_tag = label_tag(curr);
    let prev_cat = get_unified_cat_from_label(prev, label_list);
    let curr_cat = get_unified_cat_from_label(curr, label_list);
    match prev_tag {
        'O' | 'E' | 'S' => matches!(curr_tag, 'O' | 'B' | 'S'),
        'B' | 'I' => match curr_tag {
            'I' | 'E' => prev_cat == curr_cat,
            _ => false,
        },
        _ => false,
    }
}

fn transition_bias(prev: usize, curr: usize, config: &ViterbiConfig) -> f64 {
    let prev_tag = label_tag(prev);
    let curr_tag = label_tag(curr);
    match (prev_tag, curr_tag) {
        ('O', 'O') => config.transition_bias_background_stay,
        ('O', 'B') | ('O', 'S') => config.transition_bias_background_to_start,
        ('B', 'I') | ('I', 'I') => config.transition_bias_inside_to_continue,
        ('B', 'E') | ('I', 'E') => config.transition_bias_inside_to_end,
        ('E', 'O') | ('S', 'O') => config.transition_bias_end_to_background,
        ('E', 'B') | ('E', 'S') | ('S', 'B') | ('S', 'S') => config.transition_bias_end_to_start,
        _ => 0.0,
    }
}

// 🛑 [IMPORTANT] DO NOT DELETE OR RESET THE LOGIC BELOW.
fn is_match(token: &str, list: &[String]) -> bool {
    let t = token.trim().to_lowercase();
    if t.is_empty() { return false; }
    for item in list {
        let item_lower = item.to_lowercase();
        if item_lower.starts_with("suffix:") {
            let suffix = &item_lower[7..];
            if t.ends_with(suffix) { return true; }
        } else if t == item_lower {
            return true;
        }
    }
    false
}

fn is_numeric(token: &str) -> bool {
    let t = token.trim();
    !t.is_empty() && t.chars().all(|c| c.is_numeric() || c == '-' || c == ',' || c == '.')
}

pub fn viterbi_decode(
    logits: &[f32],
    seq_len: usize,
    num_labels: usize,
    config: &ViterbiConfig,
    tokens: &[String],
    label_list: &[String],
    // 🚀 [Semantic Guidance] 각 단어별 카테고리 유사도 점수 맵을 직접 수용합니다.
    semantic_scores: &std::collections::HashMap<String, Vec<f32>>, 
    // 🌟 [Dynamic Logit Biasing] 임베딩 모델이 문맥상 찾아낸 목표 카테고리와 가산점(Score)
    target_boosts: &std::collections::HashMap<&str, f32>,
) -> Vec<usize> {
    if seq_len == 0 { return vec![]; }

    // --- Pass 1: Word mapping ---
    let mut word_ids = vec![0usize; seq_len];
    let mut curr_word = 0;
    for t in 0..seq_len {
        if t > 0 && (tokens[t].starts_with(' ') || tokens[t].starts_with('\n')) { curr_word += 1; }
        word_ids[t] = curr_word;
    }

    let mut word_str = vec![String::new(); curr_word + 1];
    for t in 0..seq_len {
        let wid = word_ids[t];
        word_str[wid].push_str(tokens[t].trim());
    }

    // 🚀 [Pass 2] Semantic Guided Decoding
    let mut dp = vec![vec![NEG_INF; num_labels]; seq_len];
    let mut bp = vec![vec![0usize; num_labels]; seq_len];

    for t in 0..seq_len {
        let wid = word_ids[t];
        let wlen = word_str[wid].chars().count();
        let is_plausible_name = wlen >= 2 && wlen <= 5;

        for curr in 0..num_labels {
            let mut emission = logits[t * num_labels + curr] as f64;
            let unified_cat = get_unified_cat_from_label(curr, label_list);

            // 🚀 [Semantic Expansion] bias.json의 유사도 점수를 Viterbi Emission에 직접 투영합니다.
            if let Some(cat) = unified_cat {
                
                // 🌟 [Dynamic Logit Biasing] 임베딩 유사도 기반 타겟 점수 반영
                if let Some(&t_score) = target_boosts.get(cat) {
                    // 매칭된 타겟이면 점수에 비례하여 강력한 가산점 부여 (예: 0.5 * 20.0 = +10.0)
                    emission += (t_score as f64) * 20.0;
                } else if !target_boosts.is_empty() {
                    // 다른 타겟들이 매칭되었는데 현재 라벨은 매칭되지 않았다면 억제 페널티
                    emission -= 5.0;
                }

                let bias_cat = match cat {
                    "NAME" => "name",
                    "ADDRESS" => "address",
                    "AMOUNT" => "amount",
                    "PHONE" => "contact_number",
                    "EMAIL" => "email",
                    "FINANCE" => "finance",
                    "ORG" => "company",
                    _ => "",
                };

                if !bias_cat.is_empty() {
                    if let Some(cat_scores) = semantic_scores.get(bias_cat) {
                        if let Some(&score) = cat_scores.get(wid) {
                            // 🚀 [Embedding Space Fix] Gemma 모델의 기본 유사도가 높으므로, 진짜 연관된 단어(0.82 이상)에만 보너스를 줍니다.
                            // 수정 전 (너무 과도한 가감점)
                            // emission += (score as f64) * 15.0; 
                            // emission -= 15.0; 

                            // 수정 후 (부드러운 보정)
                            if score > 0.85 { // 문턱을 더 높여서 정말 확실할 때만
                                emission += 2.0; // 가산점을 대폭 낮춤
                            } else if score < 0.65 { // 기준을 낮춰서 웬만하면 페널티를 받지 않도록
                                emission -= 2.0; // 페널티 대폭 낮춤
                            }
                        }
                    }
                }

                // 이름 특화 추가 보정 (이름인데 길이가 너무 길거나 짧으면 억제)
                if cat == "NAME" && !is_plausible_name {
                    emission -= 40.0;
                }
            }

            if t == 0 {
                dp[0][curr] = emission + match label_tag(curr) {
                    'O' => config.transition_bias_background_stay,
                    'B' | 'S' => config.transition_bias_background_to_start,
                    _ => NEG_INF,
                };
                continue;
            }

            for prev in 0..num_labels {
                if !is_valid_transition(prev, curr, label_list) { continue; }
                let trans = transition_bias(prev, curr, config);
                let score = dp[t - 1][prev] + trans + emission;
                if score > dp[t][curr] { dp[t][curr] = score; bp[t][curr] = prev; }
            }
        }
    }

    let mut path = vec![0usize; seq_len];
    let mut best_final = 0; let mut best_score = NEG_INF;
    for s in 0..num_labels { if dp[seq_len - 1][s] > best_score { best_score = dp[seq_len - 1][s]; best_final = s; } }
    path[seq_len - 1] = best_final;
    for t in (1..seq_len).rev() { path[t - 1] = bp[t][path[t]]; }
    path
}

#[derive(Debug, Clone)]
pub struct PrivacySpan {
    pub entity_group: String,
    pub score: f32,
    pub word: String,
    pub start: usize,
    pub end: usize,
}

pub fn extract_spans(
    label_path: &[usize],
    logits: &[f32],
    num_labels: usize,
    label_list: &[String],
    _tokens: &[String],
    offsets: &[(usize, usize)],
    input_text: &str,
) -> Vec<PrivacySpan> {
    let mut spans = Vec::new();
    let seq_len = label_path.len();
    let mut current_span: Option<(usize, usize, usize)> = None;

    for t in 0..seq_len {
        let label = label_path[t];
        let tag = label_tag(label);

        if let Some((start_idx, category_idx, seq_start)) = current_span {
            let mut end_it = false;
            if tag == 'O' || tag == 'B' || tag == 'S' || get_unified_cat_from_label(label_path[t-1], label_list) != get_unified_cat_from_label(label, label_list) {
                end_it = true;
            }
            if end_it {
                let end_idx = offsets[t - 1].1;
                let mut total_score = 0.0;
                for i in seq_start..t {
                    let offset = i * num_labels;
                    let slice = &logits[offset..offset + num_labels];
                    let max_val = slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let exp_sum: f32 = slice.iter().map(|&v| (v - max_val).exp()).sum();
                    total_score += (logits[offset + label_path[i]] - max_val).exp() / exp_sum;
                }
                spans.push(PrivacySpan {
                    entity_group: label_list[1 + category_idx * 4].split('-').nth(1).unwrap_or(&label_list[1 + category_idx * 4]).to_string(),
                    score: total_score / (t - seq_start) as f32,
                    word: input_text[start_idx..end_idx].to_string(),
                    start: start_idx,
                    end: end_idx,
                });
                current_span = None;
            }
        }

        if current_span.is_none() && (tag == 'B' || tag == 'S') {
            current_span = Some((offsets[t].0, label.saturating_sub(1) / 4, t));
        }

        if tag == 'S' {
            if let Some((start_idx, category_idx, _)) = current_span {
                let offset = t * num_labels;
                let slice = &logits[offset..offset + num_labels];
                let max_val = slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp_sum: f32 = slice.iter().map(|&v| (v - max_val).exp()).sum();
                let score = (logits[offset + label_path[t]] - max_val).exp() / exp_sum;
                spans.push(PrivacySpan {
                    entity_group: label_list[1 + category_idx * 4].split('-').nth(1).unwrap_or(&label_list[1 + category_idx * 4]).to_string(),
                    score,
                    word: input_text[start_idx..offsets[t].1].to_string(),
                    start: start_idx,
                    end: offsets[t].1,
                });
                current_span = None;
            }
        }
    }
    
    if let Some((start_idx, category_idx, seq_start)) = current_span {
        let t = seq_len;
        let mut total_score = 0.0;
        for i in seq_start..t {
            let offset = i * num_labels;
            let slice = &logits[offset..offset + num_labels];
            let max_val = slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_sum: f32 = slice.iter().map(|&v| (v - max_val).exp()).sum();
            total_score += (logits[offset + label_path[i]] - max_val).exp() / exp_sum;
        }
        spans.push(PrivacySpan {
            entity_group: label_list[1 + category_idx * 4].split('-').nth(1).unwrap_or(&label_list[1 + category_idx * 4]).to_string(),
            score: total_score / (t - seq_start) as f32,
            word: input_text[start_idx..offsets[t-1].1].to_string(),
            start: start_idx,
            end: offsets[t-1].1,
        });
    }
    spans
}

pub fn calculate_span_score(token_indices: &[usize], logits: &[f32], num_labels: usize) -> f32 {
    let mut total_score = 0.0;
    for (i, &label) in token_indices.iter().enumerate() {
        let offset = i * num_labels;
        let slice = &logits[offset..offset + num_labels];
        let max_val = slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = slice.iter().map(|&v| (v - max_val).exp()).sum();
        total_score += (logits[offset + label] - max_val).exp() / exp_sum;
    }
    total_score / token_indices.len() as f32
}
