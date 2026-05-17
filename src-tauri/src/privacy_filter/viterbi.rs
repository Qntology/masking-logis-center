/// Constrained Viterbi decoder for BIOES span labelling.
///
/// The decoder enforces valid BIOES transitions:
///   O  → O, B-*, S-*
///   B-X → I-X, E-X
///   I-X → I-X, E-X
///   E-X → O, B-*, S-*
///   S-X → O, B-*, S-*
///
/// Label layout (33 labels):
///   0: O
///   For each of 8 categories (c = 0..7):
///     1 + c*4 + 0: B-category
///     1 + c*4 + 1: I-category
///     1 + c*4 + 2: E-category
///     1 + c*4 + 3: S-category

use crate::config::ViterbiConfig;

const NUM_LABELS: usize = 33;
const NEG_INF: f64 = -1e30;

/// Get the BIOES tag type for a label index.
/// Returns: 'O', 'B', 'I', 'E', or 'S'
fn label_tag(label: usize) -> char {
    if label == 0 {
        'O'
    } else {
        match (label - 1) % 4 {
            0 => 'B',
            1 => 'I',
            2 => 'E',
            3 => 'S',
            _ => unreachable!(),
        }
    }
}

/// Get the category index for a label (0-7). Returns None for O.
fn label_category(label: usize) -> Option<usize> {
    if label == 0 {
        None
    } else {
        Some((label - 1) / 4)
    }
}

/// Check if transition from `prev` to `curr` is valid under BIOES constraints.
fn is_valid_transition(prev: usize, curr: usize) -> bool {
    let prev_tag = label_tag(prev);
    let curr_tag = label_tag(curr);
    let prev_cat = label_category(prev);
    let curr_cat = label_category(curr);

    match prev_tag {
        'O' | 'E' | 'S' => {
            // Can go to O, B-*, or S-*
            matches!(curr_tag, 'O' | 'B' | 'S')
        }
        'B' | 'I' => {
            // Must continue or end the SAME category
            match curr_tag {
                'I' | 'E' => prev_cat == curr_cat,
                _ => false,
            }
        }
        _ => false,
    }
}

/// Get the transition bias for a given transition.
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

/// Run constrained Viterbi decoding on logits.
///
/// # Arguments
/// - `logits`: [seq_len, NUM_LABELS] emission scores
/// - `config`: Viterbi transition bias configuration
///
/// # Returns
/// Vector of label indices for each token position.
pub fn viterbi_decode(logits: &[f32], seq_len: usize, config: &ViterbiConfig) -> Vec<usize> {
    if seq_len == 0 {
        return vec![];
    }

    // dp[t][s] = best score ending at time t in state s
    let mut dp = vec![vec![NEG_INF; NUM_LABELS]; seq_len];
    // backpointer[t][s] = previous state for best path
    let mut bp = vec![vec![0usize; NUM_LABELS]; seq_len];

    // Initialize: first token can only start with O, B-*, or S-*
    for s in 0..NUM_LABELS {
        let tag = label_tag(s);
        if matches!(tag, 'O' | 'B' | 'S') {
            dp[0][s] = logits[s] as f64;
        }
    }

    // Forward pass
    for t in 1..seq_len {
        for curr in 0..NUM_LABELS {
            let emission = logits[t * NUM_LABELS + curr] as f64;
            let mut best_score = NEG_INF;
            let mut best_prev = 0;

            for prev in 0..NUM_LABELS {
                if !is_valid_transition(prev, curr) {
                    continue;
                }
                let score = dp[t - 1][prev]
                    + transition_bias(prev, curr, config)
                    + emission;
                if score > best_score {
                    best_score = score;
                    best_prev = prev;
                }
            }

            dp[t][curr] = best_score;
            bp[t][curr] = best_prev;
        }
    }

    // Find best final state (must be O, E-*, or S-*)
    let mut best_final = 0;
    let mut best_score = NEG_INF;
    for s in 0..NUM_LABELS {
        let tag = label_tag(s);
        if matches!(tag, 'O' | 'E' | 'S') && dp[seq_len - 1][s] > best_score {
            best_score = dp[seq_len - 1][s];
            best_final = s;
        }
    }

    // Backtrace
    let mut path = vec![0usize; seq_len];
    path[seq_len - 1] = best_final;
    for t in (1..seq_len).rev() {
        path[t - 1] = bp[t][path[t]];
    }

    path
}

/// A detected privacy span.
#[derive(Debug, Clone)]
pub struct PrivacySpan {
    pub entity_group: String,
    pub score: f32,
    pub word: String,
    pub start: usize,
    pub end: usize,
}

/// Extract spans from Viterbi-decoded label path.
///
/// Groups consecutive B-I-E or standalone S labels into spans.
pub fn extract_spans(
    label_path: &[usize],
    logits: &[f32],
    tokens: &[String],
    offsets: &[(usize, usize)],
    input_text: &str,
) -> Vec<PrivacySpan> {
    let mut spans = Vec::new();
    let seq_len = label_path.len();
    let mut i = 0;

    while i < seq_len {
        let label = label_path[i];
        let tag = label_tag(label);

        match tag {
            'S' => {
                // Single-token span
                let cat = label_category(label).unwrap();
                let cat_name = crate::config::SPAN_LABELS[cat];
                let score = compute_span_score(logits, &[i], label_path);
                let (start, end) = offsets[i];
                let word = if end > start && end <= input_text.len() {
                    input_text[start..end].to_string()
                } else {
                    tokens[i].clone()
                };
                spans.push(PrivacySpan {
                    entity_group: cat_name.to_string(),
                    score,
                    word,
                    start,
                    end,
                });
                i += 1;
            }
            'B' => {
                // Begin of multi-token span
                let cat = label_category(label).unwrap();
                let cat_name = crate::config::SPAN_LABELS[cat];
                let span_start = i;
                let char_start = offsets[i].0;
                i += 1;

                // Collect I tokens
                while i < seq_len {
                    let next_label = label_path[i];
                    let next_tag = label_tag(next_label);
                    if next_tag == 'I' && label_category(next_label) == Some(cat) {
                        i += 1;
                    } else if next_tag == 'E' && label_category(next_label) == Some(cat) {
                        i += 1;
                        break;
                    } else {
                        break;
                    }
                }

                let span_end = i;
                let char_end = offsets[span_end - 1].1;
                let token_indices: Vec<usize> = (span_start..span_end).collect();
                let score = compute_span_score(logits, &token_indices, label_path);
                let word = if char_end > char_start && char_end <= input_text.len() {
                    input_text[char_start..char_end].to_string()
                } else {
                    tokens[span_start..span_end].join("")
                };
                spans.push(PrivacySpan {
                    entity_group: cat_name.to_string(),
                    score,
                    word,
                    start: char_start,
                    end: char_end,
                });
            }
            _ => {
                i += 1;
            }
        }
    }

    spans
}

/// Compute average softmax confidence for a span.
fn compute_span_score(logits: &[f32], token_indices: &[usize], label_path: &[usize]) -> f32 {
    if token_indices.is_empty() {
        return 0.0;
    }

    let mut total_score = 0.0;
    for &t in token_indices {
        let offset = t * NUM_LABELS;
        let label = label_path[t];

        // Softmax for this position
        let max_val = logits[offset..offset + NUM_LABELS]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits[offset..offset + NUM_LABELS]
            .iter()
            .map(|&v| (v - max_val).exp())
            .sum();
        let prob = (logits[offset + label] - max_val).exp() / exp_sum;
        total_score += prob;
    }

    total_score / token_indices.len() as f32
}
