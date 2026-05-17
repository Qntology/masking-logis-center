//! Integration tests verifying the Rust inference matches the Python reference
//! implementation (HuggingFace transformers 5.6+).
//!
//! Run with:
//!   cargo test --release -- --test-threads=1
//!
//! These tests require the model weights in `./data/`.

use std::path::Path;

use privacy_filter_rs::backend::{B, Device};
use privacy_filter_rs::{PrivacyFilterInference, ViterbiConfig};

fn model_dir() -> &'static Path {
    Path::new("data")
}

fn load_engine() -> PrivacyFilterInference<B> {
    let device = <Device as Default>::default();
    PrivacyFilterInference::<B>::load(model_dir(), device)
        .expect("failed to load model from ./data")
}

// ── Tokenization ────────────────────────────────────────────────────────────

/// Verify tokenizer produces identical token IDs to the Python tokenizer.
/// Reference: AutoTokenizer.from_pretrained("openai/privacy-filter")
#[test]
fn test_tokenization_ids() {
    let tokenizer = tokenizers::Tokenizer::from_file(model_dir().join("tokenizer.json"))
        .expect("failed to load tokenizer");

    let cases: &[(&str, &[u32])] = &[
        (
            "My name is Alice Smith",
            &[5444, 1308, 382, 44045, 16627],
        ),
        (
            "You can reach me at alice.smith@example.com or call 555-0123.",
            &[3575, 665, 7627, 668, 540, 134271, 640, 68671, 81309, 1136, 503, 2421, 220, 22275, 12, 19267, 18, 13],
        ),
        (
            "The weather is nice today and the stock market went up.",
            &[976, 11122, 382, 7403, 4044, 326, 290, 6546, 2910, 5981, 869, 13],
        ),
        (
            "My name is Harry Potter and my email is harry.potter@hogwarts.edu.",
            &[5444, 1308, 382, 23564, 45666, 326, 922, 3719, 382, 3664, 1102, 1201, 346, 399, 31, 96219, 115451, 21819, 13],
        ),
    ];

    for (text, expected_ids) in cases {
        let encoding = tokenizer.encode(*text, false).unwrap();
        let ids = encoding.get_ids();
        assert_eq!(
            ids, *expected_ids,
            "token ID mismatch for: {text}"
        );
    }
}

// ── Argmax label prediction ─────────────────────────────────────────────────

/// Reference argmax labels from Python (per-token, no Viterbi).
/// These must match exactly — same model weights, same tokenizer,
/// same architecture, same label ordering.
struct LabelTestCase {
    text: &'static str,
    expected_argmax_ids: &'static [usize],
    expected_labels: &'static [&'static str],
}

const LABEL_CASES: &[LabelTestCase] = &[
    LabelTestCase {
        text: "My name is Alice Smith",
        expected_argmax_ids: &[0, 0, 0, 17, 19],
        expected_labels: &["O", "O", "O", "B-private_person", "E-private_person"],
    },
    LabelTestCase {
        text: "You can reach me at alice.smith@example.com or call 555-0123.",
        expected_argmax_ids: &[0, 0, 0, 0, 0, 13, 14, 14, 14, 15, 0, 0, 0, 21, 22, 22, 23, 0],
        expected_labels: &[
            "O", "O", "O", "O", "O",
            "B-private_email", "I-private_email", "I-private_email", "I-private_email", "E-private_email",
            "O", "O", "O",
            "B-private_phone", "I-private_phone", "I-private_phone", "E-private_phone",
            "O",
        ],
    },
    LabelTestCase {
        text: "My account number is 4532-1234-5678-9012 and my password is hunter2.",
        expected_argmax_ids: &[0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 0, 0, 0, 0, 0, 0, 0],
        expected_labels: &[
            "O", "O", "O", "O", "O",
            "B-account_number", "I-account_number", "I-account_number", "I-account_number",
            "I-account_number", "I-account_number", "I-account_number", "I-account_number",
            "I-account_number", "I-account_number", "E-account_number",
            "O", "O", "O", "O", "O", "O", "O",
        ],
    },
    LabelTestCase {
        text: "Born on January 15, 1990, Alice visited https://secret-site.com/login.",
        expected_argmax_ids: &[0, 0, 9, 10, 10, 10, 10, 10, 11, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0],
        expected_labels: &[
            "O", "O",
            "B-private_date", "I-private_date", "I-private_date", "I-private_date",
            "I-private_date", "I-private_date", "E-private_date",
            "O", "S-private_person", "O", "O", "O", "O", "O", "O", "O", "O",
        ],
    },
    LabelTestCase {
        text: "The weather is nice today and the stock market went up.",
        expected_argmax_ids: &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        expected_labels: &["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"],
    },
    LabelTestCase {
        text: "My name is Harry Potter and my email is harry.potter@hogwarts.edu.",
        expected_argmax_ids: &[0, 0, 0, 17, 19, 0, 0, 0, 0, 13, 14, 14, 14, 14, 14, 14, 14, 15, 0],
        expected_labels: &[
            "O", "O", "O", "B-private_person", "E-private_person",
            "O", "O", "O", "O",
            "B-private_email", "I-private_email", "I-private_email", "I-private_email",
            "I-private_email", "I-private_email", "I-private_email", "I-private_email",
            "E-private_email",
            "O",
        ],
    },
];

#[test]
fn test_argmax_labels() {
    let engine = load_engine();
    let all_labels = privacy_filter_rs::config::build_label_list();

    for case in LABEL_CASES {
        let rust_labels = engine.predict_argmax(case.text).unwrap();

        // Check length
        assert_eq!(
            rust_labels.len(),
            case.expected_labels.len(),
            "label count mismatch for: {}",
            case.text
        );

        // Check each label
        for (t, (rust, expected)) in rust_labels.iter().zip(case.expected_labels.iter()).enumerate() {
            assert_eq!(
                rust, expected,
                "label mismatch at token {t} for \"{}\": got {rust}, expected {expected}",
                case.text
            );
        }

        // Also verify argmax IDs
        let (_, logits) = engine.predict_logits(case.text).unwrap();
        let num_labels = all_labels.len();
        for (t, &expected_id) in case.expected_argmax_ids.iter().enumerate() {
            let offset = t * num_labels;
            let actual_id = logits[offset..offset + num_labels]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            assert_eq!(
                actual_id, expected_id,
                "argmax ID mismatch at token {t} for \"{}\": got {actual_id}, expected {expected_id}",
                case.text
            );
        }
    }
}

// ── Span extraction (end-to-end with Viterbi) ──────────────────────────────

struct SpanTestCase {
    text: &'static str,
    expected_spans: &'static [(&'static str, &'static str)], // (entity_group, substring)
}

const SPAN_CASES: &[SpanTestCase] = &[
    SpanTestCase {
        text: "My name is Alice Smith",
        expected_spans: &[("private_person", "Alice Smith")],
    },
    SpanTestCase {
        text: "You can reach me at alice.smith@example.com or call 555-0123.",
        expected_spans: &[
            ("private_email", "alice.smith@example.com"),
            ("private_phone", "555-0123"),
        ],
    },
    SpanTestCase {
        text: "My account number is 4532-1234-5678-9012 and my password is hunter2.",
        expected_spans: &[("account_number", "4532-1234-5678-9012")],
    },
    SpanTestCase {
        text: "Born on January 15, 1990, Alice visited https://secret-site.com/login.",
        expected_spans: &[
            ("private_date", "January 15, 1990"),
            ("private_person", "Alice"),
        ],
    },
    SpanTestCase {
        text: "The weather is nice today and the stock market went up.",
        expected_spans: &[],
    },
    SpanTestCase {
        text: "My name is Harry Potter and my email is harry.potter@hogwarts.edu.",
        expected_spans: &[
            ("private_person", "Harry Potter"),
            ("private_email", "harry.potter@hogwarts.edu"),
        ],
    },
];

#[test]
fn test_span_extraction() {
    let engine = load_engine();

    for case in SPAN_CASES {
        let spans = engine.predict(case.text).unwrap();

        assert_eq!(
            spans.len(),
            case.expected_spans.len(),
            "span count mismatch for \"{}\": got {:?}",
            case.text,
            spans.iter().map(|s| (&s.entity_group, &s.word)).collect::<Vec<_>>()
        );

        for (i, (span, &(expected_group, expected_word))) in
            spans.iter().zip(case.expected_spans.iter()).enumerate()
        {
            assert_eq!(
                span.entity_group, expected_group,
                "span {i} entity_group mismatch for \"{}\": got {}, expected {expected_group}",
                case.text, span.entity_group
            );
            // Trim leading space from extracted word for comparison
            let word_trimmed = span.word.trim();
            assert_eq!(
                word_trimmed, expected_word,
                "span {i} word mismatch for \"{}\": got \"{}\", expected \"{expected_word}\"",
                case.text, word_trimmed
            );
        }
    }
}

/// Verify span scores are high-confidence (>0.95) for clear PII.
#[test]
fn test_span_confidence() {
    let engine = load_engine();

    let high_confidence_cases = &[
        "My name is Alice Smith",
        "Contact me at alice@example.com",
        "My name is Harry Potter and my email is harry.potter@hogwarts.edu.",
    ];

    for text in high_confidence_cases {
        let spans = engine.predict(text).unwrap();
        for span in &spans {
            assert!(
                span.score > 0.95,
                "low confidence {:.4} for \"{}\" ({}) in: {text}",
                span.score, span.word, span.entity_group
            );
        }
    }
}

// ── No-PII texts should produce zero spans ──────────────────────────────────

#[test]
fn test_no_pii_detection() {
    let engine = load_engine();

    let clean_texts = &[
        "The weather is nice today and the stock market went up.",
        "Machine learning models are getting better every year.",
        "The quick brown fox jumps over the lazy dog.",
        "Please refer to the documentation for more details.",
    ];

    for text in clean_texts {
        let spans = engine.predict(text).unwrap();
        assert!(
            spans.is_empty(),
            "unexpected PII in \"{text}\": {:?}",
            spans.iter().map(|s| (&s.entity_group, &s.word)).collect::<Vec<_>>()
        );
    }
}

// ── Span byte offsets ───────────────────────────────────────────────────────

#[test]
fn test_span_offsets() {
    let engine = load_engine();

    let text = "My name is Harry Potter and my email is harry.potter@hogwarts.edu.";
    let spans = engine.predict(text).unwrap();

    for span in &spans {
        // Verify start/end are valid byte offsets into the original text
        assert!(
            span.end <= text.len(),
            "span end {} exceeds text length {} for \"{}\"",
            span.end, text.len(), span.word
        );
        assert!(
            span.start < span.end,
            "span start {} >= end {} for \"{}\"",
            span.start, span.end, span.word
        );
        // Verify the substring matches
        let extracted = text[span.start..span.end].trim();
        let word_trimmed = span.word.trim();
        assert_eq!(
            extracted, word_trimmed,
            "offset extraction mismatch: text[{}..{}]=\"{extracted}\" vs word=\"{word_trimmed}\"",
            span.start, span.end
        );
    }
}

// ── Viterbi decoder unit tests ──────────────────────────────────────────────

#[test]
fn test_viterbi_all_background() {
    // If all logits strongly favour O (class 0), output should be all O.
    let seq_len = 5;
    let num_labels = 33;
    let mut logits = vec![-10.0f32; seq_len * num_labels];
    for t in 0..seq_len {
        logits[t * num_labels] = 20.0; // O is dominant
    }

    let config = ViterbiConfig::default();
    let path = privacy_filter_rs::viterbi::viterbi_decode(&logits, seq_len, &config);
    assert_eq!(path, vec![0, 0, 0, 0, 0]);
}

#[test]
fn test_viterbi_single_span() {
    // Simulate: O, B-person, E-person, O
    let seq_len = 4;
    let num_labels = 33;
    let mut logits = vec![-10.0f32; seq_len * num_labels];

    // Token 0: O
    logits[0 * num_labels + 0] = 20.0;
    // Token 1: B-private_person (class 17)
    logits[1 * num_labels + 17] = 20.0;
    // Token 2: E-private_person (class 19)
    logits[2 * num_labels + 19] = 20.0;
    // Token 3: O
    logits[3 * num_labels + 0] = 20.0;

    let config = ViterbiConfig::default();
    let path = privacy_filter_rs::viterbi::viterbi_decode(&logits, seq_len, &config);
    assert_eq!(path, vec![0, 17, 19, 0]);
}

#[test]
fn test_viterbi_rejects_invalid_transitions() {
    // If logits try to force I-person without a preceding B-person,
    // Viterbi should refuse and pick a valid alternative.
    let seq_len = 3;
    let num_labels = 33;
    let mut logits = vec![-10.0f32; seq_len * num_labels];

    // Token 0: O
    logits[0 * num_labels + 0] = 20.0;
    // Token 1: Try I-private_person (class 18) — invalid from O
    logits[1 * num_labels + 18] = 20.0;
    logits[1 * num_labels + 0] = 5.0; // O is the valid fallback
    // Token 2: O
    logits[2 * num_labels + 0] = 20.0;

    let config = ViterbiConfig::default();
    let path = privacy_filter_rs::viterbi::viterbi_decode(&logits, seq_len, &config);

    // Token 1 should NOT be I-private_person (18) since that's invalid from O.
    assert_ne!(path[1], 18, "Viterbi should reject I-person after O");
}

#[test]
fn test_viterbi_single_token_span() {
    // S-private_person (class 20) should produce a valid single-token span.
    let seq_len = 3;
    let num_labels = 33;
    let mut logits = vec![-10.0f32; seq_len * num_labels];

    logits[0 * num_labels + 0] = 20.0;
    logits[1 * num_labels + 20] = 20.0; // S-private_person
    logits[2 * num_labels + 0] = 20.0;

    let config = ViterbiConfig::default();
    let path = privacy_filter_rs::viterbi::viterbi_decode(&logits, seq_len, &config);
    assert_eq!(path, vec![0, 20, 0]);
}

// ── Config parsing ──────────────────────────────────────────────────────────

#[test]
fn test_config_parsing() {
    let config = privacy_filter_rs::ModelConfig::from_file(&model_dir().join("config.json"))
        .expect("failed to parse config.json");

    assert_eq!(config.vocab_size, 200064);
    assert_eq!(config.hidden_size, 640);
    assert_eq!(config.intermediate_size, 640);
    assert_eq!(config.num_hidden_layers, 8);
    assert_eq!(config.num_attention_heads, 14);
    assert_eq!(config.num_key_value_heads, 2);
    assert_eq!(config.head_dim, 64);
    assert_eq!(config.sliding_window, 128);
    assert_eq!(config.num_local_experts, 128);
    assert_eq!(config.num_experts_per_tok, 4);
    assert_eq!(config.num_key_value_groups(), 7);
    assert_eq!(config.num_labels(), 33);
    assert_eq!(config.attention_bias, true);
}

#[test]
fn test_viterbi_config_parsing() {
    let vc = privacy_filter_rs::ViterbiConfig::from_file(
        &model_dir().join("viterbi_calibration.json"),
        "default",
    ).expect("failed to parse viterbi_calibration.json");

    assert_eq!(vc.transition_bias_background_stay, 0.0);
    assert_eq!(vc.transition_bias_background_to_start, 0.0);
    assert_eq!(vc.transition_bias_inside_to_continue, 0.0);
    assert_eq!(vc.transition_bias_inside_to_end, 0.0);
    assert_eq!(vc.transition_bias_end_to_background, 0.0);
    assert_eq!(vc.transition_bias_end_to_start, 0.0);
}

// ── Label helpers ───────────────────────────────────────────────────────────

#[test]
fn test_label_list() {
    let labels = privacy_filter_rs::config::build_label_list();
    assert_eq!(labels.len(), 33);
    assert_eq!(labels[0], "O");
    assert_eq!(labels[1], "B-account_number");
    assert_eq!(labels[2], "I-account_number");
    assert_eq!(labels[3], "E-account_number");
    assert_eq!(labels[4], "S-account_number");
    assert_eq!(labels[5], "B-private_address");
    assert_eq!(labels[17], "B-private_person");
    assert_eq!(labels[18], "I-private_person");
    assert_eq!(labels[19], "E-private_person");
    assert_eq!(labels[20], "S-private_person");
    assert_eq!(labels[32], "S-secret");
}

#[test]
fn test_label_to_category() {
    use privacy_filter_rs::config::{label_to_category, label_to_prefix};

    assert_eq!(label_to_category(0), None);
    assert_eq!(label_to_prefix(0), Some("O"));

    assert_eq!(label_to_category(1), Some("account_number"));
    assert_eq!(label_to_prefix(1), Some("B"));

    assert_eq!(label_to_category(17), Some("private_person"));
    assert_eq!(label_to_prefix(17), Some("B"));

    assert_eq!(label_to_category(20), Some("private_person"));
    assert_eq!(label_to_prefix(20), Some("S"));

    assert_eq!(label_to_category(32), Some("secret"));
    assert_eq!(label_to_prefix(32), Some("S"));
}
