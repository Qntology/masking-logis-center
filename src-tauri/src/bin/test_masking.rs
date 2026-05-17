use gemini_gui_lib::privacy_filter;
use privacy_filter::PrivacyFilterModel;
use candle_core::{Device, Tensor};
use std::path::PathBuf;
use regex::Regex;

fn mask_with_regex(text: &str) -> Vec<privacy_filter::viterbi::PrivacySpan> {
    let mut spans = Vec::new();

    // 1. Resident Registration Number (주민등록번호)
    let rrn_regex = Regex::new(r"\d{6}-[1-4]\d{6}").unwrap();
    for mat in rrn_regex.find_iter(text) {
        spans.push(privacy_filter::viterbi::PrivacySpan {
            entity_group: "RRN".to_string(),
            start: mat.start(),
            end: mat.end(),
            word: mat.as_str().to_string(),
            score: 1.0,
        });
    }

    // 2. Email (이메일)
    let email_regex = Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap();
    for mat in email_regex.find_iter(text) {
        spans.push(privacy_filter::viterbi::PrivacySpan {
            entity_group: "EMAIL".to_string(),
            start: mat.start(),
            end: mat.end(),
            word: mat.as_str().to_string(),
            score: 1.0,
        });
    }

    // 3. Phone (전화번호)
    let phone_regex = Regex::new(r"01[016789]-\d{3,4}-\d{4}").unwrap();
    for mat in phone_regex.find_iter(text) {
        spans.push(privacy_filter::viterbi::PrivacySpan {
            entity_group: "PHONE".to_string(),
            start: mat.start(),
            end: mat.end(),
            word: mat.as_str().to_string(),
            score: 1.0,
        });
    }

    // 4. Address Detail (상세 주소 - 번지, 동/호수)
    // 패턴: 주소 키워드(동/로/길 등) 뒤에 오는 숫자-숫자 또는 숫자번지
    let addr_detail_regex = Regex::new(r"([동로길리읍면])\s+(\d+(-\d+)?(번지)?)").unwrap();
    for cap in addr_detail_regex.captures_iter(text) {
        let mat = cap.get(2).unwrap(); // 두 번째 그룹 (숫자 부분)만 추출
        spans.push(privacy_filter::viterbi::PrivacySpan {
            entity_group: "BUILDINGNUMBER".to_string(),
            start: mat.start(),
            end: mat.end(),
            word: mat.as_str().to_string(),
            score: 1.0,
        });
    }

    // 추가: 아파트 동호수 (예: 101동 202호)
    let apt_detail_regex = Regex::new(r"\d+동\s*\d+호").unwrap();
    for mat in apt_detail_regex.find_iter(text) {
        spans.push(privacy_filter::viterbi::PrivacySpan {
            entity_group: "BUILDINGNUMBER".to_string(),
            start: mat.start(),
            end: mat.end(),
            word: mat.as_str().to_string(),
            score: 1.0,
        });
    }

    spans
}

fn mask_pii(text: &str, spans: &[privacy_filter::viterbi::PrivacySpan]) -> String {
    let mut masked_text = text.to_string();
    let mut sorted_spans = spans.to_vec();
    // Sort in reverse order of start position to replace from the end of the string
    sorted_spans.sort_by(|a, b| b.start.cmp(&a.start));

    for span in sorted_spans {
        if span.start < masked_text.len() && span.end <= masked_text.len() && span.start < span.end {
            let mask = format!("[{}]", span.entity_group.to_uppercase());
            masked_text.replace_range(span.start..span.end, &mask);
        }
    }
    masked_text
}

fn main() -> anyhow::Result<()> {
    #[cfg(any(feature = "cuda", feature = "metal"))]
    let device = if cfg!(feature = "cuda") {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    } else if cfg!(feature = "metal") {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let device = Device::Cpu;

    let model_path = PathBuf::from("..\\models\\privacy-filter");
    
    println!("[Test] Loading model on {:?} from {:?}...", device, model_path);
    let model = PrivacyFilterModel::load(&model_path, &device)?;
    
    let test_texts = vec![
        "안녕하세요, 제 이름은 김철수이고 서울시 강남구 역삼동 123-45에 살고 있습니다. 전화번호는 010-1234-5678입니다.",
        "제 이메일 주소는 chulsoo.kim@example.com이고, 주민등록번호는 900101-1234567입니다.",
        "결제는 국민은행 123-456-789012 계좌로 입금해 주세요.",
        "문의사항은 경기도 성남시 분당구 판교역로 166번길 25, 카카오 판교 오피스로 방문해 주세요.",
        "Hi, this is Alice. 연락처는 010.9999.8888 입니다. 이메일은 help_me@service.kr로 보내주세요. 주소는 대구시 수성구 범어동 101-1번지 102동 304호입니다.",
        "배송지 변경 요청: (06164) 서울특별시 강남구 영동대로 513 (삼성동, 코엑스) 2층 201호, 받는 사람: 홍길동, 연락처: 010-1111-2222"
    ];

    for (idx, test_text) in test_texts.iter().enumerate() {
        println!("\n--- Test Case {} ---", idx + 1);
        println!("[Test] Input: {}", test_text);
        
        // 1. Regex approach
        let mut all_spans = mask_with_regex(test_text);
        
        // 2. Model approach
        let model_spans = model.predict(test_text)?;
        
        // 3. Merge spans (avoid duplicates/overlaps)
        for m_span in model_spans {
            if !all_spans.iter().any(|s| (m_span.start >= s.start && m_span.start < s.end) || (m_span.end > s.start && m_span.end <= s.end)) {
                all_spans.push(m_span);
            }
        }

        println!("[Test] Total Detected {} spans (Regex + Model).", all_spans.len());
        for span in &all_spans {
            println!("  - {}: {} (score: {:.4})", span.entity_group, span.word, span.score);
        }
        
        let masked = mask_pii(test_text, &all_spans);
        println!("[Test] Masked: {}", masked);
    }
    
    Ok(())
}
