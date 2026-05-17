use gemini_gui_lib::privacy_filter;
use privacy_filter::PrivacyFilterModel;
use candle_core::{Device, Tensor};
use std::path::PathBuf;

fn mask_pii(text: &str, spans: &[privacy_filter::viterbi::PrivacySpan]) -> String {
    let mut masked_text = text.to_string();
    let mut sorted_spans = spans.to_vec();
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
        "문의사항은 경기도 성남시 분당구 판교역로 166, 카카오 판교 오피스로 방문해 주세요."
    ];

    let label_list = model.get_label_list();

    for (idx, test_text) in test_texts.iter().enumerate() {
        println!("\n--- Test Case {} ---", idx + 1);
        println!("[Test] Input: {}", test_text);
        
        let spans = model.predict(test_text)?;
        println!("[Test] Detected {} spans.", spans.len());
        for span in &spans {
            println!("  - {}: {} (score: {:.4})", span.entity_group, span.word, span.score);
        }
        
        let masked = mask_pii(test_text, &spans);
        println!("[Test] Masked: {}", masked);
    }
    
    Ok(())
}
