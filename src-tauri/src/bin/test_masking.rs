use gemini_gui_lib::privacy_filter;
use privacy_filter::PrivacyFilterModel;
use candle_core::Device;
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
    
    let test_text = "My name is Alice Smith and I live at 123 Maple Street, New York. My phone number is 555-0199.";
    println!("[Test] Input: {}", test_text);
    
    let spans = model.predict(test_text)?;
    println!("[Test] Detected {} spans.", spans.len());
    for span in &spans {
        println!("  - {}: {} (score: {:.4})", span.entity_group, span.word, span.score);
    }
    
    let masked = mask_pii(test_text, &spans);
    println!("[Test] Masked: {}", masked);
    
    Ok(())
}
