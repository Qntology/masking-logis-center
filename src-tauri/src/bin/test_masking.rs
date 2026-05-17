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
    
    let test_text = "My name is Alice Smith and I live at 123 Maple Street, New York. My phone number is 555-0199.";
    println!("[Test] Input: {}", test_text);
    
    // Debug logits
    let tokens = model.tokenizer.encode(test_text, false).map_err(anyhow::Error::msg)?;
    let input_ids = tokens.get_ids();
    let logits = model.forward(input_ids)?;
    let argmax = logits.argmax(candle_core::D::Minus1)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u32>()?;
    
    println!("[Test] Argmax labels for first 10 tokens:");
    for i in 0..10.min(argmax.len()) {
        println!("  token '{}' -> label {}", tokens.get_tokens()[i], argmax[i]);
    }
    
    let spans = model.predict(test_text)?;
    println!("[Test] Detected {} spans.", spans.len());
    for span in &spans {
        println!("  - {}: {} (score: {:.4})", span.entity_group, span.word, span.score);
    }
    
    let masked = mask_pii(test_text, &spans);
    println!("[Test] Masked: {}", masked);
    
    Ok(())
}
