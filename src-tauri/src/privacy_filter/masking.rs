use anyhow::Result;
use std::path::Path;
use crate::privacy_filter::PrivacyFilterModel;

pub struct PrivacyManager {
    model: PrivacyFilterModel,
}

impl PrivacyManager {
    pub fn new(model_dir: &str) -> Result<Self> {
        let device = candle_core::Device::Cpu;
        let model = PrivacyFilterModel::load(Path::new(model_dir), &device)?;
        Ok(Self { model })
    }

    pub fn mask_text(&self, text: &str) -> Result<String> {
        let spans = self.model.predict(text)?;
        let mut masked_text = text.to_string();
        for span in spans.iter().rev() {
            masked_text.replace_range(span.start..span.end, "[MASKED]");
        }
        Ok(masked_text)
    }
}
