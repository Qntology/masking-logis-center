use anyhow::Result;
use image::DynamicImage;

use base64::{Engine as _, engine::general_purpose};

pub fn get_image(path: &str) -> Result<DynamicImage> {
    if path.starts_with("data:image/") {
        // 🚀 OCR 텍스트가 뒤에 덧붙여진 경우를 대비해 순수 Base64 영역만 안전하게 분리합니다.
        let clean_path = path.split("\n---").next().unwrap_or(path).trim();
        let parts: Vec<&str> = clean_path.split(',').collect();
        if parts.len() == 2 {
            let base64_data = parts[1].trim();
            let data = general_purpose::STANDARD.decode(base64_data)?;
            return Ok(image::load_from_memory(&data)?);
        }
    }
    Ok(image::open(path)?)
}

pub fn extract_image_url(mes: &crate::params::chat::ChatCompletionParameters) -> Vec<String> {
    let mut urls = Vec::new();
    for m in &mes.messages {
        for p in &m.parts {
            if let Some(url) = &p.image_url {
                urls.push(url.clone());
            }
        }
    }
    urls
}
