use anyhow::Result;
use image::DynamicImage;

use base64::{Engine as _, engine::general_purpose};

pub fn get_image(path: &str) -> Result<DynamicImage> {
    if path.starts_with("data:image/") {
        let parts: Vec<&str> = path.split(',').collect();
        if parts.len() == 2 {
            let data = general_purpose::STANDARD.decode(parts[1])?;
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
