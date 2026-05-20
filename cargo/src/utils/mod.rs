pub mod img_utils;
pub mod video_utils;
pub mod tensor_utils;

use candle_core::{Device, DType};

pub fn get_device(device: Option<&Device>) -> Device {
    match device {
        Some(d) => d.clone(),
        None => Device::Cpu,
    }
}

pub fn get_dtype(dtype: Option<DType>, cfg_dtype: &str) -> DType {
    if let Some(d) = dtype {
        return d;
    }
    match cfg_dtype {
        "float16" => DType::F16,
        "bfloat16" => DType::BF16,
        _ => DType::F32,
    }
}

pub fn find_type_files<P: AsRef<std::path::Path>>(path: P, extension: &str) -> anyhow::Result<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some(extension) {
            files.push(path);
        }
    }
    Ok(files)
}

pub fn extract_user_text(mes: &crate::params::chat::ChatCompletionParameters) -> anyhow::Result<String> {
    for m in &mes.messages {
        if m.role == "user" {
            let mut text = String::new();
            for p in &m.parts {
                text.push_str(&p.text);
            }
            return Ok(text);
        }
    }
    Ok(String::new())
}
