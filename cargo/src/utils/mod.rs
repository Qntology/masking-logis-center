pub mod img_utils;
pub mod video_utils;
pub mod tensor_utils;
pub mod crypto;

use candle_core::{Device, DType};

pub mod paths {
    pub fn get_kv_dir(_: Option<&str>) -> std::path::PathBuf {
        crate::utils::get_app_dir().join("kv")
    }
}

pub mod direct_loader {
    pub fn save_kv_block(path: &std::path::Path, data: &[u8]) -> anyhow::Result<()> {
        if let Some(p) = path.parent() { let _ = std::fs::create_dir_all(p); }
        std::fs::write(path, data)?;
        Ok(())
    }
    pub fn load_kv_block(path: &std::path::Path) -> anyhow::Result<Vec<u8>> {
        Ok(std::fs::read(path)?)
    }
}

pub fn is_extraction_stopped() -> bool { false }
pub fn get_cuda_device(id: usize) -> candle_core::Device { candle_core::Device::new_cuda(id).unwrap_or(candle_core::Device::Cpu) }
pub fn get_logit_processor(temperature: Option<f32>, top_p: Option<f32>, _top_k: Option<usize>, seed: u64) -> candle_transformers::generation::LogitsProcessor {
    candle_transformers::generation::LogitsProcessor::new(seed, temperature.map(|v| v as f64), top_p.map(|v| v as f64))
}

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

pub fn get_app_dir() -> std::path::PathBuf {
    if let Some(mut path) = dirs::data_local_dir() {
        path.push("terminal-logis");
        let _ = std::fs::create_dir_all(&path);
        path
    } else {
        let path = std::path::PathBuf::from("terminal-logis-data");
        let _ = std::fs::create_dir_all(&path);
        path
    }
}

// 🚀 OS 커널 레벨에서 가비지 컬렉터를 강제 호출하여 RAM/VRAM 캐시를 즉시 반환하는 헬퍼 함수
pub fn force_memory_cleanup() {
    #[cfg(target_os = "windows")]
    unsafe {
        let handle = -1isize;
        let min_size = usize::MAX;
        let max_size = usize::MAX;
        let flags = 6u32; 
        windows_sys::Win32::System::Memory::SetProcessWorkingSetSizeEx(handle, min_size, max_size, flags);
    }
    #[cfg(target_os = "linux")]
    unsafe { 
        libc::malloc_trim(0); 
    }
    #[cfg(target_os = "macos")]
    unsafe { 
        extern "C" { fn malloc_zone_pressure_relief(zone: *mut libc::c_void, goal: usize) -> usize; }
        malloc_zone_pressure_relief(std::ptr::null_mut(), 0); 
    }
}
