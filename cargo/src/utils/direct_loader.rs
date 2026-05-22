use anyhow::Result;
use std::path::Path;
use std::fs;

pub fn load_kv_block(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|e| anyhow::anyhow!(e))
}

pub fn save_kv_block(path: &Path, data: &[u8]) -> Result<()> {
    if let Some(p) = path.parent() { 
        let _ = fs::create_dir_all(p); 
    }
    fs::write(path, data).map_err(|e| anyhow::anyhow!(e))
}