use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize)]
pub struct OAuthCredentials {
    pub refresh_token: String,
    pub client_id: String,
    pub client_secret: String,
}

pub fn get_auth_token() -> Result<String, Box<dyn std::error::Error>> {
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE"))?;
    let path = PathBuf::from(home).join(".gemini/oauth_creds.json");
    
    let content = fs::read_to_string(path)?;
    let creds: OAuthCredentials = serde_json::from_str(&content)?;

    // 실제로는 refresh_token을 사용하여 새로운 access_token을 받아야 합니다.
    // 여기서는 간단한 구현을 위해 refresh_token을 리턴하지만, 
    // 실제 운영 환경에서는 아래 refresh 로직이 필요합니다.
    Ok(creds.refresh_token)
}
