use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize)]
pub struct OAuthCredentials {
    pub refresh_token: Option<String>,
    pub access_token: Option<String>,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
}

pub async fn get_auth_token() -> Result<String, Box<dyn std::error::Error>> {
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE"))?;
    let path = PathBuf::from(home).join(".gemini/oauth_creds.json");
    
    let content = fs::read_to_string(&path)?;
    let mut creds: OAuthCredentials = serde_json::from_str(&content)?;

    if creds.client_id.is_none() || creds.client_secret.is_none() || creds.refresh_token.is_none() {
        if let Ok(env_val) = std::env::var("GCP_SERVICE_ACCOUNT") {
            if let Ok(env_creds) = serde_json::from_str::<OAuthCredentials>(&env_val) {
                if creds.client_id.is_none() { creds.client_id = env_creds.client_id; }
                if creds.client_secret.is_none() { creds.client_secret = env_creds.client_secret; }
                if creds.refresh_token.is_none() { creds.refresh_token = env_creds.refresh_token; }
            }
        }
    }

    // refresh_token, client_id, client_secret이 모두 존재하면 새로운 access_token을 발급받습니다.
    if let (Some(refresh_token), Some(client_id), Some(client_secret)) = (&creds.refresh_token, &creds.client_id, &creds.client_secret) {
        let client = reqwest::Client::new();
        
        // .form() 메서드 대신 직접 urlencoded 규격의 본문 문자열을 생성하여 전송합니다.
        let body_str = format!(
            "client_id={}&client_secret={}&refresh_token={}&grant_type=refresh_token",
            client_id, client_secret, refresh_token
        );
        
        let res = client.post("https://oauth2.googleapis.com/token")
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(body_str)
            .send()
            .await?;
        
        if res.status().is_success() {
            let v: serde_json::Value = res.json().await?;
            if let Some(new_access_token) = v["access_token"].as_str() {
                creds.access_token = Some(new_access_token.to_string());
                // 갱신된 토큰을 파일에 저장 (JSON 변환이 성공했을 때만 저장 진행)
                if let Ok(json_str) = serde_json::to_string_pretty(&creds) {
                    let _ = fs::write(&path, json_str);
                }
                return Ok(new_access_token.to_string());
            }
        }
    }

    // 갱신에 실패하거나 없는 경우 기존 access_token 확인
    if let Some(access_token) = creds.access_token {
        Ok(access_token)
    } else if let Some(refresh_token) = creds.refresh_token {
        // OAuth refresh_token(1//...) 형태라면 access_token으로 쓸 수 없으므로 에러 반환
        if refresh_token.starts_with("1//") {
            Err("Access token is missing and cannot be refreshed (missing client_id). Please check your oauth_creds.json.".into())
        } else {
            // 사용자가 API 키를 refresh_token 필드에 잘못 넣었을 경우를 대비한 폴백
            Ok(refresh_token)
        }
    } else {
        Err("No valid tokens found in credentials file.".into())
    }
}
