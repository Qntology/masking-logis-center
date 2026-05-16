use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::process::Command;
use tokio::sync::Mutex;
use std::sync::Arc;
use serde_json::{Value, json};
use json_patch::diff;
use futures::TryStreamExt;
use lancedb::query::{QueryBase, ExecutableQuery};

// Import Arrow traits/types directly from the arrow crate
use arrow::array::RecordBatch;
use arrow::datatypes::{Schema, Field, DataType};
use arrow::json::ReaderBuilder;

struct AppState {
    lock: Arc<Mutex<()>>,
}

// Harness Middleware: Process output, update state, and return result
#[tauri::command]
async fn execute_branching_cli(
    command: String, 
    state: tauri::State<'_, AppState>
) -> Result<String, String> {
    let _guard = state.lock.lock().await; 
    
    let path = get_creds_path()?;
    if !path.exists() {
        return Err("Not authenticated.".to_string());
    }

    let output = Command::new("node")
        .arg("../../cli/omg/index.js")
        .arg(command)
        .env("GEMINI_AUTH_FILE", path.to_str().unwrap())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|e: std::io::Error| e.to_string())?;
    
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        
        // Attempt to parse CLI output for structured JSON command
        if let Ok(json_cmd) = serde_json::from_str::<Value>(&stdout) {
            // Harness: Atomically update LanceDB
            if let Err(e) = update_lancedb_state(&json_cmd).await {
                return Err(format!("State Update Failed: {}", e));
            }
            return Ok(format!("State Updated: {}", json_cmd));
        }
        
        Ok(stdout.to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}

// Harness helper: Atomically update state in LanceDB
async fn update_lancedb_state(data: &Value) -> Result<(), String> {
    let db = lancedb::connect("data/commerce-db").execute().await.map_err(|e| e.to_string())?;
    let table = db.open_table("orders").execute().await.map_err(|e| e.to_string())?;
    
    let id = data["id"].as_i64().ok_or("Missing ID")?;
    let state = data["state"].clone();
    
    // Convert JSON to RecordBatch properly
    let schema = std::sync::Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("content", DataType::Utf8, false),
    ]));
    
    let json_data = json!([{"id": id, "content": state.to_string()}]);
    let mut reader = ReaderBuilder::new(schema)
        .build(std::io::Cursor::new(json_data.to_string()))
        .map_err(|e: arrow::error::ArrowError| e.to_string())?;
        
    let record_batch = reader.next().ok_or("No data")?.map_err(|e: arrow::error::ArrowError| e.to_string())?;

    table.add(vec![record_batch])
        .execute()
        .await
        .map_err(|e| e.to_string())?;
        
    Ok(())
}

#[tauri::command]
async fn calculate_state_diff(old_state: String, new_state: String) -> Result<String, String> {
    let old_v: Value = serde_json::from_str(&old_state).map_err(|e: serde_json::Error| e.to_string())?;
    let new_v: Value = serde_json::from_str(&new_state).map_err(|e: serde_json::Error| e.to_string())?;
    
    let patch = diff(&old_v, &new_v);
    serde_json::to_string(&patch).map_err(|e: serde_json::Error| e.to_string())
}

#[tauri::command]
async fn fts_search(query: String) -> Result<String, String> {
    let db = lancedb::connect("data/commerce-db").execute().await.map_err(|e| e.to_string())?;
    let table = db.open_table("orders").execute().await.map_err(|e| e.to_string())?;

    let results = table.query()
        .full_text_search(lancedb::index::scalar::FullTextSearchQuery::new(query))
        .execute()
        .await
        .map_err(|e: lancedb::Error| e.to_string())?;
    
    let batches: Vec<RecordBatch> = results.try_collect().await.map_err(|e| e.to_string())?;
    Ok(format!("{:?}", batches))
}

fn get_creds_path() -> Result<PathBuf, String> {
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).map_err(|e| e.to_string())?;
    let mut path = PathBuf::from(home);
    path.push(".gemini");
    path.push("oauth_creds.json");
    Ok(path)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState { lock: Arc::new(Mutex::new(())) })
        .plugin(tauri_plugin_oauth::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            save_oauth_creds, 
            execute_branching_cli, 
            fts_search, 
            calculate_state_diff
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[tauri::command]
fn save_oauth_creds(creds: String) -> Result<(), String> {
    let path = get_creds_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut file = OpenOptions::new()
        .write(true).create(true).truncate(true)
        .open(&path).map_err(|e| e.to_string())?;
    file.write_all(creds.as_bytes()).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_calculate_state_diff() {
        let old_state = r#"{"id": 1, "state": "배송중"}"#.to_string();
        let new_state = r#"{"id": 1, "state": "배송완료"}"#.to_string();
        let diff = calculate_state_diff(old_state, new_state).await.unwrap();
        
        assert!(diff.contains("replace"));
        assert!(diff.contains("/state"));
    }

    #[tokio::test]
    async fn test_harness_json_parsing() {
        let cli_stdout = r#"{"id": 1001, "state": "취소완료"}"#;
        let json_cmd: Value = serde_json::from_str(cli_stdout).unwrap();
        assert_eq!(json_cmd["id"], 1001);
        assert_eq!(json_cmd["state"], "취소완료");
    }
}
