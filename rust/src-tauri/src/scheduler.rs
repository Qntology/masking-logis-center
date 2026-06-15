use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};
use crate::store::{VectorStore, Task};
use crate::utils;
use crate::parsing::{self, PugMode};
use crate::model::LogisModel;
use serde_json::{Value, json};
use anyhow::Result;
use tauri::Emitter;
use std::sync::atomic::{AtomicBool, Ordering};

fn merge_node(obj1: &Value, obj2: &Value) -> Value {
    let mut merged = obj1.clone();
    if let (Some(m_obj), Some(o2_obj)) = (merged.as_object_mut(), obj2.as_object()) {
        for (k, v) in o2_obj {
            let is_empty = match v {
                Value::Null => true,
                Value::String(s) => s.is_empty(),
                Value::Number(n) => n.as_f64().unwrap_or(0.0) == 0.0,
                _ => false,
            };
            if !is_empty {
                m_obj.insert(k.clone(), v.clone());
            }
        }
    }
    merged
}

use tokio::sync::Notify;
use once_cell::sync::Lazy;
use once_cell::sync::OnceCell;

pub static PROGRESS_TX: OnceCell<tokio::sync::mpsc::UnboundedSender<serde_json::Value>> = OnceCell::new();

// [UI-SYNC] Instant notification system to wake up the worker
static UI_READY_SIGNAL: Lazy<Notify> = Lazy::new(|| Notify::new());
static TASK_QUEUED_SIGNAL: Lazy<Notify> = Lazy::new(|| Notify::new());
static UI_READY_FLAG: AtomicBool = AtomicBool::new(false);

pub fn mark_ui_ready() {
    UI_READY_FLAG.store(true, Ordering::SeqCst);
    UI_READY_SIGNAL.notify_waiters(); // Wake up any sleeping tasks instantly
    println!("[Scheduler] UI signaled ready. Background worker woke up.");
}

pub fn notify_new_task() {
    TASK_QUEUED_SIGNAL.notify_waiters();
}

pub async fn start_background_worker(
    store: Arc<Mutex<Option<VectorStore>>>,
    model: Arc<Mutex<Option<LogisModel>>>,
    cancellation_token: Arc<AtomicBool>,
    app_handle: tauri::AppHandle,
) {
    println!("[Scheduler] Background worker waiting for UI Ready signal...");
    
    let (ptx, mut prx) = tokio::sync::mpsc::unbounded_channel::<serde_json::Value>();
    let _ = PROGRESS_TX.set(ptx);
    let app_handle_prog = app_handle.clone();
    tokio::spawn(async move {
        use tauri::Emitter;
        while let Some(payload) = prx.recv().await {
            if let Ok(mut w) = crate::LATEST_PROGRESS_PAYLOAD.write() {
                *w = Some(payload.clone());
            }
            let _ = app_handle_prog.emit("extraction-progress", &payload);
        }
    });

    
    // 여기서 다시 spawn 하여 불필요한 DB 락 경쟁을 일으킬 필요가 없습니다.
    
    tokio::spawn(async move {
        if !UI_READY_FLAG.load(Ordering::SeqCst) {
            UI_READY_SIGNAL.notified().await;
        }
        
        let mut delay_secs = 1;
        let mut current_device_pref: Option<String> = None;
        
        let mut oom_retry_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        
        loop {
            if crate::utils::is_extraction_stopped() {
                tokio::time::sleep(Duration::from_millis(500)).await;
                continue;
            }

            let mut pending_tasks = Vec::new();
            {
                let store_opt = store.lock().await;
                if let Some(db) = store_opt.as_ref() {
                    match db.get_pending_tasks(5).await {
                        Ok(tasks) => {
                            
                            pending_tasks = tasks.into_iter().filter(|t| t.r#type != "ai_search").collect();
                        },
                        Err(e) => println!("[Scheduler] Failed to fetch tasks: {:?}", e),
                    }
                }
            }

            if pending_tasks.is_empty() {
                tokio::select! {
                    _ = sleep(Duration::from_secs(delay_secs)) => {
                        delay_secs = (delay_secs + 1).min(10); 
                    }
                    _ = TASK_QUEUED_SIGNAL.notified() => {
                        delay_secs = 1;
                        println!("[Scheduler] New task signal received. Waking up immediately.");
                    }
                }
                continue;
            } else {
                delay_secs = 1;
            }

            for task in pending_tasks {
                if cancellation_token.load(Ordering::Relaxed) {
                    println!("[Scheduler] Cancellation detected before starting task {}, skipping batch.", task.id);
                    break;
                }


                println!("[Scheduler] Processing task: {}", task.id);
                
                {
                    let store_guard = store.lock().await;
                    if let Some(db) = store_guard.as_ref() {
                        // DB의 상태값만 안전하게 1(Processing)로 동기화합니다.
                        let _ = db.update_task_status(&task.id, 1).await;
                        let _ = db.update_message_status(&task.id, 1, Some("Processing...")).await;
                        
                        
                        if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() {
                            *w = Some(json!({ "id": task.id, "ref": task.r#ref, "status": 1 }));
                        }
                    }
                }

                match process_task(task.clone(), &store, &model, &cancellation_token, &app_handle, current_device_pref.clone()).await {
                    Ok(_) => {
                        println!("[Scheduler] Task completed: {}", task.id);
                        
                        
                        if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() { 
                            if let Some(task_val) = w.as_mut() {
                                if let Some(obj) = task_val.as_object_mut() {
                                    obj.insert("status".to_string(), json!(9));
                                    obj.insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
                                }
                            }
                        }

                        // 일정 시간 뒤에 메모리를 비워주거나, 다음 작업 시작 시 덮어씌워지도록 유지합니다.

                        {
                            let mut model_lock = model.lock().await;
                            if let Some(m) = model_lock.as_ref() {
                                m.deep_purge_resources().await;
                            }
                            *model_lock = None;
                        }
                        
                        // 🌟 [말풍선 텍스트 덮어쓰기 우회] 가장 마지막으로 발행된 진행 상황(summary) 텍스트를 읽어옵니다.
                        let mut final_msg = "Task Completed".to_string();
                        if let Ok(mem) = crate::LATEST_PROGRESS_PAYLOAD.read() {
                            if let Some(payload) = mem.as_ref() {
                                if payload.get("task_id").and_then(|v| v.as_str()) == Some(task.id.as_str()) {
                                    if let Some(summary) = payload.get("summary").and_then(|v| v.as_str()) {
                                        final_msg = summary.to_string();
                                    }
                                }
                            }
                        }

                        let store_guard = store.lock().await;
                        if let Some(db) = store_guard.as_ref() {
                            let _ = db.update_task_status(&task.id, crate::logic::parse_status("complete")).await;
                            let _ = db.update_message_status(&task.id, crate::logic::parse_status("complete"), Some(&final_msg)).await;
                        }

                        current_device_pref = None; 
                        oom_retry_map.remove(&task.id); // 성공 시 장부 삭제
                    },
                    Err(e) => {
                        let err_msg = e.to_string();
                        println!("[Scheduler] Task failed: {:?}. Error: {}", task.id, err_msg);
                        
                        if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() { *w = None; }

                        {
                            let mut model_lock: tokio::sync::MutexGuard<Option<LogisModel>> = model.lock().await;
                            if let Some(m) = model_lock.as_ref() {
                                println!("[Scheduler] Error detected. Performing emergency memory release...");
                                m.deep_purge_resources().await;
                            }
                            *model_lock = None;
                        }

                        if err_msg.contains("Task cancelled") {
                             println!("[Scheduler] Task cancelled: {}", task.id);
                             
                             // 여기서 백엔드가 메시지를 다시 생성하거나 이벤트를 쏘면 UI가 좀비처럼 부활하므로 조용히 종료만 합니다.
                             current_device_pref = None;
                             continue;
                        } else if err_msg.contains("CUDA_ERROR_OUT_OF_MEMORY") || err_msg.contains("out of memory") {
                            let retries = oom_retry_map.entry(task.id.clone()).or_insert(0);
                            
                            if *retries == 0 {
                                *retries += 1;
                                println!("[Scheduler] OOM Detected! VRAM is purged. Retrying on GPU...");
                                current_device_pref = None;

                                
                                let payload = json!({
                                    "task_id": task.id,
                                    "category": "Warning", "summary": "Memory pressure detected. VRAM cleared. Retrying on GPU...", "spinner": "♻️"
                                });
                                let _ = app_handle.emit("extraction-progress", &payload);

                                
                                let log_path = crate::utils::paths::get_task_log_file(Some(&app_handle), &task.id);
                                let _ = std::fs::remove_file(&log_path);
                                
                                {
                                    let store_guard = store.lock().await;
                                    if let Some(db) = store_guard.as_ref() {
                                        let _ = db.update_task_status(&task.id, 10).await;
                                        let _ = db.update_message_status(&task.id, 10, Some("Retrying on GPU...")).await;
                                    }
                                }
                                
                                tokio::time::sleep(Duration::from_secs(2)).await;
                                continue; 
                            } else {
                                if task.r#type == "image_extraction" {
                                    let final_err = "High-resolution image exceeds VRAM capacity. Please try a smaller image.";
                                    println!("[Scheduler] GPU retry failed for Vision. Throwing error instead of freezing on CPU.");
                                    let store_guard = store.lock().await;                            
                                    if let Some(db) = store_guard.as_ref() {
                                        let _ = db.update_task_status(&task.id, crate::logic::parse_status("error")).await;
                                        let _ = db.update_message_status(&task.id, crate::logic::parse_status("error"), Some(&format!("Error: {}", final_err))).await;
                                    }
                                    let _ = app_handle.emit("extraction-progress", json!({
                                        "task_id": task.id,
                                        "category": "Error", "summary": final_err, "spinner": "❌"
                                    }));
                                    current_device_pref = None;
                                } else {
                                    println!("[Scheduler] OOM Detected twice! Activating CPU Mode for text task.");
                                    current_device_pref = Some("cpu".to_string());

                                    // 여기도 더러워진 로그 청소
                                    let log_path = crate::utils::paths::get_task_log_file(Some(&app_handle), &task.id);
                                    let _ = std::fs::remove_file(&log_path);

                                    log_task_progress(&app_handle, &task.id, &json!({
                                        "category": "Warning", "summary": "Memory pressure detected. Retrying with CPU Mode...", "spinner": "💾"
                                    }));
                                    
                                    {
                                        let store_guard = store.lock().await;
                                        if let Some(db) = store_guard.as_ref() {
                                            let _ = db.update_task_status(&task.id, 10).await;
                                            let _ = db.update_message_status(&task.id, 10, Some("Retrying in CPU Mode...")).await;
                                        }
                                    }
                                    
                                    tokio::time::sleep(Duration::from_secs(2)).await;
                                    continue;
                                }
                            }
                        } else if err_msg.contains("No valid model found") || err_msg.to_lowercase().contains("model is missing") {
                            // 🌟 [추가] 모델 파일 누락 에러 감지 시 세팅 화면으로 이동 유도
                            let final_err = "Model files are missing. Please go to Settings and download the required models.";
                            println!("[Scheduler] Model missing error detected. Redirecting user to settings.");

                            let store_guard = store.lock().await;
                            if let Some(db) = store_guard.as_ref() {
                                let _ = db.update_task_status(&task.id, crate::logic::parse_status("error")).await;
                                let _ = db.update_message_status(&task.id, crate::logic::parse_status("error"), Some(&format!("Error: {}", final_err))).await;
                            }

                            let _ = app_handle.emit("extraction-progress", json!({
                                "task_id": task.id,
                                "category": "Error",
                                "summary": final_err,
                                "spinner": "❌"
                            }));

                            // 프론트엔드에 세팅 창을 열도록 특수 이벤트 발송
                            let _ = app_handle.emit("require-model-download", json!({}));

                            current_device_pref = None;
                        } else {
                            let store_guard = store.lock().await;                            
                            if let Some(db) = store_guard.as_ref() {
                                let _ = db.update_task_status(&task.id, crate::logic::parse_status("error")).await;
                                let _ = db.update_message_status(&task.id, crate::logic::parse_status("error"), Some(&format!("Error: {}", err_msg))).await;
                            }
                            
                            let _ = app_handle.emit("extraction-progress", json!({
                                "task_id": task.id,
                                "category": "Error", "summary": format!("Failed: {}", err_msg), "spinner": "❌"
                            }));

                            current_device_pref = None;
                        }
                    }
                }
            }
            
            cancellation_token.store(false, Ordering::SeqCst);
            crate::utils::set_extraction_stop_signal(false); 
        }
    });
}

async fn process_task(
    task: Task,
    store_mutex: &Arc<Mutex<Option<VectorStore>>>,
    model_mutex: &Arc<Mutex<Option<LogisModel>>>,
    cancellation_token: &Arc<AtomicBool>,
    app_handle: &tauri::AppHandle,
    device_preference: Option<String>,
) -> Result<()> {
    
    
    let app_handle_clone = app_handle.clone();
    let tid_clone = task.id.clone();
    let emit_term = move |msg: &str| {
        println!("{}", msg);
        use tauri::Emitter;
        let _ = app_handle_clone.emit("task-console-log", serde_json::json!({"task_id": tid_clone, "text": format!("{}\n", msg)}));
    };

    
    let zero_addr = "0x0000000000000000000000000000000000000000";
    let from_addr = if task.from.is_empty() { zero_addr.to_string() } else { task.from.clone() };
    let team_id = if task.to.is_empty() || task.to == zero_addr { 
        crate::utils::hash::hash_id(&from_addr) 
    } else { 
        task.to.clone() 
    };

    emit_term("\n=======================================");
    emit_term(&format!("[PROCESS] ⚙️ Task {} started processing.", task.id));

    
    if task.r#type == "analytic_extraction" {
        return crate::analytic::process_analytic_task(
            task, store_mutex, model_mutex, cancellation_token, app_handle, device_preference
        ).await;
    }

    let kv_path = utils::paths::get_kv_dir(Some(app_handle)).join(&task.id);
    if kv_path.exists() {
        emit_term(&format!("[PROCESS] Found existing KV cache for task {}. Ready to reuse.", task.id));
    }

    
    let payload = json!({ 
        "task_id": task.id,
        "task_type": task.r#type, 
        "category": "Processing", "summary": "Starting extraction...", "spinner": "⠋" 
    });
    let _ = app_handle.emit("extraction-progress", &payload);
    log_task_progress(app_handle, &task.id, &payload);

    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    let mut task_data: Value = serde_json::from_str(&task.data_json).unwrap_or(json!({}));
    
    
    let search_mode = task_data.get("search_mode").and_then(|s| s.as_str()).unwrap_or("commerce").to_string();

    // [FIX] 작업 유형에 따라 파일명을 자동으로 결정합니다.
    let kv_name = if task.r#type == "image_extraction" {
        Some("image".to_string())
    } else {
        Some("text".to_string())
    };
    
    // [FIX] Robust device preference parsing (supports both "cpu" string and true/false boolean)
    let task_device_pref = if let Some(v) = task_data.get("device_preference") {
        if v.as_str() == Some("cpu") || v.as_bool() == Some(true) {
            Some("cpu".to_string())
        } else {
            None
        }
    } else {
        None
    };
    let effective_device_pref = task_device_pref.as_deref().or(device_preference.as_deref());
    
    let language = "english"; 

    // --- Image Extraction Logic (Qwen 3.5 Pipeline) ---
    if task.r#type == "image_extraction" {
        let image_path = task_data.get("image_path").and_then(|s| s.as_str()).unwrap_or("").to_string();
        
        if !image_path.is_empty() {
            emit_term(&format!("[PROCESS] Task type is 'image_extraction'. Bypassing Vision LLM inference and staging raw image."));
            
            // 🌟 이미지를 Base64로 인코딩하여 프론트엔드가 즉시 렌더링할 수 있게 만듭니다.
            use base64::{Engine, prelude::BASE64_STANDARD};
            let b64_img = if let Ok(bytes) = std::fs::read(&image_path) {
                format!("data:image/png;base64,{}", BASE64_STANDARD.encode(&bytes))
            } else {
                format!("file://{}", image_path)
            };

            let filename = std::path::Path::new(&image_path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("Uploaded Image")
                .to_string();

            let extracted_title = format!("[Image] {}", filename);
            let extracted_desc = "Staged Image content".to_string();

            // 🌟 [CRITICAL FIX] Base64가 아닌 원본 이미지의 파일 시스템 경로를 Pug의 img 태그 형태로 만듭니다.
            let safe_image_path = image_path.replace("\\", "/");
            let pug_image_tag = format!("img(src=\"file://{}\", alt=\"{}\")", safe_image_path, filename);

            let store_guard = store_mutex.lock().await;
            if let Some(db) = store_guard.as_ref() {
                // 🌟 [추가] 기존 아이템 여부 확인 및 속성 보존 처리
                let existing_doc = db.get_item_by_id("items", &task.id).await.unwrap_or(None);
                let is_new = existing_doc.is_none();

                let mut draft_data = json!({
                    "id": task.id.clone(),
                    "type": "draft",
                    "link": format!("file://{}", filename),
                    "html": b64_img.clone(),
                    "yaml": pug_image_tag, 
                    "title": extracted_title.clone(),
                    "description": extracted_desc.clone(),
                    "text": "Staged Image content",
                    "updated_at": chrono::Utc::now().timestamp_millis(),
                    "mode": search_mode.clone() // 🌟 [CRITICAL FIX] 삭제 시 해시 추적을 위해 mode를 반드시 저장해야 합니다.
                });

                if let Some(doc) = existing_doc {
                    if let Ok(parsed) = serde_json::from_str::<Value>(&doc.json_data) {
                        if let Some(obj) = draft_data.as_object_mut() {
                            // 이전에 마스킹된 데이터 및 커스텀 타이틀(data.title), 생성일(created_at) 보존
                            if let Some(masked) = parsed.get("masked") { obj.insert("masked".to_string(), masked.clone()); }
                            if let Some(is_masked) = parsed.get("is_masked") { obj.insert("is_masked".to_string(), is_masked.clone()); }
                            if let Some(masked_text) = parsed.get("masked_text") { obj.insert("masked_text".to_string(), masked_text.clone()); }
                            if let Some(data) = parsed.get("data") { obj.insert("data".to_string(), data.clone()); }
                            if let Some(created_at) = parsed.get("created_at") { obj.insert("created_at".to_string(), created_at.clone()); }
                            if let Some(image_text) = parsed.get("image_text") { obj.insert("image_text".to_string(), image_text.clone()); }
                        }
                    }
                } else {
                    if let Some(obj) = draft_data.as_object_mut() {
                        obj.insert("masked_text".to_string(), json!(b64_img));
                        obj.insert("created_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
                    }
                }

                let _ = db.upsert_item(
                    "items", &task.id, "draft", draft_data.clone(), None,
                    Some(&task.from), Some(&team_id), Some(&task.cc), Some(&task.bcc), Some(&task.r#ref), None
                ).await;

                // 🌟 이미지를 Pages 트리에 반영 (Hostname: Local Image, Pathname: 확장자 그룹화)
                if is_new {
                    let hostname = "Local Image".to_string();
                    
                    // 🌟 [CRITICAL FIX] 파일명 대신 확장자만 추출하여 pathname으로 설정합니다.
                    let ext = std::path::Path::new(&filename)
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("file")
                        .to_lowercase();
                    let pathname = format!(".{}", ext);
                    
                    let cc_val = task.cc.clone();
                    
                    // 중복 체크 및 ID 생성 (확장자 기준으로 page_id가 생성되어 같은 확장자끼리 취합됨)
                    let page_id = crate::utils::hash::hash_id(&format!("page_{}_{}_{}", search_mode, hostname, pathname));
                    
                    let mut page_count = 1;
                    let mut existing_page_data = json!({
                        "id": page_id.clone(),
                        "type": "pages",
                        "mode": search_mode.clone(),
                        "hostname": hostname.clone(),
                        "pathname": pathname.clone(),
                        "cc": cc_val.clone(),
                        "count": 1
                    });

                    if let Ok(Some(existing_page)) = db.get_item_by_id("pages", &page_id).await {
                        if let Ok(mut parsed) = serde_json::from_str::<Value>(&existing_page.json_data) {
                            page_count = parsed.get("count").and_then(|v| v.as_i64()).unwrap_or(0) + 1;
                            parsed.as_object_mut().unwrap().insert("count".to_string(), json!(page_count));
                            existing_page_data = parsed;
                        }
                    }

                    let _ = db.upsert_item(
                        "pages", &page_id, "pages", existing_page_data, None,
                        Some(&task.from), Some(&team_id), Some(&cc_val), Some(&task.bcc), Some(&task.r#ref), None
                    ).await;
                }
            }

            // 🌟 [CRITICAL FIX] 첫 번째 Mutex 락(store_guard)을 명시적으로 해제하여, 아래의 두 번째 락에서 데드락(Deadlock)이 발생하는 것을 원천 차단합니다!
            drop(store_guard);

            let display_summary = format!("{} - {}", extracted_title, extracted_desc);

            // 🌟 [CRITICAL FIX] 상태(1)가 UI에 덮어씌워지는 것을 방어하기 위해 Done 이벤트 발송 직전에 DB도 9로 굳힙니다.
            {
                let store_guard = store_mutex.lock().await;
                if let Some(db) = store_guard.as_ref() {
                    let _ = db.update_task_status(&task.id, 9).await;
                    let _ = db.update_message_status(&task.id, 9, Some(&display_summary)).await;
                }
            }

            let payload = json!({
                "task_id": task.id, 
                "category": "Done", 
                "summary": display_summary, 
                "spinner": "✅",
                "data": null 
            });
            
            let _ = app_handle.emit("extraction-progress", &payload);
            crate::scheduler::log_task_progress(app_handle, &task.id, &payload);

            emit_term("[PROCESS] Image staging completed.");
            return Ok(()); 
        }
    }

    if task.r#type == "mask_documents" {
        // 🌟 [CRITICAL FIX] 모델 로딩 락(Model Lock)을 마스킹 작업 내부로 강등 이동시켜, 
        // 무거운 AI 연산을 쓰지 않는 Draft(웹/이미지 스테이징) 작업들이 큐에서 막히는 병목을 원천 차단합니다!
        let model = {
            println!("[Scheduler] 🛡️ Attempting to acquire Model Lock...");
            let mut model_lock = model_mutex.lock().await;
            println!("[Scheduler] ✅ Model Lock acquired.");
            
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

            if let Some(m) = model_lock.as_ref() {
                let wants_cpu = effective_device_pref == Some("cpu");
                if m.is_cpu_mode != wants_cpu {
                    println!("[Scheduler] Device preference mismatch (Current CPU: {}, Wants CPU: {}). Reloading model...", m.is_cpu_mode, wants_cpu);
                    m.deep_purge_resources().await;
                    *model_lock = None;
                }
            }

            if model_lock.is_none() {
                println!("[Scheduler] Model not initialized. Starting LogisModel::new...");
                log_task_progress(app_handle, &task.id, &json!({ "category": "Loading Model", "summary": "Initializing AI Core..." }));
                
                match LogisModel::new(app_handle.clone(), effective_device_pref).await {
                    Ok(m) => {
                        println!("[Scheduler] LogisModel::new successful.");
                        *model_lock = Some(m);
                    },
                    Err(e) => {
                        println!("[Scheduler] ❌ LogisModel::new failed: {}", e);
                        return Err(anyhow::anyhow!("Model Load Failed: {}", e));
                    }
                }
            }
            model_lock.as_ref().unwrap().clone()
        };

        emit_term("[PROCESS] Starting batch masking for selected documents...");
        
        let uuids = task_data.get("uuids").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        if uuids.is_empty() { return Ok(()); }
        
        let total = uuids.len();
        for (idx, uuid_val) in uuids.iter().enumerate() {
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
            
            let doc_id = uuid_val.as_str().unwrap_or("");
            if doc_id.is_empty() { continue; }

            let store = {
                let store_guard = store_mutex.lock().await;
                store_guard.as_ref().ok_or_else(|| anyhow::anyhow!("Store not initialized"))?.clone()
            };

            // 🌟 [CRITICAL FIX] "items" 테이블뿐만 아니라 모든 테이블을 순회하여 문서를 찾아냅니다.
            let tables = vec!["items", "users", "pages"];
            let mut found_doc = None;
            let mut found_table = "items";

            for table in tables {
                if let Ok(Some(doc)) = store.get_item_by_id(table, doc_id).await {
                    found_doc = Some(doc);
                    found_table = table;
                    break;
                }
            }

            if let Some(doc) = found_doc {
                // 🌟 [CRITICAL FIX] 이미 마스킹된 문서(is_masked: true)라도, 
                // 새로운 니모닉 적용이나 업데이트된 프롬프트를 반영하기 위해 건너뛰지 않고 재처리하도록 제한을 해제합니다.
                // if doc.is_masked {
                //     emit_term(&format!("[EXTRACTION] Skipping document: {} (Already masked)", doc_id));
                //     continue; 
                // }

                let payload = json!({
                    "task_id": task.id,
                    "category": format!("Processing ({}/{})", idx + 1, total),
                    "summary": format!("Extracting data from draft {}...", doc_id),
                    "spinner": "⠋"
                });
                log_task_progress(app_handle, &task.id, &payload);
                emit_term(&format!("[EXTRACTION] Processing document: {} (Table: {})", doc_id, found_table));

                let mut json_data: Value = serde_json::from_str(&doc.json_data).unwrap_or(json!({}));
                let raw_html = json_data.get("html").and_then(|v| v.as_str()).unwrap_or("");
                let is_image = raw_html.starts_with("data:image") || raw_html.starts_with("file://");

                let mut target_text = String::new();
                let mut extracted_json = json!({});
                let mut masked_text = String::new(); // 🌟 변수 생명주기를 [STEP 3]까지 연장하기 위해 밖으로 빼냅니다.
                let mut doc_title = json_data.get("title").and_then(|v| v.as_str()).unwrap_or("Document").to_string(); // 🌟 [스코프 연장] STEP 3에서 접근 가능하도록 상단으로 이동
                let mut doc_desc = json_data.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string(); // 🌟 [스코프 연장] STEP 3에서 접근 가능하도록 상단으로 이동

                // 🌟 [STEP 1] 이미지인 경우 OCR을 선행하여 텍스트를 먼저 추출합니다.
                if is_image {
                    model.secure_vram_relay(crate::model::ModelSize::Qwen3_5, None, Some(cancellation_token.clone()), false, None).await?;
                    
                    let dynamic_image = if raw_html.starts_with("data:image") && raw_html.contains("base64,") {
                        let data = raw_html.split("base64,").nth(1).unwrap_or("");
                        crate::utils::img_utils::load_image_from_base64(data).ok()
                    } else if raw_html.starts_with("file://") {
                        let path = raw_html.replace("file://", "");
                        image::open(&path).ok().map(|img| image::DynamicImage::ImageRgb8(img.to_rgb8()))
                    } else {
                        None
                    };

                    let ocr_prompt = crate::parsing::ocr_image_prompt();

                    let res_ocr = model.chat_with_qwen3_5_image_spinner(
                        "You are a helpful extraction assistant.",
                        &ocr_prompt,
                        dynamic_image,
                        app_handle,
                        "extraction-progress",
                        json!({ "category": format!("Vision OCR ({}/{})", idx + 1, total), "summary": "Extracting text from image..." }),
                        1024,
                        Some(cancellation_token.clone()),
                        Some(format!("{}_{}_ocr", task.id, doc_id))
                    ).await?;

                    let ocr_json = crate::parsing::parse_json_from_llm(&res_ocr);
                    target_text = ocr_json.get("image_text").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    
                    // OCR 된 원본 텍스트를 json_data에 먼저 보존합니다.
                    if let Some(obj) = json_data.as_object_mut() {
                        obj.insert("image_text".to_string(), json!(target_text));
                    }
                } else {
                    let html_val = json_data.get("html").and_then(|v| v.as_str()).unwrap_or("");
                    let url_val = json_data.get("link").and_then(|v| v.as_str()).unwrap_or("");
                    
                    if !html_val.is_empty() {
                        // 🌟 Speedreader 조건 적용: 핵심 본문 영역이 존재할 경우 해당 영역만 추출하여 LLM 토큰 최적화
                        let speedreader_html = crate::parsing::extract_speedreader_content(html_val);
                        // 🌟 정제된 HTML 값을 가져와 태그와 속성을 모두 제거한 순수 들여쓰기 텍스트(YamlMode)로 파싱합니다.
                        target_text = crate::parsing::convert_to_clean_pug(&speedreader_html, crate::parsing::PugMode::NoAttributesMode, Some(url_val));
                    } else {
                        target_text = json_data.get("yaml").and_then(|v| v.as_str()).unwrap_or(&doc.text).to_string();
                    }

                    // 📂 [CRITICAL FIX] 프로젝트의 자체 tmp 경로를 사용하여 디버그 파일을 저장합니다.
                    let app_tmp_dir = crate::utils::paths::get_app_tmp_root(Some(app_handle));
                    let file_path = app_tmp_dir.join("debug_target_text.yaml");
                    
                    // 파일에 target_text 기록
                    if let Err(e) = std::fs::write(&file_path, &target_text) {
                        eprintln!("Failed to write debug file to tmp: {}", e);
                    } else {
                        println!("Debug file saved to: {:?}", file_path);
                    }
                }

                // 🌟 [STEP 2] 확보된 텍스트(웹페이지 PUG 또는 이미지 OCR 결과)를 대상으로 개인정보 마스킹을 수행합니다.
                if !target_text.is_empty() {
                    let task_marker_hash = crate::utils::hash::crc32(&task.id); // 🌟 스코프 상단에 해시 생성

                    // 🌟 [CRITICAL FIX] LLM 토큰 절약 및 링크 훼손 방지를 위해 href, src 속성을 해시 마커로 치환하여 장부에 격리
                    let mut link_map = std::collections::HashMap::new();
                    let mut link_counter = 0;
                    
                    if let Ok(re_double) = regex::Regex::new(r#"(?i)(href|src|data-src)\s*=\s*"([^"]*)""#) {
                        target_text = re_double.replace_all(&target_text, |caps: &regex::Captures| {
                            let original_full = caps[0].to_string();
                            let marker = format!(" [___REDACTED_LINK_{}___] ", link_counter);
                            link_map.insert(marker.trim().to_string(), original_full);
                            link_counter += 1;
                            marker
                        }).to_string();
                    }
                    
                    if let Ok(re_single) = regex::Regex::new(r#"(?i)(href|src|data-src)\s*=\s*'([^']*)'"#) {
                        target_text = re_single.replace_all(&target_text, |caps: &regex::Captures| {
                            let original_full = caps[0].to_string();
                            let marker = format!(" [___REDACTED_LINK_{}___] ", link_counter);
                            link_map.insert(marker.trim().to_string(), original_full);
                            link_counter += 1;
                            marker
                        }).to_string();
                    }

                    // 컨텍스트 크기에 따른 동적 모델 할당 (60,000 초과 시 Qwen, 이하 시 Qwen3)
                    let is_large_context = target_text.len() > 60000;
                    let target_model_size = if is_large_context { crate::model::ModelSize::Qwen } else { crate::model::ModelSize::Qwen3 };

                    // 🌟 [OOM 원인 분석용 로그] 모델에 투입되기 직전 전체 컨텍스트의 문자열 길이를 터미널에 출력합니다.
                    emit_term(&format!("[DEBUG-OOM] 현재 투입되는 컨텍스트 사이즈(문자 수): {}. 선택된 모델: {:?}", target_text.len(), target_model_size));

                    // 🌟 [CRITICAL FIX] Qwen3.5(LLM) 모델 백그라운드 로딩 트리거!
                    // 임베딩 연산과 동시에 모델이 VRAM에 올라가도록 스레드를 분리합니다.
                    let bg_model = model.clone();
                    let bg_cancel = cancellation_token.clone();
                    let llm_load_handle = tokio::spawn(async move {
                        if bg_cancel.load(Ordering::Relaxed) { return; }
                        // deep_purge를 피하기 위해 ensure_ 계열을 직접 호출하여 임베딩 모델 생존 보장
                        match target_model_size {
                            crate::model::ModelSize::Qwen => { let _ = bg_model.ensure_generator_ext(crate::model::ModelSize::Qwen, false, false).await; },
                            crate::model::ModelSize::Qwen3 => { let _ = bg_model.ensure_qwen3().await; },
                            _ => {}
                        }
                    });

                    // 🌟 16개의 마스킹 타겟 항목에 대해 (JSON_키, 추출_설명) 튜플 형태로 분리합니다.
                    let target_items = vec![
                        // 🌟 [수정] Phase 2 펌핑을 위해 조직/팀(company)을 가장 먼저 찾도록 맨 위로 끌어올립니다.
                        ("company", "the name of a company, institution, or group"),
                        // ("given_name", "person's given name"),
                        // ("middle_name", "person's middle name"),
                        // ("family_name", "person's family name or surname"),
                        // 🌟 [수정] email은 정규식으로 사전 처리되므로 LLM 추론 목록에서 제외합니다.
                        ("contact_number", "contact number"),
                        ("name", "person's name"),
                        ("username", "person's username"),
                        ("address", "physical street address"),
                        // ("age", "person's age"),
                        // ("gender_identity", "person's gender identity"),
                        // ("biological_sex", "person's biological sex"),
                        // ("eye_color", "the color of a person's eyes"),
                        // ("height", "person's physical height"),
                        // ("profession", "person's profession or field of work"),
                        // ("job_position", "person's specific job position or role"),
                        // ("department", "person's specific organizational division or department"),
                    ];

                    // 🌟 [추가] bias.json 파일을 로드하여 privacy 객체 파싱 (선행 필터링용)
                    let bias_str = include_str!("bias.json");
                    let bias_json: Value = serde_json::from_str(bias_str).unwrap_or(json!({}));

                    // 🌟 [수정] 다국어 유니코드 감지 및 빈도수 기반 Local 언어 확정 로직
                    let mut language_counts = std::collections::HashMap::new();
                    
                    for c in target_text.chars() {
                        let u = c as u32;
                        let lang = if (u >= 0x0041 && u <= 0x005A) || (u >= 0x0061 && u <= 0x007A) { "english" }
                        else if (u >= 0xAC00 && u <= 0xD7A3) || (u >= 0x1100 && u <= 0x11FF) || (u >= 0x3130 && u <= 0x318F) { "korean" }
                        else if (u >= 0x3040 && u <= 0x309F) || (u >= 0x30A0 && u <= 0x30FF) { "japanese" }
                        else if u >= 0x4E00 && u <= 0x9FFF { "chinese" }
                        else if u >= 0x0400 && u <= 0x04FF { "russian" }
                        else if u >= 0x0600 && u <= 0x06FF { "arabic" }
                        else if u >= 0x0E00 && u <= 0x0E7F { "thai" }
                        else if u >= 0x0900 && u <= 0x097F { "hindi" }
                        else if u >= 0x0980 && u <= 0x09FF { "bengali" }
                        else if u >= 0x0370 && u <= 0x03FF { "greek" }
                        else if u >= 0x0590 && u <= 0x05FF { "hebrew" }
                        else if u >= 0x1EA0 && u <= 0x1EF9 { "vietnamese" }
                        else if u >= 0x00C0 && u <= 0x00FF { "european" }
                        else { "" };

                        if !lang.is_empty() {
                            *language_counts.entry(lang).or_insert(0) += 1;
                        }
                    }
                    
                    let mut detected_languages_vec: Vec<String> = language_counts.keys().map(|s| s.to_string()).collect();
                    if detected_languages_vec.is_empty() { detected_languages_vec.push("english".to_string()); }

                    // 가장 많이 사용된 언어를 local_language로 확정
                    let local_language = language_counts.into_iter()
                        .max_by_key(|&(_, count)| count)
                        .map(|(lang, _)| lang.to_string())
                        .unwrap_or_else(|| "english".to_string());

                    emit_term(&format!("[EXTRACTION] 🌐 Detected Languages: {:?} (Local: {})", detected_languages_vec, local_language));

                    // 🌟 [추가] 언어 기반 동적 타겟 확장 로직
                    // bias.json에서는 기존 키를 사용하여 설정값을 가져오고, 컨텍스트(LLM)에만 언어 접두사를 붙여 전달합니다.
                    let mut dynamic_target_items: Vec<(String, String, String)> = Vec::new();
                    for (base_target, base_desc) in target_items {
                        // 🌟 언어 맥락이 결과 품질에 큰 영향을 미치는 고유명사 형태의 타겟들을 지정합니다.
                        if base_target == "name" || base_target == "company" || base_target == "address" || base_target == "username" || base_target == "contact_number" {
                            for lang in &detected_languages_vec {
                                let context_name = format!("{}_{}", lang, base_target);
                                let context_desc = format!("{} {}", lang, base_desc);
                                dynamic_target_items.push((context_name, base_target.to_string(), context_desc));
                            }
                        } else {
                            dynamic_target_items.push((base_target.to_string(), base_target.to_string(), base_desc.to_string()));
                        }
                    }

                    // 🌟 [추가] 코사인 유사도 산출 헬퍼 함수
                    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
                        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                        if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot_product / (norm_a * norm_b) }
                    }

                    emit_term("[EXTRACTION] 🧠 [포그라운드] Embedding 모델 로드 및 벡터 유사도 사전 검증 시작...");
                    model.ensure_embedding().await?;

                    // 🌟 전역 상태 변수들 최상단으로 호이스팅 (1차, 2차 패스 공유)
                    let mut all_matches = Vec::new();
                    masked_text = target_text.clone(); 
                    let mut skip_counter = 0; 
                    let mut skip_map = std::collections::HashMap::new(); 
                    let mut replacement_history: Vec<(String, String)> = Vec::new(); 
                    let mut domain_history: Vec<(String, String)> = Vec::new(); 
                    let task_marker_hash = crate::utils::hash::crc32(&task.id); // 🌟 [CRITICAL FIX] CRC32 해싱 기반 고유 마커 뼈대 생성

                    // 🌟 [사전 정규식 추출] email 먼저 마스킹 (1차 패스 전)
                    if let Ok(email_re) = regex::Regex::new(r"(?i)[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}") {
                        let mut found_emails = std::collections::HashSet::new();
                        for mat in email_re.find_iter(&masked_text) { found_emails.insert(mat.as_str().to_string()); }
                        for email_val in found_emails {
                            let mnemonic = crate::parsing::generate_mnemonic();
                            let upper_key = "EMAIL".to_string();
                            let final_replacement = format!("[{}: {}]", upper_key, mnemonic);
                            let skip_marker = format!("[___REDACTED_{}___]", skip_counter);
                            masked_text = masked_text.replace(&email_val, &skip_marker);
                            doc_title = doc_title.replace(&email_val, &skip_marker);
                            doc_desc = doc_desc.replace(&email_val, &skip_marker);
                            skip_map.insert(skip_marker.clone(), final_replacement);
                            replacement_history.push((email_val.clone(), skip_marker.clone()));
                            domain_history.push(("email".to_string(), email_val.clone()));
                            skip_counter += 1;
                            all_matches.push(json!({ "name": upper_key, "value": email_val, "mnemonic": mnemonic }));
                            emit_term(&format!("[EXTRACTION] 📧 이메일 정규식 사전 추출 성공: {} -> 강제 마스킹 완료", email_val));
                        }
                    }

                    // 🌟 [CRITICAL FIX] 메타데이터 노이즈(/사진=, /AFPBBNews=뉴스1 등)를 해시 마커로 치환하여 외부 장부(Shadow Map)에 격리
                    let mut noise_map = std::collections::HashMap::new();
                    let mut noise_counter = 0;
                    if let Ok(noise_re) = regex::Regex::new(r"(?i)(/(?:사진|AFPBBNews|뉴스1|기자|특파원|Copyright|ⓒ|©)[^\n]*)") {
                        masked_text = noise_re.replace_all(&masked_text, |caps: &regex::Captures| {
                            let n_marker = format!(" [___REDACTED_NOISE_{}___] ", noise_counter);
                            noise_map.insert(n_marker.trim().to_string(), caps[1].to_string());
                            noise_counter += 1;
                            n_marker
                        }).to_string();
                        
                        doc_title = noise_re.replace_all(&doc_title, |caps: &regex::Captures| {
                            let n_marker = format!(" [___REDACTED_NOISE_{}___] ", noise_counter);
                            noise_map.insert(n_marker.trim().to_string(), caps[1].to_string());
                            noise_counter += 1;
                            n_marker
                        }).to_string();

                        doc_desc = noise_re.replace_all(&doc_desc, |caps: &regex::Captures| {
                            let n_marker = format!(" [___REDACTED_NOISE_{}___] ", noise_counter);
                            noise_map.insert(n_marker.trim().to_string(), caps[1].to_string());
                            noise_counter += 1;
                            n_marker
                        }).to_string();
                    }

                    // 🌟 1차 패스용 라인 분할 (masked_text 기반)
                    let structural_tags = ["html", "body", "div", "p", "span", "thead", "tbody", "tr", "td", "th", "table"];
                    let mut lines: Vec<String> = masked_text.lines()
                        .map(|s| s.trim().trim_start_matches('|').trim().to_string())
                        .filter(|s| {
                            let s_lower = s.to_lowercase();
                            s.len() > 2 && !structural_tags.contains(&s_lower.as_str())
                        })
                        .collect();

                    // =====================================================================
                    // 🌟 [PASS 1 폐기] Company 단독 추출을 없애고 전체 NMS(PASS 2)로 통합
                    // 🌟 LLM 무조건 1차 대기 (Inference를 위해)
                    // =====================================================================
                    emit_term("[EXTRACTION] 🧠 LLM 로딩 동기화 대기 중...");
                    let _ = llm_load_handle.await;
                    model.secure_vram_relay(target_model_size, None, Some(cancellation_token.clone()), false, None).await?;

                    // =====================================================================
                    // 🌟 [PASS 2: NMS 기반 전체 추출] 동적 접두사 기반 벡터 검색 및 타이브레이커 적용
                    // =====================================================================
                    emit_term("[EXTRACTION] ⚔️ [PASS 2] 텍스트 단위 NMS 배틀 및 전체 항목 추출 시작...");
                    
                    lines = masked_text.lines()
                        .map(|s| s.trim().trim_start_matches('|').trim().to_string())
                        .filter(|s| {
                            let s_lower = s.to_lowercase();
                            s.len() > 2 && !structural_tags.contains(&s_lower.as_str())
                        })
                        .collect();
                        
                    emit_term(&format!("[EXTRACTION] 본문을 {}개의 라인으로 분할하여 순차 임베딩 및 NMS 배틀 진행 중...", lines.len()));

                    // 🌟 1. 다국어 접두사가 결합된 타겟(도메인) 및 서술어(verb_expression) 임베딩 장전
                    let mut target_biases_embs = Vec::new();
                    let mut target_prejs_embs = Vec::new();

                    for (c_name, base_target, _) in &dynamic_target_items {
                        let mut b_val = "".to_string();
                        let mut p_val = "".to_string();
                        if let Some(privacy_node) = bias_json.get("privacy").and_then(|v| v.get(base_target)) {
                            b_val = privacy_node.get("bias").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            p_val = privacy_node.get("prejudice").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        }
                        if p_val.trim().is_empty() { p_val = "random unrelated noise".to_string(); }

                        // 언어 꼬리표 추출
                        let lang_prefix = c_name.split('_').next().unwrap_or("english");
                        
                        // 🌟 다국어 동적 합성 (Dynamic Prefixing)
                        let prefixed_b_val = b_val.split(',')
                            .map(|s| format!("{} {}", lang_prefix, s.trim()))
                            .collect::<Vec<_>>().join(", ");
                        let prefixed_p_val = p_val.split(',')
                            .map(|s| format!("{} {}", lang_prefix, s.trim()))
                            .collect::<Vec<_>>().join(", ");

                        let b_emb = model.get_embedding(prefixed_b_val).await.unwrap_or_else(|_| vec![0.0; 384]);
                        let p_emb = model.get_embedding(prefixed_p_val).await.unwrap_or_else(|_| vec![0.0; 384]);
                        target_biases_embs.push(b_emb);
                        target_prejs_embs.push(p_emb);
                    }

                    // 🌟 1-1. 서술어구(verb_expression) 타이브레이커 가이드 벡터 생성 (감지된 모든 다국어 반영)
                    let mut prefixed_verb_b_vals = Vec::new();
                    
                    for lang in &detected_languages_vec {
                        let verb_val = bias_json.get("verb")
                            .and_then(|v| v.get("bias"))
                            .and_then(|v| v.get(lang))
                            .and_then(|v| v.as_str())
                            .unwrap_or("verb, predicate");
                            
                        let expr_val = bias_json.get("expression")
                            .and_then(|v| v.get("bias"))
                            .and_then(|v| v.get(lang))
                            .and_then(|v| v.as_str())
                            .unwrap_or("idiom, phrase");
                            
                        let combined_verb_expr = format!("{}, {}", verb_val, expr_val);
                            
                        let prefixed = combined_verb_expr.split(',')
                            .map(|s| format!("{} {}", lang, s.trim()))
                            .collect::<Vec<_>>().join(", ");
                            
                        prefixed_verb_b_vals.push(prefixed);
                    }
                    
                    let combined_verb_b_val = prefixed_verb_b_vals.join(", ");
                    let verb_emb = model.get_embedding(combined_verb_b_val).await.unwrap_or_else(|_| vec![0.0; 384]);

                    // 🌟 2. Sliding Window를 통한 단어 단위 청크(Chunk) 생성 및 기초 점수 산출
                    #[derive(Clone)]
                    struct ChunkSpan {
                        line_idx: usize,
                        start: usize,
                        end: usize,
                        text: String,
                        target_indices: Vec<usize>, // 🌟 [수정] 0.05점 편차 이내의 공동 우승자들을 모두 저장
                        score: f32,
                    }
                    let mut raw_spans = Vec::new();

                    let noise_marker_prefix = "[___REDACTED_NOISE_".to_string();
                    let link_marker_prefix = "[___REDACTED_LINK_".to_string();

                    for (line_idx, line) in lines.iter().enumerate() {
                        if cancellation_token.load(Ordering::Relaxed) { break; }
                        
                        let words: Vec<&str> = line.split_whitespace().collect();
                        
                        for start in 0..words.len() {
                            let max_end = words.len().min(start + 4); // 1~4 단어 조합
                            for end in (start + 1)..=max_end {
                                // 🌟 [CRITICAL FIX] 노이즈, 링크 해시 마커를 벡터 청크에서 제외하여 순수 텍스트만 임베딩합니다.
                                let clean_words: Vec<&str> = words[start..end].iter()
                                    .filter(|&&w| !w.starts_with(&noise_marker_prefix) && !w.starts_with(&link_marker_prefix))
                                    .copied()
                                    .collect();
                                
                                if clean_words.is_empty() { continue; }
                                let chunk_text = clean_words.join(" ");

                                // 특수기호만 존재하는 무의미한 텍스트 스킵
                                if chunk_text.trim().chars().all(|c| !c.is_alphanumeric()) { continue; }

                                let chunk_emb = model.get_embedding(chunk_text.clone()).await.unwrap_or_else(|_| vec![0.0; 384]);
                                let word_count = end - start;
                                let length_weight = 1.0; // 단어 개수 가중치 제거 (길이 무관 동등 점수)

                                // 🌟 서술어(verb_expression) 타이브레이커 계산 (단어 1개 이상 모두 적용)
                                let v_sim = cosine_similarity(&chunk_emb, &verb_emb);
                                // 1~2단어 0.05, 3단어 이상 0.10 차등 감점
                                let beta = if word_count <= 2 { 0.05 } else { 0.10 };
                                let verb_penalty = v_sim * beta;

                                let mut top_targets: Vec<(usize, f32)> = Vec::new();

                                // 모든 타겟 도메인과 벡터 유사도를 대결시켜 현재 청크에 적합한 도메인들을 찾음
                                for i in 0..dynamic_target_items.len() {
                                    let b_score = cosine_similarity(&chunk_emb, &target_biases_embs[i]);
                                    let p_score = cosine_similarity(&chunk_emb, &target_prejs_embs[i]);
                                    
                                    let penalty_weight = if word_count <= 2 { 0.3 } else { 0.7 };
                                    
                                    // 🌟 타이브레이커 감점을 최종 스코어에 반영
                                    let score = b_score - (p_score * penalty_weight) - verb_penalty;

                                    top_targets.push((i, score));
                                }

                                // 점수 순으로 내림차순 정렬
                                top_targets.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                                let best_score = top_targets[0].1;

                                // 🌟 2차 패스 커트라인 상향 및 0.05 편차(Margin) 공동 우승 허용 로직 적용
                                if best_score > 0.25 {
                                    let final_score = best_score * length_weight;

                                    if verb_penalty > 0.005 {
                                        emit_term(&format!("    📉 [VERB PENALTY] '{}' -> 감점: {:.4} (최종 반영 스코어: {:.4})", chunk_text, verb_penalty, final_score));
                                    }

                                    let mut selected_indices = Vec::new();
                                    
                                    for (idx, score) in top_targets {
                                        if best_score - score <= 0.05 && score > 0.25 {
                                            selected_indices.push(idx);
                                        } else {
                                            break;
                                        }
                                    }

                                    raw_spans.push(ChunkSpan {
                                        line_idx, start, end, text: chunk_text, target_indices: selected_indices, score: final_score
                                    });
                                }
                            }
                        }
                    }

                    // 🌟 3. 앞뒤 교차 문장(Context) 점수 합산 (Pass 2)
                    emit_term("  🔄 [PASS 2: CONTEXT ADJUSTMENT] Merging adjacent scores...");
                    let mut eval_spans = Vec::new();
                    for i in 0..raw_spans.len() {
                        let target = &raw_spans[i];
                        let mut prev_bonus = 0.0;
                        let mut next_bonus = 0.0;

                        for j in 0..raw_spans.len() {
                            if i == j { continue; }
                            let other = &raw_spans[j];
                            // 동일 라인 내에서 타겟 도메인 교집합이 있을 때만 보너스 교환
                            let has_common_target = other.target_indices.iter().any(|idx| target.target_indices.contains(idx));
                            if other.line_idx == target.line_idx && has_common_target {
                                if other.start < target.start && other.end > target.start && other.score > prev_bonus { prev_bonus = other.score; }
                                if other.end > target.end && other.start < target.end && other.score > next_bonus { next_bonus = other.score; }
                            }
                        }
                        
                        let final_context_score = target.score + (prev_bonus * 0.5) + (next_bonus * 0.5);
                        eval_spans.push(ChunkSpan { score: final_context_score, ..target.clone() });
                    }

                    // 🌟 4. NMS BATTLE (Pass 3) - 오버랩 충돌 해결 (승자 독식)
                    emit_term("  ⚔️ [PASS 3: NMS BATTLE] Resolving Overlaps across all targets...");
                    eval_spans.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                    let mut final_spans: Vec<ChunkSpan> = Vec::new();

                    for span in eval_spans {
                        let mut is_overlapped = false;
                        for selected in &final_spans {
                            if span.line_idx == selected.line_idx {
                                if span.start < selected.end && span.end > selected.start {
                                    is_overlapped = true;
                                    break;
                                }
                            }
                        }
                        
                        if !is_overlapped {
                            for &t_idx in &span.target_indices {
                                // 🌟 [로그 가시성 개선] 축약된 base_target 대신 다국어 정보가 결합된 context_name을 가져옵니다.
                                let (context_name, _, _) = &dynamic_target_items[t_idx];
                                let display_name = context_name.replace("_", " ");
                                emit_term(&format!("    👑 [WINNER] '{}' -> {} (Score: {:.4})", span.text, display_name, span.score));
                            }
                            final_spans.push(span);
                        } else {
                            for &t_idx in &span.target_indices {
                                // 🌟 [로그 가시성 개선] 패배(Absorbed) 로그에도 어떤 언어 트랙에서 충돌이 터졌는지 명확히 명시합니다.
                                let (context_name, _, _) = &dynamic_target_items[t_idx];
                                let display_name = context_name.replace("_", " ");
                                emit_term(&format!("    💀 [DEFEAT] '{}' -> {} (Absorbed: {:.4})", span.text, display_name, span.score));
                            }
                        }
                    }

                    // 🌟 [추가] 4.5 인접 청크 동적 스팬 확장 (Dynamic Span Expansion)
                    // 물리적으로 인접한 청크들이 만들어낼 수 있는 모든 결합 경우의 수를 파생시켜 후보군에 추가합니다.
                    emit_term("  🔗 [PASS 3.5: GAP BRIDGING] Dynamic Span Expansion for adjacent winner chunks...");
                    final_spans.sort_by(|a, b| a.line_idx.cmp(&b.line_idx).then(a.start.cmp(&b.start)));
                    
                    let mut expanded_spans: Vec<ChunkSpan> = final_spans.clone(); // 기존 단일 조각들 보존
                    
                    // 연속된 청크들을 그룹화
                    let mut contiguous_groups: Vec<Vec<ChunkSpan>> = Vec::new();
                    let mut current_group: Vec<ChunkSpan> = Vec::new();
                    
                    for span in final_spans {
                        if let Some(last) = current_group.last() {
                            // 🌟 [CRITICAL FIX] 카테고리 100% 일치 제약을 완전히 삭제! 
                            // 승리한 조각들이 같은 줄에 있고 물리적으로 맞닿아 있기만 하다면 무조건 연쇄 결합 그룹에 포함시킵니다.
                            if last.line_idx == span.line_idx && last.end == span.start {
                                // 🌟 두 청크 중 최소 하나는 '단어 1개'로 구성된 경우에만 연속성을 인정하여 과잉 그룹화 방지
                                let last_word_count = last.end - last.start;
                                let span_word_count = span.end - span.start;
                                
                                if last_word_count == 1 || span_word_count == 1 {
                                    current_group.push(span);
                                    continue;
                                }
                            }
                        }
                        if !current_group.is_empty() {
                            contiguous_groups.push(current_group.clone());
                        }
                        current_group = vec![span];
                    }
                    if !current_group.is_empty() {
                        contiguous_groups.push(current_group);
                    }
                    
                    // 각 그룹 내에서 '오직 2개의 연속된 조각(Pair)' 결합 경우의 수만 생성합니다. (제약 완전 철폐)
                    for group in contiguous_groups {
                        let n = group.len();
                        if n >= 2 {
                            let len = 2; // 🌟 [CRITICAL FIX] 무조건 2조각까지만 이어지도록 결합 길이를 2로 고정합니다.
                            for start_idx in 0..=(n - len) {
                                let sub_group = &group[start_idx..(start_idx + len)];
                                
                                let combined_text = sub_group.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" ");
                                let max_score = sub_group.iter().map(|s| s.score).fold(0.0, f32::max);
                                
                                // 🌟 파생된 거대 조각은 결합된 모든 조각들의 타겟(카테고리)을 합집합으로 가집니다. (이종 카테고리 제약 완전 철폐)
                                let mut combined_targets = Vec::new();
                                for s in sub_group {
                                    combined_targets.extend(s.target_indices.clone());
                                }
                                combined_targets.sort();
                                combined_targets.dedup();
                                
                                let new_span = ChunkSpan {
                                    line_idx: sub_group[0].line_idx,
                                    start: sub_group[0].start,
                                    end: sub_group.last().unwrap().end,
                                    text: combined_text.clone(),
                                    target_indices: combined_targets,
                                    score: max_score,
                                };
                                
                                emit_term(&format!("    🤝 [EXPANDED] 2조각 결합 파생: '{}'", combined_text));
                                expanded_spans.push(new_span);
                            }
                        }
                    }
                    let final_spans = expanded_spans;

                    // 🌟 5. NMS 승자들을 바탕으로 매칭된 라인 및 valid_targets 재조립
                    // 구조: (target_name, base_target, target_desc, bias_keyword, prejudice, is_phase2, specific_line_text, specific_candidate_text)
                    let mut valid_targets: Vec<(String, String, String, String, String, bool, String, String)> = Vec::new(); // 🌟 PUG 한 줄, 단일 키워드 1:1 맵핑

                    for span in &final_spans {
                        if span.score >= 0.10 {
                            let specific_line = lines[span.line_idx].clone();
                            let specific_candidate = span.text.clone(); // 🌟 이게 바로 NMS WINNER / EXPANDED 값입니다!

                            for &t_idx in &span.target_indices {
                                let (context_name, base_target, context_desc) = &dynamic_target_items[t_idx];
                                
                                if let Some(privacy_node) = bias_json.get("privacy").and_then(|v| v.get(base_target)) {
                                    let bias_val = privacy_node.get("bias").and_then(|v| v.as_str()).unwrap_or("");
                                    let prej_val = privacy_node.get("prejudice").and_then(|v| v.as_str()).unwrap_or("");
                                    
                                    let bias_keywords: Vec<&str> = bias_val.split(',')
                                        .map(|s| s.trim())
                                        .filter(|s| !s.is_empty() && s.chars().any(|c| c.is_alphabetic()))
                                        .collect();
                                    
                                    for keyword in bias_keywords {
                                        let lang_prefix = context_name.split('_').next().unwrap_or("english");
                                        let split_target_name = format!("{}_{}", lang_prefix, keyword);
                                        let split_target_desc = format!("{} associated with '{}'", context_desc, keyword);

                                        valid_targets.push((
                                            split_target_name,
                                            base_target.to_string(),
                                            split_target_desc,
                                            keyword.to_string(),
                                            prej_val.to_string(),
                                            false,
                                            specific_line.clone(),
                                            specific_candidate.clone() // 🌟 단일 NMS 우승 단어 1:1 매핑
                                        ));
                                    }
                                }
                            }
                        }
                    }

                    emit_term("[EXTRACTION] 🧠 2차 패스 벡터 유사도 검증 완료. 추론 준비...");
                    
                    emit_term("[EXTRACTION] ✅ LLM 2차 추론 루프 시작.");

                    let total_valid = valid_targets.len();

                    let mut p_idx = 0;
                    let mut phase2_companies: Vec<String> = Vec::new();
                    let mut phase2_executed = false; // Phase 2 진입 플래그
                    
                    // 🌟 [추가] 이미 성공적으로 마스킹된 NMS 후보를 기록하여 동의어 트랙에서 건너뛰게 하는 장부
                    let mut fully_masked_candidates = std::collections::HashSet::new();

                    // 🌟 각 속성별로 매칭이 안 될 때까지 무한 반복(loop)하며 순차적으로 처리합니다.
                    while p_idx < valid_targets.len() {
                        if cancellation_token.load(Ordering::Relaxed) { break; }
                        
                        // 🌟 [추가] Phase 2 진입 확인 및 펌핑(Pumping) 로직
                        if p_idx == total_valid && !phase2_executed {
                            phase2_executed = true;
                            if !phase2_companies.is_empty() {
                                emit_term(&format!("[EXTRACTION] 🔄 Phase 2: Generating context-aware targets for {} companies...", phase2_companies.len()));
                                
                                for company in &phase2_companies {
                                    let clean_company = company.trim();
                                    if clean_company.is_empty() { continue; }

                                    for lang in &detected_languages_vec {
                                        let targets_to_add = vec![
                                            ("name", "person's name", "reporter, player, coach, representative, council member, mr, ms, name, person, first name, full name", "company, corporate, business, team, location, place, animal, object"),
                                            ("username", "person's username", "id, username, nickname, account, user", "real name, email, password")
                                        ];

                                        for (b_target, b_desc, b_bias, b_prej) in targets_to_add {
                                            // 저장용 부모 키. ex) 레알 마드리드_korean_name
                                            let c_name_target = format!("{}_{}_{}", clean_company, lang, b_target);
                                            let c_name_desc = format!("{} associated with '{}'", b_desc, clean_company);
                                            
                                            // Vector Search
                                            let bias_emb_vec = model.get_embedding(b_bias.to_string()).await.unwrap_or_else(|_| vec![0.0; 384]);
                                            let prej_emb_vec = model.get_embedding(b_prej.to_string()).await.unwrap_or_else(|_| vec![0.0; 384]);

                                            for (i, line_text) in lines.iter().enumerate() {
                                                let line_emb = model.get_embedding(line_text.clone()).await.unwrap_or_else(|_| vec![0.0; 384]);
                                                let b_score = cosine_similarity(&line_emb, &bias_emb_vec);
                                                let p_score = cosine_similarity(&line_emb, &prej_emb_vec);
                                                let score = b_score - (p_score * 0.3);
                                                
                                                if score >= 0.10 {
                                                    emit_term(&format!("[EXTRACTION] ✅ Phase 2 Vector matched for {} in line {}", c_name_desc, i));
                                                    let specific_line = lines[i].clone();

                                                    let bias_keywords: Vec<&str> = b_bias.split(',')
                                                        .map(|s| s.trim())
                                                        .filter(|s| !s.is_empty() && s.chars().any(|c| c.is_alphabetic()))
                                                        .collect();
                                                    
                                                    for keyword in bias_keywords {
                                                        let split_target_name = format!("{}_{}", c_name_target, keyword);
                                                        let split_target_desc = format!("{} associated with '{}'", c_name_desc, keyword);

                                                        valid_targets.push((
                                                            split_target_name,
                                                            c_name_target.clone(), // 🌟 Phase 2에서는 이것을 덮어씀
                                                            split_target_desc,
                                                            keyword.to_string(),
                                                            b_prej.to_string(),
                                                            true, // 🌟 Phase 2 플래그!
                                                            specific_line.clone(),
                                                            String::new() // 🌟 Phase 2는 LLM이 찾도록 단서를 비워둠
                                                        ));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // 만약 valid_targets.len()을 넘어섰다면 진짜 끝
                        if p_idx >= valid_targets.len() { break; }
                        let (target_name, base_target, target_item, target_bias, target_prejudice, is_phase2, specific_line, specific_candidate) = valid_targets[p_idx].clone();
                        p_idx += 1;

                        // 🌟 [추가] 이미 다른 동의어 트랙에서 마스킹을 마친 후보(specific_candidate)인지 확인하고 건너뜁니다 (접근법 A)
                        if !specific_candidate.is_empty() && fully_masked_candidates.contains(&specific_candidate) {
                            emit_term(&format!("[DEBUG] 이미 다른 동의어 트랙에서 마스킹 완료된 타겟입니다. 스킵합니다: '{}'", specific_candidate));
                            continue;
                        }
                        
                        let mut current_target_found: Vec<String> = Vec::new();
                        let mut ignore_list: Vec<String> = Vec::new();
                        let mut localized_guide_str = String::new();
                        
                        ignore_list.push("[___REDACTED_".to_string());
                        ignore_list.push("REDACTED".to_string());
                        
                        for (history_target, history_val) in &domain_history {
                            if *history_target != target_name {
                                ignore_list.push(history_val.clone());
                                ignore_list.push(format!(" {}", history_val));
                                ignore_list.push(format!("\"{}", history_val));
                                ignore_list.push(format!(" \"{}", history_val));
                            }
                        }

                        let mut miss_counter = 0;
                        let mut item_extract_count = 0;
                        let mut current_temperature: f64 = 0.0;
                        let mut value_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
                        
                        // 🌟 [CRITICAL FIX] Qwen3에게 전체 문서(masked_text)를 주지 않고, 
                        // 앞서 벡터 유사도로 0.10점을 넘긴(통과한) 정확히 PUG 한 줄만 제공합니다!
                        let mut matched_context = specific_line.clone();

                        for (original_val, marker) in &replacement_history {
                            if is_phase2 {
                                if let Some(final_repl) = skip_map.get(marker) {
                                    if final_repl.contains("NAME:") || final_repl.contains("USERNAME:") {
                                        continue; 
                                    }
                                }
                            }
                            matched_context = matched_context.replace(original_val, marker);
                        }

                        loop {
                            if cancellation_token.load(Ordering::Relaxed) { break; }
                            
                            if current_temperature >= 1.0 {
                                emit_term(&format!("[EXTRACTION] 🛑 온도 1.0 도달. {} 항목 종료.", target_item));
                                break;
                            }

                            let already_found_str = if current_target_found.is_empty() {
                                "".to_string()
                            } else {
                                current_target_found.iter().map(|s| format!("\"{}\"", s.replace("\"", "\\\""))).collect::<Vec<_>>().join(", ")
                            };

                            let not_found_str = if ignore_list.is_empty() {
                                "".to_string()
                            } else {
                                ignore_list.iter().map(|s| format!("\"{}\"", s.replace("\"", "\\\""))).collect::<Vec<_>>().join(", ")
                            };

                            // 🌟 [신규 추가] Vector가 찾아낸 1개의 후보 단서를 전달
                            let candidates_str = if specific_candidate.is_empty() {
                                "".to_string()
                            } else {
                                format!("\"{}\"", specific_candidate.replace("\"", "\\\""))
                            };

                            let mut current_lang = target_name.split('_').next().unwrap_or("english");
                            if !detected_languages_vec.contains(&current_lang.to_string()) {
                                current_lang = detected_languages_vec.first().map(|s| s.as_str()).unwrap_or("english");
                            }
                            
                            let current_verb_hint = bias_json.get("verb")
                                .and_then(|v| v.get("bias"))
                                .and_then(|v| v.get(current_lang))
                                .and_then(|v| v.as_str())
                                .unwrap_or("verb, predicate");
                                
                            let current_expr_hint = bias_json.get("expression")
                                .and_then(|v| v.get("bias"))
                                .and_then(|v| v.get(current_lang))
                                .and_then(|v| v.as_str())
                                .unwrap_or("idiom, phrase");

                            let mut clean_matched_context = matched_context.clone();
                            if let Ok(marker_re) = regex::Regex::new(r"\[___REDACTED_(LINK|NOISE)_\d+___\]") {
                                clean_matched_context = marker_re.replace_all(&clean_matched_context, " ").to_string();
                            }

                            // 🌟 [CRITICAL FIX] NMS 값(specific_candidate)이 바로 input_keyword 로 할당됩니다.
                            let input_keyword = if specific_candidate.is_empty() {
                                target_bias.clone()
                            } else {
                                specific_candidate.clone()
                            };

                            let (mut system_prompt, user_prompt) = crate::parsing::build_extraction_prompt(&doc_title, &clean_matched_context, &target_name, &target_item, &target_bias, &target_prejudice, &already_found_str, &not_found_str, &candidates_str, &local_language, &localized_guide_str, &input_keyword);
                            
                            let payload = json!({ 
                                "task_id": task.id.clone(),
                                "category": format!("Masking ({}/{}) - Type {}", idx + 1, total, p_idx + 1), 
                                "summary": format!("Anonymizing {}...", target_item),
                                "spinner": "⠋"
                            });
                            let _ = app_handle.emit("extraction-progress", &payload);
                            crate::scheduler::log_task_progress(app_handle, &task.id, &payload);

                            let cancel_clone = cancellation_token.clone();
                            let system_prompt_clone = system_prompt.clone();
                            let user_prompt_clone = user_prompt.clone();
                            let session_id_clone = format!("{}_{}", task.id, doc_id);

                            let res_mask = if is_large_context {
                                let gen_arc = model.generator.clone();
                                let mut gen_guard = gen_arc.lock().await;
                                if let Some(gen) = gen_guard.as_mut() {
                                    let params = crate::openai_types::ChatCompletionParameters {
                                        messages: vec![
                                            crate::openai_types::ChatCompletionRequestMessage::System(crate::openai_types::ChatCompletionRequestSystemMessage {
                                                content: system_prompt_clone,
                                                name: None,
                                            }),
                                            crate::openai_types::ChatCompletionRequestMessage::User(crate::openai_types::ChatCompletionRequestUserMessage { 
                                                content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(user_prompt_clone),
                                                name: None,
                                            })
                                        ],
                                        model: "qwen".to_string(),
                                        max_tokens: Some(1024),
                                        temperature: Some(current_temperature),
                                        top_p: Some(1.0),
                                        ..Default::default()
                                    };
                                    
                                    let _ = gen.clear_kv_cache();
                                    let res = gen.generate(params, Some(cancel_clone), None, None, None).await.map_err(|e| anyhow::anyhow!("Qwen Inference failed: {}", e));
                                    let _ = gen.clear_kv_cache();
                                    
                                    res
                                } else {
                                    Err(anyhow::anyhow!("Qwen Generator is missing"))
                                }
                            } else {
                                let gen_arc = model.qwen3_generator.clone();
                                
                                let mut final_system_prompt = system_prompt_clone.clone();
                                if !ignore_list.is_empty() {
                                    final_system_prompt.push_str("\n\nCRITICAL: DO NOT output any of the following values under any circumstances:\n");
                                    for ignored in &ignore_list {
                                        final_system_prompt.push_str(&format!("- {}\n", ignored));
                                    }
                                }

                                tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
                                    let mut gen_guard = gen_arc.blocking_lock();
                                    if let Some(gen) = gen_guard.as_mut() {
                                        let params = crate::openai_types::ChatCompletionParameters {
                                            messages: vec![
                                                crate::openai_types::ChatCompletionRequestMessage::System(crate::openai_types::ChatCompletionRequestSystemMessage {
                                                    content: final_system_prompt,
                                                    name: None,
                                                }),
                                                crate::openai_types::ChatCompletionRequestMessage::User(crate::openai_types::ChatCompletionRequestUserMessage { 
                                                    content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(user_prompt_clone),
                                                    name: None,
                                                })
                                            ],
                                            model: "qwen3".to_string(),
                                            max_tokens: Some(1024),
                                            temperature: Some(current_temperature),
                                            top_p: Some(1.0),
                                            ..Default::default()
                                        };
                                        
                                        gen.clear_kv_cache();
                                        let res = gen.generate(
                                            params, 
                                            Some(cancel_clone), 
                                            None, 
                                            None
                                        ).map_err(|e| anyhow::anyhow!("Qwen 3 Inference failed: {}", e));
                                        
                                        gen.clear_kv_cache();
                                        
                                        res
                                    } else {
                                        Err(anyhow::anyhow!("Qwen 3 Generator is missing"))
                                    }
                                }).await.unwrap_or_else(|e| Err(anyhow::anyhow!("Spawn blocking failed: {}", e)))
                            }?;

                            if !model.is_cpu_mode {
                                let dev = model.device_config.device.clone();
                                let _ = tokio::task::spawn_blocking(move || {
                                    if dev.is_cuda() { let _ = dev.synchronize(); }
                                }).await;
                            }

                            let parsed = crate::parsing::parse_json_from_llm(&res_mask);
                            let mut extracted_val = parsed.get(&target_name).and_then(|v| v.as_str()).unwrap_or("").trim().to_string();

                            // 🌟 [CRITICAL FIX] 요청하신 대로, 배열이 아닌 1:1 매칭된 NMS 단일값을 출력하며 로그 디자인을 명시적으로 반영합니다!
                            emit_term(&format!("[DEBUG-OOM] [{}] 항목 추출 완료 (NMS 👑 [WINNER/EXPANDED]: '{}', 추출단어: '{}') - 응답 길이: {}, 결과: {}", base_target, input_keyword, extracted_val, res_mask.len(), res_mask));

                            // 🌟 [CRITICAL FIX] 아예 빈 값이거나 "..." 형태인 경우 바로 포기하지 않고 최대 3번까지 재시도합니다.
                            if extracted_val.is_empty() || extracted_val == "..." || extracted_val == "null" {
                                miss_counter += 1;
                                current_temperature += 0.2; // 🌟 온도 상승

                                emit_term(&format!("[DEBUG] 빈 값 반환 감지 (재시도 {} - 무한) (현재 온도: {:.2})", miss_counter, current_temperature));
                                
                                if current_temperature >= 1.0 {
                                    emit_term(&format!("[EXTRACTION] 🛑 온도 1.0 도달로 강제 종료."));
                                    break;
                                }

                                // 빈 값 꼼수 방지 및 재시도 유도
                                ignore_list.push("".to_string());
                                ignore_list.push("null".to_string());
                                ignore_list.push("\"\"".to_string());
                                ignore_list.push("\"null\"".to_string());
                                continue;
                            }

                            // 🌟 [STAGE 3] 추출 단어 다국어 순차 검증 (Sequential CoT Evaluation with Early Exit)
                            let is_mismatch = parsed.get("is_target_mismatch").and_then(|v| v.as_bool()).unwrap_or(false);

                            emit_term(&format!("\n======================================="));
                            emit_term(&format!("[DEBUG-EXTRACTION-STAGE3] 🎯 추출 단어 다국어 순차 검증 시작 🎯"));
                            emit_term(&format!("- 추출 단어: '{}'", extracted_val));
                            emit_term(&format!("- is_target_mismatch: {}", is_mismatch));
                            emit_term(&format!("=======================================\n"));

                            let mut is_hallucination = false;
                            
                            if is_mismatch {
                                emit_term(&format!("    💀 [REJECT] 타겟 미스매치 (is_target_mismatch=true)"));
                                is_hallucination = true;
                            } else {
                                // 🌟 [추가] 모든 검증 점수가 정확히 1.0인지 추적하는 상태 변수 선언
                                let mut all_scores_are_one = true;
                                let mut score_checked = false;

                                // 🌟 다국어 순환하며 Early Exit (하나라도 7.0 넘으면 즉시 차단)
                                for lang in &detected_languages_vec {
                                    if cancellation_token.load(Ordering::Relaxed) { break; }

                                    let verb_hint = bias_json.get("verb").and_then(|v| v.get("hints")).and_then(|v| v.get(lang)).and_then(|v| v.as_str()).unwrap_or("Does the word represent an action, state, or predicate (e.g., ends with a verb conjugation)?");
                                    let expr_hint = bias_json.get("expression").and_then(|v| v.get("hints")).and_then(|v| v.get(lang)).and_then(|v| v.as_str()).unwrap_or("Is it an idiom, conversational phrase, or full sentence?");

                                    // --- 1. Verb Score Check ---
                                    let (v_sys, v_user) = crate::parsing::build_single_property_verification_prompt(&extracted_val, lang, "verb, predicate", verb_hint);
                                    let cancel_clone_v = cancellation_token.clone();
                                    
                                    let v_res = if is_large_context {
                                        let gen_arc = model.generator.clone();
                                        let mut gen_guard = gen_arc.lock().await;
                                        if let Some(gen) = gen_guard.as_mut() {
                                            let params = crate::openai_types::ChatCompletionParameters {
                                                messages: vec![
                                                    crate::openai_types::ChatCompletionRequestMessage::System(crate::openai_types::ChatCompletionRequestSystemMessage { content: v_sys, name: None }),
                                                    crate::openai_types::ChatCompletionRequestMessage::User(crate::openai_types::ChatCompletionRequestUserMessage { content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(v_user), name: None })
                                                ],
                                                model: "qwen".to_string(), max_tokens: Some(64), temperature: Some(0.1), top_p: Some(0.1), ..Default::default()
                                            };
                                            gen.clear_kv_cache();
                                            let res = gen.generate(params, Some(cancel_clone_v), None, None, None).await.unwrap_or("{}".to_string());
                                            gen.clear_kv_cache();
                                            res
                                        } else { "{}".to_string() }
                                    } else {
                                        let gen_arc = model.qwen3_generator.clone();
                                        tokio::task::spawn_blocking(move || -> String {
                                            let mut gen_guard = gen_arc.blocking_lock();
                                            if let Some(gen) = gen_guard.as_mut() {
                                                let params = crate::openai_types::ChatCompletionParameters {
                                                    messages: vec![
                                                        crate::openai_types::ChatCompletionRequestMessage::System(crate::openai_types::ChatCompletionRequestSystemMessage { content: v_sys, name: None }),
                                                        crate::openai_types::ChatCompletionRequestMessage::User(crate::openai_types::ChatCompletionRequestUserMessage { content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(v_user), name: None })
                                                    ],
                                                    model: "qwen3".to_string(), max_tokens: Some(64), temperature: Some(0.1), top_p: Some(0.1), ..Default::default()
                                                };
                                                gen.clear_kv_cache();
                                                let res = gen.generate(params, Some(cancel_clone_v), None, None).unwrap_or("{}".to_string());
                                                gen.clear_kv_cache();
                                                res
                                            } else { "{}".to_string() }
                                        }).await.unwrap_or("{}".to_string())
                                    };

                                    if !model.is_cpu_mode {
                                        let dev = model.device_config.device.clone();
                                        let _ = tokio::task::spawn_blocking(move || { if dev.is_cuda() { let _ = dev.synchronize(); } }).await;
                                    }

                                    let v_json = crate::parsing::parse_json_from_llm(&v_res);
                                    let verb_score = v_json.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                                    
                                    emit_term(&format!("    🔍 [VERIFY] {} Verb Score: {}", lang, verb_score));
                                    
                                    score_checked = true;
                                    if verb_score != 1.0 { all_scores_are_one = false; }

                                    if verb_score >= 7.0 {
                                        emit_term(&format!("    💀 [REJECT] {} Verb Score 초과 (Early Exit)", lang));
                                        is_hallucination = true;
                                        break;
                                    }

                                    // --- 2. Expression Score Check ---
                                    let (e_sys, e_user) = crate::parsing::build_single_property_verification_prompt(&extracted_val, lang, "idiom, phrase, full sentence", expr_hint);
                                    let cancel_clone_e = cancellation_token.clone();
                                    
                                    let e_res = if is_large_context {
                                        let gen_arc = model.generator.clone();
                                        let mut gen_guard = gen_arc.lock().await;
                                        if let Some(gen) = gen_guard.as_mut() {
                                            let params = crate::openai_types::ChatCompletionParameters {
                                                messages: vec![
                                                    crate::openai_types::ChatCompletionRequestMessage::System(crate::openai_types::ChatCompletionRequestSystemMessage { content: e_sys, name: None }),
                                                    crate::openai_types::ChatCompletionRequestMessage::User(crate::openai_types::ChatCompletionRequestUserMessage { content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(e_user), name: None })
                                                ],
                                                model: "qwen".to_string(), max_tokens: Some(64), temperature: Some(0.1), top_p: Some(0.1), ..Default::default()
                                            };
                                            gen.clear_kv_cache();
                                            let res = gen.generate(params, Some(cancel_clone_e), None, None, None).await.unwrap_or("{}".to_string());
                                            gen.clear_kv_cache();
                                            res
                                        } else { "{}".to_string() }
                                    } else {
                                        let gen_arc = model.qwen3_generator.clone();
                                        tokio::task::spawn_blocking(move || -> String {
                                            let mut gen_guard = gen_arc.blocking_lock();
                                            if let Some(gen) = gen_guard.as_mut() {
                                                let params = crate::openai_types::ChatCompletionParameters {
                                                    messages: vec![
                                                        crate::openai_types::ChatCompletionRequestMessage::System(crate::openai_types::ChatCompletionRequestSystemMessage { content: e_sys, name: None }),
                                                        crate::openai_types::ChatCompletionRequestMessage::User(crate::openai_types::ChatCompletionRequestUserMessage { content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(e_user), name: None })
                                                    ],
                                                    model: "qwen3".to_string(), max_tokens: Some(64), temperature: Some(0.1), top_p: Some(0.1), ..Default::default()
                                                };
                                                gen.clear_kv_cache();
                                                let res = gen.generate(params, Some(cancel_clone_e), None, None).unwrap_or("{}".to_string());
                                                gen.clear_kv_cache();
                                                res
                                            } else { "{}".to_string() }
                                        }).await.unwrap_or("{}".to_string())
                                    };

                                    if !model.is_cpu_mode {
                                        let dev = model.device_config.device.clone();
                                        let _ = tokio::task::spawn_blocking(move || { if dev.is_cuda() { let _ = dev.synchronize(); } }).await;
                                    }

                                    let e_json = crate::parsing::parse_json_from_llm(&e_res);
                                    let expr_score = e_json.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                                    
                                    emit_term(&format!("    🔍 [VERIFY] {} Expr Score: {}", lang, expr_score));
                                    
                                    if expr_score != 1.0 { all_scores_are_one = false; }

                                    if expr_score >= 7.0 {
                                        emit_term(&format!("    💀 [REJECT] {} Expr Score 초과 (Early Exit)", lang));
                                        is_hallucination = true;
                                        break;
                                    }
                                }

                                // 🌟 [추가] 검증을 거친 모든 언어의 Verb/Expr 점수가 전부 정확히 1.0인 경우 기각
                                if !is_hallucination && score_checked && all_scores_are_one {
                                    emit_term(&format!("    💀 [REJECT] 모든 검증 점수가 1.0으로 평가됨 (조사 및 무의미한 일반명사 파편 의심)"));
                                    is_hallucination = true;
                                }
                            }

                            // 🌟 [CRITICAL FIX] 서술어/표현 점수가 7점을 넘거나 타겟 미스매치 시 즉시 환각으로 간주하고 강제 차단합니다.
                            if is_hallucination {
                                miss_counter += 1;
                                current_temperature += 0.2; // 🌟 환각이므로 온도를 올려 변형 유도
                                
                                let count = value_counts.entry(extracted_val.clone()).or_insert(0);
                                *count += 1;
                                
                                emit_term(&format!("[DEBUG] 검증 탈락 감지됨. 환각으로 간주하여 강제 기각 (재시도 {}): '{}'", miss_counter, extracted_val));
                                
                                if current_temperature >= 1.0 || *count >= 3 {
                                    emit_term(&format!("[EXTRACTION] 🛑 동일한 검증 오류 3회 누적 또는 온도 1.0 도달로 강제 종료."));
                                    break;
                                }

                                if !extracted_val.is_empty() {
                                    ignore_list.push(extracted_val.clone());
                                }
                                continue;
                            }

                            // 🌟 [CRITICAL FIX] Qwen3가 압축된 문맥(matched_context)을 읽고 있으므로, 환각 검사도 동일한 문맥에서 수행해야 완벽합니다.
                            // 🌟 [Subsumption 1단계] 검증 기준을 훼손된 masked_text가 아닌, 순수 원본(target_text)으로 변경하여 거대 덩어리 누락을 방지합니다.
                            let exists_in_context = matched_context.contains(&extracted_val);
                            let exists_in_body = target_text.contains(&extracted_val); // 🌟 masked_text -> target_text 교체
                            let exists_in_title = doc_title.contains(&extracted_val);
                            let exists_in_desc = doc_desc.contains(&extracted_val);

                            // 🌟 [CRITICAL FIX] 추출된 값이 임시 마커(해시 기반)를 포함하고 있다면 무조건 환각으로 간주하고 강제 차단합니다.
                            if extracted_val.contains("[___REDACTED_") {
                                miss_counter += 1;
                                current_temperature += 0.2; // 🌟 온도 상승
                                
                                let count = value_counts.entry(extracted_val.clone()).or_insert(0);
                                *count += 1;
                                
                                emit_term(&format!("[DEBUG] 임시 마커 추출 시도 감지, 강제 차단 (재시도 {} - 무한): '{}' (현재 온도: {:.2}, 동일값: {}회)", miss_counter, extracted_val, current_temperature, count));
                                
                                if current_temperature >= 1.0 || *count >= 3 {
                                    emit_term(&format!("[EXTRACTION] 🛑 동일한 마커 오류 3회 누적 또는 온도 1.0 도달로 강제 종료."));
                                    break;
                                }

                                ignore_list.push(extracted_val.clone());
                                continue;
                            }

                            // 🌟 [기 마스킹 단어 및 파생어 필터링] 이미 찾은 단어와 겹치거나 파생어인 경우 즉시 기각
                            let mut is_derivative = false;
                            for (orig, _) in &replacement_history {
                                // 길이가 2글자 이상인 경우에 한해서 부분 일치 검사 (너무 짧은 단어 오작동 방지)
                                if orig.chars().count() >= 2 && extracted_val.chars().count() >= 2 {
                                    if extracted_val.contains(orig) || orig.contains(&extracted_val) {
                                        // 🌟 [CRITICAL FIX] 본문(masked_text)에 해당 파생어가 마커에 종속되지 않고 독립적으로 살아있다면 허용합니다.
                                        if masked_text.contains(&extracted_val) {
                                            emit_term(&format!("[DEBUG] 파생어 '{}'가 본문에 독립적으로 존재하여 마스킹을 허용합니다. (원본: '{}')", extracted_val, orig));
                                        } else {
                                            is_derivative = true;
                                            break;
                                        }
                                    }
                                }
                            }

                            // Phase 2에서는 마커 업그레이드를 위해 의도적으로 중복 추출을 시도하므로 필터링에서 제외합니다.
                            if is_derivative && !is_phase2 {
                                // 🌟 [CRITICAL FIX] 기 마스킹 단어의 파생어가 반복 추출된다는 것은 해당 트랙이 포화 상태라는 의미입니다.
                                // 무의미하게 온도를 올리며 continue 하지 않고, 즉시 break하여 루프를 조기 종료(Early Exit)합니다.
                                emit_term(&format!("[DEBUG] 기 마스킹 단어의 파생어 반복 추출 감지. 트랙 포화로 판단하여 조기 종료(Early Break): '{}'", extracted_val));
                                break;
                            }

                            // 🌟 [과잉 추출 방지: 길이 및 어절 제한] 
                            // 개체명이 문장 통째로 나오는 것을 방어하기 위해 어절 수 및 글자 수 하드 리미트 적용
                            let word_count = extracted_val.split_whitespace().count();
                            let char_count = extracted_val.chars().count();
                            
                            // 주소(address)는 예외적으로 길 수 있으므로 기준을 다르게 적용
                            let is_address = target_name.contains("address") || target_name.contains("location");
                            let max_words = if is_address { 15 } else { 6 };
                            let max_chars = if is_address { 100 } else { 30 };

                            if word_count > max_words || char_count > max_chars {
                                miss_counter += 1;
                                current_temperature += 0.2;
                                
                                let count = value_counts.entry(extracted_val.clone()).or_insert(0);
                                *count += 1;
                                
                                emit_term(&format!("[DEBUG] 과잉 추출 감지 (어절: {}, 글자수: {}), 강제 차단: '{}'", word_count, char_count, extracted_val));
                                
                                if current_temperature >= 1.0 || *count >= 3 {
                                    emit_term(&format!("[EXTRACTION] 🛑 과잉 추출 오류 3회 누적 또는 온도 1.0 도달로 강제 종료."));
                                    break;
                                }

                                ignore_list.push(extracted_val.clone());
                                continue;
                            }

                            // 🌟 [전략 A & D 적용] 띄어쓰기 증발/변형에 대한 전역 보정 로직 (Space-Agnostic Validation)
                            if !extracted_val.is_empty() && extracted_val != "..." && extracted_val != "null" && !exists_in_context && !exists_in_body && !exists_in_title && !exists_in_desc {
                                if target_name == "contact_number" {
                                    // 연락처 특화 정규식 보정 (기존 로직 유지)
                                    let digits_only: String = extracted_val.chars().filter(|c| c.is_digit(10)).collect();
                                    if digits_only.len() >= 8 {
                                        let regex_pattern = digits_only.chars().map(|c| c.to_string()).collect::<Vec<String>>().join(r"[-.\s]*");
                                        if let Ok(re) = regex::Regex::new(&regex_pattern) {
                                            if let Some(mat) = re.find(&masked_text) {
                                                extracted_val = mat.as_str().to_string();
                                            }
                                        }
                                    }
                                } else {
                                    // 🌟 [일반 텍스트 보정] 공백을 완전히 무시한 정규식을 동적 생성하여 원본 텍스트에 존재하는 형태(띄어쓰기 포함)를 복원합니다.
                                    let no_space_val: String = extracted_val.chars().filter(|c| !c.is_whitespace()).collect();
                                    
                                    // 정규식 엔진의 부하를 막기 위해 글자 수가 2글자 이상 100글자 이하일 때만 수행
                                    if no_space_val.len() >= 2 && no_space_val.len() <= 100 {
                                        // 정규식 특수문자 이스케이프 후 띄어쓰기 허용 패턴 조립 (예: 조\s*세\s*무\s*리\s*뉴)
                                        let escaped_chars: Vec<String> = no_space_val.chars().map(|c| regex::escape(&c.to_string())).collect();
                                        let regex_pattern = escaped_chars.join(r"\s*");
                                        
                                        if let Ok(re) = regex::Regex::new(&regex_pattern) {
                                            // 본문, 제목, 압축 문맥 순으로 탐색하여 원래 형태 복원 시도
                                            if let Some(mat) = re.find(&masked_text).or_else(|| re.find(&matched_context)).or_else(|| re.find(&doc_title)) {
                                                emit_term(&format!("[DEBUG] 띄어쓰기 보정 성공: '{}' -> '{}'", extracted_val, mat.as_str()));
                                                extracted_val = mat.as_str().to_string();
                                            }
                                        }
                                    }
                                }
                            }

                            // 🌟 [CRITICAL FIX] 아예 빈 값이거나 "..." 형태인 경우 바로 포기하지 않고 최대 3번까지 재시도합니다.
                            if extracted_val.is_empty() || extracted_val == "..." || extracted_val == "null" {
                                miss_counter += 1;
                                current_temperature += 0.2; // 🌟 온도 상승
                                // empty_count += 1; // 🌟 빈 값 카운트 증가

                                emit_term(&format!("[DEBUG] 빈 값 반환 감지 (재시도 {} - 무한) (현재 온도: {:.2})", miss_counter, current_temperature));
                                
                                if current_temperature >= 1.0 {
                                    emit_term(&format!("[EXTRACTION] 🛑 온도 1.0 도달로 강제 종료."));
                                    break;
                                }

                                // 빈 값 꼼수 방지 및 재시도 유도
                                ignore_list.push("".to_string());
                                ignore_list.push("null".to_string());
                                ignore_list.push("\"\"".to_string());
                                ignore_list.push("\"null\"".to_string());
                                continue;
                            }

                            // 🌟 [환각 방지 3번 재시도 루프 (유지됨)]
                            // 값이 추출되긴 했으나 LLM이 실제로 읽은 압축 문맥(matched_context)이나 원본에 없는 환각(Hallucination)인 경우, 
                            // 무시 리스트(ignore_list)에 넣고 다시 프롬프트를 생성해 최대 3번까지 재시도(continue)합니다.
                            let re_check_context = matched_context.contains(&extracted_val);
                            let re_check_body = masked_text.contains(&extracted_val);
                            let re_check_title = doc_title.contains(&extracted_val);
                            let re_check_desc = doc_desc.contains(&extracted_val);

                            if !re_check_context && !re_check_body && !re_check_title && !re_check_desc {
                                miss_counter += 1;
                                current_temperature += 0.2; // 🌟 못 찾았으므로 온도를 높여 다음 턴에 변형을 유도
                                
                                let count = value_counts.entry(extracted_val.clone()).or_insert(0);
                                *count += 1;

                                // if current_temperature > 1.0 || *count >= 3 {
                                if current_temperature > 1.0 {
                                    emit_term(&format!("[EXTRACTION] 🛑 동일한 오탐지 값({}회) 누적 또는 온도 {:.2} 도달로 강제 종료.", count, current_temperature));
                                    break;
                                }
                                
                                // 🌟 [조건부 부분 치환 (Vector Bouncer) 적용 - 양방향 점진적 수축 로테이션]
                                // 완벽한 일치가 아니더라도 단어 단위로 쪼개서 본문에 존재하는지 확인하되, 
                                // 무조건 치환하지 않고 NMS 배틀(Pass 2)과 완벽히 동일한 스코어링 공식으로 검증하여 통과한 단어만 마스킹합니다.
                                let parts: Vec<&str> = extracted_val.split_whitespace().collect();
                                let mut partial_masked = false;

                                // 🌟 [CRITICAL FIX] 쪼개진 단어를 검증하기 위해 현재 타겟의 Bias/Prejudice 벡터를 준비합니다.
                                let lang_prefix = target_name.split('_').next().unwrap_or("english");
                                let prefixed_b_val = target_bias.split(',').map(|s| format!("{} {}", lang_prefix, s.trim())).collect::<Vec<_>>().join(", ");
                                let prefixed_p_val = target_prejudice.split(',').map(|s| format!("{} {}", lang_prefix, s.trim())).collect::<Vec<_>>().join(", ");
                                
                                let bias_emb = model.get_embedding(prefixed_b_val).await.unwrap_or_else(|_| vec![0.0; 384]);
                                let prej_emb = model.get_embedding(prefixed_p_val).await.unwrap_or_else(|_| vec![0.0; 384]);
                                
                                // 🌟 [양방향 점진적 수축 로테이션] 모든 가능한 어절 조합을 생성하고 각각의 점수를 매깁니다.
                                // 튜플 구조: (start_idx, end_idx, chunk_text, final_score, verb_penalty)
                                let mut sub_chunks: Vec<(usize, usize, String, f32, f32)> = Vec::new();

                                for start in 0..parts.len() {
                                    for end in (start + 1)..=parts.len() {
                                        let chunk_text = parts[start..end].join(" ");
                                        // 너무 짧은 조사/어미가 치환되는 것을 막기 위해 2글자 이상만 허용
                                        if chunk_text.chars().count() >= 2 && (matched_context.contains(&chunk_text) || masked_text.contains(&chunk_text) || doc_title.contains(&chunk_text) || doc_desc.contains(&chunk_text)) {
                                            
                                            // 🌟 부분 단어의 임베딩을 추출하고 Pass 2와 동일한 공식으로 심사합니다.
                                            let p_emb = model.get_embedding(chunk_text.clone()).await.unwrap_or_else(|_| vec![0.0; 384]);
                                            
                                            let b_score = cosine_similarity(&p_emb, &bias_emb);
                                            let p_score = cosine_similarity(&p_emb, &prej_emb);
                                            let v_sim = cosine_similarity(&p_emb, &verb_emb);
                                            
                                            let word_count = end - start;
                                            let length_weight = 1.0; // 단어 개수 가중치 제거 (길이 무관 동등 점수)
                                            let beta = if word_count <= 2 { 0.05 } else { 0.10 };
                                            let verb_penalty = v_sim * beta;
                                            let penalty_weight = if word_count <= 2 { 0.3 } else { 0.7 };
                                            
                                            let base_score = b_score - (p_score * penalty_weight) - verb_penalty;
                                            
                                            // Pass 2와 동일한 커트라인(0.25) 검증
                                            if base_score > 0.25 {
                                                sub_chunks.push((start, end, chunk_text, base_score * length_weight, verb_penalty));
                                            } else {
                                                emit_term(&format!("    💀 [DEFEAT] 수축 로테이션 탈락: '{}' (Score: {:.4} / VerbPenalty: {:.4})", chunk_text, base_score, verb_penalty));
                                            }
                                        }
                                    }
                                }

                                // 🌟 [NMS BATTLE] 추출된 서브 청크 중 점수가 가장 높은 것을 살리고 겹치는 하위 조각들을 흡수(기각)합니다.
                                sub_chunks.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
                                let mut final_sub_chunks: Vec<(usize, usize, String, f32, f32)> = Vec::new();

                                for cand in sub_chunks {
                                    let mut is_overlapped = false;
                                    for selected in &final_sub_chunks {
                                        // start < selected.end && end > selected.start
                                        if cand.0 < selected.1 && cand.1 > selected.0 {
                                            is_overlapped = true;
                                            break;
                                        }
                                    }
                                    if !is_overlapped {
                                        final_sub_chunks.push(cand);
                                    } else {
                                        emit_term(&format!("    💀 [DEFEAT] NMS 오버랩 흡수: '{}' (Score: {:.4})", cand.2, cand.3));
                                    }
                                }

                                // 🌟 [WINNER 마스킹 반영] 살아남은 조각들을 최종 마스킹 처리합니다.
                                for (_, _, text_val, score, _) in final_sub_chunks {
                                    emit_term(&format!("    👑 [WINNER] 부분 일치(수축 로테이션) 통과: '{}' 중 '{}' (Score: {:.4}) -> 강제 마스킹", extracted_val, text_val, score));
                                    
                                    let mnemonic = crate::parsing::generate_mnemonic();
                                    let upper_key = base_target.to_uppercase(); 
                                    let final_replacement = format!("[{}: {}]", upper_key, mnemonic);
                                    let skip_marker = format!("[___REDACTED_{}___]", skip_counter);
                                    
                                    masked_text = masked_text.replace(&text_val, &skip_marker);
                                    doc_title = doc_title.replace(&text_val, &skip_marker);
                                    doc_desc = doc_desc.replace(&text_val, &skip_marker);
                                    matched_context = matched_context.replace(&text_val, &skip_marker);
                                    
                                    skip_map.insert(skip_marker.clone(), final_replacement.clone());
                                    replacement_history.push((text_val.clone(), skip_marker.clone()));
                                    
                                    current_target_found.push(text_val.clone());
                                    domain_history.push((target_name.to_string(), text_val.clone()));
                                    
                                    if base_target == "company" {
                                        phase2_companies.push(text_val.clone());
                                    }

                                    skip_counter += 1;
                                    
                                    all_matches.push(json!({
                                        "name": upper_key,
                                        "value": text_val,
                                        "mnemonic": mnemonic
                                    }));
                                    partial_masked = true;
                                }

                                if partial_masked {
                                    ignore_list.push(extracted_val.clone());
                                    ignore_list.push(format!(" {}", extracted_val));
                                    ignore_list.push(format!("\"{}", extracted_val));
                                    
                                    miss_counter = 0;
                                    current_temperature = 0.0;
                                    item_extract_count += 1;
                                    
                                    fully_masked_candidates.insert(specific_candidate.clone()); // 🌟 완료된 후보 장부 등록

                                    // 🌟 [CRITICAL FIX] 마스킹 성공 시 무한 루프를 방지하고 다음 트랙으로 넘어가기 위해 break를 호출합니다.
                                    emit_term(&format!("[EXTRACTION] 🎯 마스킹 성공 (부분 치환). 해당 트랙 종료 후 다음 트랙으로 이동합니다."));
                                    break;
                                }

                                // 🌟 [전략 D 적용] 무작정 ignore_list에 넣기 전에 로깅하여 어떤 값이 날아갔는지 파악합니다.
                                emit_term(&format!("[DEBUG] 완전 환각/오탐지 감지 (재시도 {} - 무한): '{}' -> 온도 {:.2}로 상승", miss_counter, extracted_val, current_temperature));
                                
                                // 🌟 [CRITICAL FIX] Tokenizer 꼼수를 부리지 못하도록 공백을 포함한 변형도 함께 억제 리스트에 넣습니다.
                                // JSON 생성 환경에서 BPE 토크나이저가 쌍따옴표(")와 결합하여 완전히 다른 단일 토큰으로 
                                // 파편화시키는 현상(Logit 제어 누수)을 완벽하게 방어합니다.
                                ignore_list.push(extracted_val.clone());
                                ignore_list.push(format!(" {}", extracted_val));
                                ignore_list.push(format!("\"{}", extracted_val));
                                ignore_list.push(format!(" \"{}", extracted_val));
                                ignore_list.push(extracted_val.to_lowercase());
                                ignore_list.push(extracted_val.to_uppercase());
                                continue;
                            }

                            // 정상 추출되었으므로 연속 실패 카운터를 리셋합니다.
                            miss_counter = 0;
                            current_temperature = 0.0; // 🌟 성공했으므로 온도를 다시 0으로 차갑게 초기화
                            // 🌟 성공적으로 추출했으므로 추출 횟수 카운터를 증가시킵니다. (무한 루프 방지용)
                            item_extract_count += 1; 

                            // 🌟 [Phase 2 오버랩 방어] 앞단에 이미 치환된 마커로 동일하게 덮어쓰기!
                            if is_phase2 {
                                let mut found_marker = None;
                                for (orig, marker) in &replacement_history {
                                    if orig == &extracted_val {
                                        found_marker = Some(marker.clone());
                                        break;
                                    }
                                }
                                
                                if let Some(marker) = found_marker {
                                    emit_term(&format!("[EXTRACTION] 🔄 Phase 2 Overlap: '{}' is already masked. Upgrading marker...", extracted_val));
                                    
                                    let mnemonic = crate::parsing::generate_mnemonic();
                                    let upper_key = base_target.to_uppercase(); // ex: "레알 마드리드_KOREAN_NAME"
                                    let final_replacement = format!("[{}: {}]", upper_key, mnemonic);
                                    
                                    // skip_map 의 최종 치환 문자열을 Phase 2 타겟으로 덮어씁니다!
                                    skip_map.insert(marker.clone(), final_replacement.clone());
                                    
                                    // all_matches 업데이트
                                    for match_val in all_matches.iter_mut() {
                                        if let Some(obj) = match_val.as_object_mut() {
                                            if obj.get("value").and_then(|v| v.as_str()) == Some(extracted_val.as_str()) {
                                                obj.insert("name".to_string(), json!(upper_key.clone()));
                                                obj.insert("mnemonic".to_string(), json!(mnemonic.clone()));
                                            }
                                        }
                                    }
                                    
                                    current_target_found.push(extracted_val.clone());
                                    domain_history.push((target_name.to_string(), extracted_val.clone()));
                                    
                                    fully_masked_candidates.insert(specific_candidate.clone()); // 🌟 완료된 후보 장부 등록

                                    // 🌟 [CRITICAL FIX] 성공적으로 덮어썼으므로 루프를 종료하고 다음 트랙으로 넘어갑니다.
                                    emit_term(&format!("[EXTRACTION] 🎯 마스킹 성공 (Phase 2 병합). 해당 트랙 종료 후 다음 트랙으로 이동합니다."));
                                    break;
                                }
                            }

                            // 🌟 마스킹 니모닉 생성 및 즉시 치환 대신 해시 기반 마커로 임시 치환
                            let mnemonic = crate::parsing::generate_mnemonic();
                            let upper_key = base_target.to_uppercase(); 
                            
                            let final_replacement = format!("[{}: {}]", upper_key, mnemonic);
                            let skip_marker = format!("[___REDACTED_{}___]", skip_counter);
                            
                            // 🌟 [Subsumption 2단계] 추출된 거대 덩어리 내부에 이미 치환된 소형 마커가 존재하는지 확인하고, 
                            // 현재 masked_text에 반영된 형태(Hybrid)를 역산하여 덮어쓸 준비를 합니다.
                            let mut hybrid_val = extracted_val.clone();
                            let mut subsumed_markers = std::collections::HashSet::new();
                            for (orig, marker) in &replacement_history {
                                if extracted_val.contains(orig) && orig.chars().count() >= 2 {
                                    hybrid_val = hybrid_val.replace(orig, marker);
                                    subsumed_markers.insert(marker.clone());
                                }
                            }

                            // 🌟 [Subsumption 3단계] 흡수된 구형 소형 마커들을 장부(all_matches)에서 제거하여 새로운 범주(예: NAME)로 완벽히 세대교체합니다.
                            if !subsumed_markers.is_empty() {
                                emit_term(&format!("    🔄 [SUBSUMPTION] 거대 덩어리 발견! 기존 마커 {}개를 '{}' ({}) 속성으로 흡수 병합합니다.", subsumed_markers.len(), extracted_val, upper_key));
                                all_matches.retain(|match_val| {
                                    if let Some(obj) = match_val.as_object() {
                                        if let Some(m_val) = obj.get("value").and_then(|v| v.as_str()) {
                                            // 만약 all_matches에 들어있는 원본 값이 현재 거대 덩어리 안에 포함된다면 파기
                                            if extracted_val.contains(m_val) && m_val != extracted_val && m_val.chars().count() >= 2 {
                                                return false;
                                            }
                                        }
                                    }
                                    true
                                });
                            }
                            
                            // 🌟 [CRITICAL FIX] 원본 텍스트(본문+제목) 모두에서 임시 마커로 치환하여 이어지는 LLM 추론에서 혼선을 원천 방지합니다.
                            // 역산된 hybrid_val을 사용하여 조각난 마커들까지 통째로 하나의 거대 마커로 덮어씌웁니다.
                            masked_text = masked_text.replace(&hybrid_val, &skip_marker);
                            doc_title = doc_title.replace(&hybrid_val, &skip_marker);
                            doc_desc = doc_desc.replace(&hybrid_val, &skip_marker);
                            matched_context = matched_context.replace(&hybrid_val, &skip_marker); 
                            
                            skip_map.insert(skip_marker.clone(), final_replacement);
                            replacement_history.push((extracted_val.clone(), skip_marker.clone())); 
                            
                            current_target_found.push(extracted_val.clone());
                            domain_history.push((target_name.to_string(), extracted_val.clone()));

                            // 🌟 [파생어 자동 마스킹 적용] 띄어쓰기로 이루어진 복합어인 경우, 쪼개진 단어(파생어)들도 모두 동일한 마커로 강제 마스킹합니다.
                            let parts: Vec<&str> = extracted_val.split_whitespace().collect();
                            if parts.len() > 1 {
                                for p in parts {
                                    if p.chars().count() >= 2 {
                                        let mut hybrid_val_p = p.to_string();
                                        for (orig, marker) in &replacement_history {
                                            if p.contains(orig) && orig.chars().count() >= 2 {
                                                hybrid_val_p = hybrid_val_p.replace(orig, marker);
                                            }
                                        }
                                        masked_text = masked_text.replace(&hybrid_val_p, &skip_marker);
                                        doc_title = doc_title.replace(&hybrid_val_p, &skip_marker);
                                        doc_desc = doc_desc.replace(&hybrid_val_p, &skip_marker);
                                        matched_context = matched_context.replace(&hybrid_val_p, &skip_marker);
                                        
                                        replacement_history.push((p.to_string(), skip_marker.clone()));
                                        current_target_found.push(p.to_string());
                                        domain_history.push((target_name.to_string(), p.to_string()));
                                    }
                                }
                            }

                            if base_target == "company" {
                                phase2_companies.push(extracted_val.clone());
                            }

                            skip_counter += 1;

                            // 최종 저장용 JSON 객체를 all_matches 배열에 기록
                            all_matches.push(json!({
                                "name": upper_key, 
                                "value": extracted_val,
                                "mnemonic": mnemonic
                            }));
                            
                            fully_masked_candidates.insert(specific_candidate.clone()); // 🌟 완료된 후보 장부 등록

                            // 🌟 [CRITICAL FIX] 정상적으로 마스킹을 성공했으므로 무한 루프(continue)를 돌지 않고 즉시 탈출(break)하여 다음 트랙으로 이동합니다.
                            emit_term(&format!("[EXTRACTION] 🎯 마스킹 성공 (전체 치환). 해당 트랙 종료 후 다음 트랙으로 이동합니다."));
                            break;
                        }
                    }

                    // 🌟 [추가] 모든 추론이 끝난 후 임시 해시 마커를 실제 니모닉으로 일괄 변환합니다.
                    for i in 0..skip_counter {
                        let marker = format!("[___REDACTED_{}___]", i);
                        if let Some(final_repl) = skip_map.get(&marker) {
                            masked_text = masked_text.replace(&marker, final_repl);
                            doc_title = doc_title.replace(&marker, final_repl);
                            doc_desc = doc_desc.replace(&marker, final_repl);
                        }
                    }

                    // 🌟 [CRITICAL FIX] 마스킹이 모두 끝난 후, 외부 장부에 보관했던 메타데이터 노이즈, 링크를 정확한 포지션에 복원합니다.
                    // 마커 양옆에 주입했던 공백도 깔끔하게 제거하여 원본의 접착 상태(Glued)를 완벽히 복구합니다.
                    if let Ok(noise_re) = regex::Regex::new(r"\s*(\[___REDACTED_NOISE_\d+___\])\s*") {
                        masked_text = noise_re.replace_all(&masked_text, |caps: &regex::Captures| { noise_map.get(&caps[1]).cloned().unwrap_or_default() }).to_string();
                        doc_title = noise_re.replace_all(&doc_title, |caps: &regex::Captures| { noise_map.get(&caps[1]).cloned().unwrap_or_default() }).to_string();
                        doc_desc = noise_re.replace_all(&doc_desc, |caps: &regex::Captures| { noise_map.get(&caps[1]).cloned().unwrap_or_default() }).to_string();
                    }

                    if let Ok(link_re) = regex::Regex::new(r"\s*(\[___REDACTED_LINK_\d+___\])\s*") {
                        masked_text = link_re.replace_all(&masked_text, |caps: &regex::Captures| { link_map.get(&caps[1]).cloned().unwrap_or_default() }).to_string();
                        doc_title = link_re.replace_all(&doc_title, |caps: &regex::Captures| { link_map.get(&caps[1]).cloned().unwrap_or_default() }).to_string();
                        doc_desc = link_re.replace_all(&doc_desc, |caps: &regex::Captures| { link_map.get(&caps[1]).cloned().unwrap_or_default() }).to_string();
                    }

                    if !all_matches.is_empty() {
                        // 🌟 마스킹된 전체 텍스트도 masked 오브젝트 내부의 text 필드로 함께 캡슐화합니다.
                        extracted_json = json!({ "matches": all_matches, "text": masked_text, "title": doc_title.clone(), "description": doc_desc.clone() });
                    }
                }

                // 🌟 [STEP 3] 최종 결과물(마스킹 정보)을 DB에 업데이트합니다.
                if !extracted_json.is_null() && !extracted_json.as_object().map(|o| o.is_empty()).unwrap_or(true) {
                    if let Some(obj) = json_data.as_object_mut() {
                        obj.insert("masked".to_string(), extracted_json);
                        obj.insert("is_masked".to_string(), json!(true));
                        // 🌟 루트에 존재하던 masked_text 개별 삽입 로직은 삭제되었습니다.
                        
                        // 🌟 [추가] data 객체 내부에 masked_title을 명시적으로 반영합니다.
                        if let Some(data_obj) = obj.get_mut("data").and_then(|v| v.as_object_mut()) {
                            data_obj.insert("masked_title".to_string(), json!(doc_title.clone()));
                            data_obj.insert("masked_description".to_string(), json!(doc_desc.clone()));
                        } else {
                            obj.insert("data".to_string(), json!({ "masked_title": doc_title.clone(), "masked_description": doc_desc.clone() }));
                        }
                    }

                    // 🌟 [CRITICAL FIX] 하드코딩된 "items" 대신 문서를 찾아낸 실제 테이블(found_table)을 사용합니다!
                    let _ = store.upsert_item(
                        found_table, &doc.id, &doc.r#type, json_data, Some(doc.vector.clone()),
                        Some(&doc.from), Some(&doc.to), Some(&doc.cc), Some(&doc.bcc), Some(&doc.r#ref), Some(&doc.digest)
                    ).await;
                }
            }
        }

        let summary_msg = "Extraction & Masking complete. Refreshing list...".to_string();

        // 🌟 [CRITICAL FIX] 상태(1)가 UI에 덮어씌워지는 것을 방어하기 위해 Done 이벤트 발송 직전에 DB도 9로 굳힙니다.
        {
            let store_guard = store_mutex.lock().await;
            if let Some(db) = store_guard.as_ref() {
                let _ = db.update_task_status(&task.id, 9).await;
                let _ = db.update_message_status(&task.id, 9, Some(&summary_msg)).await;
            }
        }

        let payload = json!({
            "task_id": task.id,
            "category": "Done",
            "summary": summary_msg,
            "spinner": "✅",
            "data": null
        });
        
        let _ = app_handle.emit("extraction-progress", &payload);
        crate::scheduler::log_task_progress(app_handle, &task.id, &payload);
        return Ok(());
    }

    let mut url = task_data.get("href")
        .or_else(|| task_data.get("link"))
        .and_then(|s| s.as_str())
        .unwrap_or("")
        .to_string();

    let mut origin_candidate = task_data.get("origin")
        .or_else(|| task_data.get("domain"))
        .and_then(|s| s.as_str())
        .unwrap_or("")
        .to_string();

    
    // 브라우저 자동화 모듈이 감지한 '진짜 현재 활성화 탭 URL'을 강제로 끌어와서 완벽한 절대 주소로 병합(Join)합니다!
    {
        let state = crate::automation::LAST_DETECTED_STATE.lock().await;
        let active_tab_url = state.url.clone();
        
        if !active_tab_url.is_empty() {
            if let Ok(active_parsed) = url::Url::parse(&active_tab_url) {
                let active_origin = format!("{}://{}", active_parsed.scheme(), active_parsed.host_str().unwrap_or("localhost"));
                
                if origin_candidate.is_empty() || origin_candidate.contains("localhost") {
                    origin_candidate = active_origin;
                }
                
                if url.is_empty() {
                    url = active_tab_url;
                } else if !url.starts_with("http") {
                    
                    if let Ok(joined) = active_parsed.join(&url) {
                        url = joined.to_string();
                    }
                }
            }
        }
    }

    
    if !url.starts_with("http") && !origin_candidate.is_empty() && !origin_candidate.contains("localhost") {
        let scheme = if origin_candidate.starts_with("http") { "" } else { "http://" };
        let base_str = format!("{}{}", scheme, origin_candidate);
        if let Ok(base) = url::Url::parse(&base_str) {
            if let Ok(joined) = base.join(&url) {
                url = joined.to_string();
            }
        }
    }
    
    
    let active_task_json = json!({
        "id": task.id.clone(),
        "type": task.r#type.clone(),
        "link": url.clone(),
        "origin": origin_candidate.clone(),
        "ref": task.r#ref.clone(),
        "status": 1, 
        "created_at": task.created_at,
        "updated_at": chrono::Utc::now().timestamp_millis()
    });
    
    if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() {
        *w = Some(active_task_json.clone());
    }

    
    if url.is_empty() { 
        return Err(anyhow::anyhow!("Task missing target URL or unsupported type for background extraction.")); 
    }

    // [MEMORY] Fetch and process directly in memory
    let raw_html_content = if let Some(raw_html) = task_data.get("html").and_then(|s| s.as_str()) {
        let content = raw_html.to_string();
        if let Some(obj) = task_data.as_object_mut() { obj.remove("html"); }
        content
    } else if !url.is_empty() {
        let response = reqwest::get(&url).await?;
        let bytes = response.bytes().await?;
        
        // [ENCODING-FIX] UTF-8 First Strategy
        let (decoded_utf8, _, malformed_utf8) = encoding_rs::UTF_8.decode(&bytes);
        let utf8_str = decoded_utf8.as_ref();
        
        // Check for explicit EUC-KR/CP949 markers in the UTF-8 decoded string
        let needs_euc = utf8_str.to_lowercase().contains("charset=euc-kr") || 
                        utf8_str.to_lowercase().contains("charset=\"euc-kr\"") ||
                        utf8_str.to_lowercase().contains("charset=cp949") ||
                        utf8_str.to_lowercase().contains("charset=ks_c_5601");

        if needs_euc && malformed_utf8 {
            // Only use EUC-KR if it's explicitly requested AND UTF-8 decoding had issues
            let (decoded_euc, _, _) = encoding_rs::EUC_KR.decode(&bytes);
            decoded_euc.into_owned()
        } else {
            // Default to UTF-8 (Lossy fallback if needed)
            utf8_str.to_string()
        }
    } else {
        return Ok(());
    };

    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    let clean_html_content = parsing::pre_clean_html(&raw_html_content);
    
    let raw_pug = parsing::convert_to_clean_pug(&clean_html_content, PugMode::NoAttributesMode, Some(&url));

    // 🌟 [추가된 로직] 무거운 LLM을 VRAM에 올리기 전에 draft 타입이면 html과 yaml(PUG)을 저장하고 즉시 종료합니다.
    if task.r#type == "draft" {
        emit_term("[PROCESS] Task type is 'draft'. Bypassing LLM inference and staging raw data.");
        
        // 🌟 [OG 데이터 추출] scraper를 이용하여 HTML에서 og:title, og:description (또는 일반 title)을 추출합니다.
        // 스코프(블록)를 강제하여 `scraper::Html` 객체가 await 이전에 메모리에서 해제(Drop)되도록 격리합니다.
        let (extracted_title, extracted_desc) = {
            let document = scraper::Html::parse_document(&raw_html_content);
            
            let og_title_sel = scraper::Selector::parse("meta[property='og:title']").unwrap();
            let title_sel = scraper::Selector::parse("title").unwrap();
            let og_desc_sel = scraper::Selector::parse("meta[property='og:description']").unwrap();
            let desc_sel = scraper::Selector::parse("meta[name='description']").unwrap();

            let title = document.select(&og_title_sel).next()
                .and_then(|el| el.value().attr("content"))
                .map(|s| s.to_string())
                .unwrap_or_else(|| {
                    document.select(&title_sel).next()
                        .map(|el| el.text().collect::<String>())
                        .unwrap_or_default()
                });

            let desc = document.select(&og_desc_sel).next()
                .and_then(|el| el.value().attr("content"))
                .map(|s| s.to_string())
                .unwrap_or_else(|| {
                    document.select(&desc_sel).next()
                        .and_then(|el| el.value().attr("content"))
                        .map(|s| s.to_string())
                        .unwrap_or_default()
                });
                
            (title, desc)
        }; // <-- document 객체가 여기서 파기되므로, 이후의 await 지점을 안전하게 통과할 수 있습니다.

        let store_guard = store_mutex.lock().await;
        if let Some(db) = store_guard.as_ref() {
            // 🌟 [추가] 기존 아이템 여부 확인 및 속성 보존 처리
            let existing_doc = db.get_item_by_id("items", &task.id).await.unwrap_or(None);
            let is_new = existing_doc.is_none();

            let mut draft_data = json!({
                "id": task.id.clone(),
                "type": "draft",
                "link": url.clone(),
                "html": clean_html_content,
                "yaml": raw_pug,
                "title": extracted_title.clone(), 
                "description": extracted_desc.clone(), 
                "text": "Staged HTML and YAML content", 
                "updated_at": chrono::Utc::now().timestamp_millis(),
                "mode": search_mode.clone() // 🌟 [CRITICAL FIX] 동일하게 웹페이지 추출 시에도 mode를 누락 없이 저장합니다.
            });

            if let Some(doc) = existing_doc {
                if let Ok(parsed) = serde_json::from_str::<Value>(&doc.json_data) {
                    if let Some(obj) = draft_data.as_object_mut() {
                        // 이전에 마스킹된 데이터 및 커스텀 타이틀(data.title), 생성일(created_at) 보존
                        if let Some(masked) = parsed.get("masked") { obj.insert("masked".to_string(), masked.clone()); }
                        if let Some(is_masked) = parsed.get("is_masked") { obj.insert("is_masked".to_string(), is_masked.clone()); }
                        if let Some(masked_text) = parsed.get("masked_text") { obj.insert("masked_text".to_string(), masked_text.clone()); }
                        if let Some(data) = parsed.get("data") { obj.insert("data".to_string(), data.clone()); }
                        if let Some(created_at) = parsed.get("created_at") { obj.insert("created_at".to_string(), created_at.clone()); }
                    }
                }
            } else {
                if let Some(obj) = draft_data.as_object_mut() {
                    obj.insert("masked_text".to_string(), json!(""));
                    obj.insert("created_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
                }
            }

            let _ = db.upsert_item(
                "items", &task.id, "draft", draft_data, None,
                Some(&task.from), Some(&team_id), Some(&task.cc), Some(&task.bcc), Some(&task.r#ref), None
            ).await;

            // 🌟 URL이 존재하고 새 아이템일 경우 Pages 테이블의 count 증가
            if is_new && url.starts_with("http") {
                if let Ok(parsed_url) = url::Url::parse(&url) {
                    let hostname = parsed_url.host_str().unwrap_or("").to_string();
                    let pathname = parsed_url.path().to_string();
                    let cc_val = task.cc.clone();
                    
                    // 🌟 [CRITICAL FIX] 중복 저장 및 카운트 관리 기준을 '선택된 카테고리(모드)' 포함으로 변경합니다.
                    // 기존: hostname + pathname -> 변경: search_mode + hostname + pathname
                    let page_id = crate::utils::hash::hash_id(&format!("page_{}_{}_{}", search_mode, hostname, pathname));
                    
                    let mut page_count = 1;
                    let mut existing_page_data = json!({
                        "id": page_id.clone(),
                        "type": "pages",
                        "mode": search_mode.clone(), // 🌟 카테고리(모드) 정보 저장
                        "hostname": hostname.clone(),
                        "pathname": pathname.clone(),
                        "cc": cc_val.clone(),
                        "count": 1
                    });

                    if let Ok(Some(existing_page)) = db.get_item_by_id("pages", &page_id).await {
                        if let Ok(mut parsed) = serde_json::from_str::<Value>(&existing_page.json_data) {
                            page_count = parsed.get("count").and_then(|v| v.as_i64()).unwrap_or(0) + 1;
                            parsed.as_object_mut().unwrap().insert("count".to_string(), json!(page_count));
                            existing_page_data = parsed;
                        }
                    }

                    let _ = db.upsert_item(
                        "pages", &page_id, "pages", existing_page_data, None,
                        Some(&task.from), Some(&team_id), Some(&cc_val), Some(&task.bcc), Some(&task.r#ref), None
                    ).await;
                }
            }
        }

        // 🌟 [CRITICAL FIX] 첫 번째 Mutex 락(store_guard)을 명시적으로 해제하여, 아래의 두 번째 락에서 데드락(Deadlock)이 발생하는 것을 원천 차단합니다!
        drop(store_guard);

        // 🌟 [채팅 말풍선 텍스트 반영] 추출된 제목과 설명이 있다면 이를 바탕으로 요약 텍스트를 구성합니다.
        let display_summary = if extracted_title.is_empty() {
            "Staged HTML and YAML content".to_string()
        } else if extracted_desc.is_empty() {
            extracted_title.clone()
        } else {
            format!("{} - {}", extracted_title, extracted_desc)
        };

        // 🌟 [CRITICAL FIX] 상태(1)가 UI에 덮어씌워지는 것을 방어하기 위해 Done 이벤트 발송 직전에 DB도 9로 굳힙니다.
        {
            let store_guard = store_mutex.lock().await;
            if let Some(db) = store_guard.as_ref() {
                let _ = db.update_task_status(&task.id, 9).await;
                let _ = db.update_message_status(&task.id, 9, Some(&display_summary)).await;
            }
        }

        let payload = json!({
            "task_id": task.id, 
            "category": "Done", 
            "summary": display_summary, 
            "spinner": "✅",
            "data": null
        });
        
        let _ = app_handle.emit("extraction-progress", &payload);
        crate::scheduler::log_task_progress(app_handle, &task.id, &payload);
        
        emit_term("[PROCESS] Web page staged successfully as draft.");
        return Ok(());
    }

    Ok(())
}

pub fn log_task_progress(app: &tauri::AppHandle, task_id: &str, payload: &serde_json::Value) {
    use std::io::Write;
    use tauri::Emitter;

    
    let mut final_payload = payload.clone();
    if let Some(obj) = final_payload.as_object_mut() {
        obj.insert("task_id".to_string(), serde_json::json!(task_id));
    }

    let log_path = crate::utils::paths::get_task_log_file(Some(app), task_id);
    
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path) 
    {
        let line = format!("{}\n", final_payload.to_string());
        let _ = file.write_all(line.as_bytes());
    }

    if let Some(cat) = final_payload.get("category").and_then(|v| v.as_str()) {
        if let Ok(mut w) = crate::CURRENT_UI_CATEGORY.write() {
            *w = cat.to_string();
        }
    }
    if let Ok(mut w) = crate::LATEST_PROGRESS_PAYLOAD.write() {
        *w = Some(final_payload.clone());
    }

    let _ = app.emit("extraction-progress", &final_payload);
}

async fn wait_for_resources_settled(target_vram_mb: u64, target_ram_mb: u64, cancellation_token: Option<&Arc<AtomicBool>>) -> Result<()> {
    use nvml_wrapper::Nvml;
    use sysinfo::System;
    
    let mut sys = System::new_all();
    let nvml = Nvml::init().ok();
    
    let target_vram_bytes = target_vram_mb * 1024 * 1024;
    let target_ram_bytes = target_ram_mb * 1024 * 1024;

    let mut last_vram = 0;
    let mut stable_ticks = 0;
    let mut last_report = std::time::Instant::now();
    let start_time = std::time::Instant::now();

    println!("[RESOURCE-WATCH] Monitoring recovery (Target VRAM > {}MB)...", target_vram_mb);

    loop {
        if let Some(token) = cancellation_token {
            if token.load(Ordering::Relaxed) {
                return Err(anyhow::anyhow!("Task cancelled during resource wait"));
            }
        }

        sys.refresh_memory(); 
        let current_ram = sys.available_memory();
        let mut current_vram = 0;
        let mut has_gpu = false;

        if let Some(ref nvml_inst) = nvml {
            if let Ok(count) = nvml_inst.device_count() {
                for i in 0..count {
                    if let Ok(dev) = nvml_inst.device_by_index(i) {
                        if let Ok(mem) = dev.memory_info() {
                            if mem.free > current_vram { current_vram = mem.free; }
                            has_gpu = true;
                        }
                    }
                }
            }
        }

        let meets_vram = !has_gpu || current_vram >= target_vram_bytes;
        let meets_ram = current_ram >= target_ram_bytes;
        
        if meets_vram && meets_ram {
            break; // Perfect state reached
        }

        // [STABILITY-LOGIC] Even if below target, if memory release has stopped changing,
        // it means we've recovered all we can. Don't wait forever.
        let delta = if current_vram > last_vram { current_vram - last_vram } else { last_vram - current_vram };
        if delta < 10_000_000 { // Change < 10MB (more lenient)
            stable_ticks += 1;
        } else {
            stable_ticks = 0;
        }

        // [FAST-EXIT] If stable for 1.5 seconds OR we have at least 600MB free (enough for Embedding/0.6B)
        // This prevents being stuck at 0.7GB when target is 1.1GB.
        if (stable_ticks >= 3 && current_vram > 600_000_000) || current_vram > target_vram_bytes {
            println!("[RESOURCE-WATCH] Memory sufficient or stabilized. Proceeding with {:.2} GB free VRAM.", current_vram as f64 / 1e9);
            break;
        }

        if last_report.elapsed().as_secs() >= 2 { // Faster reporting
            println!("[RESOURCE-DIAG] Waiting... VRAM: {:.2} GB free (Target: {:.2} GB)", 
                current_vram as f64 / 1e9, target_vram_mb as f64 / 1024.0);
            last_report = std::time::Instant::now();
        }

        // Absolute maximum wait 10s (reduced from 20s)
        if start_time.elapsed().as_secs() > 10 {
            println!("[RESOURCE-WATCH] Timeout or sufficient VRAM reached. Proceeding.");
            break;
        }

        last_vram = current_vram;
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    Ok(())
}


async fn update_team_base_metrics(
    store: &crate::store::VectorStore,
    team_id: &str,
    task_cc: &str,
    items: &Vec<serde_json::Value>,
    stats_diff: std::collections::HashMap<String, (i64, i64, i64)>,
) -> anyhow::Result<()> {
    let (team_json_str, team_vector, t_from, t_to, t_cc, t_bcc, t_ref, t_digest) = match store.get_item_by_id("users", team_id).await {
        Ok(Some(doc)) => (doc.json_data, doc.vector, doc.from, doc.to, doc.cc, doc.bcc, doc.r#ref, doc.digest),
        _ => (
            json!({ "base": { "pages": {} } }).to_string(),
            vec![0.0; 384],
            "".to_string(), "".to_string(), "".to_string(), "".to_string(), "".to_string(), "".to_string()
        )
    };

    
    let mut parsed_val: serde_json::Value = serde_json::from_str(&team_json_str).unwrap_or(json!({ "base": { "pages": {} } }));
    
    
    while let Some(inner_str) = parsed_val.get("json_data").and_then(|v| v.as_str()) {
        if let Ok(inner_obj) = serde_json::from_str(inner_str) {
            parsed_val = inner_obj;
        } else {
            break;
        }
    }
    let mut team_data = parsed_val;
    
    
    if let Some(obj) = team_data.as_object_mut() {
        obj.remove("json_data");
    }
    
    // --- [블록 1 & 2: 맵 순회로 모든 타입의 통계 업데이트] ---
    for (t_name, (pages_draft_diff, pages_count_diff, global_count_diff)) in stats_diff.iter() {
        // 페이지별 통계 업데이트
        {
            let base = team_data.as_object_mut().unwrap().entry("base").or_insert(json!({ "pages": {} })).as_object_mut().unwrap();
            let pages = base.entry("pages").or_insert(json!({})).as_object_mut().unwrap();
            let cc_node = pages.entry(task_cc).or_insert(json!({})).as_object_mut().unwrap();
            let page_type_node = cc_node.entry(t_name).or_insert(json!({ "draft": 0, "count": 0 })).as_object_mut().unwrap();

            let current_draft = page_type_node.get("draft").and_then(|v| v.as_i64()).unwrap_or(0);
            let current_count = page_type_node.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
            
            page_type_node.insert("draft".to_string(), json!(0.max(current_draft + pages_draft_diff)));
            page_type_node.insert("count".to_string(), json!(0.max(current_count + pages_count_diff)));
        } 

        // 글로벌 전체 통계 업데이트 (aa.ts와 동일하게 draft는 건드리지 않고 count만 누적)
        {
            let base = team_data.as_object_mut().unwrap().entry("base").or_insert(json!({ "pages": {} })).as_object_mut().unwrap();
            let global_type_node = base.entry(t_name).or_insert(json!({ "draft": 0, "count": 0 })).as_object_mut().unwrap();
            
            let global_count = global_type_node.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
            
            // 글로벌 draft는 클라우드 로직 상 사용되지 않으므로 보존하거나 건드리지 않습니다.
            global_type_node.insert("count".to_string(), json!(0.max(global_count + global_count_diff)));
        }
    }

    // Min/Max 업데이트는 items 내의 데이터에 한해서 진행
    {
        let properties = [
            "price", "quantity", "width", "height", "length", "weight", "shipping_fee", 
            "shipping_duration", "sale_price", "supply_price", "low_stock_threshold", 
            "discount", "min_order_amount", "max_discount_amount", "usage_limit", 
            "usage_per", "started_at", "expired_at"
        ];

        for item in items {
            let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("unknown");
            let base = team_data.as_object_mut().unwrap().entry("base").or_insert(json!({ "pages": {} })).as_object_mut().unwrap();
            let global_type_node = base.entry(item_type).or_insert(json!({ "draft": 0, "count": 0 })).as_object_mut().unwrap();

            for prop in properties.iter() {
                if let Some(val) = item.get(*prop) {
                    let num_val = if val.is_number() {
                        val.as_f64().unwrap_or(0.0)
                    } else if let Some(s) = val.as_str() {
                        s.parse::<f64>().unwrap_or(0.0)
                    } else {
                        continue;
                    };

                    if num_val == 0.0 && *prop != "started_at" && *prop != "expired_at" { continue; }

                    
                    let prop_node = global_type_node.entry(*prop).or_insert(json!({ "min": 0.0, "max": 0.0 })).as_object_mut().unwrap();
                    
                    let current_min = prop_node.get("min").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let current_max = prop_node.get("max").and_then(|v| v.as_f64()).unwrap_or(0.0);

                    
                    
                    if current_min == 0.0 || num_val <= current_min { prop_node.insert("min".to_string(), json!(num_val)); }
                    if current_max == 0.0 || num_val >= current_max { prop_node.insert("max".to_string(), json!(num_val)); }
                }
            }
        }
    } // 👈 여기서 두 번째 참조가 종료됩니다.

    
    if let Some(base_json) = team_data.get("base") {
        println!("\n[DEBUG-METRICS] 최종 반영된 Base JSON 값:\n{}", serde_json::to_string_pretty(base_json).unwrap_or_default());
    }

    
    if let Some(obj) = team_data.as_object_mut() {
        obj.insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
    }

    // 5. Save back to DB (digest 파라미터에 None을 전달하여 강제 쓰기를 유도합니다)
    let _ = store.upsert_item(
        "users", 
        team_id, 
        "team", 
        team_data, 
        Some(team_vector),
        Some(&t_from),
        Some(&t_to),
        Some(&t_cc),
        Some(&t_bcc),
        Some(&t_ref),
        None
    ).await;

    
    if let Ok(Some(saved_doc)) = store.get_item_by_id("users", team_id).await {
        println!("\n==================================================");
        println!("✅ [DB-VERIFY] DB에 통계(Team) 데이터가 100% 정상 저장되었습니다!");
        println!("- 타겟 ID: {}", saved_doc.id);
        println!("- 갱신된 Timestamp: {}", saved_doc.updated_at_ts);
        
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&saved_doc.json_data) {
            if let Some(base_stats) = parsed.get("base") {
                println!("- DB 내 실제 Base 통계:\n{}", serde_json::to_string_pretty(base_stats).unwrap_or_default());
            }
        }
        println!("==================================================\n");
    } else {
        println!("\n==================================================");
        println!("🚨 [DB-VERIFY] 치명적 오류: DB에 Team 데이터가 저장되지 않았습니다!");
        println!("==================================================\n");
    }

    Ok(())
}


