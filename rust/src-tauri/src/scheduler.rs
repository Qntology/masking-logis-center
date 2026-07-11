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

// 🌟 [추가] Stanza ONNX 모델(pos.onnx, depparse.onnx) 입력용 전처리 모듈 (Vocab -> ndarray Tensor)
use std::collections::HashMap;
use std::path::Path;
use ndarray::Array2;
use onnxruntime::environment::Environment;
use onnxruntime::session::Session;
use onnxruntime::GraphOptimizationLevel;

#[derive(Debug, Clone)]
pub struct StanzaPreprocessor {
    pub word_vocab: HashMap<String, i64>,
    pub char_vocab: HashMap<char, i64>,
    pub upos_vocab: Vec<String>,
    pub word_unk_id: i64,
    pub char_unk_id: i64,
}

impl StanzaPreprocessor {
    pub fn new<P: AsRef<Path>>(vocab_path: P) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(vocab_path.as_ref())
            .map_err(|e| anyhow::anyhow!("Failed to read vocab.json: {}", e))?;
        
        let json_val: serde_json::Value = serde_json::from_str(&data)
            .map_err(|e| anyhow::anyhow!("Failed to parse vocab.json as JSON: {}", e))?;
            
        let mut word_vocab: HashMap<String, i64> = HashMap::new();
        let mut char_vocab: HashMap<char, i64> = HashMap::new();
        let mut upos_vocab = Vec::new();
        
        // 🌟 1. Word Vocab 파싱 (기존 로직 보존 및 통합)
        let word_target = if let Some(pos) = json_val.get("pos") {
            pos.get("word").unwrap_or(&json_val)
        } else if let Some(tokenize) = json_val.get("tokenize") {
            tokenize.get("main").unwrap_or(&json_val)
        } else {
            &json_val
        };

        Self::extract_vocab_from_node(word_target, &mut word_vocab);

        // 🌟 2. Char Vocab 파싱 (Stanza OOV 극복의 핵심)
        let char_target = if let Some(pos) = json_val.get("pos") {
            pos.get("char").unwrap_or(&serde_json::Value::Null)
        } else if let Some(ner) = json_val.get("ner") {
            ner.get("char").unwrap_or(&serde_json::Value::Null)
        } else {
            &serde_json::Value::Null
        };

        let mut temp_char_vocab: HashMap<String, i64> = HashMap::new();
        Self::extract_vocab_from_node(char_target, &mut temp_char_vocab);
        
        for (k, v) in temp_char_vocab {
            if let Some(c) = k.chars().next() {
                char_vocab.insert(c, v);
            }
        }

        // 🌟 3. UPOS Vocab 동적 파싱 (하드코딩을 파괴하고 파일에서 인덱스 배열 정답을 그대로 수집)
        if let Some(pos_node) = json_val.get("pos") {
            if let Some(upos_arr) = pos_node.get("upos").and_then(|v| v.as_array()) {
                for v in upos_arr {
                    if let Some(s) = v.as_str() {
                        upos_vocab.push(s.to_string());
                    }
                }
            }
        }

        if word_vocab.is_empty() {
            return Err(anyhow::anyhow!("vocab.json 내부에서 단어 매핑(Vocab) 구조를 찾을 수 없습니다."));
        }
        
        let word_unk_id = *word_vocab.get("<unk>")
            .or_else(|| word_vocab.get("<UNK>"))
            .or_else(|| word_vocab.get("[UNK]"))
            .unwrap_or(&0);
            
        let char_unk_id = *char_vocab.get(&'<').unwrap_or(&0); // '<unk>' 처리용
        
        Ok(Self { word_vocab, char_vocab, upos_vocab, word_unk_id, char_unk_id })
    }

    // 🌟 중복된 JSON 파싱 로직을 공통 헬퍼 함수로 분리
    fn extract_vocab_from_node(target_value: &serde_json::Value, vocab: &mut HashMap<String, i64>) {
        if let Some(arr) = target_value.as_array() {
            for (i, v) in arr.iter().enumerate() {
                if let Some(s) = v.as_str() {
                    vocab.insert(s.to_string(), i as i64);
                } else if let Some(obj) = v.as_object() {
                    let word_opt = obj.get("word").and_then(|w| w.as_str());
                    let id_opt = obj.get("id").and_then(|id| id.as_i64()).unwrap_or(i as i64);
                    if let Some(w) = word_opt {
                        vocab.insert(w.to_string(), id_opt);
                    } else {
                        for (k, val) in obj {
                            if let Some(id_val) = val.get("id").and_then(|id| id.as_i64()) {
                                vocab.insert(k.clone(), id_val);
                            } else if let Some(id_val) = val.as_i64() {
                                vocab.insert(k.clone(), id_val);
                            }
                        }
                    }
                }
            }
        } else {
            let target_obj = if let Some(model) = target_value.get("model") {
                model.get("vocab").and_then(|v| v.as_object())
            } else if let Some(vocab_node) = target_value.get("vocab") {
                vocab_node.as_object()
            } else if let Some(id_to_string) = target_value.get("id_to_string") {
                if let Some(obj) = id_to_string.as_object() {
                    for (id_str, word_val) in obj {
                        if let (Ok(parsed_id), Some(w)) = (id_str.parse::<i64>(), word_val.as_str()) {
                            vocab.insert(w.to_string(), parsed_id);
                        }
                    }
                }
                None
            } else {
                target_value.as_object()
            };

            if let Some(obj) = target_obj {
                for (k, v) in obj {
                    if let Some(id) = v.as_i64() {
                        vocab.insert(k.clone(), id);
                    } else if let Some(s) = v.as_str() {
                        if let Ok(parsed_id) = s.parse::<i64>() {
                            vocab.insert(k.clone(), parsed_id);
                        }
                    } else if let Some(id_val) = v.get("id").and_then(|i| i.as_i64()) {
                        vocab.insert(k.clone(), id_val);
                    } else if v.is_object() || v.is_array() {
                        if let Some(id_val) = v.get("id").and_then(|i| i.as_i64()) {
                            vocab.insert(k.clone(), id_val);
                        }
                    }
                }
            }
        }
    }

    /// 품사 태깅(pos.onnx)을 위해 분할된 단어 배열을 Word 텐서와 Wordchar(길이) 텐서로 변환합니다.
    pub fn encode_to_tensor(&self, words: &[&str], session: &Session<'static>) -> Result<Vec<ndarray::ArrayD<i64>>, anyhow::Error> {
        let seq_len = words.len();
        
        // 🌟 [CRITICAL FIX] 빈 배열(seq_len == 0)이 주어지면 ONNX LSTM Reshape 노드에서 치명적인 에러가 발생하므로 사전에 차단합니다.
        if seq_len == 0 {
            return Err(anyhow::anyhow!("입력된 단어 배열이 비어있어 ONNX 텐서 변환을 수행할 수 없습니다."));
        }

        let mut word_ids = Vec::with_capacity(seq_len);
        let mut wlen_vec = Vec::with_capacity(seq_len);
        let mut oidx_vec = Vec::with_capacity(seq_len);
        
        // 🌟 [CRITICAL FIX] Python Export 시 charmodel의 시퀀스 길이가 32로 고정(Hardcoded)되어 있습니다.
        // 동적 길이를 사용하면 ONNX Runtime에서 차원 불일치(Shape Mismatch) 에러가 발생하므로 32로 강제 고정합니다.
        let max_word_len = 32; 
        
        let mut chars_raw = ndarray::Array2::<i64>::zeros((seq_len, max_word_len));
        let mut chars_mask_raw = ndarray::Array2::<i64>::zeros((seq_len, max_word_len));

        for (w_idx, w) in words.iter().enumerate() {
            let token_id = *self.word_vocab.get(*w)
                .or_else(|| self.word_vocab.get(&w.to_lowercase()))
                .unwrap_or(&self.word_unk_id);
            word_ids.push(token_id);
            
            let w_chars: Vec<char> = w.chars().collect();
            // 🌟 [CRITICAL FIX] 32자를 초과하는 단어 길이는 ONNX Gather 연산 시 Out of Bounds 에러를 유발하므로 32로 제한(Clamp)합니다.
            let safe_wlen = w_chars.len().min(32);
            wlen_vec.push(safe_wlen as i64);
            oidx_vec.push(w_idx as i64);
            
            for (c_idx, c) in w_chars.iter().take(32).enumerate() {
                let c_id = *self.char_vocab.get(c).unwrap_or(&self.char_unk_id);
                chars_raw[[w_idx, c_idx]] = c_id;
                chars_mask_raw[[w_idx, c_idx]] = 1; // ONNX Runtime 0.0.14 대응을 위해 bool을 1/0 i64로 강제 래핑
            }
        }
        
        let word_tensor = ndarray::Array2::from_shape_vec((1, seq_len), word_ids)
            .map_err(|e| anyhow::anyhow!("Failed to build word tensor: {}", e))?.into_dyn();
        let mask_tensor = ndarray::Array2::<i64>::ones((1, seq_len)).into_dyn();
        let chars_tensor = chars_raw.into_dyn();
        let chars_mask_tensor = chars_mask_raw.into_dyn();
        let pre_tensor = ndarray::Array2::<i64>::zeros((1, seq_len)).into_dyn();
        let oidx_tensor = ndarray::Array1::from_vec(oidx_vec).into_dyn();
        let slen_tensor = ndarray::Array1::from_vec(vec![seq_len as i64]).into_dyn();
        let wlen_tensor = ndarray::Array1::from_vec(wlen_vec).into_dyn();
        
        // 🌟 [CRITICAL FIX] PyTorch ONNX Export 과정에서 사용되지 않은 입력이 제거될 수 있으므로 동적 조립하되,
        // 차원(Shape)만으로 매핑하면 pre_tensor(0)와 mask_tensor(1)가 뒤바뀌어 모든 출력이 PROPN으로 오작동합니다.
        // 반드시 input_meta.name을 파싱하여 정확한 텐서를 지정해야 문맥이 차단되는 Hallucination을 막을 수 있습니다.
        let mut final_inputs = Vec::new();

        for input_meta in &session.inputs {
            let name = input_meta.name.to_lowercase();
            
            // 🌟 [CRITICAL FIX] 섀도잉(Shadowing) 논리 결함 완벽 수정:
            // "word_mask"가 "word"에 걸리고, "wordchar_len"이 "char"에 걸려 텐서가 뒤섞이는 현상 차단.
            // 가장 구체적인 키워드(mask, len)부터 먼저 필터링하도록 계층화했습니다.
            if name.contains("mask") {
                if name.contains("char") {
                    final_inputs.push(chars_mask_tensor.clone());
                } else {
                    final_inputs.push(mask_tensor.clone());
                }
            } else if name.contains("len") || name.contains("seq") {
                if name.contains("word") || name.contains("wlen") || name.contains("char") {
                    final_inputs.push(wlen_tensor.clone());
                } else {
                    final_inputs.push(slen_tensor.clone());
                }
            } else if name.contains("pre") || name.contains("pretrained") {
                final_inputs.push(pre_tensor.clone());
            } else if name.contains("char") {
                final_inputs.push(chars_tensor.clone());
            } else if name.contains("word") {
                final_inputs.push(word_tensor.clone());
            } else if name.contains("oidx") || name.contains("orig") {
                final_inputs.push(oidx_tensor.clone());
            } else {
                // 예외 상황: 이름 기반 매칭이 불가능할 경우 차원 기반 Fallback 안전장치
                let dims = &input_meta.dimensions;
                if dims.len() == 2 && dims.get(0) == Some(&Some(1)) {
                    final_inputs.push(word_tensor.clone()); 
                } else if dims.len() == 2 && dims.get(1) == Some(&Some(32)) {
                    final_inputs.push(chars_tensor.clone());
                } else if dims.len() == 1 {
                    final_inputs.push(slen_tensor.clone());
                } else {
                    final_inputs.push(word_tensor.clone());
                }
            }
        }
        
        Ok(final_inputs)
    }
}

// 🌟 [추가] ONNX Runtime 세션을 초기화하고 보유하는 파이프라인 구조체
pub struct StanzaPipeline {
    pub preprocessor: StanzaPreprocessor,
    pub tokenize_session: Session<'static>,
    pub pos_session: Session<'static>,
}

// (로컬 라이브러리 onnxruntime crate 자체에 Send/Sync를 구현하였으므로 더 이상 unsafe 래퍼가 필요 없습니다!)

impl StanzaPipeline {
    pub fn new<P: AsRef<Path>>(base_dir: P, lang: &str) -> anyhow::Result<Self> {
        let lang_dir = base_dir.as_ref().join(lang);
        let vocab_path = lang_dir.join("vocab.json");
        let tokenize_path = lang_dir.join("tokenizer.onnx");
        let pos_path = lang_dir.join("pos.onnx");

        let preprocessor = StanzaPreprocessor::new(&vocab_path)?;

        let total_start_time = std::time::Instant::now();

        // onnxruntime 0.0.14 요구사항: Environment 할당 (static으로 메모리 릭(Leak)하여 생명주기 문제 우회)
        let env = Box::leak(Box::new(
            Environment::builder()
                .with_name("stanza_env")
                .build()
                .map_err(|e| anyhow::anyhow!("Env error: {}", e))?
        ));

        // 🌟 [onnxruntime 0.0.14 버그 우회] 
        // 구버전 라이브러리의 설계 결함으로 인해, 파일 경로 문자열의 수명(Lifetime)이 
        // Session<'static>과 동일하게 'static으로 유지되어야 컴파일이 통과됩니다.
        // 경로 문자열을 메모리에 영구 고정(Leak)하여 수명 문제를 완벽히 해결합니다.
        let tokenize_path_static: &'static str = Box::leak(tokenize_path.to_string_lossy().into_owned().into_boxed_str());
        let pos_path_static: &'static str = Box::leak(pos_path.to_string_lossy().into_owned().into_boxed_str());

        let tok_start_time = std::time::Instant::now();
        println!("[STANZA] TOKENIZER 모델 세션을 빌드합니다...");
        
        let tokenize_session = env.new_session_builder()
            .map_err(|e| anyhow::anyhow!("Tokenizer Session builder error: {}", e))?
            .with_model_from_file(tokenize_path_static)
            .map_err(|e| anyhow::anyhow!("tokenizer.onnx 모델 파일 로드 실패: {}", e))?;
            
        println!("[STANZA] ✅ TOKENIZER 모델 세션 빌드 완료! (소요 시간: {:.2}초)", tok_start_time.elapsed().as_secs_f32());

        let pos_start_time = std::time::Instant::now();
        println!("[STANZA] POS 모델 세션을 빌드합니다 (onnxruntime 0.0.14)...");
        
        let pos_session = env.new_session_builder()
            .map_err(|e| anyhow::anyhow!("POS Session builder error: {}", e))?
            .with_model_from_file(pos_path_static)
            .map_err(|e| anyhow::anyhow!("pos.onnx 모델 파일 로드 실패: {}", e))?;
            
        println!("[STANZA] ✅ POS 모델 세션 빌드 완료! (소요 시간: {:.2}초)", pos_start_time.elapsed().as_secs_f32());

        println!("[STANZA] 🚀 모든 세션 로드 완료! (총 소요 시간: {:.2}초)", total_start_time.elapsed().as_secs_f32());
        
        Ok(Self {
            preprocessor,
            tokenize_session,
            pos_session,
        })
    }
}
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
    
    let log_file_path = crate::utils::paths::get_app_tmp_root(Some(&app_handle_clone)).join(format!("{}_masking.log", tid_clone));
    let _ = std::fs::create_dir_all(crate::utils::paths::get_app_tmp_root(Some(&app_handle_clone)));
    let _ = std::fs::write(&log_file_path, "");

    let emit_state = std::sync::Arc::new(std::sync::Mutex::new((String::new(), 0usize)));

    let emit_term = move |msg: &str| {
        let mut state = emit_state.lock().unwrap();
        let (last_msg, count) = &mut *state;

        let is_spam = msg.contains("이미 다른 동의어 트랙에서 마스킹 완료된") 
            || msg.contains("[STANZA] 1차 형태소 분리 완료")
            || msg.contains("연쇄 파기(Cascade Cancellation)");

        if is_spam && msg == last_msg.as_str() {
            *count += 1;
            return;
        }

        let mut output_msg = String::new();
        if *count > 0 {
            output_msg.push_str(&format!("... (동일 로그 중복 발생으로 {}회 출력 생략) ...\n", count));
            *count = 0;
        }
        output_msg.push_str(msg);

        use std::io::Write;
        if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(&log_file_path) {
            let _ = writeln!(file, "{}", output_msg);
        }

        println!("{}", output_msg);
        use tauri::Emitter;
        let _ = app_handle_clone.emit("task-console-log", serde_json::json!({"task_id": tid_clone, "text": format!("{}\n", output_msg)}));

        *last_msg = msg.to_string();
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
                        // 🌟 [추가] 초기화 실패 시 터미널과 UI 양쪽에 상세 에러 스택을 남깁니다.
                        let err_msg = format!("LogisModel::new 초기화 실패 상세 원인: {:?}", e);
                        println!("[Scheduler] ❌ {}", err_msg);
                        log_task_progress(app_handle, &task.id, &json!({ "category": "Error", "summary": err_msg }));
                        return Err(anyhow::anyhow!("Model Load Failed: {:?}", e));
                    }
                }
            }
            model_lock.as_ref().unwrap().clone()
        };

        emit_term("[PROCESS] Starting batch masking for selected documents...");

        // 🌟 [CRITICAL FIX] Granite 모델을 매번 불러오지 않기 위해 마스킹 작업 전체 진입 전 최초 1회만 로드합니다!
        // 현재는 임베딩 모델만 사용하므로 불필요한 Granite LLM VRAM 할당을 방지하기 위해 주석 처리합니다.
        // model.ensure_granite().await?;
        
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

                    // 🌟 [변경] qwen, qwen3 대신 모두 Granite 사용
                    let is_large_context = target_text.len() > 60000;
                    let target_model_size = crate::model::ModelSize::Granite;

                    // 🌟 [OOM 원인 분석용 로그] 모델에 투입되기 직전 전체 컨텍스트의 문자열 길이를 터미널에 출력합니다.
                    emit_term(&format!("[DEBUG-OOM] 현재 투입되는 컨텍스트 사이즈(문자 수): {}. 선택된 모델: {:?}", target_text.len(), target_model_size));

                    // 🌟 Granite 모델 로딩은 상단 작업 진입부 최초 1회 로드로 이동되었습니다.

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
                        ("national_id", "national identification number or resident registration number"),
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

                    // 🌟 [CRITICAL FIX] 다국어 검증(Stage 3) 시 local_language를 무조건 가장 먼저(0번 인덱스) 검증하도록 재배열하여 불필요한 타언어(English 등) LLM 추론을 최소화합니다.
                    if let Some(pos) = detected_languages_vec.iter().position(|l| l == &local_language) {
                        let local = detected_languages_vec.remove(pos);
                        detected_languages_vec.insert(0, local);
                    }

                    emit_term(&format!("[EXTRACTION] 🌐 Detected Languages: {:?} (Local: {})", detected_languages_vec, local_language));

                    // 🌟 [추가] Stanza Pipeline 동적 로드 (로컬 언어 기준)
                    let stanza_lang_code = match local_language.as_str() {
                        "korean" => "ko",
                        "english" => "en",
                        "japanese" => "ja",
                        "chinese" => "zh-hans",
                        "french" => "fr",
                        "german" => "de",
                        "spanish" => "es",
                        "italian" => "it",
                        "portuguese" => "pt",
                        "dutch" => "nl",
                        "russian" => "ru",
                        "arabic" => "ar",
                        _ => "en",
                    };
                    
                    // =====================================================================
                    // 🌟 [CRITICAL FIX] 순차적 로딩 강제 (Concurrency Deadlock 원천 차단)
                    // LLM(Granite) 백그라운드 로딩은 상단 최초 1회 로드로 이동되었으므로 즉시 ONNX를 로드합니다.
                    // =====================================================================
                    emit_term("[EXTRACTION] 🧠 LLM 로딩 완료 확인. ONNX 로딩 진입...");
                    
                    // 🌟 [추가] LLM 모델 로딩 실패 시 상세 로그(에러 스택)를 UI 터미널에 출력합니다.
                    // if let Err(e) = model.secure_vram_relay(target_model_size, None, Some(cancellation_token.clone()), false, None).await {
                    //     let err_msg = format!("LLM 모델({:?}) 로딩 실패 상세 원인: {:?}", target_model_size, e);
                    //     emit_term(&format!("🚨 [CRITICAL ERROR] {}", err_msg));
                    //     return Err(anyhow::anyhow!(err_msg));
                    // }

                    let stanza_base_dir = crate::utils::get_app_dir().join("models").join("stanza");
                    let mut stanza_pipeline = None;
                    if stanza_base_dir.join(stanza_lang_code).exists() {
                        emit_term(&format!("[STANZA] 🧠 Loading Stanza ONNX models for '{}'...", stanza_lang_code));
                        
                        let base_dir_clone = stanza_base_dir.clone();
                        let lang_code_clone = stanza_lang_code.to_string();
                        
                        // 🌟 [UNSAFE BYPASS] 구버전 onnxruntime(0.0.14)은 내부 C++ 포인터(*mut)에 대해 
                        // Rust의 스레드 간 전송(Send) 트레이트 구현을 누락하는 설계 결함이 있습니다.
                        // 컴파일러의 락을 강제로 해제하기 위해 Unsafe 래퍼 구조체를 선언하여 전송 자격을 억지로 부여합니다.
                        struct UnsafePipelineWrapper(StanzaPipeline);
                        unsafe impl Send for UnsafePipelineWrapper {}
                        
                        let (tx, rx) = tokio::sync::oneshot::channel::<anyhow::Result<UnsafePipelineWrapper>>();
                        std::thread::spawn(move || {
                            let res = StanzaPipeline::new(base_dir_clone, &lang_code_clone).map(UnsafePipelineWrapper);
                            let _ = tx.send(res);
                        });
                        
                        let pipeline_res = rx.await.unwrap_or_else(|_| Err(anyhow::anyhow!("OS 스레드 통신 채널이 끊어졌습니다.")));

                        match pipeline_res {
                            Ok(wrapper) => {
                                stanza_pipeline = Some(wrapper.0);
                                emit_term("[STANZA] ✅ Stanza Pipeline loaded successfully.");
                            },
                            Err(e) => emit_term(&format!("[STANZA] ⚠️ Failed to load Stanza models (상세 원인): {:?}", e)),
                        }
                    } else {
                        emit_term(&format!("[STANZA] ⚠️ Stanza models for '{}' not found. Skipping POS/Depparse filtering.", stanza_lang_code));
                    }

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

                    // 🌟 [CRITICAL FIX] 일반 텍스트 교체 시 [___REDACTED_x___] 마커 내부의 숫자가 오염되는 현상을 방지하는 안전한 교체 함수
                    // (정규식을 완전히 제거하고, 단어 중간에 낀 파편 오작동을 방어하는 순수 Rust 탐색 로직으로 개선)
                    let safe_replace = |text: &str, target: &str, replacement: &str| -> String {
                        if target.is_empty() { return text.to_string(); }
                        
                        let mut result = String::with_capacity(text.len());
                        let mut current_idx = 0;
                        let target_char_count = target.chars().count();

                        while current_idx < text.len() {
                            let text_slice = &text[current_idx..];
                            let marker_idx = text_slice.find("[___REDACTED_");
                            let target_idx = text_slice.find(target);

                            match (marker_idx, target_idx) {
                                (Some(m_idx), Some(t_idx)) if m_idx <= t_idx => {
                                    let absolute_m_idx = current_idx + m_idx;
                                    if let Some(end_offset) = text[absolute_m_idx..].find("___]") {
                                        let absolute_end_idx = absolute_m_idx + end_offset + 4;
                                        result.push_str(&text[current_idx..absolute_end_idx]);
                                        current_idx = absolute_end_idx;
                                    } else {
                                        result.push_str(&text[current_idx..=absolute_m_idx]);
                                        current_idx = absolute_m_idx + 1;
                                    }
                                },
                                (Some(_), Some(t_idx)) | (None, Some(t_idx)) => {
                                    let absolute_t_idx = current_idx + t_idx;
                                    
                                    // 🌟 [Infix 방어 로직] 외래어나 다른 단어의 중간에 낀 파편(예: 에'이전'트) 치환 방지 및 영문 띄어쓰기(부분 일치) 방어
                                    let mut is_infix = false;
                                    let char_before = text[..absolute_t_idx].chars().next_back();
                                    let char_after = text[absolute_t_idx + target.len()..].chars().next();
                                    
                                    let is_ascii_target = target.chars().all(|c| c.is_ascii_alphanumeric() || c.is_whitespace() || c == '-' || c == '_');

                                    if is_ascii_target {
                                        // 영문/숫자 타겟인 경우, 앞이나 뒤에 영문자/숫자가 하나라도 붙어있으면 독립 단어가 아닌 파편(Prefix/Suffix/Infix)으로 간주
                                        let prev_is_alpha = char_before.map_or(false, |c| c.is_ascii_alphanumeric());
                                        let next_is_alpha = char_after.map_or(false, |c| c.is_ascii_alphanumeric());
                                        if prev_is_alpha || next_is_alpha {
                                            is_infix = true;
                                        }
                                    } else {
                                        // 기존 한글 및 기타 언어 로직 보완
                                        let prev_is_alnum = char_before.map_or(false, |c| c.is_alphanumeric());
                                        let next_is_alnum = char_after.map_or(false, |c| c.is_alphanumeric());
                                        
                                        if prev_is_alnum && next_is_alnum {
                                            // 앞뒤가 모두 문자로 둘러싸여 있는 경우 (예: 에'이전'트)
                                            if target_char_count <= 2 {
                                                is_infix = true;
                                            }
                                        } else if prev_is_alnum || next_is_alnum {
                                            // 앞이나 뒤 한쪽만 붙어있는 경우, 타겟이 1글자면 무조건 파편으로 간주
                                            if target_char_count <= 1 {
                                                is_infix = true;
                                            }
                                        }
                                    }
                                    
                                    if is_infix {
                                        result.push_str(&text[current_idx..absolute_t_idx + target.len()]);
                                    } else {
                                        result.push_str(&text[current_idx..absolute_t_idx]);
                                        result.push_str(replacement);
                                    }
                                    current_idx = absolute_t_idx + target.len();
                                },
                                (Some(m_idx), None) => {
                                    let absolute_m_idx = current_idx + m_idx;
                                    if let Some(end_offset) = text[absolute_m_idx..].find("___]") {
                                        let absolute_end_idx = absolute_m_idx + end_offset + 4;
                                        result.push_str(&text[current_idx..absolute_end_idx]);
                                        current_idx = absolute_end_idx;
                                    } else {
                                        result.push_str(&text[current_idx..=absolute_m_idx]);
                                        current_idx = absolute_m_idx + 1;
                                    }
                                },
                                (None, None) => {
                                    result.push_str(&text[current_idx..]);
                                    break;
                                }
                            }
                        }
                        result
                    };

                    // 🌟 [사전 정규식 추출] email 먼저 마스킹 (1차 패스 전)
                    if let Ok(email_re) = regex::Regex::new(r"(?i)[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}") {
                        let mut found_emails = std::collections::HashSet::new();
                        for mat in email_re.find_iter(&masked_text) { found_emails.insert(mat.as_str().to_string()); }
                        for email_val in found_emails {
                            let mnemonic = crate::parsing::generate_mnemonic();
                            let upper_key = "EMAIL".to_string();
                            let final_replacement = format!("[{}]", mnemonic);
                            let skip_marker = format!("[___REDACTED_{}___]", skip_counter);
                            masked_text = safe_replace(&masked_text, &email_val, &skip_marker);
                            doc_title = safe_replace(&doc_title, &email_val, &skip_marker);
                            doc_desc = safe_replace(&doc_desc, &email_val, &skip_marker);
                            skip_map.insert(skip_marker.clone(), final_replacement);
                            replacement_history.push((email_val.clone(), skip_marker.clone()));
                            domain_history.push(("email".to_string(), email_val.clone()));
                            skip_counter += 1;
                            all_matches.push(json!({ "name": upper_key, "value": email_val, "mnemonic": mnemonic }));
                            emit_term(&format!("[EXTRACTION] 📧 이메일 정규식 사전 추출 성공: {} -> 강제 마스킹 완료", email_val));
                        }
                    }

                    // 🌟 [사전 정규식 추출] 웹주소(URL) 마스킹 (http, https 모두 포함)
                    if let Ok(url_re) = regex::Regex::new(r"(?i)(https://|http://|www\.)[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/[a-zA-Z0-9./?%&=-]*)?") {
                        let mut found_urls = std::collections::HashSet::new();
                        for mat in url_re.find_iter(&masked_text) { found_urls.insert(mat.as_str().to_string()); }
                        for url_val in found_urls {
                            let mnemonic = crate::parsing::generate_mnemonic();
                            let upper_key = "URL".to_string();
                            let final_replacement = format!("[{}]", mnemonic);
                            let skip_marker = format!("[___REDACTED_{}___]", skip_counter);
                            masked_text = safe_replace(&masked_text, &url_val, &skip_marker);
                            doc_title = safe_replace(&doc_title, &url_val, &skip_marker);
                            doc_desc = safe_replace(&doc_desc, &url_val, &skip_marker);
                            skip_map.insert(skip_marker.clone(), final_replacement);
                            replacement_history.push((url_val.clone(), skip_marker.clone()));
                            domain_history.push(("url".to_string(), url_val.clone()));
                            skip_counter += 1;
                            all_matches.push(json!({ "name": upper_key, "value": url_val, "mnemonic": mnemonic }));
                            emit_term(&format!("[EXTRACTION] 🔗 웹주소 정규식 사전 추출 성공: {} -> 강제 마스킹 완료", url_val));
                        }
                    }

                    // 🌟 [CRITICAL FIX] 특수문자 제거 및 언어(단어) 추출 범용 로직
                    // 기존에 특수문자를 공백으로 덮어씌워 "FDE(Forward"가 "FDE Forward"로 쪼개지는 현상을 방지합니다.
                    // 원본 텍스트 구조를 그대로 유지하여 NMS 추출 시 괄호 등 특수문자가 훼손되지 않도록 개선합니다.
                    let mut noise_map: std::collections::HashMap<String, String> = std::collections::HashMap::new(); // 🌟 하단 복원 로직 컴파일 에러 방지용 빈 장부 유지

                    // 🌟 1차 패스용 라인 분할 (원본 텍스트 기반 유지)
                    let structural_tags = ["html", "body", "div", "p", "span", "thead", "tbody", "tr", "td", "th", "table", "ul", "li", "a", "img", "br", "h1", "h2", "h3", "h4", "h5", "h6", "strong", "em", "b", "i", "u", "s", "nav", "header", "footer", "main", "section", "article", "aside", "figure", "figcaption", "button", "input", "form", "label", "select", "textarea", "option", "iframe", "script", "style", "meta", "link", "head", "title", "svg", "path", "dl", "ol", "dd", "dt"];
                    let mut lines: Vec<String> = masked_text.lines()
                        .map(|s| s.trim().trim_start_matches('|').trim().to_string())
                        .filter(|s| {
                            let s_lower = s.to_lowercase();
                            s.len() > 2 && !structural_tags.contains(&s_lower.as_str())
                        })
                        .collect();
                        
                    // 🌟 [추가] 제목과 요약 텍스트도 마스킹 탐색 라인업에 추가합니다.
                    if !doc_title.trim().is_empty() { lines.push(doc_title.trim().to_string()); }
                    if !doc_desc.trim().is_empty() { lines.push(doc_desc.trim().to_string()); }

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
                        
                    // 🌟 [추가] 2차 패스 초기화 직후에도 제목과 요약 텍스트를 재장전합니다.
                    if !doc_title.trim().is_empty() { lines.push(doc_title.trim().to_string()); }
                    if !doc_desc.trim().is_empty() { lines.push(doc_desc.trim().to_string()); }
                        
                    emit_term(&format!("[EXTRACTION] 문서 제목, 요약, 본문을 총 {}개의 라인으로 분할하여 순차 임베딩 및 NMS 배틀 진행 중...", lines.len()));

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

                    // 🌟 [추가] 문서 제목(Title) 임베딩 생성 (NMS 경쟁 시 고유명사 타이브레이커 가중치용)
                    let title_emb = model.get_embedding(doc_title.clone()).await.unwrap_or_else(|_| vec![0.0; 384]);

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

                    let redacted_marker_prefix = "[___REDACTED_".to_string();

                    for (line_idx, line) in lines.iter().enumerate() {
                        if cancellation_token.load(Ordering::Relaxed) { break; }
                        
                        let words: Vec<&str> = line.split_whitespace().collect();
                        
                        for start in 0..words.len() {
                            let max_end = words.len().min(start + 2); // 1~2 단어 조합
                            for end in (start + 1)..=max_end {
                                // 🌟 [CRITICAL FIX] 사전 마스킹된 마커(이메일 등), 노이즈, 링크 해시 마커를 벡터 청크에서 모두 제외하여 순수 텍스트만 임베딩합니다.
                                let clean_words: Vec<&str> = words[start..end].iter()
                                    .filter(|&&w| !w.starts_with(&redacted_marker_prefix))
                                    .copied()
                                    .collect();
                                
                                if clean_words.is_empty() { continue; }
                                
                                // 🌟 [CRITICAL FIX] 단순 공백 결합이 아닌 원본 텍스트 기반 정규식 발췌 로직 적용 (PASS 2)
                                // 다중 공백(특수기호 치환 흔적 등)을 완벽히 보존하여 이후 contains 검사에서 누락되지 않도록 합니다.
                                let mut regex_pattern = String::new();
                                for (i, word) in clean_words.iter().enumerate() {
                                    let escaped_word = regex::escape(word);
                                    if i > 0 { regex_pattern.push_str(r"\s+"); }
                                    regex_pattern.push_str(&escaped_word);
                                }
                                
                                let line_text = &lines[line_idx];
                                let chunk_text = if let Ok(re) = regex::Regex::new(&regex_pattern) {
                                    if let Some(mat) = re.find(line_text) {
                                        mat.as_str().to_string()
                                    } else {
                                        clean_words.join(" ")
                                    }
                                } else {
                                    clean_words.join(" ")
                                };

                                // 특수기호만 존재하는 무의미한 텍스트 스킵
                                if chunk_text.trim().chars().all(|c| !c.is_alphanumeric()) { continue; }

                                // 🌟 [CRITICAL FIX] 1음절(글자 수 1개) 쓰레기 단어('와', '가', '의' 등)의 NMS 우승 원천 차단
                                let char_count = chunk_text.chars().filter(|c| !c.is_whitespace()).count();
                                if char_count <= 1 { continue; }

                                // 🌟 [수정] 벡터 품질 보존을 위해 임베딩 모델에 들어가는 텍스트만 특수문자를 공백으로 치환합니다.
                                let mut clean_for_emb = String::new();
                                for c in chunk_text.chars() {
                                    if c.is_alphanumeric() || c == '-' {
                                        clean_for_emb.push(c);
                                    } else {
                                        clean_for_emb.push(' ');
                                    }
                                }
                                let chunk_emb = model.get_embedding(clean_for_emb).await.unwrap_or_else(|_| vec![0.0; 384]);
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
                                    
                                    // 🌟 [추가] 문맥 의존도가 높은 고유명사 타겟(이름, 회사, 계정)에 한하여 제목 벡터 유사도 보너스를 동적으로 부여합니다.
                                    let (_, base_target, _) = &dynamic_target_items[i];
                                    let mut title_bonus = 0.0;
                                    let t_sim = cosine_similarity(&chunk_emb, &title_emb);
                                    if t_sim > 0.0 {
                                        title_bonus = t_sim * 0.15; // 제목과 벡터 방향성이 일치하는 단어에 15% 보정치 가점 부여
                                    }
                                    
                                    // 🌟 타이브레이커 감점 및 제목 보너스를 최종 스코어에 반영
                                    let score = b_score - (p_score * penalty_weight) - verb_penalty + title_bonus;

                                    top_targets.push((i, score));
                                }

                                // 점수 순으로 내림차순 정렬
                                top_targets.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                                let best_score = top_targets[0].1;

                                // 🌟 2차 패스 커트라인 조정 및 0.05 편차(Margin) 공동 우승 허용 로직 적용
                                if best_score > 0.3 {
                                    let final_score = best_score * length_weight;

                                    if verb_penalty > 0.05 {
                                        emit_term(&format!("    📉 [VERB PENALTY] '{}' -> 감점: {:.4} (최종 반영 스코어: {:.4})", chunk_text, verb_penalty, final_score));
                                    }

                                    let mut selected_indices = Vec::new();
                                    
                                    for (idx, score) in top_targets {
                                        if best_score - score <= 0.05 && score > 0.3 {
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

                    // 🌟 4. NMS BATTLE (Pass 3) - 오버랩 충돌 해결 (LLM Granite 기반)
                    emit_term("  ⚔️ [PASS 3: NMS BATTLE] Resolving Overlaps across all targets using LLM Granite...");
                    eval_spans.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                    
                    let mut final_spans: Vec<ChunkSpan> = Vec::new();
                    let mut remaining_spans = eval_spans.clone();

                    while !remaining_spans.is_empty() {
                        let current = remaining_spans.remove(0); // 가장 점수 높은 조각
                        let mut overlaps = Vec::new();
                        let mut next_remaining = Vec::new();

                        for span in remaining_spans {
                            if current.line_idx == span.line_idx && current.start < span.end && current.end > span.start {
                                overlaps.push(span);
                            } else {
                                next_remaining.push(span);
                            }
                        }

                        if overlaps.is_empty() {
                            for &t_idx in &current.target_indices {
                                let (context_name, _, _) = &dynamic_target_items[t_idx];
                                emit_term(&format!("    👑 [WINNER] '{}' -> {} (Score: {:.4})", current.text, context_name.replace("_", " "), current.score));
                            }
                            final_spans.push(current);
                        } else {
                            let mut candidates = vec![current.clone()];
                            candidates.extend(overlaps.clone());
                            
                            let mut unique_texts = Vec::new();
                            let mut unique_cands = Vec::new();
                            for cand in &candidates {
                                if !unique_texts.contains(&cand.text) {
                                    unique_texts.push(cand.text.clone());
                                    unique_cands.push(cand.clone());
                                } else {
                                    if let Some(pos) = unique_texts.iter().position(|x| x == &cand.text) {
                                        for &idx in &cand.target_indices {
                                            if !unique_cands[pos].target_indices.contains(&idx) {
                                                unique_cands[pos].target_indices.push(idx);
                                            }
                                        }
                                    }
                                }
                            }

                            if unique_cands.len() == 1 {
                                let winner = unique_cands[0].clone();
                                for &t_idx in &winner.target_indices {
                                    let (context_name, _, _) = &dynamic_target_items[t_idx];
                                    emit_term(&format!("    👑 [WINNER] '{}' -> {} (Score: {:.4})", winner.text, context_name.replace("_", " "), winner.score));
                                }
                                final_spans.push(winner);
                            } else {
                                emit_term(&format!("    ⚖️ [SCORE NMS] 중첩 충돌 발생! 벡터 점수 기반 판별: {:?}", unique_texts));
                                
                                // 점수가 가장 높은 후보를 승자로 선택 (동점일 경우 텍스트 길이가 긴 것을 선호)
                                let mut sorted_cands = unique_cands.clone();
                                sorted_cands.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal).then(b.text.len().cmp(&a.text.len())));
                                
                                let winner = sorted_cands[0].clone();
                                // winner가 원래 unique_cands에서 몇 번째인지 찾기 (로그 출력을 위해)
                                let winner_idx = unique_cands.iter().position(|x| x.text == winner.text).unwrap_or(0);
                                
                                for &t_idx in &winner.target_indices {
                                    let (context_name, _, _) = &dynamic_target_items[t_idx];
                                    emit_term(&format!("    👑 [WINNER-SCORE] '{}' -> {} (Score: {:.4})", winner.text, context_name.replace("_", " "), winner.score));
                                }
                                
                                for (i, cand) in unique_cands.iter().enumerate() {
                                    if i != winner_idx {
                                        for &t_idx in &cand.target_indices {
                                            let (context_name, _, _) = &dynamic_target_items[t_idx];
                                            emit_term(&format!("    💀 [DEFEAT-SCORE] '{}' -> {} (Rejected, Score: {:.4})", cand.text, context_name.replace("_", " "), cand.score));
                                        }
                                    }
                                }
                                
                                final_spans.push(winner);
                            }
                        }

                        remaining_spans = next_remaining;
                    }

                    // 🌟 [추가] 4.5 인접 청크 동적 스팬 확장 (Dynamic Span Expansion)
                    // 물리적으로 인접한 청크들이 만들어낼 수 있는 모든 결합 경우의 수를 파생시켜 후보군에 추가합니다.
                    emit_term("  🔗 [PASS 3.5: GAP BRIDGING] Dynamic Span Expansion for adjacent winner chunks...");
                    final_spans.sort_by(|a, b| a.line_idx.cmp(&b.line_idx).then(a.start.cmp(&b.start)));
                    
                    let mut expanded_spans: Vec<ChunkSpan> = final_spans.clone(); // 기존 단일 조각들 보존
                    
                    // 연속된 청크들을 그룹화
                    let mut contiguous_groups: Vec<Vec<ChunkSpan>> = Vec::new();
                    let mut current_group: Vec<ChunkSpan> = Vec::new();
                    
                    for span in &final_spans {
                        if let Some(last) = current_group.last() {
                            // 🌟 [CRITICAL FIX] 무분별한 이종 카테고리(예: 이름+주민번호) 결합을 막기 위해,
                            // 타겟 도메인(target_indices)에 최소 1개 이상의 교집합이 존재할 때만 결합을 허용합니다.
                            let has_common_target = last.target_indices.iter().any(|idx| span.target_indices.contains(idx));
                            
                            if last.line_idx == span.line_idx && last.end == span.start && has_common_target {
                                // 🌟 두 청크 중 최소 하나는 '단어 1개'로 구성된 경우에만 연속성을 인정하여 과잉 그룹화 방지
                                let last_word_count = last.end - last.start;
                                let span_word_count = span.end - span.start;
                                
                                if last_word_count == 1 || span_word_count == 1 {
                                    current_group.push(span.clone());
                                    continue;
                                }
                            }
                        }
                        if !current_group.is_empty() {
                            contiguous_groups.push(current_group.clone());
                        }
                        current_group = vec![span.clone()];
                    }
                    if !current_group.is_empty() {
                        contiguous_groups.push(current_group);
                    }
                    
                    // 🌟 [CRITICAL FIX] 결합 파생 조각만 별도로 격리하여 보관합니다.
                    let mut expanded_only_spans: Vec<ChunkSpan> = Vec::new(); 

                    // 🌟 [CRITICAL FIX] 2조각뿐만 아니라 최대 3조각(단어)까지 확장하여 고유명사가 파편화되는 것을 방지합니다.
                    for group in contiguous_groups {
                        let n = group.len();
                        if n >= 2 {
                            let max_len = n.min(3); // 최대 3조각 조합 허용
                            for len in 2..=max_len {
                                for start_idx in 0..=(n - len) {
                                    let sub_group = &group[start_idx..(start_idx + len)];
                                    
                                    // 🌟 [CRITICAL FIX] 단순 공백 결합이 아닌 원본 텍스트 기반 정규식 발췌 로직 적용 (PASS 3.5)
                                    // 결합되는 조각들 사이의 원래 띄어쓰기(다중 공백 등) 간격을 그대로 가져옵니다.
                                    let combined_words: Vec<&str> = sub_group.iter().map(|s| s.text.as_str()).collect();
                                    let mut regex_pattern = String::new();
                                    for (i, word) in combined_words.iter().enumerate() {
                                        let escaped_word = regex::escape(word).replace("\\ ", r"\s+");
                                        if i > 0 { regex_pattern.push_str(r"\s+"); }
                                        regex_pattern.push_str(&escaped_word);
                                    }
                                    
                                    let line_text = &lines[sub_group[0].line_idx];
                                    let combined_text = if let Ok(re) = regex::Regex::new(&regex_pattern) {
                                        if let Some(mat) = re.find(line_text) {
                                            mat.as_str().to_string()
                                        } else {
                                            combined_words.join(" ")
                                        }
                                    } else {
                                        combined_words.join(" ")
                                    };
                                    
                                    // 🌟 파생된 거대 조각은 결합된 모든 조각들의 타겟(카테고리)을 합집합으로 가집니다.
                                    let mut combined_targets = Vec::new();
                                    for s in sub_group {
                                        combined_targets.extend(s.target_indices.clone());
                                    }
                                    combined_targets.sort();
                                    combined_targets.dedup();

                                    // 🌟 [수정] Granite 임베딩을 호출하여 결합된 텍스트의 진짜 문맥 점수를 재계산합니다.
                                    let combined_emb = model.get_embedding(combined_text.clone()).await.unwrap_or_else(|_| vec![0.0; 384]);
                                    let mut real_max_score = 0.0_f32;
                                    
                                    for &t_idx in &combined_targets {
                                        let b_score = cosine_similarity(&combined_emb, &target_biases_embs[t_idx]);
                                        let p_score = cosine_similarity(&combined_emb, &target_prejs_embs[t_idx]);
                                        
                                        // 🌟 [CRITICAL FIX] "연임에 성공한", "바르셀로나에 맞서" 등 서술어/동사가 섞인 무의미한 확장을 원천 차단하기 위해 
                                        // 결합된 텍스트에 대한 verb_penalty 가중치를 대폭 상향(0.10 -> 0.25)하여 NMS 경쟁에서 패배하도록 유도합니다.
                                        let v_sim = cosine_similarity(&combined_emb, &verb_emb);
                                        let verb_penalty = v_sim * 0.25; 
                                        
                                        let mut title_bonus = 0.0;
                                        let t_sim = cosine_similarity(&combined_emb, &title_emb);
                                        if t_sim > 0.0 { title_bonus = t_sim * 0.15; }
                                        
                                        let target_score = b_score - (p_score * 0.7) - verb_penalty + title_bonus;
                                        if target_score > real_max_score {
                                            real_max_score = target_score;
                                        }
                                    }

                                    // 🌟 [CRITICAL FIX] 동사 패널티로 인해 점수가 폭락한 경우(예: '연임에 성공한'), 
                                    // 기존 조각의 점수를 무조건 상속(90%)받지 못하도록 방어 코드를 추가합니다.
                                    // 서술어 유사도(v_sim)가 특정 수치를 초과하면 상속을 완전히 끊어버려 탈락(Drop)을 유도합니다.
                                    let v_sim_for_drop = cosine_similarity(&combined_emb, &verb_emb);
                                    let final_score = if v_sim_for_drop > 0.40 {
                                        real_max_score // 동사성이 강하면 상속 없이 순수 계산 점수만 반영 (NMS 탈락 유도)
                                    } else {
                                        let inherited_max = sub_group.iter().map(|s| s.score).fold(0.0, f32::max);
                                        real_max_score.max(inherited_max * 0.9)
                                    };
                                    
                                    // 0.3 커트라인을 넘지 못하면 아예 후보군에 넣지 않음
                                    if final_score > 0.3 {
                                        let new_span = ChunkSpan {
                                            line_idx: sub_group[0].line_idx,
                                            start: sub_group[0].start,
                                            end: sub_group.last().unwrap().end,
                                            text: combined_text.clone(),
                                            target_indices: combined_targets,
                                            score: final_score,
                                        };
                                        
                                        emit_term(&format!("    🤝 [EXPANDED] {}조각 결합 파생: '{}' (재평가 스코어: {:.4})", len, combined_text, final_score));
                                        expanded_only_spans.push(new_span);
                                    } else {
                                        emit_term(&format!("    🚫 [EXPANDED DROP] 동사/서술어 패널티로 결합 기각: '{}'", combined_text));
                                    }
                                }
                            }
                        }
                    }
                    
                    // 🌟 [추가] 4.6 EXPANDED 결합 조각과 기존 조각 간의 중복(Overlap) 제거 (2차 NMS)
                    emit_term("  🧹 [PASS 3.6: EXPANDED NMS CLEANUP] Resolving Overlaps between Expanded and Original chunks...");
                    
                    let mut all_pool = final_spans.clone();
                    all_pool.extend(expanded_only_spans);

                    // 점수 기준 내림차순 정렬 (동점이면 텍스트 길이가 긴 것 우선)
                    all_pool.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal).then(b.text.len().cmp(&a.text.len())));

                    let mut cleaned_spans: Vec<ChunkSpan> = Vec::new();

                    for cand in all_pool {
                        let mut is_overlapped = false;
                        for selected in &cleaned_spans {
                            if cand.line_idx == selected.line_idx {
                                if cand.start < selected.end && cand.end > selected.start {
                                    is_overlapped = true;
                                    break;
                                }
                            }
                        }

                        if !is_overlapped {
                            cleaned_spans.push(cand);
                        } else {
                            if let Some(overlap_target) = cleaned_spans.iter().find(|s| s.line_idx == cand.line_idx && cand.start < s.end && cand.end > s.start) {
                                emit_term(&format!("    🗑️ [CLEANUP] 조각 기각됨 (중첩 흡수): '{}' (Score: {:.4}) -> 승자: '{}' (Score: {:.4})", cand.text, cand.score, overlap_target.text, overlap_target.score));
                            }
                        }
                    }
                    
                    let final_spans = cleaned_spans;

                    // 🌟 5. NMS 승자들을 바탕으로 매칭된 라인 및 valid_targets 재조립
                    // 구조: (target_name, base_target, target_desc, bias_keyword, prejudice, is_phase2, specific_line_text, specific_candidate_text)
                    let mut valid_targets: Vec<(String, String, String, String, String, bool, String, String)> = Vec::new(); // 🌟 PUG 한 줄, 단일 키워드 1:1 맵핑

                    for span in &final_spans {
                        if span.score >= 0.10 {
                            let specific_candidate = span.text.clone(); // 🌟 이게 바로 NMS WINNER / EXPANDED 값입니다!

                            // 🌟 [전체 PUG 각 줄 내용 루프 뎁스 추가]
                            for (line_idx, line_content) in lines.iter().enumerate() {
                                if line_content.contains(&specific_candidate) {
                                    let specific_line = line_content.clone();

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
                        }
                    }

                    emit_term("[EXTRACTION] 🧠 2차 패스 벡터 유사도 검증 완료. 추론 준비...");
                    
                    emit_term("[EXTRACTION] ✅ LLM 2차 추론 루프 시작.");

                    let total_valid = valid_targets.len();

                    let mut p_idx = 0;
                    let mut phase2_companies: Vec<String> = Vec::new();
                    let mut phase2_executed = false; // Phase 2 진입 플래그
                    
                    // 🌟 [추가] 이미 성공적으로 마스킹된 NMS 후보를 기록하여 동의어 트랙에서 건너뛰게 하는 장부
                    let mut fully_masked_candidates: std::collections::HashSet<String> = std::collections::HashSet::new();
                    
                    // 🌟 [추가] 할루시네이션으로 판명된 단어를 기록하여 이후 스킵하게 하는 장부
                    let mut hallucinated_candidates: std::collections::HashSet<String> = std::collections::HashSet::new();

                    // 🌟 각 속성별로 매칭이 안 될 때까지 무한 반복(loop)하며 순차적으로 처리합니다.
                    while p_idx < valid_targets.len() {
                        if cancellation_token.load(Ordering::Relaxed) { break; }

                        // 만약 valid_targets.len()을 넘어섰다면 진짜 끝
                        if p_idx >= valid_targets.len() { break; }
                        let (target_name, base_target, target_item, target_bias, target_prejudice, is_phase2, specific_line, mut specific_candidate) = valid_targets[p_idx].clone();
                        p_idx += 1;

                        // 🌟 [사전 차단] 무거운 Stanza NLP 연산에 들어가기 전에, 이미 마스킹이 완료되었거나 할루시네이션으로 판명된 타겟은 여기서 즉시 쳐냅니다!
                        if !specific_candidate.is_empty() && fully_masked_candidates.contains(&specific_candidate) {
                            emit_term(&format!("[DEBUG] 이미 다른 동의어 트랙에서 마스킹 완료된 타겟입니다. 스킵합니다: '{}'", specific_candidate));
                            continue;
                        }
                        if !specific_candidate.is_empty() && hallucinated_candidates.contains(&specific_candidate) {
                            emit_term(&format!("[DEBUG] 독성 단어 명부에 등재된 타겟입니다. 연쇄 파기(Cascade Cancellation)를 적용하여 해당 트랙을 즉시 스킵합니다: '{}'", specific_candidate));
                            continue;
                        }

                        // 🌟 [STAGE 2.5] Stanza 정제 및 전처리 (Post-NMS Trimming - 미시적 정밀 타격)
                        if let Some(stanza) = &mut stanza_pipeline {
                            if !specific_candidate.is_empty() {
                                // 🌟 [원본 보존] 문장 전체 검색을 위해 정제 전의 원본 타겟을 보존합니다.
                                let original_candidate = specific_candidate.clone();

                                // 🌟 [Plan C] 1. 범용 특수문자 완벽 분리 및 제거 (다국어 공통)
                                // 하드코딩된 기호 배열 대신, 알파벳/숫자/공백이 아닌 모든 특수문자를 범용적으로 찾아 공백으로 분리합니다.
                                let mut eval_target = String::new();
                                for c in specific_candidate.chars() {
                                    if c.is_alphanumeric() || c.is_whitespace() || c == '-' {
                                        eval_target.push(c);
                                    } else {
                                        eval_target.push(' ');
                                    }
                                }
                                eval_target = eval_target.split_whitespace().collect::<Vec<_>>().join(" ");

                                // 🌟 정제된 텍스트를 specific_candidate에 다시 덮어씌워 LLM 프롬프트에도 깨끗한 값을 전달합니다.
                                specific_candidate = eval_target.clone();

                                // 🌟 [CRITICAL FIX] 특수문자 제거 후 힌트 단어가 완전히 사라진 경우 (예: "."), 무의미한 LLM 추론 및 ONNX 에러를 방지하기 위해 트랙을 즉시 스킵합니다.
                                if specific_candidate.trim().is_empty() {
                                    emit_term("[STANZA] ⚠️ 특수문자 정제 후 힌트 단어가 완전히 사라졌습니다. 트랙을 스킵합니다.");
                                    continue;
                                }

                                // 🌟 [효율성 개선] 정제된 specific_candidate가 이미 마스킹 장부나 할루시네이션 장부에 있는지 Stanza 연산 직전에 한 번 더 검사하여 불필요한 모델 추론을 차단합니다.
                                if fully_masked_candidates.contains(&specific_candidate) {
                                    emit_term(&format!("[DEBUG] 특수기호 정제 후, 이미 마스킹 완료된 타겟으로 확인되었습니다. Stanza 연산을 스킵합니다: '{}'", specific_candidate));
                                    continue;
                                }
                                if hallucinated_candidates.contains(&specific_candidate) {
                                    emit_term(&format!("[DEBUG] 특수기호 정제 후, 독성 단어 명부에 등재된 타겟으로 확인되었습니다. 연쇄 파기(Cascade Cancellation)를 적용하여 즉시 스킵합니다: '{}'", specific_candidate));
                                    continue;
                                }

                                // 🌟 [1차 형태소 분리] 문맥 검색을 통한 안전한 폴백 매핑
                                let cand_byte_idx_opt = specific_line.find(&original_candidate);
                                let use_context = cand_byte_idx_opt.is_some();
                                let text_to_analyze = if use_context { specific_line.clone() } else { eval_target.clone() };

                                let mut ext_words_string: Vec<String> = Vec::new();
                                let mut word_spans: Vec<(String, usize, usize)> = Vec::new();
                                
                                let chars: Vec<char> = text_to_analyze.chars().collect();
                                
                                if !chars.is_empty() {
                                    let seq_len = chars.len();
                                    let mut char_ids = Vec::with_capacity(seq_len);
                                    for c in &chars {
                                        let id = *stanza.preprocessor.char_vocab.get(c).unwrap_or(&stanza.preprocessor.char_unk_id);
                                        char_ids.push(id);
                                    }
                                    
                                    if let Ok(char_tensor) = ndarray::Array2::from_shape_vec((1, seq_len), char_ids) {
                                        let char_features = ndarray::Array3::<i64>::zeros((1, seq_len, 5));
                                        let seq_lengths = ndarray::Array1::<i64>::from_vec(vec![seq_len as i64]);
                                        
                                        let inputs = vec![
                                            char_tensor.into_dyn(),
                                            char_features.into_dyn(),
                                            seq_lengths.into_dyn(),
                                        ];
                                        
                                        match stanza.tokenize_session.run::<'_, '_, '_, i64, f32, _>(inputs) {
                                            Ok(outputs) => {
                                                let output_tensor = &outputs[0];
                                                let shape = output_tensor.shape();
                                                let num_classes = *shape.last().unwrap() as usize;
                                                let is_3d = shape.len() == 3;
                                                
                                                let mut current_word = String::new();
                                                let mut word_start = 0;
                                                
                                                for i in 0..seq_len {
                                                    current_word.push(chars[i]);
                                                    
                                                    let mut max_val = std::f32::MIN;
                                                    let mut max_idx = 0;
                                                    for c_idx in 0..num_classes {
                                                        let val = if is_3d {
                                                            output_tensor[[0, i, c_idx]]
                                                        } else {
                                                            output_tensor[[i, c_idx]]
                                                        };
                                                        if val > max_val { max_val = val; max_idx = c_idx; }
                                                    }
                                                    
                                                    if max_idx > 0 || i == seq_len - 1 {
                                                        let token_str = current_word.trim().to_string();
                                                        if !token_str.is_empty() {
                                                            word_spans.push((token_str.clone(), word_start, i + 1));
                                                            ext_words_string.push(token_str);
                                                        }
                                                        current_word.clear();
                                                        word_start = i + 1;
                                                    }
                                                }
                                            },
                                            Err(_e) => {}
                                        }
                                    }
                                }
                                
                                if ext_words_string.is_empty() {
                                    ext_words_string = Vec::new();
                                    let chars: Vec<char> = text_to_analyze.chars().collect();
                                    let mut in_word = false;
                                    let mut word_start = 0;
                                    let mut current_word = String::new();
                                    for (i, &c) in chars.iter().enumerate() {
                                        if c.is_whitespace() {
                                            if in_word {
                                                word_spans.push((current_word.clone(), word_start, i));
                                                ext_words_string.push(current_word.clone());
                                                in_word = false;
                                                current_word.clear();
                                            }
                                        } else {
                                            if !in_word {
                                                in_word = true;
                                                word_start = i;
                                            }
                                            current_word.push(c);
                                        }
                                    }
                                    if in_word {
                                        word_spans.push((current_word.clone(), word_start, chars.len()));
                                        ext_words_string.push(current_word.clone());
                                    }
                                }
                                
                                let ext_words: Vec<&str> = ext_words_string.iter().map(|s| s.as_str()).collect();

                                let mut chunk_size = ext_words.len();
                                for input_meta in &stanza.pos_session.inputs {
                                    let dims = &input_meta.dimensions;
                                    if dims.len() == 2 && dims.get(1) == Some(&Some(32)) {
                                        if let Some(&Some(fixed_seq)) = dims.get(0) {
                                            chunk_size = fixed_seq as usize;
                                        }
                                    }
                                }
                                if chunk_size == 0 { chunk_size = ext_words.len(); }

                                let mut all_tags = Vec::new();
                                let mut run_success = true;
                                let mut run_error = String::new();

                                if chunk_size > 0 && !ext_words.is_empty() {
                                    for chunk in ext_words.chunks(chunk_size) {
                                        let mut padded_chunk = chunk.to_vec();
                                        let valid_len = chunk.len();
                                        while padded_chunk.len() < chunk_size {
                                            padded_chunk.push("<pad>");
                                        }

                                        match stanza.preprocessor.encode_to_tensor(&padded_chunk, &stanza.pos_session) {
                                            Ok(inputs) => {
                                                match stanza.pos_session.run::<'_, '_, '_, i64, f32, _>(inputs) {
                                                    Ok(outputs) => {
                                                        let output_tensor = &outputs[0];
                                                        let shape = output_tensor.shape();
                                                        if shape.len() == 3 {
                                                            let num_classes = shape[2] as usize;
                                                            for i in 0..valid_len {
                                                                let mut max_val = std::f32::MIN;
                                                                let mut max_idx = 0;
                                                                for c in 0..num_classes {
                                                                    let val = output_tensor[[0, i, c]];
                                                                    if val > max_val { max_val = val; max_idx = c; }
                                                                }
                                                                all_tags.push(max_idx as i64);
                                                            }
                                                        } else if shape.len() == 2 {
                                                            let num_classes = shape[1] as usize;
                                                            for i in 0..valid_len {
                                                                let mut max_val = std::f32::MIN;
                                                                let mut max_idx = 0;
                                                                for c in 0..num_classes {
                                                                    let val = output_tensor[[i, c]];
                                                                    if val > max_val { max_val = val; max_idx = c; }
                                                                }
                                                                all_tags.push(max_idx as i64);
                                                            }
                                                        }
                                                    },
                                                    Err(e) => { run_success = false; run_error = format!("{:?}", e); break; }
                                                }
                                            },
                                            Err(e) => { run_success = false; run_error = format!("{:?}", e); break; }
                                        }
                                    }
                                } else {
                                    run_success = false;
                                    run_error = "Invalid sequence length for ONNX inputs".to_string();
                                }

                                let simulated_result: Result<Vec<i64>, String> = if run_success { Ok(all_tags) } else { Err(run_error) };

                                if true {
                                    match simulated_result {
                                        Ok(tags) => {
                                            let tag_names: Vec<&str> = tags.into_iter()
                                                    .map(|id| stanza.preprocessor.upos_vocab.get(id as usize).map(|s| s.as_str()).unwrap_or("X"))
                                                    .collect();
                                                
                                            // 🌟 [CRITICAL FIX] 추출 단어가 문맥에 정확히 존재할 때만 오프셋 매핑을 수행하여 엉뚱한 마커 분할을 방지합니다.
                                            let mut candidate_words = Vec::new();
                                            let mut candidate_tags = Vec::new();
                                            
                                            if use_context {
                                                let cand_byte_idx = cand_byte_idx_opt.unwrap();
                                                let cand_start_char = specific_line[..cand_byte_idx].chars().count();
                                                let cand_end_char = cand_start_char + original_candidate.chars().count();

                                                for (w_idx, (w_str, w_start, w_end)) in word_spans.iter().enumerate() {
                                                    if *w_start < cand_end_char && *w_end > cand_start_char {
                                                        candidate_words.push(w_str.clone());
                                                        candidate_tags.push(if w_idx < tag_names.len() { tag_names[w_idx] } else { "X" });
                                                    }
                                                }
                                            }

                                            if candidate_words.is_empty() {
                                                if use_context {
                                                    candidate_words = vec![eval_target.clone()];
                                                    candidate_tags = vec!["NOUN"];
                                                } else {
                                                    candidate_words = ext_words_string.clone();
                                                    candidate_tags = tag_names.clone();
                                                }
                                            }

                                            emit_term(&format!("[STANZA] 문맥 기반 형태소 분리 완료 '{}' -> {:?}", eval_target, candidate_tags));

                                            let invalid_tags = ["PUNCT", "SYM"];
                                            let all_invalid = candidate_tags.iter().all(|&t| invalid_tags.contains(&t));
                                            // 🌟 [CRITICAL FIX] 개체명(Named Entity) 범주에 속할 수 없는 "VERB"를 허용 목록에서 제거하여 순수 동사의 LLM 진입을 원천 차단합니다.
                                            let has_noun_or_oov = candidate_tags.iter().any(|&t| t == "NOUN" || t == "PROPN" || t == "NUM" || t == "X" || t == "DET" || t == "CCONJ" || t == "PRON");
                                            
                                            let cand_char_count = specific_candidate.chars().filter(|c| !c.is_whitespace()).count();
                                            
                                            // 🌟 [CRITICAL FIX] 식별번호, 연락처 등 기호(-)가 필수적으로 포함된 고유 형식 도메인은 무조건 구제(Bypass)합니다.
                                            let is_id_domain = base_target == "national_id" || base_target == "contact_number" || base_target == "email";
                                            let rescue_oov = (cand_char_count >= 2 && all_invalid) || is_id_domain;

                                            if !rescue_oov && (all_invalid || !has_noun_or_oov) {
                                                emit_term(&format!("[STANZA] 💀 순수 수식어/조사/동사/기호 감지 (Plan B). 강제 기각: '{}'", specific_candidate));
                                                hallucinated_candidates.insert(specific_candidate.clone());
                                            } else {
                                                if rescue_oov {
                                                    emit_term(&format!("[STANZA] 🚑 OOV 및 식별번호 구제 발동 (Plan B 우회): '{}' ({} 항목). 강제 기각 및 절단을 면제합니다.", specific_candidate, base_target));
                                                }
                                                let mut trimmed_words = candidate_words.clone();
                                                let mut valid_tags_clone = candidate_tags.clone();
                                                let mut is_trimmed = false;

                                                // 🌟 [CRITICAL FIX] PUG 파이프(|) 등 순수 기호가 독립 단어로 분리되어 무한 루프를 유발하는 현상 차단
                                                let mut clean_words = Vec::new();
                                                let mut clean_tags = Vec::new();
                                                for (i, w) in trimmed_words.iter().enumerate() {
                                                    let is_pure_symbol = w.chars().all(|c| !c.is_alphanumeric());
                                                    if !is_pure_symbol {
                                                        clean_words.push(w.clone());
                                                        clean_tags.push(valid_tags_clone[i]);
                                                    } else {
                                                        is_trimmed = true; // 기호를 제거했으므로 trim 처리
                                                    }
                                                }
                                                if !clean_words.is_empty() {
                                                    trimmed_words = clean_words;
                                                    valid_tags_clone = clean_tags;
                                                }

                                                // 🌟 [CRITICAL FIX] 추출 단어 앞부분에 붙은 수식어(관형사, 부사, 접속사 등)를 잘라내는 머리 절단 로직을 추가합니다. ('전 소속팀' 등 방어)
                                                let front_drop_tags = ["DET", "ADJ", "ADV", "PUNCT", "CCONJ", "SCONJ", "PART", "ADP"];
                                                if !is_id_domain {
                                                    while let Some(first_tag) = valid_tags_clone.first() {
                                                        if front_drop_tags.contains(first_tag) && trimmed_words.len() > 1 {
                                                            // 🌟 [예외 추가] 만약 잘려나가는 단어가 영어/숫자 등 고유명사의 특징을 띤다면 보호합니다.
                                                            let first_word = &trimmed_words[0];
                                                            if first_word.chars().any(|c| c.is_ascii_alphanumeric()) {
                                                                break;
                                                            }
                                                            trimmed_words.remove(0);
                                                            valid_tags_clone.remove(0);
                                                            is_trimmed = true;
                                                        } else {
                                                            break;
                                                        }
                                                    }
                                                }

                                                // 🌟 [CRITICAL FIX] 추출 단어 끝에 꼬리로 잘못 붙은 동사(VERB), 형용사(ADJ), 부사(ADV)도 잘라내도록 꼬리 절단 태그를 대폭 보강합니다.
                                                let tail_drop_tags = ["ADP", "PUNCT", "PART", "SCONJ", "CCONJ", "DET", "VERB", "ADJ", "ADV"];
                                                if !is_id_domain {
                                                    while let Some(last_tag) = valid_tags_clone.last() {
                                                        if tail_drop_tags.contains(last_tag) && trimmed_words.len() > 1 {
                                                            // 🌟 [예외 추가] 만약 잘려나가는 단어가 영어/숫자 등 고유명사의 특징을 띤다면 보호합니다.
                                                            let last_word = trimmed_words.last().unwrap();
                                                            if last_word.chars().any(|c| c.is_ascii_alphanumeric()) {
                                                                break;
                                                            }
                                                            trimmed_words.pop();
                                                            valid_tags_clone.pop();
                                                            is_trimmed = true;
                                                        } else {
                                                            break;
                                                        }
                                                    }
                                                }
                                                
                                                let mut queue_split = false;
                                                if trimmed_words.len() >= 2 {
                                                    // 🌟 [CRITICAL FIX] 1글자 단어가 진짜 알파벳/한글/숫자일 때만 분할하되, 의미 있는 1글자 명사(NOUN, PROPN)나 숫자(NUM), 수식어가 분할 큐를 찢는 현상을 원천 방지합니다.
                                                    let protected_tags = ["ADJ", "ADV", "DET", "PART", "PRON", "NOUN", "PROPN", "NUM"];
                                                    if trimmed_words.iter().enumerate().any(|(w_idx, w)| {
                                                        let c_count = w.chars().filter(|c| !c.is_whitespace()).count();
                                                        let is_valid_char = w.chars().any(|c| c.is_alphanumeric());
                                                        let tag = valid_tags_clone.get(w_idx).copied().unwrap_or("X");
                                                        c_count == 1 && is_valid_char && !protected_tags.contains(&tag)
                                                    }) {
                                                        queue_split = true;
                                                    }
                                                }

                                                if queue_split {
                                                    let parts_display = trimmed_words.join("', '");
                                                    emit_term(&format!("[STANZA] ✂️ 1글자 단어 포함 감지. '{}' 로 분할하여 추론 큐에 독립적으로 추가합니다.", parts_display));
                                                    
                                                    for part in &trimmed_words {
                                                        let p_char_count = part.chars().filter(|c| !c.is_whitespace()).count();
                                                        if p_char_count >= 2 {
                                                            let mut clone = valid_targets[p_idx - 1].clone();
                                                            clone.7 = part.to_string();
                                                            valid_targets.push(clone);
                                                        } else {
                                                            emit_term(&format!("[STANZA] 🚫 1글자 분할 조각 기각: '{}' (무의미한 단어 확산 방지)", part));
                                                        }
                                                    }
                                                    
                                                    continue;
                                                } else if is_trimmed {
                                                    let join_str = " ";
                                                    let trimmed_candidate = trimmed_words.join(join_str);
                                                    
                                                    // 🌟 [수정된 로직] 절단 결과가 1글자 이하라면 무의미한 단어로 간주하고 해당 트랙을 즉시 기각(스킵)합니다.
                                                    let char_count = trimmed_candidate.chars().filter(|c| !c.is_whitespace()).count();
                                                    if char_count <= 1 {
                                                        emit_term(&format!("[STANZA] ✂️ 스마트 절단 결과 1글자만 남음 ('{}' -> '{}'). 무의미한 단어로 간주하여 즉시 스킵합니다.", specific_candidate, trimmed_candidate));
                                                        continue;
                                                    }
                                                    
                                                    emit_term(&format!("[STANZA] ✂️ 1차 형태소 분리 후 스마트 머리/꼬리 절단 ({}): '{}' -> '{}'", local_language, specific_candidate, trimmed_candidate));
                                                    specific_candidate = trimmed_candidate;
                                                } else {
                                                    if specific_candidate != eval_target {
                                                        specific_candidate = eval_target;
                                                    }
                                                }
                                            }
                                        },
                                        Err(e) => {
                                            emit_term(&format!("[STANZA] ⚠️ POS ONNX 추론 연산 실패 원인: {:?}", e));
                                        }
                                    }
                                }
                            }
                        }

                        // 🌟 [추가] 이미 다른 동의어 트랙에서 마스킹을 마친 후보(specific_candidate)인지 확인하고 건너뜁니다 (접근법 A)
                        if !specific_candidate.is_empty() && fully_masked_candidates.contains(&specific_candidate) {
                            emit_term(&format!("[DEBUG] 이미 다른 동의어 트랙에서 마스킹 완료된 타겟입니다. 스킵합니다: '{}'", specific_candidate));
                            continue;
                        }

                        // 🌟 [연쇄 파기(Cascade Cancellation)] 할루시네이션(본문 없음, 동사/서술어 등)으로 등재된 독성 단어라면 파생 트랙 전체를 시작도 안 하고 즉시 폐기합니다.
                        if !specific_candidate.is_empty() && hallucinated_candidates.contains(&specific_candidate) {
                            emit_term(&format!("[DEBUG] 독성 단어 명부에 등재된 타겟입니다. 연쇄 파기(Cascade Cancellation)를 적용하여 해당 트랙을 즉시 스킵합니다: '{}'", specific_candidate));
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
                            }
                        }

                        let mut item_extract_count = 0;
                        
                        // 🌟 [CRITICAL FIX] Qwen3에게 전체 문서(masked_text)를 주지 않고, 
                        // 앞서 벡터 유사도로 0.10점을 넘긴(통과한) 정확히 PUG 한 줄만 제공합니다!
                        let mut matched_context = specific_line.clone();

                        // 🌟 [CRITICAL FIX] Phase 2에서 의도적으로 마커를 해제하여 LLM에게 이름을 재노출하던 로직을 삭제합니다.
                        // 한 번 마스킹된 데이터는 Phase 2에서도 철저히 가려져야 중복 추출(Overlap) 및 무한 루프가 발생하지 않습니다.
                        for (original_val, marker) in &replacement_history {
                            matched_context = matched_context.replace(original_val, marker);
                        }

                        // 🌟 [추가] Granite 추론 전, 힌트 단어(specific_candidate)가 PUG 컨텍스트(matched_context 또는 masked_text)에 남아있는지 사전 검증합니다.
                        // '조세 무리뉴'가 마스킹된 후 '조세'가 남지 않은 경우처럼, 이미 치환되어 사라진 단어에 대한 무의미한 LLM 추론을 원천 차단합니다.
                        if !specific_candidate.is_empty() {
                            let mut cand_exists = matched_context.contains(&specific_candidate) || masked_text.contains(&specific_candidate) || doc_title.contains(&specific_candidate) || doc_desc.contains(&specific_candidate);
                            
                            // 띄어쓰기 및 특수기호 변형을 고려한 교차 검증
                            if !cand_exists && specific_candidate.chars().count() >= 2 {
                                let no_space_cand: String = specific_candidate.chars().filter(|c| c.is_alphanumeric()).collect();
                                if no_space_cand.len() >= 2 && no_space_cand.len() <= 100 {
                                    let escaped_chars: Vec<String> = no_space_cand.chars().map(|c| regex::escape(&c.to_string())).collect();
                                    let regex_pattern = escaped_chars.join(r"[^\p{L}\p{N}_]*");
                                    if let Ok(re) = regex::Regex::new(&regex_pattern) {
                                        if re.is_match(&matched_context) || re.is_match(&masked_text) || re.is_match(&doc_title) || re.is_match(&doc_desc) {
                                            cand_exists = true;
                                        }
                                    }
                                }
                            }

                            if !cand_exists {
                                // 🌟 [자투리 단어 구제 로직] 전체가 증발했더라도, 일부만 치환되고 남은 자투리(예: '페레스 회장은' -> '페레스' 마스킹 후 '회장은' 잔존)가 있는지 확인하여 구제합니다.
                                let mut leftover_candidate = specific_candidate.clone();
                                for (orig, _) in &replacement_history {
                                    if leftover_candidate.contains(orig) {
                                        leftover_candidate = leftover_candidate.replace(orig, " ");
                                    }
                                }
                                
                                // 특수문자 제거 후 순수 단어만 추출
                                let mut clean_leftovers = String::new();
                                for c in leftover_candidate.chars() {
                                    if c.is_alphanumeric() {
                                        clean_leftovers.push(c);
                                    } else {
                                        clean_leftovers.push(' ');
                                    }
                                }
                                
                                let final_leftover = clean_leftovers.split_whitespace().collect::<Vec<_>>().join(" ");
                                
                                // 자투리 단어가 2글자 이상 살아남아 있고, 본문에 존재한다면 그것으로 교체하여 트랙을 구제함
                                let leftover_char_count = final_leftover.chars().filter(|c| !c.is_whitespace()).count();
                                if leftover_char_count >= 2 && (matched_context.contains(&final_leftover) || masked_text.contains(&final_leftover) || doc_title.contains(&final_leftover) || doc_desc.contains(&final_leftover)) {
                                    emit_term(&format!("[DEBUG] ♻️ 힌트 단어('{}') 중 일부가 마스킹되었으나, 자투리 단어('{}')가 남아있어 트랙을 구제합니다.", specific_candidate, final_leftover));
                                    fully_masked_candidates.insert(specific_candidate.clone()); // 원본은 마스킹된 것으로 처리
                                    specific_candidate = final_leftover.clone();
                                    cand_exists = true;
                                }
                            }

                            if !cand_exists {
                                emit_term(&format!("[DEBUG] ⚠️ 힌트 단어('{}')가 현재 마스킹된 PUG 본문/문맥에 더 이상 존재하지 않습니다(이미 다른 마커로 치환됨). 불필요한 LLM 추론을 스킵합니다.", specific_candidate));
                                fully_masked_candidates.insert(specific_candidate.clone()); // 🌟 이미 마스킹된 것으로 간주하여 이후 동의어 트랙에서도 스킵 유도
                                continue;
                            }
                        }

                        let input_keyword = if specific_candidate.is_empty() { target_bias.clone() } else { specific_candidate.clone() };

                        // 🌟 무거운 LLM(Granite) 추론을 완전히 스킵하고 NMS/Stanza 결과를 즉시 채택합니다.
                        for _ in 0..1 {
                            if cancellation_token.load(Ordering::Relaxed) { break; }

                            let payload = json!({ 
                                "task_id": task.id.clone(),
                                "category": format!("Masking ({}/{}) - Type {}", idx + 1, total, p_idx + 1), 
                                "summary": format!("Anonymizing {}...", target_item),
                                "spinner": "⠋"
                            });
                            let _ = app_handle.emit("extraction-progress", &payload);
                            crate::scheduler::log_task_progress(app_handle, &task.id, &payload);

                            // LLM 로직 우회 후 NMS 추출 단어 직접 채택
                            let mut extracted_val = specific_candidate.clone();
                            let parsed = serde_json::json!({
                                "is_target_mismatch": false
                            });

                            emit_term(&format!("[DEBUG-OOM] [{}] 항목 추출 완료 (LLM 우회 - NMS 👑 [WINNER/EXPANDED] 직접 채택: '{}')", base_target, extracted_val));

                            // 🌟 [CRITICAL FIX] 빈 값 반환 시 재시도 없이 해당 트랙을 즉시 종료합니다.
                            if extracted_val.is_empty() || extracted_val == "..." || extracted_val == "null" {
                                emit_term(&format!("[DEBUG] 빈 값 반환 감지. 재시도 없이 트랙을 종료합니다."));
                                break;
                            }

                            // 🌟 [CRITICAL FIX] 이름/회사명 등이 1글자로 잘려나와 마스킹되는 대참사 원천 차단
                            let ext_char_count = extracted_val.chars().filter(|c| !c.is_whitespace()).count();
                            if ext_char_count <= 1 {
                                emit_term(&format!("[DEBUG] 1글자 단어 추출 감지. 무의미한 과잉 마스킹 방지를 위해 강제 기각: '{}'", extracted_val));
                                hallucinated_candidates.insert(extracted_val.clone());
                                break;
                            }

                            // 🌟 [추가] 할루시네이션(환각)으로 이미 판명된 단어라면 즉시 스킵 (동의어 트랙 등에서 재등장 방지)
                            if hallucinated_candidates.contains(&extracted_val) {
                                emit_term(&format!("[DEBUG] 이미 다른 동의어 트랙에서 할루시네이션으로 판명된 단어입니다. 스킵합니다: '{}'", extracted_val));
                                break;
                            }

                            // 🌟 [추가] 추출된 단어(extracted_val)에 대해 한 번 더 NLP(Stanza) 검증 및 정제를 수행하여 꼬리(조사)를 자르거나 명사가 아닌 경우 기각합니다.
                            let mut nlp_rejected = false;
                            let mut already_masked_skip = false;
                            if let Some(stanza) = &mut stanza_pipeline {
                                // 🌟 [CRITICAL FIX] 이미 Stage 2.5에서 NLP 검증을 통과한 specific_candidate와 동일하다면, 
                                // 마커([___REDACTED_...])로 인해 문맥이 오염된 상태에서 Stanza가 동사(VERB)로 오판하는 것을 막기 위해 중복 검증을 안전하게 건너뜁니다!
                                if !extracted_val.is_empty() && extracted_val != specific_candidate {
                                    // 🌟 [원본 보존] 문장 전체 검색을 위해 정제 전의 원본 타겟을 보존합니다.
                                    let original_ext = extracted_val.clone();

                                    let mut eval_ext = String::new();
                                    
                                    // 1차 절단: 알파벳/한글/숫자 등 일반 문자가 아닌 모든 특수기호를 범용적으로 찾아 공백으로 치환하여 단어 결합 방지
                                    for c in extracted_val.chars() {
                                        if c.is_alphanumeric() || c.is_whitespace() || c == '-' {
                                            eval_ext.push(c);
                                        } else {
                                            eval_ext.push(' ');
                                        }
                                    }
                                    eval_ext = eval_ext.split_whitespace().collect::<Vec<_>>().join(" ");

                                    // 🌟 정제된 텍스트를 extracted_val에 덮어씌워 오탐률을 줄입니다.
                                    extracted_val = eval_ext.clone();

                                    // 🌟 [CRITICAL FIX] 특수문자 제거 후 추출 단어가 완전히 사라진 경우, NLP 검증을 우회하고 강제로 기각 상태를 활성화합니다.
                                    if extracted_val.trim().is_empty() {
                                        emit_term("[STANZA-EXT] ⚠️ 특수문자 정제 후 추출 단어가 완전히 사라졌습니다. 강제 기각 처리합니다.");
                                        nlp_rejected = true;
                                    } else if hallucinated_candidates.contains(&extracted_val) {
                                        emit_term(&format!("[DEBUG] 추출 단어 정제 후 독성 단어 명부 등재 확인. NLP 연산을 스킵하고 강제 기각합니다: '{}'", extracted_val));
                                        nlp_rejected = true;
                                    } else if fully_masked_candidates.contains(&extracted_val) {
                                        emit_term(&format!("[DEBUG] 추출 단어 정제 후 이미 마스킹 완료된 타겟으로 판명. 불필요한 후속 연산을 스킵하고 트랙을 즉시 종료합니다: '{}'", extracted_val));
                                        already_masked_skip = true;
                                    } else {
                                        // 🌟 [1차 형태소 분리] 문맥 검색을 통한 안전한 폴백 매핑
                                        let cand_byte_idx_opt = matched_context.find(&original_ext);
                                        let use_context = cand_byte_idx_opt.is_some();
                                        let text_to_analyze = if use_context { matched_context.clone() } else { eval_ext.clone() };

                                        let mut ext_words_string: Vec<String> = Vec::new();
                                        let mut word_spans: Vec<(String, usize, usize)> = Vec::new();
                                        
                                        let chars: Vec<char> = text_to_analyze.chars().collect();
                                        
                                        if !chars.is_empty() {
                                            let seq_len = chars.len();
                                            let mut char_ids = Vec::with_capacity(seq_len);
                                            for c in &chars {
                                                let id = *stanza.preprocessor.char_vocab.get(c).unwrap_or(&stanza.preprocessor.char_unk_id);
                                                char_ids.push(id);
                                            }
                                            
                                            if let Ok(char_tensor) = ndarray::Array2::from_shape_vec((1, seq_len), char_ids) {
                                                let char_features = ndarray::Array3::<i64>::zeros((1, seq_len, 5));
                                                let seq_lengths = ndarray::Array1::<i64>::from_vec(vec![seq_len as i64]);
                                                
                                                let inputs = vec![
                                                    char_tensor.into_dyn(),
                                                    char_features.into_dyn(),
                                                    seq_lengths.into_dyn(),
                                                ];
                                                
                                                match stanza.tokenize_session.run::<'_, '_, '_, i64, f32, _>(inputs) {
                                                    Ok(outputs) => {
                                                        let output_tensor = &outputs[0];
                                                        let shape = output_tensor.shape();
                                                        let num_classes = *shape.last().unwrap() as usize;
                                                        let is_3d = shape.len() == 3;
                                                        
                                                        let mut current_word = String::new();
                                                        let mut word_start = 0;
                                                        
                                                        for i in 0..seq_len {
                                                            current_word.push(chars[i]);
                                                            
                                                            let mut max_val = std::f32::MIN;
                                                            let mut max_idx = 0;
                                                            for c_idx in 0..num_classes {
                                                                let val = if is_3d {
                                                                    output_tensor[[0, i, c_idx]]
                                                                } else {
                                                                    output_tensor[[i, c_idx]]
                                                                };
                                                                if val > max_val { max_val = val; max_idx = c_idx; }
                                                            }
                                                            
                                                            if max_idx > 0 || i == seq_len - 1 {
                                                                let token_str = current_word.trim().to_string();
                                                                if !token_str.is_empty() {
                                                                    word_spans.push((token_str.clone(), word_start, i + 1));
                                                                    ext_words_string.push(token_str);
                                                                }
                                                                current_word.clear();
                                                                word_start = i + 1;
                                                            }
                                                        }
                                                    },
                                                    Err(_e) => {}
                                                }
                                            }
                                        }
                                        
                                        if ext_words_string.is_empty() {
                                            ext_words_string = Vec::new();
                                            let chars: Vec<char> = text_to_analyze.chars().collect();
                                            let mut in_word = false;
                                            let mut word_start = 0;
                                            let mut current_word = String::new();
                                            for (i, &c) in chars.iter().enumerate() {
                                                if c.is_whitespace() {
                                                    if in_word {
                                                        word_spans.push((current_word.clone(), word_start, i));
                                                        ext_words_string.push(current_word.clone());
                                                        in_word = false;
                                                        current_word.clear();
                                                    }
                                        } else {
                                            if !in_word {
                                                in_word = true;
                                                word_start = i;
                                            }
                                            current_word.push(c);
                                        }
                                    }
                                    if in_word {
                                        word_spans.push((current_word.clone(), word_start, chars.len()));
                                        ext_words_string.push(current_word.clone());
                                    }
                                }
                                        
                                        let ext_words: Vec<&str> = ext_words_string.iter().map(|s| s.as_str()).collect();

                                        let mut chunk_size = ext_words.len();
                                        for input_meta in &stanza.pos_session.inputs {
                                            let dims = &input_meta.dimensions;
                                            if dims.len() == 2 && dims.get(1) == Some(&Some(32)) {
                                                if let Some(&Some(fixed_seq)) = dims.get(0) {
                                                    chunk_size = fixed_seq as usize;
                                                }
                                            }
                                        }
                                        if chunk_size == 0 { chunk_size = ext_words.len(); }

                                        let mut all_tags = Vec::new();
                                        let mut run_success = true;
                                        let mut run_error = String::new();

                                        if chunk_size > 0 && !ext_words.is_empty() {
                                            for chunk in ext_words.chunks(chunk_size) {
                                                let mut padded_chunk = chunk.to_vec();
                                                let valid_len = chunk.len();
                                                while padded_chunk.len() < chunk_size {
                                                    padded_chunk.push("<pad>");
                                                }

                                                match stanza.preprocessor.encode_to_tensor(&padded_chunk, &stanza.pos_session) {
                                                    Ok(inputs) => {
                                                        match stanza.pos_session.run::<'_, '_, '_, i64, f32, _>(inputs) {
                                                            Ok(outputs) => {
                                                                let output_tensor = &outputs[0];
                                                                let shape = output_tensor.shape();
                                                                if shape.len() == 3 {
                                                                    let num_classes = shape[2] as usize;
                                                                    for i in 0..valid_len {
                                                                        let mut max_val = std::f32::MIN;
                                                                        let mut max_idx = 0;
                                                                        for c in 0..num_classes {
                                                                            let val = output_tensor[[0, i, c]];
                                                                            if val > max_val { max_val = val; max_idx = c; }
                                                                        }
                                                                        all_tags.push(max_idx as i64);
                                                                    }
                                                                } else if shape.len() == 2 {
                                                                    let num_classes = shape[1] as usize;
                                                                    for i in 0..valid_len {
                                                                        let mut max_val = std::f32::MIN;
                                                                        let mut max_idx = 0;
                                                                        for c in 0..num_classes {
                                                                            let val = output_tensor[[i, c]];
                                                                            if val > max_val { max_val = val; max_idx = c; }
                                                                        }
                                                                        all_tags.push(max_idx as i64);
                                                                    }
                                                                }
                                                            },
                                                            Err(e) => { run_success = false; run_error = format!("{:?}", e); break; }
                                                        }
                                                    },
                                                    Err(e) => { run_success = false; run_error = format!("{:?}", e); break; }
                                                }
                                            }
                                        } else {
                                            run_success = false;
                                            run_error = "Invalid sequence length for ONNX inputs".to_string();
                                        }

                                        let simulated_result: Result<Vec<i64>, String> = if run_success { Ok(all_tags) } else { Err(run_error) };

                                        if true {
                                            if let Ok(tags) = simulated_result {
                                                let tag_names: Vec<&str> = tags.into_iter()
                                                        .map(|id| stanza.preprocessor.upos_vocab.get(id as usize).map(|s| s.as_str()).unwrap_or("X"))
                                                        .collect();
                                                    
                                                // 🌟 [CRITICAL FIX] 추출 단어가 문맥에 정확히 존재할 때만 오프셋 매핑을 수행하여 엉뚱한 마커 분할을 방지합니다.
                                                let mut candidate_words = Vec::new();
                                                let mut candidate_tags = Vec::new();
                                                
                                                if use_context {
                                                    let cand_byte_idx = cand_byte_idx_opt.unwrap();
                                                    let cand_start_char = matched_context[..cand_byte_idx].chars().count();
                                                    let cand_end_char = cand_start_char + original_ext.chars().count();

                                                    for (w_idx, (w_str, w_start, w_end)) in word_spans.iter().enumerate() {
                                                        if *w_start < cand_end_char && *w_end > cand_start_char {
                                                            candidate_words.push(w_str.clone());
                                                            candidate_tags.push(if w_idx < tag_names.len() { tag_names[w_idx] } else { "X" });
                                                        }
                                                    }
                                                }

                                                if candidate_words.is_empty() {
                                                    if use_context {
                                                        candidate_words = vec![eval_ext.clone()];
                                                        candidate_tags = vec!["NOUN"];
                                                    } else {
                                                        candidate_words = ext_words_string.clone();
                                                        candidate_tags = tag_names.clone();
                                                    }
                                                }

                                                let invalid_tags = ["PUNCT", "SYM"];
                                                let all_invalid = candidate_tags.iter().all(|&t| invalid_tags.contains(&t));
                                                // 🌟 [CRITICAL FIX] 개체명(Named Entity) 범주에 속할 수 없는 "VERB"를 허용 목록에서 제거하여 순수 동사의 강제 기각을 유도합니다.
                                                let has_noun_or_oov = candidate_tags.iter().any(|&t| t == "NOUN" || t == "PROPN" || t == "NUM" || t == "X" || t == "DET" || t == "CCONJ" || t == "PRON");
                                                
                                                let ext_char_count = extracted_val.chars().filter(|c| !c.is_whitespace()).count();
                                                
                                                // 🌟 [CRITICAL FIX] 식별번호 등 기호가 섞인 형식 도메인은 강제 기각 면제 (Bypass)
                                                let is_id_domain = base_target == "national_id" || base_target == "contact_number" || base_target == "email";
                                                let rescue_oov = (ext_char_count >= 2 && all_invalid) || is_id_domain;

                                                if !rescue_oov && (all_invalid || !has_noun_or_oov) {
                                                    emit_term(&format!("[STANZA-EXT] 💀 순수 수식어/조사/동사/기호 감지. 추출단어 강제 기각: '{}'", extracted_val));
                                                    nlp_rejected = true;
                                                } else {
                                                    if rescue_oov {
                                                        emit_term(&format!("[STANZA-EXT] 🚑 OOV 및 식별번호 구제 발동 (Plan B 우회): '{}'. 강제 기각 및 절단을 면제합니다.", extracted_val));
                                                    }
                                                    let mut trimmed_words = candidate_words.clone();
                                                    let mut valid_tags_clone = candidate_tags.clone();
                                                    let mut is_trimmed = false;

                                                    // 🌟 [CRITICAL FIX] PUG 파이프(|) 등 순수 기호가 독립 단어로 분리되어 무한 루프를 유발하는 현상 차단
                                                    let mut clean_words = Vec::new();
                                                    let mut clean_tags = Vec::new();
                                                    for (i, w) in trimmed_words.iter().enumerate() {
                                                        let is_pure_symbol = w.chars().all(|c| !c.is_alphanumeric());
                                                        if !is_pure_symbol {
                                                            clean_words.push(w.clone());
                                                            clean_tags.push(valid_tags_clone[i]);
                                                        } else {
                                                            is_trimmed = true;
                                                        }
                                                    }
                                                    if !clean_words.is_empty() {
                                                        trimmed_words = clean_words;
                                                        valid_tags_clone = clean_tags;
                                                    }

                                                    // 🌟 [CRITICAL FIX] 추출 단어 앞부분에 붙은 수식어(관형사, 부사, 접속사 등)를 잘라내는 머리 절단 로직을 추가합니다. ('전 소속팀' 등 방어)
                                                    let front_drop_tags = ["DET", "ADJ", "ADV", "PUNCT", "CCONJ", "SCONJ", "PART", "ADP"];
                                                    if !is_id_domain {
                                                        while let Some(first_tag) = valid_tags_clone.first() {
                                                            if front_drop_tags.contains(first_tag) && trimmed_words.len() > 1 {
                                                                // 🌟 [예외 추가] 만약 잘려나가는 단어가 영어/숫자 등 고유명사의 특징을 띤다면 보호합니다.
                                                                let first_word = &trimmed_words[0];
                                                                if first_word.chars().any(|c| c.is_ascii_alphanumeric()) {
                                                                    break;
                                                                }
                                                                trimmed_words.remove(0);
                                                                valid_tags_clone.remove(0);
                                                                is_trimmed = true;
                                                            } else {
                                                                break;
                                                            }
                                                        }
                                                    }

                                                    // 🌟 [CRITICAL FIX] 추출 단어 끝에 꼬리로 잘못 붙은 동사(VERB), 형용사(ADJ), 부사(ADV)도 잘라내도록 꼬리 절단 태그를 대폭 보강합니다.
                                                    let tail_drop_tags = ["ADP", "PUNCT", "PART", "SCONJ", "CCONJ", "DET", "VERB", "ADJ", "ADV"];
                                                    if !is_id_domain {
                                                        while let Some(last_tag) = valid_tags_clone.last() {
                                                            if tail_drop_tags.contains(last_tag) && trimmed_words.len() > 1 {
                                                                // 🌟 [예외 추가] 만약 잘려나가는 단어가 영어/숫자 등 고유명사의 특징을 띤다면 보호합니다.
                                                                let last_word = trimmed_words.last().unwrap();
                                                                if last_word.chars().any(|c| c.is_ascii_alphanumeric()) {
                                                                    break;
                                                                }
                                                                trimmed_words.pop();
                                                                valid_tags_clone.pop();
                                                                is_trimmed = true;
                                                            } else {
                                                                break;
                                                            }
                                                        }
                                                    }
                                                    
                                                    let mut queue_split = false;
                                                    if trimmed_words.len() >= 2 {
                                                        // 🌟 [CRITICAL FIX] 1글자 단어가 진짜 알파벳/한글/숫자일 때만 분할하되, 의미 있는 1글자 명사(NOUN, PROPN)나 숫자(NUM), 수식어가 분할 큐를 찢는 현상을 원천 방지합니다.
                                                        let protected_tags = ["ADJ", "ADV", "DET", "PART", "PRON", "NOUN", "PROPN", "NUM"];
                                                        if trimmed_words.iter().enumerate().any(|(w_idx, w)| {
                                                            let c_count = w.chars().filter(|c| !c.is_whitespace()).count();
                                                            let is_valid_char = w.chars().any(|c| c.is_alphanumeric());
                                                            let tag = valid_tags_clone.get(w_idx).copied().unwrap_or("X");
                                                            c_count == 1 && is_valid_char && !protected_tags.contains(&tag)
                                                        }) {
                                                            queue_split = true;
                                                        }
                                                    }

                                                    if queue_split {
                                                        let parts_display = trimmed_words.join("', '");
                                                        emit_term(&format!("[STANZA-EXT] ✂️ 1글자 단어 포함 감지. 추출단어를 '{}' 로 분할하여 추론 큐에 독립적으로 추가하고 현재 트랙을 종료합니다.", parts_display));
                                                        
                                                        for part in &trimmed_words {
                                                            let p_char_count = part.chars().filter(|c| !c.is_whitespace()).count();
                                                            if p_char_count >= 2 {
                                                                let mut clone = valid_targets[p_idx - 1].clone();
                                                                clone.7 = part.to_string();
                                                                valid_targets.push(clone);
                                                            } else {
                                                                emit_term(&format!("[STANZA-EXT] 🚫 1글자 분할 조각 기각: '{}' (무의미한 단어 확산 방지)", part));
                                                            }
                                                        }
                                                        
                                                        // [CRITICAL FIX] LLM 재시도 루프(loop) 내부에서 온도(temperature) 상승이나 무시 리스트(ignore_list) 갱신 없이 
                                                        // continue를 호출하면 프롬프트가 변하지 않아 영구적인 무한 루프에 빠집니다.
                                                        // 추론 큐(valid_targets)에 분할된 단어를 이미 밀어넣었으므로, 현재 트랙은 즉시 종료(break)해야 합니다.
                                                        break;
                                                    } else if is_trimmed {
                                                        let join_str = " ";
                                                        let trimmed_candidate = trimmed_words.join(join_str);
                                                        
                                                        // 🌟 [수정된 로직] 추출 단어의 사후 절단 결과가 1글자 이하라면 무의미한 단어로 간주하고 강제 기각(nlp_rejected) 처리합니다.
                                                        let char_count = trimmed_candidate.chars().filter(|c| !c.is_whitespace()).count();
                                                        if char_count <= 1 {
                                                            emit_term(&format!("[STANZA-EXT] ✂️ 스마트 절단 결과 1글자만 남음 ('{}' -> '{}'). 무의미한 단어로 간주하여 강제 기각합니다.", extracted_val, trimmed_candidate));
                                                            nlp_rejected = true;
                                                        } else {
                                                            emit_term(&format!("[STANZA-EXT] ✂️ 1차 형태소 분리 후 추출단어 스마트 머리/꼬리 절단 ({}): '{}' -> '{}'", local_language, extracted_val, trimmed_candidate));
                                                            extracted_val = trimmed_candidate;
                                                        }
                                                    } else {
                                                        if extracted_val != eval_ext {
                                                            extracted_val = eval_ext;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // NLP 검증에서 완전히 탈락한 경우(환각 등) 강제 스킵 처리
                            if nlp_rejected {
                                hallucinated_candidates.insert(extracted_val.clone());
                                if !specific_candidate.is_empty() && specific_candidate == extracted_val {
                                    hallucinated_candidates.insert(specific_candidate.clone());
                                }
                                emit_term(&format!("[EXTRACTION] 🛑 NLP 검증 탈락. 재시도 없이 트랙을 종료합니다."));
                                break;
                            }

                            if already_masked_skip {
                                break;
                            }

                            // 🌟 [방어 로직 추가] NMS 후보 단어와의 교집합(포함 관계) 및 단일 숫자/기호 강제 기각
                            let mut nms_valid = true;
                            if !specific_candidate.is_empty() {
                                let no_space_input: String = specific_candidate.chars().filter(|c| !c.is_whitespace()).collect();
                                let no_space_ext: String = extracted_val.chars().filter(|c| !c.is_whitespace()).collect();
                                
                                // 1. 단일 문자이면서 알파벳/한글 등 문자가 아닌 단순 숫자나 기호인 경우 즉시 기각
                                if no_space_ext.chars().count() == 1 && !no_space_ext.chars().next().unwrap().is_alphabetic() {
                                    nms_valid = false;
                                    emit_term(&format!("[DEBUG] 단일 숫자/기호 추출 감지. 강제 기각: '{}'", extracted_val));
                                } 
                                // 2. 입력 단어(specific_candidate)와 추출 단어 간에 포함 관계가 전혀 없는 경우 기각
                                else if !no_space_input.contains(&no_space_ext) && !no_space_ext.contains(&no_space_input) {
                                    let mut found_in_pug_loop = false;
                                    let mut corrected_val = extracted_val.clone();
                                    
                                    // 🌟 [추가 깊이] NMS 교집합이 없더라도, PUG 전체 라인을 루프 돌며 실제 존재하는지 교차 검증합니다.
                                    for line in &lines {
                                        if line.contains(&extracted_val) {
                                            found_in_pug_loop = true;
                                            break;
                                        }
                                        
                                        // 띄어쓰기 변형을 고려한 정규식 루프 검색
                                        if no_space_ext.len() >= 2 && no_space_ext.len() <= 100 {
                                            let escaped_chars: Vec<String> = no_space_ext.chars().map(|c| regex::escape(&c.to_string())).collect();
                                            let regex_pattern = escaped_chars.join(r"\s*");
                                            if let Ok(re) = regex::Regex::new(&regex_pattern) {
                                                if let Some(mat) = re.find(line) {
                                                    found_in_pug_loop = true;
                                                    corrected_val = mat.as_str().to_string();
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    
                                    if found_in_pug_loop {
                                        emit_term(&format!("[DEBUG] NMS 후보와 교집합은 없으나 PUG 본문 전체 루프 검색에서 발견됨. 마스킹 진행 허용: '{}'", corrected_val));
                                        extracted_val = corrected_val;
                                        nms_valid = true;
                                    } else {
                                        nms_valid = false;
                                        emit_term(&format!("[DEBUG] NMS 후보 단어와 교집합 없음. 강제 기각 (입력: '{}', 추출: '{}')", specific_candidate, extracted_val));
                                    }
                                }
                            }

                            if !nms_valid {
                                let no_space_ext: String = extracted_val.chars().filter(|c| !c.is_whitespace()).collect();
                                
                                // 🔍 Step 1: 좀비 단어 (이미 마스킹 완료된 정답) 판별
                                let mut is_zombie = false;
                                for masked in &fully_masked_candidates {
                                    let no_space_masked: String = masked.chars().filter(|c| !c.is_whitespace()).collect();
                                    if no_space_masked.contains(&no_space_ext) || no_space_ext.contains(&no_space_masked) {
                                        // 🌟 [CRITICAL FIX] 본문(masked_text)에 해당 단어가 독립적으로 살아있다면 좀비(할루시네이션)로 취급하지 않고 마스킹 기회를 줍니다!
                                        if masked_text.contains(&extracted_val) {
                                            is_zombie = false;
                                        } else {
                                            is_zombie = true;
                                            break;
                                        }
                                    }
                                }

                                // 🔍 Step 2: 스포일러 단어 (아직 대기 중인 정답) 판별
                                let mut is_spoiler = false;
                                if !is_zombie {
                                    for target in &valid_targets {
                                        let target_cand = &target.7; // specific_candidate
                                        if !target_cand.is_empty() {
                                            let no_space_target: String = target_cand.chars().filter(|c| !c.is_whitespace()).collect();
                                            if no_space_target.contains(&no_space_ext) || no_space_ext.contains(&no_space_target) {
                                                is_spoiler = true;
                                                break;
                                            }
                                        }
                                    }
                                }

                                if is_zombie {
                                    emit_term(&format!("[DEBUG] 🧟 좀비 단어(이미 마스킹 완료) 추출 감지. 연쇄 파기(Cascade Cancellation) 발동: '{}'", extracted_val));
                                    hallucinated_candidates.insert(extracted_val.clone());
                                    // 🌟 [Part 3 개선] 추출 단어와 힌트 단어가 동일할 때만 연대 책임 부과
                                    if !specific_candidate.is_empty() && specific_candidate == extracted_val {
                                        hallucinated_candidates.insert(specific_candidate.clone()); // 🌟 연쇄 파기 명부 등재
                                    }
                                } else if is_spoiler {
                                    emit_term(&format!("[DEBUG] 🎬 스포일러 단어(대기 중인 미래 정답) 추출 감지. 재시도 없이 트랙 종료: '{}'", extracted_val));
                                } else {
                                    emit_term(&format!("[DEBUG] ❌ 단순 오답 추출 감지. 재시도 없이 트랙 종료: '{}'", extracted_val));
                                }
                                
                                break;
                            }

                            // 🌟 [CRITICAL FIX] 추출단어가 만약에 PUG CONTENT에 있는지 없는지 먼저 체크 (ctrl + f)
                            let mut early_exists = matched_context.contains(&extracted_val) || target_text.contains(&extracted_val) || doc_title.contains(&extracted_val) || doc_desc.contains(&extracted_val);

                            // 띄어쓰기 및 기호 변형으로 인한 오탐지를 막기 위해 Space-Agnostic 보정을 선행
                            if !early_exists && extracted_val.chars().count() >= 2 {
                                let no_space_val: String = extracted_val.chars().filter(|c| c.is_alphanumeric()).collect();
                                if no_space_val.len() >= 2 && no_space_val.len() <= 100 {
                                    let escaped_chars: Vec<String> = no_space_val.chars().map(|c| regex::escape(&c.to_string())).collect();
                                    let regex_pattern = escaped_chars.join(r"[^\p{L}\p{N}_]*");
                                    if let Ok(re) = regex::Regex::new(&regex_pattern) {
                                        if let Some(mat) = re.find(&target_text).or_else(|| re.find(&matched_context)).or_else(|| re.find(&doc_title)) {
                                            extracted_val = mat.as_str().to_string();
                                            early_exists = true;
                                        }
                                    }
                                }
                            }
                            
                            // 못찾으면 환각 장부에 등록하고 즉시 스킵
                            if !early_exists {
                                emit_term(&format!("[DEBUG] 추출단어가 PUG CONTENT에 존재하지 않습니다(contains 실패). 할루시네이션으로 즉시 차단: '{}'", extracted_val));
                                hallucinated_candidates.insert(extracted_val.clone());
                                // 🌟 [Part 3 개선] 추출 단어와 힌트 단어가 동일할 때만 연대 책임 부과
                                if !specific_candidate.is_empty() && specific_candidate == extracted_val {
                                    hallucinated_candidates.insert(specific_candidate.clone()); // 🌟 연쇄 파기 명부 등재
                                }
                                
                                break;
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
                                // 🌟 [CRITICAL FIX] 추출된 단어(extracted_val) 자체의 유니코드를 분석하여 
                                // 해당 단어의 실제 언어를 Stage 3 검증의 최우선순위로 동적 재배치합니다.
                                let mut local_lang_counts = std::collections::HashMap::new();
                                for c in extracted_val.chars() {
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
                                        *local_lang_counts.entry(lang).or_insert(0) += 1;
                                    }
                                }
                                
                                let mut current_word_langs = Vec::new();
                                if let Some((best_lang, _)) = local_lang_counts.into_iter().max_by_key(|&(_, count)| count) {
                                    emit_term(&format!("    🌐 [언어 동적 매핑] 추출 단어 '{}' 분석 결과: 단일 언어({})로 검증을 제한합니다.", extracted_val, best_lang));
                                    current_word_langs.push(best_lang.to_string());
                                } else {
                                    let fallback_lang = detected_languages_vec.first().cloned().unwrap_or_else(|| "english".to_string());
                                    emit_term(&format!("    🌐 [언어 동적 매핑] 추출 단어에서 특정 유니코드 특징을 찾지 못해 단일 기본 언어({})로 검증을 제한합니다.", fallback_lang));
                                    current_word_langs.push(fallback_lang);
                                }
                            }

                            // 🌟 [CRITICAL FIX] 서술어/표현 점수가 7점을 넘거나 타겟 미스매치 시 즉시 환각으로 간주하고 강제 차단합니다.
                            if is_hallucination {
                                hallucinated_candidates.insert(extracted_val.clone()); // 🌟 이후 트랙 스킵을 위해 장부에 등록
                                // 🌟 [Part 3 개선] 추출 단어와 힌트 단어가 동일할 때만 연대 책임 부과
                                if !specific_candidate.is_empty() && specific_candidate == extracted_val {
                                    hallucinated_candidates.insert(specific_candidate.clone()); // 🌟 동사/오답 판정 시 남은 파생 트랙 일괄 취소(연쇄 파기)를 위해 등록
                                }
                                
                                emit_term(&format!("[DEBUG] 검증 탈락 감지됨. 재시도 없이 강제 기각: '{}'", extracted_val));
                                break;
                            }

                            // 🌟 [CRITICAL FIX] Qwen3가 압축된 문맥(matched_context)을 읽고 있으므로, 환각 검사도 동일한 문맥에서 수행해야 완벽합니다.
                            // 🌟 [Subsumption 1단계] 검증 기준을 훼손된 masked_text가 아닌, 순수 원본(target_text)으로 변경하여 거대 덩어리 누락을 방지합니다.
                            let exists_in_context = matched_context.contains(&extracted_val);
                            let exists_in_body = target_text.contains(&extracted_val); // 🌟 masked_text -> target_text 교체
                            let exists_in_title = doc_title.contains(&extracted_val);
                            let exists_in_desc = doc_desc.contains(&extracted_val);

                            // 🌟 [CRITICAL FIX] 추출된 값이 임시 마커(해시 기반)를 포함하고 있다면 무조건 환각으로 간주하고 강제 차단합니다.
                            if extracted_val.contains("[___REDACTED_") {
                                emit_term(&format!("[DEBUG] 임시 마커 추출 시도 감지. 재시도 없이 강제 차단: '{}'", extracted_val));
                                break;
                            }

                            // 🌟 [기 마스킹 단어 및 파생어 필터링] 이미 찾은 단어와 겹치거나 파생어인 경우 즉시 기각
                            let mut is_derivative = false;
                            for (orig, _) in &replacement_history {
                                // 길이가 2글자 이상인 경우에 한해서 부분 일치 검사 (너무 짧은 단어 오작동 방지)
                                if orig.chars().count() >= 2 && extracted_val.chars().count() >= 2 {
                                    let mut is_valid_overlap = false;

                                    if extracted_val.contains(orig) {
                                        if let Some(idx) = extracted_val.find(orig) {
                                            let cb = extracted_val[..idx].chars().next_back();
                                            let ca = extracted_val[idx + orig.len()..].chars().next();
                                            let mut is_infix = false;
                                            let is_ascii_orig = orig.chars().all(|c| c.is_ascii_alphanumeric() || c.is_whitespace() || c == '-' || c == '_');
                                            
                                            if is_ascii_orig {
                                                let prev_is_alpha = cb.map_or(false, |c| c.is_ascii_alphanumeric());
                                                let next_is_alpha = ca.map_or(false, |c| c.is_ascii_alphanumeric());
                                                if prev_is_alpha || next_is_alpha { is_infix = true; }
                                            } else {
                                                let prev_is_alnum = cb.map_or(false, |c| c.is_alphanumeric());
                                                let next_is_alnum = ca.map_or(false, |c| c.is_alphanumeric());
                                                if prev_is_alnum && next_is_alnum {
                                                    if orig.chars().count() <= 2 { is_infix = true; }
                                                } else if prev_is_alnum || next_is_alnum {
                                                    if orig.chars().count() <= 1 { is_infix = true; }
                                                }
                                            }
                                            if !is_infix { is_valid_overlap = true; }
                                        }
                                    } else if orig.contains(&extracted_val) {
                                        if let Some(idx) = orig.find(&extracted_val) {
                                            let cb = orig[..idx].chars().next_back();
                                            let ca = orig[idx + extracted_val.len()..].chars().next();
                                            let mut is_infix = false;
                                            let is_ascii_ext = extracted_val.chars().all(|c| c.is_ascii_alphanumeric() || c.is_whitespace() || c == '-' || c == '_');
                                            
                                            if is_ascii_ext {
                                                let prev_is_alpha = cb.map_or(false, |c| c.is_ascii_alphanumeric());
                                                let next_is_alpha = ca.map_or(false, |c| c.is_ascii_alphanumeric());
                                                if prev_is_alpha || next_is_alpha { is_infix = true; }
                                            } else {
                                                let prev_is_alnum = cb.map_or(false, |c| c.is_alphanumeric());
                                                let next_is_alnum = ca.map_or(false, |c| c.is_alphanumeric());
                                                if prev_is_alnum && next_is_alnum {
                                                    if extracted_val.chars().count() <= 2 { is_infix = true; }
                                                } else if prev_is_alnum || next_is_alnum {
                                                    if extracted_val.chars().count() <= 1 { is_infix = true; }
                                                }
                                            }
                                            if !is_infix { is_valid_overlap = true; }
                                        }
                                    }

                                    if is_valid_overlap {
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
                                fully_masked_candidates.insert(extracted_val.clone());
                                if !specific_candidate.is_empty() && specific_candidate == extracted_val {
                                    fully_masked_candidates.insert(specific_candidate.clone());
                                }
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
                                emit_term(&format!("[DEBUG] 과잉 추출 감지 (어절: {}, 글자수: {}). 재시도 없이 강제 차단: '{}'", word_count, char_count, extracted_val));
                                break;
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
                                    // 🌟 [일반 텍스트 보정] 공백 및 특수문자를 완전히 무시한 정규식을 동적 생성하여 원본 텍스트에 존재하는 형태(기호 포함)를 복원합니다.
                                    let no_space_val: String = extracted_val.chars().filter(|c| c.is_alphanumeric()).collect();
                                    
                                    // 정규식 엔진의 부하를 막기 위해 글자 수가 2글자 이상 100글자 이하일 때만 수행
                                    if no_space_val.len() >= 2 && no_space_val.len() <= 100 {
                                        // 정규식 특수문자 이스케이프 후 공백/특수기호 허용 패턴 조립
                                        let escaped_chars: Vec<String> = no_space_val.chars().map(|c| regex::escape(&c.to_string())).collect();
                                        let regex_pattern = escaped_chars.join(r"[^\p{L}\p{N}_]*");
                                        
                                        if let Ok(re) = regex::Regex::new(&regex_pattern) {
                                            // 본문, 제목, 압축 문맥 순으로 탐색하여 원래 형태 복원 시도
                                            if let Some(mat) = re.find(&masked_text).or_else(|| re.find(&matched_context)).or_else(|| re.find(&doc_title)) {
                                                emit_term(&format!("[DEBUG] 특수기호/띄어쓰기 보정 성공: '{}' -> '{}'", extracted_val, mat.as_str()));
                                                extracted_val = mat.as_str().to_string();
                                            }
                                        }
                                    }
                                }
                            }

                            // 🌟 [CRITICAL FIX] 빈 값 반환 시 재시도 없이 트랙 즉시 종료
                            if extracted_val.is_empty() || extracted_val == "..." || extracted_val == "null" {
                                emit_term(&format!("[DEBUG] 빈 값 반환 감지. 재시도 없이 해당 트랙을 종료합니다."));
                                break;
                            }

                            // 🌟 [환각 방지 3번 재시도 루프 (유지됨)]
                            // 값이 추출되긴 했으나 LLM이 실제로 읽은 압축 문맥(matched_context)이나 원본에 없는 환각(Hallucination)인 경우, 
                            // 무시 리스트(ignore_list)에 넣고 다시 프롬프트를 생성해 최대 3번까지 재시도(continue)합니다.
                            let re_check_context = matched_context.contains(&extracted_val);
                            let re_check_body = masked_text.contains(&extracted_val);
                            let re_check_title = doc_title.contains(&extracted_val);
                            let re_check_desc = doc_desc.contains(&extracted_val);

                            if !re_check_context && !re_check_body && !re_check_title && !re_check_desc {
                                // 🌟 [CRITICAL FIX] 원본(target_text)에는 존재하나 마스킹된 텍스트(masked_text/matched_context)에는 없는 경우,
                                // 이미 다른 트랙에 의해 부분 마스킹되어 텍스트가 훼손된 상태이므로 환각이 아닌 트랙 포화로 간주하고 조기 종료합니다.
                                if early_exists {
                                    emit_term(&format!("[DEBUG] 원본에는 존재하나 마스킹 과정에서 훼손됨. 파생어/중복 마스킹으로 간주하여 트랙 조기 종료: '{}'", extracted_val));
                                    fully_masked_candidates.insert(extracted_val.clone());
                                    if !specific_candidate.is_empty() && specific_candidate == extracted_val {
                                        fully_masked_candidates.insert(specific_candidate.clone());
                                    }
                                    break;
                                }

                                // 🌟 [조건부 부분 치환 (Vector Bouncer) 적용 - 양방향 점진적 수축 로테이션]
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
                                            
                                            // 🌟 [추가] 수축 로테이션 시에도 문맥 의존도가 높은 고유명사 타겟에 한하여 제목 벡터 유사도 보너스를 동일하게 부여합니다.
                                            let mut title_bonus = 0.0;
                                            let t_sim = cosine_similarity(&p_emb, &title_emb);
                                            if t_sim > 0.0 {
                                                title_bonus = t_sim * 0.15;
                                            }
                                            
                                            let base_score = b_score - (p_score * penalty_weight) - verb_penalty + title_bonus;
                                            
                                            // Pass 2와 동일한 커트라인(0.3) 검증
                                            if base_score > 0.3 {
                                                sub_chunks.push((start, end, chunk_text, base_score * length_weight, verb_penalty));
                                            } else {
                                                emit_term(&format!("    💀 [DEFEAT] 수축 로테이션 탈락: '{}' (Score: {:.4} / VerbPenalty: {:.4})", chunk_text, base_score, verb_penalty));
                                            }
                                        }
                                    }
                                }

                                // 🌟 [NMS BATTLE] 추출된 서브 청크 중첩을 LLM Granite 기반으로 해결합니다.
                                sub_chunks.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
                                let mut final_sub_chunks: Vec<(usize, usize, String, f32, f32)> = Vec::new();
                                let mut remaining_subs = sub_chunks.clone();

                                while !remaining_subs.is_empty() {
                                    let current = remaining_subs.remove(0);
                                    let mut overlaps = Vec::new();
                                    let mut next_remaining = Vec::new();

                                    for cand in remaining_subs {
                                        if current.0 < cand.1 && current.1 > cand.0 {
                                            overlaps.push(cand);
                                        } else {
                                            next_remaining.push(cand);
                                        }
                                    }

                                    if overlaps.is_empty() {
                                        final_sub_chunks.push(current);
                                    } else {
                                        let mut candidates = vec![current.clone()];
                                        candidates.extend(overlaps.clone());
                                        
                                        let mut unique_texts = Vec::new();
                                        let mut unique_cands = Vec::new();
                                        for cand in &candidates {
                                            if !unique_texts.contains(&cand.2) {
                                                unique_texts.push(cand.2.clone());
                                                unique_cands.push(cand.clone());
                                            }
                                        }

                                        if unique_cands.len() == 1 {
                                            final_sub_chunks.push(unique_cands[0].clone());
                                        } else {
                                            emit_term(&format!("    ⚖️ [SCORE NMS] 수축 로테이션 오버랩 발생! 벡터 점수 기반 판별: {:?}", unique_texts));
                                            
                                            let mut sorted_cands = unique_cands.clone();
                                            // 튜플 구조: (start, end, chunk_text, base_score, verb_penalty)
                                            // base_score(3번 인덱스) 기준 내림차순 정렬, 동점이면 텍스트 길이 내림차순
                                            sorted_cands.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal).then(b.2.len().cmp(&a.2.len())));

                                            let winner = sorted_cands[0].clone();
                                            let winner_idx = unique_cands.iter().position(|x| x.2 == winner.2).unwrap_or(0);

                                            emit_term(&format!("    👑 [WINNER-SCORE] 수축 로테이션 승리: '{}' (Score: {:.4})", winner.2, winner.3));
                                            
                                            for (i, cand) in unique_cands.iter().enumerate() {
                                                if i != winner_idx {
                                                    emit_term(&format!("    💀 [DEFEAT-SCORE] 수축 로테이션 기각: '{}' (Score: {:.4})", cand.2, cand.3));
                                                }
                                            }
                                            
                                            final_sub_chunks.push(winner);
                                        }
                                    }
                                    remaining_subs = next_remaining;
                                }

                                // 🌟 [WINNER 마스킹 반영] 살아남은 조각들을 최종 마스킹 처리합니다.
                                for (_, _, text_val, score, _) in final_sub_chunks {
                                    emit_term(&format!("    👑 [WINNER] 부분 일치(수축 로테이션) 통과: '{}' 중 '{}' (Score: {:.4}) -> 강제 마스킹", extracted_val, text_val, score));
                                    
                                    let mnemonic = crate::parsing::generate_mnemonic();
                                    let upper_key = base_target.to_uppercase(); 
                                    let final_replacement = format!("[{}]", mnemonic);
                                    let skip_marker = format!("[___REDACTED_{}___]", skip_counter);
                                    
                                    masked_text = safe_replace(&masked_text, &text_val, &skip_marker);
                                    doc_title = safe_replace(&doc_title, &text_val, &skip_marker);
                                    doc_desc = safe_replace(&doc_desc, &text_val, &skip_marker);
                                    matched_context = safe_replace(&matched_context, &text_val, &skip_marker);
                                    
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
                                    item_extract_count += 1;
                                    
                                    // 🌟 [Part 1 개선] 마스킹 완료 장부 등록 기준 분리
                                    fully_masked_candidates.insert(extracted_val.clone());
                                    if specific_candidate == extracted_val {
                                        fully_masked_candidates.insert(specific_candidate.clone()); // 🌟 두 단어가 일치할 때만 힌트 단어 장부 등록
                                    }

                                    emit_term(&format!("[EXTRACTION] 🎯 마스킹 성공 (부분 치환). 재시도 없이 해당 트랙을 종료합니다."));
                                    break;
                                }

                                emit_term(&format!("[DEBUG] 완전 환각/오탐지 감지. 재시도 없이 해당 트랙을 즉시 종료합니다: '{}'", extracted_val));
                                break;
                            }

                            // 정상 추출되었으므로 연속 실패 카운터를 리셋합니다.
                            link_counter = 0;
                            let current_temperature = 0.0; // 🌟 성공했으므로 온도를 다시 0으로 차갑게 초기화
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
                                    let final_replacement = format!("[{}]", mnemonic);
                                    
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
                                    
                                    // 🌟 [Part 1 개선] 마스킹 완료 장부 등록 기준 분리
                                    fully_masked_candidates.insert(extracted_val.clone());
                                    if specific_candidate == extracted_val {
                                        fully_masked_candidates.insert(specific_candidate.clone()); // 🌟 두 단어가 일치할 때만 힌트 단어 장부 등록
                                    }

                                    // 🌟 [Part 2 개선] 트랙 조기 종료(break) 조건 세분화
                                    if !specific_candidate.is_empty() && !fully_masked_candidates.contains(&specific_candidate) && masked_text.contains(&specific_candidate) {
                                        emit_term(&format!("[EXTRACTION] 🎯 마스킹 성공 (Phase 2 병합). 하지만 힌트 단어('{}')가 본문에 남아있어 트랙을 연장(continue)합니다.", specific_candidate));
                                        continue;
                                    } else {
                                        // 🌟 [CRITICAL FIX] 성공적으로 덮어썼으므로 루프를 종료하고 다음 트랙으로 넘어갑니다.
                                        emit_term(&format!("[EXTRACTION] 🎯 마스킹 성공 (Phase 2 병합). 해당 트랙 종료 후 다음 트랙으로 이동합니다."));
                                        break;
                                    }
                                }
                            }

                            // 🌟 마스킹 니모닉 생성 및 즉시 치환 대신 해시 기반 마커로 임시 치환
                            let mnemonic = crate::parsing::generate_mnemonic();
                            let upper_key = base_target.to_uppercase(); 
                            let final_replacement = format!("[{}]", mnemonic);
                            
                            // 🌟 [CRITICAL FIX] 추출된 단어(extracted_val)가 본문에는 존재하지만, 
                            // 제목이나 요약에 조사/수식어가 붙은 형태가 아닌 핵심 명사만 존재할 때 구출하는 로직입니다.
                            // 무분별한 문자 조합(O(N^2))으로 "니다" 같은 접미사가 추출되는 대참사를 막기 위해,
                            // 단어 단위(어절) 조합 및 접두사(Prefix) 기반 슬라이싱으로 안전한 청크만 생성합니다.
                            let chars_vec: Vec<char> = extracted_val.chars().collect();
                            if chars_vec.len() >= 2 {
                                let lang_prefix = target_name.split('_').next().unwrap_or("english");
                                let prefixed_b_val = target_bias.split(',').map(|s| format!("{} {}", lang_prefix, s.trim())).collect::<Vec<_>>().join(", ");
                                let prefixed_p_val = target_prejudice.split(',').map(|s| format!("{} {}", lang_prefix, s.trim())).collect::<Vec<_>>().join(", ");
                                let bias_emb = model.get_embedding(prefixed_b_val.clone()).await.unwrap_or_else(|_| vec![0.0; 384]);
                                let prej_emb = model.get_embedding(prefixed_p_val.clone()).await.unwrap_or_else(|_| vec![0.0; 384]);

                                // 🌟 안전한 후보군(Candidate Chunks) 동적 생성 (어절 단위 및 접두사 한정)
                                let mut safe_chunks = Vec::new();
                                let words: Vec<&str> = extracted_val.split_whitespace().collect();
                                for i in 0..words.len() {
                                    for j in i..words.len() {
                                        let base_chunk = words[i..=j].join(" ");
                                        if base_chunk.chars().count() >= 2 {
                                            safe_chunks.push(base_chunk.clone());
                                        }
                                        
                                        // 단일 어절일 경우, 조사/접미사를 떼어내기 위해 접두사(Prefix) 파생 단어 생성
                                        if i == j {
                                            let w_chars: Vec<char> = words[i].chars().collect();
                                            if w_chars.len() > 2 {
                                                // 앞에서부터 자르므로 "안전합니다" -> "안전", "안전합" (뒤에서 잘리는 "니다" 발생 원천 차단)
                                                for end in 2..w_chars.len() {
                                                    let prefix: String = w_chars[0..end].iter().collect();
                                                    safe_chunks.push(prefix);
                                                }
                                            }
                                        }
                                    }
                                }
                                safe_chunks.sort();
                                safe_chunks.dedup();

                                // Title 검증
                                if !doc_title.contains(&extracted_val) {
                                    let mut best_chunk = String::new();
                                    let mut best_score = 0.3_f32; 
                                    
                                    for chunk_trim in &safe_chunks {
                                        if doc_title.contains(chunk_trim) {
                                            let p_emb = model.get_embedding(chunk_trim.clone()).await.unwrap_or_else(|_| vec![0.0; 384]);
                                            let b_score = cosine_similarity(&p_emb, &bias_emb);
                                            let p_score = cosine_similarity(&p_emb, &prej_emb);
                                            let v_sim = cosine_similarity(&p_emb, &verb_emb);
                                            
                                            let word_count = chunk_trim.split_whitespace().count();
                                            let beta = if word_count <= 2 { 0.05 } else { 0.10 };
                                            let verb_penalty = v_sim * beta;
                                            let penalty_weight = if word_count <= 2 { 0.3 } else { 0.7 };
                                            
                                            let mut t_bonus = 0.0;
                                            let t_sim = cosine_similarity(&p_emb, &title_emb);
                                            if t_sim > 0.0 { t_bonus = t_sim * 0.15; }
                                            
                                            let score = b_score - (p_score * penalty_weight) - verb_penalty + t_bonus;
                                            if score > best_score {
                                                best_score = score;
                                                best_chunk = chunk_trim.clone();
                                            }
                                        }
                                    }
                                    if !best_chunk.is_empty() {
                                        emit_term(&format!("    👑 [CROSS-FIELD RESCUE] 제목(Title)에 전체 문자열이 없으나, 핵심 단어 '{}' 발견 (Score: {:.4}) -> 강제 마스킹", best_chunk, best_score));
                                        let c_mnemonic = crate::parsing::generate_mnemonic();
                                        let c_final_replacement = format!("[{}]", c_mnemonic);
                                        let c_skip_marker = format!("[___REDACTED_{}___]", skip_counter);
                                        
                                        doc_title = safe_replace(&doc_title, &best_chunk, &c_skip_marker);
                                        skip_map.insert(c_skip_marker.clone(), c_final_replacement.clone());
                                        replacement_history.push((best_chunk.clone(), c_skip_marker.clone()));
                                        current_target_found.push(best_chunk.clone());
                                        domain_history.push((target_name.to_string(), best_chunk.clone()));
                                        skip_counter += 1;
                                        
                                        all_matches.push(json!({
                                            "name": upper_key,
                                            "value": best_chunk,
                                            "mnemonic": c_mnemonic
                                        }));
                                    }
                                }

                                // Desc 검증
                                if !doc_desc.contains(&extracted_val) {
                                    let mut best_chunk = String::new();
                                    let mut best_score = 0.3_f32; 
                                    
                                    for chunk_trim in &safe_chunks {
                                        if doc_desc.contains(chunk_trim) {
                                            let p_emb = model.get_embedding(chunk_trim.clone()).await.unwrap_or_else(|_| vec![0.0; 384]);
                                            let b_score = cosine_similarity(&p_emb, &bias_emb);
                                            let p_score = cosine_similarity(&p_emb, &prej_emb);
                                            let v_sim = cosine_similarity(&p_emb, &verb_emb);
                                            
                                            let word_count = chunk_trim.split_whitespace().count();
                                            let beta = if word_count <= 2 { 0.05 } else { 0.10 };
                                            let verb_penalty = v_sim * beta;
                                            let penalty_weight = if word_count <= 2 { 0.3 } else { 0.7 };
                                            
                                            let mut t_bonus = 0.0;
                                            let t_sim = cosine_similarity(&p_emb, &title_emb);
                                            if t_sim > 0.0 { t_bonus = t_sim * 0.15; }
                                            
                                            let score = b_score - (p_score * penalty_weight) - verb_penalty + t_bonus;
                                            if score > best_score {
                                                best_score = score;
                                                best_chunk = chunk_trim.clone();
                                            }
                                        }
                                    }
                                    if !best_chunk.is_empty() {
                                        emit_term(&format!("    👑 [CROSS-FIELD RESCUE] 요약(Desc)에 전체 문자열이 없으나, 핵심 단어 '{}' 발견 (Score: {:.4}) -> 강제 마스킹", best_chunk, best_score));
                                        let c_mnemonic = crate::parsing::generate_mnemonic();
                                        let c_final_replacement = format!("[{}]", c_mnemonic);
                                        let c_skip_marker = format!("[___REDACTED_{}___]", skip_counter);
                                        
                                        doc_desc = safe_replace(&doc_desc, &best_chunk, &c_skip_marker);
                                        skip_map.insert(c_skip_marker.clone(), c_final_replacement.clone());
                                        replacement_history.push((best_chunk.clone(), c_skip_marker.clone()));
                                        current_target_found.push(best_chunk.clone());
                                        domain_history.push((target_name.to_string(), best_chunk.clone()));
                                        skip_counter += 1;
                                        
                                        all_matches.push(json!({
                                            "name": upper_key,
                                            "value": best_chunk,
                                            "mnemonic": c_mnemonic
                                        }));
                                    }
                                }
                            }

                            // 🌟 안전해진 상태에서 메인 마커 생성 (Cross-Field Rescue가 skip_counter를 업데이트했을 수 있음)
                            let skip_marker = format!("[___REDACTED_{}___]", skip_counter);

                            // 🌟 [Subsumption 2단계] 추출된 거대 덩어리 내부에 이미 치환된 소형 마커가 존재하는지 확인하고, 
                            // 현재 masked_text에 반영된 형태(Hybrid)를 역산하여 덮어쓸 준비를 합니다.
                            let mut hybrid_val = extracted_val.clone();
                            let mut subsumed_markers = std::collections::HashSet::new();
                            for (orig, marker) in &replacement_history {
                                if extracted_val.contains(orig) && orig.chars().count() >= 2 {
                                    // 🌟 [CRITICAL FIX] '에이전트가' 내부에 있는 '이전'처럼 단어 중간에 공백 없이 낀 파편(Infix)은 
                                    // safe_replace에서 마스킹 대상이 아니었으므로 역산(Subsumption)에서도 안전하게 제외합니다.
                                    let mut is_infix = false;
                                    if let Some(idx) = extracted_val.find(orig) {
                                        let cb = extracted_val[..idx].chars().next_back();
                                        let ca = extracted_val[idx + orig.len()..].chars().next();
                                        
                                        let is_ascii_orig = orig.chars().all(|c| c.is_ascii_alphanumeric() || c.is_whitespace() || c == '-' || c == '_');
                                        if is_ascii_orig {
                                            let prev_is_alpha = cb.map_or(false, |c| c.is_ascii_alphanumeric());
                                            let next_is_alpha = ca.map_or(false, |c| c.is_ascii_alphanumeric());
                                            if prev_is_alpha || next_is_alpha { is_infix = true; }
                                        } else {
                                            let prev_is_alnum = cb.map_or(false, |c| c.is_alphanumeric());
                                            let next_is_alnum = ca.map_or(false, |c| c.is_alphanumeric());
                                            if prev_is_alnum && next_is_alnum {
                                                if orig.chars().count() <= 2 { is_infix = true; }
                                            } else if prev_is_alnum || next_is_alnum {
                                                if orig.chars().count() <= 1 { is_infix = true; }
                                            }
                                        }
                                    }
                                    
                                    if !is_infix {
                                        hybrid_val = hybrid_val.replace(orig, marker);
                                        subsumed_markers.insert(marker.clone());
                                    }
                                }
                            }

                            // 🌟 [Subsumption 3단계] 흡수된 구형 소형 마커들을 장부(all_matches)에서 제거하여 새로운 범주(예: NAME)로 완벽히 세대교체합니다.
                            if !subsumed_markers.is_empty() {
                                emit_term(&format!("    🔄 [SUBSUMPTION] 거대 덩어리 발견! 기존 마커 {}개를 '{}' ({}) 속성으로 흡수 병합 평가합니다.", subsumed_markers.len(), extracted_val, upper_key));
                                all_matches.retain(|match_val| {
                                    if let Some(obj) = match_val.as_object() {
                                        if let Some(m_val) = obj.get("value").and_then(|v| v.as_str()) {
                                            if extracted_val.contains(m_val) && m_val != extracted_val && m_val.chars().count() >= 2 {
                                                // 🌟 [CRITICAL FIX] 거대 덩어리에 흡수되었더라도, 제목이나 본문 다른 곳에 소형 마커가 독립적으로 살아있다면 장부에서 삭제하지 않습니다!
                                                if let Some(m_mnemonic) = obj.get("mnemonic").and_then(|v| v.as_str()) {
                                                    let mut is_still_alive = false;
                                                    for (s_marker, s_repl) in skip_map.iter() {
                                                        if s_repl.contains(m_mnemonic) {
                                                            if masked_text.contains(s_marker) || doc_title.contains(s_marker) || doc_desc.contains(s_marker) {
                                                                is_still_alive = true;
                                                                break;
                                                            }
                                                        }
                                                    }
                                                    if is_still_alive {
                                                        emit_term(&format!("      -> 소형 마커 '{}'는 다른 필드(제목 등)에 독립적으로 살아있어 흡수를 면제합니다.", m_val));
                                                        return true; // Keep it!
                                                    }
                                                }
                                                return false; // Safely subsume
                                            }
                                        }
                                    }
                                    true
                                });
                            }
                            
                            // 🌟 [CRITICAL FIX] 원본 텍스트(본문+제목) 모두에서 임시 마커로 치환하여 이어지는 LLM 추론에서 혼선을 원천 방지합니다.
                            // 역산된 hybrid_val을 사용하여 조각난 마커들까지 통째로 하나의 거대 마커로 덮어씌웁니다.
                            masked_text = safe_replace(&masked_text, &hybrid_val, &skip_marker);
                            doc_title = safe_replace(&doc_title, &hybrid_val, &skip_marker);
                            doc_desc = safe_replace(&doc_desc, &hybrid_val, &skip_marker);
                            matched_context = safe_replace(&matched_context, &hybrid_val, &skip_marker); 
                            
                            skip_map.insert(skip_marker.clone(), final_replacement);
                            replacement_history.push((extracted_val.clone(), skip_marker.clone())); 
                            
                            current_target_found.push(extracted_val.clone());
                            domain_history.push((target_name.to_string(), extracted_val.clone()));

                            // 🌟 [파생어 자동 마스킹 제거됨] NMS 로직이 개별 단어를 충분히 커버하므로,
                            // 과도한 전역 치환으로 인한 타 트랙의 컨텍스트 훼손을 방지하기 위해 파생어 강제 마스킹을 삭제했습니다.

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
                            
                            // 🌟 [Part 1 개선] 마스킹 완료 장부 등록 기준 분리
                            fully_masked_candidates.insert(extracted_val.clone());
                            if specific_candidate == extracted_val {
                                fully_masked_candidates.insert(specific_candidate.clone()); // 🌟 두 단어가 일치할 때만 힌트 단어 장부 등록
                            }

                            // 🌟 [Part 2 개선] 트랙 조기 종료(break) 조건 세분화
                            if !specific_candidate.is_empty() && !fully_masked_candidates.contains(&specific_candidate) && masked_text.contains(&specific_candidate) {
                                emit_term(&format!("[EXTRACTION] 🎯 마스킹 성공 (전체 치환). 하지만 힌트 단어('{}')가 본문에 남아있어 트랙을 연장(continue)합니다.", specific_candidate));
                                continue;
                            } else {
                                // 🌟 [CRITICAL FIX] 정상적으로 마스킹을 성공했으므로 무한 루프(continue)를 돌지 않고 즉시 탈출(break)하여 다음 트랙으로 이동합니다.
                                emit_term(&format!("[EXTRACTION] 🎯 마스킹 성공 (전체 치환). 해당 트랙 종료 후 다음 트랙으로 이동합니다."));
                                break;
                            }
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

                    // 🌟 [FINAL SWEEP] 마지막으로 한 번 더 루프를 돌며 마스킹된 항목(all_matches)이 본문/제목/요약에 누락되어 남아있는지 체크하고 일괄 강제 마스킹합니다.
                    let mut final_sweep_count = 0;
                    for match_val in &all_matches {
                        if let Some(obj) = match_val.as_object() {
                            if let (Some(name), Some(val), Some(mnemonic)) = (obj.get("name").and_then(|v| v.as_str()), obj.get("value").and_then(|v| v.as_str()), obj.get("mnemonic").and_then(|v| v.as_str())) {
                                if val.chars().count() >= 2 && !val.starts_with("[___REDACTED") {
                                    let final_repl = format!("[{}]", mnemonic);
                                    
                                    // 1. 단순 교체 (노이즈/링크 복원 과정에서 튀어나온 텍스트 등 즉시 교체) - 중첩 마스킹 방지 적용
                                    let escaped_val = regex::escape(val);
                                    let pattern1 = format!(r"(\[[^\]]+\])|({})", escaped_val);
                                    if let Ok(re) = regex::Regex::new(&pattern1) {
                                        let replacer = |caps: &regex::Captures| {
                                            if caps.get(1).is_some() {
                                                caps[0].to_string() // 이미 마스킹된 [니모닉] 내부는 보존
                                            } else {
                                                final_repl.clone()
                                            }
                                        };
                                        let new_text = re.replace_all(&masked_text, replacer).to_string();
                                        if new_text != masked_text { masked_text = new_text; final_sweep_count += 1; }
                                        let new_title = re.replace_all(&doc_title, replacer).to_string();
                                        if new_title != doc_title { doc_title = new_title; final_sweep_count += 1; }
                                        let new_desc = re.replace_all(&doc_desc, replacer).to_string();
                                        if new_desc != doc_desc { doc_desc = new_desc; final_sweep_count += 1; }
                                    }

                                    // 2. 띄어쓰기/특수기호 변형 누락분 추적 교체 (Space-Agnostic Sweep) - 중첩 마스킹 방지 적용
                                    let no_space_val: String = val.chars().filter(|c| c.is_alphanumeric()).collect();
                                    if no_space_val.len() >= 2 && no_space_val.len() <= 100 {
                                        let escaped_chars: Vec<String> = no_space_val.chars().map(|c| regex::escape(&c.to_string())).collect();
                                        let regex_pattern = escaped_chars.join(r"[^\p{L}\p{N}_]+"); 
                                        let pattern2 = format!(r"(\[[^\]]+\])|({})", regex_pattern);
                                        if let Ok(re) = regex::Regex::new(&pattern2) {
                                            let replacer = |caps: &regex::Captures| {
                                                if caps.get(1).is_some() {
                                                    caps[0].to_string() // 이미 마스킹된 [니모닉] 내부는 보존
                                                } else {
                                                    final_repl.clone()
                                                }
                                            };
                                            let new_text = re.replace_all(&masked_text, replacer).to_string();
                                            if new_text != masked_text { masked_text = new_text; final_sweep_count += 1; }
                                            let new_title = re.replace_all(&doc_title, replacer).to_string();
                                            if new_title != doc_title { doc_title = new_title; final_sweep_count += 1; }
                                            let new_desc = re.replace_all(&doc_desc, replacer).to_string();
                                            if new_desc != doc_desc { doc_desc = new_desc; final_sweep_count += 1; }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if final_sweep_count > 0 {
                        emit_term(&format!("[EXTRACTION] 🧹 Final Sweep 완료: 변형되거나 누락된 잔여 항목 {}번 추가 마스킹 처리됨.", final_sweep_count));
                    }

                    if !all_matches.is_empty() {
                        // 🌟 마스킹된 전체 텍스트도 masked 오브젝트 내부의 text 필드로 함께 캡슐화합니다.
                        extracted_json = json!({ "matches": all_matches, "text": masked_text, "title": doc_title.clone(), "description": doc_desc.clone() });
                    }
                    
                    // 🌟 [VRAM 해제] 단일 문서에 대한 Stanza(ONNX) 추론이 모두 끝났으므로 즉시 메모리를 반환합니다.
                    drop(stanza_pipeline);
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

        // 🌟 [VRAM 해제] 마스킹 전체 작업이 종료되었으므로 LLM 및 임베딩 모델의 리소스를 VRAM에서 즉시 비웁니다.
        emit_term("[PROCESS] 🧹 마스킹 작업 완료. VRAM 반환을 위해 모든 AI 모델 리소스를 즉시 해제합니다.");
        drop(model); // 지역 변수로 잡혀있던 모델 참조(Arc 클론)를 먼저 끊어줍니다.
        
        {
            let mut model_lock = model_mutex.lock().await;
            if let Some(m) = model_lock.as_ref() {
                m.deep_purge_resources().await; // 임베딩 및 생성 모델의 메모리를 완벽 반환합니다.
            }
            *model_lock = None; // 전역 Mutex 내 모델 인스턴스 완전 소멸
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
    
    let mut raw_pug = parsing::convert_to_clean_pug(&clean_html_content, PugMode::NoAttributesMode, Some(&url));

    // 🌟 Python 패리티: [preprocess_text] 기호와 명사가 떡칠되어 Stanza 가 OOV 대참사를 내는 현상을 예방하기 위해 PUG 줄 단위 분리 전 공백 전사 레이어 장전
    if !raw_pug.is_empty() {
        if let Ok(re_space) = regex::Regex::new(r"([.,!?()\[\]{}|/\\<>])") {
            let space_applied = re_space.replace_all(&raw_pug, " $1 ").to_string();
            if let Ok(re_multi_space) = regex::Regex::new(r" +") {
                raw_pug = re_multi_space.replace_all(&space_applied, " ").to_string();
            }
        }
    }

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

