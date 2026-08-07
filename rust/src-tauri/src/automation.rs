use anyhow::anyhow;
use chromiumoxide::{Browser, BrowserConfig};
use fantoccini::ClientBuilder;
use futures::StreamExt;
use std::process::{Command, Stdio};
use std::time::Duration;
use std::path::PathBuf;
use tauri::Emitter;
use std::sync::Arc;
use once_cell::sync::Lazy;
use serde_json::json;

// Global storage to keep browser alive
pub(crate) static GLOBAL_BROWSER: Lazy<Arc<tokio::sync::Mutex<Option<Arc<Browser>>>>> = Lazy::new(|| {
    Arc::new(tokio::sync::Mutex::new(None))
});

// Driver Port (Only for Firefox/Safari)
const DRIVER_PORT: u16 = 4444;
const CHROME_DEBUG_PORT: u16 = 9222;

#[derive(serde::Serialize)]
pub struct BrowserStatus {
    pub name: String,
    pub is_supported: bool, // Chrome/Edge = Driverless Supported
    pub is_installed: bool,
    pub needs_driver: bool, // Firefox/Safari = True
}

pub async fn is_browser_reachable() -> bool {
    tokio::net::TcpStream::connect(format!("127.0.0.1:{}", CHROME_DEBUG_PORT)).await.is_ok()
}

// --- Entry Point ---
pub async fn run_browser_automation(
    browser_type: String,
    url: String,
    script: String,
    app_handle: tauri::AppHandle,
) -> anyhow::Result<String> {
    match browser_type.as_str() {
        "chrome" | "edge" => run_driverless_automation(&browser_type, &url, &script, app_handle).await,
        "firefox" | "safari" => run_driver_automation(&browser_type, &url, &script).await,
        _ => Err(anyhow!("Unknown browser type")),
    }
}

// --- Reconnection Logic ---
pub async fn try_reconnect_existing_browser(app_handle: tauri::AppHandle) -> anyhow::Result<()> {
    println!("[AUTO] Attempting to reconnect to existing browser on port {}...", CHROME_DEBUG_PORT);
    
    if tokio::net::TcpStream::connect(format!("127.0.0.1:{}", CHROME_DEBUG_PORT)).await.is_err() {
        println!("[AUTO] No existing browser detected on port {}.", CHROME_DEBUG_PORT);
        let _ = app_handle.emit("browser-status", "stopped");
        return Ok(());
    }

    let addr = format!("http://127.0.0.1:{}", CHROME_DEBUG_PORT);
    match Browser::connect(addr).await {
        Ok((browser, mut handler)) => {
            println!("[AUTO] Successfully reconnected to existing browser.");
            let browser_arc = Arc::new(browser);
            {
                let mut global = GLOBAL_BROWSER.lock().await;
                *global = Some(browser_arc.clone());
            }
            let _ = app_handle.emit("browser-status", "running");
            spawn_browser_monitor(browser_arc.clone(), app_handle.clone());
            tokio::spawn(async move {
                while let Some(h) = handler.next().await {
                    if let Err(_) = h { break; }
                }
                let mut global = GLOBAL_BROWSER.lock().await; 
                *global = None; 
                
            });
            Ok(())
        },
        Err(e) => {
            println!("[AUTO] Reconnection failed: {}", e);
            let _ = app_handle.emit("browser-status", "stopped");
            Ok(())
        }
    }
}

// Global storage for the last detected browser state
pub struct DetectedState {
    pub url: String,
    pub tab_id: String, 
    pub is_client: bool,
    pub is_admin: bool,
}

pub(crate) static LAST_DETECTED_STATE: Lazy<Arc<tokio::sync::Mutex<DetectedState>>> = Lazy::new(|| {
    Arc::new(tokio::sync::Mutex::new(DetectedState {
        url: String::new(),
        tab_id: String::new(), 
        is_client: false,
        is_admin: false,
    }))
});

// --- [모듈화 헬퍼 1] 현재 페이지(DOM)에서 포커스 및 URL 정보를 추출 ---
async fn detect_active_page_info(page: &chromiumoxide::Page) -> Option<(String, String, bool, bool)> {
    let script = r#"
        (function() {
            try {
                window.__logis_tab_id = window.__logis_tab_id || Math.random().toString(36).substring(2);
                return JSON.stringify({
                    id: window.__logis_tab_id,
                    url: window.location.href,
                    focus: document.hasFocus ? document.hasFocus() : false,
                    visible: document.visibilityState === 'visible'
                });
            } catch(e) { return null; }
        })();
    "#;

    if let Ok(Ok(res)) = tokio::time::timeout(Duration::from_millis(300), page.evaluate(script)).await {
        if let Some(val_str) = res.into_value::<String>().ok() {
            if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&val_str) {
                let tab_id = json_val.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let tab_url = json_val.get("url").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let has_focus = json_val.get("focus").and_then(|v| v.as_bool()).unwrap_or(false);
                let is_visible = json_val.get("visible").and_then(|v| v.as_bool()).unwrap_or(false);
                
                return Some((tab_id, tab_url, has_focus, is_visible));
            }
        }
    }
    None
}

// --- [모듈화 헬퍼 2] 컨텍스트 메뉴 액션(마스킹 적용/복원)을 처리 ---
async fn handle_context_menu_action(page: &chromiumoxide::Page, app_handle: &tauri::AppHandle) {
    if let Ok(Ok(res)) = tokio::time::timeout(Duration::from_millis(300), page.evaluate("window.__logis_action || ''")).await {
        if let Some(action) = res.into_value::<String>().ok() {
            if action == "recover" || action == "mask" {
                use tauri::Manager;
                let store_opt = app_handle.state::<crate::AppState>().store.clone();
                let mut dict = Vec::new();
                
                if let Some(store) = store_opt.lock().await.as_ref() {
                    if let Ok(docs) = store.get_all_items("items", 10000, 0, Some("is_masked = true".to_string())).await {
                        for doc in docs {
                            if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&doc.json_data) {
                                if let Some(matches) = json_val.get("masked").and_then(|m| m.get("matches")).and_then(|v| v.as_array()) {
                                    for m in matches {
                                        let original = m.get("value").and_then(|v| v.as_str()).unwrap_or("");
                                        let mnemonic_val = m.get("mnemonic").and_then(|v| v.as_str()).unwrap_or("");
                                        let name = m.get("name").and_then(|v| v.as_str()).unwrap_or("");
                                        if !original.is_empty() && !mnemonic_val.is_empty() {
                                            dict.push(serde_json::json!({
                                                "original": original,
                                                "mnemonic": format!("[{}: {}]", name, mnemonic_val)
                                            }));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                
                let dict_json = serde_json::to_string(&dict).unwrap_or("[]".to_string());
                let js_code = format!(r#"
                    (function() {{
                        const dict = {};
                        const direction = '{}';
                        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
                        let node;
                        let changeCount = 0;
                        while (node = walker.nextNode()) {{
                            if (node.parentElement && (node.parentElement.tagName === 'SCRIPT' || node.parentElement.tagName === 'STYLE')) continue;
                            
                            let text = node.nodeValue;
                            let changed = false;
                            for (const item of dict) {{
                                const target = direction === 'recover' ? item.mnemonic : item.original;
                                const replacement = direction === 'recover' ? item.original : item.mnemonic;
                                if (text.includes(target)) {{
                                    text = text.split(target).join(replacement);
                                    changed = true;
                                }}
                            }}
                            if (changed) {{
                                node.nodeValue = text;
                                changeCount++;
                            }}
                        }}
                        window.__logis_action = '';
                        return changeCount;
                    }})();
                "#, dict_json, action);

                let _ = page.evaluate(js_code).await;
            }
        }
    }
}

// --- [모듈화된 Main 런타임] 브라우저 모니터링 루프 ---
fn spawn_browser_monitor(browser: Arc<Browser>, app_handle: tauri::AppHandle) {
    tokio::spawn(async move {
        let mut last_detected_url = String::new();
        let mut fail_count = 0; 
        
        let mut last_focused_tab_id = String::new();

        loop {
            if crate::utils::is_extraction_stopped() {
                println!("[AUTO] Global stop signal detected. Exiting browser monitor.");
                break;
            }

            let pages = match browser.pages().await {
                Ok(p) => { fail_count = 0; p },
                Err(e) => {
                    let err_msg = e.to_string();
                    if err_msg.contains("receiver is gone") || err_msg.contains("closed") || fail_count > 5 {
                        println!("[AUTO] Browser disconnected. Exiting monitor.");
                        break;
                    }
                    fail_count += 1;
                    tokio::time::sleep(Duration::from_millis(2000)).await;
                    continue; 
                }, 
            };

            let mut active_url = String::new();
            let mut active_tab_id = String::new();

            let mut found_focus = false;
            let mut remembered_url = String::new();
            let mut remembered_visible = false;

            let mut target_page = None;

            // 1. 활성화된 탭 탐색
            for page in pages.iter().rev() {
                if let Some((tab_id, tab_url, has_focus, is_visible)) = detect_active_page_info(page).await {
                    if tab_url.starts_with("devtools://") { continue; }

                    if has_focus {
                        last_focused_tab_id = tab_id.clone();
                        active_tab_id = tab_id.clone();
                        active_url = tab_url.clone();
                        found_focus = true;
                        target_page = Some(page.clone());
                        break;
                    }

                    if !last_focused_tab_id.is_empty() && tab_id == last_focused_tab_id {
                        remembered_url = tab_url;
                        remembered_visible = is_visible;
                        target_page = Some(page.clone());
                    }
                }
            }

            // 2. 포커스를 못 찾았을 경우 Fallback 처리
            if !found_focus {
                if !remembered_url.is_empty() && remembered_visible {
                    active_url = remembered_url;
                    active_tab_id = last_focused_tab_id.clone();
                } else {
                    last_focused_tab_id.clear();
                    
                    for page in pages.iter().rev() {
                        if let Some((tab_id, tab_url, _, is_visible)) = detect_active_page_info(page).await {
                            if tab_url.starts_with("devtools://") { continue; }

                            if is_visible {
                                last_focused_tab_id = tab_id.clone();
                                active_tab_id = tab_id.clone();
                                active_url = tab_url;
                                target_page = Some(page.clone());
                                break;
                            }
                        }
                    }

                    if active_url.is_empty() {
                        active_url = "about:blank".to_string();
                    }
                }
            }

            // 3. 컨텍스트 메뉴 조작 확인
            if let Some(page) = target_page {
                handle_context_menu_action(&page, &app_handle).await;
            }

            // 4. URL 판별 (사용하지 않는 Shop 패턴 매칭 로직 제거됨)

            {
                let mut state = LAST_DETECTED_STATE.lock().await;
                state.url = active_url.clone();
                state.tab_id = active_tab_id.clone();
                state.is_client = false;
                state.is_admin = false;
            }

            // 5. 프론트엔드 UI 상태 동기화
            let payload = json!({
                "url": active_url.clone(),
                "is_client": false,
                "is_admin": false,
                "status": "running",
                "hide_button": true
            });
            
            if active_url != last_detected_url {
                let _ = app_handle.emit("browser-match-found", &payload);
                let _ = app_handle.emit("browser-status", &payload);
                
                last_detected_url = active_url;
            } else {
                let _ = app_handle.emit("browser-status", &payload);
            }
            tokio::time::sleep(Duration::from_millis(800)).await; 
        }
        
        let mut global = GLOBAL_BROWSER.lock().await;
        *global = None;
        
        let _ = app_handle.emit("browser-match-found", json!({ "url": "", "is_client": false, "is_admin": false }));
        println!("[AUTO] Browser monitor exited cleanly.");
    });
}

async fn run_driverless_automation(browser: &str, url: &str, _script: &str, app_handle: tauri::AppHandle) -> anyhow::Result<String> {
    println!("[AUTO] Request: Driverless Automation for {} (URL: {})", browser, url);
    
    // 0. Proactively try to reconnect or reuse if global exists
    let browser_arc = {
        let mut global = GLOBAL_BROWSER.lock().await;
        if global.is_none() {
            // Try to connect to 9222 before launching
            if tokio::net::TcpStream::connect(format!("127.0.0.1:{}", CHROME_DEBUG_PORT)).await.is_ok() {
                println!("[AUTO] Found existing browser on port {}. Connecting...", CHROME_DEBUG_PORT);
                if let Ok((b, mut handler)) = Browser::connect(format!("http://127.0.0.1:{}", CHROME_DEBUG_PORT)).await {
                    let b_arc = Arc::new(b);
                    *global = Some(b_arc.clone());
                    spawn_browser_monitor(b_arc.clone(), app_handle.clone());
                    
                    tokio::spawn(async move {
                        while let Some(h) = handler.next().await { if let Err(_) = h { break; } }
                        let mut g = GLOBAL_BROWSER.lock().await; *g = None; 
                        
                    });
                }
            }
        }
        global.as_ref().cloned()
    };

    let browser_arc = if let Some(b) = browser_arc {
        println!("[AUTO] Reusing/Connected browser instance.");
        let _ = app_handle.emit("browser-status", "running");
        b
    } else {
        // 1. Find Executable Path
        let exec_path = find_browser_path(browser)
            .ok_or_else(|| anyhow!("Browser executable not found for {}", browser))?;
        
        // 2. Build Config
        let build_config = || -> anyhow::Result<BrowserConfig> {
            let port_arg = format!("--remote-debugging-port={}", CHROME_DEBUG_PORT);
            
            let mut args = vec![
                "--start-maximized", 
                "--no-first-run",
                "--disable-blink-features=AutomationControlled", // [CRITICAL] Hide automation status
                "--password-store=basic", // Prevent password manager popups
                "--no-default-browser-check",
                "--force-dark-mode", // [THEME] Enable Dark Mode by default
                "--enable-features=WebUIDarkMode", // Ensure UI elements are dark
                &port_arg,
                "--remote-allow-origins=*", 
            ];

            
            let target_url = if url.is_empty() { "about:blank" } else { url };
            args.push(target_url);

            let mut builder = BrowserConfig::builder()
                .chrome_executable(&exec_path)
                .with_head()
                .no_sandbox()
                .viewport(None);

            let tmp_root = crate::utils::paths::get_app_tmp_root(None);
            let profile_dir = tmp_root.join("browser_profiles").join(browser);
            let _ = std::fs::create_dir_all(&profile_dir);
            let mut p_str = std::fs::canonicalize(&profile_dir).unwrap_or(profile_dir).to_string_lossy().to_string();
            if p_str.starts_with(r"\\?\") { p_str = p_str[4..].to_string(); }
            
            builder = builder.user_data_dir(std::path::PathBuf::from(p_str)).args(args);
            builder.build().map_err(|e| anyhow!("Config error: {}", e))
        };

        // 3. Launch
        let (browser_manager, mut handler) = Browser::launch(build_config()?).await
            .map_err(|e| anyhow!("Launch failed: {}", e))?;

        let new_arc = Arc::new(browser_manager);
        {
            let mut global = GLOBAL_BROWSER.lock().await;
            *global = Some(new_arc.clone());
        }
        let _ = app_handle.emit("browser-status", "running");
        spawn_browser_monitor(new_arc.clone(), app_handle.clone());
        tokio::spawn(async move {
            while let Some(h) = handler.next().await { if let Err(_) = h { break; } }
            let mut global = GLOBAL_BROWSER.lock().await; *global = None; 
            
        });
        new_arc
    };

    println!("[AUTO] Targeting initial page for {}...", url);
    
    
    // 🌟 [CRITICAL FIX] 500ms 강제 지연(Sleep)을 2연속 하던 로직을 폐기하고, 
    // 50ms 간격의 초고속 폴링(Fast Polling)으로 교체하여 탭이 생성되는 즉시 반응하도록 최적화합니다.
    let mut pages = vec![];
    for _ in 0..20 {
        if let Ok(p) = browser_arc.pages().await {
            if !p.is_empty() {
                pages = p;
                break;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    
    if pages.is_empty() {
        pages = browser_arc.pages().await.unwrap_or_default();
    }
    
    let nav_target = if url.is_empty() { "about:blank" } else { url };
    
    // 🌟 [CRITICAL FIX] More 버튼 등 특정 URL 요청 시 기존 탭을 무시하고 항상 '새 탭(New Tab)'으로 엽니다.
    // 새 탭을 열면 브라우저가 강제로 화면 맨 앞으로 팝업(Focus)되는 효과도 얻을 수 있어 '반응 없음' 문제가 해결됩니다.
    let page = if nav_target == "about:blank" {
        // 단순 런처 버튼 클릭 시: 기존 활성 탭을 그대로 유지
        if let Some(last_page) = pages.last() {
            last_page.clone()
        } else {
            browser_arc.new_page(nav_target).await.map_err(|e| anyhow!("Page creation failed: {}", e))?
        }
    } else {
        // 특정 URL 이동 요청 시: 무조건 새 탭 생성 (자동 팝업 및 포커스 전환)
        let p = browser_arc.new_page(nav_target).await.map_err(|e| anyhow!("Page creation failed: {}", e))?;
        
        // 만약 방금 켜진 브라우저라서 첫 탭이 빈 화면(newtab) 하나뿐이라면 깔끔하게 닫아줍니다.
        if pages.len() == 1 {
            let first_page = pages.first().unwrap();
            let first_url = first_page.url().await.unwrap_or_default().unwrap_or_default();
            if first_url.is_empty() || first_url == "about:blank" || first_url.contains("newtab") || first_url.contains("new-tab") {
                // 🌟 [CRITICAL FIX] `first_page`는 참조자(&)이므로 소유권(Ownership) 이동을 위해 .clone()을 호출한 뒤 닫습니다.
                let _ = first_page.clone().close().await;
            }
        }
        p
    };

    // [CRITICAL STEALTH] 탐지 우회 스크립트 설정
    let _ = page.evaluate_on_new_document("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})").await;

    // 🌟 [추가] 내장 브라우저에서 Alt + 우클릭 시 마스킹 복원/적용 컨텍스트 메뉴를 띄우는 스크립트 주입
    let _ = page.evaluate_on_new_document(r#"
        window.addEventListener('contextmenu', function(e) {
            if (e.altKey) {
                e.preventDefault();
                // 기존에 떠있는 커스텀 메뉴 제거
                const existingMenu = document.getElementById('logis-custom-menu');
                if (existingMenu) existingMenu.remove();

                // 커스텀 컨텍스트 메뉴 DOM 생성
                const menu = document.createElement('div');
                menu.id = 'logis-custom-menu';
                menu.style.position = 'fixed';
                menu.style.left = e.clientX + 'px';
                menu.style.top = e.clientY + 'px';
                menu.style.backgroundColor = '#ffffff';
                menu.style.border = '1px solid #d1d5db';
                menu.style.boxShadow = '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)';
                menu.style.padding = '4px 0';
                menu.style.zIndex = '999999';
                menu.style.cursor = 'pointer';
                menu.style.fontFamily = 'sans-serif';
                menu.style.fontSize = '13px';
                menu.style.borderRadius = '6px';
                menu.style.minWidth = '150px';

                const recoverBtn = document.createElement('div');
                recoverBtn.innerText = '🔓 Masking Recover';
                recoverBtn.style.padding = '8px 16px';
                recoverBtn.style.color = '#374151';
                recoverBtn.onmouseover = () => recoverBtn.style.backgroundColor = '#f3f4f6';
                recoverBtn.onmouseout = () => recoverBtn.style.backgroundColor = '#ffffff';
                recoverBtn.onclick = () => {
                    console.log("[Logis] Masking Recover Clicked!");
                    // 🌟 [추가] 폴링 루프(Rust)가 감지할 수 있도록 액션 상태를 등록
                    window.__logis_action = 'recover';
                    menu.remove();
                };

                const applyBtn = document.createElement('div');
                applyBtn.innerText = '🔒 Apply Masking';
                applyBtn.style.padding = '8px 16px';
                applyBtn.style.color = '#374151';
                applyBtn.onmouseover = () => applyBtn.style.backgroundColor = '#f3f4f6';
                applyBtn.onmouseout = () => applyBtn.style.backgroundColor = '#ffffff';
                applyBtn.onclick = () => {
                    console.log("[Logis] Apply Masking Clicked!");
                    // 🌟 [추가] 폴링 루프(Rust)가 감지할 수 있도록 액션 상태를 등록
                    window.__logis_action = 'mask';
                    menu.remove();
                };

                menu.appendChild(recoverBtn);
                menu.appendChild(applyBtn);
                document.body.appendChild(menu);
            }
        });

        // 빈 화면 클릭 시 메뉴 닫기
        window.addEventListener('click', function() {
            const existingMenu = document.getElementById('logis-custom-menu');
            if (existingMenu) existingMenu.remove();
        });
    "#).await;

    Ok(format!("Automation Started."))
}

async fn run_driver_automation(browser: &str, url: &str, script: &str) -> anyhow::Result<String> {
    let (driver_binary, port, capabilities) = match browser {
        "firefox" => {
            let name = if cfg!(target_os = "windows") { "geckodriver.exe" } 
                       else if cfg!(target_os = "macos") { "geckodriver_mac" } 
                       else { "geckodriver" };
             (name.to_string(), DRIVER_PORT, serde_json::json!({ "browserName": "firefox" }))
        },
        "safari" => {
             if cfg!(target_os = "macos") { ("/usr/bin/safaridriver".to_string(), DRIVER_PORT, serde_json::json!({ "browserName": "safari" })) } 
             else { return Err(anyhow!("Safari is only supported on macOS")); }
        },
        _ => return Err(anyhow!("Unsupported browser for driver mode")),
    };

    if !cfg!(target_os = "windows") {
        let _ = Command::new("pkill").arg("-f").arg(&driver_binary).output();
    }

    let mut child = Command::new(&driver_binary)
        .arg(format!("--port={}", port)).stdout(Stdio::null()).stderr(Stdio::null()).spawn()
        .map_err(|e| anyhow!("Driver '{}' start failed: {}", driver_binary, e))?;

    tokio::time::sleep(Duration::from_secs(2)).await;
    let mut caps_map = serde_json::map::Map::new();
    if let Some(obj) = capabilities.as_object() { caps_map.clone_from(obj); }
    
    let client = ClientBuilder::native().capabilities(caps_map).connect(&format!("http://localhost:{}", port)).await;
    if let Err(e) = client { let _ = child.kill(); return Err(anyhow!("Failed to connect to driver: {}", e)); }
    let client = client.unwrap();

    let res = async {
        client.goto(url).await?;
        let result = client.execute(script, vec![]).await?;
        Ok(format!("Driver Success ({}). Result: {}", browser, serde_json::to_string_pretty(&result).unwrap_or_default()))
    }.await;

    let _ = child.kill();
    res
}

pub async fn extract_html_from_current_tab() -> Result<String, String> {
    let target_tab_id = {
        let state = LAST_DETECTED_STATE.lock().await;
        state.tab_id.clone()
    };

    let browser_opt = {
        let global = GLOBAL_BROWSER.lock().await;
        global.as_ref().cloned()
    };
    
    if let Some(browser) = browser_opt {
        let pages = browser.pages().await.map_err(|e| e.to_string())?;
        let mut active_page = None;

        if !target_tab_id.is_empty() {
            for page in pages.iter().rev() {
                let script = "window.__logis_tab_id || ''";
                if let Ok(res) = page.evaluate(script).await {
                    if res.into_value::<String>().unwrap_or_default() == target_tab_id {
                        active_page = Some(page);
                        break;
                    }
                }
            }
        }

        if active_page.is_none() {
            for page in pages.iter().rev() {
                if let Ok(res) = page.evaluate("document.hasFocus()").await {
                    if res.into_value::<bool>().unwrap_or(false) {
                        active_page = Some(page);
                        break;
                    }
                }
            }
        }

        if active_page.is_none() {
            for page in pages.iter().rev() {
                let is_visible = match page.evaluate("document.visibilityState").await {
                    Ok(res) => res.into_value::<String>().unwrap_or_default() == "visible",
                    Err(_) => false,
                };
                if is_visible {
                    active_page = Some(page);
                    break;
                }
            }
        }

        let target_page = active_page.or(pages.last());

        if let Some(page) = target_page {
            let js_script = r#"
                (function() {
                    try {
                        const elements = document.querySelectorAll('*');
                
                        elements.forEach(el => {
                            const style = window.getComputedStyle(el);
                            let newStyle = '';
                            if (style.position === 'absolute' || style.position === 'fixed') {
                                newStyle += ` position: ${style.position};`;
                            }
                            if (style.display) {
                                newStyle += ` display: ${style.display};`;
                            }
                            
                            if (newStyle) {
                                const currentStyle = el.getAttribute('style') || '';
                                el.setAttribute('data-logis-original-style', currentStyle);
                                el.setAttribute('style', currentStyle + newStyle);
                            }
                        });
                        
                        const clone = document.documentElement.cloneNode(true);
                        
                        document.querySelectorAll('iframe').forEach(iframe => {
                            try {
                                if (iframe.contentDocument && iframe.contentDocument.documentElement) {
                                    const iframeClone = iframe.contentDocument.documentElement.cloneNode(true);
                                    clone.appendChild(iframeClone);
                                }
                            } catch(e) {}
                        });

                        document.querySelectorAll('[data-logis-original-style]').forEach(el => {
                            const original = el.getAttribute('data-logis-original-style');
                            if (original) {
                                el.setAttribute('style', original);
                            } else {
                                el.removeAttribute('style');
                            }
                            el.removeAttribute('data-logis-original-style');
                        });
                        
                        clone.querySelectorAll('[data-logis-original-style]').forEach(el => {
                            el.removeAttribute('data-logis-original-style');
                        });
                        
                        return clone.outerHTML;
                    } catch(e) {
                        return document.documentElement.outerHTML;
                    }
                })();
            "#;

            let eval_result = page.evaluate(js_script).await.map_err(|e| e.to_string())?;
            let html = eval_result.into_value::<String>().unwrap_or_default();
            
            return Ok(html);
        }
        Err("No open tabs found.".to_string())
    } else {
        Err("Browser is not running.".to_string())
    }
}

fn is_in_path(cmd: &str) -> bool {
    let check_cmd = if cfg!(target_os = "windows") { "where" } else { "which" };
    Command::new(check_cmd).arg(cmd).stdout(Stdio::null()).stderr(Stdio::null()).status().map(|s| s.success()).unwrap_or(false)
}

fn check_file_exists(path: &str) -> bool { std::path::Path::new(path).exists() }

#[cfg(target_os = "windows")]
fn find_path_in_registry(exe_name: &str) -> Option<String> {
    let queries = [
        format!(r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{}", exe_name),
        format!(r"HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{}", exe_name),
    ];
    for q in queries {
        if let Ok(output) = Command::new("reg").args(["query", &q, "/ve"]).output() {
             if output.status.success() {
                 let s = String::from_utf8_lossy(&output.stdout);
                 if let Some(line) = s.lines().find(|l| l.contains("REG_SZ")) {
                     if let Some(path_part) = line.split("REG_SZ").nth(1) { return Some(path_part.trim().to_string()); }
                 }
             }
        }
    }
    None
}
#[cfg(not(target_os = "windows"))]
fn find_path_in_registry(_: &str) -> Option<String> { None }

#[cfg(target_os = "macos")]
fn find_app_bundle(bundle_id: &str) -> Option<String> {
    let output = Command::new("mdfind").args([format!("kMDItemCFBundleIdentifier == '{}'", bundle_id)]).output();
    if let Ok(o) = output {
        let s = String::from_utf8_lossy(&o.stdout);
        if !s.trim().is_empty() {
             let app_path = s.lines().next().unwrap().trim();
             let binary_name = if bundle_id.contains("Chrome") { "Google Chrome" } else if bundle_id.contains("Edge") { "Microsoft Edge" } else if bundle_id.contains("Firefox") { "firefox" } else { return None };
             if binary_name == "firefox" { return Some(format!("{}/Contents/MacOS/firefox", app_path)); }
             return Some(format!("{}/Contents/MacOS/{}", app_path, binary_name));
        }
    }
    None
}
#[cfg(not(target_os = "macos"))]
fn find_app_bundle(_: &str) -> Option<String> { None }

fn find_browser_path(browser: &str) -> Option<PathBuf> {
    let mut potential_paths = Vec::new();
    match browser {
        "chrome" => {
            if cfg!(target_os = "windows") {
                potential_paths.push(r"C:\Program Files\Google\Chrome\Application\chrome.exe".to_string());
                potential_paths.push(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe".to_string());
                if let Some(p) = find_path_in_registry("chrome.exe") { potential_paths.push(p); }
            } else if cfg!(target_os = "macos") {
                potential_paths.push("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome".to_string());
                if let Some(p) = find_app_bundle("com.google.Chrome") { potential_paths.push(p); }
            } else {
                if let Ok(p) = which::which("google-chrome") { return Some(p); }
                if let Ok(p) = which::which("google-chrome-stable") { return Some(p); }
                if let Ok(p) = which::which("chromium") { return Some(p); }
                if let Ok(p) = which::which("chrome") { return Some(p); }
            }
        },
        "edge" => {
            if cfg!(target_os = "windows") {
                potential_paths.push(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe".to_string());
                if let Some(p) = find_path_in_registry("msedge.exe") { potential_paths.push(p); }
            } else if cfg!(target_os = "macos") {
                potential_paths.push("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge".to_string());
                if let Some(p) = find_app_bundle("com.microsoft.edgemac") { potential_paths.push(p); }
            } else {
                if let Ok(p) = which::which("microsoft-edge") { return Some(p); }
                if let Ok(p) = which::which("edge") { return Some(p); }
            }
        },
        "firefox" => {
             if cfg!(target_os = "windows") {
                 potential_paths.push(r"C:\Program Files\Mozilla Firefox\firefox.exe".to_string());
                 potential_paths.push(r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe".to_string());
             } else if cfg!(target_os = "macos") {
                 potential_paths.push("/Applications/Firefox.app/Contents/MacOS/firefox".to_string());
                 if let Some(p) = find_app_bundle("org.mozilla.firefox") { potential_paths.push(p); }
             } else { if let Ok(p) = which::which("firefox") { return Some(p); } }
        },
        _ => return None,
    };
    for p in potential_paths {
        let path = PathBuf::from(&p);
        if path.exists() { return Some(path); }
    }
    match browser {
        "chrome" => which::which("chrome").ok(),
        "edge" => which::which("msedge").ok(),
        "firefox" => which::which("firefox").ok(),
        _ => None,
    }
}

pub fn get_available_browsers() -> Vec<BrowserStatus> {
    let mut browsers = Vec::new();
    if find_browser_path("chrome").is_some() {
        browsers.push(BrowserStatus { name: "chrome".to_string(), is_supported: true, is_installed: true, needs_driver: false });
    }
    if find_browser_path("edge").is_some() {
         browsers.push(BrowserStatus { name: "edge".to_string(), is_supported: true, is_installed: true, needs_driver: false });
    }
    if find_browser_path("firefox").is_some() {
        let driver_name = if cfg!(target_os = "windows") { "geckodriver.exe" } else { "geckodriver" };
        let has_driver = is_in_path(driver_name) || check_file_exists(driver_name);
        browsers.push(BrowserStatus { name: "firefox".to_string(), is_supported: true, is_installed: true, needs_driver: !has_driver });
    }
    if cfg!(target_os = "macos") {
        browsers.push(BrowserStatus { name: "safari".to_string(), is_supported: true, is_installed: true, needs_driver: true });
    }
    browsers
}

pub async fn shutdown_browser() {
    // 1. monitor task에 종료 신호 전송
    //    monitor 루프 내부의 crate::utils::is_extraction_stopped() 체크가 true를 반환하면
    //    루프를 빠져나와 Arc<Browser>를 drop합니다.
    crate::utils::set_extraction_stop_signal(true);
    println!("[APP] Stop signal sent to monitor task. Waiting for Arc release...");

    // 2. monitor task가 Arc를 drop할 때까지 strong_count 폴링 (최대 5초)
    //    monitor 루프 간격이 800ms이므로 5초면 충분합니다.
    let mut arc_released = false;
    for attempt in 0..50 {
        let count = {
            let guard = GLOBAL_BROWSER.lock().await;
            match guard.as_ref() {
                Some(arc) => Arc::strong_count(arc),
                None => {
                    println!("[APP] GLOBAL_BROWSER is already None. Nothing to close.");
                    return;
                }
            }
        };
        if count <= 1 {
            arc_released = true;
            println!("[APP] Arc strong_count dropped to {}. Monitor task released.", count);
            break;
        }
        if attempt % 10 == 0 {
            println!("[APP] Waiting for monitor task to release Arc... (strong_count={}, attempt={})", count, attempt);
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    if !arc_released {
        println!("[APP] WARNING: Arc still referenced after 5s. Proceeding with take + fallback.");
    }

    // 3. GLOBAL_BROWSER에서 Arc를 꺼내 close 시도
    let mut guard = GLOBAL_BROWSER.lock().await;
    match guard.take() {
        Some(browser_arc) => {
            match Arc::try_unwrap(browser_arc) {
                Ok(mut browser) => {
                    println!("[APP] Arc unwrapped successfully. Calling browser.close()...");
                    if let Err(e) = browser.close().await {
                        println!("[APP] browser.close() returned error (non-fatal): {:?}", e);
                    } else {
                        println!("[APP] browser.close() succeeded. Child process terminated.");
                    }
                },
                Err(_still_referenced) => {
                    // 4. 최후 폴백: OS 레벨에서 브라우저 프로세스 강제 종료
                    println!("[APP] Arc::try_unwrap still failed. Killing browser process via OS...");
                    kill_browser_processes();
                }
            }
        },
        None => {
            println!("[APP] GLOBAL_BROWSER is already None after wait. Nothing to close.");
        }
    }
    println!("[APP] shutdown_browser() finished.");
}

#[cfg(target_os = "windows")]
fn kill_browser_processes() {
    // debug port 9222로 시작된 Chrome/Edge만 선별 kill
    // wmic로 CommandLine에 remote-debugging-port=9222가 포함된 프로세스만 종료
    let output = std::process::Command::new("wmic")
        .args([
            "process", "where",
            "CommandLine like '%--remote-debugging-port=9222%'",
            "get", "ProcessId", "/format:csv",
        ])
        .output();
    if let Ok(out) = output {
        let stdout = String::from_utf8_lossy(&out.stdout);
        for line in stdout.lines() {
            let pid_str = line.trim().trim_end_matches(',');
            if let Ok(pid) = pid_str.parse::<u32>() {
                if pid > 0 {
                    println!("[APP] Killing browser PID: {}", pid);
                    let _ = std::process::Command::new("taskkill")
                        .args(["/F", "/T", "/PID", &pid.to_string()])
                        .output();
                }
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn kill_browser_processes() {
    let _ = std::process::Command::new("pkill")
        .args(["-f", "--remote-debugging-port=9222"])
        .output();
    println!("[APP] pkill sent for debug-port browser processes.");
}

#[cfg(target_os = "linux")]
fn kill_browser_processes() {
    let _ = std::process::Command::new("pkill")
        .args(["-f", "--remote-debugging-port=9222"])
        .output();
    println!("[APP] pkill sent for debug-port browser processes.");
}