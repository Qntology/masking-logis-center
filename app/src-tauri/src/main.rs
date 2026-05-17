use chromiumoxide::browser::{Browser, BrowserConfig};
use chromiumoxide::cdp::browser_protocol::page::{AddScriptToEvaluateOnNewDocumentParams, EnableParams};
use chromiumoxide::cdp::browser_protocol::target::{EventTargetCreated, SetDiscoverTargetsParams};
use chromiumoxide::cdp::js_protocol::runtime::{AddBindingParams, EventBindingCalled};
use futures::StreamExt;
use tokio::process::Command;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use include_dir::{include_dir, Dir};

static CLI_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/../../../cli/omg");

fn extract_cli() -> PathBuf {
    let temp_dir = std::env::temp_dir().join("gemini-cli-bundle");
    // 추출 실패 시 패닉 대신 로그 출력
    if let Err(e) = CLI_DIR.extract(&temp_dir) {
        eprintln!("[Rust] Critical: Failed to extract CLI: {:?}", e);
    }
    temp_dir
}

const OVERLAY_SCRIPT: &str = r#"
(function() {
    if (window.geminiSidebarLoaded) return;
    window.geminiSidebarLoaded = true;

    function initUI() {
        const existing = document.getElementById('gemini-agent-host');
        if (existing) existing.remove();

        const host = document.createElement('div');
        host.id = 'gemini-agent-host';
        host.style.cssText = 'position:fixed; top:0; right:0; bottom:0; width:0; z-index:2147483647; pointer-events:none;';
        document.body.appendChild(host);
        
        const shadow = host.attachShadow({ mode: 'open' });
        shadow.innerHTML = `
            <style>
                #agent-container { 
                    position: fixed; top: 0; right: 0; bottom: 0; 
                    width: 350px; z-index: 2147483647;
                    background: white; border-left: 1px solid #ccc;
                    display: flex; flex-direction: column;
                    transition: transform 0.3s ease; transform: translateX(100%);
                    box-shadow: -5px 0 15px rgba(0,0,0,0.1);
                    visibility: visible !important; pointer-events: auto;
                }
                #agent-container.open { transform: translateX(0); }
                #toggle-btn { 
                    position: fixed; bottom: 30px; right: 30px; 
                    width: 60px; height: 60px; background: #007bff; color: white;
                    border-radius: 50%; cursor: pointer; z-index: 2147483647;
                    display: flex; align-items: center; justify-content: center;
                    border: 4px solid white; font-weight: bold; box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                    pointer-events: auto;
                }
                header { padding: 15px; background: #f0f0f0; font-weight: bold; color: #000; border-bottom: 1px solid #ddd; flex-shrink: 0; }
                .content { flex: 1; padding: 15px; overflow-y: auto; background: #fff; color: #000; box-sizing: border-box; }
                .footer { padding: 15px; background: #f8f9fa; border-top: 1px solid #eee; flex-shrink: 0; }
                input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
            </style>
            <button id="toggle-btn">AI</button>
            <div id="agent-container">
                <header>Gemini Agent (Native Mode)</header>
                <div class="content" id="log"><div>연결됨.</div></div>
                <div class="footer">
                    <input type="text" id="cli-input" placeholder="메시지 입력...">
                </div>
            </div>
        `;

        const container = shadow.querySelector('#agent-container');
        const toggleBtn = shadow.querySelector('#toggle-btn');
        const input = shadow.querySelector('#cli-input');
        const log = shadow.querySelector('#log');

        toggleBtn.onclick = () => { 
            console.log("[Gemini-JS] The DevTools panel has been deprecated.");
            alert('상시 최상단 Tauri 앱 인터페이스를 이용해 주세요. 개발자 도구가 필요하면 F12를 직접 눌러주세요.');
        };
        
        input.onkeypress = async (e) => {
            if (e.key === 'Enter' && input.value) {
                const cmd = input.value;
                input.value = '';
                log.innerHTML += `<div><strong>You:</strong> ${cmd}</div>`;
                if (window.gemini_rpc) window.gemini_rpc(cmd);
            }
        };

        window.addEventListener('gemini_rpc_response', (e) => {
            log.innerHTML += `<div style="color:blue"><strong>AI:</strong> ${e.detail}</div>`;
        });
    }

    if (document.readyState === 'complete') initUI();
    else window.addEventListener('load', initUI);
})();
"#;

async fn setup_page(browser: Arc<Browser>, page: chromiumoxide::Page) -> Result<(), Box<dyn std::error::Error>> {
    page.execute(EnableParams::default()).await?;
    page.execute(AddBindingParams::new("gemini_rpc")).await?;
    page.execute(AddScriptToEvaluateOnNewDocumentParams::new(OVERLAY_SCRIPT.to_string())).await?;

    let mut bindings = page.event_listener::<EventBindingCalled>().await?;
    let page_clone = page.clone();

    tokio::task::spawn(async move {
        while let Some(event) = bindings.next().await {
            if event.name == "gemini_rpc" {
                let raw_command = event.payload.trim_matches('"').to_string();

                if raw_command == "open_devtools" {
                    let _ = page_clone.bring_to_front().await;
                    let script = "alert('개발자 도구 패널 대신 최상단 Tauri 앱을 사용해 주세요.');";
                    let _ = page_clone.evaluate(script).await;
                    continue;
                }

                let response = match execute_cli(raw_command).await {
                    Ok(res) => res,
                    Err(e) => format!("Error: {}", e),
                };
                // dispatchEvent와 동시에 panel.html의 setInterval이 감지할 수 있도록 전역 변수에 응답을 기록합니다.
                let response_json = json!(response);
                let script = format!(
                    r#"
                    window.last_gemini_response = {};
                    window.dispatchEvent(new CustomEvent('gemini_rpc_response', {{ detail: {} }}));
                    "#,
                    response_json,
                    response_json
                );
                let _ = page_clone.evaluate(script).await;
            }
        }
    });
    Ok(())
}

async fn execute_cli(command: String) -> Result<String, String> {
    let cli_path = extract_cli();
    let index_js = cli_path.join("index.js");
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).map_err(|e| e.to_string())?;
    let path = PathBuf::from(home).join(".gemini/oauth_creds.json");
    
    let output = Command::new("node")
        .arg(index_js)
        .arg(&command)
        .env("GEMINI_AUTH_FILE", path.to_str().unwrap())
        .output().await.map_err(|e| e.to_string())?;
    
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}
use tauri::Manager;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_oauth::init())
        .setup(|app| {
            // 메인 윈도우 객체를 가져옵니다.
            let window = app.get_webview_window("main").unwrap();
            
            // 앱 실행 시 윈도우를 항상 최상단에 노출하도록 설정합니다.
            window.set_always_on_top(true).unwrap();
            
            // 기존의 브라우저 제어 로직을 비동기 런타임에서 별도로 실행합니다.
            tauri::async_runtime::spawn(async move {
                if let Err(e) = run_browser_service().await {
                    eprintln!("[Rust] Browser Service Error: {:?}", e);
                }
            });
            
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

async fn run_browser_service() -> Result<(), Box<dyn std::error::Error>> {
    println!("[Rust] Initializing Gemini Browser...");
    
    let port = 9222; 
    let port_arg = format!("--remote-debugging-port={}", port);
    let mut current_dir = std::env::current_dir().expect("Failed to get current dir");
    if current_dir.ends_with("src-tauri") {
        current_dir = current_dir.parent().unwrap().to_path_buf();
    }

    let config = BrowserConfig::builder()
        .with_head()
        .arg("--disable-device-emulation")
        .arg("--start-maximized")
        .arg("--no-first-run")
        .arg("--no-sandbox")
        .arg("--disable-setuid-sandbox")
        .arg(port_arg)
        .arg("--remote-allow-origins=*")
        .arg("--disable-web-security")
        .arg("--user-data-dir=C:\\temp\\gemini-profile")
        .arg("--disable-blink-features=AutomationControlled")
        .arg("--use-fake-ui-for-media-stream")
        .build()?;

    let (browser, mut handler) = Browser::launch(config).await?;
    let browser = Arc::new(browser);
    let (tx, mut rx) = tokio::sync::mpsc::channel::<()>(1);
    
    tokio::task::spawn(async move {
        while let Some(h) = handler.next().await {
            if let Err(e) = h { eprintln!("[Rust] Handler Error: {:?}", e); break; }
        }
        let _ = tx.send(()).await;
    });

    browser.execute(SetDiscoverTargetsParams::new(true)).await?;
    let mut target_events = browser.event_listener::<EventTargetCreated>().await?;
    let b_target = Arc::clone(&browser);
    
    tokio::task::spawn(async move {
        while let Some(event) = target_events.next().await {
            // TargetInfo 내부의 필드 접근 방식을 확인하여 수정
            let target_info = &event.target_info;
            if target_info.r#type == "page" {
                let tid = target_info.target_id.clone();
                let b_inner = Arc::clone(&b_target);
                tokio::task::spawn(async move {
                    // 페이지가 완전히 로드될 때까지 약간의 대기 시간을 가짐
                    tokio::time::sleep(tokio::time::Duration::from_millis(800)).await;
                    if let Ok(page) = b_inner.get_page(tid).await {
                        if let Err(e) = setup_page(Arc::clone(&b_inner), page).await {
                            eprintln!("[Rust] Error setting up page: {:?}", e);
                        }
                    }
                });
            }
        }
    });

    let initial_page = browser.new_page("https://www.google.com").await?;
    setup_page(browser.clone(), initial_page).await?;

    rx.recv().await;
    Ok(())
}