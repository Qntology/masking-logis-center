use chromiumoxide::browser::{Browser, BrowserConfig};
use chromiumoxide::handler::viewport::Viewport;
use chromiumoxide::cdp::browser_protocol::page::AddScriptToEvaluateOnNewDocumentParams;
use chromiumoxide::cdp::browser_protocol::target::EventTargetCreated;
use chromiumoxide::cdp::js_protocol::runtime::{AddBindingParams, EventBindingCalled};
use futures::StreamExt;
use std::process::Stdio;
use tokio::process::Command;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;

// Overlay Script for Injection
const OVERLAY_SCRIPT: &str = r#"
(function() {
    console.log("[Gemini] Injection script started");
    if (window.geminiSidebarLoaded) {
        console.log("[Gemini] Sidebar already loaded, skipping");
        return;
    }
    window.geminiSidebarLoaded = true;

    try {
        const host = document.createElement('div');
        host.id = 'gemini-agent-host';
        // Ensure host itself doesn't affect layout
        host.style.position = 'fixed';
        host.style.zIndex = '2147483647'; // Max z-index
        document.body.appendChild(host);
        
        console.log("[Gemini] Host element created and appended to body");

        const shadow = host.attachShadow({ mode: 'open' });
        console.log("[Gemini] Shadow root attached");

        shadow.innerHTML = `
            <style>
                :host {
                    all: initial; /* Reset all inherited styles for the host */
                }
                #agent-container { 
                    position: fixed; top: 0; right: 0; 
                    width: 350px; height: 100vh; z-index: 2147483647;
                    background: white !important; 
                    border-left: 1px solid #ccc !important; 
                    box-shadow: -2px 0 10px rgba(0,0,0,0.1) !important;
                    display: flex !important; 
                    flex-direction: column !important; 
                    overflow: hidden !important; 
                    font-family: sans-serif !important;
                    transition: transform 0.3s ease !important;
                    transform: translateX(100%);
                    visibility: visible !important;
                }
                #agent-container.open {
                    transform: translateX(0) !important;
                }
                #toggle-btn { 
                    position: fixed; bottom: 20px; right: 20px; 
                    width: 60px; height: 60px; background: #007bff !important; color: white !important;
                    border-radius: 50% !important; cursor: pointer !important; z-index: 2147483647;
                    display: flex !important; align-items: center !important; justify-content: center !important;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
                    border: 2px solid white !important; 
                    font-weight: bold !important; font-size: 16px !important;
                    visibility: visible !important;
                }
                header { background: #f8f9fa; padding: 15px; border-bottom: 1px solid #eee; font-weight: bold; display: flex; justify-content: space-between; color: black; }
                .content { flex: 1; padding: 15px; overflow-y: auto; font-size: 13px; line-height: 1.5; background: white; color: black; }
                .footer { padding: 15px; border-top: 1px solid #eee; background: #f8f9fa; }
                input { width: 100%; padding: 10px; box-sizing: border-box; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 4px; background: white; color: black; }
                button#run-btn { width: 100%; padding: 10px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }
                button#run-btn:hover { background: #0056b3; }
                .log-entry { margin-bottom: 8px; padding: 8px; border-radius: 4px; word-break: break-all; }
                .log-user { background: #e7f3ff; text-align: right; color: #0056b3; }
                .log-ai { background: #f1f1f1; color: #333; }
            </style>
            <button id="toggle-btn" title="Open Gemini AI">AI</button>
            <div id="agent-container">
                <header>
                    <span>Gemini Agent</span>
                    <button id="close-btn" style="background:none; border:none; cursor:pointer; font-size: 20px;">&times;</button>
                </header>
                <div class="content" id="log">
                    <div class="log-entry log-ai">준비되었습니다. 무엇을 도와드릴까요? (v0.9.1)</div>
                </div>
                <div class="footer">
                    <input type="text" id="cli-input" placeholder="명령어 또는 질문 입력...">
                    <button id="run-btn">실행</button>
                </div>
            </div>
        `;

        const container = shadow.querySelector('#agent-container');
        const toggleBtn = shadow.querySelector('#toggle-btn');
        const closeBtn = shadow.querySelector('#close-btn');
        const input = shadow.querySelector('#cli-input');
        const runBtn = shadow.querySelector('#run-btn');
        const log = shadow.querySelector('#log');

        console.log("[Gemini] UI Elements initialized within shadow root");

        const addLog = (text, role) => {
            const div = document.createElement('div');
            div.className = `log-entry log-${role}`;
            div.innerText = text;
            log.appendChild(div);
            log.scrollTop = log.scrollHeight;
        };

        toggleBtn.onclick = () => {
            console.log("[Gemini] Sidebar opened");
            container.classList.add('open');
            toggleBtn.style.display = 'none';
        };

        closeBtn.onclick = () => {
            console.log("[Gemini] Sidebar closed");
            container.classList.remove('open');
            toggleBtn.style.display = 'flex';
        };

        window.addEventListener('gemini_rpc_response', (e) => {
            console.log("[Gemini] Received RPC response:", e.detail);
            addLog(e.detail, 'ai');
        });

        runBtn.onclick = async () => {
            const cmd = input.value;
            if (!cmd) return;
            
            console.log("[Gemini] Sending command:", cmd);
            addLog(cmd, 'user');
            input.value = '';
            
            if (window.gemini_rpc) {
                try {
                    window.gemini_rpc(cmd);
                } catch (e) {
                    console.error("[Gemini] RPC call failed:", e);
                    addLog("에러: " + e, 'ai');
                }
            } else {
                console.warn("[Gemini] window.gemini_rpc not found");
                addLog("RPC 연결 실패 (백엔드 확인 필요)", 'ai');
            }
        };

        input.onkeypress = (e) => {
            if (e.key === 'Enter') runBtn.click();
        };
        
        console.log("[Gemini] Sidebar setup complete and visible");
    } catch (err) {
        console.error("[Gemini] Critical error during injection:", err);
    }
})();
"#;

async fn execute_cli(command: String) -> Result<String, String> {
    let path = get_creds_path()?;
    
    let output = Command::new("node")
        .arg("../../../cli/omg/index.js")
        .arg(&command)
        .env("GEMINI_AUTH_FILE", path.to_str().unwrap())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|e: std::io::Error| e.to_string())?;
    
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}

fn get_creds_path() -> Result<PathBuf, String> {
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).map_err(|e| e.to_string())?;
    let mut path = PathBuf::from(home);
    path.push(".gemini");
    path.push("oauth_creds.json");
    Ok(path)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = BrowserConfig::builder()
        .with_head()
        .viewport(Viewport {
            width: 1280,
            height: 720,
            ..Default::default()
        })
        .build()?;

    let (browser, mut handler) = Browser::launch(config).await?;
    let browser = Arc::new(browser);

    tokio::task::spawn(async move {
        while let Some(h) = handler.next().await {
            if let Err(e) = h {
                eprintln!("Handler error: {:?}", e);
                break;
            }
        }
    });

    println!("Gemini Browser Controller 시작됨.");

    let mut target_events = browser.event_listener::<EventTargetCreated>().await?;
    let browser_clone = browser.clone();

    tokio::task::spawn(async move {
        while let Some(event) = target_events.next().await {
            let target_id = event.target_info.target_id.clone();
            
            if let Ok(page) = browser_clone.get_page(target_id).await {
                // 사이드바 스크립트 등록
                let _ = page.execute(AddScriptToEvaluateOnNewDocumentParams::new(OVERLAY_SCRIPT.to_string())).await;
                
                // RPC 바인딩 등록
                let _ = page.execute(AddBindingParams::new("gemini_rpc")).await;
                
                // 바인딩 호출 이벤트 리스닝
                let mut bindings = page.event_listener::<EventBindingCalled>().await.unwrap();
                let page_clone = page.clone();
                tokio::task::spawn(async move {
                    while let Some(binding_event) = bindings.next().await {
                        if binding_event.name == "gemini_rpc" {
                            let cmd = binding_event.payload.clone();
                            let response = match execute_cli(cmd).await {
                                Ok(res) => res,
                                Err(e) => format!("Error: {}", e),
                            };
                            
                            // 결과를 다시 JS 이벤트로 전달
                            let script = format!(
                                "window.dispatchEvent(new CustomEvent('gemini_rpc_response', {{ detail: {} }}));",
                                json!(response)
                            );
                            let _ = page_clone.evaluate(script).await;
                        }
                    }
                });
            }
        }
    });

    let _page = browser.new_page("https://www.google.com").await?;

    tokio::signal::ctrl_c().await?;
    println!("종료 중...");

    Ok(())
}
