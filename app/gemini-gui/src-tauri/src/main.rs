use chromiumoxide::browser::{Browser, BrowserConfig};
use chromiumoxide::handler::viewport::Viewport;
use chromiumoxide::cdp::browser_protocol::page::AddScriptToEvaluateOnNewDocumentParams;
use chromiumoxide::cdp::browser_protocol::target::{EventTargetCreated, SetDiscoverTargetsParams};
use chromiumoxide::cdp::js_protocol::runtime::{AddBindingParams, EventBindingCalled};
use futures::StreamExt;
use std::process::Stdio;
use tokio::process::Command;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;
use std::io::{self, Write};

// [보강된 OVERLAY_SCRIPT]
const OVERLAY_SCRIPT: &str = r#"
(function() {
    console.log("[Gemini-JS] Sidebar Script Loading...");
    if (window.geminiSidebarLoaded) return;
    window.geminiSidebarLoaded = true;

    function initUI() {
        console.log("[Gemini-JS] Initializing UI...");
        const existing = document.getElementById('gemini-agent-host');
        if (existing) existing.remove();

        const host = document.createElement('div');
        host.id = 'gemini-agent-host';
        // 호스트 자체가 공간을 차지하지 않도록 고정 위치 설정
        host.style.cssText = 'position:fixed; top:0; right:0; width:0; height:0; z-index:2147483647; overflow:visible;';
        document.body.appendChild(host);
        
        const shadow = host.attachShadow({ mode: 'open' });
        shadow.innerHTML = `
            <style>
                #agent-container { 
                    position: fixed; 
                    top: 0; 
                    right: 0; 
                    bottom: 0; /* 100vh 대신 top/bottom 0으로 전체 높이 확보 */
                    width: 350px; 
                    z-index: 2147483647;
                    background: white; 
                    border-left: 1px solid #ccc;
                    display: flex; 
                    flex-direction: column;
                    transition: transform 0.3s ease; 
                    transform: translateX(100%);
                    box-shadow: -5px 0 15px rgba(0,0,0,0.1);
                    visibility: visible !important;
                    pointer-events: auto;
                }
                #agent-container.open { transform: translateX(0); }
                #toggle-btn { 
                    position: fixed; bottom: 30px; right: 30px; 
                    width: 60px; height: 60px; background: #007bff; color: white;
                    border-radius: 50%; cursor: pointer; z-index: 2147483647;
                    display: flex; align-items: center; justify-content: center;
                    border: 4px solid white; font-weight: bold; box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                    visibility: visible !important;
                }
                header { 
                    padding: 15px; background: #f0f0f0; font-weight: bold; color: #000; 
                    border-bottom: 1px solid #ddd; flex-shrink: 0;
                }
                .content { 
                    flex: 1; padding: 15px; overflow-y: auto; 
                    background: #fff; color: #000; font-family: sans-serif;
                    box-sizing: border-box;
                }
                .footer { 
                    padding: 15px; background: #f8f9fa; border-top: 1px solid #eee; 
                    flex-shrink: 0;
                }
                input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
            </style>
            <button id="toggle-btn">AI</button>
            <div id="agent-container">
                <header>Gemini Agent (v0.9.1)</header>
                <div class="content" id="log">
                    <div style="color: #666; font-style: italic;">연결됨. 명령을 입력하세요.</div>
                </div>
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
            console.log("[Gemini-JS] Opening Sidebar");
            container.classList.add('open'); 
            toggleBtn.style.display = 'none'; 
        };
        
        input.onkeypress = async (e) => {
            if (e.key === 'Enter' && input.value) {
                const cmd = input.value;
                input.value = '';
                const div = document.createElement('div');
                div.style.marginBottom = '10px';
                div.innerHTML = '<strong>You:</strong> ' + cmd;
                log.appendChild(div);
                log.scrollTop = log.scrollHeight;
                
                if (window.gemini_rpc) {
                    window.gemini_rpc(cmd);
                }
            }
        };

        window.addEventListener('gemini_rpc_response', (e) => {
            const div = document.createElement('div');
            div.style.marginBottom = '10px';
            div.style.color = '#0056b3';
            div.innerHTML = '<strong>AI:</strong> ' + e.detail;
            log.appendChild(div);
            log.scrollTop = log.scrollHeight;
        });

        // 레이아웃 강제 업데이트 함수
        function updateLayout() {
            const h = window.innerHeight;
            container.style.height = h + 'px';
            console.log("[Gemini-JS] Layout updated: " + window.innerWidth + "x" + h);
        }

        window.addEventListener('resize', updateLayout);
        updateLayout(); // 초기 실행

        console.log("[Gemini-JS] UI Initialized with robust resizing.");
    }

    if (document.body) initUI();
    else window.addEventListener('DOMContentLoaded', initUI);
})();
"#;

async fn setup_page(page: chromiumoxide::Page) -> Result<(), Box<dyn std::error::Error>> {
    let url = page.url().await?.unwrap_or_else(|| "unknown".to_string());
    println!("[Rust] >>> Setting up page: {}", url);
    io::stdout().flush().unwrap();

    // 1. 바인딩 및 스크립트 설정
    // Runtime.addBinding
    page.execute(AddBindingParams::new("gemini_rpc")).await?;
    // Page.addScriptToEvaluateOnNewDocument
    page.execute(AddScriptToEvaluateOnNewDocumentParams::new(OVERLAY_SCRIPT.to_string())).await?;
    
    // 2. 즉시 실행 (현재 페이지가 이미 로드된 경우)
    let _ = page.evaluate(OVERLAY_SCRIPT).await;

    // 3. 바인딩 리스너
    let mut bindings = page.event_listener::<EventBindingCalled>().await?;
    let page_clone = page.clone();
    tokio::task::spawn(async move {
        while let Some(event) = bindings.next().await {
            if event.name == "gemini_rpc" {
                println!("[Rust] Received RPC call: {}", event.payload);
                io::stdout().flush().unwrap();
                
                let response = match execute_cli(event.payload.clone()).await {
                    Ok(res) => res,
                    Err(e) => format!("Error: {}", e),
                };
                
                let script = format!(
                    "window.dispatchEvent(new CustomEvent('gemini_rpc_response', {{ detail: {} }}));",
                    json!(response)
                );
                let _ = page_clone.evaluate(script).await;
            }
        }
    });

    println!("[Rust] >>> Page setup complete for {}", url);
    io::stdout().flush().unwrap();
    Ok(())
}

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
    println!("[Rust] Initializing Gemini Browser (v0.9.1)...");
    io::stdout().flush().unwrap();

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

    // [중요] 핸들러 루프를 별도 스레드에서 가장 먼저 시작
    tokio::task::spawn(async move {
        println!("[Rust] Handler loop started.");
        io::stdout().flush().unwrap();
        while let Some(h) = handler.next().await {
            if let Err(e) = h {
                eprintln!("[Rust] Handler Error: {:?}", e);
            }
        }
    });

    // 타겟 감지 활성화
    browser.execute(SetDiscoverTargetsParams::new(true)).await?;

    // 새 탭 감지 리스너
    let mut target_events = browser.event_listener::<EventTargetCreated>().await?;
    let b_clone = browser.clone();
    
    tokio::task::spawn(async move {
        println!("[Rust] Event listener started.");
        io::stdout().flush().unwrap();
        while let Some(event) = target_events.next().await {
            if event.target_info.r#type == "page" {
                let tid = event.target_info.target_id.clone();
                let b_inner = b_clone.clone();
                println!("[Rust] New page target detected: {:?}", tid);
                io::stdout().flush().unwrap();
                
                tokio::task::spawn(async move {
                    // 페이지 객체가 준비될 때까지 지연
                    tokio::time::sleep(tokio::time::Duration::from_millis(800)).await;
                    if let Ok(page) = b_inner.get_page(tid).await {
                        let _ = setup_page(page).await;
                    }
                });
            }
        }
    });

    // 초기 페이지 로드
    let initial_page = browser.new_page("https://www.google.com").await?;
    setup_page(initial_page).await?;

    println!("[Rust] Initial page loaded. Press Ctrl+C to terminate.");
    io::stdout().flush().unwrap();

    tokio::signal::ctrl_c().await?;
    println!("\n[Rust] Terminating...");
    Ok(())
}
