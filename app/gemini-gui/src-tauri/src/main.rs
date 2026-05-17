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
    if !temp_dir.exists() {
        CLI_DIR.extract(&temp_dir).expect("Failed to extract CLI");
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
            console.log("[Gemini-JS] Requesting DevTools...");
            if (window.gemini_rpc) window.gemini_rpc("open_devtools");
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

async fn setup_page(page: chromiumoxide::Page) -> Result<(), Box<dyn std::error::Error>> {
    page.execute(EnableParams::default()).await?;
    page.execute(AddBindingParams::new("gemini_rpc")).await?;
    page.execute(AddScriptToEvaluateOnNewDocumentParams::new(OVERLAY_SCRIPT.to_string())).await?;
    let _ = page.evaluate(OVERLAY_SCRIPT).await;

    let mut bindings = page.event_listener::<EventBindingCalled>().await?;
    let page_clone = page.clone();
    tokio::task::spawn(async move {
        while let Some(event) = bindings.next().await {
            if event.name == "gemini_rpc" {
                if event.payload == "\"open_devtools\"" {
                    println!("[Rust] DevTools requested - manual access only for this native mode.");
                    continue;
                }
                
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
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("[Rust] Initializing Gemini Browser...");

    // Dynamic port handling for remote debugging
    let port = 9222; 
    let port_arg = format!("--remote-debugging-port={}", port);

    let config = BrowserConfig::builder()
        .with_head()
        .arg("--disable-device-emulation")
        .arg("--start-maximized")
        .arg("--disable-gpu")
        .arg("--disable-software-rasterizer")
        .arg("--disable-gpu-compositing")
        .arg("--no-first-run")
        .arg("--disable-notifications")
        .arg("--disable-extensions")
        .arg("--disable-popup-blocking")
        .arg("--blink-settings=imagesEnabled=false")
        .arg("--disable-blink-features=AutomationControlled")
        .arg("--password-store=basic")
        .arg("--no-default-browser-check")
        .arg("--force-dark-mode")
        .arg("--enable-features=WebUIDarkMode")
        .arg(port_arg)
        .arg("--remote-allow-origins=*")
        .build()?;

    let (browser, mut handler) = Browser::launch(config).await?;
    let browser = Arc::new(browser);

    let (tx, mut rx) = tokio::sync::mpsc::channel::<()>(1);
    
    // 핸들러 루프 태스크
    tokio::task::spawn(async move {
        while let Some(h) = handler.next().await {
            if let Err(e) = h { eprintln!("[Rust] Handler Error: {:?}", e); break; }
        }
        let _ = tx.send(()).await;
    });

    browser.execute(SetDiscoverTargetsParams::new(true)).await?;

    let mut target_events = browser.event_listener::<EventTargetCreated>().await?;
    let b_target = browser.clone();
    
    tokio::task::spawn(async move {
        while let Some(event) = target_events.next().await {
            if event.target_info.r#type == "page" {
                let tid = event.target_info.target_id.clone();
                let b_inner = b_target.clone();
                tokio::task::spawn(async move {
                    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                    if let Ok(page) = b_inner.get_page(tid).await {
                        let _ = setup_page(page).await;
                    }
                });
            }
        }
    });

    let initial_page = browser.new_page("https://www.google.com").await?;
    setup_page(initial_page).await?;

    tokio::select! {
        _ = tokio::signal::ctrl_c() => println!("\n[Rust] Shutting down..."),
        _ = rx.recv() => println!("\n[Rust] Browser closed, shutting down..."),
    }
    
    Ok(())
}
