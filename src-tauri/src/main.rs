mod db;
use chromiumoxide::browser::{Browser, BrowserConfig};
use chromiumoxide::cdp::browser_protocol::page::{AddScriptToEvaluateOnNewDocumentParams, EnableParams};
use chromiumoxide::cdp::browser_protocol::target::{EventTargetCreated, SetDiscoverTargetsParams};
use chromiumoxide::cdp::js_protocol::runtime::{AddBindingParams, EventBindingCalled};
use futures::StreamExt;
use tokio::process::Command;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;

fn extract_cli() -> PathBuf {
    PathBuf::new() 
}

const OVERLAY_SCRIPT: &str = r#"
(function() {
    if (window.geminiSidebarLoaded) return;
    window.geminiSidebarLoaded = true;

    function cleanAndConvertToYaml(node, depth = 0) {
        let yaml = '';
        const indent = '  '.repeat(depth);
        if (node.nodeType === Node.TEXT_NODE) {
            const text = node.textContent.trim();
            if (text) return text + '\n';
            return '';
        }
        if (node.nodeType === Node.ELEMENT_NODE) {
            if (['script', 'style', 'link', 'meta', 'noscript', 'svg', 'iframe'].includes(node.tagName.toLowerCase())) return '';
            const tagName = node.tagName.toLowerCase();
            yaml += `${indent}${tagName}:\n`;
            node.childNodes.forEach(child => { yaml += cleanAndConvertToYaml(child, depth + 1); });
        }
        return yaml;
    }

    async function generatePageId(url) {
        const msgUint8 = new TextEncoder().encode(url);
        const hashBuffer = await crypto.subtle.digest('SHA-256', msgUint8);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    }

    // 데이터 중심 정제 및 변환 함수
    function cleanAndConvertToYaml(node, depth = 0) {
        if (node.nodeType === Node.TEXT_NODE) {
            const text = node.textContent.trim();
            return text ? text + '\n' : '';
        }
        if (node.nodeType === Node.ELEMENT_NODE) {
            const ignoredTags = ['script', 'style', 'link', 'meta', 'noscript', 'svg', 'iframe', 'head', 'title', 'button', 'input', 'nav', 'footer'];
            if (ignoredTags.includes(node.tagName.toLowerCase())) return '';

            let childYaml = '';
            node.childNodes.forEach(child => {
                childYaml += cleanAndConvertToYaml(child, depth);
            });

            if (!childYaml.trim()) return '';

            const indent = '  '.repeat(depth);
            if (node.childNodes.length === 1 && node.childNodes[0].nodeType === Node.TEXT_NODE) {
                 return `${indent}- ${childYaml.trim()}\n`;
            } else {
                 return childYaml;
            }
        }
        return '';
    }

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
                header { padding: 15px; background: #f0f0f0; font-weight: bold; color: #000; border-bottom: 1px solid #ddd; display: flex; justify-content: space-between; align-items: center; }
                .content { flex: 1; padding: 15px; overflow-y: auto; background: #fff; color: #000; box-sizing: border-box; }
                .footer { padding: 15px; background: #f8f9fa; border-top: 1px solid #eee; flex-shrink: 0; }
                input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
                .staged-item { display: flex; align-items: center; margin-bottom: 10px; }
            </style>
            <button id="toggle-btn">AI</button>
            <div id="agent-container">
                <header>Staging Area (Dexie) <button id="close-btn">X</button></header>
                <div class="content" id="staged-list">
                    <div id="log"></div>
                </div>
                <div class="footer">
                    <button id="extract-btn">추출</button>
                    <button id="push-btn">Push Selected</button>
                    <input type="text" id="cli-input" placeholder="메시지 입력...">
                </div>
            </div>
        `;

        const stagedList = shadow.querySelector('#staged-list');
        const pushBtn = shadow.querySelector('#push-btn');
        const extractBtn = shadow.querySelector('#extract-btn');
        const toggleBtn = shadow.querySelector('#toggle-btn');
        const closeBtn = shadow.querySelector('#close-btn');
        const log = shadow.querySelector('#log');

        let stagedItems = [];

        // 자동 추출 함수
        async function autoExtract() {
            const pageId = await generatePageId(window.location.href);
            const yaml = cleanAndConvertToYaml(document.body);
            
            // 1. UI 로그에 YAML 표시
            log.innerHTML += `<div style="white-space: pre-wrap; font-size: 11px; margin-top: 5px;"><strong>[Auto-Extracted]:</strong>\n${yaml.substring(0, 100)}...</div>`;
            
            // 2. Dexie Staging 영역에 추가
            const item = { id: pageId, host: window.location.hostname, url: window.location.href, context: yaml, status: 'DRAFT' };
            stagedItems.push(item);
            
            const div = document.createElement('div');
            div.className = 'staged-item';
            div.innerHTML = `<input type="checkbox" data-id="${pageId}"> ${pageId.substring(0,8)}...`;
            stagedList.appendChild(div);
            
            // 3. Rust 백엔드로 전송
            if (window.gemini_rpc) window.gemini_rpc("sync_data:" + JSON.stringify(item));
        }

        toggleBtn.onclick = () => { shadow.querySelector('#agent-container').classList.toggle('open'); };
        closeBtn.onclick = () => { shadow.querySelector('#agent-container').classList.remove('open'); };
        
        extractBtn.onclick = autoExtract;

        pushBtn.onclick = () => {
            const selected = Array.from(shadow.querySelectorAll('input:checked')).map(cb => cb.dataset.id);
            const payload = stagedItems.filter(i => selected.includes(i.id));
            if (window.gemini_rpc) window.gemini_rpc("push_data:" + JSON.stringify(payload));
        };

        // 자동 실행
        window.addEventListener('load', autoExtract);
        window.addEventListener('pageshow', autoExtract);

        window.addEventListener('gemini_rpc_response', (e) => {
            log.innerHTML += `<div style="color:blue"><strong>AI:</strong> ${e.detail}</div>`;
        });
    }

    // 모든 탭/페이지 이동마다 UI 재초기화
    window.addEventListener('load', initUI);
    window.addEventListener('pageshow', initUI);

})();
"#;

async fn setup_page(browser: Arc<Browser>, page: chromiumoxide::Page) -> Result<(), Box<dyn std::error::Error>> {
    // screen_width/height를 추가하여 윈도우 내부의 가용 영역을 꽉 채우도록 설정합니다.
    page.execute(EnableParams::default()).await?;
    page.execute(AddBindingParams::new("gemini_rpc")).await?;
    page.execute(AddScriptToEvaluateOnNewDocumentParams::new(OVERLAY_SCRIPT.to_string())).await?;
    let _ = page.evaluate(OVERLAY_SCRIPT).await;
    let mut bindings = page.event_listener::<EventBindingCalled>().await?;
    let page_clone = page.clone();
    let browser_clone = browser.clone();
    tokio::task::spawn(async move {
        while let Some(event) = bindings.next().await {
            if event.name == "gemini_rpc" {
                let payload = event.payload.trim_matches('"');
                let response = if payload.starts_with("sync_data:") {
                    format!("Data staged.")
                } else if payload.starts_with("push_data:") {
                    let data = &payload["push_data:".len()..];
                    match serde_json::from_str::<Vec<db::CommerceRecord>>(data) {
                        Ok(records) => {
                            match db::save_records(records).await {
                                Ok(_) => "Data pushed successfully.".to_string(),
                                Err(e) => format!("DB Error: {}", e),
                            }
                        },
                        Err(e) => format!("JSON Error: {}", e),
                    }
                } else if payload == "open_devtools" {
                    let tid = page_clone.target_id().clone();
                    let url = format!("devtools://devtools/bundled/inspector.html?ws=localhost:9222/devtools/page/{:?}", tid);
                    let _ = browser_clone.execute(chromiumoxide::cdp::browser_protocol::target::CreateTargetParams::new(url)).await;
                    "DevTools opened".to_string()
                } else {
                    match execute_cli(payload.to_string()).await {
                        Ok(res) => res,
                        Err(e) => format!("Error: {}", e),
                    }
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
    let output = Command::new("node").arg(index_js).arg(&command).env("GEMINI_AUTH_FILE", path.to_str().unwrap()).output().await.map_err(|e| e.to_string())?;
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let port = 9222;
    let port_arg = format!("--remote-debugging-port={}", port);
    
    // 사용자가 요청한 클로저 기반 설정 로직 반영
    let mut args = vec![
        "--window-size=1920,1080", // 창 크기 강제 지정
        "--window-position=0,0",
        "--start-maximized", 
        "--disable-gpu", 
        "--disable-software-rasterizer",
        "--disable-gpu-compositing",
        "--no-first-run",
        "--disable-notifications",
        "--disable-extensions",
        "--disable-popup-blocking",
        "--blink-settings=imagesEnabled=false",
        "--disable-blink-features=AutomationControlled",
        "--password-store=basic",
        "--no-default-browser-check",
        "--force-dark-mode",
        "--enable-features=WebUIDarkMode",
        "--remote-allow-origins=*",
        "--disable-dev-shm-usage",
    ];
    
    // 원격 디버깅 포트 추가
    let port_arg_str = port_arg.as_str();
    args.push(port_arg_str);

    // 인증 여부에 따른 초기 URL 설정 및 args 추가
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).unwrap_or_default();
    let auth_path = std::path::PathBuf::from(home).join(".gemini/oauth_creds.json");
    let is_authenticated = auth_path.exists();
    let target_url = if is_authenticated { "https://www.google.com" } else { "https://aistudio.google.com/" };
    args.push(target_url);

    let config = BrowserConfig::builder()
        .with_head()
        .no_sandbox()
        .viewport(None) // 뷰포트 제한 해제
        .args(args)
        .build()
        .map_err(|e| format!("Config error: {}", e))?;
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
    let b_target = browser.clone();
    tokio::task::spawn(async move {
        while let Some(event) = target_events.next().await {
            if event.target_info.r#type == "page" {
                let tid = event.target_info.target_id.clone();
                let b_inner = b_target.clone();
                tokio::task::spawn(async move {
                    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                    if let Ok(page) = b_inner.get_page(tid).await {
                        let _ = setup_page(b_inner.clone(), page).await;
                    }
                });
            }
        }
    });
    // 인증 체크
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).unwrap_or_default();
    let auth_path = PathBuf::from(home).join(".gemini/oauth_creds.json");
    let is_authenticated = auth_path.exists();

    let start_url = if is_authenticated {
        "https://www.google.com"
    } else {
        println!("[Rust] Authentication required. Redirecting to login...");
        "https://aistudio.google.com/"
    };

    let initial_page = browser.new_page(start_url).await?;
    setup_page(browser.clone(), initial_page).await?;
    tokio::select! {
        _ = tokio::signal::ctrl_c() => println!("\n[Rust] Shutting down..."),
        _ = rx.recv() => println!("\n[Rust] Browser closed, shutting down..."),
    }
    Ok(())
}
