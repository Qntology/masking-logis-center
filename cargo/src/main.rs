use gemini_gui_lib::{db, embedding, privacy_filter, glm_ocr, params};
use privacy_filter::{PrivacyFilterModel, viterbi::PrivacySpan};
use candle_core::Device;
use glm_ocr::generate::GlmOcrGenerateModel;
use params::chat::{ChatCompletionParameters, Message, Part};
use std::sync::Mutex;
use lazy_static::lazy_static;

lazy_static! {
    static ref OCR_MODEL: Mutex<Option<GlmOcrGenerateModel>> = Mutex::new(None);
}

// Simplified stub for chat completion
async fn get_chat_completion(_messages: Vec<serde_json::Value>, _api_key: String, _model: String) -> Result<String, String> {
    Ok("[System] Gemini 서비스가 비활성화되었습니다.".to_string())
}

use chromiumoxide::browser::{Browser, BrowserConfig};
use chromiumoxide::cdp::browser_protocol::page::{AddScriptToEvaluateOnNewDocumentParams, EnableParams};
use chromiumoxide::cdp::browser_protocol::target::{EventTargetCreated, SetDiscoverTargetsParams};
use chromiumoxide::cdp::js_protocol::runtime::{AddBindingParams, EventBindingCalled};
use futures::StreamExt;
use serde_json::json;
use std::sync::Arc;

fn mask_pii(text: &str, spans: &[PrivacySpan]) -> String {
    let mut masked_text = text.to_string();
    let mut sorted_spans = spans.to_vec();
    sorted_spans.sort_by(|a, b| b.start.cmp(&a.start));

    for span in sorted_spans {
        if span.start < masked_text.len() && span.end <= masked_text.len() && span.start < span.end {
            let mask = format!("[{}]", span.entity_group.to_uppercase());
            masked_text.replace_range(span.start..span.end, &mask);
        }
    }
    masked_text
}

#[derive(serde::Deserialize)]
struct AppConfig {
    default_tab: String,
}

fn load_default_tab() -> String {
    let _ = std::fs::create_dir_all("data");
    let config_path = std::path::PathBuf::from("data/config.json");
    if let Ok(content) = std::fs::read_to_string(config_path) {
        if let Ok(config) = serde_json::from_str::<AppConfig>(&content) {
            return config.default_tab.to_uppercase();
        }
    }
    "DRAFT".to_string()
}

const OVERLAY_SCRIPT: &str = r#"
(function() {
    if (window.self !== window.top) return;
    if (window.geminiSidebarLoaded) return;
    window.geminiSidebarLoaded = true;

    async function generatePageId(url) {
        const msgUint8 = new TextEncoder().encode(url);
        const hashBuffer = await crypto.subtle.digest('SHA-256', msgUint8);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    }

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
        if (document.getElementById('gemini-agent-host')) return;
        if (!document.body) {
            window.requestAnimationFrame(initUI);
            return;
        }

        const host = document.createElement('div');
        host.id = 'gemini-agent-host';
        try {
            document.body.appendChild(host);
        } catch (e) {
            document.documentElement.appendChild(host);
        }

        const shadow = host.attachShadow({ mode: 'open' });
        const style = document.createElement('style');
        style.textContent = `
            :host { all: initial; }
            * { box-sizing: border-box !important; }
            #agent-container { 
                position: fixed; top: 50px; left: 50px; right: 50px; bottom: 50px; 
                margin: auto; border-radius: 10px; overflow: hidden;
                min-width: 300px; max-width: 760px; width:100%; z-index: 2147483648;
                background: white !important; border-left: 1px solid #ccc;
                display: flex !important; flex-direction: column;
                transition: opacity 0.3s ease-in-out; 
                box-shadow: -5px 0 15px rgba(0,0,0,0.2);
                pointer-events: none;
                opacity:0;
            }
            #agent-container.open { opacity: 1; pointer-events: auto; }
            header { padding: 12px; background: #f0f0f0 !important; font-weight: bold !important; color: #000 !important; display: flex !important; justify-content: space-between; align-items: center; flex-shrink: 0; gap: 10px; }
            .domain-filter { display: flex; gap: 5px; padding: 10px 15px; background: #fff; border-bottom: 1px solid #eee; flex-shrink: 0; overflow-x: auto; }
            .domain-filter button { padding: 4px 10px; border-radius: 20px; border: 1px solid #ddd; background: #f9f9f9; font-size: 11px; white-space: nowrap; }
            .domain-filter button.active { background: #333; color: white; border-color: #333; }
            .content { flex: 1; padding: 15px; overflow: hidden; overflow-y: scroll; background: #ffffff !important; color: #000000 !important; min-height: 0 !important; }
            #log { display: flex !important; flex-direction: column !important; gap: 10px; width: 100%; }
            #log .system { align-self: flex-start !important; text-align: left !important; color: blue !important; max-width: 85%; white-space: pre-wrap; }
            #log .user { align-self: flex-end !important; text-align: right !important; color: green !important; max-width: 85%; white-space: pre-wrap; }
            .footer { padding: 15px; flex-shrink: 0; }
            input { width: 100%; padding: 10px; border: 1px solid #ddd !important; border-radius: 4px; background: white !important; color: black !important; }
            button { cursor: pointer; padding: 5px 10px; border:0; }
        `;

        const agentContainer = document.createElement('div');
        agentContainer.id = 'agent-container';
        const header = document.createElement('header');
        header.style.flexWrap = 'wrap';
        const tabsContainer = document.createElement('div');
        tabsContainer.className = 'domain-filter';
        const tabs = ['UPDATE', 'COMMERCE', 'LOGISTICS', 'TRADE', 'CONFIG'];
        const defaultTab = (window.default_tab === 'DRAFT' ? 'UPDATE' : window.default_tab) || 'UPDATE';
        let currentTabFilter = defaultTab;

        tabs.forEach(t => {
            const btn = document.createElement('button');
            btn.textContent = t;
            btn.onclick = () => {
                currentTabFilter = t;
                Array.from(tabsContainer.children).forEach(c => { c.style.background = '#f9f9f9'; c.style.color = 'black'; });
                btn.style.background = '#333'; btn.style.color = 'white';
                renderStagedList();
            };
            tabsContainer.appendChild(btn);
        });
        header.appendChild(tabsContainer);

        const stagedList = document.createElement('div');
        stagedList.className = 'content';
        const log = document.createElement('div');
        log.id = 'log';
        stagedList.appendChild(log);

        const footer = document.createElement('div');
        footer.className = 'footer';
        const cliInput = document.createElement('input');
        cliInput.placeholder = '메시지 입력...';
        footer.appendChild(cliInput);

        agentContainer.appendChild(header);
        agentContainer.appendChild(stagedList);
        agentContainer.appendChild(footer);
        shadow.appendChild(style);
        shadow.appendChild(agentContainer);

        let stagedItems = [];
        function renderStagedList() {
            log.innerHTML = '';
            stagedItems.filter(i => i.domain === currentTabFilter || currentTabFilter === 'UPDATE').forEach(item => {
                const div = document.createElement('div');
                div.textContent = `[${item.domain}] ${item.title}`;
                log.appendChild(div);
            });
        }

        async function autoExtract() {
            const pageId = await generatePageId(window.location.href);
            const yaml = cleanAndConvertToYaml(document.body);
            const item = { id: pageId, title: document.title, domain: 'COMMERCE', context: yaml, status: 'DRAFT' };
            if (window.gemini_rpc) window.gemini_rpc("sync_data:" + JSON.stringify(item));
        }

        window.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') agentContainer.classList.toggle('open');
        });

        window.addEventListener('gemini_rpc_response', (e) => {
            const div = document.createElement('div');
            div.className = 'system';
            div.textContent = 'System: ' + JSON.stringify(e.detail);
            log.appendChild(div);
        });

        autoExtract();
    }
    initUI();
})();
"#;

async fn setup_page(browser: Arc<Browser>, page: chromiumoxide::Page, is_authenticated: bool) -> Result<(), Box<dyn std::error::Error>> {
    let _ = page.execute(AddBindingParams::new("gemini_rpc")).await;
    let default_tab = load_default_tab();
    let full_script = format!("window.is_authenticated = {};\nwindow.default_tab = \"{}\";\n{}", is_authenticated, default_tab, OVERLAY_SCRIPT);
    let _ = page.evaluate(full_script).await;
    let mut bindings = page.event_listener::<EventBindingCalled>().await?;
    let page_clone = page.clone();
    let browser_clone = browser.clone();
    
    tokio::task::spawn(async move {
        while let Some(event) = bindings.next().await {
            if event.name == "gemini_rpc" {
                let payload = event.payload.trim_matches('"').to_string();
                let response = if payload.starts_with("sync_data:") {
                    let data = &payload["sync_data:".len()..];
                    match serde_json::from_str::<db::CommerceRecord>(data) {
                        Ok(mut record) => {
                            if record.url.starts_with("file://") && record.context.contains("data:image/") {
                                if let Some(base64_part) = record.context.split("data:").nth(1) {
                                    let full_data_url = format!("data:{}", base64_part.trim());
                                    let ocr_result = {
                                        let mut model_guard = OCR_MODEL.lock().unwrap();
                                        if model_guard.is_none() {
                                            let model_path = "..\\models\\glm_ocr";
                                            let device = Device::Cpu;
                                            if let Ok(model) = GlmOcrGenerateModel::init(model_path, Some(&device), None) {
                                                *model_guard = Some(model);
                                            }
                                        }
                                        if let Some(model) = model_guard.as_mut() {
                                            let params = ChatCompletionParameters {
                                                messages: vec![Message {
                                                    role: "user".to_string(),
                                                    parts: vec![Part { text: "Extract text".to_string(), image_url: Some(full_data_url) }],
                                                }],
                                                model: "glm-ocr".to_string(),
                                                max_tokens: Some(1024),
                                                temperature: Some(0.0),
                                                top_p: None, top_k: None, repeat_penalty: None, repeat_last_n: None, seed: None,
                                            };
                                            model.generate(params).map(|res| res.choices[0].message.content.clone()).unwrap_or_default()
                                        } else { "OCR Model Load Error".to_string() }
                                    };
                                    record.context = format!("{}\n---\n[OCR]\n{}", record.context, ocr_result);
                                }
                            }
                            let updated = record.clone();
                            db::save_records(vec![record], None).await.map(|_| json!({"type":"sync_success","payload":updated}).to_string()).unwrap_or_else(|e| e.to_string())
                        },
                        Err(e) => e.to_string(),
                    }
                } else if payload == "fetch_drafts" {
                    db::fetch_drafts().await.map(|d| json!({"type":"drafts_loaded","payload":d}).to_string()).unwrap_or_else(|e| e.to_string())
                } else if payload.starts_with("gemini_chat:") {
                    "[System] Gemini 서비스 비활성화됨".to_string()
                } else { "Unknown command".to_string() };

                let script = format!("window.dispatchEvent(new CustomEvent('gemini_rpc_response', {{ detail: {} }}));", json!(response));
                let _ = page_clone.evaluate(script).await;
            }
        }
    });
    Ok(())
}

async fn run_mcp_server() -> Result<(), Box<dyn std::error::Error>> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
    let mut reader = tokio::io::BufReader::new(tokio::io::stdin()).lines();
    let mut stdout = tokio::io::stdout();
    while let Ok(Some(line)) = reader.next_line().await {
        if let Ok(req) = serde_json::from_str::<serde_json::Value>(&line) {
            let id = req["id"].clone();
            let res = json!({"jsonrpc":"2.0","id":id,"result":{"status":"Gemini Disabled"}});
            let mut s = res.to_string(); s.push('\n');
            let _ = stdout.write_all(s.as_bytes()).await;
            let _ = stdout.flush().await;
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--mcp") { return run_mcp_server().await; }

    let is_authenticated = true;
    let start_url = "about:blank";
    let browser_args = vec!["--window-size=640,480", "--force-dark-mode", start_url];

    let config = BrowserConfig::builder().with_head().no_sandbox().args(browser_args).build().map_err(|e| e.to_string())?;
    let (browser, mut handler) = Browser::launch(config).await?;
    let browser = Arc::new(browser);
    let (tx, mut rx) = tokio::sync::mpsc::channel::<()>(1);
    tokio::task::spawn(async move {
        while let Some(h) = handler.next().await { if h.is_err() { break; } }
        let _ = tx.send(()).await;
    });
    
    let mut target_events = browser.event_listener::<EventTargetCreated>().await?;
    let b_target = browser.clone();
    tokio::task::spawn(async move {
        while let Some(event) = target_events.next().await {
            if event.target_info.r#type == "page" {
                let tid = event.target_info.target_id.clone();
                let b_inner = b_target.clone();
                tokio::task::spawn(async move {
                    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
                    if let Ok(page) = b_inner.get_page(tid).await {
                        let _ = page.execute(EnableParams::default()).await;
                        let _ = setup_page(b_inner.clone(), page, true).await;
                    }
                });
            }
        }
    });

    if let Ok(pages) = browser.pages().await {
        if let Some(page) = pages.first() {
            let _ = setup_page(browser.clone(), page.clone(), true).await;
        }
    }

    tokio::select! {
        _ = tokio::signal::ctrl_c() => println!("\nShutting down..."),
        _ = rx.recv() => println!("\nBrowser closed."),
    }
    Ok(())
}
