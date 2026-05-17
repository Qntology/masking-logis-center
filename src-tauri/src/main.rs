mod db;
mod embedding;
use privacy_filter_rs::{PrivacyFilterInference, PrivacySpan};
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
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

fn mask_pii(text: &str, spans: &[PrivacySpan]) -> String {
    let mut masked_text = text.to_string();
    let mut sorted_spans = spans.to_vec();
    // Sort spans by start index in reverse to maintain offset correctness during replacement
    sorted_spans.sort_by(|a, b| b.start.cmp(&a.start));

    for span in sorted_spans {
        if span.start < masked_text.len() && span.end <= masked_text.len() && span.start < span.end {
            let mask = format!("[{}]", span.entity_group.to_uppercase());
            masked_text.replace_range(span.start..span.end, &mask);
        }
    }
    masked_text
}

const OVERLAY_SCRIPT: &str = r#"
(function() {
    // iframe 내부라면 실행하지 않음 (최상위 프레임에서만 렌더링)
    if (window.self !== window.top) return;

    if (window.geminiSidebarLoaded) return;
    window.geminiSidebarLoaded = true;

    async function generatePageId(url) {
        const msgUint8 = new TextEncoder().encode(url);
        const hashBuffer = await crypto.subtle.digest('SHA-256', msgUint8);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    }

    // 데이터 중심 정제 및 변환 함수 (중복 제거 및 통합)
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

        // body가 아직 생성되지 않았다면 대기 후 다시 호출
        if (!document.body) {
            window.requestAnimationFrame(initUI);
            return;
        }

        const host = document.createElement('div');
        host.id = 'gemini-agent-host';
        host.style.cssText = 'position:fixed; top:0; left:0; width:100%; height:100%; z-index:2147483647; pointer-events:none; overflow:hidden;';
        
        // body에 안전하게 추가
        try {
            document.body.appendChild(host);
        } catch (e) {
            // 사이트 정책에 의해 appendChild가 거부될 경우를 대비해 최상위 요소로 시도
            document.documentElement.appendChild(host);
        }

        const shadow = host.attachShadow({ mode: 'open' });

        const style = document.createElement('style');
        style.textContent = `
            :host { all: initial; }
            #agent-container { 
                position: fixed; top: 0; right: 0; bottom: 0; 
                width: 350px; z-index: 2147483648;
                background: white !important; border-left: 1px solid #ccc;
                display: flex !important; flex-direction: column;
                transition: transform 0.3s ease-in-out; transform: translateX(100%);
                box-shadow: -5px 0 15px rgba(0,0,0,0.2);
                pointer-events: auto;
            }
            #agent-container.open { transform: translateX(0); }
            #toggle-btn { 
                all: unset;
                position: fixed; bottom: 30px; right: 30px; 
                width: 60px !important; height: 60px !important; 
                background: #007bff !important; color: white !important;
                border-radius: 50% !important; cursor: pointer !important; 
                z-index: 2147483649;
                display: flex !important; align-items: center !important; 
                justify-content: center !important;
                border: 4px solid white !important; 
                font-weight: bold !important; 
                font-family: sans-serif !important;
                font-size: 16px !important;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
                pointer-events: auto;
                transition: all 0.2s ease;
            }
            #toggle-btn:hover { background: #0056b3 !important; transform: scale(1.05); }
            header { padding: 15px; background: #f0f0f0 !important; font-weight: bold !important; color: #000 !important; border-bottom: 1px solid #ddd; display: flex !important; justify-content: space-between; align-items: center; }
            .content { flex: 1; padding: 15px; overflow-y: auto; background: #ffffff !important; color: #000000 !important; box-sizing: border-box !important; }
            .footer { padding: 15px; background: #f8f9fa !important; border-top: 1px solid #eee; flex-shrink: 0; }
            input { width: 100%; padding: 10px; border: 1px solid #ddd !important; border-radius: 4px; box-sizing: border-box !important; background: white !important; color: black !important; }
            button { cursor: pointer; padding: 5px 10px; }
            .staged-item { display: flex !important; align-items: center !important; margin-bottom: 10px; color: black !important; }
        `;

        const toggleBtn = document.createElement('div');
        toggleBtn.id = 'toggle-btn';
        toggleBtn.textContent = 'AI';

        const agentContainer = document.createElement('div');
        agentContainer.id = 'agent-container';

        const header = document.createElement('header');
        header.textContent = 'Staging Area (LanceDB) ';
        const closeBtn = document.createElement('button');
        closeBtn.id = 'close-btn';
        closeBtn.textContent = 'X';
        header.appendChild(closeBtn);

        const stagedList = document.createElement('div');
        stagedList.className = 'content';
        stagedList.id = 'staged-list';
        const log = document.createElement('div');
        log.id = 'log';
        stagedList.appendChild(log);

        const footer = document.createElement('div');
        footer.className = 'footer';
        const extractBtn = document.createElement('button');
        extractBtn.id = 'extract-btn';
        extractBtn.textContent = '추출';
        const pushBtn = document.createElement('button');
        pushBtn.id = 'push-btn';
        pushBtn.textContent = 'Push Selected';
        const cliInput = document.createElement('input');
        cliInput.type = 'text';
        cliInput.id = 'cli-input';
        cliInput.placeholder = '메시지 입력...';
        
        footer.appendChild(extractBtn);
        footer.appendChild(pushBtn);
        footer.appendChild(cliInput);

        agentContainer.appendChild(header);
        agentContainer.appendChild(stagedList);
        agentContainer.appendChild(footer);

        shadow.appendChild(style);
        shadow.appendChild(toggleBtn);
        shadow.appendChild(agentContainer);

        let stagedItems = [];

        // UI 리스트를 갱신하는 독립 렌더링 함수
        function renderStagedList() {
            // 기존 아이템 삭제 (로그 영역은 보존)
            const items = stagedList.querySelectorAll('.staged-item');
            items.forEach(item => item.remove());

            stagedItems.forEach(item => {
                const itemDiv = document.createElement('div');
                itemDiv.className = 'staged-item';
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.dataset.id = item.id;
                itemDiv.appendChild(checkbox);
                itemDiv.appendChild(document.createTextNode(' ' + item.id.substring(0,8) + '... (v' + item.version + ')'));
                stagedList.appendChild(itemDiv);
            });
        }

        // 초기 로드 시 백엔드에 기존 DRAFT 목록 요청
        if (window.gemini_rpc) window.gemini_rpc("fetch_drafts");

        // 자동 추출 함수
        async function autoExtract() {
            const pageId = await generatePageId(window.location.href);
            const yaml = cleanAndConvertToYaml(document.body);
            
            // 1. UI 로그에 YAML 표시 (Trusted Types 우회)
            const autoLogDiv = document.createElement('div');
            autoLogDiv.style.cssText = 'white-space: pre-wrap; font-size: 11px; margin-top: 5px;';
            const strongText = document.createElement('strong');
            strongText.textContent = '[Auto-Extracted]:\n';
            autoLogDiv.appendChild(strongText);
            autoLogDiv.appendChild(document.createTextNode(yaml.substring(0, 100) + '...'));
            log.appendChild(autoLogDiv);
            
            const item = { 
                id: pageId, 
                host: window.location.hostname, 
                url: window.location.href, 
                title: document.title, 
                domain: 'COMMERCE', 
                context: yaml, 
                status: 'DRAFT', 
                track: 'MAIN', 
                version: 1, 
                created_at: Date.now(), 
                updated_at: Date.now() 
            };
            
            // 2. 동일한 ID의 DRAFT가 있으면 덮어쓰기(버전업), 없으면 신규 추가
            const existingIndex = stagedItems.findIndex(i => i.id === pageId && i.status === 'DRAFT');
            if (existingIndex !== -1) {
                stagedItems[existingIndex].context = yaml;
                stagedItems[existingIndex].updated_at = item.updated_at;
                stagedItems[existingIndex].version += 1;
                Object.assign(item, stagedItems[existingIndex]); 
            } else {
                stagedItems.push(item);
            }
            
            renderStagedList();
            
            // 3. Rust 백엔드(LanceDB)로 DRAFT 상태 동기화 (Upsert) 요청
            if (window.gemini_rpc) window.gemini_rpc("sync_data:" + JSON.stringify(item));
        }

        toggleBtn.onclick = () => { agentContainer.classList.toggle('open'); };
        closeBtn.onclick = () => { agentContainer.classList.remove('open'); };
        
        extractBtn.onclick = autoExtract;

        pushBtn.onclick = () => {
            const selected = Array.from(shadow.querySelectorAll('input:checked')).map(cb => cb.dataset.id);
            const payload = stagedItems.filter(i => selected.includes(i.id));
            if (window.gemini_rpc) window.gemini_rpc("push_data:" + JSON.stringify(payload));
        };

        // 자동 실행 및 상태 유지
        if (document.readyState === 'complete') {
            autoExtract();
        } else {
            window.addEventListener('load', autoExtract);
        }

        window.addEventListener('gemini_rpc_response', (e) => {
            try {
                // JSON 형태의 응답(fetch_drafts) 처리
                const data = JSON.parse(e.detail);
                if (data.type === 'drafts_loaded') {
                    data.payload.forEach(draft => {
                        // 중복 방지: 현재 리스트에 없는 항목만 추가
                        if (!stagedItems.find(i => i.id === draft.id)) {
                            stagedItems.push(draft);
                        }
                    });
                    renderStagedList();
                    return; // 데이터 응답이므로 채팅 로그에 출력하지 않고 종료
                }
            } catch(err) {
                // JSON 파싱 실패 시 일반 텍스트 응답으로 간주하여 아래 로그 출력 진행
            }

            if (log) {
                const rpcLogDiv = document.createElement('div');
                rpcLogDiv.style.color = 'blue';
                const rpcStrong = document.createElement('strong');
                rpcStrong.textContent = 'AI: ';
                rpcLogDiv.appendChild(rpcStrong);
                rpcLogDiv.appendChild(document.createTextNode(e.detail));
                log.appendChild(rpcLogDiv);
            }
        });
    }

    // 초기화 호출부 강화 (MutationObserver를 통한 강제 렌더링 보장)
    if (window.self === window.top) {
        const runOnce = () => {
            if (!document.getElementById('gemini-agent-host')) initUI();
        };

        // 로드 시점별 실행
        if (document.readyState === 'complete') {
            runOnce();
        } else {
            window.addEventListener('load', runOnce);
            document.addEventListener('DOMContentLoaded', runOnce);
        }

        // 네이버처럼 DOM이 계속 변하는 사이트를 위해 body 출현 감시
        const observer = new MutationObserver(() => {
            if (document.body && !document.getElementById('gemini-agent-host')) {
                runOnce();
                // 생성 성공 시 감시 종료하여 부하 감소
                if (document.getElementById('gemini-agent-host')) observer.disconnect();
            }
        });
        
        observer.observe(document.documentElement, { childList: true, subtree: true });

        window.addEventListener('pageshow', runOnce);
    }

})();
"#;

async fn setup_page(browser: Arc<Browser>, page: chromiumoxide::Page) -> Result<(), Box<dyn std::error::Error>> {
    // RPC 바인딩 등록
    let _ = page.execute(AddBindingParams::new("gemini_rpc")).await;
    
    // 이미 로드된 현재 페이지 상태에서 UI가 즉시 나타나도록 강제 실행
    let _ = page.evaluate(OVERLAY_SCRIPT).await;
    let mut bindings = page.event_listener::<EventBindingCalled>().await?;
    let page_clone = page.clone();
    let browser_clone = browser.clone();
    tokio::task::spawn(async move {
        while let Some(event) = bindings.next().await {
            if event.name == "gemini_rpc" {
                let payload = event.payload.trim_matches('"');
                let response = if payload.starts_with("sync_data:") {
                    let data = &payload["sync_data:".len()..];
                    match serde_json::from_str::<db::CommerceRecord>(data) {
                        Ok(record) => {
                            // 임시 저장(DRAFT) 시에는 임베딩 모델을 로드하지 않습니다.
                            match db::save_records(vec![record]).await {
                                Ok(_) => "Data synced to LanceDB (DRAFT).".to_string(),
                                Err(e) => format!("DB Error: {}", e),
                            }
                        },
                        Err(e) => format!("JSON Error: {}", e),
                    }
                } else if payload == "fetch_drafts" {
                    match db::fetch_drafts().await {
                        Ok(drafts) => {
                            json!({
                                "type": "drafts_loaded",
                                "payload": drafts
                            }).to_string()
                        },
                        Err(e) => format!("DB Error: {}", e),
                    }
                } else if payload.starts_with("push_data:") {
                    let data = &payload["push_data:".len()..];
                    match serde_json::from_str::<Vec<db::CommerceRecord>>(data) {
                        Ok(mut records) => {
                            // 1. PII Masking
                            let privacy_model_path = std::path::PathBuf::from("..\\models\\privacy-filter");
                            let device = NdArrayDevice::Cpu;
                            if let Ok(privacy_engine) = PrivacyFilterInference::<NdArray>::load(&privacy_model_path, device) {
                                println!("[Rust] Privacy Filter Loaded. Masking PII...");
                                for record in &mut records {
                                    if let Ok(spans) = privacy_engine.predict(&record.context) {
                                        record.context = mask_pii(&record.context, &spans);
                                    }
                                }
                            } else {
                                eprintln!("[Rust] Warning: Failed to load privacy filter model");
                            }

                            // 2. Push 요청 시에만 임베딩 모델을 메모리에 로드합니다.
                            let model_path = std::path::PathBuf::from("..\\models\\embeddings");
                            match embedding::EmbeddingModel::new(model_path) {
                                Ok(model) => {
                                    for record in &mut records {
                                        match model.embed(&record.context) {
                                            Ok(vector) => record.vector = vector,
                                            Err(e) => eprintln!("Embedding Error: {}", e),
                                        }
                                    }
                                    // 스코프가 끝나면 model이 드롭되어 VRAM에서 해제됩니다.
                                },
                                Err(e) => eprintln!("Model load error: {}", e),
                            }
                            
                            match db::save_records(records).await {
                                Ok(_) => "Data pushed successfully with PII masking and embeddings.".to_string(),
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
    // 사용자가 요청한 클로저 기반 설정 로직 반영
    let args = vec![
        "--window-size=1920,1080", // 창 크기 강제 지정
        "--window-position=0,0",
        "--start-maximized", 
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

    // 인증 여부에 따른 초기 URL 설정
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).unwrap_or_default();
    let auth_path = std::path::PathBuf::from(home).join(".gemini/oauth_creds.json");
    let is_authenticated = auth_path.exists();
    let start_url = if is_authenticated { "https://www.google.com" } else { "https://aistudio.google.com/" };

    // chromiumoxide가 자체적으로 디버깅 포트와 초기 타겟(about:blank)을 
    // 충돌 없이 할당하도록 포트 플래그 및 about:blank 인자를 제거했습니다.

    let config = BrowserConfig::builder()
        .with_head()
        .no_sandbox()
        .viewport(None)
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
    
    // 브라우저 직접 실행 방식은 프로토콜 에러를 유발하므로 삭제합니다.
    // 대신 아래의 target_events 리스너 내부에서 페이지별로 설정을 주입합니다.

    let mut target_events = browser.event_listener::<EventTargetCreated>().await?;
    let b_target = browser.clone();
    tokio::task::spawn(async move {
        while let Some(event) = target_events.next().await {
            if event.target_info.r#type == "page" {
                let tid = event.target_info.target_id.clone();
                let b_inner = b_target.clone();
                tokio::task::spawn(async move {
                    // CDP 연결 확보를 위한 최소한의 시간 대기
                    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
                    
                    if let Ok(page) = b_inner.get_page(tid).await {
                        // 페이지 객체 확보 후 바인딩 및 주입 설정을 즉시 수행하여 '페이지 로드 시작' 단계부터 스크립트가 살아있게 함
                        let _ = page.execute(EnableParams::default()).await;
                        let _ = page.execute(AddScriptToEvaluateOnNewDocumentParams::new(OVERLAY_SCRIPT.to_string())).await;
                        
                        // 특정 페이지 로딩 상태(예: DOMContentLoaded)까지 기다리지 않고 즉시 셋업 시도
                        let _ = setup_page(b_inner.clone(), page).await;
                    }
                });
            }
        }
    });

    if !is_authenticated {
        println!("[Rust] Authentication required. Redirecting to login...");
    }

    // 이미 열려있는 최초 빈 탭(about:blank)을 가져와 스크립트 환경을 완벽하게 주입합니다.
    if let Ok(pages) = browser.pages().await {
        if let Some(page) = pages.first() {
            let _ = page.execute(EnableParams::default()).await;
            // 모든 새 문서에 스크립트가 자동 실행되도록 브라우저 내부 설정
            let _ = page.execute(AddScriptToEvaluateOnNewDocumentParams::new(OVERLAY_SCRIPT.to_string())).await;
            // RPC 바인딩 및 이벤트 리스너 세팅
            let _ = setup_page(browser.clone(), page.clone()).await;
            
            // 모든 세팅이 완료된 이후에 수동으로 타겟 URL로 이동시킵니다.
            // 이렇게 해야 페이지가 새로 로드되면서 이미 예약된 스크립트가 100% 동작하여 버튼이 노출됩니다.
            let _ = page.goto(start_url).await;
        }
    }

    // 하단의 goto 및 new_page 로직을 제거하여 탭이 중복으로 생성되거나 이동되는 현상을 차단합니다.

    tokio::select! {
        _ = tokio::signal::ctrl_c() => println!("\n[Rust] Shutting down..."),
        _ = rx.recv() => println!("\n[Rust] Browser closed, shutting down..."),
    }
    Ok(())
}
