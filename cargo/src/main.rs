use gemini_gui_lib::{db, privacy_filter, glm_ocr, params}; // embedding 제거
use privacy_filter::viterbi::PrivacySpan; // PrivacyFilterModel 제거
use candle_core::Device;
use glm_ocr::generate::{GlmOcrGenerateModel, GenerateModel};
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
use chromiumoxide::cdp::browser_protocol::page::EnableParams; // AddScriptToEvaluateOnNewDocumentParams 제거
use chromiumoxide::cdp::browser_protocol::target::EventTargetCreated;
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
                position: fixed; top: 0; left: 0; bottom: 0; 
                margin: auto; overflow: hidden;
                min-width: 360px; max-width: 560px; width:100%; z-index: 2147483648;
                background: white !important;
                display: flex !important; flex-direction: column;
                transition: opacity 0.2s ease-in-out; 
                box-shadow: 0 10px 25px rgba(0,0,0,0.2);
                pointer-events: none;
                opacity: 0;
            }
            #agent-container.open {
                opacity: 1 !important;
                pointer-events: auto !important;
            }
            header { padding: 15px; background: #f8f9fa !important; font-weight: bold !important; color: #000 !important; border-bottom: 1px solid #eee; flex-shrink: 0; display: flex !important; justify-content: space-between; align-items: center; }
            .header-left { display: flex; align-items: center; gap: 10px; }
            .header-actions { display: flex; gap: 8px; }
            .header-actions button { padding: 6px 12px; border-radius: 4px; border: 1px solid #ddd; background: #fff; cursor: pointer; font-size: 12px; font-weight: normal; transition: all 0.2s; }
            .header-actions button:hover { background: #f0f0f0; border-color: #ccc; }
            .header-actions button:disabled { background: #e9ecef !important; color: #6c757d !important; border-color: #dee2e6 !important; cursor: not-allowed; opacity: 0.6; }
            .btn-push { background: #007bff !important; color: white !important; border-color: #0069d9 !important; }
            .btn-push:not(:disabled):hover { background: #0069d9 !important; }
            .item-row { display: flex; align-items: center; gap: 10px; padding: 8px; border-bottom: 1px solid #f0f0f0; }
            .item-row input[type="checkbox"] { cursor: pointer; }
            #main-layout { display: flex !important; flex: 1; overflow: hidden; }
            aside { width: 180px; background: #f0f0f0; border-right: 1px solid #ddd; display: flex; flex-direction: column; padding: 10px 0; flex-shrink: 0; }
            .gnb-menu { display: flex; flex-direction: column; gap: 2px; }
            .gnb-item { padding: 10px 20px; cursor: pointer; font-size: 13px; color: #333; transition: background 0.2s; }
            .gnb-item:hover { background: #e0e0e0; }
            .gnb-item.active { background: #333; color: #fff; font-weight: bold; }
            .content { flex: 1; padding: 15px; overflow-y: auto; background: #fff; }
            .content { flex: 1; padding: 15px; overflow: hidden; overflow-y: scroll; background: #ffffff !important; color: #000000 !important; min-height: 0 !important; }
            #log { display: flex !important; flex-direction: column !important; gap: 10px; width: 100%; }
            #log .system { align-self: flex-start !important; text-align: left !important; color: blue !important; max-width: 85%; white-space: pre-wrap; }
            #log .user { align-self: flex-end !important; text-align: right !important; color: green !important; max-width: 85%; white-space: pre-wrap; }
            /* footer 및 input 스타일 제거 */
            button { cursor: pointer; padding: 5px 10px; border:0; }
        `;

        const agentContainer = document.createElement('div');
        agentContainer.id = 'agent-container';
        
        const header = document.createElement('header');
        const headerLeft = document.createElement('div');
        headerLeft.className = 'header-left';

        // 전체 선택 체크박스
        const selectAllCheck = document.createElement('input');
        selectAllCheck.type = 'checkbox';
        selectAllCheck.title = 'Select All';
        selectAllCheck.onclick = (e) => {
            const checkboxes = log.querySelectorAll('.item-checkbox');
            checkboxes.forEach(cb => cb.checked = e.target.checked);
            updatePushBtnState();
        };

        const titleSpan = document.createElement('span');
        titleSpan.textContent = 'COMMERCE'; // 초기화 (이후 updateGnbUI에서 갱신됨)
        
        headerLeft.appendChild(selectAllCheck);
        headerLeft.appendChild(titleSpan);
        header.appendChild(headerLeft);

        const actionContainer = document.createElement('div');
        actionContainer.className = 'header-actions';

        function getPageMeta() {
            const ogTitle = document.querySelector('meta[property="og:title"]')?.content || document.title;
            const ogDesc = document.querySelector('meta[property="og:description"]')?.content || '';
            return ogDesc ? `${ogTitle}\n${ogDesc}` : ogTitle;
        }

        const draftBtn = document.createElement('button');
        draftBtn.textContent = 'Draft (0)';
        draftBtn.onclick = async () => {
            const pageId = await generatePageId(window.location.href);
            const yaml = cleanAndConvertToYaml(document.body);
            const item = { 
                id: pageId, 
                host: window.location.host,
                url: window.location.href,
                title: getPageMeta(), 
                domain: currentTabFilter, // 현재 GNB 탭의 도메인으로 매핑
                context: yaml, 
                status: 'DRAFT',
                track: '',
                version: 1,
                created_at: Date.now(),
                updated_at: Date.now()
            };
            if (window.rpc) {
                window.rpc("sync_data:" + JSON.stringify(item));
                alert("Draft saved for " + currentTabFilter);
            }
        };

        // Push 버튼: Privacy Filter 마스킹 후 저장 로직 실행
        const pushBtn = document.createElement('button');
        pushBtn.className = 'btn-push';
        pushBtn.textContent = 'Push (0)';
        pushBtn.disabled = true; // 초기 상태 비활성화

        function updatePushBtnState() {
            const checkedCount = log.querySelectorAll('.item-checkbox:checked').length;
            pushBtn.disabled = (checkedCount === 0);
            pushBtn.textContent = `Push (${checkedCount})`; // 선택된 카운트 반영
            
            const totalCount = log.querySelectorAll('.item-checkbox').length;
            selectAllCheck.checked = (totalCount > 0 && checkedCount === totalCount);
        }

        pushBtn.onclick = async () => {
            const checkedBoxes = log.querySelectorAll('.item-checkbox:checked');
            // 만약 개별 아이템의 전체 데이터를 다시 보내야 하는 구조라면 아래와 같이 구성합니다.
            const selectedIds = Array.from(checkedBoxes).map(cb => cb.dataset.id);
            
            if (window.rpc) {
                // 배치 처리 시에도 host 정보가 필요한 경우를 대비해 전송 객체 구조를 확인하십시오.
                window.rpc("mask_and_push_batch:" + JSON.stringify({ 
                    ids: selectedIds,
                    host: window.location.host // 필요한 경우 추가
                }));
                alert(`${selectedIds.length} items Pushed with masking`);
            }
        };

        actionContainer.appendChild(draftBtn);
        actionContainer.appendChild(pushBtn);
        header.appendChild(actionContainer);

        const mainLayout = document.createElement('div');
        mainLayout.id = 'main-layout';

        const aside = document.createElement('aside');
        const gnbMenu = document.createElement('div');
        gnbMenu.className = 'gnb-menu';

        const tabs = ['COMMERCE', 'LOGISTICS', 'TRADE', 'CONFIG'];
        const defaultTab = (window.default_tab === 'DRAFT' ? 'COMMERCE' : window.default_tab) || 'COMMERCE';
        let currentTabFilter = defaultTab;
        
        let stagedItems = []; // GNB 카운트 계산을 위해 상단으로 이동

        function updateGnbUI() {
            gnbMenu.replaceChildren();
            titleSpan.textContent = currentTabFilter; // 헤더 타이틀을 현재 활성화된 메뉴로 변경

            tabs.forEach(t => {
                const domainCount = stagedItems.filter(i => i.domain === t).length;
                const item = document.createElement('div');
                item.className = 'gnb-item' + (t === currentTabFilter ? ' active' : '');
                
                if (t === 'CONFIG') {
                    item.textContent = t;
                } else {
                    item.textContent = `${t} (${domainCount})`;
                }
                
                item.onclick = () => {
                    currentTabFilter = t;
                    updateGnbUI();
                    renderStagedList();
                };
                gnbMenu.appendChild(item);
            });
        }
        updateGnbUI();
        aside.appendChild(gnbMenu);

        const stagedList = document.createElement('div');
        stagedList.className = 'content';
        const log = document.createElement('div');
        log.id = 'log';
        stagedList.appendChild(log);

        mainLayout.appendChild(aside);
        mainLayout.appendChild(stagedList);

        agentContainer.appendChild(header);
        agentContainer.appendChild(mainLayout);
        // agentContainer.appendChild(footer); // footer 추가 코드 제거
        shadow.appendChild(style);
        shadow.appendChild(agentContainer);

        function renderStagedList() {
            log.replaceChildren(); 
            // 현재 선택된 도메인에 해당하는 항목만 엄격하게 필터링
            const filtered = stagedItems.filter(i => i.domain === currentTabFilter);
            draftBtn.textContent = `Draft (${filtered.length})`; // Draft 전체 카운트 갱신
            
            if (filtered.length === 0) {
                const empty = document.createElement('div');
                empty.style.color = '#999';
                empty.style.fontSize = '12px';
                empty.style.padding = '20px';
                empty.textContent = 'No records found for ' + currentTabFilter;
                log.appendChild(empty);
                selectAllCheck.checked = false;
                updatePushBtnState();
            } else {
                filtered.forEach(item => {
                    const row = document.createElement('div');
                    row.className = 'item-row';

                    const cb = document.createElement('input');
                    cb.type = 'checkbox';
                    cb.className = 'item-checkbox';
                    cb.dataset.id = item.id;
                    cb.onclick = () => updatePushBtnState();

                    const textContainer = document.createElement('div');
                    textContainer.style.display = 'flex';
                    textContainer.style.flexDirection = 'column';
                    textContainer.style.flex = '1';
                    textContainer.style.overflow = 'hidden';

                    const parts = item.title.split('\n');
                    const mainTitle = parts[0] || '';
                    const descText = parts.slice(1).join('\n') || '';

                    const titleSpan = document.createElement('span');
                    titleSpan.textContent = `[${item.domain}] ${mainTitle}`;
                    titleSpan.style.fontSize = '13px';
                    titleSpan.style.fontWeight = 'bold';
                    titleSpan.style.whiteSpace = 'nowrap';
                    titleSpan.style.overflow = 'hidden';
                    titleSpan.style.textOverflow = 'ellipsis';
                    textContainer.appendChild(titleSpan);

                    if (descText) {
                        const descSpan = document.createElement('span');
                        descSpan.textContent = descText;
                        descSpan.style.fontSize = '11px';
                        descSpan.style.color = '#666';
                        descSpan.style.whiteSpace = 'nowrap';
                        descSpan.style.overflow = 'hidden';
                        descSpan.style.textOverflow = 'ellipsis';
                        textContainer.appendChild(descSpan);
                    }

                    row.appendChild(cb);
                    row.appendChild(textContainer);
                    log.appendChild(row);
                });
                updatePushBtnState();
            }
        }

        async function autoExtract() {
            const pageId = await generatePageId(window.location.href);
            const yaml = cleanAndConvertToYaml(document.body);
            const item = { 
                id: pageId, 
                host: window.location.host,
                url: window.location.href,
                title: getPageMeta(), 
                domain: 'COMMERCE', 
                context: yaml, 
                status: 'DRAFT',
                track: '',
                version: 1,
                created_at: Date.now(),
                updated_at: Date.now()
            };
            if (window.rpc) window.rpc("sync_data:" + JSON.stringify(item));
        }

        window.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') agentContainer.classList.toggle('open');
        });

        window.addEventListener('rpc_response', (e) => {
            try {
                // 백엔드에서 넘겨준 문자열 JSON을 파싱
                const data = typeof e.detail === 'string' ? JSON.parse(e.detail) : e.detail;
                
                // 데이터 불러오기 응답인 경우
                if (data.type === 'drafts_loaded') {
                    stagedItems = data.payload;
                    updateGnbUI(); // 데이터가 로드된 후 GNB 메뉴의 카운트 갱신
                    renderStagedList();
                    return;
                } 
                // Draft 저장(sync) 성공 응답인 경우
                else if (data.type === 'sync_success') {
                    // 동일한 ID가 있으면 제거 후 최신 데이터 추가
                    stagedItems = stagedItems.filter(i => i.id !== data.payload.id);
                    stagedItems.push(data.payload);
                    updateGnbUI(); // 데이터 동기화 후 GNB 메뉴의 카운트 갱신
                    renderStagedList();
                    return;
                }
            } catch (err) {
                // JSON 파싱 실패시 일반 시스템 로그로 처리
            }
            
            // 기타 오류 및 시스템 메시지 출력용
            const div = document.createElement('div');
            div.className = 'system';
            div.style.padding = '10px';
            div.style.background = '#f0f4ff';
            div.style.borderRadius = '4px';
            div.textContent = 'System: ' + (typeof e.detail === 'string' ? e.detail : JSON.stringify(e.detail));
            log.appendChild(div);
            div.scrollIntoView({ behavior: 'smooth', block: 'end' });
        });

        autoExtract();
    }
    initUI();
})();
"#;

async fn setup_page(browser: Arc<Browser>, page: chromiumoxide::Page, is_authenticated: bool) -> Result<(), Box<dyn std::error::Error>> {
    let _ = page.execute(AddBindingParams::new("rpc")).await; // 바인딩명 변경
    let default_tab = load_default_tab();
    let full_script = format!("window.is_authenticated = {};\nwindow.default_tab = \"{}\";\n{}", is_authenticated, default_tab, OVERLAY_SCRIPT);
    let _ = page.evaluate(full_script).await;
    let mut bindings = page.event_listener::<EventBindingCalled>().await?;
    let page_clone = page.clone();
    let browser_clone = browser.clone();
    
    tokio::task::spawn(async move {
        while let Some(event) = bindings.next().await {
            if event.name == "rpc" { // 이벤트 수신명 변경
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
                                            let device = Device::new_cuda(0).unwrap_or(Device::Cpu); // CUDA 장치 설정으로 변경
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

                let script = format!("window.dispatchEvent(new CustomEvent('rpc_response', {{ detail: {} }}));", json!(response));
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

    let _is_authenticated = true; // 언더바 추가하여 미사용 경고 해결
    let start_url = "about:blank";

    let browser_args = vec![
        "--window-size=640,480", // 창 크기 강제 지정
        "--window-position=0,0",
        "--start-maximized", 
        "--no-first-run",
        "--disable-notifications",
        "--disable-extensions",
        "--disable-popup-blocking",
        "--disable-blink-features=AutomationControlled",
        "--password-store=basic",
        "--no-default-browser-check",
        "--force-dark-mode",
        "--enable-features=WebUIDarkMode",
        "--remote-allow-origins=*",
        "--disable-dev-shm-usage",
        start_url, // 브라우저 실행 인자에 URL을 직접 포함하여 단 1개의 정상 탭만 생성되도록 유도
    ];

    let config = BrowserConfig::builder().with_head().no_sandbox().viewport(None).args(browser_args).build().map_err(|e| e.to_string())?;
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
                    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
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
