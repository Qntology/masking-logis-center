use gemini_gui_lib::{db, embedding, privacy_filter, gemini, glm_ocr, params};
use privacy_filter::{PrivacyFilterModel, viterbi::PrivacySpan};
use candle_core::Device;
use glm_ocr::generate::GlmOcrGenerateModel;
use params::chat::{ChatCompletionParameters, Message, Part};
use std::sync::Mutex;
use lazy_static::lazy_static;

lazy_static! {
    static ref OCR_MODEL: Mutex<Option<GlmOcrGenerateModel>> = Mutex::new(None);
}

// ... rest of imports ...

// #[command]
async fn get_chat_completion(_messages: Vec<gemini::types::ChatMessage>, _api_key: String, model: String) -> Result<String, String> {
    let _client = gemini::client::GeminiClient::new(model); // Adjusted to new constructor if needed
    // Assuming a method exists or needs adjustment
    Ok("Fixing syntax error".to_string())
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
        // host.style.cssText = 'position:fixed; top:0; left:0; width:100%; height:100%; z-index:2147483647; pointer-events:none; overflow:hidden;';
        
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
            #status-select { 
                padding: 4px 8px; 
                border-radius: 8px; 
                border: 1px solid #ccc; 
                background: #fff; 
                font-size: 12px; 
                cursor: pointer; 
                outline: none;
                font-weight: bold;
                color: #333;
            }
            #status-select:focus { border-color: #000; }
            .domain-filter { display: flex; gap: 5px; padding: 10px 15px; background: #fff; border-bottom: 1px solid #eee; flex-shrink: 0; overflow-x: auto; }
            .domain-filter button { padding: 4px 10px; border-radius: 20px; border: 1px solid #ddd; background: #f9f9f9; font-size: 11px; white-space: nowrap; }
            .domain-filter button.active { background: #333; color: white; border-color: #333; }
            .content { flex: 1; padding: 15px; overflow: hidden; overflow-y: scroll; background: #ffffff !important; color: #000000 !important; min-height: 0 !important; }
            #log { display: flex !important; flex-direction: column !important; gap: 10px; width: 100%; }
            #log .system { align-self: flex-start !important; text-align: left !important; color: blue !important; max-width: 85%; white-space: pre-wrap; }
            #log .user { align-self: flex-end !important; text-align: right !important; color: green !important; max-width: 85%; white-space: pre-wrap; }
            .footer { padding: 15px; flex-shrink: 0; }
            input { width: 100%; padding: 10px; border: 1px solid #ddd !important; border-radius: 4px; background: white !important; color: black !important; }
            input:disabled { background: #eee !important; cursor: not-allowed !important; color: #888 !important; }
            button { cursor: pointer; padding: 5px 10px; border:0; }
            button:disabled { cursor: not-allowed !important; opacity: 0.6; }
            .staged-item, .user.draft, .user.draft.progressing, .user.COMMERCE, .user.LOGISTICS, .user.TRADE { display: flex !important; align-items: center !important; margin-bottom: 10px; color: black !important; }
        `;

        const agentContainer = document.createElement('div');
        agentContainer.id = 'agent-container';

        // 헤더 구성
        const header = document.createElement('header');
        header.style.flexWrap = 'wrap';
        
        const tabsContainer = document.createElement('div');
        tabsContainer.className = 'domain-filter';
        tabsContainer.style.cssText = 'display: flex; gap: 5px; flex: 1; border: none; padding: 0; background: transparent; overflow-x: auto;';
        
        // DRAFT -> UPDATE로 명칭 변경 및 CONFIG 탭 추가
        const tabs = ['UPDATE', 'COMMERCE', 'LOGISTICS', 'TRADE', 'CONFIG'];
        const defaultTab = (window.default_tab === 'DRAFT' ? 'UPDATE' : window.default_tab) || 'UPDATE';
        let currentTabFilter = defaultTab;
        
        tabs.forEach(t => {
            const btn = document.createElement('button');
            btn.textContent = t;
            btn.dataset.tab = t;
            btn.style.cssText = 'padding: 4px 10px; border-radius: 20px; border: 1px solid #ddd; background: #f9f9f9; font-size: 11px; white-space: nowrap; cursor: pointer;';
            
            // Drag and Drop: Drop Zone 이벤트 설정 (CONFIG 제외)
            if (t !== 'CONFIG') {
                btn.ondragover = (e) => {
                    e.preventDefault(); // 드롭 허용
                    btn.style.transform = 'scale(1.1)';
                };
                btn.ondragleave = () => {
                    btn.style.transform = 'scale(1)';
                };
                btn.ondrop = (e) => {
                    e.preventDefault();
                    btn.style.transform = 'scale(1)';
                    const itemId = e.dataTransfer.getData('text/plain');
                    if (itemId) {
                        const itemIndex = stagedItems.findIndex(i => i.id === itemId);
                        if (itemIndex !== -1) {
                            // 강제 도메인 업데이트 및 상태 갱신
                            stagedItems[itemIndex].domain = t;
                            stagedItems[itemIndex].status = (t === 'UPDATE') ? 'UPDATE' : 'MAIN';
                            if (window.gemini_rpc) window.gemini_rpc("sync_data:" + JSON.stringify(stagedItems[itemIndex]));
                            renderStagedList();
                        }
                    }
                };
            }

            btn.onclick = () => {
                currentTabFilter = t;
                Array.from(tabsContainer.children).forEach(c => {
                    c.style.background = '#f9f9f9';
                    c.style.color = 'black';
                    c.style.borderColor = '#ddd';
                });
                btn.style.background = '#333';
                btn.style.color = 'white';
                btn.style.borderColor = '#333';
                
                // UPDATE, CONFIG 일 때 채팅 기능 차단
                if (currentTabFilter === 'UPDATE' || currentTabFilter === 'CONFIG') {
                    cliInput.value = '';
                    cliInput.disabled = true;
                    cliInput.placeholder = currentTabFilter === 'UPDATE' ? 'UPDATE 상태에서는 질의 및 첨부가 불가능합니다.' : '설정 화면입니다.';
                    sendBtn.disabled = true;
                    attachBtn.disabled = true; 
                } else {
                    cliInput.disabled = false;
                    cliInput.placeholder = '메시지 입력...';
                    sendBtn.disabled = false;
                    attachBtn.disabled = false;
                }
                renderStagedList();
            };
            tabsContainer.appendChild(btn);
        });

        header.appendChild(tabsContainer);

        // 보기 화면 UI 설정 (리스트 vs 채팅) 토글 버튼
        const viewToggleBtn = document.createElement('button');
        let isChatView = true; // 기본값
        viewToggleBtn.textContent = '👁️ 리스트 보기';
        viewToggleBtn.style.cssText = 'margin-left: 10px; padding: 4px 8px; border-radius: 4px; background: #eee; border: 1px solid #ccc; font-size: 11px; cursor: pointer; font-weight: bold;';
        viewToggleBtn.onclick = () => {
            isChatView = !isChatView;
            viewToggleBtn.textContent = isChatView ? '👁️ 리스트 보기' : '💬 채팅 보기';
            renderStagedList(); // 뷰 전환 적용
        };
        header.appendChild(viewToggleBtn);

        // 자동 추출 함수
        async function autoExtract() {
            const pageId = await generatePageId(window.location.href);
            
            const ogTitle = document.querySelector('meta[property="og:title"]')?.content || document.title || 'No Title';
            const ogDesc = document.querySelector('meta[property="og:description"]')?.content || 'No Description';
            const bodyYaml = cleanAndConvertToYaml(document.body);
            
            // OG 태그 메타데이터와 정제된 본문 YAML을 하나의 컨텍스트로 결합
            const yaml = `og_title: ${ogTitle}\nog_description: ${ogDesc}\n---\n${bodyYaml}`;
            
            // 1. 동일한 ID(주소)의 DRAFT가 있는지 확인하고, 내용이 완전히 동일하면 건너뜁니다.
            const existingIndex = stagedItems.findIndex(i => i.id === pageId && i.status === 'DRAFT');
            if (existingIndex !== -1 && stagedItems[existingIndex].context === yaml) {
                const skipLogDiv = document.createElement('div');
                skipLogDiv.className = 'user draft';
                skipLogDiv.style.cssText = 'color: gray; font-size: 11px; margin-top: 5px; display: block !important;';
                skipLogDiv.textContent = '[Auto-Extracted]: 내용이 동일하여 추가/업데이트를 생략합니다.';
                log.appendChild(skipLogDiv);
                log.scrollTop = log.scrollHeight;
                return;
            }
            
            // 2. UI 로그에 YAML 표시 (텍스트 노드 분리 및 가독성 개선)
            const autoLogDiv = document.createElement('div');
            autoLogDiv.className = 'user draft';
            autoLogDiv.style.cssText = 'white-space: pre-wrap; font-size: 11px; margin-top: 5px; display: block !important;';
            
            const strongText = document.createElement('strong');
            strongText.textContent = '[Auto-Extracted]:';
            autoLogDiv.appendChild(strongText);
            
            // 줄바꿈을 포함한 내용 영역 추가
            const contentSpan = document.createElement('div');
            contentSpan.style.marginTop = '2px';
            contentSpan.textContent = yaml.substring(0, 200) + (yaml.length > 200 ? '...' : '');
            
            autoLogDiv.appendChild(contentSpan);
            log.appendChild(autoLogDiv);
            
            const item = { 
                id: pageId, 
                host: window.location.hostname, 
                url: window.location.href, 
                title: ogTitle, 
                // DRAFT 탭일 때는 기본값인 COMMERCE를 넣고, 그 외 탭에서는 현재 탭 이름을 도메인으로 사용
                domain: currentTabFilter === 'DRAFT' ? 'COMMERCE' : currentTabFilter, 
                context: yaml, 
                status: 'DRAFT', 
                track: 'MAIN', 
                version: 1, 
                created_at: Date.now(), 
                updated_at: Date.now() 
            };
            
            // 3. UI에 즉시 추가하지 않고 백엔드로 원본 데이터만 전송 (비동기 처리 위임)
            const draftMessage = document.createElement('div');
            draftMessage.className = 'system';
            draftMessage.style.fontSize = '11px';
            draftMessage.style.opacity = '0.7';
            draftMessage.textContent = 'System: 분석 및 저장 중...';
            log.appendChild(draftMessage);
            log.scrollTop = log.scrollHeight;

            if (window.gemini_rpc) window.gemini_rpc("sync_data:" + JSON.stringify(item));
            // renderStagedList() 호출은 백엔드 응답 수신 시점으로 미룸
        }

        

        if(window.is_authenticated){
            const rightDiv = document.createElement('div');

            const extractBtn = document.createElement('button');
            extractBtn.style.cssText = 'margin-right:10px; padding: 4px 8px; background: #ddd; color: #000; border: none; border-radius: 13px; font-weight: bold; cursor: pointer;';
            extractBtn.id = 'extract-btn';
            extractBtn.textContent = 'Draft';
            extractBtn.onclick = autoExtract;
            
            rightDiv.appendChild(extractBtn);


            const pushBtn = document.createElement('button');
            pushBtn.id = 'push-btn';
            pushBtn.style.cssText = 'padding: 4px 8px; background: #007bff; color: white; border: none; border-radius: 13px; font-weight: bold; cursor: pointer;';
            pushBtn.textContent = 'Push';

            rightDiv.appendChild(pushBtn);

            header.appendChild(rightDiv);


            pushBtn.onclick = () => {
                const selected = Array.from(shadow.querySelectorAll('input:checked')).map(cb => cb.dataset.id);
                if (selected.length === 0) return;
                
                stagedItems.forEach(item => {
                    if (selected.includes(item.id)) {
                        item.is_progressing = true;
                    }
                });
                renderStagedList();

                const payload = stagedItems.filter(i => selected.includes(i.id));
                
                // 탭 간 상태 공유를 위해 localStorage 사용 및 진행 상태 표시
                localStorage.setItem('gemini_push_status', JSON.stringify({ status: 'processing', timestamp: Date.now() }));
                showProgressLog();

                if (window.gemini_rpc) window.gemini_rpc("push_data:" + JSON.stringify(payload));
            };
        }else{
            const loginBtn = document.createElement('button');

            loginBtn.textContent = 'Login';
            loginBtn.style.cssText = 'padding: 8px; background: #007bff; color: white; border: none; border-radius: 4px; font-weight: bold; cursor: pointer;';
            loginBtn.onclick = () => {
                if (window.is_authenticated) {
                    window.location.href = "https://terminal.logis.center/";
                } else {
                    if (window.gemini_rpc) {
                        // 인증되지 않았을 경우 Rust 백엔드로 CLI 인증 명령어 전달
                        window.gemini_rpc("login");
                    }
                }
            };

            header.appendChild(loginBtn);
        }


        
                
        
        const stagedList = document.createElement('div');
        stagedList.className = 'content';
        stagedList.id = 'staged-list';
        const log = document.createElement('div');
        log.id = 'log';
        stagedList.appendChild(log);

        const footer = document.createElement('div');
        footer.className = 'footer';
        
        
        

        const chatDiv = document.createElement('div');
        chatDiv.style.cssText = 'display: flex; gap: 5px;';
        
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.id = 'file-input';
        fileInput.accept = 'image/*,application/pdf';
        fileInput.style.cssText = 'display: none; width:38px;';
        
        const attachBtn = document.createElement('button');
        attachBtn.id = 'attach-btn';
        attachBtn.textContent = '📎';
        attachBtn.title = '파일 첨부 (이미지/PDF)';
        
        const cliInput = document.createElement('input');
        cliInput.type = 'text';
        cliInput.id = 'cli-input';
        cliInput.disabled = true;
        cliInput.placeholder = '메시지 입력...';
        cliInput.style.cssText = 'flex: 1;';
        
        const sendBtn = document.createElement('button');
        sendBtn.id = 'send-btn';
        sendBtn.style.cssText = 'width:38px;';
        sendBtn.textContent = '⌲';
        
        chatDiv.appendChild(fileInput);
        chatDiv.appendChild(attachBtn);
        chatDiv.appendChild(cliInput);
        chatDiv.appendChild(sendBtn);
        
        footer.appendChild(chatDiv);

        agentContainer.appendChild(header);
        agentContainer.appendChild(stagedList);
        agentContainer.appendChild(footer);

        shadow.appendChild(style);
        shadow.appendChild(agentContainer);

        let stagedItems = [];
        let currentTabFilter = window.default_tab || 'DRAFT';

        // UI 리스트를 갱신하는 독립 렌더링 함수
        function renderStagedList() {
            // 기존 아이템, 환경설정, 메시지 폼 일괄 삭제
            const items = stagedList.querySelectorAll('.staged-item, .login-container, .config-container, .empty-msg');
            items.forEach(item => item.remove());

            // 1. CONFIG 화면 렌더링 및 조기 종료
            if (currentTabFilter === 'CONFIG') {
                log.style.display = 'none'; // 채팅 숨김
                
                const configDiv = document.createElement('div');
                configDiv.className = 'config-container';
                configDiv.style.cssText = 'padding: 20px; display: flex; flex-direction: column; gap: 15px; text-align: center;';
                
                const title = document.createElement('h3');
                title.textContent = '환경설정';
                title.style.margin = '0';
                
                const clearBtn = document.createElement('button');
                clearBtn.textContent = '🗑️ 모든 데이터 비우기';
                clearBtn.style.cssText = 'padding: 10px; background: #dc3545; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; width: 100%;';
                clearBtn.onclick = () => {
                    if(confirm('정말 모든 데이터를 비우시겠습니까? 이 작업은 되돌릴 수 없습니다.')) {
                        stagedItems = [];
                        log.innerHTML = ''; // 화면 상 로그 초기화
                        if (window.gemini_rpc) window.gemini_rpc("clear_all_data");
                        alert('데이터가 초기화되었습니다.');
                        renderStagedList();
                    }
                };
                
                configDiv.appendChild(title);
                configDiv.appendChild(clearBtn);
                stagedList.appendChild(configDiv);
                
                return;
            }

            // 2. 보기 모드 (리스트 vs 채팅) 토글 처리
            if (isChatView) {
                log.style.display = 'flex';
            } else {
                log.style.display = 'none'; // 리스트 모드일 경우 채팅 숨김
            }

            // 3. 필터링 로직 (UPDATE 명칭 일괄 처리)
            const filtered = stagedItems.filter(item => {
                if (currentTabFilter === 'UPDATE') {
                    return item.status === 'UPDATE' || item.status === 'DRAFT'; // DRAFT 하위호환
                } else {
                    return item.domain === currentTabFilter;
                }
            });

            filtered.forEach(item => {
                const itemDiv = document.createElement('div');
                if (item.is_progressing) {
                    itemDiv.className = `staged-item user ${item.domain.toLowerCase()} progressing`;
                } else {
                    itemDiv.className = `staged-item user ${item.domain.toLowerCase()} ${item.status.toLowerCase()}`;
                }
                
                // 기획 적용: Drag and Drop 속성 활성화
                itemDiv.draggable = true;
                itemDiv.ondragstart = (e) => {
                    e.dataTransfer.setData('text/plain', item.id);
                    itemDiv.style.opacity = '0.5';
                };
                itemDiv.ondragend = () => {
                    itemDiv.style.opacity = '1';
                };
                
                // 뷰 모드에 따른 리스트 아이템 표시 여부 결정
                if (isChatView) {
                    itemDiv.style.display = 'none'; // 채팅 모드에서는 리스트 요소를 그리지 않음
                }
                
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.dataset.id = item.id;
                itemDiv.appendChild(checkbox);
                
                const infoText = ` [${item.domain}] ${item.title.substring(0,20)}... (v${item.version}) ${item.status}`;
                itemDiv.appendChild(document.createTextNode(infoText));
                
                const deleteBtn = document.createElement('button');
                deleteBtn.textContent = '❌';
                deleteBtn.style.cssText = 'margin-left: auto; background: transparent; border: none; font-size: 12px; cursor: pointer; padding: 0 5px; color: #999;';
                deleteBtn.onclick = () => {
                    stagedItems = stagedItems.filter(i => i.id !== item.id);
                    renderStagedList();
                    if (window.gemini_rpc) window.gemini_rpc("delete_data:" + item.id);
                };
                itemDiv.appendChild(deleteBtn);
                
                stagedList.appendChild(itemDiv);
            });

            if (filtered.length === 0 && !isChatView) {
                const empty = document.createElement('div');
                empty.className = 'empty-msg staged-item';
                empty.style.color = '#999';
                empty.textContent = '데이터가 없습니다.';
                stagedList.appendChild(empty);
            }
        }

        // 초기 로드 시 백엔드에 기존 DRAFT 목록 요청
        if (window.gemini_rpc) window.gemini_rpc("fetch_drafts");

        

        // 초기 탭 활성화를 위한 강제 1회 클릭 호출
        const initialBtn = Array.from(tabsContainer.children).find(b => b.dataset.tab === currentTabFilter) 
                        || tabsContainer.children[0];
        if(initialBtn) initialBtn.click();

        // Esc 키 입력 시 사이드바 패널 전체를 열고 닫습니다.
        window.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !e.repeat) {
                e.preventDefault();
                const isOpen = agentContainer.classList.toggle('open'); 
                if (isOpen) {
                    if (window.gemini_rpc) window.gemini_rpc("fetch_drafts");
                    renderStagedList();
                } else {
                    const items = stagedList.querySelectorAll('.staged-item, .login-container');
                    items.forEach(item => item.remove());
                }
            }
        });

        

        // 대기 상태 표시 UI 관리 함수
        function showProgressLog() {
            if (shadow.getElementById('progress-log')) return;
            const rpcLogDiv = document.createElement('div');
            rpcLogDiv.id = 'progress-log';
            rpcLogDiv.style.color = '#ff9800'; // 진행 중임을 나타내는 주황색
            const rpcStrong = document.createElement('strong');
            rpcStrong.textContent = 'AI (Processing): ';
            rpcLogDiv.appendChild(rpcStrong);
            rpcLogDiv.appendChild(document.createTextNode('데이터 마스킹 및 임베딩 생성 중입니다... 잠시만 기다려주세요.'));
            log.appendChild(rpcLogDiv);
            log.scrollTop = log.scrollHeight;
        }

        function hideProgressLog() {
            const progressLog = shadow.getElementById('progress-log');
            if (progressLog) progressLog.remove();
        }

        

        function sendChatMessage() {
            // DRAFT 탭에서는 전송 로직이 실행되지 않도록 차단
            if (currentTabFilter === 'DRAFT') {
                alert('도메인 탭을 선택해주세요.');
                return;
            }

            const text = cliInput.value.trim();
            if (!text) return;

            // UI에 사용자 메시지 표시
            const userLogDiv = document.createElement('div');
            userLogDiv.className = 'user';
            userLogDiv.style.cssText = 'margin-top: 10px;';
            const userStrong = document.createElement('strong');
            userStrong.textContent = 'You: ';
            userLogDiv.appendChild(userStrong);
            userLogDiv.appendChild(document.createTextNode(text));
            log.appendChild(userLogDiv);
            log.scrollTop = log.scrollHeight;

            // 백엔드로 채팅 요청 전송
            if (window.gemini_rpc) {
                const payload = {
                    messages: [{ role: 'user', parts: [{ text: text }] }],
                    model: 'gemini-3.1-flash-lite-preview'
                };
                console.log("[JS] Sending gemini_chat payload:", payload);
                window.gemini_rpc("gemini_chat:" + JSON.stringify(payload));
            }
            cliInput.value = '';
            
            // 스트리밍 응답을 새 영역에 쓰기 위해 초기화
            window._currentAiMessageDiv = null;
        }

        attachBtn.onclick = () => fileInput.click();

        fileInput.addEventListener('change', async (e) => {
            // DRAFT 탭에서는 파일 처리 로직이 실행되지 않도록 차단
            if (currentTabFilter === 'DRAFT') {
                alert('도메인 탭을 선택한 후 파일을 첨부해주세요.');
                fileInput.value = ''; // 선택된 파일 초기화
                return;
            }

            const file = e.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = async (event) => {
                const base64Data = event.target.result;
                // 파일 이름과 현재 시간을 기반으로 고유 ID 생성
                const pageId = await generatePageId(file.name + Date.now());
                
                const yaml = `file_name: ${file.name}\ntype: ${file.type}\nsize: ${file.size}\ndata: ${base64Data}`;
                
                // 파일 첨부 완료를 로그 화면에 표시
                const autoLogDiv = document.createElement('div');
                autoLogDiv.className = 'user draft';
                autoLogDiv.style.cssText = 'white-space: pre-wrap; font-size: 11px; margin-top: 5px; color: purple; display: block !important;';
                const strongText = document.createElement('strong');
                strongText.textContent = '[File-Attached]:\n';
                autoLogDiv.appendChild(strongText);
                autoLogDiv.appendChild(document.createTextNode(file.name + ' (' + file.type + ') 첨부 완료'));
                log.appendChild(autoLogDiv);
                log.scrollTop = log.scrollHeight;
                
                // 파일 데이터를 DRAFT 속성으로 정의
                const item = { 
                    id: pageId, 
                    host: 'local_file', 
                    url: 'file://' + file.name, 
                    title: file.name, 
                    // DRAFT 탭일 때는 기본값인 COMMERCE를 넣고, 그 외 탭에서는 현재 탭 이름을 도메인으로 사용
                    domain: currentTabFilter === 'DRAFT' ? 'COMMERCE' : currentTabFilter, 
                    context: yaml, 
                    status: 'DRAFT', 
                    track: 'MAIN', 
                    version: 1, 
                    created_at: Date.now(), 
                    updated_at: Date.now() 
                };
                
                stagedItems.push(item);
                renderStagedList();
                
                // Rust 백엔드로 DRAFT 상태 저장 요청
                if (window.gemini_rpc) window.gemini_rpc("sync_data:" + JSON.stringify(item));
                
                fileInput.value = ''; // 연속 선택이 가능하도록 input 값 초기화
            };
            reader.readAsDataURL(file);
        });

        sendBtn.onclick = sendChatMessage;
        cliInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                // 비활성화 상태에서는 엔터 키 동작 차단
                if (currentTabFilter === 'DRAFT') {
                    e.preventDefault();
                    return;
                }
                sendChatMessage();
            }
        });

        // 자동 실행 및 상태 유지
        if (document.readyState === 'complete') {
            autoExtract();
        } else {
            window.addEventListener('load', autoExtract);
        }

        // SPA (Single Page Application) 환경에서 URL 변경을 감지하여 Draft를 자동 추출
        let lastUrl = window.location.href;
        const urlObserver = new MutationObserver(() => {
            const currentUrl = window.location.href;
            if (currentUrl !== lastUrl) {
                lastUrl = currentUrl;
                // 페이지 렌더링이 완료될 시간을 주기 위해 지연 호출
                setTimeout(autoExtract, 1000);
            }
        });
        urlObserver.observe(document, { childList: true, subtree: true });

        // 다른 브라우저 탭에 상태 전파를 위한 storage 이벤트 리스너
        window.addEventListener('storage', (e) => {
            if (e.key === 'gemini_push_status' && e.newValue) {
                try {
                    const data = JSON.parse(e.newValue);
                    if (data.status === 'processing') {
                        showProgressLog();
                    } else if (data.status === 'complete') {
                        hideProgressLog();
                        // 이벤트 발생 탭 외의 다른 탭에서도 최종 완료 메시지를 동기화 출력
                        if (log) {
                            const rpcLogDiv = document.createElement('div');
                            rpcLogDiv.style.color = 'blue';
                            const rpcStrong = document.createElement('strong');
                            rpcStrong.textContent = 'AI: ';
                            rpcLogDiv.appendChild(rpcStrong);
                            rpcLogDiv.appendChild(document.createTextNode(data.message));
                            log.appendChild(rpcLogDiv);
                            log.scrollTop = log.scrollHeight;
                        }
                    }
                } catch(err) {}
            }
        });

        window.addEventListener('gemini_rpc_response', (e) => {
            try {
                // JSON 형태의 응답(fetch_drafts, sync_success) 처리
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
                } else if (data.type === 'sync_success') {
                    const record = data.payload;
                    // 기존 리스트에서 동일 ID 제거 후 분석 완료된 새 레코드 추가 (일원화)
                    stagedItems = stagedItems.filter(i => i.id !== record.id);
                    stagedItems.push(record);
                    
                    // 드롭다운 값을 분석된 도메인으로 변경하고 리스트 갱신
                    statusSelect.value = record.domain;
                    statusSelect.onchange(); 
                    
                    renderStagedList();
                    
                    const rpcLogDiv = document.createElement('div');
                    rpcLogDiv.className = 'system';
                    rpcLogDiv.style.cssText = 'font-size: 11px; margin-top: 5px; opacity: 0.8; color: #28a745 !important;';
                    rpcLogDiv.innerHTML = `<strong>System:</strong> [${record.domain}] 분석 완료 및 DRAFT 저장됨.`;
                    log.appendChild(rpcLogDiv);
                    log.scrollTop = log.scrollHeight;
                    return;
                } else if (data.type === 'error') {
                    alert('저장 실패: ' + data.message);
                    return;
                }
            } catch(err) {
                // JSON 파싱 실패 시 일반 텍스트 응답으로 간주하여 아래 로그 출력 진행
            }

            // Push 데이터 처리 완료 응답일 경우 상태 초기화 전파
            if (typeof e.detail === 'string' && (e.detail.includes('Data pushed successfully') || e.detail.includes('DB Error:'))) {
                localStorage.setItem('gemini_push_status', JSON.stringify({ status: 'complete', message: e.detail, timestamp: Date.now() }));
                hideProgressLog();
                
                if (e.detail.includes('Data pushed successfully')) {
                    stagedItems.forEach(item => {
                        if (item.is_progressing) {
                            item.is_progressing = false;
                            item.status = 'MAIN';
                        }
                    });
                } else {
                    stagedItems.forEach(item => {
                        if (item.is_progressing) {
                            item.is_progressing = false;
                        }
                    });
                }
                renderStagedList();
            }

            if (log) {
                if (e.detail === 'Streaming started') {
                    window._currentAiMessageDiv = null;
                    return;
                }
                
                const isSystemMessage = typeof e.detail === 'string' && (
                    e.detail.includes('Data synced') || 
                    e.detail.includes('Data pushed successfully') || 
                    e.detail.includes('DB Error:') ||
                    e.detail.includes('DevTools opened') ||
                    e.detail.includes('Error:')
                );

                if (isSystemMessage) {
                    window._currentAiMessageDiv = null; // 시스템 로그 발생 시 스트리밍 텍스트 분리
                }

                // 전달된 데이터(e.detail)가 객체인 경우 텍스트 추출 로직 추가
                const rawContent = e.detail;
                const textContent = (typeof rawContent === 'object' && rawContent !== null) 
                    ? (rawContent.text || JSON.stringify(rawContent)) 
                    : rawContent;

                if (!isSystemMessage && window._currentAiMessageDiv) {
                    // 동일한 대화 응답의 스트리밍 데이터 이어서 붙이기
                    window._currentAiMessageDiv.appendChild(document.createTextNode(textContent));
                } else {
                    const rpcLogDiv = document.createElement('div');
                    rpcLogDiv.className = 'system';
                    rpcLogDiv.style.cssText = isSystemMessage ? 'font-size: 11px; margin-top: 5px; opacity: 0.7;' : 'margin-top: 5px; line-height: 1.4;';
                    const rpcStrong = document.createElement('strong');
                    rpcStrong.textContent = isSystemMessage ? 'System: ' : 'AI: ';
                    rpcLogDiv.appendChild(rpcStrong);
                    rpcLogDiv.appendChild(document.createTextNode(textContent));
                    log.appendChild(rpcLogDiv);
                    
                    if (!isSystemMessage) {
                        window._currentAiMessageDiv = rpcLogDiv;
                    }
                }
                log.scrollTop = log.scrollHeight;
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
        
        observer.observe(document, { childList: true, subtree: true });

        window.addEventListener('pageshow', runOnce);
    }

})();
"#;

async fn setup_page(browser: Arc<Browser>, page: chromiumoxide::Page, is_authenticated: bool) -> Result<(), Box<dyn std::error::Error>> {
    // RPC 바인딩 등록
    let _ = page.execute(AddBindingParams::new("gemini_rpc")).await;
    
    let default_tab = load_default_tab();
    
    // 인증 상태 변수, 기본 탭 변수, UI 스크립트를 하나로 묶어 레이스 컨디션(주입 지연) 방지
    let full_script = format!(
        "window.is_authenticated = {};\nwindow.default_tab = \"{}\";\n{}", 
        is_authenticated, default_tab, OVERLAY_SCRIPT
    );
    
    // 이미 로드된 현재 페이지 상태에서 UI가 즉시 나타나도록 강제 실행
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
                            // Image OCR Logic Integration
                            if record.url.starts_with("file://") && record.context.contains("data:image/") {
                                println!("[Rust] Image detected. Running GLM-OCR...");
                                
                                // Extract base64 image data
                                if let Some(base64_part) = record.context.split("data:").nth(1) {
                                    let full_data_url = format!("data:{}", base64_part.trim());
                                    
                                        let ocr_result = {
                                            let mut model_guard = OCR_MODEL.lock().unwrap();
                                            if model_guard.is_none() {
                                                println!("[Rust] Loading GLM-OCR model...");
                                                let model_path = "..\\models\\glm_ocr";
                                                
                                                // 0. Device selection logic
                                                #[cfg(any(feature = "cuda", feature = "metal"))]
                                                let device = if cfg!(feature = "cuda") {
                                                    Device::new_cuda(0).unwrap_or(Device::Cpu)
                                                } else if cfg!(feature = "metal") {
                                                    Device::new_metal(0).unwrap_or(Device::Cpu)
                                                } else {
                                                    Device::Cpu
                                                };
                                                #[cfg(not(any(feature = "cuda", feature = "metal")))]
                                                let device = Device::Cpu;

                                                match GlmOcrGenerateModel::init(model_path, Some(&device), None) {
                                                    Ok(model) => *model_guard = Some(model),
                                                    Err(e) => eprintln!("[Rust] OCR Model Load Error: {}", e),
                                                }
                                            }
                                            
                                            if let Some(model) = model_guard.as_mut() {
                                            let params = ChatCompletionParameters {
                                                messages: vec![Message {
                                                    role: "user".to_string(),
                                                    parts: vec![Part {
                                                        text: "Extract all text from this image.".to_string(),
                                                        image_url: Some(full_data_url),
                                                    }],
                                                }],
                                                model: "glm-ocr".to_string(),
                                                max_tokens: Some(1024),
                                                temperature: Some(0.0),
                                                top_p: None,
                                                top_k: None,
                                                repeat_penalty: None,
                                                repeat_last_n: None,
                                                seed: None,
                                            };
                                            
                                            match model.generate(params) {
                                                Ok(res) => {
                                                    if let Some(choice) = res.choices.first() {
                                                        choice.message.content.clone()
                                                    } else {
                                                        "OCR: No text found".to_string()
                                                    }
                                                },
                                                Err(e) => format!("OCR Error: {}", e),
                                            }
                                        } else {
                                            "OCR: Model not loaded".to_string()
                                        }
                                    };
                                    
                                    println!("[Rust] OCR Result: {}", ocr_result);
                                    // Append OCR result to context or replace it
                                    record.context = format!("{}\n---\n[OCR_EXTRACTED_TEXT]\n{}", record.context, ocr_result);
                                }
                            }

                            let updated_record = record.clone();
                            match db::save_records(vec![record], None).await {
                                Ok(_) => {
                                    json!({
                                        "type": "sync_success",
                                        "payload": updated_record
                                    }).to_string()
                                },
                                Err(e) => json!({"type": "error", "message": format!("DB Error: {}", e)}).to_string(),
                            }
                        },
                        Err(e) => json!({"type": "error", "message": format!("JSON Error: {}", e)}).to_string(),
                    }
                } else if payload == "clear_all_data" {
                    match db::get_or_create_table().await {
                        Ok(table) => {
                            // 테이블 내 모든 데이터 삭제
                            let _ = table.delete("id IS NOT NULL").await;
                            json!({
                                "type": "system_msg",
                                "message": "All data cleared from LanceDB"
                            }).to_string()
                        },
                        Err(e) => json!({"type": "error", "message": format!("DB Error: {}", e)}).to_string(),
                    }
                } else if payload.starts_with("delete_data:") {
                    let id_to_delete = &payload["delete_data:".len()..];
                    match db::get_or_create_table().await {
                        Ok(table) => {
                            let expr = format!("id = '{}'", id_to_delete);
                            let _ = table.delete(&expr).await;
                            format!("Item deleted from LanceDB: {}", id_to_delete)
                        },
                        Err(e) => format!("DB Error: {}", e),
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
                            // 0. Device selection logic
                            #[cfg(any(feature = "cuda", feature = "metal"))]
                            let device = if cfg!(feature = "cuda") {
                                Device::new_cuda(0).unwrap_or(Device::Cpu)
                            } else if cfg!(feature = "metal") {
                                Device::new_metal(0).unwrap_or(Device::Cpu)
                            } else {
                                Device::Cpu
                            };
                            #[cfg(not(any(feature = "cuda", feature = "metal")))]
                            let device = Device::Cpu;

                            // 1. PII Masking
                            let privacy_model_path = std::path::PathBuf::from("..\\models\\privacy-filter");
                            if let Ok(privacy_engine) = PrivacyFilterModel::load(&privacy_model_path, &device) {
                                println!("[Rust] Privacy Filter Loaded. Masking PII...");
                                for record in &mut records {
                                    if let Ok(spans) = privacy_engine.predict(&record.context) {
                                        record.masking = mask_pii(&record.context, &spans);
                                    }
                                }
                            } else {
                                eprintln!("[Rust] Warning: Failed to load privacy filter model");
                            }

                            // 2. Push 요청 시에만 임베딩 모델을 메모리에 로드합니다.
                            let model_path = std::path::PathBuf::from("..\\models\\embeddings");
                            match embedding::EmbeddingModel::new_with_device(model_path, &device) {
                                Ok(model) => {
                                    for record in &mut records {
                                        match model.embed(&record.masking) {
                                            Ok(vector) => record.vector = vector,
                                            Err(e) => eprintln!("Embedding Error: {}", e),
                                        }
                                    }
                                },
                                Err(e) => eprintln!("Model load error: {}", e),
                            }
                            
                            match db::save_records(records, None).await {
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
                } else if payload.starts_with("gemini_chat:") {
                    let data = payload["gemini_chat:".len()..].to_string();
                    let page_c = page_clone.clone();
                    tokio::spawn(async move {
                        println!("[Rust] Received gemini_chat payload: {}", data);
                        match serde_json::from_str::<serde_json::Value>(&data) {
                            Ok(v) => {
                                let messages: Vec<gemini::types::ChatMessage> = serde_json::from_value(v["messages"].clone()).unwrap_or_default();
                                let mut model = v["model"].as_str().unwrap_or("gemini-3.1-flash-lite-preview").to_string();
                                
                                // Gemini API의 /models 엔드포인트를 조회하여 제공 가능한 모델 중 가장 낮은 버전의 flash-lite를 동적으로 선택합니다.
                                let token_opt = gemini::auth::get_auth_token().await.ok();
                                if let Some(token) = token_opt {
                                    if token.starts_with("ya29.") || token.starts_with("ya29_") {
                                        // CLI OAuth 토큰은 generativelanguage 권한이 없으므로 API 조회를 생략합니다.
                                        println!("[Rust] OAuth 토큰 감지됨. 모델 목록 조회를 건너뛰고 기본 모델을 사용합니다.");
                                    } else {
                                        let url = "https://generativelanguage.googleapis.com/v1beta/models";
                                        let http_client = reqwest::Client::new();
                                        
                                        let request = http_client.get(url);
                                        let request = request.header("x-goog-api-key", &token);
                                        
                                        if let Ok(res) = request.send().await {
                                            if let Ok(models_resp) = res.json::<serde_json::Value>().await {
                                                if let Some(models_array) = models_resp["models"].as_array() {
                                                    let mut flash_lite_models: Vec<String> = models_array.iter()
                                                        .filter_map(|m| m["name"].as_str())
                                                        .map(|name| name.strip_prefix("models/").unwrap_or(name).to_string())
                                                        .filter(|name| name.contains("flash-lite"))
                                                        .collect();
                                                    
                                                    // 알파벳 및 숫자 정렬을 수행하면 낮은 버전(예: 2.0)이 배열의 가장 앞으로 오게 됩니다.
                                                    flash_lite_models.sort();
                                                    if let Some(lowest_version) = flash_lite_models.first() {
                                                        println!("[Rust] API 목록에서 가장 낮은 버전의 flash-lite 모델을 동적 선택했습니다: {}", lowest_version);
                                                        model = lowest_version.clone();
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                println!("[Rust] Using model: {}", model);
                                let client = gemini::client::GeminiClient::new(model);
                               
                                // 에러 처리에 사용할 브라우저 페이지 객체를 클로저 이동 전에 미리 복제합니다.
                                let page_c_err = page_c.clone();
                                
                                // Send 트레이트 에러를 방지하기 위해 stream_message의 결과를 바로 match로 평가하여
                                // Error 객체가 await 지점까지 생존하지 않도록 스코프를 완전히 분리(소비)합니다.
                                let err_script_opt = match client.stream_message(messages, move |chunk| {
                                    let script = format!(
                                        "window.dispatchEvent(new CustomEvent('gemini_rpc_response', {{ detail: {} }}));",
                                        json!(chunk)
                                    );
                                    // evaluate는 비동기 함수(Future)이므로 동기 클로저 내에서 tokio::spawn으로 감싸 실행을 보장합니다.
                                    let page_inner = page_c.clone();
                                    tokio::spawn(async move {
                                        if let Err(e) = page_inner.evaluate(script).await {
                                            eprintln!("[Rust] Evaluate error in stream: {}", e);
                                        }
                                    });
                                }).await {
                                    Err(e) => {
                                        eprintln!("[Rust] Stream execute error: {}", e);
                                        Some(format!(
                                            "window.dispatchEvent(new CustomEvent('gemini_rpc_response', {{ detail: {} }}));",
                                            json!(format!("Error: {}", e))
                                        ))
                                    }
                                    Ok(_) => None,
                                };

                                // 비동기(.await) 호출은 에러 객체가 메모리에서 완전히 해제된 안전한 스코프에서 진행합니다.
                                if let Some(error_script) = err_script_opt {
                                    let _ = page_c_err.evaluate(error_script).await;
                                }
                            },
                            Err(e) => eprintln!("[Rust] Failed to parse gemini_chat json: {}", e),
                        }
                    });
                    "Streaming started".to_string()
                } else if payload == "login" {
                    println!("[Rust] Login command received via RPC. Spawning auth process in background...");
                    
                    // Windows 전용 프로세스 확장 기능을 가져옵니다. (오타 수정됨)
                    #[cfg(target_os = "windows")]
                    use std::os::windows::process::CommandExt;
                    
                    // 터미널 창 생성 방지 플래그
                    const CREATE_NO_WINDOW: u32 = 0x08000000;

                    let mut command = std::process::Command::new("cmd");
                    command.args(&["/C", "gemini auth"]);
                    
                    // Windows 환경일 때만 창 숨김 플래그를 주입합니다.
                    #[cfg(target_os = "windows")]
                    command.creation_flags(CREATE_NO_WINDOW);

                    match command.spawn() {
                        Ok(_) => {
                            // 터미널 대신 사용자에게 브라우저를 확인하라는 메시지를 JSON으로 반환
                            json!({
                                "type": "sync_success", // 기존 핸들러 활용을 위해 타입 지정
                                "domain": "SYSTEM",
                                "payload": "인증 브라우저가 실행되었습니다. 로그인을 완료해주세요."
                            }).to_string()
                        },
                        Err(e) => format!("Failed to run background auth: {}", e),
                    }
                } else {
                    format!("Unknown RPC command: {}", payload)
                };
                let script = if response.starts_with('{') && response.ends_with('}') {
                    // response is already a JSON string (likely from json!().to_string())
                    format!(
                        "window.dispatchEvent(new CustomEvent('gemini_rpc_response', {{ detail: {} }}));",
                        response
                    )
                } else {
                    format!(
                        "window.dispatchEvent(new CustomEvent('gemini_rpc_response', {{ detail: {} }}));",
                        json!(response)
                    )
                };
                let _ = page_clone.evaluate(script).await;
            }
        }
    });
    Ok(())
}

async fn run_mcp_server() -> Result<(), Box<dyn std::error::Error>> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
    let stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let mut reader = tokio::io::BufReader::new(stdin).lines();

    while let Ok(Some(line)) = reader.next_line().await {
        if let Ok(req) = serde_json::from_str::<serde_json::Value>(&line) {
            let method = req["method"].as_str().unwrap_or("");
            let id = req["id"].clone();

            let response = match method {
                "initialize" => {
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {
                                "tools": {}
                            },
                            "serverInfo": {
                                "name": "terminal-logis-mcp",
                                "version": "1.0.0"
                            }
                        }
                    })
                },
                "tools/list" => {
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "tools": [
                                {
                                    "name": "fetch_drafts",
                                    "description": "로컬 데이터베이스에서 DRAFT 상태의 물류 및 상거래 데이터를 검색합니다.",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {}
                                    }
                                },
                                {
                                    "name": "ask_assistant",
                                    "description": "LanceDB 컨텍스트를 기반으로 사용자 질문에 답변하고 PII를 자동으로 마스킹합니다.",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "query": { "type": "string", "description": "사용자의 질문 내용" }
                                        },
                                        "required": ["query"]
                                    }
                                }
                            ]
                        }
                    })
                },
                "tools/call" => {
                    let tool_name = req["params"]["name"].as_str().unwrap_or("");
                    let args = &req["params"]["arguments"];

                    if tool_name == "fetch_drafts" {
                        match db::fetch_drafts().await {
                            Ok(drafts) => {
                                let text = serde_json::to_string_pretty(&drafts).unwrap_or_default();
                                json!({
                                    "jsonrpc": "2.0",
                                    "id": id,
                                    "result": {
                                        "content": [{ "type": "text", "text": text }]
                                    }
                                })
                            },
                            Err(e) => {
                                json!({
                                    "jsonrpc": "2.0",
                                    "id": id,
                                    "error": { "code": -32603, "message": e.to_string() }
                                })
                            }
                        }
                    } else if tool_name == "ask_assistant" {
                        let query = args["query"].as_str().unwrap_or("");
                        let client = gemini::client::GeminiClient::new("gemini-3.1-flash-lite-preview".to_string());
                        
                        match gemini_gui_lib::assistant::Assistant::new(client, "..\\models\\privacy-filter") {
                            Ok(assistant) => {
                                match assistant.answer_question(query).await {
                                    Ok(ans) => {
                                        json!({
                                            "jsonrpc": "2.0",
                                            "id": id,
                                            "result": {
                                                "content": [{ "type": "text", "text": ans }]
                                            }
                                        })
                                    },
                                    Err(e) => {
                                        json!({
                                            "jsonrpc": "2.0",
                                            "id": id,
                                            "error": { "code": -32603, "message": e.to_string() }
                                        })
                                    }
                                }
                            },
                            Err(e) => {
                                json!({
                                    "jsonrpc": "2.0",
                                    "id": id,
                                    "error": { "code": -32603, "message": e.to_string() }
                                })
                            }
                        }
                    } else {
                        json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "error": { "code": -32601, "message": "Method not found" }
                        })
                    }
                },
                _ => {
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": { "code": -32601, "message": "Method not found" }
                    })
                }
            };

            let mut res_str = serde_json::to_string(&response).unwrap_or_default();
            res_str.push('\n');
            let _ = stdout.write_all(res_str.as_bytes()).await;
            let _ = stdout.flush().await;
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--mcp") {
        return run_mcp_server().await;
    }

    // 인증 여부에 따른 초기 URL 설정 (URL을 먼저 계산합니다)
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).unwrap_or_default();
    let auth_path = std::path::PathBuf::from(home).join(".gemini/oauth_creds.json");
    let is_authenticated = auth_path.exists();
    let start_url = if is_authenticated { "about:blank" } else { "https://terminal.logis.center/" };

    let args = vec![
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
    let is_auth_for_events = is_authenticated;
    tokio::task::spawn(async move {
        while let Some(event) = target_events.next().await {
            if event.target_info.r#type == "page" {
                let tid = event.target_info.target_id.clone();
                let b_inner = b_target.clone();
                let is_auth_inner = is_auth_for_events;
                tokio::task::spawn(async move {
                    // CDP 연결 확보를 위한 최소한의 시간 대기
                    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
                    
                    if let Ok(page) = b_inner.get_page(tid).await {
                        // 페이지 객체 확보 후 바인딩 및 주입 설정을 즉시 수행하여 '페이지 로드 시작' 단계부터 스크립트가 살아있게 함
                        let _ = page.execute(EnableParams::default()).await;
                        
                        let default_tab = load_default_tab();
                        let full_script = format!(
                            "window.is_authenticated = {};\nwindow.default_tab = \"{}\";\n{}", 
                            is_auth_inner, default_tab, OVERLAY_SCRIPT
                        );
                        let _ = page.execute(AddScriptToEvaluateOnNewDocumentParams::new(full_script)).await;
                        
                        // 특정 페이지 로딩 상태(예: DOMContentLoaded)까지 기다리지 않고 즉시 셋업 시도
                        let _ = setup_page(b_inner.clone(), page, is_auth_inner).await;
                    }
                });
            }
        }
    });

    if !is_authenticated {
        println!("[Rust] Authentication required. Redirecting to login...");
    }

    // 브라우저 실행 인자로 주입한 시작 URL이 로드된 유일한 최초 탭을 가져와 환경을 설정합니다.
    // about:blank를 경유하지 않으므로 Target ID가 변경되지 않아 Esc 키와 RPC 통신이 영구적으로 보장됩니다.
    if let Ok(pages) = browser.pages().await {
        if let Some(page) = pages.first() {
            let _ = page.execute(EnableParams::default()).await;
            
            let default_tab = load_default_tab();
            let full_script = format!(
                "window.is_authenticated = {};\nwindow.default_tab = \"{}\";\n{}", 
                is_authenticated, default_tab, OVERLAY_SCRIPT
            );
            // 모든 새 문서에 스크립트가 자동 실행되도록 브라우저 내부 설정
            let _ = page.execute(AddScriptToEvaluateOnNewDocumentParams::new(full_script)).await;
            // RPC 바인딩 및 이벤트 리스너 세팅
            let _ = setup_page(browser.clone(), page.clone(), is_authenticated).await;
        }
    }

    tokio::select! {
        _ = tokio::signal::ctrl_c() => println!("\n[Rust] Shutting down..."),
        _ = rx.recv() => println!("\n[Rust] Browser closed, shutting down..."),
    }
    Ok(())
}
