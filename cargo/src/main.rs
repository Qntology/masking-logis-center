use terminal_logis_center::{db, privacy_filter, glm_ocr, params}; // embedding 제거
use privacy_filter::viterbi::PrivacySpan; // PrivacyFilterModel 제거
use candle_core::Device;
use glm_ocr::generate::{GlmOcrGenerateModel, GenerateModel};
use params::chat::{ChatCompletionParameters, Message, Part};
use std::sync::Mutex;
use lazy_static::lazy_static;

use std::sync::atomic::{AtomicBool, Ordering};

lazy_static! {
    static ref OCR_MODEL: Mutex<Option<GlmOcrGenerateModel>> = Mutex::new(None);
    static ref PRIVACY_MANAGER: Mutex<Option<terminal_logis_center::privacy_filter::masking::PrivacyManager>> = Mutex::new(None);
    static ref EMBEDDING_MODEL: Mutex<Option<terminal_logis_center::embedding::EmbeddingModel>> = Mutex::new(None);
    static ref GLOBAL_PROGRESS: Mutex<Option<serde_json::Value>> = Mutex::new(None);
    // 🚀 Push 작업을 실시간으로 중단하기 위한 전역 플래그입니다.
    static ref PUSH_CANCEL_SIGNAL: AtomicBool = AtomicBool::new(false);
}

// Simplified stub for chat completion
async fn _get_chat_completion(_messages: Vec<serde_json::Value>, _api_key: String, _model: String) -> Result<String, String> {
    Ok("[System] Gemini 서비스가 비활성화되었습니다.".to_string())
}

use chromiumoxide::browser::{Browser, BrowserConfig};
use chromiumoxide::cdp::browser_protocol::page::{AddScriptToEvaluateOnNewDocumentParams, EnableParams};
use chromiumoxide::cdp::browser_protocol::target::EventTargetCreated;
use chromiumoxide::cdp::js_protocol::runtime::{AddBindingParams, EventBindingCalled};
use futures::StreamExt;
use serde_json::json;
use std::sync::Arc;

// 🚀 OS 커널 레벨에서 가비지 컬렉터 강제 호출하여 RAM/VRAM 캐시를 즉시 반환하는 헬퍼 함수
fn force_memory_cleanup() {
    #[cfg(target_os = "windows")]
    unsafe {
        // aa.rs의 방식을 따라 SetProcessWorkingSetSizeEx와 플래그를 사용하여 메모리를 강제 해제합니다.
        // GetCurrentProcess()의 결과값인 -1 핸들을 직접 사용합니다.
        let handle = -1isize;
        let min_size = usize::MAX;
        let max_size = usize::MAX;
        // QUOTA_LIMITS_HARDWS_MIN_DISABLE (2) | QUOTA_LIMITS_HARDWS_MAX_DISABLE (4) = 6
        let flags = 6u32; 
        
        windows_sys::Win32::System::Memory::SetProcessWorkingSetSizeEx(handle, min_size, max_size, flags);
    }
    #[cfg(target_os = "linux")]
    unsafe { 
        // 🚀 [Linux] libc를 통해 glibc의 캐시된 힙 메모리를 OS로 즉시 강제 반환합니다.
        libc::malloc_trim(0); 
    }
    #[cfg(target_os = "macos")]
    unsafe { 
        // 🚀 [macOS] Darwin 커널의 모든 메모리 존(zone)에 대해 메모리 압박 해제를 강제 실행하여 캐시를 비웁니다.
        extern "C" { fn malloc_zone_pressure_relief(zone: *mut libc::c_void, goal: usize) -> usize; }
        malloc_zone_pressure_relief(std::ptr::null_mut(), 0); 
    }
}

fn _mask_pii(text: &str, spans: &[PrivacySpan]) -> String {
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

#[derive(serde::Deserialize, Default)]
struct AppConfig {
    #[serde(default)]
    default_tab: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    custom_tabs: Option<Vec<String>>,
    #[serde(default)]
    auto_extract: Option<bool>,
    #[serde(default)]
    enable_masking: Option<bool>,
}

fn load_app_config() -> AppConfig {
    let app_dir = terminal_logis_center::utils::get_app_dir();
    // 사용자가 설정을 저장할 때 사용하는 app_config.json을 우선적으로 읽습니다.
    let config_path = app_dir.join("app_config.json");
    if let Ok(content) = std::fs::read_to_string(&config_path) {
        if let Ok(config) = serde_json::from_str::<AppConfig>(&content) {
            return config;
        }
    }
    // 하위 호환성을 위해 config.json도 체크합니다.
    let legacy_path = app_dir.join("config.json");
    if let Ok(content) = std::fs::read_to_string(&legacy_path) {
        if let Ok(config) = serde_json::from_str::<AppConfig>(&content) {
            return config;
        }
    }
    AppConfig::default()
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

    function extractVisibleText() {
        return document.body.innerText || document.body.textContent || '';
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
            .btn-delete { background: #dc3545 !important; color: white !important; border-color: #dc3545 !important; }
            .btn-delete:hover { background: #c82333 !important; }
            .item-row-wrapper { display: flex; flex-direction: column; border-bottom: 1px solid #f0f0f0; transition: opacity 0.3s ease; }
            .item-row { display: flex; align-items: center; gap: 10px; padding: 8px; cursor: pointer; transition: background 0.2s; }
            .item-row:hover { background: #f9f9f9; }
            .item-row input[type="checkbox"] { cursor: pointer; }
            .item-detail { display: none; padding: 10px 10px 10px 32px; font-size: 11px; color: #555; background: #fafafa; border-top: 1px dashed #eee; white-space: pre-wrap; max-height: 200px; overflow-y: auto; word-break: break-all; }
            .item-detail.open { display: block; }
            /* 🚀 CSV 테이블용 추가 스타일 */
            .item-detail table { width: 100%; border-collapse: collapse; margin-top: 5px; background: white; }
            .item-detail th { background: #eee; font-weight: bold; }
            .item-detail td, .item-detail th { border: 1px solid #ddd; padding: 4px; text-align: left; }
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
            footer { padding: 10px 15px; background: #f8f9fa !important; border-top: 1px solid #eee; display: flex !important; gap: 8px; flex-shrink: 0; align-items: center; }
            footer input[type="file"] { width: 140px; font-size: 12px; cursor: pointer; }
            footer input[type="text"] { flex: 1; padding: 8px; border: 1px solid #ddd; border-radius: 4px; font-size: 13px; }
            footer button { padding: 8px 15px; background: #333 !important; color: #fff !important; border: none; border-radius: 4px; font-size: 13px; font-weight: bold; cursor: pointer; }
            footer button:hover { background: #555 !important; }
            button { cursor: pointer; padding: 5px 10px; border:0; }
        `;

        const agentContainer = document.createElement('div');
        agentContainer.id = 'agent-container';
        
        // 🚀 1. 헤더 영역을 검색창으로 변경
        const header = document.createElement('header');
        window.searchQuery = ''; // 검색어 상태 저장
        const searchInputBox = document.createElement('input');
        searchInputBox.type = 'text';
        searchInputBox.placeholder = 'Search keyword...';
        searchInputBox.style.width = '100%';
        searchInputBox.style.padding = '8px 12px';
        searchInputBox.style.border = '1px solid #ccc';
        searchInputBox.style.borderRadius = '4px';
        searchInputBox.style.fontSize = '13px';
        searchInputBox.oninput = (e) => {
            window.searchQuery = e.target.value.toLowerCase();
            window.currentPage = 1; // 🚀 검색 시 페이지 초기화
            renderStagedList();
        };
        header.appendChild(searchInputBox);

        // 🚀 UI 락을 위한 상태 변수들
        let isProcessing = false;
        let processingIds = []; // 현재 처리(Push) 중인 아이템 ID 목록
        let pushStartTime = 0; // 🚀 Race Condition 방지를 위한 Push 시작 시간 기록

        // 🚀 2. 리스트 상단으로 배치될 List Header 생성
        const listHeader = document.createElement('div');
        listHeader.style.display = 'flex';
        listHeader.style.justifyContent = 'space-between';
        listHeader.style.alignItems = 'center';
        listHeader.style.paddingBottom = '10px';
        listHeader.style.borderBottom = '1px solid #eee';
        listHeader.style.marginBottom = '10px';

        const listHeaderLeft = document.createElement('div');
        listHeaderLeft.className = 'header-left';

        // 전체 선택 체크박스
        const selectAllCheck = document.createElement('input');
        selectAllCheck.type = 'checkbox';
        selectAllCheck.title = 'Select All';
        selectAllCheck.onclick = (e) => {
            // 🚀 DOM(보이는 10개) 기준이 아닌 필터링된 전체 목록(100개 등)을 기준으로 메모리에 추가합니다.
            const query = window.searchQuery || '';
            const filtered = stagedItems.filter(i => {
                if (i.domain !== currentTabFilter) return false;
                if (!query) return true;
                const matchTitle = i.title && i.title.toLowerCase().includes(query);
                const matchContent = i.context && i.context.toLowerCase().includes(query);
                const matchMask = i.masking && i.masking.toLowerCase().includes(query);
                return matchTitle || matchContent || matchMask;
            });
            
            filtered.forEach(item => {
                if (e.target.checked) {
                    checkedSessionIds.add(item.id);
                } else {
                    checkedSessionIds.delete(item.id);
                }
            });
            
            // 현재 렌더링된 체크박스들의 뷰만 업데이트
            const checkboxes = log.querySelectorAll('.item-checkbox');
            checkboxes.forEach(cb => {
                cb.checked = checkedSessionIds.has(cb.dataset.id);
            });
            
            updatePushBtnState();
        };

        listHeaderLeft.appendChild(selectAllCheck);
        listHeader.appendChild(listHeaderLeft);

        const actionContainer = document.createElement('div');
        actionContainer.className = 'header-actions';

        function getPageMeta() {
            const ogTitle = document.querySelector('meta[property="og:title"]')?.content || document.title;
            const ogDesc = document.querySelector('meta[property="og:description"]')?.content || '';
            return ogDesc ? `${ogTitle}\n${ogDesc}` : ogTitle;
        }

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'btn-delete';
        deleteBtn.textContent = 'Delete';
        deleteBtn.style.display = 'none'; 
        deleteBtn.onclick = () => {
            // 🚀 [Fix] 처리 중인 아이템은 절대 삭제되지 않도록 필터링하여 강력히 방어합니다.
            const selectedIds = Array.from(checkedSessionIds);
            const idsToDelete = selectedIds.filter(id => !processingIds.includes(id));
            
            if (idsToDelete.length === 0) {
                alert('현재 Pushing 진행 중인 아이템은 삭제할 수 없습니다.');
                return;
            }
            if (idsToDelete.length !== selectedIds.length) {
                alert('진행 중인 아이템을 제외한 나머지 항목만 삭제됩니다.');
            }
            
            // 🚀 로컬 상태에 삭제된 ID를 기록하여 focus 이벤트로 인한 좀비 복구를 원천 차단합니다.
            idsToDelete.forEach(id => {
                deletedSessionIds.add(id);
                checkedSessionIds.delete(id); // 🚀 삭제된 아이템은 체크 유지 목록에서도 제거합니다.
            });
            
            if (window.rpc) {
                window.rpc("delete_drafts:" + JSON.stringify(idsToDelete));
            }
            stagedItems = stagedItems.filter(i => !idsToDelete.includes(i.id));
            updateGnbUI();
            renderStagedList();
        };

        const draftBtn = document.createElement('button');
        draftBtn.textContent = 'Draft (0)';
        draftBtn.onclick = async () => {
            // 🚀 전처리(Push) 중에도 Draft 추가가 가능하도록 if (isProcessing) return; 제거
            const pageId = await generatePageId(window.location.href);
            
            deletedSessionIds.delete(pageId);
            
            const extractedText = extractVisibleText();
            const item = { 
                id: pageId, 
                host: window.location.host,
                url: window.location.href,
                title: getPageMeta(), 
                domain: currentTabFilter,
                context: extractedText, 
                status: 'DRAFT',
                track: '',
                version: 1,
                created_at: Date.now(),
                updated_at: Date.now()
            };
            if (window.rpc) {
                window.rpc("sync_data:" + JSON.stringify(item));
            }
        };

        const pushBtn = document.createElement('button');
        pushBtn.className = 'btn-push';
        pushBtn.textContent = 'Push (0)';
        pushBtn.disabled = true;

        function updatePushBtnState() {
            // 🚀 DOM 기준이 아닌 세션(메모리) 기준으로 선택된 전체 ID들을 가져옵니다.
            const selectedIds = Array.from(checkedSessionIds);
            const checkedCount = selectedIds.length;

            // 🚀 체크박스가 1개 이상 선택되었을 때만 Footer(입력창 및 Submit 영역)를 노출합니다.
            if (currentTabFilter !== 'CONFIG' && currentTabFilter !== 'PROMPT') {
                footer.style.display = (checkedCount > 0) ? 'flex' : 'none';
            }

            // 🚀 작업 중일 때 Push 버튼을 Cancel 버튼으로 전환합니다.
            if (isProcessing) {
                pushBtn.disabled = false;
                pushBtn.textContent = 'Cancel';
                pushBtn.style.background = '#ffc107'; // 경고색(노란색)
                deleteBtn.disabled = true;
                draftBtn.disabled = false; // 🚀 Pushing 진행 중에도 별개로 Draft 아이템 추가가 가능하도록 활성화 유지
                return;
            }

            const draftCount = stagedItems.filter(i => selectedIds.includes(i.id) && i.status !== 'PUSHED').length;
            
            // 🚀 필수 모델 설치 여부를 검증합니다. (임베딩은 무조건 필수, 마스킹은 켜져 있을 때만 필수)
            const hasEmbedding = window.model_status && window.model_status['Embedding'];
            const hasPrivacyFilter = window.model_status && window.model_status['Privacy-Filter'];
            const isModelReady = hasEmbedding && (!window.enable_masking || hasPrivacyFilter);

            pushBtn.style.background = isModelReady ? '#007bff' : '#dc3545'; // 모델 미비 시 빨간색
            pushBtn.disabled = (draftCount === 0) || !isModelReady;
            
            if (!isModelReady && draftCount > 0) {
                pushBtn.textContent = 'Model Required';
            } else {
                pushBtn.textContent = `Push (${draftCount})`;
            }

            // 🚀 Delete 버튼에 선택된 아이템 개수를 표기합니다.
            deleteBtn.textContent = `Delete (${checkedCount})`;
            deleteBtn.style.display = (checkedCount > 0) ? 'inline-block' : 'none';
            deleteBtn.disabled = false;
            draftBtn.disabled = false;
            
            // 전체 선택 체크박스 상태 갱신
            const query = window.searchQuery || '';
            const filtered = stagedItems.filter(i => {
                if (i.domain !== currentTabFilter) return false;
                if (!query) return true;
                const matchTitle = i.title && i.title.toLowerCase().includes(query);
                const matchContent = i.context && i.context.toLowerCase().includes(query);
                const matchMask = i.masking && i.masking.toLowerCase().includes(query);
                return matchTitle || matchContent || matchMask;
            });
            const isAllChecked = filtered.length > 0 && filtered.every(i => checkedSessionIds.has(i.id));
            selectAllCheck.checked = isAllChecked;

            if (typeof updateSubmitToDrag === 'function') {
                updateSubmitToDrag();
            }
        }

        let spinnerInterval = null;
        
        function startPushSpinner() {
            if (spinnerInterval) return;
            pushBtn.disabled = true;
            deleteBtn.disabled = true;
            draftBtn.disabled = true;
            let spinnerIdx = 0;
            const spinnerFrames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            spinnerInterval = setInterval(() => {
                pushBtn.textContent = `${spinnerFrames[spinnerIdx]} Pushing...`;
                spinnerIdx = (spinnerIdx + 1) % spinnerFrames.length;
            }, 100);
        }

        function stopPushSpinner() {
            if (spinnerInterval) {
                clearInterval(spinnerInterval);
                spinnerInterval = null;
            }
        }

        pushBtn.onclick = async () => {
            // 🚀 이미 진행 중일 때 클릭하면 사용자에게 확인을 받은 후 중단(Cancel) rpc를 호출합니다.
            if (isProcessing) {
                if (confirm('현재 진행 중인 작업을 정말 중단하시겠습니까?')) {
                    if (window.rpc) {
                        window.rpc("cancel_push");
                    }
                }
                return;
            }
            
            const selectedIds = Array.from(checkedSessionIds);
            const draftIds = stagedItems.filter(i => selectedIds.includes(i.id) && i.status !== 'PUSHED').map(i => i.id);
            
            if (draftIds.length === 0) return;
            
            isProcessing = true;
            pushStartTime = Date.now();
            processingIds = draftIds;
            renderStagedList();
            startPushSpinner(); 
            updatePushBtnState(); // Cancel로 버튼 텍스트 변경 유도
            
            if (window.rpc) {
                window.rpc("mask_and_push_batch:" + JSON.stringify({ 
                    ids: draftIds,
                    host: window.location.host 
                }));
            }
        };

        actionContainer.appendChild(deleteBtn);
        actionContainer.appendChild(draftBtn);
        actionContainer.appendChild(pushBtn);
        // 🚀 기능 버튼들을 header 대신 새로 만든 listHeader에 배치합니다.
        listHeader.appendChild(actionContainer);

        const mainLayout = document.createElement('div');
        mainLayout.id = 'main-layout';

        const aside = document.createElement('aside');
        aside.style.display = 'flex';
        aside.style.flexDirection = 'column';
        aside.style.justifyContent = 'space-between';

        const gnbMenuWrapper = document.createElement('div');
        gnbMenuWrapper.style.flex = '1';
        gnbMenuWrapper.style.overflowY = 'auto';

        const gnbMenu = document.createElement('div');
        gnbMenu.className = 'gnb-menu';
        gnbMenuWrapper.appendChild(gnbMenu);

        const asideBottom = document.createElement('div');
        asideBottom.style.padding = '10px';
        asideBottom.style.display = 'flex';
        asideBottom.style.flexDirection = 'column';
        asideBottom.style.gap = '5px';
        asideBottom.style.borderTop = '1px solid #ddd';

        let dynamicTabs = window.custom_tabs || ['COMMERCE', 'LOGISTICS', 'TRADE'];
        const fixedTabs = ['TRASH', 'PROMPT', 'CONFIG']; // 🚀 휴지통 탭 추가
        const defaultTab = (window.default_tab === 'DRAFT' ? dynamicTabs[0] : window.default_tab) || dynamicTabs[0];
        let currentTabFilter = defaultTab;
        
        let isEditMode = false;
        let tempDynamicTabs = [];
        
        let stagedItems = []; 
        let savedPrompts = []; 
        let deletedSessionIds = new Set(); 
        let checkedSessionIds = new Set(); 

        function updateAsideBottomUI() {
            asideBottom.replaceChildren();
            if (isEditMode) {
                const confirmBtn = document.createElement('button');
                confirmBtn.textContent = '확인';
                confirmBtn.style.background = '#28a745';
                confirmBtn.style.color = '#fff';
                confirmBtn.style.width = '100%';
                confirmBtn.style.borderRadius = '4px';
                confirmBtn.onclick = () => {
                    // 🚀 1. 삭제된 탭 검사: 원래 있던 탭인데 임시 배열에 없으면 TRASH로 이동
                    const deletedTabs = dynamicTabs.filter(t => !tempDynamicTabs.some(temp => temp.original === t));
                    deletedTabs.forEach(delTab => {
                        stagedItems.forEach(item => {
                            if (item.domain === delTab) item.domain = 'TRASH';
                        });
                        if (window.rpc) window.rpc("rename_domain:" + JSON.stringify({ old: delTab, new: 'TRASH' }));
                    });

                    // 🚀 2. 이름이 변경된 탭 검사 (TRASH로 보내진 탭 제외)
                    tempDynamicTabs.forEach(temp => {
                        const oldTab = temp.original;
                        const newTab = temp.current ? temp.current.trim().toUpperCase() : '';
                        if (oldTab && newTab && newTab !== oldTab && !deletedTabs.includes(oldTab)) {
                            stagedItems.forEach(item => {
                                if (item.domain === oldTab) item.domain = newTab;
                            });
                            if (window.rpc) window.rpc("rename_domain:" + JSON.stringify({ old: oldTab, new: newTab }));
                        }
                    });

                    // 최종 메뉴 확정
                    dynamicTabs = tempDynamicTabs.map(t => t.current.trim().toUpperCase()).filter(t => t !== '');
                    if (dynamicTabs.length === 0) {
                        // 🚀 하드코딩된 COMMERCE 대신, 기존 설정의 첫 번째 탭을 살리거나 GENERAL을 기본값으로 사용합니다.
                        dynamicTabs = window.custom_tabs && window.custom_tabs.length > 0 ? [window.custom_tabs[0]] : ['GENERAL'];
                    }
                    isEditMode = false;
                    
                    if (!dynamicTabs.includes(currentTabFilter) && !fixedTabs.includes(currentTabFilter)) {
                        currentTabFilter = dynamicTabs[0];
                    }
                    
                    if (window.rpc) {
                        window.rpc("save_config:" + JSON.stringify({ custom_tabs: dynamicTabs }));
                    }
                    updateGnbUI();
                    renderStagedList();
                };

                const cancelBtn = document.createElement('button');
                cancelBtn.textContent = '취소';
                cancelBtn.style.background = '#6c757d';
                cancelBtn.style.color = '#fff';
                cancelBtn.style.width = '100%';
                cancelBtn.style.borderRadius = '4px';
                cancelBtn.onclick = () => {
                    isEditMode = false;
                    updateGnbUI();
                };

                asideBottom.appendChild(confirmBtn);
                asideBottom.appendChild(cancelBtn);
            } else {
                const editBtn = document.createElement('button');
                editBtn.textContent = '메뉴 수정';
                editBtn.style.background = '#007bff';
                editBtn.style.color = '#fff';
                editBtn.style.width = '100%';
                editBtn.style.borderRadius = '4px';
                editBtn.onclick = () => {
                    isEditMode = true;
                    // 🚀 인덱스 꼬임 방지를 위해 원본 이름과 현재 이름을 객체로 묶어 관리합니다.
                    tempDynamicTabs = dynamicTabs.map(t => ({ original: t, current: t }));
                    updateGnbUI();
                };
                asideBottom.appendChild(editBtn);
            }
        }

        function updateGnbUI() {
            gnbMenu.replaceChildren();

            const tabsToRender = isEditMode ? tempDynamicTabs : dynamicTabs.map(t => ({ original: t, current: t }));

            tabsToRender.forEach((tabObj, index) => {
                const t = tabObj.current;
                const domainCount = stagedItems.filter(i => i.domain === t).length;
                const item = document.createElement('div');
                item.className = 'gnb-item' + (!isEditMode && t === currentTabFilter ? ' active' : '');
                
                if (isEditMode) {
                    item.style.display = 'flex';
                    item.style.alignItems = 'center';
                    item.style.gap = '5px';
                    item.style.padding = '5px 10px';
                    
                    // 🚀 드래그 앤 드랍 순서 변경 로직 추가
                    item.draggable = true;
                    item.ondragstart = (e) => {
                        e.dataTransfer.setData('text/plain', index);
                        e.dataTransfer.effectAllowed = 'move';
                        item.style.opacity = '0.5';
                    };
                    item.ondragend = () => {
                        item.style.opacity = '1';
                    };
                    item.ondragover = (e) => {
                        e.preventDefault();
                        e.dataTransfer.dropEffect = 'move';
                    };
                    item.ondrop = (e) => {
                        e.preventDefault();
                        const fromIndex = parseInt(e.dataTransfer.getData('text/plain'), 10);
                        const toIndex = index;
                        if (fromIndex !== toIndex && !isNaN(fromIndex)) {
                            // 배열 요소 순서 변경
                            const movedItem = tempDynamicTabs.splice(fromIndex, 1)[0];
                            tempDynamicTabs.splice(toIndex, 0, movedItem);
                            updateGnbUI();
                        }
                    };

                    const dragHandle = document.createElement('span');
                    dragHandle.textContent = '≡';
                    dragHandle.style.cursor = 'grab';
                    dragHandle.style.color = '#aaa';

                    const cb = document.createElement('input');
                    cb.type = 'checkbox';
                    cb.title = 'Select';

                    const input = document.createElement('input');
                    input.type = 'text';
                    input.value = t;
                    input.style.flex = '1';
                    input.style.width = '60px';
                    input.style.fontSize = '12px';
                    input.style.padding = '2px 4px';
                    input.oninput = (e) => {
                        tempDynamicTabs[index].current = e.target.value.toUpperCase();
                    };

                    const delBtn = document.createElement('span');
                    delBtn.textContent = '🗑️';
                    delBtn.style.cursor = 'pointer';
                    delBtn.style.fontSize = '12px';
                    delBtn.title = '삭제';
                    delBtn.onclick = () => {
                        tempDynamicTabs.splice(index, 1);
                        updateGnbUI();
                    };

                    item.appendChild(dragHandle);
                    item.appendChild(cb);
                    item.appendChild(input);
                    item.appendChild(delBtn);
                } else {
                    item.textContent = `${t} (${domainCount})`;
                    item.onclick = () => {
                        currentTabFilter = t;
                        window.currentPage = 1; // 🚀 탭 전환 시 페이지 초기화
                        updateGnbUI();
                        renderStagedList();
                    };
                    
                    // 🚀 드래그 앤 드랍으로 아이템을 해당 메뉴(도메인)로 이동시키는 이벤트를 추가합니다.
                    item.ondragover = (e) => {
                        if (e.dataTransfer.types.includes('application/x-item-id')) {
                            e.preventDefault();
                            item.style.background = '#d0d0d0'; // 드랍 가능 표시 효과
                        }
                    };
                    item.ondragleave = (e) => {
                        if (e.dataTransfer.types.includes('application/x-item-id')) {
                            e.preventDefault();
                            item.style.background = ''; // 효과 원복
                        }
                    };
                    item.ondrop = (e) => {
                        if (e.dataTransfer.types.includes('application/x-item-id')) {
                            e.preventDefault();
                            item.style.background = '';
                            const droppedId = e.dataTransfer.getData('application/x-item-id');
                            if (droppedId) {
                                const targetItem = stagedItems.find(i => i.id === droppedId);
                                if (targetItem && targetItem.domain !== t) {
                                    targetItem.domain = t; // 🚀 아이템 도메인 변경
                                    // 변경된 데이터를 백엔드로 동기화하여 DB에 반영합니다.
                                    if (window.rpc) {
                                        window.rpc("sync_data:" + JSON.stringify(targetItem));
                                    }
                                    updateGnbUI();
                                    renderStagedList();
                                }
                            }
                        }
                    };
                }
                gnbMenu.appendChild(item);
            });

            if (isEditMode) {
                const addBtn = document.createElement('div');
                addBtn.className = 'gnb-item';
                addBtn.textContent = '+ 메뉴 추가';
                addBtn.style.color = '#007bff';
                addBtn.style.fontWeight = 'bold';
                addBtn.style.textAlign = 'center';
                addBtn.style.padding = '10px';
                addBtn.onclick = () => {
                    tempDynamicTabs.push({ original: null, current: 'NEW' });
                    updateGnbUI();
                };
                gnbMenu.appendChild(addBtn);
            }

            const spacer = document.createElement('div');
            spacer.style.marginTop = '20px';
            spacer.style.borderTop = '1px solid #ddd';
            spacer.style.paddingTop = '10px';
            gnbMenu.appendChild(spacer);

            fixedTabs.forEach(t => {
                // 🚀 TRASH 탭일 경우 아이템 카운트를 함께 표시합니다.
                const domainCount = t === 'TRASH' ? stagedItems.filter(i => i.domain === t).length : 0;
                const item = document.createElement('div');
                item.className = 'gnb-item' + (t === currentTabFilter && !isEditMode ? ' active' : '');
                item.textContent = t === 'TRASH' ? `${t} (${domainCount})` : t;
                item.onclick = () => {
                    if (isEditMode) return;
                    currentTabFilter = t;
                    window.currentPage = 1; // 🚀 탭 전환 시 페이지 초기화
                    updateGnbUI();
                    renderStagedList();
                };
                if (isEditMode) {
                    item.style.opacity = '0.5';
                    item.style.cursor = 'not-allowed';
                }
                gnbMenu.appendChild(item);
            });

            updateAsideBottomUI();
        }
        updateGnbUI();
        aside.appendChild(gnbMenuWrapper);
        aside.appendChild(asideBottom);

        const stagedList = document.createElement('div');
        stagedList.className = 'content';
        
        // 🚀 인피니티 스크롤 이벤트 부착
        stagedList.onscroll = () => {
            if (currentTabFilter === 'CONFIG' || currentTabFilter === 'PROMPT') return;
            // 끝에 도달했는지 확인 (여유 공간 20px 허용)
            if (stagedList.scrollTop + stagedList.clientHeight >= stagedList.scrollHeight - 20) {
                const query = window.searchQuery || '';
                const filtered = stagedItems.filter(i => {
                    if (i.domain !== currentTabFilter) return false;
                    if (!query) return true;
                    const matchTitle = i.title && i.title.toLowerCase().includes(query);
                    const matchContent = i.context && i.context.toLowerCase().includes(query);
                    const matchMask = i.masking && i.masking.toLowerCase().includes(query);
                    return matchTitle || matchContent || matchMask;
                });
                if (window.currentPage * 20 < filtered.length) {
                    window.currentPage++;
                    renderStagedList(true); // append 모드로 추가 렌더링
                }
            }
        };

        const log = document.createElement('div');
        log.id = 'log';
        // 🚀 생성해둔 listHeader를 리스트 컨테이너 최상단에 붙입니다.
        stagedList.appendChild(listHeader);
        stagedList.appendChild(log);

        mainLayout.appendChild(aside);
        mainLayout.appendChild(stagedList);

        const footer = document.createElement('footer');
        
        const fileInputWrapper = document.createElement('div');
        fileInputWrapper.style.position = 'relative';
        fileInputWrapper.style.width = '140px';
        fileInputWrapper.style.height = '30px';
        fileInputWrapper.style.display = 'none';
        fileInputWrapper.style.alignItems = 'center';
        fileInputWrapper.style.overflow = 'hidden';

        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = 'image/*, application/pdf, text/csv';
        fileInput.style.width = '100%';
        fileInput.style.fontSize = '12px';
        fileInput.style.cursor = 'pointer';

        const fileSpinner = document.createElement('div');
        fileSpinner.style.display = 'none';
        fileSpinner.style.position = 'absolute';
        fileSpinner.style.top = '0';
        fileSpinner.style.left = '0';
        fileSpinner.style.width = '100%';
        fileSpinner.style.height = '100%';
        fileSpinner.style.background = '#f8f9fa';
        fileSpinner.style.alignItems = 'center';
        fileSpinner.style.justifyContent = 'center';
        fileSpinner.style.fontSize = '12px';
        fileSpinner.style.color = '#333';
        fileSpinner.style.fontWeight = 'bold';

        fileInputWrapper.appendChild(fileInput);
        fileInputWrapper.appendChild(fileSpinner);
        
        const textInput = document.createElement('input');
        textInput.type = 'text';
        textInput.placeholder = '프롬프트 텍스트를 입력하세요...';
        textInput.setAttribute('list', 'prompt-list'); // 🚀 datalist 연결

        const promptDatalist = document.createElement('datalist');
        promptDatalist.id = 'prompt-list';
        
        const submitBtn = document.createElement('button');
        submitBtn.textContent = 'Submit';
        
        let processedFileContent = '';
        let fileSpinnerInterval = null;

        function updateSubmitToDrag() {
            const selectedIds = Array.from(checkedSessionIds);
            const pushedCount = stagedItems.filter(i => selectedIds.includes(i.id) && i.status === 'PUSHED').length;
            
            if (pushedCount > 0) {
                submitBtn.textContent = `Drag Content (${pushedCount})`;
                submitBtn.disabled = false;
                submitBtn.draggable = true;
            } else if (processedFileContent !== '' || textInput.value.trim() !== '') {
                submitBtn.textContent = 'Drag Content';
                submitBtn.disabled = false;
                submitBtn.draggable = true;
            } else {
                submitBtn.textContent = 'Submit';
                submitBtn.disabled = true;
                submitBtn.draggable = false;
            }
        }

        textInput.oninput = updateSubmitToDrag;

        fileInput.onchange = (e) => {
            const file = e.target.files[0];
            if (!file) return;

            // 🚀 이미지 파일인데 GLM-OCR이 없다면 차단 및 CONFIG 이동
            if (file.type.startsWith('image/') && !(window.model_status && window.model_status['GLM-OCR'])) {
                alert("GLM-OCR 모델이 설치되지 않았습니다. CONFIG 메뉴에서 모델을 다운로드해 주세요.");
                fileInput.value = '';
                currentTabFilter = 'CONFIG';
                updateGnbUI();
                renderStagedList();
                return;
            }

            if (file.name.toLowerCase().endsWith('.csv') || file.type === 'text/csv') {
                const reader = new FileReader();
                reader.onload = async (ev) => {
                    const textData = ev.target.result;
                    const pageId = await generatePageId(file.name + Date.now());
                    const item = { 
                        id: pageId, 
                        host: window.location.host,
                        url: "file://" + file.name,
                        title: `[File] ${file.name}`, 
                        domain: currentTabFilter,
                        context: textData, 
                        status: 'DRAFT',
                        track: '',
                        version: 1,
                        created_at: Date.now(),
                        updated_at: Date.now()
                    };
                    if (window.rpc) {
                        window.rpc("sync_data:" + JSON.stringify(item));
                    }
                };
                reader.readAsText(file);
                return;
            }

            fileSpinner.style.display = 'flex';
            fileInput.style.display = 'none';
            let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            let idx = 0;
            fileSpinnerInterval = setInterval(() => {
                fileSpinner.textContent = `${frames[idx]} Processing...`;
                idx = (idx + 1) % frames.length;
            }, 100);

            const reader = new FileReader();
            reader.onload = (ev) => {
                if (window.rpc) {
                    window.rpc("process_file:" + ev.target.result);
                }
            };
            reader.readAsDataURL(file);
        };

        // 🚀 버튼을 드래그할 때 웹 브라우저 네이티브 파일 드래그앤드랍(바탕화면 드랍)을 지원함과 동시에
        // Rust 백엔드에도 동일한 내용의 물리 파일을 백업 저장하도록 요청합니다.
        submitBtn.ondragstart = (e) => {
            if (!submitBtn.textContent.startsWith('Drag') || submitBtn.disabled) {
                e.preventDefault();
                return;
            }
            
            const selectedIds = Array.from(checkedSessionIds);
            
            const promptValue = textInput.value || 'N/A';
            const payload = {
                prompt: promptValue,
                processed_file: processedFileContent || '',
                ids: selectedIds
            };
            
            if (window.rpc) {
                // 🚀 드래그 시작 시 프롬프트를 DB(JSON)에 저장하도록 RPC 호출
                if (textInput.value.trim() !== '') {
                    window.rpc("save_prompt:" + textInput.value.trim());
                }
                window.rpc("export_to_file:" + JSON.stringify(payload));
            }

            const selectedItems = stagedItems.filter(i => selectedIds.includes(i.id) && i.status === 'PUSHED');
            let exportContent = `[Prompt]\n${textInput.value || 'N/A'}\n\n`;
            if (processedFileContent) {
                exportContent += `[File Masked Text]\n${processedFileContent}\n\n`;
            }
            exportContent += `[Selected Items (${selectedItems.length})]\n`;
            
            selectedItems.forEach(item => {
                exportContent += `\n--- ID: ${item.id} ---\n[Domain]: ${item.domain}\n[Title]: ${item.title}\n[Content]:\n${item.masking || item.context}\n`;
            });

            const utf8Bytes = new TextEncoder().encode(exportContent);
            let binary = '';
            for (let i = 0; i < utf8Bytes.length; i++) {
                binary += String.fromCharCode(utf8Bytes[i]);
            }
            const base64Str = btoa(binary);

            // 🚀 텍스트가 아닌 파일 다운로드로 인식시키기 위해 application/octet-stream 사용 및 파일명 drag.context 고정
            const fileName = "drag.context";
            const mimeType = "application/octet-stream";
            
            const file = new File([utf8Bytes], fileName, { type: mimeType });
            e.dataTransfer.items.add(file);
            // 🚀 데이터 전송 시 DownloadURL 포맷을 맞춰 브라우저 외부로 드랍 시 파일이 생성되도록 유도
            e.dataTransfer.setData('DownloadURL', `${mimeType}:${fileName}:data:${mimeType};base64,${base64Str}`);
        };

        footer.appendChild(fileInputWrapper);
        footer.appendChild(textInput);
        footer.appendChild(promptDatalist); // 🚀 datalist 추가
        footer.appendChild(submitBtn);

        agentContainer.appendChild(header);
        agentContainer.appendChild(mainLayout);
        agentContainer.appendChild(footer);
        shadow.appendChild(style);
        shadow.appendChild(agentContainer);

        agentContainer.addEventListener('dragover', (e) => {
            if (!e.dataTransfer.types.includes('Files')) return; // 🚀 파일 드래그가 아닌 메뉴 이동 등의 텍스트 드래그면 무시합니다.
            e.preventDefault();
            agentContainer.style.border = '2px dashed #007bff';
        });
        agentContainer.addEventListener('dragleave', (e) => {
            if (!e.dataTransfer.types.includes('Files')) return;
            e.preventDefault();
            agentContainer.style.border = 'none';
        });
        agentContainer.addEventListener('drop', (e) => {
            if (!e.dataTransfer.types.includes('Files')) return;
            e.preventDefault();
            agentContainer.style.border = 'none';
            
            // 🚀 추가적인 제한 없이 파일 드랍을 수용합니다.
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];

                // 🚀 이미지 드랍 시 GLM-OCR 확인 및 차단
                if (file.type.startsWith('image/') && !(window.model_status && window.model_status['GLM-OCR'])) {
                    alert("GLM-OCR 모델이 설치되지 않았습니다. CONFIG 메뉴에서 모델을 다운로드해 주세요.");
                    currentTabFilter = 'CONFIG';
                    updateGnbUI();
                    renderStagedList();
                    return;
                }

                const reader = new FileReader();
                reader.onload = async (ev) => {
                    const content = ev.target.result;
                    const pageId = await generatePageId(content + Date.now());
                    const item = { 
                        id: pageId, 
                        host: window.location.host,
                        url: file.name,
                        title: `[File] ${file.name}`, 
                        domain: currentTabFilter,
                        context: content, // 이미지면 dataURL, 텍스트면 문자열이 담깁니다.
                        status: 'DRAFT',
                        track: '',
                        version: 1,
                        created_at: Date.now(),
                        updated_at: Date.now()
                    };
                    if (window.rpc) {
                        window.rpc("sync_data:" + JSON.stringify(item));
                    }
                };
                // 🚀 파일 타입에 따라 읽기 방식을 결정합니다. 이미지면 Base64로, 그 외(CSV 등)는 텍스트로 읽습니다.
                if (file.type.startsWith('image/')) {
                    reader.readAsDataURL(file);
                } else {
                    reader.readAsText(file);
                }
            }
        });

        window.currentPage = 1;
        
        function renderStagedList(append = false) {
            if (!append) {
                log.replaceChildren(); 
            }

            // PROMPT 탭인 경우 UI 렌더링
            if (currentTabFilter === 'PROMPT') {
                if (append) return; // 🚀 PROMPT 탭은 페이징 안함
                listHeader.style.display = 'none'; // 🚀 Draft, Push 버튼이 포함된 헤더를 숨깁니다.
                footer.style.display = 'none'; // 하단 입력 영역도 숨깁니다.

                const promptWrapper = document.createElement('div');
                promptWrapper.style.padding = '20px';

                // 글로벌 딕셔너리가 없으면 빈 객체로 폴백
                const dict = window.lang_dict || {};
                const currentLang = window.default_language || 'English';
                const getText = (key) => dict[currentLang] ? dict[currentLang][key] : (dict['English'] ? dict['English'][key] : key);

                const promptTitle = document.createElement('h3');
                promptTitle.textContent = getText('prompt_title');
                promptTitle.style.marginTop = '0';
                promptTitle.style.fontSize = '14px';
                promptWrapper.appendChild(promptTitle);

                if (savedPrompts.length === 0) {
                    const emptyMsg = document.createElement('div');
                    emptyMsg.style.color = '#999';
                    emptyMsg.style.fontSize = '12px';
                    emptyMsg.style.marginTop = '20px';
                    emptyMsg.textContent = getText('prompt_empty');
                    promptWrapper.appendChild(emptyMsg);
                } else {
                    savedPrompts.forEach(p => {
                        const row = document.createElement('div');
                        row.style.display = 'flex';
                        row.style.justifyContent = 'space-between';
                        row.style.alignItems = 'center';
                        row.style.padding = '10px 0';
                        row.style.borderBottom = '1px solid #eee';
                        
                        const textSpan = document.createElement('span');
                        textSpan.textContent = p;
                        textSpan.style.fontSize = '13px';
                        textSpan.style.color = '#333';
                        textSpan.style.flex = '1';

                        const delBtn = document.createElement('button');
                        delBtn.textContent = getText('prompt_delete');
                        delBtn.style.background = '#dc3545';
                        delBtn.style.color = 'white';
                        delBtn.style.border = 'none';
                        delBtn.style.borderRadius = '3px';
                        delBtn.style.padding = '5px 10px';
                        delBtn.style.fontSize = '12px';
                        delBtn.style.cursor = 'pointer';
                        
                        delBtn.onmouseover = () => delBtn.style.background = '#c82333';
                        delBtn.onmouseout = () => delBtn.style.background = '#dc3545';
                        
                        delBtn.onclick = () => {
                            if (confirm(getText('prompt_delete_confirm'))) {
                                if (window.rpc) window.rpc("delete_prompt:" + p);
                            }
                        };

                        row.appendChild(textSpan);
                        row.appendChild(delBtn);
                        promptWrapper.appendChild(row);
                    });
                }

                log.appendChild(promptWrapper);
                return;
            }

            // CONFIG 탭인 경우 UI를 다르게 렌더링
            if (currentTabFilter === 'CONFIG') {
                if (append) return; // 🚀 CONFIG 탭은 페이징 안함
                listHeader.style.display = 'none'; // 🚀 기능 버튼과 체크박스가 묶인 헤더 전체 숨김
                footer.style.display = 'none';

                const configWrapper = document.createElement('div');
                configWrapper.style.padding = '20px';

                // 글로벌 딕셔너리가 없으면 빈 객체로 폴백
                const dict = window.lang_dict || {};
                const currentLang = window.default_language || 'English';
                
                // 🚀 선택된 언어에 해당하는 텍스트들을 가져옵니다 (데이터가 누락된 경우 기본 영어로 폴백)
                const getText = (key) => dict[currentLang] ? dict[currentLang][key] : (dict['English'] ? dict['English'][key] : key);

                const langTitle = document.createElement('h3');
                langTitle.textContent = getText('lang_title');
                langTitle.style.marginTop = '0';
                langTitle.style.fontSize = '14px';

                const langSelect = document.createElement('select');
                langSelect.style.padding = '8px';
                langSelect.style.width = '100%';
                langSelect.style.marginBottom = '10px';
                langSelect.style.fontSize = '13px';
                
                const langDesc = document.createElement('div');
                langDesc.style.fontSize = '12px';
                langDesc.style.color = '#555';
                langDesc.style.marginBottom = '30px';
                langDesc.style.padding = '10px';
                langDesc.style.background = '#f9f9f9';
                langDesc.style.borderLeft = '3px solid #007bff';
                
                const autoExtractTitle = document.createElement('h3');
                autoExtractTitle.textContent = getText('auto_extract_title') || '자동 수집 (Auto Collect)';
                autoExtractTitle.style.fontSize = '14px';

                const autoExtractDesc = document.createElement('p');
                autoExtractDesc.textContent = getText('auto_extract_desc') || '방문하는 페이지의 콘텐츠를 자동으로 수집합니다.';
                autoExtractDesc.style.fontSize = '12px';
                autoExtractDesc.style.color = '#666';
                autoExtractDesc.style.marginBottom = '15px';

                const autoExtractSelect = document.createElement('select');
                autoExtractSelect.style.padding = '8px';
                autoExtractSelect.style.width = '100%';
                autoExtractSelect.style.marginBottom = '30px';
                autoExtractSelect.style.fontSize = '13px';
                
                const optOn = document.createElement('option');
                optOn.value = 'true';
                optOn.textContent = 'ON';
                const optOff = document.createElement('option');
                optOff.value = 'false';
                optOff.textContent = 'OFF';
                
                autoExtractSelect.appendChild(optOn);
                autoExtractSelect.appendChild(optOff);
                
                // 글로벌 설정값 연동
                autoExtractSelect.value = window.auto_extract !== false ? 'true' : 'false';
                
                autoExtractSelect.onchange = (e) => {
                    const isAuto = e.target.value === 'true';
                    window.auto_extract = isAuto;
                    if (window.rpc) window.rpc("save_config:" + JSON.stringify({ auto_extract: isAuto }));
                };
                
                // 🚀 마스킹 기능 On/Off UI 추가
                const maskingTitle = document.createElement('h3');
                maskingTitle.textContent = getText('masking_title') || 'Privacy Masking';
                maskingTitle.style.fontSize = '14px';

                const maskingDesc = document.createElement('p');
                maskingDesc.textContent = getText('masking_desc') || 'Automatically masks sensitive data.';
                maskingDesc.style.fontSize = '12px';
                maskingDesc.style.color = '#666';
                maskingDesc.style.marginBottom = '15px';

                const maskingSelect = document.createElement('select');
                maskingSelect.style.padding = '8px';
                maskingSelect.style.width = '100%';
                maskingSelect.style.marginBottom = '30px';
                maskingSelect.style.fontSize = '13px';
                
                const maskOptOn = document.createElement('option');
                maskOptOn.value = 'true';
                maskOptOn.textContent = 'ON';
                const maskOptOff = document.createElement('option');
                maskOptOff.value = 'false';
                maskOptOff.textContent = 'OFF';
                
                maskingSelect.appendChild(maskOptOn);
                maskingSelect.appendChild(maskOptOff);
                
                maskingSelect.value = window.enable_masking !== false ? 'true' : 'false';
                
                maskingSelect.onchange = (e) => {
                    const isMasking = e.target.value === 'true';
                    window.enable_masking = isMasking;
                    if (window.rpc) window.rpc("save_config:" + JSON.stringify({ enable_masking: isMasking }));
                    updatePushBtnState(); // 상태 변경 시 Push 버튼 검증 재실행
                };
                
                // 🚀 모델 다운로드 UI 추가
                const modelTitle = document.createElement('h3');
                modelTitle.textContent = getText('model_title');
                modelTitle.style.fontSize = '14px';

                const modelDesc = document.createElement('p');
                modelDesc.textContent = getText('model_desc');
                modelDesc.style.fontSize = '12px';
                modelDesc.style.color = '#666';
                modelDesc.style.marginBottom = '15px';

                // 🚀 전체 다운로드 버튼 추가
                const downloadAllBtn = document.createElement('button');
                downloadAllBtn.textContent = getText('model_download_all');
                downloadAllBtn.style.background = '#17a2b8';
                downloadAllBtn.style.color = 'white';
                downloadAllBtn.style.padding = '8px 15px';
                downloadAllBtn.style.border = 'none';
                downloadAllBtn.style.borderRadius = '4px';
                downloadAllBtn.style.cursor = 'pointer';
                downloadAllBtn.style.fontWeight = 'bold';
                downloadAllBtn.style.marginBottom = '10px'; // 🚀 여백 조정
                downloadAllBtn.style.width = '100%';

                // 🚀 모델 전체 삭제 버튼 추가
                const deleteAllModelsBtn = document.createElement('button');
                deleteAllModelsBtn.textContent = getText('model_delete_all') || 'Delete All Models';
                deleteAllModelsBtn.style.background = '#dc3545';
                deleteAllModelsBtn.style.color = 'white';
                deleteAllModelsBtn.style.padding = '8px 15px';
                deleteAllModelsBtn.style.border = 'none';
                deleteAllModelsBtn.style.borderRadius = '4px';
                deleteAllModelsBtn.style.cursor = 'pointer';
                deleteAllModelsBtn.style.fontWeight = 'bold';
                deleteAllModelsBtn.style.marginBottom = '30px';
                deleteAllModelsBtn.style.width = '100%';
                
                deleteAllModelsBtn.onclick = () => {
                    if (confirm(getText('model_delete_all_confirm'))) {
                        deleteAllModelsBtn.disabled = true;
                        deleteAllModelsBtn.style.background = '#6c757d';
                        deleteAllModelsBtn.textContent = 'Deleting...';
                        if (window.rpc) {
                            window.rpc("delete_all_models");
                        }
                    }
                };

                const modelListContainer = document.createElement('div');
                modelListContainer.style.display = 'flex';
                modelListContainer.style.flexDirection = 'column';
                modelListContainer.style.gap = '10px';
                modelListContainer.style.marginBottom = '30px';

                const models = ['GLM-OCR', 'Privacy-Filter', 'Embedding'];
                
                // 다운로드 공통 로직 분리
                const triggerDownload = (m, btn, progressContainer, progressBar) => {
                    window.download_progress = window.download_progress || {};
                    window.download_progress[m] = 0; // 🚀 다운로드 시작 시 전역 기록
                    
                    btn.disabled = true;
                    btn.style.background = '#6c757d';
                    btn.style.cursor = 'not-allowed';
                    btn.textContent = getText('model_downloading');
                    progressContainer.style.display = 'block';
                    progressBar.style.width = '0%';
                    progressBar.textContent = '0%';
                    if (window.rpc) {
                        window.rpc("download_model:" + m);
                    }
                };

                downloadAllBtn.onclick = () => {
                    const toDownload = models.filter(m => !(window.model_status && window.model_status[m]));
                    if (toDownload.length === 0) {
                        alert(getText('model_all_downloaded'));
                        return;
                    }
                    if (confirm(getText('model_download_all_confirm'))) {
                        downloadAllBtn.disabled = true;
                        downloadAllBtn.style.background = '#6c757d';
                        toDownload.forEach(m => {
                            const safeId = m.replace(/[\s\(\)]+/g, '-');
                            // 🚀 Shadow DOM 내부의 요소를 올바르게 찾도록 document 대신 shadow 객체를 사용합니다.
                            const btn = shadow.getElementById('btn-download-' + safeId);
                            const pc = shadow.getElementById('progress-container-' + safeId);
                            const pb = shadow.getElementById('progress-bar-' + safeId);
                            if (btn && pc && pb && !btn.disabled) {
                                triggerDownload(m, btn, pc, pb);
                            }
                        });
                    }
                };
                
                models.forEach(m => {
                    // 서버에서 주입된 설치 상태를 기반으로 완료 여부 체크
                    const isDownloaded = window.model_status && window.model_status[m];
                    const safeId = m.replace(/[\s\(\)]+/g, '-'); // ID 규격에 맞게 변환

                    const row = document.createElement('div');
                    row.style.border = '1px solid #ddd';
                    row.style.borderRadius = '4px';
                    row.style.padding = '10px';
                    row.style.background = '#fff';

                    const topRow = document.createElement('div');
                    topRow.style.display = 'flex';
                    topRow.style.justifyContent = 'space-between';
                    topRow.style.alignItems = 'center';

                    const nameSpan = document.createElement('span');
                    nameSpan.textContent = m;
                    nameSpan.style.fontSize = '13px';
                    nameSpan.style.fontWeight = 'bold';

                    const currentProgress = (window.download_progress && window.download_progress[m] !== undefined) ? window.download_progress[m] : -1;

                    const btn = document.createElement('button');
                    btn.id = 'btn-download-' + safeId;
                    btn.style.padding = '5px 10px';
                    btn.style.fontSize = '12px';
                    btn.style.borderRadius = '3px';
                    btn.style.border = 'none';
                    btn.style.fontWeight = 'bold';

                    const progressContainer = document.createElement('div');
                    progressContainer.id = 'progress-container-' + safeId;
                    progressContainer.style.width = '100%';
                    progressContainer.style.background = '#e9ecef';
                    progressContainer.style.borderRadius = '4px';
                    progressContainer.style.overflow = 'hidden';
                    progressContainer.style.display = 'none';
                    progressContainer.style.marginTop = '10px';

                    const progressBar = document.createElement('div');
                    progressBar.id = 'progress-bar-' + safeId;
                    progressBar.style.height = '15px';
                    progressBar.style.background = '#007bff';
                    progressBar.style.transition = 'width 0.2s';
                    progressBar.style.textAlign = 'center';
                    progressBar.style.color = 'white';
                    progressBar.style.fontSize = '10px';
                    progressBar.style.lineHeight = '15px';
                    
                    // 🚀 화면 갱신(render) 시 다운로드 진행 상태를 전역 변수에서 확인하여, 버튼과 바를 완벽하게 복구합니다.
                    if (isDownloaded || currentProgress === 100) {
                        btn.textContent = getText('model_downloaded');
                        btn.style.background = '#6c757d';
                        btn.style.color = '#fff';
                        btn.disabled = true;
                        btn.style.cursor = 'not-allowed';
                    } else if (currentProgress >= 0) {
                        btn.textContent = `${getText('model_downloading')} (${currentProgress}%)`;
                        btn.style.background = '#6c757d';
                        btn.style.color = '#fff';
                        btn.disabled = true;
                        btn.style.cursor = 'not-allowed';
                        progressContainer.style.display = 'block';
                        progressBar.style.width = `${currentProgress}%`;
                        progressBar.textContent = `${currentProgress}%`;
                    } else {
                        btn.textContent = getText('model_download');
                        btn.style.background = '#28a745';
                        btn.style.color = '#fff';
                        btn.disabled = false;
                        btn.style.cursor = 'pointer';
                    }

                    topRow.appendChild(nameSpan);
                    topRow.appendChild(btn);
                    
                    progressContainer.appendChild(progressBar);

                    btn.onclick = () => {
                        const dict = window.lang_dict || {};
                        const currentLang = window.default_language || 'English';
                        const getText = (key) => dict[currentLang] ? dict[currentLang][key] : (dict['English'] ? dict['English'][key] : key);

                        if (confirm(getText('model_download_confirm'))) {
                            // 🚀 버튼 텍스트를 즉시 '다운로드 중...' 상태로 변경하여 사용자에게 알림
                            btn.textContent = getText('model_downloading');
                            triggerDownload(m, btn, progressContainer, progressBar);
                        }
                    };

                    row.appendChild(topRow);
                    row.appendChild(progressContainer);
                    modelListContainer.appendChild(row);
                });

                const resetTitle = document.createElement('h3');
                resetTitle.textContent = getText('reset_title');
                resetTitle.style.fontSize = '14px';

                const resetDesc = document.createElement('p');
                resetDesc.textContent = getText('reset_desc');
                resetDesc.style.fontSize = '12px';
                resetDesc.style.color = '#666';
                resetDesc.style.marginBottom = '15px';

                const resetBtn = document.createElement('button');
                resetBtn.textContent = getText('reset_btn');
                resetBtn.style.background = '#dc3545';
                resetBtn.style.color = 'white';
                resetBtn.style.padding = '10px 15px';
                resetBtn.style.border = 'none';
                resetBtn.style.borderRadius = '4px';
                resetBtn.style.cursor = 'pointer';
                resetBtn.style.fontWeight = 'bold';
                
                // 영어(English)가 최상단에 오도록 배열 순서 유지
                const langs = ['English', 'Chinese', 'French', 'Spanish', 'Russian', 'German', 'Japanese', 'Korean'];
                langs.forEach(l => {
                    const opt = document.createElement('option');
                    opt.value = l;
                    opt.textContent = l;
                    langSelect.appendChild(opt);
                });

                langSelect.value = currentLang;
                langDesc.textContent = getText('lang_desc');

                langSelect.onchange = (e) => {
                    const selectedLang = e.target.value;
                    window.default_language = selectedLang;
                    
                    // 🚀 언어 셀렉트박스를 바꾸는 순간 화면의 텍스트가 해당 언어로 즉시 전환됩니다.
                    const updateText = (key) => dict[selectedLang] ? dict[selectedLang][key] : (dict['English'] ? dict['English'][key] : key);
                    langTitle.textContent = updateText('lang_title');
                    langDesc.textContent = updateText('lang_desc');
                    autoExtractTitle.textContent = updateText('auto_extract_title');
                    autoExtractDesc.textContent = updateText('auto_extract_desc');
                    maskingTitle.textContent = updateText('masking_title') || 'Privacy Masking';
                    maskingDesc.textContent = updateText('masking_desc') || 'Automatically masks sensitive data.';
                    
                    modelTitle.textContent = updateText('model_title');
                    modelDesc.textContent = updateText('model_desc');
                    downloadAllBtn.textContent = updateText('model_download_all');
                    deleteAllModelsBtn.textContent = updateText('model_delete_all'); // 🚀 삭제 버튼 텍스트도 실시간 갱신
                    
                    models.forEach(m => {
                        const safeId = m.replace(/[\s\(\)]+/g, '-');
                        // 🚀 Shadow DOM 내부 요소를 올바르게 참조
                        const btn = shadow.getElementById('btn-download-' + safeId);
                        const isDownloaded = window.model_status && window.model_status[m];
                        // 현재 다운로드 중(progress/퍼센트가 표시 중)이 아니라면 텍스트를 즉시 전환합니다.
                        if (btn && !btn.disabled && !btn.textContent.includes('%') && !btn.textContent.includes('...')) {
                            btn.textContent = isDownloaded ? updateText('model_downloaded') : updateText('model_download');
                        }
                    });

                    resetTitle.textContent = updateText('reset_title');
                    resetDesc.textContent = updateText('reset_desc');
                    resetBtn.textContent = updateText('reset_btn');

                    if (window.rpc) window.rpc("save_config:" + JSON.stringify({ language: selectedLang }));
                };

                resetBtn.onmouseover = () => resetBtn.style.background = '#c82333';
                resetBtn.onmouseout = () => resetBtn.style.background = '#dc3545';
                
                resetBtn.onclick = () => {
                    // 🚀 확인 알림창 역시 설정된 언어에 따라 표시합니다.
                    const confirmMsg = dict[window.default_language] ? dict[window.default_language]['reset_confirm'] : (dict['English'] ? dict['English']['reset_confirm'] : 'Are you sure you want to delete all data?');
                    if (confirm(confirmMsg)) {
                        if (window.rpc) window.rpc("reset_all_data");
                    }
                };

                configWrapper.appendChild(langTitle);
                configWrapper.appendChild(langSelect);
                configWrapper.appendChild(langDesc);
                configWrapper.appendChild(autoExtractTitle);
                configWrapper.appendChild(autoExtractDesc);
                configWrapper.appendChild(autoExtractSelect);
                configWrapper.appendChild(maskingTitle);
                configWrapper.appendChild(maskingDesc);
                configWrapper.appendChild(maskingSelect);
                configWrapper.appendChild(modelTitle);
                configWrapper.appendChild(modelDesc);
                configWrapper.appendChild(downloadAllBtn); // 🚀 덧붙이기
                configWrapper.appendChild(deleteAllModelsBtn); // 🚀 삭제 버튼 덧붙이기
                configWrapper.appendChild(modelListContainer);
                configWrapper.appendChild(resetTitle);
                configWrapper.appendChild(resetDesc);
                configWrapper.appendChild(resetBtn);

                log.appendChild(configWrapper);
                return;
            }

            // 일반 탭 복구
            listHeader.style.display = 'flex'; // 🚀 기능 버튼과 체크박스가 묶인 헤더 전체 노출
            // 🚀 footer 노출 여부는 updatePushBtnState() 에서 동적으로 제어되므로 여기서 고정하지 않습니다.

            // 🚀 선택된 도메인 탭 검사 및 검색어 필터링 동시 적용
            const query = window.searchQuery || '';
            let filtered = stagedItems.filter(i => {
                if (i.domain !== currentTabFilter) return false;
                if (!query) return true;
                const matchTitle = i.title && i.title.toLowerCase().includes(query);
                const matchContent = i.context && i.context.toLowerCase().includes(query);
                const matchMask = i.masking && i.masking.toLowerCase().includes(query);
                return matchTitle || matchContent || matchMask;
            });
            
            // 🚀 현재 접속한 브라우저의 URL 판별
            const currentUrl = window.location.href;
            
            // 🚀 현재 접속 중인 페이지의 아이템이 항상 최상단에 오도록 정렬 (나머지는 최신순)
            filtered.sort((a, b) => {
                const aIsCurrent = (a.url === currentUrl);
                const bIsCurrent = (b.url === currentUrl);
                if (aIsCurrent && !bIsCurrent) return -1;
                if (!aIsCurrent && bIsCurrent) return 1;
                return b.updated_at - a.updated_at; // 🚀 updated_at 기준으로 변경
            });

            const draftOnlyCount = filtered.filter(i => i.status !== 'PUSHED').length;
            
            // 🚀 작업 중일 때는 Push 상태 메시지를 유지하고, 아닐 때만 카운트를 갱신합니다.
            if (!isProcessing) {
                draftBtn.textContent = `Draft (${draftOnlyCount})`;
                draftBtn.disabled = false;
            } else {
                // 🚀 작업 중이라도 Draft 버튼 자체는 활성화 상태를 유지하여 추가가 가능하게 합니다.
                draftBtn.disabled = false;
            }
            
            if (filtered.length === 0) {
                if (!append) {
                    const empty = document.createElement('div');
                    empty.style.color = '#999';
                    empty.style.fontSize = '12px';
                    empty.style.padding = '20px';
                    empty.textContent = 'No records found for ' + currentTabFilter;
                    log.appendChild(empty);
                }
                selectAllCheck.checked = false;
                updatePushBtnState();
            } else {
                // 🚀 인피니티 스크롤을 위한 Pagination 처리 (최대 20개 노출)
                const startIndex = append ? (window.currentPage - 1) * 20 : 0;
                const endIndex = window.currentPage * 20;
                const itemsToRender = filtered.slice(startIndex, endIndex);

                itemsToRender.forEach(item => {
                    const wrapper = document.createElement('div');
                    wrapper.className = 'item-row-wrapper';
                    
                    // 🚀 아이템을 드래그 앤 드랍으로 이동할 수 있도록 속성 및 이벤트를 추가합니다.
                    wrapper.draggable = true;
                    wrapper.ondragstart = (e) => {
                        e.dataTransfer.setData('application/x-item-id', item.id);
                        e.dataTransfer.effectAllowed = 'move';
                    };
                    
                    // 🚀 현재 진행 중인 아이템인 경우 시각적으로 흐리게(Opacity) 처리하고 상호작용을 막습니다.
                    if (processingIds.includes(item.id)) {
                        wrapper.style.opacity = '0.4';
                        wrapper.style.pointerEvents = 'none';
                    }

                    const row = document.createElement('div');
                    row.className = 'item-row';

                    const cb = document.createElement('input');
                    cb.type = 'checkbox';
                    cb.className = 'item-checkbox';
                    cb.dataset.id = item.id;
                    
                    // 🚀 화면 리렌더링 시, 메모리에 기록된 상태를 읽어와서 체크박스 상태를 복구합니다.
                    cb.checked = checkedSessionIds.has(item.id);
                    
                    cb.onclick = (e) => {
                        e.stopPropagation();
                        // 🚀 개별 체크박스 클릭 시 메모리 상태를 즉각 업데이트합니다.
                        if (e.target.checked) {
                            checkedSessionIds.add(item.id);
                        } else {
                            checkedSessionIds.delete(item.id);
                        }
                        updatePushBtnState();
                    };

                    const textContainer = document.createElement('div');
                    textContainer.style.display = 'flex';
                    textContainer.style.flexDirection = 'column';
                    textContainer.style.flex = '1';
                    textContainer.style.overflow = 'hidden';

                    const parts = item.title.split('\n');
                    const mainTitle = parts[0] || '';
                    const descText = parts.slice(1).join('\n') || '';

                    const titleSpan = document.createElement('span');
                    const statusBadge = item.status === 'PUSHED' ? '✅ [완료됨] ' : '';
                    
                    // 🚀 처리 중인 아이템일 경우 제목 앞에 ⏳(모래시계) 이모지를 추가하여 직관성을 극대화합니다.
                    const processingBadge = processingIds.includes(item.id) ? '⏳ [처리중...] ' : '';
                    
                    // 🚀 현재 접속한 페이지 UI 텍스트 꾸미기
                    if (item.url === currentUrl) {
                        titleSpan.textContent = `📌 [현재 페이지] ${processingBadge}${statusBadge}${mainTitle}`;
                        titleSpan.style.fontSize = '14px';
                        titleSpan.style.fontWeight = '900';
                        titleSpan.style.textDecoration = 'underline';
                    } else {
                        titleSpan.textContent = `${processingBadge}${statusBadge}${mainTitle}`;
                        titleSpan.style.fontSize = '13px';
                        titleSpan.style.fontWeight = 'bold';
                        titleSpan.style.textDecoration = 'none';
                    }
                    
                    // 🚀 처리 중인 아이템은 색상을 파란색 계열로 주어 시각적으로 분리합니다.
                    if (processingIds.includes(item.id)) {
                        titleSpan.style.color = '#007bff';
                    } else if (item.status === 'PUSHED') {
                        titleSpan.style.color = '#28a745';
                    } else {
                        titleSpan.style.color = '#000';
                    }
                    
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

                    const detailView = document.createElement('div');
                    detailView.className = 'item-detail';
                    
                    // 🚀 콘텐츠 타입에 따른 시각화 로직
                    const rawContent = item.masking || item.context || '';
                    if (rawContent.startsWith('data:image/')) {
                        // 🚀 이미지인 경우 img 태그로 렌더링
                        detailView.innerHTML = `<img src="${rawContent}" style="max-width:100%; border-radius:4px; border:1px solid #ddd;">`;
                    } else if (item.url.toLowerCase().endsWith('.csv')) {
                        // 🚀 CSV 파일인 경우 테이블로 변환
                        const rows = rawContent.split('\n').filter(r => r.trim());
                        const tableHtml = rows.map((r, i) => {
                            const cells = r.split(',').map(c => i === 0 ? `<th>${c}</th>` : `<td>${c}</td>`).join('');
                            return `<tr>${cells}</tr>`;
                        }).join('');
                        detailView.innerHTML = `<table style="width:100%; border-collapse:collapse; font-size:11px; border:1px solid #ccc;">${tableHtml}</table>`;
                        // 스타일 보정 (Shadow DOM 내부에 직접 스타일 주입)
                        const tableStyle = document.createElement('style');
                        tableStyle.textContent = `table th, table td { border: 1px solid #eee; padding: 4px; text-align: left; } table th { background: #f0f0f0; }`;
                        detailView.appendChild(tableStyle);
                    } else {
                        detailView.textContent = rawContent || '전처리된 텍스트 결과가 없습니다.';
                    }

                    row.onclick = () => detailView.classList.toggle('open');

                    row.appendChild(cb);
                    row.appendChild(textContainer);
                    wrapper.appendChild(row);
                    wrapper.appendChild(detailView);
                    log.appendChild(wrapper);
                });
                updatePushBtnState();
            }
        }

        async function autoExtract() {
            setTimeout(async () => {
                // 🚀 자동 수집 기능이 꺼져있으면 실행을 중단합니다.
                if (window.auto_extract === false) return;

                const pageId = await generatePageId(window.location.href);
                
                // 🚀 이미 삭제했던 페이지거나 현재 대기열에 이미 존재하는 페이지라면 자동 추출을 중단합니다.
                if (deletedSessionIds.has(pageId) || stagedItems.some(i => i.id === pageId)) {
                    return;
                }
                
                const extractedText = extractVisibleText();
                
                // 🚀 현재 활성화된 탭(currentTabFilter)을 기준으로 도메인을 설정하되, 설정(CONFIG)이나 프롬프트(PROMPT) 탭을 보고 있을 때는 첫 번째 기본 도메인 메뉴로 할당합니다.
                const targetDomain = (currentTabFilter === 'CONFIG' || currentTabFilter === 'PROMPT') ? dynamicTabs[0] : currentTabFilter;
                
                const item = { 
                    id: pageId, 
                    host: window.location.host,
                    url: window.location.href,
                    title: getPageMeta(), 
                    domain: targetDomain, 
                    context: extractedText, 
                    status: 'DRAFT',
                    track: '',
                    version: 1,
                    created_at: Date.now(),
                    updated_at: Date.now()
                };
                if (window.rpc) window.rpc("sync_data:" + JSON.stringify(item));
            }, 1500);
        }

        window.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') agentContainer.classList.toggle('open');
        });

        function syncStateOnReturn() {
            if (window.rpc) {
                window.rpc("fetch_drafts");
                window.rpc("check_progress");
            }
        }
        
        window.addEventListener('focus', syncStateOnReturn);
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'visible') syncStateOnReturn();
        });

        window.addEventListener('rpc_response', (e) => {
            try {
                const data = typeof e.detail === 'string' ? JSON.parse(e.detail) : e.detail;
                
                if (data.type === 'drafts_loaded') {
                    // 🚀 삭제 처리가 완료되지 않은 상태에서 서버 데이터를 가져오더라도, 로컬에서 삭제한 ID는 화면에 렌더링하지 않습니다.
                    stagedItems = data.payload.filter(i => !deletedSessionIds.has(i.id));
                    updateGnbUI(); 
                    renderStagedList();
                    return;
                } 
                else if (data.type === 'sync_success') {
                    deletedSessionIds.delete(data.payload.id);
                    
                    stagedItems = stagedItems.filter(i => i.id !== data.payload.id);
                    stagedItems.push(data.payload);
                    updateGnbUI();
                    // 🚀 작업 중이라도 리스트를 다시 그려서 새로 추가된 아이템이 즉시 보이게 합니다.
                    renderStagedList();
                    
                    // 🚀 작업 중이 아닐 때만 버튼 숫자를 일반적인 형태로 갱신합니다.
                    if (!isProcessing) {
                        const filtered = stagedItems.filter(i => i.domain === currentTabFilter);
                        const draftOnlyCount = filtered.filter(i => i.status !== 'PUSHED').length;
                        draftBtn.textContent = `Draft (${draftOnlyCount})`;
                    }
                    return;
                }
                // 🚀 진행 상황 업데이트 시 처리 중인 ID 배열을 동기화하여 현재/다른 탭 모두 흐리게 렌더링되도록 합니다.
                else if (data.type === 'push_progress') {
                    let needsRender = false;
                    if (!isProcessing) {
                        isProcessing = true; 
                        startPushSpinner(); 
                        needsRender = true;
                    }
                    if (data.payload.processing_ids && JSON.stringify(processingIds) !== JSON.stringify(data.payload.processing_ids)) {
                        processingIds = data.payload.processing_ids;
                        needsRender = true;
                    }
                    // 🚀 개별 아이템 처리가 완료될 때마다 실시간으로 화면과 메모리에 반영합니다. (순차적 완료 처리)
                    if (data.payload.completed_item) {
                        const item = data.payload.completed_item;
                        const idx = stagedItems.findIndex(i => i.id === item.id);
                        if (idx !== -1) stagedItems[idx] = item;
                        checkedSessionIds.delete(item.id); // 완료된 아이템은 체크 해제
                        needsRender = true;
                    }
                    if (needsRender) {
                        renderStagedList(); // 변경점이 있으면 즉시 렌더링하여 타 탭과 상태 일치
                    }
                    draftBtn.textContent = `Draft (${data.payload.item_display}/${data.payload.total_items}) ${data.payload.percent}%...`;
                    updatePushBtnState(); 
                    return;
                }
                else if (data.type === 'push_idle') {
                    // 🚀 로컬에서 Push를 누른 직후, 과거에 큐잉되었던 check_progress의 응답이 
                    // 뒤늦게 도착하여 진행 상태를 강제로 원복시키는 Race Condition을 방지합니다.
                    if (isProcessing && (Date.now() - pushStartTime < 3000)) {
                        return;
                    }
                    if (isProcessing) {
                        isProcessing = false;
                        processingIds = []; // 🚀 작업이 끝난 경우 진행 상태 배열 비움
                        stopPushSpinner();
                        updatePushBtnState();
                        renderStagedList();

                        // 🚀 중단 완료 후 사용자에게 시스템 로그로 피드백을 제공합니다.
                        const div = document.createElement('div');
                        div.className = 'system';
                        div.style.padding = '10px';
                        div.style.background = '#fff3cd'; // 경고 배경색
                        div.style.borderRadius = '4px';
                        div.textContent = `System: 사용자의 요청으로 작업이 중단되었습니다.`;
                        log.appendChild(div);
                        div.scrollIntoView({ behavior: 'smooth', block: 'end' });
                    }
                    return;
                }
                else if (data.type === 'delete_success') {
                    // 🚀 삭제 성공 시 화면 하단에 'System: delete_success' 노티가 출력되는 현상을 차단합니다.
                    return; 
                }
                else if (data.type === 'push_success') {
                    isProcessing = false; 
                    processingIds = []; // 🚀 성공 시 배열 비움
                    stopPushSpinner();
                    deleteBtn.disabled = false;
                    draftBtn.disabled = false;
                    
                    const updatedItems = data.payload || [];
                    const updatedIds = updatedItems.map(i => i.id);
                    
                    // 🚀 성공적으로 푸시된 아이템의 체크박스를 해제하기 위해 세션에서 제거합니다.
                    updatedIds.forEach(id => checkedSessionIds.delete(id));
                    
                    stagedItems = stagedItems.filter(i => !updatedIds.includes(i.id));
                    stagedItems.push(...updatedItems);
                    
                    updateGnbUI();
                    renderStagedList();

                    const div = document.createElement('div');
                    div.className = 'system';
                    div.style.padding = '10px';
                    div.style.background = '#e6fffa';
                    div.style.borderRadius = '4px';
                    div.textContent = `System: Successfully masked, vectorized, and pushed ${updatedItems.length} items.`;
                    log.appendChild(div);
                    div.scrollIntoView({ behavior: 'smooth', block: 'end' });
                    return;
                }
                else if (data.type === 'file_processed') {
                    if (fileSpinnerInterval) clearInterval(fileSpinnerInterval);
                    fileSpinner.textContent = 'Done!';
                    setTimeout(() => {
                        fileSpinner.style.display = 'none';
                        fileInput.style.display = 'block';
                    }, 2000);
                    
                    processedFileContent = data.payload.masked;
                    updateSubmitToDrag();
                    
                    const div = document.createElement('div');
                    div.className = 'system';
                    div.style.padding = '10px';
                    div.style.background = '#e6fffa';
                    div.style.borderRadius = '4px';
                    div.textContent = `System: File OCR & Masking completed. Ready to export.`;
                    log.appendChild(div);
                    div.scrollIntoView({ behavior: 'smooth', block: 'end' });
                    return;
                }
                else if (data.type === 'export_success') {
                    // 🚀 Rust에서 실제 파일 저장이 완료되면 시스템 로그에 저장 경로를 표시합니다.
                    const div = document.createElement('div');
                    div.className = 'system';
                    div.style.padding = '10px';
                    div.style.background = '#e6fffa';
                    div.style.borderRadius = '4px';
                    div.textContent = `System: 파일이 성공적으로 디스크에 저장되었습니다. 경로: ${data.payload}`;
                    log.appendChild(div);
                    div.scrollIntoView({ behavior: 'smooth', block: 'end' });
                    return;
                }
                else if (data.type === 'reset_success') {
                    stagedItems = [];
                    checkedSessionIds.clear();
                    deletedSessionIds.clear();
                    updateGnbUI();
                    renderStagedList();
                    
                    const div = document.createElement('div');
                    div.className = 'system';
                    div.style.padding = '10px';
                    div.style.background = '#e6fffa';
                    div.style.borderRadius = '4px';
                    div.textContent = `System: ${data.message}`;
                    log.appendChild(div);
                    div.scrollIntoView({ behavior: 'smooth', block: 'end' });
                    return;
                }
                else if (data.type === 'error') {
                    window.download_progress = {}; // 🚀 에러 발생 시 멈춰있는 다운로드 UI 초기화
                    isProcessing = false; 
                    processingIds = []; // 🚀 에러 발생 시 배열 비움
                    stopPushSpinner();
                    if (fileSpinnerInterval) {
                        clearInterval(fileSpinnerInterval);
                        fileSpinner.style.display = 'none';
                        fileInput.style.display = 'block';
                    }
                    deleteBtn.disabled = false;
                    draftBtn.disabled = false;
                    updatePushBtnState(); 
                    renderStagedList(); 
                    
                    const div = document.createElement('div');
                    div.className = 'system';
                    div.style.padding = '10px';
                    div.style.background = '#ffe6e6';
                    div.style.color = '#d8000c';
                    div.style.borderRadius = '4px';
                    div.textContent = 'Error: ' + data.message;
                    log.appendChild(div);
                    div.scrollIntoView({ behavior: 'smooth', block: 'end' });
                    
                    // 🚀 에러 메시지가 권한, 폴더, 네트워크 등 치명적 오류인 경우 사용자에게 즉각 알림을 띄웁니다.
                    const msgLower = data.message.toLowerCase();
                    if (msgLower.includes('permission') || msgLower.includes('access') || msgLower.includes('권한') || msgLower.includes('denied') || msgLower.includes('실패')) {
                        alert("시스템 또는 네트워크 오류가 발생했습니다.\n관리자 권한으로 실행하거나 디스크 여유 공간 및 인터넷 연결을 확인해 주세요.\n\n상세 원인: " + data.message);
                    }
                    return;
                }
                else if (data.type === 'prompts_loaded') {
                    // 🚀 로드된 프롬프트 목록으로 datalist 옵션 구성 및 상태 업데이트
                    savedPrompts = data.payload || [];
                    promptDatalist.replaceChildren();
                    savedPrompts.forEach(pText => {
                        const opt = document.createElement('option');
                        opt.value = pText;
                        promptDatalist.appendChild(opt);
                    });
                    
                    // PROMPT 탭에 있을 경우 리스트 즉시 갱신
                    if (currentTabFilter === 'PROMPT') {
                        renderStagedList();
                    }
                    return;
                }
                else if (data.type === 'download_progress') {
                    window.download_progress = window.download_progress || {};
                    window.download_progress[data.model] = data.percent; // 🚀 전역 변수에 실시간 갱신

                    if (currentTabFilter === 'CONFIG') {
                        const safeId = data.model.replace(/[\s\(\)]+/g, '-');
                        // 🚀 백엔드의 다운로드 진행 퍼센트를 화면에 갱신하기 위해 Shadow DOM을 조회합니다.
                        const pb = shadow.getElementById('progress-bar-' + safeId);
                        const pc = shadow.getElementById('progress-container-' + safeId);
                        const btn = shadow.getElementById('btn-download-' + safeId);

                        const dict = window.lang_dict || {};
                        const currentLang = window.default_language || 'English';
                        const getText = (key) => dict[currentLang] ? dict[currentLang][key] : (dict['English'] ? dict['English'][key] : key);

                        if (pc && pc.style.display === 'none') {
                            pc.style.display = 'block';
                        }
                        
                        // 🚀 버튼 텍스트에 진행률(%)을 직접 표기합니다.
                        if (btn) {
                            btn.textContent = `${getText('model_downloading')} (${data.percent}%)`;
                            btn.disabled = true;
                            btn.style.background = '#6c757d';
                        }

                        if (pb) {
                            pb.style.width = `${data.percent}%`;
                            pb.textContent = `${data.percent}%`;
                        }
                    }
                    return;
                }
                else if (data.type === 'download_complete') {
                    window.download_progress = window.download_progress || {};
                    window.download_progress[data.model] = 100; // 🚀 다운로드 완전 종료 마킹

                    // 메모리 상태 동기화 (재진입 시 렌더링 유지)
                    if (!window.model_status) window.model_status = {};
                    window.model_status[data.model] = true;

                    if (currentTabFilter === 'CONFIG') {
                        const safeId = data.model.replace(/[\s\(\)]+/g, '-');
                        // 🚀 다운로드 완료 시 UI 상태 갱신을 위해 Shadow DOM을 조회합니다.
                        const pb = shadow.getElementById('progress-bar-' + safeId);
                        const pc = shadow.getElementById('progress-container-' + safeId);
                        const btn = shadow.getElementById('btn-download-' + safeId);
                        
                        const dict = window.lang_dict || {};
                        const currentLang = window.default_language || 'English';
                        const getText = (key) => dict[currentLang] ? dict[currentLang][key] : (dict['English'] ? dict['English'][key] : key);

                        if (pb && btn && pc) {
                            pb.style.width = `100%`;
                            pb.textContent = `완료!`;
                            setTimeout(() => {
                                pc.style.display = 'none';
                                btn.textContent = getText('model_downloaded');
                            }, 800);
                        }
                    }
                    const div = document.createElement('div');
                    div.className = 'system';
                    div.style.padding = '10px';
                    div.style.background = '#e6fffa';
                    div.style.borderRadius = '4px';
                    div.textContent = `System: ${data.model} 모델 다운로드가 완료되었습니다.`;
                    log.appendChild(div);
                    div.scrollIntoView({ behavior: 'smooth', block: 'end' });
                    return;
                }
                else if (data.type === 'delete_models_success') {
                    // 🚀 전역 상태를 비워 모든 모델을 미설치 상태로 되돌립니다.
                    window.model_status = {};
                    window.download_progress = {};
                    
                    const dict = window.lang_dict || {};
                    const currentLang = window.default_language || 'English';
                    const getText = (key) => dict[currentLang] ? dict[currentLang][key] : (dict['English'] ? dict['English'][key] : key);
                    
                    alert(getText('model_delete_success') || 'All models have been successfully deleted.');
                    
                    if (currentTabFilter === 'CONFIG') {
                        renderStagedList();
                    }
                    
                    const div = document.createElement('div');
                    div.className = 'system';
                    div.style.padding = '10px';
                    div.style.background = '#e6fffa';
                    div.style.borderRadius = '4px';
                    div.textContent = `System: 설치된 모든 모델 파일이 디스크에서 완전히 제거되었습니다.`;
                    log.appendChild(div);
                    div.scrollIntoView({ behavior: 'smooth', block: 'end' });
                    return;
                }
            } catch (err) {
            }
            
            // 🚀 시스템 메시지가 JSON 객체 문자열이거나 처리되지 않은 내부 RPC 응답일 경우, 화면에 코드 덩어리(HTML 파일처럼)로 노출되는 현상을 원천 차단합니다.
            const detailStr = typeof e.detail === 'string' ? e.detail : JSON.stringify(e.detail);
            if (!detailStr.trim().startsWith('{') && !detailStr.trim().startsWith('[')) {
                const div = document.createElement('div');
                div.className = 'system';
                div.style.padding = '10px';
                div.style.background = '#f0f4ff';
                div.style.borderRadius = '4px';
                div.textContent = 'System: ' + detailStr;
                log.appendChild(div);
                div.scrollIntoView({ behavior: 'smooth', block: 'end' });
            }
        });

        autoExtract();
        
        setTimeout(() => {
            if (window.rpc) {
                window.rpc("fetch_drafts");
                window.rpc("check_progress");
                window.rpc("fetch_prompts"); // 🚀 초기 구동 시 프롬프트 목록 불러오기
            }
        }, 300);
    }
    initUI();
})();
"#;

async fn setup_page(browser: Arc<Browser>, page: chromiumoxide::Page, is_authenticated: bool) -> Result<(), Box<dyn std::error::Error>> {
    // 외부 스크립트 차단 우회를 위해 예측 불가능한 랜덤 바인딩명 및 전역 변수명 생성
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let rpc_binding_name = format!("__sys_rpc_{:x}", now);
    let sidebar_var_name = format!("__sys_sidebar_{:x}", now);

    let _ = page.execute(AddBindingParams::new(&rpc_binding_name)).await; // 바인딩명 변경
    
    let app_config = load_app_config();
    let default_tab = app_config.default_tab.unwrap_or_else(|| "DRAFT".to_string()).to_uppercase();
    let default_language = app_config.language.unwrap_or_else(|| "English".to_string());
    let custom_tabs_str = app_config.custom_tabs.map(|tabs| serde_json::to_string(&tabs).unwrap_or_else(|_| "null".to_string())).unwrap_or_else(|| "null".to_string());
    let auto_extract = app_config.auto_extract.unwrap_or(true); // 🚀 기본값은 활성화(true)
    let enable_masking = app_config.enable_masking.unwrap_or(true); // 🚀 마스킹 기능도 기본값 활성화(true)
    
    let app_dir = terminal_logis_center::utils::get_app_dir();

    // 🚀 마스킹 단어 사전 파일(adjectives.txt, nouns.txt)을 바이너리에 내장하고 AppData 폴더로 무조건 복사합니다.
    let _ = std::fs::write(app_dir.join("adjectives.txt"), include_str!("../adjectives.txt"));
    let _ = std::fs::write(app_dir.join("nouns.txt"), include_str!("../nouns.txt"));

    // 🚀 [초기 실행 시 파일 자동 연결] 
    // GGUF 모델 파일 유무와 관계없이, 앱이 실행될 때마다 프로젝트 내부에 있는 최신 JSON 설정 파일들을 
    // AppData의 구동 폴더로 무조건 복사(덮어쓰기)하여 파일 누락을 원천 차단합니다.
    let glm_weights = app_dir.join("models").join("glm_ocr").join("GLM-OCR-Q8_0.gguf");
    let privacy_weights = app_dir.join("models").join("privacy-filter").join("model.safetensors");
    let embed_weights = app_dir.join("models").join("embeddings").join("embeddinggemma-300m-Q4_0.gguf");

    {
        // 🚀 슬래시(/)가 섞여서 출력되는 경로 표기 문제를 해결하기 위해 .join("models").join("...") 형식으로 분리합니다.
        let base = app_dir.join("models").join("glm_ocr");
        let _ = std::fs::create_dir_all(&base);
        let _ = std::fs::write(base.join("config.json"), include_str!("../models/glm_ocr/config.json"));
        let _ = std::fs::write(base.join("tokenizer_config.json"), include_str!("../models/glm_ocr/tokenizer_config.json"));
        let _ = std::fs::write(base.join("tokenizer.json"), include_str!("../models/glm_ocr/tokenizer.json")); 
        let _ = std::fs::write(base.join("preprocessor_config.json"), include_str!("../models/glm_ocr/preprocessor_config.json")); // 🚀 추가
    }
    {
        let base = app_dir.join("models").join("privacy-filter");
        let _ = std::fs::create_dir_all(&base);
        let _ = std::fs::write(base.join("config.json"), include_str!("../models/privacy-filter/config.json"));
        let _ = std::fs::write(base.join("tokenizer_config.json"), include_str!("../models/privacy-filter/tokenizer_config.json"));
        let _ = std::fs::write(base.join("tokenizer.json"), include_str!("../models/privacy-filter/tokenizer.json")); 
    }
    {
        let base = app_dir.join("models").join("embeddings");
        let _ = std::fs::create_dir_all(&base);
        let _ = std::fs::write(base.join("config.json"), include_str!("../models/embeddings/config.json"));
        let _ = std::fs::write(base.join("tokenizer.json"), include_str!("../models/embeddings/tokenizer.json")); 
    }

    let glm_mmproj = app_dir.join("models").join("glm_ocr").join("mmproj-GLM-OCR-Q8_0.gguf"); 

    // 🚀 [무결성 검증] 단순히 파일이 존재하는 것뿐만 아니라, 다운로드가 끊겨 생성된 쓰레기 파일(예: 10MB 미만)인지 용량까지 엄격하게 검사하여 앱 크래시를 원천 차단합니다.
    let is_valid_model = |p: &std::path::PathBuf| -> bool {
        p.exists() && std::fs::metadata(p).map(|m| m.len()).unwrap_or(0) > 10_000_000 // 최소 10MB 이상이어야 정상 가중치 파일로 인정
    };

    let glm_exists = is_valid_model(&glm_weights) && is_valid_model(&glm_mmproj);
    let privacy_exists = is_valid_model(&privacy_weights);
    let embed_exists = is_valid_model(&embed_weights);
    let model_status_str = json!({
        "GLM-OCR": glm_exists,
        "Privacy-Filter": privacy_exists,
        "Embedding": embed_exists
    }).to_string();
    
    // 🚀 컴파일 시 빌드에 language.json 파일을 아예 내장(Embed)시켜버려, 배포 후에도 경로 문제 없이 다국어를 100% 보장합니다.
    let lang_dict_str = include_str!("language.json").to_string();

    // 🚀 [시스템 파일 자동 연결] 모델 구동에 필요한 JSON 설정 파일들을 바이너리에 내장합니다.
    // 사용자가 무거운 모델 파일(.gguf)만 받아도 즉시 연동되도록 하기 위함입니다.
    let glm_config = include_str!("../models/glm_ocr/config.json");
    let glm_tokenizer_config = include_str!("../models/glm_ocr/tokenizer_config.json");
    let glm_tokenizer = include_str!("../models/glm_ocr/tokenizer.json"); // 🚀 추가
    let glm_preprocessor = include_str!("../models/glm_ocr/preprocessor_config.json"); // 🚀 추가
    let privacy_config = include_str!("../models/privacy-filter/config.json");
    let privacy_tokenizer_config = include_str!("../models/privacy-filter/tokenizer_config.json");
    let privacy_tokenizer = include_str!("../models/privacy-filter/tokenizer.json"); // 🚀 추가
    let embed_config = include_str!("../models/embeddings/config.json");
    let embed_tokenizer = include_str!("../models/embeddings/tokenizer.json"); // 🚀 추가

    // 헬퍼 클로저: 특정 모델 폴더에 내장된 JSON 파일들을 자동으로 생성/복원합니다.
    let ensure_model_configs = |app_dir: &std::path::Path, model_type: &str| {
        let base = app_dir.join("models").join(match model_type {
            "GLM-OCR" => "glm_ocr",
            "Privacy-Filter" => "privacy-filter",
            "Embedding" => "embeddings",
            _ => return,
        });
        let _ = std::fs::create_dir_all(&base);
        match model_type {
            "GLM-OCR" => {
                let _ = std::fs::write(base.join("config.json"), glm_config);
                let _ = std::fs::write(base.join("tokenizer_config.json"), glm_tokenizer_config);
                let _ = std::fs::write(base.join("tokenizer.json"), glm_tokenizer); // 🚀 추가
                let _ = std::fs::write(base.join("preprocessor_config.json"), glm_preprocessor); // 🚀 추가
            },
            "Privacy-Filter" => {
                let _ = std::fs::write(base.join("config.json"), privacy_config);
                let _ = std::fs::write(base.join("tokenizer_config.json"), privacy_tokenizer_config);
                let _ = std::fs::write(base.join("tokenizer.json"), privacy_tokenizer); // 🚀 추가
            },
            "Embedding" => {
                let _ = std::fs::write(base.join("config.json"), embed_config);
                let _ = std::fs::write(base.join("tokenizer.json"), embed_tokenizer); // 🚀 추가
            },
            _ => (),
        }
    };
    
    // OVERLAY_SCRIPT 내의 예측 가능한 전역 변수(window.rpc 등)를 랜덤 생성한 이름으로 동적 치환
    let overlay_script_replaced = OVERLAY_SCRIPT
        .replace("window.rpc", &format!("window.{}", rpc_binding_name))
        .replace("window.geminiSidebarLoaded", &format!("window.{}", sidebar_var_name));

    let full_script = format!("window.is_authenticated = {};\nwindow.default_tab = \"{}\";\nwindow.default_language = \"{}\";\nwindow.custom_tabs = {};\nwindow.auto_extract = {};\nwindow.enable_masking = {};\nwindow.model_status = {};\nwindow.lang_dict = {};\n{}", is_authenticated, default_tab, default_language, custom_tabs_str, auto_extract, enable_masking, model_status_str, lang_dict_str, overlay_script_replaced);
    // 페이지가 새로고침되거나 다른 페이지로 이동하더라도 스크립트가 유지되도록 등록합니다.
    let _ = page.execute(AddScriptToEvaluateOnNewDocumentParams::new(&full_script)).await;
    let _ = page.evaluate(full_script).await;
    let mut bindings = page.event_listener::<EventBindingCalled>().await?;
    let page_clone = page.clone();
    let _browser_clone = browser.clone();
    
    tokio::task::spawn(async move {
        let app_dir = terminal_logis_center::utils::get_app_dir();
        while let Some(event) = bindings.next().await {
            if event.name == rpc_binding_name { // 이벤트 수신명 변경
                let payload = event.payload.trim_matches('"').to_string();
                let response = if payload.starts_with("sync_data:") {
                    // 🚀 [Fix] RPC 이벤트 루프(UI 스레드)가 멈추지 않도록 무거운 OCR/저장 로직을 백그라운드 태스크로 분리합니다.
                    let data = payload["sync_data:".len()..].to_string();
                    let browser_c = _browser_clone.clone();
                    let app_dir_c = app_dir.clone();
                    
                    tokio::task::spawn(async move {
                        let result_str = match serde_json::from_str::<db::CommerceRecord>(&data) {
                            Ok(mut record) => {
                                use terminal_logis_center::harness::DefaultHarness;
                                let harness = DefaultHarness;
                                
                                if record.url.starts_with("file://") && record.context.contains("data:image/") {
                                    if let Some(base64_part) = record.context.split("data:").nth(1) {
                                        let full_data_url = format!("data:{}", base64_part.trim());
                                        let ocr_result = {
                                            let mut model_guard = OCR_MODEL.lock().unwrap();
                                            if model_guard.is_none() {
                                                let model_path = app_dir_c.join("models").join("glm_ocr");
                                                let model_path_str = model_path.to_string_lossy().to_string();
                                                let ocr_device = Device::new_cuda(0).unwrap_or(Device::Cpu); 
                                                if let Ok(model) = GlmOcrGenerateModel::init(&model_path_str, Some(&ocr_device), None) {
                                                    *model_guard = Some(model);
                                                }
                                            }
                                            let res = if let Some(model) = model_guard.as_mut() {
                                                let params = ChatCompletionParameters {
                                                    messages: vec![Message {
                                                        role: "user".to_string(),
                                                        parts: vec![Part { text: "Extract text from image and return as JSON format".to_string(), image_url: Some(full_data_url.to_string()) }],
                                                    }],
                                                    model: "glm-ocr".to_string(),
                                                    max_tokens: Some(2048),
                                                    temperature: Some(0.2),
                                                    top_p: Some(0.95),
                                                    top_k: None, repeat_penalty: Some(1.2), repeat_last_n: Some(64), seed: Some(42),
                                                };
                                                model.generate(params).map(|res| res.choices[0].message.content.clone()).unwrap_or_default()
                                            } else { "OCR Model Load Error".to_string() };
                                            
                                            *model_guard = None;
                                            res
                                        };
                                        let display_ocr = ocr_result.replace("```markdown", "").replace("```", "").trim().to_string();
                                        record.context = format!("{}\n---\n[OCR 결과]\n{}", record.context, display_ocr);
                                    } 
                                }
                                println!("[System] 동기화 수신: {} (ID: {})", record.title, record.id);

                                let is_image = record.context.contains("data:image/") || record.url.starts_with("file://");
                                if !is_image && record.context.contains('<') && record.context.contains('>') {
                                    let cleaned_context = harness.clean_html(&record.context);
                                    record.context = cleaned_context;
                                    println!("[System] 동기화: 웹페이지 태그 평탄화 수행됨.");
                                } else if !is_image {
                                    println!("[System] 동기화: 순수 텍스트 감지 (평탄화 건너뜀).");
                                }

                                let updated = record.clone();
                                db::save_records(vec![record], None).await.map(|_| json!({"type":"sync_success","payload":updated}).to_string()).unwrap_or_else(|e| e.to_string())
                            },
                            Err(e) => e.to_string(),
                        };
                        
                        // 🚀 데이터 처리 완료 후, 현재 열려있는 모든 탭에 성공 신호를 전송하여 화면을 즉시 동기화합니다.
                        let script = format!("window.dispatchEvent(new CustomEvent('rpc_response', {{ detail: {} }}));", json!(result_str));
                        if let Ok(pages) = browser_c.pages().await {
                            for p in pages {
                                let _ = p.evaluate(script.clone()).await;
                            }
                        }
                    });
                    json!({"type": "sync_started"}).to_string()
                } else if payload == "fetch_drafts" {
                    db::fetch_drafts().await.map(|d| json!({"type":"drafts_loaded","payload":d}).to_string()).unwrap_or_else(|e| e.to_string())
                } else if payload.starts_with("delete_drafts:") {
                    let data = &payload["delete_drafts:".len()..];
                    if let Ok(ids) = serde_json::from_str::<Vec<String>>(data) {
                        if let Ok(table) = db::get_or_create_table().await {
                            for id in ids {
                                let expr = format!("id = '{}'", id);
                                let _ = table.delete(&expr).await;
                            }
                        }
                    }
                    json!({"type":"delete_success"}).to_string()
                } else if payload.starts_with("mask_and_push_batch:") {
                    // 🚀 배치 작업 시작 전 중단 신호를 초기화합니다.
                    PUSH_CANCEL_SIGNAL.store(false, Ordering::SeqCst);
                    let data = payload["mask_and_push_batch:".len()..].to_string();
                    let browser_c = _browser_clone.clone();
                    let app_dir_c = app_dir.clone();
                    
                    // 🚀 긴 시간이 걸리는 Pushing 연산을 백그라운드로 분리하여 탭 프리징을 방지합니다.
                    tokio::task::spawn(async move {
                        let result_str = if let Ok(req) = serde_json::from_str::<serde_json::Value>(&data) {
                            if let Some(ids) = req.get("ids").and_then(|i| i.as_array()) {
                                let id_strings: Vec<String> = ids.iter().filter_map(|i| i.as_str().map(String::from)).collect();
                                
                                let mut target_records = Vec::new();
                                if let Ok(drafts) = db::fetch_drafts().await {
                                    target_records = drafts.into_iter().filter(|r| id_strings.contains(&r.id)).collect();
                                }
                                
                                if let Ok(table) = db::get_or_create_table().await {
                                    let mut has_error = None;
                                    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);

                                    let total_items = target_records.len();
                                    let total_steps = total_items * 3;
                                    let mut current_step = 0;

                                    let mut active_processing_ids = id_strings.clone();

                                    {
                                        let initial_payload = json!({"item_display": 1, "total_items": total_items, "percent": 0, "processing_ids": active_processing_ids.clone()});
                                        *GLOBAL_PROGRESS.lock().unwrap() = Some(initial_payload.clone());
                                        let script = format!("window.dispatchEvent(new CustomEvent('rpc_response', {{ detail: {} }}));", json!(json!({"type": "push_progress", "payload": initial_payload})));
                                        // 🚀 [Fix] 새 탭을 열어도 진행 상황을 인지할 수 있도록 모든 브라우저 페이지에 브로드캐스트합니다.
                                        if let Ok(pages) = browser_c.pages().await {
                                            for p in pages { let _ = p.evaluate(script.clone()).await; }
                                        }
                                    }

                                    let needs_ocr = target_records.iter().any(|r| r.context.starts_with("data:image/") || r.context.starts_with("data:application/pdf"));
                                    
                                    use terminal_logis_center::harness::DefaultHarness;
                                    let harness = DefaultHarness;
                                    for record in &mut target_records {
                                        let is_image = record.context.starts_with("data:image/") || record.url.starts_with("file://");
                                        if !is_image {
                                            let raw_content = record.context.clone();
                                            if raw_content.contains('<') && raw_content.contains('>') {
                                                record.context = harness.clean_html(&raw_content);
                                                println!("[System] 웹페이지 HTML 태그 평탄화 완료. (ID: {})", record.id);
                                            } else {
                                                println!("[System] 웹페이지 순수 텍스트 유지됨. (ID: {})", record.id);
                                            }
                                        }
                                    }

                                    if needs_ocr {
                                        {
                                            let mut model_guard = OCR_MODEL.lock().unwrap();
                                            if model_guard.is_none() {
                                                let model_path = app_dir_c.join("models").join("glm_ocr");
                                                let model_path_str = model_path.to_string_lossy().to_string();
                                                match GlmOcrGenerateModel::init(&model_path_str, Some(&device), None) {
                                                    Ok(model) => *model_guard = Some(model),
                                                    Err(e) => {
                                                        let err_msg = format!("OCR Init Error: {:?}", e);
                                                        println!("[Error] {}", err_msg);
                                                        has_error = Some(err_msg);
                                                    }
                                                }
                                            }
                                        }

                                        for (idx, record) in target_records.iter_mut().enumerate() {
                                            if PUSH_CANCEL_SIGNAL.load(Ordering::SeqCst) {
                                                has_error = Some("Push operation cancelled by user.".to_string());
                                                break;
                                            }

                                            let is_image = record.context.starts_with("data:image/") || record.context.starts_with("data:application/pdf");
                                            if is_image && has_error.is_none() {
                                                let mut ocr_success = false;
                                                let mut cleaned_ocr = String::new();
                                                {
                                                    let mut model_guard = OCR_MODEL.lock().unwrap();
                                                    if let Some(model) = model_guard.as_mut() {
                                                        let params = ChatCompletionParameters {
                                                            messages: vec![Message {
                                                                role: "user".to_string(),
                                                                parts: vec![Part { text: "Extract text from image and return as JSON format".to_string(), image_url: Some(record.context.clone()) }],
                                                            }],
                                                            model: "glm-ocr".to_string(),
                                                            max_tokens: Some(2048),
                                                            temperature: Some(0.2),
                                                            top_p: Some(0.95),
                                                            top_k: None, repeat_penalty: Some(1.2), repeat_last_n: Some(64), seed: Some(42),
                                                        };
                                                        match model.generate(params) {
                                                            Ok(res) => {
                                                                let raw_ocr = res.choices[0].message.content.clone();
                                                                cleaned_ocr = raw_ocr.replace("```markdown", "").replace("```", "").trim().to_string();
                                                                ocr_success = true;
                                                            },
                                                            Err(e) => {
                                                                has_error = Some(format!("OCR Error: {}", e));
                                                            }
                                                        }
                                                    }
                                                }
                                                if ocr_success {
                                                    record.context = cleaned_ocr.clone();
                                                }
                                            }
                                            
                                            current_step += 1;
                                            let percent = (current_step as f64 / total_steps as f64 * 100.0) as usize;
                                            let payload = json!({"item_display": idx + 1, "total_items": total_items, "percent": percent, "processing_ids": active_processing_ids.clone()});
                                            *GLOBAL_PROGRESS.lock().unwrap() = Some(payload.clone());
                                            
                                            // 🚀 진행 상황 브로드캐스트
                                            let script = format!("window.dispatchEvent(new CustomEvent('rpc_response', {{ detail: {} }}));", json!(json!({"type": "push_progress", "payload": payload})));
                                            if let Ok(pages) = browser_c.pages().await {
                                                for p in pages { let _ = p.evaluate(script.clone()).await; }
                                            }
                                        }

                                        {
                                            let mut model_guard = OCR_MODEL.lock().unwrap();
                                            *model_guard = None; 
                                        }
                                        force_memory_cleanup();
                                        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
                                    } else {
                                        current_step += total_items; 
                                    }

                                    let enable_masking = load_app_config().enable_masking.unwrap_or(true);
                                    if has_error.is_none() {
                                        if enable_masking {
                                            {
                                                let mut pm_guard = PRIVACY_MANAGER.lock().unwrap();
                                                if pm_guard.is_none() {
                                                    let pm_path = app_dir_c.join("models").join("privacy-filter");
                                                    let pm_path_str = pm_path.to_string_lossy().to_string();
                                                    *pm_guard = terminal_logis_center::privacy_filter::masking::PrivacyManager::new(&pm_path_str, &device).ok();
                                                }
                                            }
                                            
                                            let mut privacy_session = terminal_logis_center::privacy_filter::masking::PrivacySession::new();

                                            for (idx, record) in target_records.iter_mut().enumerate() {
                                                if PUSH_CANCEL_SIGNAL.load(Ordering::SeqCst) {
                                                    has_error = Some("Push operation cancelled by user.".to_string());
                                                    break;
                                                }

                                                let mut masked_success = false;
                                                let mut masked_text = String::new();
                                                {
                                                    let pm_guard = PRIVACY_MANAGER.lock().unwrap();
                                                    if let Some(pm) = pm_guard.as_ref() {
                                                        masked_text = pm.mask_text_with_session(&record.context, &mut privacy_session, idx).unwrap_or_else(|e| {
                                                            has_error = Some(format!("Masking failed: {}", e));
                                                            record.context.clone()
                                                        });
                                                        masked_success = true;
                                                    } else {
                                                        has_error = Some("Privacy Filter 모델 로드에 실패했습니다.".to_string());
                                                    }
                                                }
                                                if masked_success {
                                                    record.masking = masked_text;
                                                }

                                                current_step += 1;
                                                let percent = (current_step as f64 / total_steps as f64 * 100.0) as usize;
                                                let payload = json!({"item_display": idx + 1, "total_items": total_items, "percent": percent, "processing_ids": active_processing_ids.clone()});
                                                *GLOBAL_PROGRESS.lock().unwrap() = Some(payload.clone());
                                                
                                                // 🚀 진행 상황 브로드캐스트
                                                let script = format!("window.dispatchEvent(new CustomEvent('rpc_response', {{ detail: {} }}));", json!(json!({"type": "push_progress", "payload": payload})));
                                                if let Ok(pages) = browser_c.pages().await {
                                                    for p in pages { let _ = p.evaluate(script.clone()).await; }
                                                }
                                            }
                                            
                                            {
                                                let mut pm_guard = PRIVACY_MANAGER.lock().unwrap();
                                                *pm_guard = None; 
                                            }
                                            force_memory_cleanup();
                                        } else {
                                            for record in target_records.iter_mut() {
                                                record.masking = record.context.clone();
                                            }
                                            current_step += total_items; 
                                        }
                                    } else {
                                        current_step += total_items;
                                    }

                                    if has_error.is_none() {
                                        let needs_embedding = target_records.iter().any(|r| !r.masking.trim().is_empty());
                                        
                                        if needs_embedding {
                                            {
                                                let mut em_guard = EMBEDDING_MODEL.lock().unwrap();
                                                if em_guard.is_none() {
                                                    let em_path = app_dir_c.join("models").join("embeddings");
                                                    let em_path_str = em_path.to_string_lossy().to_string();
                                                    *em_guard = terminal_logis_center::embedding::EmbeddingModel::new_with_device(&em_path_str, &device).ok();
                                                }
                                            }
                                            for (idx, record) in target_records.iter_mut().enumerate() {
                                                if PUSH_CANCEL_SIGNAL.load(Ordering::SeqCst) {
                                                    has_error = Some("Push operation cancelled by user.".to_string());
                                                    break;
                                                }

                                                let text_to_embed = record.masking.trim();
                                                if text_to_embed.is_empty() {
                                                    record.vector = vec![0.0; 768];
                                                } else {
                                                    let em_guard = EMBEDDING_MODEL.lock().unwrap();
                                                    if let Some(em) = em_guard.as_ref() {
                                                        record.vector = em.embed(text_to_embed).unwrap_or_else(|e| {
                                                            has_error = Some(format!("Embedding failed: {}", e));
                                                            vec![0.0; 768]
                                                        });
                                                    } else {
                                                        has_error = Some("Embedding 모델 로드에 실패했습니다.".to_string());
                                                    }
                                                }

                                                record.status = "PUSHED".to_string();
                                                let expr = format!("id = '{}'", record.id);
                                                let _ = table.delete(&expr).await;
                                                let _ = db::save_records(vec![record.clone()], None).await;

                                                active_processing_ids.retain(|id| id != &record.id);

                                                current_step += 1;
                                                let percent = (current_step as f64 / total_steps as f64 * 100.0) as usize;
                                                let payload = json!({
                                                    "item_display": idx + 1, 
                                                    "total_items": total_items, 
                                                    "percent": percent, 
                                                    "processing_ids": active_processing_ids.clone(),
                                                    "completed_item": record.clone()
                                                });
                                                *GLOBAL_PROGRESS.lock().unwrap() = Some(payload.clone());
                                                
                                                // 🚀 진행 상황 브로드캐스트
                                                let script = format!("window.dispatchEvent(new CustomEvent('rpc_response', {{ detail: {} }}));", json!(json!({"type": "push_progress", "payload": payload})));
                                                if let Ok(pages) = browser_c.pages().await {
                                                    for p in pages { let _ = p.evaluate(script.clone()).await; }
                                                }
                                            }
                                            
                                            {
                                                let mut em_guard = EMBEDDING_MODEL.lock().unwrap();
                                                *em_guard = None; 
                                            }
                                            force_memory_cleanup();
                                        } else {
                                            for (idx, record) in target_records.iter_mut().enumerate() {
                                                if PUSH_CANCEL_SIGNAL.load(Ordering::SeqCst) {
                                                    has_error = Some("Push operation cancelled by user.".to_string());
                                                    break;
                                                }

                                                record.vector = vec![0.0; 768];
                                                record.status = "PUSHED".to_string();
                                                let expr = format!("id = '{}'", record.id);
                                                let _ = table.delete(&expr).await;
                                                let _ = db::save_records(vec![record.clone()], None).await;

                                                active_processing_ids.retain(|id| id != &record.id);
                                                current_step += 1;
                                                let percent = (current_step as f64 / total_steps as f64 * 100.0) as usize;
                                                let payload = json!({
                                                    "item_display": idx + 1, 
                                                    "total_items": total_items, 
                                                    "percent": percent, 
                                                    "processing_ids": active_processing_ids.clone(),
                                                    "completed_item": record.clone()
                                                });
                                                *GLOBAL_PROGRESS.lock().unwrap() = Some(payload.clone());
                                                
                                                // 🚀 진행 상황 브로드캐스트
                                                let script = format!("window.dispatchEvent(new CustomEvent('rpc_response', {{ detail: {} }}));", json!(json!({"type": "push_progress", "payload": payload})));
                                                if let Ok(pages) = browser_c.pages().await {
                                                    for p in pages { let _ = p.evaluate(script.clone()).await; }
                                                }
                                            }
                                        }
                                    }

                                    *GLOBAL_PROGRESS.lock().unwrap() = None; 
                                    
                                    if PUSH_CANCEL_SIGNAL.load(Ordering::SeqCst) {
                                        has_error = Some("Push operation cancelled by user.".to_string());
                                    }

                                    if let Some(err_msg) = has_error {
                                        json!({"type": "error", "message": err_msg}).to_string()
                                    } else {
                                        json!({"type": "push_success", "payload": target_records}).to_string()
                                    }
                                } else {
                                    json!({"type": "error", "message": "Failed to access database table."}).to_string()
                                }
                            } else {
                                json!({"type": "error", "message": "Invalid ids in request."}).to_string()
                            }
                        } else {
                            json!({"type": "error", "message": "Invalid request payload format."}).to_string()
                        };
                        
                        // 🚀 푸시 완료 응답 역시 모든 창에 브로드캐스트합니다.
                        let script = format!("window.dispatchEvent(new CustomEvent('rpc_response', {{ detail: {} }}));", json!(result_str));
                        if let Ok(pages) = browser_c.pages().await {
                            for p in pages {
                                let _ = p.evaluate(script.clone()).await;
                            }
                        }
                    });
                    json!({"type": "push_started"}).to_string()
                } else if payload == "cancel_push" {
                    // 🚀 중단 신호를 켭니다. mask_and_push_batch의 각 Phase 진입 전후에 이 플래그를 체크하게 됩니다.
                    PUSH_CANCEL_SIGNAL.store(true, Ordering::SeqCst);
                    json!({"type": "push_idle"}).to_string()
                } else if payload.starts_with("process_file:") {
                    let full_data_url = payload["process_file:".len()..].to_string();
                    let browser_c = _browser_clone.clone();
                    let app_dir_c = app_dir.clone();

                    tokio::task::spawn(async move {
                        let mut ocr_result = String::new();
                        let mut masked_result = String::new();
                        let mut has_error = None;

                        // 1. OCR Extract
                        {
                            {
                                let mut model_guard = OCR_MODEL.lock().unwrap();
                                if model_guard.is_none() {
                                    let model_path = app_dir_c.join("models").join("glm_ocr");
                                    let model_path_str = model_path.to_string_lossy().to_string();
                                    let ocr_device = Device::new_cuda(0).unwrap_or(Device::Cpu);
                                    match GlmOcrGenerateModel::init(&model_path_str, Some(&ocr_device), None) {
                                        Ok(model) => *model_guard = Some(model),
                                        Err(e) => {
                                            let err_msg = format!("OCR Init Error: {:?}", e);
                                            println!("[Error] {}", err_msg);
                                            has_error = Some(err_msg);
                                        }
                                    }
                                }
                                if let Some(model) = model_guard.as_mut() {
                                    let params = ChatCompletionParameters {
                                        messages: vec![Message {
                                            role: "user".to_string(),
                                            parts: vec![Part { text: "Extract text from image and return as JSON format".to_string(), image_url: Some(full_data_url.to_string()) }],
                                        }],
                                        model: "glm-ocr".to_string(),
                                        max_tokens: Some(2048),
                                        temperature: Some(0.2),
                                        top_p: Some(0.95),
                                        top_k: None, repeat_penalty: Some(1.2), repeat_last_n: Some(64), seed: Some(42),
                                    };
                                    match model.generate(params) {
                                        Ok(res) => {
                                            ocr_result = res.choices[0].message.content.clone();
                                            println!("[GlmOcr] 단일 파일 생성된 텍스트 결과:\n{}", ocr_result);
                                        },
                                        Err(e) => {
                                            let err_msg = format!("OCR Error: {}", e);
                                            println!("[Error] 단일 파일 OCR 처리 중 예외 발생: {:?}", e);
                                            has_error = Some(err_msg);
                                        }
                                    }
                                } else {
                                    has_error = Some("OCR 모델 로드에 실패했습니다.".to_string());
                                }
                                *model_guard = None;
                            } 
                            force_memory_cleanup(); 
                            tokio::time::sleep(tokio::time::Duration::from_millis(300)).await; 
                        }

                        // 2. Privacy Filter
                        if has_error.is_none() && !ocr_result.is_empty() {
                            let mut pm_guard = PRIVACY_MANAGER.lock().unwrap();
                            let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
                            if pm_guard.is_none() {
                                let pm_path = app_dir_c.join("models").join("privacy-filter");
                                let pm_path_str = pm_path.to_string_lossy().to_string();
                                *pm_guard = terminal_logis_center::privacy_filter::masking::PrivacyManager::new(&pm_path_str, &device).ok();
                            }
                            if let Some(pm) = pm_guard.as_ref() {
                                masked_result = pm.mask_text(&ocr_result).unwrap_or_else(|e| {
                                    has_error = Some(format!("Masking failed: {}", e));
                                    ocr_result.clone()
                                });
                            } else {
                                has_error = Some("Privacy Filter 모델 로드에 실패했습니다.".to_string());
                                masked_result = ocr_result.clone();
                            }
                            *pm_guard = None;
                            force_memory_cleanup(); 
                        }

                        let result_str = if let Some(err_msg) = has_error {
                            json!({"type": "error", "message": err_msg}).to_string()
                        } else {
                            println!("[System] [단일 파일 처리] 최종 전처리 결과:\n{}", masked_result);
                            json!({"type": "file_processed", "payload": {"ocr": ocr_result, "masked": masked_result}}).to_string()
                        };

                        let script = format!("window.dispatchEvent(new CustomEvent('rpc_response', {{ detail: {} }}));", json!(result_str));
                        if let Ok(pages) = browser_c.pages().await {
                            for p in pages {
                                let _ = p.evaluate(script.clone()).await;
                            }
                        }
                    });
                    json!({"type": "process_started"}).to_string()
                } else if payload.starts_with("export_to_file:") {
                    // 🚀 웹 브라우저의 드래그앤드랍 제약을 해결하기 위해 Rust 백엔드에서 물리적 파일 생성을 담당합니다.
                    let data = &payload["export_to_file:".len()..];
                    let response_json = if let Ok(req) = serde_json::from_str::<serde_json::Value>(data) {
                        let prompt = req.get("prompt").and_then(|v| v.as_str()).unwrap_or("N/A");
                        let processed_file = req.get("processed_file").and_then(|v| v.as_str()).unwrap_or("");
                        let ids: Vec<String> = req.get("ids")
                            .and_then(|v| v.as_array())
                            .map(|arr| arr.iter().filter_map(|i| i.as_str().map(String::from)).collect())
                            .unwrap_or_default();
                            
                        let mut export_content = format!("[Prompt]\n{}\n\n", prompt);
                        if !processed_file.is_empty() {
                            export_content.push_str(&format!("[File Masked Text]\n{}\n\n", processed_file));
                        }
                        
                        let mut selected_items = Vec::new();
                        if let Ok(drafts) = db::fetch_drafts().await {
                            selected_items = drafts.into_iter()
                                .filter(|r| ids.contains(&r.id) && r.status == "PUSHED")
                                .collect();
                        }
                        
                        export_content.push_str(&format!("[Selected Items ({})]\n", selected_items.len()));
                        
                        for item in selected_items {
                            let content = if !item.masking.is_empty() { &item.masking } else { &item.context };
                            export_content.push_str(&format!("\n--- ID: {} ---\n[Domain]: {}\n[Title]: {}\n[Content]:\n{}\n", 
                                item.id, item.domain, item.title, content));
                        }
                        
                        // 🚀 OS 호환성을 위해 AppData 폴더를 사용합니다.
                        let export_dir = app_dir.join("exports");
                        let _ = std::fs::create_dir_all(&export_dir);
                        let file_path = export_dir.join("drag.context").to_string_lossy().to_string();
                        
                        match std::fs::write(&file_path, export_content) {
                            Ok(_) => {
                                println!("[System] 성공적으로 파일이 업데이트되었습니다: {}", file_path);
                                json!({"type": "export_success", "payload": file_path}).to_string()
                            },
                            Err(e) => json!({"type": "error", "message": format!("File Write Error: {}", e)}).to_string(),
                        }
                    } else {
                        json!({"type": "error", "message": "Invalid export payload format."}).to_string()
                    };
                    response_json
                } else if payload.starts_with("gemini_chat:") {
                    "[System] Gemini 서비스 비활성화됨".to_string()
                } else if payload == "check_progress" {
                    if let Some(progress) = GLOBAL_PROGRESS.lock().unwrap().clone() {
                        json!({"type": "push_progress", "payload": progress}).to_string()
                    } else {
                        json!({"type": "push_idle"}).to_string()
                    }
                } else if payload == "reset_all_data" {
                    if let Ok(_) = db::reset_all_records().await {
                        json!({"type": "reset_success", "message": "모든 데이터가 성공적으로 초기화되었습니다."}).to_string()
                    } else {
                        json!({"type": "error", "message": "데이터 초기화에 실패했습니다."}).to_string()
                    }
                } else if payload.starts_with("save_config:") {
                    let config_data = &payload["save_config:".len()..];
                    let config_path = app_dir.join("app_config.json");
                    
                    let mut current_config: serde_json::Value = std::fs::read_to_string(&config_path)
                        .ok()
                        .and_then(|s| serde_json::from_str(&s).ok())
                        .unwrap_or_else(|| json!({}));
                    
                    if let Ok(new_config) = serde_json::from_str::<serde_json::Value>(config_data) {
                        if let Some(obj) = current_config.as_object_mut() {
                            if let Some(new_obj) = new_config.as_object() {
                                for (k, v) in new_obj {
                                    obj.insert(k.clone(), v.clone());
                                }
                            }
                        }
                        if let Ok(_) = std::fs::write(&config_path, current_config.to_string()) {
                            json!({"type": "config_saved"}).to_string()
                        } else {
                            json!({"type": "error", "message": "설정 저장 실패"}).to_string()
                        }
                    } else {
                        json!({"type": "error", "message": "잘못된 JSON 페이로드"}).to_string()
                    }
                } else if payload.starts_with("rename_domain:") {
                    // 🚀 프론트엔드에서 메뉴명을 바꿀 때 발생하며, DB 전체에 저장된 해당 도메인명 기록을 전부 갱신합니다.
                    let data = &payload["rename_domain:".len()..];
                    if let Ok(req) = serde_json::from_str::<serde_json::Value>(data) {
                        if let (Some(old_dom), Some(new_dom)) = (req.get("old").and_then(|v| v.as_str()), req.get("new").and_then(|v| v.as_str())) {
                            if let Ok(drafts) = db::fetch_drafts().await {
                                let mut to_update: Vec<_> = drafts.into_iter().filter(|r| r.domain == old_dom).collect();
                                for record in &mut to_update {
                                    record.domain = new_dom.to_string();
                                }
                                let _ = db::save_records(to_update, None).await;
                            }
                        }
                    }
                    json!({"type": "domain_renamed"}).to_string()
                } else if payload.starts_with("save_prompt:") {
                    let prompt_data = &payload["save_prompt:".len()..];
                    let prompt_path = app_dir.join("prompts.json");
                    let mut prompts: Vec<String> = std::fs::read_to_string(&prompt_path)
                        .ok()
                        .and_then(|s| serde_json::from_str(&s).ok())
                        .unwrap_or_default();
                    if !prompts.contains(&prompt_data.to_string()) && !prompt_data.is_empty() {
                        prompts.push(prompt_data.to_string());
                        let _ = std::fs::write(&prompt_path, serde_json::to_string(&prompts).unwrap_or_default());
                    }
                    json!({"type": "prompts_loaded", "payload": prompts}).to_string()
                } else if payload.starts_with("delete_prompt:") {
                    let prompt_data = &payload["delete_prompt:".len()..];
                    let prompt_path = app_dir.join("prompts.json");
                    let mut prompts: Vec<String> = std::fs::read_to_string(&prompt_path)
                        .ok()
                        .and_then(|s| serde_json::from_str(&s).ok())
                        .unwrap_or_default();
                    prompts.retain(|p| p != prompt_data);
                    let _ = std::fs::write(&prompt_path, serde_json::to_string(&prompts).unwrap_or_default());
                    json!({"type": "prompts_loaded", "payload": prompts}).to_string()
                } else if payload == "fetch_prompts" {
                    let prompt_path = app_dir.join("prompts.json");
                    let prompts: Vec<String> = std::fs::read_to_string(&prompt_path)
                        .ok()
                        .and_then(|s| serde_json::from_str(&s).ok())
                        .unwrap_or_default();
                    json!({"type": "prompts_loaded", "payload": prompts}).to_string()
                } else if payload.starts_with("download_model:") {
                    let model_name = payload["download_model:".len()..].to_string();
                    let page_c = page_clone.clone();
                    let app_dir_clone = app_dir.clone(); // 🚀 복제본을 생성하여 태스크 안으로 이동시킵니다.
                    
                    // 🚀 비동기 다운로드 태스크 스폰
                    tokio::task::spawn(async move {
                        let folder_name = match model_name.as_str() {
                            "GLM-OCR" => "glm_ocr",
                            "Privacy-Filter" => "privacy-filter",
                            "Embedding" => "embeddings",
                            _ => "unknown"
                        };

                        let dir_path = app_dir_clone.join("models").join(folder_name);
                        
                        // 🚀 디렉토리 생성 중 권한 오류 발생 시 에러 반환 및 사용자 알림 연동
                        if let Err(e) = std::fs::create_dir_all(&dir_path) {
                            println!("[Error] 디렉토리 생성 실패 (권한 문제일 수 있습니다): {:?}", e);
                            let err_script = format!("window.dispatchEvent(new CustomEvent('rpc_response', {{ detail: {} }}));", json!(json!({"type": "error", "message": format!("폴더 생성 권한이 없습니다. ({}): {}", dir_path.display(), e)})));
                            let _ = page_c.evaluate(err_script).await;
                            return;
                        }

                        // 🚀 [시스템 자동 연결] 다운로드 시작 전, 시스템에 내장된 JSON 설정 파일들을 먼저 폴더에 생성합니다.
                        // 이제 사용자는 무거운 가중치(.gguf, .safetensors) 파일만 다운로드 받으면 됩니다.
                        let base_config_dir = app_dir_clone.clone();
                        let m_name_for_config = model_name.clone();
                        
                        // JSON 파일들 복원
                        let glm_c = include_str!("../models/glm_ocr/config.json");
                        let glm_t = include_str!("../models/glm_ocr/tokenizer_config.json");
                        let glm_tok = include_str!("../models/glm_ocr/tokenizer.json"); // 🚀 추가
                        let glm_prep = include_str!("../models/glm_ocr/preprocessor_config.json"); // 🚀 추가
                        let priv_c = include_str!("../models/privacy-filter/config.json");
                        let priv_t = include_str!("../models/privacy-filter/tokenizer_config.json");
                        let priv_tok = include_str!("../models/privacy-filter/tokenizer.json"); // 🚀 추가
                        let emb_c = include_str!("../models/embeddings/config.json");
                        let emb_tok = include_str!("../models/embeddings/tokenizer.json"); // 🚀 추가

                        let target_base = base_config_dir.join("models").join(folder_name);
                        let _ = std::fs::create_dir_all(&target_base);
                        match m_name_for_config.as_str() {
                            "GLM-OCR" => {
                                let _ = std::fs::write(target_base.join("config.json"), glm_c);
                                let _ = std::fs::write(target_base.join("tokenizer_config.json"), glm_t);
                                let _ = std::fs::write(target_base.join("tokenizer.json"), glm_tok); // 🚀 추가
                                let _ = std::fs::write(target_base.join("preprocessor_config.json"), glm_prep); // 🚀 추가
                            },
                            "Privacy-Filter" => {
                                let _ = std::fs::write(target_base.join("config.json"), priv_c);
                                let _ = std::fs::write(target_base.join("tokenizer_config.json"), priv_t);
                                let _ = std::fs::write(target_base.join("tokenizer.json"), priv_tok); // 🚀 추가
                            },
                            "Embedding" => {
                                let _ = std::fs::write(target_base.join("config.json"), emb_c);
                                let _ = std::fs::write(target_base.join("tokenizer.json"), emb_tok); // 🚀 추가
                            },
                            _ => (),
                        }

                        // 🚀 실제 Hugging Face 다운로드 URL 리스트 매핑 (가중치 파일만 받도록 최소화)
                        let files_to_download = match model_name.as_str() {
                            "GLM-OCR" => vec![
                                ("https://huggingface.co/ggml-org/GLM-OCR-GGUF/resolve/main/GLM-OCR-Q8_0.gguf", "GLM-OCR-Q8_0.gguf"),
                                ("https://huggingface.co/ggml-org/GLM-OCR-GGUF/resolve/main/mmproj-GLM-OCR-Q8_0.gguf", "mmproj-GLM-OCR-Q8_0.gguf"),
                            ],
                            "Privacy-Filter" => vec![
                                ("https://huggingface.co/OpenMed/privacy-filter-multilingual/resolve/main/model.safetensors", "model.safetensors"),
                            ],
                            "Embedding" => vec![
                                ("https://huggingface.co/unsloth/embeddinggemma-300m-GGUF/resolve/main/embeddinggemma-300m-Q4_0.gguf", "embeddinggemma-300m-Q4_0.gguf"),
                            ],
                            _ => vec![]
                        };

                        if files_to_download.is_empty() {
                            let err_script = format!("window.dispatchEvent(new CustomEvent('rpc_response', {{ detail: {} }}));", json!(json!({"type": "error", "message": format!("다운로드할 파일 목록이 없습니다: {}", model_name)})));
                            let _ = page_c.evaluate(err_script).await;
                            return;
                        }

                        let total_files = files_to_download.len();
                        let client = reqwest::Client::new();
                        let mut has_error = false;

                        for (file_idx, (url, filename)) in files_to_download.iter().enumerate() {
                            let file_path = dir_path.join(filename);
                            let tmp_path = dir_path.join(format!("{}.tmp", filename));
                            
                            // 🚀 [손상 파일 스킵 방지] 가중치 파일(.gguf, .safetensors)은 최소 10MB 이상일 때만 정상으로 인정하여 스킵합니다.
                            // 만약 다운로드가 끊긴 0~9MB짜리 파일이 있다면 스킵하지 않고 처음부터 다시(tmp로) 덮어씌웁니다.
                            let min_size = if filename.ends_with(".gguf") || filename.ends_with(".safetensors") { 10_000_000 } else { 0 };
                            if file_path.exists() && std::fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0) > min_size {
                                let percent = (((file_idx as f64 + 1.0) / total_files as f64) * 100.0) as u32;
                                let progress_script = format!("window.dispatchEvent(new CustomEvent('rpc_response', {{ detail: {} }}));", json!(json!({"type": "download_progress", "model": model_name, "percent": percent})));
                                let _ = page_c.evaluate(progress_script).await;
                                continue;
                            }

                            match client.get(*url).send().await {
                                Ok(res) => {
                                    if !res.status().is_success() {
                                        let err_msg = format!("파일 다운로드 실패 (HTTP {}): {}", res.status(), url);
                                        println!("[Error] {}", err_msg);
                                        let err_script = format!("window.dispatchEvent(new CustomEvent('rpc_response', {{ detail: {} }}));", json!(json!({"type": "error", "message": err_msg})));
                                        let _ = page_c.evaluate(err_script).await;
                                        has_error = true;
                                        break;
                                    }
                                    
                                    let total_size = res.content_length().unwrap_or(0) as f64;
                                    let mut downloaded = 0.0;
                                    
                                    // 🚀 [원자적 다운로드] 불완전 다운로드 문제를 막기 위해 .tmp 파일로 먼저 기록합니다.
                                    match tokio::fs::File::create(&tmp_path).await {
                                        Ok(mut file) => {
                                            use tokio::io::AsyncWriteExt;
                                            let mut stream = res.bytes_stream();
                                            let mut write_error = false;
                                            
                                            while let Some(chunk_result) = stream.next().await {
                                                match chunk_result {
                                                    Ok(chunk) => {
                                                        if let Err(e) = file.write_all(&chunk).await {
                                                            let err_msg = format!("파일 쓰기 실패 (디스크 용량 부족 또는 권한 문제): {:?}", e);
                                                            println!("[Error] {}", err_msg);
                                                            let err_script = format!("window.dispatchEvent(new CustomEvent('rpc_response', {{ detail: {} }}));", json!(json!({"type": "error", "message": err_msg})));
                                                            let _ = page_c.evaluate(err_script).await;
                                                            write_error = true;
                                                            break;
                                                        }
                                                        downloaded += chunk.len() as f64;
                                                        
                                                        // 전체 파일 개수 대비 현재 파일의 다운로드 퍼센트 계산
                                                        let file_progress = if total_size > 0.0 { downloaded / total_size } else { 0.0 };
                                                        let percent = (((file_idx as f64 + file_progress) / total_files as f64) * 100.0) as u32;
                                                        
                                                        let progress_script = format!("window.dispatchEvent(new CustomEvent('rpc_response', {{ detail: {} }}));", json!(json!({"type": "download_progress", "model": model_name, "percent": percent})));
                                                        let _ = page_c.evaluate(progress_script).await;
                                                    },
                                                    Err(e) => {
                                                        let err_msg = format!("네트워크 스트림 읽기 실패 (인터넷 끊김 등): {:?}", e);
                                                        println!("[Error] {}", err_msg);
                                                        let err_script = format!("window.dispatchEvent(new CustomEvent('rpc_response', {{ detail: {} }}));", json!(json!({"type": "error", "message": err_msg})));
                                                        let _ = page_c.evaluate(err_script).await;
                                                        write_error = true;
                                                        break;
                                                    }
                                                }
                                            }
                                            
                                            if write_error {
                                                // 🚀 에러 발생 시 쓰다 만 tmp 파일을 제거합니다.
                                                let _ = std::fs::remove_file(&tmp_path);
                                                has_error = true;
                                                break;
                                            } else {
                                                // 🚀 성공적으로 다운로드가 끝났을 때만 실제 파일명으로 덮어씌웁니다. (이 시점에 찌꺼기 파일이 완벽하게 초기화 복구됩니다)
                                                let _ = std::fs::rename(&tmp_path, &file_path);
                                            }
                                        },
                                        Err(e) => {
                                            let err_msg = format!("파일 생성 실패 ({} 권한을 확인해주세요): {:?}", tmp_path.display(), e);
                                            println!("[Error] {}", err_msg);
                                            let err_script = format!("window.dispatchEvent(new CustomEvent('rpc_response', {{ detail: {} }}));", json!(json!({"type": "error", "message": err_msg})));
                                            let _ = page_c.evaluate(err_script).await;
                                            has_error = true;
                                            break;
                                        }
                                    }
                                },
                                Err(e) => {
                                    let err_msg = format!("네트워크 연결 실패 (인터넷 확인 필요): {:?}", e);
                                    println!("[Error] {}", err_msg);
                                    let err_script = format!("window.dispatchEvent(new CustomEvent('rpc_response', {{ detail: {} }}));", json!(json!({"type": "error", "message": err_msg})));
                                    let _ = page_c.evaluate(err_script).await;
                                    has_error = true;
                                    break;
                                }
                            }
                            if has_error { break; }
                        }
                        
                        if !has_error {
                            let complete_script = format!("window.dispatchEvent(new CustomEvent('rpc_response', {{ detail: {} }}));", json!(json!({"type": "download_complete", "model": model_name})));
                            let _ = page_c.evaluate(complete_script).await;
                        }
                    });
                    
                    json!({"type": "download_started"}).to_string()
                } else if payload == "delete_all_models" {
                    // 🚀 OS의 파일 시스템을 통해 models 디렉토리 전체를 강제로 삭제합니다.
                    let models_dir = app_dir.join("models");
                    if models_dir.exists() {
                        if let Err(e) = std::fs::remove_dir_all(&models_dir) {
                            println!("[Error] 모델 디렉토리 삭제 실패: {:?}", e);
                            json!({"type": "error", "message": format!("디렉토리를 삭제할 수 없습니다: {}", e)}).to_string()
                        } else {
                            println!("[System] 모델 디렉토리 전체 삭제 성공");
                            json!({"type": "delete_models_success"}).to_string()
                        }
                    } else {
                        // 애초에 폴더가 없으면 성공으로 간주
                        json!({"type": "delete_models_success"}).to_string()
                    }
                } else { "Unknown command".to_string() };

                let script = format!("window.dispatchEvent(new CustomEvent('rpc_response', {{ detail: {} }}));", json!(response));
                
                // 🚀 특정 탭(page_clone)에만 응답을 보내던 것을, 현재 열려있는 모든 브라우저 탭에 브로드캐스트하여 실시간 동기화 문제를 원천 해결합니다.
                if let Ok(pages) = _browser_clone.pages().await {
                    for p in pages {
                        let _ = p.evaluate(script.clone()).await;
                    }
                }
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
        _ = tokio::signal::ctrl_c() => {
            println!("\n[System] 앱 종료 신호(Ctrl+C) 감지. 크롬 브라우저 프로세스를 함께 종료합니다...");
            // 🚀 앱이 종료될 때 OS 레벨에서 강제 종료를 호출하여 자식 프로세스인 Chromium도 완벽하게 정리되도록 합니다.
            std::process::exit(0);
        },
        _ = rx.recv() => {
            // 🚀 크롬 브라우저의 'X' 버튼을 눌러 모든 창이 닫히면 채널 핸들러가 이를 감지하여 이곳이 실행됩니다.
            println!("\n[System] 크롬 브라우저가 종료되었습니다. 앱을 안전하게 종료합니다...");
            std::process::exit(0);
        },
    }
}
