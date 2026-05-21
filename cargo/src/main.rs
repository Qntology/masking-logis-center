use gemini_gui_lib::{db, privacy_filter, glm_ocr, params}; // embedding 제거
use privacy_filter::viterbi::PrivacySpan; // PrivacyFilterModel 제거
use candle_core::Device;
use glm_ocr::generate::{GlmOcrGenerateModel, GenerateModel};
use params::chat::{ChatCompletionParameters, Message, Part};
use std::sync::Mutex;
use lazy_static::lazy_static;

lazy_static! {
    static ref OCR_MODEL: Mutex<Option<GlmOcrGenerateModel>> = Mutex::new(None);
    static ref PRIVACY_MANAGER: Mutex<Option<gemini_gui_lib::privacy_filter::masking::PrivacyManager>> = Mutex::new(None);
    static ref EMBEDDING_MODEL: Mutex<Option<gemini_gui_lib::embedding::EmbeddingModel>> = Mutex::new(None);
    static ref GLOBAL_PROGRESS: Mutex<Option<serde_json::Value>> = Mutex::new(None);
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
    unsafe { extern "C" { fn malloc_trim(pad: usize) -> i32; } malloc_trim(0); }
    #[cfg(target_os = "macos")]
    unsafe { extern "C" { fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize; } malloc_zone_pressure_relief(std::ptr::null_mut(), 0); }
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

    // 🚀 복잡하고 실패 확률이 높은 커스텀 파서를 제거하고,
    // 브라우저가 직접 렌더링된 텍스트만 깔끔하게 추출하는 innerText를 활용합니다.
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
            .item-row-wrapper { display: flex; flex-direction: column; border-bottom: 1px solid #f0f0f0; }
            .item-row { display: flex; align-items: center; gap: 10px; padding: 8px; cursor: pointer; transition: background 0.2s; }
            .item-row:hover { background: #f9f9f9; }
            .item-row input[type="checkbox"] { cursor: pointer; }
            .item-detail { display: none; padding: 10px 10px 10px 32px; font-size: 11px; color: #555; background: #fafafa; border-top: 1px dashed #eee; white-space: pre-wrap; max-height: 200px; overflow-y: auto; word-break: break-all; }
            .item-detail.open { display: block; }
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
        
        const header = document.createElement('header');
        const headerLeft = document.createElement('div');
        headerLeft.className = 'header-left';

        // 🚀 UI가 다시 그려질 때 버튼 텍스트가 덮어씌워지는 것을 막는 핵심 락(Lock) 변수
        let isProcessing = false;

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

        // Delete 버튼 생성 (초기에는 숨김 처리)
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'btn-delete';
        deleteBtn.textContent = 'Delete';
        deleteBtn.style.display = 'none'; 
        deleteBtn.onclick = () => {
            if (isProcessing) return;
            const checkedBoxes = log.querySelectorAll('.item-checkbox:checked');
            const selectedIds = Array.from(checkedBoxes).map(cb => cb.dataset.id);
            if (window.rpc) {
                window.rpc("delete_drafts:" + JSON.stringify(selectedIds));
            }
            // 선택된 아이템을 UI 데이터 배열에서 즉시 제거하고 화면을 갱신합니다.
            stagedItems = stagedItems.filter(i => !selectedIds.includes(i.id));
            updateGnbUI();
            renderStagedList();
        };

        const draftBtn = document.createElement('button');
        draftBtn.textContent = 'Draft (0)';
        draftBtn.onclick = async () => {
            if (isProcessing) return;
            const pageId = await generatePageId(window.location.href);
            const extractedText = extractVisibleText();
            const item = { 
                id: pageId, 
                host: window.location.host,
                url: window.location.href,
                title: getPageMeta(), 
                domain: currentTabFilter, // 현재 GNB 탭의 도메인으로 매핑
                context: extractedText, 
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
            // 🚀 백엔드가 작업 중일 때는 UI 갱신을 멈추고 버튼 비활성화를 유지합니다.
            if (isProcessing) {
                pushBtn.disabled = true;
                deleteBtn.disabled = true;
                draftBtn.disabled = true;
                return;
            }

            const checkedBoxes = Array.from(log.querySelectorAll('.item-checkbox:checked'));
            const checkedCount = checkedBoxes.length;
            
            // 상태별 개수 카운트
            const selectedIds = checkedBoxes.map(cb => cb.dataset.id);
            const draftCount = stagedItems.filter(i => selectedIds.includes(i.id) && i.status !== 'PUSHED').length;
            
            pushBtn.disabled = (draftCount === 0);
            pushBtn.textContent = `Push (${draftCount})`; // 완료되지 않은 DRAFT 항목만 Push 카운트에 반영
            
            // 체크된 항목이 1개 이상일 경우 삭제 버튼을 노출합니다.
            deleteBtn.style.display = (checkedCount > 0) ? 'inline-block' : 'none';
            
            const totalCount = log.querySelectorAll('.item-checkbox').length;
            selectAllCheck.checked = (totalCount > 0 && checkedCount === totalCount);

            // 체크박스 상태가 변경될 때 하단(Footer)의 Drag 버튼 상태도 즉시 갱신합니다.
            if (typeof updateSubmitToDrag === 'function') {
                updateSubmitToDrag();
            }
        }

        // 🚀 스피너 구동 로직을 독립적인 함수로 분리하여 어떤 탭에서든 즉시 실행되게 만듭니다.
        let spinnerInterval = null;
        
        function startPushSpinner() {
            if (spinnerInterval) return; // 이미 동작 중이면 무시
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
            if (isProcessing) return;
            
            const checkedBoxes = Array.from(log.querySelectorAll('.item-checkbox:checked'));
            const selectedIds = checkedBoxes.map(cb => cb.dataset.id);
            const draftIds = stagedItems.filter(i => selectedIds.includes(i.id) && i.status !== 'PUSHED').map(i => i.id);
            
            if (draftIds.length === 0) return;
            
            isProcessing = true;
            startPushSpinner(); // 스피너 강제 기동
            
            if (window.rpc) {
                window.rpc("mask_and_push_batch:" + JSON.stringify({ 
                    ids: draftIds,
                    host: window.location.host 
                }));
            }
        };

        actionContainer.appendChild(deleteBtn); // Draft 버튼의 왼쪽에 위치하도록 먼저 append
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

        // Footer UI 생성 (이미지, 텍스트 입력 폼 및 Drag & Drop Export)
        const footer = document.createElement('footer');
        
        const fileInputWrapper = document.createElement('div');
        fileInputWrapper.style.position = 'relative';
        fileInputWrapper.style.width = '140px';
        fileInputWrapper.style.height = '30px';
        fileInputWrapper.style.display = 'none'; // 파일 인풋 숨김 처리
        fileInputWrapper.style.alignItems = 'center';
        fileInputWrapper.style.overflow = 'hidden';

        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = 'image/*, application/pdf, text/csv';
        fileInput.style.width = '100%';
        fileInput.style.fontSize = '12px';
        fileInput.style.cursor = 'pointer';

        // 파일 처리 중 보여줄 스피너 영역
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
        
        const submitBtn = document.createElement('button');
        submitBtn.textContent = 'Submit';
        
        // 상태 저장용 변수들
        let processedFileContent = '';
        let fileSpinnerInterval = null;

        // 프롬프트, 파일 또는 체크된 항목이 준비되면 Submit을 Drag 버튼으로 전환
        function updateSubmitToDrag() {
            const checkedBoxes = Array.from(log.querySelectorAll('.item-checkbox:checked'));
            const selectedIds = checkedBoxes.map(cb => cb.dataset.id);
            // 완료된(PUSHED) 항목만 Drag 카운트에 반영
            const pushedCount = stagedItems.filter(i => selectedIds.includes(i.id) && i.status === 'PUSHED').length;
            
            if (pushedCount > 0) {
                submitBtn.textContent = `Drag (${pushedCount})`; // 완료된 카운트 반영
                submitBtn.draggable = true;
            } else if (processedFileContent !== '' || textInput.value.trim() !== '') {
                submitBtn.textContent = 'Drag';
                submitBtn.draggable = true;
            } else {
                submitBtn.textContent = 'Submit';
                submitBtn.draggable = false;
            }
        }

        textInput.oninput = updateSubmitToDrag;

        // 파일 선택 시 스피너 동작 및 백엔드로 처리 요청 (기존 로직 유지하되 UI 숨김상태)
        fileInput.onchange = (e) => {
            const file = e.target.files[0];
            if (!file) return;

            // 🚀 CSV 파일인 경우 무거운 OCR 엔진을 거치지 않고, 텍스트를 바로 파싱하여 Draft 대기열로 밀어넣습니다.
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
                        alert(file.name + " 파일이 " + currentTabFilter + " 대기열에 추가되었습니다.");
                    }
                };
                reader.readAsText(file); // OCR용 DataURL이 아닌 순수 텍스트로 읽습니다.
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

        // Drag 이벤트 발생 시 선택된 항목들과 텍스트를 파일로 바인딩하여 바탕화면으로 내보내기
        submitBtn.ondragstart = (e) => {
            if (!submitBtn.textContent.startsWith('Drag')) {
                e.preventDefault();
                return;
            }
            const checkedBoxes = Array.from(log.querySelectorAll('.item-checkbox:checked'));
            const selectedIds = checkedBoxes.map(cb => cb.dataset.id);
            // 드래그 시에는 완료된(PUSHED) 항목만 추출
            const selectedItems = stagedItems.filter(i => selectedIds.includes(i.id) && i.status === 'PUSHED');

            let exportContent = `[Prompt]\n${textInput.value || 'N/A'}\n\n`;
            if (processedFileContent) {
                exportContent += `[File Masked Text]\n${processedFileContent}\n\n`;
            }
            exportContent += `[Selected Items (${selectedItems.length})]\n`;
            
            selectedItems.forEach(item => {
                exportContent += `\n--- ID: ${item.id} ---\n[Domain]: ${item.domain}\n[Title]: ${item.title}\n[Content]:\n${item.masking || item.context}\n`;
            });

            // 🚀 브라우저 보안 정책상 로컬 경로(temp)를 직접 참조할 수는 없으므로,
            // 메모리 상에서 즉시 File 객체를 생성하여 드래그 이벤트에 주입합니다.
            // 이렇게 하면 다른 웹 페이지의 파일 업로드/드롭존에서 실제 파일로 인식하게 됩니다.
            const file = new File([exportContent], "export.txt", { type: "text/plain" });
            e.dataTransfer.items.add(file);

            // 기존 일반 텍스트 데이터 유지 (텍스트 입력창 드롭 대비)
            e.dataTransfer.setData('text/plain', exportContent);

            // OS 바탕화면 등 외부로 드롭할 경우를 위해 DownloadURL 방식도 함께 유지
            const utf8Bytes = new TextEncoder().encode(exportContent);
            let binary = '';
            for (let i = 0; i < utf8Bytes.length; i++) {
                binary += String.fromCharCode(utf8Bytes[i]);
            }
            const base64Str = btoa(binary);
            e.dataTransfer.setData('DownloadURL', `text/plain:export.txt:data:text/plain;base64,${base64Str}`);
        };

        footer.appendChild(fileInputWrapper);
        footer.appendChild(textInput);
        footer.appendChild(submitBtn);

        agentContainer.appendChild(header);
        agentContainer.appendChild(mainLayout);
        agentContainer.appendChild(footer); // 생성한 footer 영역 마운트
        shadow.appendChild(style);
        shadow.appendChild(agentContainer);

        // agentContainer 영역에 Drag & Drop 시 Draft Item 추가 로직
        agentContainer.addEventListener('dragover', (e) => {
            e.preventDefault();
            agentContainer.style.border = '2px dashed #007bff';
        });
        agentContainer.addEventListener('dragleave', (e) => {
            e.preventDefault();
            agentContainer.style.border = 'none';
        });
        agentContainer.addEventListener('drop', (e) => {
            e.preventDefault();
            agentContainer.style.border = 'none';
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                const reader = new FileReader();
                reader.onload = async (ev) => {
                    const dataUrl = ev.target.result;
                    const pageId = await generatePageId(dataUrl + Date.now());
                    const item = { 
                        id: pageId, 
                        host: window.location.host,
                        url: file.name,
                        title: `[File] ${file.name}`, 
                        domain: currentTabFilter, // 현재 활성화된 도메인으로 등록
                        context: dataUrl, 
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
                reader.readAsDataURL(file);
            }
        });

        function renderStagedList() {
            log.replaceChildren(); 
            // 현재 선택된 도메인에 해당하는 항목만 엄격하게 필터링
            const filtered = stagedItems.filter(i => i.domain === currentTabFilter);
            const draftOnlyCount = filtered.filter(i => i.status !== 'PUSHED').length;
            
            // 🚀 백엔드 처리 중이 아닐 때만 텍스트를 Draft 형태로 갱신합니다.
            if (!isProcessing) {
                draftBtn.textContent = `Draft (${draftOnlyCount})`;
            }
            
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
                    const wrapper = document.createElement('div');
                    wrapper.className = 'item-row-wrapper';

                    const row = document.createElement('div');
                    row.className = 'item-row';

                    const cb = document.createElement('input');
                    cb.type = 'checkbox';
                    cb.className = 'item-checkbox';
                    cb.dataset.id = item.id;
                    cb.onclick = (e) => {
                        e.stopPropagation(); // 🚀 체크박스 클릭 시 아코디언이 열리는 것을 방지합니다.
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
                    const statusBadge = item.status === 'PUSHED' ? '✅ [PUSHED] ' : '';
                    titleSpan.textContent = `${statusBadge}[${item.domain}] ${mainTitle}`;
                    titleSpan.style.fontSize = '13px';
                    titleSpan.style.fontWeight = 'bold';
                    titleSpan.style.color = item.status === 'PUSHED' ? '#28a745' : '#000';
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

                    // 🚀 전처리 결과를 보여줄 아코디언 상세 뷰를 생성합니다.
                    const detailView = document.createElement('div');
                    detailView.className = 'item-detail';
                    detailView.textContent = item.masking || item.context || '전처리된 텍스트 결과가 없습니다.';

                    // 🚀 row 클릭 시 아코디언 토글 이벤트를 바인딩합니다.
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
            // 🚀 SPA(React/Vue 등) 웹페이지의 DOM 렌더링이 완료될 시간을 주기 위해 1.5초 지연 후 텍스트를 추출합니다.
            setTimeout(async () => {
                const pageId = await generatePageId(window.location.href);
                const extractedText = extractVisibleText();
                const item = { 
                    id: pageId, 
                    host: window.location.host,
                    url: window.location.href,
                    title: getPageMeta(), 
                    domain: 'COMMERCE', 
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

        // 🚀 유저가 다른 탭에 있다가 화면으로 돌아올 때 즉시 최신 진행 상태를 물어보고 동기화합니다.
        window.addEventListener('focus', () => {
            if (window.rpc) {
                window.rpc("fetch_drafts");
                window.rpc("check_progress");
            }
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
                // 🚀 진행 상황 (퍼센트) 실시간 업데이트 응답인 경우
                else if (data.type === 'push_progress') {
                    if (!isProcessing) {
                        isProcessing = true; // 강제 락 활성화
                        startPushSpinner(); // 다른 탭에서도 즉시 스피너 가동
                    }
                    draftBtn.textContent = `Draft (${data.payload.item_display}/${data.payload.total_items}) ${data.payload.percent}%...`;
                    updatePushBtnState(); // 버튼 비활성화 갱신
                    return;
                }
                // 🚀 새로고침/포커스 시 백엔드가 작업 중이지 않음을 확인한 경우
                else if (data.type === 'push_idle') {
                    if (isProcessing) {
                        isProcessing = false;
                        stopPushSpinner();
                        updatePushBtnState();
                        renderStagedList();
                    }
                    return;
                }
                // 마스킹 및 Push 대기열 작업 완료 응답인 경우
                else if (data.type === 'push_success') {
                    isProcessing = false; // 작업 완료 시 락 해제
                    stopPushSpinner();
                    deleteBtn.disabled = false;
                    draftBtn.disabled = false;
                    
                    const updatedItems = data.payload || [];
                    const updatedIds = updatedItems.map(i => i.id);
                    
                    // 삭제하지 않고 최신(마스킹 및 벡터화 완료) 데이터로 교체하여 목록 유지
                    stagedItems = stagedItems.filter(i => !updatedIds.includes(i.id));
                    stagedItems.push(...updatedItems);
                    
                    updateGnbUI();
                    renderStagedList();

                    // 성공 시스템 로그
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
                // 파일 처리 (OCR + Masking) 완료 응답인 경우
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
                // 프로세스 중 에러가 발생한 경우
                else if (data.type === 'error') {
                    isProcessing = false; // 에러 시 락 해제
                    stopPushSpinner();
                    if (fileSpinnerInterval) {
                        clearInterval(fileSpinnerInterval);
                        fileSpinner.style.display = 'none';
                        fileInput.style.display = 'block';
                    }
                    deleteBtn.disabled = false;
                    draftBtn.disabled = false;
                    updatePushBtnState(); // 버튼 텍스트와 상태를 원래대로 원복
                    renderStagedList(); // 퍼센트 텍스트로 변한 버튼을 다시 "Draft (N)" 형태로 롤백합니다.
                    
                    const div = document.createElement('div');
                    div.className = 'system';
                    div.style.padding = '10px';
                    div.style.background = '#ffe6e6';
                    div.style.color = '#d8000c';
                    div.style.borderRadius = '4px';
                    div.textContent = 'Error: ' + data.message;
                    log.appendChild(div);
                    div.scrollIntoView({ behavior: 'smooth', block: 'end' });
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
        
        // 🚀 새로고침/새 탭 오픈 시 데이터를 불러오고, 백엔드에 진행 중인 작업이 있는지(check_progress) 확인합니다.
        setTimeout(() => {
            if (window.rpc) {
                window.rpc("fetch_drafts");
                window.rpc("check_progress");
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
    let default_tab = load_default_tab();
    
    // OVERLAY_SCRIPT 내의 예측 가능한 전역 변수(window.rpc 등)를 랜덤 생성한 이름으로 동적 치환
    let overlay_script_replaced = OVERLAY_SCRIPT
        .replace("window.rpc", &format!("window.{}", rpc_binding_name))
        .replace("window.geminiSidebarLoaded", &format!("window.{}", sidebar_var_name));

    let full_script = format!("window.is_authenticated = {};\nwindow.default_tab = \"{}\";\n{}", is_authenticated, default_tab, overlay_script_replaced);
    // 페이지가 새로고침되거나 다른 페이지로 이동하더라도 스크립트가 유지되도록 등록합니다.
    let _ = page.execute(AddScriptToEvaluateOnNewDocumentParams::new(&full_script)).await;
    let _ = page.evaluate(full_script).await;
    let mut bindings = page.event_listener::<EventBindingCalled>().await?;
    let page_clone = page.clone();
    let _browser_clone = browser.clone();
    
    tokio::task::spawn(async move {
        while let Some(event) = bindings.next().await {
            if event.name == rpc_binding_name { // 이벤트 수신명 변경
                let payload = event.payload.trim_matches('"').to_string();
                let response = if payload.starts_with("sync_data:") {
                    let data = &payload["sync_data:".len()..];
                    match serde_json::from_str::<db::CommerceRecord>(data) {
                        Ok(mut record) => {
                            // 🚀 [Fix] 미사용 트레이트 Harness 임포트를 제거하고 DefaultHarness 구조체만 사용합니다.
                            use gemini_gui_lib::harness::DefaultHarness;
                            let harness = DefaultHarness;
                            
                            if record.url.starts_with("file://") && record.context.contains("data:image/") {
                                if let Some(base64_part) = record.context.split("data:").nth(1) {
                                    let full_data_url = format!("data:{}", base64_part.trim());
                                    let ocr_result = {
                                        let mut model_guard = OCR_MODEL.lock().unwrap();
                                        if model_guard.is_none() {
                                            let model_path = "..\\models\\glm_ocr";
                                            // 🚀 SSD 오프로딩이 적용되었으므로 당당하게 4GB CUDA VRAM을 100% 활용합니다.
                                            let ocr_device = Device::new_cuda(0).unwrap_or(Device::Cpu); 
                                            if let Ok(model) = GlmOcrGenerateModel::init(model_path, Some(&ocr_device), None) {
                                                *model_guard = Some(model);
                                            }
                                        }
                                        let res = if let Some(model) = model_guard.as_mut() {
                                            let params = ChatCompletionParameters {
                                                messages: vec![Message {
                                                    role: "user".to_string(),
                                                    parts: vec![Part { text: "Extract text".to_string(), image_url: Some(full_data_url.to_string()) }],
                                                }],
                                                model: "glm-ocr".to_string(),
                                                max_tokens: Some(2048),
                                                temperature: Some(0.2),
                                                top_p: Some(0.95),
                                                top_k: None, repeat_penalty: Some(1.2), repeat_last_n: Some(64), seed: Some(42),
                                            };
                                            model.generate(params).map(|res| res.choices[0].message.content.clone()).unwrap_or_default()
                                        } else { "OCR Model Load Error".to_string() };
                                        
                                        // ★ VRAM 해제: 다음 작업을 위해 OCR 모델 메모리를 즉시 반환합니다.
                                        *model_guard = None;
                                        res
                                    };
                                    // 🚀 OCR 결과물에서도 마크다운 껍데기를 제거하여 가독성을 높입니다.
                                    let display_ocr = ocr_result.replace("```markdown", "").replace("```", "").trim().to_string();
                                    record.context = format!("{}\n---\n[OCR 결과]\n{}", record.context, display_ocr);
                                } 
                            }
                            // 🚀 [Fix] 동기화 시점에 로그를 남겨 어떤 아이템이 들어오는지 명확히 합니다.
                            println!("[System] 동기화 수신: {} (ID: {})", record.title, record.id);

                            // 🚀 [Fix] 텍스트가 비어있지 않고 실제 HTML 태그를 포함한 경우에만 평탄화를 수행하여 데이터 증발을 원천 차단합니다.
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
                    }
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
                    let data = &payload["mask_and_push_batch:".len()..];
                    let response_json = if let Ok(req) = serde_json::from_str::<serde_json::Value>(data) {
                        if let Some(ids) = req.get("ids").and_then(|i| i.as_array()) {
                            let id_strings: Vec<String> = ids.iter().filter_map(|i| i.as_str().map(String::from)).collect();
                            
                            let mut target_records = Vec::new();
                            if let Ok(drafts) = db::fetch_drafts().await {
                                target_records = drafts.into_iter().filter(|r| id_strings.contains(&r.id)).collect();
                            }
                            
                            if let Ok(table) = db::get_or_create_table().await {
                                let mut has_error = None;
                                // 현재 가용한 최적의 장치(CUDA 0번 우선)를 할당합니다.
                                let device = Device::new_cuda(0).unwrap_or(Device::Cpu);

                                // ★ 전체 처리율 계산을 위한 단계 설정 (Phase 0, 1, 2)
                                let total_items = target_records.len();
                                let total_steps = total_items * 3;
                                let mut current_step = 0;

                                // ★ Phase 0: OCR 일괄 처리 및 VRAM 해제
                                let needs_ocr = target_records.iter().any(|r| r.context.starts_with("data:image/") || r.context.starts_with("data:application/pdf"));
                                
                                // 🚀 [Phase 0-1] 웹페이지 텍스트 정제 (데이터 증발 방지 로직 적용)
                                use gemini_gui_lib::harness::DefaultHarness;
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
                                        
                                        let text_len = record.context.len();
                                        let preview_text: String = record.context.chars().take(100).collect();
                                        println!("[Debug:Phase0-1] 웹페이지 텍스트 미리보기 (총 {}바이트) :\n{}", text_len, preview_text);
                                    }
                                }

                                // 🚀 [Phase 0-2] 이미지/PDF 전용 OCR 처리
                                if needs_ocr {
                                    {
                                        let mut model_guard = OCR_MODEL.lock().unwrap();
                                        if model_guard.is_none() {
                                            let model_path = "..\\models\\glm_ocr";
                                            match GlmOcrGenerateModel::init(model_path, Some(&device), None) {
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
                                        let is_image = record.context.starts_with("data:image/") || record.context.starts_with("data:application/pdf");
                                        let item_type = if is_image { "이미지" } else { "텍스트" };
                                        println!("[System] Phase 0 - [{}]순서-{} 처리 시작 (ID: {})", idx, item_type, record.id);
                                        
                                        if is_image && has_error.is_none() {
                                            let mut ocr_success = false;
                                            let mut cleaned_ocr = String::new();
                                            {
                                                let mut model_guard = OCR_MODEL.lock().unwrap();
                                                if let Some(model) = model_guard.as_mut() {
                                                    let params = ChatCompletionParameters {
                                                        messages: vec![Message {
                                                            role: "user".to_string(),
                                                            parts: vec![Part { text: "Extract text from image".to_string(), image_url: Some(record.context.clone()) }],
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
                                                            let err_msg = format!("OCR Error: {}", e);
                                                            println!("[Error] 이미지 OCR 중 예외 발생: {:?}", e);
                                                            has_error = Some(err_msg);
                                                        }
                                                    }
                                                }
                                            }
                                            if ocr_success {
                                                record.context = cleaned_ocr.clone();
                                                println!("[System] 이미지 OCR 완료 및 마크다운 태그 정제됨. (ID: {})", record.id);
                                                println!("[GlmOcr] 배치 작업 생성된 텍스트 결과:\n{}", cleaned_ocr);
                                            }
                                        }
                                        
                                        // 🚀 실시간 퍼센트 전송 및 브로드캐스트
                                        current_step += 1;
                                        let percent = (current_step as f64 / total_steps as f64 * 100.0) as usize;
                                        let payload = json!({"item_display": idx + 1, "total_items": total_items, "percent": percent});
                                        *GLOBAL_PROGRESS.lock().unwrap() = Some(payload.clone());
                                        let script = format!("window.dispatchEvent(new CustomEvent('rpc_response', {{ detail: {} }}));", json!(json!({"type": "push_progress", "payload": payload})));
                                        if let Ok(pages) = _browser_clone.pages().await { for p in pages { let _ = p.evaluate(script.clone()).await; } }
                                    }

                                    {
                                        let mut model_guard = OCR_MODEL.lock().unwrap();
                                        *model_guard = None; // VRAM 완전 해제
                                    }
                                    force_memory_cleanup();
                                    tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
                                    println!("[System] === Phase 0: 이미지 전용 OCR 처리 종료 (VRAM 해제됨) ===\n");
                                } else {
                                    current_step += total_items; // OCR 패스 시 퍼센트 점프
                                }

                                // ★ Phase 1: Privacy Filter 일괄 마스킹 및 VRAM 해제
                                if has_error.is_none() {
                                    println!("\n[System] === Phase 1: Privacy Filter 마스킹 시작 ===");
                                    {
                                        let mut pm_guard = PRIVACY_MANAGER.lock().unwrap();
                                        if pm_guard.is_none() {
                                            println!("[System] PrivacyManager 모델을 GPU 메모리에 로드 중...");
                                            *pm_guard = gemini_gui_lib::privacy_filter::masking::PrivacyManager::new("..\\models\\privacy-filter", &device).ok();
                                        }
                                    }
                                    
                                    for (idx, record) in target_records.iter_mut().enumerate() {
                                        let mut masked_success = false;
                                        let mut masked_text = String::new();
                                        {
                                            let pm_guard = PRIVACY_MANAGER.lock().unwrap();
                                            if let Some(pm) = pm_guard.as_ref() {
                                                println!("[System] 마스킹 진행 중 (Record ID: {})", record.id);
                                                masked_text = pm.mask_text(&record.context).unwrap_or_else(|e| {
                                                    let err_str = format!("Masking failed: {}", e);
                                                    println!("[Error] Rust Backend Error Caught: {}", err_str);
                                                    has_error = Some(err_str);
                                                    record.context.clone()
                                                });
                                                masked_success = true;
                                            } else {
                                                let err_str = "Privacy Filter 모델 로드에 실패했습니다.".to_string();
                                                println!("[Error] {}", err_str);
                                                has_error = Some(err_str);
                                            }
                                        }
                                        if masked_success {
                                            record.masking = masked_text;
                                            println!("[System] [Record ID: {}] 최종 전처리 결과:\n{}", record.id, record.masking);
                                        }

                                        // 🚀 실시간 퍼센트 전송 및 브로드캐스트
                                        current_step += 1;
                                        let percent = (current_step as f64 / total_steps as f64 * 100.0) as usize;
                                        let payload = json!({"item_display": idx + 1, "total_items": total_items, "percent": percent});
                                        *GLOBAL_PROGRESS.lock().unwrap() = Some(payload.clone());
                                        let script = format!("window.dispatchEvent(new CustomEvent('rpc_response', {{ detail: {} }}));", json!(json!({"type": "push_progress", "payload": payload})));
                                        if let Ok(pages) = _browser_clone.pages().await { for p in pages { let _ = p.evaluate(script.clone()).await; } }
                                    }
                                    
                                    {
                                        let mut pm_guard = PRIVACY_MANAGER.lock().unwrap();
                                        *pm_guard = None; // VRAM 완전 해제
                                    }
                                    force_memory_cleanup();
                                    println!("[System] === Phase 1: Privacy Filter 마스킹 종료 (VRAM 해제됨) ===\n");
                                } else {
                                    current_step += total_items;
                                }

                                // ★ Phase 2: Embedding 일괄 벡터화 및 VRAM 해제
                                if has_error.is_none() {
                                    let needs_embedding = target_records.iter().any(|r| !r.masking.trim().is_empty());
                                    
                                    if needs_embedding {
                                        {
                                            let mut em_guard = EMBEDDING_MODEL.lock().unwrap();
                                            if em_guard.is_none() {
                                                *em_guard = gemini_gui_lib::embedding::EmbeddingModel::new_with_device("..\\models\\embeddings", &device).ok();
                                            }
                                        }
                                        for (idx, record) in target_records.iter_mut().enumerate() {
                                            let text_to_embed = record.masking.trim();
                                            if text_to_embed.is_empty() {
                                                println!("[System] Phase 2 - [{}]순서 임베딩 건너뜐 (빈 텍스트)", idx);
                                                record.vector = vec![0.0; 768];
                                            } else {
                                                println!("[System] Phase 2 - [{}]순서 임베딩 진행 중...", idx);
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
                                            
                                            // 🚀 실시간 퍼센트 전송 및 브로드캐스트
                                            current_step += 1;
                                            let percent = (current_step as f64 / total_steps as f64 * 100.0) as usize;
                                            let payload = json!({"item_display": idx + 1, "total_items": total_items, "percent": percent});
                                            *GLOBAL_PROGRESS.lock().unwrap() = Some(payload.clone());
                                            let script = format!("window.dispatchEvent(new CustomEvent('rpc_response', {{ detail: {} }}));", json!(json!({"type": "push_progress", "payload": payload})));
                                            if let Ok(pages) = _browser_clone.pages().await { for p in pages { let _ = p.evaluate(script.clone()).await; } }
                                        }
                                        
                                        {
                                            let mut em_guard = EMBEDDING_MODEL.lock().unwrap();
                                            *em_guard = None; // VRAM 완전 해제
                                        }
                                        force_memory_cleanup();
                                    } else {
                                        println!("[System] === Phase 2: 임베딩 대상 텍스트가 모두 비어있어 모델 로드를 완전히 건너뜁니다. ===");
                                        for record in &mut target_records {
                                            record.vector = vec![0.0; 768];
                                        }
                                        current_step += total_items;
                                        let percent = (current_step as f64 / total_steps as f64 * 100.0) as usize;
                                        let payload = json!({"item_display": total_items, "total_items": total_items, "percent": percent});
                                        *GLOBAL_PROGRESS.lock().unwrap() = Some(payload.clone());
                                        let script = format!("window.dispatchEvent(new CustomEvent('rpc_response', {{ detail: {} }}));", json!(json!({"type": "push_progress", "payload": payload})));
                                        if let Ok(pages) = _browser_clone.pages().await { for p in pages { let _ = p.evaluate(script.clone()).await; } }
                                    }
                                }

                                // DB 갱신 처리
                                *GLOBAL_PROGRESS.lock().unwrap() = None; // 🚀 작업 완료 시 전역 진행률을 안전하게 초기화합니다.
                                if let Some(err_msg) = has_error {
                                    json!({"type": "error", "message": err_msg}).to_string()
                                } else {
                                    for record in &mut target_records {
                                        record.status = "PUSHED".to_string();
                                        let expr = format!("id = '{}'", record.id);
                                        let _ = table.delete(&expr).await;
                                    }
                                    // 마스킹과 임베딩 적용된 레코드 LanceDB 저장
                                    match db::save_records(target_records.clone(), None).await {
                                        Ok(_) => json!({"type": "push_success", "payload": target_records}).to_string(),
                                        Err(e) => json!({"type": "error", "message": format!("DB Save Error: {}", e)}).to_string(),
                                    }
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
                    response_json
                } else if payload.starts_with("process_file:") {
                    let full_data_url = &payload["process_file:".len()..];
                    let mut ocr_result = String::new();
                    let mut masked_result = String::new();
                    let mut has_error = None;

                    // 1. OCR Extract
                    {
                        {
                            let mut model_guard = OCR_MODEL.lock().unwrap();
                            if model_guard.is_none() {
                                let model_path = "..\\models\\glm_ocr";
                                // 🚀 수동 파일 업로드 시에도 CUDA VRAM 기반 SSD 오프로딩을 적극 적용합니다.
                                let ocr_device = Device::new_cuda(0).unwrap_or(Device::Cpu);
                                match GlmOcrGenerateModel::init(model_path, Some(&ocr_device), None) {
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
                                        parts: vec![Part { text: "Extract text".to_string(), image_url: Some(full_data_url.to_string()) }],
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
                                        println!("[Error] 단일 파일 OCR 처리 중 예외 발생: {:?}", e); // 🌟 Rust 로그 추가
                                        has_error = Some(err_msg);
                                    }
                                }
                            } else {
                                let err_msg = "OCR 모델 로드에 실패했습니다.".to_string();
                                println!("[Error] {}", err_msg); // 🌟 Rust 로그 추가
                                has_error = Some(err_msg);
                            }
                            // ★ VRAM 해제
                            *model_guard = None;
                        } // 뮤텍스 락 스코프 종료로 인한 자동 해제
                        
                        force_memory_cleanup(); // 🚀 OS 커널 레벨 메모리 강제 회수
                        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await; // 확실한 VRAM 해제 텀 부여
                        
                        // 🚀 단일 파일 처리 시에도 VRAM 해제 여부를 즉각 확인할 수 있도록 로그를 추가합니다.
                        println!("[System] === 단일 파일 GLM OCR 처리 종료 (VRAM 해제됨) ===\n");
                    }

                    // 2. Privacy Filter
                    if has_error.is_none() && !ocr_result.is_empty() {
                        let mut pm_guard = PRIVACY_MANAGER.lock().unwrap();
                        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
                        if pm_guard.is_none() {
                            *pm_guard = gemini_gui_lib::privacy_filter::masking::PrivacyManager::new("..\\models\\privacy-filter", &device).ok();
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
                        // ★ VRAM 해제
                        *pm_guard = None;
                        force_memory_cleanup(); // 🚀 OS 커널 레벨 메모리 강제 회수
                    }

                    if let Some(err_msg) = has_error {
                        json!({"type": "error", "message": err_msg}).to_string()
                    } else {
                        // 🚀 단일 파일 처리 완료 후 최종 텍스트 결과를 콘솔에 출력합니다.
                        println!("[System] [단일 파일 처리] 최종 전처리 결과:\n{}", masked_result);
                        json!({"type": "file_processed", "payload": {"ocr": ocr_result, "masked": masked_result}}).to_string()
                    }
                } else if payload.starts_with("gemini_chat:") {
                    "[System] Gemini 서비스 비활성화됨".to_string()
                } else if payload == "check_progress" {
                    if let Some(progress) = GLOBAL_PROGRESS.lock().unwrap().clone() {
                        json!({"type": "push_progress", "payload": progress}).to_string()
                    } else {
                        json!({"type": "push_idle"}).to_string()
                    }
                } else { "Unknown command".to_string() };

                let script = format!("window.dispatchEvent(new CustomEvent('rpc_response', {{ detail: {} }}));", json!(response));
                if let Ok(pages) = _browser_clone.pages().await {
                    for p in pages { let _ = p.evaluate(script.clone()).await; }
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
        _ = tokio::signal::ctrl_c() => println!("\nShutting down..."),
        _ = rx.recv() => println!("\nBrowser closed."),
    }
    Ok(())
}
