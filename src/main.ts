import { invoke } from "@tauri-apps/api/core";
import { db } from "./db";
import { start } from '@fabianlars/tauri-plugin-oauth';

const openTabs: { hash: string, url: string }[] = [];

async function login() {
  try {
    const port = await start((url: string) => {
      console.log("Received redirect URL:", url);
    });
    
    const authUrl = `https://accounts.google.com/o/oauth2/v2/auth?client_id=YOUR_CLIENT_ID&redirect_uri=http://localhost:${port}&response_type=code&scope=email profile`;
    window.open(authUrl, '_blank');
    
    alert("로그인 페이지가 열렸습니다.");
  } catch (error) {
    console.error("Login failed:", error);
    alert("로그인 실패: " + error);
  }
}

async function openNewTab() {
  const urlInput = document.querySelector("#url-input") as HTMLInputElement;
  const url = urlInput.value || "https://www.google.com";

  try {
    const tabHash = await invoke<string>("open_new_tab", { url });
    openTabs.push({ hash: tabHash, url });
    updateTabList();
  } catch (e) {
    console.error("탭 열기 실패:", e);
  }
}

function updateTabList() {
  const tabBar = document.querySelector("#tab-bar") as HTMLElement;
  if (!tabBar) return;
  tabBar.innerHTML = openTabs.map(tab => 
    `<button class="tab-btn" onclick="alert('Focusing tab: ${tab.hash.substring(0, 8)}')">${tab.url.substring(0, 15)}...</button>`
  ).join('');
}

async function saveMessage(channelId: string, role: 'user' | 'assistant', content: string) {
  const id = await db.conversation_log.add({
    channel_id: channelId,
    role,
    content,
    timestamp: Date.now(),
    status: 'pending'
  });

  try {
    // Direct sync to Rust backend
    const result = await invoke<string>("sync_data", { 
      data: JSON.stringify({ id, role, content, channel_id: channelId }) 
    });
    console.log("Sync success:", result);
    
    await db.conversation_log.update(id, { status: 'committed' });
  } catch (error) {
    console.error("Sync failed:", error);
  }
}

async function runCommand() {
  const inputEl = document.querySelector("#cli-input") as HTMLInputElement;
  const outputEl = document.querySelector("#cli-output") as HTMLElement;

  if (!inputEl || !outputEl) return;

  const content = inputEl.value;
  await saveMessage('trunk', 'user', content);

  try {
    // Direct chat to Rust backend
    const output = await invoke<string>("gemini_chat", { 
      payload: JSON.stringify({
        messages: [{ role: 'user', parts: [{ text: content }] }],
        model: 'gemini-3.1-flash-lite-preview'
      })
    });
    outputEl.textContent = output;
  } catch (error) {
    outputEl.textContent = "Error: " + error;
  }
}

window.addEventListener("DOMContentLoaded", () => {
  document.querySelector("#login-btn")?.addEventListener("click", login);
  document.querySelector("#new-tab-btn")?.addEventListener("click", openNewTab);
  document.querySelector("#run-btn")?.addEventListener("click", runCommand);
});
