import { invoke } from "@tauri-apps/api/core";
import { db } from "./db";

async function saveMessage(channelId: string, role: 'user' | 'assistant', content: string) {
  // 1. Save to local cache (Dexie)
  const id = await db.conversation_log.add({
    channel_id: channelId,
    role,
    content,
    timestamp: Date.now(),
    status: 'pending'
  });

  // 2. Sync to Backend (LanceDB) via Harness IPC
  try {
    const result = await invoke<string>("execute_branching_cli", { 
      command: JSON.stringify({ id, role, content, channel_id: channelId }) 
    });
    console.log("Sync success:", result);
    
    // 3. Mark as committed
    await db.conversation_log.update(id, { status: 'committed' });
  } catch (error) {
    console.error("Sync failed:", error);
    // Handle error (e.g., mark as error in UI)
  }
}

async function runCommand() {
  const inputEl = document.querySelector("#cli-input") as HTMLInputElement;
  const outputEl = document.querySelector("#cli-output") as HTMLElement;

  if (!inputEl || !outputEl) return;

  const content = inputEl.value;
  await saveMessage('trunk', 'user', content);

  try {
    const output = await invoke<string>("execute_branching_cli", { command: content });
    outputEl.textContent = output;
  } catch (error) {
    outputEl.textContent = "Error: " + error;
  }
}

window.addEventListener("DOMContentLoaded", () => {
  document.querySelector("#run-btn")?.addEventListener("click", runCommand);
});
