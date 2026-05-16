import { invoke } from "@tauri-apps/api/core";

async function login() {
  alert("OAuth login temporarily disabled.");
}

async function runCommand() {
  const inputEl = document.querySelector("#cli-input") as HTMLInputElement;
  const outputEl = document.querySelector("#cli-output") as HTMLElement;

  if (!inputEl || !outputEl) return;

  try {
    const output = await invoke<string>("execute_branching_cli", { command: inputEl.value });
    outputEl.textContent = output;
  } catch (error) {
    outputEl.textContent = "Error: " + error;
  }
}

async function renderDiff(oldState: any, newState: any) {
  const diffOutputEl = document.querySelector("#diff-output") as HTMLElement;
  if (!diffOutputEl) return;

  try {
    const patchStr = await invoke<string>("calculate_state_diff", { 
        oldState: JSON.stringify(oldState), 
        newState: JSON.stringify(newState) 
    });
    const patch = JSON.parse(patchStr);

    let html = "<h3>State Changes:</h3><pre>";
    patch.forEach((op: any) => {
        if (op.op === 'add') html += `<span style="color:green">+ ${op.path}: ${JSON.stringify(op.value)}</span>\n`;
        else if (op.op === 'remove') html += `<span style="color:red">- ${op.path}</span>\n`;
        else if (op.op === 'replace') html += `<span style="color:orange">~ ${op.path}: ${JSON.stringify(op.value)}</span>\n`;
    });
    html += "</pre>";
    diffOutputEl.innerHTML = html;
  } catch (error) {
    diffOutputEl.textContent = "Diff Error: " + error;
  }
}

window.addEventListener("DOMContentLoaded", () => {
  document.querySelector("#login-btn")?.addEventListener("click", login);
  document.querySelector("#run-btn")?.addEventListener("click", runCommand);
  renderDiff({a: 1}, {a: 2, b: 3});
});
