# PRD: AI Commerce Agent Branching & Pruning Architecture

## 1. Overview
This PRD defines an AI commerce agent architecture that utilizes a "Branching & Pruning" model to manage large commerce contexts (orders, products) efficiently within Gemini 3.1 Flash-Lite's token limits and cost constraints.

## 2. Core Principles
- **Trunk & Branch (Branching):** Main Trunk manages the overall context (Read-Only). Sub-Branches are created for event-based operations (Write/Modify) to prevent context pollution in the main channel.
- **State-Based Overwrite (Pruning):** AI does not retain event logs. It maintains a single "source of truth" (YAML/CSV) and overwrites the state, preventing "infinite history" token explosion.
- **Zero-Prompt UX:** The user interacts via structured UI buttons, which map to backend prompts.
- **Session Reset:** Sub-channels are short-lived, ensuring Gemini's KV cache is refreshed and pollution is eliminated.

## 3. Component Architecture
- **Trunk Channel:** The persistent main LLM session.
- **Branch Channel:** Ephemeral sub-sessions for specific tasks (e.g., partial cancellation).
- **Harness (Middleware):** Pre-processes web browser data (DOM flattening), manages state, and handles event triggering.

## 4. Operational Flow
1. **Event Trigger:** User clicks a button (e.g., "Partial Cancel").
2. **Branching:** System opens a temporary Sub-Branch session, inheriting Main Trunk context.
3. **Execution:** AI processes the event, creates a command (JSON), and shuts down the Sub-Branch.
4. **State Update:** System updates the central YAML/DB state.
5. **Sync:** Main Trunk reloads the updated state for the next interaction.

## 5. Technical Requirements
- Support for stateful command execution.
- Automated web browser data pruning.
- Periodic re-caching of the Base state.
