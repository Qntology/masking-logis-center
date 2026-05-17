import Dexie, { Table } from 'dexie';

export interface ConversationLog {
  id?: number;
  channel_id: string; // 'trunk' or branch ID
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  status: 'pending' | 'committed'; // Sync status
}

export interface SessionState {
  id?: number;
  active_branch: string | null;
  last_updated: number;
}

export class GeminiDB extends Dexie {
  conversation_log!: Table<ConversationLog>;
  session_state!: Table<SessionState>;

  constructor() {
    super('GeminiDB');
    this.version(1).stores({
      conversation_log: '++id, channel_id, timestamp, status',
      session_state: '++id, active_branch'
    });
  }
}

export const db = new GeminiDB();
