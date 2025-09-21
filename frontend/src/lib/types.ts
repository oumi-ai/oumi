/**
 * Type definitions for the frontend chat application
 */

export interface Session {
  id: string;
  name: string;
  description?: string;
  modelId?: string;
  createdAt: string;
  updatedAt: string;
  conversationIds: string[];
  metadata?: {
    [key: string]: any;
  };
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  attachments?: any[];
  branchId?: string;
  meta?: {
    authorName?: string;            // Human-friendly name (user or model label)
    authorType?: 'user' | 'ai' | 'system';
    modelName?: string;             // If AI generated/edited
    engine?: string;                // Provider/engine id
    createdAt?: number;             // Epoch ms when message was created
    durationMs?: number;            // Generation duration (AI)
    [k: string]: any;
  };
}

// Phase A – Versioned message primitives (non-breaking)
export interface MessageVersion {
  id: string;          // stable id for this version
  role: 'user' | 'assistant' | 'system';
  content: string;     // raw content (markdown/plain)
  timestamp: number;   // when this version was created
  attachments?: any[];
  meta?: {
    editor?: 'user' | 'system' | 'regen' | string;
    sourceVersionId?: string; // for regen/edit lineage
    authorName?: string;
    authorType?: 'user' | 'ai' | 'system';
    modelName?: string;
    engine?: string;
    createdAt?: number;
    durationMs?: number;
    [k: string]: any;
  };
}

export interface MessageNode {
  id: string;                 // node identity within a conversation
  versions: MessageVersion[]; // ordered chronologically
}

// Merge persistence (Phase B placeholder)
export interface MergeRecord {
  id: string;
  timestamp: string;
  sourceBranchId: string;
  targetBranchId: string;
  sourceNodeId?: string;
  targetNodeId?: string;
  chosenVersionId?: string;
  meta?: { [k: string]: any };
}

export interface ConversationBranch {
  id: string;
  name: string;
  isActive: boolean;
  messageCount: number;
  createdAt: string;
  lastActive: string;
  preview?: string;
  parentId?: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  branches?: { [branchId: string]: { messages: Message[] } };
  createdAt: string;
  updatedAt: string;
}

export interface ModelConfigMetadata {
  display_name: string;
  description: string;
  engine: string;
  context_length: number;
  model_family: string;
  model_name?: string;
  filename?: string;
  config_path?: string;
}

export interface GenerationParams {
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  topK?: number;
  contextLength?: number;
  stream?: boolean;
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
}

export interface DownloadProgress {
  filename: string;
  progress: number;       // 0-100
  downloaded: string;     // "1.2GB" 
  total: string;          // "1.8GB"
  speed: string;          // "15.2MB/s"
  isComplete: boolean;
  estimatedTimeRemaining?: string; // "00:37"
}

export interface DownloadState {
  isDownloading: boolean;
  downloads: Map<string, DownloadProgress>;
  overallProgress: number;
  totalFiles: number;
  completedFiles: number;
  hasError: boolean;
  errorMessage?: string;
}

export interface DownloadErrorEvent {
  message: string;
  timestamp: string;
  filename?: string;
}

export interface ConfigOption {
  id: string;
  config_path: string;
  relative_path: string;
  display_name: string;
  model_name: string;
  engine: string;
  context_length: number;
  model_family: string;
  filename: string;
}

export interface ChatCompletionRequest {
  messages: Array<{
    role: 'user' | 'assistant' | 'system';
    content: string;
  }>;
  session_id?: string;
  branch_id?: string;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  stream?: boolean;
}

export interface ChatCompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string;
    };
    finish_reason: string;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

// API Provider types
export interface ApiProvider {
  id: string;
  name: string;
  displayName: string;
  description: string;
  website: string;
  keyName: string; // e.g., "OPENAI_API_KEY"
  keyPlaceholder: string; // e.g., "sk-..."
  baseUrl?: string;
  icon?: string;
  models: ApiModel[];
  pricing?: ApiPricing;
  requiresKey: boolean;
  testEndpoint?: string;
}

export interface ApiModel {
  id: string;
  name: string;
  displayName: string;
  description: string;
  contextLength: number;
  inputCost?: number; // per 1M tokens
  outputCost?: number; // per 1M tokens
  isMultimodal?: boolean;
  tags?: string[];
}

export interface ApiPricing {
  currency: string;
  unit: string; // "1M tokens", "1K tokens", etc.
  inputCostPer1M?: number;
  outputCostPer1M?: number;
}

export interface ApiKeyConfig {
  providerId: string;
  // In Electron builds, do not store full key value in renderer.
  // Keep this optional for compatibility; prefer using `last4` + metadata.
  keyValue?: string;
  // Convenience for UI display without storing secrets
  last4?: string;
  isActive: boolean;
  createdAt: string;
  lastValidated?: string;
  isValid?: boolean;
  validationError?: string;
  usage?: ApiUsageStats;
}

export interface ApiUsageStats {
  totalRequests: number;
  totalTokens: number;
  totalCost: number;
  lastReset: string;
  monthlyLimit?: number;
  monthlyUsed: number;
}

export interface ApiValidationResult {
  isValid: boolean;
  error?: string;
  details?: {
    model?: string;
    organization?: string;
    rateLimit?: number;
  };
}

// ChatHistory artifact used for full-session save/load
export interface ChatHistory {
  version: string;              // schema version
  created_at: string;           // ISO timestamp
  saved_by?: string;            // optional user display name
  session_id: string;           // session identifier
  model_info?: {
    name?: string;
    engine?: string;
    context_length?: number;
  };
  current_conversation_id?: string;
  current_branch_id?: string;
  conversations: Array<{
    id: string;
    title: string;
    updatedAt: string;
    branches: { [branchId: string]: { messages: Message[]; metadata?: any } };
    nodeGraph?: {
      nodes?: { [id: string]: MessageNode };
      timelines?: { [branchId: string]: string[] };
      heads?: { [branchId: string]: { [nodeId: string]: string } };
      tombstones?: { [branchId: string]: { [nodeId: string]: boolean } };
      merges?: MergeRecord[];
    };
  }>;
}

// Settings types
export interface AppSettings {
  apiKeys: Record<string, ApiKeyConfig>;
  selectedProvider: string;
  selectedModel: string;
  usageMonitoring: boolean;
  autoValidateKeys: boolean;
  // When true, the app creates a fresh session on each launch
  startNewSessionOnLaunch: boolean;
  notifications: {
    lowBalance: boolean;
    highUsage: boolean;
    keyExpiry: boolean;
  };
  autoSave: {
    enabled: boolean;
    intervalMinutes: number;
  };
  // HuggingFace credentials for improved model recommendations
  huggingFace: {
    username?: string;
    token?: string; // Personal Access Token
  };
  // User profile
  user?: {
    displayName?: string;
  };
}
