/**
 * Type definitions for the frontend chat application
 */

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  attachments?: any[];
  branchId?: string;
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

// Settings types
export interface AppSettings {
  apiKeys: Record<string, ApiKeyConfig>;
  selectedProvider: string;
  selectedModel: string;
  usageMonitoring: boolean;
  autoValidateKeys: boolean;
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
}
