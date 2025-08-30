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