/**
 * API client for communicating with the backend server
 */

import { 
  ApiResponse, 
  ChatCompletionRequest, 
  ChatCompletionResponse, 
  ConfigOption, 
  ConversationBranch, 
  Message 
} from './types';

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl?: string) {
    // Use runtime check for environment variable, avoiding Node.js process.env
    this.baseUrl = baseUrl || 
                  (typeof window !== 'undefined' ? 
                    (window as any).NEXT_PUBLIC_BACKEND_URL : undefined) ||
                  'http://localhost:9000'; // Match backend default port
  }

  // Allow dynamic URL updates for debugging or runtime discovery
  updateBaseUrl(newBaseUrl: string): void {
    this.baseUrl = newBaseUrl;
  }

  // Get current base URL
  getBaseUrl(): string {
    return this.baseUrl;
  }

  private async fetchApi<T = any>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseUrl}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });

      const data = await response.json();
      
      if (!response.ok) {
        return {
          success: false,
          message: data.message || response.statusText,
          error: data.error || response.statusText,
        };
      }

      return {
        success: true,
        data,
      };
    } catch (error) {
      return {
        success: false,
        message: error instanceof Error ? error.message : 'Network error',
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  // Health check
  async health(): Promise<ApiResponse> {
    return this.fetchApi('/health');
  }

  // Chat completions
  async chatCompletion(request: ChatCompletionRequest): Promise<ApiResponse<ChatCompletionResponse>> {
    return this.fetchApi('/v1/chat/completions', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // Stream chat completions
  async streamChatCompletion(
    request: ChatCompletionRequest,
    onChunk: (chunk: string) => void
  ): Promise<ApiResponse> {
    const url = `${this.baseUrl}/v1/chat/completions`;
    
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ...request, stream: true }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        return {
          success: false,
          message: errorData.message || response.statusText,
        };
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        return {
          success: false,
          message: 'No response body',
        };
      }

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter(line => line.trim());
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') {
              return { success: true };
            }
            
            try {
              const parsed = JSON.parse(data);
              const content = parsed.choices?.[0]?.delta?.content;
              if (content) {
                onChunk(content);
              }
            } catch (e) {
              console.warn('Failed to parse SSE data:', data);
            }
          }
        }
      }

      return { success: true };
    } catch (error) {
      return {
        success: false,
        message: error instanceof Error ? error.message : 'Stream error',
      };
    }
  }

  // Configuration management
  async getConfigs(): Promise<ApiResponse<{ configs: ConfigOption[] }>> {
    return this.fetchApi('/v1/oumi/configs');
  }

  async getModels(): Promise<ApiResponse<{ data: Array<{ id: string; config_metadata?: any }> }>> {
    return this.fetchApi('/v1/models');
  }

  // Branch management
  async getBranches(sessionId: string): Promise<ApiResponse<{ 
    branches: ConversationBranch[];
    current_branch?: string;
  }>> {
    return this.fetchApi(`/v1/oumi/branches?session_id=${encodeURIComponent(sessionId)}`);
  }

  async createBranch(
    sessionId: string,
    name: string,
    parentBranchId?: string
  ): Promise<ApiResponse<{ branch: ConversationBranch }>> {
    return this.fetchApi(`/v1/oumi/branches`, {
      method: 'POST',
      body: JSON.stringify({ 
        action: "create",                    // Required by backend
        session_id: sessionId,
        name, 
        from_branch: parentBranchId          // Backend expects 'from_branch' not 'parent_branch_id'
      }),
    });
  }

  async switchBranch(
    sessionId: string,
    branchId: string
  ): Promise<ApiResponse> {
    return this.fetchApi(`/v1/oumi/command`, {
      method: 'POST',
      body: JSON.stringify({ 
        command: 'switch',
        args: [branchId],
        session_id: sessionId
      }),
    });
  }

  async deleteBranch(
    sessionId: string,
    branchId: string
  ): Promise<ApiResponse> {
    return this.fetchApi(`/v1/oumi/command`, {
      method: 'POST',
      body: JSON.stringify({ 
        command: 'branch_delete',
        args: [branchId],
        session_id: sessionId
      }),
    });
  }

  // Conversation management
  async getConversation(
    sessionId: string,
    branchId: string = 'main'
  ): Promise<ApiResponse<{ conversation: Message[] }>> {
    return this.fetchApi(`/v1/oumi/conversation?session_id=${encodeURIComponent(sessionId)}&branch_id=${encodeURIComponent(branchId)}`);
  }

  async sendMessage(
    content: string,
    sessionId: string,
    branchId: string = 'main'
  ): Promise<ApiResponse<Message>> {
    return this.fetchApi(`/api/sessions/${sessionId}/branches/${branchId}/messages`, {
      method: 'POST',
      body: JSON.stringify({ content }),
    });
  }

  // Command execution
  async executeCommand(
    command: string,
    args: string[] = []
  ): Promise<ApiResponse> {
    console.log(`🌐 API Client: Executing command '${command}' with args:`, args);
    const response = await this.fetchApi('/v1/oumi/command', {
      method: 'POST',
      body: JSON.stringify({ command, args }),
    });
    console.log(`🌐 API Client: Command '${command}' response:`, response);
    return response;
  }

  // System monitoring
  async getSystemStats(sessionId?: string): Promise<ApiResponse> {
    const url = sessionId ? `/v1/oumi/system_stats?session_id=${encodeURIComponent(sessionId)}` : '/v1/oumi/system_stats';
    return this.fetchApi(url);
  }

  async getModelStats(): Promise<ApiResponse> {
    return this.fetchApi('/v1/models');
  }

  async clearModel(): Promise<ApiResponse> {
    return this.fetchApi('/v1/oumi/clear_model', {
      method: 'POST'
    });
  }

  // File operations
  async uploadFile(file: File): Promise<ApiResponse> {
    const formData = new FormData();
    formData.append('file', file);

    return this.fetchApi('/api/files/upload', {
      method: 'POST',
      body: formData,
      headers: {}, // Let browser set Content-Type for FormData
    });
  }
}

// Create and export a singleton instance
const apiClient = new ApiClient();
export default apiClient;