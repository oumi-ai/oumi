/**
 * Unified API client that works in both web and Electron environments
 */

import apiClient from './api';  // Original HTTP API client
import electronAPI from './electron-api';  // Electron IPC API client

import { 
  ApiResponse, 
  ChatCompletionRequest, 
  ChatCompletionResponse, 
  ConfigOption, 
  ConversationBranch, 
  Message 
} from './types';

class UnifiedApiClient {
  private webClient = apiClient;
  private electronClient = electronAPI;

  /**
   * Determine if we're running in Electron
   */
  public isElectron(): boolean {
    return this.electronClient.isElectronApp();
  }

  /**
   * Get the appropriate client based on environment
   */
  private getClient() {
    return this.isElectron() ? this.electronClient : this.webClient;
  }

  // Server control (Electron only)
  async startServer(): Promise<ApiResponse> {
    if (this.isElectron()) {
      return this.electronClient.startServer();
    } else {
      return { success: true, message: 'Server control not available in web version' };
    }
  }

  async stopServer(): Promise<ApiResponse> {
    if (this.isElectron()) {
      return this.electronClient.stopServer();
    } else {
      return { success: true, message: 'Server control not available in web version' };
    }
  }

  async restartServer(): Promise<ApiResponse> {
    if (this.isElectron()) {
      return this.electronClient.restartServer();
    } else {
      return { success: true, message: 'Server control not available in web version' };
    }
  }

  async getServerStatus(): Promise<ApiResponse> {
    if (this.isElectron()) {
      return this.electronClient.getServerStatus();
    } else {
      return { success: true, data: { running: true, url: 'N/A', port: 'N/A' } };
    }
  }

  // Health check
  async health(): Promise<ApiResponse> {
    return this.getClient().health();
  }

  // Chat completions
  async chatCompletion(request: ChatCompletionRequest): Promise<ApiResponse<ChatCompletionResponse>> {
    return this.getClient().chatCompletion(request);
  }

  // Stream chat completions
  async streamChatCompletion(
    request: ChatCompletionRequest,
    onChunk: (chunk: string) => void
  ): Promise<ApiResponse> {
    return this.getClient().streamChatCompletion(request, onChunk);
  }

  // Configuration management
  async getConfigs(): Promise<ApiResponse<{ configs: ConfigOption[] }>> {
    return this.getClient().getConfigs();
  }

  async getModels(): Promise<ApiResponse<{ data: Array<{ id: string; config_metadata?: any }> }>> {
    return this.getClient().getModels();
  }

  // Branch management
  async getBranches(sessionId: string = 'default'): Promise<ApiResponse<{ 
    branches: ConversationBranch[];
    current_branch?: string;
  }>> {
    return this.getClient().getBranches(sessionId);
  }

  async createBranch(
    sessionId: string = 'default',
    name: string,
    parentBranchId?: string
  ): Promise<ApiResponse<{ branch: ConversationBranch }>> {
    return this.getClient().createBranch(sessionId, name, parentBranchId);
  }

  async switchBranch(
    sessionId: string = 'default',
    branchId: string
  ): Promise<ApiResponse> {
    return this.getClient().switchBranch(sessionId, branchId);
  }

  async deleteBranch(
    sessionId: string = 'default',
    branchId: string
  ): Promise<ApiResponse> {
    return this.getClient().deleteBranch(sessionId, branchId);
  }

  // Conversation management
  async getConversation(
    sessionId: string = 'default',
    branchId: string = 'main'
  ): Promise<ApiResponse<{ conversation: Message[] }>> {
    return this.getClient().getConversation(sessionId, branchId);
  }

  async sendMessage(
    content: string,
    sessionId: string = 'default',
    branchId: string = 'main'
  ): Promise<ApiResponse<Message>> {
    return this.getClient().sendMessage(content, sessionId, branchId);
  }

  // Command execution
  async executeCommand(
    command: string,
    args: string[] = []
  ): Promise<ApiResponse> {
    return this.getClient().executeCommand(command, args);
  }

  // System monitoring
  async getSystemStats(): Promise<ApiResponse> {
    return this.getClient().getSystemStats();
  }

  async getModelStats(): Promise<ApiResponse> {
    return this.getClient().getModelStats();
  }

  // File operations (Electron-specific, with fallbacks)
  async uploadFile(file: File): Promise<ApiResponse> {
    if (this.isElectron()) {
      // For Electron, we could handle file operations differently
      throw new Error('File upload not yet implemented for Electron');
    } else {
      return this.webClient.uploadFile(file);
    }
  }

  // Electron-specific methods (graceful degradation for web)
  async showSaveDialog(options: any): Promise<string | null> {
    if (this.isElectron()) {
      return this.electronClient.showSaveDialog(options);
    } else {
      // Web fallback - could show a modal or use browser download
      console.warn('Save dialog not available in web version');
      return null;
    }
  }

  async showOpenDialog(options: any): Promise<string[] | null> {
    if (this.isElectron()) {
      return this.electronClient.showOpenDialog(options);
    } else {
      // Web fallback - could use file input
      console.warn('Open dialog not available in web version');
      return null;
    }
  }

  async saveConversationToFile(conversation: Message[], filename?: string): Promise<boolean> {
    if (this.isElectron()) {
      return this.electronClient.saveConversationToFile(conversation, filename);
    } else {
      // Web fallback - download as file
      try {
        const content = JSON.stringify({
          version: '1.0',
          timestamp: new Date().toISOString(),
          conversation
        }, null, 2);

        const blob = new Blob([content], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename || `conversation-${new Date().toISOString().slice(0, 10)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        return true;
      } catch (error) {
        console.error('Error downloading conversation:', error);
        return false;
      }
    }
  }

  async loadConversationFromFile(): Promise<Message[] | null> {
    if (this.isElectron()) {
      return this.electronClient.loadConversationFromFile();
    } else {
      // Web fallback - use file input
      return new Promise((resolve) => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';
        input.onchange = async (e) => {
          const file = (e.target as HTMLInputElement).files?.[0];
          if (!file) {
            resolve(null);
            return;
          }

          try {
            const content = await file.text();
            const data = JSON.parse(content);
            resolve(data.conversation || null);
          } catch (error) {
            console.error('Error loading conversation:', error);
            resolve(null);
          }
        };
        input.click();
      });
    }
  }

  // Storage methods (with localStorage fallback)
  async getStorageItem(key: string, defaultValue?: any): Promise<any> {
    if (this.isElectron()) {
      return this.electronClient.getStorageItem(key, defaultValue);
    } else {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue;
    }
  }

  async setStorageItem(key: string, value: any): Promise<void> {
    if (this.isElectron()) {
      return this.electronClient.setStorageItem(key, value);
    } else {
      localStorage.setItem(key, JSON.stringify(value));
    }
  }

  async deleteStorageItem(key: string): Promise<void> {
    if (this.isElectron()) {
      return this.electronClient.deleteStorageItem(key);
    } else {
      localStorage.removeItem(key);
    }
  }

  async clearStorage(): Promise<void> {
    if (this.isElectron()) {
      return this.electronClient.clearStorage();
    } else {
      localStorage.clear();
    }
  }

  // App control methods (Electron-specific)
  async getVersion(): Promise<string> {
    if (this.isElectron()) {
      return this.electronClient.getVersion();
    } else {
      return 'Web Version';
    }
  }

  reload(): void {
    if (this.isElectron()) {
      this.electronClient.reload();
    } else {
      window.location.reload();
    }
  }

  quit(): void {
    if (this.isElectron()) {
      this.electronClient.quit();
    } else {
      // Web can't quit, but could close tab
      window.close();
    }
  }

  toggleDevTools(): void {
    if (this.isElectron()) {
      this.electronClient.toggleDevTools();
    } else if (process.env.NODE_ENV === 'development') {
      console.log('Dev tools toggle not available in web version');
    }
  }

  toggleFullScreen(): void {
    if (this.isElectron()) {
      this.electronClient.toggleFullScreen();
    } else {
      // Web fullscreen API
      if (document.fullscreenElement) {
        document.exitFullscreen();
      } else {
        document.documentElement.requestFullscreen();
      }
    }
  }

  zoom(direction: 'in' | 'out' | 'reset'): void {
    if (this.isElectron()) {
      this.electronClient.zoom(direction);
    } else {
      // Web zoom fallback (limited browser support)
      const currentZoom = parseFloat(document.body.style.zoom || '1');
      switch (direction) {
        case 'in':
          document.body.style.zoom = Math.min(currentZoom + 0.1, 3).toString();
          break;
        case 'out':
          document.body.style.zoom = Math.max(currentZoom - 0.1, 0.5).toString();
          break;
        case 'reset':
          document.body.style.zoom = '1';
          break;
      }
    }
  }

  // Event system methods (Electron-specific)
  on(channel: string, listener: (...args: any[]) => void): void {
    if (this.isElectron()) {
      this.electronClient.on(channel, listener);
    }
  }

  off(channel: string, listener: (...args: any[]) => void): void {
    if (this.isElectron()) {
      this.electronClient.off(channel, listener);
    }
  }

  send(channel: string, ...args: any[]): void {
    if (this.isElectron()) {
      this.electronClient.send(channel, ...args);
    }
  }

  async invoke(channel: string, ...args: any[]): Promise<any> {
    if (this.isElectron()) {
      return this.electronClient.invoke(channel, ...args);
    }
    return null;
  }

  // Platform information
  getPlatform(): { os: string; arch: string; version: string } {
    if (this.isElectron()) {
      return this.electronClient.getPlatform();
    } else {
      return {
        os: 'web',
        arch: 'unknown', 
        version: 'unknown'
      };
    }
  }

  // Get appropriate base URL for web client
  getBaseUrl(): string {
    if (this.isElectron()) {
      return 'N/A (Using IPC)';
    } else {
      return this.webClient.getBaseUrl();
    }
  }

  // Update base URL for web client (ignored in Electron)
  updateBaseUrl(newBaseUrl: string): void {
    if (!this.isElectron()) {
      this.webClient.updateBaseUrl(newBaseUrl);
    }
  }
}

// Create and export singleton instance
const unifiedApiClient = new UnifiedApiClient();
export default unifiedApiClient;