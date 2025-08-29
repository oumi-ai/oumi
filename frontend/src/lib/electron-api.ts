/**
 * Electron API client - replaces HTTP calls with IPC communication
 */

import { 
  ApiResponse, 
  ChatCompletionRequest, 
  ChatCompletionResponse, 
  ConfigOption, 
  ConversationBranch, 
  Message 
} from './types';

declare global {
  interface Window {
    electronAPI: {
      // Application control
      app: {
        getVersion: () => Promise<string>;
        quit: () => void;
        reload: () => void;
        toggleDevTools: () => void;
        toggleFullScreen: () => void;
        zoom: (direction: 'in' | 'out' | 'reset') => void;
      };

      // File system operations
      files: {
        showSaveDialog: (options: any) => Promise<string | null>;
        showOpenDialog: (options: any) => Promise<string[] | null>;
        writeFile: (filePath: string, content: string) => Promise<boolean>;
        readFile: (filePath: string) => Promise<string | null>;
        exists: (filePath: string) => Promise<boolean>;
      };

      // Server control
      server: {
        start: () => Promise<any>;
        stop: () => Promise<any>;
        restart: () => Promise<any>;
        status: () => Promise<any>;
      };

      // Chat API operations
      chat: {
        health: () => Promise<any>;
        getSystemStats: () => Promise<any>;
        getModelStats: () => Promise<any>;
        chatCompletion: (request: any) => Promise<any>;
        streamChatCompletion: (request: any, onChunk: (chunk: string) => void) => Promise<any>;
        getConfigs: () => Promise<any>;
        getModels: () => Promise<any>;
        getBranches: (sessionId?: string) => Promise<any>;
        createBranch: (sessionId: string, name: string, parentBranchId?: string) => Promise<any>;
        switchBranch: (sessionId: string, branchId: string) => Promise<any>;
        deleteBranch: (sessionId: string, branchId: string) => Promise<any>;
        getConversation: (sessionId?: string, branchId?: string) => Promise<any>;
        sendMessage: (content: string, sessionId?: string, branchId?: string) => Promise<any>;
        executeCommand: (command: string, args?: string[]) => Promise<any>;
      };

      // Real-time events
      events: {
        on: (channel: string, listener: (...args: any[]) => void) => void;
        off: (channel: string, listener: (...args: any[]) => void) => void;
        send: (channel: string, ...args: any[]) => void;
        invoke: (channel: string, ...args: any[]) => Promise<any>;
      };

      // Storage operations
      storage: {
        get: (key: string, defaultValue?: any) => Promise<any>;
        set: (key: string, value: any) => Promise<void>;
        delete: (key: string) => Promise<void>;
        clear: () => Promise<void>;
      };

      // Platform information
      platform: {
        os: string;
        arch: string;
        version: string;
      };
    };
  }
}

class ElectronApiClient {
  private isElectron: boolean;

  constructor() {
    this.isElectron = typeof window !== 'undefined' && 'electronAPI' in window;
  }

  // Check if running in Electron environment
  public isElectronApp(): boolean {
    return this.isElectron;
  }

  // App control methods
  public async getVersion(): Promise<string> {
    if (!this.isElectron) return 'Web Version';
    return window.electronAPI.app.getVersion();
  }

  public quit(): void {
    if (this.isElectron) {
      window.electronAPI.app.quit();
    }
  }

  public reload(): void {
    if (this.isElectron) {
      window.electronAPI.app.reload();
    } else {
      window.location.reload();
    }
  }

  public toggleDevTools(): void {
    if (this.isElectron) {
      window.electronAPI.app.toggleDevTools();
    }
  }

  public toggleFullScreen(): void {
    if (this.isElectron) {
      window.electronAPI.app.toggleFullScreen();
    }
  }

  public zoom(direction: 'in' | 'out' | 'reset'): void {
    if (this.isElectron) {
      window.electronAPI.app.zoom(direction);
    }
  }

  // Server control methods
  public async startServer(): Promise<ApiResponse> {
    if (!this.isElectron) {
      throw new Error('Server control only available in Electron app');
    }
    return window.electronAPI.server.start();
  }

  public async stopServer(): Promise<ApiResponse> {
    if (!this.isElectron) {
      throw new Error('Server control only available in Electron app');
    }
    return window.electronAPI.server.stop();
  }

  public async restartServer(): Promise<ApiResponse> {
    if (!this.isElectron) {
      throw new Error('Server control only available in Electron app');
    }
    return window.electronAPI.server.restart();
  }

  public async getServerStatus(): Promise<ApiResponse> {
    if (!this.isElectron) {
      throw new Error('Server control only available in Electron app');
    }
    return window.electronAPI.server.status();
  }

  // File system methods
  public async showSaveDialog(options: any): Promise<string | null> {
    if (!this.isElectron) return null;
    return window.electronAPI.files.showSaveDialog(options);
  }

  public async showOpenDialog(options: any): Promise<string[] | null> {
    if (!this.isElectron) return null;
    return window.electronAPI.files.showOpenDialog(options);
  }

  public async writeFile(filePath: string, content: string): Promise<boolean> {
    if (!this.isElectron) return false;
    return window.electronAPI.files.writeFile(filePath, content);
  }

  public async readFile(filePath: string): Promise<string | null> {
    if (!this.isElectron) return null;
    return window.electronAPI.files.readFile(filePath);
  }

  public async fileExists(filePath: string): Promise<boolean> {
    if (!this.isElectron) return false;
    return window.electronAPI.files.exists(filePath);
  }

  // Chat API methods - same interface as original ApiClient
  public async health(): Promise<ApiResponse> {
    if (!this.isElectron) {
      throw new Error('Health check only available in Electron app');
    }
    return window.electronAPI.chat.health();
  }

  public async chatCompletion(request: ChatCompletionRequest): Promise<ApiResponse<ChatCompletionResponse>> {
    if (!this.isElectron) {
      throw new Error('Chat completion only available in Electron app');
    }
    return window.electronAPI.chat.chatCompletion(request);
  }

  public async streamChatCompletion(
    request: ChatCompletionRequest,
    onChunk: (chunk: string) => void
  ): Promise<ApiResponse> {
    if (!this.isElectron) {
      throw new Error('Streaming chat completion only available in Electron app');
    }
    return window.electronAPI.chat.streamChatCompletion(request, onChunk);
  }

  public async getConfigs(): Promise<ApiResponse<{ configs: ConfigOption[] }>> {
    if (!this.isElectron) {
      throw new Error('Config access only available in Electron app');
    }
    return window.electronAPI.chat.getConfigs();
  }

  public async getModels(): Promise<ApiResponse<{ data: Array<{ id: string; config_metadata?: any }> }>> {
    if (!this.isElectron) {
      throw new Error('Model access only available in Electron app');
    }
    return window.electronAPI.chat.getModels();
  }

  public async getBranches(sessionId: string = 'default'): Promise<ApiResponse<{ 
    branches: ConversationBranch[];
    current_branch?: string;
  }>> {
    if (!this.isElectron) {
      throw new Error('Branch access only available in Electron app');
    }
    return window.electronAPI.chat.getBranches(sessionId);
  }

  public async createBranch(
    sessionId: string = 'default',
    name: string,
    parentBranchId?: string
  ): Promise<ApiResponse<{ branch: ConversationBranch }>> {
    if (!this.isElectron) {
      throw new Error('Branch creation only available in Electron app');
    }
    return window.electronAPI.chat.createBranch(sessionId, name, parentBranchId);
  }

  public async switchBranch(
    sessionId: string = 'default',
    branchId: string
  ): Promise<ApiResponse> {
    if (!this.isElectron) {
      throw new Error('Branch switching only available in Electron app');
    }
    return window.electronAPI.chat.switchBranch(sessionId, branchId);
  }

  public async deleteBranch(
    sessionId: string = 'default',
    branchId: string
  ): Promise<ApiResponse> {
    if (!this.isElectron) {
      throw new Error('Branch deletion only available in Electron app');
    }
    return window.electronAPI.chat.deleteBranch(sessionId, branchId);
  }

  public async getConversation(
    sessionId: string = 'default',
    branchId: string = 'main'
  ): Promise<ApiResponse<{ conversation: Message[] }>> {
    if (!this.isElectron) {
      throw new Error('Conversation access only available in Electron app');
    }
    return window.electronAPI.chat.getConversation(sessionId, branchId);
  }

  public async sendMessage(
    content: string,
    sessionId: string = 'default',
    branchId: string = 'main'
  ): Promise<ApiResponse<Message>> {
    if (!this.isElectron) {
      throw new Error('Message sending only available in Electron app');
    }
    return window.electronAPI.chat.sendMessage(content, sessionId, branchId);
  }

  public async executeCommand(
    command: string,
    args: string[] = []
  ): Promise<ApiResponse> {
    if (!this.isElectron) {
      throw new Error('Command execution only available in Electron app');
    }
    console.log(`🖥️  Electron API: Executing command '${command}' with args:`, args);
    const response = await window.electronAPI.chat.executeCommand(command, args);
    console.log(`🖥️  Electron API: Command '${command}' response:`, response);
    return response;
  }

  public async getSystemStats(): Promise<ApiResponse> {
    if (!this.isElectron) {
      throw new Error('System stats only available in Electron app');
    }
    return window.electronAPI.chat.getSystemStats();
  }

  public async getModelStats(): Promise<ApiResponse> {
    if (!this.isElectron) {
      throw new Error('Model stats only available in Electron app');
    }
    return window.electronAPI.chat.getModelStats();
  }

  // Event system methods
  public on(channel: string, listener: (...args: any[]) => void): void {
    if (this.isElectron) {
      window.electronAPI.events.on(channel, listener);
    }
  }

  public off(channel: string, listener: (...args: any[]) => void): void {
    if (this.isElectron) {
      window.electronAPI.events.off(channel, listener);
    }
  }

  public send(channel: string, ...args: any[]): void {
    if (this.isElectron) {
      window.electronAPI.events.send(channel, ...args);
    }
  }

  public async invoke(channel: string, ...args: any[]): Promise<any> {
    if (!this.isElectron) return null;
    return window.electronAPI.events.invoke(channel, ...args);
  }

  // Storage methods
  public async getStorageItem(key: string, defaultValue?: any): Promise<any> {
    if (!this.isElectron) {
      // Fall back to localStorage for web version
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue;
    }
    return window.electronAPI.storage.get(key, defaultValue);
  }

  public async setStorageItem(key: string, value: any): Promise<void> {
    if (!this.isElectron) {
      // Fall back to localStorage for web version
      localStorage.setItem(key, JSON.stringify(value));
      return;
    }
    return window.electronAPI.storage.set(key, value);
  }

  public async deleteStorageItem(key: string): Promise<void> {
    if (!this.isElectron) {
      localStorage.removeItem(key);
      return;
    }
    return window.electronAPI.storage.delete(key);
  }

  public async clearStorage(): Promise<void> {
    if (!this.isElectron) {
      localStorage.clear();
      return;
    }
    return window.electronAPI.storage.clear();
  }

  // Platform information
  public getPlatform(): { os: string; arch: string; version: string } {
    if (!this.isElectron) {
      return {
        os: 'web',
        arch: 'unknown',
        version: 'unknown'
      };
    }
    return window.electronAPI.platform;
  }

  // File operations for conversation save/load
  public async saveConversationToFile(conversation: Message[], filename?: string): Promise<boolean> {
    if (!this.isElectron) return false;

    try {
      const filePath = filename || await this.showSaveDialog({
        title: 'Save Conversation',
        defaultPath: `conversation-${new Date().toISOString().slice(0, 10)}.json`,
        filters: [
          { name: 'JSON Files', extensions: ['json'] },
          { name: 'All Files', extensions: ['*'] }
        ]
      });

      if (!filePath) return false;

      const content = JSON.stringify({
        version: '1.0',
        timestamp: new Date().toISOString(),
        conversation
      }, null, 2);

      return await this.writeFile(filePath, content);
    } catch (error) {
      console.error('Error saving conversation:', error);
      return false;
    }
  }

  public async loadConversationFromFile(): Promise<Message[] | null> {
    if (!this.isElectron) return null;

    try {
      const filePaths = await this.showOpenDialog({
        title: 'Load Conversation',
        filters: [
          { name: 'JSON Files', extensions: ['json'] },
          { name: 'All Files', extensions: ['*'] }
        ],
        properties: ['openFile']
      });

      if (!filePaths || filePaths.length === 0) return null;

      const content = await this.readFile(filePaths[0]);
      if (!content) return null;

      const data = JSON.parse(content);
      return data.conversation || null;
    } catch (error) {
      console.error('Error loading conversation:', error);
      return null;
    }
  }
}

// Create and export singleton instance
const electronAPI = new ElectronApiClient();
export default electronAPI;