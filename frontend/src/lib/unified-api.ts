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
   * Alias for isElectron() to match component expectations
   */
  public isElectronApp(): boolean {
    return this.electronClient.isElectronApp();
  }

  /**
   * Get the appropriate client based on environment
   */
  private getClient() {
    return this.isElectron() ? this.electronClient : this.webClient;
  }

  // Server control (Electron only)
  async startServer(configPath?: string, systemPrompt?: string): Promise<ApiResponse> {
    if (this.isElectron()) {
      return this.electronClient.startServer(configPath, systemPrompt);
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

  async testModel(configPath: string): Promise<ApiResponse<{ success: boolean; message?: string }>> {
    if (this.isElectron()) {
      return this.electronClient.testModel(configPath);
    } else {
      return { success: true, data: { success: true, message: 'Model testing not available in web version' } };
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
  async getBranches(sessionId: string): Promise<ApiResponse<{ 
    branches: ConversationBranch[];
    current_branch?: string;
  }>> {
    return this.getClient().getBranches(sessionId);
  }

  async createBranch(
    sessionId: string,
    name: string,
    parentBranchId?: string
  ): Promise<ApiResponse<{ branch: ConversationBranch }>> {
    return this.getClient().createBranch(sessionId, name, parentBranchId);
  }

  async switchBranch(
    sessionId: string,
    branchId: string
  ): Promise<ApiResponse> {
    return this.getClient().switchBranch(sessionId, branchId);
  }

  async deleteBranch(
    sessionId: string,
    branchId: string
  ): Promise<ApiResponse> {
    return this.getClient().deleteBranch(sessionId, branchId);
  }

  // Conversation management
  async getConversation(
    sessionId: string,
    branchId: string = 'main'
  ): Promise<ApiResponse<{ conversation: Message[] }>> {
    return this.getClient().getConversation(sessionId, branchId);
  }

  async regenNode(params: { assistantId?: string; userMessageId?: string; prompt?: string; sessionId?: string; branchId?: string; historyMode?: 'none'|'last_user'|'full' }): Promise<ApiResponse<{ assistant: { id: string; content: string } }>> {
    // Fill session/branch if missing
    let sessionId = params.sessionId;
    let branchId = params.branchId;
    try {
      if (!sessionId || !branchId) {
        const storeMod: any = await import('./store');
        const st = storeMod.useChatStore?.getState?.();
        sessionId = sessionId || st?.getCurrentSessionId?.() || st?.currentSessionId;
        branchId = branchId || st?.currentBranchId || 'main';
      }
    } catch {}

    const payload = {
      assistantId: params.assistantId,
      userMessageId: params.userMessageId,
      prompt: params.prompt,
      sessionId,
      branchId,
      historyMode: params.historyMode || 'last_user',
    };

    const client: any = this.getClient();
    if (typeof client.regenNode === 'function') {
      return client.regenNode(payload);
    }
    // Web fallback
    return this.webClient.regenNode({
      assistantId: payload.assistantId,
      userMessageId: payload.userMessageId,
      prompt: payload.prompt,
      sessionId: payload.sessionId,
      branchId: payload.branchId,
      historyMode: payload.historyMode,
    });
  }

  async listConversations(sessionId: string): Promise<ApiResponse<{ conversations: any[] }>> {
    // Scope strictly to the provided sessionId to prevent cross-session leakage
    try {
      const key = `conversations_${sessionId}`;
      const conversations = await this.getStorageItem(key, []);

      // Normalize and sort by last modified (newest first)
      const normalized = Array.isArray(conversations) ? conversations : [];
      normalized.sort((a, b) =>
        new Date(b.lastModified || b.updatedAt || 0).getTime() -
        new Date(a.lastModified || a.updatedAt || 0).getTime()
      );

      return { success: true, data: { conversations: normalized } };
    } catch (error) {
      console.error('Error listing conversations:', error);
      return { success: false, error: 'Failed to list conversations', data: { conversations: [] } };
    }
  }

  /**
   * Lists all conversations across all sessions
   * 
   * @returns ApiResponse with conversations from all sessions with sessionId attached
   */
  async listAllConversations(): Promise<ApiResponse<{ conversations: any[] }>> {
    try {
      // Get all keys from storage
      const allKeys = await this.getAllStorageKeys();
      
      // Filter keys that match conversations_* pattern
      const conversationKeys = allKeys.filter(key => key.startsWith('conversations_'));
      
      // Get conversations from all session keys
      const allConversations = [];
      for (const key of conversationKeys) {
        // Extract sessionId from the key (format: conversations_${sessionId})
        const sessionId = key.replace('conversations_', '');
        
        // Get conversations for this session
        const conversations = await this.getStorageItem(key, []);
        
        // Add sessionId to each conversation and add to master list
        if (Array.isArray(conversations)) {
          const conversationsWithSession = conversations.map(conv => ({
            ...conv,
            sessionId: sessionId // Add sessionId to each conversation
          }));
          allConversations.push(...conversationsWithSession);
        }
      }
      
      // Sort all conversations by lastModified (newest first)
      allConversations.sort((a, b) =>
        new Date(b.lastModified || b.updatedAt || 0).getTime() -
        new Date(a.lastModified || a.updatedAt || 0).getTime()
      );
      
      return { 
        success: true, 
        data: { 
          conversations: allConversations 
        } 
      };
    } catch (error) {
      console.error('Error listing all conversations:', error);
      return { 
        success: false, 
        error: 'Failed to list all conversations', 
        data: { conversations: [] } 
      };
    }
  }

  /**
   * Returns flattened branch nodes for a session.
   * Each node represents a conversation+branch pair with summary metadata.
   */
  async listSessionNodes(sessionId: string): Promise<ApiResponse<{ nodes: any[] }>> {
    try {
      const key = `conversations_${sessionId}`;
      const conversations = await this.getStorageItem(key, []);
      const nodes: any[] = [];
      if (Array.isArray(conversations)) {
        for (const conv of conversations) {
          const convId = conv.id || conv.filename;
          const convName = conv.name || conv.filename || 'Untitled Conversation';
          const convLast = conv.lastModified || conv.updatedAt || new Date().toISOString();
          const convPreview = conv.preview || 'No preview available';
          const convCount = conv.messageCount || 0;

          // Always include main/root node even if no branches array is present
          nodes.push({
            conversationId: convId,
            branchId: 'main',
            name: convName,
            lastActive: convLast,
            messageCount: convCount,
            preview: convPreview,
            sessionId,
            isRoot: true,
          });

          const branches = Array.isArray(conv.branches) ? conv.branches : [];
          for (const br of branches) {
            // Skip duplicating main if present in branches
            if (br.id === 'main') continue;
            nodes.push({
              conversationId: convId,
              branchId: br.id,
              name: br.name || br.id,
              lastActive: br.lastActive || convLast,
              messageCount: br.messageCount || 0,
              preview: br.preview || '',
              parentId: br.parentId,
              sessionId,
              isRoot: false,
            });
          }
        }
      }
      // Sort by lastActive desc, keeping current order for ties
      nodes.sort((a, b) => new Date(b.lastActive || 0).getTime() - new Date(a.lastActive || 0).getTime());
      return { success: true, data: { nodes } };
    } catch (error) {
      console.error('Error listing session nodes:', error);
      return { success: false, error: 'Failed to list session nodes', data: { nodes: [] } };
    }
  }

  async loadConversation(
    sessionId: string, 
    conversationId: string,
    targetBranchId?: string
  ): Promise<ApiResponse<{ messages: Message[]; nodeGraph?: any; currentBranchId?: string }>> {
    if (this.isElectron()) {
      // For Electron, use storage fallback for now until backend method is implemented
      try {
        const conversationKey = `conversation_${sessionId}_${conversationId}`;
        const conversationData = await this.getStorageItem(conversationKey);
        
        if (!conversationData) {
          return {
            success: false,
            error: 'Conversation not found'
          };
        }

        const messages = conversationData.messages || conversationData.conversation || [];
        const nodeGraph = conversationData.nodeGraph || undefined;
        const currentBranchId = conversationData.currentBranchId || conversationData.current_branch || undefined;

        // If targetBranchId is provided, we would normally load into that branch
        // For now, just return the messages
        return {
          success: true,
          data: { messages, nodeGraph, currentBranchId }
        };
      } catch (error) {
        console.error('Error loading conversation:', error);
        return {
          success: false,
          error: 'Failed to load conversation'
        };
      }
    } else {
      // Web fallback - load from localStorage
      try {
        const conversationKey = `conversation_${sessionId}_${conversationId}`;
        const storedConversation = localStorage.getItem(conversationKey);
        
        if (!storedConversation) {
          return {
            success: false,
            error: 'Conversation not found'
          };
        }

        const conversationData = JSON.parse(storedConversation);
        const messages = conversationData.messages || conversationData.conversation || [];
        const nodeGraph = conversationData.nodeGraph || undefined;
        const currentBranchId = conversationData.currentBranchId || conversationData.current_branch || undefined;

        // If targetBranchId is provided, we would normally load into that branch
        // For now, just return the messages
        return {
          success: true,
          data: { messages, nodeGraph, currentBranchId }
        };
      } catch (error) {
        console.error('Error loading conversation:', error);
        return {
          success: false,
          error: 'Failed to load conversation'
        };
      }
    }
  }

  async deleteConversation(
    sessionId: string,
    conversationId: string
  ): Promise<ApiResponse> {
    if (this.isElectron()) {
      // For Electron, use storage fallback for now until backend method is implemented
      try {
        const conversationKey = `conversation_${sessionId}_${conversationId}`;
        const conversationsKey = `conversations_${sessionId}`;
        
        // Remove the conversation data
        await this.deleteStorageItem(conversationKey);
        
        // Update the conversations list
        const storedConversations = await this.getStorageItem(conversationsKey, []);
        const updatedConversations = storedConversations.filter((conv: any) => conv.id !== conversationId);
        await this.setStorageItem(conversationsKey, updatedConversations);
        
        return {
          success: true,
          message: 'Conversation deleted successfully'
        };
      } catch (error) {
        console.error('Error deleting conversation:', error);
        return {
          success: false,
          error: 'Failed to delete conversation'
        };
      }
    } else {
      // Web fallback - remove from localStorage
      try {
        const conversationKey = `conversation_${sessionId}_${conversationId}`;
        const conversationsKey = `conversations_${sessionId}`;
        
        // Remove the conversation data
        localStorage.removeItem(conversationKey);
        
        // Update the conversations list
        const storedConversations = localStorage.getItem(conversationsKey);
        if (storedConversations) {
          const conversations = JSON.parse(storedConversations);
          const updatedConversations = conversations.filter((conv: any) => conv.id !== conversationId);
          localStorage.setItem(conversationsKey, JSON.stringify(updatedConversations));
        }
        
        return {
          success: true,
          message: 'Conversation deleted successfully'
        };
      } catch (error) {
        console.error('Error deleting conversation:', error);
        return {
          success: false,
          error: 'Failed to delete conversation'
        };
      }
    }
  }

  async sendMessage(
    content: string,
    sessionId: string,
    branchId: string = 'main'
  ): Promise<ApiResponse<Message>> {
    return this.getClient().sendMessage(content, sessionId, branchId);
  }

  // Command execution
  async executeCommand(
    command: string,
    args: string[] = []
  ): Promise<ApiResponse> {
    // Attach session + branch automatically when available
    let sessionId: string | undefined;
    let branchId: string | undefined;
    try {
      const storeMod: any = await import('./store');
      const st = storeMod.useChatStore?.getState?.();
      if (st) {
        sessionId = st.getCurrentSessionId?.() || st.currentSessionId;
        branchId = st.currentBranchId || 'main';
      }
    } catch {}
    return this.getClient().executeCommand(command, args, sessionId, branchId);
  }

  async executeCommandAdvanced(
    command: string,
    args: string[] = [],
    extras?: { sessionId?: string; branchId?: string; messageId?: string; index?: number; payload?: string }
  ): Promise<ApiResponse> {
    // Auto-fill session/branch if not provided
    let sessionId = extras?.sessionId;
    let branchId = extras?.branchId;
    try {
      if (!sessionId || !branchId) {
        const storeMod: any = await import('./store');
        const st = storeMod.useChatStore?.getState?.();
        sessionId = sessionId || st?.getCurrentSessionId?.() || st?.currentSessionId;
        branchId = branchId || st?.currentBranchId || 'main';
      }
    } catch {}
    const payload = {
      sessionId,
      branchId,
      messageId: extras?.messageId,
      index: extras?.index,
      payload: extras?.payload,
    };
    const client: any = this.getClient();
    if (typeof client.executeCommandAdvanced === 'function') {
      return client.executeCommandAdvanced(command, args, payload);
    }
    // Fallback to basic executeCommand if advanced not available
    return client.executeCommand(command, args, sessionId, branchId);
  }

  // System monitoring
  async getSystemStats(sessionId?: string): Promise<ApiResponse> {
    return this.getClient().getSystemStats(sessionId);
  }

  async getModelStats(): Promise<ApiResponse> {
    return this.getClient().getModelStats();
  }

  async clearModel(): Promise<ApiResponse> {
    return this.getClient().clearModel();
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

  // Enhanced conversation management with automatic persistence
  async saveConversation(
    sessionId: string,
    conversationId: string,
    conversationData: any
  ): Promise<ApiResponse> {
    try {
      // Save individual conversation (attach node graph if available from store)
      const conversationKey = `conversation_${sessionId}_${conversationId}`;
      let payload = conversationData;
      try {
        const storeMod: any = await import('./store');
        const storeState = storeMod.useChatStore?.getState?.();
        if (storeState) {
          const nodeGraph = {
            nodes: storeState.messageNodes?.[conversationId] || {},
            timelines: storeState.branchTimelines?.[conversationId] || {},
            heads: storeState.branchHeads?.[conversationId] || {},
            tombstones: storeState.branchTombstones?.[conversationId] || {},
            merges: storeState.merges?.[conversationId] || [],
          };
          if (Object.keys(nodeGraph.nodes).length > 0) {
            payload = { ...conversationData, nodeGraph };
          }
        }
      } catch (e) {
        console.debug('saveConversation: nodeGraph not attached', e?.toString?.());
      }
      await this.setStorageItem(conversationKey, payload);
      
      // Update conversations list
      const conversationsKey = `conversations_${sessionId}`;
      const existingConversations = await this.getStorageItem(conversationsKey, []);
      
      const conversationIndex = existingConversations.findIndex((c: any) => c.id === conversationId);

      // Derive message count and preview from branches first, falling back to flat messages
      let messageCount = Array.isArray(conversationData?.messages) ? conversationData.messages.length : 0;
      let preview = 'No messages';
      const branchSummaries: Array<{
        id: string;
        name?: string;
        messageCount: number;
        lastActive?: string;
        preview?: string;
        parentId?: string;
      }> = [];
      
      if (conversationData?.branches && typeof conversationData.branches === 'object') {
        let latestMsg: any = null;
        let total = 0;
        for (const [branchId, branch] of Object.entries(conversationData.branches) as any) {
          const msgs: any[] = Array.isArray(branch?.messages) ? branch.messages : [];
          total += msgs.length;
          if (msgs.length > 0) {
            const candidate = msgs[msgs.length - 1];
            // Choose the message with the newest timestamp when available
            const candidateTs = typeof candidate?.timestamp === 'number' ? candidate.timestamp : 0;
            const latestTs = typeof latestMsg?.timestamp === 'number' ? latestMsg.timestamp : -1;
            if (candidateTs >= latestTs) {
              latestMsg = candidate;
            }
          }
          // Per-branch summary
          const last = msgs.length > 0 ? msgs[msgs.length - 1] : null;
          const meta = (branch && typeof branch === 'object' && branch.metadata) ? branch.metadata : {};
          branchSummaries.push({
            id: String(branchId),
            name: meta?.name,
            parentId: meta?.parentId,
            messageCount: msgs.length,
            lastActive: last?.timestamp ? new Date(last.timestamp).toISOString() : (meta?.lastActive || undefined),
            preview: last?.content ? String(last.content).slice(0, 100) : undefined,
          });
        }
        messageCount = total;
        if (latestMsg?.content) {
          const text = String(latestMsg.content);
          preview = text.slice(0, 100);
        }
      } else if (messageCount > 0) {
        const last = conversationData.messages[conversationData.messages.length - 1];
        if (last?.content) preview = String(last.content).slice(0, 100);
      }

      // Do not index empty conversations in the summary list
      if (messageCount === 0) {
        return {
          success: true,
          message: 'Conversation saved (empty, not indexed)'
        };
      }

      // Try to get current branch id for pinning/highlighting
      let currentBranchId: string | undefined = conversationData?.currentBranchId || conversationData?.current_branch;
      if (!currentBranchId) {
        try {
          const branchesResp: any = await this.getBranches(sessionId);
          if (branchesResp?.success && branchesResp.data?.current_branch) {
            currentBranchId = branchesResp.data.current_branch;
          }
        } catch {}
      }

      const conversationEntry: any = {
        id: conversationId,
        name: conversationData.title || 'Untitled Conversation',
        lastModified: conversationData.updatedAt || new Date().toISOString(),
        messageCount,
        preview,
        filename: conversationId
      };
      // Persist branch summaries when available
      if (branchSummaries.length > 0) {
        conversationEntry.branches = branchSummaries;
      }
      if (currentBranchId) {
        conversationEntry.currentBranchId = currentBranchId;
      }

      if (conversationIndex >= 0) {
        existingConversations[conversationIndex] = conversationEntry;
      } else {
        existingConversations.push(conversationEntry);
      }

      await this.setStorageItem(conversationsKey, existingConversations);

      return {
        success: true,
        message: 'Conversation saved successfully'
      };
    } catch (error) {
      console.error('Failed to save conversation:', error);
      return {
        success: false,
        error: 'Failed to save conversation'
      };
    }
  }

  async clearStorage(): Promise<void> {
    if (this.isElectron()) {
      return this.electronClient.clearStorage();
    } else {
      localStorage.clear();
    }
  }

  async getAllStorageKeys(): Promise<string[]> {
    if (this.isElectron()) {
      return this.electronClient.getAllStorageKeys();
    } else {
      const keys: string[] = [];
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key) keys.push(key);
      }
      return keys;
    }
  }

  async resetWelcomeSettings(): Promise<ApiResponse> {
    if (this.isElectron()) {
      return this.electronClient.resetWelcomeSettings();
    } else {
      // Web fallback: clear welcome-related localStorage items
      const welcomeKeys = ['hasCompletedWelcome', 'selectedConfig', 'systemPrompt', 'enableWelcomeCaching'];
      welcomeKeys.forEach(key => localStorage.removeItem(key));
      return { success: true, message: 'Welcome settings reset' };
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


  // Python environment setup methods (Electron-specific)
  async isEnvironmentSetupNeeded(): Promise<boolean> {
    if (this.isElectron()) {
      return this.electronClient.isEnvironmentSetupNeeded();
    } else {
      return false;
    }
  }

  async getPythonUserDataPath(): Promise<string> {
    if (this.isElectron()) {
      return this.electronClient.getPythonUserDataPath();
    } else {
      return '';
    }
  }

  async cancelPythonSetup(): Promise<void> {
    if (this.isElectron()) {
      return this.electronClient.cancelPythonSetup();
    }
  }

  async rebuildPythonEnvironment(): Promise<{ success: boolean; message: string }> {
    if (this.isElectron()) {
      return this.electronClient.rebuildPythonEnvironment();
    } else {
      return { success: false, message: 'Rebuild not available in web version' };
    }
  }

  async removeEnvironment(): Promise<{ success: boolean; message: string }> {
    if (this.isElectron()) {
      return this.electronClient.removeEnvironment();
    } else {
      return { success: false, message: 'Remove environment not available in web version' };
    }
  }

  async getSystemChangeInfo(): Promise<{ hasChanged: boolean; changes: string[]; shouldRebuild: boolean } | null> {
    if (this.isElectron()) {
      return this.electronClient.getSystemChangeInfo();
    } else {
      return null;
    }
  }

  // System detection methods
  async getSystemCapabilities(): Promise<any> {
    if (this.isElectron()) {
      return this.electronClient.getSystemCapabilities();
    } else {
      // Return null for web version - capabilities detection requires system access
      return null;
    }
  }

  async getSystemInfo(): Promise<any> {
    if (this.isElectron()) {
      return this.electronClient.getSystemInfo();
    } else {
      return null;
    }
  }

  // Secure API Key management (Electron only)
  async storeApiKey(providerId: string, keyValue: string, isActive: boolean = true): Promise<ApiResponse> {
    if (this.isElectron()) {
      return this.electronClient.storeApiKey(providerId, keyValue, isActive);
    } else {
      return { success: false, message: 'API key storage only available in Electron version' };
    }
  }

  async getApiKey(providerId: string): Promise<ApiResponse> {
    if (this.isElectron()) {
      return this.electronClient.getApiKey(providerId);
    } else {
      return { success: false, message: 'API key access only available in Electron version' };
    }
  }

  async getAllApiKeys(): Promise<ApiResponse> {
    if (this.isElectron()) {
      return this.electronClient.getAllApiKeys();
    } else {
      return { success: false, message: 'API key access only available in Electron version' };
    }
  }

  async removeApiKey(providerId: string): Promise<ApiResponse> {
    if (this.isElectron()) {
      return this.electronClient.removeApiKey(providerId);
    } else {
      return { success: false, message: 'API key removal only available in Electron version' };
    }
  }

  async updateApiKeyStatus(providerId: string, updates: any): Promise<ApiResponse> {
    if (this.isElectron()) {
      return this.electronClient.updateApiKeyStatus(providerId, updates);
    } else {
      return { success: false, message: 'API key updates only available in Electron version' };
    }
  }

  async validateApiKeyWithOumi(providerId: string): Promise<ApiResponse> {
    if (this.isElectron()) {
      return this.electronClient.validateApiKeyWithOumi(providerId);
    } else {
      return { success: false, message: 'API key validation only available in Electron version' };
    }
  }

  async getEnvironmentSystemInfo(): Promise<any> {
    if (this.isElectron()) {
      // First try the full environment system info (requires Oumi backend)
      try {
        const fullInfo = await this.electronClient.getEnvironmentSystemInfo();
        if (fullInfo && fullInfo.platform && fullInfo.platform !== 'unknown') {
          return fullInfo;
        }
      } catch (error) {
        console.debug('Full system info not available, trying basic fallback...');
      }
      
      // Fallback to basic system info (uses lightweight Python script)
      try {
        const basicInfo = await this.electronClient.getBasicSystemInfo();
        if (basicInfo && basicInfo.platform && basicInfo.platform !== 'unknown') {
          return basicInfo;
        }
      } catch (error) {
        console.warn('Basic system info detection also failed:', error);
      }
      
      return null;
    } else {
      return null;
    }
  }

  onSetupProgress(callback: (progress: any) => void): void {
    if (this.isElectron()) {
      this.electronClient.onSetupProgress(callback);
    }
  }

  offSetupProgress(callback: (progress: any) => void): void {
    if (this.isElectron()) {
      this.electronClient.offSetupProgress(callback);
    }
  }

  onSetupError(callback: (error: string) => void): void {
    if (this.isElectron()) {
      this.electronClient.onSetupError(callback);
    }
  }

  offSetupError(callback: (error: string) => void): void {
    if (this.isElectron()) {
      this.electronClient.offSetupError(callback);
    }
  }
}

// Create and export singleton instance
const unifiedApiClient = new UnifiedApiClient();
export default unifiedApiClient;
