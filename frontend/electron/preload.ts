/**
 * Electron preload script - secure IPC bridge between main and renderer processes
 */

import { contextBridge, ipcRenderer, IpcRendererEvent } from 'electron';

// API interface for renderer process
export interface ElectronAPI {
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
    start: (configPath?: string, systemPrompt?: string) => Promise<any>;
    stop: () => Promise<any>;
    restart: () => Promise<any>;
    status: () => Promise<any>;
  };

  // Chat API operations (replacing HTTP calls)
  chat: {
    // Health and system info
    health: () => Promise<any>;
    getSystemStats: () => Promise<any>;
    getModelStats: () => Promise<any>;

    // Chat operations
    chatCompletion: (request: any) => Promise<any>;
    streamChatCompletion: (request: any, onChunk: (chunk: string) => void) => Promise<any>;

    // Configuration
    getConfigs: () => Promise<any>;
    getModels: () => Promise<any>;

    // Branch management
    getBranches: (sessionId?: string) => Promise<any>;
    createBranch: (sessionId: string, name: string, parentBranchId?: string) => Promise<any>;
    switchBranch: (sessionId: string, branchId: string) => Promise<any>;
    deleteBranch: (sessionId: string, branchId: string) => Promise<any>;

    // Conversation management
    getConversation: (sessionId?: string, branchId?: string) => Promise<any>;
    sendMessage: (content: string, sessionId?: string, branchId?: string) => Promise<any>;

    // Command execution
    executeCommand: (command: string, args?: string[]) => Promise<any>;
  };

  // Real-time event system (replacing WebSocket)
  events: {
    // Listen for events from main process
    on: (channel: string, listener: (...args: any[]) => void) => void;
    off: (channel: string, listener: (...args: any[]) => void) => void;
    
    // Send events to main process
    send: (channel: string, ...args: any[]) => void;
    
    // Invoke and wait for response
    invoke: (channel: string, ...args: any[]) => Promise<any>;
  };

  // Storage operations
  storage: {
    get: (key: string, defaultValue?: any) => Promise<any>;
    set: (key: string, value: any) => Promise<void>;
    delete: (key: string) => Promise<void>;
    clear: () => Promise<void>;
    resetWelcomeSettings: () => Promise<any>;
  };

  // Platform information
  platform: {
    os: string;
    arch: string;
    version: string;
  };

  // Menu message handlers
  onMenuMessage: (channel: string, callback: (...args: any[]) => void) => void;
  removeMenuListener: (channel: string, callback: (...args: any[]) => void) => void;
}

// Create the API object
const electronAPI: ElectronAPI = {
  app: {
    getVersion: () => ipcRenderer.invoke('app:get-version'),
    quit: () => ipcRenderer.send('app:quit'),
    reload: () => ipcRenderer.send('app:reload'),
    toggleDevTools: () => ipcRenderer.send('app:toggle-dev-tools'),
    toggleFullScreen: () => ipcRenderer.send('app:toggle-full-screen'),
    zoom: (direction) => ipcRenderer.send('app:zoom', direction)
  },

  files: {
    showSaveDialog: (options) => ipcRenderer.invoke('files:show-save-dialog', options),
    showOpenDialog: (options) => ipcRenderer.invoke('files:show-open-dialog', options),
    writeFile: (filePath, content) => ipcRenderer.invoke('files:write-file', filePath, content),
    readFile: (filePath) => ipcRenderer.invoke('files:read-file', filePath),
    exists: (filePath) => ipcRenderer.invoke('files:exists', filePath)
  },

  server: {
    start: (configPath?: string, systemPrompt?: string) => ipcRenderer.invoke('server:start', configPath, systemPrompt),
    stop: () => ipcRenderer.invoke('server:stop'),
    restart: () => ipcRenderer.invoke('server:restart'),
    status: () => ipcRenderer.invoke('server:status')
  },

  chat: {
    health: () => ipcRenderer.invoke('chat:health'),
    getSystemStats: () => ipcRenderer.invoke('chat:get-system-stats'),
    getModelStats: () => ipcRenderer.invoke('chat:get-model-stats'),

    chatCompletion: (request) => ipcRenderer.invoke('chat:completion', request),
    streamChatCompletion: async (request, onChunk) => {
      // Set up stream listener
      const streamId = Math.random().toString(36).substr(2, 9);
      
      const cleanup = () => {
        ipcRenderer.removeAllListeners(`chat:stream-chunk:${streamId}`);
        ipcRenderer.removeAllListeners(`chat:stream-end:${streamId}`);
        ipcRenderer.removeAllListeners(`chat:stream-error:${streamId}`);
      };

      return new Promise((resolve, reject) => {
        // Listen for chunks
        ipcRenderer.on(`chat:stream-chunk:${streamId}`, (_, chunk) => {
          onChunk(chunk);
        });

        // Listen for stream end
        ipcRenderer.on(`chat:stream-end:${streamId}`, (_, result) => {
          cleanup();
          resolve(result);
        });

        // Listen for errors
        ipcRenderer.on(`chat:stream-error:${streamId}`, (_, error) => {
          cleanup();
          reject(new Error(error));
        });

        // Start the stream
        ipcRenderer.invoke('chat:stream-completion', request, streamId).catch(reject);
      });
    },

    getConfigs: () => ipcRenderer.invoke('chat:get-configs'),
    getModels: () => ipcRenderer.invoke('chat:get-models'),

    getBranches: (sessionId = 'default') => ipcRenderer.invoke('chat:get-branches', sessionId),
    createBranch: (sessionId, name, parentBranchId) => 
      ipcRenderer.invoke('chat:create-branch', sessionId, name, parentBranchId),
    switchBranch: (sessionId, branchId) => 
      ipcRenderer.invoke('chat:switch-branch', sessionId, branchId),
    deleteBranch: (sessionId, branchId) => 
      ipcRenderer.invoke('chat:delete-branch', sessionId, branchId),

    getConversation: (sessionId = 'default', branchId = 'main') => 
      ipcRenderer.invoke('chat:get-conversation', sessionId, branchId),
    sendMessage: (content, sessionId = 'default', branchId = 'main') => 
      ipcRenderer.invoke('chat:send-message', content, sessionId, branchId),

    executeCommand: (command, args = []) => 
      ipcRenderer.invoke('chat:execute-command', command, args)
  },

  events: {
    on: (channel: string, listener: (...args: any[]) => void) => {
      const wrappedListener = (_: IpcRendererEvent, ...args: any[]) => listener(...args);
      ipcRenderer.on(channel, wrappedListener);
    },
    
    off: (channel: string, listener: (...args: any[]) => void) => {
      ipcRenderer.removeListener(channel, listener);
    },
    
    send: (channel: string, ...args: any[]) => {
      ipcRenderer.send(channel, ...args);
    },
    
    invoke: (channel: string, ...args: any[]) => {
      return ipcRenderer.invoke(channel, ...args);
    }
  },

  storage: {
    get: (key, defaultValue) => ipcRenderer.invoke('storage:get', key, defaultValue),
    set: (key, value) => ipcRenderer.invoke('storage:set', key, value),
    delete: (key) => ipcRenderer.invoke('storage:delete', key),
    clear: () => ipcRenderer.invoke('storage:clear'),
    resetWelcomeSettings: () => ipcRenderer.invoke('storage:reset-welcome-settings')
  },

  platform: {
    os: process.platform,
    arch: process.arch,
    version: process.version
  },

  // Menu message handlers
  onMenuMessage: (channel: string, callback: (...args: any[]) => void) => {
    ipcRenderer.on(channel, callback);
  },

  removeMenuListener: (channel: string, callback: (...args: any[]) => void) => {
    ipcRenderer.removeListener(channel, callback);
  }
};

// Expose the API to the renderer process
contextBridge.exposeInMainWorld('electronAPI', electronAPI);

// Type declaration for global window object (used in renderer)
declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}

// Validate that context isolation is working
window.addEventListener('DOMContentLoaded', () => {
  const replaceText = (selector: string, text: string) => {
    const element = document.getElementById(selector);
    if (element) element.innerText = text;
  };

  for (const dependency of ['chrome', 'node', 'electron']) {
    replaceText(`${dependency}-version`, (process.versions as any)[dependency]);
  }
});

export {};