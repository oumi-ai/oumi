/**
 * IPC Handlers - Bridge between Electron renderer and Python backend
 */

import { ipcMain, dialog, app, BrowserWindow } from 'electron';
import { promises as fs } from 'fs';
import Store from 'electron-store';
import log from 'electron-log';
import { PythonServerManager, DownloadProgress, DownloadErrorEvent } from './python-manager';

const store = new Store();

/**
 * Set up all IPC handlers
 */
export function setupIpcHandlers(pythonManager: PythonServerManager): void {
  // App control handlers
  setupAppHandlers();
  
  // File system handlers
  setupFileHandlers();
  
  // Storage handlers
  setupStorageHandlers();
  
  // Chat API handlers (proxy to Python backend)
  setupChatHandlers(pythonManager);
  
  // Download progress handlers
  setupDownloadHandlers(pythonManager);
  
  log.info('IPC handlers set up successfully');
}

/**
 * App control handlers
 */
function setupAppHandlers(): void {
  ipcMain.handle('app:get-version', () => {
    return app.getVersion();
  });

  ipcMain.on('app:quit', () => {
    app.quit();
  });

  ipcMain.on('app:reload', () => {
    const focusedWindow = BrowserWindow.getFocusedWindow();
    if (focusedWindow) {
      focusedWindow.reload();
    }
  });

  ipcMain.on('app:toggle-dev-tools', () => {
    const focusedWindow = BrowserWindow.getFocusedWindow();
    if (focusedWindow) {
      focusedWindow.webContents.toggleDevTools();
    }
  });

  ipcMain.on('app:toggle-full-screen', () => {
    const focusedWindow = BrowserWindow.getFocusedWindow();
    if (focusedWindow) {
      focusedWindow.setFullScreen(!focusedWindow.isFullScreen());
    }
  });

  ipcMain.on('app:zoom', (_, direction: 'in' | 'out' | 'reset') => {
    const focusedWindow = BrowserWindow.getFocusedWindow();
    if (focusedWindow) {
      const webContents = focusedWindow.webContents;
      const currentZoom = webContents.getZoomLevel();

      switch (direction) {
        case 'in':
          webContents.setZoomLevel(Math.min(currentZoom + 0.5, 3));
          break;
        case 'out':
          webContents.setZoomLevel(Math.max(currentZoom - 0.5, -3));
          break;
        case 'reset':
          webContents.setZoomLevel(0);
          break;
      }
    }
  });
}

/**
 * File system handlers
 */
function setupFileHandlers(): void {
  ipcMain.handle('files:show-save-dialog', async (_, options) => {
    const focusedWindow = BrowserWindow.getFocusedWindow();
    if (!focusedWindow) return null;

    const result = await dialog.showSaveDialog(focusedWindow, {
      filters: [
        { name: 'JSON Files', extensions: ['json'] },
        { name: 'Text Files', extensions: ['txt'] },
        { name: 'All Files', extensions: ['*'] }
      ],
      ...options
    });

    return result.canceled ? null : result.filePath;
  });

  ipcMain.handle('files:show-open-dialog', async (_, options) => {
    const focusedWindow = BrowserWindow.getFocusedWindow();
    if (!focusedWindow) return null;

    const result = await dialog.showOpenDialog(focusedWindow, {
      filters: [
        { name: 'JSON Files', extensions: ['json'] },
        { name: 'Text Files', extensions: ['txt'] },
        { name: 'All Files', extensions: ['*'] }
      ],
      properties: ['openFile'],
      ...options
    });

    return result.canceled ? null : result.filePaths;
  });

  ipcMain.handle('files:write-file', async (_, filePath: string, content: string) => {
    try {
      await fs.writeFile(filePath, content, 'utf8');
      return true;
    } catch (error) {
      log.error('File write error:', error);
      return false;
    }
  });

  ipcMain.handle('files:read-file', async (_, filePath: string) => {
    try {
      const content = await fs.readFile(filePath, 'utf8');
      return content;
    } catch (error) {
      log.error('File read error:', error);
      return null;
    }
  });

  ipcMain.handle('files:exists', async (_, filePath: string) => {
    try {
      await fs.access(filePath);
      return true;
    } catch {
      return false;
    }
  });
}

/**
 * Storage handlers
 */
function setupStorageHandlers(): void {
  ipcMain.handle('storage:get', (_, key: string, defaultValue?: any) => {
    return (store as any).get(key, defaultValue);
  });

  ipcMain.handle('storage:set', (_, key: string, value: any) => {
    (store as any).set(key, value);
  });

  ipcMain.handle('storage:delete', (_, key: string) => {
    (store as any).delete(key);
  });

  ipcMain.handle('storage:clear', () => {
    (store as any).clear();
  });

  // Reset welcome screen settings
  ipcMain.handle('storage:reset-welcome-settings', () => {
    const welcomeKeys = ['hasCompletedWelcome', 'selectedConfig', 'systemPrompt', 'enableWelcomeCaching'];
    welcomeKeys.forEach(key => {
      (store as any).delete(key);
    });
    log.info('Welcome screen settings reset');
    return { success: true };
  });
}

/**
 * Chat API handlers - proxy to Python backend
 */
function setupChatHandlers(pythonManager: PythonServerManager): void {
  // Server control handlers
  ipcMain.handle('server:start', async (_, configPath?: string, systemPrompt?: string) => {
    try {
      if (pythonManager.isServerRunning()) {
        return { success: true, url: pythonManager.getServerUrl() };
      }
      
      // Set the config path if provided
      if (configPath) {
        pythonManager.setConfigPath(configPath);
        log.info('Set server config path to:', configPath);
      }
      
      // Set the system prompt if provided
      if (systemPrompt) {
        pythonManager.setSystemPrompt(systemPrompt);
        log.info('Set server system prompt to:', systemPrompt.substring(0, 100) + '...');
      }
      
      const serverUrl = await pythonManager.start();
      return { success: true, url: serverUrl };
    } catch (error) {
      log.error('Failed to start server:', error);
      return { 
        success: false, 
        message: error instanceof Error ? error.message : 'Failed to start server'
      };
    }
  });

  ipcMain.handle('server:stop', async () => {
    try {
      await pythonManager.stop();
      return { success: true };
    } catch (error) {
      log.error('Failed to stop server:', error);
      return { 
        success: false, 
        message: error instanceof Error ? error.message : 'Failed to stop server'
      };
    }
  });

  ipcMain.handle('server:restart', async () => {
    try {
      const serverUrl = await pythonManager.restart();
      return { success: true, url: serverUrl };
    } catch (error) {
      log.error('Failed to restart server:', error);
      return { 
        success: false, 
        message: error instanceof Error ? error.message : 'Failed to restart server'
      };
    }
  });

  ipcMain.handle('server:status', () => {
    return {
      running: pythonManager.isServerRunning(),
      url: pythonManager.getServerUrl(),
      port: pythonManager.getPort()
    };
  });

  ipcMain.handle('server:test-model', async (_, configPath: string) => {
    try {
      log.info('Starting model test for config:', configPath);
      const result = await pythonManager.testModel(configPath);
      log.info('Model test result:', result);
      return result;
    } catch (error) {
      log.error('Model test error:', error);
      return { 
        success: false, 
        message: error instanceof Error ? error.message : 'Unknown error during model test'
      };
    }
  });

  const getBaseUrl = () => pythonManager.getServerUrl();

  // Helper function to make HTTP requests to Python backend
  async function proxyToPython(endpoint: string, options: RequestInit = {}): Promise<any> {
    const url = `${getBaseUrl()}${endpoint}`;
    
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
      log.error(`Proxy error for ${endpoint}:`, error);
      return {
        success: false,
        message: error instanceof Error ? error.message : 'Network error',
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  // Health and system info
  ipcMain.handle('chat:health', () => proxyToPython('/health'));
  ipcMain.handle('chat:get-system-stats', () => proxyToPython('/v1/oumi/system_stats'));
  ipcMain.handle('chat:get-model-stats', () => proxyToPython('/v1/models'));

  // Chat completion
  ipcMain.handle('chat:completion', (_, request) => 
    proxyToPython('/v1/chat/completions', {
      method: 'POST',
      body: JSON.stringify(request),
    })
  );

  // Streaming chat completion
  ipcMain.handle('chat:stream-completion', async (event, request, streamId) => {
    const url = `${getBaseUrl()}/v1/chat/completions`;
    
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...request, stream: true }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        event.sender.send(`chat:stream-error:${streamId}`, errorData.message || response.statusText);
        return;
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        event.sender.send(`chat:stream-error:${streamId}`, 'No response body');
        return;
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
              event.sender.send(`chat:stream-end:${streamId}`, { success: true });
              return;
            }
            
            try {
              const parsed = JSON.parse(data);
              const content = parsed.choices?.[0]?.delta?.content;
              if (content) {
                event.sender.send(`chat:stream-chunk:${streamId}`, content);
              }
            } catch (e) {
              log.warn('Failed to parse SSE data:', data);
            }
          }
        }
      }

      event.sender.send(`chat:stream-end:${streamId}`, { success: true });
    } catch (error) {
      event.sender.send(`chat:stream-error:${streamId}`, 
        error instanceof Error ? error.message : 'Stream error');
    }
  });

  // Configuration
  ipcMain.handle('chat:get-configs', () => proxyToPython('/v1/oumi/configs'));
  ipcMain.handle('chat:get-models', () => proxyToPython('/v1/models'));

  // Branch management
  ipcMain.handle('chat:get-branches', (_, sessionId) => 
    proxyToPython(`/v1/oumi/branches?session_id=${sessionId}`)
  );

  ipcMain.handle('chat:create-branch', (_, sessionId, name, parentBranchId) => 
    proxyToPython('/v1/oumi/branches', {
      method: 'POST',
      body: JSON.stringify({ 
        action: "create",
        session_id: sessionId,
        name, 
        from_branch: parentBranchId
      }),
    })
  );

  ipcMain.handle('chat:switch-branch', (_, sessionId, branchId) => 
    proxyToPython('/v1/oumi/command', {
      method: 'POST',
      body: JSON.stringify({ 
        command: 'switch',
        args: [branchId],
        session_id: sessionId
      }),
    })
  );

  ipcMain.handle('chat:delete-branch', (_, sessionId, branchId) => 
    proxyToPython('/v1/oumi/command', {
      method: 'POST',
      body: JSON.stringify({ 
        command: 'branch_delete',
        args: [branchId],
        session_id: sessionId
      }),
    })
  );

  // Conversation management
  ipcMain.handle('chat:get-conversation', (_, sessionId, branchId) => 
    proxyToPython(`/v1/oumi/conversation?session_id=${sessionId}&branch_id=${branchId}`)
  );

  ipcMain.handle('chat:send-message', (_, content, sessionId, branchId) => 
    proxyToPython(`/api/sessions/${sessionId}/branches/${branchId}/messages`, {
      method: 'POST',
      body: JSON.stringify({ content }),
    })
  );

  // Command execution
  ipcMain.handle('chat:execute-command', (_, command, args) => {
    log.info(`Executing command: ${command} with args:`, args);
    return proxyToPython('/v1/oumi/command', {
      method: 'POST',
      body: JSON.stringify({ command, args }),
    });
  });
}

/**
 * Download progress handlers
 */
function setupDownloadHandlers(pythonManager: PythonServerManager): void {
  // Set up progress monitoring callbacks
  pythonManager.setDownloadProgressCallback((progress: DownloadProgress) => {
    // Broadcast to all renderer processes
    broadcastToRenderer('server:download-progress', progress);
  });

  pythonManager.setDownloadErrorCallback((error: DownloadErrorEvent) => {
    // Broadcast download errors to all renderer processes
    broadcastToRenderer('server:download-error', error);
  });
}

/**
 * Broadcast events to all renderer windows
 */
export function broadcastToRenderer(channel: string, ...args: any[]): void {
  BrowserWindow.getAllWindows().forEach(window => {
    window.webContents.send(channel, ...args);
  });
}