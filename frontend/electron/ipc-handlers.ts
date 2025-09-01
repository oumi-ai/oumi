/**
 * IPC Handlers - Bridge between Electron renderer and Python backend
 */

import { ipcMain, dialog, app, BrowserWindow } from 'electron';
import { promises as fs } from 'fs';
import Store from 'electron-store';
import log from 'electron-log';
import { PythonServerManager, DownloadProgress, DownloadErrorEvent } from './python-manager';
import { apiKeyManager, ApiKeyConfig } from './api-key-manager';

const store = new Store();

/**
 * Set up all IPC handlers
 */
export function setupIpcHandlers(pythonManager: PythonServerManager): void {
  log.info('Setting up IPC handlers - starting...');
  
  try {
    // App control handlers
    log.info('Setting up app handlers...');
    setupAppHandlers();
    
    // Loading screen control
    ipcMain.handle('app:hide-loading-screen', async () => {
      log.info('Hiding loading screen - React is ready');
      // Find the main window and execute JavaScript to hide loading screen
      const { BrowserWindow } = require('electron');
      const windows = BrowserWindow.getAllWindows();
      const mainWindow = windows.find((win: any) => !win.isDestroyed());
      if (mainWindow) {
        await mainWindow.webContents.executeJavaScript(`
          const loadingScreen = document.getElementById('electron-loading-screen');
          if (loadingScreen) {
            loadingScreen.style.transition = 'opacity 0.3s ease-out';
            loadingScreen.style.opacity = '0';
            setTimeout(() => {
              loadingScreen.style.display = 'none';
            }, 300);
          }
        `);
        return { success: true, message: 'Loading screen hidden' };
      }
      return { success: false, message: 'Main window not found' };
    });
    
    // File system handlers
    log.info('Setting up file handlers...');
    setupFileHandlers();
    
    // Storage handlers
    log.info('Setting up storage handlers...');
    setupStorageHandlers();
    
    // Chat API handlers (proxy to Python backend)
    log.info('Setting up chat handlers...');
    setupChatHandlers(pythonManager);
    
    // Download progress handlers
    log.info('Setting up download handlers...');
    setupDownloadHandlers(pythonManager);
    
    // Config discovery handlers
    log.info('Setting up config handlers...');
    setupConfigHandlers();
    
    // Python environment setup handlers
    log.info('Setting up Python environment handlers...');
    setupPythonEnvironmentHandlers(pythonManager);
    
    // Logger handlers
    log.info('Setting up logger handlers...');
    setupLoggerHandlers();
    
    // System detection handlers
    log.info('Setting up system detection handlers...');
    setupSystemDetectionHandlers();
    
    // API key management handlers
    log.info('Setting up API key management handlers...');
    setupApiKeyHandlers();
    
    log.info('IPC handlers set up successfully');
  } catch (error) {
    log.error('Error setting up IPC handlers:', error);
    throw error;
  }
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
      
      // Return in ApiResponse format
      return {
        success: true,
        data: result
      };
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
 * Config discovery handlers
 */
function setupConfigHandlers(): void {
  ipcMain.handle('config:discover-bundled', async () => {
    try {
      const path = require('path');
      const fs = require('fs').promises;
      const yaml = require('js-yaml');

      // Configs are inside OUMI_ROOT, regardless of environment  
      const oumiRoot = app.isPackaged
        ? path.join(process.resourcesPath, 'python')  // In packaged app, python dir IS the oumi root
        : path.join(__dirname, '../../../');          // In dev, go up to oumi root
        
      const configsPath = path.resolve(oumiRoot, 'configs');

      log.info(`[ConfigDiscovery] OUMI_ROOT: ${oumiRoot}`);
      log.info(`[ConfigDiscovery] Discovering configs in: ${configsPath}`);

      const configs = await discoverConfigsRecursive(configsPath, configsPath);
      
      log.info(`Discovered ${configs.length} configurations`);
      
      return {
        success: true,
        data: {
          generated_at: new Date().toISOString(),
          version: "1.0",
          total_configs: configs.length,
          configs: configs
        }
      };
    } catch (error) {
      log.error('Error discovering configs:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  });
}

/**
 * Recursively discover config files
 */
async function discoverConfigsRecursive(dir: string, baseDir: string): Promise<any[]> {
  const path = require('path');
  const fs = require('fs').promises;
  const yaml = require('js-yaml');
  
  const configs: any[] = [];

  try {
    const items = await fs.readdir(dir);

    for (const item of items) {
      const fullPath = path.join(dir, item);
      const stat = await fs.stat(fullPath);

      if (stat.isDirectory()) {
        // Recursively search subdirectories
        const subConfigs = await discoverConfigsRecursive(fullPath, baseDir);
        configs.push(...subConfigs);
      } else if (item.match(/\.(yaml|yml)$/)) {
        // Only process inference configs
        const relativePath = path.relative(baseDir, fullPath);
        if (relativePath.includes('inference') || relativePath.includes('infer')) {
          try {
            const configData = await parseConfigFile(fullPath, relativePath);
            if (configData) {
              configs.push(configData);
            }
          } catch (error) {
            log.warn(`Failed to parse config ${fullPath}:`, error);
          }
        }
      }
    }
  } catch (error) {
    log.warn(`Failed to read directory ${dir}:`, error);
  }

  return configs;
}

/**
 * Parse a YAML config file and extract metadata
 */
async function parseConfigFile(filePath: string, relativePath: string): Promise<any | null> {
  const fs = require('fs').promises;
  const yaml = require('js-yaml');
  const path = require('path');

  try {
    const content = await fs.readFile(filePath, 'utf8');
    const config = yaml.load(content) as any;

    if (!config || typeof config !== 'object') {
      return null;
    }

    // Extract model information
    const modelName = config.model?.model_name || 'Unknown Model';
    const engine = config.engine || 'UNKNOWN';
    const contextLength = config.model?.model_max_length || config.model?.context_length || 2048;

    // Extract model family from path
    const family = extractModelFamily(relativePath);
    
    // Create display name
    const fileName = path.basename(filePath, path.extname(filePath));
    const cleanFileName = fileName.replace(/_infer$/, '').replace(/_/g, ' ');
    
    let displayName;
    if (family === 'openai' || family === 'anthropic' || family === 'gemini' || family === 'vertex') {
      displayName = `${family.toUpperCase()} - ${cleanFileName}`;
    } else {
      displayName = `${family} - ${cleanFileName}`;
    }

    return {
      id: relativePath.replace(/[/\\]/g, '_').replace(/\.(yaml|yml)$/, ''),
      config_path: relativePath, // Keep relative for runtime resolution
      relative_path: relativePath,
      display_name: displayName,
      model_name: modelName,
      engine: engine,
      context_length: contextLength,
      model_family: family,
      size_category: categorizeModelSize(displayName)
    };
  } catch (error) {
    log.warn(`Failed to parse config ${filePath}:`, error);
    return null;
  }
}

/**
 * Extract model family from config path
 */
function extractModelFamily(configPath: string): string {
  let match = configPath.match(/recipes[/\\]([^/\\]+)/);
  if (match) return match[1];
  
  match = configPath.match(/apis[/\\]([^/\\]+)/);
  if (match) return match[1];
  
  match = configPath.match(/projects[/\\]([^/\\]+)/);
  if (match) return match[1];
  
  return 'unknown';
}

/**
 * Categorize model size based on display name
 */
function categorizeModelSize(displayName: string): string {
  const name = displayName.toLowerCase();
  if (name.includes('135m') || name.includes('1b')) return 'small';
  if (name.includes('3b') || name.includes('7b') || name.includes('8b')) return 'medium';
  if (name.includes('20b') || name.includes('24b') || name.includes('30b') || name.includes('32b') || name.includes('70b')) return 'large';
  if (name.includes('120b') || name.includes('405b')) return 'xl';
  return 'medium';
}

/**
 * Python environment setup handlers
 */
function setupPythonEnvironmentHandlers(pythonManager: PythonServerManager): void {
  // Check if environment setup is needed
  ipcMain.handle('python:is-setup-needed', async () => {
    try {
      const isNeeded = await pythonManager.isEnvironmentSetupNeeded();
      return isNeeded;
    } catch (error) {
      log.error('Failed to check if Python setup is needed:', error);
      return false;
    }
  });

  // Get user data path
  ipcMain.handle('python:get-user-data-path', async () => {
    try {
      return pythonManager.getUserDataPath();
    } catch (error) {
      log.error('Failed to get Python user data path:', error);
      return '';
    }
  });

  // Cancel setup
  ipcMain.handle('python:cancel-setup', async () => {
    try {
      // Cancel setup through environment manager if possible
      // This would need to be implemented in python-manager
      log.info('Cancelling Python environment setup');
    } catch (error) {
      log.error('Failed to cancel Python setup:', error);
    }
  });

  // Rebuild environment
  ipcMain.handle('python:rebuild-environment', async () => {
    try {
      log.info('Rebuilding Python environment');
      await pythonManager.rebuildEnvironment();
      return { success: true, message: 'Environment rebuilt successfully' };
    } catch (error) {
      log.error('Failed to rebuild Python environment:', error);
      return { success: false, message: error instanceof Error ? error.message : 'Unknown error' };
    }
  });

  // Get system change information
  ipcMain.handle('python:get-system-change-info', async () => {
    try {
      return pythonManager.getSystemChangeInfo();
    } catch (error) {
      log.error('Failed to get system change info:', error);
      return null;
    }
  });

  // Get environment system information
  ipcMain.handle('python:get-environment-system-info', async () => {
    try {
      return pythonManager.getEnvironmentSystemInfo();
    } catch (error) {
      log.error('Failed to get environment system info:', error);
      return null;
    }
  });

  // Get basic system information using lightweight script (fallback)
  ipcMain.handle('python:get-basic-system-info', async () => {
    const { executePythonScript } = require('./python-utils');
    const path = require('path');
    
    try {
      // Path to our lightweight system info script
      // In development: __dirname = dist/electron, so go up to project root
      // In production: __dirname = resources/app.asar/dist/electron, so go up to app.asar then to scripts
      let scriptPath;
      if (process.env.NODE_ENV === 'development' || __dirname.includes('dist/electron')) {
        // Development or compiled to dist/electron
        scriptPath = path.join(__dirname, '../../scripts/system-info.py');
      } else {
        // Production build (likely in app.asar)
        scriptPath = path.join(__dirname, '../../../scripts/system-info.py');
      }
      
      log.info('Looking for system info script at:', scriptPath);
      
      // Check if script exists
      const fs = require('fs');
      if (!fs.existsSync(scriptPath)) {
        log.error('System info script not found at:', scriptPath);
        return {
          platform: 'unknown',
          architecture: 'unknown',
          totalRAM: 8,
          cudaAvailable: false,
          cudaDevices: []
        };
      }
      
      log.info(`Running system info script: ${scriptPath}`);
      const result = await executePythonScript(scriptPath, [], {
        timeout: 5000
      });
      
      if (result.success && result.stdout) {
        try {
          const systemInfo = JSON.parse(result.stdout);
          log.info('Successfully detected system info using lightweight script:', systemInfo);
          return systemInfo;
        } catch (parseError) {
          log.error('Failed to parse system info JSON:', parseError);
          return {
            platform: 'unknown',
            architecture: 'unknown',
            totalRAM: 8,
            cudaAvailable: false,
            cudaDevices: []
          };
        }
      } else {
        log.warn('System info script failed:', result.error || result.stderr);
        return {
          platform: 'unknown',
          architecture: 'unknown',
          totalRAM: 8,
          cudaAvailable: false,
          cudaDevices: []
        };
      }
    } catch (error) {
      log.error('Exception in basic system info detection:', error);
      return {
        platform: 'unknown',
        architecture: 'unknown',
        totalRAM: 8,
        cudaAvailable: false,
        cudaDevices: []
      };
    }
  });

  // Remove environment
  ipcMain.handle('python:remove-environment', async () => {
    try {
      log.info('Removing Python environment');
      await pythonManager.removeEnvironment();
      return { success: true, message: 'Environment removed successfully' };
    } catch (error) {
      log.error('Failed to remove Python environment:', error);
      return { success: false, message: error instanceof Error ? error.message : 'Unknown error' };
    }
  });

  // Set up progress forwarding
  pythonManager.setSetupProgressCallback((progress) => {
    broadcastToRenderer('python:setup-progress', progress);
  });

  // Error events would be handled similarly
  // Note: This assumes python-manager will have error callbacks added
  
  log.info('Python environment IPC handlers set up');
}

/**
 * Logger handlers - for file-based logging
 */
function setupLoggerHandlers(): void {
  ipcMain.handle('logger:write', async (_, entry: any) => {
    try {
      // Use electron-log for file logging (it handles file rotation and formatting)
      const logMessage = `[${entry.timestamp}] ${entry.level} [${entry.component}] ${entry.message}`;
      
      switch (entry.level) {
        case 'DEBUG':
          log.debug(logMessage, entry.data || '');
          break;
        case 'INFO':
          log.info(logMessage, entry.data || '');
          break;
        case 'WARN':
          log.warn(logMessage, entry.data || '');
          break;
        case 'ERROR':
          log.error(logMessage, entry.data || '');
          break;
        default:
          log.info(logMessage, entry.data || '');
      }
      
      return { success: true };
    } catch (error) {
      // Fallback to console if logging fails
      console.error('Failed to write log entry:', error);
      return { 
        success: false, 
        error: error instanceof Error ? error.message : String(error) 
      };
    }
  });
}

/**
 * System detection handlers
 */
function setupSystemDetectionHandlers(): void {
  // Get system capabilities for hardware detection
  ipcMain.handle('system:get-capabilities', async () => {
    try {
      const { SystemDetector } = await import('./system-detector');
      const systemInfo = await SystemDetector.detectSystem();
      
      // Convert SystemInfo to SystemCapabilities format expected by renderer
      return {
        platform: systemInfo.platform,
        architecture: systemInfo.architecture,
        totalRAM: systemInfo.totalRAM,
        cudaAvailable: systemInfo.cudaAvailable,
        cudaDevices: systemInfo.cudaDevices.map(device => ({
          vram: device.vram
        }))
      };
    } catch (error) {
      log.error('Failed to detect system capabilities:', error);
      // Return basic fallback info
      return {
        platform: process.platform,
        architecture: process.arch,
        totalRAM: 8, // Fallback to 8GB
        cudaAvailable: false,
        cudaDevices: []
      };
    }
  });

  // Get detailed system information
  ipcMain.handle('system:get-info', async () => {
    try {
      const { SystemDetector } = await import('./system-detector');
      return await SystemDetector.detectSystem();
    } catch (error) {
      log.error('Failed to detect system info:', error);
      throw error;
    }
  });
}

/**
 * API Key Management handlers
 */
function setupApiKeyHandlers(): void {
  // Store API key securely
  ipcMain.handle('apikey:store', async (_, config: ApiKeyConfig) => {
    try {
      apiKeyManager.storeApiKey(config);
      return { success: true };
    } catch (error) {
      log.error('Failed to store API key:', error);
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Failed to store API key' 
      };
    }
  });

  // Get API key (without exposing the actual key value to renderer)
  ipcMain.handle('apikey:get', async (_, providerId: string) => {
    try {
      const keyConfig = apiKeyManager.getApiKey(providerId);
      if (!keyConfig) {
        return { success: false, error: 'API key not found' };
      }

      // Return config without the actual key value for security
      return {
        success: true,
        data: {
          providerId: keyConfig.providerId,
          isActive: keyConfig.isActive,
          isValid: keyConfig.isValid,
          lastValidated: keyConfig.lastValidated,
          details: keyConfig.details
        }
      };
    } catch (error) {
      log.error('Failed to get API key:', error);
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Failed to retrieve API key' 
      };
    }
  });

  // Get all API keys (without key values)
  ipcMain.handle('apikey:get-all', async () => {
    try {
      const keys = apiKeyManager.getAllApiKeys();
      return { success: true, data: keys };
    } catch (error) {
      log.error('Failed to get API keys:', error);
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Failed to retrieve API keys' 
      };
    }
  });

  // Remove API key
  ipcMain.handle('apikey:remove', async (_, providerId: string) => {
    try {
      const success = apiKeyManager.removeApiKey(providerId);
      return { success };
    } catch (error) {
      log.error('Failed to remove API key:', error);
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Failed to remove API key' 
      };
    }
  });

  // Update API key status
  ipcMain.handle('apikey:update-status', async (_, providerId: string, updates: any) => {
    try {
      const success = apiKeyManager.updateApiKeyStatus(providerId, updates);
      return { success };
    } catch (error) {
      log.error('Failed to update API key status:', error);
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Failed to update API key status' 
      };
    }
  });

  // Validate API key using Oumi
  ipcMain.handle('apikey:validate-with-oumi', async (_, providerId: string) => {
    try {
      const result = await apiKeyManager.validateApiKeyWithOumi(providerId);
      return { success: true, data: result };
    } catch (error) {
      log.error('Failed to validate API key:', error);
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Failed to validate API key' 
      };
    }
  });

  // Test API key with specific config
  ipcMain.handle('apikey:test-with-config', async (_, providerId: string, configPath: string) => {
    try {
      const apiKey = apiKeyManager.getApiKey(providerId);
      if (!apiKey) {
        return { success: false, error: 'API key not found' };
      }

      // This would extend the existing testModel logic for API keys
      // For now, delegate to the existing validation method
      const result = await apiKeyManager.validateApiKeyWithOumi(providerId);
      return { success: true, data: result };
    } catch (error) {
      log.error('Failed to test API key with config:', error);
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Failed to test API key' 
      };
    }
  });

  // Check for legacy keys that need migration
  ipcMain.handle('apikey:check-migration-needed', async () => {
    try {
      const needsMigration = apiKeyManager.hasLegacyKeys();
      return { success: true, data: { needsMigration } };
    } catch (error) {
      log.error('Failed to check migration status:', error);
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Failed to check migration status' 
      };
    }
  });

  // Clear all API keys (security reset)
  ipcMain.handle('apikey:clear-all', async () => {
    try {
      apiKeyManager.clearAllKeys();
      return { success: true };
    } catch (error) {
      log.error('Failed to clear API keys:', error);
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Failed to clear API keys' 
      };
    }
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