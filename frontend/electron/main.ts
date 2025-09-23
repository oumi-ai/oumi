/**
 * Electron main process for Chatterley Desktop Application
 */

import { app, BrowserWindow, Menu, ipcMain, dialog, shell, session } from 'electron';
import { autoUpdater } from 'electron-updater';
import log from 'electron-log';
import Store from 'electron-store';
import * as path from 'path';
import { PythonServerManager } from './python-manager';
import { createApplicationMenu } from './menu';
import { setupIpcHandlers } from './ipc-handlers';

// Initialize logging
log.transports.file.level = 'info';
log.transports.console.level = 'debug';

// Initialize persistent storage with corrupted JSON recovery
function createMainStore(): Store {
  const opts: Store.Options<any> = {
    name: 'chatterley-config',
    defaults: {
      windowBounds: { width: 1400, height: 900 },
      pythonPort: 9000,
      lastSession: 'default'
    },
    clearInvalidConfig: true,
    fileExtension: 'json'
  };

  try {
    return new Store(opts);
  } catch (err) {
    // Backup and recreate on parse errors
    try {
      const fs = require('fs');
      const storePath = require('path').join(app.getPath('userData'), `${opts.name}.${opts.fileExtension}`);
      if (fs.existsSync(storePath)) {
        const backupPath = `${storePath}.bak-${Date.now()}`;
        fs.renameSync(storePath, backupPath);
        log.warn(`[Main] Backed up corrupted store to: ${backupPath}`);
      }
    } catch (backupErr) {
      log.warn('[Main] Failed backing up corrupted store:', backupErr);
    }
    return new Store(opts);
  }
}

const store = createMainStore();

class ChatterleyApp {
  private mainWindow: BrowserWindow | null = null;
  private setupWindow: BrowserWindow | null = null;
  private pythonManager: PythonServerManager | null = null;
  private isDevelopment: boolean;

  constructor() {
    // Force production mode when debugging production build
    this.isDevelopment = process.env.ELECTRON_DEBUG_PRODUCTION !== '1' && 
                         (process.env.NODE_ENV === 'development' || !app.isPackaged);
    
    // Enhanced debugging for production builds
    if (process.env.ELECTRON_DEBUG_PRODUCTION === '1') {
      log.info('🐛 Production Debug Mode Enabled');
      log.info(`App packaged: ${app.isPackaged}`);
      log.info(`Node ENV: ${process.env.NODE_ENV}`);
      log.info(`App path: ${app.getAppPath()}`);
      log.info(`User data: ${app.getPath('userData')}`);
      log.info(`Development mode: ${this.isDevelopment}`);
    }
    
    this.setupEventHandlers();
  }

  private setupEventHandlers(): void {
    // App event handlers
    app.whenReady().then(() => this.onReady());
    app.on('window-all-closed', () => this.onWindowAllClosed());
    app.on('activate', () => this.onActivate());
    app.on('before-quit', () => this.onBeforeQuit());

    // Auto-updater events
    autoUpdater.on('checking-for-update', () => {
      log.info('Checking for updates...');
    });

    autoUpdater.on('update-available', () => {
      log.info('Update available');
      this.showUpdateNotification();
    });

    autoUpdater.on('update-not-available', () => {
      log.info('Update not available');
    });

    autoUpdater.on('error', (err) => {
      log.error('Error in auto-updater:', err);
    });
  }

  private async onReady(): Promise<void> {
    try {
      // In debug modes, proactively clear Electron caches at startup
      if (this.isDevelopment || process.env.ELECTRON_DEBUG_PRODUCTION === '1') {
        try {
          await this.clearCachesOnStartup();
        } catch (err) {
          log.warn('Failed to clear Electron caches on startup:', err);
        }
      }

      // Initialize Python server manager (but don't start server yet)
      const pythonPortVal = (store as any).get('pythonPort');
      const pythonPort = typeof pythonPortVal === 'number'
        ? pythonPortVal
        : Number.parseInt(String(pythonPortVal || ''), 10) || 9000;
      this.pythonManager = new PythonServerManager(pythonPort);
      
      log.info('Python server manager initialized - checking environment setup');
      
      // Check if Python environment setup is needed
      const isSetupNeeded = await this.pythonManager.isEnvironmentSetupNeeded();
      
      if (isSetupNeeded) {
        log.info('Python environment setup needed - showing setup screen');
        
        // Show setup progress window BEFORE environment setup
        this.createSetupWindow();
        
        // Set up IPC handlers for setup progress communication
        this.setupSetupIpcHandlers();
        
        // Perform environment setup
        try {
          log.info('Starting Python environment setup...');
          await this.pythonManager.rebuildEnvironment();
          log.info('Python environment setup completed successfully');
          
          // Close setup window and proceed to main application
          this.closeSetupWindow();
          await this.initializeMainApplication();
          
        } catch (error) {
          log.error('Failed to set up Python environment:', error);
          this.showSetupError(error instanceof Error ? error.message : 'Unknown error occurred during setup');
          return;
        }
      } else {
        log.info('Python environment already exists and is valid - proceeding to main app');
        await this.initializeMainApplication();
      }

    } catch (error) {
      log.error('Failed to initialize application:', error);
      dialog.showErrorBox(
        'Initialization Error',
        'Failed to start Chatterley. Please check the logs and try again.'
      );
      app.quit();
    }
  }

  private async initializeMainApplication(): Promise<void> {
    // Create main window
    this.createMainWindow();

    // Set up IPC handlers
    log.info('About to set up IPC handlers...');
    try {
      setupIpcHandlers(this.pythonManager!);
      log.info('IPC handlers setup completed successfully');
    } catch (error) {
      log.error('Failed to set up IPC handlers:', error);
      throw error;
    }

    // Create application menu
    const menu = createApplicationMenu(this.mainWindow!);
    Menu.setApplicationMenu(menu);

    // Check for updates in production
    if (!this.isDevelopment) {
      setTimeout(() => autoUpdater.checkForUpdatesAndNotify(), 2000);
    }
  }

  private async clearCachesOnStartup(): Promise<void> {
    log.info('🧹 Debug mode: clearing Electron caches on startup');
    try {
      // Clear Chromium HTTP cache
      await session.defaultSession.clearCache();
    } catch (e) {
      log.warn('clearCache failed:', e);
    }

    try {
      // Clear storage data: service workers, caches, local storage, etc.
      await session.defaultSession.clearStorageData({
        // Electron types allow: 'cookies' | 'filesystem' | 'indexdb' | 'localstorage' | 'shadercache' | 'websql' | 'serviceworkers' | 'cachestorage'
        storages: ['cookies', 'filesystem', 'indexdb', 'localstorage', 'shadercache', 'websql', 'serviceworkers', 'cachestorage'],
        quotas: ['temporary', 'syncable']
      });
    } catch (e) {
      log.warn('clearStorageData failed:', e);
    }

    // Best-effort removal of on-disk cache directories
    try {
      const fs = require('fs');
      const path = require('path');
      const userData = app.getPath('userData');
      const dirs = ['Cache', 'Code Cache', 'GPUCache'];
      for (const d of dirs) {
        const p = path.join(userData, d);
        if (fs.existsSync(p)) {
          fs.rmSync(p, { recursive: true, force: true });
          log.info(`🧹 removed cache dir: ${p}`);
        }
        // Nested Cache_Data sometimes appears
        const cd = path.join(userData, 'Cache', 'Cache_Data');
        if (fs.existsSync(cd)) {
          fs.rmSync(cd, { recursive: true, force: true });
          log.info(`🧹 removed cache dir: ${cd}`);
        }
      }
    } catch (e) {
      log.warn('Filesystem cache cleanup failed:', e);
    }
  }

  private createMainWindow(): void {
    const bounds = store.get('windowBounds') as { width: number; height: number };

    this.mainWindow = new BrowserWindow({
      width: bounds.width,
      height: bounds.height,
      minWidth: 800,
      minHeight: 600,
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        sandbox: false,
        preload: path.join(__dirname, 'preload.js'),
        webSecurity: !this.isDevelopment,
        allowRunningInsecureContent: false,
        experimentalFeatures: false
      },
      titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
      show: false, // Don't show until ready
      icon: this.getAppIcon()
    });

    // Load the application
    let startUrl: string;
    
    if (this.isDevelopment) {
      const nextPort = process.env.NEXT_DEV_PORT || '3000';
      startUrl = `http://localhost:${nextPort}`;
    } else {
      // In production, look for the Next.js static export
      const indexPath = path.join(__dirname, '../../out/index.html');
      log.info(`Loading production app from: ${indexPath}`);
      
      // Enhanced debugging for production builds
      if (process.env.ELECTRON_DEBUG_PRODUCTION === '1') {
        const fs = require('fs');
        log.info(`Index file exists: ${fs.existsSync(indexPath)}`);
        log.info(`Out directory contents:`, fs.readdirSync(path.dirname(indexPath)).slice(0, 10));
      }
      
      startUrl = `file://${indexPath}`;
    }

    log.info(`Loading app from: ${startUrl}`);
    this.mainWindow.loadURL(startUrl);

    // Enhanced error handling for production debugging
    this.mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription, validatedURL) => {
      log.error(`Failed to load ${validatedURL}: ${errorCode} - ${errorDescription}`);
      if (process.env.ELECTRON_DEBUG_PRODUCTION === '1') {
        this.mainWindow?.webContents.openDevTools();
      }
    });

    this.mainWindow.webContents.on('console-message', (event, level, message, line, sourceId) => {
      if (process.env.ELECTRON_DEBUG_PRODUCTION === '1' || this.isDevelopment) {
        log.info(`Renderer Console [${level}]: ${message}`);
      }
    });

    // Show window when ready
    this.mainWindow.once('ready-to-show', () => {
      this.mainWindow?.show();
      
      if (this.isDevelopment || process.env.ELECTRON_DEBUG_PRODUCTION === '1') {
        this.mainWindow?.webContents.openDevTools();
      }
    });

    // Save window bounds on resize
    this.mainWindow.on('resize', () => {
      if (this.mainWindow) {
        store.set('windowBounds', this.mainWindow.getBounds());
      }
    });

    // Handle window closed
    this.mainWindow.on('closed', () => {
      this.mainWindow = null;
    });

    // Handle external links
    this.mainWindow.webContents.setWindowOpenHandler(({ url }) => {
      shell.openExternal(url);
      return { action: 'deny' };
    });
  }

  private getAppIcon(): string | undefined {
    const iconPath = path.join(__dirname, '../assets/icon');
    
    switch (process.platform) {
      case 'darwin':
        return `${iconPath}.icns`;
      case 'win32':
        return `${iconPath}.ico`;
      case 'linux':
        return `${iconPath}.png`;
      default:
        return undefined;
    }
  }

  private onWindowAllClosed(): void {
    // On macOS, keep app running even when all windows are closed
    if (process.platform !== 'darwin') {
      app.quit();
    }
  }

  private onActivate(): void {
    // On macOS, re-create window when dock icon is clicked
    if (BrowserWindow.getAllWindows().length === 0) {
      this.createMainWindow();
    }
  }

  private async onBeforeQuit(): Promise<void> {
    try {
      // Gracefully stop Python server
      if (this.pythonManager) {
        await this.pythonManager.stop();
      }
    } catch (error) {
      log.error('Error stopping Python server:', error);
    }
  }

  private showUpdateNotification(): void {
    if (!this.mainWindow) return;

    dialog.showMessageBox(this.mainWindow, {
      type: 'info',
      title: 'Update Available',
      message: 'A new version of Chatterley is available.',
      detail: 'The update will be downloaded and installed automatically when you restart the application.',
      buttons: ['OK']
    });
  }

  // Public methods for menu actions
  public getMainWindow(): BrowserWindow | null {
    return this.mainWindow;
  }

  public openDevTools(): void {
    if (this.mainWindow) {
      this.mainWindow.webContents.openDevTools();
    }
  }

  public reload(): void {
    if (this.mainWindow) {
      this.mainWindow.reload();
    }
  }

  public toggleFullScreen(): void {
    if (this.mainWindow) {
      this.mainWindow.setFullScreen(!this.mainWindow.isFullScreen());
    }
  }

  public zoom(direction: 'in' | 'out' | 'reset'): void {
    if (!this.mainWindow) return;

    const webContents = this.mainWindow.webContents;
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

  private createSetupWindow(): void {
    this.setupWindow = new BrowserWindow({
      width: 600,
      height: 400,
      resizable: false,
      minimizable: false,
      maximizable: false,
      show: false,
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        preload: path.join(__dirname, 'preload.js'),
        webSecurity: !this.isDevelopment
      },
      titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
      icon: this.getAppIcon()
    });

    // Create a simple setup page
    const setupHtml = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>Setting up Chatterley</title>
        <style>
          body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 40px;
            background: #f5f5f5;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            box-sizing: border-box;
          }
          .setup-container {
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            text-align: center;
            max-width: 400px;
            width: 100%;
          }
          h1 { color: #333; margin-bottom: 20px; }
          .progress { color: #666; margin-bottom: 30px; }
          .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #007AFF;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
          }
          @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
          .error { color: #d32f2f; margin-top: 20px; display: none; }
          .details { font-size: 14px; color: #888; margin-top: 15px; }
        </style>
      </head>
      <body>
        <div class="setup-container">
          <h1>Setting up Chatterley</h1>
          <div class="spinner"></div>
          <div class="progress" id="progress">Preparing Python environment...</div>
          <div class="details" id="details">This may take a few minutes on first launch.</div>
          <div class="error" id="error"></div>
        </div>
        <script>
          window.electronAPI.python.onSetupProgress((progress) => {
            document.getElementById('progress').textContent = progress.message;
          });
          
          window.electronAPI.python.onSetupError((error) => {
            document.querySelector('.spinner').style.display = 'none';
            document.getElementById('error').style.display = 'block';
            document.getElementById('error').textContent = 'Setup failed: ' + error;
          });
        </script>
      </body>
      </html>
    `;

    this.setupWindow.loadURL(`data:text/html;charset=UTF-8,${encodeURIComponent(setupHtml)}`);
    
    this.setupWindow.once('ready-to-show', () => {
      this.setupWindow?.show();
    });

    this.setupWindow.on('closed', () => {
      this.setupWindow = null;
    });
  }

  private closeSetupWindow(): void {
    if (this.setupWindow) {
      this.setupWindow.close();
      this.setupWindow = null;
    }
  }

  private showSetupError(message: string): void {
    if (this.setupWindow) {
      this.setupWindow.webContents.send('python:setup-error', message);
    }
  }

  private setupSetupIpcHandlers(): void {
    // Set up progress forwarding for the setup window
    if (this.pythonManager) {
      this.pythonManager.setSetupProgressCallback((progress) => {
        if (this.setupWindow) {
          this.setupWindow.webContents.send('python:setup-progress', progress);
        }
      });
    }
  }

}

// Global app instance
let appInstance: ChatterleyApp;

// Initialize the application
if (!app.requestSingleInstanceLock()) {
  // Another instance is already running
  app.quit();
} else {
  appInstance = new ChatterleyApp();

  // Handle second instance
  app.on('second-instance', () => {
    const mainWindow = appInstance.getMainWindow();
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });
}

// Export app instance for menu and IPC handlers
export { appInstance };
