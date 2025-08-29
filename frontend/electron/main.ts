/**
 * Electron main process for Oumi Chat Desktop Application
 */

import { app, BrowserWindow, Menu, ipcMain, dialog, shell } from 'electron';
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

// Initialize persistent storage
const store = new Store({
  name: 'oumi-chat-config',
  defaults: {
    windowBounds: { width: 1400, height: 900 },
    pythonPort: 9000,
    lastSession: 'default'
  }
});

class OumiChatApp {
  private mainWindow: BrowserWindow | null = null;
  private pythonManager: PythonServerManager | null = null;
  private isDevelopment: boolean;

  constructor() {
    this.isDevelopment = process.env.NODE_ENV === 'development' || !app.isPackaged;
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
      // Initialize Python server manager (but don't start server yet)
      const pythonPort = (store as any).get('pythonPort') || 9000;
      this.pythonManager = new PythonServerManager(pythonPort);
      
      log.info('Python server manager initialized - server will start when config is selected');

      // Create main window
      this.createMainWindow();

      // Set up IPC handlers
      setupIpcHandlers(this.pythonManager);

      // Create application menu
      const menu = createApplicationMenu(this.mainWindow!);
      Menu.setApplicationMenu(menu);

      // Check for updates in production
      if (!this.isDevelopment) {
        setTimeout(() => autoUpdater.checkForUpdatesAndNotify(), 2000);
      }

    } catch (error) {
      log.error('Failed to initialize application:', error);
      dialog.showErrorBox(
        'Initialization Error',
        'Failed to start Oumi Chat. Please check the logs and try again.'
      );
      app.quit();
    }
  }

  private createMainWindow(): void {
    const bounds = (store as any).get('windowBounds') as { width: number; height: number };

    this.mainWindow = new BrowserWindow({
      width: bounds.width,
      height: bounds.height,
      minWidth: 800,
      minHeight: 600,
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        preload: path.join(__dirname, 'preload.js'),
        webSecurity: !this.isDevelopment
      },
      titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
      show: false, // Don't show until ready
      icon: this.getAppIcon()
    });

    // Load the application
    let startUrl: string;
    
    if (this.isDevelopment) {
      startUrl = 'http://localhost:3000';
    } else {
      // In production, look for the Next.js static export
      const indexPath = path.join(__dirname, '../../out/index.html');
      log.info(`Loading production app from: ${indexPath}`);
      startUrl = `file://${indexPath}`;
    }

    log.info(`Loading app from: ${startUrl}`);
    this.mainWindow.loadURL(startUrl);

    // Show window when ready
    this.mainWindow.once('ready-to-show', () => {
      this.mainWindow?.show();
      
      if (this.isDevelopment) {
        this.mainWindow?.webContents.openDevTools();
      }
    });

    // Save window bounds on resize
    this.mainWindow.on('resize', () => {
      if (this.mainWindow) {
        (store as any).set('windowBounds', this.mainWindow.getBounds());
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
      message: 'A new version of Oumi Chat is available.',
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
}

// Global app instance
let appInstance: OumiChatApp;

// Initialize the application
if (!app.requestSingleInstanceLock()) {
  // Another instance is already running
  app.quit();
} else {
  appInstance = new OumiChatApp();

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