/**
 * Native application menu system
 */

import { Menu, MenuItemConstructorOptions, BrowserWindow, dialog, shell, app } from 'electron';
import log from 'electron-log';

export function createApplicationMenu(mainWindow: BrowserWindow): Menu {
  const isMac = process.platform === 'darwin';

  const template: MenuItemConstructorOptions[] = [
    // macOS App Menu
    ...(isMac ? [{
      label: app.getName(),
      submenu: [
        { label: 'About Chatterley', role: 'about' as const },
        { type: 'separator' as const },
        { label: 'Services', role: 'services' as const, submenu: [] },
        { type: 'separator' as const },
        { label: 'Hide Chatterley', accelerator: 'Command+H', role: 'hide' as const },
        { label: 'Hide Others', accelerator: 'Command+Shift+H', role: 'hideOthers' as const },
        { label: 'Show All', role: 'unhide' as const },
        { type: 'separator' as const },
        { label: 'Quit', accelerator: 'Command+Q', click: () => app.quit() }
      ]
    }] : []),

    // File Menu
    {
      label: 'File',
      submenu: [
        {
          label: 'New Chat',
          accelerator: 'CmdOrCtrl+N',
          click: () => {
            mainWindow.webContents.send('menu:new-chat');
          }
        },
        { type: 'separator' },
        // Save/Load conversation menu items removed per request
        {
          label: 'Browse for Config...',
          accelerator: 'CmdOrCtrl+Shift+O',
          click: async () => {
            try {
              const result = await dialog.showOpenDialog(mainWindow, {
                title: 'Select Configuration File',
                filters: [
                  { name: 'YAML Config Files', extensions: ['yaml', 'yml'] },
                  { name: 'All Files', extensions: ['*'] }
                ],
                properties: ['openFile']
              });

              if (!result.canceled && result.filePaths.length > 0) {
                mainWindow.webContents.send('menu:browse-config', result.filePaths[0]);
              }
            } catch (error) {
              log.error('Error in browse config dialog:', error);
            }
          }
        },
        // Export Chat menu removed per request
        { type: 'separator' },
        {
          label: 'Preferences',
          accelerator: isMac ? 'Cmd+,' : 'Ctrl+,',
          click: () => {
            mainWindow.webContents.send('menu:preferences');
          }
        },
        { type: 'separator' },
        {
          label: 'Reset Welcome Settings',
          click: () => {
            mainWindow.webContents.send('menu:reset-welcome-settings');
          }
        },
        {
          label: 'Reset Chat History...',
          click: async () => {
            log.info('🧹 Menu: Reset Chat History clicked');
            const result = await dialog.showMessageBox(mainWindow, {
              type: 'warning',
              title: 'Reset Chat History',
              message: 'Are you sure you want to reset all chat history?',
              detail: 'This will permanently delete all threads, conversations, messages, attachments, and vector indexes. This action cannot be undone.',
              buttons: ['Cancel', 'Reset without backup', 'Backup and reset'],
              defaultId: 0,
              cancelId: 0
            });

            log.info(`🧹 Menu: Dialog result - response: ${result.response}`);
            if (result.response === 1) {
              log.info('🧹 Menu: Sending menu:reset-history message to renderer');
              mainWindow.webContents.send('menu:reset-history');
              log.info('🧹 Menu: Message sent successfully');
            } else if (result.response === 2) {
              log.info('🧹 Menu: Sending menu:backup-and-reset-history message to renderer');
              mainWindow.webContents.send('menu:backup-and-reset-history');
              log.info('🧹 Menu: Message sent successfully');
            } else {
              log.info('🧹 Menu: User cancelled reset');
            }
          }
        },
        {
          label: 'Rebuild Python Environment',
          click: async () => {
            log.info('🔧 Menu: Rebuild Python Environment clicked');
            const result = await dialog.showMessageBox(mainWindow, {
              type: 'warning',
              title: 'Rebuild Python Environment',
              message: 'Are you sure you want to rebuild the Python environment?',
              detail: 'This will delete the existing Python environment and recreate it from scratch. This may take a few minutes.',
              buttons: ['Cancel', 'Rebuild'],
              defaultId: 0,
              cancelId: 0
            });

            log.info(`🔧 Menu: Dialog result - response: ${result.response}`);
            if (result.response === 1) {
              log.info('🔧 Menu: Sending menu:rebuild-python-environment message to renderer');
              mainWindow.webContents.send('menu:rebuild-python-environment');
              log.info('🔧 Menu: Message sent successfully');
            } else {
              log.info('🔧 Menu: User cancelled rebuild');
            }
          }
        },
        ...(!isMac ? [
          { type: 'separator' as const },
          { label: 'Exit', accelerator: 'Ctrl+Q', click: () => app.quit() }
        ] : [])
      ]
    },

    // Edit Menu
    {
      label: 'Edit',
      submenu: [
        { label: 'Undo', accelerator: 'CmdOrCtrl+Z', role: 'undo' },
        { label: 'Redo', accelerator: 'Shift+CmdOrCtrl+Z', role: 'redo' },
        { type: 'separator' },
        { label: 'Cut', accelerator: 'CmdOrCtrl+X', role: 'cut' },
        { label: 'Copy', accelerator: 'CmdOrCtrl+C', role: 'copy' },
        { label: 'Paste', accelerator: 'CmdOrCtrl+V', role: 'paste' },
        { label: 'Select All', accelerator: 'CmdOrCtrl+A', role: 'selectAll' },
        { type: 'separator' },
        {
          label: 'Find',
          accelerator: 'CmdOrCtrl+F',
          click: () => {
            mainWindow.webContents.send('menu:find');
          }
        },
        {
          label: 'Clear Conversation',
          accelerator: 'CmdOrCtrl+Shift+Delete',
          click: () => {
            mainWindow.webContents.send('menu:clear-conversation');
          }
        }
      ]
    },

    // View Menu
    {
      label: 'View',
      submenu: [
        {
          label: 'Reload',
          accelerator: 'CmdOrCtrl+R',
          click: () => mainWindow.reload()
        },
        {
          label: 'Force Reload',
          accelerator: 'CmdOrCtrl+Shift+R',
          click: () => mainWindow.webContents.reloadIgnoringCache()
        },
        {
          label: 'Toggle Developer Tools',
          accelerator: process.platform === 'darwin' ? 'Alt+Cmd+I' : 'Ctrl+Shift+I',
          click: () => mainWindow.webContents.toggleDevTools()
        },
        { type: 'separator' },
        {
          label: 'Actual Size',
          accelerator: 'CmdOrCtrl+0',
          click: () => mainWindow.webContents.setZoomLevel(0)
        },
        {
          label: 'Zoom In',
          accelerator: 'CmdOrCtrl+Plus',
          click: () => {
            const currentZoom = mainWindow.webContents.getZoomLevel();
            mainWindow.webContents.setZoomLevel(Math.min(currentZoom + 0.5, 3));
          }
        },
        {
          label: 'Zoom Out',
          accelerator: 'CmdOrCtrl+-',
          click: () => {
            const currentZoom = mainWindow.webContents.getZoomLevel();
            mainWindow.webContents.setZoomLevel(Math.max(currentZoom - 0.5, -3));
          }
        },
        { type: 'separator' },
        {
          label: 'Toggle Branch Tree',
          accelerator: 'CmdOrCtrl+B',
          click: () => {
            mainWindow.webContents.send('menu:toggle-branch-tree');
          }
        },
        {
          label: 'Toggle Control Panel',
          accelerator: 'CmdOrCtrl+T',
          click: () => {
            mainWindow.webContents.send('menu:toggle-control-panel');
          }
        },
        { type: 'separator' },
        {
          label: 'Toggle Fullscreen',
          accelerator: isMac ? 'Ctrl+Cmd+F' : 'F11',
          click: () => mainWindow.setFullScreen(!mainWindow.isFullScreen())
        }
      ]
    },

    // Chat Menu
    {
      label: 'Chat',
      submenu: [
        {
          label: 'Send Message',
          accelerator: 'Enter',
          click: () => {
            mainWindow.webContents.send('menu:send-message');
          }
        },
        {
          label: 'New Line in Message',
          accelerator: 'Shift+Enter',
          click: () => {
            mainWindow.webContents.send('menu:new-line');
          }
        },
        { type: 'separator' },
        {
          label: 'Regenerate Last Response',
          accelerator: 'CmdOrCtrl+R',
          click: () => {
            mainWindow.webContents.send('menu:regenerate');
          }
        },
        {
          label: 'Stop Generation',
          accelerator: 'Escape',
          click: () => {
            mainWindow.webContents.send('menu:stop-generation');
          }
        },
        { type: 'separator' },
        {
          label: 'Create Branch',
          accelerator: 'CmdOrCtrl+Shift+B',
          click: () => {
            mainWindow.webContents.send('menu:create-branch');
          }
        },
        {
          label: 'Switch Branch',
          accelerator: 'CmdOrCtrl+Shift+S',
          click: () => {
            mainWindow.webContents.send('menu:switch-branch');
          }
        },
        { type: 'separator' },
        {
          label: 'Model Settings',
          accelerator: 'CmdOrCtrl+M',
          click: () => {
            mainWindow.webContents.send('menu:model-settings');
          }
        }
      ]
    },

    // Window Menu
    {
      label: 'Window',
      submenu: [
        { label: 'Minimize', accelerator: 'CmdOrCtrl+M', role: 'minimize' },
        { label: 'Close', accelerator: 'CmdOrCtrl+W', role: 'close' },
        ...(isMac ? [
          { type: 'separator' as const },
          { label: 'Bring All to Front', role: 'front' as const }
        ] : [])
      ]
    },

    // Help Menu
    {
      label: 'Help',
      submenu: [
        {
          label: 'Documentation',
          click: () => {
            shell.openExternal('https://oumi.ai/docs');
          }
        },
        {
          label: 'GitHub Repository',
          click: () => {
            shell.openExternal('https://github.com/oumi-ai/oumi');
          }
        },
        {
          label: 'Report Issue',
          click: () => {
            shell.openExternal('https://github.com/oumi-ai/oumi/issues');
          }
        },
        { type: 'separator' },
        {
          label: 'Keyboard Shortcuts',
          accelerator: 'CmdOrCtrl+?',
          click: () => {
            mainWindow.webContents.send('menu:show-shortcuts');
          }
        },
        { type: 'separator' },
        {
          label: 'Check for Updates',
          click: () => {
            mainWindow.webContents.send('menu:check-updates');
          }
        },
        ...(!isMac ? [{
          label: 'About Chatterley',
          click: () => {
            showAboutDialog(mainWindow);
          }
        }] : [])
      ]
    }
  ];

  return Menu.buildFromTemplate(template);
}

/**
 * Show about dialog
 */
function showAboutDialog(mainWindow: BrowserWindow): void {
  dialog.showMessageBox(mainWindow, {
    type: 'info',
    title: 'About Chatterley',
    message: 'Chatterley',
    detail: `Version: ${app.getVersion()}\n\nA cross-platform desktop application for conversing with AI models.\n\nBuilt with Electron and powered by the Oumi AI platform.`,
    buttons: ['OK']
  });
}

/**
 * Create context menu for chat messages
 */
export function createChatContextMenu(mainWindow: BrowserWindow, messageData?: any): Menu {
  const template: MenuItemConstructorOptions[] = [
    {
      label: 'Copy Message',
      accelerator: 'CmdOrCtrl+C',
      click: () => {
        mainWindow.webContents.send('context-menu:copy-message', messageData);
      }
    },
    {
      label: 'Copy as Markdown',
      click: () => {
        mainWindow.webContents.send('context-menu:copy-markdown', messageData);
      }
    },
    { type: 'separator' },
    {
      label: 'Edit Message',
      click: () => {
        mainWindow.webContents.send('context-menu:edit-message', messageData);
      }
    },
    {
      label: 'Delete Message',
      click: () => {
        mainWindow.webContents.send('context-menu:delete-message', messageData);
      }
    },
    { type: 'separator' },
    {
      label: 'Regenerate Response',
      click: () => {
        mainWindow.webContents.send('context-menu:regenerate', messageData);
      }
    },
    {
      label: 'Branch from Here',
      click: () => {
        mainWindow.webContents.send('context-menu:branch-from-here', messageData);
      }
    }
  ];

  return Menu.buildFromTemplate(template);
}
