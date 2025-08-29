# Oumi Chat Desktop Application

A cross-platform desktop application for conversing with AI models, built with Electron and powered by the Oumi AI platform.

## Features

- **Native Desktop Experience**: Full native menus, keyboard shortcuts, and OS integration
- **Cross-Platform**: Runs on macOS, Windows, and Linux
- **Embedded Backend**: Python server automatically starts with the application
- **File Operations**: Save and load conversations with native file dialogs
- **Real-time Communication**: IPC-based communication for optimal performance
- **Offline Storage**: Persistent conversation history and settings

## Development

### Prerequisites

- Node.js 18+ and npm
- Python 3.9+ with Oumi dependencies
- Conda environment named `oumi` (recommended)

### Setup

1. Install dependencies:
```bash
npm install
```

2. Compile Electron TypeScript files:
```bash
npm run electron:compile
```

3. Build the Next.js application:
```bash
npm run build:electron
```

### Running in Development

To run the Electron app in development mode:

```bash
npm run electron:dev
```

This will:
- Start the Next.js development server
- Compile Electron TypeScript files
- Launch Electron pointing to the dev server

### Building for Production

To build the application for distribution:

```bash
npm run electron:dist
```

This creates platform-specific packages in the `dist/packages` directory.

## Architecture

### Frontend
- **Framework**: Next.js 15 with TypeScript and React
- **Styling**: Tailwind CSS
- **State Management**: Zustand
- **UI Components**: Custom components with Lucide icons

### Backend Integration
- **IPC Communication**: Secure communication between renderer and main process
- **Python Server**: Embedded Oumi backend server
- **Port Management**: Dynamic port discovery to avoid conflicts

### Security
- **Context Isolation**: Renderer process runs in isolated context
- **Preload Script**: Secure API exposure to renderer
- **No Node Integration**: Renderer cannot directly access Node.js APIs

## Available Scripts

- `npm run dev` - Start Next.js development server
- `npm run build` - Build Next.js for web
- `npm run build:electron` - Build Next.js for Electron
- `npm run electron:compile` - Compile Electron TypeScript files
- `npm run electron:dev` - Run Electron in development mode
- `npm run electron:pack` - Package application without distribution
- `npm run electron:dist` - Build and package for distribution

## File Structure

```
frontend/
├── electron/                 # Electron main process files
│   ├── main.ts              # Main Electron process
│   ├── preload.ts           # Preload script for IPC
│   ├── python-manager.ts    # Python server lifecycle
│   ├── ipc-handlers.ts      # IPC message handlers
│   └── menu.ts              # Native application menus
├── src/                     # React application source
│   ├── components/          # React components
│   ├── lib/                 # Utilities and API clients
│   │   ├── unified-api.ts   # Unified API client
│   │   ├── electron-api.ts  # Electron IPC client
│   │   └── api.ts           # HTTP API client (web)
│   └── app/                 # Next.js app directory
└── dist/                    # Built files
    ├── electron/            # Compiled Electron files
    └── packages/            # Distribution packages
```

## Platform Support

- **macOS**: DMG installer and ZIP archive (Intel + Apple Silicon)
- **Windows**: NSIS installer and portable executable
- **Linux**: AppImage and Snap packages

## Configuration

The application uses Electron Store for persistent configuration:

- **Window bounds**: Automatically saved and restored
- **Python server port**: Configurable port for backend
- **Last session**: Remembers last active session

## Troubleshooting

### Python Server Issues
- Ensure conda environment `oumi` is available
- Check that Oumi dependencies are installed
- Look for error logs in Electron dev tools

### Build Issues
- Clear `node_modules` and reinstall: `rm -rf node_modules && npm install`
- Clear Next.js cache: `rm -rf .next`
- Clear Electron build cache: `rm -rf dist`

### Port Conflicts
- The application automatically finds available ports
- Default port range: 9000-9099
- Check for processes using these ports: `lsof -i :9000-9099`

## Contributing

1. Follow the existing code style and patterns
2. Run tests before submitting changes
3. Update documentation for new features
4. Ensure cross-platform compatibility

## License

Apache License 2.0 - see LICENSE file for details.