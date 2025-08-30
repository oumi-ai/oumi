# Chatterley - Build and Distribution Guide

This guide covers building and distributing Chatterley across platforms.

## Quick Start

### Development Build
```bash
# Quick development build (no distribution)
./scripts/dev-build.sh
```

### Production Build
```bash
# Build for current platform
./scripts/build-local.sh

# Build for specific platform
./scripts/build-local.sh mac
./scripts/build-local.sh win
./scripts/build-local.sh linux

# Build for all platforms
./scripts/build-local.sh all
```

## Build Scripts

### NPM Scripts

| Script | Description |
|--------|-------------|
| `npm run dist:mac` | Build macOS DMG and ZIP |
| `npm run dist:win` | Build Windows NSIS installer and portable |
| `npm run dist:linux` | Build Linux AppImage, DEB, and RPM |
| `npm run dist:all` | Build for all platforms |
| `npm run electron:dev` | Run in development mode |
| `npm run electron:pack` | Package without distribution |

### Shell Scripts

| Script | Description |
|--------|-------------|
| `./scripts/build-local.sh` | Interactive build script |
| `./scripts/dev-build.sh` | Quick development build |
| `./scripts/version.js` | Version management |

## Platform-Specific Builds

### macOS (DMG + ZIP)
**Requirements:**
- macOS machine (for native builds)
- Xcode Command Line Tools
- Apple Developer ID (for code signing)

**Build:**
```bash
npm run dist:mac
```

**Output:**
- `Chatterley-{version}-mac-{arch}.dmg` - Installer
- `Chatterley-{version}-mac-{arch}.zip` - Portable

**Code Signing (Optional):**
Set environment variables:
```bash
export APPLE_ID="your-apple-id@example.com"
export APPLE_ID_PASSWORD="app-specific-password"
export APPLE_TEAM_ID="YOUR_TEAM_ID"
```

### Windows (NSIS + Portable)
**Requirements:**
- Windows machine or cross-compilation setup
- Code signing certificate (optional)

**Build:**
```bash
npm run dist:win
```

**Output:**
- `Chatterley Setup {version}.exe` - NSIS installer
- `Chatterley {version}.exe` - Portable executable

**Code Signing (Optional):**
Set environment variables:
```bash
export WIN_CSC_LINK="path/to/certificate.p12"
export WIN_CSC_KEY_PASSWORD="certificate-password"
```

### Linux (AppImage + DEB + RPM)
**Requirements:**
- Linux machine
- Standard build tools (`build-essential`)

**Build:**
```bash
npm run dist:linux
```

**Output:**
- `Chatterley-{version}.AppImage` - Universal Linux app
- `chatterley_{version}_amd64.deb` - Debian package
- `chatterley-{version}.x86_64.rpm` - Red Hat package

## Auto-Update System

Chatterley includes automatic updates using `electron-updater`:

### Configuration
Updates are configured in `package.json`:
```json
{
  "publish": [
    {
      "provider": "github",
      "owner": "oumi-ai",
      "repo": "oumi-chat-desktop"
    }
  ]
}
```

### Update Flow
1. App checks for updates on startup (production only)
2. Downloads updates in background
3. Notifies user when ready
4. Installs on next app restart

### Manual Update Check
Users can check for updates via:
- **macOS**: `Chatterley` → `Check for Updates`
- **Windows/Linux**: `Help` → `Check for Updates`

## Version Management

### Update Version
```bash
# Increment patch version (1.0.0 -> 1.0.1)
node scripts/version.js patch

# Increment minor version (1.0.0 -> 1.1.0)
node scripts/version.js minor

# Increment major version (1.0.0 -> 2.0.0)
node scripts/version.js major

# Set specific version
node scripts/version.js 1.2.3

# Show current version
node scripts/version.js --current
```

### Release Process
1. Update version: `node scripts/version.js patch`
2. Test build locally: `./scripts/build-local.sh`
3. Push to GitHub: `git push origin main --tags`
4. GitHub Actions builds and creates release automatically

## CI/CD Pipeline

GitHub Actions automatically builds releases when tags are pushed:

### Workflow Triggers
- **Tags**: `v*` (e.g., `v1.0.0`)
- **Manual**: Workflow dispatch

### Build Matrix
- **macOS**: Latest (Intel + Apple Silicon)
- **Windows**: Latest (x64)
- **Linux**: Latest (x64)

### Artifacts
- Automatically uploaded to GitHub Releases
- Available as build artifacts for 30 days

## Configuration Files

### Core Configuration
- `package.json` - Main electron-builder configuration
- `build/entitlements.mac.plist` - macOS security entitlements
- `scripts/notarize.js` - macOS notarization script

### Build Resources
```
build/
├── entitlements.mac.plist    # macOS entitlements
└── icon.{icns,ico,png}      # Platform-specific icons (create these)

assets/
├── icon.icns               # macOS icon
├── icon.ico                # Windows icon
└── icon.png                # Linux icon
```

## Troubleshooting

### Common Issues

#### "Cannot find module" errors
```bash
# Clean and reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

#### macOS Gatekeeper issues
```bash
# Remove quarantine attribute
xattr -cr "/path/to/Chatterley.app"
```

#### Windows SmartScreen warnings
- Builds need to be code-signed to avoid warnings
- Submit to Microsoft for reputation building

#### Linux permission errors
```bash
# Make AppImage executable
chmod +x "Chatterley-*.AppImage"
```

### Debug Builds
```bash
# Build with debug info
DEBUG=electron-builder npm run dist:mac

# Build directory only (no packaging)
npm run electron:pack -- --dir
```

### Build Logs
- **Location**: `~/Library/Logs/Chatterley/` (macOS)
- **Location**: `%APPDATA%/Chatterley/logs/` (Windows)  
- **Location**: `~/.config/Chatterley/logs/` (Linux)

## Security Considerations

### Code Signing
- **macOS**: Prevents "Unknown Developer" warnings
- **Windows**: Prevents SmartScreen warnings
- **Required**: For distribution and auto-updates

### Notarization (macOS)
- **Required**: For macOS 10.15+ compatibility
- **Process**: Automatic via `scripts/notarize.js`
- **Requirements**: Apple Developer account

### Sandboxing
- **Status**: Disabled (Python integration requires full access)
- **Alternative**: Hardened Runtime with entitlements

## Performance Optimization

### Build Size Reduction
- Excludes dev dependencies and test files
- Compresses with maximum settings
- Filters unnecessary Python cache files

### Bundle Analysis
```bash
# Analyze bundle size
npx electron-builder --publish=never --analyze
```

## Support Matrix

| Platform | Architecture | Status | Notes |
|----------|-------------|--------|-------|
| macOS | x64 | ✅ | Intel Macs |
| macOS | arm64 | ✅ | Apple Silicon |
| Windows | x64 | ✅ | Windows 10+ |
| Windows | arm64 | ⚠️ | Experimental |
| Linux | x64 | ✅ | Ubuntu 18.04+ |
| Linux | arm64 | ❌ | Not supported |

## Resources

- [electron-builder Documentation](https://www.electron.build/)
- [electron-updater Documentation](https://github.com/electron-userland/electron-updater)
- [Apple Code Signing Guide](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution)
- [Windows Code Signing Guide](https://docs.microsoft.com/en-us/windows/msix/package/sign-app-package-using-signtool)