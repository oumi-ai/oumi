#!/bin/bash

# Local build script for Chatterley Desktop
# Usage: ./scripts/build-local.sh [platform]
# Platforms: mac, win, linux, all

set -e

PLATFORM=${1:-"$(uname | tr '[:upper:]' '[:lower:]')"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Best-effort cleanup so rebuilds don't fail on existing artifacts/symlinks
cleanup_macos_dmg_volume() {
    # Determine product name and version from package.json
    local product_name
    local version
    product_name=$(node -p "(p=> (p.build && p.build.productName) || p.productName || p.name || 'App')(require('./package.json'))" 2>/dev/null || echo "App")
    version=$(node -p "require('./package.json').version" 2>/dev/null || echo "")

    # If we couldn't determine, do nothing
    if [[ -z "$product_name" || -z "$version" ]]; then
        return 0
    fi

    local volume="/Volumes/${product_name} ${version}"

    # Detach if mounted
    if mount | grep -Fq "$volume"; then
        echo "🔌 Detaching existing DMG volume: $volume"
        hdiutil detach -force "$volume" || true
    fi

    # Remove any stale mount directory or symlinks inside
    if [[ -d "$volume" ]]; then
        echo "🧹 Removing stale volume directory: $volume"
        rm -rf "$volume" || true
    fi
}

echo "🚀 Building Chatterley Desktop for platform: $PLATFORM"
echo "📁 Project directory: $PROJECT_DIR"

cd "$PROJECT_DIR"

# Check if we're in the frontend directory
if [[ ! -f "package.json" ]]; then
    echo "❌ Error: Must run from frontend directory"
    exit 1
fi

# Install dependencies if needed
if [[ ! -d "node_modules" ]]; then
    echo "📦 Installing npm dependencies..."
    npm install
fi

# Clean existing build artifacts
if [[ -d "dist" ]]; then
    echo "🧹 Removing existing dist directory..."
    rm -rf dist
fi

# Generate static configs
echo "⚙️ Generating static configs..."
npm run generate-configs

# Download Python distributions based on platform
case "$PLATFORM" in
    "mac" | "darwin")
        echo "🍎 Building for macOS..."
        echo "🧽 Pre-cleaning any stale DMG mounts/symlinks..."
        cleanup_macos_dmg_volume
        npm run dist:mac
        ;;
    "win" | "windows")
        echo "🪟 Building for Windows..."
        npm run dist:win
        ;;
    "linux")
        echo "🐧 Building for Linux..."
        npm run dist:linux
        ;;
    "all")
        echo "🌍 Building for all platforms..."
        echo "📦 Downloading Python distributions for all platforms..."
        npm run download-python -- --all-platforms
        echo "🍎 Building macOS packages..."
        echo "🧽 Pre-cleaning any stale DMG mounts/symlinks..."
        cleanup_macos_dmg_volume
        npm run dist:mac
        echo "🪟 Building Windows packages..."
        npm run dist:win
        echo "🐧 Building Linux packages..."
        npm run dist:linux
        ;;
    *)
        echo "❌ Unknown platform: $PLATFORM"
        echo "Supported platforms: mac, win, linux, all"
        exit 1
        ;;
esac

echo "✅ Build completed! Check dist/packages/ for installers."

# Show build artifacts
echo ""
echo "📦 Build artifacts:"
if [[ -d "dist/packages" ]]; then
    ls -la dist/packages/
else
    echo "No build artifacts found."
fi
