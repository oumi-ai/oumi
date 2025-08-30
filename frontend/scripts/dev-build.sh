#!/bin/bash

# Development build script for quick testing
# Usage: ./scripts/dev-build.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🔧 Development build for Chatterley Desktop"

cd "$PROJECT_DIR"

# Generate configs and build frontend
echo "⚙️ Generating configs and building Next.js app..."
npm run build:electron

# Compile Electron TypeScript
echo "📝 Compiling Electron TypeScript..."
npm run electron:compile

# Package with electron-builder (no distribution)
echo "📦 Packaging with electron-builder..."
npx electron-builder --dir

echo "✅ Development build completed!"
echo "🚀 Run the app from: dist/mac/Chatterley.app (or equivalent for your platform)"