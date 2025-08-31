#!/usr/bin/env node

/**
 * Debug Electron Production Build Script
 * 
 * This script helps debug production build issues by:
 * 1. Building the production version
 * 2. Running Electron with enhanced logging
 * 3. Opening DevTools automatically
 * 4. Providing helpful debugging information
 */

const { spawn, execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

console.log('🐛 Electron Production Debug Mode');
console.log('================================\n');

// Check if we need to build first
const buildRequired = process.argv.includes('--build') || !fs.existsSync(path.join(__dirname, '..', 'out'));

if (buildRequired) {
  console.log('📦 Building production version...');
  try {
    execSync('npm run build:electron && npm run electron:compile', { 
      stdio: 'inherit',
      cwd: path.join(__dirname, '..')
    });
    console.log('✅ Build complete!\n');
  } catch (error) {
    console.error('❌ Build failed:', error.message);
    process.exit(1);
  }
}

console.log('🚀 Starting Electron with debugging enabled...\n');
console.log('Debug features enabled:');
console.log('  • DevTools will open automatically');
console.log('  • Console logging enhanced');
console.log('  • Network requests logged');
console.log('  • File access attempts logged');
console.log('  • IPC messages logged\n');

// Set debugging environment variables
// Note: We don't set NODE_ENV here to avoid conflicts with Next.js dev server
const env = {
  ...process.env,
  DEBUG: '*',
  ELECTRON_IS_DEV: '1', // This enables DevTools while using production build
  ELECTRON_DEBUG_PRODUCTION: '1',
  DEBUG_COLORS: '1',
};

// Launch Electron with debugging
const electronProcess = spawn('electron', [
  path.join(__dirname, '..', 'dist', 'electron', 'main.js'),
  '--inspect=9229',
  '--remote-debugging-port=9222',
  '--enable-logging',
  '--log-level=0', // Show all logs
  '--v=1', // Verbose logging
], {
  env,
  stdio: 'inherit',
  cwd: path.join(__dirname, '..')
});

electronProcess.on('close', (code) => {
  console.log(`\n🏁 Electron process exited with code ${code}`);
});

electronProcess.on('error', (error) => {
  console.error('❌ Failed to start Electron:', error.message);
});

// Handle shutdown gracefully
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down Electron debug session...');
  electronProcess.kill('SIGTERM');
});

console.log('💡 Tips for debugging:');
console.log('  • Press Ctrl+Shift+I in the app to open DevTools');
console.log('  • Check the Console tab for JavaScript errors');
console.log('  • Check the Network tab for failed requests');
console.log('  • Check the Sources tab for file loading issues');
console.log('  • Main process logs appear in this terminal');
console.log('  • Use Chrome DevTools at chrome://inspect for main process debugging\n');