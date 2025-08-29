#!/usr/bin/env node

/**
 * Script to wait for Next.js dev server to be ready on any available port
 * This replaces the hardcoded wait-on http://localhost:3000
 */

const http = require('http');
const { spawn } = require('child_process');

// Ports to check (in order of preference)
const PORTS_TO_CHECK = [3000, 3001, 3002, 3003, 3004, 3005];
const MAX_ATTEMPTS = 60; // 30 seconds total
const RETRY_DELAY = 500; // 500ms between attempts

async function checkPort(port) {
  return new Promise((resolve) => {
    const req = http.get(`http://localhost:${port}`, (res) => {
      resolve(res.statusCode < 400);
    });
    
    req.on('error', () => {
      resolve(false);
    });
    
    req.setTimeout(1000, () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function findNextPort() {
  for (const port of PORTS_TO_CHECK) {
    if (await checkPort(port)) {
      return port;
    }
  }
  return null;
}

async function waitForNext() {
  console.log('🔍 Waiting for Next.js dev server to be ready...');
  
  let attempts = 0;
  
  while (attempts < MAX_ATTEMPTS) {
    const port = await findNextPort();
    
    if (port) {
      console.log(`✅ Next.js dev server ready on http://localhost:${port}`);
      return port;
    }
    
    attempts++;
    if (attempts < MAX_ATTEMPTS) {
      process.stdout.write(`⏳ Waiting... (${attempts}/${MAX_ATTEMPTS})\r`);
      await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
    }
  }
  
  throw new Error('❌ Next.js dev server failed to start within timeout period');
}

async function startElectron() {
  try {
    const port = await waitForNext();
    
    // Start Electron with the detected port
    console.log('🚀 Starting Electron...');
    const electronProcess = spawn('electron', ['dist/electron/main.js'], {
      stdio: 'inherit',
      env: {
        ...process.env,
        NEXT_DEV_PORT: port.toString()
      }
    });
    
    electronProcess.on('exit', (code) => {
      console.log(`\n📱 Electron exited with code ${code}`);
      process.exit(code);
    });
    
    // Handle cleanup
    process.on('SIGINT', () => {
      console.log('\n🛑 Shutting down...');
      electronProcess.kill('SIGINT');
    });
    
    process.on('SIGTERM', () => {
      electronProcess.kill('SIGTERM');
    });
    
  } catch (error) {
    console.error(error.message);
    process.exit(1);
  }
}

// Run the script
startElectron();