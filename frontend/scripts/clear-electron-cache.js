#!/usr/bin/env node
/*
 * Clear Electron and app caches to avoid stale data during debug runs.
 */
const fs = require('fs');
const path = require('path');
const os = require('os');

function rmrf(p) {
  try {
    if (fs.existsSync(p)) {
      fs.rmSync(p, { recursive: true, force: true });
      console.log(`[cache] removed: ${p}`);
    }
  } catch (e) {
    console.warn(`[cache] failed to remove ${p}: ${e.message}`);
  }
}

function resolveBase() {
  const platform = process.platform;
  if (platform === 'darwin') {
    return path.join(os.homedir(), 'Library', 'Application Support');
  }
  if (platform === 'win32') {
    return process.env.APPDATA || path.join(os.homedir(), 'AppData', 'Roaming');
  }
  // linux and others
  return process.env.XDG_CONFIG_HOME || path.join(os.homedir(), '.config');
}

function main() {
  const base = resolveBase();
  const apps = ['Electron', 'Chatterley'];
  const dirs = ['Cache', 'Code Cache', 'GPUCache'];

  console.log(`[cache] base dir: ${base}`);
  for (const app of apps) {
    for (const d of dirs) {
      const target = path.join(base, app, d);
      rmrf(target);
      // Some Electron versions store nested Cache_Data
      const cacheData = path.join(base, app, 'Cache', 'Cache_Data');
      rmrf(cacheData);
    }
  }
}

main();

