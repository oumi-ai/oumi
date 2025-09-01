#!/usr/bin/env node

const { execSync } = require('child_process');
const path = require('path');

/**
 * Cleanup script to unmount any lingering DMG volumes
 * This prevents "File exists" errors when building multiple architectures
 */

const DMG_VOLUME_NAME = 'Chatterley 0.1.0';
const MOUNT_PATH = `/Volumes/${DMG_VOLUME_NAME}`;

console.log('🧹 Cleaning up DMG volumes...');

try {
  // Check if volume is mounted
  execSync(`test -d "${MOUNT_PATH}"`, { stdio: 'ignore' });
  
  console.log(`📀 Found mounted volume: ${MOUNT_PATH}`);
  
  // Unmount the volume
  execSync(`diskutil unmount "${MOUNT_PATH}"`, { stdio: 'pipe' });
  console.log('✅ Successfully unmounted volume');
  
} catch (error) {
  // Volume not mounted or unmount failed - this is normal
  console.log('✅ No cleanup needed');
}

console.log('🧹 DMG cleanup complete');