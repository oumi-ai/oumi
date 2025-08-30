#!/usr/bin/env node

/**
 * Version management script for Chatterley Desktop
 * Usage:
 *   node scripts/version.js patch    # 1.0.0 -> 1.0.1
 *   node scripts/version.js minor    # 1.0.0 -> 1.1.0
 *   node scripts/version.js major    # 1.0.0 -> 2.0.0
 *   node scripts/version.js 1.2.3    # Set specific version
 *   node scripts/version.js --current # Show current version
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const packageJsonPath = path.join(__dirname, '../package.json');

function getCurrentVersion() {
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
  return packageJson.version;
}

function setVersion(newVersion) {
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
  packageJson.version = newVersion;
  fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2) + '\n');
  return newVersion;
}

function incrementVersion(type) {
  const current = getCurrentVersion();
  const parts = current.split('.').map(Number);
  
  switch (type) {
    case 'major':
      parts[0]++;
      parts[1] = 0;
      parts[2] = 0;
      break;
    case 'minor':
      parts[1]++;
      parts[2] = 0;
      break;
    case 'patch':
      parts[2]++;
      break;
    default:
      throw new Error(`Invalid increment type: ${type}`);
  }
  
  return parts.join('.');
}

function isValidVersion(version) {
  return /^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?$/.test(version);
}

function main() {
  const args = process.argv.slice(2);
  
  if (args.length === 0) {
    console.log('Usage: node scripts/version.js <patch|minor|major|version|--current>');
    process.exit(1);
  }
  
  const command = args[0];
  
  if (command === '--current') {
    console.log(getCurrentVersion());
    return;
  }
  
  let newVersion;
  
  if (['patch', 'minor', 'major'].includes(command)) {
    newVersion = incrementVersion(command);
  } else if (isValidVersion(command)) {
    newVersion = command;
  } else {
    console.error(`Invalid version or command: ${command}`);
    process.exit(1);
  }
  
  const oldVersion = getCurrentVersion();
  setVersion(newVersion);
  
  console.log(`Version updated: ${oldVersion} -> ${newVersion}`);
  
  // Create git tag if in a git repository
  try {
    execSync('git rev-parse --git-dir', { stdio: 'ignore' });
    
    console.log('Creating git commit and tag...');
    execSync(`git add package.json`);
    execSync(`git commit -m "chore: bump version to ${newVersion}"`);
    execSync(`git tag -a v${newVersion} -m "Release v${newVersion}"`);
    
    console.log(`Git tag v${newVersion} created.`);
    console.log('To push: git push origin main --tags');
  } catch (error) {
    console.log('Not a git repository or git not available - skipping tag creation');
  }
}

if (require.main === module) {
  main();
}