#!/usr/bin/env node
/**
 * Download Python Distributions Script for Chatterley
 * Downloads cross-platform python-build-standalone distributions for bundling
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const { execSync } = require('child_process');

// Configuration
const PYTHON_VERSION = '3.11';  // Use Python 3.11 for good compatibility
const GITHUB_API_URL = 'https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest';

// Platform-specific distribution patterns
const DISTRIBUTIONS = {
  'darwin-x64': `cpython-${PYTHON_VERSION}.*-x86_64-apple-darwin-install_only.tar.gz`,
  'darwin-arm64': `cpython-${PYTHON_VERSION}.*-aarch64-apple-darwin-install_only.tar.gz`,  
  'win32-x64': `cpython-${PYTHON_VERSION}.*-x86_64-pc-windows-msvc-shared-install_only.tar.gz`,
  'linux-x64': `cpython-${PYTHON_VERSION}.*-x86_64-unknown-linux-gnu-install_only.tar.gz`
};

const FRONTEND_DIR = path.resolve(__dirname, '..');
const PYTHON_DIST_DIR = path.join(FRONTEND_DIR, 'python-dist');

console.log('🐍 Downloading Python distributions for Chatterley...');
console.log(`Frontend dir: ${FRONTEND_DIR}`);
console.log(`Python dist dir: ${PYTHON_DIST_DIR}`);

/**
 * Download a file from URL to local path
 */
function downloadFile(url, filepath) {
  return new Promise((resolve, reject) => {
    console.log(`📥 Downloading: ${path.basename(filepath)}`);
    
    const file = fs.createWriteStream(filepath);
    
    const request = https.get(url, (response) => {
      if (response.statusCode === 302 || response.statusCode === 301) {
        // Follow redirect
        downloadFile(response.headers.location, filepath).then(resolve).catch(reject);
        return;
      }
      
      if (response.statusCode !== 200) {
        reject(new Error(`Download failed: ${response.statusCode} ${response.statusMessage}`));
        return;
      }
      
      const totalSize = parseInt(response.headers['content-length'], 10);
      let downloadedSize = 0;
      let lastProgress = 0;
      
      response.pipe(file);
      
      response.on('data', (chunk) => {
        downloadedSize += chunk.length;
        const progress = Math.round((downloadedSize / totalSize) * 100);
        
        if (progress > lastProgress + 5) { // Update every 5%
          console.log(`   📊 Progress: ${progress}% (${(downloadedSize / 1024 / 1024).toFixed(1)}MB / ${(totalSize / 1024 / 1024).toFixed(1)}MB)`);
          lastProgress = progress;
        }
      });
      
      file.on('finish', () => {
        file.close();
        console.log(`   ✅ Downloaded: ${path.basename(filepath)}`);
        resolve();
      });
    });
    
    request.on('error', (error) => {
      fs.unlink(filepath, () => {}); // Clean up partial file
      reject(error);
    });
    
    file.on('error', (error) => {
      fs.unlink(filepath, () => {}); // Clean up partial file
      reject(error);
    });
  });
}

/**
 * Extract tar.gz file
 */
function extractTarGz(filepath, extractDir) {
  console.log(`📂 Extracting: ${path.basename(filepath)}`);
  
  try {
    // Create extract directory
    fs.mkdirSync(extractDir, { recursive: true });
    
    // Extract using system tar command (cross-platform)
    if (process.platform === 'win32') {
      // On Windows, use 7-zip or built-in tar if available
      try {
        execSync(`tar -xzf "${filepath}" -C "${extractDir}"`, { stdio: 'inherit' });
      } catch (error) {
        throw new Error('Failed to extract on Windows. Please ensure tar is available or use WSL.');
      }
    } else {
      // Unix-like systems
      execSync(`tar -xzf "${filepath}" -C "${extractDir}"`, { stdio: 'inherit' });
    }
    
    console.log(`   ✅ Extracted to: ${extractDir}`);
    
    // Clean up tar file
    fs.unlinkSync(filepath);
    console.log(`   🗑️  Cleaned up: ${path.basename(filepath)}`);
    
  } catch (error) {
    throw new Error(`Extraction failed: ${error.message}`);
  }
}

/**
 * Test Python distribution
 */
function testPythonDistribution(platformDir) {
  console.log(`🧪 Testing Python distribution...`);
  
  const platform = process.platform;
  let pythonPath;
  
  if (platform === 'win32') {
    pythonPath = path.join(platformDir, 'python.exe');
  } else {
    pythonPath = path.join(platformDir, 'bin', 'python3');
  }
  
  if (!fs.existsSync(pythonPath)) {
    throw new Error(`Python executable not found at: ${pythonPath}`);
  }
  
  try {
    const output = execSync(`"${pythonPath}" --version`, { encoding: 'utf8' });
    console.log(`   ✅ Python version: ${output.trim()}`);
    
    // Test basic functionality
    const testOutput = execSync(`"${pythonPath}" -c "import sys; print('Python test OK')"`, { encoding: 'utf8' });
    console.log(`   ✅ Python test: ${testOutput.trim()}`);
    
    return true;
  } catch (error) {
    console.error(`   ❌ Python test failed: ${error.message}`);
    return false;
  }
}

/**
 * Get current platform identifier
 */
function getCurrentPlatform() {
  const platform = process.platform;
  const arch = process.arch;
  
  if (platform === 'darwin') {
    return arch === 'arm64' ? 'darwin-arm64' : 'darwin-x64';
  } else if (platform === 'win32') {
    return 'win32-x64';
  } else if (platform === 'linux') {
    return 'linux-x64';
  } else {
    throw new Error(`Unsupported platform: ${platform}-${arch}`);
  }
}

/**
 * Fetch latest release information from GitHub
 */
async function getLatestRelease() {
  return new Promise((resolve, reject) => {
    const request = https.get(GITHUB_API_URL, {
      headers: {
        'User-Agent': 'Chatterley-Build-Script'
      }
    }, (response) => {
      let data = '';
      
      response.on('data', chunk => {
        data += chunk;
      });
      
      response.on('end', () => {
        try {
          const release = JSON.parse(data);
          resolve(release);
        } catch (error) {
          reject(new Error(`Failed to parse GitHub API response: ${error.message}`));
        }
      });
    });
    
    request.on('error', (error) => {
      reject(new Error(`Failed to fetch latest release: ${error.message}`));
    });
  });
}

/**
 * Find matching asset for platform
 */
function findMatchingAsset(assets, platform) {
  const pattern = DISTRIBUTIONS[platform];
  if (!pattern) {
    throw new Error(`No distribution pattern defined for platform: ${platform}`);
  }
  
  const regex = new RegExp(pattern);
  const asset = assets.find(asset => regex.test(asset.name));
  
  if (!asset) {
    console.log(`Available assets for ${platform}:`);
    assets.forEach(asset => {
      if (asset.name.includes('cpython') && asset.name.includes(PYTHON_VERSION)) {
        console.log(`  - ${asset.name}`);
      }
    });
    throw new Error(`No matching asset found for platform ${platform} with pattern: ${pattern}`);
  }
  
  return asset;
}

/**
 * Download and setup Python distribution for a platform
 */
async function setupPlatformDistribution(platform, asset, targetOnly = false) {
  console.log(`\n🔧 Setting up ${platform} distribution...`);
  
  const filename = asset.name;
  const downloadPath = path.join(PYTHON_DIST_DIR, filename);
  const platformDir = path.join(PYTHON_DIST_DIR, platform);
  
  // Clean existing platform directory
  if (fs.existsSync(platformDir)) {
    console.log(`🧹 Cleaning existing ${platform} distribution...`);
    fs.rmSync(platformDir, { recursive: true, force: true });
  }
  
  // Download
  await downloadFile(asset.browser_download_url, downloadPath);
  
  // Extract
  extractTarGz(downloadPath, platformDir);
  
  // Find the extracted Python directory (usually something like 'python')
  const contents = fs.readdirSync(platformDir);
  const pythonDir = contents.find(item => {
    const itemPath = path.join(platformDir, item);
    return fs.statSync(itemPath).isDirectory();
  });
  
  if (pythonDir) {
    // Move contents up one level to flatten structure
    const sourcePath = path.join(platformDir, pythonDir);
    const tempPath = path.join(PYTHON_DIST_DIR, `temp-${platform}`);
    
    // Move to temp location
    fs.renameSync(sourcePath, tempPath);
    
    // Remove old platform dir and replace with temp
    fs.rmSync(platformDir, { recursive: true, force: true });
    fs.renameSync(tempPath, platformDir);
  }
  
  // Test the distribution if it's for current platform
  if (!targetOnly && platform === getCurrentPlatform()) {
    const success = testPythonDistribution(platformDir);
    if (!success) {
      throw new Error(`Python distribution test failed for ${platform}`);
    }
  }
  
  console.log(`✅ ${platform} distribution ready`);
}

/**
 * Main function
 */
async function main() {
  try {
    // Create python-dist directory
    if (fs.existsSync(PYTHON_DIST_DIR)) {
      console.log('🧹 Cleaning existing python-dist directory...');
      fs.rmSync(PYTHON_DIST_DIR, { recursive: true, force: true });
    }
    fs.mkdirSync(PYTHON_DIST_DIR, { recursive: true });
    
    // Get latest release
    console.log('🔍 Fetching latest python-build-standalone release...');
    const release = await getLatestRelease();
    console.log(`📋 Latest release: ${release.tag_name}`);
    console.log(`📅 Published: ${new Date(release.published_at).toLocaleDateString()}`);
    
    // Determine which platforms to download
    const currentPlatform = getCurrentPlatform();
    console.log(`🖥️  Current platform: ${currentPlatform}`);
    
    // For now, just download current platform (can be expanded for CI builds)
    const platforms = [currentPlatform];
    
    // Check if we should download all platforms (CI mode)
    if (process.argv.includes('--all-platforms')) {
      platforms.push(...Object.keys(DISTRIBUTIONS).filter(p => p !== currentPlatform));
    }
    
    console.log(`📦 Will download distributions for: ${platforms.join(', ')}`);
    
    // Download and setup each platform
    for (const platform of platforms) {
      const asset = findMatchingAsset(release.assets, platform);
      console.log(`🎯 Found asset for ${platform}: ${asset.name}`);
      
      await setupPlatformDistribution(platform, asset, platform !== currentPlatform);
    }
    
    // Create distribution info file
    const distInfo = {
      version: release.tag_name,
      python_version: PYTHON_VERSION,
      created_at: new Date().toISOString(),
      platforms: platforms,
      release_url: release.html_url
    };
    
    fs.writeFileSync(
      path.join(PYTHON_DIST_DIR, 'distribution-info.json'),
      JSON.stringify(distInfo, null, 2)
    );
    
    console.log('\n🎉 Python distributions download complete!');
    console.log(`📁 Distributions location: ${PYTHON_DIST_DIR}`);
    console.log(`📏 Total size: ${execSync(`du -sh "${PYTHON_DIST_DIR}" 2>/dev/null | cut -f1 || echo "Unknown"`, { encoding: 'utf8' }).trim()}`);
    
    console.log('\n📊 Distribution summary:');
    platforms.forEach(platform => {
      const platformDir = path.join(PYTHON_DIST_DIR, platform);
      if (fs.existsSync(platformDir)) {
        try {
          const size = execSync(`du -sh "${platformDir}" 2>/dev/null | cut -f1 || echo "Unknown"`, { encoding: 'utf8' }).trim();
          console.log(`   ${platform}: ${size}`);
        } catch {
          console.log(`   ${platform}: Available`);
        }
      }
    });
    
    console.log('\n📝 Next steps:');
    console.log('1. The Python distributions will be bundled with the Electron app');
    console.log('2. The PythonEnvironmentManager will use these for creating virtual environments');
    console.log('3. Run the app build to include these distributions in the final package');
    
  } catch (error) {
    console.error('❌ Error:', error.message);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  main();
}