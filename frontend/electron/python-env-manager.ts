/**
 * Python Environment Manager - handles setup and management of standalone Python environments
 * for Chatterley without requiring conda installation
 */

import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { app } from 'electron';
import log from 'electron-log';
import { SystemDetector, SystemInfo } from './system-detector';

export interface SetupProgress {
  step: string;
  progress: number;      // 0-100
  message: string;
  isComplete: boolean;
  estimatedTimeRemaining?: string;
}

export interface EnvironmentInfo {
  path: string;
  pythonPath: string;
  isValid: boolean;
  createdAt?: string;
  lastUsed?: string;
  systemInfo?: SystemInfo;
}

export class PythonEnvironmentManager {
  private progressCallback?: (progress: SetupProgress) => void;
  private isSettingUp: boolean = false;
  private setupProcess: ChildProcess | null = null;
  
  constructor() {
    // Initialize logging
    log.info('[PythonEnvManager] Initialized');
  }

  /**
   * Get the user app data directory for storing the Python environment
   */
  private getUserDataDir(): string {
    const platform = process.platform;
    const homeDir = os.homedir();
    
    switch (platform) {
      case 'darwin':
        return path.join(homeDir, 'Library', 'Application Support', 'Chatterley');
      case 'win32':
        return path.join(process.env.APPDATA || path.join(homeDir, 'AppData', 'Roaming'), 'Chatterley');
      case 'linux':
        return path.join(process.env.XDG_DATA_HOME || path.join(homeDir, '.local', 'share'), 'Chatterley');
      default:
        return path.join(homeDir, '.chatterley');
    }
  }

  /**
   * Get the path to the bundled Python distribution
   */
  private getBundledPythonPath(): string {
    const resourcesPath = app.isPackaged 
      ? process.resourcesPath 
      : path.join(__dirname, '../..');
      
    const pythonDist = path.join(resourcesPath, 'python-dist');
    
    // Platform-specific directory structure
    const platform = process.platform;
    const arch = process.arch;
    
    let platformDir: string;
    if (platform === 'darwin') {
      platformDir = `darwin-${arch}`; // darwin-arm64, darwin-x64
    } else if (platform === 'win32') {
      platformDir = `win32-${arch}`;  // win32-x64
    } else {
      platformDir = `linux-${arch}`;  // linux-x64
    }
    
    // Find the Python executable in the platform-specific directory
    if (platform === 'win32') {
      return path.join(pythonDist, platformDir, 'python.exe');
    } else {
      return path.join(pythonDist, platformDir, 'bin', 'python');
    }
  }

  /**
   * Get the path where we'll create the Chatterley Python environment
   */
  private getEnvironmentPath(): string {
    return path.join(this.getUserDataDir(), 'python-env');
  }

  /**
   * Get info file path for storing environment metadata
   */
  private getEnvironmentInfoPath(): string {
    return path.join(this.getEnvironmentPath(), 'chatterley-env-info.json');
  }

  /**
   * Check if the Python environment already exists and is valid
   */
  public async checkEnvironment(): Promise<EnvironmentInfo> {
    const envPath = this.getEnvironmentPath();
    const infoPath = this.getEnvironmentInfoPath();
    
    log.info(`[PythonEnvManager] Checking environment at: ${envPath}`);
    
    const result: EnvironmentInfo = {
      path: envPath,
      pythonPath: '',
      isValid: false
    };

    if (!fs.existsSync(envPath)) {
      log.info('[PythonEnvManager] Environment directory does not exist');
      return result;
    }

    // Find Python executable in the environment
    const platform = process.platform;
    const pythonExe = platform === 'win32' 
      ? path.join(envPath, 'Scripts', 'python.exe')
      : path.join(envPath, 'bin', 'python');
    
    if (!fs.existsSync(pythonExe)) {
      log.info('[PythonEnvManager] Python executable not found in environment');
      return result;
    }

    result.pythonPath = pythonExe;

    // Load environment info if available
    if (fs.existsSync(infoPath)) {
      try {
        const info = JSON.parse(fs.readFileSync(infoPath, 'utf8'));
        result.createdAt = info.createdAt;
        result.lastUsed = info.lastUsed;
        result.systemInfo = info.systemInfo;
      } catch (error) {
        log.warn('[PythonEnvManager] Failed to read environment info:', error);
      }
    }

    // Test if the environment is functional
    try {
      const isValid = await this.testEnvironment(pythonExe);
      result.isValid = isValid;
      
      if (isValid) {
        // Update last used timestamp, preserving system info
        await this.updateEnvironmentInfo({
          createdAt: result.createdAt || new Date().toISOString(),
          lastUsed: new Date().toISOString(),
          systemInfo: result.systemInfo
        });
      }
      
      return result;
    } catch (error) {
      log.error('[PythonEnvManager] Environment test failed:', error);
      return result;
    }
  }

  /**
   * Test if the Python environment has oumi installed and working
   */
  private async testEnvironment(pythonPath: string): Promise<boolean> {
    return new Promise((resolve) => {
      const testProcess = spawn(pythonPath, ['-c', 'import oumi; print("OK")'], {
        stdio: 'pipe',
        timeout: 10000
      });

      let output = '';
      let hasOutput = false;

      testProcess.stdout?.on('data', (data) => {
        output += data.toString();
        hasOutput = true;
      });

      testProcess.on('close', (code) => {
        const success = code === 0 && hasOutput && output.trim().includes('OK');
        log.info(`[PythonEnvManager] Environment test result: ${success} (code: ${code}, output: ${output.trim()})`);
        resolve(success);
      });

      testProcess.on('error', (error) => {
        log.error('[PythonEnvManager] Environment test error:', error);
        resolve(false);
      });
    });
  }

  /**
   * Set up the Python environment from scratch
   */
  public async setupEnvironment(): Promise<EnvironmentInfo> {
    if (this.isSettingUp) {
      throw new Error('Environment setup already in progress');
    }

    this.isSettingUp = true;
    log.info('[PythonEnvManager] Starting environment setup...');

    try {
      // Step 1: Check bundled Python exists
      await this.reportProgress('checking', 5, 'Checking bundled Python...');
      const bundledPython = this.getBundledPythonPath();
      
      if (!fs.existsSync(bundledPython)) {
        throw new Error(`Bundled Python not found at: ${bundledPython}`);
      }

      // Step 2: Create environment directory
      await this.reportProgress('creating', 15, 'Creating environment directory...');
      const envPath = this.getEnvironmentPath();
      
      // Clean existing environment if it exists
      if (fs.existsSync(envPath)) {
        log.info('[PythonEnvManager] Removing existing environment');
        fs.rmSync(envPath, { recursive: true, force: true });
      }
      
      fs.mkdirSync(envPath, { recursive: true });

      // Step 3: Create virtual environment
      await this.reportProgress('venv', 25, 'Creating virtual environment...');
      await this.createVirtualEnvironment(bundledPython, envPath);

      // Step 4: Get environment Python path
      const platform = process.platform;
      const envPython = platform === 'win32' 
        ? path.join(envPath, 'Scripts', 'python.exe')
        : path.join(envPath, 'bin', 'python');

      // Step 5: Install uv
      await this.reportProgress('uv', 45, 'Installing uv package manager...');
      await this.installUv(envPython);

      // Step 6: Install oumi dependencies
      await this.reportProgress('oumi', 65, 'Installing Oumi dependencies...');
      try {
        await this.installOumiDependencies(envPython, envPath);
      } catch (error) {
        log.error('[PythonEnvManager] Environment setup failed during Oumi installation:', error);
        throw error; // Re-throw to be caught by outer catch block
      }

      // Step 7: Test installation
      await this.reportProgress('testing', 90, 'Testing installation...');
      const isValid = await this.testEnvironment(envPython);
      
      if (!isValid) {
        throw new Error('Environment test failed after installation');
      }

      // Step 8: Detect system information and save environment info
      await this.reportProgress('finishing', 95, 'Detecting system capabilities...');
      
      let systemInfo: SystemInfo;
      try {
        systemInfo = await SystemDetector.detectSystem();
        log.info('[PythonEnvManager] System detection completed:', {
          platform: systemInfo.platform,
          architecture: systemInfo.architecture,
          totalRAM: `${systemInfo.totalRAM}GB`,
          cudaAvailable: systemInfo.cudaAvailable,
          cudaDevices: systemInfo.cudaDevices.length
        });
      } catch (error) {
        log.warn('[PythonEnvManager] System detection failed:', error);
        // Create minimal system info if detection fails
        systemInfo = {
          architecture: process.arch,
          platform: process.platform,
          platformVersion: 'Unknown',
          cpuModel: 'Unknown',
          totalRAM: Math.round(os.totalmem() / (1024 * 1024 * 1024)),
          availableRAM: Math.round(os.freemem() / (1024 * 1024 * 1024)),
          cudaAvailable: false,
          cudaDevices: [],
          detectedAt: new Date().toISOString(),
          fingerprint: 'unknown'
        };
      }
      
      const now = new Date().toISOString();
      await this.updateEnvironmentInfo({
        createdAt: now,
        lastUsed: now,
        systemInfo
      });

      await this.reportProgress('complete', 100, 'Setup complete!', true);

      log.info('[PythonEnvManager] Environment setup completed successfully');
      
      return {
        path: envPath,
        pythonPath: envPython,
        isValid: true,
        createdAt: now,
        lastUsed: now,
        systemInfo
      };

    } catch (error) {
      log.error('[PythonEnvManager] Environment setup failed:', error);
      throw error;
    } finally {
      this.isSettingUp = false;
      this.setupProcess = null;
    }
  }

  /**
   * Create virtual environment using bundled Python
   */
  private async createVirtualEnvironment(pythonPath: string, envPath: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const venvProcess = spawn(pythonPath, ['-m', 'venv', envPath], {
        stdio: 'pipe'
      });

      let stderr = '';
      
      venvProcess.stderr?.on('data', (data) => {
        stderr += data.toString();
      });

      venvProcess.on('close', (code) => {
        if (code === 0) {
          log.info('[PythonEnvManager] Virtual environment created successfully');
          resolve();
        } else {
          log.error(`[PythonEnvManager] Virtual environment creation failed with code ${code}`);
          log.error(`[PythonEnvManager] Stderr: ${stderr}`);
          reject(new Error(`Failed to create virtual environment: ${stderr}`));
        }
      });

      venvProcess.on('error', (error) => {
        log.error('[PythonEnvManager] Virtual environment creation error:', error);
        reject(error);
      });
    });
  }

  /**
   * Install uv package manager
   */
  private async installUv(pythonPath: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const installProcess = spawn(pythonPath, ['-m', 'pip', 'install', 'uv'], {
        stdio: 'pipe'
      });

      this.setupProcess = installProcess;
      let stderr = '';
      
      installProcess.stderr?.on('data', (data) => {
        stderr += data.toString();
      });

      installProcess.on('close', (code) => {
        if (code === 0) {
          log.info('[PythonEnvManager] uv installed successfully');
          resolve();
        } else {
          log.error(`[PythonEnvManager] uv installation failed with code ${code}`);
          log.error(`[PythonEnvManager] Stderr: ${stderr}`);
          reject(new Error(`Failed to install uv: ${stderr}`));
        }
      });

      installProcess.on('error', (error) => {
        log.error('[PythonEnvManager] uv installation error:', error);
        reject(error);
      });
    });
  }

  /**
   * Determine required extras based on system capabilities
   */
  private async getRequiredExtras(): Promise<string[]> {
    const extras: string[] = [];
    
    // Always include interactive (for prompt_toolkit, etc.)
    extras.push('interactive');
    
    try {
      // Detect system capabilities
      const systemInfo = await SystemDetector.detectSystem();
      
      if (systemInfo.cudaAvailable && systemInfo.cudaDevices.length > 0) {
        // CUDA is available - include GPU and quantization extras
        extras.push('gpu', 'quantization');
        log.info(`[PythonEnvManager] CUDA detected: ${systemInfo.cudaDevices.length} device(s) - including GPU extras`);
      } else {
        // No CUDA available - include llama_cpp for CPU inference
        extras.push('llama_cpp');
        log.info('[PythonEnvManager] No CUDA detected - including llama_cpp for CPU-only installation');
        
        // For Windows and Linux without CUDA, also include ci_cpu
        if (systemInfo.platform === 'win32' || systemInfo.platform === 'linux') {
          extras.push('ci_cpu');
          log.info('[PythonEnvManager] Including ci_cpu extras for Windows/Linux CPU-only installation');
        }
      }
      
    } catch (error) {
      log.warn('[PythonEnvManager] System detection failed, using safe defaults:', error);
      // If system detection fails, use conservative approach
      
      // Add llama_cpp for CPU inference as fallback
      extras.push('llama_cpp');
      
      // Add ci_cpu for Windows/Linux as a safe fallback
      if (process.platform === 'win32' || process.platform === 'linux') {
        extras.push('ci_cpu');
      }
    }
    
    return extras;
  }

  /**
   * Install Oumi and dependencies using uv
   */
  private async installOumiDependencies(pythonPath: string, envPath: string): Promise<void> {
    // Get path to bundled oumi source
    const resourcesPath = app.isPackaged 
      ? process.resourcesPath 
      : path.join(__dirname, '../..');  // In development, use project root
    const oumiSourcePath = app.isPackaged
      ? path.join(resourcesPath, 'python', 'oumi')  // In packaged app: resources/python/oumi/
      : path.join(__dirname, '../../..');           // In development: project root (has pyproject.toml)
    
    log.info(`[PythonEnvManager] Resource path: ${resourcesPath}`);
    log.info(`[PythonEnvManager] Oumi source path: ${oumiSourcePath}`);
    
    // Verify the path exists
    if (!fs.existsSync(oumiSourcePath)) {
      const errorMsg = `Oumi source path does not exist: ${oumiSourcePath}`;
      log.error(`[PythonEnvManager] ${errorMsg}`);
      throw new Error(errorMsg);
    }
    
    // Verify pyproject.toml exists
    const pyprojectPath = path.join(oumiSourcePath, 'pyproject.toml');
    if (!fs.existsSync(pyprojectPath)) {
      const errorMsg = `pyproject.toml not found at: ${pyprojectPath}`;
      log.error(`[PythonEnvManager] ${errorMsg}`);
      throw new Error(errorMsg);
    }
    
    log.info(`[PythonEnvManager] Found pyproject.toml at: ${pyprojectPath}`);
    
    // Determine appropriate extras based on system capabilities
    const extras = await this.getRequiredExtras();
    log.info(`[PythonEnvManager] Selected extras: ${extras.join(', ')}`);
    await this.reportProgress('oumi', 70, `Installing Oumi with extras: ${extras.join(', ')}...`);

    return new Promise((resolve, reject) => {
      
      // Use uv to install oumi in development mode with appropriate extras
      const uvPath = process.platform === 'win32' 
        ? path.join(envPath, 'Scripts', 'uv.exe')
        : path.join(envPath, 'bin', 'uv');
        
      const packageSpec = extras.length > 0 
        ? `${oumiSourcePath}[${extras.join(',')}]`  // Install with extras
        : oumiSourcePath;                            // Install without extras
      
      log.info(`[PythonEnvManager] Installing: ${packageSpec}`);
        
      const installProcess = spawn(uvPath, [
        'pip', 'install', 
        '-e', packageSpec,
        '-v'  // Verbose output to get more progress info
      ], {
        stdio: 'pipe',
        env: {
          ...process.env,
          VIRTUAL_ENV: envPath,
          PATH: `${path.dirname(uvPath)}:${process.env.PATH}`
        }
      });

      this.setupProcess = installProcess;
      let stderr = '';
      let stdout = '';
      let installedPackages = 0;
      
      installProcess.stdout?.on('data', (data) => {
        const output = data.toString();
        stdout += output;
        
        // Log all output
        const lines = output.trim().split('\n');
        lines.forEach(async (line: string) => {
          if (line.trim()) {
            log.info(`[PythonEnvManager] Install: ${line.trim()}`);
            
            // Parse installation progress
            await this.parseInstallationProgress(line, installedPackages);
          }
        });
      });
      
      installProcess.stderr?.on('data', (data) => {
        const output = data.toString();
        stderr += output;
        
        // Log stderr and look for progress info there too
        const lines = output.trim().split('\n');
        lines.forEach(async (line: string) => {
          if (line.trim()) {
            log.info(`[PythonEnvManager] Install stderr: ${line.trim()}`);
            
            // Parse installation progress from stderr too
            await this.parseInstallationProgress(line, installedPackages);
          }
        });
      });

      installProcess.on('close', (code) => {
        if (code === 0) {
          log.info('[PythonEnvManager] Oumi dependencies installed successfully');
          resolve();
        } else {
          log.error(`[PythonEnvManager] Oumi installation failed with code ${code}`);
          log.error(`[PythonEnvManager] Stderr: ${stderr}`);
          reject(new Error(`Failed to install oumi: ${stderr}`));
        }
      });

      installProcess.on('error', (error) => {
        log.error('[PythonEnvManager] Oumi installation error:', error);
        reject(error);
      });
    });
  }

  /**
   * Parse installation progress from uv output and report to UI
   */
  private async parseInstallationProgress(line: string, installedPackages: number): Promise<void> {
    try {
      // Look for package download/install patterns
      if (line.includes('Downloading') && line.includes('(')) {
        const packageMatch = line.match(/Downloading\s+([^\s]+)/);
        if (packageMatch) {
          const packageName = packageMatch[1];
          await this.reportProgress('oumi', 75 + (installedPackages * 2), `Downloading ${packageName}...`);
        }
      } else if (line.includes('Installing') && line.includes('(')) {
        const packageMatch = line.match(/Installing\s+([^\s]+)/);
        if (packageMatch) {
          const packageName = packageMatch[1];
          installedPackages++;
          await this.reportProgress('oumi', 75 + (installedPackages * 2), `Installing ${packageName}...`);
        }
      } else if (line.includes('Collecting')) {
        const packageMatch = line.match(/Collecting\s+([^\s]+)/);
        if (packageMatch) {
          const packageName = packageMatch[1];
          await this.reportProgress('oumi', 70 + (installedPackages * 1), `Resolving dependencies for ${packageName}...`);
        }
      } else if (line.includes('Building wheel')) {
        const packageMatch = line.match(/Building wheel.*?for\s+([^\s]+)/);
        if (packageMatch) {
          const packageName = packageMatch[1];
          await this.reportProgress('oumi', 80 + (installedPackages * 1), `Building ${packageName}...`);
        }
      } else if (line.includes('Successfully installed')) {
        await this.reportProgress('oumi', 85, 'Installation completed, verifying packages...');
      } else if (line.toLowerCase().includes('resolving') || line.toLowerCase().includes('downloading')) {
        // Generic progress for any resolving/downloading activity
        await this.reportProgress('oumi', 72, 'Resolving package dependencies...');
      }
    } catch (error) {
      // Don't let progress parsing errors fail the installation
      log.warn('[PythonEnvManager] Error parsing installation progress:', error);
    }
  }

  /**
   * Update environment info file
   */
  private async updateEnvironmentInfo(info: { createdAt: string; lastUsed: string; systemInfo?: SystemInfo }): Promise<void> {
    const infoPath = this.getEnvironmentInfoPath();
    const os = require('os');
    
    // Debug architecture detection
    log.info('[PythonEnvManager] Architecture Detection Debug:');
    log.info(`  process.arch: ${process.arch}`);
    log.info(`  os.arch(): ${os.arch()}`);
    log.info(`  process.platform: ${process.platform}`);
    
    // If systemInfo is not provided, detect it now
    let systemInfo = info.systemInfo;
    if (!systemInfo) {
      log.info('[PythonEnvManager] SystemInfo not provided, detecting now...');
      try {
        const { SystemDetector } = await import('./system-detector');
        systemInfo = await SystemDetector.detectSystem();
        log.info('[PythonEnvManager] SystemInfo detected:', systemInfo);
      } catch (error) {
        log.error('[PythonEnvManager] Failed to detect system info:', error);
        // Create minimal system info as fallback
        systemInfo = {
          platform: process.platform,
          architecture: process.arch,
          cpuModel: 'Unknown',
          totalRAM: 0,
          availableRAM: 0,
          platformVersion: 'Unknown',
          cudaAvailable: false,
          cudaDevices: [],
          detectedAt: new Date().toISOString(),
          fingerprint: 'unknown'
        };
      }
    }
    
    const envInfo = {
      version: '1.1', // Updated version to include system info
      platform: process.platform,
      arch: process.arch, // Use process.arch consistently everywhere
      ...info,
      systemInfo // Ensure systemInfo is always included
    };
    
    try {
      fs.writeFileSync(infoPath, JSON.stringify(envInfo, null, 2));
      log.info('[PythonEnvManager] Environment info updated with arch:', envInfo.arch);
      log.info('[PythonEnvManager] Environment info updated with systemInfo:', !!envInfo.systemInfo);
    } catch (error) {
      log.warn('[PythonEnvManager] Failed to update environment info:', error);
    }
  }

  /**
   * Report progress to callback if available
   */
  private async reportProgress(step: string, progress: number, message: string, isComplete = false): Promise<void> {
    if (this.progressCallback) {
      this.progressCallback({
        step,
        progress,
        message,
        isComplete
      });
    }
    
    // Small delay to ensure UI updates
    await new Promise(resolve => setTimeout(resolve, 100));
  }

  /**
   * Set progress callback for setup updates
   */
  public setProgressCallback(callback: (progress: SetupProgress) => void): void {
    this.progressCallback = callback;
  }

  /**
   * Cancel ongoing setup
   */
  public cancelSetup(): void {
    if (this.setupProcess) {
      log.info('[PythonEnvManager] Cancelling setup...');
      this.setupProcess.kill('SIGTERM');
      this.setupProcess = null;
    }
    this.isSettingUp = false;
  }

  /**
   * Check if setup is currently in progress
   */
  public isSetupInProgress(): boolean {
    return this.isSettingUp;
  }

  /**
   * Get user data directory path for display to user
   */
  public getUserDataPath(): string {
    return this.getUserDataDir();
  }

  /**
   * Remove the Python environment (cleanup)
   */
  public async removeEnvironment(): Promise<void> {
    const envPath = this.getEnvironmentPath();
    if (fs.existsSync(envPath)) {
      log.info('[PythonEnvManager] Removing Python environment');
      fs.rmSync(envPath, { recursive: true, force: true });
    }
  }

  /**
   * Check if the system has changed significantly since environment creation
   */
  public async checkSystemChanges(environmentInfo: EnvironmentInfo): Promise<{ hasChanged: boolean; changes: string[]; shouldRebuild: boolean }> {
    try {
      if (!environmentInfo.systemInfo) {
        return { 
          hasChanged: true, 
          changes: ['No system information available from previous setup'],
          shouldRebuild: true 
        };
      }

      const currentSystem = await SystemDetector.detectSystem();
      const oldSystem = environmentInfo.systemInfo;
      const changes: string[] = [];
      
      // Check for significant changes
      if (oldSystem.platform !== currentSystem.platform) {
        changes.push(`Platform changed from ${oldSystem.platform} to ${currentSystem.platform}`);
      }
      
      if (oldSystem.architecture !== currentSystem.architecture) {
        changes.push(`Architecture changed from ${oldSystem.architecture} to ${currentSystem.architecture}`);
      }
      
      // Check CUDA availability changes
      if (oldSystem.cudaAvailable !== currentSystem.cudaAvailable) {
        const status = currentSystem.cudaAvailable ? 'available' : 'unavailable';
        const oldStatus = oldSystem.cudaAvailable ? 'available' : 'unavailable';
        changes.push(`CUDA changed from ${oldStatus} to ${status}`);
      }
      
      // Check CUDA device changes (if CUDA is available)
      if (currentSystem.cudaAvailable && oldSystem.cudaAvailable) {
        const oldDeviceCount = oldSystem.cudaDevices.length;
        const newDeviceCount = currentSystem.cudaDevices.length;
        
        if (oldDeviceCount !== newDeviceCount) {
          changes.push(`CUDA device count changed from ${oldDeviceCount} to ${newDeviceCount}`);
        }
        
        // Check total VRAM changes
        const oldVRAM = oldSystem.cudaDevices.reduce((sum, device) => sum + device.vram, 0);
        const newVRAM = currentSystem.cudaDevices.reduce((sum, device) => sum + device.vram, 0);
        const vramDiff = Math.abs(oldVRAM - newVRAM);
        
        if (vramDiff > 1) { // More than 1GB difference
          changes.push(`Total VRAM changed from ${oldVRAM.toFixed(1)}GB to ${newVRAM.toFixed(1)}GB`);
        }
      }
      
      // Check significant RAM changes (more than 25% difference)
      const ramDiff = Math.abs(oldSystem.totalRAM - currentSystem.totalRAM);
      if (ramDiff > oldSystem.totalRAM * 0.25) {
        changes.push(`RAM changed from ${oldSystem.totalRAM}GB to ${currentSystem.totalRAM}GB`);
      }
      
      // Determine if rebuild is recommended
      const shouldRebuild = changes.some(change => 
        change.includes('Platform changed') || 
        change.includes('Architecture changed') ||
        change.includes('CUDA changed') ||
        change.includes('device count changed')
      );
      
      return {
        hasChanged: changes.length > 0,
        changes,
        shouldRebuild
      };
      
    } catch (error) {
      log.error('[PythonEnvManager] Error checking system changes:', error);
      return { 
        hasChanged: true, 
        changes: ['Unable to detect system changes'],
        shouldRebuild: false 
      };
    }
  }

  /**
   * Force rebuild the Python environment
   * This will delete the existing environment and recreate it from scratch
   */
  public async rebuildEnvironment(): Promise<EnvironmentInfo> {
    try {
      log.info('[PythonEnvManager] Starting environment rebuild');
      
      // Cancel any ongoing setup
      if (this.setupProcess) {
        this.setupProcess.kill();
        this.setupProcess = null;
      }
      
      // Remove the existing environment
      await this.removeEnvironment();
      
      // Report that we're starting the rebuild
      await this.reportProgress('checking', 0, 'Starting environment rebuild...');
      
      // Set up the environment from scratch
      return await this.setupEnvironment();
    } catch (error) {
      log.error('[PythonEnvManager] Error during environment rebuild:', error);
      throw error;
    }
  }
}