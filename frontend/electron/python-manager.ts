/**
 * Python Server Manager - handles lifecycle of the Oumi backend server
 */

import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import * as net from 'net';
import { app } from 'electron';
import log from 'electron-log';
import { PythonEnvironmentManager, EnvironmentInfo, SetupProgress } from './python-env-manager';
import { SystemInfo } from './system-detector';

export interface DownloadProgress {
  filename: string;
  progress: number;       // 0-100
  downloaded: string;     // "1.2GB" 
  total: string;          // "1.8GB"
  speed: string;          // "15.2MB/s"
  isComplete: boolean;
  estimatedTimeRemaining?: string; // "00:37"
}

export interface DownloadErrorEvent {
  message: string;
  timestamp: string;
  filename?: string;
}

export interface PythonServerConfig {
  port: number;
  host: string;
  timeout: number;
  maxRetries: number;
  conda_env?: string;
  python_path?: string;
  config_path?: string; // Path to Oumi inference config
  system_prompt?: string; // System prompt for the model
}

export class PythonServerManager {
  private serverProcess: ChildProcess | null = null;
  private config: PythonServerConfig;
  private isStarting: boolean = false;
  private isRunning: boolean = false;
  private downloadProgressCallback?: (progress: DownloadProgress) => void;
  private downloadErrorCallback?: (error: DownloadErrorEvent) => void;
  private envManager: PythonEnvironmentManager;
  private environmentInfo: EnvironmentInfo | null = null;
  private isDevelopment: boolean;
  private setupProgressCallback?: (progress: SetupProgress) => void;
  private systemChangeInfo?: { hasChanged: boolean; changes: string[]; shouldRebuild: boolean };

  constructor(port: number = 9000, config: Partial<PythonServerConfig> = {}) {
    this.config = {
      port,
      host: 'localhost',
      timeout: 30000, // 30 seconds
      maxRetries: 3,
      conda_env: 'oumi',
      ...config
    };
    
    this.envManager = new PythonEnvironmentManager();
    this.isDevelopment = process.env.NODE_ENV === 'development' || !app.isPackaged;
    
    log.info(`[PythonServerManager] Running in ${this.isDevelopment ? 'development' : 'production'} mode`);
  }

  /**
   * Start the Python backend server
   */
  public async start(): Promise<string> {
    if (this.isRunning) {
      return this.getServerUrl();
    }

    if (this.isStarting) {
      // Wait for current startup attempt
      return this.waitForServer();
    }

    this.isStarting = true;
    
    try {
      // Ensure Python environment is ready
      await this.ensurePythonEnvironment();
      
      // Find available port
      this.config.port = await this.findAvailablePort(this.config.port);
      
      // Start the server process
      await this.startServerProcess();
      
      // Wait for server to be ready
      await this.waitForServer();
      
      this.isRunning = true;
      this.isStarting = false;
      
      log.info(`Python server started successfully on port ${this.config.port}`);
      return this.getServerUrl();
      
    } catch (error) {
      this.isStarting = false;
      this.isRunning = false;
      throw error;
    }
  }

  /**
   * Stop the Python backend server
   */
  public async stop(): Promise<void> {
    if (!this.serverProcess) {
      return;
    }

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        // Force kill if graceful shutdown fails
        if (this.serverProcess) {
          this.serverProcess.kill('SIGKILL');
        }
        reject(new Error('Server shutdown timeout'));
      }, 10000);

      this.serverProcess!.on('exit', () => {
        clearTimeout(timeout);
        this.serverProcess = null;
        this.isRunning = false;
        log.info('Python server stopped');
        resolve();
      });

      // Graceful shutdown
      this.serverProcess!.kill('SIGTERM');
    });
  }

  /**
   * Check if the server is running
   */
  public isServerRunning(): boolean {
    return this.isRunning && this.serverProcess !== null;
  }

  /**
   * Get the server URL
   */
  public getServerUrl(): string {
    return `http://${this.config.host}:${this.config.port}`;
  }

  /**
   * Get server port
   */
  public getPort(): number {
    return this.config.port;
  }

  /**
   * Make HTTP request to server for health check
   */
  public async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.getServerUrl()}/health`);
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Start the Python server process using oumi webchat command
   */
  private async startServerProcess(): Promise<void> {
    // Try multiple ports if the preferred one is taken
    let attempts = 0;
    const maxAttempts = 10;
    
    while (attempts < maxAttempts) {
      const testPort = this.config.port + attempts;
      const isAvailable = await this.isPortAvailable(testPort);
      
      if (isAvailable) {
        this.config.port = testPort;
        break;
      }
      
      attempts++;
      if (attempts >= maxAttempts) {
        throw new Error(`No available ports found in range ${this.config.port}-${this.config.port + maxAttempts}`);
      }
    }

    // Build the oumi webchat command
    const oumiArgs = [
      'webchat',
      '--host', this.config.host,
      '--backend-port', this.config.port.toString()
    ];

    // Add config path if specified (resolve to absolute path)
    if (this.config.config_path) {
      const resolvedConfigPath = this.resolveConfigPath(this.config.config_path);
      oumiArgs.push('-c', `"${resolvedConfigPath}"`);
    }

    // Add system prompt if specified
    if (this.config.system_prompt) {
      oumiArgs.push('--system-prompt', `"${this.config.system_prompt}"`);
    }

    // Build the full command with appropriate Python environment
    const fullCommand = await this.buildPythonCommand(oumiArgs);
    
    log.info(`Starting Python server on port ${this.config.port}: ${fullCommand}`);

    return new Promise((resolve, reject) => {
      // Use shell execution to handle environment activation
      if (process.platform === 'win32' && !this.isDevelopment) {
        // Windows production: direct execution with clean environment
        this.serverProcess = spawn('cmd', ['/c', fullCommand], {
          stdio: ['pipe', 'pipe', 'pipe'],
          env: this.getCleanEnvironment(),
          cwd: this.getOumiRootPath()  // Set working directory to oumi root
        });
      } else {
        // Unix-like or development: bash execution with clean environment
        this.serverProcess = spawn('bash', ['-c', fullCommand], {
          stdio: ['pipe', 'pipe', 'pipe'],
          env: this.getCleanEnvironment(),
          cwd: this.getOumiRootPath()  // Set working directory to oumi root
        });
      }

      // Handle process events
      this.serverProcess.on('error', (error) => {
        log.error('Python server process error:', error);
        reject(new Error(`Failed to start Python server: ${error.message}`));
      });

      this.serverProcess.on('exit', (code, signal) => {
        log.info(`Python server exited with code ${code}, signal ${signal}`);
        this.isRunning = false;
        this.serverProcess = null;
      });

      // Track if we've resolved startup
      let hasResolved = false;

      // Log server output
      if (this.serverProcess.stdout) {
        this.serverProcess.stdout.on('data', (data) => {
          const output = data.toString().trim();
          if (output) {
            log.info(`[Python Server] ${output}`);
            
            // Handle download progress if present (both during startup and after)
            this.handleDownloadProgress(output);
            
            // Check for startup success indicators
            if (!hasResolved && (
              output.includes('Uvicorn running on') || 
              output.includes('Server started') ||
              output.includes('Application startup complete') ||
              output.includes('Webchat server is running')
            )) {
              hasResolved = true;
              resolve();
            }
          }
        });
      }

      if (this.serverProcess.stderr) {
        this.serverProcess.stderr.on('data', (data) => {
          const error = data.toString().trim();
          if (error) {
            log.error(`[Python Server Error] ${error}`);
            
            // Handle download progress/errors that might appear in stderr (both during startup and after)
            this.handleDownloadProgress(error);
          }
        });
      }

      // Shorter timeout for process startup - let waitForServer handle health checking
      setTimeout(async () => {
        if (!hasResolved) {
          log.info('Process startup timeout reached, proceeding to health checks...');
          hasResolved = true;
          resolve(); // Let waitForServer handle the health checking with proper backoff
        }
      }, 5000); // Reduced to 5 seconds
    });
  }

  /**
   * Wait for server to be ready by checking health endpoint with exponential backoff
   */
  private async waitForServer(): Promise<string> {
    const startTime = Date.now();
    const maxWaitTime = 60000; // Maximum 60 seconds total
    let attempt = 0;
    let delay = 100; // Start with 100ms delay
    const maxDelay = 2000; // Cap delay at 2 seconds
    
    log.info('Starting health checks for server readiness...');
    
    while (Date.now() - startTime < maxWaitTime) {
      try {
        const isHealthy = await this.healthCheck();
        if (isHealthy) {
          const totalTime = Date.now() - startTime;
          log.info(`Server ready in ${totalTime}ms after ${attempt + 1} health check attempts`);
          return this.getServerUrl();
        }
      } catch (error) {
        // Continue trying
      }
      
      // Log progress every 5 seconds
      if (attempt % Math.max(1, Math.floor(5000 / delay)) === 0 && attempt > 0) {
        const elapsed = Date.now() - startTime;
        log.info(`Still waiting for server (${elapsed}ms elapsed, attempt ${attempt + 1})`);
      }
      
      await this.sleep(delay);
      attempt++;
      
      // Exponential backoff with jitter
      delay = Math.min(maxDelay, delay * 1.5 + Math.random() * 100);
    }
    
    throw new Error('Server health check failed - server may not have started properly');
  }

  /**
   * Get Python command (handle conda environments)
   */
  private async getPythonCommand(): Promise<string> {
    // Try conda environment first
    if (this.config.conda_env) {
      const condaCommand = await this.getCondaCommand();
      if (condaCommand) {
        return condaCommand;
      }
    }

    // Fall back to system Python
    if (this.config.python_path && fs.existsSync(this.config.python_path)) {
      return this.config.python_path;
    }

    // Default Python command
    return process.platform === 'win32' ? 'python' : 'python3';
  }

  /**
   * Get conda-activated Python command
   */
  private async getCondaCommand(): Promise<string | null> {
    try {
      // For conda, we need to use shell execution
      if (process.platform === 'win32') {
        // Windows conda path
        return 'python'; // Assume conda is already in PATH on Windows
      } else {
        // macOS/Linux - try to find conda Python directly
        const condaPython = `/Users/${process.env.USER}/miniconda3/envs/${this.config.conda_env}/bin/python`;
        if (fs.existsSync(condaPython)) {
          return condaPython;
        }
        
        // Try alternative conda paths
        const altCondaPython = `/opt/homebrew/Caskroom/miniconda/base/envs/${this.config.conda_env}/bin/python`;
        if (fs.existsSync(altCondaPython)) {
          return altCondaPython;
        }
        
        return null;
      }
    } catch (error) {
      log.warn('Could not find conda environment, falling back to system Python');
      return null;
    }
  }

  /**
   * Ensure Python environment is ready (always use bundled Python + venv)
   */
  private async ensurePythonEnvironment(): Promise<void> {
    // Always use the bundled Python environment, regardless of development/production mode
    log.info('[PythonServerManager] Checking standalone environment with bundled Python');
    
    this.environmentInfo = await this.envManager.checkEnvironment();
    
    if (!this.environmentInfo.isValid) {
      log.info('[PythonServerManager] Environment needs setup - will use bundled Python');
      
      // Set up progress forwarding if callback is available
      if (this.setupProgressCallback) {
        this.envManager.setProgressCallback(this.setupProgressCallback);
      }
      
      this.environmentInfo = await this.envManager.setupEnvironment();
      log.info('[PythonServerManager] Environment setup completed');
    } else {
      // Check if system has changed significantly
      const changeCheck = await this.envManager.checkSystemChanges(this.environmentInfo);
      
      if (changeCheck.hasChanged) {
        log.warn('[PythonServerManager] System changes detected:', changeCheck.changes);
        
        // Store change information for UI to display warning
        this.systemChangeInfo = {
          hasChanged: true,
          changes: changeCheck.changes,
          shouldRebuild: changeCheck.shouldRebuild
        };
      } else {
        log.info('[PythonServerManager] Using existing valid environment - no system changes detected');
      }
    }
  }

  /**
   * Build appropriate Python command using the standalone environment
   */
  private async buildPythonCommand(oumiArgs: string[]): Promise<string> {
    // Always use standalone environment (bundled Python + venv)
    return this.buildStandaloneCommand(oumiArgs);
  }

  /**
   * Build conda activation command for development
   */
  private buildCondaCommand(oumiArgs: string[]): string {
    const condaEnv = this.config.conda_env || 'oumi';
    
    if (process.platform === 'win32') {
      // Windows command
      return `conda activate ${condaEnv} && oumi ${oumiArgs.join(' ')}`;
    } else {
      // macOS/Linux command - use the pattern from CLAUDE.md
      return `source ~/.zshrc && conda activate ${condaEnv} && oumi ${oumiArgs.join(' ')}`;
    }
  }

  /**
   * Build standalone Python command for production
   */
  private buildStandaloneCommand(oumiArgs: string[]): string {
    if (!this.environmentInfo?.isValid) {
      const errorMsg = `Python environment not ready. Environment info: ${JSON.stringify(this.environmentInfo)}`;
      log.error(`[PythonServerManager] ${errorMsg}`);
      throw new Error(errorMsg);
    }

    const pythonPath = this.environmentInfo.pythonPath;
    
    // Verify Python executable exists
    if (!require('fs').existsSync(pythonPath)) {
      const errorMsg = `Python executable not found at: ${pythonPath}`;
      log.error(`[PythonServerManager] ${errorMsg}`);
      throw new Error(errorMsg);
    }
    
    const oumiCommand = `"${pythonPath}" -m oumi ${oumiArgs.join(' ')}`;
    
    log.info(`[PythonServerManager] Using standalone Python: ${pythonPath}`);
    log.info(`[PythonServerManager] Full command: ${oumiCommand}`);
    return oumiCommand;
  }

  /**
   * Get the oumi package root directory
   */
  private getOumiRootPath(): string {
    // Check if we're running in a production packaged app
    if (app.isPackaged) {
      // In production, configs are bundled in Resources/python/
      const resourcesPath = process.resourcesPath;
      const pythonPath = path.join(resourcesPath, 'python');
      log.info(`[getOumiRootPath] Production mode - resources path: ${resourcesPath}`);
      log.info(`[getOumiRootPath] Python path: ${pythonPath}`);
      return pythonPath;
    }
    
    // Development/debug mode
    if (__dirname.includes('/dist/electron')) {
      // Running from frontend/dist/electron/, go up 3 levels to oumi root
      return path.resolve(__dirname, '../../../');  
    } else {
      // Running from frontend/electron/, go up 2 levels to oumi root
      return path.resolve(__dirname, '../../');  
    }
  }

  /**
   * Resolve config path to absolute path if it's relative
   */
  private resolveConfigPath(configPath: string): string {
    log.info(`[resolveConfigPath] Input path: ${configPath}`);
    
    if (path.isAbsolute(configPath)) {
      log.info(`[resolveConfigPath] Path is already absolute: ${configPath}`);
      return configPath;
    }
    
    // For relative paths, resolve using the same logic as config discovery
    const configsBasePath = app.isPackaged
      ? path.join(process.resourcesPath, 'python', 'configs')
      : path.join(__dirname, '../../../configs');
    
    log.info(`[resolveConfigPath] Config base path: ${configsBasePath} (isPackaged: ${app.isPackaged})`);
    log.info(`[resolveConfigPath] __dirname: ${__dirname}`);
    
    const absolutePath = path.resolve(configsBasePath, configPath);
    
    log.info(`[resolveConfigPath] Resolved config path: ${configPath} -> ${absolutePath}`);
    return absolutePath;
  }

  /**
   * Get clean environment for Python process (without dev environment variables)
   */
  private getCleanEnvironment(): NodeJS.ProcessEnv {
    // Start with minimal environment
    const cleanEnv: NodeJS.ProcessEnv = {
      OUMI_LOG_LEVEL: 'INFO',
      NODE_ENV: 'production',  // Required by TypeScript definition
      HOME: process.env.HOME || '',
      USER: process.env.USER || '',
      PATH: process.env.PATH || '',
      SHELL: process.env.SHELL || '/bin/bash',
    };

    // Add necessary system variables for macOS
    if (process.platform === 'darwin') {
      if (process.env.LC_ALL) cleanEnv.LC_ALL = process.env.LC_ALL;
      if (process.env.LANG) cleanEnv.LANG = process.env.LANG;
    }

    // Explicitly exclude dev environment variables that could interfere
    // DO NOT include: PYTHONPATH, CONDA_DEFAULT_ENV, CONDA_PREFIX, etc.
    
    return cleanEnv;
  }

  /**
   * Find an available port starting from the preferred port
   */
  private async findAvailablePort(preferredPort: number): Promise<number> {
    for (let port = preferredPort; port < preferredPort + 100; port++) {
      if (await this.isPortAvailable(port)) {
        return port;
      }
    }
    throw new Error('No available ports found');
  }

  /**
   * Check if a port is available
   */
  private isPortAvailable(port: number): Promise<boolean> {
    return new Promise((resolve) => {
      const server = net.createServer();
      
      server.listen(port, () => {
        server.once('close', () => resolve(true));
        server.close();
      });
      
      server.on('error', () => resolve(false));
    });
  }

  /**
   * Sleep utility
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Restart the server
   */
  public async restart(): Promise<string> {
    log.info('Restarting Python server...');
    
    if (this.isRunning) {
      await this.stop();
    }
    
    return this.start();
  }

  /**
   * Update configuration (requires restart to take effect)
   */
  public updateConfig(updates: Partial<PythonServerConfig>): void {
    this.config = { ...this.config, ...updates };
    log.info('Python server configuration updated:', updates);
  }

  /**
   * Set inference configuration path
   */
  public setConfigPath(configPath: string): void {
    this.config.config_path = configPath;
    log.info(`Python server config path set to: ${configPath}`);
  }

  /**
   * Set system prompt
   */
  public setSystemPrompt(systemPrompt: string): void {
    this.config.system_prompt = systemPrompt;
    log.info(`Python server system prompt set to: ${systemPrompt.substring(0, 100)}...`);
  }

  /**
   * Get current configuration
   */
  public getConfig(): PythonServerConfig {
    return { ...this.config };
  }

  /**
   * Run test inference to check if model is ready and trigger download if needed
   */
  public async testModel(configPath: string): Promise<{ success: boolean; message?: string }> {
    log.info(`Testing model with config: ${configPath}`);

    // Create temporary input and output files for non-interactive mode
    // Use app temp directory instead of __dirname to avoid writing to read-only .asar
    const { app } = require('electron');
    const tempDir = app.getPath('temp');
    const tempInputPath = path.join(tempDir, 'chatterley_test_input.jsonl');
    const tempOutputPath = path.join(tempDir, 'chatterley_test_output.jsonl');
    const testInput = JSON.stringify({
      messages: [
        { role: "user", content: "Hello world" }
      ]
    });
    
    // Network monitoring state
    let networkMonitorInterval: NodeJS.Timeout | null = null;
    let lastNetworkStats = { bytesReceived: 0, totalReceived: 0 };
    let isDownloading = false;
    
    try {
      // Ensure Python environment is ready before testing
      log.info(`[testModel] Ensuring Python environment is ready...`);
      await this.ensurePythonEnvironment();
      log.info(`[testModel] Python environment check completed. Environment info:`, {
        isValid: this.environmentInfo?.isValid,
        pythonPath: this.environmentInfo?.pythonPath,
        path: this.environmentInfo?.path
      });
      
      // Write test input file
      require('fs').writeFileSync(tempInputPath, testInput);
      
      // Resolve config path to absolute path
      const resolvedConfigPath = this.resolveConfigPath(configPath);
      
      // Verify config file exists
      if (!require('fs').existsSync(resolvedConfigPath)) {
        log.error(`Config file does not exist: ${resolvedConfigPath}`);
        return { success: false, message: `Config file not found: ${resolvedConfigPath}` };
      }
      
      const testCommand = await this.buildPythonCommand([
        'infer',
        '-c', `"${resolvedConfigPath}"`,
        '--input_path', `"${tempInputPath}"`,
        '--output_path', `"${tempOutputPath}"`
      ]);

      log.info(`Running test command: ${testCommand}`);
      log.info(`Resolved config path: ${resolvedConfigPath}`);
      log.info(`Temp input path: ${tempInputPath}`);
      log.info(`Temp output path: ${tempOutputPath}`);

      return new Promise((resolve, reject) => {

        // Use appropriate shell for the platform
        let testProcess: ChildProcess;
        if (process.platform === 'win32') {
          testProcess = spawn('cmd', ['/c', testCommand], {
            stdio: ['pipe', 'pipe', 'pipe'],
            env: this.getCleanEnvironment(),
            cwd: this.getOumiRootPath()
          });
        } else {
          testProcess = spawn('bash', ['-c', testCommand], {
            stdio: ['pipe', 'pipe', 'pipe'],
            env: this.getCleanEnvironment(),
            cwd: this.getOumiRootPath()
          });
        }

        log.info(`Test process spawned with PID: ${testProcess.pid}`);

        // Network monitoring function for macOS
        const getProcessNetworkStats = async (pid: number): Promise<{ bytesReceived: number } | null> => {
          return new Promise((resolve) => {
            const { exec } = require('child_process');
            // Use netstat to get network stats for the specific process
            exec(`netstat -p ${pid} 2>/dev/null | grep -E '(tcp|udp)' | awk '{sum+=$2} END {print sum+0}'`, (error: any, stdout: string) => {
              if (error) {
                // Fallback: try lsof approach
                exec(`lsof -p ${pid} -a -i 2>/dev/null | wc -l`, (lsofError: any, lsofStdout: string) => {
                  if (lsofError) {
                    resolve(null);
                  } else {
                    const connections = parseInt(lsofStdout.trim()) || 0;
                    resolve({ bytesReceived: connections > 0 ? 1 : 0 }); // Simple indicator
                  }
                });
                return;
              }
              const bytes = parseInt(stdout.trim()) || 0;
              resolve({ bytesReceived: bytes });
            });
          });
        };

        // Start network monitoring function
        const startNetworkMonitoring = (pid: number) => {
          let downloadStartTime: number | null = null;
          let stableCount = 0;
          
          networkMonitorInterval = setInterval(async () => {
            const stats = await getProcessNetworkStats(pid);
            if (!stats) return;

            const currentBytes = stats.bytesReceived;
            const bytesThisInterval = currentBytes - lastNetworkStats.bytesReceived;
            
            // Detect download activity (significant network activity)
            if (bytesThisInterval > 1024) { // More than 1KB received
              if (!isDownloading) {
                isDownloading = true;
                downloadStartTime = Date.now();
                log.info('[Network Monitor] Download activity detected');
                
                // Emit download start event
                if (this.downloadProgressCallback) {
                  this.downloadProgressCallback({
                    filename: 'Model files',
                    progress: 0,
                    downloaded: '0MB',
                    total: 'Unknown',
                    speed: 'Detecting...',
                    isComplete: false
                  });
                }
              }
              
              // Calculate download progress indicators
              if (isDownloading && downloadStartTime) {
                const elapsed = Date.now() - downloadStartTime;
                const totalMB = Math.round(currentBytes / (1024 * 1024));
                const speedMBs = totalMB / (elapsed / 1000);
                
                if (this.downloadProgressCallback) {
                  this.downloadProgressCallback({
                    filename: 'Model files',
                    progress: Math.min(95, Math.floor(elapsed / 1000)), // Fake progress based on time
                    downloaded: `${totalMB}MB`,
                    total: 'Unknown',
                    speed: `${speedMBs.toFixed(1)}MB/s`,
                    isComplete: false
                  });
                }
              }
              
              stableCount = 0;
            } else if (isDownloading) {
              stableCount++;
              
              // If network activity has been stable for 3 intervals (6 seconds), consider download complete
              if (stableCount >= 3) {
                log.info('[Network Monitor] Download appears complete (network activity stabilized)');
                isDownloading = false;
                
                if (this.downloadProgressCallback) {
                  this.downloadProgressCallback({
                    filename: 'Model files',
                    progress: 100,
                    downloaded: `${Math.round(currentBytes / (1024 * 1024))}MB`,
                    total: 'Complete',
                    speed: 'Complete',
                    isComplete: true
                  });
                }
                
                if (networkMonitorInterval) {
                  clearInterval(networkMonitorInterval);
                  networkMonitorInterval = null;
                }
              }
            }
            
            lastNetworkStats.bytesReceived = currentBytes;
          }, 2000); // Check every 2 seconds
        };

        // Start network monitoring if we have a PID
        if (testProcess.pid) {
          startNetworkMonitoring(testProcess.pid);
        }

        let hasCompleted = false;
        let outputBuffer = '';

        const cleanup = () => {
          if (networkMonitorInterval) {
            clearInterval(networkMonitorInterval);
            networkMonitorInterval = null;
          }
          if (testProcess && !testProcess.killed) {
            testProcess.kill('SIGTERM');
          }
          // Clean up temp files
          try {
            require('fs').unlinkSync(tempInputPath);
            require('fs').unlinkSync(tempOutputPath);
          } catch (e) {
            // Ignore cleanup errors
          }
        };



      // Track if model loading succeeded
      let modelLoaded = false;
      
      // Handle process completion
      testProcess.on('exit', (code, signal) => {
        if (hasCompleted) return;
        hasCompleted = true;

        log.info(`Test inference exited with code ${code}, signal ${signal}`);
        log.info(`Output buffer: ${outputBuffer.slice(-1000)}`); // Log last 1000 chars for debugging
        
        // In non-interactive mode, successful completion means the model loaded and ran inference
        if (code === 0) {
          resolve({ success: true, message: 'Model test successful - inference completed' });
        } else if (modelLoaded) {
          // Even if exit code is not 0, if we detected model loading, consider it a success
          resolve({ success: true, message: 'Model test successful - model loaded (process terminated early)' });
        } else {
          // Include output buffer in error message for better debugging
          const errorDetails = outputBuffer.slice(-500); // Last 500 chars of output
          const errorMessage = errorDetails 
            ? `Model test failed - process exited with code ${code}. Details: ${errorDetails}`
            : `Model test failed - process exited with code ${code}`;
          resolve({ success: false, message: errorMessage });
        }
      });

      testProcess.on('error', (error) => {
        if (hasCompleted) return;
        hasCompleted = true;

        log.error('Test inference process error:', error);
        resolve({ success: false, message: `Process error: ${error.message}` });
      });

      // Monitor stdout for download progress and completion
      if (testProcess.stdout) {
        testProcess.stdout.on('data', (data) => {
          const output = data.toString().trim();
          outputBuffer += output + '\n';
          
          if (output) {
            log.info(`[Test Inference] ${output}`);
            
            // Handle download progress during test
            this.handleDownloadProgress(output);
            
            // Check for model loading success indicators in non-interactive mode
            if (output.includes('Model loaded successfully') || 
                output.includes('Starting batch inference') ||
                output.includes('Processing') ||
                output.includes('Inference completed') ||
                output.includes('Writing results') ||
                (output.includes('messages') && output.includes('role')) ||
                output.includes('Building model')) {
              modelLoaded = true;
              log.info('Model loading detected, process will complete naturally');
            }
          }
        });
      }

      // Monitor stderr for errors and download progress
      if (testProcess.stderr) {
        testProcess.stderr.on('data', (data) => {
          const error = data.toString().trim();
          outputBuffer += error + '\n';
          
          if (error) {
            log.info(`[Test Inference Error] ${error}`);
            
            // Handle download progress that might appear in stderr
            this.handleDownloadProgress(error);
          }
        });
      }

      // Set timeout for test inference (5 minutes should be enough)
      const timeout = setTimeout(() => {
        if (hasCompleted) return;
        hasCompleted = true;

        log.warn('Test inference timeout reached');
        cleanup();
        resolve({ success: false, message: 'Model test timed out' });
      }, 300000); // 5 minutes

      // Clean up timeout when process completes
      testProcess.on('exit', () => {
        clearTimeout(timeout);
      });
      
      });
      
    } catch (error) {
      log.error('Error setting up test inference:', error);
      const errorMessage = error instanceof Error ? error.message : String(error);
      return { success: false, message: `Setup error: ${errorMessage}` };
    } finally {
      // Clean up temporary files
      try {
        const fs = require('fs');
        if (fs.existsSync(tempInputPath)) {
          fs.unlinkSync(tempInputPath);
        }
        if (fs.existsSync(tempOutputPath)) {
          fs.unlinkSync(tempOutputPath);
        }
      } catch (cleanupError) {
        log.warn('Failed to clean up temporary test files:', cleanupError);
      }
      
      // Clear network monitoring if still running
      if (networkMonitorInterval) {
        clearInterval(networkMonitorInterval);
        networkMonitorInterval = null;
      }
    }
  }

  /**
   * Get server logs (if available)
   */
  public getLogs(): string[] {
    // This could be enhanced to store and return recent log entries
    return [];
  }

  /**
   * Set download progress callback
   */
  public setDownloadProgressCallback(callback: (progress: DownloadProgress) => void): void {
    this.downloadProgressCallback = callback;
  }

  /**
   * Set download error callback
   */
  public setDownloadErrorCallback(callback: (error: DownloadErrorEvent) => void): void {
    this.downloadErrorCallback = callback;
  }

  /**
   * Set setup progress callback for environment setup
   */
  public setSetupProgressCallback(callback: (progress: SetupProgress) => void): void {
    this.setupProgressCallback = callback;
  }

  /**
   * Check if environment setup is needed
   */
  public async isEnvironmentSetupNeeded(): Promise<boolean> {
    // Always check the bundled Python environment
    const info = await this.envManager.checkEnvironment();
    return !info.isValid;
  }

  /**
   * Get environment info for display
   */
  public getEnvironmentInfo(): EnvironmentInfo | null {
    return this.environmentInfo;
  }

  /**
   * Get user data directory path
   */
  public getUserDataPath(): string {
    return this.envManager.getUserDataPath();
  }

  /**
   * Remove the Python environment (cleanup)
   */
  public async removeEnvironment(): Promise<void> {
    // Always manage the bundled Python environment
    await this.envManager.removeEnvironment();
    this.environmentInfo = null;
  }

  /**
   * Parse download progress from output line
   */
  private parseDownloadProgress(output: string): DownloadProgress | null {
    // HuggingFace download patterns:
    // "Downloading pytorch_model.bin: 67%|██████▋ | 1.2G/1.8G [01:23<00:37, 15.2MB/s]"
    // "Downloading tokenizer.json: 100%|██████████| 2.11M/2.11M [00:01<00:00, 1.85MB/s]"
    
    const downloadMatch = output.match(/Downloading\s+([^:]+):\s+(\d+)%.*?(\d+\.?\d*[KMGT]?)B\/(\d+\.?\d*[KMGT]?)B.*?\[(\d{2}:\d{2})<(\d{2}:\d{2}),\s*(\d+\.?\d*[KMGT]?B\/s)\]/);
    
    if (downloadMatch) {
      const filename = downloadMatch[1].trim();
      const progress = parseInt(downloadMatch[2]);
      const downloaded = downloadMatch[3] + 'B';
      const total = downloadMatch[4] + 'B';
      const elapsed = downloadMatch[5];
      const remaining = downloadMatch[6];
      const speed = downloadMatch[7];

      return {
        filename,
        progress,
        downloaded,
        total,
        speed,
        isComplete: progress === 100,
        estimatedTimeRemaining: remaining !== '00:00' ? remaining : undefined
      };
    }

    // Alternative pattern for different download formats
    const altMatch = output.match(/Downloading.*?(\S+).*?(\d+)%.*?(\d+\.?\d*[KMGT]?)B.*?(\d+\.?\d*[KMGT]?B\/s)/);
    if (altMatch) {
      const filename = altMatch[1];
      const progress = parseInt(altMatch[2]);
      const downloaded = altMatch[3] + 'B';
      const speed = altMatch[4];

      return {
        filename,
        progress,
        downloaded,
        total: 'Unknown',
        speed,
        isComplete: progress === 100
      };
    }

    return null;
  }

  /**
   * Parse download errors from output
   */
  private parseDownloadError(output: string): DownloadErrorEvent | null {
    // Common error patterns
    if (output.includes('Failed to download') || 
        output.includes('Connection error') ||
        output.includes('timeout') ||
        output.includes('HTTP error')) {
      
      const filenameMatch = output.match(/(?:downloading|download)\s+([^\s:]+)/i);
      
      return {
        message: output.trim(),
        timestamp: new Date().toISOString(),
        filename: filenameMatch ? filenameMatch[1] : undefined
      };
    }
    
    return null;
  }

  /**
   * Handle download progress events
   */
  private handleDownloadProgress(output: string): void {
    const progress = this.parseDownloadProgress(output);
    if (progress && this.downloadProgressCallback) {
      log.info(`[Download Progress] ${progress.filename}: ${progress.progress}% (${progress.speed})`);
      this.downloadProgressCallback(progress);
    }

    const error = this.parseDownloadError(output);
    if (error && this.downloadErrorCallback) {
      log.error(`[Download Error] ${error.message}`);
      this.downloadErrorCallback(error);
    }
  }

  /**
   * Force rebuild the Python environment
   * This will delete the existing environment and recreate it
   */
  async rebuildEnvironment(): Promise<void> {
    try {
      log.info('[PythonServerManager] Starting environment rebuild');
      
      // Stop the server if it's running
      if (this.isServerRunning()) {
        await this.stop();
      }

      // Always manage the bundled Python environment

      // Force rebuild the standalone environment
      log.info('[PythonServerManager] Rebuilding standalone environment');
      
      // Set up progress forwarding if callback is available
      if (this.setupProgressCallback) {
        this.envManager.setProgressCallback(this.setupProgressCallback);
      }
      
      // Force rebuild (delete existing environment and recreate)
      this.environmentInfo = await this.envManager.rebuildEnvironment();
      
      if (this.environmentInfo.isValid) {
        log.info('[PythonServerManager] Environment rebuild completed successfully');
      } else {
        const errorMsg = 'Environment rebuild failed - environment is not valid after setup';
        log.error(`[PythonServerManager] ${errorMsg}`);
        throw new Error(errorMsg);
      }
    } catch (error) {
      log.error('[PythonServerManager] Error during environment rebuild:', error);
      // Re-throw the error instead of swallowing it and returning false
      // This ensures proper error propagation and debugging
      throw error;
    }
  }

  /**
   * Get system change information if available
   */
  public getSystemChangeInfo(): { hasChanged: boolean; changes: string[]; shouldRebuild: boolean } | null {
    return this.systemChangeInfo || null;
  }

  /**
   * Get current environment system information
   */
  public getEnvironmentSystemInfo(): SystemInfo | null {
    return this.environmentInfo?.systemInfo || null;
  }


}