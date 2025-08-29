/**
 * Python Server Manager - handles lifecycle of the Oumi backend server
 */

import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import * as net from 'net';
import log from 'electron-log';

export interface PythonServerConfig {
  port: number;
  host: string;
  timeout: number;
  maxRetries: number;
  conda_env?: string;
  python_path?: string;
  config_path?: string; // Path to Oumi inference config
}

export class PythonServerManager {
  private serverProcess: ChildProcess | null = null;
  private config: PythonServerConfig;
  private isStarting: boolean = false;
  private isRunning: boolean = false;

  constructor(port: number = 9000, config: Partial<PythonServerConfig> = {}) {
    this.config = {
      port,
      host: 'localhost',
      timeout: 30000, // 30 seconds
      maxRetries: 3,
      conda_env: 'oumi',
      ...config
    };
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
   * Start the Python server process
   */
  private async startServerProcess(): Promise<void> {
    const pythonCommand = await this.getPythonCommand();
    
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

    const serverArgs = [
      '-m', 'oumi.webchat.server',
      '--host', this.config.host,
      '--port', this.config.port.toString()
    ];

    // Add config path if specified
    if (this.config.config_path) {
      serverArgs.push('--config', this.config.config_path);
    }

    log.info(`Starting Python server on port ${this.config.port}: ${pythonCommand} ${serverArgs.join(' ')}`);

    return new Promise((resolve, reject) => {
      this.serverProcess = spawn(pythonCommand, serverArgs, {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: {
          ...process.env,
          PYTHONPATH: this.getPythonPath(),
          OUMI_LOG_LEVEL: 'INFO'
        }
      });

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
            
            // Check for startup success indicators
            if (!hasResolved && (
              output.includes('Uvicorn running on') || 
              output.includes('Server started') ||
              output.includes('Application startup complete')
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
          }
        });
      }

      // Timeout for server startup - try health check as fallback
      setTimeout(async () => {
        if (!hasResolved) {
          log.info('Server startup timeout reached, checking health endpoint...');
          
          // Try health check as fallback
          const isHealthy = await this.healthCheck();
          if (isHealthy) {
            log.info('Server is responding to health check, considering it started');
            hasResolved = true;
            resolve();
          } else {
            reject(new Error('Python server startup timeout - not responding to health checks'));
          }
        }
      }, this.config.timeout);
    });
  }

  /**
   * Wait for server to be ready by checking health endpoint
   */
  private async waitForServer(): Promise<string> {
    const maxAttempts = 60; // 30 seconds with 500ms intervals
    let attempts = 0;

    while (attempts < maxAttempts) {
      try {
        const isHealthy = await this.healthCheck();
        if (isHealthy) {
          return this.getServerUrl();
        }
      } catch (error) {
        // Continue trying
      }

      await this.sleep(500);
      attempts++;
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
   * Get Python path for PYTHONPATH environment variable
   */
  private getPythonPath(): string {
    // Path to the Oumi package (parent directory of frontend)
    const oumiPath = path.resolve(__dirname, '../../src');
    const existingPath = process.env.PYTHONPATH || '';
    
    return existingPath ? `${oumiPath}:${existingPath}` : oumiPath;
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
   * Get current configuration
   */
  public getConfig(): PythonServerConfig {
    return { ...this.config };
  }

  /**
   * Get server logs (if available)
   */
  public getLogs(): string[] {
    // This could be enhanced to store and return recent log entries
    return [];
  }
}