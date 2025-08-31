/**
 * System Detection Utility - Detects hardware and software capabilities
 */

import { spawn } from 'child_process';
import * as os from 'os';
import * as fs from 'fs';
import * as path from 'path';
import log from 'electron-log';

export interface SystemInfo {
  architecture: string;      // x64, arm64, etc.
  platform: string;         // darwin, win32, linux
  platformVersion: string;  // OS version
  cpuModel: string;         // CPU model name
  totalRAM: number;         // Total RAM in GB
  availableRAM: number;     // Available RAM in GB
  cudaAvailable: boolean;   // CUDA availability
  cudaVersion?: string;     // CUDA version if available
  cudaDevices: CudaDevice[]; // CUDA devices info
  detectedAt: string;       // ISO timestamp when detected
  fingerprint: string;     // Unique fingerprint for change detection
}

export interface CudaDevice {
  id: number;
  name: string;
  vram: number;            // VRAM in GB
  driverVersion?: string;  // Driver version
}

export class SystemDetector {
  /**
   * Detect complete system information
   */
  public static async detectSystem(): Promise<SystemInfo> {
    try {
      log.info('[SystemDetector] Starting system detection');
      
      const basicInfo = this.getBasicSystemInfo();
      const cudaInfo = await this.detectCUDA();
      const detectedAt = new Date().toISOString();
      
      const systemInfo: SystemInfo = {
        ...basicInfo,
        ...cudaInfo,
        detectedAt,
        fingerprint: this.generateFingerprint({ ...basicInfo, ...cudaInfo })
      };
      
      log.info('[SystemDetector] System detection completed:', {
        platform: systemInfo.platform,
        architecture: systemInfo.architecture,
        totalRAM: `${systemInfo.totalRAM}GB`,
        cudaAvailable: systemInfo.cudaAvailable,
        cudaDevices: systemInfo.cudaDevices.length
      });
      
      return systemInfo;
    } catch (error) {
      log.error('[SystemDetector] Error during system detection:', error);
      throw error;
    }
  }

  /**
   * Get basic system information using Node.js os module
   */
  private static getBasicSystemInfo() {
    const platform = os.platform();
    const architecture = process.arch; // Use process.arch for consistency with env manager
    const cpuModel = os.cpus()[0]?.model || 'Unknown CPU';
    const totalRAM = Math.round(os.totalmem() / (1024 * 1024 * 1024) * 100) / 100; // GB
    const availableRAM = Math.round(os.freemem() / (1024 * 1024 * 1024) * 100) / 100; // GB
    
    let platformVersion = os.release();
    
    // Get more detailed OS version information
    if (platform === 'darwin') {
      // macOS version mapping
      const release = os.release();
      const majorVersion = parseInt(release.split('.')[0]);
      if (majorVersion >= 23) platformVersion = 'macOS 14+ (Sonoma or later)';
      else if (majorVersion >= 22) platformVersion = 'macOS 13 (Ventura)';
      else if (majorVersion >= 21) platformVersion = 'macOS 12 (Monterey)';
      else if (majorVersion >= 20) platformVersion = 'macOS 11 (Big Sur)';
      else platformVersion = `macOS ${majorVersion - 4}`;
    } else if (platform === 'win32') {
      // Windows version detection
      try {
        const version = os.version();
        if (version.includes('Windows 11')) platformVersion = 'Windows 11';
        else if (version.includes('Windows 10')) platformVersion = 'Windows 10';
        else platformVersion = version;
      } catch {
        platformVersion = 'Windows (Unknown Version)';
      }
    }
    
    return {
      architecture,
      platform,
      platformVersion,
      cpuModel,
      totalRAM,
      availableRAM
    };
  }

  /**
   * Detect CUDA availability and devices
   */
  private static async detectCUDA(): Promise<{ cudaAvailable: boolean; cudaVersion?: string; cudaDevices: CudaDevice[] }> {
    try {
      log.info('[SystemDetector] Detecting CUDA...');
      
      // Try to run nvidia-smi to detect CUDA
      const nvidiaSmiResult = await this.runCommand('nvidia-smi', [
        '--query-gpu=index,name,memory.total,driver_version', 
        '--format=csv,noheader,nounits'
      ], { timeout: 5000 });
      
      if (nvidiaSmiResult.success && nvidiaSmiResult.output) {
        const lines = nvidiaSmiResult.output.trim().split('\n');
        const cudaDevices: CudaDevice[] = lines.map(line => {
          const [id, name, vramMB, driverVersion] = line.split(', ').map(s => s.trim());
          return {
            id: parseInt(id),
            name: name,
            vram: Math.round(parseInt(vramMB) / 1024 * 100) / 100, // Convert MB to GB
            driverVersion: driverVersion
          };
        });
        
        // Try to get CUDA version
        let cudaVersion: string | undefined;
        const nvccResult = await this.runCommand('nvcc', ['--version'], { timeout: 3000 });
        if (nvccResult.success && nvccResult.output) {
          const versionMatch = nvccResult.output.match(/release (\d+\.\d+)/);
          if (versionMatch) {
            cudaVersion = versionMatch[1];
          }
        }
        
        log.info(`[SystemDetector] CUDA detected: ${cudaDevices.length} device(s), version: ${cudaVersion || 'unknown'}`);
        
        return {
          cudaAvailable: true,
          cudaVersion,
          cudaDevices
        };
      }
      
      // CUDA not available
      log.info('[SystemDetector] CUDA not available');
      return {
        cudaAvailable: false,
        cudaDevices: []
      };
      
    } catch (error) {
      log.warn('[SystemDetector] CUDA detection failed:', error);
      return {
        cudaAvailable: false,
        cudaDevices: []
      };
    }
  }

  /**
   * Run a command and return result
   */
  private static async runCommand(
    command: string, 
    args: string[] = [], 
    options: { timeout?: number } = {}
  ): Promise<{ success: boolean; output?: string; error?: string }> {
    return new Promise((resolve) => {
      const process = spawn(command, args, { 
        stdio: 'pipe',
        shell: true 
      });
      
      let stdout = '';
      let stderr = '';
      
      // Set timeout
      const timeoutMs = options.timeout || 10000;
      const timeout = setTimeout(() => {
        process.kill();
        resolve({ success: false, error: 'Command timeout' });
      }, timeoutMs);
      
      process.stdout?.on('data', (data) => {
        stdout += data.toString();
      });
      
      process.stderr?.on('data', (data) => {
        stderr += data.toString();
      });
      
      process.on('close', (code) => {
        clearTimeout(timeout);
        if (code === 0) {
          resolve({ success: true, output: stdout });
        } else {
          resolve({ success: false, error: stderr || `Command exited with code ${code}` });
        }
      });
      
      process.on('error', (error) => {
        clearTimeout(timeout);
        resolve({ success: false, error: error.message });
      });
    });
  }

  /**
   * Generate a system fingerprint for change detection
   */
  private static generateFingerprint(info: Partial<SystemInfo>): string {
    const key = `${info.platform}-${info.architecture}-${info.totalRAM}-${info.cudaAvailable}-${info.cudaDevices?.length || 0}`;
    return Buffer.from(key).toString('base64');
  }

  /**
   * Compare two system fingerprints
   */
  public static hasSystemChanged(oldFingerprint: string, newFingerprint: string): boolean {
    return oldFingerprint !== newFingerprint;
  }

  /**
   * Get system capabilities summary for display
   */
  public static getSystemCapabilitiesSummary(systemInfo: SystemInfo): string {
    const parts = [];
    
    // Platform and architecture
    parts.push(`${systemInfo.platformVersion} (${systemInfo.architecture})`);
    
    // RAM
    parts.push(`${systemInfo.totalRAM}GB RAM`);
    
    // CUDA info
    if (systemInfo.cudaAvailable && systemInfo.cudaDevices.length > 0) {
      const totalVRAM = systemInfo.cudaDevices.reduce((sum, device) => sum + device.vram, 0);
      parts.push(`CUDA (${systemInfo.cudaDevices.length} GPU${systemInfo.cudaDevices.length > 1 ? 's' : ''}, ${totalVRAM}GB VRAM)`);
    } else {
      parts.push('No CUDA');
    }
    
    return parts.join(' • ');
  }

  /**
   * Determine the best engine for the current system
   */
  public static getBestEngineForSystem(systemInfo: SystemInfo): string[] {
    const engines = [];
    
    if (systemInfo.platform === 'darwin') {
      // macOS: Prefer LLAMACPP for efficiency, then NATIVE
      engines.push('LLAMACPP', 'NATIVE');
    } else if (systemInfo.cudaAvailable && systemInfo.cudaDevices.length > 0) {
      // CUDA available: Prefer VLLM for performance, then NATIVE
      engines.push('VLLM', 'NATIVE');
    } else {
      // CPU-only: Prefer NATIVE, then LLAMACPP
      engines.push('NATIVE', 'LLAMACPP');
    }
    
    return engines;
  }

  /**
   * Get recommended model sizes based on available VRAM/RAM
   */
  public static getRecommendedModelSizes(systemInfo: SystemInfo): string[] {
    const totalVRAM = systemInfo.cudaDevices.reduce((sum, device) => sum + device.vram, 0);
    const effectiveMemory = Math.max(totalVRAM, systemInfo.totalRAM * 0.5); // Use 50% of RAM if no GPU
    
    const sizes = [];
    
    if (effectiveMemory >= 24) {
      sizes.push('large', 'medium', 'small'); // 70B, 30B, 8B models
    } else if (effectiveMemory >= 16) {
      sizes.push('medium', 'small'); // 30B, 8B models
    } else if (effectiveMemory >= 8) {
      sizes.push('small'); // 8B models
    } else {
      sizes.push('small'); // Small models only, maybe with quantization
    }
    
    return sizes;
  }
}