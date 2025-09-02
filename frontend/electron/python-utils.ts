/**
 * Python Execution Utilities - Universal functions for running Python commands in Electron
 * This ensures consistent Python path resolution and error handling across the application
 */

import { spawn, SpawnOptions, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import { app } from 'electron';
import log from 'electron-log';

export interface PythonExecutionOptions {
  env?: NodeJS.ProcessEnv;
  cwd?: string;
  timeout?: number;
  stdio?: 'pipe' | 'inherit' | 'ignore';
}

export interface PythonExecutionResult {
  success: boolean;
  stdout: string;
  stderr: string;
  exitCode: number | null;
  error?: string;
}

/**
 * Get potential Python executable paths in order of preference
 */
export function getPythonCommands(): string[] {
  const commands: string[] = [];
  
  // 1. Try bundled Python first (production builds)
  try {
    const bundledPythonPath = getBundledPythonPath();
    if (bundledPythonPath && fs.existsSync(bundledPythonPath)) {
      commands.push(bundledPythonPath);
      log.info('[PythonUtils] Found bundled Python:', bundledPythonPath);
    }
  } catch (error) {
    log.debug('[PythonUtils] No bundled Python found:', error);
  }

  // 2. Try conda environment (if available)
  try {
    const condaPython = getCondaPythonPath();
    if (condaPython && fs.existsSync(condaPython)) {
      commands.push(condaPython);
      log.info('[PythonUtils] Found conda Python:', condaPython);
    }
  } catch (error) {
    log.debug('[PythonUtils] No conda Python found:', error);
  }

  // 3. Try system Python installations
  const systemCommands = process.platform === 'win32' 
    ? ['python.exe', 'python3.exe', 'py.exe']
    : ['python3', 'python'];
  
  commands.push(...systemCommands);
  
  log.info('[PythonUtils] Python command priority order:', commands);
  return commands;
}

/**
 * Get bundled Python path for production builds
 */
function getBundledPythonPath(): string | null {
  if (!app.isPackaged) {
    return null; // No bundled Python in development
  }

  const resourcesPath = process.resourcesPath;
  const platform = process.platform;

  if (platform === 'win32') {
    // Windows: flattened structure at root level
    const winPath = path.join(resourcesPath, 'python-dist', 'python.exe');
    if (fs.existsSync(winPath)) {
      return winPath;
    }
    
    // Alternative Windows path structure  
    const altWinPath = path.join(resourcesPath, 'python', 'python.exe');
    if (fs.existsSync(altWinPath)) {
      return altWinPath;
    }
  } else {
    // Unix-like: flattened structure in bin/ subdirectory
    const unixPath = path.join(resourcesPath, 'python-dist', 'bin', 'python3');
    if (fs.existsSync(unixPath)) {
      return unixPath;
    }
    
    // Fallback to 'python' instead of 'python3'
    const unixPath2 = path.join(resourcesPath, 'python-dist', 'bin', 'python');
    if (fs.existsSync(unixPath2)) {
      return unixPath2;
    }
    
    // Alternative Unix path structure
    const altUnixPath = path.join(resourcesPath, 'python', 'bin', 'python3');
    if (fs.existsSync(altUnixPath)) {
      return altUnixPath;
    }
    
    const altUnixPath2 = path.join(resourcesPath, 'python', 'bin', 'python');
    if (fs.existsSync(altUnixPath2)) {
      return altUnixPath2;
    }
  }

  return null;
}

/**
 * Get conda Python path if available
 */
function getCondaPythonPath(): string | null {
  const condaEnv = process.env.CONDA_DEFAULT_ENV;
  const condaPrefix = process.env.CONDA_PREFIX;
  
  if (condaPrefix) {
    const pythonExe = process.platform === 'win32' ? 'python.exe' : 'python';
    const condaPythonPath = path.join(condaPrefix, 'bin', pythonExe);
    
    if (fs.existsSync(condaPythonPath)) {
      return condaPythonPath;
    }
    
    // Windows conda structure
    const winCondaPythonPath = path.join(condaPrefix, pythonExe);
    if (fs.existsSync(winCondaPythonPath)) {
      return winCondaPythonPath;
    }
  }
  
  return null;
}

/**
 * Execute Python code directly with automatic Python path resolution
 */
export async function executePythonCode(
  code: string, 
  options: PythonExecutionOptions = {}
): Promise<PythonExecutionResult> {
  const commands = getPythonCommands();
  let lastError = '';

  for (const pythonCmd of commands) {
    try {
      log.info(`[PythonUtils] Trying Python command: ${pythonCmd}`);
      const result = await runPythonProcess(pythonCmd, ['-c', code], options);
      
      if (result.success || result.exitCode === 0) {
        log.info(`[PythonUtils] Successfully executed with: ${pythonCmd}`);
        return result;
      } else if (result.stderr && !result.stderr.includes('ENOENT')) {
        // Python was found but there was an execution error
        log.warn(`[PythonUtils] Python execution failed with ${pythonCmd}:`, result.stderr);
        return result;
      }
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
      log.debug(`[PythonUtils] Failed to execute with ${pythonCmd}:`, lastError);
      continue;
    }
  }

  return {
    success: false,
    stdout: '',
    stderr: lastError,
    exitCode: -1,
    error: `No working Python executable found. Tried: ${commands.join(', ')}`
  };
}

/**
 * Execute Python script file with automatic Python path resolution
 */
export async function executePythonScript(
  scriptPath: string,
  args: string[] = [],
  options: PythonExecutionOptions = {}
): Promise<PythonExecutionResult> {
  if (!fs.existsSync(scriptPath)) {
    return {
      success: false,
      stdout: '',
      stderr: `Script not found: ${scriptPath}`,
      exitCode: -1,
      error: `Python script not found at path: ${scriptPath}`
    };
  }

  const commands = getPythonCommands();
  let lastError = '';

  for (const pythonCmd of commands) {
    try {
      log.info(`[PythonUtils] Trying to run script with: ${pythonCmd} ${scriptPath}`);
      const result = await runPythonProcess(pythonCmd, [scriptPath, ...args], options);
      
      if (result.success || result.exitCode === 0) {
        log.info(`[PythonUtils] Successfully executed script with: ${pythonCmd}`);
        return result;
      } else if (result.stderr && !result.stderr.includes('ENOENT')) {
        // Python was found but there was an execution error
        log.warn(`[PythonUtils] Script execution failed with ${pythonCmd}:`, result.stderr);
        return result;
      }
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
      log.debug(`[PythonUtils] Failed to execute script with ${pythonCmd}:`, lastError);
      continue;
    }
  }

  return {
    success: false,
    stdout: '',
    stderr: lastError,
    exitCode: -1,
    error: `No working Python executable found. Tried: ${commands.join(', ')}`
  };
}

/**
 * Run a Python process with the given command and arguments
 */
function runPythonProcess(
  pythonCmd: string,
  args: string[],
  options: PythonExecutionOptions
): Promise<PythonExecutionResult> {
  return new Promise((resolve) => {
    const spawnOptions: SpawnOptions = {
      env: options.env || process.env,
      cwd: options.cwd,
      stdio: options.stdio || 'pipe'
    };

    let childProcess: ChildProcess;
    
    try {
      childProcess = spawn(pythonCmd, args, spawnOptions);
    } catch (error) {
      resolve({
        success: false,
        stdout: '',
        stderr: error instanceof Error ? error.message : String(error),
        exitCode: -1,
        error: `Failed to spawn Python process: ${error}`
      });
      return;
    }

    let stdout = '';
    let stderr = '';

    // Set up timeout if specified
    let timeoutId: NodeJS.Timeout | null = null;
    if (options.timeout) {
      timeoutId = setTimeout(() => {
        if (childProcess && !childProcess.killed) {
          childProcess.kill('SIGKILL');
          resolve({
            success: false,
            stdout,
            stderr: stderr + '\nProcess killed due to timeout',
            exitCode: -1,
            error: `Python process timed out after ${options.timeout}ms`
          });
        }
      }, options.timeout);
    }

    // Collect stdout if available
    if (childProcess.stdout) {
      childProcess.stdout.on('data', (data: Buffer) => {
        stdout += data.toString();
      });
    }

    // Collect stderr if available
    if (childProcess.stderr) {
      childProcess.stderr.on('data', (data: Buffer) => {
        stderr += data.toString();
      });
    }

    // Handle process completion
    childProcess.on('close', (code: number | null) => {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }

      const success = code === 0;
      resolve({
        success,
        stdout: stdout.trim(),
        stderr: stderr.trim(),
        exitCode: code,
        error: success ? undefined : `Process exited with code ${code}`
      });
    });

    // Handle process errors
    childProcess.on('error', (error: Error) => {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }

      resolve({
        success: false,
        stdout,
        stderr: stderr + error.message,
        exitCode: -1,
        error: `Process error: ${error.message}`
      });
    });
  });
}

/**
 * Check if Python is available and working
 */
export async function checkPythonAvailability(): Promise<{ 
  available: boolean; 
  version?: string; 
  command?: string; 
  error?: string 
}> {
  try {
    const result = await executePythonCode('import sys; print(sys.version)');
    
    if (result.success && result.stdout) {
      return {
        available: true,
        version: result.stdout.trim(),
        command: 'python' // We don't expose the exact command for security
      };
    }

    return {
      available: false,
      error: result.error || result.stderr || 'Unknown error'
    };
  } catch (error) {
    return {
      available: false,
      error: error instanceof Error ? error.message : String(error)
    };
  }
}

/**
 * Test if a specific Python package is available
 */
export async function checkPythonPackage(packageName: string): Promise<{
  available: boolean;
  version?: string;
  error?: string;
}> {
  try {
    const testCode = `
try:
    import ${packageName}
    print(getattr(${packageName}, '__version__', 'unknown'))
except ImportError as e:
    print('IMPORT_ERROR: ' + str(e))
except Exception as e:
    print('ERROR: ' + str(e))
`;

    const result = await executePythonCode(testCode);
    
    if (result.success && result.stdout) {
      const output = result.stdout.trim();
      
      if (output.startsWith('IMPORT_ERROR:') || output.startsWith('ERROR:')) {
        return {
          available: false,
          error: output
        };
      }
      
      return {
        available: true,
        version: output !== 'unknown' ? output : undefined
      };
    }

    return {
      available: false,
      error: result.error || result.stderr || 'Package test failed'
    };
  } catch (error) {
    return {
      available: false,
      error: error instanceof Error ? error.message : String(error)
    };
  }
}