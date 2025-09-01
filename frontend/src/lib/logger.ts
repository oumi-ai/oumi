/**
 * File-based logging system for Chatterley
 * Reduces console spam while maintaining detailed debug information
 */

export interface LogEntry {
  timestamp: string;
  level: 'DEBUG' | 'INFO' | 'WARN' | 'ERROR';
  component: string;
  message: string;
  data?: any;
}

export interface LoggerConfig {
  enableConsole: boolean;
  enableFile: boolean;
  logLevel: 'DEBUG' | 'INFO' | 'WARN' | 'ERROR';
  maxFileSize: number; // KB
}

class Logger {
  private config: LoggerConfig = {
    enableConsole: false, // Disabled by default to reduce spam
    enableFile: true,
    logLevel: 'DEBUG',
    maxFileSize: 1024 // 1MB
  };

  private logs: LogEntry[] = [];
  private maxLogs = 1000; // Keep last 1000 log entries in memory

  constructor() {
    // Load config from localStorage if available
    if (typeof window !== 'undefined') {
      try {
        const savedConfig = localStorage.getItem('chatterley-logger-config');
        if (savedConfig) {
          this.config = { ...this.config, ...JSON.parse(savedConfig) };
        }
      } catch (error) {
        // Ignore localStorage errors
      }
    }
  }

  /**
   * Update logger configuration
   */
  public configure(config: Partial<LoggerConfig>): void {
    this.config = { ...this.config, ...config };
    
    if (typeof window !== 'undefined') {
      try {
        localStorage.setItem('chatterley-logger-config', JSON.stringify(this.config));
      } catch (error) {
        // Ignore localStorage errors
      }
    }
  }

  /**
   * Check if logging level should be processed
   */
  private shouldLog(level: LogEntry['level']): boolean {
    const levels = ['DEBUG', 'INFO', 'WARN', 'ERROR'];
    const currentIndex = levels.indexOf(this.config.logLevel);
    const messageIndex = levels.indexOf(level);
    return messageIndex >= currentIndex;
  }

  /**
   * Add log entry
   */
  private addLogEntry(level: LogEntry['level'], component: string, message: string, data?: any): void {
    if (!this.shouldLog(level)) return;

    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      component,
      message,
      data
    };

    // Add to memory logs
    this.logs.push(entry);
    if (this.logs.length > this.maxLogs) {
      this.logs = this.logs.slice(-this.maxLogs);
    }

    // Console output (if enabled)
    if (this.config.enableConsole) {
      const logMessage = `[${entry.timestamp}] ${level} [${component}] ${message}`;
      
      switch (level) {
        case 'DEBUG':
          console.debug(logMessage, data || '');
          break;
        case 'INFO':
          console.info(logMessage, data || '');
          break;
        case 'WARN':
          console.warn(logMessage, data || '');
          break;
        case 'ERROR':
          console.error(logMessage, data || '');
          break;
      }
    }

    // File output (Electron only)
    if (this.config.enableFile && typeof window !== 'undefined' && window.electronAPI?.logger) {
      window.electronAPI.logger.writeLog(entry).catch((error: any) => {
        // Only log to console if electron logging fails (to avoid recursion)
        console.warn('[Logger] Failed to write to file:', error);
      });
    }
  }

  /**
   * Debug level logging
   */
  public debug(component: string, message: string, data?: any): void {
    this.addLogEntry('DEBUG', component, message, data);
  }

  /**
   * Info level logging
   */
  public info(component: string, message: string, data?: any): void {
    this.addLogEntry('INFO', component, message, data);
  }

  /**
   * Warning level logging
   */
  public warn(component: string, message: string, data?: any): void {
    this.addLogEntry('WARN', component, message, data);
  }

  /**
   * Error level logging
   */
  public error(component: string, message: string, data?: any): void {
    this.addLogEntry('ERROR', component, message, data);
  }

  /**
   * Get recent log entries
   */
  public getRecentLogs(count = 100, level?: LogEntry['level']): LogEntry[] {
    let logs = this.logs;
    
    if (level) {
      logs = logs.filter(log => log.level === level);
    }
    
    return logs.slice(-count);
  }

  /**
   * Clear all logs
   */
  public clear(): void {
    this.logs = [];
  }

  /**
   * Export logs as JSON string
   */
  public exportLogs(): string {
    return JSON.stringify(this.logs, null, 2);
  }

  /**
   * Enable console logging temporarily (for debugging)
   */
  public enableConsoleTemporarily(durationMs = 30000): void {
    const originalConfig = this.config.enableConsole;
    this.config.enableConsole = true;
    
    setTimeout(() => {
      this.config.enableConsole = originalConfig;
    }, durationMs);
  }
}

// Export singleton instance
export const logger = new Logger();

// Export class for testing
export { Logger };

// Logger uses existing electronAPI types from preload.ts