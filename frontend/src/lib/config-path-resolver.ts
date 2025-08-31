/**
 * Unified Config Path Resolver
 * 
 * Single source of truth for all config path resolution across the Chatterley frontend.
 * Handles development vs production environments consistently.
 */

export interface ConfigPathResolver {
  resolveConfigPath(relativePath: string): string;
  getConfigsBasePath(): string;
  isProduction(): boolean;
  loadStaticConfigs(): Promise<any>;
}

class UnifiedConfigPathResolver implements ConfigPathResolver {
  private cachedBasePath?: string;
  private cachedIsProduction?: boolean;
  private staticConfigs?: any;

  /**
   * Determine if we're running in production mode
   */
  public isProduction(): boolean {
    if (this.cachedIsProduction === undefined) {
      // Check multiple indicators for production mode
      const isElectron = typeof window !== 'undefined' && 'electronAPI' in window;
      
      if (isElectron) {
        // In Electron, use multiple indicators for production mode
        // Check if running from a packaged app by examining the process
        try {
          this.cachedIsProduction = window.location.protocol === 'file:' ||
            process.env.NODE_ENV === 'production' ||
            !window.location.hostname.includes('localhost');
        } catch {
          // Fallback if process is not available
          this.cachedIsProduction = window.location.protocol === 'file:';
        }
      } else {
        // In web mode, check environment and URL patterns
        this.cachedIsProduction = process.env.NODE_ENV === 'production' || 
          window.location.protocol === 'file:' ||
          !window.location.hostname.includes('localhost');
      }
    }
    return this.cachedIsProduction;
  }

  /**
   * Get the base path for config files based on environment
   */
  public getConfigsBasePath(): string {
    if (this.cachedBasePath) {
      return this.cachedBasePath;
    }

    const isElectron = typeof window !== 'undefined' && 'electronAPI' in window;
    
    if (isElectron) {
      if (this.isProduction()) {
        // Production Electron: configs are bundled in Resources/python/configs/
        this.cachedBasePath = 'python/configs';
      } else {
        // Development Electron: configs are in oumi/configs/ relative to frontend
        this.cachedBasePath = '../../../configs';
      }
    } else {
      // Web mode: configs should be served from public directory
      this.cachedBasePath = './configs';
    }

    console.log(`[ConfigPathResolver] Base path: ${this.cachedBasePath} (production: ${this.isProduction()})`);
    return this.cachedBasePath;
  }

  /**
   * Resolve a relative config path to the appropriate absolute or resolvable path
   */
  public resolveConfigPath(relativePath: string): string {
    // If already absolute, return as-is
    if (this.isAbsolutePath(relativePath)) {
      return relativePath;
    }

    // Remove leading slash if present
    const cleanPath = relativePath.startsWith('/') ? relativePath.slice(1) : relativePath;
    
    const basePath = this.getConfigsBasePath();
    const resolvedPath = `${basePath}/${cleanPath}`;
    
    console.log(`[ConfigPathResolver] Resolved: ${relativePath} -> ${resolvedPath}`);
    return resolvedPath;
  }

  /**
   * Load static configs with consistent error handling
   */
  public async loadStaticConfigs(): Promise<any> {
    if (this.staticConfigs) {
      return this.staticConfigs;
    }

    try {
      // Try multiple potential locations for static configs
      const locations = [
        './static-configs.json',
        '/static-configs.json', 
        'static-configs.json'
      ];

      for (const location of locations) {
        try {
          console.log(`[ConfigPathResolver] Trying to load static configs from: ${location}`);
          const response = await fetch(location);
          if (response.ok) {
            this.staticConfigs = await response.json();
            console.log(`[ConfigPathResolver] Successfully loaded ${this.staticConfigs.configs?.length || 0} configs from: ${location}`);
            return this.staticConfigs;
          }
        } catch (err) {
          console.warn(`[ConfigPathResolver] Failed to load from ${location}:`, err);
        }
      }

      // If all locations fail, try Electron's bundled config discovery
      const isElectron = typeof window !== 'undefined' && 'electronAPI' in window;
      if (isElectron && window.electronAPI?.config?.discoverBundled) {
        console.log('[ConfigPathResolver] Falling back to Electron bundled config discovery');
        try {
          const result = await window.electronAPI.config.discoverBundled();
          if (result.success && result.data) {
            this.staticConfigs = result.data;
            console.log(`[ConfigPathResolver] Successfully discovered ${this.staticConfigs.configs?.length || 0} bundled configs`);
            return this.staticConfigs;
          }
        } catch (err) {
          console.warn('[ConfigPathResolver] Electron config discovery failed:', err);
        }
      }

      throw new Error('Failed to load configs from all attempted locations');
    } catch (error) {
      console.error('[ConfigPathResolver] Error loading static configs:', error);
      throw error;
    }
  }

  /**
   * Get a specific config by ID with proper path resolution
   */
  public async getConfigById(configId: string): Promise<{ config: any; resolvedPath: string } | null> {
    try {
      const staticConfigs = await this.loadStaticConfigs();
      const config = staticConfigs.configs?.find((cfg: any) => cfg.id === configId);
      
      if (!config) {
        console.warn(`[ConfigPathResolver] Config not found: ${configId}`);
        return null;
      }

      const resolvedPath = this.resolveConfigPath(config.config_path || config.relative_path);
      
      console.log(`[ConfigPathResolver] Found config ${configId}:`);
      console.log(`[ConfigPathResolver]   Original path: ${config.config_path || config.relative_path}`);
      console.log(`[ConfigPathResolver]   Resolved path: ${resolvedPath}`);
      
      return {
        config,
        resolvedPath
      };
    } catch (error) {
      console.error(`[ConfigPathResolver] Error getting config ${configId}:`, error);
      return null;
    }
  }

  /**
   * Clear cached values (useful for testing or environment changes)
   */
  public clearCache(): void {
    this.cachedBasePath = undefined;
    this.cachedIsProduction = undefined;
    this.staticConfigs = undefined;
  }

  /**
   * Check if a path is absolute
   */
  private isAbsolutePath(path: string): boolean {
    // Unix absolute path
    if (path.startsWith('/')) return true;
    
    // Windows absolute path  
    if (/^[A-Za-z]:\\/.test(path)) return true;
    
    // URL
    if (path.includes('://')) return true;
    
    return false;
  }
}

// Export singleton instance
export const configPathResolver = new UnifiedConfigPathResolver();

// Export class for testing
export { UnifiedConfigPathResolver };