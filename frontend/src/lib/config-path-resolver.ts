/**
 * Unified Config Path Resolver
 * 
 * Single source of truth for all config path resolution across the Chatterley frontend.
 * Handles development vs production environments consistently.
 */

export interface ConfigPathResolver {
  loadStaticConfigs(): Promise<any>;
  getConfigById(configId: string): Promise<{ config: any; configPath: string } | null>;
}

class UnifiedConfigPathResolver implements ConfigPathResolver {
  private staticConfigs?: any;

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
   * Get a specific config by ID - returns original path for backend resolution
   */
  public async getConfigById(configId: string): Promise<{ config: any; configPath: string } | null> {
    try {
      const staticConfigs = await this.loadStaticConfigs();
      const config = staticConfigs.configs?.find((cfg: any) => cfg.id === configId);
      
      if (!config) {
        console.warn(`[ConfigPathResolver] Config not found: ${configId}`);
        return null;
      }

      const originalPath = config.config_path || config.relative_path;
      
      console.log(`[ConfigPathResolver] Found config ${configId}:`);
      console.log(`[ConfigPathResolver]   Config path (for backend): ${originalPath}`);
      
      return {
        config,
        configPath: originalPath
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
    this.staticConfigs = undefined;
  }
}

// Export singleton instance
export const configPathResolver = new UnifiedConfigPathResolver();

// Export class for testing
export { UnifiedConfigPathResolver };