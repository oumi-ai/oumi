/**
 * Unified Config Path Resolver
 * 
 * Single source of truth for all config path resolution across the Chatterley frontend.
 * Handles development vs production environments consistently.
 */

import { logger } from './logger';

export interface ConfigPathResolver {
  loadStaticConfigs(): Promise<any>;
  getConfigById(configId: string): Promise<{ config: any; configPath: string } | null>;
}

class UnifiedConfigPathResolver implements ConfigPathResolver {
  private staticConfigs?: any;
  private lastLoadAttempt: number = 0;
  private loadAttemptInterval: number = 5000; // Don't retry failed loads more than once per 5 seconds

  /**
   * Load static configs with consistent error handling
   */
  public async loadStaticConfigs(): Promise<any> {
    logger.debug('ConfigPathResolver', `loadStaticConfigs called, cached: ${!!this.staticConfigs}`);
    
    if (this.staticConfigs) {
      logger.debug('ConfigPathResolver', `Returning cached configs (${this.staticConfigs.configs?.length || 0} configs)`);
      return this.staticConfigs;
    }

    // Rate limit failed attempts to prevent spam
    const now = Date.now();
    if (now - this.lastLoadAttempt < this.loadAttemptInterval) {
      logger.debug('ConfigPathResolver', 'Skipping load attempt due to rate limiting');
      throw new Error('Config loading rate limited - waiting between attempts');
    }
    this.lastLoadAttempt = now;

    try {
      // Try multiple potential locations for static configs
      const locations = [
        './static-configs.json',
        '/static-configs.json', 
        'static-configs.json'
      ];

      logger.debug('ConfigPathResolver', `Attempting to load from ${locations.length} locations`);
      const errors: string[] = [];

      for (const location of locations) {
        try {
          logger.debug('ConfigPathResolver', `Trying location: ${location}`);
          const response = await fetch(location);
          
          if (response.ok) {
            this.staticConfigs = await response.json();
            logger.info('ConfigPathResolver', `Successfully loaded ${this.staticConfigs.configs?.length || 0} configs from: ${location}`);
            return this.staticConfigs;
          } else {
            errors.push(`${location}: ${response.status} ${response.statusText}`);
          }
        } catch (err) {
          errors.push(`${location}: ${err instanceof Error ? err.message : 'Unknown error'}`);
        }
      }

      // If all locations fail, try Electron's bundled config discovery
      const isElectron = typeof window !== 'undefined' && 'electronAPI' in window;
      if (isElectron && window.electronAPI?.config?.discoverBundled) {
        logger.debug('ConfigPathResolver', 'Falling back to Electron bundled config discovery');
        try {
          const result = await window.electronAPI.config.discoverBundled();
          if (result.success && result.data) {
            this.staticConfigs = result.data;
            logger.info('ConfigPathResolver', `Successfully discovered ${this.staticConfigs.configs?.length || 0} bundled configs`);
            return this.staticConfigs;
          } else {
            errors.push(`Electron discovery: ${result.error || 'Unknown error'}`);
          }
        } catch (err) {
          errors.push(`Electron discovery: ${err instanceof Error ? err.message : 'Unknown error'}`);
        }
      }

      const errorMessage = 'Failed to load configs from all attempted locations';
      logger.warn('ConfigPathResolver', errorMessage, { attempts: errors });
      throw new Error(errorMessage);
    } catch (error) {
      logger.error('ConfigPathResolver', 'Error loading static configs', error);
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
        logger.warn('ConfigPathResolver', `Config not found: ${configId}`);
        return null;
      }

      const originalPath = config.config_path || config.relative_path;
      
      logger.debug('ConfigPathResolver', `Found config ${configId}`, {
        configPath: originalPath,
        displayName: config.display_name
      });
      
      return {
        config,
        configPath: originalPath
      };
    } catch (error) {
      logger.error('ConfigPathResolver', `Error getting config ${configId}`, error);
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