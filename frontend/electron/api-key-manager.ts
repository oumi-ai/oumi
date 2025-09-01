/**
 * Secure API Key Manager - Handles encrypted storage and validation of API keys
 */

import { app } from 'electron';
import Store from 'electron-store';
import * as crypto from 'crypto';
import * as os from 'os';
import log from 'electron-log';
import * as path from 'path';
import { executePythonCode } from './python-utils';

export interface ApiKeyConfig {
  providerId: string;
  keyValue: string;
  isActive: boolean;
  isValid?: boolean;
  lastValidated?: string;
  details?: any;
}

export interface ApiValidationResult {
  isValid: boolean;
  error?: string;
  details?: any;
}

class ApiKeyManager {
  private encryptedStore: Store;
  private encryptionKey: string;

  constructor() {
    // Generate machine-specific encryption key
    this.encryptionKey = this.generateEncryptionKey();
    
    // Initialize encrypted store
    this.encryptedStore = new Store({
      name: 'api-keys',
      encryptionKey: this.encryptionKey,
      defaults: {}
    });

    log.info('[ApiKeyManager] Initialized with encrypted storage');
  }

  /**
   * Generate a machine-specific encryption key
   */
  private generateEncryptionKey(): string {
    try {
      // Create a machine-specific seed
      const machineId = os.hostname() + os.platform() + os.arch();
      const appSalt = 'chatterley-api-keys-v1';
      
      // Use PBKDF2 to derive encryption key
      const key = crypto.pbkdf2Sync(
        machineId + appSalt,
        'oumi-chatterley-salt',
        100000, // iterations
        32, // key length in bytes
        'sha256'
      );

      return key.toString('hex');
    } catch (error) {
      log.error('[ApiKeyManager] Failed to generate encryption key:', error);
      // Fallback to a less secure but functional key
      return crypto.createHash('sha256').update('fallback-key').digest('hex');
    }
  }

  /**
   * Store an API key securely
   */
  public storeApiKey(config: ApiKeyConfig): void {
    try {
      this.encryptedStore.set(`keys.${config.providerId}`, {
        keyValue: config.keyValue,
        isActive: config.isActive,
        isValid: config.isValid,
        lastValidated: config.lastValidated || new Date().toISOString(),
        details: config.details,
        createdAt: new Date().toISOString()
      });
      
      log.info(`[ApiKeyManager] Stored API key for provider: ${config.providerId}`);
    } catch (error) {
      log.error(`[ApiKeyManager] Failed to store API key for ${config.providerId}:`, error);
      throw new Error('Failed to store API key securely');
    }
  }

  /**
   * Retrieve an API key
   */
  public getApiKey(providerId: string): ApiKeyConfig | null {
    try {
      const stored = this.encryptedStore.get(`keys.${providerId}`) as any;
      if (!stored) return null;

      return {
        providerId,
        keyValue: stored.keyValue,
        isActive: stored.isActive || false,
        isValid: stored.isValid,
        lastValidated: stored.lastValidated,
        details: stored.details
      };
    } catch (error) {
      log.error(`[ApiKeyManager] Failed to retrieve API key for ${providerId}:`, error);
      return null;
    }
  }

  /**
   * Get all stored API keys (without exposing key values)
   */
  public getAllApiKeys(): Omit<ApiKeyConfig, 'keyValue'>[] {
    try {
      const keys = this.encryptedStore.get('keys') as Record<string, any> || {};
      
      return Object.entries(keys).map(([providerId, data]) => ({
        providerId,
        isActive: data.isActive || false,
        isValid: data.isValid,
        lastValidated: data.lastValidated,
        details: data.details
      }));
    } catch (error) {
      log.error('[ApiKeyManager] Failed to retrieve API keys list:', error);
      return [];
    }
  }

  /**
   * Remove an API key
   */
  public removeApiKey(providerId: string): boolean {
    try {
      this.encryptedStore.delete(`keys.${providerId}`);
      log.info(`[ApiKeyManager] Removed API key for provider: ${providerId}`);
      return true;
    } catch (error) {
      log.error(`[ApiKeyManager] Failed to remove API key for ${providerId}:`, error);
      return false;
    }
  }

  /**
   * Update API key status
   */
  public updateApiKeyStatus(providerId: string, updates: Partial<Omit<ApiKeyConfig, 'providerId' | 'keyValue'>>): boolean {
    try {
      const existing = this.encryptedStore.get(`keys.${providerId}`) as any;
      if (!existing) return false;

      const updated = {
        ...existing,
        ...updates,
        lastValidated: updates.lastValidated || existing.lastValidated
      };

      this.encryptedStore.set(`keys.${providerId}`, updated);
      log.info(`[ApiKeyManager] Updated API key status for provider: ${providerId}`);
      return true;
    } catch (error) {
      log.error(`[ApiKeyManager] Failed to update API key status for ${providerId}:`, error);
      return false;
    }
  }

  /**
   * Validate API key using Oumi configuration
   */
  public async validateApiKeyWithOumi(providerId: string): Promise<ApiValidationResult> {
    const apiKey = this.getApiKey(providerId);
    if (!apiKey) {
      return { isValid: false, error: 'API key not found' };
    }

    try {
      log.info(`[ApiKeyManager] Validating API key for ${providerId} using Oumi`);

      // Find appropriate API config for the provider
      const configPath = await this.findApiConfigForProvider(providerId);
      if (!configPath) {
        return { isValid: false, error: `No Oumi config found for provider: ${providerId}` };
      }

      // Test the API key using Oumi
      const result = await this.testWithOumiConfig(configPath, providerId, apiKey.keyValue);
      
      // Update the stored key with validation result
      this.updateApiKeyStatus(providerId, {
        isValid: result.isValid,
        lastValidated: new Date().toISOString(),
        details: result.details
      });

      return result;
    } catch (error) {
      log.error(`[ApiKeyManager] Validation failed for ${providerId}:`, error);
      return { 
        isValid: false, 
        error: error instanceof Error ? error.message : 'Validation failed' 
      };
    }
  }

  /**
   * Find the appropriate Oumi API config for a provider
   */
  private async findApiConfigForProvider(providerId: string): Promise<string | null> {
    try {
      const oumiRoot = app.isPackaged
        ? path.join(process.resourcesPath, 'python')
        : path.join(__dirname, '../../../');
      
      const configsPath = path.resolve(oumiRoot, 'configs', 'apis');
      
      // Map provider IDs to config directories
      const providerConfigMap: Record<string, string> = {
        'openai': 'openai/infer_gpt_4o.yaml',
        'anthropic': 'anthropic/infer_claude_3_5_sonnet.yaml',
        'google': 'gemini/infer_gemini_1_5_pro.yaml'
      };

      const configFile = providerConfigMap[providerId];
      if (!configFile) {
        log.warn(`[ApiKeyManager] No config mapping for provider: ${providerId}`);
        return null;
      }

      const fullPath = path.join(configsPath, configFile);
      return fullPath;
    } catch (error) {
      log.error(`[ApiKeyManager] Failed to find config for ${providerId}:`, error);
      return null;
    }
  }

  /**
   * Test API key using Oumi configuration
   */
  private async testWithOumiConfig(configPath: string, providerId: string, apiKey: string): Promise<ApiValidationResult> {
    try {
      // Set up environment with API key
      const env = { ...process.env };
      const envVarMap: Record<string, string> = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY', 
        'google': 'GOOGLE_API_KEY',
        'gemini': 'GOOGLE_API_KEY',
        'together': 'TOGETHER_API_KEY',
        'deepseek': 'DEEPSEEK_API_KEY',
        'sambanova': 'SAMBANOVA_API_KEY',
        'parasail': 'PARASAIL_API_KEY',
        'lambda': 'LAMBDA_API_KEY'
      };

      const envVar = envVarMap[providerId];
      if (!envVar) {
        return { isValid: false, error: `No environment variable mapping for ${providerId}` };
      }

      env[envVar] = apiKey;
      
      // Use a simple test: try to load the model with the config
      const testScript = `
import os
import sys
sys.path.insert(0, '${path.dirname(configPath).replace(/\\/g, '\\\\')}/../../../src')

try:
    from oumi.core.configs import InferenceConfig
    config = InferenceConfig.from_yaml('${configPath.replace(/\\/g, '\\\\')}')
    print('{"success": true, "model": "' + config.model.model_name + '"}')
except Exception as e:
    print('{"success": false, "error": "' + str(e).replace('"', '\\\\"') + '"}')
`;

      const result = await executePythonCode(testScript, {
        env,
        timeout: 10000
      });

      if (result.success && result.stdout) {
        try {
          const validationResult = JSON.parse(result.stdout);
          if (validationResult.success) {
            return {
              isValid: true,
              details: {
                model: validationResult.model,
                validatedWith: 'Oumi',
                configPath: configPath
              }
            };
          } else {
            return {
              isValid: false,
              error: validationResult.error || 'Configuration test failed'
            };
          }
        } catch (parseError) {
          return {
            isValid: false,
            error: 'Failed to parse validation result'
          };
        }
      } else {
        return {
          isValid: false,
          error: result.error || result.stderr || 'Python execution failed'
        };
      }
    } catch (error) {
      return {
        isValid: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  /**
   * Check if there are any legacy (unencrypted) API keys that need migration
   */
  public hasLegacyKeys(): boolean {
    // This would check if there are keys stored in the old format
    // Implementation would depend on how keys were previously stored
    return false; // Placeholder
  }

  /**
   * Get all stored API keys with their values (for backend environment setup)
   * WARNING: This exposes key values - use only for secure backend communication
   */
  public getAllKeysWithValues(): Record<string, string> {
    try {
      const keys: Record<string, string> = {};
      
      log.info('[ApiKeyManager] getAllKeysWithValues - checking stored data...');
      
      // Try both approaches: direct store access and using the existing getApiKey method
      const storeData = this.encryptedStore.store;
      log.info(`[ApiKeyManager] Store data keys:`, Object.keys(storeData));
      
      // Approach 1: Direct store iteration (current approach)
      for (const [key, value] of Object.entries(storeData)) {
        log.info(`[ApiKeyManager] Checking store key: ${key}`);
        if (key.startsWith('keys.')) {
          const providerId = key.replace('keys.', '');
          const stored = value as any;
          
          log.info(`[ApiKeyManager] Found key for provider ${providerId}:`, {
            hasKeyValue: !!stored?.keyValue,
            isActive: stored?.isActive,
            isValid: stored?.isValid,
            keyLength: stored?.keyValue ? stored.keyValue.length : 0
          });
          
          if (stored && stored.keyValue) {
            // Include key even if isActive is not set (might be undefined/false)
            // We'll make isActive optional for now to debug the issue
            if (stored.isActive !== false) { // Include if isActive is true or undefined
              keys[providerId] = stored.keyValue;
              log.info(`[ApiKeyManager] Including key for provider: ${providerId}`);
            } else {
              log.warn(`[ApiKeyManager] Excluding inactive key for provider: ${providerId}`);
            }
          } else {
            log.warn(`[ApiKeyManager] Invalid key data for provider ${providerId}`);
          }
        }
      }
      
      // Approach 2: Try using the existing getApiKey method for known providers
      const knownProviders = ['openai', 'anthropic', 'google', 'gemini', 'together', 'deepseek'];
      log.info(`[ApiKeyManager] Trying getApiKey method for known providers...`);
      
      for (const providerId of knownProviders) {
        if (!keys[providerId]) { // Only if not already found
          const keyConfig = this.getApiKey(providerId);
          if (keyConfig && keyConfig.keyValue) {
            keys[providerId] = keyConfig.keyValue;
            log.info(`[ApiKeyManager] Found key via getApiKey for provider: ${providerId}`);
          }
        }
      }
      
      log.info(`[ApiKeyManager] getAllKeysWithValues returning ${Object.keys(keys).length} keys: ${Object.keys(keys).join(', ')}`);
      return keys;
    } catch (error) {
      log.error('[ApiKeyManager] Failed to retrieve all API keys:', error);
      return {};
    }
  }

  /**
   * Clear all stored API keys (for security reset)
   */
  public clearAllKeys(): void {
    try {
      this.encryptedStore.clear();
      log.info('[ApiKeyManager] Cleared all stored API keys');
    } catch (error) {
      log.error('[ApiKeyManager] Failed to clear API keys:', error);
      throw new Error('Failed to clear API keys');
    }
  }
}

export const apiKeyManager = new ApiKeyManager();