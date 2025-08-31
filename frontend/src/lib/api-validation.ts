/**
 * API Key Validation Service
 */

import { ApiProvider, ApiValidationResult } from './types';
import { getProviderById } from './api-providers';

interface ValidationRequest {
  provider: ApiProvider;
  apiKey: string;
}

class ApiValidationService {
  private cache = new Map<string, { result: ApiValidationResult; timestamp: number }>();
  private readonly CACHE_TTL = 5 * 60 * 1000; // 5 minutes

  async validateKey(providerId: string, apiKey: string): Promise<ApiValidationResult> {
    const provider = getProviderById(providerId);
    if (!provider) {
      return {
        isValid: false,
        error: 'Unknown provider',
      };
    }

    // Check cache first
    const cacheKey = `${providerId}:${apiKey}`;
    const cached = this.cache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < this.CACHE_TTL) {
      return cached.result;
    }

    try {
      const result = await this.performValidation({ provider, apiKey });
      
      // Cache the result
      this.cache.set(cacheKey, {
        result,
        timestamp: Date.now(),
      });

      return result;
    } catch (error) {
      const errorResult: ApiValidationResult = {
        isValid: false,
        error: error instanceof Error ? error.message : 'Validation failed',
      };

      // Cache error results for a shorter time
      this.cache.set(cacheKey, {
        result: errorResult,
        timestamp: Date.now(),
      });

      return errorResult;
    }
  }

  private async performValidation({ provider, apiKey }: ValidationRequest): Promise<ApiValidationResult> {
    // Basic format validation first
    const formatValidation = this.validateFormat(provider, apiKey);
    if (!formatValidation.isValid) {
      return formatValidation;
    }

    // Perform actual API validation
    switch (provider.id) {
      case 'openai':
        return this.validateOpenAI(apiKey);
      case 'anthropic':
        return this.validateAnthropic(apiKey);
      case 'google':
        return this.validateGoogle(apiKey);
      case 'groq':
        return this.validateGroq(apiKey);
      case 'together':
        return this.validateTogether(apiKey);
      default:
        return this.validateGeneric(provider, apiKey);
    }
  }

  private validateFormat(provider: ApiProvider, apiKey: string): ApiValidationResult {
    if (!apiKey || apiKey.trim().length === 0) {
      return { isValid: false, error: 'API key cannot be empty' };
    }

    const key = apiKey.trim();

    switch (provider.id) {
      case 'openai':
        if (!key.startsWith('sk-') || key.length < 40) {
          return { isValid: false, error: 'OpenAI API keys should start with "sk-" and be at least 40 characters long' };
        }
        break;
      
      case 'anthropic':
        if (!key.startsWith('sk-ant-') || key.length < 30) {
          return { isValid: false, error: 'Anthropic API keys should start with "sk-ant-" and be at least 30 characters long' };
        }
        break;
      
      case 'google':
        if (!key.startsWith('AI') || key.length < 20) {
          return { isValid: false, error: 'Google API keys should start with "AI" and be at least 20 characters long' };
        }
        break;
      
      case 'groq':
        if (!key.startsWith('gsk_') || key.length < 20) {
          return { isValid: false, error: 'Groq API keys should start with "gsk_" and be at least 20 characters long' };
        }
        break;
      
      default:
        if (key.length < 10) {
          return { isValid: false, error: 'API key seems too short' };
        }
    }

    return { isValid: true };
  }

  private async validateOpenAI(apiKey: string): Promise<ApiValidationResult> {
    try {
      const response = await fetch('https://api.openai.com/v1/models', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
      });

      if (response.status === 401) {
        return { isValid: false, error: 'Invalid API key' };
      }

      if (response.status === 429) {
        return { isValid: false, error: 'Rate limit exceeded' };
      }

      if (!response.ok) {
        return { isValid: false, error: `API error: ${response.status}` };
      }

      const data = await response.json();
      const models = data.data || [];
      
      return {
        isValid: true,
        details: {
          model: models[0]?.id || 'Unknown',
          rateLimit: parseInt(response.headers.get('x-ratelimit-limit-requests') || '0'),
        },
      };
    } catch (error) {
      if (error instanceof TypeError && error.message.includes('fetch')) {
        return { isValid: false, error: 'Network error - please check your connection' };
      }
      return { isValid: false, error: 'Failed to validate API key' };
    }
  }

  private async validateAnthropic(apiKey: string): Promise<ApiValidationResult> {
    try {
      // Anthropic doesn't have a simple models endpoint, so we'll make a minimal request
      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'x-api-key': apiKey,
          'Content-Type': 'application/json',
          'anthropic-version': '2023-06-01',
        },
        body: JSON.stringify({
          model: 'claude-3-haiku-20240307',
          max_tokens: 1,
          messages: [{ role: 'user', content: 'Hi' }],
        }),
      });

      if (response.status === 401) {
        return { isValid: false, error: 'Invalid API key' };
      }

      if (response.status === 429) {
        return { isValid: false, error: 'Rate limit exceeded' };
      }

      // For Anthropic, even a 400 (bad request) with valid auth means the key works
      if (response.status === 400 || response.status === 200) {
        return {
          isValid: true,
          details: {
            model: 'claude-3-haiku-20240307',
          },
        };
      }

      return { isValid: false, error: `API error: ${response.status}` };
    } catch (error) {
      if (error instanceof TypeError && error.message.includes('fetch')) {
        return { isValid: false, error: 'Network error - please check your connection' };
      }
      return { isValid: false, error: 'Failed to validate API key' };
    }
  }

  private async validateGoogle(apiKey: string): Promise<ApiValidationResult> {
    try {
      const response = await fetch(`https://generativelanguage.googleapis.com/v1/models?key=${apiKey}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.status === 400) {
        const data = await response.json().catch(() => null);
        if (data?.error?.code === 400) {
          return { isValid: false, error: 'Invalid API key' };
        }
      }

      if (response.status === 429) {
        return { isValid: false, error: 'Rate limit exceeded' };
      }

      if (!response.ok) {
        return { isValid: false, error: `API error: ${response.status}` };
      }

      const data = await response.json();
      const models = data.models || [];

      return {
        isValid: true,
        details: {
          model: models[0]?.name?.split('/').pop() || 'gemini-pro',
        },
      };
    } catch (error) {
      if (error instanceof TypeError && error.message.includes('fetch')) {
        return { isValid: false, error: 'Network error - please check your connection' };
      }
      return { isValid: false, error: 'Failed to validate API key' };
    }
  }

  private async validateGroq(apiKey: string): Promise<ApiValidationResult> {
    try {
      const response = await fetch('https://api.groq.com/openai/v1/models', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
      });

      if (response.status === 401) {
        return { isValid: false, error: 'Invalid API key' };
      }

      if (response.status === 429) {
        return { isValid: false, error: 'Rate limit exceeded' };
      }

      if (!response.ok) {
        return { isValid: false, error: `API error: ${response.status}` };
      }

      const data = await response.json();
      const models = data.data || [];

      return {
        isValid: true,
        details: {
          model: models[0]?.id || 'llama3-groq-70b-8192-tool-use-preview',
        },
      };
    } catch (error) {
      if (error instanceof TypeError && error.message.includes('fetch')) {
        return { isValid: false, error: 'Network error - please check your connection' };
      }
      return { isValid: false, error: 'Failed to validate API key' };
    }
  }

  private async validateTogether(apiKey: string): Promise<ApiValidationResult> {
    try {
      const response = await fetch('https://api.together.xyz/v1/models', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
      });

      if (response.status === 401) {
        return { isValid: false, error: 'Invalid API key' };
      }

      if (response.status === 429) {
        return { isValid: false, error: 'Rate limit exceeded' };
      }

      if (!response.ok) {
        return { isValid: false, error: `API error: ${response.status}` };
      }

      const data = await response.json();
      const models = data.data || [];

      return {
        isValid: true,
        details: {
          model: models[0]?.id || 'meta-llama/Llama-3-70b-chat-hf',
        },
      };
    } catch (error) {
      if (error instanceof TypeError && error.message.includes('fetch')) {
        return { isValid: false, error: 'Network error - please check your connection' };
      }
      return { isValid: false, error: 'Failed to validate API key' };
    }
  }

  private async validateGeneric(provider: ApiProvider, apiKey: string): Promise<ApiValidationResult> {
    // For providers without specific validation logic, just return format validation
    return {
      isValid: true,
      details: {
        model: provider.models[0]?.name || 'Unknown',
      },
    };
  }

  clearCache(): void {
    this.cache.clear();
  }

  getCacheSize(): number {
    return this.cache.size;
  }
}

export const apiValidationService = new ApiValidationService();