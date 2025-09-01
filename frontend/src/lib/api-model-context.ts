/**
 * API Model Context Length Utilities
 * 
 * TODO: These context lengths should eventually be fetched from the actual API providers
 * rather than being hardcoded. Each provider should expose their model context limits.
 */

interface ApiModelContextInfo {
  contextLength: number;
  isDynamic?: boolean; // Whether the context length can vary
  notes?: string;
}

/**
 * Get context length information for API-based models
 * This is a temporary solution until we can fetch real context limits from providers
 */
export function getApiModelContextLength(modelName: string, engine: string): ApiModelContextInfo {
  const engineLower = engine.toLowerCase();
  const modelLower = modelName.toLowerCase();

  // OpenAI Models
  if (engineLower.includes('openai')) {
    if (modelLower.includes('gpt-4o')) {
      return { contextLength: 128000, notes: 'GPT-4o series' };
    }
    if (modelLower.includes('gpt-4-turbo') || modelLower.includes('gpt-4-1106') || modelLower.includes('gpt-4-0125')) {
      return { contextLength: 128000, notes: 'GPT-4 Turbo' };
    }
    if (modelLower.includes('gpt-4')) {
      return { contextLength: 8192, notes: 'GPT-4 base' };
    }
    if (modelLower.includes('gpt-3.5-turbo-16k')) {
      return { contextLength: 16384, notes: 'GPT-3.5 Turbo 16K' };
    }
    if (modelLower.includes('gpt-3.5-turbo')) {
      return { contextLength: 4096, notes: 'GPT-3.5 Turbo' };
    }
    if (modelLower.includes('o1')) {
      return { contextLength: 200000, notes: 'o1 series' };
    }
    // Default for unknown OpenAI models
    return { contextLength: 128000, isDynamic: true, notes: 'OpenAI API (estimated)' };
  }

  // Anthropic Models
  if (engineLower.includes('anthropic')) {
    if (modelLower.includes('claude-3-5-sonnet')) {
      return { contextLength: 200000, notes: 'Claude 3.5 Sonnet' };
    }
    if (modelLower.includes('claude-3')) {
      return { contextLength: 200000, notes: 'Claude 3 series' };
    }
    if (modelLower.includes('claude-2')) {
      return { contextLength: 100000, notes: 'Claude 2 series' };
    }
    // Default for unknown Anthropic models
    return { contextLength: 200000, isDynamic: true, notes: 'Anthropic API (estimated)' };
  }

  // Google Models
  if (engineLower.includes('google') || engineLower.includes('gemini')) {
    if (modelLower.includes('gemini-1.5')) {
      return { contextLength: 1000000, notes: 'Gemini 1.5 Pro/Flash' };
    }
    if (modelLower.includes('gemini-pro')) {
      return { contextLength: 30720, notes: 'Gemini Pro' };
    }
    if (modelLower.includes('gemini')) {
      return { contextLength: 30720, notes: 'Gemini series' };
    }
    // Default for unknown Google models
    return { contextLength: 32768, isDynamic: true, notes: 'Google API (estimated)' };
  }

  // Cohere Models
  if (engineLower.includes('cohere')) {
    return { contextLength: 128000, isDynamic: true, notes: 'Cohere API (estimated)' };
  }

  // Together API Models
  if (engineLower === 'together' || engineLower.includes('together')) {
    // Together hosts various models with different context lengths
    if (modelLower.includes('llama-3') || modelLower.includes('llama3')) {
      return { contextLength: 8192, notes: 'Llama 3 via Together' };
    }
    if (modelLower.includes('mixtral')) {
      return { contextLength: 32768, notes: 'Mixtral via Together' };
    }
    return { contextLength: 32768, isDynamic: true, notes: 'Together API (estimated)' };
  }

  // DeepSeek Models  
  if (engineLower === 'deepseek' || engineLower.includes('deepseek')) {
    if (modelLower.includes('v3')) {
      return { contextLength: 128000, notes: 'DeepSeek V3' };
    }
    return { contextLength: 64000, isDynamic: true, notes: 'DeepSeek API (estimated)' };
  }

  // SambaNova Models
  if (engineLower === 'sambanova' || engineLower.includes('sambanova')) {
    return { contextLength: 128000, isDynamic: true, notes: 'SambaNova API (estimated)' };
  }

  // Parasail Models  
  if (engineLower === 'parasail' || engineLower.includes('parasail')) {
    return { contextLength: 32768, isDynamic: true, notes: 'Parasail API (estimated)' };
  }

  // Lambda AI Models
  if (engineLower === 'lambda' || engineLower.includes('lambda')) {
    return { contextLength: 32768, isDynamic: true, notes: 'Lambda AI API (estimated)' };
  }

  // Mistral AI Models (if supported via API)
  if (engineLower.includes('mistral')) {
    if (modelLower.includes('large')) {
      return { contextLength: 128000, notes: 'Mistral Large' };
    }
    if (modelLower.includes('medium')) {
      return { contextLength: 32768, notes: 'Mistral Medium' };
    }
    return { contextLength: 32768, isDynamic: true, notes: 'Mistral API (estimated)' };
  }

  // Default for unknown API providers
  return { contextLength: 32768, isDynamic: true, notes: 'API provider (estimated)' };
}

/**
 * Check if a model is an API-based model (doesn't use local resources)
 * Based on Oumi's InferenceEngineType definitions
 */
export function isApiBasedModel(engine: string): boolean {
  const engineLower = engine.toLowerCase();
  
  // Oumi's API-based inference engines that don't require local resources
  return engineLower === 'openai' ||
         engineLower === 'anthropic' ||
         engineLower === 'together' ||
         engineLower === 'deepseek' ||
         engineLower === 'sambanova' ||
         engineLower === 'parasail' ||
         engineLower === 'lambda' ||
         engineLower === 'google_gemini' ||
         engineLower === 'gemini' ||
         engineLower === 'google_vertex' ||
         engineLower === 'vertex' ||
         // Legacy/partial matches for backward compatibility
         engineLower.includes('openai') ||
         engineLower.includes('anthropic') ||
         engineLower.includes('google') ||
         engineLower.includes('gemini') ||
         engineLower.includes('together') ||
         engineLower.includes('deepseek') ||
         engineLower === 'api'; // Generic API engine
}

/**
 * Get formatted context length for display
 */
export function formatContextLength(contextLength: number | null | undefined, engine?: string): string {
  if (!contextLength && engine && isApiBasedModel(engine)) {
    // For API models without context length, use our estimation
    const apiInfo = getApiModelContextLength('', engine);
    if (apiInfo.isDynamic) {
      return `~${apiInfo.contextLength.toLocaleString()}`;
    }
    return apiInfo.contextLength.toLocaleString();
  }
  
  if (!contextLength) {
    return 'Dynamic';
  }
  
  return contextLength.toLocaleString();
}