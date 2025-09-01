/**
 * HuggingFace service for fetching model metadata with user authentication
 */

interface ModelMetadata {
  parameterCount: number;
  tags: string[];
  isSpecialist: boolean;
  lastUpdated: string;
  error?: string;
}

interface HuggingFaceCredentials {
  username?: string;
  token?: string;
}

export class HuggingFaceService {
  private static cache = new Map<string, { data: ModelMetadata; expires: number }>();
  private static CACHE_DURATION = 24 * 60 * 60 * 1000; // 24 hours

  /**
   * Fetch model metadata with optional authentication
   */
  /**
   * Check if a model is a HuggingFace model (not an API-only model)
   */
  private static isHuggingFaceModel(modelName: string): boolean {
    // Skip API-only models that don't exist on HuggingFace
    const apiOnlyPrefixes = [
      'claude-', 'gpt-', 'chatgpt-', 'o1-', 'o3-',
      'gemini-', 'vertex-', 'meta/llama-', 'Unknown Model'
    ];
    
    return !apiOnlyPrefixes.some(prefix => 
      modelName.toLowerCase().startsWith(prefix.toLowerCase())
    );
  }

  static async fetchModelMetadata(
    modelName: string,
    credentials?: HuggingFaceCredentials
  ): Promise<ModelMetadata | null> {
    if (!modelName) return null;
    
    // Skip API-only models
    if (!this.isHuggingFaceModel(modelName)) {
      return null;
    }

    // Check cache first
    const cached = this.cache.get(modelName);
    if (cached && Date.now() < cached.expires) {
      return cached.data;
    }

    try {
      const response = await fetch('/api/huggingface/metadata', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          modelName,
          username: credentials?.username,
          token: credentials?.token,
        }),
      });

      if (!response.ok) {
        console.warn(`HF metadata API failed with status ${response.status}`);
        return null;
      }

      const result = await response.json();
      
      let metadata: ModelMetadata;
      if (result.success) {
        metadata = result.metadata;
      } else {
        // Use fallback data if API call failed
        metadata = result.fallback;
      }

      // Cache the result
      this.cache.set(modelName, {
        data: metadata,
        expires: Date.now() + this.CACHE_DURATION,
      });

      return metadata;
    } catch (error) {
      // Only log actual network/API errors, not expected failures for API-only models
      if (error instanceof TypeError && error.message === 'Failed to fetch') {
        // Likely network issue or model doesn't exist - don't spam console
        return null;
      }
      console.warn(`Failed to fetch HF metadata for ${modelName}:`, error);
      return null;
    }
  }

  /**
   * Enhanced config enhancement with real-time HF metadata
   */
  static async enhanceConfigsWithMetadata(
    configs: any[],
    credentials?: HuggingFaceCredentials
  ): Promise<any[]> {
    // Get unique model names to minimize API calls, filtering out API-only models
    const uniqueModels = new Set<string>();
    configs.forEach(config => {
      if (config.model_name && this.isHuggingFaceModel(config.model_name)) {
        uniqueModels.add(config.model_name);
      }
    });

    // Fetch metadata for all unique models with rate limiting
    const metadataMap = new Map<string, ModelMetadata>();
    const models = Array.from(uniqueModels);
    
    // Process models in batches to avoid overwhelming the API
    const BATCH_SIZE = 5;
    for (let i = 0; i < models.length; i += BATCH_SIZE) {
      const batch = models.slice(i, i + BATCH_SIZE);
      
      const batchPromises = batch.map(async (modelName, index) => {
        // Add a small delay to avoid rate limiting
        await new Promise(resolve => setTimeout(resolve, index * 200));
        const metadata = await this.fetchModelMetadata(modelName, credentials);
        if (metadata) {
          metadataMap.set(modelName, metadata);
        }
      });
      
      await Promise.all(batchPromises);
    }

    // Enhance configs with fetched metadata
    return configs.map(config => {
      const metadata = metadataMap.get(config.model_name);
      
      // Update config with metadata if available
      if (metadata && metadata.parameterCount > 0) {
        // Use real parameter count for accurate size categorization
        const parameterCount = metadata.parameterCount;
        let sizeCategory = 'medium'; // default
        
        if (parameterCount <= 3) sizeCategory = 'small';
        else if (parameterCount <= 30) sizeCategory = 'medium';
        else sizeCategory = 'large';
        
        return {
          ...config,
          size_category: sizeCategory,
          parameter_count: parameterCount,
          is_specialist: metadata.isSpecialist,
          hf_tags: metadata.tags,
          hf_enhanced: true, // Flag to indicate this was enhanced with fresh HF data
        };
      }

      // Return original config if no metadata available
      return config;
    });
  }

  /**
   * Clear the metadata cache
   */
  static clearCache(): void {
    this.cache.clear();
  }
}