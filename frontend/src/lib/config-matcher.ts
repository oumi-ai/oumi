/**
 * Configuration Matcher - Analyzes system capabilities and recommends the best configurations
 */

export interface SystemCapabilities {
  platform: string;         // darwin, win32, linux
  architecture: string;     // x64, arm64
  totalRAM: number;         // GB
  cudaAvailable: boolean;
  cudaDevices: Array<{
    vram: number;           // GB
  }>;
}

export interface ConfigOption {
  id: string;
  engine: string;
  model_family: string;
  size_category: string;
  display_name: string;
  recommended?: boolean;
}

export interface ConfigRecommendation {
  goodMatch: boolean;
  reason: string;
  score: number; // 0-100
  warnings?: string[];
}

export class ConfigMatcher {
  /**
   * Evaluate how well a configuration matches the current system
   */
  public static evaluateConfig(
    config: ConfigOption,
    system: SystemCapabilities
  ): ConfigRecommendation {
    let score = 50; // Base score
    let warnings: string[] = [];
    let reason = '';

    // Platform-specific engine preferences
    const engineScore = this.evaluateEngine(config.engine, system);
    score += engineScore.score;
    if (engineScore.reason) {
      reason += engineScore.reason;
    }
    warnings.push(...engineScore.warnings);

    // Model size vs available memory
    const memoryScore = this.evaluateModelSize(config, system);
    score += memoryScore.score;
    if (memoryScore.reason) {
      reason = reason ? `${reason}; ${memoryScore.reason}` : memoryScore.reason;
    }
    warnings.push(...memoryScore.warnings);

    // Special cases and bonuses
    const specialScore = this.evaluateSpecialCases(config, system);
    score += specialScore.score;
    if (specialScore.reason) {
      reason = reason ? `${reason}; ${specialScore.reason}` : specialScore.reason;
    }
    warnings.push(...specialScore.warnings);

    // Clamp score between 0-100
    score = Math.max(0, Math.min(100, score));

    return {
      goodMatch: score >= 70,
      reason: reason || this.getDefaultReason(config, system),
      score,
      warnings: warnings.filter(Boolean)
    };
  }

  /**
   * Evaluate engine compatibility with system
   */
  private static evaluateEngine(
    engine: string,
    system: SystemCapabilities
  ): { score: number; reason: string; warnings: string[] } {
    const engineLower = engine.toLowerCase();
    const warnings: string[] = [];

    // macOS preferences
    if (system.platform === 'darwin') {
      if (engineLower === 'llamacpp') {
        return {
          score: 25,
          reason: 'LlamaCPP optimized for macOS',
          warnings
        };
      }
      if (engineLower === 'native') {
        return {
          score: 15,
          reason: 'Native engine works well on macOS',
          warnings
        };
      }
      if (engineLower === 'vllm' && system.cudaDevices.length === 0) {
        warnings.push('VLLM may not perform optimally without CUDA on macOS');
        return {
          score: -10,
          reason: '',
          warnings
        };
      }
    }

    // Windows/Linux with CUDA
    if (system.platform !== 'darwin' && system.cudaAvailable && system.cudaDevices.length > 0) {
      if (engineLower === 'vllm') {
        return {
          score: 25,
          reason: 'VLLM optimized for CUDA performance',
          warnings
        };
      }
      if (engineLower === 'native') {
        return {
          score: 10,
          reason: 'Native engine supports CUDA',
          warnings
        };
      }
    }

    // CPU-only systems
    if (!system.cudaAvailable || system.cudaDevices.length === 0) {
      if (engineLower === 'native') {
        return {
          score: 15,
          reason: 'Native engine efficient for CPU inference',
          warnings
        };
      }
      if (engineLower === 'llamacpp') {
        return {
          score: 10,
          reason: 'LlamaCPP optimized for CPU',
          warnings
        };
      }
      if (engineLower === 'vllm') {
        warnings.push('VLLM designed for GPU acceleration, may be slower on CPU');
        return {
          score: -15,
          reason: '',
          warnings
        };
      }
    }

    return { score: 0, reason: '', warnings };
  }

  /**
   * Evaluate model size vs available memory
   */
  private static evaluateModelSize(
    config: ConfigOption,
    system: SystemCapabilities
  ): { score: number; reason: string; warnings: string[] } {
    const warnings: string[] = [];
    const totalVRAM = system.cudaDevices.reduce((sum, device) => sum + device.vram, 0);
    const effectiveMemory = Math.max(totalVRAM, system.totalRAM * 0.4); // Use 40% of RAM if no GPU

    const sizeCategory = config.size_category.toLowerCase();
    
    // Estimate memory requirements (rough approximations)
    let requiredMemory = 2; // Default for small models
    if (sizeCategory === 'large') {
      requiredMemory = 40; // 70B models
    } else if (sizeCategory === 'medium') {
      requiredMemory = 16; // 7B-30B models
    } else if (sizeCategory === 'small') {
      requiredMemory = 4; // 1B-8B models
    }

    // Check if GGUF quantization might help
    const isQuantized = config.display_name.toLowerCase().includes('gguf') || 
                       config.display_name.toLowerCase().includes('q4') ||
                       config.display_name.toLowerCase().includes('q8');
    
    if (isQuantized) {
      requiredMemory *= 0.5; // Quantized models use roughly half the memory
    }

    const memoryRatio = effectiveMemory / requiredMemory;

    if (memoryRatio >= 2) {
      return {
        score: 20,
        reason: 'Plenty of memory available',
        warnings
      };
    } else if (memoryRatio >= 1.5) {
      return {
        score: 10,
        reason: 'Good memory fit',
        warnings
      };
    } else if (memoryRatio >= 1) {
      return {
        score: 0,
        reason: 'Adequate memory',
        warnings: ['Model may use most available memory']
      };
    } else if (memoryRatio >= 0.8) {
      warnings.push('Model may exceed available memory, consider quantized version');
      return {
        score: -10,
        reason: '',
        warnings
      };
    } else {
      warnings.push('Insufficient memory - model likely to fail or be very slow');
      return {
        score: -25,
        reason: '',
        warnings
      };
    }
  }

  /**
   * Evaluate special cases and combinations
   */
  private static evaluateSpecialCases(
    config: ConfigOption,
    system: SystemCapabilities
  ): { score: number; reason: string; warnings: string[] } {
    const warnings: string[] = [];
    const displayName = config.display_name.toLowerCase();

    // GGUF + macOS bonus
    if (system.platform === 'darwin' && displayName.includes('gguf') && displayName.includes('macos')) {
      return {
        score: 15,
        reason: 'macOS-optimized GGUF model',
        warnings
      };
    }

    // GGUF quantization for limited memory
    const totalVRAM = system.cudaDevices.reduce((sum, device) => sum + device.vram, 0);
    const effectiveMemory = Math.max(totalVRAM, system.totalRAM * 0.4);
    
    if (effectiveMemory < 16 && displayName.includes('gguf')) {
      return {
        score: 10,
        reason: 'Quantized model good for limited memory',
        warnings
      };
    }

    // High VRAM systems prefer non-quantized
    if (totalVRAM > 20 && !displayName.includes('gguf') && config.engine.toLowerCase() === 'vllm') {
      return {
        score: 10,
        reason: 'High-performance setup for powerful GPU',
        warnings
      };
    }

    // API models always score well (they don't use local resources)
    if (config.model_family === 'openai' || config.model_family === 'anthropic' || config.model_family === 'gemini') {
      return {
        score: 20,
        reason: 'Cloud API - no local resource requirements',
        warnings
      };
    }

    return { score: 0, reason: '', warnings };
  }

  /**
   * Get default reason when no specific reasons apply
   */
  private static getDefaultReason(config: ConfigOption, system: SystemCapabilities): string {
    const engineName = config.engine.toLowerCase();
    
    if (engineName === 'native') {
      return 'General-purpose engine';
    } else if (engineName === 'vllm') {
      return system.cudaAvailable ? 'High-performance GPU engine' : 'GPU-optimized engine';
    } else if (engineName === 'llamacpp') {
      return 'CPU-optimized engine';
    }
    
    return 'Compatible configuration';
  }

  /**
   * Get system capabilities summary for display
   */
  public static getSystemSummary(system: SystemCapabilities): string {
    const parts = [];
    
    // Platform
    const platformName = system.platform === 'darwin' ? 'macOS' : 
                        system.platform === 'win32' ? 'Windows' : 'Linux';
    parts.push(`${platformName} ${system.architecture}`);
    
    // RAM
    parts.push(`${system.totalRAM}GB RAM`);
    
    // CUDA
    if (system.cudaAvailable && system.cudaDevices.length > 0) {
      const totalVRAM = system.cudaDevices.reduce((sum, device) => sum + device.vram, 0);
      parts.push(`${system.cudaDevices.length} CUDA GPU${system.cudaDevices.length > 1 ? 's' : ''} (${totalVRAM.toFixed(1)}GB VRAM)`);
    } else {
      parts.push('CPU only');
    }
    
    return parts.join(' • ');
  }

  /**
   * Sort configurations by recommendation score
   */
  public static sortConfigsByRecommendation(
    configs: ConfigOption[],
    system: SystemCapabilities
  ): Array<ConfigOption & { recommendation?: ConfigRecommendation }> {
    return configs
      .map(config => ({
        ...config,
        recommendation: this.evaluateConfig(config, system)
      }))
      .sort((a, b) => (b.recommendation?.score || 0) - (a.recommendation?.score || 0));
  }
}