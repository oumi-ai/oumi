/**
 * Model switching component with branch-specific model selection
 */

"use client";

import React from 'react';
import { Bot, ChevronDown, RefreshCw, Check, AlertTriangle, Search, X, Zap, Brain, Cpu, Gem, Waves, FlaskConical, Building2 } from 'lucide-react';
import { useChatStore } from '@/lib/store';
import apiClient from '@/lib/unified-api';
import { ModelConfigMetadata } from '@/lib/types';
import { formatContextLength } from '@/lib/api-model-context';

interface ConfigOption {
  id: string;
  config_path: string;
  relative_path: string;
  display_name: string;
  model_name: string;
  engine: string;
  context_length: number;
  model_family: string;
  filename: string;
}

// Engine display strictly reflects the engine reported by the active config.
// No heuristics here — we rely on the backend/config source of truth.

const getEngineAbbreviation = (engine: string) => {
  switch (engine.toUpperCase()) {
    case 'LLAMACPP': return 'LLAMA';
    case 'NATIVE': return 'NATVE';
    case 'VLLM': return 'VLLM';
    case 'OPENAI': return 'OPENAI';
    case 'ANTHROPIC': return 'ANTHRO';
    default: return engine.slice(0, 5).toUpperCase();
  }
};

const getEngineColor = (engine: string) => {
  switch (engine.toUpperCase()) {
    case 'NATIVE': return 'bg-blue-100 text-blue-800';
    case 'VLLM': return 'bg-green-100 text-green-800';
    case 'LLAMACPP': return 'bg-purple-100 text-purple-800';
    case 'OPENAI': return 'bg-orange-100 text-orange-800';
    case 'ANTHROPIC': return 'bg-red-100 text-red-800';
    default: return 'bg-gray-100 text-gray-800';
  }
};

const getFamilyIcon = (family: string) => {
  switch (family.toLowerCase()) {
    case 'llama3_1':
    case 'llama3_2': 
    case 'llama3_3':
    case 'llama4': 
      return <Building2 size={16} className="text-orange-500" />; // Meta
    case 'qwen3':
    case 'qwen2_5': 
      return <Zap size={16} className="text-red-500" />; // Alibaba/Qwen
    case 'gemma3': 
      return <Gem size={16} className="text-blue-500" />; // Google
    case 'phi3':
    case 'phi4': 
      return <Brain size={16} className="text-green-500" />; // Microsoft
    case 'deepseek_r1': 
      return <Waves size={16} className="text-cyan-500" />; // DeepSeek
    case 'gpt_oss': 
      return <FlaskConical size={16} className="text-purple-500" />; // Research/OSS
    default: 
      return <Cpu size={16} className="text-gray-500" />; // Generic
  }
};

interface ModelSwitcherProps {
  className?: string;
}

export default function ModelSwitcher({ className = '' }: ModelSwitcherProps) {
  const { currentBranchId } = useChatStore();
  const [currentModel, setCurrentModel] = React.useState<string>('');
  const [availableConfigs, setAvailableConfigs] = React.useState<ConfigOption[]>([]);
  const [isDropdownOpen, setIsDropdownOpen] = React.useState(false);
  const [isLoading, setIsLoading] = React.useState(false);
  const [loadingMessage, setLoadingMessage] = React.useState<string>('');
  const [error, setError] = React.useState<string | null>(null);
  const [isInitialized, setIsInitialized] = React.useState(false);
  const [searchTerm, setSearchTerm] = React.useState('');
  const [currentModelConfigMetadata, setCurrentModelConfigMetadata] = React.useState<ModelConfigMetadata | null>(null);
  const dropdownRef = React.useRef<HTMLDivElement>(null);
  const searchInputRef = React.useRef<HTMLInputElement>(null);

  // Load current model and available configs on mount
  React.useEffect(() => {
    const loadData = async () => {
      try {
        // Load available configs first
        const configsResponse = await apiClient.getConfigs();
        if (configsResponse.success && configsResponse.data?.configs) {
          const sanitized = configsResponse.data.configs.map((c: any) => ({
            id: c.id ?? c.relative_path ?? c.config_path ?? c.filename ?? '',
            config_path: c.config_path ?? '',
            relative_path: c.relative_path ?? '',
            display_name: typeof c.display_name === 'string' && c.display_name.length > 0
              ? c.display_name
              : (c.model_name || c.filename || c.relative_path || 'Unknown'),
            model_name: c.model_name ?? '',
            engine: c.engine ?? 'UNKNOWN',
            context_length: typeof c.context_length === 'number' ? c.context_length : 0,
            model_family: c.model_family ?? 'unknown',
            filename: c.filename ?? '',
          }));
          setAvailableConfigs(sanitized);
          console.log(`📋 Loaded ${configsResponse.data.configs.length} inference configurations`);
        }

        // Then load current model with enhanced config metadata
        const modelResponse = await apiClient.getModels();
        if (modelResponse.success && modelResponse.data?.data?.[0]) {
          const model = modelResponse.data.data[0];
          setCurrentModel(model.id);
          
          // CRITICAL FIX: Extract and cache config metadata from server
          if (model.config_metadata) {
            setCurrentModelConfigMetadata(model.config_metadata);
            console.log(`🎯 Current model with metadata: ${model.id}`, model.config_metadata);
          } else {
            setCurrentModelConfigMetadata(null);
            console.log(`🎯 Current model (no metadata): ${model.id}`);
          }
        }

        setIsInitialized(true);
      } catch (error) {
        console.error('Failed to load model data:', error);
        setError('Failed to load model information');
        setIsInitialized(true);
      }
    };

    loadData();
  }, []);

  // Refresh model info when currentModel changes
  React.useEffect(() => {
    if (currentModel && availableConfigs.length > 0) {
      console.log(`🔄 Updating model info for: ${currentModel}`);
      // Force re-render by updating the key or triggering state update
      setIsInitialized(true);
    }
  }, [currentModel, availableConfigs]);

  // Close dropdown when clicking outside
  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsDropdownOpen(false);
        setSearchTerm('');
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Focus search input when dropdown opens
  React.useEffect(() => {
    if (isDropdownOpen && searchInputRef.current) {
      searchInputRef.current.focus();
    }
  }, [isDropdownOpen]);

  // Filter configs based on search term
  const filteredConfigs = React.useMemo(() => {
    if (!searchTerm) return availableConfigs;

    const term = searchTerm.toLowerCase();
    const safe = (v: unknown) => (typeof v === 'string' ? v.toLowerCase() : '');
    return availableConfigs.filter(config =>
      safe(config.display_name).includes(term) ||
      safe(config.model_name).includes(term) ||
      safe(config.filename).includes(term) ||
      safe(config.engine).includes(term) ||
      safe(config.model_family).includes(term)
    );
  }, [searchTerm, availableConfigs]);

  // Group filtered configs by model family
  const groupedFilteredConfigs = React.useMemo(() => {
    return filteredConfigs.reduce((acc, config) => {
      const family = (config.model_family && typeof config.model_family === 'string')
        ? config.model_family
        : 'unknown';
      if (!acc[family]) {
        acc[family] = [];
      }
      acc[family].push(config);
      return acc;
    }, {} as Record<string, ConfigOption[]>);
  }, [filteredConfigs]);

  const handleModelSwitch = async (configPath: string) => {
    if (configPath === currentModel) {
      setIsDropdownOpen(false);
      return;
    }

    setIsLoading(true);
    setError(null);
    setIsDropdownOpen(false);
    
    // Show descriptive loading messages
    const selectedConfig = availableConfigs.find(config => config.config_path === configPath);
    if (selectedConfig) {
      setLoadingMessage(`Switching to ${selectedConfig.display_name}...`);
    } else {
      setLoadingMessage('Loading model from path...');
    }
    
    // Add a short delay to show loading message, then update for potential downloading
    setTimeout(() => {
      if (isLoading) {
        setLoadingMessage('Downloading model if needed... This may take several minutes.');
      }
    }, 2000);

    try {
      // Clear model from memory before switching to ensure clean state
      console.log('🧹 Clearing model before model switch...');
      const clearResult = await apiClient.clearModel();
      if (clearResult.success) {
        console.log('✅ Model cleared successfully before model switch');
      } else {
        console.warn('⚠️ Model clear failed, continuing with model switch:', clearResult.message);
      }
      
      // Use the command API to switch models using config path
      console.log(`🔄 Attempting to switch model using config: ${configPath}`);
      const response = await apiClient.executeCommand('swap', [configPath]);
      
      console.log('🔄 Model switch response:', response);
      
      if (response.success) {
        try { const { showToast } = await import('@/lib/toastBus'); showToast({ message: '✅ Model switched successfully', variant: 'success' }); } catch {}
        // CRITICAL FIX: Reload model information from server after successful swap
        try {
          const modelResponse = await apiClient.getModels();
          if (modelResponse.success && modelResponse.data?.data?.[0]) {
            const model = modelResponse.data.data[0];
            setCurrentModel(model.id);
            
            // Extract and cache updated config metadata after swap
            if (model.config_metadata) {
              setCurrentModelConfigMetadata(model.config_metadata);
              console.log(`🔄 Updated model with metadata: ${model.id}`, model.config_metadata);
            } else {
              setCurrentModelConfigMetadata(null);
              console.log(`🔄 Updated model (no metadata): ${model.id}`);
            }
          } else {
            // Fallback to config path if server response fails
            setCurrentModel(configPath);
            setCurrentModelConfigMetadata(null);
            console.warn('⚠️ Could not refresh model info from server, using config path');
          }
        } catch (refreshError) {
          console.error('❌ Error refreshing model info:', refreshError);
          // Fallback to config path if refresh fails
          setCurrentModel(configPath);
          setCurrentModelConfigMetadata(null);
        }
        
        setIsDropdownOpen(false);
        setSearchTerm('');
        console.log(`✅ Successfully switched to config: ${configPath}`);
        
        // Show success message temporarily
        setError(null);
      } else {
        const msg = response.message || 'Failed to switch model';
        try { const { showToast } = await import('@/lib/toastBus'); showToast({ message: `❌ ${msg}`, variant: 'error' }); } catch {}
        throw new Error(msg);
      }
    } catch (err) {
      console.error('❌ Model switch error:', err);
      setError(err instanceof Error ? err.message : 'Failed to switch model');
      try { const { showToast } = await import('@/lib/toastBus'); showToast({ message: '❌ Model switch failed', variant: 'error' }); } catch {}
    } finally {
      setIsLoading(false);
      setLoadingMessage('');
    }
  };

  const getCurrentModelInfo = () => {
    // Debug logging disabled to reduce console clutter
    // console.log(`🔍 Getting model info - isInitialized: ${isInitialized}, currentModel: ${currentModel}, configsCount: ${availableConfigs.length}`);
    
    if (!isInitialized) {
      return {
        displayName: 'Loading...',
        description: 'Loading model information',
        engine: 'UNKNOWN',
        contextLength: 0,
        modelFamily: 'unknown',
      };
    }

    if (!currentModel) {
      return {
        displayName: 'No Model Selected',
        description: 'No model currently loaded',
        engine: 'NONE',
        contextLength: 0,
        modelFamily: 'unknown',
      };
    }

    // CRITICAL FIX: Use cached config metadata from server if available
    if (currentModelConfigMetadata) {
      console.log(`✅ Using server's active config metadata:`, currentModelConfigMetadata);
      return {
        displayName: currentModelConfigMetadata.display_name,
        description: currentModelConfigMetadata.description,
        engine: (currentModelConfigMetadata.engine || 'UNKNOWN').toUpperCase(),
        contextLength: currentModelConfigMetadata.context_length,
        modelFamily: currentModelConfigMetadata.model_family,
      };
    }

    // Fallback: try to find matching config from scanned configs
    let matchingConfig = availableConfigs.find(config => 
      config.config_path === currentModel || 
      config.relative_path === currentModel ||
      config.model_name === currentModel ||
      config.id === currentModel
    );

    // If no exact match, try partial matching on model name
    if (!matchingConfig && currentModel.includes('/')) {
      const modelName = currentModel.split('/').pop() || currentModel;
      matchingConfig = availableConfigs.find(config => 
        (typeof config.model_name === 'string' && config.model_name.includes(modelName)) ||
        (typeof config.display_name === 'string' && config.display_name.toLowerCase().includes(modelName.toLowerCase()))
      );
    }

    if (matchingConfig) {
      console.log(`✅ Found matching config:`, matchingConfig);
      return {
        displayName: matchingConfig.display_name,
        description: `${matchingConfig.model_name} (${matchingConfig.filename})`,
        engine: (matchingConfig.engine || 'UNKNOWN').toUpperCase(),
        contextLength: matchingConfig.context_length,
        modelFamily: matchingConfig.model_family,
      };
    }

    // Final fallback for unknown models
    console.log(`⚠️ No matching config found for model: ${currentModel}`);
    const fallbackName = currentModel.split('/').pop() || currentModel;
    return {
      displayName: fallbackName,
      description: 'Custom model (not in config list)',
      engine: 'NATIVE', // Conservative default
      contextLength: 8192, // Conservative default
      modelFamily: 'unknown',
    };
  };

  const currentModelInfo = getCurrentModelInfo();

  return (
    <div className={`bg-card rounded-lg p-4 border space-y-4 ${className}`}>
      <div className="flex items-center gap-2">
        <Bot size={16} />
        <span className="text-sm font-semibold text-foreground">Model Configuration</span>
        <div className="ml-auto text-xs text-muted-foreground">
          Branch: {currentBranchId}
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 p-3 bg-destructive/10 rounded-lg border border-destructive/20">
          <AlertTriangle size={14} className="text-destructive" />
          <span className="text-sm text-destructive">{error}</span>
        </div>
      )}

      <div className="space-y-3">
        {/* Current Model Display */}
        <div className="space-y-2">
          <label className="text-xs font-medium text-muted-foreground">Current Model</label>
          <div className="relative" ref={dropdownRef}>
            <button
              onClick={() => setIsDropdownOpen(!isDropdownOpen)}
              disabled={isLoading}
              className="w-full flex items-center justify-between p-3 bg-muted hover:bg-muted/80 rounded-lg transition-colors disabled:opacity-50"
            >
              <div className="flex items-center gap-3 min-w-0 flex-1">
                <div className="flex items-center gap-2 min-w-0 flex-1">
                  {getFamilyIcon(currentModelInfo.modelFamily)}
                  <div className="text-left min-w-0 flex-1">
                    <div className="font-medium text-sm text-foreground truncate">
                      {isLoading ? 'Switching Model...' : currentModelInfo.displayName}
                    </div>
                    <div className="text-xs text-muted-foreground truncate">
                      {isLoading ? loadingMessage : currentModelInfo.description}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2 flex-shrink-0">
                  <span className={`px-2 py-1 rounded text-xs font-medium ${getEngineColor(currentModelInfo.engine)}`}>
                    {getEngineAbbreviation(currentModelInfo.engine)}
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {isLoading && <RefreshCw size={14} className="animate-spin text-primary" />}
                <ChevronDown size={16} className={`transition-transform ${isDropdownOpen && !isLoading ? 'rotate-180' : ''} ${isLoading ? 'opacity-50' : ''}`} />
              </div>
            </button>

            {/* Dropdown */}
            {isDropdownOpen && (
              <div className="absolute top-full left-0 right-0 mt-1 bg-card border border-border rounded-lg shadow-xl z-[60] max-h-96 overflow-hidden backdrop-blur-sm">
                {/* Current model full name header */}
                <div className="sticky top-0 bg-card border-b border-border p-3">
                  <div className="flex items-center justify-between">
                    <div className="text-sm font-medium text-foreground">
                      {currentModelInfo.displayName}
                    </div>
                    <span className={`ml-2 px-2 py-0.5 rounded text-xs font-medium ${getEngineColor(currentModelInfo.engine)}`}>
                      {getEngineAbbreviation(currentModelInfo.engine)}
                    </span>
                  </div>
                </div>
                {/* Search input */}
                <div className="sticky top-0 bg-card border-b border-border p-3">
                  <div className="relative">
                    <Search size={14} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground" />
                    <input
                      ref={searchInputRef}
                      type="text"
                      placeholder="Search models..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="w-full pl-9 pr-8 py-2 bg-muted border border-border rounded-md text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                      onKeyDown={(e) => {
                        if (e.key === 'Escape') {
                          setSearchTerm('');
                          setIsDropdownOpen(false);
                        } else if (e.key === 'Enter') {
                          if (filteredConfigs.length === 1) {
                            // If there's only one result, switch to it on Enter
                            handleModelSwitch(filteredConfigs[0].config_path);
                          } else if (filteredConfigs.length === 0 && searchTerm.trim()) {
                            // If no results but we have a search term, use it as custom model
                            handleModelSwitch(searchTerm.trim());
                          }
                        }
                      }}
                    />
                    {searchTerm && (
                      <button
                        onClick={() => setSearchTerm('')}
                        className="absolute right-2 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground"
                      >
                        <X size={14} />
                      </button>
                    )}
                  </div>
                </div>

                {/* Results */}
                <div className="max-h-80 overflow-y-auto p-2">
                  {filteredConfigs.length === 0 && searchTerm ? (
                    <div className="space-y-3">
                      <div className="p-4 text-center text-muted-foreground">
                        <Search size={24} className="mx-auto mb-2 opacity-50" />
                        <div className="text-sm">No models found matching "{searchTerm}"</div>
                      </div>
                      {/* Custom model option */}
                      <div className="border-t pt-3">
                        <button
                          onClick={() => handleModelSwitch(searchTerm)}
                          className="w-full flex items-center justify-between p-3 hover:bg-muted rounded text-left transition-colors"
                        >
                          <div className="flex items-center gap-3 flex-1">
                            <div className="flex-1">
                              <div className="font-medium text-sm text-foreground">
                                Use "{searchTerm}" as custom model
                              </div>
                              <div className="text-xs text-muted-foreground">
                                Load model from HuggingFace Hub or local path
                              </div>
                            </div>
                          </div>
                          <div className="text-xs text-primary">
                            Enter ↵
                          </div>
                        </button>
                      </div>
                    </div>
                  ) : filteredConfigs.length === 0 ? (
                    <div className="p-4 text-center text-muted-foreground">
                      <Bot size={24} className="mx-auto mb-2 opacity-50" />
                      <div className="text-sm">Start typing to search configs</div>
                    </div>
                  ) : (
                    Object.entries(groupedFilteredConfigs).map(([family, configs]) => (
                    <div key={family} className="mb-4 last:mb-0">
                      <div className="px-2 py-1 text-xs font-semibold text-muted-foreground uppercase tracking-wide flex items-center gap-2">
                        {getFamilyIcon(family)} {family} Models
                      </div>
                      <div className="space-y-1">
                        {configs.map((config) => (
                          <button
                            key={config.id}
                            onClick={() => handleModelSwitch(config.config_path)}
                            className="w-full flex items-center justify-between p-2 hover:bg-muted rounded text-left transition-colors"
                          >
                            <div className="flex items-center gap-3 flex-1">
                              <div className="flex-1">
                                <div className="font-medium text-sm text-foreground flex items-center gap-2">
                                  {config.display_name}
                                </div>
                                <div className="text-xs text-muted-foreground">
                                  {config.model_name}
                                </div>
                                <div className="text-xs text-muted-foreground mt-1">
                                  Context: {formatContextLength(config.context_length, config.engine)} tokens • {config.filename}
                                </div>
                              </div>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className={`px-2 py-1 rounded text-xs font-medium ${getEngineColor(config.engine)}`}>
                                {getEngineAbbreviation(config.engine)}
                              </span>
                              {config.config_path === currentModel && (
                                <Check size={14} className="text-green-600" />
                              )}
                            </div>
                          </button>
                        ))}
                      </div>
                    </div>
                    ))
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Model Info */}
        <div className="grid grid-cols-2 gap-3 pt-3 border-t">
          <div className="text-center p-2 bg-muted rounded">
            <div className="text-xs text-muted-foreground">Context Length</div>
            <div className="font-mono text-sm text-foreground">
              {formatContextLength(currentModelInfo.contextLength, currentModelInfo.engine)}
            </div>
          </div>
          <div className="text-center p-2 bg-muted rounded">
            <div className="text-xs text-muted-foreground">Engine</div>
            <div className="font-mono text-sm text-foreground">
              {getEngineAbbreviation(currentModelInfo.engine)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
