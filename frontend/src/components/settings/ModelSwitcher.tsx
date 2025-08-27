/**
 * Model switching component with branch-specific model selection
 */

"use client";

import React from 'react';
import { Bot, ChevronDown, RefreshCw, Check, AlertTriangle, Search, X } from 'lucide-react';
import { useChatStore } from '@/lib/store';
import apiClient from '@/lib/api';

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
    case 'llama4': return '🦙';
    case 'qwen3':
    case 'qwen2_5': return '🐧';
    case 'gemma3': return '💎';
    case 'phi3':
    case 'phi4': return '🔷';
    case 'deepseek_r1': return '🌊';
    case 'gpt_oss': return '🔬';
    default: return '🤖';
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
  const [error, setError] = React.useState<string | null>(null);
  const [isInitialized, setIsInitialized] = React.useState(false);
  const [searchTerm, setSearchTerm] = React.useState('');
  const dropdownRef = React.useRef<HTMLDivElement>(null);
  const searchInputRef = React.useRef<HTMLInputElement>(null);

  // Load current model and available configs on mount
  React.useEffect(() => {
    const loadData = async () => {
      try {
        // Load current model and available configs in parallel
        const [modelResponse, configsResponse] = await Promise.all([
          apiClient.getModels(),
          apiClient.getConfigs(),
        ]);

        // Set current model
        if (modelResponse.success && modelResponse.data?.data?.[0]) {
          const model = modelResponse.data.data[0];
          setCurrentModel(model.id);
        }

        // Set available configs
        if (configsResponse.success && configsResponse.data?.configs) {
          setAvailableConfigs(configsResponse.data.configs);
          console.log(`📋 Loaded ${configsResponse.data.configs.length} inference configurations`);
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
    return availableConfigs.filter(config => 
      config.display_name.toLowerCase().includes(term) ||
      config.model_name.toLowerCase().includes(term) ||
      config.filename.toLowerCase().includes(term) ||
      config.engine.toLowerCase().includes(term) ||
      config.model_family.toLowerCase().includes(term)
    );
  }, [searchTerm, availableConfigs]);

  // Group filtered configs by model family
  const groupedFilteredConfigs = React.useMemo(() => {
    return filteredConfigs.reduce((acc, config) => {
      if (!acc[config.model_family]) {
        acc[config.model_family] = [];
      }
      acc[config.model_family].push(config);
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

    try {
      // Use the command API to switch models using config path
      console.log(`🔄 Attempting to switch model using config: ${configPath}`);
      const response = await apiClient.executeCommand('swap', [configPath]);
      
      console.log('🔄 Model switch response:', response);
      
      if (response.success) {
        setCurrentModel(configPath);
        setIsDropdownOpen(false);
        setSearchTerm('');
        console.log(`✅ Successfully switched to config: ${configPath}`);
        
        // Show success message temporarily
        setError(null);
      } else {
        throw new Error(response.message || 'Failed to switch model');
      }
    } catch (err) {
      console.error('❌ Model switch error:', err);
      setError(err instanceof Error ? err.message : 'Failed to switch model');
    } finally {
      setIsLoading(false);
    }
  };

  const getCurrentModelInfo = () => {
    if (!isInitialized || !currentModel) {
      return {
        displayName: 'Loading...',
        description: 'Loading model information',
        engine: 'UNKNOWN',
        contextLength: 0,
        modelFamily: 'unknown',
      };
    }

    // Find matching config
    const matchingConfig = availableConfigs.find(config => 
      config.config_path === currentModel || 
      config.relative_path === currentModel ||
      config.model_name === currentModel
    );

    if (matchingConfig) {
      return {
        displayName: matchingConfig.display_name,
        description: `${matchingConfig.model_name} (${matchingConfig.filename})`,
        engine: matchingConfig.engine,
        contextLength: matchingConfig.context_length,
        modelFamily: matchingConfig.model_family,
      };
    }

    // Fallback for unknown models
    return {
      displayName: currentModel.split('/').pop() || currentModel,
      description: 'Loaded from configuration',
      engine: 'LLAMACPP', // Default based on the config we're testing with
      contextLength: 16384, // Based on Gemma config
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
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2">
                  <span className="text-lg">{getFamilyIcon(currentModelInfo.modelFamily)}</span>
                  <div className="text-left">
                    <div className="font-medium text-sm text-foreground">
                      {currentModelInfo.displayName}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {currentModelInfo.description}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`px-2 py-1 rounded text-xs font-medium ${getEngineColor(currentModelInfo.engine)}`}>
                    {currentModelInfo.engine}
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {isLoading && <RefreshCw size={14} className="animate-spin" />}
                <ChevronDown size={16} className={`transition-transform ${isDropdownOpen ? 'rotate-180' : ''}`} />
              </div>
            </button>

            {/* Dropdown */}
            {isDropdownOpen && (
              <div className="absolute top-full left-0 right-0 mt-1 bg-popover border rounded-lg shadow-lg z-50 max-h-96 overflow-hidden">
                {/* Search input */}
                <div className="sticky top-0 bg-popover border-b p-3">
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
                      <div className="px-2 py-1 text-xs font-semibold text-muted-foreground uppercase tracking-wide">
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
                                  Context: {config.context_length.toLocaleString()} tokens • {config.filename}
                                </div>
                              </div>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className={`px-2 py-1 rounded text-xs font-medium ${getEngineColor(config.engine)}`}>
                                {config.engine}
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
              {currentModelInfo.contextLength.toLocaleString()}
            </div>
          </div>
          <div className="text-center p-2 bg-muted rounded">
            <div className="text-xs text-muted-foreground">Engine</div>
            <div className="font-mono text-sm text-foreground">
              {currentModelInfo.engine}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}