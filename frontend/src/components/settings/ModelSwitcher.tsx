/**
 * Model switching component with branch-specific model selection
 */

"use client";

import React from 'react';
import { Bot, ChevronDown, RefreshCw, Check, AlertTriangle } from 'lucide-react';
import { useChatStore } from '@/lib/store';
import apiClient from '@/lib/api';

interface ModelOption {
  id: string;
  name: string;
  displayName: string;
  description: string;
  engine: string;
  category: 'small' | 'medium' | 'large' | 'api';
  contextLength: number;
  isRecommended?: boolean;
}

// Predefined model options based on common Oumi configurations
const MODEL_OPTIONS: ModelOption[] = [
  // Small models (< 3B)
  {
    id: 'microsoft/Phi-3.5-mini-instruct',
    name: 'microsoft/Phi-3.5-mini-instruct',
    displayName: 'Phi-3.5 Mini',
    description: 'Lightweight 3.8B model, fast and efficient',
    engine: 'NATIVE',
    category: 'small',
    contextLength: 128000,
    isRecommended: true,
  },
  {
    id: 'HuggingFaceTB/SmolLM2-1.7B-Instruct',
    name: 'HuggingFaceTB/SmolLM2-1.7B-Instruct',
    displayName: 'SmolLM2 1.7B',
    description: 'Tiny but capable 1.7B model',
    engine: 'NATIVE',
    category: 'small',
    contextLength: 8192,
  },
  
  // Medium models (3B - 30B)
  {
    id: 'meta-llama/Llama-3.1-8B-Instruct',
    name: 'meta-llama/Llama-3.1-8B-Instruct',
    displayName: 'Llama 3.1 8B',
    description: 'Balanced performance and efficiency',
    engine: 'VLLM',
    category: 'medium',
    contextLength: 131072,
    isRecommended: true,
  },
  {
    id: 'Qwen/Qwen3-14B-Instruct',
    name: 'Qwen/Qwen3-14B-Instruct',
    displayName: 'Qwen3 14B',
    description: 'Strong reasoning and coding abilities',
    engine: 'VLLM',
    category: 'medium',
    contextLength: 32768,
  },
  
  // Large models (30B+)
  {
    id: 'meta-llama/Llama-3.1-70B-Instruct',
    name: 'meta-llama/Llama-3.1-70B-Instruct',
    displayName: 'Llama 3.1 70B',
    description: 'High-performance large model',
    engine: 'VLLM',
    category: 'large',
    contextLength: 131072,
  },
  {
    id: 'Qwen/Qwen3-72B-Instruct',
    name: 'Qwen/Qwen3-72B-Instruct',
    displayName: 'Qwen3 72B',
    description: 'Top-tier reasoning capabilities',
    engine: 'VLLM',
    category: 'large',
    contextLength: 32768,
  },
  
  // API models
  {
    id: 'gpt-4o',
    name: 'gpt-4o',
    displayName: 'GPT-4o',
    description: 'OpenAI\'s latest multimodal model',
    engine: 'OPENAI',
    category: 'api',
    contextLength: 128000,
  },
  {
    id: 'claude-3.5-sonnet',
    name: 'claude-3.5-sonnet',
    displayName: 'Claude 3.5 Sonnet',
    description: 'Anthropic\'s advanced reasoning model',
    engine: 'ANTHROPIC',
    category: 'api',
    contextLength: 200000,
  },
];

const getCategoryColor = (category: ModelOption['category']) => {
  switch (category) {
    case 'small': return 'bg-blue-100 text-blue-800';
    case 'medium': return 'bg-green-100 text-green-800';
    case 'large': return 'bg-purple-100 text-purple-800';
    case 'api': return 'bg-orange-100 text-orange-800';
    default: return 'bg-gray-100 text-gray-800';
  }
};

const getCategoryIcon = (category: ModelOption['category']) => {
  switch (category) {
    case 'small': return '🚀';
    case 'medium': return '⚖️';
    case 'large': return '🧠';
    case 'api': return '☁️';
    default: return '🤖';
  }
};

interface ModelSwitcherProps {
  className?: string;
}

export default function ModelSwitcher({ className = '' }: ModelSwitcherProps) {
  const { currentBranchId } = useChatStore();
  const [currentModel, setCurrentModel] = React.useState<string>('meta-llama/Llama-3.1-8B-Instruct');
  const [isDropdownOpen, setIsDropdownOpen] = React.useState(false);
  const [isLoading, setIsLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const dropdownRef = React.useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleModelSwitch = async (modelId: string) => {
    if (modelId === currentModel) {
      setIsDropdownOpen(false);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Use the command API to switch models
      const response = await apiClient.executeCommand('swap', [modelId]);
      
      if (response.success) {
        setCurrentModel(modelId);
        setIsDropdownOpen(false);
      } else {
        throw new Error(response.message || 'Failed to switch model');
      }
    } catch (err) {
      console.error('Model switch error:', err);
      setError(err instanceof Error ? err.message : 'Failed to switch model');
    } finally {
      setIsLoading(false);
    }
  };

  const getCurrentModelOption = () => {
    return MODEL_OPTIONS.find(model => model.id === currentModel) || {
      id: currentModel,
      name: currentModel,
      displayName: currentModel.split('/').pop() || currentModel,
      description: 'Custom model',
      engine: 'UNKNOWN',
      category: 'medium' as const,
      contextLength: 4096,
    };
  };

  const currentModelOption = getCurrentModelOption();
  const groupedModels = MODEL_OPTIONS.reduce((acc, model) => {
    if (!acc[model.category]) {
      acc[model.category] = [];
    }
    acc[model.category].push(model);
    return acc;
  }, {} as Record<string, ModelOption[]>);

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
                  <span className="text-lg">{getCategoryIcon(currentModelOption.category)}</span>
                  <div className="text-left">
                    <div className="font-medium text-sm text-foreground">
                      {currentModelOption.displayName}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {currentModelOption.description}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`px-2 py-1 rounded text-xs font-medium ${getCategoryColor(currentModelOption.category)}`}>
                    {currentModelOption.engine}
                  </span>
                  {currentModelOption.isRecommended && (
                    <span className="px-2 py-1 rounded text-xs font-medium bg-green-100 text-green-800">
                      ✨ Recommended
                    </span>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-2">
                {isLoading && <RefreshCw size={14} className="animate-spin" />}
                <ChevronDown size={16} className={`transition-transform ${isDropdownOpen ? 'rotate-180' : ''}`} />
              </div>
            </button>

            {/* Dropdown */}
            {isDropdownOpen && (
              <div className="absolute top-full left-0 right-0 mt-1 bg-popover border rounded-lg shadow-lg z-50 max-h-96 overflow-y-auto">
                <div className="p-2">
                  {Object.entries(groupedModels).map(([category, models]) => (
                    <div key={category} className="mb-4 last:mb-0">
                      <div className="px-2 py-1 text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                        {getCategoryIcon(category as ModelOption['category'])} {category} Models
                      </div>
                      <div className="space-y-1">
                        {models.map((model) => (
                          <button
                            key={model.id}
                            onClick={() => handleModelSwitch(model.id)}
                            className="w-full flex items-center justify-between p-2 hover:bg-muted rounded text-left transition-colors"
                          >
                            <div className="flex items-center gap-3 flex-1">
                              <div className="flex-1">
                                <div className="font-medium text-sm text-foreground flex items-center gap-2">
                                  {model.displayName}
                                  {model.isRecommended && <span className="text-xs">✨</span>}
                                </div>
                                <div className="text-xs text-muted-foreground">
                                  {model.description}
                                </div>
                                <div className="text-xs text-muted-foreground mt-1">
                                  Context: {model.contextLength.toLocaleString()} tokens
                                </div>
                              </div>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className={`px-2 py-1 rounded text-xs font-medium ${getCategoryColor(model.category)}`}>
                                {model.engine}
                              </span>
                              {model.id === currentModel && (
                                <Check size={14} className="text-green-600" />
                              )}
                            </div>
                          </button>
                        ))}
                      </div>
                    </div>
                  ))}
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
              {currentModelOption.contextLength.toLocaleString()}
            </div>
          </div>
          <div className="text-center p-2 bg-muted rounded">
            <div className="text-xs text-muted-foreground">Engine</div>
            <div className="font-mono text-sm text-foreground">
              {currentModelOption.engine}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}