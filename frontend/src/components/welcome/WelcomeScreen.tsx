/**
 * Welcome screen for Oumi Chat Desktop - Model configuration selection
 */

"use client";

import React from 'react';
import { Bot, Search, Zap, Settings, ArrowRight, Loader2, AlertCircle, CheckCircle2, MessageSquare, Wand2, BookOpen, Heart, Briefcase, Code, Gamepad2 } from 'lucide-react';
import apiClient from '@/lib/unified-api';

interface ConfigOption {
  id: string;
  config_path: string;
  relative_path: string;
  display_name: string;
  model_name: string;
  engine: string;
  context_length: number;
  model_family: string;
  size_category: string;
  recommended?: boolean;
}

interface SystemPromptPreset {
  id: string;
  name: string;
  icon: React.ReactNode;
  description: string;
  prompt: string;
  category: 'general' | 'creative' | 'professional' | 'personal';
}

interface WelcomeScreenProps {
  onConfigSelected: (configId: string, systemPrompt?: string) => void;
}

export default function WelcomeScreen({ onConfigSelected }: WelcomeScreenProps) {
  const [configs, setConfigs] = React.useState<ConfigOption[]>([]);
  const [filteredConfigs, setFilteredConfigs] = React.useState<ConfigOption[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [searchTerm, setSearchTerm] = React.useState('');
  const [selectedEngine, setSelectedEngine] = React.useState<string>('all');
  const [selectedSize, setSelectedSize] = React.useState<string>('all');
  const [starting, setStarting] = React.useState(false);
  
  // System prompt state
  const [showSystemPrompt, setShowSystemPrompt] = React.useState(false);
  const [selectedConfig, setSelectedConfig] = React.useState<string | null>(null);
  const [systemPrompt, setSystemPrompt] = React.useState('');
  const [selectedPreset, setSelectedPreset] = React.useState<string>('default');

  // System prompt presets
  const systemPromptPresets: SystemPromptPreset[] = [
    {
      id: 'default',
      name: 'Default Assistant',
      icon: <Bot className="w-5 h-5" />,
      description: 'Helpful, harmless, and honest AI assistant',
      prompt: 'You are a helpful, harmless, and honest AI assistant. Provide accurate, thoughtful responses while being respectful and ethical.',
      category: 'general'
    },
    {
      id: 'creative_writer',
      name: 'Creative Writer',
      icon: <BookOpen className="w-5 h-5" />,
      description: 'Storytelling and creative writing companion',
      prompt: 'You are a creative writing assistant with expertise in storytelling, world-building, and narrative techniques. Help users craft compelling stories, develop characters, and explore creative ideas with imagination and literary flair.',
      category: 'creative'
    },
    {
      id: 'roleplay',
      name: 'Roleplay Partner',
      icon: <Gamepad2 className="w-5 h-5" />,
      description: 'Immersive character roleplay and scenarios',
      prompt: 'You are a skilled roleplay partner who can embody different characters and scenarios. Maintain character consistency, create engaging dialogue, and help build immersive fictional worlds while respecting boundaries.',
      category: 'creative'
    },
    {
      id: 'therapist',
      name: 'Supportive Listener',
      icon: <Heart className="w-5 h-5" />,
      description: 'Empathetic support and reflection (not professional therapy)',
      prompt: 'You are a compassionate, empathetic listener who provides emotional support and thoughtful reflection. Use active listening, ask clarifying questions, and offer gentle guidance. Remember: you are not a licensed therapist - encourage professional help when appropriate.',
      category: 'personal'
    },
    {
      id: 'coding_mentor',
      name: 'Coding Mentor',
      icon: <Code className="w-5 h-5" />,
      description: 'Programming guidance and code review',
      prompt: 'You are an experienced software engineer and coding mentor. Provide clear explanations, best practices, code reviews, and debugging help. Focus on teaching concepts, writing clean code, and helping users become better programmers.',
      category: 'professional'
    },
    {
      id: 'business_advisor',
      name: 'Business Advisor',
      icon: <Briefcase className="w-5 h-5" />,
      description: 'Strategic business insights and planning',
      prompt: 'You are a knowledgeable business advisor with expertise in strategy, operations, and entrepreneurship. Provide practical advice, help analyze business decisions, and offer insights on market trends and business development.',
      category: 'professional'
    },
    {
      id: 'teacher',
      name: 'Patient Teacher',
      icon: <Wand2 className="w-5 h-5" />,
      description: 'Educational explanations and learning support',
      prompt: 'You are a patient, encouraging teacher who excels at explaining complex topics in simple terms. Use examples, analogies, and step-by-step breakdowns. Adapt your teaching style to the user\'s level and learning preferences.',
      category: 'general'
    }
  ];

  React.useEffect(() => {
    loadConfigs();
  }, []);

  React.useEffect(() => {
    filterConfigs();
  }, [configs, searchTerm, selectedEngine, selectedSize]);

  // Update system prompt when preset changes
  React.useEffect(() => {
    const preset = systemPromptPresets.find(p => p.id === selectedPreset);
    if (preset) {
      setSystemPrompt(preset.prompt);
    }
  }, [selectedPreset, systemPromptPresets]);

  const loadConfigs = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await apiClient.getConfigs();
      
      if (response.success && response.data) {
        const processedConfigs = response.data.configs.map((config: any) => ({
          ...config,
          // Extract model family from path
          model_family: extractModelFamily(config.relative_path),
          // Categorize model size
          size_category: categorizeModelSize(config.display_name),
          // Mark some as recommended
          recommended: isRecommended(config)
        }));
        
        setConfigs(processedConfigs);
      } else {
        throw new Error(response.message || 'Failed to load configurations');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load configurations');
    } finally {
      setLoading(false);
    }
  };

  const extractModelFamily = (path: string): string => {
    const match = path.match(/recipes\/([^/]+)/);
    return match ? match[1] : 'unknown';
  };

  const categorizeModelSize = (displayName: string): string => {
    const name = displayName.toLowerCase();
    if (name.includes('135m') || name.includes('1b')) return 'small';
    if (name.includes('3b') || name.includes('7b') || name.includes('8b')) return 'medium';
    if (name.includes('20b') || name.includes('30b') || name.includes('70b')) return 'large';
    if (name.includes('120b') || name.includes('405b')) return 'xl';
    return 'medium';
  };

  const isRecommended = (config: any): boolean => {
    const name = config.display_name.toLowerCase();
    const engine = config.engine.toLowerCase();
    
    // Recommend GGUF configs for macOS (efficient)
    if (name.includes('gguf') && name.includes('macos')) return true;
    
    // Recommend smaller models for general use
    if ((name.includes('8b') || name.includes('7b')) && engine === 'native') return true;
    
    // Recommend instruct models over base models
    if (name.includes('instruct') && !name.includes('120b') && !name.includes('405b')) return true;
    
    return false;
  };

  const filterConfigs = () => {
    let filtered = [...configs];

    // Search filter
    if (searchTerm) {
      filtered = filtered.filter(config => 
        config.display_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        config.model_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        config.model_family.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Engine filter
    if (selectedEngine !== 'all') {
      filtered = filtered.filter(config => 
        config.engine.toLowerCase() === selectedEngine.toLowerCase()
      );
    }

    // Size filter
    if (selectedSize !== 'all') {
      filtered = filtered.filter(config => config.size_category === selectedSize);
    }

    // Sort by recommended first, then by model family and size
    filtered.sort((a, b) => {
      if (a.recommended && !b.recommended) return -1;
      if (!a.recommended && b.recommended) return 1;
      
      if (a.model_family !== b.model_family) {
        return a.model_family.localeCompare(b.model_family);
      }
      
      return a.display_name.localeCompare(b.display_name);
    });

    setFilteredConfigs(filtered);
  };

  const handleConfigSelect = (configId: string) => {
    setSelectedConfig(configId);
    setShowSystemPrompt(true);
  };

  const handleStartChat = async () => {
    if (!selectedConfig) return;
    
    setStarting(true);
    try {
      // Small delay for UI feedback
      await new Promise(resolve => setTimeout(resolve, 500));
      onConfigSelected(selectedConfig, systemPrompt.trim() || undefined);
    } catch (err) {
      setError('Failed to start with selected configuration');
      setStarting(false);
    }
  };

  const handleBackToModels = () => {
    setShowSystemPrompt(false);
    setSelectedConfig(null);
  };

  const getEngineIcon = (engine: string) => {
    switch (engine.toLowerCase()) {
      case 'vllm': return <Zap className="w-4 h-4 text-blue-500" />;
      case 'native': return <Bot className="w-4 h-4 text-green-500" />;
      case 'llamacpp': return <Settings className="w-4 h-4 text-purple-500" />;
      default: return <Bot className="w-4 h-4 text-gray-500" />;
    }
  };

  const getSizeColor = (size: string) => {
    switch (size) {
      case 'small': return 'text-green-600 bg-green-100';
      case 'medium': return 'text-blue-600 bg-blue-100';
      case 'large': return 'text-orange-600 bg-orange-100';
      case 'xl': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  if (starting) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="bg-white rounded-lg shadow-lg p-8 max-w-md w-full mx-4 text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-6"></div>
          <h2 className="text-xl font-semibold mb-2">Starting Oumi Chat</h2>
          <p className="text-gray-600">Loading your selected model configuration...</p>
        </div>
      </div>
    );
  }

  // System Prompt Configuration Screen
  if (showSystemPrompt && selectedConfig) {
    const config = configs.find(c => c.id === selectedConfig);
    
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-50 to-pink-100">
        <div className="container mx-auto px-4 py-8">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="flex items-center justify-center mb-4">
              <MessageSquare className="w-12 h-12 text-purple-600 mr-3" />
              <h1 className="text-4xl font-bold text-gray-900">Customize Your AI</h1>
            </div>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Choose how your AI assistant should behave. You can use a preset or create your own system prompt.
            </p>
          </div>

          <div className="max-w-4xl mx-auto">
            {/* Selected Config Summary */}
            <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
              <h2 className="text-xl font-semibold mb-2">Selected Model</h2>
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-medium text-gray-900">{config?.display_name}</h3>
                  <p className="text-sm text-gray-600">{config?.model_name}</p>
                </div>
                <button 
                  onClick={handleBackToModels}
                  className="text-blue-600 hover:text-blue-700 text-sm font-medium"
                >
                  Change Model
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Preset Selection */}
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h2 className="text-xl font-semibold mb-4">Choose a Preset</h2>
                <div className="space-y-3">
                  {systemPromptPresets.map((preset) => (
                    <div 
                      key={preset.id}
                      className={`p-4 border rounded-lg cursor-pointer transition-all ${
                        selectedPreset === preset.id 
                          ? 'border-purple-500 bg-purple-50' 
                          : 'border-gray-200 hover:border-purple-300'
                      }`}
                      onClick={() => setSelectedPreset(preset.id)}
                    >
                      <div className="flex items-start space-x-3">
                        <div className="flex-shrink-0 text-purple-600">
                          {preset.icon}
                        </div>
                        <div className="flex-1 min-w-0">
                          <h3 className="font-medium text-gray-900">{preset.name}</h3>
                          <p className="text-sm text-gray-600 mt-1">{preset.description}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Custom System Prompt */}
              <div className="bg-white rounded-lg shadow-lg p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-semibold">System Prompt</h2>
                  <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
                    {systemPrompt.length} characters
                  </span>
                </div>
                
                <textarea
                  value={systemPrompt}
                  onChange={(e) => {
                    setSystemPrompt(e.target.value);
                    setSelectedPreset('custom');
                  }}
                  placeholder="Enter a custom system prompt to define how your AI assistant should behave..."
                  className="w-full h-64 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 resize-none"
                />
                
                <div className="mt-4 space-y-2">
                  <h4 className="font-medium text-gray-900">Tips:</h4>
                  <ul className="text-sm text-gray-600 space-y-1">
                    <li>• Be specific about the AI's role and expertise</li>
                    <li>• Include desired tone (formal, casual, friendly, etc.)</li>
                    <li>• Mention any constraints or guidelines</li>
                    <li>• Keep it clear and concise</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="mt-8 flex justify-center space-x-4">
              <button 
                onClick={handleBackToModels}
                className="px-6 py-3 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300 transition-colors font-medium"
              >
                ← Back to Models
              </button>
              <button 
                onClick={handleStartChat}
                className="px-8 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors font-medium flex items-center"
              >
                Start Chat
                <ArrowRight className="w-5 h-5 ml-2" />
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <Bot className="w-12 h-12 text-blue-600 mr-3" />
            <h1 className="text-4xl font-bold text-gray-900">Oumi Chat</h1>
          </div>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Welcome to Oumi Chat Desktop. Select a model configuration to get started with your AI conversations.
          </p>
        </div>

        {/* Content */}
        <div className="max-w-6xl mx-auto">
          {loading ? (
            <div className="bg-white rounded-lg shadow-lg p-8 text-center">
              <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4 text-blue-600" />
              <p className="text-gray-600">Loading available configurations...</p>
            </div>
          ) : error ? (
            <div className="bg-white rounded-lg shadow-lg p-8 text-center">
              <AlertCircle className="w-8 h-8 mx-auto mb-4 text-red-500" />
              <h3 className="text-lg font-semibold mb-2 text-red-700">Error Loading Configurations</h3>
              <p className="text-gray-600 mb-4">{error}</p>
              <button 
                onClick={loadConfigs}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Retry
              </button>
            </div>
          ) : (
            <>
              {/* Filters */}
              <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* Search */}
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                    <input
                      type="text"
                      placeholder="Search models..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>

                  {/* Engine Filter */}
                  <select
                    value={selectedEngine}
                    onChange={(e) => setSelectedEngine(e.target.value)}
                    className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="all">All Engines</option>
                    <option value="native">Native (CPU/GPU)</option>
                    <option value="vllm">vLLM (GPU)</option>
                    <option value="llamacpp">LlamaCPP (CPU)</option>
                  </select>

                  {/* Size Filter */}
                  <select
                    value={selectedSize}
                    onChange={(e) => setSelectedSize(e.target.value)}
                    className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="all">All Sizes</option>
                    <option value="small">Small (1-3B)</option>
                    <option value="medium">Medium (7-8B)</option>
                    <option value="large">Large (20-70B)</option>
                    <option value="xl">Extra Large (100B+)</option>
                  </select>
                </div>
              </div>

              {/* Model Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredConfigs.map((config) => (
                  <div 
                    key={config.id}
                    className="bg-white rounded-lg shadow-lg p-6 hover:shadow-xl transition-shadow cursor-pointer relative"
                    onClick={() => handleConfigSelect(config.id)}
                  >
                    {config.recommended && (
                      <div className="absolute top-2 right-2">
                        <CheckCircle2 className="w-5 h-5 text-green-500" />
                      </div>
                    )}
                    
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex items-center">
                        {getEngineIcon(config.engine)}
                        <span className="ml-2 text-sm font-medium text-gray-600 uppercase">
                          {config.engine}
                        </span>
                      </div>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSizeColor(config.size_category)}`}>
                        {config.size_category}
                      </span>
                    </div>

                    <h3 className="text-lg font-semibold mb-2 text-gray-900">
                      {config.display_name}
                    </h3>
                    
                    <p className="text-sm text-gray-600 mb-3">
                      {config.model_name}
                    </p>

                    <div className="flex items-center justify-between text-sm text-gray-500 mb-4">
                      <span>Context: {config.context_length.toLocaleString()}</span>
                      <span className="capitalize">{config.model_family}</span>
                    </div>

                    <div className="flex items-center text-blue-600 hover:text-blue-700">
                      <span className="text-sm font-medium">Select this model</span>
                      <ArrowRight className="w-4 h-4 ml-2" />
                    </div>
                  </div>
                ))}
              </div>

              {filteredConfigs.length === 0 && (
                <div className="bg-white rounded-lg shadow-lg p-8 text-center">
                  <p className="text-gray-600">No configurations match your filters. Try adjusting your search criteria.</p>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}