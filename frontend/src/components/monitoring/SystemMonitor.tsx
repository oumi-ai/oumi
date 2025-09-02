/**
 * System monitoring component with live CPU, RAM, GPU, context usage, and network activity
 */

"use client";

import React from 'react';
import { Activity, Cpu, HardDrive, Zap, MessageSquare, Wifi, WifiOff, Clock, Bot, Play, Square, RefreshCw } from 'lucide-react';
import apiClient from '@/lib/unified-api';
import { useChatStore } from '@/lib/store';

interface SystemStats {
  cpu_percent?: number;
  ram_used_gb: number;
  ram_total_gb: number;
  ram_percent: number;
  gpu_vram_used_gb?: number;
  gpu_vram_total_gb?: number;
  gpu_vram_percent?: number;
  context_used_tokens: number;
  context_max_tokens: number;
  context_percent: number;
  conversation_turns?: number;
}

interface ModelStatus {
  loaded: boolean;
  modelName?: string;
  lastTested?: number;
  testResult?: 'success' | 'failure' | 'unknown';
}

interface NetworkActivity {
  activeRequests: number;
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  averageResponseTime: number;
  lastRequestTime?: number;
  connectionStatus: 'connected' | 'disconnected' | 'slow';
}

interface ProgressBarProps {
  value: number;
  max: number;
  label: string;
  icon: React.ReactNode;
  color: string;
  formatValue?: (value: number) => string;
  formatMax?: (max: number) => string;
}

const ProgressBar: React.FC<ProgressBarProps> = ({
  value,
  max,
  label,
  icon,
  color,
  formatValue = (v) => v.toString(),
  formatMax = (m) => m.toString(),
}) => {
  const percentage = max > 0 ? Math.min((value / max) * 100, 100) : 0;
  
  // Determine color based on usage level
  let barColor = 'bg-green-500';
  if (percentage >= 80) {
    barColor = 'bg-red-500';
  } else if (percentage >= 60) {
    barColor = 'bg-yellow-500';
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={`p-1 rounded ${color}`}>
            {icon}
          </div>
          <span className="text-sm font-medium text-foreground">{label}</span>
        </div>
        <div className="text-xs text-muted-foreground">
          {formatValue(value)} / {formatMax(max)}
        </div>
      </div>
      <div className="w-full bg-muted rounded-full h-2">
        <div
          className={`h-2 rounded-full transition-all duration-300 ${barColor}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <div className="text-xs text-muted-foreground text-center">
        {percentage.toFixed(1)}%
      </div>
    </div>
  );
};

interface SystemMonitorProps {
  className?: string;
  updateInterval?: number; // milliseconds
}

export default function SystemMonitor({ 
  className = '',
  updateInterval = 2000 // 2 seconds
}: SystemMonitorProps) {
  const { getCurrentSessionId } = useChatStore();
  const [stats, setStats] = React.useState<SystemStats | null>(null);
  const [networkActivity, setNetworkActivity] = React.useState<NetworkActivity>({
    activeRequests: 0,
    totalRequests: 0,
    successfulRequests: 0,
    failedRequests: 0,
    averageResponseTime: 0,
    connectionStatus: 'disconnected'
  });
  const [modelStatus, setModelStatus] = React.useState<ModelStatus>({
    loaded: false,
    testResult: 'unknown'
  });
  const [isLoading, setIsLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [isModelActionLoading, setIsModelActionLoading] = React.useState(false);
  const intervalRef = React.useRef<NodeJS.Timeout | null>(null);
  const requestTimesRef = React.useRef<number[]>([]);
  const requestCountRef = React.useRef({ total: 0, successful: 0, failed: 0 });
  // Keep latest model status and last auto-tested model name to avoid stale closures and repeated tests
  const modelStatusRef = React.useRef<ModelStatus>(modelStatus);
  const lastAutoTestedModelRef = React.useRef<string | null>(null);

  React.useEffect(() => {
    modelStatusRef.current = modelStatus;
  }, [modelStatus]);

  const trackNetworkRequest = (success: boolean, responseTime: number) => {
    requestCountRef.current.total++;
    if (success) {
      requestCountRef.current.successful++;
    } else {
      requestCountRef.current.failed++;
    }
    
    // Keep last 10 response times for average calculation
    requestTimesRef.current.push(responseTime);
    if (requestTimesRef.current.length > 10) {
      requestTimesRef.current.shift();
    }
    
    const avgResponseTime = requestTimesRef.current.reduce((a, b) => a + b, 0) / requestTimesRef.current.length;
    const successRate = requestCountRef.current.successful / requestCountRef.current.total;
    
    let connectionStatus: 'connected' | 'disconnected' | 'slow' = 'connected';
    if (avgResponseTime > 3000) {
      connectionStatus = 'slow';
    } else if (successRate < 0.8) {
      connectionStatus = 'disconnected';
    }
    
    setNetworkActivity({
      activeRequests: 0, // Will be updated when requests are in flight
      totalRequests: requestCountRef.current.total,
      successfulRequests: requestCountRef.current.successful,
      failedRequests: requestCountRef.current.failed,
      averageResponseTime: avgResponseTime,
      lastRequestTime: Date.now(),
      connectionStatus
    });
  };

  // Model status and control functions
  const checkModelStatus = async () => {
    try {
      // Get current model information
      const modelResponse = await apiClient.getModels();
      if (modelResponse.success && modelResponse.data?.data?.[0]) {
        const model = modelResponse.data.data[0];
        setModelStatus(prev => ({
          ...prev,
          // Only update the model name - preserve loaded status and test results
          modelName: model.id,
        }));
        // If we have a model name but no test result yet, kick off a one-time background test
        if (
          model.id &&
          (!modelStatusRef.current.testResult || modelStatusRef.current.testResult === 'unknown') &&
          lastAutoTestedModelRef.current !== model.id &&
          !isModelActionLoading
        ) {
          lastAutoTestedModelRef.current = model.id;
          // Fire and forget; do not block UI
          void testModel(model.id);
        }
      } else {
        // Only reset if we currently have a model name but API says no model
        setModelStatus(prev => {
          if (prev.modelName) {
            return {
              ...prev,
              loaded: false, // If no model available, definitely not loaded
              modelName: undefined,
              testResult: 'unknown',
            };
          }
          return prev; // Don't change anything if we already know there's no model
        });
      }
    } catch (error) {
      console.error('Failed to check model status:', error);
      // Don't reset status on API errors - could be temporary network issues
      // Only log the error, preserve current status
    }
  };

  // Allow explicit model name for fresh reads after status checks
  const testModel = async (explicitModelName?: string) => {
    const name = explicitModelName ?? modelStatusRef.current.modelName;
    if (!name) return;
    
    setIsModelActionLoading(true);
    try {
      // Get the config path for the current model using UnifiedConfigPathResolver
      // First, try to get the currently selected config from storage
      const selectedConfig = await apiClient.getStorageItem('selectedConfig', null);
      let configPath = selectedConfig;
      
      // If no selected config, try to find it using UnifiedConfigPathResolver
      if (!configPath) {
        try {
          const { configPathResolver } = await import('@/lib/config-path-resolver');
          
          // Try to find a config that matches the current model name
          const configsResponse = await apiClient.getConfigs();
          if (configsResponse.success && configsResponse.data?.configs) {
            const matchingConfig = configsResponse.data.configs.find((config: any) => 
              config.id === name || config.display_name?.includes(name) || name.includes(config.id)
            );
            
            if (matchingConfig && matchingConfig.id) {
              const resolvedConfig = await configPathResolver.getConfigById(matchingConfig.id);
              if (resolvedConfig) {
                configPath = resolvedConfig.configPath;
              }
            }
          }
        } catch (error) {
          console.warn('Failed to resolve config path for model testing:', error);
        }
      }
      
      // Don't test if no valid config path found
      if (!configPath) {
        console.warn('No valid config path found for model:', name);
        return;
      }
      
      const response = await apiClient.testModel(configPath);
      const success = response.success && response.data?.success;
      
      setModelStatus(prev => ({
        loaded: Boolean(success), // Only set loaded to true if test succeeds
        modelName: name,
        lastTested: Date.now(),
        testResult: success ? 'success' : 'failure',
      }));
      
      console.log(success ? '✅ Model test successful' : '❌ Model test failed', response.data?.message);
    } catch (error) {
      console.error('Model test error:', error);
      setModelStatus(prev => ({
        loaded: false, // Test failed, so not loaded
        modelName: name,
        lastTested: Date.now(),
        testResult: 'failure',
      }));
    } finally {
      setIsModelActionLoading(false);
    }
  };

  const unloadModel = async () => {
    setIsModelActionLoading(true);
    try {
      const response = await apiClient.clearModel();
      if (response.success) {
        setModelStatus(prev => ({
          ...prev,
          loaded: false,
          testResult: 'unknown',
          lastTested: undefined,
        }));
        console.log('🧹 Model unloaded successfully');
      } else {
        throw new Error(response.message || 'Failed to unload model');
      }
    } catch (error) {
      console.error('Failed to unload model:', error);
    } finally {
      setIsModelActionLoading(false);
    }
  };

  const reloadModel = async () => {
    setIsModelActionLoading(true);
    try {
      // First unload the model
      await unloadModel();
      
      // Wait a bit then check if model is available (but not loaded until tested)
      setTimeout(async () => {
        await checkModelStatus();
        // Fetch latest model name fresh to avoid stale state, then test it
        try {
          const mr = await apiClient.getModels();
          const name = mr.success ? mr.data?.data?.[0]?.id : undefined;
          if (name) {
            setModelStatus(prev => ({ ...prev, modelName: name }));
            await testModel(name);
          }
        } catch (e) {
          console.warn('Reload: failed to re-fetch models before test', e);
        }
        setIsModelActionLoading(false);
      }, 1000);
    } catch (error) {
      console.error('Failed to reload model:', error);
      setIsModelActionLoading(false);
    }
  };

  const fetchStats = async () => {
    const startTime = Date.now();
    setNetworkActivity(prev => ({ ...prev, activeRequests: prev.activeRequests + 1 }));
    
    try {
      const sessionId = getCurrentSessionId();
      const response = await apiClient.getSystemStats(sessionId);
      const responseTime = Date.now() - startTime;
      
      if (response.success && response.data) {
        setStats(response.data);
        setError(null);
        trackNetworkRequest(true, responseTime);
      } else {
        trackNetworkRequest(false, responseTime);
        throw new Error(response.message || 'Failed to fetch system stats');
      }
    } catch (err) {
      const responseTime = Date.now() - startTime;
      trackNetworkRequest(false, responseTime);
      console.error('System monitor error:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch stats');
    } finally {
      setNetworkActivity(prev => ({ ...prev, activeRequests: Math.max(0, prev.activeRequests - 1) }));
      setIsLoading(false);
    }
  };

  // Start polling when component mounts
  React.useEffect(() => {
    const fetchAll = async () => {
      await fetchStats(); // System stats
      await checkModelStatus(); // Model status
    };
    
    fetchAll(); // Initial load
    
    intervalRef.current = setInterval(fetchAll, updateInterval);
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [updateInterval]);

  // Format file size
  const formatGB = (gb: number) => `${gb.toFixed(1)}GB`;
  const formatTokens = (tokens: number) => tokens.toLocaleString();

  if (error) {
    return (
      <div className={`bg-card rounded-lg p-4 border border-destructive/20 ${className}`}>
        <div className="flex items-center gap-2 text-destructive">
          <Activity size={16} />
          <span className="text-sm font-medium">System Monitor Error</span>
        </div>
        <p className="text-xs text-muted-foreground mt-2">{error}</p>
        <button
          onClick={fetchStats}
          className="mt-2 text-xs text-primary hover:underline"
        >
          Retry
        </button>
      </div>
    );
  }

  if (isLoading || !stats) {
    return (
      <div className={`bg-card rounded-lg p-4 border ${className}`}>
        <div className="flex items-center gap-2 text-muted-foreground">
          <Activity size={16} className="animate-pulse" />
          <span className="text-sm">Loading system stats...</span>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-card rounded-lg p-4 border space-y-4 ${className}`}>
      <div className="flex items-center gap-2 text-foreground">
        <Activity size={16} />
        <span className="text-sm font-semibold">System & Network Monitor</span>
        <div className="ml-auto">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
        </div>
      </div>

      <div className="space-y-4">
        {/* CPU Usage */}
        {stats.cpu_percent !== undefined && (
          <ProgressBar
            value={stats.cpu_percent}
            max={100}
            label="CPU Usage"
            icon={<Cpu size={12} className="text-blue-600" />}
            color="bg-blue-100"
            formatValue={(v) => `${v.toFixed(1)}%`}
            formatMax={() => "100%"}
          />
        )}

        {/* RAM Usage */}
        <ProgressBar
          value={stats.ram_used_gb}
          max={stats.ram_total_gb}
          label="Memory (RAM)"
          icon={<HardDrive size={12} className="text-purple-600" />}
          color="bg-purple-100"
          formatValue={formatGB}
          formatMax={formatGB}
        />

        {/* GPU VRAM Usage (if available) */}
        {stats.gpu_vram_total_gb && stats.gpu_vram_used_gb !== undefined && (
          <ProgressBar
            value={stats.gpu_vram_used_gb}
            max={stats.gpu_vram_total_gb}
            label="GPU VRAM"
            icon={<Zap size={12} className="text-green-600" />}
            color="bg-green-100"
            formatValue={formatGB}
            formatMax={formatGB}
          />
        )}

        {/* Context Window Usage */}
        <ProgressBar
          value={stats.context_used_tokens}
          max={stats.context_max_tokens}
          label="Context Window"
          icon={<MessageSquare size={12} className="text-orange-600" />}
          color="bg-orange-100"
          formatValue={formatTokens}
          formatMax={formatTokens}
        />

        {/* Network Activity */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <div className="p-1 rounded bg-cyan-100">
              {networkActivity.connectionStatus === 'connected' ? (
                <Wifi size={12} className="text-cyan-600" />
              ) : networkActivity.connectionStatus === 'slow' ? (
                <Clock size={12} className="text-yellow-600" />
              ) : (
                <WifiOff size={12} className="text-red-600" />
              )}
            </div>
            <span className="text-sm font-medium text-foreground">Network Activity</span>
            <div className="ml-auto flex items-center gap-1">
              {networkActivity.activeRequests > 0 && (
                <div className="w-2 h-2 bg-cyan-500 rounded-full animate-pulse" />
              )}
              <span className="text-xs text-muted-foreground capitalize">
                {networkActivity.connectionStatus}
              </span>
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Requests:</span>
              <span className="font-mono">{networkActivity.totalRequests}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Active:</span>
              <span className="font-mono">{networkActivity.activeRequests}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-green-600">Success:</span>
              <span className="font-mono">{networkActivity.successfulRequests}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-red-600">Failed:</span>
              <span className="font-mono">{networkActivity.failedRequests}</span>
            </div>
          </div>
          
          {networkActivity.averageResponseTime > 0 && (
            <div className="flex justify-between text-xs pt-1 border-t">
              <span className="text-muted-foreground">Avg Response:</span>
              <span className="font-mono">
                {networkActivity.averageResponseTime < 1000 
                  ? `${Math.round(networkActivity.averageResponseTime)}ms`
                  : `${(networkActivity.averageResponseTime / 1000).toFixed(1)}s`
                }
              </span>
            </div>
          )}
        </div>

        {/* Model Status */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <div className="p-1 rounded bg-blue-100">
              <Bot size={12} className="text-blue-600" />
            </div>
            <span className="text-sm font-medium text-foreground">Model Status</span>
            <div className="ml-auto flex items-center gap-1">
              <div 
                className={`w-2 h-2 rounded-full ${
                  modelStatus.testResult === 'success' ? 'bg-green-500' :
                  modelStatus.testResult === 'failure' ? 'bg-red-500' :
                  'bg-gray-400'
                }`} 
              />
              <span className={`text-xs capitalize ${
                modelStatus.loaded ? 'text-green-600' : 'text-red-600'
              }`}>
                {modelStatus.loaded ? 'Loaded' : 'Unloaded'}
              </span>
            </div>
          </div>
          
          <div className="grid grid-cols-1 gap-2 text-xs">
            {modelStatus.modelName && (
              <div className="flex justify-between">
                <span className="text-muted-foreground">Model:</span>
                <span className="font-mono text-right truncate ml-2" title={modelStatus.modelName}>
                  {modelStatus.modelName.length > 20 
                    ? `${modelStatus.modelName.slice(0, 17)}...`
                    : modelStatus.modelName
                  }
                </span>
              </div>
            )}
            
            {modelStatus.lastTested && (
              <div className="flex justify-between">
                <span className="text-muted-foreground">Last Test:</span>
                <span className={`font-mono ${
                  modelStatus.testResult === 'success' ? 'text-green-600' :
                  modelStatus.testResult === 'failure' ? 'text-red-600' :
                  'text-gray-600'
                }`}>
                  {new Date(modelStatus.lastTested).toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit'
                  })}
                </span>
              </div>
            )}
          </div>
          
          {/* Model Control Buttons */}
          <div className="flex gap-1 pt-1">
            <button
              onClick={() => testModel()}
              disabled={!modelStatus.modelName || isModelActionLoading}
              className="flex items-center gap-1 px-2 py-1 text-xs bg-blue-100 hover:bg-blue-200 text-blue-700 rounded disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              title="Test model functionality"
            >
              {isModelActionLoading ? (
                <RefreshCw size={10} className="animate-spin" />
              ) : (
                <Play size={10} />
              )}
              Test
            </button>
            
            <button
              onClick={modelStatus.loaded ? unloadModel : reloadModel}
              disabled={isModelActionLoading}
              className={`flex items-center gap-1 px-2 py-1 text-xs rounded disabled:opacity-50 disabled:cursor-not-allowed transition-colors ${
                modelStatus.loaded 
                  ? 'bg-red-100 hover:bg-red-200 text-red-700'
                  : 'bg-green-100 hover:bg-green-200 text-green-700'
              }`}
              title={modelStatus.loaded ? 'Unload model from memory' : 'Reload model into memory'}
            >
              {isModelActionLoading ? (
                <RefreshCw size={10} className="animate-spin" />
              ) : modelStatus.loaded ? (
                <Square size={10} />
              ) : (
                <Play size={10} />
              )}
              {modelStatus.loaded ? 'Unload' : 'Reload'}
            </button>
          </div>
        </div>

        {/* Conversation Stats */}
        {stats.conversation_turns !== undefined && (
          <div className="flex items-center justify-between pt-2 border-t">
            <div className="flex items-center gap-2">
              <MessageSquare size={12} className="text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Conversation Turns</span>
            </div>
            <span className="text-xs font-mono text-foreground">
              {stats.conversation_turns}
            </span>
          </div>
        )}
      </div>

      {/* Last update indicator */}
      <div className="text-xs text-muted-foreground text-center pt-2 border-t">
        Updates every {updateInterval / 1000}s
      </div>
    </div>
  );
}
