/**
 * API Key Management and Settings Component
 */

"use client";

import React, { useState, useCallback } from 'react';
import { 
  Key, 
  Eye, 
  EyeOff, 
  Plus, 
  Check, 
  X, 
  AlertCircle, 
  DollarSign,
  Settings,
  Shield,
  RefreshCw,
  Trash2,
  ExternalLink,
  Zap
} from 'lucide-react';
import { useChatStore } from '@/lib/store';
import { API_PROVIDERS, getAllProviders, formatCost, calculateCost } from '@/lib/api-providers';
import { apiValidationService } from '@/lib/api-validation';
import { ApiProvider, ApiKeyConfig, ApiValidationResult } from '@/lib/types';

interface ApiKeyInputProps {
  provider: ApiProvider;
  existingKey?: ApiKeyConfig;
  onSave: (key: string) => void;
  onCancel: () => void;
  onRemove?: () => void;
}

function ApiKeyInput({ provider, existingKey, onSave, onCancel, onRemove }: ApiKeyInputProps) {
  const [keyValue, setKeyValue] = useState(existingKey?.keyValue || '');
  const [showKey, setShowKey] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [validationResult, setValidationResult] = useState<ApiValidationResult | null>(null);

  const validateKey = useCallback(async (key: string): Promise<ApiValidationResult> => {
    setIsValidating(true);
    
    try {
      const result = await apiValidationService.validateKey(provider.id, key);
      return result;
    } catch (error) {
      return {
        isValid: false,
        error: error instanceof Error ? error.message : 'Failed to validate API key',
      };
    } finally {
      setIsValidating(false);
    }
  }, [provider.id]);

  const handleSave = async () => {
    if (!keyValue.trim()) return;
    
    const result = await validateKey(keyValue);
    setValidationResult(result);
    
    if (result.isValid) {
      onSave(keyValue);
    }
  };

  const maskKey = (key: string) => {
    if (key.length <= 8) return key;
    return key.slice(0, 4) + '•'.repeat(Math.min(key.length - 8, 20)) + key.slice(-4);
  };

  return (
    <div className="bg-card border rounded-lg p-4 space-y-4">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 bg-primary/10 rounded-lg flex items-center justify-center">
          <Key size={16} className="text-primary" />
        </div>
        <div>
          <h3 className="font-semibold text-sm">{provider.displayName}</h3>
          <p className="text-xs text-muted-foreground">{provider.description}</p>
        </div>
        <a 
          href={provider.website} 
          target="_blank" 
          rel="noopener noreferrer"
          className="ml-auto text-muted-foreground hover:text-foreground transition-colors"
        >
          <ExternalLink size={14} />
        </a>
      </div>

      <div className="space-y-3">
        <div className="relative">
          <input
            type={showKey ? 'text' : 'password'}
            placeholder={provider.keyPlaceholder}
            value={keyValue}
            onChange={(e) => setKeyValue(e.target.value)}
            className="w-full px-3 py-2 pr-10 bg-background border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary"
          />
          <button
            type="button"
            onClick={() => setShowKey(!showKey)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
          >
            {showKey ? <EyeOff size={14} /> : <Eye size={14} />}
          </button>
        </div>

        {validationResult && (
          <div className={`flex items-center gap-2 text-xs p-2 rounded ${
            validationResult.isValid 
              ? 'bg-green-50 text-green-700 border border-green-200' 
              : 'bg-red-50 text-red-700 border border-red-200'
          }`}>
            {validationResult.isValid ? (
              <>
                <Check size={12} />
                <span>API key validated successfully</span>
                {validationResult.details?.organization && (
                  <span className="text-green-600">
                    • {validationResult.details.organization}
                  </span>
                )}
              </>
            ) : (
              <>
                <AlertCircle size={12} />
                <span>{validationResult.error}</span>
              </>
            )}
          </div>
        )}

        <div className="flex gap-2">
          <button
            onClick={handleSave}
            disabled={!keyValue.trim() || isValidating}
            className="flex items-center gap-2 px-3 py-2 bg-primary text-primary-foreground text-xs rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50"
          >
            {isValidating ? (
              <RefreshCw size={12} className="animate-spin" />
            ) : (
              <Check size={12} />
            )}
            {existingKey ? 'Update' : 'Save'} Key
          </button>
          
          <button
            onClick={onCancel}
            className="px-3 py-2 bg-muted text-muted-foreground text-xs rounded-lg hover:bg-muted/80 transition-colors"
          >
            Cancel
          </button>
          
          {existingKey && onRemove && (
            <button
              onClick={onRemove}
              className="ml-auto px-3 py-2 bg-red-50 text-red-700 text-xs rounded-lg hover:bg-red-100 transition-colors"
            >
              <Trash2 size={12} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

interface ProviderCardProps {
  provider: ApiProvider;
  apiKey?: ApiKeyConfig;
  onAddKey: () => void;
  onEditKey: () => void;
  onRemoveKey: () => void;
  onToggleActive: (isActive: boolean) => void;
}

function ProviderCard({ 
  provider, 
  apiKey, 
  onAddKey, 
  onEditKey, 
  onRemoveKey, 
  onToggleActive 
}: ProviderCardProps) {
  const hasKey = !!apiKey;
  const isActive = apiKey?.isActive || false;
  const isValidated = apiKey?.isValid;

  const totalCost = apiKey?.usage?.totalCost || 0;
  const totalTokens = apiKey?.usage?.totalTokens || 0;
  const totalRequests = apiKey?.usage?.totalRequests || 0;

  const statusColor = !hasKey 
    ? 'text-muted-foreground' 
    : isValidated === false 
      ? 'text-red-500'
      : isActive 
        ? 'text-green-500' 
        : 'text-yellow-500';

  const statusIcon = !hasKey 
    ? <Key size={14} className={statusColor} />
    : isValidated === false 
      ? <AlertCircle size={14} className={statusColor} />
      : isActive 
        ? <Check size={14} className={statusColor} />
        : <X size={14} className={statusColor} />;

  return (
    <div className="bg-card border rounded-lg p-4">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
            <Key size={18} className="text-primary" />
          </div>
          <div>
            <h3 className="font-semibold">{provider.displayName}</h3>
            <p className="text-sm text-muted-foreground">{provider.description}</p>
            <div className="flex items-center gap-1 mt-1">
              {statusIcon}
              <span className={`text-xs ${statusColor}`}>
                {!hasKey 
                  ? 'No API key' 
                  : isValidated === false 
                    ? 'Invalid key'
                    : isActive 
                      ? 'Active' 
                      : 'Inactive'}
              </span>
            </div>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          {hasKey && (
            <>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={isActive}
                  onChange={(e) => onToggleActive(e.target.checked)}
                  className="rounded"
                />
                <span className="text-xs text-muted-foreground">Active</span>
              </label>
              <button
                onClick={onEditKey}
                className="p-1 text-muted-foreground hover:text-foreground transition-colors"
                title="Edit API key"
              >
                <Settings size={14} />
              </button>
            </>
          )}
          <a 
            href={provider.website} 
            target="_blank" 
            rel="noopener noreferrer"
            className="p-1 text-muted-foreground hover:text-foreground transition-colors"
            title="Visit provider website"
          >
            <ExternalLink size={14} />
          </a>
        </div>
      </div>

      {/* Models */}
      <div className="mb-4">
        <h4 className="text-xs font-medium text-muted-foreground mb-2">
          Available Models ({provider.models.length})
        </h4>
        <div className="grid grid-cols-1 gap-2">
          {provider.models.slice(0, 3).map((model) => (
            <div key={model.id} className="flex items-center justify-between text-xs p-2 bg-muted/50 rounded">
              <div>
                <span className="font-medium">{model.displayName}</span>
                <div className="flex items-center gap-2 mt-1">
                  {model.tags?.slice(0, 3).map((tag) => (
                    <span key={tag} className="px-1.5 py-0.5 bg-primary/10 text-primary rounded-full text-[10px]">
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
              <div className="text-right">
                <div className="text-muted-foreground">
                  {(model.contextLength / 1000).toFixed(0)}K ctx
                </div>
                {model.inputCost && (
                  <div className="text-muted-foreground">
                    ${model.inputCost}/$
                    {model.outputCost}/1M
                  </div>
                )}
              </div>
            </div>
          ))}
          {provider.models.length > 3 && (
            <div className="text-xs text-muted-foreground text-center py-1">
              +{provider.models.length - 3} more models
            </div>
          )}
        </div>
      </div>

      {/* Usage Stats */}
      {hasKey && totalRequests > 0 && (
        <div className="mb-4 p-3 bg-muted/30 rounded-lg">
          <h4 className="text-xs font-medium text-muted-foreground mb-2 flex items-center gap-1">
            <DollarSign size={12} />
            Usage This Month
          </h4>
          <div className="grid grid-cols-3 gap-3 text-xs">
            <div className="text-center">
              <div className="font-mono text-foreground">{totalRequests.toLocaleString()}</div>
              <div className="text-muted-foreground">Requests</div>
            </div>
            <div className="text-center">
              <div className="font-mono text-foreground">{(totalTokens / 1000).toFixed(1)}K</div>
              <div className="text-muted-foreground">Tokens</div>
            </div>
            <div className="text-center">
              <div className="font-mono text-foreground">{formatCost(totalCost)}</div>
              <div className="text-muted-foreground">Cost</div>
            </div>
          </div>
        </div>
      )}

      {/* Action Button */}
      <button
        onClick={hasKey ? onEditKey : onAddKey}
        className={`w-full py-2 px-4 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2 ${
          hasKey 
            ? 'bg-muted hover:bg-muted/80 text-foreground' 
            : 'bg-primary hover:bg-primary/90 text-primary-foreground'
        }`}
      >
        {hasKey ? (
          <>
            <Settings size={14} />
            Manage Key
          </>
        ) : (
          <>
            <Plus size={14} />
            Add API Key
          </>
        )}
      </button>
    </div>
  );
}

export default function ApiSettings() {
  const { settings, addApiKey, updateApiKey, removeApiKey, setActiveApiKey, updateSettings } = useChatStore();
  const [editingProvider, setEditingProvider] = useState<string | null>(null);
  const [showQuickSetup, setShowQuickSetup] = useState(false);

  const providers = getAllProviders();
  const hasAnyKeys = Object.keys(settings.apiKeys).length > 0;
  const activeKeys = Object.values(settings.apiKeys).filter(key => key.isActive).length;

  const handleSaveKey = (providerId: string, keyValue: string) => {
    if (settings.apiKeys[providerId]) {
      updateApiKey(providerId, { 
        keyValue,
        lastValidated: new Date().toISOString(),
        isValid: true,
      });
    } else {
      addApiKey(providerId, keyValue);
    }
    setEditingProvider(null);
  };

  const handleRemoveKey = (providerId: string) => {
    removeApiKey(providerId);
    setEditingProvider(null);
  };

  const popularProviders = providers.filter(p => ['openai', 'anthropic', 'google'].includes(p.id));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <Shield size={20} />
            API Key Management
          </h2>
          <p className="text-sm text-muted-foreground mt-1">
            Securely manage your API keys for different AI providers
          </p>
        </div>
        <div className="text-right text-xs text-muted-foreground">
          <div>{activeKeys} active keys</div>
          <div>{Object.keys(settings.apiKeys).length} total providers</div>
        </div>
      </div>

      {/* Quick Setup CTA */}
      {!hasAnyKeys && (
        <div className="bg-gradient-to-r from-primary/5 to-primary/10 border border-primary/20 rounded-lg p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
              <Zap size={20} className="text-primary" />
            </div>
            <div>
              <h3 className="font-semibold">Quick Setup</h3>
              <p className="text-sm text-muted-foreground">
                Get started with popular AI providers
              </p>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-3 mb-4">
            {popularProviders.map((provider) => (
              <button
                key={provider.id}
                onClick={() => setEditingProvider(provider.id)}
                className="p-3 bg-background border rounded-lg hover:bg-muted/50 transition-colors"
              >
                <div className="font-medium text-sm">{provider.displayName}</div>
                <div className="text-xs text-muted-foreground mt-1">
                  {provider.models.length} models
                </div>
              </button>
            ))}
          </div>
          <button
            onClick={() => setShowQuickSetup(true)}
            className="text-sm text-primary hover:underline"
          >
            View setup guide →
          </button>
        </div>
      )}

      {/* Editing Form */}
      {editingProvider && (
        <ApiKeyInput
          provider={providers.find(p => p.id === editingProvider)!}
          existingKey={settings.apiKeys[editingProvider]}
          onSave={(key) => handleSaveKey(editingProvider, key)}
          onCancel={() => setEditingProvider(null)}
          onRemove={settings.apiKeys[editingProvider] ? () => handleRemoveKey(editingProvider) : undefined}
        />
      )}

      {/* Provider Cards */}
      <div className="grid gap-4">
        {providers.map((provider) => (
          <ProviderCard
            key={provider.id}
            provider={provider}
            apiKey={settings.apiKeys[provider.id]}
            onAddKey={() => setEditingProvider(provider.id)}
            onEditKey={() => setEditingProvider(provider.id)}
            onRemoveKey={() => handleRemoveKey(provider.id)}
            onToggleActive={(isActive) => setActiveApiKey(provider.id, isActive)}
          />
        ))}
      </div>

      {/* Settings */}
      <div className="bg-card border rounded-lg p-4">
        <h3 className="font-semibold mb-4 flex items-center gap-2">
          <Settings size={16} />
          Settings
        </h3>
        <div className="space-y-4">
          <label className="flex items-center justify-between">
            <div>
              <div className="font-medium text-sm">Usage Monitoring</div>
              <div className="text-xs text-muted-foreground">
                Track API costs and token usage
              </div>
            </div>
            <input
              type="checkbox"
              checked={settings.usageMonitoring}
              onChange={(e) => updateSettings({ usageMonitoring: e.target.checked })}
              className="rounded"
            />
          </label>
          
          <label className="flex items-center justify-between">
            <div>
              <div className="font-medium text-sm">Auto-validate Keys</div>
              <div className="text-xs text-muted-foreground">
                Automatically validate API keys when added
              </div>
            </div>
            <input
              type="checkbox"
              checked={settings.autoValidateKeys}
              onChange={(e) => updateSettings({ autoValidateKeys: e.target.checked })}
              className="rounded"
            />
          </label>

          <div>
            <label className="block font-medium text-sm mb-2">Monthly Cost Limit</label>
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">$</span>
              <input
                type="number"
                min="0"
                step="10"
                placeholder="50"
                value={settings.maxMonthlyCost || ''}
                onChange={(e) => updateSettings({ 
                  maxMonthlyCost: e.target.value ? parseFloat(e.target.value) : undefined 
                })}
                className="flex-1 px-3 py-2 bg-background border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary"
              />
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Get notified when approaching this limit
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}