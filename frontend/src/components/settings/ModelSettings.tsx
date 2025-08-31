/**
 * Interactive model settings component with sliders and controls
 */

"use client";

import React from 'react';
import { Settings, Thermometer, Hash, Target, RotateCcw, FileText } from 'lucide-react';
import { useChatStore } from '@/lib/store';
import apiClient from '@/lib/unified-api';

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  description?: string;
  icon?: React.ReactNode;
  formatValue?: (value: number) => string;
}

const Slider: React.FC<SliderProps> = ({
  label,
  value,
  min,
  max,
  step,
  onChange,
  description,
  icon,
  formatValue = (v) => v.toString(),
}) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(parseFloat(e.target.value));
  };

  const percentage = ((value - min) / (max - min)) * 100;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {icon && <div className="text-primary">{icon}</div>}
          <div>
            <label className="text-sm font-medium text-foreground">
              {label}
            </label>
            {description && (
              <p className="text-xs text-muted-foreground">{description}</p>
            )}
          </div>
        </div>
        <div className="text-sm font-mono text-foreground bg-muted px-2 py-1 rounded">
          {formatValue(value)}
        </div>
      </div>
      
      <div className="relative">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={handleChange}
          className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer slider"
          style={{
            background: `linear-gradient(to right, hsl(var(--primary)) 0%, hsl(var(--primary)) ${percentage}%, hsl(var(--muted)) ${percentage}%, hsl(var(--muted)) 100%)`
          }}
        />
        <div className="flex justify-between text-xs text-muted-foreground mt-1">
          <span>{formatValue(min)}</span>
          <span>{formatValue(max)}</span>
        </div>
      </div>
    </div>
  );
};

interface ModelSettingsProps {
  className?: string;
}

export default function ModelSettings({ className = '' }: ModelSettingsProps) {
  const { generationParams, updateGenerationParam } = useChatStore();
  
  // Default values for reset
  const defaultParams = {
    temperature: 0.7,
    maxTokens: 2048,
    topP: 0.9,
    contextLength: 8192,
  };


  const handleReset = () => {
    Object.entries(defaultParams).forEach(([key, value]) => {
      updateGenerationParam(key as keyof typeof defaultParams, value);
    });
  };

  return (
    <div className={`bg-card rounded-lg p-4 border space-y-6 ${className}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Settings size={16} />
          <span className="text-sm font-semibold text-foreground">Model Settings</span>
        </div>
        <button
          onClick={handleReset}
          className="flex items-center gap-1 px-2 py-1 text-xs text-muted-foreground hover:text-foreground transition-colors rounded hover:bg-muted"
          title="Reset to defaults"
        >
          <RotateCcw size={12} />
          Reset
        </button>
      </div>

      <div className="space-y-6">
        {/* Temperature Slider */}
        <Slider
          label="Temperature"
          value={generationParams.temperature ?? 0.7}
          min={0.0}
          max={2.0}
          step={0.1}
          onChange={(value) => updateGenerationParam('temperature', value)}
          description="Controls randomness in responses"
          icon={<Thermometer size={14} />}
          formatValue={(v) => v.toFixed(1)}
        />

        {/* Max Tokens Slider */}
        <Slider
          label="Max Tokens"
          value={generationParams.maxTokens ?? 2048}
          min={1}
          max={8192}
          step={32}
          onChange={(value) => updateGenerationParam('maxTokens', Math.round(value))}
          description="Maximum length of response"
          icon={<Hash size={14} />}
          formatValue={(v) => Math.round(v).toLocaleString()}
        />

        {/* Top-p Slider */}
        <Slider
          label="Top-p (Nucleus Sampling)"
          value={generationParams.topP ?? 0.9}
          min={0.0}
          max={1.0}
          step={0.05}
          onChange={(value) => updateGenerationParam('topP', value)}
          description="Controls diversity of token selection"
          icon={<Target size={14} />}
          formatValue={(v) => v.toFixed(2)}
        />

        {/* Context Length Slider */}
        <Slider
          label="Max Context Length"
          value={generationParams.contextLength ?? 8192}
          min={512}
          max={131072}
          step={512}
          onChange={(value) => updateGenerationParam('contextLength', Math.round(value))}
          description="Maximum tokens the model can process (input + output)"
          icon={<FileText size={14} />}
          formatValue={(v) => Math.round(v).toLocaleString() + ' tokens'}
        />

        {/* Settings Info */}
        <div className="pt-4 border-t space-y-2">
          <h4 className="text-xs font-medium text-foreground">Quick Presets</h4>
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={() => {
                updateGenerationParam('temperature', 0.2);
                updateGenerationParam('topP', 0.8);
              }}
              className="px-3 py-2 text-xs bg-muted hover:bg-muted/80 rounded transition-colors text-foreground"
            >
              🎯 Focused
            </button>
            <button
              onClick={() => {
                updateGenerationParam('temperature', 0.7);
                updateGenerationParam('topP', 0.9);
              }}
              className="px-3 py-2 text-xs bg-muted hover:bg-muted/80 rounded transition-colors text-foreground"
            >
              ⚖️ Balanced
            </button>
            <button
              onClick={() => {
                updateGenerationParam('temperature', 1.2);
                updateGenerationParam('topP', 0.95);
              }}
              className="px-3 py-2 text-xs bg-muted hover:bg-muted/80 rounded transition-colors text-foreground"
            >
              🎨 Creative
            </button>
            <button
              onClick={() => {
                updateGenerationParam('temperature', 1.8);
                updateGenerationParam('topP', 0.98);
              }}
              className="px-3 py-2 text-xs bg-muted hover:bg-muted/80 rounded transition-colors text-foreground"
            >
              🚀 Wild
            </button>
          </div>
        </div>

        {/* Current Settings Summary */}
        <div className="pt-4 border-t">
          <h4 className="text-xs font-medium text-foreground mb-2">Current Settings</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            <div className="text-center p-2 bg-muted rounded">
              <div className="font-mono text-foreground">{(generationParams.temperature ?? 0.7).toFixed(1)}</div>
              <div className="text-muted-foreground">Temp</div>
            </div>
            <div className="text-center p-2 bg-muted rounded">
              <div className="font-mono text-foreground">{(generationParams.maxTokens ?? 2048).toLocaleString()}</div>
              <div className="text-muted-foreground">Max Tokens</div>
            </div>
            <div className="text-center p-2 bg-muted rounded">
              <div className="font-mono text-foreground">{(generationParams.topP ?? 0.9).toFixed(2)}</div>
              <div className="text-muted-foreground">Top-p</div>
            </div>
            <div className="text-center p-2 bg-muted rounded">
              <div className="font-mono text-foreground">{(generationParams.contextLength ?? 8192).toLocaleString()}</div>
              <div className="text-muted-foreground">Context</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}