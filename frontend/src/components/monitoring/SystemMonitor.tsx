/**
 * System monitoring component with live CPU, RAM, GPU, and context usage
 */

"use client";

import React from 'react';
import { Activity, Cpu, HardDrive, Zap, MessageSquare } from 'lucide-react';
import apiClient from '@/lib/unified-api';

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
  const [stats, setStats] = React.useState<SystemStats | null>(null);
  const [isLoading, setIsLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const intervalRef = React.useRef<NodeJS.Timeout | null>(null);

  const fetchStats = async () => {
    try {
      const response = await apiClient.getSystemStats();
      
      if (response.success && response.data) {
        setStats(response.data);
        setError(null);
      } else {
        throw new Error(response.message || 'Failed to fetch system stats');
      }
    } catch (err) {
      console.error('System monitor error:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch stats');
    } finally {
      setIsLoading(false);
    }
  };

  // Start polling when component mounts
  React.useEffect(() => {
    fetchStats(); // Initial load
    
    intervalRef.current = setInterval(fetchStats, updateInterval);
    
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
        <span className="text-sm font-semibold">System Monitor</span>
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