'use client';

import { useState, useEffect } from 'react';
import { AlertTriangle, RefreshCw, X, Monitor, Cpu, HardDrive, Zap } from 'lucide-react';
import apiClient from '@/lib/unified-api';

interface SystemChangeInfo {
  hasChanged: boolean;
  changes: string[];
  shouldRebuild: boolean;
}

interface SystemChangeWarningProps {
  onRebuild?: () => void;
  onDismiss?: () => void;
}

export default function SystemChangeWarning({ onRebuild, onDismiss }: SystemChangeWarningProps) {
  const [changeInfo, setChangeInfo] = useState<SystemChangeInfo | null>(null);
  const [systemInfo, setSystemInfo] = useState<any>(null);
  const [isVisible, setIsVisible] = useState(false);
  const [isRebuilding, setIsRebuilding] = useState(false);

  useEffect(() => {
    checkSystemChanges();
  }, []);

  const checkSystemChanges = async () => {
    try {
      if (!apiClient.isElectronApp()) return;

      const [changeInfo, systemInfo] = await Promise.all([
        apiClient.getSystemChangeInfo(),
        apiClient.getEnvironmentSystemInfo()
      ]);

      setChangeInfo(changeInfo);
      setSystemInfo(systemInfo);
      setIsVisible(changeInfo?.hasChanged || false);
    } catch (error) {
      console.error('Failed to check system changes:', error);
    }
  };

  const handleRebuild = async () => {
    try {
      setIsRebuilding(true);
      const result = await apiClient.rebuildPythonEnvironment();
      
      if (result.success) {
        setIsVisible(false);
        if (onRebuild) {
          onRebuild();
        }
      } else {
        alert(`Rebuild failed: ${result.message}`);
      }
    } catch (error) {
      console.error('Rebuild failed:', error);
      alert('Rebuild failed. Please try again.');
    } finally {
      setIsRebuilding(false);
    }
  };

  const handleDismiss = () => {
    setIsVisible(false);
    if (onDismiss) {
      onDismiss();
    }
  };

  const getChangeIcon = (change: string) => {
    if (change.includes('Platform') || change.includes('Architecture')) {
      return <Monitor className="w-4 h-4" />;
    } else if (change.includes('CUDA')) {
      return <Zap className="w-4 h-4" />;
    } else if (change.includes('RAM')) {
      return <HardDrive className="w-4 h-4" />;
    } else {
      return <Cpu className="w-4 h-4" />;
    }
  };

  const getSystemCapabilitiesSummary = () => {
    if (!systemInfo) return 'System information not available';
    
    const parts = [];
    parts.push(`${systemInfo.platformVersion || 'Unknown OS'} (${systemInfo.architecture || 'Unknown'})`);
    parts.push(`${systemInfo.totalRAM || 0}GB RAM`);
    
    if (systemInfo.cudaAvailable && systemInfo.cudaDevices?.length > 0) {
      const totalVRAM = systemInfo.cudaDevices.reduce((sum: number, device: any) => sum + (device.vram || 0), 0);
      parts.push(`CUDA (${systemInfo.cudaDevices.length} GPU${systemInfo.cudaDevices.length > 1 ? 's' : ''}, ${totalVRAM.toFixed(1)}GB VRAM)`);
    } else {
      parts.push('No CUDA');
    }
    
    return parts.join(' • ');
  };

  if (!isVisible || !changeInfo) {
    return null;
  }

  return (
    <div className="fixed top-4 right-4 z-50 max-w-md">
      <div className="bg-card border-2 border-orange-200 dark:border-orange-800 rounded-lg shadow-lg p-4">
        {/* Header */}
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center">
            <AlertTriangle className="w-5 h-5 text-orange-500 mr-2" />
            <h3 className="font-medium text-foreground">
              {changeInfo.shouldRebuild ? 'System Changes Detected' : 'System Changes Noted'}
            </h3>
          </div>
          <button
            onClick={handleDismiss}
            className="text-muted-foreground hover:text-foreground transition-colors"
            aria-label="Dismiss"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Current System Info */}
        <div className="mb-3 p-2 bg-muted/50 rounded text-xs">
          <div className="font-medium text-foreground mb-1">Current System:</div>
          <div className="text-muted-foreground">{getSystemCapabilitiesSummary()}</div>
        </div>

        {/* Changes List */}
        <div className="mb-4">
          <div className="text-sm font-medium text-foreground mb-2">Changes detected:</div>
          <div className="space-y-1">
            {changeInfo.changes.map((change, index) => (
              <div key={index} className="flex items-center text-xs text-muted-foreground">
                {getChangeIcon(change)}
                <span className="ml-2">{change}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-2">
          {changeInfo.shouldRebuild && (
            <button
              onClick={handleRebuild}
              disabled={isRebuilding}
              className="flex-1 flex items-center justify-center px-3 py-2 bg-orange-500 hover:bg-orange-600 text-white rounded-md transition-colors disabled:opacity-50 text-sm"
            >
              {isRebuilding ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Rebuilding...
                </>
              ) : (
                <>
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Rebuild Environment
                </>
              )}
            </button>
          )}
          <button
            onClick={handleDismiss}
            className="px-3 py-2 border border-border hover:border-primary/50 rounded-md transition-colors text-sm"
          >
            {changeInfo.shouldRebuild ? 'Later' : 'Dismiss'}
          </button>
        </div>

        {/* Rebuild recommendation */}
        {changeInfo.shouldRebuild && (
          <p className="mt-3 text-xs text-muted-foreground">
            💡 Rebuilding your environment is recommended for optimal performance and compatibility.
          </p>
        )}
      </div>
    </div>
  );
}