/**
 * Main Settings Screen Component
 */

"use client";

import React, { useState } from 'react';
import { 
  Settings,
  Key,
  Sliders,
  Monitor,
  Bell,
  Shield,
  Database,
  Download,
  Palette,
  HelpCircle,
  Eye,
  EyeOff,
  Clock,
  Save,
  Cpu,
  HardDrive,
  Zap
} from 'lucide-react';
import ApiSettings from './ApiSettings';
import ModelSettings from './ModelSettings';
import { useChatStore } from '@/lib/store';
import { useAutoSave } from '@/hooks/useAutoSave';
import apiClient from '@/lib/unified-api';
import { SystemCapabilities } from '@/lib/config-matcher';

type SettingsTab = 'api' | 'model' | 'system' | 'notifications' | 'about';

interface ExtendedSystemInfo extends SystemCapabilities {
  // Additional browser/frontend detected information
  nodeVersion: string;
  electronVersion?: string;
  userAgent: string;
  language: string;
  timezone: string;
  screenResolution: string;
  browserCores: number;
  estimatedMemoryGB?: number;
}

// Function to get additional browser/frontend system information
function getBrowserSystemInfo() {
  const getNodeVersion = (): string => {
    // Try to get from process if available (Electron context)
    if (typeof process !== 'undefined' && process.versions) {
      return process.versions.node || 'Unknown';
    }
    return 'Unknown';
  };

  const getElectronVersion = (): string | undefined => {
    // Try to get from process if available (Electron context)
    if (typeof process !== 'undefined' && process.versions) {
      return process.versions.electron;
    }
    return undefined;
  };

  const getScreenResolution = (): string => {
    const width = window.screen.width;
    const height = window.screen.height;
    const pixelRatio = window.devicePixelRatio || 1;
    
    if (pixelRatio > 1) {
      return `${width}×${height} (${pixelRatio}x scaling)`;
    }
    
    return `${width}×${height}`;
  };

  const getEstimatedMemoryGB = (): number | undefined => {
    // @ts-ignore - navigator.deviceMemory is experimental
    const deviceMemory = navigator.deviceMemory;
    return deviceMemory;
  };

  return {
    nodeVersion: getNodeVersion(),
    electronVersion: getElectronVersion(),
    userAgent: navigator.userAgent,
    language: navigator.language || 'Unknown',
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'Unknown',
    screenResolution: getScreenResolution(),
    browserCores: navigator.hardwareConcurrency || 0,
    estimatedMemoryGB: getEstimatedMemoryGB(),
  };
}

// Function to combine backend and frontend system information
async function getCombinedSystemInfo(): Promise<ExtendedSystemInfo | null> {
  try {
    // Get browser-detected information
    const browserInfo = getBrowserSystemInfo();
    
    // Get backend system information if in Electron environment
    if (apiClient.isElectronApp()) {
      const backendSystemInfo = await apiClient.getEnvironmentSystemInfo();
      
      if (backendSystemInfo) {
        // Combine backend and frontend information
        return {
          ...backendSystemInfo,
          ...browserInfo,
        };
      }
    }
    
    // Fallback: create system info from browser detection only
    return {
      // SystemCapabilities fields (fallback values)
      platform: 'Unknown',
      architecture: 'Unknown', 
      totalRAM: browserInfo.estimatedMemoryGB || 0,
      cudaAvailable: false,
      cudaDevices: [],
      
      // Browser-detected fields
      ...browserInfo,
    };
  } catch (error) {
    console.error('Failed to get system information:', error);
    return null;
  }
}

interface TabButtonProps {
  id: SettingsTab;
  icon: React.ReactNode;
  label: string;
  description: string;
  isActive: boolean;
  onClick: (id: SettingsTab) => void;
  badge?: string;
}

function TabButton({ id, icon, label, description, isActive, onClick, badge }: TabButtonProps) {
  return (
    <button
      onClick={() => onClick(id)}
      className={`w-full p-3 text-left rounded-lg transition-colors relative ${
        isActive 
          ? 'bg-primary text-primary-foreground' 
          : 'hover:bg-muted text-foreground'
      }`}
    >
      <div className="flex items-start gap-3">
        <div className={`mt-0.5 ${isActive ? 'text-primary-foreground' : 'text-primary'}`}>
          {icon}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <div className="font-medium text-sm">{label}</div>
            {badge && (
              <span className={`px-1.5 py-0.5 text-[10px] rounded-full ${
                isActive 
                  ? 'bg-primary-foreground/20 text-primary-foreground' 
                  : 'bg-primary/10 text-primary'
              }`}>
                {badge}
              </span>
            )}
          </div>
          <div className={`text-xs mt-0.5 ${
            isActive ? 'text-primary-foreground/70' : 'text-muted-foreground'
          }`}>
            {description}
          </div>
        </div>
      </div>
    </button>
  );
}

function SystemSettings() {
  const { settings, updateSettings } = useChatStore();
  const [showHfToken, setShowHfToken] = useState(false);
  const { isAutoSaveEnabled, autoSaveInterval, lastSaved, isSaving } = useAutoSave();

  const handleHuggingFaceUpdate = (field: 'username' | 'token', value: string) => {
    updateSettings({
      huggingFace: {
        ...(settings.huggingFace || {}),
        [field]: value || undefined,
      },
    });
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Monitor size={20} />
          System Settings
        </h2>
        <p className="text-sm text-muted-foreground mt-1">
          Configure application behavior and performance
        </p>
      </div>

      {/* HuggingFace Integration */}
      <div className="bg-card border rounded-lg p-4 space-y-4">
        <div className="flex items-center gap-2">
          <h3 className="font-semibold">HuggingFace Integration</h3>
          <div className="text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 px-2 py-0.5 rounded-full">
            Optional
          </div>
        </div>
        
        <p className="text-xs text-muted-foreground">
          ⚠️ Model recommendations work better with HuggingFace authentication. This allows access to more model metadata and improved size detection.
        </p>

        <div className="space-y-3">
          <div>
            <label className="block font-medium text-sm mb-2">
              HuggingFace Username <span className="text-muted-foreground font-normal">(optional)</span>
            </label>
            <input
              type="text"
              value={settings.huggingFace?.username || ''}
              onChange={(e) => handleHuggingFaceUpdate('username', e.target.value)}
              placeholder="your-username"
              className="w-full px-3 py-2 bg-background border rounded-lg text-sm focus:ring-2 focus:ring-primary focus:border-primary"
            />
          </div>

          <div>
            <label className="block font-medium text-sm mb-2">
              Personal Access Token <span className="text-muted-foreground font-normal">(optional)</span>
            </label>
            <div className="relative">
              <input
                type={showHfToken ? 'text' : 'password'}
                value={settings.huggingFace?.token || ''}
                onChange={(e) => handleHuggingFaceUpdate('token', e.target.value)}
                placeholder="hf_..."
                className="w-full px-3 py-2 pr-10 bg-background border rounded-lg text-sm focus:ring-2 focus:ring-primary focus:border-primary"
              />
              <button
                type="button"
                onClick={() => setShowHfToken(!showHfToken)}
                className="absolute inset-y-0 right-0 px-3 flex items-center text-muted-foreground hover:text-foreground"
              >
                {showHfToken ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Create a <a 
                href="https://huggingface.co/settings/tokens" 
                target="_blank" 
                rel="noopener noreferrer" 
                className="text-primary hover:underline"
              >
                personal access token
              </a> for enhanced model metadata access
            </p>
          </div>
        </div>
      </div>


      <div className="bg-card border rounded-lg p-4 space-y-4">
        <h3 className="font-semibold">Storage</h3>

        {/* Start new session on launch */}
        <label className="flex items-center justify-between">
          <div>
            <div className="font-medium text-sm flex items-center gap-2">
              <Clock className="w-4 h-4" />
              Start New Session on Launch
            </div>
            <div className="text-xs text-muted-foreground">
              When enabled, the app creates a fresh session each time it starts
            </div>
          </div>
          <input 
            type="checkbox" 
            checked={settings.startNewSessionOnLaunch}
            onChange={(e) => updateSettings({ startNewSessionOnLaunch: e.target.checked })}
            className="rounded" 
          />
        </label>

        <label className="flex items-center justify-between">
          <div>
            <div className="font-medium text-sm flex items-center gap-2">
              <Save className="w-4 h-4" />
              Auto-save Conversations
              {isSaving && <span className="text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 px-2 py-0.5 rounded-full animate-pulse">Saving...</span>}
            </div>
            <div className="text-xs text-muted-foreground">
              Every {autoSaveInterval} minutes • {lastSaved ? `Last saved ${lastSaved.toLocaleTimeString()}` : 'Not saved yet'}
            </div>
          </div>
          <input 
            type="checkbox" 
            checked={settings.autoSave?.enabled || false}
            onChange={(e) => updateSettings({
              autoSave: {
                ...(settings.autoSave || {}),
                enabled: e.target.checked,
              },
            })}
            className="rounded" 
          />
        </label>

        {/* Auto-save interval setting */}
        {settings.autoSave?.enabled && (
          <div className="pl-6 space-y-2">
            <label className="block text-sm font-medium">Save Interval</label>
            <select
              value={settings.autoSave?.intervalMinutes || 5}
              onChange={(e) => updateSettings({
                autoSave: {
                  ...(settings.autoSave || {}),
                  intervalMinutes: parseInt(e.target.value),
                },
              })}
              className="w-full px-3 py-2 bg-background border rounded-lg text-sm focus:ring-2 focus:ring-primary focus:border-primary"
            >
              <option value={1}>1 minute</option>
              <option value={2}>2 minutes</option>
              <option value={5}>5 minutes</option>
              <option value={10}>10 minutes</option>
              <option value={15}>15 minutes</option>
              <option value={30}>30 minutes</option>
            </select>
          </div>
        )}

        <div className="pt-3 border-t">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Cache Size <span className="text-xs text-orange-600 bg-orange-100 dark:text-orange-400 dark:bg-orange-900/30 px-1.5 py-0.5 rounded-full font-medium">PLACEHOLDER</span></span>
            <span className="text-muted-foreground">247 MB</span>
          </div>
          <button className="mt-2 text-sm text-muted-foreground cursor-not-allowed" disabled>
            Clear Cache <span className="text-xs text-orange-600 bg-orange-100 dark:text-orange-400 dark:bg-orange-900/30 px-1 py-0.5 rounded-full font-medium">PLACEHOLDER</span>
          </button>
        </div>
      </div>
    </div>
  );
}

function NotificationSettings() {
  const { settings, updateSettings } = useChatStore();

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Bell size={20} />
          Notifications
        </h2>
        <p className="text-sm text-muted-foreground mt-1">
          Control when and how you receive notifications
        </p>
      </div>

      <div className="bg-card border rounded-lg p-4 space-y-4">
        <h3 className="font-semibold">API Usage Alerts</h3>
        
        <label className="flex items-center justify-between">
          <div>
            <div className="font-medium text-sm">Low Balance Warning</div>
            <div className="text-xs text-muted-foreground">
              Notify when API credits are running low
            </div>
          </div>
          <input 
            type="checkbox" 
            checked={settings.notifications.lowBalance}
            onChange={(e) => updateSettings({
              notifications: { ...settings.notifications, lowBalance: e.target.checked }
            })}
            className="rounded" 
          />
        </label>

        <label className="flex items-center justify-between">
          <div>
            <div className="font-medium text-sm">High Usage Alert</div>
            <div className="text-xs text-muted-foreground">
              Notify when approaching monthly limits
            </div>
          </div>
          <input 
            type="checkbox" 
            checked={settings.notifications.highUsage}
            onChange={(e) => updateSettings({
              notifications: { ...settings.notifications, highUsage: e.target.checked }
            })}
            className="rounded" 
          />
        </label>

        <label className="flex items-center justify-between">
          <div>
            <div className="font-medium text-sm">Key Expiry Warnings</div>
            <div className="text-xs text-muted-foreground">
              Notify before API keys expire
            </div>
          </div>
          <input 
            type="checkbox" 
            checked={settings.notifications.keyExpiry}
            onChange={(e) => updateSettings({
              notifications: { ...settings.notifications, keyExpiry: e.target.checked }
            })}
            className="rounded" 
          />
        </label>
      </div>

      <div className="bg-card border rounded-lg p-4 space-y-4">
        <h3 className="font-semibold">System Notifications</h3>
        
        <label className="flex items-center justify-between opacity-50">
          <div>
            <div className="font-medium text-sm">Model Download Complete <span className="text-xs text-orange-600 bg-orange-100 dark:text-orange-400 dark:bg-orange-900/30 px-1.5 py-0.5 rounded-full font-medium">PLACEHOLDER</span></div>
            <div className="text-xs text-muted-foreground">
              Notify when model downloads finish
            </div>
          </div>
          <input type="checkbox" defaultChecked className="rounded" disabled />
        </label>

        <label className="flex items-center justify-between opacity-50">
          <div>
            <div className="font-medium text-sm">Update Available <span className="text-xs text-orange-600 bg-orange-100 dark:text-orange-400 dark:bg-orange-900/30 px-1.5 py-0.5 rounded-full font-medium">PLACEHOLDER</span></div>
            <div className="text-xs text-muted-foreground">
              Notify when app updates are available
            </div>
          </div>
          <input type="checkbox" defaultChecked className="rounded" disabled />
        </label>

        <label className="flex items-center justify-between opacity-50">
          <div>
            <div className="font-medium text-sm">System Errors <span className="text-xs text-orange-600 bg-orange-100 dark:text-orange-400 dark:bg-orange-900/30 px-1.5 py-0.5 rounded-full font-medium">PLACEHOLDER</span></div>
            <div className="text-xs text-muted-foreground">
              Show notifications for system errors
            </div>
          </div>
          <input type="checkbox" defaultChecked className="rounded" disabled />
        </label>
      </div>
    </div>
  );
}

function AboutSettings() {
  const [systemSpecs, setSystemSpecs] = React.useState<ExtendedSystemInfo | null>(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    // Get combined system specs on component mount
    const loadSystemSpecs = async () => {
      try {
        const specs = await getCombinedSystemInfo();
        setSystemSpecs(specs);
      } catch (error) {
        console.error('Failed to detect system specs:', error);
      } finally {
        setLoading(false);
      }
    };

    loadSystemSpecs();
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <HelpCircle size={20} />
          About Chatterley
        </h2>
        <p className="text-sm text-muted-foreground mt-1">
          Application information and resources
        </p>
      </div>

      <div className="bg-card border rounded-lg p-4">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-16 h-16 bg-primary/10 rounded-lg flex items-center justify-center">
            <Settings size={24} className="text-primary" />
          </div>
          <div>
            <h3 className="text-lg font-semibold">Chatterley</h3>
            <p className="text-sm text-muted-foreground">Desktop AI Chat Application</p>
            <p className="text-xs text-muted-foreground mt-1">Version 0.1.0</p>
          </div>
        </div>

        <div className="space-y-3 text-sm">
          <div className="flex items-center justify-between py-2 border-b">
            <span className="text-muted-foreground">Built with</span>
            <span>Oumi AI Platform</span>
          </div>
          <div className="flex items-center justify-between py-2 border-b">
            <span className="text-muted-foreground">Framework</span>
            <span>Electron + Next.js</span>
          </div>
          <div className="flex items-center justify-between py-2 border-b">
            <span className="text-muted-foreground">License</span>
            <span>Proprietary</span>
          </div>
          <div className="flex items-center justify-between py-2">
            <span className="text-muted-foreground">Last Update</span>
            <span>2024-12-31</span>
          </div>
        </div>

        <div className="mt-6 pt-4 border-t space-y-2">
          <button className="w-full py-2 px-4 bg-muted text-muted-foreground cursor-not-allowed rounded-lg text-sm opacity-50" disabled>
            Check for Updates <span className="text-xs text-orange-600 bg-orange-100 dark:text-orange-400 dark:bg-orange-900/30 px-1.5 py-0.5 rounded-full font-medium ml-2">PLACEHOLDER</span>
          </button>
          <div className="flex gap-2">
            <button className="flex-1 py-2 px-4 bg-muted text-muted-foreground cursor-not-allowed rounded-lg text-sm opacity-50" disabled>
              View Logs <span className="text-xs text-orange-600 bg-orange-100 dark:text-orange-400 dark:bg-orange-900/30 px-1.5 py-0.5 rounded-full font-medium ml-1">PLACEHOLDER</span>
            </button>
            <button className="flex-1 py-2 px-4 bg-muted text-muted-foreground cursor-not-allowed rounded-lg text-sm opacity-50" disabled>
              Report Issue <span className="text-xs text-orange-600 bg-orange-100 dark:text-orange-400 dark:bg-orange-900/30 px-1.5 py-0.5 rounded-full font-medium ml-1">PLACEHOLDER</span>
            </button>
          </div>
        </div>
      </div>

      {/* System Specifications */}
      <div className="bg-card border rounded-lg p-4">
        <h3 className="font-semibold mb-4 flex items-center gap-2">
          <Monitor size={16} />
          System Specifications
        </h3>
        
        {loading ? (
          <div className="text-sm text-muted-foreground">
            <div className="animate-pulse">Loading system specifications...</div>
          </div>
        ) : systemSpecs ? (
          <div className="space-y-4 text-sm">
            {/* System Hardware */}
            <div>
              <h4 className="font-medium mb-3 text-foreground">Hardware & Platform</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="flex items-center justify-between py-2 border-b">
                  <span className="text-muted-foreground flex items-center gap-2">
                    <Monitor size={14} />
                    Platform
                  </span>
                  <span className="text-right">
                    {systemSpecs.platform === 'darwin' ? 'macOS' : 
                     systemSpecs.platform === 'win32' ? 'Windows' :
                     systemSpecs.platform || 'Unknown'}
                  </span>
                </div>
                <div className="flex items-center justify-between py-2 border-b">
                  <span className="text-muted-foreground flex items-center gap-2">
                    <Cpu size={14} />
                    Architecture
                  </span>
                  <span className="text-right">{systemSpecs.architecture || 'Unknown'}</span>
                </div>
                <div className="flex items-center justify-between py-2 border-b">
                  <span className="text-muted-foreground flex items-center gap-2">
                    <Zap size={14} />
                    CPU Cores
                  </span>
                  <span className="text-right">
                    {systemSpecs.browserCores > 0 ? `${systemSpecs.browserCores} cores` : 'Unknown'}
                  </span>
                </div>
                <div className="flex items-center justify-between py-2 border-b">
                  <span className="text-muted-foreground flex items-center gap-2">
                    <HardDrive size={14} />
                    System RAM
                  </span>
                  <span className="text-right">
                    {systemSpecs.totalRAM > 0 ? `${systemSpecs.totalRAM} GB` : 
                     systemSpecs.estimatedMemoryGB ? `~${systemSpecs.estimatedMemoryGB} GB` : 'Unknown'}
                  </span>
                </div>
              </div>
            </div>

            {/* GPU Information */}
            {systemSpecs.cudaDevices && systemSpecs.cudaDevices.length > 0 && (
              <div>
                <h4 className="font-medium mb-3 text-foreground">Graphics & Compute</h4>
                <div className="space-y-2">
                  <div className="flex items-center justify-between py-2 border-b">
                    <span className="text-muted-foreground">CUDA Available</span>
                    <span className="text-right">
                      {systemSpecs.cudaAvailable ? (
                        <span className="text-green-600 dark:text-green-400">✓ Yes</span>
                      ) : (
                        <span className="text-red-600 dark:text-red-400">✗ No</span>
                      )}
                    </span>
                  </div>
                  {systemSpecs.cudaDevices.map((device, index) => (
                    <div key={index} className="flex items-center justify-between py-2 border-b">
                      <span className="text-muted-foreground">GPU {index + 1} VRAM</span>
                      <span className="text-right">{device.vram} GB</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Runtime Information */}
            <div>
              <h4 className="font-medium mb-3 text-foreground">Runtime Environment</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {systemSpecs.electronVersion && (
                  <div className="flex items-center justify-between py-2 border-b">
                    <span className="text-muted-foreground">Electron</span>
                    <span className="text-right">v{systemSpecs.electronVersion}</span>
                  </div>
                )}
                <div className="flex items-center justify-between py-2 border-b">
                  <span className="text-muted-foreground">Node.js</span>
                  <span className="text-right">
                    {systemSpecs.nodeVersion !== 'Unknown' ? `v${systemSpecs.nodeVersion}` : systemSpecs.nodeVersion}
                  </span>
                </div>
                <div className="flex items-center justify-between py-2 border-b">
                  <span className="text-muted-foreground">Language</span>
                  <span className="text-right">{systemSpecs.language}</span>
                </div>
                <div className="flex items-center justify-between py-2 border-b">
                  <span className="text-muted-foreground">Timezone</span>
                  <span className="text-right">{systemSpecs.timezone}</span>
                </div>
                <div className="flex items-center justify-between py-2">
                  <span className="text-muted-foreground">Display</span>
                  <span className="text-right text-xs">{systemSpecs.screenResolution}</span>
                </div>
              </div>
            </div>

            {/* User Agent (collapsible) */}
            <details className="pt-3 border-t">
              <summary className="cursor-pointer text-muted-foreground hover:text-foreground text-xs">
                Show User Agent String
              </summary>
              <div className="mt-2 p-2 bg-muted rounded text-xs font-mono break-all">
                {systemSpecs.userAgent}
              </div>
            </details>
          </div>
        ) : (
          <div className="text-sm text-muted-foreground">
            <div className="flex items-center gap-2">
              <span className="text-yellow-600 dark:text-yellow-400">⚠</span>
              Failed to detect system specifications
            </div>
          </div>
        )}
      </div>

      <div className="bg-card border rounded-lg p-4">
        <h3 className="font-semibold mb-4">Resources</h3>
        <div className="space-y-2 opacity-50">
          <a href="#" className="block text-sm text-muted-foreground cursor-not-allowed" onClick={(e) => e.preventDefault()}>
            📖 Documentation <span className="text-xs text-orange-600 bg-orange-100 dark:text-orange-400 dark:bg-orange-900/30 px-1.5 py-0.5 rounded-full font-medium ml-2">PLACEHOLDER</span>
          </a>
          <a href="#" className="block text-sm text-muted-foreground cursor-not-allowed" onClick={(e) => e.preventDefault()}>
            💬 Community Support <span className="text-xs text-orange-600 bg-orange-100 dark:text-orange-400 dark:bg-orange-900/30 px-1.5 py-0.5 rounded-full font-medium ml-2">PLACEHOLDER</span>
          </a>
          <a href="#" className="block text-sm text-muted-foreground cursor-not-allowed" onClick={(e) => e.preventDefault()}>
            🐛 Bug Reports <span className="text-xs text-orange-600 bg-orange-100 dark:text-orange-400 dark:bg-orange-900/30 px-1.5 py-0.5 rounded-full font-medium ml-2">PLACEHOLDER</span>
          </a>
          <a href="#" className="block text-sm text-muted-foreground cursor-not-allowed" onClick={(e) => e.preventDefault()}>
            💡 Feature Requests <span className="text-xs text-orange-600 bg-orange-100 dark:text-orange-400 dark:bg-orange-900/30 px-1.5 py-0.5 rounded-full font-medium ml-2">PLACEHOLDER</span>
          </a>
        </div>
      </div>
    </div>
  );
}

export default function SettingsScreen() {
  const [activeTab, setActiveTab] = useState<SettingsTab>('api');
  const { settings } = useChatStore();

  const apiKeyCount = Object.keys(settings.apiKeys).length;
  const activeKeyCount = Object.values(settings.apiKeys).filter(key => key.isActive).length;

  const tabs: Array<{
    id: SettingsTab;
    icon: React.ReactNode;
    label: string;
    description: string;
    badge?: string;
  }> = [
    {
      id: 'api',
      icon: <Key size={16} />,
      label: 'API Keys',
      description: 'Manage AI provider credentials',
      badge: apiKeyCount > 0 ? `${activeKeyCount}/${apiKeyCount}` : undefined,
    },
    {
      id: 'model',
      icon: <Sliders size={16} />,
      label: 'Model Settings',
      description: 'Configure generation parameters',
    },
    {
      id: 'system',
      icon: <Monitor size={16} />,
      label: 'System',
      description: 'Performance and storage settings',
    },
    {
      id: 'notifications',
      icon: <Bell size={16} />,
      label: 'Notifications',
      description: 'Control alerts and notifications',
    },
    {
      id: 'about',
      icon: <HelpCircle size={16} />,
      label: 'About',
      description: 'App info and resources',
    },
  ];

  const renderContent = () => {
    switch (activeTab) {
      case 'api':
        return <ApiSettings />;
      case 'model':
        return <ModelSettings />;
      case 'system':
        return <SystemSettings />;
      case 'notifications':
        return <NotificationSettings />;
      case 'about':
        return <AboutSettings />;
      default:
        return <ApiSettings />;
    }
  };

  return (
    <div className="h-full flex">
      {/* Sidebar */}
      <div className="w-64 border-r bg-card/50 p-4">
        <div className="mb-6">
          <h1 className="text-xl font-bold flex items-center gap-2">
            <Settings size={20} />
            Settings
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Configure your Chatterley experience
          </p>
        </div>

        <div className="space-y-2">
          {tabs.map((tab) => (
            <TabButton
              key={tab.id}
              id={tab.id}
              icon={tab.icon}
              label={tab.label}
              description={tab.description}
              badge={tab.badge}
              isActive={activeTab === tab.id}
              onClick={setActiveTab}
            />
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-6 max-w-4xl">
          {renderContent()}
        </div>
      </div>
    </div>
  );
}
