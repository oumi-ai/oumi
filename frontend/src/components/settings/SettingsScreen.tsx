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
  EyeOff
} from 'lucide-react';
import ApiSettings from './ApiSettings';
import ModelSettings from './ModelSettings';
import { useChatStore } from '@/lib/store';

type SettingsTab = 'api' | 'model' | 'system' | 'notifications' | 'about';

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
        <h3 className="font-semibold">Performance</h3>
        
        <label className="flex items-center justify-between">
          <div>
            <div className="font-medium text-sm">Enable Hardware Acceleration</div>
            <div className="text-xs text-muted-foreground">
              Use GPU acceleration when available
            </div>
          </div>
          <input type="checkbox" defaultChecked className="rounded" />
        </label>

        <label className="flex items-center justify-between">
          <div>
            <div className="font-medium text-sm">Stream Responses</div>
            <div className="text-xs text-muted-foreground">
              Show responses as they're generated
            </div>
          </div>
          <input type="checkbox" defaultChecked className="rounded" />
        </label>

        <div>
          <label className="block font-medium text-sm mb-2">Response Buffer Size</label>
          <select className="w-full px-3 py-2 bg-background border rounded-lg text-sm">
            <option>Small (1KB)</option>
            <option>Medium (4KB)</option>
            <option selected>Large (8KB)</option>
          </select>
        </div>
      </div>

      <div className="bg-card border rounded-lg p-4 space-y-4">
        <h3 className="font-semibold">Storage</h3>
        
        <div>
          <label className="block font-medium text-sm mb-2">Chat History Retention</label>
          <select className="w-full px-3 py-2 bg-background border rounded-lg text-sm">
            <option>1 week</option>
            <option>1 month</option>
            <option selected>3 months</option>
            <option>1 year</option>
            <option>Forever</option>
          </select>
        </div>

        <label className="flex items-center justify-between">
          <div>
            <div className="font-medium text-sm">Auto-save Conversations</div>
            <div className="text-xs text-muted-foreground">
              Automatically save conversations locally
            </div>
          </div>
          <input type="checkbox" defaultChecked className="rounded" />
        </label>

        <div className="pt-3 border-t">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Cache Size</span>
            <span>247 MB</span>
          </div>
          <button className="mt-2 text-sm text-primary hover:underline">
            Clear Cache
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
        
        <label className="flex items-center justify-between">
          <div>
            <div className="font-medium text-sm">Model Download Complete</div>
            <div className="text-xs text-muted-foreground">
              Notify when model downloads finish
            </div>
          </div>
          <input type="checkbox" defaultChecked className="rounded" />
        </label>

        <label className="flex items-center justify-between">
          <div>
            <div className="font-medium text-sm">Update Available</div>
            <div className="text-xs text-muted-foreground">
              Notify when app updates are available
            </div>
          </div>
          <input type="checkbox" defaultChecked className="rounded" />
        </label>

        <label className="flex items-center justify-between">
          <div>
            <div className="font-medium text-sm">System Errors</div>
            <div className="text-xs text-muted-foreground">
              Show notifications for system errors
            </div>
          </div>
          <input type="checkbox" defaultChecked className="rounded" />
        </label>
      </div>
    </div>
  );
}

function AboutSettings() {
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
          <button className="w-full py-2 px-4 bg-muted hover:bg-muted/80 rounded-lg text-sm transition-colors">
            Check for Updates
          </button>
          <div className="flex gap-2">
            <button className="flex-1 py-2 px-4 bg-muted hover:bg-muted/80 rounded-lg text-sm transition-colors">
              View Logs
            </button>
            <button className="flex-1 py-2 px-4 bg-muted hover:bg-muted/80 rounded-lg text-sm transition-colors">
              Report Issue
            </button>
          </div>
        </div>
      </div>

      <div className="bg-card border rounded-lg p-4">
        <h3 className="font-semibold mb-4">Resources</h3>
        <div className="space-y-2">
          <a href="#" className="block text-sm text-primary hover:underline">
            📖 Documentation
          </a>
          <a href="#" className="block text-sm text-primary hover:underline">
            💬 Community Support
          </a>
          <a href="#" className="block text-sm text-primary hover:underline">
            🐛 Bug Reports
          </a>
          <a href="#" className="block text-sm text-primary hover:underline">
            💡 Feature Requests
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