/**
 * Control panel with system monitoring, model settings, and branch management
 */

"use client";

import React from 'react';
import { PanelLeft, PanelLeftClose, History } from 'lucide-react';
import SystemMonitor from '@/components/monitoring/SystemMonitor';
import ModelSettings from '@/components/settings/ModelSettings';
import ModelSwitcher from '@/components/settings/ModelSwitcher';
import ChatHistorySidebar from '@/components/history/ChatHistorySidebar';

interface ControlPanelProps {
  className?: string;
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
}

export default function ControlPanel({ 
  className = '', 
  isCollapsed = false,
  onToggleCollapse 
}: ControlPanelProps) {
  const [showChatHistory, setShowChatHistory] = React.useState(false);

  if (isCollapsed) {
    return (
      <div className={`bg-card border-r flex flex-col items-center p-2 space-y-4 ${className}`}>
        <button
          onClick={onToggleCollapse}
          className="p-2 hover:bg-muted rounded-lg transition-colors text-muted-foreground hover:text-foreground"
          title="Expand control panel"
        >
          <PanelLeft size={20} />
        </button>
        
        {/* Collapsed indicators */}
        <div className="space-y-3">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" title="System Monitor" />
          <div className="w-2 h-2 bg-blue-500 rounded-full" title="Model Settings" />
          <div className="w-2 h-2 bg-purple-500 rounded-full" title="Branch Manager" />
          <div className="w-2 h-2 bg-orange-500 rounded-full" title="Chat History" />
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-card border-r flex ${className}`}>
      {/* Main Control Panel */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <h2 className="font-semibold text-foreground">Control Panel</h2>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowChatHistory(!showChatHistory)}
              className={`p-1 hover:bg-muted rounded transition-colors ${
                showChatHistory 
                  ? 'text-orange-600 bg-orange-100 dark:bg-orange-900/30' 
                  : 'text-muted-foreground hover:text-foreground'
              }`}
              title={showChatHistory ? 'Hide chat history' : 'Show chat history'}
            >
              <History size={16} />
            </button>
            <button
              onClick={onToggleCollapse}
              className="p-1 hover:bg-muted rounded transition-colors text-muted-foreground hover:text-foreground"
              title="Collapse panel"
            >
              <PanelLeftClose size={16} />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {/* System Monitoring */}
          <SystemMonitor updateInterval={3000} />
          
          {/* Model Configuration */}
          <ModelSwitcher />
          
          {/* Model Settings */}
          <ModelSettings />
        </div>
      </div>

      {/* Chat History Sidebar */}
      {showChatHistory && (
        <ChatHistorySidebar className="w-96" />
      )}
    </div>
  );
}