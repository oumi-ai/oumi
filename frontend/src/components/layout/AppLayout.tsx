/**
 * Main application layout with chat and branch tree
 */

"use client";

import React from 'react';
import ChatInterface from '@/components/chat/ChatInterface';
import BranchTree from '@/components/branches/BranchTree';
import ControlPanel from '@/components/layout/ControlPanel';
import { useChatStore } from '@/lib/store';
import apiClient from '@/lib/unified-api';
import { useConversationCommand, COMMAND_CONFIGS } from '@/hooks/useConversationCommand';
import { Maximize2, Minimize2, Settings, RotateCcw, PanelLeft, PanelLeftClose } from 'lucide-react';

export default function AppLayout() {
  const [isBranchTreeExpanded, setIsBranchTreeExpanded] = React.useState(true);
  const [isControlPanelExpanded, setIsControlPanelExpanded] = React.useState(true);
  const [isInitialized, setIsInitialized] = React.useState(false);
  const { clearMessages, currentBranchId, generationParams, setBranches, setCurrentBranch, setMessages } = useChatStore();
  const { executeCommand, isExecuting } = useConversationCommand();

  // Handle menu messages from Electron
  React.useEffect(() => {
    if (!apiClient.isElectron || !apiClient.isElectron()) return;

    const handleResetWelcomeMessage = async () => {
      await handleResetWelcomeSettings();
    };

    // Listen for menu messages (Electron-specific)
    if (typeof window !== 'undefined' && window.electronAPI) {
      window.electronAPI.onMenuMessage('menu:reset-welcome-settings', handleResetWelcomeMessage);
    }

    return () => {
      if (typeof window !== 'undefined' && window.electronAPI) {
        window.electronAPI.removeMenuListener('menu:reset-welcome-settings', handleResetWelcomeMessage);
      }
    };
  }, []);

  // Handle welcome settings reset
  const handleResetWelcomeSettings = async () => {
    try {
      const result = await apiClient.resetWelcomeSettings();
      if (result.success) {
        // Show success feedback - could use a toast or alert
        if (window.confirm('Welcome settings have been reset! The app will reload to show the welcome screen.')) {
          // Reload the app to show welcome screen
          if (apiClient.isElectron && apiClient.isElectron()) {
            apiClient.reload();
          } else {
            window.location.reload();
          }
        }
      } else {
        alert('Failed to reset welcome settings. Please try again.');
      }
    } catch (error) {
      console.error('Error resetting welcome settings:', error);
      alert('Failed to reset welcome settings. Please try again.');
    }
  };

  // Initialize app state from backend on first load
  React.useEffect(() => {
    const initializeApp = async () => {
      try {
        console.log('🔄 Initializing app state from backend...');
        
        // Load branches to get the current branch
        const branchesResponse = await apiClient.getBranches('default');
        if (branchesResponse.success && branchesResponse.data) {
          const { branches, current_branch } = branchesResponse.data;
          
          // Transform backend branches to frontend format
          const transformedBranches = branches.map((branch: any) => ({
            id: branch.id,
            name: branch.name,
            isActive: branch.id === current_branch,
            messageCount: branch.message_count || 0,
            createdAt: branch.created_at,
            lastActive: branch.last_active || branch.created_at,
            preview: branch.message_count > 0 ? `${branch.message_count} messages` : 'Empty branch'
          }));
          
          console.log(`📋 Loaded ${transformedBranches.length} branches, current: ${current_branch}`);
          setBranches(transformedBranches);
          if (current_branch && current_branch !== currentBranchId) {
            setCurrentBranch(current_branch);
          }
        }
        
        setIsInitialized(true);
        console.log('✅ App state initialized');
      } catch (error) {
        console.error('❌ Failed to initialize app state:', error);
        // Still mark as initialized to prevent infinite loading
        setIsInitialized(true);
      }
    };

    if (!isInitialized) {
      initializeApp();
    }
  }, [isInitialized, currentBranchId, setBranches, setCurrentBranch]);

  const handleClearConversation = async () => {
    if (confirm('Are you sure you want to clear this conversation? This action cannot be undone.')) {
      try {
        // Clear messages in the UI immediately for responsiveness
        clearMessages();
        
        // Execute clear command which will refresh conversation and branches
        const result = await executeCommand('clear', [], COMMAND_CONFIGS.clear);
        
        if (!result.success && result.message) {
          console.error('Failed to clear conversation:', result.message);
        }
      } catch (error) {
        console.error('Error clearing conversation:', error);
        // Messages were already cleared in UI, so we don't revert that
      }
    }
  };

  // Show loading state during initialization
  if (!isInitialized) {
    return (
      <div className="flex h-screen bg-background items-center justify-center">
        <div className="text-center space-y-4">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
          <p className="text-muted-foreground">Loading conversation...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen bg-background">
      {/* Header */}
      <div className="absolute top-0 left-0 right-0 z-10 bg-card border-b border-border shadow-sm">
        <div className="flex items-center justify-between px-4 py-3">
          {/* Left section */}
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-semibold text-foreground">
              Oumi WebChat
            </h1>
            <div className="text-sm text-muted-foreground">
              Branch: {currentBranchId}
            </div>
          </div>

          {/* Right section */}
          <div className="flex items-center gap-2">
            {/* Clear conversation */}
            <button
              onClick={handleClearConversation}
              className="p-2 rounded-md hover:bg-accent text-muted-foreground hover:text-foreground disabled:opacity-50"
              title="Clear conversation"
              disabled={isExecuting}
            >
              <RotateCcw size={18} className={isExecuting ? 'animate-spin' : ''} />
            </button>

            {/* Control panel toggle */}
            <button
              onClick={() => setIsControlPanelExpanded(!isControlPanelExpanded)}
              className={`p-2 rounded-md hover:bg-accent text-muted-foreground hover:text-foreground ${
                isControlPanelExpanded ? 'bg-accent' : ''
              }`}
              title={isControlPanelExpanded ? 'Hide control panel' : 'Show control panel'}
            >
              {isControlPanelExpanded ? <PanelLeftClose size={18} /> : <PanelLeft size={18} />}
            </button>

            {/* Branch tree toggle */}
            <button
              onClick={() => setIsBranchTreeExpanded(!isBranchTreeExpanded)}
              className="p-2 rounded-md hover:bg-accent text-muted-foreground hover:text-foreground"
              title={isBranchTreeExpanded ? 'Collapse branches' : 'Expand branches'}
            >
              {isBranchTreeExpanded ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
            </button>
          </div>
        </div>

      </div>

      {/* Main content */}
      <div className="flex flex-1 pt-16">
        {/* Control panel sidebar */}
        <div className={`transition-all duration-200 ${
          isControlPanelExpanded ? 'w-80' : 'w-16'
        }`}>
          <ControlPanel 
            className="min-h-screen" 
            isCollapsed={!isControlPanelExpanded}
            onToggleCollapse={() => setIsControlPanelExpanded(!isControlPanelExpanded)}
          />
        </div>

        {/* Chat interface */}
        <div className={`flex-1 transition-all duration-200 ${
          isBranchTreeExpanded ? 'mr-80' : ''
        }`}>
          <ChatInterface className="min-h-screen" />
        </div>

        {/* Branch tree sidebar */}
        <div className={`fixed right-0 top-16 w-80 min-h-screen transition-transform duration-200 ${
          isBranchTreeExpanded ? 'translate-x-0' : 'translate-x-full'
        }`}>
          <BranchTree className="min-h-screen" />
        </div>
      </div>
    </div>
  );
}