/**
 * Main application layout with chat and branch tree
 */

"use client";

import React from 'react';
import ChatInterface from '@/components/chat/ChatInterface';
import BranchTree from '@/components/branches/BranchTree';
import ControlPanel from '@/components/layout/ControlPanel';
import SystemChangeWarning from '@/components/monitoring/SystemChangeWarning';
import { useChatStore } from '@/lib/store';
import apiClient from '@/lib/unified-api';
import { useConversationCommand, COMMAND_CONFIGS } from '@/hooks/useConversationCommand';
import { Maximize2, Minimize2, Settings, RotateCcw, PanelLeft, PanelLeftClose, X, Search, History } from 'lucide-react';
import SettingsScreen from '@/components/settings/SettingsScreen';
import ChatHistorySidebar from '@/components/history/ChatHistorySidebar';
import SearchHistoryWindow from '@/components/search/SearchHistoryWindow';
import { ChatInterfaceRef } from '@/components/chat/ChatInterface';

export default function AppLayout() {
  const [isBranchTreeExpanded, setIsBranchTreeExpanded] = React.useState(true);
  const [isControlPanelExpanded, setIsControlPanelExpanded] = React.useState(true);
  const [showChatHistory, setShowChatHistory] = React.useState(true);
  const [isInitialized, setIsInitialized] = React.useState(false);
  const [showSettings, setShowSettings] = React.useState(false);
  const [showSearchHistory, setShowSearchHistory] = React.useState(false);
  const { clearMessages, currentBranchId, generationParams, setBranches, setCurrentBranch, setMessages } = useChatStore();
  const { executeCommand, isExecuting } = useConversationCommand();
  const chatInterfaceRef = React.useRef<ChatInterfaceRef | null>(null);

  // Handle keyboard shortcuts
  React.useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // ESC key to close modals
      if (event.key === 'Escape') {
        if (showSearchHistory) {
          setShowSearchHistory(false);
        } else if (showSettings) {
          setShowSettings(false);
        }
        return;
      }
      
      // Ctrl+F to open search
      if (event.ctrlKey && event.key === 'f') {
        event.preventDefault(); // Prevent browser find
        setShowSearchHistory(true);
        return;
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [showSettings, showSearchHistory]);

  // Handle React ready state and menu messages from Electron
  React.useEffect(() => {
    console.log('🔧 [AppLayout] React component mounted, setting up...');
    
    const setupElectronIntegration = () => {
      if (!apiClient.isElectron || !apiClient.isElectron()) {
        console.log('🔧 [AppLayout] Not in Electron environment');
        return undefined;
      }

      // Setup menu message handlers
      const handleModelSettings = () => {
        console.log('🔧 [AppLayout] Opening Model Settings from menu');
        setShowSettings(true);
      };

      const handleToggleBranchTree = () => {
        console.log('🔧 [AppLayout] Toggling Branch Tree from menu');
        setIsBranchTreeExpanded(!isBranchTreeExpanded);
      };

      const handleToggleControlPanel = () => {
        console.log('🔧 [AppLayout] Toggling Control Panel from menu');
        setIsControlPanelExpanded(!isControlPanelExpanded);
      };

      const handleClearConversationMenu = () => {
        console.log('🔧 [AppLayout] Clear Conversation from menu');
        handleClearConversation();
      };

      const handleNewChat = async () => {
        console.log('🔧 [AppLayout] New Chat from menu');
        try {
          // Clear the current conversation and start fresh
          clearMessages();
          const result = await executeCommand('clear', [], COMMAND_CONFIGS.clear);
          if (!result.success && result.message) {
            console.error('Failed to start new chat:', result.message);
          }
        } catch (error) {
          console.error('Error starting new chat:', error);
        }
      };

      const handlePreferences = () => {
        console.log('🔧 [AppLayout] Opening Preferences from menu');
        setShowSettings(true);
      };

      const handleFind = () => {
        console.log('🔧 [AppLayout] Opening Find/Search from menu');
        setShowSearchHistory(true);
      };

      const handleSaveConversation = async (filePath: string) => {
        console.log('🔧 [AppLayout] Save Conversation from menu:', filePath);
        try {
          const response = await apiClient.executeCommand('save', [filePath]);
          if (response.success) {
            console.log('Conversation saved successfully');
            // Could show a toast notification here instead of alert
            alert('Conversation saved successfully!');
          } else {
            console.error('Failed to save conversation:', response.message);
            alert('Failed to save conversation: ' + (response.message || 'Unknown error'));
          }
        } catch (error) {
          console.error('Error saving conversation:', error);
          alert('Error saving conversation');
        }
      };

      const handleRegenerateLastResponse = () => {
        console.log('🔧 [AppLayout] Regenerate Last Response from menu');
        if (chatInterfaceRef.current) {
          chatInterfaceRef.current.regenerateLastResponse();
        }
      };

      const handleStopGeneration = () => {
        console.log('🔧 [AppLayout] Stop Generation from menu');
        if (chatInterfaceRef.current) {
          chatInterfaceRef.current.stopGeneration();
        }
      };

      // Register menu handlers
      if (window.electronAPI) {
        window.electronAPI.onMenuMessage('menu:model-settings', handleModelSettings);
        window.electronAPI.onMenuMessage('menu:toggle-branch-tree', handleToggleBranchTree);
        window.electronAPI.onMenuMessage('menu:toggle-control-panel', handleToggleControlPanel);
        window.electronAPI.onMenuMessage('menu:clear-conversation', handleClearConversationMenu);
        window.electronAPI.onMenuMessage('menu:new-chat', handleNewChat);
        window.electronAPI.onMenuMessage('menu:preferences', handlePreferences);
        window.electronAPI.onMenuMessage('menu:find', handleFind);
        window.electronAPI.onMenuMessage('menu:save-conversation', handleSaveConversation);
        window.electronAPI.onMenuMessage('menu:regenerate', handleRegenerateLastResponse);
        window.electronAPI.onMenuMessage('menu:stop-generation', handleStopGeneration);
      }

      console.log('🔧 [AppLayout] Electron menu handlers registered');

      // Cleanup function
      return () => {
        if (window.electronAPI) {
          window.electronAPI.removeMenuListener('menu:model-settings', handleModelSettings);
          window.electronAPI.removeMenuListener('menu:toggle-branch-tree', handleToggleBranchTree);
          window.electronAPI.removeMenuListener('menu:toggle-control-panel', handleToggleControlPanel);
          window.electronAPI.removeMenuListener('menu:clear-conversation', handleClearConversationMenu);
          window.electronAPI.removeMenuListener('menu:new-chat', handleNewChat);
          window.electronAPI.removeMenuListener('menu:preferences', handlePreferences);
          window.electronAPI.removeMenuListener('menu:find', handleFind);
          window.electronAPI.removeMenuListener('menu:save-conversation', handleSaveConversation);
          window.electronAPI.removeMenuListener('menu:regenerate', handleRegenerateLastResponse);
          window.electronAPI.removeMenuListener('menu:stop-generation', handleStopGeneration);
        }
      };
    };

    const cleanup = setupElectronIntegration();

    return cleanup;
  }, [isBranchTreeExpanded, isControlPanelExpanded, clearMessages, executeCommand]);


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
            <img 
              src="./images/chatterley-logo.png" 
              alt="Chatterley Logo"
              className="w-8 h-8"
              onError={(e) => {
                // Hide if logo not found
                e.currentTarget.style.display = 'none';
              }}
            />
            <h1 className="text-xl font-semibold text-foreground">
              Chatterley: Powered by Oumi
            </h1>
            <div className="text-sm text-muted-foreground">
              Branch: {currentBranchId}
            </div>
          </div>

          {/* Right section */}
          <div className="flex items-center gap-2">
            {/* Search & History */}
            <button
              onClick={() => setShowSearchHistory(true)}
              className="p-2 rounded-md hover:bg-accent text-muted-foreground hover:text-foreground"
              title="Search & History (Ctrl+F)"
            >
              <Search size={18} />
            </button>

            {/* Settings */}
            <button
              onClick={() => setShowSettings(true)}
              className="p-2 rounded-md hover:bg-accent text-muted-foreground hover:text-foreground"
              title="Settings"
            >
              <Settings size={18} />
            </button>

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
          (isBranchTreeExpanded || showChatHistory) ? 'mr-80' : ''
        }`}>
          <ChatInterface 
            className="min-h-screen" 
            onRef={(ref) => { chatInterfaceRef.current = ref; }}
          />
        </div>

        {/* Right sidebar with Branch tree and Chat history */}
        <div className={`fixed right-0 top-16 w-80 min-h-screen transition-transform duration-200 ${
          (isBranchTreeExpanded || showChatHistory) ? 'translate-x-0' : 'translate-x-full'
        }`}>
          <div className="h-full flex flex-col">
            {/* Right sidebar header with toggle button */}
            <div className="bg-card border-b p-3 flex items-center justify-between">
              <h3 className="font-medium text-foreground text-sm">Sidebar</h3>
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
            </div>
            {/* Branch Tree - shows when branch tree is expanded */}
            {isBranchTreeExpanded && (
              <div className={`${showChatHistory ? 'flex-1' : 'h-full'} min-h-0`}>
                <BranchTree className="h-full" />
              </div>
            )}
            
            {/* Chat History - shows when chat history is enabled */}
            {showChatHistory && (
              <div className={`${isBranchTreeExpanded ? 'flex-1' : 'h-full'} min-h-0 ${isBranchTreeExpanded ? 'border-t' : ''}`}>
                <ChatHistorySidebar className="h-full" />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Search & History Window */}
      <SearchHistoryWindow
        isOpen={showSearchHistory}
        onClose={() => setShowSearchHistory(false)}
        onNavigateToMessage={(conversationId, messageId, branchId) => {
          // TODO: Implement navigation to specific message
          console.log('Navigate to message:', { conversationId, messageId, branchId });
        }}
      />

      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="w-full h-full max-w-7xl max-h-[90vh] bg-background border border-border rounded-lg shadow-2xl overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b border-border">
              <h2 className="text-lg font-semibold">Settings</h2>
              <button
                onClick={() => setShowSettings(false)}
                className="p-2 rounded-md hover:bg-accent text-muted-foreground hover:text-foreground"
                title="Close settings"
              >
                <X size={18} />
              </button>
            </div>
            <div className="h-full overflow-hidden">
              <SettingsScreen />
            </div>
          </div>
        </div>
      )}

      {/* System change warning */}
      <SystemChangeWarning />
    </div>
  );
}