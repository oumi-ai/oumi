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
import ConfirmationDialog from '@/components/ui/ConfirmationDialog';
import SettingsScreen from '@/components/settings/SettingsScreen';
import ChatHistorySidebar from '@/components/history/ChatHistorySidebar';
import SearchHistoryWindow from '@/components/search/SearchHistoryWindow';
import { ChatInterfaceRef } from '@/components/chat/ChatInterface';
import ToastContainer from '@/components/ui/ToastContainer';

export default function AppLayout() {
  const [isBranchTreeExpanded, setIsBranchTreeExpanded] = React.useState(true);
  const [isControlPanelExpanded, setIsControlPanelExpanded] = React.useState(true);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = React.useState(false);
  const [showChatHistory, setShowChatHistory] = React.useState(true);
  const [isInitialized, setIsInitialized] = React.useState(false);
  const [showSettings, setShowSettings] = React.useState(false);
  const [showSearchHistory, setShowSearchHistory] = React.useState(false);
  const [showResetConfirmation, setShowResetConfirmation] = React.useState(false);
  const [resetWithBackup, setResetWithBackup] = React.useState(false);
  const [isResetting, setIsResetting] = React.useState(false);
  const [resetProgress, setResetProgress] = React.useState<string[]>([]);
  const [resetSuccess, setResetSuccess] = React.useState<string | undefined>(undefined);
  const { clearMessages, currentBranchId, generationParams, setCurrentBranch, setMessages, getCurrentSessionId } = useChatStore();
  // Note: setBranches is no longer needed as branches are derived on demand
  const { executeCommand, isExecuting } = useConversationCommand();
  const chatInterfaceRef = React.useRef<ChatInterfaceRef | null>(null);

  // Keep latest function references for handlers registered once
  const executeCommandRef = React.useRef(executeCommand);
  React.useEffect(() => { executeCommandRef.current = executeCommand; }, [executeCommand]);
  const clearMessagesRef = React.useRef(clearMessages);
  React.useEffect(() => { clearMessagesRef.current = clearMessages; }, [clearMessages]);

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
        setIsBranchTreeExpanded(prev => !prev);
      };

      const handleToggleControlPanel = () => {
        console.log('🔧 [AppLayout] Toggling Control Panel from menu');
        setIsControlPanelExpanded(prev => !prev);
      };

      const handleClearConversationMenu = () => {
        console.log('🔧 [AppLayout] Clear Conversation from menu');
        handleClearConversation();
      };

      const handleNewChat = async () => {
        console.log('🔧 [AppLayout] New Chat from menu');
        try {
          // Clear the current conversation and start fresh
          clearMessagesRef.current();
          const result = await executeCommandRef.current('clear', [], COMMAND_CONFIGS.clear);
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

      // Removed save conversation menu handler per request

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

      const handleResetHistory = () => {
        console.log('🔧 [AppLayout] Reset History from menu');
        setResetWithBackup(false);
        setShowResetConfirmation(true);
      };

      const handleBackupAndResetHistory = () => {
        console.log('🔧 [AppLayout] Backup and Reset History from menu');
        setResetWithBackup(true);
        setShowResetConfirmation(true);
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
        // Removed: menu:save-conversation handler registration
        window.electronAPI.onMenuMessage('menu:regenerate', handleRegenerateLastResponse);
        window.electronAPI.onMenuMessage('menu:stop-generation', handleStopGeneration);
        window.electronAPI.onMenuMessage('menu:reset-history', handleResetHistory);
        window.electronAPI.onMenuMessage('menu:backup-and-reset-history', handleBackupAndResetHistory);
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
          // Removed: menu:save-conversation handler cleanup
          window.electronAPI.removeMenuListener('menu:regenerate', handleRegenerateLastResponse);
          window.electronAPI.removeMenuListener('menu:stop-generation', handleStopGeneration);
          window.electronAPI.removeMenuListener('menu:reset-history', handleResetHistory);
          window.electronAPI.removeMenuListener('menu:backup-and-reset-history', handleBackupAndResetHistory);
        }
      };
    };

    const cleanup = setupElectronIntegration();

    return cleanup;
  }, []);


  // Initialize app state from backend on first load
  React.useEffect(() => {
    const initializeApp = async () => {
      try {
        console.log('🔄 Initializing app state from backend...');
        
        // Load branches to get the current branch
        const sessionId = getCurrentSessionId();
        const branchesResponse = await apiClient.getBranches(sessionId);
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
          // Note: setBranches is no longer needed since branches are derived on demand
          // The branches will be available via getBranches()
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
  }, [isInitialized, currentBranchId, setCurrentBranch]);

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

  // Reset history handlers
  const createBackup = async (): Promise<boolean> => {
    try {
      setResetProgress(prev => [...prev, "Starting backup..."]);
      
      // Get current session ID
      const sessionId = getCurrentSessionId();
      
      // Get all branches for the current session
      const branchesResponse = await apiClient.getBranches(sessionId);
      if (!branchesResponse.success || !branchesResponse.data) {
        throw new Error("Failed to retrieve branches");
      }
      const data = branchesResponse.data;
      const branches = data?.branches ?? [];
      setResetProgress(prev => [...prev, `Found ${branches.length} branches to backup`]);
      
      // Prepare backup data structure
      const backupData: {
        version: string;
        timestamp: string;
        session_id: string;
        branches: Record<string, any>;
      } = {
        version: "1.0",
        timestamp: new Date().toISOString(),
        session_id: sessionId,
        branches: {}
      };
      
      // For each branch, get its conversation history
      for (const branch of branches) {
        setResetProgress(prev => [...prev, `Backing up branch: ${branch.name || branch.id}`]);
        
        const conversationResponse = await apiClient.getConversation(sessionId, branch.id);
        if (conversationResponse.success) {
          backupData.branches[branch.id] = {
            name: branch.name || branch.id,
            created_at: branch.createdAt,
            last_active: branch.lastActive || branch.createdAt,
            conversation: (conversationResponse.data?.conversation) || []
          };
        }
      }
      
      // Save backup to file
      const filename = `chat-backup-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
      const saved = await apiClient.saveConversationToFile(backupData as any, filename);
      
      if (saved) {
        setResetProgress(prev => [...prev, `Backup saved successfully to ${filename}`]);
        return true;
      } else {
        throw new Error("Failed to save backup file");
      }
    } catch (error) {
      console.error("Backup creation error:", error);
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      setResetProgress(prev => [...prev, `Backup error: ${errorMessage}`]);
      return false;
    }
  };
  
  const resetHistory = async (): Promise<boolean> => {
    try {
      setResetProgress(prev => [...prev, "Starting reset operation..."]);

      // Clear UI state first for immediate feedback
      clearMessages();

      const sessionId = getCurrentSessionId();
      setResetProgress(prev => [...prev, `Resetting session: ${sessionId}`]);

      const response = await fetch('/v1/oumi/reset_history', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          confirm_phrase: "RESET",
          scope: "current",
        }),
      });

      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();

      if (result.success) {
        setResetProgress(prev => [...prev, `Reset successful. Deleted: ${JSON.stringify(result.deleted || {})}`]);

        // Re-initialize state from backend (best-effort)
        try {
          const branchesResponse = await apiClient.getBranches(sessionId);
          if (branchesResponse.success && branchesResponse.data) {
            const { current_branch } = branchesResponse.data;
            if (current_branch && current_branch !== currentBranchId) {
              setCurrentBranch(current_branch);
            }
          }
        } catch (err) {
          // Non-fatal; continue with main branch implied
          setResetProgress(prev => [...prev, `Warning: Could not get branch information. Using main branch.`]);
        }

        setResetSuccess("Chat history has been successfully reset.");
        return true;
      } else {
        throw new Error(result.message || "Reset operation failed");
      }
    } catch (error: any) {
      console.error("Reset error:", error);
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      setResetProgress(prev => [...prev, `Reset error: ${errorMessage}`]);
      return false;
    }
  };
  
  const handleResetConfirm = async () => {
    setIsResetting(true);
    try {
      await resetHistory();
    } finally {
      setTimeout(() => {
        setIsResetting(false);
        // Auto-close after 3 seconds
        setTimeout(() => {
          setShowResetConfirmation(false);
          setResetProgress([]);
          setResetSuccess(undefined);
        }, 3000);
      }, 500);
    }
  };
  
  const handleBackupAndResetConfirm = async () => {
    setIsResetting(true);
    try {
      const backupSuccess = await createBackup();
      if (backupSuccess) {
        await resetHistory();
      } else {
        setResetProgress(prev => [...prev, "Reset operation cancelled due to backup failure"]);
      }
    } finally {
      setTimeout(() => {
        setIsResetting(false);
        // Auto-close after 3 seconds if successful
        if (resetSuccess) {
          setTimeout(() => {
            setShowResetConfirmation(false);
            setResetProgress([]);
            setResetSuccess(undefined);
          }, 3000);
        }
      }, 500);
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
    <div className="flex h-screen overflow-hidden bg-background">
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
              <button
                onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
                className="p-2 rounded-md hover:bg-accent text-muted-foreground hover:text-foreground"
                title={isSidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
              >
                {isSidebarCollapsed ? <PanelLeft size={18} /> : <PanelLeftClose size={18} />}
              </button>
          </div>
        </div>

      </div>

      {/* Main content */}
      <div className="flex flex-1 min-h-0 pt-16">
        {/* Control panel sidebar */}
        <div className={`transition-all duration-200 ${
          isControlPanelExpanded ? 'w-80' : 'w-16'
        }`}>
          <ControlPanel 
            className="h-full" 
            isCollapsed={!isControlPanelExpanded}
            onToggleCollapse={() => setIsControlPanelExpanded(!isControlPanelExpanded)}
          />
        </div>

        {/* Chat interface */}
        <div className="flex-1 min-h-0 transition-all duration-200">
          <ChatInterface 
            className="h-full" 
            onRef={(ref) => { chatInterfaceRef.current = ref; }}
          />
        </div>

        {/* Right sidebar with Branch tree and Chat history */}
        <div className={`transition-all duration-200 w-80 ${
          (isBranchTreeExpanded || showChatHistory) && !isSidebarCollapsed ? '' : 'hidden'
        }`}>
          <div className="flex flex-col h-full overflow-hidden">
            {/* Right sidebar header with toggle button */}
            <div className="bg-card border-b p-3 flex items-center justify-between sticky top-0 z-10">
              <h3 className="font-medium text-foreground text-sm">Sidebar</h3>
              <div className="flex items-center gap-1">
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
                  onClick={() => setIsSidebarCollapsed(true)}
                  className="p-1 hover:bg-muted rounded transition-colors text-muted-foreground hover:text-foreground"
                  title="Collapse sidebar"
                >
                  <PanelLeftClose size={16} />
                </button>
              </div>
            </div>
            {/* Branch Tree - shows when branch tree is expanded */}
            {isBranchTreeExpanded && (
              <div className={`${showChatHistory ? 'flex-1' : 'h-full'} min-h-0 overflow-y-auto overscroll-contain`}>
                <BranchTree className="h-full" />
              </div>
            )}
            
            {/* Chat History - shows when chat history is enabled */}
            {showChatHistory && (
              <div className={`${isBranchTreeExpanded ? 'flex-1' : 'h-full'} min-h-0 overflow-y-auto overscroll-contain ${isBranchTreeExpanded ? 'border-t' : ''}`}>
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

      {/* Reset Chat History Confirmation Dialog */}
      <ConfirmationDialog
        isOpen={showResetConfirmation}
        title="Reset Chat History"
        message="Are you sure you want to reset all chat history?"
        detail="This will permanently delete all threads, conversations, messages, attachments, and vector indexes. This action cannot be undone."
        confirmationText="RESET"
        confirmLabel="Reset"
        alternateLabel={resetWithBackup ? "Backup and Reset" : undefined}
        dangerous={true}
        onConfirm={handleResetConfirm}
        onAlternate={resetWithBackup ? handleBackupAndResetConfirm : undefined}
        onCancel={() => setShowResetConfirmation(false)}
        isLoading={isResetting}
        progressDetails={resetProgress}
        successMessage={resetSuccess}
      />

      {/* Global toasts */}
      <ToastContainer />
    </div>
  );
}
