/**
 * Main application layout with chat and branch tree
 */

"use client";

import React from 'react';
import ChatInterface from '@/components/chat/ChatInterface';
import BranchTree from '@/components/branches/BranchTree';
import { useChatStore } from '@/lib/store';
import apiClient from '@/lib/api';
import { Maximize2, Minimize2, Settings, RotateCcw } from 'lucide-react';

export default function AppLayout() {
  const [isBranchTreeExpanded, setIsBranchTreeExpanded] = React.useState(true);
  const [isSettingsOpen, setIsSettingsOpen] = React.useState(false);
  const [isInitialized, setIsInitialized] = React.useState(false);
  const { clearMessages, currentBranchId, generationParams, setBranches, setCurrentBranch, setMessages } = useChatStore();

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

  const handleClearConversation = () => {
    if (confirm('Are you sure you want to clear this conversation? This action cannot be undone.')) {
      clearMessages();
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
    <div className="flex h-screen bg-background">
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
              className="p-2 rounded-md hover:bg-accent text-muted-foreground hover:text-foreground"
              title="Clear conversation"
            >
              <RotateCcw size={18} />
            </button>

            {/* Settings */}
            <button
              onClick={() => setIsSettingsOpen(!isSettingsOpen)}
              className={`p-2 rounded-md hover:bg-accent text-muted-foreground hover:text-foreground ${
                isSettingsOpen ? 'bg-accent' : ''
              }`}
              title="Settings"
            >
              <Settings size={18} />
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

        {/* Settings panel */}
        {isSettingsOpen && (
          <div className="absolute right-4 top-full mt-2 w-80 bg-card border border-border rounded-lg shadow-lg p-4 z-20">
            <h3 className="font-semibold text-foreground mb-3">Generation Settings</h3>
            
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-foreground mb-1">
                  Temperature: {generationParams.temperature}
                </label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={generationParams.temperature}
                  onChange={(e) => {
                    useChatStore.getState().setGenerationParams({
                      temperature: parseFloat(e.target.value),
                    });
                  }}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-foreground mb-1">
                  Max Tokens: {generationParams.maxTokens}
                </label>
                <input
                  type="range"
                  min="100"
                  max="4096"
                  step="100"
                  value={generationParams.maxTokens}
                  onChange={(e) => {
                    useChatStore.getState().setGenerationParams({
                      maxTokens: parseInt(e.target.value),
                    });
                  }}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-foreground mb-1">
                  Top P: {generationParams.topP}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={generationParams.topP}
                  onChange={(e) => {
                    useChatStore.getState().setGenerationParams({
                      topP: parseFloat(e.target.value),
                    });
                  }}
                  className="w-full"
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Main content */}
      <div className="flex flex-1 pt-16">
        {/* Chat interface */}
        <div className={`flex-1 transition-all duration-200 ${
          isBranchTreeExpanded ? 'mr-80' : ''
        }`}>
          <ChatInterface className="h-full" />
        </div>

        {/* Branch tree sidebar */}
        <div className={`fixed right-0 top-16 bottom-0 w-80 transition-transform duration-200 ${
          isBranchTreeExpanded ? 'translate-x-0' : 'translate-x-full'
        }`}>
          <BranchTree className="h-full" />
        </div>
      </div>

      {/* Settings overlay backdrop */}
      {isSettingsOpen && (
        <div
          className="fixed inset-0 z-10"
          onClick={() => setIsSettingsOpen(false)}
        />
      )}
    </div>
  );
}