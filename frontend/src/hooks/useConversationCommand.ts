/**
 * Custom hook for executing conversation commands with smart state management
 * Avoids full page reloads that trigger welcome screens
 */

import { useState } from 'react';
import apiClient from '@/lib/unified-api';
import { useChatStore } from '@/lib/store';

interface CommandOptions {
  /** Wait time after command execution before refreshing (for async operations like regen) */
  waitMs?: number;
  /** Whether to refresh conversation data after command */
  refreshConversation?: boolean;
  /** Whether to refresh branch data after command */
  refreshBranches?: boolean;
  /** Fallback to page reload if refresh fails */
  fallbackToReload?: boolean;
  /** Success message to show */
  successMessage?: string;
  /** Error message prefix */
  errorPrefix?: string;
}

interface CommandResult {
  success: boolean;
  message?: string;
  data?: any;
}

export function useConversationCommand() {
  const [isExecuting, setIsExecuting] = useState(false);
  const { setMessages, setCurrentBranch } = useChatStore();
  // Note: setBranches is no longer needed as branches are derived on demand

  /**
   * Refresh conversation data from backend
   */
  const refreshConversation = async (): Promise<boolean> => {
    try {
      const {
        getCurrentSessionId,
        currentConversationId,
        currentBranchId,
      } = useChatStore.getState();

      // Fetch messages for the CURRENT branch, not always 'main'
      const conversationResponse = await apiClient.getConversation(getCurrentSessionId(), currentBranchId || 'main');
      if (conversationResponse.success && conversationResponse.data?.conversation && currentConversationId) {
        setMessages(currentConversationId, currentBranchId || 'main', conversationResponse.data.conversation);
        console.log(`🔄 Conversation refreshed for branch '${currentBranchId || 'main'}' with ${conversationResponse.data.conversation.length} messages`);
        return true;
      }
      return false;
    } catch (error) {
      console.error('Error refreshing conversation:', error);
      return false;
    }
  };

  /**
   * Refresh branch data from backend
   */
  const refreshBranches = async (): Promise<boolean> => {
    try {
      const { getCurrentSessionId } = useChatStore.getState();
      const branchesResponse = await apiClient.getBranches(getCurrentSessionId());
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
        
        console.log(`🌿 Branches updated from backend:`, transformedBranches);
        // Note: We don't need to setBranches anymore since they are derived on demand
        // The branches will be available via getBranches() which reads from the store
        if (current_branch) {
          setCurrentBranch(current_branch);
        }
        console.log(`🌿 Branches refreshed: ${transformedBranches.length} branches, current: ${current_branch}`);
        return true;
      }
      return false;
    } catch (error) {
      console.error('Error refreshing branches:', error);
      return false;
    }
  };

  /**
   * Execute a conversation command with smart state management
   */
  const executeCommand = async (
    command: string,
    args: string[] = [],
    options: CommandOptions = {}
  ): Promise<CommandResult> => {
    const {
      waitMs = 0,
      refreshConversation: shouldRefreshConversation = true,
      refreshBranches: shouldRefreshBranches = false,
      fallbackToReload = true,
      successMessage,
      errorPrefix = 'Command failed'
    } = options;

    if (isExecuting) {
      return { success: false, message: 'Another command is already executing' };
    }

    setIsExecuting(true);
    
    try {
      console.log(`🚀 Executing command '${command}' with args:`, args);
      const response = await apiClient.executeCommand(command, args);
      console.log(`🚀 Command '${command}' response:`, response);

      if (!response.success) {
        const errorMessage = response.message || 'Unknown error';
        
        // Handle index synchronization errors
        if (errorMessage.includes('out of bounds') || 
            errorMessage.includes('Invalid message index') ||
            errorMessage.includes('out of range')) {
          console.warn('❌ Index out of sync with backend - refreshing to sync state');
          if (fallbackToReload) {
            setTimeout(() => window.location.reload(), 1000);
            return { success: false, message: 'Refreshing to sync conversation state...' };
          }
        }
        
        return { success: false, message: `${errorPrefix}: ${errorMessage}` };
      }

      // Wait if specified (for async operations like regeneration)
      if (waitMs > 0) {
        console.log(`⏳ Waiting ${waitMs}ms for command completion...`);
        await new Promise(resolve => setTimeout(resolve, waitMs));
      }

      // Refresh data as requested
      let refreshSucceeded = true;
      
      if (shouldRefreshConversation) {
        refreshSucceeded = await refreshConversation();
        if (!refreshSucceeded && fallbackToReload) {
          console.warn('Failed to refresh conversation, falling back to page reload');
          setTimeout(() => window.location.reload(), 500);
          return { success: true, message: 'Refreshing page to sync state...' };
        }
      }
      
      if (shouldRefreshBranches) {
        const branchRefreshSucceeded = await refreshBranches();
        if (!branchRefreshSucceeded && fallbackToReload) {
          console.warn('Failed to refresh branches, falling back to page reload');
          setTimeout(() => window.location.reload(), 500);
          return { success: true, message: 'Refreshing page to sync state...' };
        }
        refreshSucceeded = refreshSucceeded && branchRefreshSucceeded;
      }

      const resultMessage = successMessage || response.message || 'Command completed successfully';
      console.log(`✅ Command '${command}' completed successfully`);
      
      return { 
        success: true, 
        message: resultMessage,
        data: response.data 
      };

    } catch (error) {
      console.error(`Error executing command '${command}':`, error);
      
      if (fallbackToReload) {
        console.warn('Command execution failed, falling back to page reload');
        setTimeout(() => window.location.reload(), 1000);
        return { success: false, message: 'Refreshing page due to error...' };
      }
      
      return { 
        success: false, 
        message: `${errorPrefix}: ${error instanceof Error ? error.message : 'Unknown error'}` 
      };
    } finally {
      setIsExecuting(false);
    }
  };

  return {
    executeCommand,
    refreshConversation,
    refreshBranches,
    isExecuting
  };
}

// Predefined command configurations for common operations
export const COMMAND_CONFIGS = {
  // Message operations
  delete: {
    refreshConversation: true,
    refreshBranches: true, // Update message counts
    errorPrefix: 'Failed to delete message'
  },
  
  regen: {
    waitMs: 3000, // Wait for regeneration to complete
    refreshConversation: true,
    errorPrefix: 'Failed to regenerate message'
  },
  
  edit: {
    waitMs: 1000, // Brief wait for edit to propagate
    refreshConversation: true,
    errorPrefix: 'Failed to edit message'
  },
  
  // Conversation operations
  clear: {
    refreshConversation: true,
    refreshBranches: true, // Update message counts
    successMessage: 'Conversation cleared',
    errorPrefix: 'Failed to clear conversation'
  },
  
  // File operations
  save: {
    refreshConversation: false,
    refreshBranches: false,
    errorPrefix: 'Failed to save conversation'
  },
  
  load: {
    waitMs: 1000, // Wait for load to complete
    refreshConversation: true,
    refreshBranches: true,
    successMessage: 'Conversation loaded successfully',
    errorPrefix: 'Failed to load conversation'
  },
  
  // Model operations
  swap: {
    waitMs: 2000, // Wait for model to initialize
    refreshConversation: true,
    refreshBranches: true,
    successMessage: 'Model switched successfully',
    errorPrefix: 'Failed to switch model'
  },

  // Branch operations
  branch_from: {
    waitMs: 1500, // Wait for branch creation to complete
    refreshConversation: false, // Don't refresh current conversation
    refreshBranches: true, // Update branch list
    successMessage: 'New branch created successfully',
    errorPrefix: 'Failed to create branch from this point'
  }
} as const;
