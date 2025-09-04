/**
 * Store Adapter Utilities
 * 
 * This file provides adapter utilities to help transition between the legacy
 * flat message array format and the new branch-aware normalized structure.
 */

import type { Message, Conversation, ConversationBranch } from '../types';

/**
 * Converts legacy conversation format to the new branch-aware format
 */
export function adaptLegacyConversation(
  legacyConversation: Omit<Conversation, 'branches'> & { messages: Message[] }
): Conversation {
  // Create the branch structure with the main branch having all messages
  const branchStructure: { [branchId: string]: { messages: Message[] } } = {
    main: { 
      messages: legacyConversation.messages 
    }
  };

  // Return the adapted conversation
  return {
    ...legacyConversation,
    branches: branchStructure
  };
}

/**
 * Extracts all messages from all branches of a conversation
 * Used for backward compatibility with code expecting a flat message array
 */
export function flattenBranchMessages(
  conversation: Conversation
): Message[] {
  // If the conversation still has the legacy messages array, use it
  if (conversation.messages && conversation.messages.length > 0) {
    return conversation.messages;
  }

  // If branches exist, collect all messages
  if (conversation.branches) {
    // Start with the main branch if it exists
    if (conversation.branches.main) {
      return conversation.branches.main.messages;
    }
    
    // Otherwise collect from the first available branch
    const firstBranchId = Object.keys(conversation.branches)[0];
    if (firstBranchId) {
      return conversation.branches[firstBranchId].messages;
    }
  }

  // Fallback to empty array if no messages found
  return [];
}

/**
 * Converts branch-specific message structure to legacy flat message array
 * This helps with backward compatibility for APIs and components
 * expecting the old format
 */
export function normalizedToLegacy(
  conversationId: string,
  branchId: string,
  normalizedMessages: {
    [conversationId: string]: {
      [branchId: string]: Message[]
    }
  }
): Message[] {
  if (
    !normalizedMessages[conversationId] || 
    !normalizedMessages[conversationId][branchId]
  ) {
    return [];
  }
  
  return normalizedMessages[conversationId][branchId];
}

/**
 * Converts a flat message array to the normalized branch structure
 */
export function legacyToNormalized(
  conversationId: string,
  branchId: string,
  messages: Message[]
): {
  [conversationId: string]: {
    [branchId: string]: Message[]
  }
} {
  return {
    [conversationId]: {
      [branchId]: messages
    }
  };
}

/**
 * Builds a conversation branch structure from branch-specific message storage
 */
export function buildBranchStructure(
  conversationId: string,
  normalizedMessages: {
    [conversationId: string]: {
      [branchId: string]: Message[]
    }
  },
  branchMetadata: ConversationBranch[] = []
): { [branchId: string]: { messages: Message[] } } {
  // If no normalized messages for this conversation, return empty structure
  if (!normalizedMessages[conversationId]) {
    return { main: { messages: [] } };
  }
  
  const branchIds = Object.keys(normalizedMessages[conversationId]);
  const result: { [branchId: string]: { messages: Message[] } } = {};
  
  // Ensure 'main' is always first if it exists
  if (branchIds.includes('main')) {
    result.main = {
      messages: normalizedMessages[conversationId].main
    };
  }
  
  // Add all other branches
  for (const branchId of branchIds) {
    if (branchId === 'main') continue; // Skip main as we already added it
    
    result[branchId] = {
      messages: normalizedMessages[conversationId][branchId]
    };
  }
  
  // If no branches were found, ensure at least main exists
  if (Object.keys(result).length === 0) {
    result.main = { messages: [] };
  }
  
  return result;
}

/**
 * Combines messages from different branches when creating a new branch
 * or merging branches
 */
export function combineBranchMessages(
  baseMessages: Message[],
  additionalMessages: Message[]
): Message[] {
  // This is a simple concatenation, but could be extended with 
  // more sophisticated merging logic in the future
  return [...baseMessages, ...additionalMessages];
}

/**
 * Gets metadata about branches from the normalized message structure
 */
export function getBranchMetadata(
  conversationId: string,
  normalizedMessages: {
    [conversationId: string]: {
      [branchId: string]: Message[]
    }
  },
  currentBranchId: string = 'main'
): ConversationBranch[] {
  if (!normalizedMessages[conversationId]) {
    return [
      {
        id: 'main',
        name: 'Main',
        isActive: true,
        messageCount: 0,
        createdAt: new Date().toISOString(),
        lastActive: new Date().toISOString()
      }
    ];
  }
  
  const branchIds = Object.keys(normalizedMessages[conversationId]);
  return branchIds.map(branchId => {
    const messages = normalizedMessages[conversationId][branchId];
    const lastMessage = messages.length > 0 ? messages[messages.length - 1] : null;
    
    return {
      id: branchId,
      name: branchId === 'main' ? 'Main' : `Branch ${branchId}`,
      isActive: branchId === currentBranchId,
      messageCount: messages.length,
      createdAt: lastMessage ? new Date(lastMessage.timestamp).toISOString() : new Date().toISOString(),
      lastActive: lastMessage ? new Date(lastMessage.timestamp).toISOString() : new Date().toISOString(),
      preview: lastMessage ? lastMessage.content.slice(0, 50) + (lastMessage.content.length > 50 ? '...' : '') : undefined
    };
  });
}

/**
 * SessionManager adapter for frontend store
 * This is a simplified version of the session manager pattern
 * used in the backend (see oumi/webchat/core/session_manager.py)
 */
export class SessionManager {
  private static instance: SessionManager;
  private currentSessionId: string;

  private constructor() {
    // Generate a session ID on first load
    this.currentSessionId = this.generateSessionId();
  }

  /**
   * Get the singleton instance
   */
  public static getInstance(): SessionManager {
    if (!SessionManager.instance) {
      SessionManager.instance = new SessionManager();
    }
    return SessionManager.instance;
  }

  /**
   * Get the current session ID
   */
  public static getCurrentSessionId(): string {
    return SessionManager.getInstance().currentSessionId;
  }

  /**
   * Generate a new session ID
   */
  private generateSessionId(): string {
    return `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Create a new session (resets the current session ID)
   */
  public createNewSession(): string {
    this.currentSessionId = this.generateSessionId();
    return this.currentSessionId;
  }
  
  /**
   * Reset to a fresh session
   */
  public resetToFreshSession(): string {
    return this.createNewSession();
  }
  
  /**
   * Start a new session (static helper)
   */
  public static startNewSession(): string {
    return SessionManager.getInstance().createNewSession();
  }
  
  /**
   * Reset to a fresh session (static helper)
   */
  public static resetToFreshSession(): string {
    return SessionManager.getInstance().resetToFreshSession();
  }
}

/**
 * Helper to auto-save conversations to the backend
 * This is called after changes to the conversation store
 */
export async function autoSaveConversation(conversation: Conversation): Promise<void> {
  // Log the save event for debugging
  console.debug(`Auto-saving conversation ${conversation.id}`, conversation.title);

  // Import the API client dynamically to avoid circular dependencies
  const unifiedApiClient = (await import('../unified-api')).default;

  try {
    // Save the conversation to persistent storage
    await unifiedApiClient.saveConversation(
      SessionManager.getCurrentSessionId(),
      conversation.id,
      conversation
    );
  } catch (error) {
    console.error('Failed to auto-save conversation:', error);
  }
}