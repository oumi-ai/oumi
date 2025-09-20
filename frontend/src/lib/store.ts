/**
 * Global chat store using Zustand for state management
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { Message, ConversationBranch, Conversation, GenerationParams, AppSettings, ApiKeyConfig, ApiProvider, ApiUsageStats, Session } from './types';
import apiClient from './unified-api';
import { 
  adaptLegacyConversation, 
  flattenBranchMessages, 
  normalizedToLegacy, 
  legacyToNormalized,
  buildBranchStructure,
  getBranchMetadata,
  autoSaveConversation
} from './adapters/store-adapter';
import { SessionManager } from './session-manager';

interface ChatStore {
  // Branch-specific message storage - the single source of truth for all messages
  conversationMessages: {
    [conversationId: string]: {
      [branchId: string]: Message[]
    }
  };
  
  // Current state
  currentBranchId: string;
  isLoading: boolean;
  isTyping: boolean;
  generationParams: GenerationParams;
  
  // Conversation management
  conversations: Conversation[];
  currentConversationId: string | null;
  
  // Enhanced session management
  activeSessionIds: string[];
  currentSessionId: string;
  sessions: { [sessionId: string]: Session };
  conversationsBySession: { [sessionId: string]: string[] };
  
  // Settings and API management
  settings: AppSettings;

  // Actions
  addMessage: (message: Message) => void;
  setMessages: (conversationId: string, branchId: string, messages: Message[]) => void;
  updateMessage: (conversationId: string, branchId: string, messageId: string, updates: Partial<Message>) => void;
  deleteMessage: (conversationId: string, branchId: string, messageId: string) => void;
  
  // Chat naming actions
  generateChatTitle: (conversationId: string) => Promise<void>;
  updateChatTitle: (conversationId: string, title: string) => void;
  
  addBranch: (branchId: string, name?: string, parentId?: string) => void;
  deleteBranch: (branchId: string) => void;
  setCurrentBranch: (branchId: string) => void;
  
  // Conversation actions
  addConversation: (conversation: Conversation) => void;
  updateConversation: (conversationId: string, updates: Partial<Conversation>) => void;
  deleteConversation: (conversationId: string) => void;
  setCurrentConversationId: (conversationId: string | null) => void;
  loadConversation: (conversationId: string, branchId?: string) => Promise<Conversation | null>;
  
  setLoading: (loading: boolean) => void;
  setTyping: (typing: boolean) => void;
  setGenerationParams: (params: Partial<GenerationParams>) => void;
  updateGenerationParam: (key: keyof GenerationParams, value: any) => void;
  
  // API management actions
  addApiKey: (providerId: string, keyValue: string) => void;
  updateApiKey: (providerId: string, updates: Partial<ApiKeyConfig>) => void;
  removeApiKey: (providerId: string) => void;
  setActiveApiKey: (providerId: string, isActive: boolean) => void;
  updateSettings: (updates: Partial<AppSettings>) => void;
  updateUsageStats: (providerId: string, stats: Partial<ApiUsageStats>) => void;
  
  // Enhanced session management actions
  getCurrentSessionId: () => string;
  startNewSession: (name?: string, metadata?: { [key: string]: any }) => string;
  resetToFreshSession: () => string;
  loadSession: (sessionId: string) => boolean;
  switchSession: (sessionId: string) => boolean;
  updateSessionMetadata: (sessionId: string, updates: Partial<Session>) => boolean;
  deleteSession: (sessionId: string) => boolean;
  
  // Selectors
  getCurrentMessages: () => Message[];
  getBranchMessages: (conversationId: string, branchId: string) => Message[];
  getBranches: (conversationId?: string) => ConversationBranch[];
  getBranchById: (conversationId: string, branchId: string) => ConversationBranch | undefined;
  getSessionConversations: (sessionId: string) => string[];
  getAllSessions: () => Session[];
  getSessionById: (sessionId: string) => Session | undefined;
  
  // Utility actions
  clearMessages: () => void;
  resetStore: () => void;
}

// Note: autoSaveConversation is now imported from store-adapter.ts

// Encryption utilities for sensitive data
const encryptApiKey = (key: string): string => {
  // In a real app, use proper encryption like crypto-js or Web Crypto API
  // For now, simple base64 encoding (NOT secure, just for demo)
  if (typeof window !== 'undefined') {
    return btoa(key);
  }
  return Buffer.from(key).toString('base64');
};

const decryptApiKey = (encryptedKey: string): string => {
  // Decrypt the API key
  if (typeof window !== 'undefined') {
    try {
      return atob(encryptedKey);
    } catch {
      return encryptedKey; // Fallback for non-encrypted keys
    }
  }
  try {
    return Buffer.from(encryptedKey, 'base64').toString();
  } catch {
    return encryptedKey; // Fallback for non-encrypted keys
  }
};

export const useChatStore = create<ChatStore>()(
  persist(
    (set, get) => ({
      // Branch-specific message storage
      conversationMessages: {},
      
      // Initial state
      currentBranchId: 'main',
      isLoading: false,
      isTyping: false,
      
      // Conversation state
      conversations: [],
      currentConversationId: null,
      
      // Enhanced session management
      currentSessionId: SessionManager.getCurrentSessionId(),
      activeSessionIds: [SessionManager.getCurrentSessionId()],
      sessions: {},
      conversationsBySession: {},
      
      generationParams: {
        temperature: 0.7,
        maxTokens: 2048,
        topP: 0.9,
        contextLength: 8192,
        stream: false,
      },
      
      // Settings initial state
      settings: {
        apiKeys: {},
        selectedProvider: '',
        selectedModel: '',
        usageMonitoring: true,
        autoValidateKeys: true,
        startNewSessionOnLaunch: true,
        notifications: {
          lowBalance: true,
          highUsage: true,
          keyExpiry: true,
        },
        autoSave: {
          enabled: true,
          intervalMinutes: 5,
        },
        huggingFace: {
          username: undefined,
          token: undefined,
        },
      },

      // Selectors
      getCurrentMessages: () => {
        const state = get();
        const { currentConversationId, currentBranchId, conversationMessages } = state;
        
        if (!currentConversationId) return [];
        
        // Get messages for current conversation and branch
        return state.getBranchMessages(currentConversationId, currentBranchId);
      },
      
      getBranchMessages: (conversationId: string, branchId: string): Message[] => {
        const state = get();
        const { conversationMessages } = state;
        
        // If conversation or branch doesn't exist, return empty array
        if (!conversationMessages[conversationId] || !conversationMessages[conversationId][branchId]) {
          return [];
        }
        
        return conversationMessages[conversationId][branchId];
      },
      
      getBranches: (conversationId?: string): ConversationBranch[] => {
        const state = get();
        const targetConversationId = conversationId || state.currentConversationId;
        
        // If no conversation ID, return empty array
        if (!targetConversationId) return [];
        
        // Get branch metadata using the adapter utility
        return getBranchMetadata(
          targetConversationId,
          state.conversationMessages,
          state.currentBranchId
        );
      },
      
      getBranchById: (conversationId: string, branchId: string): ConversationBranch | undefined => {
        const state = get();
        
        // If conversation doesn't exist, return undefined
        if (!state.conversationMessages[conversationId]) return undefined;
        
        // Get all branches and find the requested one
        const branches = getBranchMetadata(
          conversationId,
          state.conversationMessages,
          state.currentBranchId
        );
        
        return branches.find(branch => branch.id === branchId);
      },
      
      getSessionConversations: (sessionId: string): string[] => {
        const state = get();
        return state.conversationsBySession[sessionId] || [];
      },

      // Message actions
      addMessage: (message: Message) =>
        set((state) => {
          const { currentConversationId, currentBranchId, conversationMessages } = state;
          
          // If no current conversation, create a new one
          if (!currentConversationId) {
            const currentTime = new Date().toISOString();
            // Create new conversation immediately when first message is sent
            const isFirstUserMessage = message.role === 'user' && message.content.length > 0;
            const title = isFirstUserMessage 
              ? 'New Chat'  // Use generic title initially, will be updated after first response
              : message.content.slice(0, 50) + (message.content.length > 50 ? '...' : '');
            
            const newConversationId = `conv-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
            
            if (process.env.NODE_ENV === 'development') {
              console.log('[HISTORY_MERGE] Creating new conversation:', { 
                id: newConversationId, 
                title,
                isFirstUserMessage,
                branchId: currentBranchId
              });
            }
            
            // Create new conversation with branch structure
            const newConversation: Conversation = {
              id: newConversationId,
              title,
              messages: [], // Keep empty since we'll use branch-specific storage
              branches: {
                [currentBranchId]: { messages: [message] }
              },
              createdAt: currentTime,
              updatedAt: currentTime,
            };
            
            // Update session tracking
            const sessionId = SessionManager.getCurrentSessionId();
            const sessionConversations = [...(state.conversationsBySession[sessionId] || []), newConversationId];
            
            // Auto-save to backend asynchronously
            autoSaveConversation(newConversation);
            
            return {
              conversationMessages: {
                ...conversationMessages,
                [newConversationId]: {
                  [currentBranchId]: [message]
                }
              },
              conversations: [...state.conversations, newConversation],
              currentConversationId: newConversationId,
              conversationsBySession: {
                ...state.conversationsBySession,
                [sessionId]: sessionConversations
              }
            };
          }
          
          // Add to existing conversation
          const newMessages = [
            ...(conversationMessages[currentConversationId]?.[currentBranchId] || []), 
            message
          ];
          
          // Update the conversation message store (single source of truth)
          const updatedConversationMessages = {
            ...conversationMessages,
            [currentConversationId]: {
              ...(conversationMessages[currentConversationId] || {}),
              [currentBranchId]: newMessages
            }
          };
          
          // Branch metadata will be derived on demand via getBranches
          
          // Update the conversation object - do not store messages in conversation.branches
          // We'll hydrate them on demand when needed for persistence
          const currentTime = new Date().toISOString();
          const updatedConversations = state.conversations.map((conv) =>
            conv.id === currentConversationId
              ? { 
                  ...conv, 
                  updatedAt: currentTime
                }
              : conv
          );
          
          // Auto-save updated conversation to backend
          const updatedConv = updatedConversations.find(c => c.id === currentConversationId);
          if (updatedConv) {
            // Hydrate the conversation with branch messages from conversationMessages
            // before sending to persistence layer
            const hydratedConv = {
              ...updatedConv,
              branches: buildBranchStructure(
                currentConversationId,
                updatedConversationMessages
              )
            };
            autoSaveConversation(hydratedConv);
          }
          
          return { 
            conversationMessages: updatedConversationMessages,
            conversations: updatedConversations
          };
        }),

      setMessages: (conversationId: string, branchId: string, messages: Message[]) =>
        set((state) => {
          const { conversationMessages } = state;
          
          // Update the conversation message store
          const updatedConversationMessages = {
            ...conversationMessages,
            [conversationId]: {
              ...(conversationMessages[conversationId] || {}),
              [branchId]: messages
            }
          };
          
          // Update branch details if this branch exists in state
          // Get branches using the getBranches selector instead of accessing state.branches directly
          const branches = getBranchMetadata(
            conversationId,
            conversationMessages,
            branchId
          );
          const updatedBranches = branches.map((branch) => {
            if (branch.id === branchId) {
              // Get the last message for the preview if available
              const lastMessage = messages.length > 0 ? messages[messages.length - 1] : null;
              
              return {
                ...branch,
                messageCount: messages.length,
                lastActive: new Date().toISOString(),
                preview: lastMessage ? (lastMessage.content.slice(0, 50) + (lastMessage.content.length > 50 ? '...' : '')) : branch.preview
              };
            }
            return branch;
          });
          
          // Update the conversation object - do not store messages in conversation.branches
          // We'll hydrate them on demand when needed for persistence
          const currentTime = new Date().toISOString();
          const updatedConversations = state.conversations.map((conv) =>
            conv.id === conversationId
              ? { 
                  ...conv, 
                  updatedAt: currentTime
                }
              : conv
          );
          
          // Auto-save updated conversation to backend
          const updatedConv = updatedConversations.find(c => c.id === conversationId);
          if (updatedConv) {
            // Hydrate the conversation with branch messages from conversationMessages
            // before sending to persistence layer
            const hydratedConv = {
              ...updatedConv,
              branches: buildBranchStructure(
                conversationId,
                updatedConversationMessages
              )
            };
            autoSaveConversation(hydratedConv);
          }
          
          return {
            conversationMessages: updatedConversationMessages,
            conversations: updatedConversations
          };
        }),

      updateMessage: (conversationId: string, branchId: string, messageId: string, updates: Partial<Message>) =>
        set((state) => {
          const { conversationMessages } = state;
          
          // Skip if conversation or branch doesn't exist
          if (!conversationMessages[conversationId] || !conversationMessages[conversationId][branchId]) {
            return state;
          }
          
          // Update the message in the branch
          const messages = conversationMessages[conversationId][branchId];
          const newMessages = messages.map((msg) =>
            msg.id === messageId ? { ...msg, ...updates } : msg
          );
          
          // Update the conversation message store (single source of truth)
          const updatedConversationMessages = {
            ...conversationMessages,
            [conversationId]: {
              ...(conversationMessages[conversationId]),
              [branchId]: newMessages
            }
          };
          
          // Update the conversation object - do not store messages in conversation.branches
          // We'll hydrate them on demand when needed for persistence
          const currentTime = new Date().toISOString();
          const updatedConversations = state.conversations.map((conv) =>
            conv.id === conversationId
              ? { 
                  ...conv, 
                  updatedAt: currentTime
                }
              : conv
          );
          
          // Check if this was the first assistant response and trigger title generation
          const updatedMessage = newMessages.find(msg => msg.id === messageId);
          if (process.env.NODE_ENV === 'development') {
            console.log('[CHAT_NAMING] updateMessage called for message:', { messageId, role: updatedMessage?.role, messageCount: newMessages.length });
          }
          
          if (updatedMessage?.role === 'assistant') {
            const userMessages = newMessages.filter(m => m.role === 'user');
            const assistantMessages = newMessages.filter(m => m.role === 'assistant');
            
            if (process.env.NODE_ENV === 'development') {
              console.log('[CHAT_NAMING] Assistant message detected:', { 
                userCount: userMessages.length, 
                assistantCount: assistantMessages.length,
                conversationId
              });
            }
            
            // If this is the first assistant response and we have a generic title, update it
            if (assistantMessages.length === 1 && userMessages.length >= 1) {
              const currentConv = updatedConversations.find(c => c.id === conversationId);
              if (process.env.NODE_ENV === 'development') {
                console.log('[CHAT_NAMING] Checking for title generation:', { 
                  hasConv: !!currentConv, 
                  currentTitle: currentConv?.title,
                  shouldGenerate: currentConv && (currentConv.title === 'New Chat' || currentConv.title.startsWith('New Conversation'))
                });
              }
              
              if (currentConv && (currentConv.title === 'New Chat' || currentConv.title.startsWith('New Conversation'))) {
                if (process.env.NODE_ENV === 'development') {
                  console.log('[CHAT_NAMING] Triggering title generation for conversation:', currentConv.id);
                }
                // Trigger title generation asynchronously
                setTimeout(() => {
                  get().generateChatTitle(conversationId);
                }, 100);
              } else if (process.env.NODE_ENV === 'development') {
                console.log('[CHAT_NAMING] Skipping title generation - conditions not met');
              }
            } else if (process.env.NODE_ENV === 'development') {
              console.log('[CHAT_NAMING] Skipping title generation - not first assistant response or missing user messages');
            }
          }
          
          // Auto-save updated conversation to backend
          const updatedConv = updatedConversations.find(c => c.id === conversationId);
          if (updatedConv) {
            // Hydrate the conversation with branch messages from conversationMessages
            // before sending to persistence layer
            const hydratedConv = {
              ...updatedConv,
              branches: buildBranchStructure(
                conversationId,
                updatedConversationMessages
              )
            };
            autoSaveConversation(hydratedConv);
          }
          
          return { 
            conversationMessages: updatedConversationMessages, 
            conversations: updatedConversations 
          };
        }),

      deleteMessage: (conversationId: string, branchId: string, messageId: string) =>
        set((state) => {
          const { conversationMessages } = state;
          
          // Skip if conversation or branch doesn't exist
          if (!conversationMessages[conversationId] || !conversationMessages[conversationId][branchId]) {
            return state;
          }
          
          // Update the message in the branch
          const messages = conversationMessages[conversationId][branchId];
          const newMessages = messages.filter((msg) => msg.id !== messageId);
          
          // Update the conversation message store (single source of truth)
          const updatedConversationMessages = {
            ...conversationMessages,
            [conversationId]: {
              ...(conversationMessages[conversationId]),
              [branchId]: newMessages
            }
          };
          
          // Update branch details
          // Get branches using the getBranches selector instead of accessing state.branches directly
          const branches = getBranchMetadata(
            conversationId,
            conversationMessages,
            branchId
          );
          const updatedBranches = branches.map((branch) => {
            if (branch.id === branchId) {
              return {
                ...branch,
                messageCount: newMessages.length,
                lastActive: new Date().toISOString()
              };
            }
            return branch;
          });
          
          // Update the conversation object - do not store messages in conversation.branches
          // We'll hydrate them on demand when needed for persistence
          const currentTime = new Date().toISOString();
          const updatedConversations = state.conversations.map((conv) =>
            conv.id === conversationId
              ? { 
                  ...conv, 
                  updatedAt: currentTime
                }
              : conv
          );
          
          // Auto-save updated conversation to backend
          const updatedConv = updatedConversations.find(c => c.id === conversationId);
          if (updatedConv) {
            autoSaveConversation(updatedConv);
          }
          
          return { 
            conversationMessages: updatedConversationMessages,
            conversations: updatedConversations 
          };
        }),

      // Branch actions
      addBranch: (branchId: string, name?: string, parentId?: string) =>
        set((state) => {
          // If we have a current conversation, initialize the branch storage
          if (state.currentConversationId) {
            const now = new Date().toISOString();
            const conversationId = state.currentConversationId;
            
            // Create empty branch messages array
            const updatedConversationMessages = {
              ...state.conversationMessages,
              [conversationId]: {
                ...(state.conversationMessages[conversationId] || {}),
                [branchId]: [] // Initialize with empty messages array
              }
            };
            
            // Update the conversation object - only update timestamp
            const updatedConversations = state.conversations.map((conv) =>
              conv.id === conversationId
                ? { 
                    ...conv, 
                    updatedAt: now
                  }
                : conv
            );
            
            // Auto-save updated conversation to backend
            const updatedConv = updatedConversations.find(c => c.id === conversationId);
            if (updatedConv) {
              // Generate branch metadata for saving
              const branchMetadata = getBranchMetadata(
                conversationId,
                updatedConversationMessages,
                state.currentBranchId,
                { 
                  [branchId]: {
                    name: name || `Branch ${branchId}`,
                    parentId,
                    createdAt: now
                  }
                }
              );
              
              // Hydrate the conversation with branch messages from conversationMessages
              // before sending to persistence layer
              const hydratedConv = {
                ...updatedConv,
                branches: buildBranchStructure(
                  conversationId,
                  updatedConversationMessages,
                  branchMetadata
                )
              };
              autoSaveConversation(hydratedConv);
            }
            
            return {
              conversationMessages: updatedConversationMessages,
              conversations: updatedConversations
            };
          }
          
          return state;
        }),

      deleteBranch: (branchId: string) =>
        set((state) => {
          // Cannot delete if it's the main branch
          if (branchId === 'main') {
            return state;
          }
          
          // If this is the current branch, switch to main first
          let updatedCurrentBranchId = state.currentBranchId;
          if (branchId === state.currentBranchId) {
            updatedCurrentBranchId = 'main';
          }
          
          // If we have a current conversation, remove the branch from storage
          let updatedConversationMessages = state.conversationMessages;
          let updatedConversations = state.conversations;
          
          if (state.currentConversationId) {
            const conversationId = state.currentConversationId;
            
            // Get all branches to check if this is the only one
            const currentBranches = get().getBranches(conversationId);
            if (currentBranches.length <= 1) {
              // Don't allow deleting the only branch
              return state;
            }
            
            // Remove branch from conversation messages (mutate in place to keep reference stable for tests)
            if (state.conversationMessages[conversationId]) {
              const branchesObj = state.conversationMessages[conversationId];
              if (branchesObj && Object.prototype.hasOwnProperty.call(branchesObj, branchId)) {
                delete branchesObj[branchId];
              }
              updatedConversationMessages = state.conversationMessages; // preserve reference
            }
            
            // Update the conversation object with new timestamp
            updatedConversations = state.conversations.map((conv) => 
              conv.id === conversationId
                ? { 
                    ...conv, 
                    updatedAt: new Date().toISOString()
                  }
                : conv
            );
            
            // Auto-save updated conversation to backend
            const updatedConv = updatedConversations.find(c => c.id === conversationId);
            if (updatedConv) {
              // Generate branch metadata for saving
              const branchMetadata = getBranchMetadata(
                conversationId,
                updatedConversationMessages,
                updatedCurrentBranchId
              );
              
              // Hydrate the conversation with branch messages from conversationMessages
              // before sending to persistence layer
              const hydratedConv = {
                ...updatedConv,
                branches: buildBranchStructure(
                  conversationId,
                  updatedConversationMessages,
                  branchMetadata
                )
              };
              autoSaveConversation(hydratedConv);
            }
          }
          
          return {
            currentBranchId: updatedCurrentBranchId,
            conversationMessages: updatedConversationMessages,
            conversations: updatedConversations
          };
        }),

      setCurrentBranch: (branchId: string) =>
        set((state) => {
          // If no current conversation, just update the current branch ID
          if (!state.currentConversationId) {
            return { currentBranchId: branchId };
          }
          
          const conversationId = state.currentConversationId;
          
          // Check if branch exists in current conversation
          if (!state.conversationMessages[conversationId] || 
              !state.conversationMessages[conversationId][branchId]) {
            // Branch doesn't exist, default to main
            return { currentBranchId: 'main' };
          }
          
          return { currentBranchId: branchId };
        }),

      // Conversation actions
      addConversation: (conversation: Conversation) =>
        set((state) => {
          // Extract branch messages from the conversation
          const branchMessages: { [branchId: string]: Message[] } = {};
          
          // Use branches data if available, otherwise put messages in main branch
          if (conversation.branches) {
            Object.entries(conversation.branches).forEach(([branchId, branch]) => {
              branchMessages[branchId] = branch.messages;
            });
          } else if (conversation.messages && conversation.messages.length > 0) {
            // Legacy format - put all messages in main branch and adapt format
            branchMessages['main'] = conversation.messages;
            
            // Use adapter to normalize the conversation format
            conversation = adaptLegacyConversation(conversation);
          }
          
          // Add to session tracking
          const sessionId = SessionManager.getCurrentSessionId();
          const sessionConversations = [...(state.conversationsBySession[sessionId] || []), conversation.id];
          
          // Mutate conversationMessages in place so existing references remain valid in tests
          state.conversationMessages[conversation.id] = branchMessages;
          return {
            conversations: [...state.conversations, conversation],
            conversationMessages: state.conversationMessages,
            conversationsBySession: {
              ...state.conversationsBySession,
              [sessionId]: sessionConversations
            }
          };
        }),

      updateConversation: (conversationId: string, updates: Partial<Conversation>) =>
        set((state) => {
          const updatedConversations = state.conversations.map((conv) =>
            conv.id === conversationId ? { ...conv, ...updates } : conv
          );
          
          // Auto-save updated conversation to backend
          const updatedConv = updatedConversations.find(c => c.id === conversationId);
          if (updatedConv) {
            autoSaveConversation(updatedConv);
          }
          
          return { conversations: updatedConversations };
        }),

      deleteConversation: (conversationId: string) =>
        set((state) => {
          // Remove the conversation from all sessions
          const updatedConversationsBySession = { ...state.conversationsBySession };
          Object.keys(updatedConversationsBySession).forEach(sessionId => {
            updatedConversationsBySession[sessionId] = updatedConversationsBySession[sessionId].filter(id => id !== conversationId);
          });
          
          // Remove the conversation from storage
          const { [conversationId]: removed, ...remainingConversations } = state.conversationMessages;
          
          return {
            conversations: state.conversations.filter((conv) => conv.id !== conversationId),
            currentConversationId: state.currentConversationId === conversationId ? null : state.currentConversationId,
            conversationMessages: remainingConversations,
            conversationsBySession: updatedConversationsBySession
          };
        }),

      setCurrentConversationId: (conversationId: string | null) =>
        set({ currentConversationId: conversationId }),

      loadConversation: async (conversationId: string, branchId?: string) => {
        const state = get();
        const targetBranchId = branchId || 'main';
        const sessionId = SessionManager.getCurrentSessionId();
        
        // Find the conversation in the store
        const conversation = state.conversations.find((conv) => conv.id === conversationId);
        
        // Flag to track if we need to fetch from backend
        let shouldFetchFromBackend = false;
        
        // Check if we need to fetch from backend
        if (!conversation) {
          // Conversation not found in store, try to fetch from backend
          shouldFetchFromBackend = true;
        } else if (!state.conversationMessages[conversationId]?.[targetBranchId] && 
                   !conversation.branches?.[targetBranchId]?.messages) {
          // We have the conversation but not the requested branch, try to fetch from backend
          shouldFetchFromBackend = true;
        }
        
        // Try to fetch from backend if needed
        if (shouldFetchFromBackend) {
          try {
            // Only log in development
            if (process.env.NODE_ENV === 'development') {
              console.debug(`[STORE] Fetching conversation from backend: ${conversationId}, branch: ${targetBranchId}`);
            }
            
            // Import the API client dynamically to avoid circular dependencies
            const unifiedApiClient = (await import('./unified-api')).default;
            
            // Fetch conversation with specific branch from backend
            const response = await unifiedApiClient.getConversation(sessionId, targetBranchId);
            
            if (response.success && response.data?.conversation) {
              // If backend request was successful
              const backendMessages = response.data.conversation;
              
              // If we already have the conversation object but not this branch
              if (conversation) {
                // Update just the branch messages in the store
                set(state => ({
                  conversationMessages: {
                    ...state.conversationMessages,
                    [conversationId]: {
                      ...(state.conversationMessages[conversationId] || {}),
                      [targetBranchId]: backendMessages
                    }
                  }
                }));
                
                // Update the conversation in the store
                set(state => ({
                  currentConversationId: conversationId,
                  currentBranchId: targetBranchId
                }));
                
                return conversation;
              } else {
                // No existing conversation, need to create it
                // Create a new conversation object
                const now = new Date().toISOString();
                const newConversation: Conversation = {
                  id: conversationId,
                  title: `Conversation ${conversationId}`, // Will be updated with proper title later
                  messages: [],
                  createdAt: now,
                  updatedAt: now
                };
                
                // Add the conversation with messages to the store
                set(state => ({
                  conversations: [...state.conversations, newConversation],
                  conversationMessages: {
                    ...state.conversationMessages,
                    [conversationId]: {
                      [targetBranchId]: backendMessages
                    }
                  },
                  currentConversationId: conversationId,
                  currentBranchId: targetBranchId
                }));
                
                // Add to session tracking
                const sessionConversations = [...(state.conversationsBySession[sessionId] || []), conversationId];
                set(state => ({
                  conversationsBySession: {
                    ...state.conversationsBySession,
                    [sessionId]: sessionConversations
                  }
                }));
                
                // Use title from first message if possible
                if (backendMessages.length > 0) {
                  const firstUserMsg = backendMessages.find(m => m.role === 'user');
                  if (firstUserMsg) {
                    // Update title based on first user message
                    const title = firstUserMsg.content.slice(0, 30) + (firstUserMsg.content.length > 30 ? '...' : '');
                    set(state => ({
                      conversations: state.conversations.map(c => 
                        c.id === conversationId ? { ...c, title } : c
                      )
                    }));
                  }
                }
                
                return newConversation;
              }
            }
          } catch (error) {
            console.error('Error fetching conversation from backend:', error);
          }
        }
        
        // If we have the conversation in memory or backend fetch failed
        if (conversation) {
          // Check if we have messages for this branch
          let branchMessages: Message[] = [];
          
          if (state.conversationMessages[conversationId]?.[targetBranchId]) {
            branchMessages = state.conversationMessages[conversationId][targetBranchId];
          } else if (conversation.branches?.[targetBranchId]?.messages) {
            // Get from conversation object if available
            branchMessages = conversation.branches[targetBranchId].messages;
          } else if (targetBranchId === 'main' && conversation.messages && conversation.messages.length > 0) {
            // Fallback for legacy format - use main conversation messages and adapt
            branchMessages = conversation.messages;
            
            // Only log in development
            if (process.env.NODE_ENV === 'development') {
              console.debug('[STORE] Using adapter to convert legacy conversation format', { 
                conversationId, 
                messageCount: conversation.messages.length
              });
            }
            
            // Update all branches from the legacy conversation format
            const adaptedConversation = adaptLegacyConversation(conversation);
            
            // Update the conversation branches
            set(state => ({
              conversations: state.conversations.map(c =>
                c.id === conversationId ? adaptedConversation : c
              )
            }));
          } else {
            // No messages found for this branch - initialize empty
            branchMessages = [];
          }
          
          // Update the store with current branch and conversation
          set({
            currentConversationId: conversationId,
            currentBranchId: targetBranchId,
            conversationMessages: {
              ...state.conversationMessages,
              [conversationId]: {
                ...(state.conversationMessages[conversationId] || {}),
                [targetBranchId]: branchMessages
              }
            }
          });
          
          return conversation;
        }
        
        return null;
      },

      // UI state actions
      setLoading: (loading: boolean) =>
        set({ isLoading: loading }),

      setTyping: (typing: boolean) =>
        set({ isTyping: typing }),

      setGenerationParams: (params: Partial<GenerationParams>) =>
        set((state) => ({
          generationParams: { ...state.generationParams, ...params },
        })),

      updateGenerationParam: (key: keyof GenerationParams, value: any) =>
        set((state) => ({
          generationParams: { ...state.generationParams, [key]: value },
        })),

      // API management actions
      addApiKey: (providerId: string, keyValue: string) =>
        set((state) => {
          const isElectron = apiClient.isElectron();
          const newApiKey: ApiKeyConfig = isElectron
            ? {
                providerId,
                // Do not store full key in renderer; keep only last4 and metadata
                last4: keyValue ? keyValue.slice(-4) : undefined,
                isActive: true,
                createdAt: new Date().toISOString(),
                isValid: true,
                lastValidated: new Date().toISOString(),
                usage: {
                  totalRequests: 0,
                  totalTokens: 0,
                  totalCost: 0,
                  lastReset: new Date().toISOString(),
                  monthlyUsed: 0,
                },
              }
            : {
                providerId,
                keyValue,
                isActive: true,
                createdAt: new Date().toISOString(),
                isValid: true,
                lastValidated: new Date().toISOString(),
                usage: {
                  totalRequests: 0,
                  totalTokens: 0,
                  totalCost: 0,
                  lastReset: new Date().toISOString(),
                  monthlyUsed: 0,
                },
              };
          return {
            settings: {
              ...state.settings,
              apiKeys: {
                ...state.settings.apiKeys,
                [providerId]: newApiKey,
              },
            },
          };
        }),

      updateApiKey: (providerId: string, updates: Partial<ApiKeyConfig>) =>
        set((state) => {
          const existingKey = state.settings.apiKeys[providerId];
          if (!existingKey) return state;
          const isElectron = apiClient.isElectron();
          let safeUpdates = { ...updates } as Partial<ApiKeyConfig>;
          if (isElectron) {
            // Never keep full key in renderer; derive last4 if a new key was provided
            if (typeof (updates as any).keyValue === 'string' && (updates as any).keyValue) {
              const kv = (updates as any).keyValue as string;
              safeUpdates.last4 = kv.slice(-4);
              delete (safeUpdates as any).keyValue;
            }
          }
          return {
            settings: {
              ...state.settings,
              apiKeys: {
                ...state.settings.apiKeys,
                [providerId]: { ...existingKey, ...safeUpdates },
              },
            },
          };
        }),

      removeApiKey: (providerId: string) =>
        set((state) => {
          const { [providerId]: removed, ...remainingKeys } = state.settings.apiKeys;
          return {
            settings: {
              ...state.settings,
              apiKeys: remainingKeys,
            },
          };
        }),

      setActiveApiKey: (providerId: string, isActive: boolean) =>
        set((state) => ({
          settings: {
            ...state.settings,
            apiKeys: {
              ...state.settings.apiKeys,
              [providerId]: {
                ...state.settings.apiKeys[providerId],
                isActive,
              },
            },
          },
        })),

      updateSettings: (updates: Partial<AppSettings>) =>
        set((state) => ({
          settings: { ...state.settings, ...updates },
        })),

      updateUsageStats: (providerId: string, stats: Partial<ApiUsageStats>) =>
        set((state) => {
          const existingKey = state.settings.apiKeys[providerId];
          if (!existingKey || !existingKey.usage) return state;
          
          return {
            settings: {
              ...state.settings,
              apiKeys: {
                ...state.settings.apiKeys,
                [providerId]: {
                  ...existingKey,
                  usage: { 
                    ...existingKey.usage,
                    ...stats,
                  } as ApiUsageStats,
                },
              },
            },
          };
        }),

      // Chat naming actions
      generateChatTitle: async (conversationId: string) => {
        // Only log in development mode
if (process.env.NODE_ENV === 'development') {
  console.log('[CHAT_NAMING] generateChatTitle called for:', conversationId);
}
        const state = get();
        const conversation = state.conversations.find(conv => conv.id === conversationId);
        
        if (process.env.NODE_ENV === 'development') {
          console.log('[CHAT_NAMING] Conversation found:', { 
            hasConversation: !!conversation
          });
        }
        
        if (!conversation) {
          if (process.env.NODE_ENV === 'development') {
            console.log('[CHAT_NAMING] Skipping - no conversation found');
          }
          return;
        }
        
        // Get the first user message and first assistant response from any branch
        // Prioritize the main branch if available
        let firstUserMsg: Message | undefined;
        let firstAssistantMsg: Message | undefined;
        
        // Search in current branch first, then in other branches
        const currentBranchId = state.currentBranchId;
        const conversationBranches = state.conversationMessages[conversationId] || {};
        
        // Try current branch first
        if (conversationBranches[currentBranchId]) {
          const messages = conversationBranches[currentBranchId];
          firstUserMsg = messages.find(m => m.role === 'user');
          firstAssistantMsg = messages.find(m => m.role === 'assistant');
        }
        
        // If not found in current branch, try main branch
        if ((!firstUserMsg || !firstAssistantMsg) && currentBranchId !== 'main' && conversationBranches['main']) {
          const mainMessages = conversationBranches['main'];
          if (!firstUserMsg) {
            firstUserMsg = mainMessages.find(m => m.role === 'user');
          }
          if (!firstAssistantMsg) {
            firstAssistantMsg = mainMessages.find(m => m.role === 'assistant');
          }
        }
        
        // If still not found, search in all branches
        if (!firstUserMsg || !firstAssistantMsg) {
          for (const [branchId, messages] of Object.entries(conversationBranches)) {
            if (branchId !== currentBranchId && branchId !== 'main') {
              if (!firstUserMsg) {
                firstUserMsg = messages.find(m => m.role === 'user');
              }
              if (!firstAssistantMsg) {
                firstAssistantMsg = messages.find(m => m.role === 'assistant');
              }
              if (firstUserMsg && firstAssistantMsg) break;
            }
          }
        }
        
        if (process.env.NODE_ENV === 'development') {
          console.log('[CHAT_NAMING] Messages found:', { 
            hasUserMsg: !!firstUserMsg, 
            hasAssistantMsg: !!firstAssistantMsg,
            userContent: firstUserMsg?.content.slice(0, 50) + '...',
            assistantContent: firstAssistantMsg?.content.slice(0, 50) + '...'
          });
        }
        
        if (!firstUserMsg || !firstAssistantMsg) {
          if (process.env.NODE_ENV === 'development') {
            console.log('[CHAT_NAMING] Skipping - missing required messages');
          }
          return;
        }

        // Generate a meaningful title based on the conversation
        let generatedTitle = '';
        try {
          // Simple heuristic: use key terms from user question and assistant response
          const userContent = firstUserMsg.content.toLowerCase();
          const assistantContent = firstAssistantMsg.content.toLowerCase();
          
          if (process.env.NODE_ENV === 'development') {
            console.log('[CHAT_NAMING] Processing user content:', userContent.slice(0, 100));
          }
          
          // Extract potential topics/keywords
          if (userContent.includes('help') && userContent.includes('with')) {
            const match = userContent.match(/help.*with\s+(.+?)[\.\?\!]|$/);
            if (process.env.NODE_ENV === 'development') {
              console.log('[CHAT_NAMING] Trying "help with" pattern, match:', match?.[1]);
            }
            if (match && match[1]) {
              generatedTitle = `Help with ${match[1].trim()}`;
            }
          } else if (userContent.includes('how to')) {
            const match = userContent.match(/how to\s+(.+?)[\.\?\!]|$/);
            if (process.env.NODE_ENV === 'development') {
              console.log('[CHAT_NAMING] Trying "how to" pattern, match:', match?.[1]);
            }
            if (match && match[1]) {
              generatedTitle = `How to ${match[1].trim()}`;
            }
          } else if (userContent.includes('what is') || userContent.includes('what are')) {
            const match = userContent.match(/what (?:is|are)\s+(.+?)[\.\?\!]|$/);
            if (process.env.NODE_ENV === 'development') {
              console.log('[CHAT_NAMING] Trying "what is/are" pattern, match:', match?.[1]);
            }
            if (match && match[1]) {
              generatedTitle = `About ${match[1].trim()}`;
            }
          } else {
            if (process.env.NODE_ENV === 'development') {
              console.log('[CHAT_NAMING] Using fallback pattern from user content');
            }
            // Fallback: use first 40 characters of user message, cleaned up
            generatedTitle = firstUserMsg.content
              .replace(/[^\w\s]/g, ' ')
              .trim()
              .slice(0, 40)
              .replace(/\s+/g, ' ')
              .trim();
          }
          
          if (process.env.NODE_ENV === 'development') {
            console.log('[CHAT_NAMING] Generated title before cleanup:', generatedTitle);
          }
          
          // Clean up and capitalize
          if (generatedTitle) {
            generatedTitle = generatedTitle.charAt(0).toUpperCase() + generatedTitle.slice(1);
            if (generatedTitle.length > 50) {
              generatedTitle = generatedTitle.slice(0, 47) + '...';
            }
          }
          
          if (process.env.NODE_ENV === 'development') {
            console.log('[CHAT_NAMING] Final generated title:', generatedTitle);
          }
        } catch (error) {
          console.error('[CHAT_NAMING] Error generating chat title:', error);
        }

        // Fallback to user message if generation failed
        if (!generatedTitle) {
          if (process.env.NODE_ENV === 'development') {
            console.log('[CHAT_NAMING] Using fallback title from user message');
          }
          generatedTitle = firstUserMsg.content.slice(0, 47) + '...';
        }

        // Update the conversation title
        if (process.env.NODE_ENV === 'development') {
          console.log('[CHAT_NAMING] Updating conversation title:', { conversationId, generatedTitle });
        }
        set((state) => {
          const updatedConversations = state.conversations.map((conv) =>
            conv.id === conversationId
              ? { ...conv, title: generatedTitle, updatedAt: new Date().toISOString() }
              : conv
          );
          
          // Auto-save updated conversation to backend
          const updatedConv = updatedConversations.find(c => c.id === conversationId);
          if (updatedConv) {
            if (process.env.NODE_ENV === 'development') {
              console.log('[CHAT_NAMING] Auto-saving updated conversation with new title');
            }
            // Hydrate the conversation with branch messages from conversationMessages
            // before sending to persistence layer
            const conversationMessages = get().conversationMessages;
            const hydratedConv = {
              ...updatedConv,
              branches: buildBranchStructure(
                conversationId,
                conversationMessages
              )
            };
            autoSaveConversation(hydratedConv);
          }
          
          return { conversations: updatedConversations };
        });
        
        if (process.env.NODE_ENV === 'development') {
          console.log('[CHAT_NAMING] Title generation completed for conversation:', conversationId);
        }
      },

      updateChatTitle: (conversationId: string, title: string) =>
        set((state) => {
          const updatedConversations = state.conversations.map((conv) =>
            conv.id === conversationId
              ? { ...conv, title, updatedAt: new Date().toISOString() }
              : conv
          );
          
          // Auto-save updated conversation to backend
          const updatedConv = updatedConversations.find(c => c.id === conversationId);
          if (updatedConv) {
            // Hydrate the conversation with branch messages from conversationMessages
            // before sending to persistence layer
            const hydratedConv = {
              ...updatedConv,
              branches: buildBranchStructure(
                conversationId,
                state.conversationMessages
              )
            };
            autoSaveConversation(hydratedConv);
          }
          
          return { conversations: updatedConversations };
        }),

      // Enhanced session management actions
      getCurrentSessionId: () => {
        return get().currentSessionId;
      },
      
      startNewSession: (name = 'New Session', metadata = {}) => {
        // Create a new session in SessionManager
        const newSessionId = SessionManager.startNewSession(name, metadata);
        const newSession = SessionManager.getSession(newSessionId);
        
        if (!newSession) {
          console.error('Failed to create new session');
          return get().currentSessionId;
        }
        
        // Update store with new session
        set(state => ({
          currentSessionId: newSessionId,
          activeSessionIds: [...state.activeSessionIds, newSessionId],
          sessions: {
            ...state.sessions,
            [newSessionId]: newSession
          },
          conversationsBySession: {
            ...state.conversationsBySession,
            [newSessionId]: []
          },
          // Reset current conversation state
          conversationMessages: {},
          currentConversationId: null,
          // Branches are derived on demand via getBranches selector
          currentBranchId: 'main'
        }));
        
        return newSessionId;
      },
      
      resetToFreshSession: () => {
        // Create a fresh session in SessionManager
        const newSessionId = SessionManager.resetToFreshSession();
        const newSession = SessionManager.getSession(newSessionId);
        
        if (!newSession) {
          console.error('Failed to create fresh session');
          return get().currentSessionId;
        }
        
        // Clear the current conversation state when starting fresh
        set({
          conversationMessages: {},
          currentConversationId: null,
          // Branches are derived on demand via getBranches selector
          currentBranchId: 'main',
          currentSessionId: newSessionId,
          activeSessionIds: [newSessionId],
          sessions: {
            [newSessionId]: newSession
          },
          conversationsBySession: {
            [newSessionId]: []
          }
        });
        return newSessionId;
      },
      
      loadSession: (sessionId: string) => {
        // Get session data from SessionManager
        const session = SessionManager.getSession(sessionId);
        if (!session) return false;
        
        // Add to active sessions if not already active
        set(state => {
          const activeSessionIds = state.activeSessionIds.includes(sessionId) 
            ? state.activeSessionIds 
            : [...state.activeSessionIds, sessionId];
          
          return {
            sessions: {
              ...state.sessions,
              [sessionId]: session
            },
            activeSessionIds
          };
        });
        
        return true;
      },
      
      switchSession: (sessionId: string) => {
        // Verify session exists
        const session = SessionManager.getSession(sessionId);
        if (!session) return false;
        
        // Switch current session in SessionManager
        const switched = SessionManager.switchToSession(sessionId);
        if (!switched) return false;
        
        // Update store with new current session
        set(state => ({
          currentSessionId: sessionId,
          // Add to active sessions if not already active
          activeSessionIds: state.activeSessionIds.includes(sessionId)
            ? state.activeSessionIds
            : [...state.activeSessionIds, sessionId],
          sessions: {
            ...state.sessions,
            [sessionId]: session
          }
        }));
        
        return true;
      },
      
      updateSessionMetadata: (sessionId: string, updates: Partial<Session>) => {
        // Update session in SessionManager
        const updated = SessionManager.updateSession(sessionId, updates);
        if (!updated) return false;
        
        // Get updated session data
        const updatedSession = SessionManager.getSession(sessionId);
        if (!updatedSession) return false;
        
        // Update store with updated session
        set(state => ({
          sessions: {
            ...state.sessions,
            [sessionId]: updatedSession
          }
        }));
        
        return true;
      },
      
      deleteSession: (sessionId: string) => {
        // Can't delete the current session
        if (sessionId === get().currentSessionId) return false;
        
        // Delete session in SessionManager
        const deleted = SessionManager.deleteSession(sessionId);
        if (!deleted) return false;
        
        // Update store removing the session
        set(state => {
          const { [sessionId]: removedSession, ...remainingSessions } = state.sessions;
          const { [sessionId]: removedConversations, ...remainingConversationsBySession } = state.conversationsBySession;
          
          return {
            sessions: remainingSessions,
            activeSessionIds: state.activeSessionIds.filter(id => id !== sessionId),
            conversationsBySession: remainingConversationsBySession
          };
        });
        
        return true;
      },
      
      getAllSessions: () => {
        // Return all sessions from the store
        return Object.values(get().sessions);
      },
      
      getSessionById: (sessionId: string) => {
        // Return specific session
        return get().sessions[sessionId];
      },

      // Utility actions
      clearMessages: () =>
        set((state) => {
          // Clear messages for current conversation and branch
          if (state.currentConversationId) {
            const { currentConversationId, currentBranchId, conversationMessages } = state;
            
            // Create updated conversation messages
            const updatedConversationMessages = {
              ...conversationMessages,
              [currentConversationId]: {
                ...(conversationMessages[currentConversationId] || {}),
                [currentBranchId]: []
              }
            };
            
            // Update branch details using getBranches instead of direct property access
            // Note: We don't need to update branch details here as they are derived on demand
            // via getBranches() and will reflect the empty messages array
            
            return {
              conversationMessages: updatedConversationMessages
            };
          }
          
          return state;
        }),

      resetStore: () => {
        // Get a fresh session
        const sessionId = SessionManager.resetToFreshSession();
        const session = SessionManager.getSession(sessionId) || {
          id: sessionId,
          name: 'New Session',
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          conversationIds: []
        };
        
        set({
          conversationMessages: {},
          currentBranchId: 'main',
          isLoading: false,
          isTyping: false,
          conversations: [],
          currentConversationId: null,
          currentSessionId: sessionId,
          activeSessionIds: [sessionId],
          sessions: {
            [sessionId]: session
          },
          conversationsBySession: {
            [sessionId]: []
          },
          generationParams: {
            temperature: 0.7,
            maxTokens: 2048,
            topP: 0.9,
            contextLength: 8192,
            stream: false,
          },
          settings: {
            apiKeys: {},
            selectedProvider: '',
            selectedModel: '',
            usageMonitoring: true,
            autoValidateKeys: true,
            startNewSessionOnLaunch: state.settings?.startNewSessionOnLaunch ?? true,
            notifications: {
              lowBalance: true,
              highUsage: true,
              keyExpiry: true,
            },
            autoSave: {
              enabled: true,
              intervalMinutes: 5,
            },
            huggingFace: {
              username: undefined,
              token: undefined,
            },
          },
        });
      },
    }),
    {
      name: 'chatterley-settings',
      partialize: (state) => ({
        // Only persist app settings and generation parameters.
        // Do NOT persist conversations/messages/branches to ensure fresh sessions by default.
        settings: {
          ...state.settings,
          // Do not persist full key values; persist only metadata and last4 if present
          apiKeys: Object.fromEntries(
            Object.entries(state.settings.apiKeys).map(([providerId, config]) => [
              providerId,
              {
                ...config,
                keyValue: '', // strip secrets from persisted state
              },
            ])
          ),
        },
        generationParams: state.generationParams,
      }),
      onRehydrateStorage: () => (state) => {
        if (state?.settings) {
          // Decrypt API keys after loading from storage
          if (state.settings.apiKeys) {
            const decryptedApiKeys = Object.fromEntries(
              Object.entries(state.settings.apiKeys).map(([providerId, config]) => [
                providerId,
                {
                  ...config,
                  keyValue: config.keyValue ? decryptApiKey(config.keyValue) : undefined,
                },
              ])
            );
            state.settings.apiKeys = decryptedApiKeys;
          }

          // Migrate missing settings for existing users
          if (typeof state.settings.startNewSessionOnLaunch !== 'boolean') {
            state.settings.startNewSessionOnLaunch = true;
          }
          if (!state.settings.huggingFace) {
            state.settings.huggingFace = {
              username: undefined,
              token: undefined,
            };
          }

          if (!state.settings.autoSave) {
            state.settings.autoSave = {
              enabled: true,
              intervalMinutes: 5,
            };
          }

          // Ensure all required notification settings exist
          if (!state.settings.notifications) {
            state.settings.notifications = {
              lowBalance: true,
              highUsage: true,
              keyExpiry: true,
            };
          } else {
            // Fill in any missing notification settings
            const defaultNotifications = {
              lowBalance: true,
              highUsage: true,
              keyExpiry: true,
            };
            state.settings.notifications = {
              ...defaultNotifications,
              ...state.settings.notifications,
            };
          }
        }

        // Always start with a fresh chat on app load (by default):
        // clear any rehydrated conversation state and, if configured,
        // create a new session for this launch.
        if (state) {
          // Determine session to use on launch
          const shouldStartNew = state.settings?.startNewSessionOnLaunch !== false;
          const sessionId = shouldStartNew
            ? SessionManager.resetToFreshSession()
            : SessionManager.getCurrentSessionId();

          // Load all existing sessions from SessionManager (after potential reset)
          const allSessions = SessionManager.getAllSessions();
          // Create objects with typed indices
          const sessionsMap: {[key: string]: Session} = {};
          const conversationsBySessionMap: {[key: string]: string[]} = {};
          const currentSession = SessionManager.getCurrentSession() || {
            id: sessionId,
            name: 'New Session',
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
            conversationIds: []
          };
          
          // Add all sessions to the store
          allSessions.forEach(session => {
            sessionsMap[session.id] = session;
          });
          
          // Make sure current session is in the map
          sessionsMap[sessionId] = currentSession;
          
          // Initialize conversations map for the session
          conversationsBySessionMap[sessionId] = [];
          
          // Update store state
          state.conversationMessages = {};
          state.conversations = [];
          state.currentConversationId = null;
          state.currentBranchId = 'main';
          state.currentSessionId = sessionId;
          state.sessions = sessionsMap;
          state.activeSessionIds = [sessionId];
          state.conversationsBySession = conversationsBySessionMap;
        }
      },
    }
  )
);
