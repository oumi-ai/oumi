/**
 * Global chat store using Zustand for state management
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { Message, ConversationBranch, Conversation, GenerationParams, AppSettings, ApiKeyConfig, ApiProvider, ApiUsageStats } from './types';
import apiClient from './unified-api';
import { 
  adaptLegacyConversation, 
  flattenBranchMessages, 
  normalizedToLegacy, 
  legacyToNormalized,
  buildBranchStructure,
  getBranchMetadata,
  SessionManager,
  autoSaveConversation
} from './adapters/store-adapter';

interface ChatStore {
  // Branch-specific message storage
  conversationMessages: {
    [conversationId: string]: {
      [branchId: string]: Message[]
    }
  };
  
  // Current state
  branches: ConversationBranch[];
  currentBranchId: string;
  isLoading: boolean;
  isTyping: boolean;
  generationParams: GenerationParams;
  
  // Conversation management
  conversations: Conversation[];
  currentConversationId: string | null;
  
  // Session management
  activeSessionIds: string[];
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
  
  setBranches: (branches: ConversationBranch[]) => void;
  addBranch: (branch: ConversationBranch) => void;
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
  
  // Session management actions
  getCurrentSessionId: () => string;
  startNewSession: () => string;
  resetToFreshSession: () => string;
  
  // Selectors
  getCurrentMessages: () => Message[];
  getBranchMessages: (conversationId: string, branchId: string) => Message[];
  getSessionConversations: (sessionId: string) => string[];
  
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
      branches: [
        {
          id: 'main',
          name: 'Main',
          isActive: true,
          messageCount: 0,
          createdAt: new Date().toISOString(),
          lastActive: new Date().toISOString(),
          preview: 'New conversation',
        },
      ],
      currentBranchId: 'main',
      isLoading: false,
      isTyping: false,
      
      // Conversation state
      conversations: [],
      currentConversationId: null,
      
      // Session management
      activeSessionIds: [SessionManager.getCurrentSessionId()],
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
            
            console.log('[HISTORY_MERGE] Creating new conversation:', { 
              id: newConversationId, 
              title,
              isFirstUserMessage,
              branchId: currentBranchId
            });
            
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
          
          // Update the conversation message store
          const updatedConversationMessages = {
            ...conversationMessages,
            [currentConversationId]: {
              ...(conversationMessages[currentConversationId] || {}),
              [currentBranchId]: newMessages
            }
          };
          
          // Update branch details
          const updatedBranches = state.branches.map((branch) => {
            if (branch.id === currentBranchId) {
              return {
                ...branch,
                messageCount: newMessages.length,
                lastActive: new Date().toISOString(),
                preview: message.content.slice(0, 50) + (message.content.length > 50 ? '...' : '')
              };
            }
            return branch;
          });
          
          // Update the conversation object
          const currentTime = new Date().toISOString();
          const updatedConversations = state.conversations.map((conv) =>
            conv.id === currentConversationId
              ? { 
                  ...conv, 
                  updatedAt: currentTime,
                  branches: {
                    ...(conv.branches || {}),
                    [currentBranchId]: { 
                      messages: newMessages
                    }
                  }
                }
              : conv
          );
          
          // Auto-save updated conversation to backend
          const updatedConv = updatedConversations.find(c => c.id === currentConversationId);
          if (updatedConv) {
            autoSaveConversation(updatedConv);
          }
          
          return { 
            conversationMessages: updatedConversationMessages,
            branches: updatedBranches,
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
          const updatedBranches = state.branches.map((branch) => {
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
          
          // Update the conversation object if it exists
          const currentTime = new Date().toISOString();
          const updatedConversations = state.conversations.map((conv) =>
            conv.id === conversationId
              ? { 
                  ...conv, 
                  updatedAt: currentTime,
                  branches: {
                    ...(conv.branches || {}),
                    [branchId]: { 
                      messages: messages
                    }
                  }
                }
              : conv
          );
          
          return {
            conversationMessages: updatedConversationMessages,
            branches: updatedBranches,
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
          
          // Update the conversation message store
          const updatedConversationMessages = {
            ...conversationMessages,
            [conversationId]: {
              ...(conversationMessages[conversationId]),
              [branchId]: newMessages
            }
          };
          
          // Update the conversation object
          const currentTime = new Date().toISOString();
          const updatedConversations = state.conversations.map((conv) =>
            conv.id === conversationId
              ? { 
                  ...conv, 
                  updatedAt: currentTime,
                  branches: {
                    ...(conv.branches || {}),
                    [branchId]: { 
                      messages: newMessages
                    }
                  }
                }
              : conv
          );
          
          // Check if this was the first assistant response and trigger title generation
          const updatedMessage = newMessages.find(msg => msg.id === messageId);
          console.log('[CHAT_NAMING] updateMessage called for message:', { messageId, role: updatedMessage?.role, messageCount: newMessages.length });
          
          if (updatedMessage?.role === 'assistant') {
            const userMessages = newMessages.filter(m => m.role === 'user');
            const assistantMessages = newMessages.filter(m => m.role === 'assistant');
            
            console.log('[CHAT_NAMING] Assistant message detected:', { 
              userCount: userMessages.length, 
              assistantCount: assistantMessages.length,
              conversationId
            });
            
            // If this is the first assistant response and we have a generic title, update it
            if (assistantMessages.length === 1 && userMessages.length >= 1) {
              const currentConv = updatedConversations.find(c => c.id === conversationId);
              console.log('[CHAT_NAMING] Checking for title generation:', { 
                hasConv: !!currentConv, 
                currentTitle: currentConv?.title,
                shouldGenerate: currentConv && (currentConv.title === 'New Chat' || currentConv.title.startsWith('New Conversation'))
              });
              
              if (currentConv && (currentConv.title === 'New Chat' || currentConv.title.startsWith('New Conversation'))) {
                console.log('[CHAT_NAMING] Triggering title generation for conversation:', currentConv.id);
                // Trigger title generation asynchronously
                setTimeout(() => {
                  get().generateChatTitle(conversationId);
                }, 100);
              } else {
                console.log('[CHAT_NAMING] Skipping title generation - conditions not met');
              }
            } else {
              console.log('[CHAT_NAMING] Skipping title generation - not first assistant response or missing user messages');
            }
          }
          
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
          
          // Update the conversation message store
          const updatedConversationMessages = {
            ...conversationMessages,
            [conversationId]: {
              ...(conversationMessages[conversationId]),
              [branchId]: newMessages
            }
          };
          
          // Update branch details
          const updatedBranches = state.branches.map((branch) => {
            if (branch.id === branchId) {
              return {
                ...branch,
                messageCount: newMessages.length,
                lastActive: new Date().toISOString()
              };
            }
            return branch;
          });
          
          // Update the conversation object
          const currentTime = new Date().toISOString();
          const updatedConversations = state.conversations.map((conv) =>
            conv.id === conversationId
              ? { 
                  ...conv, 
                  updatedAt: currentTime,
                  branches: {
                    ...(conv.branches || {}),
                    [branchId]: { 
                      messages: newMessages
                    }
                  }
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
            branches: updatedBranches, 
            conversations: updatedConversations 
          };
        }),

      // Branch actions
      setBranches: (branches: ConversationBranch[]) =>
        set({ branches }),

      addBranch: (branch: ConversationBranch) =>
        set((state) => {
          // If we have a current conversation, initialize the branch storage
          if (state.currentConversationId) {
            // Create empty branch messages array
            const conversationId = state.currentConversationId;
            const updatedConversationMessages = {
              ...state.conversationMessages,
              [conversationId]: {
                ...(state.conversationMessages[conversationId] || {}),
                [branch.id]: [] // Initialize with empty messages array
              }
            };
            
            // Update the conversation object
            const updatedConversations = state.conversations.map((conv) =>
              conv.id === conversationId
                ? { 
                    ...conv, 
                    updatedAt: new Date().toISOString(),
                    branches: {
                      ...(conv.branches || {}),
                      [branch.id]: { 
                        messages: []
                      }
                    }
                  }
                : conv
            );
            
            return {
              branches: [...state.branches, branch],
              conversationMessages: updatedConversationMessages,
              conversations: updatedConversations
            };
          }
          
          return {
            branches: [...state.branches, branch],
          };
        }),

      deleteBranch: (branchId: string) =>
        set((state) => {
          // Cannot delete if it's the only branch or the main branch
          if (state.branches.length === 1 || branchId === 'main') {
            return state;
          }
          
          // If this is the current branch, switch to main first
          let updatedCurrentBranchId = state.currentBranchId;
          if (branchId === state.currentBranchId) {
            updatedCurrentBranchId = 'main';
          }
          
          // Update branch list with active status
          const updatedBranches = state.branches
            .filter((branch) => branch.id !== branchId)
            .map((branch) => ({
              ...branch,
              isActive: branch.id === updatedCurrentBranchId
            }));
          
          // If we have a current conversation, remove the branch from storage
          let updatedConversationMessages = state.conversationMessages;
          let updatedConversations = state.conversations;
          
          if (state.currentConversationId) {
            const conversationId = state.currentConversationId;
            
            // Remove branch from conversation messages
            if (state.conversationMessages[conversationId]) {
              const { [branchId]: removed, ...remainingBranches } = state.conversationMessages[conversationId];
              updatedConversationMessages = {
                ...state.conversationMessages,
                [conversationId]: remainingBranches
              };
            }
            
            // Remove branch from conversation object
            updatedConversations = state.conversations.map((conv) => {
              if (conv.id === conversationId && conv.branches) {
                const { [branchId]: removed, ...remainingBranches } = conv.branches;
                return { 
                  ...conv, 
                  updatedAt: new Date().toISOString(),
                  branches: remainingBranches
                };
              }
              return conv;
            });
            
            // Auto-save updated conversation to backend
            const updatedConv = updatedConversations.find(c => c.id === conversationId);
            if (updatedConv) {
              autoSaveConversation(updatedConv);
            }
          }
          
          return {
            branches: updatedBranches,
            currentBranchId: updatedCurrentBranchId,
            conversationMessages: updatedConversationMessages,
            conversations: updatedConversations
          };
        }),

      setCurrentBranch: (branchId: string) =>
        set((state) => ({
          currentBranchId: branchId,
          branches: state.branches.map((branch) => ({
            ...branch,
            isActive: branch.id === branchId,
          })),
        })),

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
          
          return {
            conversations: [...state.conversations, conversation],
            conversationMessages: {
              ...state.conversationMessages,
              [conversation.id]: branchMessages
            },
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
        // Find the conversation in the store
        const conversation = state.conversations.find((conv) => conv.id === conversationId);
        
        if (conversation) {
          const targetBranchId = branchId || 'main';
          
          // Check if we have messages for this branch
          let branchMessages: Message[] = [];
          
          if (state.conversationMessages[conversationId]?.[targetBranchId]) {
            branchMessages = state.conversationMessages[conversationId][targetBranchId];
          } else if (conversation.branches?.[targetBranchId]) {
            // Get from conversation object if available
            branchMessages = conversation.branches[targetBranchId].messages;
          } else if (targetBranchId === 'main' && conversation.messages && conversation.messages.length > 0) {
            // Fallback for legacy format - use main conversation messages and adapt
            branchMessages = conversation.messages;
            
            // Use adapter to convert legacy format to normalized format
            console.log('[STORE] Using adapter to convert legacy conversation format', { 
              conversationId, 
              messageCount: conversation.messages.length
            });
            
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
        console.log('[CHAT_NAMING] generateChatTitle called for:', conversationId);
        const state = get();
        const conversation = state.conversations.find(conv => conv.id === conversationId);
        
        console.log('[CHAT_NAMING] Conversation found:', { 
          hasConversation: !!conversation
        });
        
        if (!conversation) {
          console.log('[CHAT_NAMING] Skipping - no conversation found');
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
        
        console.log('[CHAT_NAMING] Messages found:', { 
          hasUserMsg: !!firstUserMsg, 
          hasAssistantMsg: !!firstAssistantMsg,
          userContent: firstUserMsg?.content.slice(0, 50) + '...',
          assistantContent: firstAssistantMsg?.content.slice(0, 50) + '...'
        });
        
        if (!firstUserMsg || !firstAssistantMsg) {
          console.log('[CHAT_NAMING] Skipping - missing required messages');
          return;
        }

        // Generate a meaningful title based on the conversation
        let generatedTitle = '';
        try {
          // Simple heuristic: use key terms from user question and assistant response
          const userContent = firstUserMsg.content.toLowerCase();
          const assistantContent = firstAssistantMsg.content.toLowerCase();
          
          console.log('[CHAT_NAMING] Processing user content:', userContent.slice(0, 100));
          
          // Extract potential topics/keywords
          if (userContent.includes('help') && userContent.includes('with')) {
            const match = userContent.match(/help.*with\s+(.+?)[\.\?\!]|$/);
            console.log('[CHAT_NAMING] Trying "help with" pattern, match:', match?.[1]);
            if (match && match[1]) {
              generatedTitle = `Help with ${match[1].trim()}`;
            }
          } else if (userContent.includes('how to')) {
            const match = userContent.match(/how to\s+(.+?)[\.\?\!]|$/);
            console.log('[CHAT_NAMING] Trying "how to" pattern, match:', match?.[1]);
            if (match && match[1]) {
              generatedTitle = `How to ${match[1].trim()}`;
            }
          } else if (userContent.includes('what is') || userContent.includes('what are')) {
            const match = userContent.match(/what (?:is|are)\s+(.+?)[\.\?\!]|$/);
            console.log('[CHAT_NAMING] Trying "what is/are" pattern, match:', match?.[1]);
            if (match && match[1]) {
              generatedTitle = `About ${match[1].trim()}`;
            }
          } else {
            console.log('[CHAT_NAMING] Using fallback pattern from user content');
            // Fallback: use first 40 characters of user message, cleaned up
            generatedTitle = firstUserMsg.content
              .replace(/[^\w\s]/g, ' ')
              .trim()
              .slice(0, 40)
              .replace(/\s+/g, ' ')
              .trim();
          }
          
          console.log('[CHAT_NAMING] Generated title before cleanup:', generatedTitle);
          
          // Clean up and capitalize
          if (generatedTitle) {
            generatedTitle = generatedTitle.charAt(0).toUpperCase() + generatedTitle.slice(1);
            if (generatedTitle.length > 50) {
              generatedTitle = generatedTitle.slice(0, 47) + '...';
            }
          }
          
          console.log('[CHAT_NAMING] Final generated title:', generatedTitle);
        } catch (error) {
          console.error('[CHAT_NAMING] Error generating chat title:', error);
        }

        // Fallback to user message if generation failed
        if (!generatedTitle) {
          console.log('[CHAT_NAMING] Using fallback title from user message');
          generatedTitle = firstUserMsg.content.slice(0, 47) + '...';
        }

        // Update the conversation title
        console.log('[CHAT_NAMING] Updating conversation title:', { conversationId, generatedTitle });
        set((state) => {
          const updatedConversations = state.conversations.map((conv) =>
            conv.id === conversationId
              ? { ...conv, title: generatedTitle, updatedAt: new Date().toISOString() }
              : conv
          );
          
          // Auto-save updated conversation to backend
          const updatedConv = updatedConversations.find(c => c.id === conversationId);
          if (updatedConv) {
            console.log('[CHAT_NAMING] Auto-saving updated conversation with new title');
            autoSaveConversation(updatedConv);
          }
          
          return { conversations: updatedConversations };
        });
        
        console.log('[CHAT_NAMING] Title generation completed for conversation:', conversationId);
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
            autoSaveConversation(updatedConv);
          }
          
          return { conversations: updatedConversations };
        }),

      // Session management actions
      getCurrentSessionId: () => SessionManager.getCurrentSessionId(),
      
      startNewSession: () => SessionManager.startNewSession(),
      
      resetToFreshSession: () => {
        const newSessionId = SessionManager.resetToFreshSession();
        // Also clear the current conversation state when starting fresh
        set({
          conversationMessages: {},
          currentConversationId: null,
          branches: [
            {
              id: 'main',
              name: 'Main',
              isActive: true,
              messageCount: 0,
              createdAt: new Date().toISOString(),
              lastActive: new Date().toISOString(),
              preview: 'New conversation',
            },
          ],
          currentBranchId: 'main',
          activeSessionIds: [newSessionId],
          conversationsBySession: {
            [newSessionId]: []
          }
        });
        return newSessionId;
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
            
            // Update branch details
            const updatedBranches = state.branches.map((branch) => {
              if (branch.id === currentBranchId) {
                return {
                  ...branch,
                  messageCount: 0,
                  lastActive: new Date().toISOString(),
                  preview: 'Empty conversation'
                };
              }
              return branch;
            });
            
            return {
              conversationMessages: updatedConversationMessages,
              branches: updatedBranches
            };
          }
          
          return state;
        }),

      resetStore: () =>
        set({
          conversationMessages: {},
          branches: [
            {
              id: 'main',
              name: 'Main',
              isActive: true,
              messageCount: 0,
              createdAt: new Date().toISOString(),
              lastActive: new Date().toISOString(),
              preview: 'New conversation',
            },
          ],
          currentBranchId: 'main',
          isLoading: false,
          isTyping: false,
          conversations: [],
          currentConversationId: null,
          activeSessionIds: [SessionManager.getCurrentSessionId()],
          conversationsBySession: {},
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
        }),
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

        // Always start with a fresh chat on app load: clear any rehydrated
        // conversation state, messages, branches, and start a new session.
        if (state) {
          // Start fresh session for the new app instance
          const sessionId = SessionManager.resetToFreshSession();
          
          state.conversationMessages = {};
          state.conversations = [];
          state.currentConversationId = null;
          state.branches = [
            {
              id: 'main',
              name: 'Main',
              isActive: true,
              messageCount: 0,
              createdAt: new Date().toISOString(),
              lastActive: new Date().toISOString(),
              preview: 'New conversation',
            },
          ];
          state.currentBranchId = 'main';
          state.activeSessionIds = [sessionId];
          state.conversationsBySession = {
            [sessionId]: []
          };
        }
      },
    }
  )
);
