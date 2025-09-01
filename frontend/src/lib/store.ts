/**
 * Global chat store using Zustand for state management
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { Message, ConversationBranch, Conversation, GenerationParams, AppSettings, ApiKeyConfig, ApiProvider, ApiUsageStats } from './types';

interface ChatStore {
  // Current state
  messages: Message[];
  branches: ConversationBranch[];
  currentBranchId: string;
  isLoading: boolean;
  isTyping: boolean;
  generationParams: GenerationParams;
  
  // Conversation management
  conversations: Conversation[];
  currentConversationId: string | null;
  
  // Settings and API management
  settings: AppSettings;

  // Actions
  addMessage: (message: Message) => void;
  setMessages: (messages: Message[]) => void;
  updateMessage: (messageId: string, updates: Partial<Message>) => void;
  deleteMessage: (messageId: string) => void;
  
  setBranches: (branches: ConversationBranch[]) => void;
  addBranch: (branch: ConversationBranch) => void;
  deleteBranch: (branchId: string) => void;
  setCurrentBranch: (branchId: string) => void;
  
  // Conversation actions
  addConversation: (conversation: Conversation) => void;
  updateConversation: (conversationId: string, updates: Partial<Conversation>) => void;
  deleteConversation: (conversationId: string) => void;
  setCurrentConversationId: (conversationId: string | null) => void;
  loadConversation: (conversationId: string) => Promise<Conversation | null>;
  
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
  
  // Utility actions
  clearMessages: () => void;
  resetStore: () => void;
}

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
      // Initial state
      messages: [],
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

      // Message actions
      addMessage: (message: Message) =>
        set((state) => {
          const newMessages = [...state.messages, message];
          const updatedState = { messages: newMessages };
          
          // Auto-save: Update current conversation or create new one
          if (state.settings.autoSave?.enabled !== false) {
            const currentTime = new Date().toISOString();
            
            if (state.currentConversationId) {
              // Update existing conversation
              const updatedConversations = state.conversations.map((conv) =>
                conv.id === state.currentConversationId
                  ? { ...conv, messages: newMessages, updatedAt: currentTime }
                  : conv
              );
              return { ...updatedState, conversations: updatedConversations };
            } else {
              // Create new conversation
              const firstMessage = newMessages[0];
              const title = firstMessage 
                ? firstMessage.content.slice(0, 50) + (firstMessage.content.length > 50 ? '...' : '')
                : 'New Conversation';
              
              const newConversation: Conversation = {
                id: `conv-${Date.now()}`,
                title,
                messages: newMessages,
                createdAt: currentTime,
                updatedAt: currentTime,
              };
              
              return {
                ...updatedState,
                conversations: [...state.conversations, newConversation],
                currentConversationId: newConversation.id,
              };
            }
          }
          
          return updatedState;
        }),

      setMessages: (messages: Message[]) =>
        set({ messages }),

      updateMessage: (messageId: string, updates: Partial<Message>) =>
        set((state) => {
          const newMessages = state.messages.map((msg) =>
            msg.id === messageId ? { ...msg, ...updates } : msg
          );
          const updatedState = { messages: newMessages };
          
          // Auto-save: Update current conversation
          if (state.settings.autoSave?.enabled !== false && state.currentConversationId) {
            const currentTime = new Date().toISOString();
            const updatedConversations = state.conversations.map((conv) =>
              conv.id === state.currentConversationId
                ? { ...conv, messages: newMessages, updatedAt: currentTime }
                : conv
            );
            return { ...updatedState, conversations: updatedConversations };
          }
          
          return updatedState;
        }),

      deleteMessage: (messageId: string) =>
        set((state) => ({
          messages: state.messages.filter((msg) => msg.id !== messageId),
        })),

      // Branch actions
      setBranches: (branches: ConversationBranch[]) =>
        set({ branches }),

      addBranch: (branch: ConversationBranch) =>
        set((state) => ({
          branches: [...state.branches, branch],
        })),

      deleteBranch: (branchId: string) =>
        set((state) => ({
          branches: state.branches.filter((branch) => branch.id !== branchId),
        })),

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
        set((state) => ({
          conversations: [...state.conversations, conversation],
        })),

      updateConversation: (conversationId: string, updates: Partial<Conversation>) =>
        set((state) => ({
          conversations: state.conversations.map((conv) =>
            conv.id === conversationId ? { ...conv, ...updates } : conv
          ),
        })),

      deleteConversation: (conversationId: string) =>
        set((state) => ({
          conversations: state.conversations.filter((conv) => conv.id !== conversationId),
          currentConversationId: state.currentConversationId === conversationId ? null : state.currentConversationId,
        })),

      setCurrentConversationId: (conversationId: string | null) =>
        set({ currentConversationId: conversationId }),

      loadConversation: async (conversationId: string) => {
        const state = get();
        const conversation = state.conversations.find((conv) => conv.id === conversationId);
        if (conversation) {
          // Load the conversation data into current state
          set({
            messages: conversation.messages,
            currentConversationId: conversationId,
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
          const newApiKey: ApiKeyConfig = {
            providerId,
            keyValue, // Store as-is, encryption handled by persist middleware
            isActive: true,
            createdAt: new Date().toISOString(),
            isValid: true, // Assume valid since we just added it
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
          
          return {
            settings: {
              ...state.settings,
              apiKeys: {
                ...state.settings.apiKeys,
                [providerId]: { ...existingKey, ...updates },
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

      // Utility actions
      clearMessages: () =>
        set({ messages: [] }),

      resetStore: () =>
        set({
          messages: [],
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
        settings: {
          ...state.settings,
          // Encrypt API keys before persisting
          apiKeys: Object.fromEntries(
            Object.entries(state.settings.apiKeys).map(([providerId, config]) => [
              providerId,
              {
                ...config,
                keyValue: encryptApiKey(config.keyValue),
              },
            ])
          ),
        },
        generationParams: state.generationParams,
        // Persist conversation history
        conversations: state.conversations,
        currentConversationId: state.currentConversationId,
        messages: state.messages,
        branches: state.branches,
        currentBranchId: state.currentBranchId,
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
                  keyValue: decryptApiKey(config.keyValue),
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
      },
    }
  )
);