/**
 * Global chat store using Zustand for state management
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { Message, ConversationBranch, Conversation, GenerationParams, AppSettings, ApiKeyConfig, ApiProvider, ApiUsageStats } from './types';
import apiClient from './unified-api';
import { SessionManager } from './session-manager';

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
  
  // Session management actions
  getCurrentSessionId: () => string;
  startNewSession: () => string;
  resetToFreshSession: () => string;
  
  // Utility actions
  clearMessages: () => void;
  resetStore: () => void;
}

// Auto-save helper function
const autoSaveConversation = async (conversation: Conversation) => {
  try {
    const sessionId = SessionManager.getCurrentSessionId();
    console.log('[HISTORY_MERGE] Auto-saving conversation to backend:', { 
      sessionId, 
      conversationId: conversation.id, 
      title: conversation.title,
      messageCount: conversation.messages.length 
    });
    await apiClient.saveConversation(sessionId, conversation.id, conversation);
    console.log('[HISTORY_MERGE] Auto-save completed successfully');
  } catch (error) {
    console.warn('[HISTORY_MERGE] Failed to auto-save conversation to backend:', error);
    // Don't block the UI - this is just a backup save
  }
};

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
          const currentTime = new Date().toISOString();
          
          // Always create or update conversations for immediate availability
          if (!state.currentConversationId) {
            // Create new conversation immediately when first message is sent
            const isFirstUserMessage = message.role === 'user' && newMessages.length === 1;
            const title = isFirstUserMessage 
              ? 'New Chat'  // Use generic title initially, will be updated after first response
              : message.content.slice(0, 50) + (message.content.length > 50 ? '...' : '');
            
            const newConversation: Conversation = {
              id: `conv-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
              title,
              messages: newMessages,
              createdAt: currentTime,
              updatedAt: currentTime,
            };
            
            console.log('[HISTORY_MERGE] Creating new conversation:', { 
              id: newConversation.id, 
              title: newConversation.title, 
              isFirstUserMessage,
              messageCount: newMessages.length 
            });
            
            // Auto-save to backend asynchronously
            autoSaveConversation(newConversation);
            
            return {
              messages: newMessages,
              conversations: [...state.conversations, newConversation],
              currentConversationId: newConversation.id,
            };
          } else {
            // Update existing conversation
            const updatedConversations = state.conversations.map((conv) =>
              conv.id === state.currentConversationId
                ? { ...conv, messages: newMessages, updatedAt: currentTime }
                : conv
            );
            
            // Auto-save updated conversation to backend
            const updatedConv = updatedConversations.find(c => c.id === state.currentConversationId);
            if (updatedConv) {
              autoSaveConversation(updatedConv);
            }
            
            return { 
              messages: newMessages, 
              conversations: updatedConversations 
            };
          }
        }),

      setMessages: (messages: Message[]) =>
        set({ messages }),

      updateMessage: (messageId: string, updates: Partial<Message>) =>
        set((state) => {
          const newMessages = state.messages.map((msg) =>
            msg.id === messageId ? { ...msg, ...updates } : msg
          );
          
          // Always update current conversation if it exists
          if (state.currentConversationId) {
            const currentTime = new Date().toISOString();
            const updatedConversations = state.conversations.map((conv) =>
              conv.id === state.currentConversationId
                ? { ...conv, messages: newMessages, updatedAt: currentTime }
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
                conversationId: state.currentConversationId 
              });
              
              // If this is the first assistant response and we have a generic title, update it
              if (assistantMessages.length === 1 && userMessages.length >= 1) {
                const currentConv = updatedConversations.find(c => c.id === state.currentConversationId);
                console.log('[CHAT_NAMING] Checking for title generation:', { 
                  hasConv: !!currentConv, 
                  currentTitle: currentConv?.title,
                  shouldGenerate: currentConv && (currentConv.title === 'New Chat' || currentConv.title.startsWith('New Conversation'))
                });
                
                if (currentConv && (currentConv.title === 'New Chat' || currentConv.title.startsWith('New Conversation'))) {
                  console.log('[CHAT_NAMING] Triggering title generation for conversation:', currentConv.id);
                  // Trigger title generation asynchronously
                  setTimeout(() => {
                    get().generateChatTitle(state.currentConversationId!);
                  }, 100);
                } else {
                  console.log('[CHAT_NAMING] Skipping title generation - conditions not met');
                }
              } else {
                console.log('[CHAT_NAMING] Skipping title generation - not first assistant response or missing user messages');
              }
            }
            
            // Auto-save updated conversation to backend
            const updatedConv = updatedConversations.find(c => c.id === state.currentConversationId);
            if (updatedConv) {
              autoSaveConversation(updatedConv);
            }
            
            return { 
              messages: newMessages, 
              conversations: updatedConversations 
            };
          }
          
          return { messages: newMessages };
        }),

      deleteMessage: (messageId: string) =>
        set((state) => {
          const newMessages = state.messages.filter((msg) => msg.id !== messageId);
          
          // Update current conversation if it exists
          if (state.currentConversationId) {
            const currentTime = new Date().toISOString();
            const updatedConversations = state.conversations.map((conv) =>
              conv.id === state.currentConversationId
                ? { ...conv, messages: newMessages, updatedAt: currentTime }
                : conv
            );
            
            // Auto-save updated conversation to backend
            const updatedConv = updatedConversations.find(c => c.id === state.currentConversationId);
            if (updatedConv) {
              autoSaveConversation(updatedConv);
            }
            
            return { 
              messages: newMessages, 
              conversations: updatedConversations 
            };
          }
          
          return { messages: newMessages };
        }),

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

      // Chat naming actions
      generateChatTitle: async (conversationId: string) => {
        console.log('[CHAT_NAMING] generateChatTitle called for:', conversationId);
        const state = get();
        const conversation = state.conversations.find(conv => conv.id === conversationId);
        
        console.log('[CHAT_NAMING] Conversation found:', { 
          hasConversation: !!conversation, 
          messageCount: conversation?.messages.length || 0,
          title: conversation?.title 
        });
        
        if (!conversation || conversation.messages.length < 2) {
          console.log('[CHAT_NAMING] Skipping - no conversation or insufficient messages');
          return;
        }

        // Get the first user message and first assistant response
        const firstUserMsg = conversation.messages.find(m => m.role === 'user');
        const firstAssistantMsg = conversation.messages.find(m => m.role === 'assistant');
        
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
          messages: [],
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
        });
        return newSessionId;
      },

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
        // Only persist app settings and generation parameters.
        // Do NOT persist conversations/messages/branches to ensure fresh sessions by default.
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

        // Always start with a fresh chat on app load: clear any rehydrated
        // conversation state, messages, branches, and start a new session.
        if (state) {
          // Start fresh session for the new app instance
          SessionManager.resetToFreshSession();
          
          state.messages = [];
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
        }
      },
    }
  )
);
