/**
 * Global chat store using Zustand for state management
 */

import { create } from 'zustand';
import { Message, ConversationBranch, GenerationParams } from './types';

interface ChatStore {
  // Current state
  messages: Message[];
  branches: ConversationBranch[];
  currentBranchId: string;
  isLoading: boolean;
  isTyping: boolean;
  generationParams: GenerationParams;

  // Actions
  addMessage: (message: Message) => void;
  setMessages: (messages: Message[]) => void;
  updateMessage: (messageId: string, updates: Partial<Message>) => void;
  deleteMessage: (messageId: string) => void;
  
  setBranches: (branches: ConversationBranch[]) => void;
  addBranch: (branch: ConversationBranch) => void;
  deleteBranch: (branchId: string) => void;
  setCurrentBranch: (branchId: string) => void;
  
  setLoading: (loading: boolean) => void;
  setTyping: (typing: boolean) => void;
  setGenerationParams: (params: Partial<GenerationParams>) => void;
  
  // Utility actions
  clearMessages: () => void;
  resetStore: () => void;
}

export const useChatStore = create<ChatStore>((set, get) => ({
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
  generationParams: {
    temperature: 0.7,
    max_tokens: 2048,
    top_p: 0.9,
    stream: false,
  },

  // Message actions
  addMessage: (message: Message) =>
    set((state) => ({
      messages: [...state.messages, message],
    })),

  setMessages: (messages: Message[]) =>
    set({ messages }),

  updateMessage: (messageId: string, updates: Partial<Message>) =>
    set((state) => ({
      messages: state.messages.map((msg) =>
        msg.id === messageId ? { ...msg, ...updates } : msg
      ),
    })),

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

  // UI state actions
  setLoading: (loading: boolean) =>
    set({ isLoading: loading }),

  setTyping: (typing: boolean) =>
    set({ isTyping: typing }),

  setGenerationParams: (params: Partial<GenerationParams>) =>
    set((state) => ({
      generationParams: { ...state.generationParams, ...params },
    })),

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
        max_tokens: 2048,
        top_p: 0.9,
        stream: false,
      },
    }),
}));