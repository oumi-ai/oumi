/**
 * Tests for the branch-aware store implementation
 * 
 * This file contains tests for the new branch-aware data structure, selectors,
 * and compatibility adapters.
 */

import { useChatStore } from './store';
import { adaptLegacyConversation, flattenBranchMessages } from './adapters/store-adapter';
import { Message, Conversation } from './types';

// Mock message data for testing
const mockMessage1: Message = {
  id: 'msg1',
  role: 'user',
  content: 'Hello, how are you?',
  timestamp: Date.now() - 1000
};

const mockMessage2: Message = {
  id: 'msg2',
  role: 'assistant',
  content: "I'm an AI assistant and I'm functioning well. How can I help you today?",
  timestamp: Date.now()
};

// Test the branch-aware store functionality
describe('Branch-Aware Store', () => {
  // Clear store state before each test
  beforeEach(() => {
    useChatStore.setState({
      conversationMessages: {},
      conversations: [],
      currentConversationId: null,
      currentBranchId: 'main',
      branches: [
        {
          id: 'main',
          name: 'Main',
          isActive: true,
          messageCount: 0,
          createdAt: new Date().toISOString(),
          lastActive: new Date().toISOString(),
        }
      ],
    });
  });

  describe('Branch-specific message storage', () => {
    it('should store messages for specific branches', () => {
      const store = useChatStore.getState();
      const conversationId = 'test-conversation';
      const branchId1 = 'main';
      const branchId2 = 'branch2';
      
      // Add conversation and set as current
      store.addConversation({
        id: conversationId,
        title: 'Test Conversation',
        messages: [],
        branches: {},
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      });
      store.setCurrentConversationId(conversationId);
      
      // Add messages to different branches
      store.setMessages(conversationId, branchId1, [mockMessage1]);
      store.setMessages(conversationId, branchId2, [mockMessage2]);
      
      // Verify branch isolation
      const branch1Messages = store.getBranchMessages(conversationId, branchId1);
      const branch2Messages = store.getBranchMessages(conversationId, branchId2);
      
      expect(branch1Messages).toHaveLength(1);
      expect(branch2Messages).toHaveLength(1);
      expect(branch1Messages[0].content).toBe(mockMessage1.content);
      expect(branch2Messages[0].content).toBe(mockMessage2.content);
    });

    it('should update branch messages independently', () => {
      const store = useChatStore.getState();
      const conversationId = 'test-conversation';
      const branchId1 = 'main';
      const branchId2 = 'branch2';
      
      // Add conversation and branches
      store.addConversation({
        id: conversationId,
        title: 'Test Conversation',
        messages: [],
        branches: {
          [branchId1]: { messages: [mockMessage1] },
          [branchId2]: { messages: [mockMessage2] }
        },
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      });
      store.setCurrentConversationId(conversationId);
      
      // Add a message to branch1 only
      const newMessage: Message = {
        id: 'msg3',
        role: 'user',
        content: 'This is a new message',
        timestamp: Date.now()
      };
      
      // Set current branch to branch1
      store.setCurrentBranch(branchId1);
      
      // Add message
      store.addMessage(newMessage);
      
      // Verify branch isolation
      const branch1Messages = store.getBranchMessages(conversationId, branchId1);
      const branch2Messages = store.getBranchMessages(conversationId, branchId2);
      
      expect(branch1Messages).toHaveLength(2);
      expect(branch2Messages).toHaveLength(1);
      expect(branch1Messages[1].content).toBe(newMessage.content);
    });
  });

  describe('Selectors', () => {
    it('getCurrentMessages should return messages for current branch', () => {
      const store = useChatStore.getState();
      const conversationId = 'test-conversation';
      const branchId1 = 'main';
      const branchId2 = 'branch2';
      
      // Add conversation with branches
      store.addConversation({
        id: conversationId,
        title: 'Test Conversation',
        messages: [],
        branches: {
          [branchId1]: { messages: [mockMessage1] },
          [branchId2]: { messages: [mockMessage2] }
        },
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      });
      
      // Set current conversation and branch
      store.setCurrentConversationId(conversationId);
      store.setCurrentBranch(branchId1);
      
      // Test selector
      expect(store.getCurrentMessages()).toHaveLength(1);
      expect(store.getCurrentMessages()[0].content).toBe(mockMessage1.content);
      
      // Switch branch
      store.setCurrentBranch(branchId2);
      
      // Test selector after switch
      expect(store.getCurrentMessages()).toHaveLength(1);
      expect(store.getCurrentMessages()[0].content).toBe(mockMessage2.content);
    });
    
    it('getBranchMessages should return messages for specific branch', () => {
      const store = useChatStore.getState();
      const conversationId = 'test-conversation';
      const branchId1 = 'main';
      const branchId2 = 'branch2';
      
      // Add conversation with branches
      store.setMessages(conversationId, branchId1, [mockMessage1]);
      store.setMessages(conversationId, branchId2, [mockMessage2]);
      
      // Test selector
      expect(store.getBranchMessages(conversationId, branchId1)).toHaveLength(1);
      expect(store.getBranchMessages(conversationId, branchId2)).toHaveLength(1);
      expect(store.getBranchMessages(conversationId, branchId1)[0].content).toBe(mockMessage1.content);
      expect(store.getBranchMessages(conversationId, branchId2)[0].content).toBe(mockMessage2.content);
    });
  });

  describe('Compatibility adapters', () => {
    it('adaptLegacyConversation should convert legacy format to branch-aware', () => {
      // Create legacy conversation
      const legacyConversation: Conversation = {
        id: 'legacy-conversation',
        title: 'Legacy Conversation',
        messages: [mockMessage1, mockMessage2],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      };
      
      // Adapt to branch-aware format
      const adaptedConversation = adaptLegacyConversation(legacyConversation);
      
      // Test adapted conversation
      expect(adaptedConversation.branches).toBeDefined();
      expect(adaptedConversation.branches?.main).toBeDefined();
      expect(adaptedConversation.branches?.main.messages).toHaveLength(2);
      expect(adaptedConversation.branches?.main.messages[0].content).toBe(mockMessage1.content);
      expect(adaptedConversation.branches?.main.messages[1].content).toBe(mockMessage2.content);
    });
    
    it('flattenBranchMessages should extract all messages from conversation', () => {
      // Create branch-aware conversation
      const branchConversation: Conversation = {
        id: 'branch-conversation',
        title: 'Branch Conversation',
        messages: [],
        branches: {
          main: { messages: [mockMessage1] },
          branch2: { messages: [mockMessage2] }
        },
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      };
      
      // Use adapter to flatten messages
      const messages = flattenBranchMessages(branchConversation);
      
      // By default, should return main branch messages
      expect(messages).toHaveLength(1);
      expect(messages[0].content).toBe(mockMessage1.content);
    });
  });

  describe('Branch operations', () => {
    it('should add a new branch', () => {
      const store = useChatStore.getState();
      const conversationId = 'test-conversation';
      const newBranchId = 'new-branch';
      
      // Add conversation
      store.addConversation({
        id: conversationId,
        title: 'Test Conversation',
        messages: [],
        branches: { main: { messages: [] } },
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      });
      store.setCurrentConversationId(conversationId);
      
      // Add new branch
      store.addBranch({
        id: newBranchId,
        name: 'New Branch',
        isActive: false,
        messageCount: 0,
        createdAt: new Date().toISOString(),
        lastActive: new Date().toISOString(),
      });
      
      // Verify branch was added
      expect(store.branches).toHaveLength(2);
      expect(store.branches[1].id).toBe(newBranchId);
      
      // Verify branch has empty messages array
      expect(store.getBranchMessages(conversationId, newBranchId)).toHaveLength(0);
    });
    
    it('should delete a branch', () => {
      const store = useChatStore.getState();
      const conversationId = 'test-conversation';
      const branchToDeleteId = 'branch-to-delete';
      
      // Add conversation with multiple branches
      store.addConversation({
        id: conversationId,
        title: 'Test Conversation',
        messages: [],
        branches: {
          main: { messages: [] },
          [branchToDeleteId]: { messages: [] }
        },
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      });
      store.setCurrentConversationId(conversationId);
      
      // Add branches to store branch list
      store.setBranches([
        {
          id: 'main',
          name: 'Main',
          isActive: true,
          messageCount: 0,
          createdAt: new Date().toISOString(),
          lastActive: new Date().toISOString(),
        },
        {
          id: branchToDeleteId,
          name: 'Branch to Delete',
          isActive: false,
          messageCount: 0,
          createdAt: new Date().toISOString(),
          lastActive: new Date().toISOString(),
        }
      ]);
      
      // Delete branch
      store.deleteBranch(branchToDeleteId);
      
      // Verify branch was deleted
      expect(store.branches).toHaveLength(1);
      expect(store.branches[0].id).toBe('main');
      
      // Verify branch messages were deleted
      expect(store.conversationMessages[conversationId][branchToDeleteId]).toBeUndefined();
    });
    
    it('should switch between branches', () => {
      const store = useChatStore.getState();
      const conversationId = 'test-conversation';
      const branch1Id = 'main';
      const branch2Id = 'branch2';
      
      // Add conversation with branches
      store.addConversation({
        id: conversationId,
        title: 'Test Conversation',
        messages: [],
        branches: {
          [branch1Id]: { messages: [mockMessage1] },
          [branch2Id]: { messages: [mockMessage2] }
        },
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      });
      
      // Add branches to store branch list
      store.setBranches([
        {
          id: branch1Id,
          name: 'Main',
          isActive: true,
          messageCount: 1,
          createdAt: new Date().toISOString(),
          lastActive: new Date().toISOString(),
        },
        {
          id: branch2Id,
          name: 'Branch 2',
          isActive: false,
          messageCount: 1,
          createdAt: new Date().toISOString(),
          lastActive: new Date().toISOString(),
        }
      ]);
      
      // Set current conversation
      store.setCurrentConversationId(conversationId);
      
      // Set branch 1 as current
      store.setCurrentBranch(branch1Id);
      
      // Verify current messages are from branch 1
      expect(store.getCurrentMessages()).toHaveLength(1);
      expect(store.getCurrentMessages()[0].content).toBe(mockMessage1.content);
      
      // Switch to branch 2
      store.setCurrentBranch(branch2Id);
      
      // Verify current messages are now from branch 2
      expect(store.getCurrentMessages()).toHaveLength(1);
      expect(store.getCurrentMessages()[0].content).toBe(mockMessage2.content);
      
      // Verify branch isActive status was updated
      const updatedBranch1 = store.branches.find(b => b.id === branch1Id);
      const updatedBranch2 = store.branches.find(b => b.id === branch2Id);
      
      expect(updatedBranch1?.isActive).toBe(false);
      expect(updatedBranch2?.isActive).toBe(true);
    });
  });

  describe('Session management', () => {
    it('should associate conversations with sessions', () => {
      const store = useChatStore.getState();
      const sessionId = store.getCurrentSessionId();
      const conversationId = 'test-conversation';
      
      // Add conversation
      store.addConversation({
        id: conversationId,
        title: 'Test Conversation',
        messages: [],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      });
      
      // Verify conversation is associated with session
      const sessionConversations = store.getSessionConversations(sessionId);
      expect(sessionConversations).toContain(conversationId);
    });
  });
});