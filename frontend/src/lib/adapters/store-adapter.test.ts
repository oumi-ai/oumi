/**
 * Tests for store-adapter.ts
 * 
 * This file contains tests to verify the functionality of the store adapter utility
 * which helps migrate between legacy and branch-aware formats.
 */

import { 
  adaptLegacyConversation, 
  flattenBranchMessages,
  normalizedToLegacy,
  legacyToNormalized,
  buildBranchStructure,
  getBranchMetadata
} from './store-adapter';

import { Message, Conversation } from '../types';

// Mock test data
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

describe('Store Adapter', () => {
  describe('adaptLegacyConversation', () => {
    it('should convert legacy conversation to branch-aware format', () => {
      // Create a legacy format conversation
      const legacyConversation: any = {
        id: 'conv1',
        title: 'Test Conversation',
        messages: [mockMessage1, mockMessage2],
        createdAt: '2023-10-15T12:00:00Z',
        updatedAt: '2023-10-15T12:01:00Z'
      };

      // Adapt to new format
      const result = adaptLegacyConversation(legacyConversation);

      // Verify structure
      expect(result.id).toBe('conv1');
      expect(result.title).toBe('Test Conversation');
      expect(result.messages).toEqual([mockMessage1, mockMessage2]); // Should keep original messages
      expect(result.branches).toBeDefined();
      // Add non-null assertion since we've already checked branches is defined
      expect(result.branches!.main).toBeDefined();
      expect(result.branches!.main.messages).toEqual([mockMessage1, mockMessage2]);
    });
  });

  describe('flattenBranchMessages', () => {
    it('should extract messages from main branch', () => {
      const conversation: Conversation = {
        id: 'conv1',
        title: 'Test Conversation',
        messages: [], // Empty legacy array
        branches: {
          main: {
            messages: [mockMessage1, mockMessage2]
          }
        },
        createdAt: '2023-10-15T12:00:00Z',
        updatedAt: '2023-10-15T12:01:00Z'
      };

      const result = flattenBranchMessages(conversation);
      
      expect(result).toEqual([mockMessage1, mockMessage2]);
    });

    it('should fall back to legacy messages if branches are empty', () => {
      const conversation: Conversation = {
        id: 'conv1',
        title: 'Test Conversation',
        messages: [mockMessage1, mockMessage2],
        branches: {}, // Empty branches
        createdAt: '2023-10-15T12:00:00Z',
        updatedAt: '2023-10-15T12:01:00Z'
      };

      const result = flattenBranchMessages(conversation);
      
      expect(result).toEqual([mockMessage1, mockMessage2]);
    });
  });

  describe('normalizedToLegacy', () => {
    it('should extract messages from normalized format', () => {
      const normalizedMessages = {
        'conv1': {
          'main': [mockMessage1, mockMessage2],
          'branch1': [mockMessage1]
        }
      };

      const result = normalizedToLegacy('conv1', 'main', normalizedMessages);
      
      expect(result).toEqual([mockMessage1, mockMessage2]);
    });

    it('should return empty array for non-existent conversation', () => {
      const normalizedMessages = {
        'conv1': {
          'main': [mockMessage1, mockMessage2]
        }
      };

      const result = normalizedToLegacy('conv2', 'main', normalizedMessages);
      
      expect(result).toEqual([]);
    });
  });

  describe('legacyToNormalized', () => {
    it('should convert flat message array to normalized structure', () => {
      const messages = [mockMessage1, mockMessage2];
      
      const result = legacyToNormalized('conv1', 'main', messages);
      
      expect(result).toEqual({
        'conv1': {
          'main': [mockMessage1, mockMessage2]
        }
      });
    });
  });

  describe('buildBranchStructure', () => {
    it('should build branch structure from normalized messages', () => {
      const normalizedMessages = {
        'conv1': {
          'main': [mockMessage1, mockMessage2],
          'branch1': [mockMessage1]
        }
      };
      
      const result = buildBranchStructure('conv1', normalizedMessages);
      
      expect(result).toEqual({
        'main': {
          messages: [mockMessage1, mockMessage2]
        },
        'branch1': {
          messages: [mockMessage1]
        }
      });
    });

    it('should ensure main branch exists even if no messages', () => {
      const normalizedMessages = {
        'conv1': {
          'branch1': [mockMessage1]
        }
      };
      
      const result = buildBranchStructure('conv1', normalizedMessages);
      
      expect(result.main).toBeDefined();
      expect(result.branch1).toBeDefined();
    });
  });

  describe('getBranchMetadata', () => {
    it('should generate branch metadata from messages', () => {
      const normalizedMessages = {
        'conv1': {
          'main': [mockMessage1, mockMessage2],
          'branch1': [mockMessage1]
        }
      };
      
      const result = getBranchMetadata('conv1', normalizedMessages, 'main');
      
      expect(result).toHaveLength(2);
      expect(result[0].id).toBe('main');
      expect(result[0].isActive).toBe(true);
      expect(result[0].messageCount).toBe(2);
      
      expect(result[1].id).toBe('branch1');
      expect(result[1].isActive).toBe(false);
      expect(result[1].messageCount).toBe(1);
    });
  });
});