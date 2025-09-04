/**
 * Tests for the SessionManager class
 */

import { SessionManager } from '../session-manager';
import { Session } from '../types';

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: jest.fn((key: string) => store[key] || null),
    setItem: jest.fn((key: string, value: string) => {
      store[key] = value.toString();
    }),
    clear: jest.fn(() => {
      store = {};
    }),
    removeItem: jest.fn((key: string) => {
      delete store[key];
    }),
  };
})();

Object.defineProperty(window, 'localStorage', { value: localStorageMock });

describe('SessionManager', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    localStorageMock.clear();
    // Reset SessionManager singleton instance
    // @ts-ignore - Access to private static property for testing
    SessionManager.instance = undefined;
  });

  describe('Session Creation', () => {
    it('should create a new session with default name', () => {
      const sessionId = SessionManager.startNewSession();
      expect(sessionId).toBeDefined();
      
      const session = SessionManager.getSession(sessionId);
      expect(session).toBeDefined();
      expect(session?.name).toBe('New Session');
      expect(session?.conversationIds).toEqual([]);
      expect(session?.metadata).toEqual({});
    });

    it('should create a new session with custom name and metadata', () => {
      const metadata = { modelId: 'gpt-4', purpose: 'research' };
      const sessionId = SessionManager.startNewSession('Research Project', metadata);
      
      const session = SessionManager.getSession(sessionId);
      expect(session?.name).toBe('Research Project');
      expect(session?.metadata).toEqual(metadata);
    });

    it('should reset to a fresh session', () => {
      // First create a session with some data
      const sessionId1 = SessionManager.startNewSession('First Session');
      SessionManager.addConversation('conv1');
      
      // Now reset to a fresh session
      const sessionId2 = SessionManager.resetToFreshSession();
      
      // Should be different IDs
      expect(sessionId2).not.toBe(sessionId1);
      
      // Current session should be the new one
      expect(SessionManager.getCurrentSessionId()).toBe(sessionId2);
      
      // Original session should still exist
      const session1 = SessionManager.getSession(sessionId1);
      expect(session1).toBeDefined();
      expect(session1?.conversationIds).toEqual(['conv1']);
      
      // New session should be empty
      const session2 = SessionManager.getSession(sessionId2);
      expect(session2?.conversationIds).toEqual([]);
    });
  });

  describe('Session Management', () => {
    it('should update session metadata', () => {
      const sessionId = SessionManager.startNewSession();
      
      // Update session metadata
      const updates = { 
        name: 'Updated Name',
        description: 'Test description',
        metadata: { modelId: 'gpt-3.5' }
      };
      
      const result = SessionManager.updateCurrentSession(updates);
      expect(result).toBe(true);
      
      // Check session was updated
      const session = SessionManager.getCurrentSession();
      expect(session?.name).toBe('Updated Name');
      expect(session?.description).toBe('Test description');
      expect(session?.metadata).toEqual({ modelId: 'gpt-3.5' });
    });

    it('should switch between sessions', () => {
      // Create two sessions
      const session1Id = SessionManager.startNewSession('Session 1');
      const session2Id = SessionManager.startNewSession('Session 2');
      
      // Current session should be session2
      expect(SessionManager.getCurrentSessionId()).toBe(session2Id);
      
      // Switch to session1
      const result = SessionManager.switchToSession(session1Id);
      expect(result).toBe(true);
      
      // Current session should now be session1
      expect(SessionManager.getCurrentSessionId()).toBe(session1Id);
    });

    it('should add conversations to a session', () => {
      const sessionId = SessionManager.startNewSession();
      
      // Add conversations to the session
      const added1 = SessionManager.addConversation('conv1');
      const added2 = SessionManager.addConversation('conv2');
      
      expect(added1).toBe(true);
      expect(added2).toBe(true);
      
      // Check conversations were added
      const session = SessionManager.getCurrentSession();
      expect(session?.conversationIds).toContain('conv1');
      expect(session?.conversationIds).toContain('conv2');
      expect(session?.conversationIds.length).toBe(2);
    });

    it('should delete a session', () => {
      // Create two sessions
      const session1Id = SessionManager.startNewSession('Session 1');
      const session2Id = SessionManager.startNewSession('Session 2');
      
      // Delete session1
      const result = SessionManager.deleteSession(session1Id);
      expect(result).toBe(true);
      
      // session1 should no longer exist
      const session1 = SessionManager.getSession(session1Id);
      expect(session1).toBeUndefined();
      
      // Current session should still be session2
      expect(SessionManager.getCurrentSessionId()).toBe(session2Id);
    });

    it('should get all sessions', () => {
      // Create a few sessions
      SessionManager.startNewSession('Session 1');
      SessionManager.startNewSession('Session 2');
      SessionManager.startNewSession('Session 3');
      
      // Get all sessions
      const sessions = SessionManager.getAllSessions();
      
      // Should have 3 sessions
      expect(sessions.length).toBe(3);
      
      // Check session names
      const names = sessions.map(s => s.name);
      expect(names).toContain('Session 1');
      expect(names).toContain('Session 2');
      expect(names).toContain('Session 3');
    });
  });

  describe('Persistence', () => {
    it('should save sessions to localStorage', () => {
      // Create some sessions
      const session1Id = SessionManager.startNewSession('Session 1');
      SessionManager.addConversation('conv1');
      
      const session2Id = SessionManager.startNewSession('Session 2');
      SessionManager.addConversation('conv2');
      
      // localStorage should have been called
      expect(localStorageMock.setItem).toHaveBeenCalled();
      
      // Reset the SessionManager singleton
      // @ts-ignore - Access to private static property for testing
      SessionManager.instance = undefined;
      
      // Re-initialize SessionManager (should load from localStorage)
      const currentId = SessionManager.getCurrentSessionId();
      
      // Should match the last session we created
      expect(currentId).toBe(session2Id);
      
      // Check sessions were loaded correctly
      const sessions = SessionManager.getAllSessions();
      expect(sessions.length).toBe(2);
      
      // Check session data was preserved
      const session1 = SessionManager.getSession(session1Id);
      expect(session1?.name).toBe('Session 1');
      expect(session1?.conversationIds).toContain('conv1');
      
      const session2 = SessionManager.getSession(session2Id);
      expect(session2?.name).toBe('Session 2');
      expect(session2?.conversationIds).toContain('conv2');
    });
  });
});