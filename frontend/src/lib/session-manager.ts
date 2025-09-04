/**
 * SessionManager - Advanced session management utility for chat sessions
 * 
 * Provides:
 * - Unique session ID generation
 * - Persistent session storage (localStorage)
 * - Session metadata tracking
 * - Multi-session support
 * 
 * IMPORTANT: This is the canonical SessionManager implementation.
 * Do not create alternative implementations.
 */

import { Session } from './types';

/**
 * SessionManager singleton class for managing user sessions
 */
export class SessionManager {
  private static instance: SessionManager;
  private currentSessionId: string;
  private sessions: Map<string, Session>;
  private storageKey = 'oumi_sessions';

  private constructor() {
    // Load existing sessions from storage
    this.sessions = new Map<string, Session>();
    this.loadSessions();
    
    // Get last active session or create a new one
    const lastSessionId = localStorage.getItem('oumi_current_session');
    if (lastSessionId && this.sessions.has(lastSessionId)) {
      this.currentSessionId = lastSessionId;
    } else {
      this.currentSessionId = this.createNewSession('New Session');
    }
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
   * Get the current session data
   */
  public static getCurrentSession(): Session | undefined {
    const instance = SessionManager.getInstance();
    return instance.sessions.get(instance.currentSessionId);
  }

  /**
   * Get all sessions
   */
  public static getAllSessions(): Session[] {
    const instance = SessionManager.getInstance();
    return Array.from(instance.sessions.values());
  }

  /**
   * Get a session by ID
   */
  public static getSession(sessionId: string): Session | undefined {
    return SessionManager.getInstance().sessions.get(sessionId);
  }

  /**
   * Load sessions from storage
   */
  private loadSessions(): void {
    try {
      const sessionsJson = localStorage.getItem(this.storageKey);
      if (sessionsJson) {
        const sessionsArray: Session[] = JSON.parse(sessionsJson);
        sessionsArray.forEach(session => {
          this.sessions.set(session.id, session);
        });
      }
    } catch (error) {
      console.error('Failed to load sessions from storage:', error);
    }
  }

  /**
   * Save sessions to storage
   */
  private saveSessions(): void {
    try {
      const sessionsArray = Array.from(this.sessions.values());
      localStorage.setItem(this.storageKey, JSON.stringify(sessionsArray));
      localStorage.setItem('oumi_current_session', this.currentSessionId);
    } catch (error) {
      console.error('Failed to save sessions to storage:', error);
    }
  }

  /**
   * Generate a new session ID
   */
  private generateSessionId(): string {
    return `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Create a new session with the given name
   */
  public createNewSession(name: string = 'New Session', metadata?: { [key: string]: any }): string {
    const sessionId = this.generateSessionId();
    const now = new Date().toISOString();
    
    const newSession: Session = {
      id: sessionId,
      name: name,
      createdAt: now,
      updatedAt: now,
      conversationIds: [],
      metadata: metadata || {}
    };
    
    this.sessions.set(sessionId, newSession);
    this.currentSessionId = sessionId;
    this.saveSessions();
    
    return sessionId;
  }
  
  /**
   * Reset to a fresh session
   */
  public resetToFreshSession(): string {
    return this.createNewSession('New Session');
  }
  
  /**
   * Start a new session (static helper)
   */
  public static startNewSession(name: string = 'New Session', metadata?: { [key: string]: any }): string {
    return SessionManager.getInstance().createNewSession(name, metadata);
  }
  
  /**
   * Reset to a fresh session (static helper)
   */
  public static resetToFreshSession(): string {
    return SessionManager.getInstance().resetToFreshSession();
  }
  
  /**
   * Update session metadata
   */
  public updateSession(sessionId: string, updates: Partial<Session>): boolean {
    const session = this.sessions.get(sessionId);
    if (!session) return false;
    
    const updatedSession = {
      ...session,
      ...updates,
      updatedAt: new Date().toISOString()
    };
    
    this.sessions.set(sessionId, updatedSession);
    this.saveSessions();
    return true;
  }
  
  /**
   * Update current session metadata (static helper)
   */
  public static updateCurrentSession(updates: Partial<Session>): boolean {
    const instance = SessionManager.getInstance();
    return instance.updateSession(instance.currentSessionId, updates);
  }
  
  /**
   * Update a session by ID (static helper)
   */
  public static updateSession(sessionId: string, updates: Partial<Session>): boolean {
    const instance = SessionManager.getInstance();
    return instance.updateSession(sessionId, updates);
  }
  
  /**
   * Switch to a different session
   */
  public switchSession(sessionId: string): boolean {
    if (!this.sessions.has(sessionId)) return false;
    
    this.currentSessionId = sessionId;
    this.saveSessions();
    return true;
  }
  
  /**
   * Switch to a session (static helper)
   */
  public static switchToSession(sessionId: string): boolean {
    return SessionManager.getInstance().switchSession(sessionId);
  }
  
  /**
   * Add a conversation to a session
   */
  public addConversationToSession(sessionId: string, conversationId: string): boolean {
    const session = this.sessions.get(sessionId);
    if (!session) return false;
    
    if (!session.conversationIds.includes(conversationId)) {
      session.conversationIds.push(conversationId);
      session.updatedAt = new Date().toISOString();
      this.sessions.set(sessionId, session);
      this.saveSessions();
    }
    
    return true;
  }
  
  /**
   * Add conversation to current session (static helper)
   */
  public static addConversation(conversationId: string): boolean {
    const instance = SessionManager.getInstance();
    return instance.addConversationToSession(instance.currentSessionId, conversationId);
  }
  
  /**
   * Delete a session and all associated conversations
   */
  public deleteSession(sessionId: string): boolean {
    if (!this.sessions.has(sessionId)) return false;
    
    this.sessions.delete(sessionId);
    
    // If we deleted the current session, switch to another one or create new
    if (this.currentSessionId === sessionId) {
      const sessionIds = Array.from(this.sessions.keys());
      if (sessionIds.length > 0) {
        this.currentSessionId = sessionIds[0];
      } else {
        this.currentSessionId = this.createNewSession('New Session');
      }
    }
    
    this.saveSessions();
    return true;
  }
  
  /**
   * Delete session (static helper)
   */
  public static deleteSession(sessionId: string): boolean {
    return SessionManager.getInstance().deleteSession(sessionId);
  }

  /**
   * Get the session start time
   */
  public static getSessionStartTime(sessionId?: string): string | null {
    const instance = SessionManager.getInstance();
    const targetSessionId = sessionId || instance.currentSessionId;
    const session = instance.sessions.get(targetSessionId);
    return session ? session.createdAt : null;
  }

  /**
   * Check if we have an active session
   */
  public static hasActiveSession(): boolean {
    const instance = SessionManager.getInstance();
    return instance.sessions.has(instance.currentSessionId);
  }

  /**
   * Clear the current session (for fresh starts)
   */
  public static clearSession(): void {
    const instance = SessionManager.getInstance();
    instance.deleteSession(instance.currentSessionId);
  }
}