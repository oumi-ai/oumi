/**
 * Session management utility for chat sessions
 * Generates unique session IDs based on timestamps
 */

export class SessionManager {
  private static currentSessionId: string | null = null;
  private static sessionStartTime: number | null = null;

  /**
   * Generate a session ID from a timestamp
   */
  private static generateSessionId(timestamp: number): string {
    // Create a hash of the timestamp for a unique but deterministic session ID
    const hash = timestamp.toString(36) + Math.random().toString(36).substring(2, 8);
    return `session_${hash}`;
  }

  /**
   * Start a new session with current timestamp
   */
  static startNewSession(): string {
    const now = Date.now();
    this.sessionStartTime = now;
    this.currentSessionId = this.generateSessionId(now);
    console.log(`🆔 Started new session: ${this.currentSessionId}`);
    return this.currentSessionId;
  }

  /**
   * Get the current session ID, creating a new one if none exists
   */
  static getCurrentSessionId(): string {
    if (!this.currentSessionId) {
      return this.startNewSession();
    }
    return this.currentSessionId;
  }

  /**
   * Get the session start time
   */
  static getSessionStartTime(): number | null {
    return this.sessionStartTime;
  }

  /**
   * Check if we have an active session
   */
  static hasActiveSession(): boolean {
    return this.currentSessionId !== null;
  }

  /**
   * Clear the current session (for fresh starts)
   */
  static clearSession(): void {
    console.log(`🧹 Cleared session: ${this.currentSessionId}`);
    this.currentSessionId = null;
    this.sessionStartTime = null;
  }

  /**
   * Reset to a fresh session (clear + start new)
   */
  static resetToFreshSession(): string {
    this.clearSession();
    return this.startNewSession();
  }
}