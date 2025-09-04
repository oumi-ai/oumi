/**
 * SessionSwitcher component for managing and switching between sessions
 */

"use client";

import React, { useState, useEffect } from 'react';
import { Plus, History, Settings, ChevronDown, Check, Edit, Trash2, Calendar, MessageSquare } from 'lucide-react';
import { useChatStore } from '@/lib/store';
import { Session } from '@/lib/types';

interface SessionSwitcherProps {
  className?: string;
}

export default function SessionSwitcher({ className = '' }: SessionSwitcherProps) {
  const {
    sessions,
    currentSessionId,
    getAllSessions,
    switchSession,
    startNewSession,
    updateSessionMetadata,
    deleteSession
  } = useChatStore();
  
  const [isOpen, setIsOpen] = useState(false);
  const [allSessions, setAllSessions] = useState<Session[]>([]);
  const [editingSession, setEditingSession] = useState<string | null>(null);
  const [editName, setEditName] = useState('');
  const [editDescription, setEditDescription] = useState('');
  
  // Get current session
  const currentSession = useChatStore.getState().getSessionById(currentSessionId);

  // Load all sessions
  useEffect(() => {
    const sessions = getAllSessions();
    // Sort by most recently updated first
    const sortedSessions = [...sessions].sort((a, b) => 
      new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
    );
    setAllSessions(sortedSessions);
  }, [getAllSessions, sessions]);

  const handleSwitchSession = (sessionId: string) => {
    if (switchSession(sessionId)) {
      setIsOpen(false);
    }
  };

  const handleCreateSession = () => {
    const newSessionId = startNewSession();
    setIsOpen(false);
  };
  
  const handleEditClick = (session: Session, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingSession(session.id);
    setEditName(session.name);
    setEditDescription(session.description || '');
  };
  
  const handleDeleteClick = (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm('Are you sure you want to delete this session? All conversations in this session will be removed.')) {
      deleteSession(sessionId);
    }
  };
  
  const handleSaveEdit = (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    updateSessionMetadata(sessionId, {
      name: editName,
      description: editDescription,
      updatedAt: new Date().toISOString()
    });
    setEditingSession(null);
  };
  
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString(undefined, { 
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };
  
  return (
    <div className={`relative ${className}`}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-2 px-3 py-2 rounded-md border border-border hover:bg-muted transition-colors"
      >
        <History className="w-4 h-4" />
        <span>{currentSession?.name || 'Session'}</span>
        <ChevronDown className="w-4 h-4" />
      </button>

      {isOpen && (
        <div className="absolute z-50 mt-2 w-64 rounded-md shadow-lg bg-background border border-border">
          <div className="max-h-80 overflow-y-auto">
            {/* Session list */}
            {allSessions.map((session) => (
              <div
                key={session.id}
                onClick={() => handleSwitchSession(session.id)}
                className={`px-4 py-3 hover:bg-muted transition-colors cursor-pointer ${
                  session.id === currentSessionId ? 'bg-muted' : ''
                }`}
              >
                {editingSession === session.id ? (
                  <div onClick={(e) => e.stopPropagation()} className="space-y-2">
                    <input
                      type="text"
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      className="w-full px-2 py-1 text-sm border rounded"
                      placeholder="Session name"
                    />
                    <textarea
                      value={editDescription}
                      onChange={(e) => setEditDescription(e.target.value)}
                      className="w-full px-2 py-1 text-sm border rounded"
                      placeholder="Description (optional)"
                      rows={2}
                    />
                    <div className="flex justify-end space-x-2">
                      <button
                        onClick={() => setEditingSession(null)}
                        className="px-2 py-1 text-xs rounded border hover:bg-muted"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={(e) => handleSaveEdit(session.id, e)}
                        className="px-2 py-1 text-xs rounded bg-primary text-primary-foreground hover:bg-primary/80"
                      >
                        Save
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="flex justify-between">
                      <div className="font-medium">
                        {session.name}
                        {session.id === currentSessionId && (
                          <span className="ml-2">
                            <Check className="w-3.5 h-3.5 inline-block text-primary" />
                          </span>
                        )}
                      </div>
                      <div className="flex space-x-1">
                        <button
                          onClick={(e) => handleEditClick(session, e)}
                          className="text-muted-foreground hover:text-foreground transition-colors"
                        >
                          <Edit className="w-3.5 h-3.5" />
                        </button>
                        {session.id !== currentSessionId && (
                          <button
                            onClick={(e) => handleDeleteClick(session.id, e)}
                            className="text-muted-foreground hover:text-destructive transition-colors"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        )}
                      </div>
                    </div>
                    {session.description && (
                      <div className="text-xs text-muted-foreground mt-1">
                        {session.description}
                      </div>
                    )}
                    <div className="flex items-center space-x-3 text-xs text-muted-foreground mt-2">
                      <div className="flex items-center space-x-1">
                        <Calendar className="w-3 h-3" />
                        <span>{formatDate(session.updatedAt)}</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <MessageSquare className="w-3 h-3" />
                        <span>{session.conversationIds.length} conversations</span>
                      </div>
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>
          
          {/* Create new session button */}
          <div className="border-t border-border">
            <button
              onClick={handleCreateSession}
              className="flex items-center space-x-2 w-full px-4 py-3 text-sm text-left hover:bg-muted transition-colors"
            >
              <Plus className="w-4 h-4" />
              <span>Create New Session</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}