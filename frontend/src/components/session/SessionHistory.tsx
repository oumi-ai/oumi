/**
 * SessionHistory component to display conversations grouped by session
 */

"use client";

import React, { useState, useEffect } from 'react';
import { Calendar, Clock, ChevronDown, ChevronRight, MessageSquare, Search, Filter } from 'lucide-react';
import { useChatStore } from '@/lib/store';
import { Session, Conversation } from '@/lib/types';

interface SessionHistoryProps {
  className?: string;
  onConversationSelect?: (conversationId: string) => void;
}

export default function SessionHistory({ className = '', onConversationSelect }: SessionHistoryProps) {
  const { 
    conversations, 
    sessions,
    currentSessionId,
    currentConversationId,
    getAllSessions,
    getSessionConversations,
    loadConversation 
  } = useChatStore();
  
  // Local state
  const [expandedSessions, setExpandedSessions] = useState<Set<string>>(new Set([currentSessionId]));
  const [searchQuery, setSearchQuery] = useState('');
  const [filterDate, setFilterDate] = useState<string>('all'); // 'all', 'today', 'week', 'month'
  const [sortBy, setSortBy] = useState<string>('date'); // 'date', 'name'
  const [allSessions, setAllSessions] = useState<Session[]>([]);
  
  // Load sessions
  useEffect(() => {
    const loadedSessions = getAllSessions();
    // Sort by most recently updated first
    const sortedSessions = [...loadedSessions].sort((a, b) => 
      new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
    );
    setAllSessions(sortedSessions);
  }, [getAllSessions, sessions]);
  
  // Toggle session expansion
  const toggleSessionExpansion = (sessionId: string) => {
    const newExpanded = new Set(expandedSessions);
    if (expandedSessions.has(sessionId)) {
      newExpanded.delete(sessionId);
    } else {
      newExpanded.add(sessionId);
    }
    setExpandedSessions(newExpanded);
  };
  
  // Handle conversation selection
  const handleConversationClick = async (conversationId: string) => {
    await loadConversation(conversationId);
    if (onConversationSelect) {
      onConversationSelect(conversationId);
    }
  };
  
  // Format date for display
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString(undefined, { 
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };
  
  // Format relative time
  const formatRelativeTime = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return formatDate(dateString);
  };
  
  // Filter conversations by date
  const filterConversationsByDate = (conversations: Conversation[]) => {
    if (filterDate === 'all') return conversations;
    
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const oneWeekAgo = new Date(now);
    oneWeekAgo.setDate(now.getDate() - 7);
    const oneMonthAgo = new Date(now);
    oneMonthAgo.setMonth(now.getMonth() - 1);
    
    return conversations.filter(conv => {
      const convDate = new Date(conv.updatedAt);
      if (filterDate === 'today') {
        return convDate >= today;
      } else if (filterDate === 'week') {
        return convDate >= oneWeekAgo;
      } else if (filterDate === 'month') {
        return convDate >= oneMonthAgo;
      }
      return true;
    });
  };
  
  // Filter and sort conversations
  const getFilteredConversations = (sessionId: string) => {
    // Get conversation IDs for this session
    const sessionConvIds = getSessionConversations(sessionId);
    
    // Get full conversation objects
    let sessionConversations = conversations.filter(
      conv => sessionConvIds.includes(conv.id)
    );
    
    // Apply date filter
    sessionConversations = filterConversationsByDate(sessionConversations);
    
    // Apply search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      sessionConversations = sessionConversations.filter(
        conv => conv.title.toLowerCase().includes(query)
      );
    }
    
    // Apply sorting
    if (sortBy === 'date') {
      sessionConversations.sort((a, b) => 
        new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
      );
    } else if (sortBy === 'name') {
      sessionConversations.sort((a, b) => a.title.localeCompare(b.title));
    }
    
    return sessionConversations;
  };
  
  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Search and filters */}
      <div className="p-3 border-b border-border">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search conversations..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-4 py-2 text-sm rounded-md border border-border bg-background"
          />
        </div>
        
        <div className="flex items-center justify-between mt-3 text-xs">
          <div className="flex items-center space-x-1.5">
            <Filter className="h-3.5 w-3.5 text-muted-foreground" />
            <select
              value={filterDate}
              onChange={(e) => setFilterDate(e.target.value)}
              className="px-1 py-1 rounded border border-border bg-background"
            >
              <option value="all">All time</option>
              <option value="today">Today</option>
              <option value="week">This week</option>
              <option value="month">This month</option>
            </select>
          </div>
          
          <div className="flex items-center space-x-1.5">
            <span className="text-muted-foreground">Sort:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="px-1 py-1 rounded border border-border bg-background"
            >
              <option value="date">Recent</option>
              <option value="name">Name</option>
            </select>
          </div>
        </div>
      </div>
      
      {/* Sessions list */}
      <div className="flex-1 overflow-y-auto">
        {allSessions.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
            <MessageSquare className="h-12 w-12 opacity-20 mb-2" />
            <p>No sessions found</p>
          </div>
        ) : (
          <div className="divide-y divide-border">
            {allSessions.map(session => {
              const sessionConversations = getFilteredConversations(session.id);
              if (sessionConversations.length === 0 && searchQuery) {
                return null; // Hide sessions with no matching conversations
              }
              
              return (
                <div key={session.id} className="bg-background">
                  {/* Session header */}
                  <div 
                    className={`flex items-center py-3 px-4 cursor-pointer hover:bg-muted transition-colors ${
                      session.id === currentSessionId ? 'bg-muted/50' : ''
                    }`}
                    onClick={() => toggleSessionExpansion(session.id)}
                  >
                    <div className="mr-2 text-muted-foreground">
                      {expandedSessions.has(session.id) ? (
                        <ChevronDown className="h-4 w-4" />
                      ) : (
                        <ChevronRight className="h-4 w-4" />
                      )}
                    </div>
                    
                    <div className="flex-1">
                      <div className="font-medium">
                        {session.name}
                        {session.id === currentSessionId && (
                          <span className="ml-2 text-xs px-1.5 py-0.5 rounded-full bg-primary/20 text-primary">
                            Current
                          </span>
                        )}
                      </div>
                      <div className="flex items-center space-x-3 text-xs text-muted-foreground mt-1">
                        <div className="flex items-center space-x-1">
                          <Calendar className="h-3 w-3" />
                          <span>{formatDate(session.createdAt)}</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <MessageSquare className="h-3 w-3" />
                          <span>
                            {sessionConversations.length} {sessionConversations.length === 1 ? 'conversation' : 'conversations'}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {/* Conversations list */}
                  {expandedSessions.has(session.id) && (
                    <div className="divide-y divide-border/50 bg-muted/30">
                      {sessionConversations.length === 0 ? (
                        <div className="px-8 py-3 text-sm text-muted-foreground italic">
                          {searchQuery ? 'No conversations match your search' : 'No conversations in this session'}
                        </div>
                      ) : (
                        sessionConversations.map(conversation => (
                          <div
                            key={conversation.id}
                            className={`px-8 py-3 cursor-pointer hover:bg-muted transition-colors ${
                              conversation.id === currentConversationId ? 'bg-primary/10 border-l-4 border-l-primary' : ''
                            }`}
                            onClick={() => handleConversationClick(conversation.id)}
                          >
                            <div className="font-medium text-sm">{conversation.title}</div>
                            <div className="flex items-center space-x-3 text-xs text-muted-foreground mt-1">
                              <div className="flex items-center space-x-1">
                                <Clock className="h-3 w-3" />
                                <span>{formatRelativeTime(conversation.updatedAt)}</span>
                              </div>
                              <div className="flex items-center space-x-1">
                                <MessageSquare className="h-3 w-3" />
                                <span>
                                  {useChatStore.getState().getBranchMessages(conversation.id, 'main').length} messages
                                </span>
                              </div>
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}