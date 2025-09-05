/**
 * Chat History Sidebar - Browse and load previous conversations
 */

"use client";

import React from 'react';
import { History, MessageSquare, Clock, Download, Search, Trash2, RefreshCw, ArrowLeft } from 'lucide-react';
import apiClient from '@/lib/unified-api';
import { useChatStore } from '@/lib/store';

interface ConversationEntry {
  id: string;
  name: string;
  lastModified: string;
  messageCount: number;
  preview: string;
  size?: string;
}

interface ConversationPreview {
  id: string;
  name: string;
  messages: Array<{
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: string;
  }>;
  messageCount: number;
  lastModified: string;
}

interface ChatHistorySidebarProps {
  className?: string;
}

export default function ChatHistorySidebar({ className = '' }: ChatHistorySidebarProps) {
  const { 
    currentBranchId, 
    conversations: storeConversations, 
    currentConversationId, 
    loadConversation: loadStoreConversation,
    setCurrentConversationId,
    setMessages,
    deleteConversation: deleteStoreConversation,
    getCurrentSessionId,
    getBranchMessages
  } = useChatStore();
  const [conversations, setConversations] = React.useState<ConversationEntry[]>([]);
  const [selectedConversation, setSelectedConversation] = React.useState<string | null>(null);
  const [conversationPreview, setConversationPreview] = React.useState<ConversationPreview | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [loadingPreview, setLoadingPreview] = React.useState(false);
  const [loadingConversation, setLoadingConversation] = React.useState(false);
  const [searchTerm, setSearchTerm] = React.useState('');
  const [error, setError] = React.useState<string | null>(null);

  // Load conversation list on mount
  React.useEffect(() => {
    loadConversations();
  }, []);

  // Sync with store conversations for real-time updates - merge with existing backend conversations
  React.useEffect(() => {
    if (storeConversations && storeConversations.length > 0) {
      console.log('[HISTORY_MERGE] Store conversations updated:', storeConversations.length, 'conversations');
      
      const convertedStoreConversations = storeConversations.map((conv) => ({
        id: conv.id,
        name: conv.title || 'Untitled Conversation',
        lastModified: conv.updatedAt || conv.createdAt,
        messageCount: conv.messages?.length || 0,
        preview: conv.messages && conv.messages.length > 0 
          ? conv.messages[conv.messages.length - 1].content.slice(0, 100)
          : 'No messages yet'
      }));

      // Merge store conversations with existing backend conversations
      setConversations(prevConversations => {
        console.log('[HISTORY_MERGE] Merging store conversations with existing backend conversations:', {
          backendCount: prevConversations.length,
          storeCount: convertedStoreConversations.length
        });
        
        const mergedConversations = [...prevConversations];
        let addedCount = 0;
        let updatedCount = 0;
        
        convertedStoreConversations.forEach(storeConv => {
          const existingIndex = mergedConversations.findIndex(conv => conv.id === storeConv.id);
          if (existingIndex >= 0) {
            // Update existing conversation
            mergedConversations[existingIndex] = storeConv;
            updatedCount++;
          } else {
            // Add new conversation
            mergedConversations.push(storeConv);
            addedCount++;
          }
        });

        console.log('[HISTORY_MERGE] Merge completed:', {
          totalConversations: mergedConversations.length,
          addedCount,
          updatedCount
        });

        // Sort by last modified (newest first)
        mergedConversations.sort((a, b) => 
          new Date(b.lastModified).getTime() - new Date(a.lastModified).getTime()
        );

        return mergedConversations;
      });
      setLoading(false);
    }
  }, [storeConversations]);

  const loadConversations = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const sessionId = getCurrentSessionId();
      console.log('[HISTORY_MERGE] Loading conversations from backend for session:', sessionId);
      const response = await apiClient.listConversations(sessionId);
      
      if (response.success && response.data?.conversations) {
        console.log('[HISTORY_MERGE] Backend returned', response.data.conversations.length, 'conversations');
        
        const conversations = response.data.conversations.map((conv: any) => ({
          id: conv.id || conv.filename,
          name: conv.name || conv.filename || 'Untitled Conversation',
          lastModified: conv.lastModified || conv.modified || new Date().toISOString(),
          messageCount: conv.messageCount || 0,
          preview: conv.preview || 'No preview available',
          size: conv.size || undefined
        }));
        
        // Sort by last modified (newest first)
        conversations.sort((a, b) => new Date(b.lastModified).getTime() - new Date(a.lastModified).getTime());
        
        console.log('[HISTORY_MERGE] Setting backend conversations:', conversations.length);
        setConversations(conversations);
      } else {
        console.log('[HISTORY_MERGE] No conversations returned from backend or request failed:', response.message);
        throw new Error(response.message || 'Failed to load conversations');
      }
    } catch (error) {
      console.error('[HISTORY_MERGE] Failed to load conversations:', error);
      setError(error instanceof Error ? error.message : 'Failed to load conversations');
      setConversations([]);
    } finally {
      setLoading(false);
    }
  };

  const loadConversationPreview = async (conversationId: string) => {
    try {
      setLoadingPreview(true);
      setError(null);
      
      // First try to load from store for faster access
      const storeConversation = storeConversations.find(c => c.id === conversationId);
      
      if (storeConversation) {
        // Get messages from the conversation (prefer branches, fallback to flat messages)
        const messages = storeConversation.branches?.main?.messages || storeConversation.messages || [];
        setConversationPreview({
          id: conversationId,
          name: storeConversation.title || 'Unknown Conversation',
          messages: messages.slice(0, 6).map(msg => ({
            role: msg.role,
            content: msg.content,
            timestamp: msg.timestamp ? new Date(msg.timestamp).toISOString() : new Date().toISOString()
          })), // Show first 6 messages as preview
          messageCount: messages.length,
          lastModified: storeConversation.updatedAt || storeConversation.createdAt
        });
      } else {
        // Fallback to API if not in store
        const sessionId = getCurrentSessionId();
        const response = await apiClient.loadConversation(sessionId, conversationId);
        
        if (response.success && response.data) {
          const messages = response.data.messages || [];
          const conversationData = conversations.find(c => c.id === conversationId);
          
          setConversationPreview({
            id: conversationId,
            name: conversationData?.name || 'Unknown Conversation',
            messages: messages.slice(0, 6).map(msg => ({
              role: msg.role,
              content: msg.content,
              timestamp: typeof msg.timestamp === 'number' ? new Date(msg.timestamp).toISOString() : msg.timestamp
            })), // Show first 6 messages as preview
            messageCount: messages.length,
            lastModified: conversationData?.lastModified || new Date().toISOString()
          });
        } else {
          throw new Error(response.message || 'Failed to load conversation preview');
        }
      }
    } catch (error) {
      console.error('Failed to load conversation preview:', error);
      setError(error instanceof Error ? error.message : 'Failed to load preview');
      setConversationPreview(null);
    } finally {
      setLoadingPreview(false);
    }
  };

  const loadConversationIntoBranch = async (conversationId: string) => {
    if (!conversationId) return;
    
    try {
      setLoadingConversation(true);
      setError(null);
      
      // Load the conversation from store - much faster and real-time
      const conversation = await loadStoreConversation(conversationId);
      
      if (conversation) {
        // Show success message
        alert('✅ Conversation loaded successfully into the current branch!');
        
        // Close the preview
        setSelectedConversation(null);
        setConversationPreview(null);
      } else {
        // Fallback to API if not in store
        const sessionId = getCurrentSessionId();
        const response = await apiClient.loadConversation(sessionId, conversationId, currentBranchId);
        
        if (response.success) {
          // Update store with loaded messages
          if (response.data?.messages) {
            setMessages(conversationId, currentBranchId || 'main', response.data.messages);
            setCurrentConversationId(conversationId);
          }
          
          alert('✅ Conversation loaded successfully into the current branch!');
          setSelectedConversation(null);
          setConversationPreview(null);
        } else {
          throw new Error(response.message || 'Failed to load conversation');
        }
      }
    } catch (error) {
      console.error('Failed to load conversation:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to load conversation';
      alert(`❌ ${errorMessage}`);
    } finally {
      setLoadingConversation(false);
    }
  };

  const deleteConversation = async (conversationId: string) => {
    if (!confirm('Are you sure you want to delete this conversation? This action cannot be undone.')) {
      return;
    }
    
    try {
      // Delete from store first for immediate UI update
      deleteStoreConversation(conversationId);
      
      // Clear preview if it was the deleted conversation
      if (selectedConversation === conversationId) {
        setSelectedConversation(null);
        setConversationPreview(null);
      }
      
      // If this was the current conversation, clear it
      if (currentConversationId === conversationId) {
        setCurrentConversationId(null);
        // The setMessages function now requires 3 parameters
        setMessages('', 'main', []);
      }
      
      // Also delete from backend/API for persistence
      try {
        const sessionId = getCurrentSessionId();
        await apiClient.deleteConversation(sessionId, conversationId);
      } catch (apiError) {
        console.warn('Failed to delete from backend, but deleted locally:', apiError);
      }
      
      alert('✅ Conversation deleted successfully');
    } catch (error) {
      console.error('Failed to delete conversation:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to delete conversation';
      alert(`❌ ${errorMessage}`);
    }
  };

  const handleConversationClick = (conversationId: string) => {
    if (selectedConversation === conversationId) {
      // Toggle off if already selected
      setSelectedConversation(null);
      setConversationPreview(null);
    } else {
      setSelectedConversation(conversationId);
      loadConversationPreview(conversationId);
    }
  };

  const filteredConversations = React.useMemo(() => {
    if (!searchTerm.trim()) return conversations;
    
    const term = searchTerm.toLowerCase();
    return conversations.filter(conv => 
      conv.name.toLowerCase().includes(term) ||
      conv.preview.toLowerCase().includes(term)
    );
  }, [conversations, searchTerm]);

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else if (diffDays === 1) {
      return 'Yesterday';
    } else if (diffDays < 7) {
      return `${diffDays} days ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  return (
    <div className={`bg-card border-l flex flex-col h-full ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center gap-2">
          <History size={18} />
          <h3 className="font-semibold text-foreground">Chat History</h3>
        </div>
        <button
          onClick={loadConversations}
          disabled={loading}
          className="p-1 hover:bg-muted rounded transition-colors text-muted-foreground hover:text-foreground disabled:opacity-50"
          title="Refresh conversations"
        >
          <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {/* Search */}
      <div className="p-3 border-b">
        <div className="relative">
          <Search size={16} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search conversations..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-9 pr-3 py-2 text-sm bg-background border border-border rounded-md focus:ring-2 focus:ring-primary focus:border-primary"
          />
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 flex">
        {/* Conversation List */}
        <div className="flex-1 overflow-y-auto">
          {error && (
            <div className="p-3 text-sm text-red-600 bg-red-50 dark:bg-red-900/20 border-b">
              {error}
            </div>
          )}
          
          {loading ? (
            <div className="p-4 text-center">
              <RefreshCw size={20} className="animate-spin mx-auto mb-2 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">Loading conversations...</p>
            </div>
          ) : filteredConversations.length === 0 ? (
            <div className="p-4 text-center">
              <MessageSquare size={32} className="mx-auto mb-3 text-muted-foreground opacity-50" />
              <p className="text-sm text-muted-foreground">
                {searchTerm ? 'No conversations match your search' : 'No saved conversations found'}
              </p>
            </div>
          ) : (
            <div className="space-y-1 p-2">
              {filteredConversations.map((conversation) => (
                <div
                  key={conversation.id}
                  className={`p-3 rounded-lg cursor-pointer transition-colors group relative ${
                    selectedConversation === conversation.id
                      ? 'bg-primary/10 border border-primary/20'
                      : currentConversationId === conversation.id
                      ? 'bg-accent/50 border border-accent/30'  // Active conversation style
                      : 'hover:bg-muted'
                  }`}
                  onClick={() => handleConversationClick(conversation.id)}
                >
                  {/* Delete button */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteConversation(conversation.id);
                    }}
                    className="absolute top-2 right-2 p-1 opacity-0 group-hover:opacity-100 hover:bg-red-100 dark:hover:bg-red-900/30 rounded transition-all"
                    title="Delete conversation"
                  >
                    <Trash2 size={12} className="text-red-600" />
                  </button>

                  <div className="pr-6">
                    <div className="flex items-center gap-2 mb-1">
                      <h4 className="font-medium text-sm text-foreground truncate">
                        {conversation.name}
                      </h4>
                      {currentConversationId === conversation.id && (
                        <div className="flex-shrink-0 w-2 h-2 bg-green-500 rounded-full" title="Active conversation" />
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground line-clamp-2 mb-2">
                      {conversation.preview}
                    </p>
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <div className="flex items-center gap-1">
                        <Clock size={10} />
                        {formatDate(conversation.lastModified)}
                      </div>
                      <div className="flex items-center gap-1">
                        <MessageSquare size={10} />
                        {conversation.messageCount} messages
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Conversation Preview */}
        {selectedConversation && (
          <div className="w-80 border-l bg-background overflow-y-auto">
            {loadingPreview ? (
              <div className="p-4 text-center">
                <RefreshCw size={20} className="animate-spin mx-auto mb-2 text-muted-foreground" />
                <p className="text-sm text-muted-foreground">Loading preview...</p>
              </div>
            ) : conversationPreview ? (
              <div className="p-4">
                <div className="flex items-center gap-2 mb-4">
                  <button
                    onClick={() => {
                      setSelectedConversation(null);
                      setConversationPreview(null);
                    }}
                    className="p-1 hover:bg-muted rounded transition-colors text-muted-foreground hover:text-foreground flex items-center gap-1"
                    title="Back to conversation list"
                  >
                    <ArrowLeft size={14} />
                    <span className="text-xs">Back</span>
                  </button>
                </div>
                <div className="mb-4">
                  <h4 className="font-semibold text-foreground mb-2">{conversationPreview.name}</h4>
                  <div className="text-xs text-muted-foreground mb-4">
                    {conversationPreview.messageCount} messages • {formatDate(conversationPreview.lastModified)}
                  </div>
                  
                  <button
                    onClick={() => loadConversationIntoBranch(selectedConversation)}
                    disabled={loadingConversation}
                    className="w-full bg-primary hover:bg-primary/90 disabled:opacity-50 text-primary-foreground py-2 px-3 rounded text-sm font-medium transition-colors flex items-center justify-center gap-2"
                  >
                    <Download size={14} />
                    {loadingConversation ? 'Loading...' : 'Load into Current Branch'}
                  </button>
                </div>

                <div className="space-y-3">
                  <h5 className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Preview</h5>
                  {conversationPreview.messages.map((message, index) => (
                    <div key={index} className="text-xs">
                      <div className={`font-medium mb-1 ${
                        message.role === 'user' ? 'text-blue-600' : 'text-green-600'
                      }`}>
                        {message.role === 'user' ? 'You' : 'Assistant'}
                      </div>
                      <div className="text-muted-foreground line-clamp-3 whitespace-pre-wrap">
                        {message.content}
                      </div>
                    </div>
                  ))}
                  {conversationPreview.messageCount > conversationPreview.messages.length && (
                    <div className="text-xs text-muted-foreground italic text-center pt-2">
                      ... {conversationPreview.messageCount - conversationPreview.messages.length} more messages
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="p-4 text-center">
                <p className="text-sm text-muted-foreground">Failed to load preview</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}