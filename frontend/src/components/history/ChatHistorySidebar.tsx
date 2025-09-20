/**
 * Chat History Sidebar - Browse and load previous conversations
 */

"use client";

import React from 'react';
import { History, MessageSquare, Clock, Download, Search, Trash2, RefreshCw, ArrowLeft, Users, User, ChevronDown, ChevronRight, GitBranch } from 'lucide-react';
import apiClient from '@/lib/unified-api';
import { useChatStore } from '@/lib/store';

interface ConversationEntry {
  id: string;
  name: string;
  lastModified: string;
  messageCount: number;
  preview: string;
  size?: string;
  sessionId?: string; // Added for cross-session support
  branches?: Array<{
    id: string;
    name?: string;
    messageCount: number;
    lastActive?: string;
    preview?: string;
    parentId?: string;
  }>; // Optional persisted branch summaries
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
  sessionId?: string; // Added for cross-session support
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
    getBranchMessages,
    getBranches,
    setCurrentBranch
  } = useChatStore();
  const [conversations, setConversations] = React.useState<ConversationEntry[]>([]);
  const [selectedConversation, setSelectedConversation] = React.useState<string | null>(null);
  const [conversationPreview, setConversationPreview] = React.useState<ConversationPreview | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [loadingPreview, setLoadingPreview] = React.useState(false);
  const [loadingConversation, setLoadingConversation] = React.useState(false);
  const [searchTerm, setSearchTerm] = React.useState('');
  const [error, setError] = React.useState<string | null>(null);
  const [viewMode, setViewMode] = React.useState<'current' | 'all' | 'nodes'>('current');
  const [nodes, setNodes] = React.useState<any[]>([]);
  const [groupNodes, setGroupNodes] = React.useState<boolean>(true);
  const [expandedConversations, setExpandedConversations] = React.useState<Record<string, boolean>>({});

  // Track active branch message count to refresh Active Branch card after new messages
  const activeBranchVersion = useChatStore((state) => {
    const cid = state.currentConversationId;
    const bid = state.currentBranchId;
    if (!cid || !bid) return 0;
    const msgs = state.conversationMessages[cid]?.[bid] || [];
    return msgs.length;
  });

  const toggleExpanded = (conversationId: string) => {
    setExpandedConversations(prev => ({ ...prev, [conversationId]: !prev[conversationId] }));
  };

  const handleSwitchBranch = async (branchId: string) => {
    try {
      const sessionId = getCurrentSessionId();
      const resp = await apiClient.switchBranch(sessionId, branchId);
      if (!resp.success) {
        console.warn('Failed to switch branch from history:', resp.message);
      }
      setCurrentBranch(branchId);
      if (currentConversationId) {
        await loadStoreConversation(currentConversationId, branchId);
      }
    } catch (e) {
      console.error('Error switching branch from history:', e);
      setCurrentBranch(branchId);
    }
  };

  // Load conversation list on mount and when viewMode changes
  React.useEffect(() => {
    if (viewMode === 'nodes') {
      (async () => {
        setLoading(true);
        try {
          const sessionId = getCurrentSessionId();
          const resp = await apiClient.listSessionNodes(sessionId);
          if (resp.success && resp.data?.nodes) {
            setNodes(resp.data.nodes);
          } else {
            setNodes([]);
          }
        } catch (e) {
          console.error('Failed to load nodes:', e);
          setNodes([]);
        } finally {
          setLoading(false);
        }
      })();
    } else {
      loadConversations();
    }
  }, [viewMode]);

  // Sync with store conversations for real-time updates - merge with existing backend conversations
  React.useEffect(() => {
    if (storeConversations && storeConversations.length > 0) {
      console.log('[HISTORY_MERGE] Store conversations updated:', storeConversations.length, 'conversations');
      
      // With branch-aware storage, conversation messages live in the
      // branch store, not on conv.messages. Use the CURRENT BRANCH to
      // compute counts and preview for accurate sidebar summaries when
      // switching branches.
      const convertedStoreConversations = storeConversations.map((conv) => {
        const branchMessages = getBranchMessages(conv.id, currentBranchId) || [];
        const last = branchMessages.length > 0 ? branchMessages[branchMessages.length - 1] : undefined;
        return {
          id: conv.id,
          name: conv.title || 'Untitled Conversation',
          lastModified: conv.updatedAt || conv.createdAt,
          messageCount: branchMessages.length,
          preview: last ? String(last.content).slice(0, 100) : 'No messages yet'
        } as ConversationEntry;
      });

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
  }, [storeConversations, currentBranchId]);

  const loadConversations = async () => {
    try {
      setLoading(true);
      setError(null);
      
      let response;
      const currentSessionId = getCurrentSessionId();
      
      if (viewMode === 'current') {
        // Load conversations for current session only
        console.log('[HISTORY_MERGE] Loading conversations from backend for current session:', currentSessionId);
        response = await apiClient.listConversations(currentSessionId);
      } else {
        // Load conversations from all sessions
        console.log('[HISTORY_MERGE] Loading conversations from backend for ALL sessions');
        response = await apiClient.listAllConversations();
      }
      
      if (response.success && response.data?.conversations) {
        console.log('[HISTORY_MERGE] Backend returned', response.data.conversations.length, 'conversations');
        
        const conversations = response.data.conversations.map((conv: any) => ({
          id: conv.id || conv.filename,
          name: conv.name || conv.filename || 'Untitled Conversation',
          lastModified: conv.lastModified || conv.modified || new Date().toISOString(),
          messageCount: conv.messageCount || 0,
          preview: conv.preview || 'No preview available',
          size: conv.size || undefined,
          sessionId: conv.sessionId || currentSessionId, // Use provided sessionId or default to current
          branches: Array.isArray(conv.branches) ? conv.branches : undefined,
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

  // Pinned card for active branch in Current Session (memoized)
  const activeBranchCard = React.useMemo(() => {
    if (viewMode !== 'current' || !currentConversationId) return null as React.ReactNode;
    const branches = getBranches(currentConversationId);
    const active = branches.find(b => b.isActive) || branches.find(b => b.id === currentBranchId);
    const msgs = active ? getBranchMessages(currentConversationId, active.id) : [];
    const last = msgs.length > 0 ? msgs[msgs.length - 1] : undefined;
    if (!active) return null as React.ReactNode;
    return (
      <div className="p-3 mb-2 rounded-lg border border-primary/30 bg-primary/5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <GitBranch size={14} className="text-primary" />
            <span className="text-sm font-medium text-foreground">Active Branch</span>
          </div>
          <span className="text-xs text-muted-foreground">{msgs.length} message{msgs.length !== 1 ? 's' : ''}</span>
        </div>
        <div className="mt-1 text-sm font-semibold text-foreground truncate">{active.name}</div>
        {last && (
          <div className="text-xs text-muted-foreground truncate mt-0.5">{String(last.content).slice(0, 120)}</div>
        )}
      </div>
    );
  }, [viewMode, currentConversationId, currentBranchId, getBranches, getBranchMessages, activeBranchVersion]);

  const loadConversationPreview = async (conversationId: string) => {
    try {
      setLoadingPreview(true);
      setError(null);
      
      // Get the conversation entry which might contain the sessionId
      const conversationEntry = conversations.find(c => c.id === conversationId);
      const sourceSessionId = conversationEntry?.sessionId || getCurrentSessionId();
      
      // First try to load from store for faster access (only works for current session)
      const isCurrentSession = sourceSessionId === getCurrentSessionId();
      const storeConversation = isCurrentSession ? storeConversations.find(c => c.id === conversationId) : null;
      
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
          lastModified: storeConversation.updatedAt || storeConversation.createdAt,
          sessionId: sourceSessionId
        });
      } else {
        // Fallback to API, using the source sessionId
        console.log(`[PREVIEW] Loading conversation ${conversationId} from session ${sourceSessionId}`);
        const response = await apiClient.loadConversation(sourceSessionId, conversationId);
        
        if (response.success && response.data) {
          const messages = response.data.messages || [];
          
          setConversationPreview({
            id: conversationId,
            name: conversationEntry?.name || 'Unknown Conversation',
            messages: messages.slice(0, 6).map(msg => ({
              role: msg.role,
              content: msg.content,
              timestamp: typeof msg.timestamp === 'number' ? new Date(msg.timestamp).toISOString() : msg.timestamp
            })), // Show first 6 messages as preview
            messageCount: messages.length,
            lastModified: conversationEntry?.lastModified || new Date().toISOString(),
            sessionId: sourceSessionId
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
      
      // Get source sessionId from conversation preview or entry
      const sourceSessionId = conversationPreview?.sessionId || 
                             conversations.find(c => c.id === conversationId)?.sessionId || 
                             getCurrentSessionId();
      
      const isFromCurrentSession = sourceSessionId === getCurrentSessionId();
      const currentSessionId = getCurrentSessionId();
      
      // For conversations from current session, try to load from store first
      if (isFromCurrentSession) {
        // Load the conversation from store - much faster and real-time
        const conversation = await loadStoreConversation(conversationId);
        
        if (conversation) {
          // Show success message
          alert('✅ Conversation loaded successfully into the current branch!');
          
          // Close the preview
          setSelectedConversation(null);
          setConversationPreview(null);
          return;
        }
      }
      
      // If not found in store or from different session, load from the source session
      console.log(`[LOAD] Loading conversation ${conversationId} from session ${sourceSessionId} into current session ${currentSessionId}`);
      const response = await apiClient.loadConversation(sourceSessionId, conversationId);
      
      if (response.success && response.data?.messages) {
        // Create a new conversation ID if importing from another session
        const targetConversationId = isFromCurrentSession ? conversationId : `imported-${Date.now()}-${conversationId.slice(-8)}`;
        
        // Update store with loaded messages
        setMessages(targetConversationId, currentBranchId || 'main', response.data.messages);
        setCurrentConversationId(targetConversationId);
        
        // If from another session, update the conversation title to indicate its source
        if (!isFromCurrentSession) {
          const conversationName = conversationPreview?.name || 
                                  conversations.find(c => c.id === conversationId)?.name || 
                                  'Imported Conversation';
          const sourceSessionDisplay = sourceSessionId.slice(0, 8); // Shorter ID for display
          
          // Update conversation with new title indicating the source
          const storeConversation = storeConversations.find(c => c.id === targetConversationId);
          if (storeConversation) {
            // Use updateChatTitle if available
            if (typeof useChatStore.getState().updateChatTitle === 'function') {
              useChatStore.getState().updateChatTitle(
                targetConversationId, 
                `${conversationName} (from ${sourceSessionDisplay})`
              );
            }
          }
        }
        
        alert('✅ Conversation loaded successfully into the current branch!');
        setSelectedConversation(null);
        setConversationPreview(null);
      } else {
        throw new Error(response.message || 'Failed to load conversation');
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
      // Get the source session for this conversation
      const conversationEntry = conversations.find(c => c.id === conversationId);
      const sourceSessionId = conversationEntry?.sessionId || getCurrentSessionId();
      const isFromCurrentSession = sourceSessionId === getCurrentSessionId();
      
      // Delete from UI immediately for responsive UX
      setConversations(prev => prev.filter(c => !(c.id === conversationId && c.sessionId === sourceSessionId)));
      
      // Clear preview if it was the deleted conversation
      if (selectedConversation === conversationId) {
        setSelectedConversation(null);
        setConversationPreview(null);
      }
      
      if (isFromCurrentSession) {
        // Delete from store only if it's from the current session
        deleteStoreConversation(conversationId);
        
        // If this was the current conversation, clear it
        if (currentConversationId === conversationId) {
          setCurrentConversationId(null);
          // The setMessages function now requires 3 parameters
          setMessages('', 'main', []);
        }
      }
      
      // Delete from backend/API for persistence
      try {
        await apiClient.deleteConversation(sourceSessionId, conversationId);
      } catch (apiError) {
        console.warn(`Failed to delete from backend for session ${sourceSessionId}, but deleted locally:`, apiError);
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
        <div className="flex items-center gap-2">
          <button
            onClick={loadConversations}
            disabled={loading}
            className="p-1 hover:bg-muted rounded transition-colors text-muted-foreground hover:text-foreground disabled:opacity-50"
            title="Refresh conversations"
          >
            <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>
      
      {/* Toggle between Current Session and All Chats */}
      <div className="flex border-b">
        <button 
          className={`flex-1 py-2 text-sm font-medium flex justify-center items-center gap-1 ${
            viewMode === 'current' 
              ? 'text-primary border-b-2 border-primary' 
              : 'text-muted-foreground hover:text-foreground hover:bg-muted'
          }`}
          onClick={() => setViewMode('current')}
        >
          <User size={14} />
          <span>Current Session</span>
        </button>
        <button 
          className={`flex-1 py-2 text-sm font-medium flex justify-center items-center gap-1 ${
            viewMode === 'all' 
              ? 'text-primary border-b-2 border-primary' 
              : 'text-muted-foreground hover:text-foreground hover:bg-muted'
          }`}
          onClick={() => setViewMode('all')}
        >
          <Users size={14} />
          <span>All Chats</span>
        </button>
        <button 
          className={`flex-1 py-2 text-sm font-medium flex justify-center items-center gap-1 ${
            viewMode === 'nodes' 
              ? 'text-primary border-b-2 border-primary' 
              : 'text-muted-foreground hover:text-foreground hover:bg-muted'
          }`}
          onClick={() => setViewMode('nodes')}
        >
          <GitBranch size={14} />
          <span>Branches</span>
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
        {viewMode === 'nodes' && (
          <div className="mt-2 text-xs text-muted-foreground flex items-center justify-end gap-2">
            <span>Group by conversation</span>
            <input type="checkbox" checked={groupNodes} onChange={(e) => setGroupNodes(e.target.checked)} />
          </div>
        )}
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
          ) : viewMode === 'nodes' ? (
            <div className="space-y-2 p-2">
              {/* Node-centric list */}
              {(() => {
                const sessionId = getCurrentSessionId();
                // Filter by search
                const filtered = nodes.filter(n => {
                  const text = `${n.name || ''} ${n.preview || ''}`.toLowerCase();
                  const term = searchTerm.toLowerCase();
                  return !term || text.includes(term);
                });
                if (groupNodes) {
                  const groups: Record<string, any[]> = {};
                  const convNames: Record<string, string> = {};
                  for (const n of filtered) {
                    groups[n.conversationId] = groups[n.conversationId] || [];
                    groups[n.conversationId].push(n);
                    if (n.isRoot) convNames[n.conversationId] = n.name;
                  }
                  const convIds = Object.keys(groups);
                  return convIds.length === 0 ? (
                    <div className="p-4 text-center text-muted-foreground text-sm">No branches found</div>
                  ) : convIds.map(cid => (
                    <div key={cid} className="border rounded-lg">
                      <div className="px-3 py-2 text-xs font-semibold text-muted-foreground border-b">{convNames[cid] || cid}</div>
                      <div className="p-2 space-y-1">
                        {groups[cid].filter(n => !n.isRoot).map(n => (
                          <button
                            key={`${n.conversationId}:${n.branchId}`}
                            onClick={() => handleSwitchBranch(n.branchId)}
                            className={`w-full text-left p-2 rounded hover:bg-muted transition-colors ${n.branchId === currentBranchId && cid === currentConversationId ? 'bg-primary/10 border border-primary/20' : 'border border-transparent'}`}
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <GitBranch size={12} className={n.branchId === currentBranchId && cid === currentConversationId ? 'text-primary' : 'text-muted-foreground'} />
                                <span className="text-sm text-foreground truncate">{n.name || n.branchId}</span>
                              </div>
                              <span className="text-xs text-muted-foreground">{n.messageCount} msg</span>
                            </div>
                            {n.preview && <div className="text-xs text-muted-foreground truncate mt-0.5">{String(n.preview).slice(0, 100)}</div>}
                          </button>
                        ))}
                      </div>
                    </div>
                  ));
                }
                // Flattened
                return filtered.map(n => (
                  <button
                    key={`${n.conversationId}:${n.branchId}`}
                    onClick={() => handleSwitchBranch(n.branchId)}
                    className={`w-full text-left p-2 rounded hover:bg-muted transition-colors ${n.branchId === currentBranchId && n.conversationId === currentConversationId ? 'bg-primary/10 border border-primary/20' : 'border border-transparent'}`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <GitBranch size={12} className={n.branchId === currentBranchId && n.conversationId === currentConversationId ? 'text-primary' : 'text-muted-foreground'} />
                        <span className="text-sm text-foreground truncate">{n.name || n.branchId}</span>
                      </div>
                      <span className="text-xs text-muted-foreground">{n.messageCount} msg</span>
                    </div>
                    {n.preview && <div className="text-xs text-muted-foreground truncate mt-0.5">{String(n.preview).slice(0, 100)}</div>}
                  </button>
                ));
              })()}
            </div>
          ) : (
            <div className="space-y-1 p-2">
              {viewMode === 'current' && activeBranchCard}
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
                  {/* Expand/collapse branches */}
                  <button
                    onClick={(e) => { e.stopPropagation(); toggleExpanded(conversation.id); }}
                    className="absolute left-2 top-2 p-1 text-muted-foreground hover:text-foreground"
                    title={expandedConversations[conversation.id] ? 'Collapse' : 'Expand'}
                  >
                    {expandedConversations[conversation.id] ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                  </button>
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

                  <div className="pl-6 pr-6">
                    <div className="flex items-center gap-2 mb-1">
                      <h4 className="font-medium text-sm text-foreground truncate">
                        {conversation.name}
                      </h4>
                      {currentConversationId === conversation.id && (
                        <div className="flex-shrink-0 w-2 h-2 bg-green-500 rounded-full" title="Active conversation" />
                      )}
                      {/* Show session indicator badge for non-current sessions in "All Chats" view */}
                      {viewMode === 'all' && conversation.sessionId && conversation.sessionId !== getCurrentSessionId() && (
                        <div 
                          className="text-xs px-1.5 py-0.5 bg-amber-100 dark:bg-amber-900/30 text-amber-800 dark:text-amber-300 rounded"
                          title={`From session: ${conversation.sessionId}`}
                        >
                          {conversation.sessionId.substring(0, 6)}
                        </div>
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

                    {/* Branch list (expanded) */}
                    {expandedConversations[conversation.id] && (
                      <div className="mt-2 border-t pt-2 space-y-1">
                        {(() => {
                          const storeBranches = getBranches(conversation.id) || [];
                          // Heuristic: if store has only a default main with 0 messages and no others,
                          // prefer persisted summaries from backend.
                          const usePersisted = (
                            (!storeBranches || storeBranches.length === 0) ||
                            (storeBranches.length === 1 && storeBranches[0].id === 'main' && storeBranches[0].messageCount === 0 && (conversation.branches?.length || 0) > 0)
                          );
                          const renderBranches: Array<any> = usePersisted
                            ? (conversation.branches || []).map(br => ({ id: br.id, name: br.name || (br.id === 'main' ? 'Main' : br.id), messageCount: br.messageCount }))
                            : storeBranches;
                          return renderBranches.map((branch: any) => {
                            const bMsgs = getBranchMessages(conversation.id, branch.id) || [];
                            const bLast = bMsgs.length > 0 ? bMsgs[bMsgs.length - 1] : undefined;
                            const isActiveBranch = branch.id === currentBranchId && conversation.id === currentConversationId;
                            return (
                              <button
                                key={branch.id}
                                onClick={(e) => { e.stopPropagation(); handleSwitchBranch(branch.id); }}
                                className={`w-full text-left p-2 rounded hover:bg-muted transition-colors ${isActiveBranch ? 'bg-primary/10 border border-primary/20' : 'border border-transparent'}`}
                              >
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center gap-2">
                                    <GitBranch size={12} className={isActiveBranch ? 'text-primary' : 'text-muted-foreground'} />
                                    <span className="text-sm text-foreground truncate">{branch.name || (branch.id === 'main' ? 'Main' : branch.id)}</span>
                                  </div>
                                  <span className="text-xs text-muted-foreground">{(bMsgs.length || branch.messageCount || 0)} msg</span>
                                </div>
                                {bLast && (
                                  <div className="text-xs text-muted-foreground truncate mt-0.5">{String(bLast.content).slice(0, 100)}</div>
                                )}
                              </button>
                            );
                          });
                        })()}
                      </div>
                    )}
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
