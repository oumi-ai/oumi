/**
 * Search & History Window - Comprehensive search and conversation history browser
 */

"use client";

import React from 'react';
import { X, Search, History, MessageCircle, Calendar, Clock, Filter, ChevronDown, ChevronRight, FileText, User, Bot, Regex, Type, Settings2 } from 'lucide-react';
import { useChatStore } from '@/lib/store';
import { Message, Conversation } from '@/lib/types';

interface SearchResult {
  conversationId: string;
  conversationTitle: string;
  messageId: string;
  messageIndex: number;
  branchId?: string;
  content: string;
  role: 'user' | 'assistant' | 'system';
  timestamp: number;
  matchStart: number;
  matchEnd: number;
  context: string; // Surrounding text for context
}

interface SearchHistoryWindowProps {
  isOpen: boolean;
  onClose: () => void;
  onNavigateToMessage?: (conversationId: string, messageId: string, branchId?: string) => void;
}

export default function SearchHistoryWindow({ isOpen, onClose, onNavigateToMessage }: SearchHistoryWindowProps) {
  const { conversations, currentConversationId, loadConversation, getCurrentMessages } = useChatStore();
  
  // Get current messages using the selector
  const currentMessages = getCurrentMessages();
  
  // Search state
  const [searchQuery, setSearchQuery] = React.useState('');
  const [searchType, setSearchType] = React.useState<'text' | 'regex'>('text');
  const [isSearching, setIsSearching] = React.useState(false);
  const [searchResults, setSearchResults] = React.useState<SearchResult[]>([]);
  const [caseSensitive, setCaseSensitive] = React.useState(false);
  
  // History state
  const [selectedConversation, setSelectedConversation] = React.useState<string | null>(null);
  const [expandedConversations, setExpandedConversations] = React.useState<Set<string>>(new Set());
  const [sortBy, setSortBy] = React.useState<'date' | 'title'>('date');
  const [filterRole, setFilterRole] = React.useState<'all' | 'user' | 'assistant' | 'system'>('all');
  
  // Tab state
  const [activeTab, setActiveTab] = React.useState<'search' | 'history'>('search');

  /**
   * Perform search across all conversations and branches
   */
  const performSearch = React.useCallback(async () => {
    if (!searchQuery.trim()) {
      setSearchResults([]);
      return;
    }

    setIsSearching(true);
    const results: SearchResult[] = [];

    try {
      // Create search pattern
      let searchPattern: RegExp;
      if (searchType === 'regex') {
        try {
          const flags = caseSensitive ? 'g' : 'gi';
          searchPattern = new RegExp(searchQuery, flags);
        } catch (error) {
          console.error('Invalid regex pattern:', error);
          setIsSearching(false);
          return;
        }
      } else {
        const escapedQuery = searchQuery.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const flags = caseSensitive ? 'g' : 'gi';
        searchPattern = new RegExp(escapedQuery, flags);
      }

      // Search through all conversations
      const { conversationMessages, getBranchMessages } = useChatStore.getState();
      
      for (const conversation of conversations) {
        // Search all branches for this conversation
        const convBranches = Object.keys(conversationMessages[conversation.id] || {});
        
        for (const branchId of convBranches) {
          const branchMessages = getBranchMessages(conversation.id, branchId);
          await searchInMessages(
            branchMessages,
            conversation.id,
            conversation.title,
            searchPattern,
            results,
            branchId
          );
        }
      }

      // Also search current messages if they're not part of a saved conversation
      if (currentMessages && currentMessages.length > 0) {
        const currentConversationTitle = currentConversationId 
          ? conversations.find(c => c.id === currentConversationId)?.title || 'Current Conversation'
          : 'Current Conversation';
        
        // Only search current messages if they're not already part of a saved conversation
        const isCurrentConversationSaved = currentConversationId && 
          conversations.find(c => c.id === currentConversationId);
        
        if (!isCurrentConversationSaved) {
          await searchInMessages(
            currentMessages,
            currentConversationId || 'current',
            currentConversationTitle,
            searchPattern,
            results
          );
        }
      }

      // Sort results by timestamp (newest first)
      results.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
      setSearchResults(results);
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setIsSearching(false);
    }
  }, [searchQuery, searchType, caseSensitive, conversations]);

  /**
   * Search within a set of messages
   */
  const searchInMessages = async (
    messages: Message[],
    conversationId: string,
    conversationTitle: string,
    searchPattern: RegExp,
    results: SearchResult[],
    branchId?: string
  ) => {
    for (let i = 0; i < messages.length; i++) {
      const message = messages[i];
      if (filterRole !== 'all' && message.role !== filterRole) continue;

      const content = message.content;
      let match;
      searchPattern.lastIndex = 0; // Reset regex state

      while ((match = searchPattern.exec(content)) !== null) {
        // Create context around the match (50 chars before/after)
        const contextStart = Math.max(0, match.index - 50);
        const contextEnd = Math.min(content.length, match.index + match[0].length + 50);
        const context = content.slice(contextStart, contextEnd);

        results.push({
          conversationId,
          conversationTitle,
          messageId: message.id,
          messageIndex: i,
          branchId,
          content: message.content,
          role: message.role,
          timestamp: message.timestamp,
          matchStart: match.index,
          matchEnd: match.index + match[0].length,
          context: contextStart > 0 ? '...' + context : context + (contextEnd < content.length ? '...' : '')
        });

        // Prevent infinite loop with global regex
        if (!searchPattern.global) break;
      }
    }
  };

  /**
   * Handle navigation to a specific message
   */
  const handleNavigateToResult = (result: SearchResult) => {
    if (onNavigateToMessage) {
      onNavigateToMessage(result.conversationId, result.messageId, result.branchId);
      onClose();
    }
  };

  /**
   * Load and display conversation messages
   */
  const handleConversationSelect = async (conversationId: string) => {
    if (selectedConversation === conversationId) {
      setSelectedConversation(null);
    } else {
      setSelectedConversation(conversationId);
      // Ensure conversation is loaded
      await loadConversation(conversationId);
    }
  };

  /**
   * Toggle conversation expansion
   */
  const toggleConversationExpansion = (conversationId: string) => {
    const newExpanded = new Set(expandedConversations);
    if (newExpanded.has(conversationId)) {
      newExpanded.delete(conversationId);
    } else {
      newExpanded.add(conversationId);
    }
    setExpandedConversations(newExpanded);
  };

  /**
   * Format timestamp for display
   */
  const formatTimestamp = (timestamp: number | string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) {
      return date.toLocaleTimeString();
    } else if (diffDays === 1) {
      return 'Yesterday ' + date.toLocaleTimeString();
    } else if (diffDays < 7) {
      return diffDays + ' days ago';
    } else {
      return date.toLocaleDateString();
    }
  };

  /**
   * Highlight search matches in text
   */
  const highlightMatches = (text: string, query: string, isRegex: boolean, caseSensitive: boolean) => {
    if (!query) return text;

    try {
      let pattern: RegExp;
      if (isRegex) {
        const flags = caseSensitive ? 'g' : 'gi';
        pattern = new RegExp(query, flags);
      } else {
        const escapedQuery = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const flags = caseSensitive ? 'g' : 'gi';
        pattern = new RegExp(escapedQuery, flags);
      }

      return text.replace(pattern, '<mark class="bg-yellow-200 dark:bg-yellow-800 px-1 rounded">$&</mark>');
    } catch (error) {
      return text;
    }
  };

  // Trigger search when query changes
  React.useEffect(() => {
    const timeoutId = setTimeout(() => {
      if (searchQuery.trim()) {
        performSearch();
      } else {
        setSearchResults([]);
      }
    }, 300); // Debounce search

    return () => clearTimeout(timeoutId);
  }, [performSearch, searchQuery]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-background rounded-lg shadow-xl w-full max-w-4xl h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border">
          <div className="flex items-center space-x-4">
            <div className="flex bg-muted rounded-lg p-1">
              <button
                onClick={() => setActiveTab('search')}
                className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm transition-colors ${
                  activeTab === 'search' 
                    ? 'bg-background text-foreground shadow-sm' 
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                <Search className="w-4 h-4" />
                <span>Search</span>
              </button>
              <button
                onClick={() => setActiveTab('history')}
                className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm transition-colors ${
                  activeTab === 'history' 
                    ? 'bg-background text-foreground shadow-sm' 
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                <History className="w-4 h-4" />
                <span>History</span>
              </button>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden">
          {activeTab === 'search' ? (
            <div className="h-full flex flex-col">
              {/* Search Controls */}
              <div className="p-4 border-b border-border space-y-4">
                <div className="flex items-center space-x-4">
                  <div className="flex-1 relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
                    <input
                      type="text"
                      placeholder="Search conversations..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="w-full pl-10 pr-4 py-2 border border-border rounded-lg bg-background text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                    />
                  </div>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => setSearchType(searchType === 'text' ? 'regex' : 'text')}
                      className={`p-2 rounded-lg border transition-colors ${
                        searchType === 'regex'
                          ? 'bg-primary text-primary-foreground'
                          : 'border-border hover:bg-muted'
                      }`}
                      title={searchType === 'regex' ? 'Switch to text search' : 'Switch to regex search'}
                    >
                      {searchType === 'regex' ? <Regex className="w-4 h-4" /> : <Type className="w-4 h-4" />}
                    </button>
                    <button
                      onClick={() => setCaseSensitive(!caseSensitive)}
                      className={`px-3 py-2 rounded-lg border text-sm font-mono transition-colors ${
                        caseSensitive
                          ? 'bg-primary text-primary-foreground'
                          : 'border-border hover:bg-muted'
                      }`}
                      title={caseSensitive ? 'Case sensitive' : 'Case insensitive'}
                    >
                      Aa
                    </button>
                  </div>
                </div>
                
                {/* Filters */}
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-2">
                    <Filter className="w-4 h-4 text-muted-foreground" />
                    <select
                      value={filterRole}
                      onChange={(e) => setFilterRole(e.target.value as 'all' | 'user' | 'assistant' | 'system')}
                      className="px-2 py-1 text-sm border border-border rounded bg-background text-foreground"
                    >
                      <option value="all">All messages</option>
                      <option value="user">User messages</option>
                      <option value="assistant">Assistant messages</option>
                      <option value="system">System messages</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Search Results */}
              <div className="flex-1 overflow-y-auto p-4">
                {isSearching ? (
                  <div className="flex items-center justify-center py-12">
                    <div className="flex items-center space-x-2 text-muted-foreground">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
                      <span>Searching...</span>
                    </div>
                  </div>
                ) : searchResults.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                    <Search className="w-12 h-12 mb-4 opacity-50" />
                    <p className="text-lg font-medium">
                      {searchQuery ? 'No results found' : 'Enter a search query'}
                    </p>
                    <p className="text-sm">
                      {searchQuery ? 'Try different keywords or check your regex pattern' : 'Search across all conversations and branches'}
                    </p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <p className="text-sm text-muted-foreground">
                      Found {searchResults.length} result{searchResults.length !== 1 ? 's' : ''}
                    </p>
                    {searchResults.map((result, index) => (
                      <div
                        key={index}
                        onClick={() => handleNavigateToResult(result)}
                        className="p-4 border border-border rounded-lg hover:bg-muted cursor-pointer transition-colors"
                      >
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex items-center space-x-2">
                            {result.role === 'user' ? (
                              <User className="w-4 h-4 text-blue-500" />
                            ) : result.role === 'assistant' ? (
                              <Bot className="w-4 h-4 text-green-500" />
                            ) : (
                              <Settings2 className="w-4 h-4 text-gray-500" />
                            )}
                            <span className="text-sm font-medium">
                              {result.conversationTitle}
                              {result.branchId && (
                                <span className="ml-2 text-xs text-muted-foreground">
                                  (Branch: {result.branchId})
                                </span>
                              )}
                            </span>
                          </div>
                          <span className="text-xs text-muted-foreground">
                            {formatTimestamp(result.timestamp)}
                          </span>
                        </div>
                        <div 
                          className="text-sm text-muted-foreground"
                          dangerouslySetInnerHTML={{
                            __html: highlightMatches(result.context, searchQuery, searchType === 'regex', caseSensitive)
                          }}
                        />
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="h-full flex flex-col">
              {/* History Controls */}
              <div className="p-4 border-b border-border">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-2">
                      <span className="text-sm text-muted-foreground">Sort by:</span>
                      <select
                        value={sortBy}
                        onChange={(e) => setSortBy(e.target.value as 'date' | 'title')}
                        className="px-2 py-1 text-sm border border-border rounded bg-background text-foreground"
                      >
                        <option value="date">Date</option>
                        <option value="title">Title</option>
                      </select>
                    </div>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {conversations.length} conversation{conversations.length !== 1 ? 's' : ''}
                  </div>
                </div>
              </div>

              {/* Conversation List */}
              <div className="flex-1 overflow-y-auto">
                {conversations.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                    <MessageCircle className="w-12 h-12 mb-4 opacity-50" />
                    <p className="text-lg font-medium">No conversations yet</p>
                    <p className="text-sm">Start chatting to see your history here</p>
                  </div>
                ) : (
                  <div className="divide-y divide-border">
                    {[...conversations]
                      .sort((a, b) => {
                        if (sortBy === 'title') {
                          return a.title.localeCompare(b.title);
                        } else {
                          return new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime();
                        }
                      })
                      .map((conversation) => (
                        <div key={conversation.id}>
                          <div
                            onClick={() => handleConversationSelect(conversation.id)}
                            className={`p-4 hover:bg-muted cursor-pointer transition-colors ${
                              selectedConversation === conversation.id ? 'bg-muted' : ''
                            } ${
                              currentConversationId === conversation.id ? 'border-l-4 border-l-primary' : ''
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex items-center space-x-3">
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    toggleConversationExpansion(conversation.id);
                                  }}
                                  className="text-muted-foreground hover:text-foreground"
                                >
                                  {expandedConversations.has(conversation.id) ? (
                                    <ChevronDown className="w-4 h-4" />
                                  ) : (
                                    <ChevronRight className="w-4 h-4" />
                                  )}
                                </button>
                                <MessageCircle className="w-4 h-4 text-muted-foreground" />
                                <div>
                                  <h3 className="font-medium text-foreground">{conversation.title}</h3>
                                  <div className="flex items-center space-x-2 text-xs text-muted-foreground">
                                    <Clock className="w-3 h-3" />
                                    <span>{formatTimestamp(conversation.updatedAt)}</span>
                                    <span>•</span>
                                    <span>{useChatStore.getState().getBranchMessages(conversation.id, 'main').length} messages</span>
                                    {Object.keys(conversation.branches || {}).length > 0 && (
                                      <>
                                        <span>•</span>
                                        <span>{Object.keys(conversation.branches || {}).length} branches</span>
                                      </>
                                    )}
                                  </div>
                                </div>
                              </div>
                              {currentConversationId === conversation.id && (
                                <div className="text-xs bg-primary text-primary-foreground px-2 py-1 rounded-full">
                                  Current
                                </div>
                              )}
                            </div>
                          </div>
                          
                          {/* Expanded conversation details */}
                          {expandedConversations.has(conversation.id) && (
                            <div className="bg-muted/50 p-4 space-y-2">
                              {useChatStore.getState().getBranchMessages(conversation.id, 'main').slice(0, 3).map((message, index) => (
                                <div key={message.id} className="flex items-start space-x-2 text-sm">
                                  {message.role === 'user' ? (
                                    <User className="w-3 h-3 text-blue-500 mt-0.5 flex-shrink-0" />
                                  ) : (
                                    <Bot className="w-3 h-3 text-green-500 mt-0.5 flex-shrink-0" />
                                  )}
                                  <p className="text-muted-foreground truncate">
                                    {message.content.length > 100 
                                      ? message.content.substring(0, 100) + '...' 
                                      : message.content
                                    }
                                  </p>
                                </div>
                              ))}
                              {useChatStore.getState().getBranchMessages(conversation.id, 'main').length > 3 && (
                                <p className="text-xs text-muted-foreground">
                                  +{useChatStore.getState().getBranchMessages(conversation.id, 'main').length - 3} more messages
                                </p>
                              )}
                            </div>
                          )}
                        </div>
                      ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}