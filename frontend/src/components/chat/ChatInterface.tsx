/**
 * Main chat interface component
 */

"use client";

import React from 'react';
import { useChatStore } from '@/lib/store';
import { useAutoSave } from '@/hooks/useAutoSave';
import { Message } from '@/lib/types';
import apiClient from '@/lib/unified-api';
import { isValidCommand, parseCommand } from '@/lib/constants';
import ChatHistory from './ChatHistory';
import MessageInput from './MessageInput';

interface ChatInterfaceProps {
  className?: string;
  onRef?: (ref: ChatInterfaceRef) => void;
}

export interface ChatInterfaceRef {
  regenerateLastResponse: () => void;
  stopGeneration: () => void;
  sendMessage: (message: string) => void;
}

export default function ChatInterface({ className = '', onRef }: ChatInterfaceProps) {
  const {
    messages,
    isLoading,
    isTyping,
    currentBranchId,
    addMessage,
    setLoading,
    setTyping,
    setMessages,
    setBranches,
    generationParams,
  } = useChatStore();
  
  // Initialize auto-save functionality
  useAutoSave();

  // State for stopping generation
  const [shouldStop, setShouldStop] = React.useState(false);

  // Expose methods to parent via onRef callback
  React.useEffect(() => {
    if (onRef) {
      onRef({
        regenerateLastResponse: handleRegenerateLastResponse,
        stopGeneration: handleStopGeneration,
        sendMessage: handleSendMessage,
      });
    }
  }, [onRef]);

  // Load conversation history on mount and branch changes
  React.useEffect(() => {
    loadConversation();
  }, [currentBranchId]);

  const refreshBranches = async () => {
    try {
      const response = await apiClient.getBranches('default');
      if (response.success && response.data) {
        const { branches } = response.data;
        
        // Transform backend branches to frontend format
        const transformedBranches = branches.map((branch: any) => ({
          id: branch.id,
          name: branch.name,
          isActive: branch.id === currentBranchId,
          messageCount: branch.message_count || 0,
          createdAt: branch.created_at,
          lastActive: branch.last_active || branch.created_at,
          preview: branch.message_count > 0 ? `${branch.message_count} messages` : 'Empty branch'
        }));
        
        setBranches(transformedBranches);
      }
    } catch (error) {
      console.error('Failed to refresh branches:', error);
    }
  };

  const loadConversation = async () => {
    try {
      setLoading(true);
      const response = await apiClient.getConversation('default', currentBranchId);
      
      if (response.success && response.data) {
        // Transform backend messages to frontend format
        const transformedMessages: Message[] = response.data.conversation.map((msg: any) => ({
          id: msg.id || `${msg.role}-${Date.now()}-${Math.random()}`,
          role: msg.role,
          content: msg.content,
          timestamp: new Date(msg.timestamp || Date.now()).getTime(),
          attachments: msg.attachments,
        }));
        
        setMessages(transformedMessages);
      }
    } catch (error) {
      console.error('Failed to load conversation:', error);
      // Don't show error for empty conversations
      if (error instanceof Error && !error.message.includes('not found')) {
        const errorMessage: Message = {
          id: `error-${Date.now()}`,
          role: 'assistant',
          content: `❌ Failed to load conversation: ${error.message}`,
          timestamp: Date.now(),
        };
        setMessages([errorMessage]);
      }
    } finally {
      setLoading(false);
    }
  };

  // Handler for regenerating the last response
  const handleRegenerateLastResponse = async () => {
    const lastUserMessage = messages.slice().reverse().find(msg => msg.role === 'user');
    if (lastUserMessage && !isLoading && !isTyping) {
      // Remove the last assistant message if it exists
      const lastMessageIndex = messages.length - 1;
      if (lastMessageIndex >= 0 && messages[lastMessageIndex].role === 'assistant') {
        const updatedMessages = messages.slice(0, -1);
        setMessages(updatedMessages);
      }
      
      // Regenerate response for the last user message
      await handleChatMessage(lastUserMessage.content);
    }
  };

  // Handler for stopping generation
  const handleStopGeneration = () => {
    setShouldStop(true);
    setLoading(false);
    setTyping(false);
  };

  const handleSendMessage = async (content: string, attachments?: any[]) => {
    // Check if it's a valid command and block it
    if (isValidCommand(content)) {
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: `❌ Commands cannot be executed through the chat input. Please use the UI controls and buttons instead.`,
        timestamp: Date.now(),
      };
      addMessage(errorMessage);
      return;
    }

    // Create user message
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content,
      timestamp: Date.now(),
      attachments,
    };

    // Add user message to store
    addMessage(userMessage);

    // Handle regular chat message (no command handling anymore)
    await handleChatMessage(content);
  };

  // Internal method for UI elements to execute commands (bypasses user input blocking)
  const executeCommand = async (command: string) => {
    setLoading(true);
    
    try {
      // Parse command using shared utility
      const parsed = parseCommand(command);
      if (!parsed) {
        throw new Error('Invalid command format');
      }

      const { name: commandName, args } = parsed;

      // Execute command via API
      const response = await apiClient.executeCommand(commandName, args);

      if (response.success && response.data) {
        // Add command result as system message
        const resultMessage: Message = {
          id: `system-${Date.now()}`,
          role: 'assistant',
          content: response.data.message || 'Command executed successfully',
          timestamp: Date.now(),
        };
        addMessage(resultMessage);
      } else {
        throw new Error(response.message || 'Command failed');
      }
    } catch (error) {
      // Add error message
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: `❌ Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: Date.now(),
      };
      addMessage(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleChatMessage = async (content: string) => {
    setTyping(true);
    
    try {
      // Prepare messages for API
      const apiMessages = messages
        .filter(msg => msg.role !== 'system') // Exclude system messages from API
        .map(msg => ({
          role: msg.role,
          content: msg.content,
        }));

      // Add the current user message
      apiMessages.push({ role: 'user', content });

      // Send to chat API
      const response = await apiClient.chatCompletion({
        messages: apiMessages,
        session_id: 'default',
        branch_id: currentBranchId,
        temperature: generationParams.temperature,
        max_tokens: generationParams.maxTokens,
        top_p: generationParams.topP,
      });

      if (response.success && response.data) {
        // Add assistant response
        const assistantMessage: Message = {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: response.data.choices?.[0]?.message?.content || 'No response generated',
          timestamp: Date.now(),
        };
        addMessage(assistantMessage);
        
        // Refresh branch data after successful chat exchange
        await refreshBranches();
      } else {
        throw new Error(response.message || 'Failed to get response');
      }
    } catch (error) {
      console.error('Chat message error:', error);
      
      // Add error message
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: error instanceof Error && error.message.includes('Backend may still be loading')
          ? '🔄 Backend is starting up. Please wait for the model to load and try again.'
          : `❌ Error: ${error instanceof Error ? error.message : 'Failed to send message'}`,
        timestamp: Date.now(),
      };
      addMessage(errorMessage);
    } finally {
      setTyping(false);
    }
  };

  const handleAttachFiles = async (files: FileList) => {
    // PLACEHOLDER: File attachment not fully implemented
    console.log('PLACEHOLDER: Files to attach:', Array.from(files).map(f => f.name));
    
    // For now, just show a placeholder message
    const attachmentMessage: Message = {
      id: `attachment-${Date.now()}`,
      role: 'system',
      content: `📎 PLACEHOLDER: File attachment feature coming soon. Selected files: ${Array.from(files).map(f => f.name).join(', ')}`,
      timestamp: Date.now(),
    };
    addMessage(attachmentMessage);
  };

  return (
    <div className={`flex flex-col h-full bg-background ${className}`}>
      {/* Chat history */}
      <div className="flex-1 relative">
        <ChatHistory
          messages={messages}
          isTyping={isTyping}
          isLoading={isLoading}
        />
      </div>

      {/* Message input */}
      <MessageInput
        onSendMessage={handleSendMessage}
        onAttachFiles={handleAttachFiles}
        disabled={isLoading}
        isLoading={isLoading || isTyping}
      />
    </div>
  );
}