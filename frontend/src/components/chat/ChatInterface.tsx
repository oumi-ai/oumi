/**
 * Main chat interface component
 */

"use client";

import React from 'react';
import { generateDisplayName } from '@/lib/nameGen';
import { useChatStore } from '@/lib/store';
import { useAutoSave } from '@/hooks/useAutoSave';
import { Message, ChatCompletionRequest } from '@/lib/types';
import apiClient from '@/lib/unified-api';
import { isValidCommand, parseCommand } from '@/lib/constants';
import ChatHistory from './ChatHistory';
import MessageInput, { PreparedAttachment } from './MessageInput';

const deriveOmniCapability = (
  metadata: any,
  modelId?: string | null
): boolean | undefined => {
  if (metadata && typeof metadata.is_omni_capable === 'boolean') {
    return metadata.is_omni_capable;
  }
  const source = metadata?.model_name ?? modelId ?? '';
  const lower = String(source).toLowerCase();
  if (!lower) return undefined;
  const isOmni = lower.includes('omni') && lower.includes('qwen');
  return isOmni;
};

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
    isLoading,
    isTyping,
    currentBranchId,
    currentConversationId,
    settings,
    addMessage,
    setLoading,
    setTyping,
    setMessages,
    generationParams,
    updateMessage,
    getCurrentSessionId,
    getCurrentMessages,
  } = useChatStore();
  
  // Note: setBranches is no longer needed as branches are derived on demand from state
  
  // Get messages using the getCurrentMessages selector
  const messages = getCurrentMessages();
  
  // Initialize auto-save functionality
  useAutoSave();

  // State for stopping generation
  const [shouldStop, setShouldStop] = React.useState(false);
  
  // Ref to track current streaming message ID
  const currentStreamingMessageId = React.useRef<string | null>(null);

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

  // Only load conversation history when switching between existing branches/conversations
  // For fresh sessions, we start with empty messages (as configured in store.ts)
  React.useEffect(() => {
    // Only load conversation if we have an active conversation ID AND
    // the current messages array is empty (meaning we're switching TO a conversation)
    // This prevents loading when we're actively adding messages to the current conversation
    if (currentConversationId && messages.length === 0) {
      loadConversation();
    }
  }, [currentBranchId, currentConversationId]);

  const refreshBranches = async () => {
    try {
      const response = await apiClient.getBranches(getCurrentSessionId());
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
        
        // Note: setBranches is no longer needed since branches are derived on demand
        console.log('Branches updated successfully (will be available via getBranches)');
      }
    } catch (error) {
      console.error('Failed to refresh branches:', error);
    }
  };

  const loadConversation = async () => {
    try {
      setLoading(true);
      const response = await apiClient.getConversation(getCurrentSessionId(), currentBranchId);
      
      if (response.success && response.data) {
        const normalizeTs = (t: any): number => {
          if (typeof t === 'number') {
            // If seconds (10 digits), convert to ms
            return t < 1e11 ? Math.round(t * 1000) : Math.round(t);
          }
          if (typeof t === 'string') {
            const f = parseFloat(t);
            if (!isNaN(f)) return f < 1e11 ? Math.round(f * 1000) : Math.round(f);
          }
          return Date.now();
        };
        // Transform backend messages to frontend format with minimal metadata
        const convLen = Array.isArray(response.data?.conversation) ? (response.data!.conversation as any[]).length : 0;
        const transformedMessages: Message[] = (response.data?.conversation as any[]).map((msg: any, i: number) => {
          const ts = normalizeTs(msg.timestamp);
          const md = (msg.metadata || msg.meta || {}) as any;
          const modelName = md.model_name ?? md.modelName ?? (msg.role === 'assistant' ? settings.selectedModel : undefined);
          const engine = md.engine ?? (msg.role === 'assistant' ? settings.selectedProvider : undefined);
          const durationMs = md.duration_ms ?? md.durationMs;
          if (convLen > 0 && i === convLen - 1) {
            console.log('[CHAT_LOAD] last msg meta', md, 'mapped', { modelName, engine, durationMs });
          }
          return {
            id: msg.id || `${msg.role}-${Date.now()}-${Math.random()}`,
            role: msg.role,
            content: msg.content,
            timestamp: ts,
            attachments: msg.attachments,
            meta: {
              authorType: msg.role === 'assistant' ? 'ai' : (msg.role === 'user' ? 'user' : 'system'),
              authorName: msg.role === 'assistant' ? (modelName || 'AI') : (settings.user?.displayName || 'You'),
              modelName,
              engine,
              createdAt: ts,
              ...(typeof durationMs === 'number' ? { durationMs } : {}),
            }
          };
        });
        
        // Use the branch-specific setMessages
        // The setMessages function now requires 3 parameters
        setMessages(currentConversationId || '', currentBranchId, transformedMessages);
        console.log('[CHAT_LOAD] setMessages with', transformedMessages.length, 'messages for', currentConversationId, currentBranchId);
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
        // Use the branch-specific setMessages
        // The setMessages function now requires 3 parameters
        setMessages(currentConversationId || '', currentBranchId, [errorMessage]);
      }
    } finally {
      setLoading(false);
    }
  };

  // Handler for regenerating the last response (id-first, backend regen_node)
  const handleRegenerateLastResponse = async () => {
    if (isLoading || isTyping) return;
    const lastAssistant = [...messages].reverse().find(m => m.role === 'assistant');
    const lastUser = [...messages].reverse().find(m => m.role === 'user');
    if (!lastAssistant && !lastUser) return;

    try {
      setLoading(true);
      const resp = await apiClient.regenNode({
        assistantId: lastAssistant?.id,
        userMessageId: lastAssistant ? undefined : lastUser?.id,
        sessionId: getCurrentSessionId(),
        branchId: currentBranchId || 'main',
        historyMode: 'last_user',
      });
      if (!resp.success) {
        throw new Error(resp.message || 'Regen failed');
      }
      // Reload conversation to reflect regenerated assistant
      await loadConversation();
      await refreshBranches();
    } catch (e) {
      console.error('regenNode failed:', e);
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: `❌ Failed to regenerate: ${e instanceof Error ? e.message : 'Unknown error'}`,
        timestamp: Date.now(),
      };
      addMessage(errorMessage);
    } finally {
      setLoading(false);
      setTyping(false);
    }
  };

  // Handler for stopping generation
  const handleStopGeneration = () => {
    setShouldStop(true);
    setLoading(false);
    setTyping(false);
    
    // Clear streaming state
    currentStreamingMessageId.current = null;
  };

  const handleSendMessage = async (content: string, attachments?: PreparedAttachment[]) => {
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
    const displayName = settings.user?.displayName || generateDisplayName();
    const createdAt = Date.now();
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content,
      timestamp: createdAt,
      attachments,
      meta: {
        authorName: displayName,
        authorType: 'user',
        createdAt,
      }
    };

    // Add user message to store immediately
    addMessage(userMessage);

    // Use requestAnimationFrame to ensure the UI has rendered the user message
    // before starting API processing. This prevents timing issues where the 
    // user message might not appear in the chat history.
    requestAnimationFrame(async () => {
      // Handle regular chat message (no command handling anymore)
      await handleChatMessage(content, attachments);
    });
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

  // Ensure model is loaded, attempt auto-reload if not
  const ensureModelLoaded = async () => {
    try {
      // Check if model is currently loaded
      const modelResponse = await apiClient.getModels();
      console.log('[ChatInterface] ensureModelLoaded -> getModels response:', modelResponse);
      if (modelResponse.success && modelResponse.data?.data?.[0]) {
        try {
          const modelEntry = modelResponse.data.data[0];
          const md: any = modelEntry.config_metadata;
          console.log('[ChatInterface] config metadata from getModels:', md);
          const derived = deriveOmniCapability(md, modelEntry.id);
          if (typeof derived === 'boolean') {
            console.log('[ChatInterface] Setting isOmniCapable from getModels:', derived);
            setIsOmniCapable(derived);
          }
        } catch (metaError) {
          console.warn('[ChatInterface] Failed to interpret config metadata from getModels:', metaError);
        }
        // Model is loaded, we're good
        return;
      }
      
      console.log('🔄 No model loaded, attempting auto-reload...');
      
      // Add system message to inform user about auto-reload
      const autoReloadMessage: Message = {
        id: `system-${Date.now()}`,
        role: 'assistant',
        content: '🔄 No model is currently loaded. Attempting to load the selected model...',
        timestamp: Date.now(),
      };
      addMessage(autoReloadMessage);
      
      // The backend uses lazy loading - making a test request should trigger model loading
      // We'll make a simple health check to trigger the lazy loading
      const healthResponse = await apiClient.health();
      console.log('[ChatInterface] ensureModelLoaded -> health response:', healthResponse);
      
      // Wait a moment for potential model loading
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Check again if model is now loaded
      const recheckResponse = await apiClient.getModels();
      console.log('[ChatInterface] ensureModelLoaded -> recheck getModels response:', recheckResponse);
      if (recheckResponse.success && recheckResponse.data?.data?.[0]) {
        try {
          const modelEntry = recheckResponse.data.data[0];
          const md: any = modelEntry.config_metadata;
          console.log('[ChatInterface] config metadata from recheck:', md);
          const derived = deriveOmniCapability(md, modelEntry.id);
          if (typeof derived === 'boolean') {
            console.log('[ChatInterface] Setting isOmniCapable from recheck:', derived);
            setIsOmniCapable(derived);
          }
        } catch (metaError) {
          console.warn('[ChatInterface] Failed to interpret config metadata from recheck:', metaError);
        }
        console.log('✅ Model auto-loaded successfully');
        
        const successMessage: Message = {
          id: `system-${Date.now()}`,
          role: 'assistant',
          content: '✅ Model loaded successfully. You can now send your message.',
          timestamp: Date.now(),
        };
        addMessage(successMessage);
      } else {
        throw new Error('Failed to auto-load model');
      }
      
    } catch (error) {
      console.error('❌ Model auto-reload failed:', error);
      
      const errorMessage: Message = {
        id: `system-${Date.now()}`,
        role: 'assistant',
        content: '❌ Failed to load model automatically. Please use the Model Configuration panel to select and load a model, or check the System Monitor to reload the current model.',
        timestamp: Date.now(),
      };
      addMessage(errorMessage);
      
      throw new Error('Model not available');
    }
  };

  const buildContentParts = (text: string, attachments?: PreparedAttachment[]) => {
    if (!attachments || attachments.length === 0) {
      return text; // plain string
    }
    const map = new Map(attachments.map(a => [a.id, a] as const));
    const parts: Array<any> = [];
    const re = /(\[attachment:[^\]]+\])/g;
    const tokens = text.split(re).filter(Boolean);
    for (const token of tokens) {
      const m = token.match(/^\[attachment:([^\]]+)\]$/);
      if (m) {
        const att = map.get(m[1]);
        if (!att) continue;
        if (att.type === 'image') {
          parts.push({ type: 'image_url', content: att.dataUrl ?? att.base64 ?? '' });
        } else if (att.type === 'audio') {
          parts.push({ type: 'audio_url', content: att.dataUrl ?? att.base64 ?? '' });
        } else if (att.type === 'video') {
          parts.push({ type: 'video_url', content: att.dataUrl ?? att.base64 ?? '' });
        } else if (att.type === 'document' && att.dataUrl) {
          parts.push({ type: 'text', content: att.dataUrl });
        }
      } else if (token.trim().length > 0) {
        parts.push({ type: 'text', content: token });
      }
    }
    return parts.length > 0 ? parts : text;
  };

  const [isOmniCapable, setIsOmniCapable] = React.useState(false);

  React.useEffect(() => {
    void ensureModelLoaded();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  React.useEffect(() => {
    if (settings.selectedModel) {
      void ensureModelLoaded();
    }
  // eslint-disable-next-line react-hooks-exhaustive-deps
  }, [settings.selectedModel]);

  const handleChatMessage = async (content: string, attachments?: PreparedAttachment[]) => {
    setTyping(true);
    setShouldStop(false);
    
    try {
      // Prepare messages for API
      const apiMessages: ChatCompletionRequest['messages'] = messages
        .filter(msg => msg.role !== 'system') // Exclude system messages from API
        .map(msg => ({
          role: (msg.role as 'user' | 'assistant' | 'system'),
          content: (msg.content as any),
        }));

      // Add the current user message (possibly multimodal)
      const contentOrParts = isOmniCapable ? buildContentParts(content, attachments) : content;
      apiMessages.push({ role: 'user', content: contentOrParts as any });

      // Auto-reload model if needed before attempting chat
      await ensureModelLoaded();

      // Check if streaming is enabled
      const useStreaming = generationParams.stream ?? false;
      
      if (useStreaming) {
        console.log('🔄 Using streaming mode');
        
        // Create initial assistant message for progressive updates
        const assistantMessageId = `assistant-${Date.now()}`;
        const start = performance.now();
        const assistantMessage: Message = {
          id: assistantMessageId,
          role: 'assistant',
          content: '', // Start empty for streaming
          timestamp: Date.now(),
          meta: {
            authorType: 'ai',
            authorName: settings.selectedModel || 'AI',
            modelName: settings.selectedModel || undefined,
            engine: settings.selectedProvider || undefined,
            createdAt: Date.now(),
          }
        };
        
        addMessage(assistantMessage);
        currentStreamingMessageId.current = assistantMessageId;
        
        // Track accumulated content for streaming
        let accumulatedContent = '';
        
        // Use streaming API
        const response = await apiClient.streamChatCompletion({
          messages: apiMessages,
          session_id: getCurrentSessionId(),
          branch_id: currentBranchId,
          temperature: generationParams.temperature,
          max_tokens: generationParams.maxTokens,
          top_p: generationParams.topP,
          stream: true, // Explicitly enable streaming
        }, (chunk: string) => {
          // Progressive update callback - update the message with each chunk
          if (currentStreamingMessageId.current && !shouldStop) {
            accumulatedContent += chunk;
            // The updateMessage function now requires 4 parameters
            updateMessage(
              currentConversationId || '',
              currentBranchId,
              currentStreamingMessageId.current, 
              { content: accumulatedContent }
            );
          }
        });
        
        currentStreamingMessageId.current = null;
        
        if (!response.success) {
          throw new Error(response.message || 'Streaming failed');
        }
        
        console.log('✅ Streaming completed successfully');
        // Attach duration metadata
        const durationMs = Math.max(0, Math.round(performance.now() - start));
        updateMessage(
          currentConversationId || '',
          currentBranchId,
          assistantMessageId,
          { meta: { ...(assistantMessage.meta || {}), durationMs } } as any
        );
        
      } else {
        console.log('🔄 Using non-streaming mode');
        
        // Use non-streaming API (original behavior)
        const start = performance.now();
        const response = await apiClient.chatCompletion({
          messages: apiMessages,
          session_id: getCurrentSessionId(),
          branch_id: currentBranchId,
          temperature: generationParams.temperature,
          max_tokens: generationParams.maxTokens,
          top_p: generationParams.topP,
          stream: false, // Explicitly disable streaming
        });

        if (response.success && response.data) {
          // Add complete assistant response
          const durationMs = Math.max(0, Math.round(performance.now() - start));
          const assistantMessage: Message = {
            id: `assistant-${Date.now()}`,
            role: 'assistant',
            content: response.data.choices?.[0]?.message?.content || 'No response generated',
            timestamp: Date.now(),
            meta: {
              authorType: 'ai',
              authorName: response.data.model || settings.selectedModel || 'AI',
              modelName: response.data.model || settings.selectedModel || undefined,
              engine: settings.selectedProvider || undefined,
              createdAt: Date.now(),
              durationMs,
            }
          };
          addMessage(assistantMessage);
        } else {
          throw new Error(response.message || 'Failed to get response');
        }
      }
        
      // Refresh branch data after successful chat exchange
      await refreshBranches();
      
    } catch (error) {
      console.error('Chat message error:', error);
      
      // Clear streaming state on error
      currentStreamingMessageId.current = null;
      
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
    <div className={`flex flex-col h-full min-h-0 bg-background ${className}`}>
      {/* Chat history (internal scroll) */}
      <div className="flex-1 min-h-0 relative">
        <ChatHistory
          messages={messages}
          isTyping={isTyping}
          isLoading={isLoading}
        />
      </div>

      {/* Message input pinned to bottom */}
      <div className="sticky bottom-0 z-10 border-t border-border bg-background">
        <MessageInput
          onSendMessage={handleSendMessage}
          onAttachFiles={handleAttachFiles}
          disabled={isLoading}
          isLoading={isLoading || isTyping}
          isOmniCapable={isOmniCapable}
        />
      </div>
    </div>
  );
}
