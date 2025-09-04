/**
 * Individual chat message component
 */

"use client";

import React from 'react';
import { Message } from '@/lib/types';
import { User, Bot, Copy, Check, Trash2, RefreshCw, Edit3, Save, GitBranch } from 'lucide-react';
import MarkdownRenderer from '@/components/ui/MarkdownRenderer';
import { useChatStore } from '@/lib/store';
import { useConversationCommand, COMMAND_CONFIGS } from '@/hooks/useConversationCommand';

interface ChatMessageProps {
  message: Message;
  isLatest?: boolean;
  messageIndex?: number; // Position in conversation for targeted operations
}

export default function ChatMessage({ message, isLatest = false, messageIndex }: ChatMessageProps) {
  const [copied, setCopied] = React.useState(false);
  const [isEditing, setIsEditing] = React.useState(false);
  const [editContent, setEditContent] = React.useState(message.content);
  const [actionInProgress, setActionInProgress] = React.useState<string | null>(null);
  const { updateMessage, deleteMessage, addMessage, branches } = useChatStore();
  const { executeCommand, isExecuting } = useConversationCommand();

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error('Failed to copy message:', error);
    }
  };

  const handleDelete = async () => {
    if (!confirm('Are you sure you want to delete this message?')) return;
    
    setActionInProgress('delete');
    try {
      const args = messageIndex !== undefined ? [messageIndex.toString()] : [];
      const result = await executeCommand('delete', args, COMMAND_CONFIGS.delete);
      
      if (!result.success && result.message) {
        alert(result.message);
      }
    } catch (error) {
      console.error('Error deleting message:', error);
      alert('Error deleting message');
    } finally {
      setActionInProgress(null);
    }
  };

  const handleRegen = async () => {
    setActionInProgress('regen');
    try {
      const args = messageIndex !== undefined ? [messageIndex.toString()] : [];
      const result = await executeCommand('regen', args, COMMAND_CONFIGS.regen);
      
      if (!result.success && result.message) {
        alert(result.message);
      }
    } catch (error) {
      console.error('Error regenerating message:', error);
      alert('Error regenerating message');
    } finally {
      setActionInProgress(null);
    }
  };

  const handleEdit = () => {
    setIsEditing(true);
    setEditContent(message.content);
  };

  const handleSaveEdit = async () => {
    if (editContent.trim() === '') {
      alert('Message content cannot be empty');
      return;
    }

    setActionInProgress('save');
    try {
      if (messageIndex !== undefined) {
        const args = [messageIndex.toString(), editContent.trim()];
        const result = await executeCommand('edit', args, COMMAND_CONFIGS.edit);
        
        if (result.success) {
          // Update local state immediately for responsive UI
          // The updateMessage function now requires 4 parameters
          updateMessage(
            currentConversationId || '',
            currentBranchId || 'main',
            message.id, 
            { content: editContent }
          );
          setIsEditing(false);
          console.log('✅ Message edited and persisted to backend');
        } else if (result.message) {
          alert(result.message);
        }
      } else {
        // Fallback: update locally if no messageIndex
        // The updateMessage function now requires 4 parameters
        updateMessage(
          currentConversationId || '',
          currentBranchId || 'main',
          message.id, 
          { content: editContent }
        );
        setIsEditing(false);
        console.warn('⚠️  Message edited locally only (no messageIndex provided)');
      }
    } catch (error) {
      console.error('❌ Error saving edit:', error);
      alert('Error saving edit');
    } finally {
      setActionInProgress(null);
    }
  };

  const handleCancelEdit = () => {
    setIsEditing(false);
    setEditContent(message.content);
  };

  const handleCreateBranch = async () => {
    // Check branch limit (currently limited to 5 branches total)
    if (branches.length >= 5) {
      alert(
        '🌳 Branch Limit Reached\n\n' +
        'You can only have up to 5 active branches at a time for now. ' +
        'This limit may be increased in future versions after further development.\n\n' +
        'Please delete an existing branch before creating a new one.'
      );
      return;
    }

    if (messageIndex === undefined) {
      alert('❌ Cannot create branch: Message position unknown');
      return;
    }
    
    console.log(`🌿 ChatMessage: Creating branch from message index ${messageIndex}`);
    setActionInProgress('branch');
    try {
      const result = await executeCommand(
        'branch_from', 
        [messageIndex.toString()], 
        COMMAND_CONFIGS.branch_from
      );
      
      console.log(`🌿 ChatMessage: Branch creation result:`, result);
      
      if (!result.success && result.message) {
        alert(`❌ ${result.message}`);
      } else if (result.success) {
        // Show success message briefly
        setTimeout(() => {
          alert('✅ New branch created! Switch to it from the branch panel to continue this conversation thread.');
        }, 100);
      }
    } catch (error) {
      console.error('Error creating branch:', error);
      alert('❌ Failed to create branch: An unexpected error occurred');
    } finally {
      setActionInProgress(null);
    }
  };


  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';

  // Don't render system messages in the chat
  if (isSystem) {
    return null;
  }

  return (
    <div
      className={`group flex gap-3 px-4 py-6 ${
        isUser 
          ? 'bg-muted' 
          : 'bg-card'
      } ${isLatest ? 'border-b border-blue-200' : ''}`}
    >
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser
            ? 'bg-blue-500 text-white'
            : 'bg-green-500 text-white'
        }`}
      >
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>

      {/* Message content */}
      <div className="flex-1 space-y-2 overflow-hidden">
        {/* Role label */}
        <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
          {isUser ? 'You' : 'Assistant'}
        </div>

        {/* Message text */}
        <div className="text-foreground leading-relaxed break-words">
          {isEditing ? (
            // Editing mode for both user and assistant messages
            <div className="space-y-3">
              <textarea
                value={editContent}
                onChange={(e) => setEditContent(e.target.value)}
                className="w-full h-32 p-3 border border-border bg-input text-foreground placeholder:text-muted-foreground rounded-md resize-vertical focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                placeholder={isUser ? "Edit your message..." : "Edit the assistant's response..."}
                disabled={actionInProgress === 'save' || isExecuting}
              />
              <div className="flex gap-2">
                <button
                  onClick={handleSaveEdit}
                  disabled={actionInProgress === 'save' || isExecuting}
                  className="px-3 py-1 bg-primary hover:bg-primary/90 disabled:opacity-50 text-primary-foreground rounded text-sm flex items-center gap-1 transition-colors"
                >
                  <Save size={12} />
                  {actionInProgress === 'save' ? 'Saving...' : 'Save'}
                </button>
                <button
                  onClick={handleCancelEdit}
                  disabled={actionInProgress === 'save' || isExecuting}
                  className="px-3 py-1 bg-muted hover:bg-muted/80 disabled:opacity-50 text-muted-foreground rounded text-sm transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : isUser ? (
            // User messages - render as plain text with line breaks
            <div className="whitespace-pre-wrap">{message.content}</div>
          ) : (
            // Assistant messages - render as markdown
            <MarkdownRenderer content={message.content} />
          )}
        </div>

        {/* Attachments */}
        {message.attachments && message.attachments.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-3">
            {message.attachments.map((attachment) => (
              <div
                key={attachment.id}
                className="flex items-center gap-2 px-3 py-2 bg-gray-100 rounded-md text-sm"
              >
                <span className="text-gray-600">{attachment.filename}</span>
                <span className="text-xs text-gray-400">
                  ({(attachment.size / 1024).toFixed(1)} KB)
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Timestamp and actions */}
        <div className="flex items-center gap-2 pt-2 flex-wrap">
          <span className="text-xs text-gray-400">
            {new Date(message.timestamp).toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </span>

          {/* Action buttons - always visible on mobile, hover on desktop */}
          <div className="flex gap-1 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 transition-opacity">
            {/* Copy button - for all messages */}
            <button
              onClick={handleCopy}
              className="p-1 rounded hover:bg-gray-200"
              title="Copy message"
              disabled={!!actionInProgress || isExecuting}
            >
              {copied ? (
                <Check size={14} className="text-green-600" />
              ) : (
                <Copy size={14} className="text-gray-500" />
              )}
            </button>

            {/* Message actions for all types */}
            {!isEditing && (
              <>
                {/* Delete button - for all messages */}
                <button
                  onClick={handleDelete}
                  className="p-1 rounded hover:bg-red-100"
                  title="Delete this message"
                  disabled={actionInProgress === 'delete' || isExecuting}
                >
                  <Trash2 size={14} className={actionInProgress === 'delete' ? 'text-gray-400' : 'text-red-600'} />
                </button>

                {/* Edit button - for all messages */}
                <button
                  onClick={handleEdit}
                  className="p-1 rounded hover:bg-yellow-100"
                  title="Edit this message"
                  disabled={!!actionInProgress || isExecuting}
                >
                  <Edit3 size={14} className="text-yellow-600" />
                </button>

                {/* Assistant-only actions */}
                {!isUser && (
                  <>
                    {/* Regenerate button */}
                    <button
                      onClick={handleRegen}
                      className="p-1 rounded hover:bg-blue-100"
                      title="Regenerate response"
                      disabled={actionInProgress === 'regen' || isExecuting}
                    >
                      <RefreshCw size={14} className={actionInProgress === 'regen' ? 'text-gray-400 animate-spin' : 'text-blue-600'} />
                    </button>

                    {/* Branch from this point button */}
                    <button
                      onClick={handleCreateBranch}
                      className="p-1 rounded hover:bg-purple-100"
                      title="Create new branch from this assistant response"
                      disabled={actionInProgress === 'branch' || isExecuting || messageIndex === undefined}
                    >
                      <GitBranch size={14} className={actionInProgress === 'branch' ? 'text-gray-400' : 'text-purple-600'} />
                    </button>
                  </>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}