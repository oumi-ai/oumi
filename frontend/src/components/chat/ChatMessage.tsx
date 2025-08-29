/**
 * Individual chat message component
 */

"use client";

import React from 'react';
import { Message } from '@/lib/types';
import { User, Bot, Copy, Check, Trash2, RefreshCw, Edit3, Save } from 'lucide-react';
import MarkdownRenderer from '@/components/ui/MarkdownRenderer';
import { useChatStore } from '@/lib/store';
import apiClient from '@/lib/api';

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
  const { updateMessage, deleteMessage, addMessage } = useChatStore();

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
      // Use position-specific delete if messageIndex is available
      const args = messageIndex !== undefined ? [messageIndex.toString()] : [];
      const response = await apiClient.executeCommand('delete', args);
      if (response.success) {
        deleteMessage(message.id);
        // Refresh the conversation to show updated state
        setTimeout(() => window.location.reload(), 500);
      } else {
        console.error('Failed to delete message:', response.message);
        alert('Failed to delete message: ' + (response.message || 'Unknown error'));
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
      // Use position-specific regeneration if messageIndex is available
      const args = messageIndex !== undefined ? [messageIndex.toString()] : [];
      const response = await apiClient.executeCommand('regen', args);
      if (response.success) {
        // The backend will handle regeneration, refresh to show new response
        setTimeout(() => window.location.reload(), 1000);
      } else {
        console.error('Failed to regenerate message:', response.message);
        alert('Failed to regenerate message: ' + (response.message || 'Unknown error'));
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
      // Update message locally for now
      updateMessage(message.id, { content: editContent });
      setIsEditing(false);
      
      // TODO: Send update to backend to persist the change
      console.log('WYSIWYG edit saved locally. Backend persistence not yet implemented.');
    } catch (error) {
      console.error('Error saving edit:', error);
      alert('Error saving edit');
    } finally {
      setActionInProgress(null);
    }
  };

  const handleCancelEdit = () => {
    setIsEditing(false);
    setEditContent(message.content);
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
          {isUser ? (
            // User messages - render as plain text with line breaks
            <div className="whitespace-pre-wrap">{message.content}</div>
          ) : isEditing ? (
            // Assistant message editing mode
            <div className="space-y-3">
              <textarea
                value={editContent}
                onChange={(e) => setEditContent(e.target.value)}
                className="w-full h-32 p-3 border border-gray-300 rounded-md resize-vertical focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Edit the assistant's response..."
                disabled={actionInProgress === 'save'}
              />
              <div className="flex gap-2">
                <button
                  onClick={handleSaveEdit}
                  disabled={actionInProgress === 'save'}
                  className="px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50 text-sm flex items-center gap-1"
                >
                  <Save size={12} />
                  {actionInProgress === 'save' ? 'Saving...' : 'Save'}
                </button>
                <button
                  onClick={handleCancelEdit}
                  disabled={actionInProgress === 'save'}
                  className="px-3 py-1 bg-gray-600 text-white rounded hover:bg-gray-700 disabled:opacity-50 text-sm"
                >
                  Cancel
                </button>
              </div>
            </div>
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
              disabled={!!actionInProgress}
            >
              {copied ? (
                <Check size={14} className="text-green-600" />
              ) : (
                <Copy size={14} className="text-gray-500" />
              )}
            </button>

            {/* Assistant-only actions */}
            {!isUser && !isEditing && (
              <>
                {/* Delete button */}
                <button
                  onClick={handleDelete}
                  className="p-1 rounded hover:bg-red-100"
                  title="Delete this message"
                  disabled={actionInProgress === 'delete'}
                >
                  <Trash2 size={14} className={actionInProgress === 'delete' ? 'text-gray-400' : 'text-red-600'} />
                </button>

                {/* Regenerate button */}
                <button
                  onClick={handleRegen}
                  className="p-1 rounded hover:bg-blue-100"
                  title="Regenerate response"
                  disabled={actionInProgress === 'regen'}
                >
                  <RefreshCw size={14} className={actionInProgress === 'regen' ? 'text-gray-400 animate-spin' : 'text-blue-600'} />
                </button>

                {/* Edit button */}
                <button
                  onClick={handleEdit}
                  className="p-1 rounded hover:bg-yellow-100"
                  title="Edit this response"
                  disabled={!!actionInProgress}
                >
                  <Edit3 size={14} className="text-yellow-600" />
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}