/**
 * Individual chat message component
 */

"use client";

import React from 'react';
import { Message } from '@/lib/types';
import { User, Bot, Copy, Check } from 'lucide-react';

interface ChatMessageProps {
  message: Message;
  isLatest?: boolean;
}

export default function ChatMessage({ message, isLatest = false }: ChatMessageProps) {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error('Failed to copy message:', error);
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
          ? 'bg-gray-50' 
          : 'bg-white'
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
        <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
          {isUser ? 'You' : 'Assistant'}
        </div>

        {/* Message text */}
        <div
          className="prose prose-sm max-w-none text-gray-800 leading-relaxed whitespace-pre-wrap break-words"
        >
          {message.content}
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
        <div className="flex items-center gap-3 pt-2">
          <span className="text-xs text-gray-400">
            {new Date(message.timestamp).toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </span>

          {/* Copy button */}
          <button
            onClick={handleCopy}
            className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-gray-200"
            title="Copy message"
          >
            {copied ? (
              <Check size={14} className="text-green-600" />
            ) : (
              <Copy size={14} className="text-gray-500" />
            )}
          </button>
        </div>
      </div>
    </div>
  );
}