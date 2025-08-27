/**
 * Message input component with send functionality
 */

"use client";

import React from 'react';
import { Send, Paperclip, Loader2 } from 'lucide-react';

interface MessageInputProps {
  onSendMessage: (message: string) => void;
  onAttachFiles?: (files: FileList) => void;
  disabled?: boolean;
  isLoading?: boolean;
  placeholder?: string;
}

export default function MessageInput({
  onSendMessage,
  onAttachFiles,
  disabled = false,
  isLoading = false,
  placeholder = "Type your message or /command...",
}: MessageInputProps) {
  const [message, setMessage] = React.useState('');
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !disabled && !isLoading) {
      onSendMessage(message.trim());
      setMessage('');
      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setMessage(e.target.value);
    
    // Auto-resize textarea
    const textarea = e.target;
    textarea.style.height = 'auto';
    const newHeight = Math.min(textarea.scrollHeight, 120); // Max height of 120px
    textarea.style.height = `${newHeight}px`;
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0 && onAttachFiles) {
      onAttachFiles(e.target.files);
      // Reset file input
      e.target.value = '';
    }
  };

  const isCommand = message.trim().startsWith('/');

  return (
    <div className="border-t border-border bg-card p-4">
      <form onSubmit={handleSubmit} className="flex items-end gap-3">
        {/* File attachment button */}
        {onAttachFiles && (
          <>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              className="hidden"
              onChange={handleFileSelect}
              accept="image/*,.pdf,.txt,.json,.csv,.md"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={disabled || isLoading}
              className="flex-shrink-0 p-2 rounded-md border border-border hover:bg-accent disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              title="Attach files"
            >
              <Paperclip size={20} className="text-muted-foreground" />
            </button>
          </>
        )}

        {/* Message input */}
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={disabled || isLoading}
            rows={1}
            className={`w-full resize-none border rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed transition-all text-input bg-input placeholder:text-muted-foreground ${
              isCommand 
                ? 'border-purple-500 bg-purple-900/20' 
                : 'border-border'
            }`}
            style={{ minHeight: '40px' }}
          />
          
          {/* Command indicator */}
          {isCommand && (
            <div className="absolute -top-6 left-0 text-xs text-purple-400 font-medium">
              Command mode
            </div>
          )}
        </div>

        {/* Send button */}
        <button
          type="submit"
          disabled={!message.trim() || disabled || isLoading}
          className="flex-shrink-0 bg-primary hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed text-primary-foreground p-2 rounded-md transition-colors flex items-center justify-center min-w-[40px] min-h-[40px]"
        >
          {isLoading ? (
            <Loader2 size={20} className="animate-spin" />
          ) : (
            <Send size={20} />
          )}
        </button>
      </form>
      
      {/* Helper text */}
      <div className="mt-2 text-xs text-muted-foreground flex items-center justify-between">
        <span>
          Press Enter to send, Shift+Enter for new line
        </span>
        {isCommand && (
          <span className="text-purple-400">
            Available commands: /clear, /delete, /regen, /help
          </span>
        )}
      </div>
    </div>
  );
}