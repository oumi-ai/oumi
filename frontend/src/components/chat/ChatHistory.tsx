/**
 * Chat history component with message list and scrolling
 */

"use client";

import React from 'react';
import { Message } from '@/lib/types';
import ChatMessage from './ChatMessage';
import TypingIndicator from './TypingIndicator';

interface ChatHistoryProps {
  messages: Message[];
  isTyping?: boolean;
  isLoading?: boolean;
  className?: string;
}

export default function ChatHistory({
  messages,
  isTyping = false,
  isLoading = false,
  className = '',
}: ChatHistoryProps) {
  const scrollAreaRef = React.useRef<HTMLDivElement>(null);
  const isAtBottomRef = React.useRef(true);

  // Auto-scroll to bottom when new messages arrive
  React.useEffect(() => {
    if (scrollAreaRef.current && isAtBottomRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  // Track if user is at bottom of scroll
  const handleScroll = () => {
    if (scrollAreaRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = scrollAreaRef.current;
      const threshold = 100; // pixels from bottom
      isAtBottomRef.current = scrollTop + clientHeight >= scrollHeight - threshold;
    }
  };

  const scrollToBottom = () => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTo({
        top: scrollAreaRef.current.scrollHeight,
        behavior: 'smooth',
      });
      isAtBottomRef.current = true;
    }
  };

  const isEmpty = messages.length === 0 && !isLoading;

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Messages container */}
      <div
        ref={scrollAreaRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto chat-scroll"
        style={{ scrollBehavior: 'smooth' }}
      >
        {isEmpty ? (
          // Empty state
          <div className="flex items-center justify-center h-full text-center p-8">
            <div className="max-w-md space-y-4">
              <div className="text-6xl">💬</div>
              <h3 className="text-lg font-semibold text-gray-700">
                Start a conversation
              </h3>
              <p className="text-gray-500">
                Ask a question, give a command, or just say hello!
              </p>
              <div className="text-sm text-gray-400 space-y-1">
                <p>You can use commands like:</p>
                <div className="flex flex-wrap gap-2 justify-center">
                  {['/help', '/clear', '/regen'].map((cmd) => (
                    <code
                      key={cmd}
                      className="px-2 py-1 bg-gray-100 rounded text-purple-600"
                    >
                      {cmd}
                    </code>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ) : (
          // Messages list
          <div className="divide-y divide-gray-100">
            {messages.map((message, index) => (
              <ChatMessage
                key={message.id}
                message={message}
                isLatest={index === messages.length - 1}
              />
            ))}
            
            {/* Typing indicator */}
            {isTyping && (
              <div className="px-4 py-6">
                <TypingIndicator />
              </div>
            )}
          </div>
        )}
      </div>

      {/* Scroll to bottom button */}
      {!isAtBottomRef.current && messages.length > 0 && (
        <div className="absolute bottom-4 right-4">
          <button
            onClick={scrollToBottom}
            className="bg-blue-600 hover:bg-blue-700 text-white p-3 rounded-full shadow-lg transition-colors"
            title="Scroll to bottom"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 14l-7 7m0 0l-7-7m7 7V3"
              />
            </svg>
          </button>
        </div>
      )}
    </div>
  );
}