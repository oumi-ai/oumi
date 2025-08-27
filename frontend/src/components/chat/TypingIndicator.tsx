/**
 * Typing indicator component for assistant responses
 */

import React from 'react';
import { Bot } from 'lucide-react';

export default function TypingIndicator() {
  return (
    <div className="flex gap-3">
      {/* Avatar */}
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-green-500 text-white flex items-center justify-center">
        <Bot size={16} />
      </div>

      {/* Typing animation */}
      <div className="flex-1 space-y-2">
        <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
          Assistant
        </div>
        
        <div className="flex items-center space-x-1">
          <div className="flex space-x-1">
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"></div>
            <div 
              className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"
              style={{ animationDelay: '0.2s' }}
            ></div>
            <div 
              className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"
              style={{ animationDelay: '0.4s' }}
            ></div>
          </div>
          <span className="text-sm text-gray-500 ml-2">
            Assistant is typing...
          </span>
        </div>
      </div>
    </div>
  );
}