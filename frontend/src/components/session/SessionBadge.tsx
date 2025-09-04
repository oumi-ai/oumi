/**
 * SessionBadge component to display the current session in the UI
 */

"use client";

import React from 'react';
import { History, Clock } from 'lucide-react';
import { useChatStore } from '@/lib/store';
import { Session } from '@/lib/types';

interface SessionBadgeProps {
  className?: string;
  showDetails?: boolean;
}

export default function SessionBadge({ className = '', showDetails = false }: SessionBadgeProps) {
  const { currentSessionId } = useChatStore();
  const session = useChatStore.getState().getSessionById(currentSessionId);
  
  if (!session) return null;
  
  // Calculate time elapsed since session creation
  const getTimeElapsed = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    
    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
  };
  
  return (
    <div className={`inline-flex items-center ${className}`}>
      <div className="flex items-center px-2 py-1 rounded-md bg-primary/10 text-primary border border-primary/20">
        <History className="w-3 h-3 mr-1.5" />
        <span className="text-xs font-medium">{session.name}</span>
        
        {showDetails && (
          <>
            <div className="mx-1.5 h-3 w-px bg-primary/30" />
            <div className="flex items-center">
              <Clock className="w-3 h-3 mr-1" />
              <span className="text-xs">{getTimeElapsed(session.createdAt)}</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}