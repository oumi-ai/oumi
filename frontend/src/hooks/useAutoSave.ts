/**
 * Auto-save hook for conversations
 */

import { useEffect, useRef, useState } from 'react';
import { useChatStore } from '@/lib/store';

export function useAutoSave() {
  const { settings, getCurrentMessages } = useChatStore();
  const messages = getCurrentMessages();
  const [lastSaved, setLastSaved] = useState<Date | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const lastMessagesLength = useRef(messages.length);

  const saveConversation = async () => {
    if (messages.length === 0) return;
    
    setIsSaving(true);
    try {
      // The Zustand persist middleware handles the actual saving
      // We just need to trigger a state update to ensure persistence
      await new Promise(resolve => setTimeout(resolve, 100)); // Simulate save time
      setLastSaved(new Date());
    } catch (error) {
      console.error('Auto-save failed:', error);
    } finally {
      setIsSaving(false);
    }
  };

  useEffect(() => {
    if (!settings.autoSave?.enabled) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }

    const intervalMs = (settings.autoSave.intervalMinutes || 5) * 60 * 1000;

    // Save immediately if messages have changed
    if (messages.length !== lastMessagesLength.current && messages.length > 0) {
      saveConversation();
      lastMessagesLength.current = messages.length;
    }

    // Set up interval for periodic saves
    intervalRef.current = setInterval(() => {
      if (messages.length > 0) {
        saveConversation();
      }
    }, intervalMs);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [settings.autoSave, messages.length]);

  return {
    isAutoSaveEnabled: settings.autoSave?.enabled || false,
    autoSaveInterval: settings.autoSave?.intervalMinutes || 5,
    lastSaved,
    isSaving,
  };
}