import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';

import ChatHistorySidebar from '../ChatHistorySidebar';
import { useChatStore } from '@/lib/store';
import apiClient from '@/lib/unified-api';

describe('ChatHistorySidebar session-scoped merge', () => {
  beforeEach(() => {
    // Reset store to a known state with current session S1
    useChatStore.setState({
      conversationMessages: {},
      conversations: [],
      currentConversationId: null,
      currentBranchId: 'main',
      currentSessionId: 'S1',
    } as any);

    jest.spyOn(apiClient as any, 'getStorageItem').mockReset();
    jest.spyOn(apiClient as any, 'listConversations').mockReset();
    jest.spyOn(apiClient as any, 'loadConversation').mockReset();
  });

  it('shows only current session conversations, ignoring others', async () => {
    // Prepare storage: conversations for S1 and S2
    const s1Key = 'conversations_S1';
    const s2Key = 'conversations_S2';

    const s1Convs = [
      { id: 'conv_s1_a', name: 'S1 Conv A', lastModified: '2025-01-02T10:00:00Z', messageCount: 2, preview: '...' },
      { id: 'conv_s1_b', name: 'S1 Conv B', lastModified: '2025-01-01T10:00:00Z', messageCount: 1, preview: '...' },
    ];
    const s2Convs = [
      { id: 'conv_s2_x', name: 'S2 Conv X', lastModified: '2025-01-03T10:00:00Z', messageCount: 3, preview: '...' },
    ];

    // Spy getStorageItem to return data only when asked for the exact key
    (apiClient.getStorageItem as jest.Mock).mockImplementation(async (key: string, def: any) => {
      if (key === s1Key) return s1Convs;
      if (key === s2Key) return s2Convs;
      return def;
    });

    // listConversations under the hood will call getStorageItem with `conversations_${sessionId}`
    (apiClient.listConversations as jest.Mock).mockImplementation(async (sessionId: string) => {
      const key = `conversations_${sessionId}`;
      const conversations = await apiClient.getStorageItem(key, []);
      return { success: true, data: { conversations } };
    });

    render(<ChatHistorySidebar />);

    // Expect S1 conversations to appear
    expect(await screen.findByText('S1 Conv A')).toBeInTheDocument();
    expect(await screen.findByText('S1 Conv B')).toBeInTheDocument();

    // Ensure S2 is not shown
    expect(screen.queryByText('S2 Conv X')).toBeNull();

    // Verify that listConversations was called with current session only
    expect((apiClient.listConversations as jest.Mock)).toHaveBeenCalledWith('S1');
  });

  it('loads conversation into current branch using current session only', async () => {
    // Arrange storage for S1
    const s1Key = 'conversations_S1';
    const s1Convs = [
      { id: 'conv_s1_a', name: 'S1 Conv A', lastModified: '2025-01-02T10:00:00Z', messageCount: 2, preview: '...' },
    ];
    (apiClient.getStorageItem as jest.Mock).mockImplementation(async (key: string, def: any) => {
      if (key === s1Key) return s1Convs;
      return def;
    });
    (apiClient.listConversations as jest.Mock).mockImplementation(async (sessionId: string) => {
      const key = `conversations_${sessionId}`;
      const conversations = await apiClient.getStorageItem(key, []);
      return { success: true, data: { conversations } };
    });

    // Mock loadConversation to capture session usage
    (apiClient.loadConversation as jest.Mock).mockResolvedValue({ success: true, data: { messages: [] } });

    render(<ChatHistorySidebar />);

    // Wait for list to render, then click the conversation to open preview
    const item = await screen.findByText('S1 Conv A');
    fireEvent.click(item);

    // Click the load button in the preview
    const loadButton = await screen.findByRole('button', { name: /Load into Current Branch/i });
    fireEvent.click(loadButton);

    await waitFor(() => {
      expect((apiClient.loadConversation as jest.Mock)).toHaveBeenCalled();
    });

    // Assert it used the current session id
    const [calledSessionId, calledConversationId] = (apiClient.loadConversation as jest.Mock).mock.calls[0].slice(0, 2);
    expect(calledSessionId).toBe('S1');
    expect(calledConversationId).toBe('conv_s1_a');
  });
});

