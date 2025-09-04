# Store Adapters

This directory contains adapter utilities for the Oumi chat application's store management. The adapters provide functions to help with:

1. Transitioning between legacy and branch-aware data structures
2. Converting between normalized and flat data formats
3. Providing utilities for branch operations
4. Auto-saving conversations

## Key Components

### `store-adapter.ts`

Primary adapter utility with functions for migrating between data formats:

- **adaptLegacyConversation** - Converts a legacy flat conversation to a branch-aware format
- **flattenBranchMessages** - Extracts all messages from all branches (for backwards compatibility)
- **normalizedToLegacy** - Converts branch-specific messages to legacy flat array
- **legacyToNormalized** - Converts flat message array to normalized branch structure
- **buildBranchStructure** - Creates branch structure from normalized message storage
- **getBranchMetadata** - Generates branch metadata from message structure
- **autoSaveConversation** - Helper for persisting conversations

> **Important**: `SessionManager` has been moved to `/src/lib/session-manager.ts` to avoid duplication.

## Usage Examples

### Converting Legacy Format to Branch-Aware

```typescript
import { adaptLegacyConversation } from './adapters/store-adapter';

// Legacy conversation with flat message array
const legacyConversation = {
  id: 'conv1',
  title: 'Test Conversation',
  messages: [...],  // Flat message array
  createdAt: '2023-10-15T12:00:00Z',
  updatedAt: '2023-10-15T12:01:00Z'
};

// Convert to branch-aware format
const branchAwareConversation = adaptLegacyConversation(legacyConversation);

// Result has branches object with main branch containing messages
// {
//   id: 'conv1',
//   title: 'Test Conversation',
//   messages: [...],  // Original array kept for compatibility
//   branches: {
//     main: { messages: [...] }  // Same messages in branch format
//   },
//   createdAt: '2023-10-15T12:00:00Z',
//   updatedAt: '2023-10-15T12:01:00Z'
// }
```

### Working with Normalized Branch Messages

```typescript
import { 
  normalizedToLegacy,
  legacyToNormalized
} from './adapters/store-adapter';

// Extract messages for specific branch in a conversation
const messages = normalizedToLegacy('conv1', 'main', conversationMessages);

// Convert back to normalized format
const normalized = legacyToNormalized('conv1', 'main', messages);
```

### Session Management

```typescript
import { SessionManager } from '../session-manager';

// Get current session ID for grouping conversations
const sessionId = SessionManager.getCurrentSessionId();

// Get the current session with metadata
const session = SessionManager.getCurrentSession();

// Create a new session with name and metadata
const newSessionId = SessionManager.startNewSession('Project Research', { modelId: 'gpt-4o' });

// Add conversation to current session
SessionManager.addConversation('conversation-123');

// Update session metadata
SessionManager.updateCurrentSession({ name: 'Updated Session Name' });
```

## Testing

The adapter utilities include test coverage in `store-adapter.test.ts`.