# Branch-Aware Store Refactoring

## Overview

This document summarizes the changes implemented for the branch-aware store refactoring in the Oumi webchat frontend. The goal of this refactoring was to implement a normalized, branch-aware data structure for storing and managing conversation messages.

## Key Changes

### 1. Normalized Branch-Aware Data Structure

We implemented a normalized store structure with branch-specific message storage:

```typescript
conversationMessages: {
  [conversationId: string]: {
    [branchId: string]: Message[]
  }
}
```

This structure provides complete isolation of messages between branches, allowing for independent branch operations without interference.

### 2. Derived State Selectors

To efficiently access data from the normalized structure, we added selectors:

- `getCurrentMessages()` - Returns messages for the active conversation and branch
- `getBranchMessages(conversationId, branchId)` - Returns messages for a specific branch in a conversation
- `getSessionConversations(sessionId)` - Returns conversation IDs associated with a specific session

These selectors abstract away the complexity of the normalized structure, providing a clean interface for components.

### 3. Backward Compatibility

To support existing code and ensure smooth migration, we created adapter utilities in `store-adapter.ts`:

- `adaptLegacyConversation()` - Converts legacy flat message array to branch-aware format
- `flattenBranchMessages()` - Extracts all messages from branches (for backward compatibility)
- `normalizedToLegacy()` - Converts from normalized to flat structure
- `legacyToNormalized()` - Converts from flat to normalized structure

### 4. Updated UI Components

We updated key UI components to work with the new branch-aware structure:

- **ChatInterface.tsx** - Now uses `getCurrentMessages()` selector
- **BranchTree.tsx** - Updated to use branch-specific message storage
- **ChatHistorySidebar.tsx** - Modified to use adapters for displaying branch contents

### 5. Session Management

Added support for cross-session tracking with:

- `activeSessionIds` - Tracks active session IDs
- `conversationsBySession` - Maps sessions to conversations
- `SessionManager` - Singleton class for session management

### 6. Test Coverage

- Added unit tests for store functionality in `store.test.ts`
- Created a manual testing guide in `BRANCH_TESTING.md`

## Implementation Details

### Store Structure

The store now maintains these key state elements:

```typescript
{
  // Normalized branch-specific message storage
  conversationMessages: { ... },
  
  // Branch metadata
  branches: [ ... ],
  currentBranchId: string,
  
  // Conversation management
  conversations: [ ... ],
  currentConversationId: string | null,
  
  // Session tracking
  activeSessionIds: string[],
  conversationsBySession: { [sessionId: string]: string[] }
}
```

### API Changes

#### Selectors

```typescript
// Get messages for current conversation and branch
getCurrentMessages: () => Message[];

// Get messages for specific branch
getBranchMessages: (conversationId: string, branchId: string) => Message[];

// Get conversations for a specific session
getSessionConversations: (sessionId: string) => string[];
```

#### Branch Operations

```typescript
// Add a new branch
addBranch: (branch: ConversationBranch) => void;

// Delete a branch
deleteBranch: (branchId: string) => void;

// Switch to a different branch
setCurrentBranch: (branchId: string) => void;
```

#### Message Operations

```typescript
// Add a message to current branch
addMessage: (message: Message) => void;

// Set messages for a specific branch
setMessages: (conversationId: string, branchId: string, messages: Message[]) => void;

// Update a specific message
updateMessage: (conversationId: string, branchId: string, messageId: string, updates: Partial<Message>) => void;
```

## Migration Strategy

The implementation includes these key strategies for smooth migration:

1. **Feature Flags**: New code paths check for existence of currentConversationId before using branch-specific operations

2. **Fallbacks**: Legacy methods are preserved for backward compatibility

3. **Adapters**: Utility functions convert between old and new formats

4. **Progressive Enhancement**: Components are updated one at a time to use the new structure

## Known Limitations

- Some UI components still rely on the legacy flat message structure
- Full branch comparison/merging operations are not yet implemented (placeholder UI exists)
- Real-time updates across multiple sessions could benefit from further optimization

## Next Steps

1. Complete UI component updates for all remaining components
2. Implement branch comparison and merging functionality
3. Optimize real-time updates and cross-session synchronization
4. Add comprehensive error handling for branch operations
5. Remove legacy code paths once migration is complete

## Testing Notes

Refer to `BRANCH_TESTING.md` for detailed manual testing procedures for the new branch-aware store functionality.