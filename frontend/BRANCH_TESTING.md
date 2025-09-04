# Branch-Aware Store Testing Guide

This document provides instructions for manually testing the new branch-aware store implementation in the Oumi webchat application.

## Setup

1. Make sure you have the latest code with the branch-aware store changes
2. Start the frontend application: `npm run dev`
3. Open the application in a browser

## Test Cases

### Case 1: Basic Branch Isolation

**Objective**: Verify that messages in different branches are properly isolated.

1. Start a new conversation by typing a message in the chat input
2. Create a new branch by clicking the "New Branch" button in the branch panel
3. Enter a name for the branch (e.g., "Test Branch") and confirm
4. Send a unique message in this new branch
5. Switch back to the main branch
6. Verify that the message sent in the test branch is not visible in the main branch
7. Switch back to the test branch and verify the unique message is still there

**Expected Result**: Each branch maintains its own set of messages. Messages added in one branch should not appear in other branches.

### Case 2: Branch Creation and Deletion

**Objective**: Verify branch management operations.

1. Start a new conversation
2. Create multiple branches (at least 3) with different names
3. Send a unique message in each branch
4. Delete one of the branches
5. Verify that the remaining branches still have their unique messages
6. Create a new branch with the same name as the deleted one
7. Verify that this new branch starts empty and doesn't inherit messages from the deleted branch

**Expected Result**: Branch deletion only affects the specific branch being deleted. New branches always start empty regardless of previous branches with the same name.

### Case 3: Conversation Loading

**Objective**: Test that conversations load correctly with their branch structure.

1. Start a new conversation and create several branches
2. Send unique messages in each branch
3. Start a completely new conversation
4. Use the history sidebar to load the first conversation
5. Verify that all branches are preserved
6. Switch between branches and verify each branch has its correct messages

**Expected Result**: Loading a conversation should restore all branches and their respective messages.

### Case 4: Cross-Session Persistence

**Objective**: Verify that branch data persists across sessions.

1. Start a new conversation and create several branches
2. Send unique messages in each branch
3. Close the browser and reopen the application
4. Load the previous conversation from history
5. Verify that all branches are preserved with their messages intact

**Expected Result**: Branch structure and messages should persist across sessions.

### Case 5: Concurrent Branch Operations

**Objective**: Test branch operations while messages are being processed.

1. Start a new conversation
2. Send a message that will trigger a longer response
3. While the assistant is responding, try to:
   - Create a new branch
   - Switch to a different branch
   - Delete a branch
4. Verify that these operations don't cause unexpected behavior

**Expected Result**: Branch operations should be safely queued or handled while other operations are in progress.

### Case 6: Session Association

**Objective**: Verify that conversations are properly associated with sessions.

1. Start a new session (or restart the application)
2. Create a new conversation with a unique message
3. Check the history sidebar to confirm the conversation appears
4. Start another new session (or restart in incognito mode)
5. Verify that the conversation is still accessible in the history

**Expected Result**: Conversations should be accessible across multiple sessions.

## Reporting Issues

If you encounter any issues during testing:

1. Note the specific test case and step where the issue occurred
2. Capture the browser console logs (if possible)
3. Note any error messages that appeared
4. Report these details in the issue tracker

## Debugging Tips

- Check browser console for errors
- Inspect the store state using Redux DevTools
- For the Zustand store, you can debug by adding these temporary lines:
  ```javascript
  // Temporary debug code
  console.log('Current store state:', useChatStore.getState());
  ```
- Monitor network requests to see API interactions