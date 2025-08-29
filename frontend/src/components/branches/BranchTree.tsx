/**
 * Advanced branch tree component with multiple view modes and management features
 */

"use client";

import React from 'react';
import { useChatStore } from '@/lib/store';
import { ConversationBranch, Message } from '@/lib/types';
import { Plus, GitBranch, Trash2, TreePine, List, MoreVertical, Shuffle, GitMerge } from 'lucide-react';
import apiClient from '@/lib/api';
import BranchTreeVisualization from './BranchTreeVisualization';
import BranchInheritanceView from './BranchInheritanceView';
import BranchContextMenu from './BranchContextMenu';
import BranchMergeDialog from './BranchMergeDialog';

interface BranchTreeProps {
  className?: string;
}

export default function BranchTree({ className = '' }: BranchTreeProps) {
  const {
    branches,
    currentBranchId,
    messages,
    setBranches,
    setCurrentBranch,
    addBranch,
    deleteBranch,
    setMessages,
  } = useChatStore();
  
  const [isCreating, setIsCreating] = React.useState(false);
  const [newBranchName, setNewBranchName] = React.useState('');
  const [viewMode, setViewMode] = React.useState<'list' | 'tree' | 'inheritance'>('list');
  const [contextMenu, setContextMenu] = React.useState<{
    branch: ConversationBranch;
    position: { x: number; y: number };
  } | null>(null);
  const [mergeDialog, setMergeDialog] = React.useState<{
    sourceBranch: ConversationBranch;
    targetBranch: ConversationBranch;
  } | null>(null);

  // Load branches on component mount with retry mechanism
  React.useEffect(() => {
    let retryTimeout: NodeJS.Timeout;
    
    const loadWithRetry = async () => {
      // First attempt immediately
      await loadBranches();
      
      // Retry after 2 seconds in case backend is starting up
      retryTimeout = setTimeout(async () => {
        console.log('Retrying branch load after backend startup delay...');
        await loadBranches();
      }, 2000);
    };
    
    loadWithRetry();
    
    // Cleanup timeout on unmount
    return () => {
      if (retryTimeout) {
        clearTimeout(retryTimeout);
      }
    };
  }, []);

  const loadBranches = async () => {
    try {
      const response = await apiClient.getBranches();
      if (response.success && response.data) {
        const formattedBranches: ConversationBranch[] = response.data.branches.map((branch: any) => ({
          id: branch.id,
          name: branch.name,
          isActive: branch.id === response.data?.current_branch,
          messageCount: branch.message_count || 0,
          createdAt: branch.created_at,
          lastActive: branch.last_active,
          preview: branch.preview || 'Empty branch',
          parentId: branch.parent,
        }));
        setBranches(formattedBranches);
        setCurrentBranch(response.data?.current_branch || 'main');
      } else if (!response.success) {
        // Backend may not be ready yet, set up default branch
        console.warn('Backend not ready, setting up default branch:', response.message);
        setBranches([{
          id: 'main',
          name: 'main',
          isActive: true,
          messageCount: 0,
          createdAt: new Date().toISOString(),
          lastActive: new Date().toISOString(),
          preview: 'Empty branch',
          parentId: undefined,
        }]);
        setCurrentBranch('main');
      }
    } catch (error) {
      console.warn('Backend connection failed, setting up default branch:', error);
      // Set up default state when backend is not available
      setBranches([{
        id: 'main',
        name: 'main',
        isActive: true,
        messageCount: 0,
        createdAt: new Date().toISOString(),
        lastActive: new Date().toISOString(),
        preview: 'Empty branch',
        parentId: undefined,
      }]);
      setCurrentBranch('main');
    }
  };

  const handleCreateBranch = async (branchName?: string, fromBranchId?: string) => {
    const name = branchName || newBranchName.trim() || `branch_${Date.now()}`;
    const parentId = fromBranchId || currentBranchId;

    setIsCreating(true);
    try {
      const response = await apiClient.createBranch('default', name, parentId);
      
      if (response.success && response.data && response.data.branch) {
        const branchData = response.data.branch;
        const newBranch: ConversationBranch = {
          id: branchData.id || `branch_${Date.now()}`,
          name: branchData.name || name,
          isActive: branchData.is_active || false,
          messageCount: branchData.message_count || 0,
          createdAt: branchData.created_at || new Date().toISOString(),
          lastActive: branchData.last_active || new Date().toISOString(),
          preview: branchData.message_count > 0 ? `${branchData.message_count} messages` : 'Empty branch',
          parentId: parentId,
        };
        
        addBranch(newBranch);
        if (!branchName) {
          setNewBranchName(''); // Only clear input if it came from list view
        }
        
        // Reload branches to get accurate data
        await loadBranches();
      } else {
        console.error('Failed to create branch:', response.message || 'Unknown error');
      }
    } catch (error) {
      console.error('Failed to create branch:', error);
    } finally {
      setIsCreating(false);
    }
  };

  const handleSwitchBranch = async (branchId: string) => {
    if (branchId === currentBranchId) return;

    try {
      const response = await apiClient.switchBranch('default', branchId);
      
      if (response.success && response.data) {
        setCurrentBranch(branchId);
        
        // Load branch conversation history from API response
        if (response.data.conversation) {
          const transformedMessages: Message[] = response.data.conversation.map((msg: any) => ({
            id: msg.id || `${msg.role}-${Date.now()}-${Math.random()}`,
            role: msg.role,
            content: msg.content,
            timestamp: new Date(msg.timestamp || Date.now()).getTime(),
            attachments: msg.attachments,
          }));
          setMessages(transformedMessages);
        } else {
          setMessages([]);
        }
        
        await loadBranches();
      }
    } catch (error) {
      console.error('Failed to switch branch:', error);
      // Still update UI even if conversation loading fails
      setCurrentBranch(branchId);
      setMessages([]);
    }
  };

  const handleDeleteBranch = async (branchId: string) => {
    if (branchId === 'main') {
      alert('Cannot delete main branch');
      return;
    }

    if (!confirm('Are you sure you want to delete this branch?')) {
      return;
    }

    try {
      const response = await apiClient.deleteBranch('default', branchId);
      
      if (response.success) {
        deleteBranch(branchId);
        
        // Switch to main if we deleted the current branch
        if (branchId === currentBranchId) {
          await handleSwitchBranch('main');
        }
        
        await loadBranches();
      }
    } catch (error) {
      console.error('Failed to delete branch:', error);
    }
  };

  // Handle branch renaming
  const handleRenameBranch = async (branchId: string, newName: string) => {
    try {
      // For now, simulate rename - in real implementation this would call API
      const updatedBranches = branches.map(branch =>
        branch.id === branchId ? { ...branch, name: newName } : branch
      );
      setBranches(updatedBranches);
      
      console.log(`Rename branch ${branchId} to ${newName}`); // TODO: Implement API call
    } catch (error) {
      console.error('Failed to rename branch:', error);
    }
  };

  // Handle branch merging
  const handleMergeBranch = async (sourceBranchId: string, targetBranchId: string, strategy: 'append' | 'interleave' | 'replace') => {
    try {
      console.log(`Merge ${sourceBranchId} into ${targetBranchId} using ${strategy}`); // TODO: Implement API call
      
      // For now, just archive the source branch (remove it)
      deleteBranch(sourceBranchId);
      
      // If we merged the current branch, switch to target
      if (sourceBranchId === currentBranchId) {
        await handleSwitchBranch(targetBranchId);
      }
      
      await loadBranches();
    } catch (error) {
      console.error('Failed to merge branch:', error);
    }
  };

  // Handle right-click context menu
  const handleContextMenu = (e: React.MouseEvent, branch: ConversationBranch) => {
    e.preventDefault();
    e.stopPropagation();
    
    setContextMenu({
      branch,
      position: { x: e.clientX, y: e.clientY }
    });
  };

  // Handle merge dialog
  const handleOpenMergeDialog = (sourceBranch: ConversationBranch, targetBranch: ConversationBranch) => {
    setMergeDialog({ sourceBranch, targetBranch });
  };

  // Get current branch object
  const currentBranch = branches.find(b => b.id === currentBranchId);

  return (
    <div className={`bg-card border-l border-border ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-foreground flex items-center gap-2">
            <GitBranch size={18} />
            Conversation Branches
          </h3>
          
          {/* View mode toggle */}
          <div className="flex bg-muted rounded p-1">
            <button
              onClick={() => setViewMode('list')}
              className={`p-1 rounded text-xs transition-colors ${
                viewMode === 'list' 
                  ? 'bg-card text-foreground shadow-sm' 
                  : 'text-muted-foreground hover:text-foreground'
              }`}
              title="List view"
            >
              <List size={14} />
            </button>
            <button
              onClick={() => setViewMode('tree')}
              className={`p-1 rounded text-xs transition-colors ${
                viewMode === 'tree' 
                  ? 'bg-card text-foreground shadow-sm' 
                  : 'text-muted-foreground hover:text-foreground'
              }`}
              title="Tree view"
            >
              <TreePine size={14} />
            </button>
            <button
              onClick={() => setViewMode('inheritance')}
              className={`p-1 rounded text-xs transition-colors ${
                viewMode === 'inheritance' 
                  ? 'bg-card text-foreground shadow-sm' 
                  : 'text-muted-foreground hover:text-foreground'
              }`}
              title="Inheritance view"
            >
              <Shuffle size={14} />
            </button>
          </div>
        </div>
      </div>

      {/* Main content area - List, Tree, or Inheritance view */}
      {viewMode === 'list' ? (
        /* Branch list view */
        <div className="flex-1 overflow-y-auto">
          {branches.map((branch) => (
            <div
              key={branch.id}
              className={`group flex items-center justify-between p-3 hover:bg-accent cursor-pointer border-b border-border ${
                branch.isActive ? 'bg-primary/10 border-primary/30' : ''
              }`}
              onClick={() => handleSwitchBranch(branch.id)}
              onContextMenu={(e) => handleContextMenu(e, branch)}
            >
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <div
                    className={`w-2 h-2 rounded-full ${
                      branch.isActive ? 'bg-primary' : 'bg-muted-foreground'
                    }`}
                  />
                  <span
                    className={`font-medium truncate ${
                      branch.isActive ? 'text-primary' : 'text-foreground'
                    }`}
                  >
                    {branch.name}
                  </span>
                </div>
                
                <div className="mt-1 text-xs text-muted-foreground">
                  {branch.messageCount} messages
                </div>
                
                {branch.preview && branch.preview !== 'Empty branch' && (
                  <div className="mt-1 text-xs text-muted-foreground/70 truncate">
                    {branch.preview}
                  </div>
                )}
              </div>

              {/* Action buttons */}
              <div className="flex items-center gap-1">
                {/* Context menu button */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleContextMenu(e, branch);
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-accent text-muted-foreground"
                  title="More options"
                >
                  <MoreVertical size={14} />
                </button>

                {/* Delete button */}
                {branch.id !== 'main' && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteBranch(branch.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-900/20 text-red-400 hover:text-red-300"
                    title="Delete branch"
                  >
                    <Trash2 size={14} />
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      ) : viewMode === 'tree' ? (
        /* Tree visualization view */
        <div className="flex-1">
          <BranchTreeVisualization
            onBranchSelect={handleSwitchBranch}
            onCreateBranch={(fromBranchId, name) => {
              // Create branch from the selected parent
              handleCreateBranch(name, fromBranchId);
            }}
            onDeleteBranch={handleDeleteBranch}
            className="h-full"
          />
        </div>
      ) : (
        /* Inheritance view */
        <div className="flex-1">
          {currentBranch ? (
            <BranchInheritanceView
              currentBranch={currentBranch}
              allBranches={branches}
              messages={messages}
              onSwitchBranch={handleSwitchBranch}
              className="h-full"
            />
          ) : (
            <div className="flex items-center justify-center h-full text-gray-500">
              <div className="text-center">
                <Shuffle size={32} className="mx-auto mb-2 opacity-50" />
                <div className="text-sm">No current branch selected</div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Create new branch - only show in list view */}
      {viewMode === 'list' && (
        <div className="p-4 border-t border-border">
          <div className="space-y-2">
            <input
              type="text"
              value={newBranchName}
              onChange={(e) => setNewBranchName(e.target.value)}
              placeholder="Branch name..."
              className="w-full px-3 py-2 text-sm border border-border bg-input text-input-foreground placeholder:text-muted-foreground rounded focus:outline-none focus:ring-2 focus:ring-primary"
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  handleCreateBranch();
                }
              }}
            />
            <button
              onClick={() => handleCreateBranch()}
              disabled={isCreating}
              className="w-full bg-primary hover:bg-primary/90 disabled:opacity-50 text-primary-foreground py-2 px-3 rounded text-sm font-medium transition-colors flex items-center justify-center gap-2"
            >
              <Plus size={16} />
              {isCreating ? 'Creating...' : 'New Branch'}
            </button>
          </div>
        </div>
      )}

      {/* Context Menu */}
      {contextMenu && (
        <BranchContextMenu
          branch={contextMenu.branch}
          isVisible={true}
          position={contextMenu.position}
          onClose={() => setContextMenu(null)}
          onSwitchTo={handleSwitchBranch}
          onCreateChild={(parentId, name) => handleCreateBranch(name, parentId)}
          onRename={handleRenameBranch}
          onDelete={handleDeleteBranch}
          onMergeBranch={(sourceBranchId, targetBranchId) => {
            const sourceBranch = branches.find(b => b.id === sourceBranchId);
            const targetBranch = branches.find(b => b.id === targetBranchId);
            if (sourceBranch && targetBranch) {
              handleOpenMergeDialog(sourceBranch, targetBranch);
            }
            setContextMenu(null);
          }}
          availableBranches={branches}
        />
      )}

      {/* Merge Dialog */}
      {mergeDialog && (
        <BranchMergeDialog
          isOpen={true}
          onClose={() => setMergeDialog(null)}
          sourceBranch={mergeDialog.sourceBranch}
          targetBranch={mergeDialog.targetBranch}
          sourceMessages={messages.filter(m => m.branchId === mergeDialog.sourceBranch.id)}
          targetMessages={messages.filter(m => m.branchId === mergeDialog.targetBranch.id)}
          onConfirmMerge={handleMergeBranch}
        />
      )}
    </div>
  );
}