/**
 * Context menu for branch operations
 */

"use client";

import React from 'react';
import { MoreVertical, GitBranch, Copy, Trash2, Edit2, ArrowRight } from 'lucide-react';
import { ConversationBranch } from '@/lib/types';

interface BranchContextMenuProps {
  branch: ConversationBranch;
  isVisible: boolean;
  position: { x: number; y: number };
  onClose: () => void;
  onSwitchTo: (branchId: string) => void;
  onCreateChild: (parentBranchId: string, name: string) => void;
  onRename: (branchId: string, newName: string) => void;
  onDelete: (branchId: string) => void;
  onMergeBranch?: (sourceBranchId: string, targetBranchId: string) => void;
  availableBranches: ConversationBranch[];
}

export default function BranchContextMenu({
  branch,
  isVisible,
  position,
  onClose,
  onSwitchTo,
  onCreateChild,
  onRename,
  onDelete,
  onMergeBranch,
  availableBranches,
}: BranchContextMenuProps) {
  const [showRename, setShowRename] = React.useState(false);
  const [showCreateChild, setShowCreateChild] = React.useState(false);
  const [showMergeOptions, setShowMergeOptions] = React.useState(false);
  const [newName, setNewName] = React.useState('');
  const menuRef = React.useRef<HTMLDivElement>(null);

  // Handle clicks outside menu
  React.useEffect(() => {
    if (!isVisible) return;

    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        onClose();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isVisible, onClose]);

  // Reset states when menu visibility changes
  React.useEffect(() => {
    if (!isVisible) {
      setShowRename(false);
      setShowCreateChild(false);
      setShowMergeOptions(false);
      setNewName('');
    }
  }, [isVisible]);

  if (!isVisible) return null;

  const handleRename = (e: React.FormEvent) => {
    e.preventDefault();
    if (newName.trim()) {
      onRename(branch.id, newName.trim());
      onClose();
    }
  };

  const handleCreateChild = (e: React.FormEvent) => {
    e.preventDefault();
    if (newName.trim()) {
      onCreateChild(branch.id, newName.trim());
      onClose();
    }
  };

  const mergeCandidates = availableBranches.filter(
    b => b.id !== branch.id && b.parentId !== branch.id
  );

  return (
    <div
      ref={menuRef}
      className="fixed bg-white border border-gray-200 rounded-lg shadow-lg z-50 py-1 min-w-48"
      style={{
        left: position.x,
        top: position.y,
      }}
    >
      {/* Switch to branch */}
      {!branch.isActive && (
        <button
          onClick={() => {
            onSwitchTo(branch.id);
            onClose();
          }}
          className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 flex items-center gap-2"
        >
          <ArrowRight size={14} />
          Switch to Branch
        </button>
      )}

      {/* Rename branch */}
      {!showRename ? (
        <button
          onClick={() => {
            setShowRename(true);
            setNewName(branch.name);
          }}
          className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 flex items-center gap-2"
        >
          <Edit2 size={14} />
          Rename Branch
        </button>
      ) : (
        <form onSubmit={handleRename} className="px-3 py-2">
          <input
            type="text"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            className="w-full text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
            placeholder="Branch name..."
            autoFocus
            onBlur={() => setShowRename(false)}
          />
        </form>
      )}

      {/* Create child branch */}
      {!showCreateChild ? (
        <button
          onClick={() => {
            setShowCreateChild(true);
            setNewName('');
          }}
          className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 flex items-center gap-2"
        >
          <GitBranch size={14} />
          Create Child Branch
        </button>
      ) : (
        <form onSubmit={handleCreateChild} className="px-3 py-2">
          <input
            type="text"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            className="w-full text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
            placeholder="New branch name..."
            autoFocus
            onBlur={() => setShowCreateChild(false)}
          />
        </form>
      )}

      {/* Merge branch options */}
      {onMergeBranch && mergeCandidates.length > 0 && (
        <>
          {!showMergeOptions ? (
            <button
              onClick={() => setShowMergeOptions(true)}
              className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 flex items-center gap-2"
            >
              <Copy size={14} />
              Merge Into...
            </button>
          ) : (
            <div className="border-t border-gray-100">
              <div className="px-3 py-1 text-xs font-medium text-gray-500">
                Merge into:
              </div>
              {mergeCandidates.slice(0, 5).map((candidate) => (
                <button
                  key={candidate.id}
                  onClick={() => {
                    onMergeBranch(branch.id, candidate.id);
                    onClose();
                  }}
                  className="w-full text-left px-5 py-1 text-xs text-gray-600 hover:bg-gray-50"
                >
                  {candidate.name}
                </button>
              ))}
              {mergeCandidates.length > 5 && (
                <div className="px-5 py-1 text-xs text-gray-400">
                  +{mergeCandidates.length - 5} more...
                </div>
              )}
            </div>
          )}
        </>
      )}

      {/* Delete branch */}
      {branch.id !== 'main' && (
        <>
          <div className="border-t border-gray-100 my-1" />
          <button
            onClick={() => {
              if (confirm(`Delete branch "${branch.name}"? This cannot be undone.`)) {
                onDelete(branch.id);
                onClose();
              }
            }}
            className="w-full text-left px-3 py-2 text-sm text-red-600 hover:bg-red-50 flex items-center gap-2"
          >
            <Trash2 size={14} />
            Delete Branch
          </button>
        </>
      )}
    </div>
  );
}