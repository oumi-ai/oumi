/**
 * Dialog for merging branches with conflict resolution
 */

"use client";

import React from 'react';
import { X, AlertTriangle, Check, GitMerge, MessageSquare } from 'lucide-react';
import { ConversationBranch, Message } from '@/lib/types';

interface BranchMergeDialogProps {
  isOpen: boolean;
  onClose: () => void;
  sourceBranch: ConversationBranch | null;
  targetBranch: ConversationBranch | null;
  sourceMessages: Message[];
  targetMessages: Message[];
  onConfirmMerge: (
    sourceBranchId: string,
    targetBranchId: string,
    mergeStrategy: 'append' | 'interleave' | 'replace'
  ) => void;
}

type MergeStrategy = 'append' | 'interleave' | 'replace';

export default function BranchMergeDialog({
  isOpen,
  onClose,
  sourceBranch,
  targetBranch,
  sourceMessages,
  targetMessages,
  onConfirmMerge,
}: BranchMergeDialogProps) {
  const [mergeStrategy, setMergeStrategy] = React.useState<MergeStrategy>('append');
  const [isProcessing, setIsProcessing] = React.useState(false);

  // Reset state when dialog opens/closes
  React.useEffect(() => {
    if (isOpen) {
      setMergeStrategy('append');
      setIsProcessing(false);
    }
  }, [isOpen]);

  if (!isOpen || !sourceBranch || !targetBranch) {
    return null;
  }

  // Analyze merge conflicts and preview
  const mergeAnalysis = React.useMemo(() => {
    const conflicts: string[] = [];
    const warnings: string[] = [];

    // Check for potential conflicts
    if (sourceMessages.length === 0) {
      warnings.push('Source branch has no messages to merge');
    }

    if (targetMessages.length === 0) {
      warnings.push('Target branch is empty');
    }

    // Check for overlapping content
    const sourceContent = sourceMessages.map(m => m.content.toLowerCase()).join(' ');
    const targetContent = targetMessages.map(m => m.content.toLowerCase()).join(' ');
    
    if (sourceContent.length > 0 && targetContent.length > 0) {
      // Simple overlap detection (could be more sophisticated)
      const sourceWords = new Set(sourceContent.split(/\s+/));
      const targetWords = new Set(targetContent.split(/\s+/));
      const intersection = new Set([...sourceWords].filter(x => targetWords.has(x)));
      
      if (intersection.size > Math.min(sourceWords.size, targetWords.size) * 0.5) {
        warnings.push('High content similarity detected between branches');
      }
    }

    // Check for timing conflicts
    if (sourceMessages.length > 0 && targetMessages.length > 0) {
      const sourceLatest = Math.max(...sourceMessages.map(m => new Date(m.timestamp).getTime()));
      const targetLatest = Math.max(...targetMessages.map(m => new Date(m.timestamp).getTime()));
      
      if (Math.abs(sourceLatest - targetLatest) < 60000) { // Within 1 minute
        conflicts.push('Recent activity detected in both branches');
      }
    }

    return { conflicts, warnings };
  }, [sourceMessages, targetMessages]);

  const handleMerge = async () => {
    if (!sourceBranch || !targetBranch) return;
    
    setIsProcessing(true);
    
    try {
      onConfirmMerge(sourceBranch.id, targetBranch.id, mergeStrategy);
      onClose();
    } catch (error) {
      console.error('Merge failed:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const getMergePreview = (): string => {
    switch (mergeStrategy) {
      case 'append':
        return `All ${sourceMessages.length} messages from "${sourceBranch.name}" will be added to the end of "${targetBranch.name}".`;
      case 'interleave':
        return `Messages from both branches will be merged by timestamp, creating a unified conversation timeline.`;
      case 'replace':
        return `All messages in "${targetBranch.name}" will be replaced with messages from "${sourceBranch.name}".`;
      default:
        return '';
    }
  };

  const getResultingMessageCount = (): number => {
    switch (mergeStrategy) {
      case 'append':
        return targetMessages.length + sourceMessages.length;
      case 'interleave':
        return Math.max(targetMessages.length, sourceMessages.length) + 
               Math.min(targetMessages.length, sourceMessages.length);
      case 'replace':
        return sourceMessages.length;
      default:
        return 0;
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl m-4">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center gap-3">
            <GitMerge size={24} className="text-blue-600" />
            <div>
              <h2 className="text-lg font-semibold text-gray-900">
                Merge Branches
              </h2>
              <p className="text-sm text-gray-600">
                Merge "{sourceBranch.name}" into "{targetBranch.name}"
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Branch summary */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-blue-50 rounded-lg p-4">
              <h3 className="font-medium text-blue-900 mb-2">Source Branch</h3>
              <div className="text-sm text-blue-700">
                <div className="font-medium">{sourceBranch.name}</div>
                <div className="flex items-center gap-2 mt-1">
                  <MessageSquare size={12} />
                  <span>{sourceMessages.length} messages</span>
                </div>
              </div>
            </div>
            
            <div className="bg-green-50 rounded-lg p-4">
              <h3 className="font-medium text-green-900 mb-2">Target Branch</h3>
              <div className="text-sm text-green-700">
                <div className="font-medium">{targetBranch.name}</div>
                <div className="flex items-center gap-2 mt-1">
                  <MessageSquare size={12} />
                  <span>{targetMessages.length} messages</span>
                </div>
              </div>
            </div>
          </div>

          {/* Conflicts and warnings */}
          {(mergeAnalysis.conflicts.length > 0 || mergeAnalysis.warnings.length > 0) && (
            <div className="space-y-3">
              {mergeAnalysis.conflicts.map((conflict, index) => (
                <div key={index} className="flex items-start gap-2 p-3 bg-red-50 border border-red-200 rounded-lg">
                  <AlertTriangle size={16} className="text-red-600 flex-shrink-0 mt-0.5" />
                  <div className="text-sm text-red-800">{conflict}</div>
                </div>
              ))}
              
              {mergeAnalysis.warnings.map((warning, index) => (
                <div key={index} className="flex items-start gap-2 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                  <AlertTriangle size={16} className="text-yellow-600 flex-shrink-0 mt-0.5" />
                  <div className="text-sm text-yellow-800">{warning}</div>
                </div>
              ))}
            </div>
          )}

          {/* Merge strategy selection */}
          <div>
            <h3 className="font-medium text-gray-900 mb-3">Merge Strategy</h3>
            <div className="space-y-3">
              <label className="flex items-start gap-3 p-3 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer">
                <input
                  type="radio"
                  name="mergeStrategy"
                  value="append"
                  checked={mergeStrategy === 'append'}
                  onChange={(e) => setMergeStrategy(e.target.value as MergeStrategy)}
                  className="mt-0.5"
                />
                <div className="flex-1">
                  <div className="font-medium text-sm text-gray-900">
                    Append Messages
                  </div>
                  <div className="text-xs text-gray-600 mt-1">
                    Add source messages to the end of target branch (recommended)
                  </div>
                </div>
              </label>

              <label className="flex items-start gap-3 p-3 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer">
                <input
                  type="radio"
                  name="mergeStrategy"
                  value="interleave"
                  checked={mergeStrategy === 'interleave'}
                  onChange={(e) => setMergeStrategy(e.target.value as MergeStrategy)}
                  className="mt-0.5"
                />
                <div className="flex-1">
                  <div className="font-medium text-sm text-gray-900">
                    Interleave by Timestamp
                  </div>
                  <div className="text-xs text-gray-600 mt-1">
                    Merge messages chronologically based on timestamps
                  </div>
                </div>
              </label>

              <label className="flex items-start gap-3 p-3 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer">
                <input
                  type="radio"
                  name="mergeStrategy"
                  value="replace"
                  checked={mergeStrategy === 'replace'}
                  onChange={(e) => setMergeStrategy(e.target.value as MergeStrategy)}
                  className="mt-0.5"
                />
                <div className="flex-1">
                  <div className="font-medium text-sm text-gray-900">
                    Replace Target Messages
                  </div>
                  <div className="text-xs text-gray-600 mt-1">
                    Replace all target messages with source messages (destructive)
                  </div>
                </div>
              </label>
            </div>
          </div>

          {/* Merge preview */}
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="font-medium text-gray-900 mb-2">Merge Preview</h3>
            <p className="text-sm text-gray-700 mb-3">
              {getMergePreview()}
            </p>
            <div className="flex items-center justify-between text-xs text-gray-600">
              <span>Resulting message count: {getResultingMessageCount()}</span>
              <span>Source branch will be archived after merge</span>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center justify-between p-6 bg-gray-50 border-t border-gray-200">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
          >
            Cancel
          </button>
          
          <div className="flex items-center gap-3">
            {mergeAnalysis.conflicts.length > 0 && (
              <div className="text-xs text-red-600 flex items-center gap-1">
                <AlertTriangle size={12} />
                {mergeAnalysis.conflicts.length} conflict(s)
              </div>
            )}
            
            <button
              onClick={handleMerge}
              disabled={isProcessing || mergeAnalysis.conflicts.length > 0}
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
            >
              {isProcessing ? (
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                <Check size={16} />
              )}
              {isProcessing ? 'Merging...' : 'Confirm Merge'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}