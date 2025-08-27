/**
 * Visual representation of conversation inheritance between branches
 */

"use client";

import React from 'react';
import { ConversationBranch, Message } from '@/lib/types';
import { GitBranch, MessageSquare, Clock, ArrowDown, Shuffle } from 'lucide-react';

interface BranchInheritanceViewProps {
  currentBranch: ConversationBranch;
  allBranches: ConversationBranch[];
  messages: Message[];
  onSwitchBranch: (branchId: string) => void;
  className?: string;
}

interface InheritanceNode {
  branch: ConversationBranch;
  messages: Message[];
  isAncestor: boolean;
  isDescendant: boolean;
  isCurrent: boolean;
  depth: number;
}

export default function BranchInheritanceView({
  currentBranch,
  allBranches,
  messages,
  onSwitchBranch,
  className = '',
}: BranchInheritanceViewProps) {
  // Build inheritance chain
  const inheritanceChain = React.useMemo((): InheritanceNode[] => {
    const branchMap = new Map(allBranches.map(b => [b.id, b]));
    const messagesByBranch = new Map<string, Message[]>();
    
    // Group messages by branch
    messages.forEach(msg => {
      const branchId = msg.branchId || 'main';
      if (!messagesByBranch.has(branchId)) {
        messagesByBranch.set(branchId, []);
      }
      messagesByBranch.get(branchId)!.push(msg);
    });

    const nodes: InheritanceNode[] = [];

    // Get ancestry path (from root to current)
    const getAncestryPath = (branchId: string): string[] => {
      const path: string[] = [];
      let current = branchMap.get(branchId);
      
      while (current) {
        path.unshift(current.id);
        current = current.parentId ? branchMap.get(current.parentId) : null;
      }
      
      return path;
    };

    // Get all descendants
    const getDescendants = (branchId: string): string[] => {
      const descendants: string[] = [];
      const queue = [branchId];
      
      while (queue.length > 0) {
        const currentId = queue.shift()!;
        const children = allBranches.filter(b => b.parentId === currentId);
        
        children.forEach(child => {
          descendants.push(child.id);
          queue.push(child.id);
        });
      }
      
      return descendants;
    };

    const ancestryPath = getAncestryPath(currentBranch.id);
    const descendants = getDescendants(currentBranch.id);

    // Create nodes for ancestry path
    ancestryPath.forEach((branchId, index) => {
      const branch = branchMap.get(branchId);
      if (branch) {
        nodes.push({
          branch,
          messages: messagesByBranch.get(branchId) || [],
          isAncestor: index < ancestryPath.length - 1,
          isDescendant: false,
          isCurrent: branchId === currentBranch.id,
          depth: index,
        });
      }
    });

    // Add immediate children
    const immediateChildren = allBranches.filter(b => b.parentId === currentBranch.id);
    immediateChildren.forEach(child => {
      nodes.push({
        branch: child,
        messages: messagesByBranch.get(child.id) || [],
        isAncestor: false,
        isDescendant: true,
        isCurrent: false,
        depth: ancestryPath.length,
      });
    });

    return nodes;
  }, [currentBranch, allBranches, messages]);

  // Calculate inheritance statistics
  const stats = React.useMemo(() => {
    const ancestors = inheritanceChain.filter(n => n.isAncestor);
    const descendants = inheritanceChain.filter(n => n.isDescendant);
    
    const totalInheritedMessages = ancestors.reduce((sum, node) => 
      sum + node.messages.length, 0
    );
    
    const totalDescendantMessages = descendants.reduce((sum, node) => 
      sum + node.messages.length, 0
    );

    return {
      ancestorCount: ancestors.length,
      descendantCount: descendants.length,
      inheritedMessages: totalInheritedMessages,
      descendantMessages: totalDescendantMessages,
    };
  }, [inheritanceChain]);

  const renderNode = (node: InheritanceNode, index: number) => {
    const { branch, messages, isAncestor, isDescendant, isCurrent } = node;
    
    return (
      <div key={branch.id} className="relative">
        {/* Connection line */}
        {index > 0 && (
          <div className="absolute -top-4 left-4 w-0.5 h-4 bg-gray-300"></div>
        )}

        <div
          className={`
            flex items-start gap-3 p-3 rounded-lg border transition-colors cursor-pointer
            ${isCurrent 
              ? 'bg-blue-50 border-blue-200 ring-1 ring-blue-300' 
              : isAncestor 
                ? 'bg-green-50 border-green-200 hover:bg-green-100' 
                : isDescendant 
                  ? 'bg-purple-50 border-purple-200 hover:bg-purple-100'
                  : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
            }
          `}
          onClick={() => !isCurrent && onSwitchBranch(branch.id)}
        >
          {/* Branch indicator */}
          <div className="flex-shrink-0 mt-1">
            {isCurrent ? (
              <div className="w-2 h-2 bg-blue-500 rounded-full ring-2 ring-blue-200"></div>
            ) : isAncestor ? (
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            ) : (
              <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
            )}
          </div>

          {/* Branch content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between mb-1">
              <div className="font-medium text-sm text-gray-900 truncate">
                {branch.name}
                {isCurrent && <span className="ml-2 text-xs text-blue-600">(current)</span>}
              </div>
              <div className="text-xs text-gray-500">
                {new Date(branch.lastActive).toLocaleDateString()}
              </div>
            </div>

            <div className="flex items-center gap-3 text-xs text-gray-600 mb-2">
              <div className="flex items-center gap-1">
                <MessageSquare size={10} />
                <span>{messages.length} messages</span>
              </div>
              <div className="flex items-center gap-1">
                <Clock size={10} />
                <span>{new Date(branch.lastActive).toLocaleTimeString()}</span>
              </div>
            </div>

            {branch.preview && branch.preview !== 'Empty branch' && (
              <div className="text-xs text-gray-500 bg-white/50 rounded p-2 border border-gray-100">
                {branch.preview.length > 80 
                  ? `${branch.preview.substring(0, 80)}...` 
                  : branch.preview
                }
              </div>
            )}
          </div>

          {/* Type indicator */}
          <div className="flex-shrink-0">
            {isAncestor && (
              <div className="text-green-600 text-xs font-medium">Inherited</div>
            )}
            {isDescendant && (
              <div className="text-purple-600 text-xs font-medium">Child</div>
            )}
            {isCurrent && (
              <div className="text-blue-600 text-xs font-medium">Current</div>
            )}
          </div>
        </div>

        {/* Arrow between nodes */}
        {index < inheritanceChain.length - 1 && !isDescendant && (
          <div className="flex justify-center py-2">
            <ArrowDown size={16} className="text-gray-400" />
          </div>
        )}
      </div>
    );
  };

  return (
    <div className={`bg-white rounded-lg border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-100">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-gray-800 flex items-center gap-2">
            <Shuffle size={18} />
            Branch Inheritance
          </h3>
          <div className="text-xs text-gray-500">
            {inheritanceChain.length} related branches
          </div>
        </div>
      </div>

      {/* Statistics */}
      <div className="p-4 bg-gray-50 border-b border-gray-100">
        <div className="grid grid-cols-2 gap-4 text-center">
          <div>
            <div className="text-lg font-bold text-green-600">
              {stats.inheritedMessages}
            </div>
            <div className="text-xs text-gray-600">
              Inherited Messages
            </div>
          </div>
          <div>
            <div className="text-lg font-bold text-purple-600">
              {stats.descendantMessages}
            </div>
            <div className="text-xs text-gray-600">
              Child Messages  
            </div>
          </div>
        </div>
      </div>

      {/* Inheritance chain */}
      <div className="p-4">
        {inheritanceChain.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            <GitBranch size={32} className="mx-auto mb-2 opacity-50" />
            <div className="text-sm">No related branches found</div>
          </div>
        ) : (
          <div className="space-y-0">
            {inheritanceChain
              .filter(n => n.isAncestor || n.isCurrent)
              .map((node, index) => renderNode(node, index))
            }
            
            {/* Children section */}
            {inheritanceChain.some(n => n.isDescendant) && (
              <>
                <div className="flex justify-center py-3">
                  <div className="flex items-center gap-2 text-xs text-gray-500 bg-gray-100 px-3 py-1 rounded-full">
                    <GitBranch size={12} />
                    Child Branches
                  </div>
                </div>
                <div className="grid gap-2 pl-6">
                  {inheritanceChain
                    .filter(n => n.isDescendant)
                    .map((node, index) => (
                      <div key={node.branch.id} className="relative">
                        {/* Connection line to parent */}
                        <div className="absolute -left-6 top-4 w-6 h-0.5 bg-gray-300"></div>
                        <div className="absolute -left-6 top-0 w-0.5 h-4 bg-gray-300"></div>
                        {renderNode(node, index)}
                      </div>
                    ))
                  }
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="p-4 bg-gray-50 border-t border-gray-100">
        <div className="text-xs font-medium text-gray-700 mb-2">Legend:</div>
        <div className="grid grid-cols-3 gap-3 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <span className="text-gray-600">Inherited</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
            <span className="text-gray-600">Current</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
            <span className="text-gray-600">Children</span>
          </div>
        </div>
      </div>
    </div>
  );
}