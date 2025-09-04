/**
 * Advanced D3.js tree visualization for conversation branches
 */

"use client";

import React from 'react';
import Tree from 'react-d3-tree';
import { useChatStore } from '@/lib/store';
import { ConversationBranch } from '@/lib/types';
import { GitBranch, MessageCircle, Calendar, Eye } from 'lucide-react';

interface TreeNode {
  name: string;
  attributes?: {
    id: string;
    messageCount: number;
    lastActive: string;
    preview: string;
    isActive: boolean;
  };
  children?: TreeNode[];
}

interface BranchTreeVisualizationProps {
  className?: string;
  onBranchSelect?: (branchId: string) => void;
  onCreateBranch?: (fromBranchId: string, name: string) => void;
  onDeleteBranch?: (branchId: string) => void;
}

export default function BranchTreeVisualization({
  className = '',
  onBranchSelect,
  onCreateBranch,
  onDeleteBranch,
}: BranchTreeVisualizationProps) {
  const { currentBranchId, getBranches } = useChatStore();
  // Get branches using the selector
  const branches = getBranches();
  const [selectedNode, setSelectedNode] = React.useState<string | null>(null);
  const [treeTranslate, setTreeTranslate] = React.useState({ x: 0, y: 0 });
  const [isCreatingBranch, setIsCreatingBranch] = React.useState(false);
  const [newBranchName, setNewBranchName] = React.useState('');
  const treeContainerRef = React.useRef<HTMLDivElement>(null);

  // Initialize tree position
  React.useEffect(() => {
    if (treeContainerRef.current) {
      const { offsetWidth, offsetHeight } = treeContainerRef.current;
      setTreeTranslate({
        x: offsetWidth / 2,
        y: offsetHeight / 4,
      });
    }
  }, []);

  // Convert branches to D3 tree format
  const buildTreeData = React.useMemo((): TreeNode => {
    const branchMap = new Map<string, ConversationBranch>();
    const childrenMap = new Map<string, ConversationBranch[]>();

    // Index branches
    branches.forEach((branch: ConversationBranch) => {
      branchMap.set(branch.id, branch);
      const parentId = branch.parentId || 'root';
      if (!childrenMap.has(parentId)) {
        childrenMap.set(parentId, []);
      }
      childrenMap.get(parentId)!.push(branch);
    });

    // Build tree recursively
    const buildNode = (branchId: string): TreeNode => {
      const branch = branchMap.get(branchId);
      if (!branch) {
        // Root node
        const children = childrenMap.get('root') || [];
        return {
          name: 'Conversation Tree',
          children: children.map((child) => buildNode(child.id)),
        };
      }

      const children = childrenMap.get(branchId) || [];
      const lastActiveDate = new Date(branch.lastActive).toLocaleDateString();
      
      return {
        name: branch.name,
        attributes: {
          id: branch.id,
          messageCount: branch.messageCount,
          lastActive: lastActiveDate,
          preview: branch.preview || 'No preview',
          isActive: branch.isActive,
        },
        children: children.map((child) => buildNode(child.id)),
      };
    };

    return buildNode('root');
  }, [branches]);

  // Handle branch creation
  const handleCreateBranch = () => {
    if (!selectedNode || !newBranchName.trim() || !onCreateBranch) return;
    
    setIsCreatingBranch(true);
    onCreateBranch(selectedNode, newBranchName.trim());
    setNewBranchName('');
    setIsCreatingBranch(false);
  };

  // Handle branch deletion
  const handleDeleteBranch = (branchId: string) => {
    if (branchId === 'main' || !onDeleteBranch) return;
    
    if (confirm('Are you sure you want to delete this branch?')) {
      onDeleteBranch(branchId);
      if (selectedNode === branchId) {
        setSelectedNode(null);
      }
    }
  };

  // Custom node rendering
  const renderCustomNode = ({ nodeDatum, toggleNode }: any) => {
    const isRoot = !nodeDatum.attributes;
    const isActive = nodeDatum.attributes?.isActive;
    const isSelected = selectedNode === nodeDatum.attributes?.id;

    return (
      <g>
        {/* Node circle */}
        <circle
          r={isRoot ? 8 : 6}
          fill={
            isRoot
              ? '#6366f1'
              : isActive
              ? '#22c55e'
              : isSelected
              ? '#3b82f6'
              : '#94a3b8'
          }
          stroke={isSelected ? '#1d4ed8' : isActive ? '#16a34a' : '#64748b'}
          strokeWidth={isSelected || isActive ? 2 : 1}
          onClick={() => {
            if (!isRoot && nodeDatum.attributes) {
              setSelectedNode(nodeDatum.attributes.id);
              onBranchSelect?.(nodeDatum.attributes.id);
            }
          }}
          style={{ cursor: isRoot ? 'default' : 'pointer' }}
        />

        {/* Node label */}
        <text
          dy=".31em"
          x={isRoot ? 12 : 10}
          textAnchor="start"
          fontSize="12px"
          fill="#374151"
          fontWeight={isActive ? 'bold' : 'normal'}
          style={{ cursor: isRoot ? 'default' : 'pointer' }}
          onClick={() => {
            if (!isRoot && nodeDatum.attributes) {
              setSelectedNode(nodeDatum.attributes.id);
              onBranchSelect?.(nodeDatum.attributes.id);
            }
          }}
        >
          {nodeDatum.name}
        </text>

        {/* Message count indicator */}
        {!isRoot && nodeDatum.attributes && nodeDatum.attributes.messageCount > 0 && (
          <text
            dy="-.5em"
            x="0"
            textAnchor="middle"
            fontSize="10px"
            fill="#6b7280"
            fontWeight="bold"
          >
            {nodeDatum.attributes.messageCount}
          </text>
        )}

        {/* Active indicator */}
        {isActive && (
          <circle
            r="2"
            cx="8"
            cy="-8"
            fill="#22c55e"
            stroke="#ffffff"
            strokeWidth="1"
          />
        )}
      </g>
    );
  };

  // Node details panel
  const selectedBranch = selectedNode
    ? branches.find((b: ConversationBranch) => b.id === selectedNode)
    : null;

  return (
    <div className={`flex flex-col h-full bg-white ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-100">
        <h3 className="font-semibold text-gray-800 flex items-center gap-2">
          <GitBranch size={18} />
          Branch Tree Visualization
        </h3>
        <p className="text-xs text-gray-500 mt-1">
          Click on branches to switch conversations
        </p>
      </div>

      {/* Tree visualization */}
      <div className="flex-1 relative" ref={treeContainerRef}>
        {treeTranslate.x > 0 && (
          <Tree
            data={buildTreeData}
            translate={treeTranslate}
            orientation="vertical"
            pathFunc="step"
            separation={{ siblings: 1.5, nonSiblings: 2 }}
            nodeSize={{ x: 120, y: 80 }}
            renderCustomNodeElement={renderCustomNode}
            enableLegacyTransitions={true}
            collapsible={false}
            zoom={0.8}
            scaleExtent={{ min: 0.3, max: 3 }}
          />
        )}

        {/* Legend */}
        <div className="absolute top-2 left-2 bg-white/90 backdrop-blur-sm border rounded-lg p-3 text-xs">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
              <span>Active Branch</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-blue-500"></div>
              <span>Selected Branch</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-gray-400"></div>
              <span>Inactive Branch</span>
            </div>
          </div>
        </div>
      </div>

      {/* Selected node details */}
      {selectedBranch && (
        <div className="border-t border-gray-100 p-4 bg-gray-50">
          <h4 className="font-medium text-gray-800 mb-3 flex items-center gap-2">
            <Eye size={16} />
            Branch Details
          </h4>
          
          <div className="space-y-3">
            <div>
              <div className="text-sm font-medium text-gray-700">
                {selectedBranch.name}
              </div>
              <div className="text-xs text-gray-500">
                ID: {selectedBranch.id}
              </div>
            </div>

            <div className="flex items-center gap-4 text-xs text-gray-600">
              <div className="flex items-center gap-1">
                <MessageCircle size={12} />
                <span>{selectedBranch.messageCount} messages</span>
              </div>
              <div className="flex items-center gap-1">
                <Calendar size={12} />
                <span>{new Date(selectedBranch.lastActive).toLocaleDateString()}</span>
              </div>
            </div>

            {selectedBranch.preview && selectedBranch.preview !== 'Empty branch' && (
              <div>
                <div className="text-xs font-medium text-gray-700 mb-1">Preview:</div>
                <div className="text-xs text-gray-600 bg-white rounded p-2 border">
                  {selectedBranch.preview.length > 100
                    ? `${selectedBranch.preview.substring(0, 100)}...`
                    : selectedBranch.preview
                  }
                </div>
              </div>
            )}

            <div className="space-y-2">
              {selectedBranch.id !== currentBranchId && (
                <button
                  onClick={() => onBranchSelect?.(selectedBranch.id)}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-3 rounded text-sm font-medium transition-colors"
                >
                  Switch to Branch
                </button>
              )}

              {/* Create branch from selected */}
              {onCreateBranch && !isCreatingBranch && (
                <div className="space-y-2">
                  <input
                    type="text"
                    value={newBranchName}
                    onChange={(e) => setNewBranchName(e.target.value)}
                    placeholder="New branch name..."
                    className="w-full px-2 py-1 text-xs border rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        handleCreateBranch();
                      }
                    }}
                  />
                  <button
                    onClick={handleCreateBranch}
                    disabled={!newBranchName.trim()}
                    className="w-full bg-green-600 hover:bg-green-700 disabled:opacity-50 text-white py-1 px-3 rounded text-xs font-medium transition-colors"
                  >
                    Create Child Branch
                  </button>
                </div>
              )}

              {/* Delete branch */}
              {selectedBranch.id !== 'main' && onDeleteBranch && (
                <button
                  onClick={() => handleDeleteBranch(selectedBranch.id)}
                  className="w-full bg-red-600 hover:bg-red-700 text-white py-1 px-3 rounded text-xs font-medium transition-colors"
                >
                  Delete Branch
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}