import { useState } from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { ChevronDown, ChevronUp, Eye, EyeOff, Copy, Check } from 'lucide-react'
import { SampleConversation } from '@/components/results/SampleConversation'
import type { ResultGroup, Conversation, AnalysisResult } from '@/types/eval'

interface GroupedResultsViewProps {
  /** Title for the section */
  title: string
  /** Groups to display */
  groups: ResultGroup[]
  /** All conversations for drill-down */
  conversations: Conversation[]
  /** Analysis results for each conversation (optional, for SampleConversation) */
  analysisResults?: AnalysisResult[]
  /** Message when no groups */
  emptyMessage?: string
  /** Show percentages next to counts */
  showPercentages?: boolean
  /** Sort order */
  sortBy?: 'count' | 'name'
  /** For duplicates: mark first index as "keep", rest as "duplicate" */
  markFirstAsKeep?: boolean
}

/**
 * Reusable component for displaying grouped dataset-level results.
 * 
 * Used for:
 * - Duplicate groups (deduplication analyzer)
 * - Language distribution (future)
 * - Label/category distribution (future)
 * - System prompt groups (future)
 */
export function GroupedResultsView({
  title,
  groups,
  conversations,
  analysisResults,
  emptyMessage = 'No groups found',
  showPercentages = true,
  sortBy = 'count',
  markFirstAsKeep = false,
}: GroupedResultsViewProps) {
  const [isOpen, setIsOpen] = useState(true)
  const [expandedGroups, setExpandedGroups] = useState<Set<number>>(new Set())
  const [copiedIndices, setCopiedIndices] = useState<number | null>(null)

  // Sort groups
  const sortedGroups = [...groups].sort((a, b) => {
    if (sortBy === 'count') {
      return b.count - a.count // Largest first
    }
    return a.name.localeCompare(b.name)
  })

  const totalItems = groups.reduce((sum, g) => sum + g.count, 0)

  const toggleGroup = (index: number) => {
    setExpandedGroups(prev => {
      const next = new Set(prev)
      if (next.has(index)) {
        next.delete(index)
      } else {
        next.add(index)
      }
      return next
    })
  }

  const copyIndices = async (indices: number[], groupIndex: number) => {
    try {
      await navigator.clipboard.writeText(indices.join(', '))
      setCopiedIndices(groupIndex)
      setTimeout(() => setCopiedIndices(null), 2000)
    } catch (e) {
      console.error('Failed to copy indices:', e)
    }
  }

  if (groups.length === 0) {
    return (
      <div className="text-center py-4 text-muted-foreground text-sm">
        {emptyMessage}
      </div>
    )
  }

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen} className="mt-6">
      <CollapsibleTrigger asChild>
        <Button 
          variant="ghost" 
          className="w-full justify-between p-4 h-auto hover:bg-muted/50"
        >
          <div className="flex items-center gap-2">
            <span className="font-medium">{title}</span>
            <Badge variant="secondary" className="text-xs">
              {groups.length} groups
            </Badge>
            {totalItems > 0 && (
              <span className="text-sm text-muted-foreground">
                ({totalItems} items)
              </span>
            )}
          </div>
          {isOpen ? (
            <ChevronUp className="h-4 w-4" />
          ) : (
            <ChevronDown className="h-4 w-4" />
          )}
        </Button>
      </CollapsibleTrigger>

      <CollapsibleContent>
        <div className="space-y-3 px-2 pb-4">
          {sortedGroups.map((group, groupIndex) => {
            const isExpanded = expandedGroups.has(groupIndex)
            const percentage = group.percentage ?? 
              (totalItems > 0 ? (group.count / totalItems) * 100 : 0)

            return (
              <Card key={groupIndex} className="overflow-hidden">
                <div 
                  className="p-4 cursor-pointer hover:bg-muted/30 transition-colors"
                  onClick={() => toggleGroup(groupIndex)}
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 min-w-0">
                      {/* Group header */}
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="font-medium">{group.name}</span>
                        <Badge variant="outline" className="text-xs">
                          {group.count} {group.count === 1 ? 'item' : 'items'}
                        </Badge>
                        {showPercentages && (
                          <span className="text-xs text-muted-foreground">
                            ({percentage.toFixed(1)}%)
                          </span>
                        )}
                        {/* Show metadata badges */}
                        {group.metadata?.similarity !== undefined && (
                          <Badge variant="secondary" className="text-xs">
                            {(group.metadata.similarity as number) === 1.0 
                              ? 'Exact match' 
                              : `${((group.metadata.similarity as number) * 100).toFixed(0)}% similar`}
                          </Badge>
                        )}
                      </div>

                      {/* Sample text preview */}
                      {group.sample_text && (
                        <p className="text-sm text-muted-foreground mt-1 line-clamp-2">
                          "{group.sample_text}"
                        </p>
                      )}

                      {/* Indices (collapsed view) */}
                      {!isExpanded && group.indices.length <= 10 && (
                        <p className="text-xs text-muted-foreground mt-1 font-mono">
                          Indices: {group.indices.join(', ')}
                        </p>
                      )}
                    </div>

                    {/* Expand button */}
                    <Button variant="ghost" size="sm" className="shrink-0">
                      {isExpanded ? (
                        <>
                          <EyeOff className="h-4 w-4 mr-1" />
                          Hide
                        </>
                      ) : (
                        <>
                          <Eye className="h-4 w-4 mr-1" />
                          View
                        </>
                      )}
                    </Button>
                  </div>
                </div>

                {/* Expanded content */}
                {isExpanded && (
                  <CardContent className="pt-0 pb-4 border-t bg-muted/20">
                    {/* Actions bar */}
                    <div className="flex items-center justify-between py-2 mb-3">
                      <span className="text-sm text-muted-foreground">
                        Showing {group.indices.length} conversations
                      </span>
                      <Button
                        variant="outline"
                        size="sm"
                        className="text-xs"
                        onClick={(e) => {
                          e.stopPropagation()
                          copyIndices(group.indices, groupIndex)
                        }}
                      >
                        {copiedIndices === groupIndex ? (
                          <>
                            <Check className="h-3 w-3 mr-1" />
                            Copied!
                          </>
                        ) : (
                          <>
                            <Copy className="h-3 w-3 mr-1" />
                            Copy Indices
                          </>
                        )}
                      </Button>
                    </div>

                    {/* Conversations list */}
                    <div className="space-y-2">
                      {group.indices.map((convIndex, i) => {
                        const conversation = conversations[convIndex]
                        if (!conversation) {
                          return (
                            <div 
                              key={convIndex} 
                              className="p-3 rounded border border-dashed text-sm text-muted-foreground"
                            >
                              Conversation at index {convIndex} not found
                            </div>
                          )
                        }

                        // Get analysis result for this conversation if available
                        const analysisResult = analysisResults?.[convIndex] || {}

                        return (
                          <div key={convIndex} className="relative">
                            {/* Keep/Duplicate badge for deduplication */}
                            {markFirstAsKeep && (
                              <Badge 
                                variant={i === 0 ? 'default' : 'destructive'}
                                className="absolute top-3 right-3 z-10 text-xs"
                              >
                                {i === 0 ? 'Keep' : 'Duplicate'}
                              </Badge>
                            )}
                            <SampleConversation
                              index={convIndex}
                              conversation={conversation}
                              analysisResult={analysisResult}
                              keyPrefix={`group-${groupIndex}-conv-${convIndex}`}
                            />
                          </div>
                        )
                      })}
                    </div>
                  </CardContent>
                )}
              </Card>
            )
          })}
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}
