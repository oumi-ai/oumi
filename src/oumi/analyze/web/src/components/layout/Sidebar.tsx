import { useState } from 'react'
import { Search, CheckCircle, XCircle, Clock, BarChart3, Plus } from 'lucide-react'
import { cn, formatRelativeTime } from '@/lib/utils'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import type { EvalMetadata } from '@/types/eval'

interface SidebarProps {
  evals: EvalMetadata[]
  selectedId: string | null
  onSelect: (id: string) => void
  isLoading: boolean
  onNewAnalysis?: () => void
}

export function Sidebar({ evals, selectedId, onSelect, isLoading, onNewAnalysis }: SidebarProps) {
  const [searchQuery, setSearchQuery] = useState('')

  const filteredEvals = evals.filter(
    (e) =>
      e.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      e.id.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const selectedEval = evals.find((e) => e.id === selectedId)

  return (
    <div className="w-80 border-r bg-muted/30 flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <BarChart3 className="h-6 w-6 text-primary" />
            <h1 className="text-xl font-semibold">Oumi Analyze</h1>
          </div>
          {onNewAnalysis && (
            <Button variant="outline" size="sm" onClick={onNewAnalysis}>
              <Plus className="h-4 w-4" />
            </Button>
          )}
        </div>
        
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search evaluations..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-4 py-2 text-sm bg-background border rounded-md focus:outline-none focus:ring-2 focus:ring-ring"
          />
        </div>
      </div>

      {/* Eval List */}
      <ScrollArea className="flex-1">
        <div className="p-2">
          {isLoading ? (
            <div className="text-center text-muted-foreground py-8">
              Loading evaluations...
            </div>
          ) : filteredEvals.length === 0 ? (
            <div className="text-center text-muted-foreground py-8">
              {searchQuery ? 'No matching evaluations' : 'No evaluations found'}
            </div>
          ) : (
            <div className="space-y-1">
              {filteredEvals.map((evalMeta) => (
                <EvalListItem
                  key={evalMeta.id}
                  eval={evalMeta}
                  isSelected={evalMeta.id === selectedId}
                  onClick={() => onSelect(evalMeta.id)}
                />
              ))}
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Quick Stats for Selected */}
      {selectedEval && (
        <>
          <Separator />
          <div className="p-4 space-y-3">
            <h3 className="text-sm font-medium text-muted-foreground">Quick Stats</h3>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Samples</span>
                <span className="font-medium">{selectedEval.sample_count}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Analyzers</span>
                <span className="font-medium">{selectedEval.analyzer_count}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Pass Rate</span>
                <span className="font-medium">
                  {selectedEval.pass_rate !== null
                    ? `${selectedEval.pass_rate.toFixed(1)}%`
                    : 'N/A'}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Tests</span>
                <span className="font-medium">
                  {selectedEval.tests_passed}/{selectedEval.test_count}
                </span>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

interface EvalListItemProps {
  eval: EvalMetadata
  isSelected: boolean
  onClick: () => void
}

function EvalListItem({ eval: evalMeta, isSelected, onClick }: EvalListItemProps) {
  const passRate = evalMeta.pass_rate
  const hasTests = evalMeta.test_count > 0

  return (
    <button
      onClick={onClick}
      className={cn(
        'w-full text-left p-3 rounded-lg transition-colors',
        'hover:bg-accent',
        isSelected && 'bg-accent'
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <p className="font-medium truncate">{evalMeta.name}</p>
          <div className="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
            <Clock className="h-3 w-3" />
            <span>{formatRelativeTime(evalMeta.created_at)}</span>
          </div>
        </div>
        
        {hasTests && (
          <div className="flex items-center gap-1">
            {passRate !== null && passRate >= 100 ? (
              <Badge variant="success" className="flex items-center gap-1">
                <CheckCircle className="h-3 w-3" />
                {passRate.toFixed(0)}%
              </Badge>
            ) : passRate !== null && passRate > 0 ? (
              <Badge variant="warning" className="flex items-center gap-1">
                {passRate.toFixed(0)}%
              </Badge>
            ) : passRate !== null ? (
              <Badge variant="error" className="flex items-center gap-1">
                <XCircle className="h-3 w-3" />
                {passRate.toFixed(0)}%
              </Badge>
            ) : (
              <Badge variant="secondary">N/A</Badge>
            )}
          </div>
        )}
      </div>

      <div className="flex items-center gap-3 mt-2 text-xs text-muted-foreground">
        <span>{evalMeta.sample_count} samples</span>
        <span>{evalMeta.analyzer_count} analyzers</span>
        {hasTests && (
          <span>
            {evalMeta.tests_passed}/{evalMeta.test_count} tests
          </span>
        )}
      </div>
    </button>
  )
}
