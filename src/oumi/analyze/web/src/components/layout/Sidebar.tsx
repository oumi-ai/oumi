import { useState, useRef } from 'react'
import { Search, CheckCircle, XCircle, Clock, BarChart3, Plus, Trash2 } from 'lucide-react'
import { cn, formatRelativeTime } from '@/lib/utils'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import type { EvalMetadata } from '@/types/eval'

interface SidebarProps {
  evals: EvalMetadata[]
  selectedId: string | null
  onSelect: (id: string) => void
  onDelete?: (id: string) => void
  isLoading: boolean
  onNewAnalysis?: () => void
  onLogoClick?: () => void
}

export function Sidebar({ evals, selectedId, onSelect, onDelete, isLoading, onNewAnalysis, onLogoClick }: SidebarProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [deleteTarget, setDeleteTarget] = useState<EvalMetadata | null>(null)

  const filteredEvals = evals.filter(
    (e) =>
      e.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      e.id.toLowerCase().includes(searchQuery.toLowerCase())
  )


  return (
    <div className="w-80 border-r bg-muted/30 flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b">
        <div className="flex items-center justify-between mb-4">
          <button 
            onClick={onLogoClick}
            className="flex items-center gap-2 hover:opacity-80 transition-opacity"
          >
            <BarChart3 className="h-6 w-6 text-primary" />
            <h1 className="text-xl font-semibold">Oumi Analyze</h1>
          </button>
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
                  onDelete={onDelete ? () => setDeleteTarget(evalMeta) : undefined}
                />
              ))}
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={!!deleteTarget} onOpenChange={(open) => !open && setDeleteTarget(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Analysis</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete "{deleteTarget?.name}"? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (deleteTarget && onDelete) {
                  onDelete(deleteTarget.id)
                }
                setDeleteTarget(null)
              }}
              className="bg-red-500 hover:bg-red-600"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}

interface EvalListItemProps {
  eval: EvalMetadata
  isSelected: boolean
  onClick: () => void
  onDelete?: () => void
}

function EvalListItem({ eval: evalMeta, isSelected, onClick, onDelete }: EvalListItemProps) {
  const passRate = evalMeta.pass_rate
  const hasTests = evalMeta.test_count > 0
  const [offsetX, setOffsetX] = useState(0)
  const [isDragging, setIsDragging] = useState(false)
  const startX = useRef(0)
  const DELETE_THRESHOLD = 80

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!onDelete) return
    startX.current = e.clientX
    setIsDragging(true)
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return
    const diff = startX.current - e.clientX
    // Only allow sliding left (positive diff)
    setOffsetX(Math.max(0, Math.min(diff, DELETE_THRESHOLD + 20)))
  }

  const handleMouseUp = () => {
    if (!isDragging) return
    setIsDragging(false)
    
    if (offsetX >= DELETE_THRESHOLD) {
      // Keep it open to show delete button
      setOffsetX(DELETE_THRESHOLD)
    } else {
      // Snap back
      setOffsetX(0)
    }
  }

  const handleMouseLeave = () => {
    if (isDragging) {
      setIsDragging(false)
      if (offsetX < DELETE_THRESHOLD) {
        setOffsetX(0)
      }
    }
  }

  const handleDeleteClick = () => {
    setOffsetX(0)
    onDelete?.()
  }

  return (
    <div className="relative overflow-hidden rounded-lg">
      {/* Delete button background */}
      {onDelete && (
        <button
          onClick={handleDeleteClick}
          className={cn(
            "absolute right-0 top-0 bottom-0 flex items-center justify-center bg-red-500 text-white transition-all",
            offsetX > 0 ? "w-20" : "w-0"
          )}
        >
          <Trash2 className="h-5 w-5" />
        </button>
      )}

      {/* Main content */}
      <div
        className={cn(
          'relative w-full text-left p-3 rounded-lg cursor-pointer bg-background',
          'hover:bg-accent',
          isSelected && 'bg-accent',
          isDragging ? '' : 'transition-transform duration-200'
        )}
        style={{ transform: `translateX(-${offsetX}px)` }}
        onClick={() => offsetX === 0 && onClick()}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      >
        <div>
          <p className="font-medium truncate">{evalMeta.name}</p>
          <div className="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
            <Clock className="h-3 w-3 shrink-0" />
            <span>{formatRelativeTime(evalMeta.created_at)}</span>
          </div>
        </div>

        <div className="flex items-center justify-between gap-3 mt-2 text-xs text-muted-foreground">
          <div className="flex items-center gap-3">
            <span>{evalMeta.sample_count} samples</span>
            <span>{evalMeta.analyzer_count} analyzers</span>
            {hasTests && (
              <span>
                {evalMeta.tests_passed}/{evalMeta.test_count} tests
              </span>
            )}
          </div>
          
          {hasTests && passRate !== null && (
            <Badge 
              variant={passRate >= 1.0 ? "success" : passRate > 0 ? "warning" : "error"} 
              className="flex items-center gap-1"
            >
              {passRate >= 1.0 && <CheckCircle className="h-3 w-3" />}
              {passRate === 0 && <XCircle className="h-3 w-3" />}
              {Math.round(passRate * 100)}%
            </Badge>
          )}
          {hasTests && passRate === null && (
            <Badge variant="secondary">N/A</Badge>
          )}
        </div>
      </div>
    </div>
  )
}
