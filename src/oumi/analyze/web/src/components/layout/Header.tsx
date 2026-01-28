import { ReactNode, useState, useRef, useEffect } from 'react'
import { Calendar, Database, Pencil, Check, X, Trash2 } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import { formatRelativeTime } from '@/lib/utils'
import type { EvalData } from '@/types/eval'
import { computeTestSummary } from '@/types/eval'

interface HeaderProps {
  evalData: EvalData
  children?: ReactNode
  onRename?: (newName: string) => void
  onDelete?: () => void
}

export function Header({ evalData, children, onRename, onDelete }: HeaderProps) {
  const { metadata } = evalData
  const testSummary = computeTestSummary(evalData.test_results)
  const [isEditing, setIsEditing] = useState(false)
  const [editName, setEditName] = useState(metadata.name)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus()
      inputRef.current.select()
    }
  }, [isEditing])

  const handleSave = () => {
    if (editName.trim() && editName !== metadata.name) {
      onRename?.(editName.trim())
    }
    setIsEditing(false)
  }

  const handleCancel = () => {
    setEditName(metadata.name)
    setIsEditing(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleSave()
    if (e.key === 'Escape') handleCancel()
  }

  return (
    <div className="border-b bg-background px-6 py-4">
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2">
            {isEditing ? (
              <div className="flex items-center gap-2">
                <Input
                  ref={inputRef}
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  onKeyDown={handleKeyDown}
                  className="text-2xl font-semibold h-10 w-80"
                />
                <Button variant="ghost" size="icon" onClick={handleSave} className="h-8 w-8">
                  <Check className="h-4 w-4 text-green-500" />
                </Button>
                <Button variant="ghost" size="icon" onClick={handleCancel} className="h-8 w-8">
                  <X className="h-4 w-4 text-red-500" />
                </Button>
              </div>
            ) : (
              <>
                <h1 className="text-2xl font-semibold">{metadata.name}</h1>
                {onRename && (
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setIsEditing(true)}
                    className="h-8 w-8 text-muted-foreground hover:text-foreground"
                  >
                    <Pencil className="h-4 w-4" />
                  </Button>
                )}
              </>
            )}
          </div>
          <div className="flex items-center gap-4 mt-2 text-sm text-muted-foreground">
            <div className="flex items-center gap-1">
              <Calendar className="h-4 w-4" />
              <span>{formatRelativeTime(metadata.created_at)}</span>
            </div>
            <div className="flex items-center gap-1">
              <Database className="h-4 w-4" />
              <span>{metadata.sample_count} samples</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* Pass Rate Badge */}
          {testSummary.total_tests > 0 && (
            <div className="text-center">
              <div className="text-2xl font-bold">
                {testSummary.pass_rate.toFixed(0)}%
              </div>
              <div className="text-xs text-muted-foreground">Pass Rate</div>
            </div>
          )}

          {/* Test Stats */}
          <div className="flex gap-2">
            {testSummary.passed_tests > 0 && (
              <Badge variant="success">
                {testSummary.passed_tests} Passed
              </Badge>
            )}
            {testSummary.failed_tests > 0 && (
              <Badge variant="error">
                {testSummary.failed_tests} Failed
              </Badge>
            )}
            {testSummary.error_tests > 0 && (
              <Badge variant="warning">
                {testSummary.error_tests} Errors
              </Badge>
            )}
          </div>

          {/* Action buttons from parent */}
          {children}

          {/* Delete button with confirmation dialog */}
          {onDelete && (
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-9 w-9 text-muted-foreground hover:text-red-500 hover:bg-red-50"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Delete Analysis</AlertDialogTitle>
                  <AlertDialogDescription>
                    Are you sure you want to delete "{metadata.name}"? This action cannot be undone.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    onClick={onDelete}
                    className="bg-red-500 hover:bg-red-600"
                  >
                    Delete
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          )}
        </div>
      </div>
    </div>
  )
}
