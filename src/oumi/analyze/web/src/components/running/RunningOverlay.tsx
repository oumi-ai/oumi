import { Progress } from '@/components/ui/progress'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Loader2, CheckCircle2, XCircle, Terminal, X } from 'lucide-react'
import type { JobStatus } from '@/hooks/useEvals'

interface RunningOverlayProps {
  jobStatus: JobStatus | null
  onCancel?: () => void
  onRetry?: () => void
}

export function RunningOverlay({ jobStatus, onCancel, onRetry }: RunningOverlayProps) {
  if (!jobStatus) return null

  const progressPercent = Math.round((jobStatus.progress / jobStatus.total) * 100)
  const isComplete = jobStatus.status === 'completed'
  const isFailed = jobStatus.status === 'failed'
  const isRunning = jobStatus.status === 'running' || jobStatus.status === 'pending'

  return (
    <div className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50 flex items-center justify-center">
      <Card className="w-full max-w-2xl mx-4">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              {isComplete ? (
                <CheckCircle2 className="h-5 w-5 text-green-500" />
              ) : isFailed ? (
                <XCircle className="h-5 w-5 text-red-500" />
              ) : (
                <Loader2 className="h-5 w-5 animate-spin" />
              )}
              {isComplete ? 'Analysis Complete' : isFailed ? 'Analysis Failed' : 'Running Analysis'}
            </CardTitle>
            {isRunning && onCancel && (
              <Button variant="ghost" size="icon" onClick={onCancel}>
                <X className="h-4 w-4" />
              </Button>
            )}
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Status Message */}
          <p className="text-sm text-muted-foreground">
            {jobStatus.message || 'Starting analysis...'}
          </p>

          {/* Progress Bar */}
          {isRunning && (
            <div>
              <Progress value={progressPercent} className="h-2" />
              <p className="text-xs text-muted-foreground mt-1">
                {jobStatus.progress} / {jobStatus.total} samples
              </p>
            </div>
          )}

          {/* Error Message */}
          {isFailed && jobStatus.error && (
            <div className="p-3 bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 rounded-lg">
              <p className="text-sm text-red-600 dark:text-red-400">{jobStatus.error}</p>
            </div>
          )}

          {/* Success Message */}
          {isComplete && (
            <div className="p-3 bg-green-50 dark:bg-green-950 border border-green-200 dark:border-green-800 rounded-lg">
              <p className="text-sm text-green-600 dark:text-green-400">
                Analysis completed! Redirecting to results...
              </p>
            </div>
          )}

          {/* Log Output */}
          {jobStatus.log_lines && jobStatus.log_lines.length > 0 && (
            <div>
              <div className="flex items-center gap-2 mb-2">
                <Terminal className="h-4 w-4" />
                <span className="text-sm font-medium">Output Log</span>
              </div>
              <ScrollArea className="h-32 border rounded-md bg-zinc-950 p-3">
                <pre className="text-xs text-zinc-300 font-mono whitespace-pre-wrap">
                  {jobStatus.log_lines.join('\n')}
                </pre>
              </ScrollArea>
            </div>
          )}

          {/* Actions */}
          {isFailed && (
            <div className="flex items-center justify-end gap-2 pt-2">
              {onCancel && (
                <Button variant="outline" onClick={onCancel}>
                  Close
                </Button>
              )}
              {onRetry && (
                <Button onClick={onRetry}>
                  Retry
                </Button>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
