import { ReactNode } from 'react'
import { Calendar, Database, FileText } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { formatRelativeTime } from '@/lib/utils'
import type { EvalData } from '@/types/eval'
import { computeTestSummary } from '@/types/eval'

interface HeaderProps {
  evalData: EvalData
  children?: ReactNode
}

export function Header({ evalData, children }: HeaderProps) {
  const { metadata } = evalData
  const testSummary = computeTestSummary(evalData.test_results)

  return (
    <div className="border-b bg-background px-6 py-4">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-semibold">{metadata.name}</h1>
          <div className="flex items-center gap-4 mt-2 text-sm text-muted-foreground">
            <div className="flex items-center gap-1">
              <Calendar className="h-4 w-4" />
              <span>{formatRelativeTime(metadata.created_at)}</span>
            </div>
            <div className="flex items-center gap-1">
              <Database className="h-4 w-4" />
              <span>{metadata.sample_count} samples</span>
            </div>
            {metadata.config_path && (
              <div className="flex items-center gap-1">
                <FileText className="h-4 w-4" />
                <span className="truncate max-w-[200px]" title={metadata.config_path}>
                  {metadata.config_path.split('/').pop()}
                </span>
              </div>
            )}
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
        </div>
      </div>
    </div>
  )
}
