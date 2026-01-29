import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { TestSummary as TestSummaryType } from '@/types/eval'
import { CheckCircle2, XCircle, AlertCircle, AlertTriangle } from 'lucide-react'

interface TestSummaryProps {
  summary: TestSummaryType
}

export function TestSummary({ summary }: TestSummaryProps) {
  const passRate = summary.total_tests > 0 
    ? Math.round((summary.passed_tests / summary.total_tests) * 100) 
    : 0

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
      {/* Pass Rate Card */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Pass Rate
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2">
            <span className={`text-2xl font-bold ${
              passRate >= 80 ? 'text-green-600' : 
              passRate >= 50 ? 'text-yellow-600' : 'text-red-600'
            }`}>
              {passRate}%
            </span>
          </div>
        </CardContent>
      </Card>

      {/* Passed Card */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-1">
            <CheckCircle2 className="h-4 w-4 text-green-600" />
            Passed
          </CardTitle>
        </CardHeader>
        <CardContent>
          <span className="text-2xl font-bold text-green-600">
            {summary.passed_tests}
          </span>
        </CardContent>
      </Card>

      {/* Failed Card */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-1">
            <XCircle className="h-4 w-4 text-red-600" />
            Failed
          </CardTitle>
        </CardHeader>
        <CardContent>
          <span className="text-2xl font-bold text-red-600">
            {summary.failed_tests}
          </span>
        </CardContent>
      </Card>

      {/* Severity Breakdown */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-1">
            <AlertCircle className="h-4 w-4" />
            Severity
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 flex-wrap">
            {summary.high_severity_failures > 0 && (
              <Badge variant="destructive" className="text-xs">
                <AlertTriangle className="h-3 w-3 mr-1" />
                {summary.high_severity_failures} High
              </Badge>
            )}
            {summary.medium_severity_failures > 0 && (
              <Badge variant="secondary" className="text-xs bg-yellow-100 text-yellow-800 hover:bg-yellow-100">
                {summary.medium_severity_failures} Med
              </Badge>
            )}
            {summary.low_severity_failures > 0 && (
              <Badge variant="outline" className="text-xs">
                {summary.low_severity_failures} Low
              </Badge>
            )}
            {summary.high_severity_failures === 0 && 
             summary.medium_severity_failures === 0 && 
             summary.low_severity_failures === 0 && (
              <span className="text-sm text-muted-foreground">None</span>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
