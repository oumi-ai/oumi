import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { SampleConversation } from './SampleConversation'
import type { TestResult, EvalData, AnalysisResult } from '@/types/eval'
import { CheckCircle2, XCircle, ChevronDown, ChevronUp, AlertTriangle, AlertCircle, Info } from 'lucide-react'
import { cn } from '@/lib/utils'

interface TestCardProps {
  test: TestResult
  evalData: EvalData
}

const severityConfig = {
  high: {
    icon: AlertTriangle,
    className: 'border-red-500 bg-red-50 dark:bg-red-950',
    badge: 'destructive' as const,
  },
  medium: {
    icon: AlertCircle,
    className: 'border-yellow-500 bg-yellow-50 dark:bg-yellow-950',
    badge: 'secondary' as const,
  },
  low: {
    icon: Info,
    className: 'border-gray-300 bg-gray-50 dark:bg-gray-900',
    badge: 'outline' as const,
  },
}

function getProblematicIndices(
  test: TestResult, 
  analysisResults: Record<string, AnalysisResult[]>
): number[] {
  // Get the metric name (e.g., "query_quality" from "query_quality.score")
  const metricParts = test.metric?.split('.') || []
  const analyzerName = metricParts[0]
  
  if (!analyzerName || !analysisResults[analyzerName]) {
    // Fall back to sample_indices from test
    return test.sample_indices || []
  }
  
  const results = analysisResults[analyzerName]
  const problematicIndices: number[] = []
  
  // Filter to only samples with actual issues
  const sampleIndices = test.sample_indices || []
  for (const idx of sampleIndices) {
    const result = results[idx]
    if (!result) continue
    
    // Check if this sample actually has an issue
    const hasIssue = 
      result.passed === false ||
      result.error !== undefined ||
      (result.score !== undefined && test.threshold !== null && result.score < test.threshold)
    
    if (hasIssue) {
      problematicIndices.push(idx)
    }
  }
  
  return problematicIndices.length > 0 ? problematicIndices : sampleIndices.slice(0, 5)
}

function getAnalysisResultForSample(
  sampleIndex: number,
  metric: string,
  analysisResults: Record<string, AnalysisResult[]>
): AnalysisResult {
  const analyzerName = metric?.split('.')[0] || ''
  const results = analysisResults[analyzerName]
  
  if (results && results[sampleIndex]) {
    return results[sampleIndex]
  }
  
  // Return empty result if not found
  return {}
}

export function TestCard({ test, evalData }: TestCardProps) {
  const [isExpanded, setIsExpanded] = useState(!test.passed)
  const [showAllSamples, setShowAllSamples] = useState(false)
  
  const severity = severityConfig[test.severity] || severityConfig.low
  const SeverityIcon = severity.icon
  
  const problematicIndices = getProblematicIndices(test, evalData.analysis_results)
  const displayIndices = showAllSamples ? problematicIndices : problematicIndices.slice(0, 3)
  
  return (
    <Card className={cn(
      'mb-4 border-l-4',
      test.passed ? 'border-l-green-500' : severity.className.split(' ')[0]
    )}>
      <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
        <CollapsibleTrigger asChild>
          <CardHeader className="cursor-pointer hover:bg-muted/50 transition-colors">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                {test.passed ? (
                  <CheckCircle2 className="h-5 w-5 text-green-600" />
                ) : (
                  <XCircle className="h-5 w-5 text-red-600" />
                )}
                
                <div>
                  <CardTitle className="text-base flex items-center gap-2">
                    {test.title || test.test_id}
                    <Badge variant={severity.badge} className="text-xs">
                      <SeverityIcon className="h-3 w-3 mr-1" />
                      {test.severity}
                    </Badge>
                  </CardTitle>
                  {test.description && (
                    <p className="text-sm text-muted-foreground mt-1">
                      {test.description}
                    </p>
                  )}
                </div>
              </div>
              
              <div className="flex items-center gap-4">
                <div className="text-right">
                  <div className="text-sm">
                    <span className="font-medium">{test.affected_count}</span>
                    <span className="text-muted-foreground">/{test.total_count}</span>
                  </div>
                  <div className={cn(
                    'text-xs',
                    test.passed ? 'text-green-600' : 'text-red-600'
                  )}>
                    {test.affected_percentage?.toFixed(1)}%
                  </div>
                </div>
                
                <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                  {isExpanded ? (
                    <ChevronUp className="h-4 w-4" />
                  ) : (
                    <ChevronDown className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </div>
          </CardHeader>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <CardContent className="pt-0 space-y-4">
            {/* Test Details */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-3 bg-muted/50 rounded-lg">
              <div>
                <span className="text-xs text-muted-foreground">Test ID</span>
                <p className="text-sm font-mono">{test.test_id}</p>
              </div>
              <div>
                <span className="text-xs text-muted-foreground">Metric</span>
                <p className="text-sm font-mono">{test.metric}</p>
              </div>
              {/* Show value threshold with operator if available */}
              {test.details?.value !== undefined && (
                <div>
                  <span className="text-xs text-muted-foreground">Condition</span>
                  <p className="text-sm font-mono">{String(test.details.operator || '')} {String(test.details.value)}</p>
                </div>
              )}
              {/* Show max percentage limit if set */}
              {test.details?.max_percentage !== undefined && test.details.max_percentage !== null && (
                <div>
                  <span className="text-xs text-muted-foreground">Max % Allowed</span>
                  <p className="text-sm">{String(test.details.max_percentage)}%</p>
                </div>
              )}
              {/* Show min percentage limit if set */}
              {test.details?.min_percentage !== undefined && test.details.min_percentage !== null && (
                <div>
                  <span className="text-xs text-muted-foreground">Min % Required</span>
                  <p className="text-sm">{String(test.details.min_percentage)}%</p>
                </div>
              )}
              {test.actual_value !== null && (
                <div>
                  <span className="text-xs text-muted-foreground">Actual</span>
                  <p className="text-sm">{test.actual_value}</p>
                </div>
              )}
            </div>

            {/* Error message */}
            {test.error && (
              <div className="p-3 bg-red-50 dark:bg-red-950 rounded-lg border border-red-200 dark:border-red-800">
                <span className="text-sm text-red-700 dark:text-red-300">{test.error}</span>
              </div>
            )}

            {/* Dataset-level test - show value comparison instead of samples */}
            {test.total_count === 1 ? (
              <div className="space-y-3">
                <h4 className="text-sm font-medium">Dataset-Level Result</h4>
                <div className="p-4 bg-muted/50 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-xs text-muted-foreground block mb-1">Metric</span>
                      <span className="font-mono text-sm">{test.metric}</span>
                    </div>
                    <div className="text-right">
                      <span className="text-xs text-muted-foreground block mb-1">Actual Value</span>
                      {(() => {
                        // Extract actual value from failure reasons or details
                        const failureReasons = test.details?.failure_reasons as Record<string, string> | undefined
                        const reason = failureReasons?.['0'] || ''
                        // Parse value from reason like "1 does not satisfy > 2" or "0.05 < 0.1"
                        const valueMatch = reason.match(/^([\d.]+)/)
                        const actualValue = valueMatch ? valueMatch[1] : 
                          (test.details?.passing_count !== undefined ? 
                            `${test.details.passing_count}/${test.total_count} passing` : 
                            'N/A')
                        return (
                          <span className={cn(
                            "font-mono text-lg font-semibold",
                            test.passed ? "text-green-600" : "text-red-600"
                          )}>
                            {actualValue}
                          </span>
                        )
                      })()}
                    </div>
                    <div className="text-center px-4">
                      <span className="text-xs text-muted-foreground block mb-1">Condition</span>
                      <span className="font-mono text-sm">
                        {String(test.details?.operator ?? '')} {String(test.details?.value ?? '')}
                      </span>
                    </div>
                    <div>
                      <span className="text-xs text-muted-foreground block mb-1">Result</span>
                      {test.passed ? (
                        <Badge variant="default" className="bg-green-600">PASSED</Badge>
                      ) : (
                        <Badge variant="destructive">FAILED</Badge>
                      )}
                    </div>
                  </div>
                  {(() => {
                    const reasons = test.details?.failure_reasons as Record<string, string> | undefined
                    const reason = reasons?.['0']
                    if (!test.passed && reason) {
                      return (
                        <div className="mt-3 pt-3 border-t border-border">
                          <span className="text-xs text-muted-foreground">Reason: </span>
                          <span className="text-sm text-red-600 dark:text-red-400">
                            {String(reason)}
                          </span>
                        </div>
                      )
                    }
                    return null
                  })()}
                </div>
                <p className="text-xs text-muted-foreground">
                  This is a dataset-level metric that produces a single result for the entire dataset.
                </p>
              </div>
            ) : problematicIndices.length > 0 && evalData.conversations.length > 0 ? (
              /* Sample Conversations for conversation-level tests */
              <div className="space-y-3">
                <h4 className="text-sm font-medium">
                  Sample Conversations with Issues ({problematicIndices.length} affected)
                </h4>
                
                {displayIndices.map((sampleIndex, i) => {
                  const conversation = evalData.conversations[sampleIndex]
                  if (!conversation) return null
                  
                  const analysisResult = getAnalysisResultForSample(
                    sampleIndex,
                    test.metric,
                    evalData.analysis_results
                  )
                  
                  // Get failure reason from test details
                  const failureReasons = test.details?.failure_reasons as Record<string, string> | undefined
                  const failureReason = failureReasons?.[String(sampleIndex)]
                  
                  return (
                    <SampleConversation
                      key={`${test.test_id}-sample-${sampleIndex}`}
                      index={sampleIndex}
                      conversation={conversation}
                      analysisResult={analysisResult}
                      keyPrefix={`${test.test_id}-${i}`}
                      failureReason={failureReason}
                      metricName={test.metric}
                    />
                  )
                })}
                
                {problematicIndices.length > 3 && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowAllSamples(!showAllSamples)}
                    className="w-full"
                  >
                    {showAllSamples 
                      ? 'Show fewer samples' 
                      : `Show all ${problematicIndices.length} samples`
                    }
                  </Button>
                )}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                No sample conversations available.
              </p>
            )}
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  )
}
