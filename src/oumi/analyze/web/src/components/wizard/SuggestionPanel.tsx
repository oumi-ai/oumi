import { useState } from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  ChevronDown,
  ChevronUp,
  Sparkles,
  Check,
  Plus,
  Loader2,
  AlertCircle,
  X,
  Lightbulb,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type {
  AnalyzerSuggestion,
  CustomMetricSuggestion,
  TestSuggestion,
} from '@/hooks/useSuggestions'

interface SuggestionPanelProps {
  /** Current status of suggestion loading */
  status: 'idle' | 'loading' | 'success' | 'error'
  /** Error message if status is error */
  error?: string | null
  /** Whether the panel has been dismissed */
  isDismissed: boolean
  /** Callback to dismiss the panel */
  onDismiss: () => void
  /** Callback to show the panel again */
  onUndismiss: () => void
  /** Panel type determines which suggestions to show */
  type: 'analyzers' | 'tests'
  /** Analyzer suggestions (for type='analyzers') */
  analyzerSuggestions?: AnalyzerSuggestion[]
  /** Custom metric suggestions (for type='analyzers') */
  customMetricSuggestions?: CustomMetricSuggestion[]
  /** Test suggestions (for type='tests') */
  testSuggestions?: TestSuggestion[]
  /** Set of already-applied analyzer IDs */
  appliedAnalyzers?: Set<string>
  /** Set of already-applied custom metric IDs */
  appliedCustomMetrics?: Set<string>
  /** Set of already-applied test IDs */
  appliedTests?: Set<string>
  /** Callback when user applies an analyzer suggestion */
  onApplyAnalyzer?: (suggestion: AnalyzerSuggestion) => void
  /** Callback when user applies a custom metric suggestion */
  onApplyCustomMetric?: (suggestion: CustomMetricSuggestion) => void
  /** Callback when user applies a test suggestion */
  onApplyTest?: (suggestion: TestSuggestion) => void
  /** Callback to apply all suggestions */
  onApplyAll?: () => void
}

/**
 * A collapsible panel showing AI-powered suggestions for configuration.
 * 
 * This component displays suggestions from the LLM and allows users to
 * apply them individually or all at once.
 */
export function SuggestionPanel({
  status,
  error,
  isDismissed,
  onDismiss,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  onUndismiss: _onUndismiss, // Not used in this component - handled by SuggestionPanelMinimized
  type,
  analyzerSuggestions = [],
  customMetricSuggestions = [],
  testSuggestions = [],
  appliedAnalyzers = new Set(),
  appliedCustomMetrics = new Set(),
  appliedTests = new Set(),
  onApplyAnalyzer,
  onApplyCustomMetric,
  onApplyTest,
  onApplyAll,
}: SuggestionPanelProps) {
  const [isOpen, setIsOpen] = useState(true)

  // Don't show if idle or dismissed
  if (status === 'idle' || isDismissed) {
    return null
  }

  // Calculate counts for display
  const unappliedAnalyzers = analyzerSuggestions.filter(a => !appliedAnalyzers.has(a.id))
  const unappliedCustomMetrics = customMetricSuggestions.filter(m => !appliedCustomMetrics.has(m.id))
  const unappliedTests = testSuggestions.filter(t => !appliedTests.has(t.id))

  const totalSuggestions = type === 'analyzers'
    ? analyzerSuggestions.length + customMetricSuggestions.length
    : testSuggestions.length

  const unappliedCount = type === 'analyzers'
    ? unappliedAnalyzers.length + unappliedCustomMetrics.length
    : unappliedTests.length

  // Loading state
  if (status === 'loading') {
    return (
      <Card className="border-primary/30 bg-primary/5 mb-4">
        <CardContent className="p-4">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary/10">
              <Loader2 className="h-4 w-4 text-primary animate-spin" />
            </div>
            <div className="flex-1">
              <p className="text-sm font-medium">Analyzing your data...</p>
              <p className="text-xs text-muted-foreground">
                AI is reviewing sample conversations to suggest configuration
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  // Error state
  if (status === 'error') {
    return (
      <Card className="border-destructive/30 bg-destructive/5 mb-4">
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-8 h-8 rounded-full bg-destructive/10">
                <AlertCircle className="h-4 w-4 text-destructive" />
              </div>
              <div className="flex-1">
                <p className="text-sm font-medium">Couldn't generate suggestions</p>
                <p className="text-xs text-muted-foreground">
                  {error || 'Please configure manually or try again'}
                </p>
              </div>
            </div>
            <Button variant="ghost" size="sm" onClick={onDismiss}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    )
  }

  // No suggestions
  if (totalSuggestions === 0) {
    return null
  }

  // All applied
  if (unappliedCount === 0) {
    return (
      <Card className="border-green-500/30 bg-green-500/5 mb-4">
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-8 h-8 rounded-full bg-green-500/10">
                <Check className="h-4 w-4 text-green-600" />
              </div>
              <div className="flex-1">
                <p className="text-sm font-medium">All suggestions applied</p>
                <p className="text-xs text-muted-foreground">
                  You've added all {totalSuggestions} AI-recommended {type === 'analyzers' ? 'analyzers' : 'tests'}
                </p>
              </div>
            </div>
            <Button variant="ghost" size="sm" onClick={onDismiss}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    )
  }

  // Success state with suggestions
  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen} className="mb-4">
      <Card className="border-primary/30 bg-gradient-to-r from-primary/5 to-transparent">
        <CollapsibleTrigger asChild>
          <CardContent className="p-4 cursor-pointer hover:bg-primary/5 transition-colors">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary/10">
                  <Sparkles className="h-4 w-4 text-primary" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-medium">AI Suggestions</p>
                    <Badge variant="secondary" className="text-xs">
                      {unappliedCount} remaining
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Based on analysis of your sample conversations
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation()
                    onDismiss()
                  }}
                  className="h-8 w-8 p-0"
                >
                  <X className="h-4 w-4" />
                </Button>
                {isOpen ? (
                  <ChevronUp className="h-4 w-4 text-muted-foreground" />
                ) : (
                  <ChevronDown className="h-4 w-4 text-muted-foreground" />
                )}
              </div>
            </div>
          </CardContent>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <div className="px-4 pb-4">
            <ScrollArea className="max-h-[300px]">
              <div className="space-y-3">
                {/* Analyzer suggestions */}
                {type === 'analyzers' && unappliedAnalyzers.length > 0 && (
                  <div className="space-y-2">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                      Analyzers
                    </p>
                    {unappliedAnalyzers.map((suggestion) => (
                      <SuggestionItem
                        key={suggestion.id}
                        title={suggestion.id}
                        reason={suggestion.reason}
                        onApply={() => onApplyAnalyzer?.(suggestion)}
                      />
                    ))}
                  </div>
                )}

                {/* Custom metric suggestions */}
                {type === 'analyzers' && unappliedCustomMetrics.length > 0 && (
                  <div className="space-y-2">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                      Custom Metrics
                    </p>
                    {unappliedCustomMetrics.map((suggestion) => (
                      <SuggestionItem
                        key={suggestion.id}
                        title={suggestion.id}
                        reason={suggestion.reason}
                        subtitle={suggestion.description}
                        onApply={() => onApplyCustomMetric?.(suggestion)}
                      />
                    ))}
                  </div>
                )}

                {/* Test suggestions */}
                {type === 'tests' && unappliedTests.length > 0 && (
                  <div className="space-y-2">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                      Tests
                    </p>
                    {unappliedTests.map((suggestion) => (
                      <SuggestionItem
                        key={suggestion.id}
                        title={suggestion.title || suggestion.id}
                        reason={suggestion.reason}
                        subtitle={`${suggestion.type} on ${suggestion.metric}`}
                        severity={suggestion.severity}
                        onApply={() => onApplyTest?.(suggestion)}
                      />
                    ))}
                  </div>
                )}
              </div>
            </ScrollArea>

            {/* Apply all button */}
            {unappliedCount > 1 && (
              <div className="mt-3 pt-3 border-t">
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full"
                  onClick={onApplyAll}
                >
                  <Plus className="h-4 w-4 mr-2" />
                  Apply All {unappliedCount} Suggestions
                </Button>
              </div>
            )}
          </div>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  )
}

/**
 * Individual suggestion item
 */
interface SuggestionItemProps {
  title: string
  reason: string
  subtitle?: string
  severity?: 'low' | 'medium' | 'high'
  onApply: () => void
}

function SuggestionItem({ title, reason, subtitle, severity, onApply }: SuggestionItemProps) {
  return (
    <div className="flex items-start gap-3 p-3 rounded-lg bg-background/50 border border-border/50 hover:border-primary/30 transition-colors">
      <div className="flex items-center justify-center w-6 h-6 rounded-full bg-primary/10 mt-0.5">
        <Lightbulb className="h-3 w-3 text-primary" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <p className="text-sm font-medium">{title}</p>
          {severity && (
            <Badge
              variant="outline"
              className={cn(
                'text-xs',
                severity === 'high' && 'border-red-500/50 text-red-600',
                severity === 'medium' && 'border-yellow-500/50 text-yellow-600',
                severity === 'low' && 'border-green-500/50 text-green-600'
              )}
            >
              {severity}
            </Badge>
          )}
        </div>
        {subtitle && (
          <p className="text-xs text-muted-foreground mt-0.5">{subtitle}</p>
        )}
        <p className="text-xs text-muted-foreground mt-1">{reason}</p>
      </div>
      <Button
        variant="ghost"
        size="sm"
        className="h-8 px-3 text-primary hover:text-primary hover:bg-primary/10"
        onClick={onApply}
      >
        <Plus className="h-4 w-4 mr-1" />
        Add
      </Button>
    </div>
  )
}

/**
 * Compact button to show suggestions panel when dismissed
 */
export function SuggestionPanelMinimized({
  suggestionCount,
  onClick,
}: {
  suggestionCount: number
  onClick: () => void
}) {
  if (suggestionCount === 0) return null

  return (
    <Button
      variant="outline"
      size="sm"
      className="mb-4 gap-2 border-primary/30 text-primary hover:bg-primary/5"
      onClick={onClick}
    >
      <Sparkles className="h-4 w-4" />
      Show AI Suggestions ({suggestionCount})
    </Button>
  )
}
