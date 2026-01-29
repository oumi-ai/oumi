import { useState, useCallback } from 'react'

const API_BASE_URL = '/api'

/**
 * Suggested analyzer configuration
 */
export interface AnalyzerSuggestion {
  id: string
  reason: string
  params?: Record<string, unknown>
}

/**
 * Suggested custom metric configuration
 */
export interface CustomMetricSuggestion {
  id: string
  function: string
  reason: string
  output_schema: Array<{ name: string; type: string; description?: string }>
  description?: string
}

/**
 * Suggested test configuration
 */
export interface TestSuggestion {
  id: string
  type: 'threshold' | 'percentage' | 'range'
  metric: string
  reason: string
  title?: string
  description?: string
  severity?: 'low' | 'medium' | 'high'
  operator?: string
  value?: number
  condition?: string
  max_percentage?: number
  min_percentage?: number
  min_value?: number
  max_value?: number
}

/**
 * Complete suggestion response from the API
 */
export interface SuggestionResponse {
  analyzers: AnalyzerSuggestion[]
  custom_metrics: CustomMetricSuggestion[]
  tests: TestSuggestion[]
  error?: string | null
}

/**
 * Dataset configuration for suggestion request
 */
export interface DatasetConfig {
  dataset_path?: string
  dataset_name?: string
  split?: string
  subset?: string
  sample_count?: number
  /** User's description of their goals and what issues they want to check */
  user_query?: string
}

/**
 * Suggestion state
 */
type SuggestionStatus = 'idle' | 'loading' | 'success' | 'error'

/**
 * Fetch suggestions from the API
 */
async function fetchSuggestions(config: DatasetConfig): Promise<SuggestionResponse> {
  const response = await fetch(`${API_BASE_URL}/suggest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dataset_path: config.dataset_path,
      dataset_name: config.dataset_name,
      split: config.split || 'train',
      subset: config.subset,
      sample_count: config.sample_count || 1,
      user_query: config.user_query,
    }),
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.error || `Failed to get suggestions: ${response.statusText}`)
  }

  return response.json()
}

/**
 * Hook to manage AI-powered suggestions for analyzers, metrics, and tests.
 * 
 * The suggestion system analyzes sample conversations from the dataset
 * and recommends appropriate configuration using an LLM.
 * 
 * Usage:
 * ```tsx
 * const { 
 *   suggestions, 
 *   status, 
 *   triggerSuggestions, 
 *   appliedAnalyzers,
 *   markAnalyzerApplied,
 * } = useSuggestions()
 * 
 * // Trigger when user completes dataset step
 * triggerSuggestions({ dataset_path: '/path/to/data.jsonl' })
 * 
 * // Mark suggestion as applied
 * markAnalyzerApplied('length')
 * ```
 */
export function useSuggestions() {
  const [status, setStatus] = useState<SuggestionStatus>('idle')
  const [suggestions, setSuggestions] = useState<SuggestionResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  
  // Track which suggestions have been applied
  const [appliedAnalyzers, setAppliedAnalyzers] = useState<Set<string>>(new Set())
  const [appliedCustomMetrics, setAppliedCustomMetrics] = useState<Set<string>>(new Set())
  const [appliedTests, setAppliedTests] = useState<Set<string>>(new Set())
  
  // Track if panel is dismissed
  const [isDismissed, setIsDismissed] = useState(false)

  /**
   * Trigger suggestion generation for a dataset
   */
  const triggerSuggestions = useCallback(async (config: DatasetConfig) => {
    // Don't re-fetch if already loading
    if (status === 'loading') return

    // Reset applied state for new suggestions
    setAppliedAnalyzers(new Set())
    setAppliedCustomMetrics(new Set())
    setAppliedTests(new Set())
    setIsDismissed(false)
    
    setStatus('loading')
    setError(null)

    try {
      const result = await fetchSuggestions(config)
      
      if (result.error) {
        setError(result.error)
        setStatus('error')
      } else {
        setSuggestions(result)
        setStatus('success')
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Failed to get suggestions'
      setError(message)
      setStatus('error')
    }
  }, [status])

  /**
   * Mark an analyzer suggestion as applied
   */
  const markAnalyzerApplied = useCallback((id: string) => {
    setAppliedAnalyzers(prev => new Set([...prev, id]))
  }, [])

  /**
   * Mark a custom metric suggestion as applied
   */
  const markCustomMetricApplied = useCallback((id: string) => {
    setAppliedCustomMetrics(prev => new Set([...prev, id]))
  }, [])

  /**
   * Mark a test suggestion as applied
   */
  const markTestApplied = useCallback((id: string) => {
    setAppliedTests(prev => new Set([...prev, id]))
  }, [])

  /**
   * Mark all analyzer suggestions as applied
   */
  const markAllAnalyzersApplied = useCallback(() => {
    if (suggestions?.analyzers) {
      setAppliedAnalyzers(new Set(suggestions.analyzers.map(a => a.id)))
    }
  }, [suggestions])

  /**
   * Mark all custom metric suggestions as applied
   */
  const markAllCustomMetricsApplied = useCallback(() => {
    if (suggestions?.custom_metrics) {
      setAppliedCustomMetrics(new Set(suggestions.custom_metrics.map(m => m.id)))
    }
  }, [suggestions])

  /**
   * Mark all test suggestions as applied
   */
  const markAllTestsApplied = useCallback(() => {
    if (suggestions?.tests) {
      setAppliedTests(new Set(suggestions.tests.map(t => t.id)))
    }
  }, [suggestions])

  /**
   * Dismiss the suggestion panel
   */
  const dismiss = useCallback(() => {
    setIsDismissed(true)
  }, [])

  /**
   * Show the suggestion panel again
   */
  const undismiss = useCallback(() => {
    setIsDismissed(false)
  }, [])

  /**
   * Reset all state
   */
  const reset = useCallback(() => {
    setStatus('idle')
    setSuggestions(null)
    setError(null)
    setAppliedAnalyzers(new Set())
    setAppliedCustomMetrics(new Set())
    setAppliedTests(new Set())
    setIsDismissed(false)
  }, [])

  /**
   * Get unapplied analyzer suggestions
   */
  const unappliedAnalyzers = suggestions?.analyzers.filter(
    a => !appliedAnalyzers.has(a.id)
  ) || []

  /**
   * Get unapplied custom metric suggestions
   */
  const unappliedCustomMetrics = suggestions?.custom_metrics.filter(
    m => !appliedCustomMetrics.has(m.id)
  ) || []

  /**
   * Get unapplied test suggestions
   */
  const unappliedTests = suggestions?.tests.filter(
    t => !appliedTests.has(t.id)
  ) || []

  /**
   * Check if there are any unapplied suggestions
   */
  const hasUnappliedSuggestions = 
    unappliedAnalyzers.length > 0 || 
    unappliedCustomMetrics.length > 0 || 
    unappliedTests.length > 0

  return {
    // State
    status,
    suggestions,
    error,
    isDismissed,
    
    // Derived state
    unappliedAnalyzers,
    unappliedCustomMetrics,
    unappliedTests,
    hasUnappliedSuggestions,
    
    // Applied tracking
    appliedAnalyzers,
    appliedCustomMetrics,
    appliedTests,
    
    // Actions
    triggerSuggestions,
    markAnalyzerApplied,
    markCustomMetricApplied,
    markTestApplied,
    markAllAnalyzersApplied,
    markAllCustomMetricsApplied,
    markAllTestsApplied,
    dismiss,
    undismiss,
    reset,
  }
}
