/**
 * TypeScript types matching Python Pydantic models from storage.py and testing/results.py
 */

// Severity levels for test failures
export type TestSeverity = 'high' | 'medium' | 'low'

// Metadata for a saved eval
export interface EvalMetadata {
  id: string
  name: string
  config_path: string | null
  created_at: string
  dataset_path: string | null
  sample_count: number
  pass_rate: number | null
  analyzer_count: number
  test_count: number
  tests_passed: number
  tests_failed: number
}

// Result of a single test execution
export interface TestResult {
  test_id: string
  passed: boolean
  severity: TestSeverity
  title: string
  description: string
  metric: string
  affected_count: number
  total_count: number
  affected_percentage: number
  threshold: number | null
  actual_value: number | null
  sample_indices: number[]
  error: string | null
  details: Record<string, unknown>
}

// Summary of all test results
export interface TestSummary {
  results: TestResult[]
  total_tests: number
  passed_tests: number
  failed_tests: number
  error_tests: number
  pass_rate: number
  high_severity_failures: number
  medium_severity_failures: number
  low_severity_failures: number
}

// Analysis result from an analyzer (generic structure)
export interface AnalysisResult {
  // Common fields
  score?: number
  passed?: boolean
  label?: string
  reasoning?: string
  category?: string
  error?: string
  
  // Length analyzer fields
  total_tokens?: number
  num_messages?: number
  avg_tokens_per_message?: number
  
  // Allow additional fields
  [key: string]: unknown
}

// Message in a conversation
export interface Message {
  role: 'user' | 'assistant' | 'system' | string
  content: string | MessageContent[]
}

// For multimodal content
export interface MessageContent {
  type?: string
  text?: string
  image_url?: string
}

// Conversation structure
export interface Conversation {
  messages: Message[]
  metadata?: Record<string, unknown>
}

// Full eval data including results
export interface EvalData {
  metadata: EvalMetadata
  config: Record<string, unknown>
  analysis_results: Record<string, AnalysisResult[]>
  test_results: TestSummary | { results: TestResult[] } | { tests: TestResult[] }
  conversations: Conversation[]
}

// Helper to get test results array from different formats
export function getTestResults(testResults: EvalData['test_results']): TestResult[] {
  if ('results' in testResults && Array.isArray(testResults.results)) {
    return testResults.results
  }
  if ('tests' in testResults && Array.isArray(testResults.tests)) {
    return testResults.tests
  }
  return []
}

// Helper to compute test summary from results
export function computeTestSummary(testResults: EvalData['test_results']): TestSummary {
  const results = getTestResults(testResults)
  const total = results.length
  const passed = results.filter(r => r.passed && !r.error).length
  const errors = results.filter(r => r.error).length
  const failed = total - passed - errors
  
  return {
    results,
    total_tests: total,
    passed_tests: passed,
    failed_tests: failed,
    error_tests: errors,
    pass_rate: total > 0 ? Math.round(100 * passed / total * 10) / 10 : 0,
    high_severity_failures: results.filter(r => !r.passed && r.severity === 'high').length,
    medium_severity_failures: results.filter(r => !r.passed && r.severity === 'medium').length,
    low_severity_failures: results.filter(r => !r.passed && r.severity === 'low').length,
  }
}

// Index file structure
export interface EvalIndex {
  evals: EvalMetadata[]
}
