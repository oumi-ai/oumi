import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useState, useEffect, useCallback } from 'react'
import type { EvalIndex, EvalData, EvalMetadata } from '@/types/eval'

const DATA_BASE_URL = '/data'
const API_BASE_URL = '/api'

/**
 * Job status from the backend
 */
export interface JobStatus {
  id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  total: number
  message: string
  error: string | null
  eval_id: string | null
  log_lines: string[]
}

/**
 * Fetch the index of all available evals
 */
async function fetchEvalIndex(): Promise<EvalIndex> {
  const response = await fetch(`${DATA_BASE_URL}/index.json`)
  if (!response.ok) {
    throw new Error(`Failed to fetch eval index: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Fetch a specific eval by ID
 */
async function fetchEval(evalId: string): Promise<EvalData> {
  const response = await fetch(`${DATA_BASE_URL}/evals/${evalId}.json`)
  if (!response.ok) {
    throw new Error(`Failed to fetch eval ${evalId}: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Hook to fetch the list of all available evals
 */
export function useEvalList() {
  return useQuery<EvalMetadata[], Error>({
    queryKey: ['evals'],
    queryFn: async () => {
      const index = await fetchEvalIndex()
      return index.evals ?? []
    },
  })
}

/**
 * Hook to fetch a specific eval by ID
 */
export function useEval(evalId: string | null) {
  return useQuery<EvalData, Error>({
    queryKey: ['eval', evalId],
    queryFn: () => {
      if (!evalId) throw new Error('No eval ID provided')
      return fetchEval(evalId)
    },
    enabled: !!evalId,
  })
}

/**
 * Start an analysis job
 */
async function startAnalysis(yamlConfig: string): Promise<{ job_id: string }> {
  const response = await fetch(`${API_BASE_URL}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config: yamlConfig }),
  })
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.error || `Failed to start analysis: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Run tests only, reusing cached analyzer results
 */
async function runTestsOnly(yamlConfig: string, parentEvalId: string): Promise<{ 
  status: string
  eval_id: string
  message: string 
}> {
  const response = await fetch(`${API_BASE_URL}/run-tests-only`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      config: yamlConfig,
      parent_eval_id: parentEvalId,
    }),
  })
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.error || `Failed to run tests: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Fetch job status
 */
async function fetchJobStatus(jobId: string): Promise<JobStatus> {
  const response = await fetch(`${API_BASE_URL}/jobs/${jobId}`)
  if (!response.ok) {
    throw new Error(`Failed to fetch job status: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Rename an eval
 */
async function renameEvalApi(evalId: string, newName: string): Promise<{ success: boolean }> {
  const response = await fetch(`${API_BASE_URL}/rename`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ eval_id: evalId, name: newName }),
  })
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.error || `Failed to rename: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Hook to rename an eval
 */
export function useRenameEval() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ evalId, newName }: { evalId: string; newName: string }) => 
      renameEvalApi(evalId, newName),
    onSuccess: (_, variables) => {
      // Refresh both evals list and the specific eval
      queryClient.invalidateQueries({ queryKey: ['evals'] })
      queryClient.invalidateQueries({ queryKey: ['eval', variables.evalId] })
    },
  })
}

/**
 * Delete an eval
 */
async function deleteEvalApi(evalId: string): Promise<{ success: boolean }> {
  const response = await fetch(`${API_BASE_URL}/delete`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ eval_id: evalId }),
  })
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.error || `Failed to delete: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Hook to delete an eval
 */
export function useDeleteEval() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (evalId: string) => deleteEvalApi(evalId),
    onSuccess: (_, evalId) => {
      // Refresh evals list and remove the specific eval from cache
      queryClient.invalidateQueries({ queryKey: ['evals'] })
      queryClient.removeQueries({ queryKey: ['eval', evalId] })
    },
  })
}

/**
 * Hook to run an analysis and track progress
 */
export function useRunAnalysis() {
  const queryClient = useQueryClient()
  const [jobId, setJobId] = useState<string | null>(null)
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const [isPolling, setIsPolling] = useState(false)
  const [isRunningTestsOnly, setIsRunningTestsOnly] = useState(false)

  // Poll for job status
  useEffect(() => {
    if (!jobId || !isPolling) return

    const pollInterval = setInterval(async () => {
      try {
        const status = await fetchJobStatus(jobId)
        setJobStatus(status)

        // Stop polling when job is done
        if (status.status === 'completed' || status.status === 'failed') {
          setIsPolling(false)
          // Refresh evals list on completion
          if (status.status === 'completed') {
            queryClient.invalidateQueries({ queryKey: ['evals'] })
          }
        }
      } catch (error) {
        console.error('Error polling job status:', error)
      }
    }, 1000) // Poll every second

    return () => clearInterval(pollInterval)
  }, [jobId, isPolling, queryClient])

  const startMutation = useMutation({
    mutationFn: startAnalysis,
    onSuccess: (data) => {
      setJobId(data.job_id)
      setJobStatus({
        id: data.job_id,
        status: 'pending',
        progress: 0,
        total: 100,
        message: 'Starting analysis...',
        error: null,
        eval_id: null,
        log_lines: [],
      })
      setIsPolling(true)
    },
  })

  // Mutation for running tests only (synchronous, no polling needed)
  const testsOnlyMutation = useMutation({
    mutationFn: ({ config, parentEvalId }: { config: string; parentEvalId: string }) => 
      runTestsOnly(config, parentEvalId),
    onSuccess: (data) => {
      setJobStatus({
        id: 'tests-only',
        status: 'completed',
        progress: 100,
        total: 100,
        message: data.message,
        error: null,
        eval_id: data.eval_id,
        log_lines: ['Tests re-run using cached analyzer results'],
      })
      queryClient.invalidateQueries({ queryKey: ['evals'] })
    },
    onError: (error: Error) => {
      setJobStatus({
        id: 'tests-only',
        status: 'failed',
        progress: 0,
        total: 100,
        message: 'Failed to run tests',
        error: error.message,
        eval_id: null,
        log_lines: [],
      })
    },
  })

  const run = useCallback((yamlConfig: string) => {
    setJobId(null)
    setJobStatus(null)
    setIsRunningTestsOnly(false)
    startMutation.mutate(yamlConfig)
  }, [startMutation])

  // Run tests only, reusing cached results from parent
  const runTestsOnlyCached = useCallback((yamlConfig: string, parentEvalId: string) => {
    setJobId(null)
    setJobStatus({
      id: 'tests-only',
      status: 'running',
      progress: 50,
      total: 100,
      message: 'Re-running tests with cached analyzer results...',
      error: null,
      eval_id: null,
      log_lines: [],
    })
    setIsRunningTestsOnly(true)
    testsOnlyMutation.mutate({ config: yamlConfig, parentEvalId })
  }, [testsOnlyMutation])

  const reset = useCallback(() => {
    setJobId(null)
    setJobStatus(null)
    setIsPolling(false)
    setIsRunningTestsOnly(false)
  }, [])

  return {
    run,
    runTestsOnlyCached,
    reset,
    jobId,
    jobStatus,
    isStarting: startMutation.isPending || testsOnlyMutation.isPending,
    startError: startMutation.error || testsOnlyMutation.error,
    isRunningTestsOnly,
  }
}
