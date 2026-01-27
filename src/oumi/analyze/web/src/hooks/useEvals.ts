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
 * Hook to run an analysis and track progress
 */
export function useRunAnalysis() {
  const queryClient = useQueryClient()
  const [jobId, setJobId] = useState<string | null>(null)
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const [isPolling, setIsPolling] = useState(false)

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

  const run = useCallback((yamlConfig: string) => {
    setJobId(null)
    setJobStatus(null)
    startMutation.mutate(yamlConfig)
  }, [startMutation])

  const reset = useCallback(() => {
    setJobId(null)
    setJobStatus(null)
    setIsPolling(false)
  }, [])

  return {
    run,
    reset,
    jobId,
    jobStatus,
    isStarting: startMutation.isPending,
    startError: startMutation.error,
  }
}
