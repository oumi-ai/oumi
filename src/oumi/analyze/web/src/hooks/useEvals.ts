import { useQuery } from '@tanstack/react-query'
import type { EvalIndex, EvalData, EvalMetadata } from '@/types/eval'

const DATA_BASE_URL = '/data'

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
