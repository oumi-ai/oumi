import { ScoreDistribution } from './ScoreDistribution'
import { PassRateChart } from './PassRateChart'
import type { EvalData } from '@/types/eval'

interface ChartsViewProps {
  evalData: EvalData
}

export function ChartsView({ evalData }: ChartsViewProps) {
  const { analysis_results } = evalData

  // Get analyzers that have score fields
  const analyzersWithScores = Object.entries(analysis_results).filter(
    ([, results]) => results.some((r) => r.score !== undefined)
  )

  // Check if any analyzer has passed/failed data
  const hasPassedData = Object.values(analysis_results).some((results) =>
    results.some((r) => r.passed !== undefined)
  )

  if (analyzersWithScores.length === 0 && !hasPassedData) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        <p>No chart data available for this evaluation.</p>
        <p className="text-sm mt-2">
          Charts are generated from analyzers that produce scores or pass/fail results.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Pass Rate Chart */}
      {hasPassedData && (
        <PassRateChart analysisResults={analysis_results} />
      )}

      {/* Score Distributions */}
      {analyzersWithScores.length > 0 && (
        <div>
          <h2 className="text-lg font-semibold mb-4">Score Distributions</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {analyzersWithScores.map(([name, results]) => (
              <ScoreDistribution
                key={name}
                analyzerName={name}
                results={results}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
