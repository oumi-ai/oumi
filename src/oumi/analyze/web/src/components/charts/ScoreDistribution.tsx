import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type { AnalysisResult } from '@/types/eval'

interface ScoreDistributionProps {
  analyzerName: string
  results: AnalysisResult[]
}

export function ScoreDistribution({ analyzerName, results }: ScoreDistributionProps) {
  // Extract scores from results
  const scores = results
    .map((r) => r.score)
    .filter((s): s is number => s !== undefined && s !== null)

  if (scores.length === 0) {
    return null
  }

  // Create histogram bins (0-10, 10-20, ..., 90-100)
  const bins = Array.from({ length: 10 }, (_, i) => ({
    range: `${i * 10}-${(i + 1) * 10}`,
    count: 0,
  }))

  scores.forEach((score) => {
    const binIndex = Math.min(Math.floor(score / 10), 9)
    bins[binIndex].count++
  })

  // Calculate stats
  const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length
  const minScore = Math.min(...scores)
  const maxScore = Math.max(...scores)

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{analyzerName} - Score Distribution</CardTitle>
        <div className="flex gap-4 text-sm text-muted-foreground">
          <span>Avg: {avgScore.toFixed(1)}</span>
          <span>Min: {minScore.toFixed(1)}</span>
          <span>Max: {maxScore.toFixed(1)}</span>
          <span>N: {scores.length}</span>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={bins}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis
                dataKey="range"
                className="text-xs"
                tick={{ fill: 'hsl(var(--muted-foreground))' }}
              />
              <YAxis
                className="text-xs"
                tick={{ fill: 'hsl(var(--muted-foreground))' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--popover))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '0.5rem',
                }}
                labelStyle={{ color: 'hsl(var(--popover-foreground))' }}
              />
              <Bar
                dataKey="count"
                fill="hsl(var(--primary))"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}
