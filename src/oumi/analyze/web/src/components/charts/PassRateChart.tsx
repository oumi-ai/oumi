import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type { AnalysisResult } from '@/types/eval'

interface PassRateChartProps {
  analysisResults: Record<string, AnalysisResult[]>
}

export function PassRateChart({ analysisResults }: PassRateChartProps) {
  // Calculate pass rate for each analyzer
  const data = Object.entries(analysisResults)
    .map(([name, results]) => {
      const passedResults = results.filter((r) => r.passed === true)
      const failedResults = results.filter((r) => r.passed === false)
      const totalWithPassedField = passedResults.length + failedResults.length

      if (totalWithPassedField === 0) {
        return null
      }

      const passRate = (passedResults.length / totalWithPassedField) * 100

      return {
        name,
        passRate: Math.round(passRate * 10) / 10,
        passed: passedResults.length,
        total: totalWithPassedField,
      }
    })
    .filter((d): d is NonNullable<typeof d> => d !== null)

  if (data.length === 0) {
    return null
  }

  // Color based on pass rate
  const getColor = (passRate: number) => {
    if (passRate >= 80) return 'hsl(142, 76%, 36%)' // green
    if (passRate >= 50) return 'hsl(45, 93%, 47%)' // yellow
    return 'hsl(0, 84%, 60%)' // red
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Pass Rate by Analyzer</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis
                type="number"
                domain={[0, 100]}
                className="text-xs"
                tick={{ fill: 'hsl(var(--muted-foreground))' }}
                tickFormatter={(v) => `${v}%`}
              />
              <YAxis
                type="category"
                dataKey="name"
                className="text-xs"
                tick={{ fill: 'hsl(var(--muted-foreground))' }}
                width={120}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--popover))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '0.5rem',
                }}
                labelStyle={{ color: 'hsl(var(--popover-foreground))' }}
                formatter={(value: number, _name, props) => [
                  `${value}% (${props.payload.passed}/${props.payload.total})`,
                  'Pass Rate',
                ]}
              />
              <Bar dataKey="passRate" radius={[0, 4, 4, 0]}>
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getColor(entry.passRate)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}
