import { useState, useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from 'recharts'
import type { EvalData, AnalysisResult } from '@/types/eval'

interface ChartsViewProps {
  evalData: EvalData
}

// Theme-consistent color palette for charts
const CHART_COLORS = [
  'hsl(217 91% 60%)',    // Vibrant blue
  'hsl(160 60% 45%)',    // Teal
  'hsl(30 80% 55%)',     // Orange
  'hsl(280 65% 60%)',    // Purple
  'hsl(340 75% 55%)',    // Pink
  'hsl(45 90% 50%)',     // Yellow
  'hsl(120 60% 45%)',    // Green
  'hsl(190 70% 50%)',    // Cyan
]

// Primary color for histograms
const HISTOGRAM_COLOR = 'hsl(217 91% 60%)' // Vibrant blue

// Tooltip style for consistency
const tooltipStyle = {
  contentStyle: {
    backgroundColor: 'hsl(var(--popover))',
    border: '1px solid hsl(var(--border))',
    borderRadius: '0.5rem',
  },
  labelStyle: { color: 'hsl(var(--popover-foreground))' },
  itemStyle: { color: 'hsl(var(--popover-foreground))' },
}

// Helper to get numeric fields from results
function getNumericFields(results: AnalysisResult[]): string[] {
  if (results.length === 0) return []
  
  const sample = results[0]
  const numericFields: string[] = []
  
  for (const [key, value] of Object.entries(sample)) {
    // Include all numeric fields including score
    if (typeof value === 'number') {
      numericFields.push(key)
    }
  }
  
  return numericFields
}

// Helper to get categorical fields from results
function getCategoricalFields(results: AnalysisResult[]): string[] {
  if (results.length === 0) return []
  
  const sample = results[0]
  const categoricalFields: string[] = []
  
  for (const [key, value] of Object.entries(sample)) {
    if (typeof value === 'string' && !['reasoning', 'error'].includes(key)) {
      categoricalFields.push(key)
    }
    if (typeof value === 'boolean') {
      categoricalFields.push(key)
    }
  }
  
  return categoricalFields
}

// Create histogram data from numeric values
function createHistogramData(values: number[], bins: number = 10): { range: string; count: number }[] {
  if (values.length === 0) return []
  
  const min = Math.min(...values)
  const max = Math.max(...values)
  
  if (min === max) {
    return [{ range: String(min), count: values.length }]
  }
  
  const binWidth = (max - min) / bins
  const histogram: { range: string; count: number }[] = []
  
  for (let i = 0; i < bins; i++) {
    const binStart = min + i * binWidth
    const binEnd = min + (i + 1) * binWidth
    const count = values.filter(v => 
      i === bins - 1 ? v >= binStart && v <= binEnd : v >= binStart && v < binEnd
    ).length
    
    histogram.push({
      range: `${Math.round(binStart)}-${Math.round(binEnd)}`,
      count,
    })
  }
  
  return histogram.filter(h => h.count > 0)
}

// Create pie chart data from categorical values
function createPieData(values: (string | boolean)[]): { name: string; value: number }[] {
  const counts: Record<string, number> = {}
  
  for (const value of values) {
    const key = String(value)
    counts[key] = (counts[key] || 0) + 1
  }
  
  return Object.entries(counts).map(([name, value]) => ({ name, value }))
}

export function ChartsView({ evalData }: ChartsViewProps) {
  const { analysis_results } = evalData
  const analyzerNames = Object.keys(analysis_results)
  
  const [selectedAnalyzer, setSelectedAnalyzer] = useState<string>(
    analyzerNames.length > 0 ? analyzerNames[0] : ''
  )

  // Get fields for selected analyzer
  const selectedResults = analysis_results[selectedAnalyzer] || []
  const numericFields = useMemo(() => getNumericFields(selectedResults), [selectedResults])
  const categoricalFields = useMemo(() => getCategoricalFields(selectedResults), [selectedResults])

  if (analyzerNames.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        <p>No analysis results available.</p>
        <p className="text-sm mt-2">
          Run an analysis to see charts and visualizations.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Analyzer-specific charts */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Analyzer Metrics</CardTitle>
            <div className="flex items-center gap-2">
              <Label htmlFor="analyzer-select" className="text-sm">Analyzer:</Label>
              <Select value={selectedAnalyzer} onValueChange={setSelectedAnalyzer}>
                <SelectTrigger className="w-48" id="analyzer-select">
                  <SelectValue placeholder="Select analyzer" />
                </SelectTrigger>
                <SelectContent>
                  {analyzerNames.map((name) => (
                    <SelectItem key={name} value={name}>
                      {name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {numericFields.length === 0 && categoricalFields.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <p>No chartable fields for this analyzer.</p>
            </div>
          ) : (
            <div className="space-y-8">
              {/* Numeric field histograms */}
              {numericFields.length > 0 && (
                <div>
                  <h3 className="text-md font-medium mb-4">Numeric Distributions</h3>
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {numericFields.map((field) => {
                      const values = selectedResults
                        .map(r => r[field])
                        .filter((v): v is number => typeof v === 'number')
                      const histogramData = createHistogramData(values)
                      const avgValue = values.reduce((a, b) => a + b, 0) / values.length
                      const minValue = Math.min(...values)
                      const maxValue = Math.max(...values)
                      
                      return (
                        <Card key={field}>
                          <CardHeader>
                            <CardTitle className="text-base">
                              {field.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                            </CardTitle>
                            <div className="flex gap-4 text-sm text-muted-foreground">
                              <span>Avg: {avgValue.toFixed(1)}</span>
                              <span>Min: {minValue.toFixed(1)}</span>
                              <span>Max: {maxValue.toFixed(1)}</span>
                              <span>N: {values.length}</span>
                            </div>
                          </CardHeader>
                          <CardContent>
                            <div className="h-64">
                              <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={histogramData}>
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
                                  <Tooltip {...tooltipStyle} />
                                  <Bar 
                                    dataKey="count" 
                                    fill={HISTOGRAM_COLOR}
                                    radius={[4, 4, 0, 0]}
                                  />
                                </BarChart>
                              </ResponsiveContainer>
                            </div>
                          </CardContent>
                        </Card>
                      )
                    })}
                  </div>
                </div>
              )}

              {/* Categorical field pie charts */}
              {categoricalFields.length > 0 && (
                <div>
                  <h3 className="text-md font-medium mb-4">Categorical Distributions</h3>
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {categoricalFields.map((field) => {
                      const values = selectedResults
                        .map(r => r[field])
                        .filter((v): v is string | boolean => 
                          typeof v === 'string' || typeof v === 'boolean'
                        )
                      const pieData = createPieData(values)
                      const total = pieData.reduce((sum, d) => sum + d.value, 0)
                      
                      return (
                        <Card key={field}>
                          <CardHeader>
                            <CardTitle className="text-base">
                              {field.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                            </CardTitle>
                            <div className="flex gap-4 text-sm text-muted-foreground">
                              <span>Categories: {pieData.length}</span>
                              <span>N: {total}</span>
                            </div>
                          </CardHeader>
                          <CardContent>
                            <div className="h-64">
                              <ResponsiveContainer width="100%" height="100%">
                                <PieChart>
                                  <Pie
                                    data={pieData}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={50}
                                    outerRadius={80}
                                    paddingAngle={2}
                                    dataKey="value"
                                    label={({ name, percent }) => 
                                      `${name} (${(percent * 100).toFixed(0)}%)`
                                    }
                                    labelLine={false}
                                  >
                                    {pieData.map((_, index) => (
                                      <Cell 
                                        key={`cell-${index}`} 
                                        fill={CHART_COLORS[index % CHART_COLORS.length]} 
                                      />
                                    ))}
                                  </Pie>
                                  <Tooltip {...tooltipStyle} />
                                  <Legend 
                                    wrapperStyle={{ color: 'hsl(var(--foreground))' }}
                                  />
                                </PieChart>
                              </ResponsiveContainer>
                            </div>
                          </CardContent>
                        </Card>
                      )
                    })}
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
