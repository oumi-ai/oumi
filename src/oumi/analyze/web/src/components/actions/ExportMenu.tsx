import { useState } from 'react'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import type { EvalData } from '@/types/eval'
import { Download, FileJson, FileSpreadsheet, FileText, Check } from 'lucide-react'

interface ExportMenuProps {
  evalData: EvalData
}

function downloadFile(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

function convertToCSV(data: Record<string, unknown>[]): string {
  if (data.length === 0) return ''
  
  // Get all unique keys
  const keys = new Set<string>()
  data.forEach(item => {
    Object.keys(item).forEach(key => keys.add(key))
  })
  const headers = Array.from(keys)
  
  // Create CSV rows
  const rows = data.map(item => {
    return headers.map(header => {
      const value = item[header]
      if (value === null || value === undefined) return ''
      if (typeof value === 'object') return JSON.stringify(value)
      const str = String(value)
      // Escape quotes and wrap in quotes if needed
      if (str.includes(',') || str.includes('"') || str.includes('\n')) {
        return `"${str.replace(/"/g, '""')}"`
      }
      return str
    }).join(',')
  })
  
  return [headers.join(','), ...rows].join('\n')
}

export function ExportMenu({ evalData }: ExportMenuProps) {
  const [exportedFormat, setExportedFormat] = useState<string | null>(null)
  const evalName = evalData.metadata.name || evalData.metadata.id

  const handleExportJSON = () => {
    const content = JSON.stringify(evalData, null, 2)
    downloadFile(content, `${evalName}.json`, 'application/json')
    setExportedFormat('json')
    setTimeout(() => setExportedFormat(null), 2000)
  }

  const handleExportAnalysisCSV = () => {
    // Flatten analysis results into rows
    const rows: Record<string, unknown>[] = []
    
    Object.entries(evalData.analysis_results).forEach(([analyzer, results]) => {
      results.forEach((result, index) => {
        rows.push({
          sample_index: index,
          analyzer,
          ...result
        })
      })
    })
    
    const content = convertToCSV(rows)
    downloadFile(content, `${evalName}_analysis.csv`, 'text/csv')
    setExportedFormat('csv-analysis')
    setTimeout(() => setExportedFormat(null), 2000)
  }

  const handleExportTestsCSV = () => {
    const tests = 'results' in evalData.test_results 
      ? evalData.test_results.results 
      : 'tests' in evalData.test_results 
        ? evalData.test_results.tests 
        : []
    
    const rows = tests.map(test => ({
      test_id: test.test_id,
      title: test.title,
      passed: test.passed,
      severity: test.severity,
      metric: test.metric,
      affected_count: test.affected_count,
      total_count: test.total_count,
      affected_percentage: test.affected_percentage,
      threshold: test.threshold,
      error: test.error || ''
    }))
    
    const content = convertToCSV(rows)
    downloadFile(content, `${evalName}_tests.csv`, 'text/csv')
    setExportedFormat('csv-tests')
    setTimeout(() => setExportedFormat(null), 2000)
  }

  const handleExportConfig = () => {
    const content = JSON.stringify(evalData.config, null, 2)
    downloadFile(content, `${evalName}_config.json`, 'application/json')
    setExportedFormat('config')
    setTimeout(() => setExportedFormat(null), 2000)
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm">
          <Download className="h-4 w-4 mr-2" />
          Export
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        <DropdownMenuLabel>Export Format</DropdownMenuLabel>
        <DropdownMenuSeparator />
        
        <DropdownMenuItem onClick={handleExportJSON}>
          <FileJson className="h-4 w-4 mr-2" />
          Full Evaluation (JSON)
          {exportedFormat === 'json' && <Check className="h-4 w-4 ml-auto" />}
        </DropdownMenuItem>
        
        <DropdownMenuItem onClick={handleExportConfig}>
          <FileText className="h-4 w-4 mr-2" />
          Configuration (JSON)
          {exportedFormat === 'config' && <Check className="h-4 w-4 ml-auto" />}
        </DropdownMenuItem>
        
        <DropdownMenuSeparator />
        
        <DropdownMenuItem onClick={handleExportAnalysisCSV}>
          <FileSpreadsheet className="h-4 w-4 mr-2" />
          Analysis Results (CSV)
          {exportedFormat === 'csv-analysis' && <Check className="h-4 w-4 ml-auto" />}
        </DropdownMenuItem>
        
        <DropdownMenuItem onClick={handleExportTestsCSV}>
          <FileSpreadsheet className="h-4 w-4 mr-2" />
          Test Results (CSV)
          {exportedFormat === 'csv-tests' && <Check className="h-4 w-4 ml-auto" />}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
