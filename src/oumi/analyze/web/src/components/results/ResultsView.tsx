import { TestSummary } from './TestSummary'
import { TestCard } from './TestCard'
import type { EvalData } from '@/types/eval'
import { getTestResults, computeTestSummary } from '@/types/eval'
import { AlertTriangle, CheckCircle2, Settings2 } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface ResultsViewProps {
  evalData: EvalData
  onEditTests?: () => void
}

export function ResultsView({ evalData, onEditTests }: ResultsViewProps) {
  const testResults = getTestResults(evalData.test_results)
  const summary = computeTestSummary(evalData.test_results)

  const failedTests = testResults.filter(t => !t.passed)
  const passedTests = testResults.filter(t => t.passed)

  if (testResults.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-muted-foreground">No tests configured for this evaluation.</p>
        <p className="text-sm text-muted-foreground mt-2">
          Add tests to your configuration to validate analysis results.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <TestSummary summary={summary} />

      {/* Failed Tests Section */}
      {failedTests.length > 0 && (
        <section>
          <div className="flex items-center justify-between mb-4">
            <h2 className="flex items-center gap-2 text-lg font-semibold text-red-600">
              <AlertTriangle className="h-5 w-5" />
              Failed Tests ({failedTests.length})
            </h2>
            {onEditTests && (
              <Button variant="outline" size="sm" onClick={onEditTests}>
                <Settings2 className="h-4 w-4 mr-2" />
                Edit Tests
              </Button>
            )}
          </div>
          <div className="space-y-4">
            {failedTests.map(test => (
              <TestCard 
                key={test.test_id} 
                test={test} 
                evalData={evalData}
              />
            ))}
          </div>
        </section>
      )}

      {/* Passed Tests Section */}
      {passedTests.length > 0 && (
        <section>
          <div className="flex items-center justify-between mb-4">
            <h2 className="flex items-center gap-2 text-lg font-semibold text-green-600">
              <CheckCircle2 className="h-5 w-5" />
              Passed Tests ({passedTests.length})
            </h2>
            {/* Show Edit Tests button here if no failed tests */}
            {onEditTests && failedTests.length === 0 && (
              <Button variant="outline" size="sm" onClick={onEditTests}>
                <Settings2 className="h-4 w-4 mr-2" />
                Edit Tests
              </Button>
            )}
          </div>
          <div className="space-y-4">
            {passedTests.map(test => (
              <TestCard 
                key={test.test_id} 
                test={test} 
                evalData={evalData}
              />
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
