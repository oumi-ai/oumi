import { useState } from 'react'
import { useEvalList, useEval } from '@/hooks/useEvals'
import { Sidebar } from '@/components/layout/Sidebar'
import { Header } from '@/components/layout/Header'
import { ResultsView } from '@/components/results/ResultsView'
import { ChartsView } from '@/components/charts/ChartsView'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

function App() {
  const [selectedEvalId, setSelectedEvalId] = useState<string | null>(null)
  const { data: evals, isLoading: evalsLoading } = useEvalList()
  const { data: evalData, isLoading: evalLoading } = useEval(selectedEvalId)

  return (
    <div className="flex h-screen bg-background">
      <Sidebar
        evals={evals ?? []}
        selectedId={selectedEvalId}
        onSelect={setSelectedEvalId}
        isLoading={evalsLoading}
      />
      
      <main className="flex-1 flex flex-col overflow-hidden">
        {selectedEvalId && evalData ? (
          <>
            <Header evalData={evalData} />
            <div className="flex-1 overflow-auto p-6">
              <Tabs defaultValue="results" className="w-full">
                <TabsList className="mb-4">
                  <TabsTrigger value="results">Results</TabsTrigger>
                  <TabsTrigger value="charts">Charts</TabsTrigger>
                </TabsList>
                <TabsContent value="results">
                  <ResultsView evalData={evalData} />
                </TabsContent>
                <TabsContent value="charts">
                  <ChartsView evalData={evalData} />
                </TabsContent>
              </Tabs>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <h2 className="text-2xl font-semibold mb-2">Oumi Analyze</h2>
              <p className="text-muted-foreground">
                {evalsLoading || evalLoading
                  ? 'Loading...'
                  : 'Select an evaluation from the sidebar to view results'}
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
