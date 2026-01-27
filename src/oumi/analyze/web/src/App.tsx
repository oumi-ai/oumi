import { useState } from 'react'
import { useEvalList, useEval, useRunAnalysis, useRenameEval, useDeleteEval } from '@/hooks/useEvals'
import { Sidebar } from '@/components/layout/Sidebar'
import { Header } from '@/components/layout/Header'
import { ResultsView } from '@/components/results/ResultsView'
import { ChartsView } from '@/components/charts/ChartsView'
import { ConfigEditor } from '@/components/config/ConfigEditor'
import { SetupWizard } from '@/components/wizard/SetupWizard'
import { ExportMenu } from '@/components/actions/ExportMenu'
import { RunningOverlay } from '@/components/running/RunningOverlay'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Plus, BarChart3, FileCode, TestTube, PieChart, CheckCircle } from 'lucide-react'

function App() {
  const [selectedEvalId, setSelectedEvalId] = useState<string | null>(null)
  const [showWizard, setShowWizard] = useState(false)
  const [editConfig, setEditConfig] = useState<Record<string, unknown> | null>(null)
  const [isRunningFromConfig, setIsRunningFromConfig] = useState(false)
  const [showCopiedDialog, setShowCopiedDialog] = useState(false)
  const { data: evals, isLoading: evalsLoading, refetch } = useEvalList()
  const { data: evalData, isLoading: evalLoading } = useEval(selectedEvalId)
  const { run, reset, jobStatus } = useRunAnalysis()
  const renameEval = useRenameEval()
  const deleteEval = useDeleteEval()

  const handleRename = (newName: string) => {
    if (selectedEvalId) {
      renameEval.mutate({ evalId: selectedEvalId, newName })
    }
  }

  const handleDelete = (evalId?: string) => {
    const idToDelete = evalId || selectedEvalId
    if (idToDelete) {
      deleteEval.mutate(idToDelete, {
        onSuccess: () => {
          if (idToDelete === selectedEvalId) {
            setSelectedEvalId(null)
          }
        }
      })
    }
  }

  const handleEditInWizard = () => {
    if (evalData) {
      setEditConfig(evalData.config)
      setShowWizard(true)
    }
  }

  const handleWizardComplete = (yamlConfig: string) => {
    // Copy to clipboard and show notification
    navigator.clipboard.writeText(yamlConfig)
    setShowCopiedDialog(true)
  }

  const handleRunComplete = (evalId: string | null) => {
    // Refresh evals list and select the new eval
    refetch().then(() => {
      if (evalId) {
        setSelectedEvalId(evalId)
      }
      setShowWizard(false)
      setEditConfig(null)
      setIsRunningFromConfig(false)
    })
  }

  const handleCloseWizard = () => {
    setShowWizard(false)
    setEditConfig(null)
  }

  const handleRunFromConfig = (yamlConfig: string) => {
    setIsRunningFromConfig(true)
    run(yamlConfig)
  }

  const handleCancelRun = () => {
    setIsRunningFromConfig(false)
    reset()
  }

  // Handle job completion when running from config
  if (isRunningFromConfig && jobStatus?.status === 'completed') {
    // Auto-redirect after completion
    setTimeout(() => {
      handleRunComplete(jobStatus.eval_id)
    }, 1500)
  }

  // Show wizard view
  if (showWizard) {
    return (
      <div className="flex h-screen bg-background">
        <Sidebar
          evals={evals ?? []}
          selectedId={selectedEvalId}
          onSelect={(id) => {
            setSelectedEvalId(id)
            handleCloseWizard()
          }}
          onDelete={handleDelete}
          isLoading={evalsLoading}
          onLogoClick={() => setSelectedEvalId(null)}
          onNewAnalysis={() => {
            setEditConfig(null)
            setShowWizard(true)
          }}
        />
        <main className="flex-1 flex flex-col overflow-auto p-6">
          <SetupWizard
            onComplete={handleWizardComplete}
            onRunComplete={handleRunComplete}
            onCancel={handleCloseWizard}
            initialConfig={editConfig ?? undefined}
          />
        </main>
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-background">
      <Sidebar
        evals={evals ?? []}
        selectedId={selectedEvalId}
        onSelect={setSelectedEvalId}
        onDelete={handleDelete}
        isLoading={evalsLoading}
        onNewAnalysis={() => setShowWizard(true)}
        onLogoClick={() => setSelectedEvalId(null)}
      />
      
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Running overlay when re-running from config */}
        {isRunningFromConfig && (
          <RunningOverlay 
            jobStatus={jobStatus} 
            onCancel={handleCancelRun}
            onRetry={() => evalData && handleRunFromConfig(JSON.stringify(evalData.config))}
          />
        )}

        {selectedEvalId && evalData ? (
          <>
            <Header evalData={evalData} onRename={handleRename} onDelete={() => handleDelete()}>
              <ExportMenu evalData={evalData} />
            </Header>
            <div className="flex-1 overflow-auto p-6">
              <Tabs defaultValue="results" className="w-full">
                <TabsList className="mb-4">
                  <TabsTrigger value="results" className="gap-2">
                    <TestTube className="h-4 w-4" />
                    Results
                  </TabsTrigger>
                  <TabsTrigger value="charts" className="gap-2">
                    <PieChart className="h-4 w-4" />
                    Charts
                  </TabsTrigger>
                  <TabsTrigger value="config" className="gap-2">
                    <FileCode className="h-4 w-4" />
                    Config
                  </TabsTrigger>
                </TabsList>
                <TabsContent value="results">
                  <ResultsView evalData={evalData} />
                </TabsContent>
                <TabsContent value="charts">
                  <ChartsView evalData={evalData} />
                </TabsContent>
                <TabsContent value="config">
                  <ConfigEditor 
                    evalData={evalData} 
                    onRunAnalysis={handleRunFromConfig} 
                    onEditInWizard={handleEditInWizard}
                  />
                </TabsContent>
              </Tabs>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center max-w-md">
              <BarChart3 className="h-16 w-16 mx-auto mb-4 text-muted-foreground/50" />
              <h2 className="text-2xl font-semibold mb-2">Oumi Analyze</h2>
              <p className="text-muted-foreground mb-6">
                {evalsLoading || evalLoading
                  ? 'Loading evaluations...'
                  : 'Select an evaluation from the sidebar or create a new analysis to get started.'}
              </p>
              {!evalsLoading && !evalLoading && (
                <Button onClick={() => setShowWizard(true)} size="lg">
                  <Plus className="h-5 w-5 mr-2" />
                  Create New Analysis
                </Button>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Config Copied Success Dialog */}
      <Dialog open={showCopiedDialog} onOpenChange={setShowCopiedDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <CheckCircle className="h-5 w-5 text-green-500" />
              Configuration Copied
            </DialogTitle>
            <DialogDescription className="pt-2">
              Your configuration has been copied to the clipboard.
            </DialogDescription>
          </DialogHeader>
          <div className="bg-muted p-3 rounded-md">
            <p className="text-sm font-medium mb-1">Run it with:</p>
            <code className="text-xs bg-background p-2 rounded block">
              oumi analyze --config your_config.yaml --typed
            </code>
          </div>
          <DialogFooter>
            <Button onClick={() => setShowCopiedDialog(false)}>
              Done
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default App
