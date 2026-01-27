import { useState, useCallback } from 'react'
import { useEvalList, useEval, useRunAnalysis, useRenameEval, useDeleteEval } from '@/hooks/useEvals'
import yaml from 'js-yaml'
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
  const [wizardInitialStep, setWizardInitialStep] = useState(0)
  const [isRunningFromConfig, setIsRunningFromConfig] = useState(false)
  const [showCopiedDialog, setShowCopiedDialog] = useState(false)
  const { data: evals, isLoading: evalsLoading, refetch } = useEvalList()
  const { data: evalData, isLoading: evalLoading } = useEval(selectedEvalId)
  const { run, runTestsOnlyCached, reset, jobStatus } = useRunAnalysis()
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
      // Extract base name by stripping any existing version suffix (_v2, _v3, etc.) or timestamps
      const currentName = evalData.metadata.name
      const baseName = currentName
        .replace(/_v\d+$/, '')  // Remove _v2, _v3, etc.
        .replace(/_\d{4}-\d{2}-\d{2}_\d{4}$/, '')  // Remove timestamp suffix
        .replace(/_\d{4}-\d{2}-\d{2}_\d{4}_\d{4}-\d{2}-\d{2}_\d{4}$/, '')  // Remove double timestamps
      
      // Find the highest existing version number for this base name
      const versionPattern = new RegExp(`^${baseName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}_v(\\d+)$`)
      let maxVersion = 1
      for (const evalItem of evals || []) {
        // Check if this eval matches the base name directly (it's version 1)
        if (evalItem.name === baseName) {
          maxVersion = Math.max(maxVersion, 1)
        }
        // Check if this eval has a version suffix
        const match = evalItem.name.match(versionPattern)
        if (match) {
          maxVersion = Math.max(maxVersion, parseInt(match[1], 10))
        }
      }
      
      // Create new name with next version number
      const newVersion = maxVersion + 1
      const newName = `${baseName}_v${newVersion}`
      
      const configWithName = {
        ...evalData.config,
        eval_name: newName,
        // Store parent reference for linking
        parent_eval_id: evalData.metadata.id,
      }
      setEditConfig(configWithName)
      setWizardInitialStep(0)
      setShowWizard(true)
    }
  }

  // Open wizard at tests step for quick test editing
  const handleEditTests = () => {
    if (evalData) {
      // Same name generation logic as handleEditInWizard
      const currentName = evalData.metadata.name
      const baseName = currentName
        .replace(/_v\d+$/, '')
        .replace(/_\d{4}-\d{2}-\d{2}_\d{4}$/, '')
        .replace(/_\d{4}-\d{2}-\d{2}_\d{4}_\d{4}-\d{2}-\d{2}_\d{4}$/, '')
      
      const versionPattern = new RegExp(`^${baseName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}_v(\\d+)$`)
      let maxVersion = 1
      for (const evalItem of evals || []) {
        if (evalItem.name === baseName) {
          maxVersion = Math.max(maxVersion, 1)
        }
        const match = evalItem.name.match(versionPattern)
        if (match) {
          maxVersion = Math.max(maxVersion, parseInt(match[1], 10))
        }
      }
      
      const newVersion = maxVersion + 1
      const newName = `${baseName}_v${newVersion}`
      
      const configWithName = {
        ...evalData.config,
        eval_name: newName,
        parent_eval_id: evalData.metadata.id,
      }
      setEditConfig(configWithName)
      setWizardInitialStep(2) // Start at Tests step
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
    setWizardInitialStep(0)
  }

  // Check if only tests changed between two configs
  // We compare: analyzer ids/instance_ids, custom_metric ids, and dataset settings
  // We intentionally DON'T compare params deeply (whitespace in multiline strings causes false negatives)
  const onlyTestsChanged = useCallback((oldConfig: Record<string, unknown>, newConfig: Record<string, unknown>): boolean => {
    // Compare analyzers by id and instance_id only (not params - too fragile with multiline strings)
    const oldAnalyzers = (oldConfig.analyzers as Array<{id: string; instance_id?: string}>) || []
    const newAnalyzers = (newConfig.analyzers as Array<{id: string; instance_id?: string}>) || []
    
    if (oldAnalyzers.length !== newAnalyzers.length) {
      console.log('Analyzer count mismatch:', oldAnalyzers.length, 'vs', newAnalyzers.length)
      return false
    }
    
    // Sort both arrays by instance_id for comparison
    const sortedOld = [...oldAnalyzers].sort((a, b) => (a.instance_id || a.id || '').localeCompare(b.instance_id || b.id || ''))
    const sortedNew = [...newAnalyzers].sort((a, b) => (a.instance_id || a.id || '').localeCompare(b.instance_id || b.id || ''))
    
    for (let i = 0; i < sortedOld.length; i++) {
      const oldA = sortedOld[i]
      const newA = sortedNew[i]
      
      if (oldA.id !== newA.id) {
        console.log('Analyzer id mismatch:', oldA.id, 'vs', newA.id)
        return false
      }
      if ((oldA.instance_id || oldA.id) !== (newA.instance_id || newA.id)) {
        console.log('Analyzer instance_id mismatch:', oldA.instance_id, 'vs', newA.instance_id)
        return false
      }
    }
    
    // Compare custom_metrics by ID only (not function content - too fragile)
    const oldCustomMetrics = (oldConfig.custom_metrics as Array<{id: string}>) || []
    const newCustomMetrics = (newConfig.custom_metrics as Array<{id: string}>) || []
    
    if (oldCustomMetrics.length !== newCustomMetrics.length) {
      console.log('Custom metrics count mismatch:', oldCustomMetrics.length, 'vs', newCustomMetrics.length)
      return false
    }
    
    const oldMetricIds = oldCustomMetrics.map(m => m.id).sort()
    const newMetricIds = newCustomMetrics.map(m => m.id).sort()
    
    for (let i = 0; i < oldMetricIds.length; i++) {
      if (oldMetricIds[i] !== newMetricIds[i]) {
        console.log('Custom metric id mismatch:', oldMetricIds[i], 'vs', newMetricIds[i])
        return false
      }
    }
    
    // Compare dataset settings (must be the same)
    if (oldConfig.dataset_path !== newConfig.dataset_path) {
      console.log('Dataset path mismatch:', oldConfig.dataset_path, 'vs', newConfig.dataset_path)
      return false
    }
    if (oldConfig.dataset_name !== newConfig.dataset_name) {
      console.log('Dataset name mismatch:', oldConfig.dataset_name, 'vs', newConfig.dataset_name)
      return false
    }
    if (oldConfig.sample_count !== newConfig.sample_count) {
      console.log('Sample count mismatch:', oldConfig.sample_count, 'vs', newConfig.sample_count)
      return false
    }
    
    return true
  }, [])

  const handleRunFromConfig = useCallback((yamlConfig: string) => {
    setIsRunningFromConfig(true)
    
    // Try to detect if only tests changed
    if (evalData && selectedEvalId) {
      try {
        const newConfig = yaml.load(yamlConfig) as Record<string, unknown>
        const oldConfig = evalData.config
        
        console.log('=== Comparing configs ===')
        console.log('Old analyzers:', oldConfig.analyzers)
        console.log('New analyzers:', newConfig.analyzers)
        console.log('Old custom_metrics:', oldConfig.custom_metrics)
        console.log('New custom_metrics:', newConfig.custom_metrics)
        console.log('Old dataset_path:', oldConfig.dataset_path)
        console.log('New dataset_path:', newConfig.dataset_path)
        
        const testsOnly = onlyTestsChanged(oldConfig, newConfig)
        console.log('Only tests changed?', testsOnly)
        
        if (testsOnly) {
          console.log('✅ Using cached results - calling runTestsOnlyCached')
          runTestsOnlyCached(yamlConfig, selectedEvalId)
          return
        } else {
          console.log('❌ Config changed - running full analysis')
        }
      } catch (e) {
        console.warn('Could not parse config for comparison:', e)
      }
    } else {
      console.log('No evalData or selectedEvalId - running full analysis')
    }
    
    // Run full analysis
    run(yamlConfig)
  }, [evalData, selectedEvalId, onlyTestsChanged, run, runTestsOnlyCached])

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
            initialStep={wizardInitialStep}
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
                  <ResultsView evalData={evalData} onEditTests={handleEditTests} />
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
