import { useState, useCallback, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Checkbox } from '@/components/ui/checkbox'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Progress } from '@/components/ui/progress'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import Editor from '@monaco-editor/react'
import { 
  ChevronRight, 
  ChevronLeft, 
  Database, 
  Gauge,
  TestTube,
  FileCode,
  Check,
  Plus,
  Trash2,
  Upload,
  Play,
  Loader2,
  CheckCircle2,
  XCircle,
  Terminal,
  X
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useRunAnalysis } from '@/hooks/useEvals'

// LLM analyzer types that need special handling (use id: llm with criteria param)
const LLM_ANALYZER_TYPES = ['usefulness', 'safety', 'coherence', 'factuality'] as const

// Available analyzers with their parameters and metrics
const AVAILABLE_ANALYZERS = {
  length: {
    name: 'Length Analyzer',
    description: 'Analyze text length in words, characters, and tokens',
    params: [
      { key: 'count_tokens', type: 'boolean', default: false, label: 'Count Tokens' },
      { key: 'tiktoken_encoding', type: 'string', default: 'cl100k_base', label: 'Tiktoken Encoding' },
      { key: 'compute_role_stats', type: 'boolean', default: false, label: 'Compute Role Stats' },
    ],
    metrics: ['total_words', 'total_chars', 'total_tokens', 'num_messages', 'avg_words_per_message']
  },
  usefulness: {
    name: 'Usefulness Analyzer',
    description: 'Evaluate response usefulness using LLM',
    params: [
      { key: 'model_name', type: 'string', default: 'gpt-4o-mini', label: 'Model' },
      { key: 'api_provider', type: 'select', options: ['openai', 'anthropic'] as const, default: 'openai', label: 'API Provider' },
      { key: 'target_scope', type: 'select', options: ['conversation', 'last_turn', 'first_user'] as const, default: 'conversation', label: 'Target Scope' },
    ],
    metrics: ['score', 'passed', 'label', 'reasoning']
  },
  safety: {
    name: 'Safety Analyzer',
    description: 'Check content for safety issues',
    params: [
      { key: 'model_name', type: 'string', default: 'gpt-4o-mini', label: 'Model' },
      { key: 'api_provider', type: 'select', options: ['openai', 'anthropic'] as const, default: 'openai', label: 'API Provider' },
    ],
    metrics: ['score', 'passed', 'label', 'reasoning']
  },
  coherence: {
    name: 'Coherence Analyzer',
    description: 'Evaluate conversation coherence',
    params: [
      { key: 'model_name', type: 'string', default: 'gpt-4o-mini', label: 'Model' },
      { key: 'api_provider', type: 'select', options: ['openai', 'anthropic'] as const, default: 'openai', label: 'API Provider' },
    ],
    metrics: ['score', 'passed', 'label', 'reasoning']
  },
  factuality: {
    name: 'Factuality Analyzer',
    description: 'Check factual accuracy of responses',
    params: [
      { key: 'model_name', type: 'string', default: 'gpt-4o-mini', label: 'Model' },
      { key: 'api_provider', type: 'select', options: ['openai', 'anthropic'] as const, default: 'openai', label: 'API Provider' },
    ],
    metrics: ['score', 'passed', 'label', 'reasoning']
  },
}

type AnalyzerKey = keyof typeof AVAILABLE_ANALYZERS

interface WizardConfig {
  name: string
  datasetPath: string
  datasetName: string
  sampleCount: number
  outputPath: string
  analyzers: {
    id: string
    type: AnalyzerKey
    params: Record<string, unknown>
  }[]
  tests: {
    id: string
    type: 'threshold' | 'percentage' | 'range'
    metric: string
    title: string
    description: string
    severity: 'low' | 'medium' | 'high'
    operator?: string
    value?: number
    minPercentage?: number
    maxPercentage?: number
    minValue?: number
    maxValue?: number
  }[]
}

interface SetupWizardProps {
  onComplete?: (yamlConfig: string) => void
  onRunComplete?: (evalId: string | null) => void
  onCancel?: () => void
  /** Initial config to edit (for edit mode) */
  initialConfig?: Record<string, unknown>
}

/** Migrate old param names to new ones */
function migrateParams(params: Record<string, unknown>): Record<string, unknown> {
  const migrated = { ...params }
  // Migrate 'model' -> 'model_name' for LLM analyzers
  if ('model' in migrated && !('model_name' in migrated)) {
    migrated.model_name = migrated.model
    delete migrated.model
  }
  // Add default api_provider if missing for LLM analyzers
  if ('model_name' in migrated && !('api_provider' in migrated)) {
    migrated.api_provider = 'openai'
  }
  return migrated
}

/** Parse a config object into WizardConfig format */
function parseConfigToWizard(config: Record<string, unknown>): WizardConfig {
  const wizardConfig: WizardConfig = {
    name: (config.eval_name as string) || '',
    datasetPath: (config.dataset_path as string) || '',
    datasetName: (config.dataset_name as string) || '',
    sampleCount: (config.sample_count as number) || 100,
    outputPath: (config.output_path as string) || './analysis_output',
    analyzers: [],
    tests: [],
  }

  // Parse analyzers
  const analyzers = config.analyzers as Array<Record<string, unknown>> | undefined
  if (analyzers && Array.isArray(analyzers)) {
    wizardConfig.analyzers = analyzers.map((a) => {
      let analyzerType = (a.id as string) || 'length'
      const rawParams = (a.params as Record<string, unknown>) || {}
      
      // For LLM analyzers (id: llm), use instance_id or criteria param as the type
      if (analyzerType === 'llm') {
        const instanceId = a.instance_id as string | undefined
        const criteria = rawParams.criteria as string | undefined
        analyzerType = instanceId || criteria || 'usefulness'
      }
      
      // Migrate old param names to new ones
      const params = migrateParams(rawParams)
      
      // Check if analyzer type is supported, fallback to length
      const isSupported = AVAILABLE_ANALYZERS[analyzerType as AnalyzerKey] !== undefined
      
      return {
        id: analyzerType,
        type: (isSupported ? analyzerType : 'length') as AnalyzerKey,
        params,
      }
    })
  }

  // Parse tests
  const tests = config.tests as Array<Record<string, unknown>> | undefined
  if (tests && Array.isArray(tests)) {
    wizardConfig.tests = tests.map((t) => ({
      id: (t.id as string) || `test_${Math.random().toString(36).slice(2, 6)}`,
      type: (t.type as 'threshold' | 'percentage' | 'range') || 'threshold',
      metric: (t.metric as string) || '',
      title: (t.title as string) || '',
      description: (t.description as string) || '',
      severity: (t.severity as 'low' | 'medium' | 'high') || 'medium',
      operator: t.operator as string | undefined,
      value: t.value as number | undefined,
      minPercentage: (t.min_percentage as number) ?? (t.minPercentage as number | undefined),
      maxPercentage: (t.max_percentage as number) ?? (t.maxPercentage as number | undefined),
      minValue: (t.min_value as number) ?? (t.minValue as number | undefined),
      maxValue: (t.max_value as number) ?? (t.maxValue as number | undefined),
    }))
  }

  return wizardConfig
}

const STEPS = [
  { id: 'dataset', title: 'Dataset', icon: Database },
  { id: 'analyzers', title: 'Analyzers', icon: Gauge },
  { id: 'tests', title: 'Tests', icon: TestTube },
  { id: 'review', title: 'Review', icon: FileCode },
]

function generateYaml(config: WizardConfig): string {
  const lines: string[] = []
  
  // Analysis name (optional)
  if (config.name.trim()) {
    lines.push(`eval_name: ${config.name.trim()}`)
  }
  
  // Dataset config
  if (config.datasetPath) {
    lines.push(`dataset_path: ${config.datasetPath}`)
  } else if (config.datasetName) {
    lines.push(`dataset_name: ${config.datasetName}`)
  }
  
  lines.push(`sample_count: ${config.sampleCount}`)
  lines.push(`output_path: ${config.outputPath}`)
  lines.push('')
  
  // Analyzers
  lines.push('analyzers:')
  config.analyzers.forEach(analyzer => {
    const isLlmAnalyzer = (LLM_ANALYZER_TYPES as readonly string[]).includes(analyzer.type)
    
    if (isLlmAnalyzer) {
      // For LLM analyzers, use id: llm with criteria param to ensure correct metric paths
      // This avoids the issue where UsefulnessAnalyzer etc. override analyzer_id
      lines.push(`  - id: llm`)
      lines.push(`    instance_id: ${analyzer.type}`)
      lines.push(`    params:`)
      lines.push(`      criteria: ${analyzer.type}`)
      const paramEntries = Object.entries(analyzer.params).filter(([, v]) => v !== undefined && v !== '')
      paramEntries.forEach(([key, value]) => {
        lines.push(`      ${key}: ${value}`)
      })
    } else {
      // For non-LLM analyzers (like length), use the type directly
      lines.push(`  - id: ${analyzer.type}`)
      lines.push(`    instance_id: ${analyzer.type}`)
      const paramEntries = Object.entries(analyzer.params).filter(([, v]) => v !== undefined && v !== '')
      if (paramEntries.length > 0) {
        lines.push('    params:')
        paramEntries.forEach(([key, value]) => {
          lines.push(`      ${key}: ${value}`)
        })
      }
    }
  })
  lines.push('')
  
  // Tests
  if (config.tests.length > 0) {
    lines.push('tests:')
    config.tests.forEach(test => {
      lines.push(`  - id: ${test.id}`)
      lines.push(`    type: ${test.type}`)
      lines.push(`    metric: ${test.metric}`)
      lines.push(`    title: "${test.title}"`)
      lines.push(`    description: "${test.description}"`)
      lines.push(`    severity: ${test.severity}`)
      
      if (test.type === 'threshold') {
        if (test.operator) lines.push(`    operator: "${test.operator}"`)
        if (test.value !== undefined) lines.push(`    value: ${test.value}`)
        if (test.minPercentage !== undefined) lines.push(`    min_percentage: ${test.minPercentage}`)
      } else if (test.type === 'percentage') {
        if (test.minPercentage !== undefined) lines.push(`    min_percentage: ${test.minPercentage}`)
        if (test.maxPercentage !== undefined) lines.push(`    max_percentage: ${test.maxPercentage}`)
      } else if (test.type === 'range') {
        if (test.minValue !== undefined) lines.push(`    min_value: ${test.minValue}`)
        if (test.maxValue !== undefined) lines.push(`    max_value: ${test.maxValue}`)
      }
    })
  }
  
  return lines.join('\n')
}

export function SetupWizard({ onComplete, onRunComplete, onCancel, initialConfig }: SetupWizardProps) {
  const [currentStep, setCurrentStep] = useState(0)
  const [isRunning, setIsRunning] = useState(false)
  const [config, setConfig] = useState<WizardConfig>(() => {
    if (initialConfig) {
      return parseConfigToWizard(initialConfig)
    }
    return {
      name: '',
      datasetPath: '',
      datasetName: '',
      sampleCount: 100,
      outputPath: './analysis_output',
      analyzers: [],
      tests: [],
    }
  })
  
  const isEditMode = !!initialConfig

  const { run, reset, jobStatus, isStarting } = useRunAnalysis()

  // Handle job completion
  useEffect(() => {
    if (jobStatus?.status === 'completed') {
      // Wait a moment to show the success state
      const timer = setTimeout(() => {
        onRunComplete?.(jobStatus.eval_id)
      }, 1500)
      return () => clearTimeout(timer)
    }
  }, [jobStatus?.status, jobStatus?.eval_id, onRunComplete])

  const updateConfig = useCallback((updates: Partial<WizardConfig>) => {
    setConfig(prev => ({ ...prev, ...updates }))
  }, [])

  const addAnalyzer = useCallback((type: AnalyzerKey) => {
    const analyzer = AVAILABLE_ANALYZERS[type]
    const defaultParams: Record<string, unknown> = {}
    analyzer.params.forEach(p => {
      defaultParams[p.key] = p.default
    })
    
    setConfig(prev => ({
      ...prev,
      analyzers: [
        ...prev.analyzers,
        {
          id: type,
          type,
          params: defaultParams
        }
      ]
    }))
  }, [])

  const removeAnalyzer = useCallback((index: number) => {
    setConfig(prev => ({
      ...prev,
      analyzers: prev.analyzers.filter((_, i) => i !== index)
    }))
  }, [])

  const updateAnalyzerParam = useCallback((index: number, key: string, value: unknown) => {
    setConfig(prev => ({
      ...prev,
      analyzers: prev.analyzers.map((a, i) => 
        i === index ? { ...a, params: { ...a.params, [key]: value } } : a
      )
    }))
  }, [])

  const addTest = useCallback(() => {
    const testId = `test_${config.tests.length + 1}`
    setConfig(prev => ({
      ...prev,
      tests: [
        ...prev.tests,
        {
          id: testId,
          type: 'threshold',
          metric: '',
          title: '',
          description: '',
          severity: 'medium',
          operator: '>',
          value: 0,
        }
      ]
    }))
  }, [config.tests.length])

  const removeTest = useCallback((index: number) => {
    setConfig(prev => ({
      ...prev,
      tests: prev.tests.filter((_, i) => i !== index)
    }))
  }, [])

  const updateTest = useCallback((index: number, updates: Partial<WizardConfig['tests'][0]>) => {
    setConfig(prev => ({
      ...prev,
      tests: prev.tests.map((t, i) => i === index ? { ...t, ...updates } : t)
    }))
  }, [])

  const getAvailableMetrics = useCallback(() => {
    const metrics: string[] = []
    config.analyzers.forEach(analyzer => {
      const analyzerDef = AVAILABLE_ANALYZERS[analyzer.type]
      analyzerDef.metrics.forEach(m => {
        // Use analyzer.type as the prefix since instance_id matches the type
        metrics.push(`${analyzer.type}.${m}`)
      })
    })
    return metrics
  }, [config.analyzers])

  const handleRunAnalysis = useCallback(() => {
    const yaml = generateYaml(config)
    setIsRunning(true)
    run(yaml)
  }, [config, run])

  const handleCopyConfig = useCallback(() => {
    const yaml = generateYaml(config)
    navigator.clipboard.writeText(yaml)
    onComplete?.(yaml)
  }, [config, onComplete])

  const handleBackFromRunning = useCallback(() => {
    setIsRunning(false)
    reset()
  }, [reset])

  const canProceed = () => {
    switch (currentStep) {
      case 0:
        return config.datasetPath || config.datasetName
      case 1:
        return config.analyzers.length > 0
      case 2:
        return true // Tests are optional
      case 3:
        return true
      default:
        return false
    }
  }

  const renderDatasetStep = () => (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium mb-2">Analysis Setup</h3>
        <p className="text-sm text-muted-foreground mb-4">
          Name your analysis and choose a dataset source.
        </p>
      </div>

      <div className="space-y-4">
        <div>
          <Label htmlFor="analysisName">Analysis Name</Label>
          <Input
            id="analysisName"
            placeholder="My Analysis"
            className="mt-1.5"
            value={config.name}
            onChange={(e) => updateConfig({ name: e.target.value })}
          />
          <p className="text-xs text-muted-foreground mt-1">
            Optional. Leave blank to auto-generate from config filename.
          </p>
        </div>

        <Separator />

        <div>
          <Label htmlFor="datasetPath">Local File Path</Label>
          <div className="flex gap-2 mt-1.5">
            <Input
              id="datasetPath"
              placeholder="/path/to/dataset.jsonl"
              value={config.datasetPath}
              onChange={(e) => updateConfig({ datasetPath: e.target.value, datasetName: '' })}
            />
            <Button variant="outline" size="icon">
              <Upload className="h-4 w-4" />
            </Button>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <Separator className="flex-1" />
          <span className="text-sm text-muted-foreground">or</span>
          <Separator className="flex-1" />
        </div>

        <div>
          <Label htmlFor="datasetName">HuggingFace Dataset</Label>
          <Input
            id="datasetName"
            placeholder="username/dataset_name"
            className="mt-1.5"
            value={config.datasetName}
            onChange={(e) => updateConfig({ datasetName: e.target.value, datasetPath: '' })}
          />
        </div>

        <Separator />

        <div className="grid grid-cols-2 gap-4">
          <div>
            <Label htmlFor="sampleCount">Sample Count</Label>
            <Input
              id="sampleCount"
              type="number"
              className="mt-1.5"
              value={config.sampleCount}
              onChange={(e) => updateConfig({ sampleCount: parseInt(e.target.value) || 100 })}
            />
          </div>
          <div>
            <Label htmlFor="outputPath">Output Path</Label>
            <Input
              id="outputPath"
              className="mt-1.5"
              value={config.outputPath}
              onChange={(e) => updateConfig({ outputPath: e.target.value })}
            />
          </div>
        </div>
      </div>
    </div>
  )

  const renderAnalyzersStep = () => (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium mb-2">Select Analyzers</h3>
        <p className="text-sm text-muted-foreground mb-4">
          Choose the analyzers you want to run on your dataset.
        </p>
      </div>

      {/* Available Analyzers */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {Object.entries(AVAILABLE_ANALYZERS).map(([key, analyzer]) => {
          const isSelected = config.analyzers.some(a => a.type === key)
          return (
            <Card
              key={key}
              className={cn(
                'cursor-pointer transition-colors',
                isSelected ? 'border-primary bg-primary/5' : 'hover:border-muted-foreground/50'
              )}
              onClick={() => !isSelected && addAnalyzer(key as AnalyzerKey)}
            >
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">{analyzer.name}</h4>
                    <p className="text-xs text-muted-foreground mt-1">
                      {analyzer.description}
                    </p>
                  </div>
                  {isSelected && (
                    <Badge variant="default" className="ml-2">
                      <Check className="h-3 w-3 mr-1" />
                      Added
                    </Badge>
                  )}
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Selected Analyzers Configuration */}
      {config.analyzers.length > 0 && (
        <>
          <Separator />
          <div>
            <h4 className="font-medium mb-3">Configure Selected Analyzers</h4>
            <div className="space-y-4">
              {config.analyzers.map((analyzer, index) => {
                const analyzerDef = AVAILABLE_ANALYZERS[analyzer.type]
                return (
                  <Card key={index}>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <h5 className="font-medium">{analyzerDef.name}</h5>
                          <p className="text-xs text-muted-foreground">
                            Metrics: {analyzerDef.metrics.join(', ')}
                          </p>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => removeAnalyzer(index)}
                        >
                          <Trash2 className="h-4 w-4 text-red-500" />
                        </Button>
                      </div>
                      
                      {analyzerDef.params.length > 0 && (
                        <div className="grid grid-cols-2 gap-3">
                          {analyzerDef.params.map(param => (
                            <div key={param.key}>
                              <Label className="text-xs">{param.label}</Label>
                              {param.type === 'boolean' ? (
                                <div className="flex items-center gap-2 mt-1">
                                  <Checkbox
                                    checked={analyzer.params[param.key] as boolean}
                                    onCheckedChange={(checked) => 
                                      updateAnalyzerParam(index, param.key, checked)
                                    }
                                  />
                                </div>
                              ) : param.type === 'select' && 'options' in param ? (
                                <Select
                                  value={analyzer.params[param.key] as string}
                                  onValueChange={(value) => 
                                    updateAnalyzerParam(index, param.key, value)
                                  }
                                >
                                  <SelectTrigger className="mt-1 h-8 text-xs">
                                    <SelectValue />
                                  </SelectTrigger>
                                  <SelectContent>
                                    {(param.options as readonly string[]).map((opt: string) => (
                                      <SelectItem key={opt} value={opt}>{opt}</SelectItem>
                                    ))}
                                  </SelectContent>
                                </Select>
                              ) : (
                                <Input
                                  className="mt-1 h-8 text-xs"
                                  value={analyzer.params[param.key] as string}
                                  onChange={(e) => 
                                    updateAnalyzerParam(index, param.key, e.target.value)
                                  }
                                />
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </CardContent>
                  </Card>
                )
              })}
            </div>
          </div>
        </>
      )}
    </div>
  )

  const renderTestsStep = () => {
    const availableMetrics = getAvailableMetrics()
    
    return (
      <div className="space-y-6">
        <div>
          <h3 className="text-lg font-medium mb-2">Define Tests</h3>
          <p className="text-sm text-muted-foreground mb-4">
            Create tests to validate your analysis results. Tests are optional but recommended.
          </p>
        </div>

        <Button onClick={addTest} variant="outline" className="w-full">
          <Plus className="h-4 w-4 mr-2" />
          Add Test
        </Button>

        {config.tests.map((test, index) => (
          <Card key={index}>
            <CardContent className="p-4 space-y-4">
              <div className="flex items-center justify-between">
                <h5 className="font-medium">Test {index + 1}</h5>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => removeTest(index)}
                >
                  <Trash2 className="h-4 w-4 text-red-500" />
                </Button>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label className="text-xs">Test ID</Label>
                  <Input
                    className="mt-1 h-8 text-xs"
                    value={test.id}
                    onChange={(e) => updateTest(index, { id: e.target.value })}
                  />
                </div>
                <div>
                  <Label className="text-xs">Title</Label>
                  <Input
                    className="mt-1 h-8 text-xs"
                    value={test.title}
                    onChange={(e) => updateTest(index, { title: e.target.value })}
                  />
                </div>
              </div>

              <div>
                <Label className="text-xs">Description</Label>
                <Input
                  className="mt-1 h-8 text-xs"
                  value={test.description}
                  onChange={(e) => updateTest(index, { description: e.target.value })}
                />
              </div>

              <div className="grid grid-cols-3 gap-3">
                <div>
                  <Label className="text-xs">Metric</Label>
                  <Select
                    value={test.metric}
                    onValueChange={(value) => updateTest(index, { metric: value })}
                  >
                    <SelectTrigger className="mt-1 h-8 text-xs">
                      <SelectValue placeholder="Select metric" />
                    </SelectTrigger>
                    <SelectContent>
                      {availableMetrics.map(m => (
                        <SelectItem key={m} value={m}>{m}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label className="text-xs">Test Type</Label>
                  <Select
                    value={test.type}
                    onValueChange={(value) => updateTest(index, { 
                      type: value as 'threshold' | 'percentage' | 'range' 
                    })}
                  >
                    <SelectTrigger className="mt-1 h-8 text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="threshold">Threshold</SelectItem>
                      <SelectItem value="percentage">Percentage</SelectItem>
                      <SelectItem value="range">Range</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label className="text-xs">Severity</Label>
                  <Select
                    value={test.severity}
                    onValueChange={(value) => updateTest(index, { 
                      severity: value as 'low' | 'medium' | 'high' 
                    })}
                  >
                    <SelectTrigger className="mt-1 h-8 text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="low">Low</SelectItem>
                      <SelectItem value="medium">Medium</SelectItem>
                      <SelectItem value="high">High</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {/* Type-specific fields */}
              {test.type === 'threshold' && (
                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <Label className="text-xs">Operator</Label>
                    <Select
                      value={test.operator}
                      onValueChange={(value) => updateTest(index, { operator: value })}
                    >
                      <SelectTrigger className="mt-1 h-8 text-xs">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value=">">{'>'}</SelectItem>
                        <SelectItem value=">=">{'>='}</SelectItem>
                        <SelectItem value="<">{'<'}</SelectItem>
                        <SelectItem value="<=">{'<='}</SelectItem>
                        <SelectItem value="==">{'=='}</SelectItem>
                        <SelectItem value="!=">{'!='}</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label className="text-xs">Value</Label>
                    <Input
                      className="mt-1 h-8 text-xs"
                      type="number"
                      value={test.value}
                      onChange={(e) => updateTest(index, { value: parseFloat(e.target.value) })}
                    />
                  </div>
                  <div>
                    <Label className="text-xs">Min %</Label>
                    <Input
                      className="mt-1 h-8 text-xs"
                      type="number"
                      placeholder="Optional"
                      value={test.minPercentage ?? ''}
                      onChange={(e) => updateTest(index, { 
                        minPercentage: e.target.value ? parseFloat(e.target.value) : undefined 
                      })}
                    />
                  </div>
                </div>
              )}

              {test.type === 'percentage' && (
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <Label className="text-xs">Min Percentage</Label>
                    <Input
                      className="mt-1 h-8 text-xs"
                      type="number"
                      value={test.minPercentage ?? ''}
                      onChange={(e) => updateTest(index, { 
                        minPercentage: e.target.value ? parseFloat(e.target.value) : undefined 
                      })}
                    />
                  </div>
                  <div>
                    <Label className="text-xs">Max Percentage</Label>
                    <Input
                      className="mt-1 h-8 text-xs"
                      type="number"
                      value={test.maxPercentage ?? ''}
                      onChange={(e) => updateTest(index, { 
                        maxPercentage: e.target.value ? parseFloat(e.target.value) : undefined 
                      })}
                    />
                  </div>
                </div>
              )}

              {test.type === 'range' && (
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <Label className="text-xs">Min Value</Label>
                    <Input
                      className="mt-1 h-8 text-xs"
                      type="number"
                      value={test.minValue ?? ''}
                      onChange={(e) => updateTest(index, { 
                        minValue: e.target.value ? parseFloat(e.target.value) : undefined 
                      })}
                    />
                  </div>
                  <div>
                    <Label className="text-xs">Max Value</Label>
                    <Input
                      className="mt-1 h-8 text-xs"
                      type="number"
                      value={test.maxValue ?? ''}
                      onChange={(e) => updateTest(index, { 
                        maxValue: e.target.value ? parseFloat(e.target.value) : undefined 
                      })}
                    />
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        ))}

        {config.tests.length === 0 && (
          <div className="text-center py-8 text-muted-foreground">
            <TestTube className="h-12 w-12 mx-auto mb-3 opacity-50" />
            <p>No tests added yet.</p>
            <p className="text-sm">Tests help validate your analysis results automatically.</p>
          </div>
        )}
      </div>
    )
  }

  const renderReviewStep = () => {
    const yamlConfig = generateYaml(config)
    
    return (
      <div className="space-y-6">
        <div>
          <h3 className="text-lg font-medium mb-2">Review Configuration</h3>
          <p className="text-sm text-muted-foreground mb-4">
            Review your configuration and make any final adjustments.
          </p>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <Card>
            <CardContent className="p-4 text-center">
              <Database className="h-8 w-8 mx-auto mb-2 text-blue-500" />
              <p className="text-sm font-medium">Dataset</p>
              <p className="text-xs text-muted-foreground truncate">
                {config.datasetPath || config.datasetName || 'Not set'}
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4 text-center">
              <Gauge className="h-8 w-8 mx-auto mb-2 text-green-500" />
              <p className="text-sm font-medium">Analyzers</p>
              <p className="text-xs text-muted-foreground">
                {config.analyzers.length} selected
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4 text-center">
              <TestTube className="h-8 w-8 mx-auto mb-2 text-purple-500" />
              <p className="text-sm font-medium">Tests</p>
              <p className="text-xs text-muted-foreground">
                {config.tests.length} defined
              </p>
            </CardContent>
          </Card>
        </div>

        <div>
          <Label className="mb-2 block">Generated YAML Configuration</Label>
          <div className="border rounded-md overflow-hidden">
            <Editor
              height="300px"
              language="yaml"
              theme="vs-dark"
              value={yamlConfig}
              options={{
                minimap: { enabled: false },
                fontSize: 12,
                lineNumbers: 'on',
                scrollBeyondLastLine: false,
                wordWrap: 'on',
                tabSize: 2,
                automaticLayout: true,
                readOnly: true,
              }}
            />
          </div>
        </div>
      </div>
    )
  }

  const renderRunningStep = () => {
    const progressPercent = jobStatus ? Math.round((jobStatus.progress / jobStatus.total) * 100) : 0
    const isComplete = jobStatus?.status === 'completed'
    const isFailed = jobStatus?.status === 'failed'

    return (
      <div className="space-y-6">
        <div className="text-center py-8">
          {/* Status Icon */}
          <div className="mb-6">
            {isComplete ? (
              <CheckCircle2 className="h-16 w-16 mx-auto text-green-500" />
            ) : isFailed ? (
              <XCircle className="h-16 w-16 mx-auto text-red-500" />
            ) : (
              <Loader2 className="h-16 w-16 mx-auto text-primary animate-spin" />
            )}
          </div>

          {/* Status Message */}
          <h3 className="text-xl font-semibold mb-2">
            {isComplete ? 'Analysis Complete!' : isFailed ? 'Analysis Failed' : 'Running Analysis...'}
          </h3>
          <p className="text-muted-foreground mb-6">
            {jobStatus?.message || 'Starting analysis...'}
          </p>

          {/* Progress Bar */}
          {!isComplete && !isFailed && (
            <div className="max-w-md mx-auto mb-6">
              <Progress value={progressPercent} className="h-3" />
              <p className="text-sm text-muted-foreground mt-2">
                {jobStatus?.progress || 0} / {jobStatus?.total || 100} samples processed
              </p>
            </div>
          )}

          {/* Error Message */}
          {isFailed && jobStatus?.error && (
            <div className="max-w-md mx-auto mb-6 p-4 bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 rounded-lg">
              <p className="text-sm text-red-600 dark:text-red-400">{jobStatus.error}</p>
            </div>
          )}

          {/* Success Message */}
          {isComplete && (
            <div className="max-w-md mx-auto mb-6 p-4 bg-green-50 dark:bg-green-950 border border-green-200 dark:border-green-800 rounded-lg">
              <p className="text-sm text-green-600 dark:text-green-400">
                Analysis completed successfully! Redirecting to results...
              </p>
            </div>
          )}
        </div>

        {/* Log Output */}
        {jobStatus?.log_lines && jobStatus.log_lines.length > 0 && (
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Terminal className="h-4 w-4" />
              <Label>Output Log</Label>
            </div>
            <ScrollArea className="h-48 border rounded-md bg-zinc-950 p-4">
              <pre className="text-xs text-zinc-300 font-mono whitespace-pre-wrap">
                {jobStatus.log_lines.join('\n')}
              </pre>
            </ScrollArea>
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center justify-center gap-4 pt-4">
          {isFailed && (
            <>
              <Button variant="outline" onClick={handleBackFromRunning}>
                <ChevronLeft className="h-4 w-4 mr-2" />
                Back to Config
              </Button>
              <Button onClick={handleRunAnalysis}>
                <Play className="h-4 w-4 mr-2" />
                Retry
              </Button>
            </>
          )}
        </div>
      </div>
    )
  }

  // Show running view
  if (isRunning) {
    return (
      <Card className="w-full max-w-4xl mx-auto">
        <CardHeader className="relative">
          <Button
            variant="ghost"
            size="icon"
            onClick={handleBackFromRunning}
            className="absolute right-4 top-4 h-8 w-8 text-muted-foreground hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </Button>
          <CardTitle>Running Analysis</CardTitle>
          <CardDescription>
            Your analysis is being processed. This may take a few minutes.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {renderRunningStep()}
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader className="relative">
        <Button
          variant="ghost"
          size="icon"
          onClick={onCancel}
          className="absolute right-4 top-4 h-8 w-8 text-muted-foreground hover:text-foreground"
        >
          <X className="h-4 w-4" />
        </Button>
        <CardTitle>{isEditMode ? 'Edit Analysis' : 'Create New Analysis'}</CardTitle>
        <CardDescription>
          {isEditMode 
            ? 'Modify the configuration and re-run the analysis.'
            : 'Set up a new dataset analysis by following these steps.'}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {/* Progress Steps */}
        <div className="flex items-center justify-between mb-8">
          {STEPS.map((step, index) => {
            const Icon = step.icon
            const isActive = index === currentStep
            const isCompleted = index < currentStep
            
            return (
              <div key={step.id} className="flex items-center">
                <div
                  className={cn(
                    'flex items-center justify-center w-10 h-10 rounded-full border-2 transition-colors',
                    isActive ? 'border-primary bg-primary text-primary-foreground' :
                    isCompleted ? 'border-primary bg-primary/10 text-primary' :
                    'border-muted-foreground/30 text-muted-foreground'
                  )}
                >
                  {isCompleted ? (
                    <Check className="h-5 w-5" />
                  ) : (
                    <Icon className="h-5 w-5" />
                  )}
                </div>
                <span className={cn(
                  'ml-2 text-sm font-medium hidden sm:block',
                  isActive ? 'text-foreground' : 'text-muted-foreground'
                )}>
                  {step.title}
                </span>
                {index < STEPS.length - 1 && (
                  <ChevronRight className="h-4 w-4 mx-4 text-muted-foreground" />
                )}
              </div>
            )
          })}
        </div>

        {/* Step Content */}
        <div className="min-h-[400px]">
          {currentStep === 0 && renderDatasetStep()}
          {currentStep === 1 && renderAnalyzersStep()}
          {currentStep === 2 && renderTestsStep()}
          {currentStep === 3 && renderReviewStep()}
        </div>

        {/* Navigation */}
        <div className="flex items-center justify-between mt-8 pt-6 border-t">
          <Button
            variant="outline"
            onClick={() => currentStep > 0 ? setCurrentStep(currentStep - 1) : onCancel?.()}
          >
            <ChevronLeft className="h-4 w-4 mr-2" />
            {currentStep === 0 ? 'Cancel' : 'Back'}
          </Button>
          
          {currentStep < STEPS.length - 1 ? (
            <Button
              onClick={() => setCurrentStep(currentStep + 1)}
              disabled={!canProceed()}
            >
              Next
              <ChevronRight className="h-4 w-4 ml-2" />
            </Button>
          ) : (
            <div className="flex gap-2">
              <Button variant="outline" onClick={handleCopyConfig}>
                <FileCode className="h-4 w-4 mr-2" />
                Copy Config
              </Button>
              <Button onClick={handleRunAnalysis} disabled={isStarting}>
                {isStarting ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Play className="h-4 w-4 mr-2" />
                )}
                Run Analysis
              </Button>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
