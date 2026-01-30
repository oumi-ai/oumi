import { useState, useCallback, useEffect, useRef } from 'react'
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
import { Textarea } from '@/components/ui/textarea'
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
  X,
  Code,
  Sparkles,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useRunAnalysis } from '@/hooks/useEvals'
import { useSuggestions, type AnalyzerSuggestion, type CustomMetricSuggestion, type TestSuggestion } from '@/hooks/useSuggestions'
import { SuggestionPanel, SuggestionPanelMinimized } from './SuggestionPanel'

// LLM analyzer types that need special handling (use id: llm with criteria param)
const LLM_ANALYZER_TYPES = ['usefulness', 'safety', 'coherence', 'factuality', 'instruction_following', 'custom_llm'] as const

// Available analyzers with their parameters and metrics
const AVAILABLE_ANALYZERS = {
  // --- Non-LLM Analyzers (fast, cheap) ---
  length: {
    name: 'Length Analyzer',
    description: 'Compute token length metrics for conversations',
    params: [
      { key: 'tiktoken_encoding', type: 'string', default: 'cl100k_base', label: 'Tiktoken Encoding' },
      { key: 'compute_role_stats', type: 'boolean', default: false, label: 'Compute Role Stats' },
    ],
    metrics: ['total_tokens', 'num_messages', 'avg_tokens_per_message']
  },
  quality: {
    name: 'Data Quality Analyzer',
    description: 'Basic quality checks: empty turns, invalid values, truncation, refusals, tag balance',
    params: [
      { key: 'check_turn_pattern', type: 'boolean', default: true, label: 'Check Turn Pattern' },
      { key: 'check_empty_content', type: 'boolean', default: true, label: 'Check Empty Content' },
      { key: 'check_invalid_values', type: 'boolean', default: true, label: 'Check Invalid Values' },
      { key: 'check_truncation', type: 'boolean', default: true, label: 'Check Truncation' },
      { key: 'check_refusals', type: 'boolean', default: true, label: 'Check Policy Refusals' },
      { key: 'check_tags', type: 'boolean', default: true, label: 'Check Tag Balance' },
    ],
    metrics: [
      // Boolean indicators
      'has_alternating_turns', 'has_empty_turns', 'has_invalid_values',
      'fits_4k_context', 'fits_8k_context', 'appears_truncated', 'ends_mid_sentence',
      'has_policy_refusal', 'has_think_tags', 'has_unbalanced_tags', 'passes_basic_quality',
      // Numeric counts
      'num_consecutive_same_role', 'empty_turn_count', 'estimated_tokens', 'refusal_count'
    ]
  },
  turn_stats: {
    name: 'Turn Statistics Analyzer',
    description: 'Compute turn counts, per-role statistics, and response length ratios',
    params: [
      { key: 'include_system_in_counts', type: 'boolean', default: false, label: 'Include System in Counts' },
    ],
    metrics: [
      'num_turns', 'num_user_turns', 'num_assistant_turns', 'has_system_message',
      'avg_user_chars', 'avg_assistant_chars', 'total_user_chars', 'total_assistant_chars',
      'response_ratio', 'assistant_turn_ratio', 'first_turn_role', 'last_turn_role'
    ]
  },
  deduplication: {
    name: 'Deduplication Analyzer',
    description: 'Detect duplicate conversations across the dataset (dataset-level)',
    params: [
      { key: 'hash_method', type: 'select', options: ['normalized', 'exact'] as const, default: 'normalized', label: 'Hash Method' },
      { key: 'include_system', type: 'boolean', default: false, label: 'Include System Messages' },
      { key: 'include_roles', type: 'boolean', default: true, label: 'Include Role Prefixes' },
    ],
    metrics: [
      'total_conversations', 'unique_conversations', 'duplicate_count', 'duplicate_ratio',
      'num_duplicate_groups', 'largest_group_size'
    ]
  },
  // --- LLM-based Analyzers ---
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
  instruction_following: {
    name: 'Instruction Following Analyzer',
    description: 'Evaluate how well responses follow given instructions',
    params: [
      { key: 'model_name', type: 'string', default: 'gpt-4o-mini', label: 'Model' },
      { key: 'api_provider', type: 'select', options: ['openai', 'anthropic'] as const, default: 'openai', label: 'API Provider' },
      { key: 'target_scope', type: 'select', options: ['conversation', 'last_turn', 'first_user'] as const, default: 'conversation', label: 'Target Scope' },
    ],
    metrics: ['score', 'passed', 'label', 'reasoning']
  },
  custom_llm: {
    name: 'Custom LLM Analyzer',
    description: 'Create your own LLM-as-judge with a custom prompt',
    params: [
      { key: 'criteria_name', type: 'string', default: 'custom_eval', label: 'Metric Name (ID)' },
      { key: 'prompt_template', type: 'textarea', default: 'Evaluate the following conversation:\n\n{target}\n\nProvide a score from 0-100.', label: 'Prompt Template' },
      { key: 'model_name', type: 'string', default: 'gpt-4o-mini', label: 'Model' },
      { key: 'api_provider', type: 'select', options: ['openai', 'anthropic'] as const, default: 'openai', label: 'API Provider' },
      { key: 'target_scope', type: 'select', options: ['conversation', 'last_turn', 'first_user'] as const, default: 'conversation', label: 'Target Scope' },
    ],
    metrics: ['score', 'passed', 'label', 'reasoning']
  },
}

// Default custom metric template
const DEFAULT_CUSTOM_METRIC_FUNCTION = `def compute(conversation):
    """
    Extract or compute a custom metric from the conversation.
    
    Args:
        conversation: A Conversation object with:
            - messages: List of Message objects (role, content)
            - metadata: Dict with any metadata fields
    
    Returns:
        A dict with your metric values, e.g.:
        {"label_id": 5, "label_name": "billing"}
    """
    metadata = conversation.metadata or {}
    return {
        "value": metadata.get("some_field", "default")
    }`

type AnalyzerKey = keyof typeof AVAILABLE_ANALYZERS

interface CustomMetric {
  id: string
  scope: string
  function: string
  description?: string
  output_schema?: Array<{name: string; type: string; description?: string}>
  depends_on?: string[]
}

interface WizardConfig {
  name: string
  parentEvalId?: string  // For linking derived analyses
  datasetPath: string
  datasetName: string
  split: string
  subset?: string  // HuggingFace dataset config/subset
  sampleCount: number
  outputPath: string
  analyzers: {
    id: string
    type: AnalyzerKey
    instanceId?: string  // For custom LLM analyzers, stores the original instance_id
    params: Record<string, unknown>
  }[]
  customMetrics: CustomMetric[]
  tests: {
    id: string
    type: 'threshold' | 'percentage' | 'range'
    metric: string
    title: string
    description: string
    severity: 'low' | 'medium' | 'high'
    operator?: string
    value?: number | boolean  // Can be number or boolean for threshold tests
    valueType?: 'number' | 'boolean'  // Track the value type for UI
    condition?: string  // For percentage tests, e.g., '== True'
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
  /** Initial step to start on (0=Dataset, 1=Analyzers, 2=Tests) */
  initialStep?: number
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
    parentEvalId: (config.parent_eval_id as string) || undefined,
    datasetPath: (config.dataset_path as string) || '',
    datasetName: (config.dataset_name as string) || '',
    split: (config.split as string) || 'train',
    sampleCount: (config.sample_count as number) || 100,
    outputPath: (config.output_path as string) || './analysis_output',
    analyzers: [],
    customMetrics: [],
    tests: [],
  }

  // Parse analyzers
  const analyzers = config.analyzers as Array<Record<string, unknown>> | undefined
  if (analyzers && Array.isArray(analyzers)) {
    wizardConfig.analyzers = analyzers.map((a) => {
      let analyzerType = (a.id as string) || 'length'
      const rawParams = (a.params as Record<string, unknown>) || {}
      const instanceId = a.instance_id as string | undefined
      let customInstanceId: string | undefined = undefined
      
      // For LLM analyzers (id: llm), determine the analyzer type
      if (analyzerType === 'llm') {
        const criteria = rawParams.criteria as string | undefined
        const criteriaName = rawParams.criteria_name as string | undefined
        const resolvedType = instanceId || criteria || criteriaName || 'usefulness'
        
        // Check if it's a known LLM analyzer type
        const isKnownLlmType = AVAILABLE_ANALYZERS[resolvedType as AnalyzerKey] !== undefined
        if (isKnownLlmType) {
          analyzerType = resolvedType
        } else {
          // Custom LLM analyzer - use 'usefulness' UI but preserve the custom instance_id
          analyzerType = 'usefulness'
          customInstanceId = resolvedType
        }
      }
      
      // Migrate old param names to new ones
      const params = migrateParams(rawParams)
      
      // Check if analyzer type is supported
      const isSupported = AVAILABLE_ANALYZERS[analyzerType as AnalyzerKey] !== undefined
      
      return {
        id: analyzerType,
        type: (isSupported ? analyzerType : 'length') as AnalyzerKey,
        instanceId: customInstanceId,
        params,
      }
    })
  }

  // Parse custom metrics
  const customMetrics = config.custom_metrics as Array<Record<string, unknown>> | undefined
  if (customMetrics && Array.isArray(customMetrics)) {
    wizardConfig.customMetrics = customMetrics.map((m) => ({
      id: (m.id as string) || '',
      scope: (m.scope as string) || 'conversation',
      function: (m.function as string) || '',
      description: m.description as string | undefined,
      output_schema: m.output_schema as Array<{name: string; type: string; description?: string}> | undefined,
      depends_on: m.depends_on as string[] | undefined,
    }))
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
      condition: t.condition as string | undefined,
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

/** Format a value for YAML output, handling multiline strings properly */
function formatYamlValue(key: string, value: unknown, indent: string): string[] {
  const strValue = String(value)
  
  // Check if it's a multiline string
  if (typeof value === 'string' && strValue.includes('\n')) {
    const lines = [`${indent}${key}: |`]
    strValue.split('\n').forEach(line => {
      lines.push(`${indent}  ${line}`)
    })
    return lines
  }
  
  // Check if value needs quoting (contains special chars)
  if (typeof value === 'string' && /[:#{}[\],&*?|<>=!%@`]/.test(strValue)) {
    return [`${indent}${key}: "${strValue.replace(/"/g, '\\"')}"`]
  }
  
  return [`${indent}${key}: ${strValue}`]
}

function generateYaml(config: WizardConfig): string {
  const lines: string[] = []
  
  // Analysis name (optional)
  if (config.name.trim()) {
    lines.push(`eval_name: ${config.name.trim()}`)
  }
  
  // Parent eval ID (for linking derived analyses)
  if (config.parentEvalId) {
    lines.push(`parent_eval_id: ${config.parentEvalId}`)
  }
  
  // Dataset config
  if (config.datasetPath) {
    lines.push(`dataset_path: ${config.datasetPath}`)
  } else if (config.datasetName) {
    lines.push(`dataset_name: ${config.datasetName}`)
    if (config.subset) {
      lines.push(`subset: ${config.subset}`)
    }
    lines.push(`split: ${config.split || 'train'}`)
  }
  
  lines.push(`sample_count: ${config.sampleCount}`)
  lines.push(`output_path: ${config.outputPath}`)
  lines.push('')
  
  // Analyzers
  lines.push('analyzers:')
  config.analyzers.forEach(analyzer => {
    const isLlmAnalyzer = (LLM_ANALYZER_TYPES as readonly string[]).includes(analyzer.type)
    
    if (isLlmAnalyzer) {
      // For LLM analyzers, use criteria_name - it controls the metric prefix
      // For custom_llm, get criteria_name from params; for presets, use the type
      const isCustomLlm = analyzer.type === 'custom_llm'
      const instanceId = isCustomLlm 
        ? (analyzer.params.criteria_name as string) || 'custom_eval'
        : (analyzer.instanceId || analyzer.type)
      
      lines.push(`  - id: llm`)
      lines.push(`    instance_id: ${instanceId}`)
      lines.push(`    params:`)
      lines.push(`      criteria_name: ${instanceId}`)
      
      // Filter out criteria/criteria_name since we're already outputting it above
      const paramEntries = Object.entries(analyzer.params).filter(
        ([k, v]) => v !== undefined && v !== '' && !['criteria', 'criteria_name'].includes(k)
      )
      paramEntries.forEach(([key, value]) => {
        lines.push(...formatYamlValue(key, value, '      '))
      })
    } else {
      // For non-LLM analyzers (like length), use the type directly
      lines.push(`  - id: ${analyzer.type}`)
      lines.push(`    instance_id: ${analyzer.type}`)
      const paramEntries = Object.entries(analyzer.params).filter(([, v]) => v !== undefined && v !== '')
      if (paramEntries.length > 0) {
        lines.push('    params:')
        paramEntries.forEach(([key, value]) => {
          lines.push(...formatYamlValue(key, value, '      '))
        })
      }
    }
  })
  lines.push('')
  
  // Custom metrics
  if (config.customMetrics.length > 0) {
    lines.push('custom_metrics:')
    config.customMetrics.forEach(metric => {
      lines.push(`  - id: ${metric.id}`)
      lines.push(`    scope: ${metric.scope}`)
      // Use YAML literal block for multiline function
      lines.push(`    function: |`)
      metric.function.split('\n').forEach(line => {
        lines.push(`      ${line}`)
      })
      if (metric.description) {
        lines.push(`    description: "${metric.description}"`)
      }
      // Output schema
      if (metric.output_schema && (metric.output_schema as unknown[]).length > 0) {
        lines.push(`    output_schema:`)
        ;(metric.output_schema as Array<{name: string; type?: string; description?: string}>).forEach(field => {
          lines.push(`      - name: ${field.name}`)
          if (field.type) lines.push(`        type: ${field.type}`)
          if (field.description) lines.push(`        description: "${field.description}"`)
        })
      } else {
        lines.push(`    output_schema: []`)
      }
      if (metric.depends_on && metric.depends_on.length > 0) {
        lines.push(`    depends_on:`)
        metric.depends_on.forEach(dep => {
          lines.push(`      - ${dep}`)
        })
      } else {
        lines.push(`    depends_on: []`)
      }
    })
    lines.push('')
  }
  
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
        if (test.maxPercentage !== undefined) lines.push(`    max_percentage: ${test.maxPercentage}`)
        if (test.minPercentage !== undefined) lines.push(`    min_percentage: ${test.minPercentage}`)
      } else if (test.type === 'percentage') {
        // Percentage tests require a condition - use provided or default to '== True'
        const condition = test.condition || '== True'
        lines.push(`    condition: "${condition}"`)
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

export function SetupWizard({ onComplete, onRunComplete, onCancel, initialConfig, initialStep = 0 }: SetupWizardProps) {
  const [currentStep, setCurrentStep] = useState(initialStep)
  const [isRunning, setIsRunning] = useState(false)
  const [config, setConfig] = useState<WizardConfig>(() => {
    if (initialConfig) {
      return parseConfigToWizard(initialConfig)
    }
    return {
      name: '',
      datasetPath: '',
      datasetName: '',
      split: 'train',
      sampleCount: 100,
      outputPath: './analysis_output',
      analyzers: [],
      customMetrics: [],
      tests: [],
    }
  })

  // Update config when initialConfig changes (e.g., opening wizard for different eval)
  useEffect(() => {
    if (initialConfig) {
      setConfig(parseConfigToWizard(initialConfig))
      setCurrentStep(initialStep)
    }
  }, [initialConfig, initialStep])
  
  const isEditMode = !!initialConfig

  const { run, runTestsOnlyCached, reset, jobStatus, isStarting } = useRunAnalysis()

  // AI-powered suggestions
  const {
    status: suggestionsStatus,
    suggestions,
    error: suggestionsError,
    isDismissed: suggestionsDismissed,
    userPrompt: suggestionsUserPrompt,
    unappliedAnalyzers,
    unappliedCustomMetrics,
    // Note: unappliedTests not used - we compute filtered version based on selected analyzers
    appliedAnalyzers,
    appliedCustomMetrics,
    appliedTests,
    triggerSuggestions,
    markAnalyzerApplied,
    markCustomMetricApplied,
    markTestApplied,
    markAllAnalyzersApplied,
    markAllCustomMetricsApplied,
    markAllTestsApplied,
    dismiss: dismissSuggestions,
    undismiss: undismissSuggestions,
    editPrompt: editSuggestionsPrompt,
  } = useSuggestions()

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

  // File upload handling
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)

  // HuggingFace dataset info
  const [datasetInfo, setDatasetInfo] = useState<{
    splits: string[]
    configs: string[]
    loading: boolean
    error: string | null
    loadedDataset: string | null  // Track which dataset the info is for
  }>({
    splits: [],
    configs: [],
    loading: false,
    error: null,
    loadedDataset: null,
  })

  const fetchDatasetInfo = useCallback(async (datasetName: string) => {
    if (!datasetName.trim()) return

    setDatasetInfo(prev => ({ ...prev, loading: true, error: null }))

    try {
      const response = await fetch('/api/dataset-info', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_name: datasetName }),
      })

      const result = await response.json()

      if (result.error) {
        setDatasetInfo(prev => ({
          ...prev,
          loading: false,
          error: result.error,
          loadedDataset: null,
        }))
      } else {
        setDatasetInfo({
          splits: result.splits || ['train'],
          configs: result.configs || [],
          loading: false,
          error: null,
          loadedDataset: datasetName,
        })
        // Auto-select first split if current split not in list
        if (result.splits && result.splits.length > 0) {
          if (!result.splits.includes(config.split)) {
            updateConfig({ split: result.splits[0] })
          }
        }
      }
    } catch (e) {
      setDatasetInfo(prev => ({
        ...prev,
        loading: false,
        error: e instanceof Error ? e.message : 'Failed to fetch dataset info',
        loadedDataset: null,
      }))
    }
  }, [config.split, updateConfig])

  const handleFileSelect = useCallback(async (file: File) => {
    setIsUploading(true)
    try {
      // Read file content
      const content = await file.text()
      
      // Upload to server
      const response = await fetch('/api/upload-dataset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filename: file.name,
          content: content
        })
      })
      
      const result = await response.json()
      
      if (result.success && result.path) {
        updateConfig({ 
          datasetPath: result.path,
          datasetName: '' 
        })
      } else {
        console.error('Upload failed:', result.error)
        // Fallback to just filename with a warning
        updateConfig({ 
          datasetPath: file.name,
          datasetName: '' 
        })
      }
    } catch (error) {
      console.error('Error uploading file:', error)
      // Fallback to just filename
      updateConfig({ 
        datasetPath: file.name,
        datasetName: '' 
      })
    } finally {
      setIsUploading(false)
    }
  }, [updateConfig])

  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleFileSelect(file)
    }
  }, [handleFileSelect])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) {
      handleFileSelect(file)
    }
  }, [handleFileSelect])

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

  // Custom Metrics management
  const addCustomMetric = useCallback(() => {
    const metricId = `custom_${config.customMetrics.length + 1}`
    setConfig(prev => ({
      ...prev,
      customMetrics: [
        ...prev.customMetrics,
        {
          id: metricId,
          scope: 'conversation',
          function: DEFAULT_CUSTOM_METRIC_FUNCTION,
        }
      ]
    }))
  }, [config.customMetrics.length])

  const removeCustomMetric = useCallback((index: number) => {
    setConfig(prev => ({
      ...prev,
      customMetrics: prev.customMetrics.filter((_, i) => i !== index)
    }))
  }, [])

  const updateCustomMetric = useCallback((index: number, updates: Partial<CustomMetric>) => {
    setConfig(prev => ({
      ...prev,
      customMetrics: prev.customMetrics.map((m, i) => 
        i === index ? { ...m, ...updates } : m
      )
    }))
  }, [])

  const addTest = useCallback(() => {
    const testId = `test_${config.tests.length + 1}`
    // Add new test at the beginning so it's immediately visible
    setConfig(prev => ({
      ...prev,
      tests: [
        {
          id: testId,
          type: 'threshold',
          metric: '',
          title: '',
          description: '',
          severity: 'medium',
          operator: '>',
          value: 0,
        },
        ...prev.tests,
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

  // Handle step navigation
  const handleNextStep = useCallback(() => {
    setCurrentStep(currentStep + 1)
  }, [currentStep])

  // State for AI suggestion user query - initialize from persisted prompt if available
  const [suggestionQuery, setSuggestionQuery] = useState(suggestionsUserPrompt || '')
  
  // Sync local query state when persisted prompt changes (e.g., loaded from localStorage)
  useEffect(() => {
    if (suggestionsUserPrompt && !suggestionQuery) {
      setSuggestionQuery(suggestionsUserPrompt)
    }
  }, [suggestionsUserPrompt, suggestionQuery])

  // Request AI suggestions with optional user query
  const requestAISuggestions = useCallback(() => {
    if (!config.datasetPath && !config.datasetName) return
    triggerSuggestions({
      dataset_path: config.datasetPath || undefined,
      dataset_name: config.datasetName || undefined,
      split: config.split || 'train',
      subset: config.subset || undefined,
      sample_count: 1,
      user_query: suggestionQuery || undefined,
    })
  }, [config.datasetPath, config.datasetName, config.split, config.subset, suggestionQuery, triggerSuggestions])


  // Apply an analyzer suggestion
  const applyAnalyzerSuggestion = useCallback((suggestion: AnalyzerSuggestion) => {
    // Check if already added
    if (config.analyzers.some(a => a.type === suggestion.id)) {
      markAnalyzerApplied(suggestion.id)
      return
    }

    // Get default params from AVAILABLE_ANALYZERS
    const analyzerKey = suggestion.id as AnalyzerKey
    const analyzerDef = AVAILABLE_ANALYZERS[analyzerKey]
    if (!analyzerDef) {
      console.warn(`Unknown analyzer: ${suggestion.id}`)
      return
    }

    const defaultParams: Record<string, unknown> = {}
    analyzerDef.params.forEach(p => {
      defaultParams[p.key] = suggestion.params?.[p.key] ?? p.default
    })

    setConfig(prev => ({
      ...prev,
      analyzers: [
        ...prev.analyzers,
        {
          id: suggestion.id,
          type: analyzerKey,
          params: defaultParams,
        }
      ]
    }))
    markAnalyzerApplied(suggestion.id)
  }, [config.analyzers, markAnalyzerApplied])

  // Apply a custom metric suggestion
  const applyCustomMetricSuggestion = useCallback((suggestion: CustomMetricSuggestion) => {
    // Check if already added
    if (config.customMetrics.some(m => m.id === suggestion.id)) {
      markCustomMetricApplied(suggestion.id)
      return
    }

    setConfig(prev => ({
      ...prev,
      customMetrics: [
        ...prev.customMetrics,
        {
          id: suggestion.id,
          scope: 'conversation',
          function: suggestion.function,
          description: suggestion.description,
          output_schema: suggestion.output_schema,
        }
      ]
    }))
    markCustomMetricApplied(suggestion.id)
  }, [config.customMetrics, markCustomMetricApplied])

  // Apply a test suggestion
  const applyTestSuggestion = useCallback((suggestion: TestSuggestion) => {
    // Check if already added
    if (config.tests.some(t => t.id === suggestion.id)) {
      markTestApplied(suggestion.id)
      return
    }

    setConfig(prev => ({
      ...prev,
      tests: [
        {
          id: suggestion.id,
          type: suggestion.type,
          metric: suggestion.metric,
          title: suggestion.title || '',
          description: suggestion.description || '',
          severity: suggestion.severity || 'medium',
          operator: suggestion.operator,
          value: suggestion.value,
          condition: suggestion.condition,
          maxPercentage: suggestion.max_percentage,
          minPercentage: suggestion.min_percentage,
          minValue: suggestion.min_value,
          maxValue: suggestion.max_value,
        },
        ...prev.tests,
      ]
    }))
    markTestApplied(suggestion.id)
  }, [config.tests, markTestApplied])

  // Apply all analyzer and custom metric suggestions
  const applyAllAnalyzerSuggestions = useCallback(() => {
    suggestions?.analyzers.forEach(s => {
      if (!appliedAnalyzers.has(s.id)) {
        applyAnalyzerSuggestion(s)
      }
    })
    suggestions?.custom_metrics.forEach(s => {
      if (!appliedCustomMetrics.has(s.id)) {
        applyCustomMetricSuggestion(s)
      }
    })
    markAllAnalyzersApplied()
    markAllCustomMetricsApplied()
  }, [suggestions, appliedAnalyzers, appliedCustomMetrics, applyAnalyzerSuggestion, applyCustomMetricSuggestion, markAllAnalyzersApplied, markAllCustomMetricsApplied])

  // Get the metric prefixes (analyzer IDs) from selected analyzers and custom metrics
  const getAvailableMetricPrefixes = useCallback((): Set<string> => {
    const prefixes = new Set<string>()
    
    // Add prefixes from selected analyzers
    config.analyzers.forEach(analyzer => {
      if (analyzer.type === 'custom_llm') {
        prefixes.add((analyzer.params.criteria_name as string) || 'custom_eval')
      } else {
        prefixes.add(analyzer.instanceId || analyzer.type)
      }
    })
    
    // Add prefixes from custom metrics
    config.customMetrics.forEach(customMetric => {
      prefixes.add(customMetric.id)
    })
    
    return prefixes
  }, [config.analyzers, config.customMetrics])

  // Check if a test's metric is relevant to the selected analyzers
  const isTestRelevant = useCallback((metric: string, availablePrefixes: Set<string>): boolean => {
    // Extract the analyzer/metric prefix from "AnalyzerName.field_name"
    const prefix = metric.split('.')[0]
    return availablePrefixes.has(prefix)
  }, [])

  // Apply all test suggestions (only relevant ones)
  const applyAllTestSuggestions = useCallback(() => {
    const availableMetrics = getAvailableMetricPrefixes()
    suggestions?.tests.forEach(s => {
      if (!appliedTests.has(s.id) && isTestRelevant(s.metric, availableMetrics)) {
        applyTestSuggestion(s)
      }
    })
    markAllTestsApplied()
  }, [suggestions, appliedTests, applyTestSuggestion, markAllTestsApplied, getAvailableMetricPrefixes, isTestRelevant])

  // Get filtered test suggestions (only tests for selected analyzers)
  const getRelevantTestSuggestions = useCallback(() => {
    if (!suggestions?.tests) return []
    const availablePrefixes = getAvailableMetricPrefixes()
    return suggestions.tests.filter(test => isTestRelevant(test.metric, availablePrefixes))
  }, [suggestions?.tests, getAvailableMetricPrefixes, isTestRelevant])

  const getAvailableMetrics = useCallback(() => {
    const metrics: string[] = []
    // Metrics from analyzers
    config.analyzers.forEach(analyzer => {
      const analyzerDef = AVAILABLE_ANALYZERS[analyzer.type]
      // For custom_llm, use criteria_name from params; for presets use instanceId or type
      let metricPrefix: string
      if (analyzer.type === 'custom_llm') {
        metricPrefix = (analyzer.params.criteria_name as string) || 'custom_eval'
      } else {
        metricPrefix = analyzer.instanceId || analyzer.type
      }
      analyzerDef.metrics.forEach(m => {
        metrics.push(`${metricPrefix}.${m}`)
      })
    })
    // Metrics from custom metrics
    config.customMetrics.forEach(customMetric => {
      // Get field names from output_schema
      const schemaFields = (customMetric.output_schema as Array<{name?: string}>)
        ?.map(s => s.name)
        .filter(Boolean) || []
      
      if (schemaFields.length > 0) {
        schemaFields.forEach(field => {
          metrics.push(`${customMetric.id}.${field}`)
        })
      } else {
        // No schema defined yet - show hint
        metrics.push(`${customMetric.id}.* (define output schema)`)
      }
    })
    return metrics
  }, [config.analyzers, config.customMetrics])

  // Check if only tests changed by comparing generated configs (excluding tests section)
  const onlyTestsChanged = useCallback((): boolean => {
    if (!initialConfig) return false
    
    // Parse original config to wizard format and generate YAML
    const parsedInitial = parseConfigToWizard(initialConfig)
    const initialYaml = generateYaml(parsedInitial)
    const currentYaml = generateYaml(config)
    
    // Remove the tests section from both for comparison
    // Tests section starts with "tests:" and goes to end of file
    const stripTests = (yaml: string) => {
      const testsIndex = yaml.indexOf('\ntests:')
      return testsIndex === -1 ? yaml : yaml.substring(0, testsIndex)
    }
    
    const initialWithoutTests = stripTests(initialYaml)
    const currentWithoutTests = stripTests(currentYaml)
    
    const matches = initialWithoutTests === currentWithoutTests
    
    if (matches) {
      console.log('onlyTestsChanged: ✅ Only tests changed - can use cache')
    } else {
      console.log('onlyTestsChanged: ❌ Analyzers or other config changed')
      console.log('Initial (no tests):', initialWithoutTests)
      console.log('Current (no tests):', currentWithoutTests)
    }
    
    return matches
  }, [initialConfig, config])

  const handleRunAnalysis = useCallback(() => {
    const yaml = generateYaml(config)
    setIsRunning(true)
    
    // If only tests changed and we have a parent eval, reuse cached results
    const parentEvalId = config.parentEvalId
    if (parentEvalId && onlyTestsChanged()) {
      console.log('Only tests changed - reusing cached analyzer results')
      runTestsOnlyCached(yaml, parentEvalId)
    } else {
      run(yaml)
    }
  }, [config, run, runTestsOnlyCached, onlyTestsChanged])

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
          <div 
            className={`flex gap-2 mt-1.5 p-2 rounded-md border-2 border-dashed transition-colors ${
              isDragging ? 'border-primary bg-primary/5' : 'border-transparent'
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <Input
              id="datasetPath"
              placeholder="/path/to/dataset.jsonl"
              value={config.datasetPath}
              onChange={(e) => updateConfig({ datasetPath: e.target.value, datasetName: '' })}
              className="flex-1"
            />
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileInputChange}
              accept=".jsonl,.json"
              className="hidden"
            />
            <Button 
              variant="outline" 
              size="icon"
              onClick={() => fileInputRef.current?.click()}
              title="Browse for file"
              disabled={isUploading}
            >
              {isUploading ? (
                <span className="animate-spin">⏳</span>
              ) : (
                <Upload className="h-4 w-4" />
              )}
            </Button>
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            {isUploading 
              ? 'Uploading...' 
              : 'Type the full path directly, or drop/upload a file (creates a temp copy)'}
          </p>
        </div>

        <div className="flex items-center gap-4">
          <Separator className="flex-1" />
          <span className="text-sm text-muted-foreground">or</span>
          <Separator className="flex-1" />
        </div>

        <div className="space-y-3">
          <div>
            <Label htmlFor="datasetName">HuggingFace Dataset</Label>
            <div className="flex gap-2 mt-1.5">
              <Input
                id="datasetName"
                placeholder="HuggingFaceH4/ultrachat_200k"
                className="flex-1"
                value={config.datasetName}
                onChange={(e) => {
                  updateConfig({ datasetName: e.target.value, datasetPath: '' })
                  // Clear dataset info if name changed
                  if (e.target.value !== datasetInfo.loadedDataset) {
                    setDatasetInfo(prev => ({ ...prev, loadedDataset: null, splits: [], configs: [] }))
                  }
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && config.datasetName) {
                    e.preventDefault()
                    fetchDatasetInfo(config.datasetName)
                  }
                }}
              />
              <Button
                variant="outline"
                onClick={() => fetchDatasetInfo(config.datasetName)}
                disabled={!config.datasetName || datasetInfo.loading}
                title="Load dataset info to see available splits"
              >
                {datasetInfo.loading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  'Load'
                )}
              </Button>
            </div>
            {datasetInfo.error && (
              <p className="text-xs text-destructive mt-1">{datasetInfo.error}</p>
            )}
            {datasetInfo.loadedDataset && !datasetInfo.error && (
              <p className="text-xs text-green-600 mt-1">
                Dataset found with {datasetInfo.splits.length} split(s)
                {datasetInfo.configs.length > 0 && ` and ${datasetInfo.configs.length} config(s)`}
              </p>
            )}
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="split">Split</Label>
              {datasetInfo.splits.length > 0 ? (
                <Select
                  value={config.split}
                  onValueChange={(value) => updateConfig({ split: value })}
                >
                  <SelectTrigger className="mt-1.5">
                    <SelectValue placeholder="Select split" />
                  </SelectTrigger>
                  <SelectContent>
                    {datasetInfo.splits.map((split) => (
                      <SelectItem key={split} value={split}>
                        {split}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              ) : (
                <Input
                  id="split"
                  placeholder="train"
                  className="mt-1.5"
                  value={config.split}
                  onChange={(e) => updateConfig({ split: e.target.value })}
                />
              )}
            </div>
            {datasetInfo.configs.length > 0 && (
              <div>
                <Label htmlFor="subset">Config/Subset</Label>
                <Select
                  value={config.subset || '__default__'}
                  onValueChange={(value) => updateConfig({ subset: value === '__default__' ? undefined : value })}
                >
                  <SelectTrigger className="mt-1.5">
                    <SelectValue placeholder="Default" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__default__">Default</SelectItem>
                    {datasetInfo.configs.map((cfg) => (
                      <SelectItem key={cfg} value={cfg}>
                        {cfg}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}
          </div>
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
          Choose the analyzers you want to run on your dataset. Use AI to help you configure based on your goals.
        </p>
      </div>

      {/* AI Suggestions Section */}
      {suggestionsStatus === 'idle' ? (
        <Card className="border-primary/30 bg-gradient-to-r from-primary/5 to-primary/10">
          <CardContent className="p-4">
            <div className="flex items-start gap-3 mb-3">
              <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary/10 mt-0.5">
                <Sparkles className="h-4 w-4 text-primary" />
              </div>
              <div className="flex-1">
                <h4 className="font-medium text-sm">Get AI Suggestions</h4>
                <p className="text-xs text-muted-foreground mt-1">
                  Let AI analyze your dataset and suggest analyzers, custom metrics, and tests to prepare your data for SFT fine-tuning.
                </p>
              </div>
            </div>
            
            <div className="space-y-3 ml-11">
              <div>
                <Label htmlFor="suggestion-query" className="text-xs font-medium">
                  Describe your goals (optional)
                </Label>
                <Textarea
                  id="suggestion-query"
                  className="mt-1.5 text-sm"
                  placeholder="e.g., This is a coding assistant dataset. I want to check for incomplete responses, ensure code blocks are properly formatted, and catch any conversations where the model refused to help..."
                  value={suggestionQuery}
                  onChange={(e) => setSuggestionQuery(e.target.value)}
                  rows={3}
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Describe your dataset and what issues you want to catch. Leave empty for general suggestions.
                </p>
              </div>
              
              <Button 
                onClick={requestAISuggestions}
                disabled={!config.datasetPath && !config.datasetName}
                className="w-full"
              >
                <Sparkles className="h-4 w-4 mr-2" />
                Analyze Dataset & Get Suggestions
              </Button>
            </div>
          </CardContent>
        </Card>
      ) : suggestionsDismissed && (unappliedAnalyzers.length > 0 || unappliedCustomMetrics.length > 0) ? (
        <SuggestionPanelMinimized
          suggestionCount={unappliedAnalyzers.length + unappliedCustomMetrics.length}
          onClick={undismissSuggestions}
        />
      ) : (
        <SuggestionPanel
          status={suggestionsStatus}
          error={suggestionsError}
          isDismissed={suggestionsDismissed}
          userPrompt={suggestionsUserPrompt}
          onDismiss={dismissSuggestions}
          onUndismiss={undismissSuggestions}
          type="analyzers"
          analyzerSuggestions={suggestions?.analyzers || []}
          customMetricSuggestions={suggestions?.custom_metrics || []}
          appliedAnalyzers={appliedAnalyzers}
          appliedCustomMetrics={appliedCustomMetrics}
          onApplyAnalyzer={applyAnalyzerSuggestion}
          onApplyCustomMetric={applyCustomMetricSuggestion}
          onApplyAll={applyAllAnalyzerSuggestions}
          onEdit={editSuggestionsPrompt}
        />
      )}

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
                const displayName = analyzer.instanceId 
                  ? `Custom LLM: ${analyzer.instanceId}` 
                  : analyzerDef.name
                const metricPrefix = analyzer.instanceId || analyzer.type
                return (
                  <Card key={index}>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <h5 className="font-medium">{displayName}</h5>
                          <p className="text-xs text-muted-foreground">
                            Metrics: {analyzerDef.metrics.map(m => `${metricPrefix}.${m}`).join(', ')}
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
                        <div className="space-y-3">
                          <div className="grid grid-cols-2 gap-3">
                            {analyzerDef.params.filter(p => p.type !== 'textarea').map(param => (
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
                          {/* Textarea params (like prompt_template) get full width */}
                          {analyzerDef.params.filter(p => p.type === 'textarea').map(param => (
                            <div key={param.key}>
                              <Label className="text-xs">{param.label}</Label>
                              <p className="text-xs text-muted-foreground mb-1">
                                Use {'{target}'} as placeholder for the conversation content
                              </p>
                              <textarea
                                className="mt-1 w-full h-32 text-xs font-mono p-2 border rounded-md bg-background resize-y"
                                value={analyzer.params[param.key] as string}
                                onChange={(e) => 
                                  updateAnalyzerParam(index, param.key, e.target.value)
                                }
                              />
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

      {/* Custom Python Metrics Section */}
      <Separator className="my-6" />
      <div>
        <div className="flex items-center justify-between mb-2">
          <div>
            <h3 className="text-lg font-medium flex items-center gap-2">
              <Code className="h-5 w-5 text-orange-500" />
              Custom Python Metrics
            </h3>
            <p className="text-sm text-muted-foreground mt-1">
              Extract custom fields from conversations using Python functions.
            </p>
          </div>
          <Button variant="outline" size="sm" onClick={addCustomMetric}>
            <Plus className="h-4 w-4 mr-2" />
            Add Metric
          </Button>
        </div>

        {config.customMetrics.length > 0 && (
          <div className="space-y-4 mt-4">
            {config.customMetrics.map((metric, index) => (
              <Card key={index}>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h5 className="font-medium">Custom Metric {index + 1}</h5>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeCustomMetric(index)}
                    >
                      <Trash2 className="h-4 w-4 text-red-500" />
                    </Button>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-3 mb-3">
                    <div>
                      <Label className="text-xs">Metric ID</Label>
                      <Input
                        className="mt-1 h-8 text-xs"
                        placeholder="e.g., ground_truth"
                        value={metric.id}
                        onChange={(e) => updateCustomMetric(index, { id: e.target.value })}
                      />
                    </div>
                    <div>
                      <Label className="text-xs">Scope</Label>
                      <Select
                        value={metric.scope}
                        onValueChange={(value) => updateCustomMetric(index, { scope: value })}
                      >
                        <SelectTrigger className="mt-1 h-8 text-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="conversation">Conversation</SelectItem>
                          <SelectItem value="message">Message</SelectItem>
                          <SelectItem value="dataset">Dataset</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  
                  {/* Output Schema */}
                  <div className="mb-3">
                    <div className="flex items-center justify-between mb-2">
                      <Label className="text-xs">Output Schema (required)</Label>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 text-xs"
                        onClick={() => {
                          const currentSchema = (metric.output_schema as Array<{name: string; type: string; description?: string}>) || []
                          updateCustomMetric(index, {
                            output_schema: [...currentSchema, { name: '', type: 'string', description: '' }]
                          })
                        }}
                      >
                        <Plus className="h-3 w-3 mr-1" />
                        Add Field
                      </Button>
                    </div>
                    <p className="text-xs text-muted-foreground mb-2">
                      Define the fields your function returns (must match your return dict keys)
                    </p>
                    
                    {((metric.output_schema as Array<{name: string; type: string; description?: string}>) || []).length === 0 ? (
                      <div className="text-xs text-muted-foreground italic p-2 border border-dashed rounded">
                        No output fields defined. Click "Add Field" to define what your function returns.
                      </div>
                    ) : (
                      <div className="space-y-2">
                        {((metric.output_schema as Array<{name: string; type: string; description?: string}>) || []).map((field, fieldIdx) => (
                          <div key={fieldIdx} className="flex items-center gap-2 p-2 bg-muted/30 rounded">
                            <Input
                              className="h-7 text-xs flex-1"
                              placeholder="field_name"
                              value={field.name}
                              onChange={(e) => {
                                const currentSchema = [...((metric.output_schema as Array<{name: string; type: string; description?: string}>) || [])]
                                currentSchema[fieldIdx] = { ...currentSchema[fieldIdx], name: e.target.value }
                                updateCustomMetric(index, { output_schema: currentSchema })
                              }}
                            />
                            <Select
                              value={field.type}
                              onValueChange={(value) => {
                                const currentSchema = [...((metric.output_schema as Array<{name: string; type: string; description?: string}>) || [])]
                                currentSchema[fieldIdx] = { ...currentSchema[fieldIdx], type: value }
                                updateCustomMetric(index, { output_schema: currentSchema })
                              }}
                            >
                              <SelectTrigger className="h-7 text-xs w-24">
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="string">string</SelectItem>
                                <SelectItem value="int">int</SelectItem>
                                <SelectItem value="float">float</SelectItem>
                                <SelectItem value="bool">bool</SelectItem>
                                <SelectItem value="any">any</SelectItem>
                              </SelectContent>
                            </Select>
                            <Input
                              className="h-7 text-xs flex-1"
                              placeholder="description (optional)"
                              value={field.description || ''}
                              onChange={(e) => {
                                const currentSchema = [...((metric.output_schema as Array<{name: string; type: string; description?: string}>) || [])]
                                currentSchema[fieldIdx] = { ...currentSchema[fieldIdx], description: e.target.value }
                                updateCustomMetric(index, { output_schema: currentSchema })
                              }}
                            />
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-7 w-7 p-0"
                              onClick={() => {
                                const currentSchema = [...((metric.output_schema as Array<{name: string; type: string; description?: string}>) || [])]
                                currentSchema.splice(fieldIdx, 1)
                                updateCustomMetric(index, { output_schema: currentSchema })
                              }}
                            >
                              <X className="h-3 w-3 text-red-500" />
                            </Button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  <div>
                    <Label className="text-xs">Python Function</Label>
                    <p className="text-xs text-muted-foreground mb-1">
                      Return a dict with your metric values. Access conversation.messages and conversation.metadata.
                    </p>
                    <textarea
                      className="mt-1 w-full h-40 text-xs font-mono p-2 border rounded-md bg-muted/50 resize-y"
                      value={metric.function}
                      onChange={(e) => updateCustomMetric(index, { function: e.target.value })}
                    />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
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

        {/* AI Suggestions Panel for Tests - filtered to only show relevant tests */}
        {(() => {
          const relevantTests = getRelevantTestSuggestions()
          const unappliedRelevantTests = relevantTests.filter(t => !appliedTests.has(t.id))
          
          if (suggestionsDismissed && unappliedRelevantTests.length > 0) {
            return (
              <SuggestionPanelMinimized
                suggestionCount={unappliedRelevantTests.length}
                onClick={undismissSuggestions}
              />
            )
          }
          
          return (
            <SuggestionPanel
              status={suggestionsStatus}
              error={suggestionsError}
              isDismissed={suggestionsDismissed}
              userPrompt={suggestionsUserPrompt}
              onDismiss={dismissSuggestions}
              onUndismiss={undismissSuggestions}
              type="tests"
              testSuggestions={relevantTests}
              appliedTests={appliedTests}
              onApplyTest={applyTestSuggestion}
              onApplyAll={applyAllTestSuggestions}
              onEdit={editSuggestionsPrompt}
            />
          )
        })()}

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
                  <div className="relative mt-1">
                    <Input
                      className="h-8 text-xs pr-8"
                      placeholder="e.g., length.total_tokens"
                      value={test.metric}
                      onChange={(e) => updateTest(index, { metric: e.target.value })}
                      list={`metrics-${index}`}
                    />
                    <datalist id={`metrics-${index}`}>
                      {availableMetrics.filter(m => !m.includes('(specify')).map(m => (
                        <option key={m} value={m} />
                      ))}
                    </datalist>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Type or select from suggestions
                  </p>
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
                <div className="space-y-3">
                  <p className="text-xs text-muted-foreground">
                    Flag samples where metric {test.operator || '>'} value. Use <strong>Max %</strong> to allow up to X% violations, 
                    or <strong>Min %</strong> to require at least X% match the condition.
                  </p>
                  <div className="grid grid-cols-3 gap-3">
                    <div>
                      <Label className="text-xs">Value Type</Label>
                      <Select
                        value={test.valueType || 'number'}
                        onValueChange={(value) => updateTest(index, { 
                          valueType: value as 'number' | 'boolean',
                          value: value === 'boolean' ? true : 0,
                          // Reset operator to == when switching to boolean if current operator is not valid
                          operator: value === 'boolean' && !['==', '!='].includes(test.operator || '') ? '==' : test.operator
                        })}
                      >
                        <SelectTrigger className="mt-1 h-8 text-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="number">Number</SelectItem>
                          <SelectItem value="boolean">Boolean</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
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
                          {(test.valueType || 'number') === 'boolean' ? (
                            <>
                              <SelectItem value="==">{'=='}</SelectItem>
                              <SelectItem value="!=">{'!='}</SelectItem>
                            </>
                          ) : (
                            <>
                              <SelectItem value=">">{'>'}</SelectItem>
                              <SelectItem value=">=">{'>='}</SelectItem>
                              <SelectItem value="<">{'<'}</SelectItem>
                              <SelectItem value="<=">{'<='}</SelectItem>
                              <SelectItem value="==">{'=='}</SelectItem>
                              <SelectItem value="!=">{'!='}</SelectItem>
                            </>
                          )}
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label className="text-xs">Value</Label>
                      {(test.valueType || 'number') === 'boolean' ? (
                        <Select
                          value={String(test.value ?? true)}
                          onValueChange={(value) => updateTest(index, { value: value === 'true' })}
                        >
                          <SelectTrigger className="mt-1 h-8 text-xs">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="true">True</SelectItem>
                            <SelectItem value="false">False</SelectItem>
                          </SelectContent>
                        </Select>
                      ) : (
                        <Input
                          className="mt-1 h-8 text-xs"
                          type="number"
                          value={typeof test.value === 'number' ? test.value : 0}
                          onChange={(e) => updateTest(index, { value: parseFloat(e.target.value) })}
                        />
                      )}
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <Label className="text-xs">Max % Allowed</Label>
                      <Input
                        className="mt-1 h-8 text-xs"
                        type="number"
                        placeholder="e.g., 10 (at most 10% violate)"
                        value={test.maxPercentage ?? ''}
                        onChange={(e) => updateTest(index, { 
                          maxPercentage: e.target.value ? parseFloat(e.target.value) : undefined 
                        })}
                      />
                    </div>
                    <div>
                      <Label className="text-xs">Min % Required</Label>
                      <Input
                        className="mt-1 h-8 text-xs"
                        type="number"
                        placeholder="e.g., 90 (at least 90% match)"
                        value={test.minPercentage ?? ''}
                        onChange={(e) => updateTest(index, { 
                          minPercentage: e.target.value ? parseFloat(e.target.value) : undefined 
                        })}
                      />
                    </div>
                  </div>
                </div>
              )}

              {test.type === 'percentage' && (
                <div className="space-y-3">
                  <p className="text-xs text-muted-foreground">
                    Set the required percentage of samples that must match the condition (e.g., passed == True).
                    Use <strong>Min %</strong> to require at least X% pass, or <strong>Max %</strong> to allow at most X% to match.
                  </p>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <Label className="text-xs">Min Percentage</Label>
                      <Input
                        className="mt-1 h-8 text-xs"
                        type="number"
                        placeholder="e.g., 80 (at least 80% pass)"
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
                        placeholder="e.g., 10 (at most 10% fail)"
                        value={test.maxPercentage ?? ''}
                        onChange={(e) => updateTest(index, { 
                          maxPercentage: e.target.value ? parseFloat(e.target.value) : undefined 
                        })}
                      />
                    </div>
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

        {/* Custom Metrics (read-only display) */}
        {config.customMetrics.length > 0 && (
          <>
            <Separator className="my-6" />
            <div>
              <h3 className="text-lg font-medium mb-2 flex items-center gap-2">
                <Code className="h-5 w-5 text-orange-500" />
                Custom Metrics
                <Badge variant="secondary" className="ml-2">{config.customMetrics.length}</Badge>
              </h3>
              <p className="text-sm text-muted-foreground mb-4">
                These custom metrics are defined in your config and will be preserved.
                Edit them directly in the YAML/JSON config if needed.
              </p>
            </div>

            <div className="space-y-3">
              {config.customMetrics.map((metric, index) => (
                <Card key={index} className="bg-muted/50">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <Badge variant="outline">{metric.id}</Badge>
                        <span className="text-xs text-muted-foreground">scope: {metric.scope}</span>
                      </div>
                      {metric.depends_on && metric.depends_on.length > 0 && (
                        <span className="text-xs text-muted-foreground">
                          depends on: {metric.depends_on.join(', ')}
                        </span>
                      )}
                    </div>
                    <div className="text-xs font-mono bg-background p-2 rounded max-h-24 overflow-auto">
                      <pre className="whitespace-pre-wrap">{metric.function.slice(0, 200)}{metric.function.length > 200 ? '...' : ''}</pre>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </>
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

        <div className={cn("grid gap-4", config.customMetrics.length > 0 ? "grid-cols-4" : "grid-cols-3")}>
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
          {config.customMetrics.length > 0 && (
            <Card>
              <CardContent className="p-4 text-center">
                <Code className="h-8 w-8 mx-auto mb-2 text-orange-500" />
                <p className="text-sm font-medium">Custom Metrics</p>
                <p className="text-xs text-muted-foreground">
                  {config.customMetrics.length} defined
                </p>
              </CardContent>
            </Card>
          )}
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
                {jobStatus?.progress && jobStatus?.total 
                  ? `${jobStatus.progress} / ${jobStatus.total} samples processed`
                  : 'Processing...'}
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
              onClick={handleNextStep}
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
