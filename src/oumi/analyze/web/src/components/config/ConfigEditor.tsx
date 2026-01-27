import { useState, useCallback } from 'react'
import Editor from '@monaco-editor/react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import type { EvalData } from '@/types/eval'
import { Copy, Check, FileCode, Settings, Play } from 'lucide-react'

interface ConfigEditorProps {
  evalData: EvalData
  onRunAnalysis?: (config: string) => void
}

// Convert config object to YAML string
function configToYaml(config: Record<string, unknown>): string {
  const toYamlValue = (value: unknown, indent: number = 0): string => {
    const spaces = '  '.repeat(indent)
    
    if (value === null || value === undefined) {
      return 'null'
    }
    
    if (typeof value === 'string') {
      // Check if needs quoting
      if (value.includes('\n') || value.includes(':') || value.includes('#')) {
        return `"${value.replace(/"/g, '\\"')}"`
      }
      return value
    }
    
    if (typeof value === 'number' || typeof value === 'boolean') {
      return String(value)
    }
    
    if (Array.isArray(value)) {
      if (value.length === 0) return '[]'
      return value.map((item) => {
        if (typeof item === 'object' && item !== null) {
          const objYaml = Object.entries(item)
            .map(([k, v], j) => {
              const prefix = j === 0 ? '- ' : '  '
              return `${spaces}${prefix}${k}: ${toYamlValue(v, indent + 1)}`
            })
            .join('\n')
          return objYaml
        }
        return `${spaces}- ${toYamlValue(item, indent + 1)}`
      }).join('\n')
    }
    
    if (typeof value === 'object') {
      const entries = Object.entries(value)
      if (entries.length === 0) return '{}'
      return entries
        .map(([k, v]) => `${spaces}${k}: ${toYamlValue(v, indent + 1)}`)
        .join('\n')
    }
    
    return String(value)
  }
  
  return Object.entries(config)
    .map(([key, value]) => {
      if (Array.isArray(value) || (typeof value === 'object' && value !== null)) {
        return `${key}:\n${toYamlValue(value, 1)}`
      }
      return `${key}: ${toYamlValue(value)}`
    })
    .join('\n\n')
}

export function ConfigEditor({ evalData, onRunAnalysis }: ConfigEditorProps) {
  const [yamlContent, setYamlContent] = useState(() => configToYaml(evalData.config))
  const [copied, setCopied] = useState(false)
  const [hasChanges, setHasChanges] = useState(false)

  const handleEditorChange = useCallback((value: string | undefined) => {
    if (value !== undefined) {
      setYamlContent(value)
      setHasChanges(true)
    }
  }, [])

  const handleCopy = useCallback(async () => {
    await navigator.clipboard.writeText(yamlContent)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }, [yamlContent])

  const handleRun = useCallback(() => {
    if (onRunAnalysis) {
      onRunAnalysis(yamlContent)
    }
  }, [yamlContent, onRunAnalysis])

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-lg">
            <FileCode className="h-5 w-5" />
            Configuration
            {hasChanges && (
              <Badge variant="secondary" className="text-xs">Modified</Badge>
            )}
          </CardTitle>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleCopy}
              className="h-8"
            >
              {copied ? (
                <>
                  <Check className="h-4 w-4 mr-1" />
                  Copied
                </>
              ) : (
                <>
                  <Copy className="h-4 w-4 mr-1" />
                  Copy
                </>
              )}
            </Button>
            {onRunAnalysis && (
              <Button
                size="sm"
                onClick={handleRun}
                className="h-8"
              >
                <Play className="h-4 w-4 mr-1" />
                Run Analysis
              </Button>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="yaml" className="w-full">
          <TabsList className="mb-3">
            <TabsTrigger value="yaml" className="text-xs">
              <FileCode className="h-3 w-3 mr-1" />
              YAML
            </TabsTrigger>
            <TabsTrigger value="json" className="text-xs">
              <Settings className="h-3 w-3 mr-1" />
              JSON
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="yaml" className="mt-0">
            <div className="border rounded-md overflow-hidden">
              <Editor
                height="400px"
                language="yaml"
                theme="vs-dark"
                value={yamlContent}
                onChange={handleEditorChange}
                options={{
                  minimap: { enabled: false },
                  fontSize: 13,
                  lineNumbers: 'on',
                  scrollBeyondLastLine: false,
                  wordWrap: 'on',
                  tabSize: 2,
                  automaticLayout: true,
                  readOnly: false,
                }}
              />
            </div>
          </TabsContent>
          
          <TabsContent value="json" className="mt-0">
            <div className="border rounded-md overflow-hidden">
              <Editor
                height="400px"
                language="json"
                theme="vs-dark"
                value={JSON.stringify(evalData.config, null, 2)}
                options={{
                  minimap: { enabled: false },
                  fontSize: 13,
                  lineNumbers: 'on',
                  scrollBeyondLastLine: false,
                  wordWrap: 'on',
                  tabSize: 2,
                  automaticLayout: true,
                  readOnly: true,
                }}
              />
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
