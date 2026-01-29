import { useState } from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import type { Conversation, AnalysisResult, Message, MessageContent } from '@/types/eval'
import { ChevronDown, ChevronUp, User, Bot, Settings, Info } from 'lucide-react'
import { cn, truncateText } from '@/lib/utils'

interface SampleConversationProps {
  index: number
  conversation: Conversation
  analysisResult: AnalysisResult
  keyPrefix?: string
  failureReason?: string
  metricName?: string
}

const TRUNCATE_LENGTH = 300

function getMessageContent(message: Message): string {
  if (typeof message.content === 'string') {
    return message.content
  }
  // Handle array of content (multimodal)
  return message.content
    .map((c: MessageContent) => c.text || '[non-text content]')
    .join(' ')
}

function MessageBubble({ message }: { message: Message }) {
  const [expanded, setExpanded] = useState(false)
  const content = getMessageContent(message)
  const isTruncated = content.length > TRUNCATE_LENGTH
  const displayContent = expanded ? content : truncateText(content, TRUNCATE_LENGTH)

  const roleConfig = {
    user: {
      icon: User,
      bgColor: 'bg-blue-50 dark:bg-blue-950',
      borderColor: 'border-blue-200 dark:border-blue-800',
      label: 'User',
    },
    assistant: {
      icon: Bot,
      bgColor: 'bg-green-50 dark:bg-green-950',
      borderColor: 'border-green-200 dark:border-green-800',
      label: 'Assistant',
    },
    system: {
      icon: Settings,
      bgColor: 'bg-gray-50 dark:bg-gray-900',
      borderColor: 'border-gray-200 dark:border-gray-700',
      label: 'System',
    },
  }

  const config = roleConfig[message.role as keyof typeof roleConfig] || roleConfig.user
  const Icon = config.icon

  return (
    <div className={cn('rounded-lg border p-3', config.bgColor, config.borderColor)}>
      <div className="flex items-center gap-2 mb-2">
        <Icon className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm font-medium">{config.label}</span>
      </div>
      <div className="text-sm whitespace-pre-wrap break-words">
        {displayContent}
      </div>
      {isTruncated && (
        <Button
          variant="ghost"
          size="sm"
          className="mt-2 h-6 text-xs"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? 'Show less' : 'Show more'}
        </Button>
      )}
    </div>
  )
}

export function SampleConversation({ 
  index, 
  conversation, 
  analysisResult,
  keyPrefix = '',
  failureReason,
  metricName
}: SampleConversationProps) {
  const [isOpen, setIsOpen] = useState(false)

  const score = analysisResult.score
  const passed = analysisResult.passed
  const label = analysisResult.label
  const reasoning = analysisResult.reasoning
  const error = analysisResult.error

  return (
    <Card className="mb-3">
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <CollapsibleTrigger asChild>
          <div className="flex items-center justify-between p-4 cursor-pointer hover:bg-muted/50 transition-colors">
            <div className="flex items-center gap-3">
              <span className="text-sm font-medium text-muted-foreground">
                Sample {index + 1}
              </span>
              
              {/* Score */}
              {score !== undefined && (
                <Badge variant="outline" className={cn(
                  'text-xs',
                  score >= 80 ? 'border-green-500 text-green-700' :
                  score >= 50 ? 'border-yellow-500 text-yellow-700' :
                  'border-red-500 text-red-700'
                )}>
                  Score: {score}
                </Badge>
              )}

              {/* Pass/Fail */}
              {passed !== undefined && (
                <Badge variant={passed ? 'default' : 'destructive'} className="text-xs">
                  {passed ? 'Passed' : 'Failed'}
                </Badge>
              )}

              {/* Label */}
              {label && (
                <Badge variant="secondary" className="text-xs">
                  {label}
                </Badge>
              )}

              {/* Error */}
              {error && (
                <Badge variant="destructive" className="text-xs">
                  Error
                </Badge>
              )}

              {/* Failure Reason (for non-LLM tests) */}
              {failureReason && !reasoning && (
                <span className="text-xs text-red-600 dark:text-red-400 font-mono">
                  {failureReason}
                </span>
              )}
            </div>

            <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
              {isOpen ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )}
            </Button>
          </div>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <CardContent className="pt-0 pb-4 px-4 space-y-4">
            {/* Failure Reason for non-LLM tests */}
            {failureReason && !reasoning && (
              <div className="bg-red-50 dark:bg-red-950 rounded-lg p-3 border border-red-200 dark:border-red-800">
                <div className="flex items-center gap-2 mb-2">
                  <Info className="h-4 w-4 text-red-600 dark:text-red-400" />
                  <span className="text-sm font-medium text-red-700 dark:text-red-300">Failed Condition</span>
                </div>
                <p className="text-sm font-mono text-red-700 dark:text-red-300">
                  {metricName}: {failureReason}
                </p>
              </div>
            )}

            {/* Reasoning (for LLM tests) */}
            {reasoning && (
              <div className="bg-muted/50 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-2">
                  <Info className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium">Reasoning</span>
                </div>
                <p className="text-sm text-muted-foreground">{reasoning}</p>
              </div>
            )}

            {/* Error message */}
            {error && (
              <div className="bg-red-50 dark:bg-red-950 rounded-lg p-3 border border-red-200 dark:border-red-800">
                <span className="text-sm text-red-700 dark:text-red-300">{error}</span>
              </div>
            )}

            {/* Messages */}
            <div className="space-y-3">
              <h4 className="text-sm font-medium">Conversation</h4>
              {conversation.messages.map((message, msgIndex) => (
                <MessageBubble 
                  key={`${keyPrefix}-msg-${msgIndex}`} 
                  message={message}
                />
              ))}
            </div>

            {/* Metadata */}
            {conversation.metadata && Object.keys(conversation.metadata).length > 0 && (
              <Collapsible>
                <CollapsibleTrigger asChild>
                  <Button variant="ghost" size="sm" className="text-xs">
                    <Settings className="h-3 w-3 mr-1" />
                    Metadata
                    <ChevronDown className="h-3 w-3 ml-1" />
                  </Button>
                </CollapsibleTrigger>
                <CollapsibleContent>
                  <pre className="mt-2 p-3 bg-muted rounded-lg text-xs overflow-auto max-h-40">
                    {JSON.stringify(conversation.metadata, null, 2)}
                  </pre>
                </CollapsibleContent>
              </Collapsible>
            )}
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  )
}
