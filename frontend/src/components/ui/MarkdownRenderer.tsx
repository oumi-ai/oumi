/**
 * Lightweight markdown renderer using markdown-to-jsx
 * No Node.js polyfills required - perfect for Electron renderer
 */

"use client";

import React from 'react';
import Markdown from 'markdown-to-jsx';
import hljs from 'highlight.js';
import katex from 'katex';
import { Copy, Check } from 'lucide-react';

// Import KaTeX CSS
import 'katex/dist/katex.min.css';
// Import syntax highlighting CSS
import 'highlight.js/styles/github-dark.css';

interface CodeBlockProps {
  className?: string;
  children?: React.ReactNode;
}

const CodeBlock = ({ className, children, ...props }: CodeBlockProps) => {
  const [copied, setCopied] = React.useState(false);
  const match = /language-(\w+)/.exec(className || '');
  const language = match ? match[1] : '';
  const codeRef = React.useRef<HTMLElement>(null);

  React.useEffect(() => {
    if (codeRef.current && language && hljs.getLanguage(language)) {
      hljs.highlightElement(codeRef.current);
    }
  }, [language, children]);

  const handleCopy = async () => {
    const text = String(children).replace(/\n$/, '');
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error('Failed to copy code:', error);
    }
  };

  return (
    <div className="relative group">
      <div className="flex items-center justify-between bg-muted px-4 py-2 rounded-t-lg border-b">
        <span className="text-xs font-medium text-muted-foreground uppercase">
          {language || 'text'}
        </span>
        <button
          onClick={handleCopy}
          className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-muted-foreground/20"
          title="Copy code"
        >
          {copied ? (
            <Check size={14} className="text-green-600" />
          ) : (
            <Copy size={14} className="text-muted-foreground" />
          )}
        </button>
      </div>
      <pre className="overflow-x-auto bg-card rounded-b-lg">
        <code 
          ref={codeRef}
          className={`${className} block p-4 text-sm`} 
          {...props}
        >
          {children}
        </code>
      </pre>
    </div>
  );
};

const InlineCode = ({ children, ...props }: { children?: React.ReactNode }) => (
  <code className="px-1.5 py-0.5 bg-muted text-foreground rounded text-sm font-mono" {...props}>
    {children}
  </code>
);

// Math rendering component
const MathDisplay = ({ children }: { children: string }) => {
  const mathHtml = React.useMemo(() => {
    try {
      return katex.renderToString(children, {
        displayMode: true,
        throwOnError: false,
      });
    } catch (error) {
      console.warn('KaTeX rendering error:', error);
      return children;
    }
  }, [children]);

  return (
    <div className="my-4 text-center">
      <div 
        className="inline-block p-4 bg-muted/30 rounded-lg"
        dangerouslySetInnerHTML={{ __html: mathHtml }}
      />
    </div>
  );
};

const InlineMath = ({ children }: { children: string }) => {
  const mathHtml = React.useMemo(() => {
    try {
      return katex.renderToString(children, {
        displayMode: false,
        throwOnError: false,
      });
    } catch (error) {
      console.warn('KaTeX rendering error:', error);
      return children;
    }
  }, [children]);

  return (
    <span 
      className="inline-math"
      dangerouslySetInnerHTML={{ __html: mathHtml }}
    />
  );
};

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

export default function MarkdownRenderer({ content, className = '' }: MarkdownRendererProps) {
  // Pre-process content for math expressions
  const processedContent = React.useMemo(() => {
    let processed = content;
    
    // Handle display math ($$...$$)
    processed = processed.replace(/\$\$([^$]+)\$\$/g, (match, math) => {
      return `<MathDisplay>${math.trim()}</MathDisplay>`;
    });
    
    // Handle inline math ($...$)
    processed = processed.replace(/\$([^$\n]+)\$/g, (match, math) => {
      return `<InlineMath>${math.trim()}</InlineMath>`;
    });
    
    return processed;
  }, [content]);

  return (
    <div className={`prose prose-sm max-w-none dark:prose-invert ${className}`}>
      <Markdown
        options={{
          overrides: {
            // Code blocks
            code: CodeBlock,
            
            // Inline code
            inlineCode: InlineCode,
            
            // Math components
            MathDisplay: {
              component: MathDisplay,
            },
            InlineMath: {
              component: InlineMath,
            },
            
            // Custom blockquote styling
            blockquote: {
              component: ({ children }: { children?: React.ReactNode }) => (
                <blockquote className="border-l-4 border-primary pl-4 py-2 bg-muted/50 rounded-r italic">
                  {children}
                </blockquote>
              ),
            },
            
            // Custom table styling
            table: {
              component: ({ children }: { children?: React.ReactNode }) => (
                <div className="overflow-x-auto">
                  <table className="min-w-full border-collapse border border-border">
                    {children}
                  </table>
                </div>
              ),
            },
            
            th: {
              component: ({ children }: { children?: React.ReactNode }) => (
                <th className="border border-border px-4 py-2 bg-muted font-semibold text-left">
                  {children}
                </th>
              ),
            },
            
            td: {
              component: ({ children }: { children?: React.ReactNode }) => (
                <td className="border border-border px-4 py-2">
                  {children}
                </td>
              ),
            },
            
            // Custom link styling
            a: {
              component: ({ href, children }: { href?: string; children?: React.ReactNode }) => (
                <a 
                  href={href} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  {children}
                </a>
              ),
            },
            
            // Custom list styling
            ul: {
              component: ({ children }: { children?: React.ReactNode }) => (
                <ul className="list-disc list-inside space-y-1">
                  {children}
                </ul>
              ),
            },
            
            ol: {
              component: ({ children }: { children?: React.ReactNode }) => (
                <ol className="list-decimal list-inside space-y-1">
                  {children}
                </ol>
              ),
            },
            
            // Custom heading styling
            h1: {
              component: ({ children }: { children?: React.ReactNode }) => (
                <h1 className="text-2xl font-bold border-b border-border pb-2 mb-4">
                  {children}
                </h1>
              ),
            },
            
            h2: {
              component: ({ children }: { children?: React.ReactNode }) => (
                <h2 className="text-xl font-semibold mb-3">
                  {children}
                </h2>
              ),
            },
            
            h3: {
              component: ({ children }: { children?: React.ReactNode }) => (
                <h3 className="text-lg font-medium mb-2">
                  {children}
                </h3>
              ),
            },
            
            // Custom paragraph styling
            p: {
              component: ({ children }: { children?: React.ReactNode }) => (
                <p className="mb-3 leading-relaxed">
                  {children}
                </p>
              ),
            },
          },
        }}
      >
        {processedContent}
      </Markdown>
    </div>
  );
}