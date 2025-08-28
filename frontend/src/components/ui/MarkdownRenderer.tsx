/**
 * Markdown renderer with syntax highlighting and math support
 */

"use client";

import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import { Copy, Check } from 'lucide-react';

// Import KaTeX CSS
import 'katex/dist/katex.min.css';
// Import syntax highlighting CSS
import 'highlight.js/styles/github-dark.css';

interface CodeBlockProps {
  inline?: boolean;
  className?: string;
  children: React.ReactNode;
}

const CodeBlock = ({ inline, className, children, ...props }: CodeBlockProps) => {
  const [copied, setCopied] = React.useState(false);
  const match = /language-(\w+)/.exec(className || '');
  const language = match ? match[1] : '';

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

  if (inline) {
    return (
      <code className="px-1.5 py-0.5 bg-muted text-foreground rounded text-sm font-mono" {...props}>
        {children}
      </code>
    );
  }

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
        <code className={`${className} block p-4 text-sm`} {...props}>
          {children}
        </code>
      </pre>
    </div>
  );
};

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

export default function MarkdownRenderer({ content, className = '' }: MarkdownRendererProps) {
  return (
    <div className={`prose prose-sm max-w-none dark:prose-invert ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[
          rehypeKatex,
          [rehypeHighlight, { detect: true, ignoreMissing: true }]
        ]}
        components={{
          // Custom code block rendering
          code: CodeBlock,
          
          // Custom blockquote styling
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-primary pl-4 py-2 bg-muted/50 rounded-r italic">
              {children}
            </blockquote>
          ),
          
          // Custom table styling
          table: ({ children }) => (
            <div className="overflow-x-auto">
              <table className="min-w-full border-collapse border border-border">
                {children}
              </table>
            </div>
          ),
          
          th: ({ children }) => (
            <th className="border border-border px-4 py-2 bg-muted font-semibold text-left">
              {children}
            </th>
          ),
          
          td: ({ children }) => (
            <td className="border border-border px-4 py-2">
              {children}
            </td>
          ),
          
          // Custom link styling
          a: ({ href, children }) => (
            <a 
              href={href} 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-primary hover:underline"
            >
              {children}
            </a>
          ),
          
          // Custom list styling
          ul: ({ children }) => (
            <ul className="list-disc list-inside space-y-1">
              {children}
            </ul>
          ),
          
          ol: ({ children }) => (
            <ol className="list-decimal list-inside space-y-1">
              {children}
            </ol>
          ),
          
          // Custom heading styling
          h1: ({ children }) => (
            <h1 className="text-2xl font-bold border-b border-border pb-2 mb-4">
              {children}
            </h1>
          ),
          
          h2: ({ children }) => (
            <h2 className="text-xl font-semibold mb-3">
              {children}
            </h2>
          ),
          
          h3: ({ children }) => (
            <h3 className="text-lg font-medium mb-2">
              {children}
            </h3>
          ),
          
          // Custom paragraph styling
          p: ({ children }) => (
            <p className="mb-3 leading-relaxed">
              {children}
            </p>
          ),
          
          // Math display styling
          div: ({ className, children }) => {
            if (className?.includes('math-display')) {
              return (
                <div className="my-4 text-center">
                  <div className="inline-block p-4 bg-muted/30 rounded-lg">
                    {children}
                  </div>
                </div>
              );
            }
            return <div className={className}>{children}</div>;
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}