/**
 * ErrorDialog - Reusable error dialog component with informative messages
 */

import React from 'react';
import { AlertCircle, X, RefreshCw, ArrowLeft, Bug } from 'lucide-react';

export interface ErrorInfo {
  title: string;
  message: string;
  details?: string;
  actions?: {
    primary?: {
      label: string;
      action: () => void;
    };
    secondary?: {
      label: string;
      action: () => void;
    };
  };
  canDismiss?: boolean;
}

interface ErrorDialogProps {
  error: ErrorInfo | null;
  onClose: () => void;
}

export default function ErrorDialog({ error, onClose }: ErrorDialogProps) {
  const [showDetails, setShowDetails] = React.useState(false);

  if (!error) return null;

  const handleClose = () => {
    if (error.canDismiss !== false) {
      onClose();
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-card rounded-lg shadow-xl max-w-md w-full border border-border">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border">
          <div className="flex items-center space-x-3">
            <AlertCircle className="w-6 h-6 text-red-500" />
            <h2 className="text-lg font-semibold text-foreground">{error.title}</h2>
          </div>
          {error.canDismiss !== false && (
            <button
              onClick={handleClose}
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>

        {/* Content */}
        <div className="p-6">
          <p className="text-muted-foreground mb-4">{error.message}</p>

          {/* Details section (collapsible) */}
          {error.details && (
            <div className="mb-4">
              <button
                onClick={() => setShowDetails(!showDetails)}
                className="flex items-center space-x-2 text-sm text-primary hover:opacity-80 transition-opacity"
              >
                <Bug className="w-4 h-4" />
                <span>{showDetails ? 'Hide' : 'Show'} Technical Details</span>
              </button>
              
              {showDetails && (
                <div className="mt-3 p-3 bg-muted rounded-lg border">
                  <pre className="text-xs text-muted-foreground whitespace-pre-wrap overflow-auto max-h-32">
                    {error.details}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center justify-end space-x-3 p-6 border-t border-border bg-muted/50">
          {error.actions?.secondary && (
            <button
              onClick={error.actions.secondary.action}
              className="px-4 py-2 text-muted-foreground hover:text-foreground transition-colors font-medium"
            >
              {error.actions.secondary.label}
            </button>
          )}
          
          {error.actions?.primary && (
            <button
              onClick={error.actions.primary.action}
              className="px-6 py-2 bg-primary text-primary-foreground rounded-lg hover:opacity-90 transition-opacity font-medium flex items-center space-x-2"
            >
              <span>{error.actions.primary.label}</span>
              {error.actions.primary.label.toLowerCase().includes('retry') && (
                <RefreshCw className="w-4 h-4" />
              )}
              {error.actions.primary.label.toLowerCase().includes('back') && (
                <ArrowLeft className="w-4 h-4" />
              )}
            </button>
          )}
          
          {/* Default dismiss button if no primary action */}
          {!error.actions?.primary && error.canDismiss !== false && (
            <button
              onClick={handleClose}
              className="px-6 py-2 bg-primary text-primary-foreground rounded-lg hover:opacity-90 transition-opacity font-medium"
            >
              OK
            </button>
          )}
        </div>
      </div>
    </div>
  );
}