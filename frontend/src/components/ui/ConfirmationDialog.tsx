/**
 * ConfirmationDialog - A reusable confirmation dialog component for destructive operations
 */

import React from 'react';
import { AlertTriangle, X, CheckCircle } from 'lucide-react';

export interface ConfirmationDialogProps {
  /**
   * Whether the dialog is open
   */
  isOpen: boolean;
  
  /**
   * The title of the dialog
   */
  title: string;
  
  /**
   * The main message of the dialog
   */
  message: string;
  
  /**
   * Optional additional details
   */
  detail?: string;
  
  /**
   * Optional confirmation text that user needs to type
   */
  confirmationText?: string;
  
  /**
   * Label for the confirm button
   */
  confirmLabel: string;
  
  /**
   * Label for the cancel button
   */
  cancelLabel?: string;
  
  /**
   * Optional label for an alternate action button
   */
  alternateLabel?: string;
  
  /**
   * Whether this is a dangerous/destructive operation
   */
  dangerous?: boolean;
  
  /**
   * Callback when the confirm button is clicked
   */
  onConfirm: () => void;
  
  /**
   * Callback when the cancel button is clicked
   */
  onCancel: () => void;
  
  /**
   * Optional callback for the alternate action button
   */
  onAlternate?: () => void;
  
  /**
   * Whether to show a loading indicator while processing
   */
  isLoading?: boolean;
  
  /**
   * Optional progress details when operation is in progress
   */
  progressDetails?: string[];
  
  /**
   * Optional success message to show after completion
   */
  successMessage?: string;
}

export default function ConfirmationDialog({
  isOpen,
  title,
  message,
  detail,
  confirmationText,
  confirmLabel,
  cancelLabel = 'Cancel',
  alternateLabel,
  dangerous = false,
  onConfirm,
  onCancel,
  onAlternate,
  isLoading = false,
  progressDetails = [],
  successMessage
}: ConfirmationDialogProps) {
  const [inputText, setInputText] = React.useState('');
  const [showProgress, setShowProgress] = React.useState(false);
  const [isSuccess, setIsSuccess] = React.useState(false);
  
  // Reset state when dialog opens or closes
  React.useEffect(() => {
    if (isOpen) {
      setInputText('');
      setShowProgress(false);
      setIsSuccess(false);
    }
  }, [isOpen]);
  
  // Show progress UI when loading starts
  React.useEffect(() => {
    if (isLoading && isOpen) {
      setShowProgress(true);
    }
  }, [isLoading, isOpen]);
  
  // Show success UI when success message is provided and not loading
  React.useEffect(() => {
    if (successMessage && !isLoading && isOpen) {
      setIsSuccess(true);
      // Auto-close after success
      const timer = setTimeout(() => {
        onCancel();
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [successMessage, isLoading, isOpen, onCancel]);
  
  // Don't render if dialog is closed
  if (!isOpen) return null;
  
  // Check if confirmation text matches
  const isConfirmationValid = !confirmationText || inputText === confirmationText;
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-card rounded-lg shadow-xl max-w-md w-full border border-border overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border">
          <div className="flex items-center space-x-3">
            {isSuccess ? (
              <CheckCircle className="w-6 h-6 text-green-500" />
            ) : (
              <AlertTriangle className={`w-6 h-6 ${dangerous ? 'text-red-500' : 'text-amber-500'}`} />
            )}
            <h2 className="text-lg font-semibold text-foreground">
              {isSuccess ? 'Operation Complete' : title}
            </h2>
          </div>
          <button
            onClick={onCancel}
            className="text-muted-foreground hover:text-foreground transition-colors"
            disabled={isLoading}
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {isSuccess ? (
            <p className="text-foreground mb-2 flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-green-500" />
              {successMessage}
            </p>
          ) : showProgress ? (
            <div>
              <p className="text-muted-foreground mb-4">
                Please wait while the operation completes...
              </p>
              <div className="my-4">
                <div className="w-full bg-muted rounded-full h-2 mb-4">
                  <div className="bg-primary h-2 rounded-full animate-pulse" 
                    style={{ width: '100%' }}></div>
                </div>
                
                {/* Progress details */}
                {progressDetails.length > 0 && (
                  <div className="mt-3 max-h-32 overflow-y-auto text-sm text-muted-foreground bg-muted/50 rounded-md p-2">
                    {progressDetails.map((detail, i) => (
                      <p key={i} className="mb-1">{detail}</p>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <>
              <p className="text-foreground mb-2">{message}</p>
              {detail && <p className="text-sm text-muted-foreground mt-2 mb-4">{detail}</p>}
              
              {/* Confirmation text input */}
              {confirmationText && (
                <div className="mt-4">
                  <p className="text-sm text-muted-foreground mb-2">
                    Type <span className="font-medium">{confirmationText}</span> to confirm:
                  </p>
                  <input
                    type="text"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    placeholder={confirmationText}
                    className="w-full px-3 py-2 bg-background text-foreground border border-border rounded-md"
                    autoFocus
                  />
                </div>
              )}
            </>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center justify-end space-x-3 p-6 border-t border-border bg-muted/50">
          {!isSuccess && !isLoading && (
            <>
              <button
                onClick={onCancel}
                className="px-4 py-2 text-muted-foreground hover:text-foreground transition-colors font-medium"
              >
                {cancelLabel}
              </button>
              
              {alternateLabel && onAlternate && (
                <button
                  onClick={onAlternate}
                  className="px-4 py-2 bg-amber-500 text-white rounded-lg hover:opacity-90 transition-opacity font-medium"
                  disabled={!isConfirmationValid || isLoading}
                >
                  {alternateLabel}
                </button>
              )}
              
              <button
                onClick={onConfirm}
                className={`px-6 py-2 ${dangerous ? 'bg-red-500' : 'bg-primary'} text-white rounded-lg hover:opacity-90 transition-opacity font-medium`}
                disabled={!isConfirmationValid || isLoading}
              >
                {confirmLabel}
              </button>
            </>
          )}
          
          {isSuccess && (
            <button
              onClick={onCancel}
              className="px-6 py-2 bg-primary text-white rounded-lg hover:opacity-90 transition-opacity font-medium"
            >
              Close
            </button>
          )}
        </div>
      </div>
    </div>
  );
}