/**
 * useErrorHandler - Custom hook for centralized error handling
 */

import { useState, useCallback } from 'react';
import { ErrorInfo } from '@/components/ui/ErrorDialog';

export interface ErrorHandlerOptions {
  showTechnicalDetails?: boolean;
  logToConsole?: boolean;
  canDismiss?: boolean;
}

export default function useErrorHandler() {
  const [currentError, setCurrentError] = useState<ErrorInfo | null>(null);

  const showError = useCallback((
    title: string,
    message: string,
    error?: Error | string,
    options: ErrorHandlerOptions & {
      actions?: ErrorInfo['actions'];
    } = {}
  ) => {
    const {
      showTechnicalDetails = true,
      logToConsole = true,
      canDismiss = true,
      actions
    } = options;

    // Log to console for debugging
    if (logToConsole) {
      console.error(`[${title}] ${message}`, error);
    }

    // Format technical details
    let details: string | undefined;
    if (showTechnicalDetails && error) {
      if (error instanceof Error) {
        details = `Error: ${error.message}\n\nStack Trace:\n${error.stack || 'No stack trace available'}`;
      } else if (typeof error === 'string') {
        details = error;
      }
    }

    setCurrentError({
      title,
      message,
      details,
      actions,
      canDismiss
    });
  }, []);

  const showModelTestError = useCallback((error: string, onRetry?: () => void, onGoBack?: () => void) => {
    showError(
      'Model Test Failed',
      'The selected model could not be loaded or tested. This might be due to insufficient resources, missing dependencies, or network issues.',
      error,
      {
        actions: {
          primary: onRetry ? {
            label: 'Retry Test',
            action: onRetry
          } : undefined,
          secondary: onGoBack ? {
            label: 'Choose Different Model',
            action: onGoBack
          } : undefined
        }
      }
    );
  }, [showError]);

  const showServerError = useCallback((error: string, onRestart?: () => void, onGoBack?: () => void) => {
    showError(
      'Server Error',
      'The chat server encountered an error and could not start properly. This might be due to port conflicts or configuration issues.',
      error,
      {
        actions: {
          primary: onRestart ? {
            label: 'Restart Server',
            action: onRestart
          } : undefined,
          secondary: onGoBack ? {
            label: 'Go Back',
            action: onGoBack
          } : undefined
        }
      }
    );
  }, [showError]);

  const showConfigError = useCallback((error: string, onRetry?: () => void) => {
    showError(
      'Configuration Error',
      'There was a problem loading the model configuration. The configuration file might be corrupted or inaccessible.',
      error,
      {
        actions: {
          primary: onRetry ? {
            label: 'Retry',
            action: onRetry
          } : undefined,
          secondary: {
            label: 'Choose Different Model',
            action: () => window.location.reload()
          }
        }
      }
    );
  }, [showError]);

  const showNetworkError = useCallback((error: string, onRetry?: () => void) => {
    showError(
      'Network Error',
      'Could not connect to the chat server. The server might be starting up, or there might be a network issue.',
      error,
      {
        actions: {
          primary: onRetry ? {
            label: 'Retry Connection',
            action: onRetry
          } : undefined
        }
      }
    );
  }, [showError]);

  const showDownloadError = useCallback((error: string, onRetry?: () => void, onSkip?: () => void) => {
    showError(
      'Download Failed',
      'Failed to download model files. This might be due to network issues or insufficient disk space.',
      error,
      {
        actions: {
          primary: onRetry ? {
            label: 'Retry Download',
            action: onRetry
          } : undefined,
          secondary: onSkip ? {
            label: 'Try Different Model',
            action: onSkip
          } : undefined
        }
      }
    );
  }, [showError]);

  const showGenericError = useCallback((error: string, onGoBack?: () => void) => {
    showError(
      'Something went wrong',
      'An unexpected error occurred. The application will return to the previous screen.',
      error,
      {
        actions: {
          primary: onGoBack ? {
            label: 'Go Back',
            action: onGoBack
          } : {
            label: 'OK',
            action: () => setCurrentError(null)
          }
        }
      }
    );
  }, [showError]);

  const clearError = useCallback(() => {
    setCurrentError(null);
  }, []);

  return {
    currentError,
    showError,
    showModelTestError,
    showServerError,
    showConfigError,
    showNetworkError,
    showDownloadError,
    showGenericError,
    clearError
  };
}