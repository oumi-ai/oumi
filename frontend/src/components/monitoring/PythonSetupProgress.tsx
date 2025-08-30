'use client';

import { useState, useEffect, useCallback } from 'react';
import { Bot, Download, Settings, CheckCircle, AlertCircle, X, FolderOpen } from 'lucide-react';

interface SetupProgress {
  step: string;
  progress: number;
  message: string;
  isComplete: boolean;
  estimatedTimeRemaining?: string;
}

interface PythonSetupProgressProps {
  isVisible: boolean;
  onCancel?: () => void;
  onComplete?: () => void;
  onError?: (error: string) => void;
}

const SETUP_STEPS = [
  { id: 'checking', label: 'Checking Requirements', icon: Settings },
  { id: 'creating', label: 'Creating Environment', icon: FolderOpen },
  { id: 'venv', label: 'Setting up Virtual Environment', icon: Settings },
  { id: 'uv', label: 'Installing Package Manager', icon: Download },
  { id: 'oumi', label: 'Installing AI Dependencies', icon: Bot },
  { id: 'testing', label: 'Testing Installation', icon: CheckCircle },
  { id: 'finishing', label: 'Finalizing Setup', icon: CheckCircle },
  { id: 'complete', label: 'Setup Complete', icon: CheckCircle }
];

export default function PythonSetupProgress({ 
  isVisible, 
  onCancel, 
  onComplete, 
  onError 
}: PythonSetupProgressProps) {
  const [progress, setProgress] = useState<SetupProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [startTime, setStartTime] = useState<Date | null>(null);
  const [userDataPath, setUserDataPath] = useState<string>('');

  // Get user data path for display
  useEffect(() => {
    if (isVisible && typeof window !== 'undefined' && window.electronAPI) {
      window.electronAPI.python.getUserDataPath()
        .then((path: string) => setUserDataPath(path))
        .catch(console.error);
    }
  }, [isVisible]);

  // Setup progress handler
  const handleProgress = useCallback((progressData: SetupProgress) => {
    setProgress(progressData);
    
    if (progressData.isComplete && onComplete) {
      onComplete();
    }
  }, [onComplete]);

  // Setup error handler  
  const handleError = useCallback((errorMessage: string) => {
    setError(errorMessage);
    if (onError) {
      onError(errorMessage);
    }
  }, [onError]);

  // Initialize progress monitoring when visible
  useEffect(() => {
    if (isVisible && typeof window !== 'undefined' && window.electronAPI) {
      setStartTime(new Date());
      setError(null);
      setProgress(null);
      
      // Start monitoring setup progress
      window.electronAPI.python.onSetupProgress(handleProgress);
      window.electronAPI.python.onSetupError(handleError);
      
      return () => {
        // Cleanup listeners when component unmounts or becomes hidden
        window.electronAPI.python.offSetupProgress(handleProgress);
        window.electronAPI.python.offSetupError(handleError);
      };
    }
  }, [isVisible, handleProgress, handleError]);

  // Calculate elapsed time
  const getElapsedTime = useCallback(() => {
    if (!startTime) return '0:00';
    
    const now = new Date();
    const elapsed = Math.floor((now.getTime() - startTime.getTime()) / 1000);
    const minutes = Math.floor(elapsed / 60);
    const seconds = elapsed % 60;
    
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }, [startTime]);

  // Get current step info
  const getCurrentStep = () => {
    if (!progress) return null;
    
    return SETUP_STEPS.find(step => step.id === progress.step) || {
      id: progress.step,
      label: progress.message,
      icon: Settings
    };
  };

  // Handle cancel
  const handleCancel = () => {
    if (window.electronAPI) {
      window.electronAPI.python.cancelSetup();
    }
    if (onCancel) {
      onCancel();
    }
  };

  if (!isVisible) {
    return null;
  }

  const currentStep = getCurrentStep();
  const currentProgress = progress?.progress || 0;
  const isError = !!error;

  return (
    <div className="fixed inset-0 bg-background/95 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-card border rounded-lg shadow-lg p-8 w-full max-w-md mx-4 relative">
        {/* Close button (only show if not in progress or if error) */}
        {(isError || currentProgress === 0 || currentProgress === 100) && onCancel && (
          <button 
            onClick={handleCancel}
            className="absolute top-4 right-4 text-muted-foreground hover:text-foreground transition-colors"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        )}

        {/* Header */}
        <div className="text-center mb-6">
          <div className="flex items-center justify-center mb-4">
            <Bot className="w-12 h-12 text-primary mr-3" />
            <div>
              <h1 className="text-2xl font-bold text-foreground">Setting up Chatterley</h1>
              <p className="text-muted-foreground">Preparing your AI environment...</p>
            </div>
          </div>
        </div>

        {/* Error State */}
        {isError && (
          <div className="mb-6 p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
            <div className="flex items-center mb-2">
              <AlertCircle className="w-5 h-5 text-destructive mr-2" />
              <span className="font-medium text-destructive">Setup Failed</span>
            </div>
            <p className="text-sm text-muted-foreground mb-3">{error}</p>
            <button 
              onClick={handleCancel}
              className="px-4 py-2 bg-destructive text-destructive-foreground rounded-md hover:bg-destructive/90 transition-colors"
            >
              Close
            </button>
          </div>
        )}

        {/* Progress State */}
        {!isError && (
          <>
            {/* Current Step */}
            {currentStep && (
              <div className="mb-6">
                <div className="flex items-center mb-3">
                  <currentStep.icon className="w-5 h-5 text-primary mr-3" />
                  <span className="font-medium text-foreground">{currentStep.label}</span>
                </div>
                
                <div className="text-sm text-muted-foreground mb-2">
                  {progress?.message || 'Initializing...'}
                </div>

                {/* Progress Bar */}
                <div className="w-full bg-muted rounded-full h-2 mb-2">
                  <div 
                    className="bg-primary h-2 rounded-full transition-all duration-500 ease-out"
                    style={{ width: `${currentProgress}%` }}
                  />
                </div>

                {/* Progress Text */}
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>{currentProgress}% complete</span>
                  <span>Elapsed: {getElapsedTime()}</span>
                </div>

                {/* Estimated Time */}
                {progress?.estimatedTimeRemaining && (
                  <div className="text-xs text-muted-foreground mt-1 text-center">
                    Est. remaining: {progress.estimatedTimeRemaining}
                  </div>
                )}
              </div>
            )}

            {/* Step List */}
            <div className="mb-6">
              <h3 className="font-medium text-foreground mb-3">Setup Progress</h3>
              <div className="space-y-2">
                {SETUP_STEPS.map((step, index) => {
                  const isCurrentStep = currentStep?.id === step.id;
                  const isCompleted = progress && (
                    SETUP_STEPS.findIndex(s => s.id === progress.step) > index ||
                    (isCurrentStep && progress.isComplete)
                  );
                  
                  return (
                    <div 
                      key={step.id}
                      className={`flex items-center p-2 rounded ${
                        isCurrentStep ? 'bg-primary/10 border border-primary/20' : ''
                      }`}
                    >
                      <div className="flex items-center flex-1">
                        {isCompleted ? (
                          <CheckCircle className="w-4 h-4 text-primary mr-3" />
                        ) : isCurrentStep ? (
                          <div className="w-4 h-4 mr-3 flex items-center justify-center">
                            <div className="w-2 h-2 bg-primary rounded-full animate-pulse" />
                          </div>
                        ) : (
                          <div className="w-4 h-4 mr-3 flex items-center justify-center">
                            <div className="w-2 h-2 bg-muted-foreground/30 rounded-full" />
                          </div>
                        )}
                        <span className={`text-sm ${
                          isCompleted ? 'text-primary' : 
                          isCurrentStep ? 'text-foreground font-medium' : 
                          'text-muted-foreground'
                        }`}>
                          {step.label}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Environment Info */}
            {userDataPath && (
              <div className="p-4 bg-muted/50 rounded-lg mb-6">
                <h4 className="font-medium text-foreground mb-2">Environment Location</h4>
                <p className="text-xs text-muted-foreground font-mono break-all">
                  {userDataPath}
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  Files will be installed here. You can remove them later if needed.
                </p>
              </div>
            )}

            {/* Cancel Button (only during active setup) */}
            {currentProgress > 0 && currentProgress < 100 && onCancel && (
              <div className="text-center">
                <button 
                  onClick={handleCancel}
                  className="px-4 py-2 text-muted-foreground hover:text-foreground border border-border hover:border-primary/50 rounded-md transition-colors"
                >
                  Cancel Setup
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}