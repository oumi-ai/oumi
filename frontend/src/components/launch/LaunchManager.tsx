/**
 * Launch Manager - handles app initialization flow
 */

"use client";

import React from 'react';
import WelcomeScreen from '@/components/welcome/WelcomeScreen';
import AppLayout from '@/components/layout/AppLayout';
import DownloadProgressMonitor from '@/components/monitoring/DownloadProgressMonitor';
import { Loader2, AlertTriangle } from 'lucide-react';
import apiClient from '@/lib/unified-api';
import { DownloadState, DownloadProgress, DownloadErrorEvent } from '@/lib/types';

interface LaunchManagerProps {}

type LaunchState = 'welcome' | 'initializing' | 'ready' | 'error';

interface LaunchError {
  message: string;
  details?: string;
  canRetry?: boolean;
}

export default function LaunchManager({}: LaunchManagerProps) {
  const [launchState, setLaunchState] = React.useState<LaunchState>('welcome');
  const [selectedConfig, setSelectedConfig] = React.useState<string | null>(null);
  const [error, setError] = React.useState<LaunchError | null>(null);
  const [initProgress, setInitProgress] = React.useState<string>('Preparing...');
  const [downloadState, setDownloadState] = React.useState<DownloadState>({
    isDownloading: false,
    downloads: new Map(),
    overallProgress: 0,
    totalFiles: 0,
    completedFiles: 0,
    hasError: false
  });

  const handleConfigSelected = async (configId: string, systemPrompt?: string) => {
    setSelectedConfig(configId);
    setLaunchState('initializing');
    setError(null);
    
    console.log('🚀 Starting with config:', configId);
    if (systemPrompt) {
      console.log('🎭 Using system prompt:', systemPrompt.substring(0, 100) + '...');
    }

    try {
      // Step 1: Validate config selection and get config path
      setInitProgress('Validating configuration...');
      
      // Load config path from static configs
      let configPath: string | undefined;
      try {
        const response = await fetch('/static-configs.json');
        if (response.ok) {
          const data = await response.json();
          const selectedConfigData = data.configs?.find((cfg: any) => cfg.id === configId);
          configPath = selectedConfigData?.config_path;
          console.log('📍 Config path:', configPath);
        }
      } catch (err) {
        console.warn('Failed to load config path from static configs:', err);
      }
      
      await new Promise(resolve => setTimeout(resolve, 500));

      // Step 2: Start backend server if in Electron
      if (apiClient.isElectron && apiClient.isElectron()) {
        setInitProgress('Starting backend server...');
        
        // Check if server is already running
        const serverStatus = await apiClient.getServerStatus();
        if (!serverStatus.success || !serverStatus.data?.running) {
          // Start the server with the selected config and system prompt
          const startResult = await apiClient.startServer(configPath, systemPrompt);
          if (!startResult.success) {
            throw new Error(`Failed to start server: ${startResult.message}`);
          }
        }
      }
      
      // Wait for backend to be ready
      setInitProgress('Connecting to backend server...');
      let attempts = 0;
      const maxAttempts = 30; // 15 seconds max
      
      while (attempts < maxAttempts) {
        try {
          const healthResponse = await apiClient.health();
          if (healthResponse.success) {
            break;
          }
        } catch (err) {
          // Continue trying
        }
        
        attempts++;
        await new Promise(resolve => setTimeout(resolve, 500));
        
        if (attempts < maxAttempts) {
          setInitProgress(`Connecting to backend server... (${attempts}/${maxAttempts})`);
        }
      }

      if (attempts >= maxAttempts) {
        throw new Error('Backend server failed to respond within timeout period');
      }

      // Step 3: Load selected configuration
      setInitProgress('Loading model configuration...');
      
      // If this is an Electron app, we might want to set the initial config
      if (apiClient.isElectron && apiClient.isElectron()) {
        // Store the selected config and system prompt for future use
        await apiClient.setStorageItem('selectedConfig', configId);
        await apiClient.setStorageItem('hasCompletedWelcome', true);
        
        if (systemPrompt) {
          await apiClient.setStorageItem('systemPrompt', systemPrompt);
        }
      }

      // Step 4: Wait for any model downloads to complete
      setInitProgress('Checking for model downloads...');
      
      // Give the server a moment to start any downloads
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // If we're in Electron, monitor for downloads
      if (apiClient.isElectron && apiClient.isElectron()) {
        // Wait for downloads to complete or timeout after 10 seconds if no downloads detected
        const maxWaitTime = 10000; // 10 seconds
        const startTime = Date.now();
        
        while (Date.now() - startTime < maxWaitTime) {
          // If downloads are actively happening, wait for them to complete
          if (downloadState.isDownloading) {
            setInitProgress(`Downloading model files... ${downloadState.overallProgress.toFixed(1)}% complete`);
            await new Promise(resolve => setTimeout(resolve, 1000));
            continue;
          }
          
          // If we have completed downloads, we're done
          if (downloadState.downloads.size > 0 && !downloadState.isDownloading) {
            setInitProgress('Model download complete. Initializing chat interface...');
            break;
          }
          
          // Check if server is ready to handle API requests
          try {
            const healthResponse = await apiClient.health();
            if (healthResponse.success) {
              // Server is responding, no downloads detected, we can proceed
              break;
            }
          } catch (err) {
            // Server might still be busy, wait a bit more
          }
          
          await new Promise(resolve => setTimeout(resolve, 500));
        }
      }
      
      // Final initialization step
      setInitProgress('Initializing chat interface...');
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Ready to show main app
      setLaunchState('ready');
      
    } catch (err) {
      console.error('Launch initialization error:', err);
      setError({
        message: 'Failed to initialize Oumi Chat',
        details: err instanceof Error ? err.message : 'Unknown error occurred',
        canRetry: true
      });
      setLaunchState('error');
    }
  };

  const handleRetryLaunch = () => {
    if (selectedConfig) {
      handleConfigSelected(selectedConfig);
    } else {
      setLaunchState('welcome');
    }
  };

  const handleBackToWelcome = () => {
    setSelectedConfig(null);
    setError(null);
    setLaunchState('welcome');
  };

  // Add download progress monitoring
  React.useEffect(() => {
    if (!apiClient.isElectron || !apiClient.isElectron()) return;

    const handleDownloadProgress = (progress: DownloadProgress) => {
      setDownloadState(prev => {
        const newDownloads = new Map(prev.downloads);
        newDownloads.set(progress.filename, progress);
        
        const completed = Array.from(newDownloads.values()).filter(d => d.isComplete).length;
        const total = newDownloads.size;
        const overall = total > 0 ? (completed / total) * 100 : 0;
        
        return {
          ...prev,
          isDownloading: total > completed,
          downloads: newDownloads,
          overallProgress: overall,
          totalFiles: total,
          completedFiles: completed,
          hasError: false // Clear error when progress continues
        };
      });
    };

    const handleDownloadError = (error: DownloadErrorEvent) => {
      setDownloadState(prev => ({
        ...prev,
        hasError: true,
        errorMessage: error.message,
        isDownloading: false
      }));
    };

    // Add event listeners if electronAPI is available
    if (typeof window !== 'undefined' && window.electronAPI) {
      window.electronAPI.onDownloadProgress(handleDownloadProgress);
      window.electronAPI.onDownloadError(handleDownloadError);
    }
    
    return () => {
      if (typeof window !== 'undefined' && window.electronAPI) {
        window.electronAPI.removeDownloadProgressListener(handleDownloadProgress);
        window.electronAPI.removeDownloadErrorListener(handleDownloadError);
      }
    };
  }, []);

  // Check if user has opted into welcome screen caching (default: always show welcome)
  React.useEffect(() => {
    const checkPreviousSetup = async () => {
      try {
        // Check for debug flag to force welcome screen
        const forceWelcome = new URLSearchParams(window.location.search).get('welcome') === 'true';
        
        if (apiClient.isElectron && apiClient.isElectron() && !forceWelcome) {
          // NEW: Check if user has explicitly enabled welcome screen caching (default: false)
          const welcomeCachingEnabled = await apiClient.getStorageItem('enableWelcomeCaching', false);
          
          if (welcomeCachingEnabled) {
            const hasCompleted = await apiClient.getStorageItem('hasCompletedWelcome', false);
            const savedConfig = await apiClient.getStorageItem('selectedConfig', null);
            const savedPrompt = await apiClient.getStorageItem('systemPrompt', null);
            
            if (hasCompleted && savedConfig) {
              // Skip welcome screen and go straight to initialization
              setSelectedConfig(savedConfig);
              setTimeout(() => {
                handleConfigSelected(savedConfig, savedPrompt);
              }, 100); // Small delay to ensure state is updated
              return;
            }
          }
        }
        
        // DEFAULT: Always show welcome screen unless explicitly cached
        setLaunchState('welcome');
      } catch (err) {
        console.error('Error checking previous setup:', err);
        setLaunchState('welcome');
      }
    };

    checkPreviousSetup();
  }, []);

  // Render based on launch state
  switch (launchState) {
    case 'welcome':
      return <WelcomeScreen onConfigSelected={handleConfigSelected} />;
    
    case 'initializing':
      return (
        <div className="min-h-screen bg-background flex items-center justify-center">
          <div className="bg-card rounded-lg shadow-lg p-8 max-w-2xl w-full mx-4">
            {/* Show download progress if downloading */}
            {downloadState.isDownloading || downloadState.downloads.size > 0 ? (
              <DownloadProgressMonitor downloadState={downloadState} />
            ) : (
              /* Standard initialization UI */
              <div className="text-center">
                <Loader2 className="w-12 h-12 animate-spin mx-auto mb-6 text-primary" />
                <h2 className="text-xl font-semibold mb-2 text-foreground">Initializing Oumi Chat</h2>
                <p className="text-muted-foreground mb-4">{initProgress}</p>
                
                {/* Progress indicator */}
                <div className="w-full bg-muted rounded-full h-2">
                  <div 
                    className="bg-primary h-2 rounded-full transition-all duration-300 ease-out"
                    style={{ 
                      width: launchState === 'initializing' ? '60%' : '0%'
                    }}
                  ></div>
                </div>
              </div>
            )}
          </div>
        </div>
      );

    case 'error':
      return (
        <div className="min-h-screen bg-background flex items-center justify-center">
          <div className="bg-card rounded-lg shadow-lg p-8 max-w-md w-full mx-4 text-center">
            <AlertTriangle className="w-12 h-12 mx-auto mb-6 text-red-500" />
            <h2 className="text-xl font-semibold mb-2 text-red-700">Initialization Failed</h2>
            <p className="text-muted-foreground mb-2">{error?.message}</p>
            {error?.details && (
              <p className="text-sm text-muted-foreground mb-6 bg-muted p-3 rounded">
                {error.details}
              </p>
            )}
            
            <div className="flex flex-col sm:flex-row gap-3">
              {error?.canRetry && (
                <button 
                  onClick={handleRetryLaunch}
                  className="flex-1 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:opacity-90 transition-opacity"
                >
                  Retry
                </button>
              )}
              <button 
                onClick={handleBackToWelcome}
                className="flex-1 px-4 py-2 bg-muted text-foreground rounded-lg hover:bg-accent transition-colors"
              >
                Back to Welcome
              </button>
            </div>
          </div>
        </div>
      );

    case 'ready':
      return <AppLayout />;

    default:
      return (
        <div className="min-h-screen bg-background flex items-center justify-center">
          <div className="bg-card rounded-lg shadow-lg p-8 max-w-md w-full mx-4 text-center">
            <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4 text-primary" />
            <p className="text-muted-foreground">Loading...</p>
          </div>
        </div>
      );
  }
}