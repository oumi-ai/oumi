/**
 * Launch Manager - handles app initialization flow
 */

"use client";

import React from 'react';
import WelcomeScreen from '@/components/welcome/WelcomeScreen';
import AppLayout from '@/components/layout/AppLayout';
import DownloadProgressMonitor from '@/components/monitoring/DownloadProgressMonitor';
import ErrorDialog from '@/components/ui/ErrorDialog';
import useErrorHandler from '@/hooks/useErrorHandler';
import { Loader2, AlertTriangle } from 'lucide-react';
import apiClient from '@/lib/unified-api';
import { DownloadState, DownloadProgress, DownloadErrorEvent } from '@/lib/types';
import { configPathResolver } from '@/lib/config-path-resolver';
import { SystemCapabilities } from '@/lib/config-matcher';
import { logger } from '@/lib/logger';

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
  const [systemCapabilities, setSystemCapabilities] = React.useState<SystemCapabilities | null>(null);
  const [isLoadingSystemInfo, setIsLoadingSystemInfo] = React.useState(true);
  
  // Error handling
  const { 
    currentError, 
    showServerError, 
    showNetworkError,
    clearError 
  } = useErrorHandler();
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
      
      // Load config using unified path resolver (no frontend resolution)
      let configPath: string | undefined;
      try {
        const configResult = await configPathResolver.getConfigById(configId);
        if (configResult) {
          configPath = configResult.configPath;
          console.log('📍 Config path (raw, for backend):', configPath);
          console.log('📍 Config details:', configResult.config.display_name);
        } else {
          console.warn('Config not found:', configId);
        }
      } catch (err) {
        console.warn('Failed to load config path from unified resolver:', err);
      }
      
      await new Promise(resolve => setTimeout(resolve, 500));

      // Step 2: Start backend server if in Electron
      if (apiClient.isElectron && apiClient.isElectron()) {
        setInitProgress('Starting backend server...');
        
        // Check if server is already running
        const serverStatus = await apiClient.getServerStatus();
        if (!serverStatus.success || !serverStatus.data?.running) {
          // Start the server with the selected config and system prompt
          console.log('🚀 Starting server with config:', configPath);
          const startResult = await apiClient.startServer(configPath, systemPrompt);
          console.log('🚀 Server start result:', startResult);
          
          if (!startResult.success) {
            console.error('❌ Server start failed:', {
              success: startResult.success,
              message: startResult.message,
              error: startResult.error,
              configPath,
              stack: new Error().stack
            });
            throw new Error(`Failed to start server: ${startResult.message || startResult.error || 'Unknown error'}`);
          }
        }
      }
      
      // Wait for backend to be ready with exponential backoff
      setInitProgress('Connecting to backend server...');
      const startTime = Date.now();
      const maxWaitTime = 60000; // 60 seconds total
      let attempt = 0;
      let delay = 200; // Start with 200ms delay
      const maxDelay = 2000; // Cap at 2 seconds
      
      while (Date.now() - startTime < maxWaitTime) {
        try {
          const healthResponse = await apiClient.health();
          console.log(`🏥 Health check attempt ${attempt + 1}:`, healthResponse);
          
          if (healthResponse.success) {
            console.log('✅ Health check passed, server is ready');
            break;
          } else {
            console.warn(`⚠️ Health check failed (attempt ${attempt + 1}):`, healthResponse);
          }
        } catch (err) {
          console.warn(`❌ Health check error (attempt ${attempt + 1}):`, err);
          // Continue trying
        }
        
        // Update progress every few attempts
        if (attempt % 5 === 0) {
          const elapsed = Math.floor((Date.now() - startTime) / 1000);
          setInitProgress(`Connecting to backend server... (${elapsed}s elapsed)`);
          console.log(`🕐 Health check progress: ${elapsed}s elapsed, ${attempt + 1} attempts`);
        }
        
        await new Promise(resolve => setTimeout(resolve, delay));
        attempt++;
        
        // Exponential backoff with jitter
        delay = Math.min(maxDelay, delay * 1.3 + Math.random() * 100);
      }

      if (Date.now() - startTime >= maxWaitTime) {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        console.error('🚨 Server health check timeout:', {
          elapsed: `${elapsed}s`,
          attempts: attempt + 1,
          maxWaitTime: `${maxWaitTime / 1000}s`,
          configPath,
          stack: new Error().stack
        });
        throw new Error(`Server health check failed - server may not have started properly (${attempt + 1} attempts over ${elapsed}s)`);
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

      // Step 4: Initialize app state
      setInitProgress('Initializing chat interface...');
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Ready to show main app
      setLaunchState('ready');
      
    } catch (err) {
      console.error('Launch initialization error:', err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      
      // Determine the type of error and show appropriate dialog
      if (errorMessage.includes('Failed to start server')) {
        showServerError(
          errorMessage,
          () => {
            clearError();
            handleRetryLaunch();
          },
          () => {
            clearError();
            setLaunchState('welcome');
          }
        );
      } else if (errorMessage.includes('Backend server failed to respond')) {
        showNetworkError(
          errorMessage,
          () => {
            clearError();
            handleRetryLaunch();
          }
        );
      } else {
        // Generic error
        setError({
          message: 'Failed to initialize Chatterley',
          details: errorMessage,
          canRetry: true
        });
        setLaunchState('error');
      }
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

  // Handle welcome settings reset from menu
  const handleResetWelcomeSettings = async () => {
    try {
      const result = await apiClient.resetWelcomeSettings();
      if (result.success) {
        // Show success feedback - could use a toast or alert
        if (window.confirm('Welcome settings have been reset! The app will reload to show the welcome screen.')) {
          // Reload the app to show welcome screen
          if (apiClient.isElectron && apiClient.isElectron()) {
            apiClient.reload();
          } else {
            window.location.reload();
          }
        }
      } else {
        alert('Failed to reset welcome settings. Please try again.');
      }
    } catch (error) {
      console.error('Error resetting welcome settings:', error);
      alert('Failed to reset welcome settings. Please try again.');
    }
  };

  // Handle Python environment rebuild from menu
  const handleRebuildPythonEnvironment = async () => {
    console.log('🔧 [LaunchManager] Starting Python environment rebuild...');
    try {
      console.log('🔧 [LaunchManager] Removing existing Python environment...');
      
      // First, remove the existing environment
      const removeResult = await apiClient.removeEnvironment();
      console.log('🔧 [LaunchManager] Remove result:', removeResult);
      
      if (!removeResult.success) {
        alert(`Failed to remove existing Python environment: ${removeResult.message}`);
        console.error('❌ [LaunchManager] Environment removal failed:', removeResult.message);
        return;
      }
      
      console.log('✅ [LaunchManager] Existing environment removed successfully');
      
      // Show success message and restart instruction
      alert('Python environment has been cleared successfully!\n\nPlease restart the application to set up the environment again with the latest dependencies.');
      
      // Reset to welcome state to allow user to reconfigure
      setLaunchState('welcome');
      setSelectedConfig(null);
      setError(null);
      
      console.log('✅ [LaunchManager] Environment rebuild initiated - user should restart the app');
      
    } catch (error) {
      console.error('❌ [LaunchManager] Error rebuilding Python environment:', error);
      alert('Failed to rebuild Python environment. Please try again.');
    }
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

  // Load system capabilities first, then hide loading screen when React is ready
  React.useEffect(() => {
    const initializeApp = async () => {
      if (typeof window !== 'undefined' && window.electronAPI) {
        logger.debug('LaunchManager', 'React mounted, loading system capabilities');
        
        // Step 1: Load system capabilities
        try {
          const systemInfo = await apiClient.getEnvironmentSystemInfo();
          if (systemInfo) {
            const capabilities: SystemCapabilities = {
              platform: systemInfo.platform,
              architecture: systemInfo.architecture,
              totalRAM: systemInfo.totalRAM,
              cudaAvailable: systemInfo.cudaAvailable,
              cudaDevices: systemInfo.cudaDevices || []
            };
            setSystemCapabilities(capabilities);
            logger.info('LaunchManager', 'System capabilities loaded successfully', {
              platform: capabilities.platform,
              arch: capabilities.architecture,
              ram: `${capabilities.totalRAM}GB`,
              cuda: capabilities.cudaAvailable,
              gpus: capabilities.cudaDevices.length
            });
          } else {
            logger.warn('LaunchManager', 'No system capabilities available - smart recommendations disabled');
          }
        } catch (error) {
          logger.error('LaunchManager', 'Failed to load system capabilities', error);
        } finally {
          setIsLoadingSystemInfo(false);
        }
        
        // Step 2: Hide the loading screen
        console.log('🔧 [LaunchManager] Hiding loading screen...');
        try {
          await window.electronAPI.app.hideLoadingScreen();
          console.log('🔧 [LaunchManager] Loading screen hidden successfully');
        } catch (error) {
          console.error('🔧 [LaunchManager] Failed to hide loading screen:', error);
        }
      } else {
        // Web version - no system capabilities, just mark as loaded
        setIsLoadingSystemInfo(false);
      }
    };

    initializeApp();

    // Set up menu message handlers (available in all states)
    const setupMenuHandlers = () => {
      if (typeof window !== 'undefined' && window.electronAPI) {
        const handleResetWelcomeMessage = async () => {
          console.log('🔧 [LaunchManager] Received menu:reset-welcome-settings message');
          await handleResetWelcomeSettings();
        };

        const handleRebuildEnvironmentMessage = async () => {
          console.log('🔧 [LaunchManager] Received menu:rebuild-python-environment message');
          await handleRebuildPythonEnvironment();
        };

        console.log('🔧 [LaunchManager] Registering menu listeners');
        window.electronAPI.onMenuMessage('menu:reset-welcome-settings', handleResetWelcomeMessage);
        window.electronAPI.onMenuMessage('menu:rebuild-python-environment', handleRebuildEnvironmentMessage);

        return () => {
          console.log('🔧 [LaunchManager] Cleaning up menu listeners');
          window.electronAPI.removeMenuListener('menu:reset-welcome-settings', handleResetWelcomeMessage);
          window.electronAPI.removeMenuListener('menu:rebuild-python-environment', handleRebuildEnvironmentMessage);
        };
      }
    };

    setupMenuHandlers();
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
      // Show loading spinner while system info is being detected
      if (isLoadingSystemInfo) {
        return (
          <div className="min-h-screen bg-background flex items-center justify-center">
            <div className="bg-card rounded-lg shadow-lg p-8 max-w-md w-full mx-4 text-center">
              <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4 text-primary" />
              <p className="text-muted-foreground">Detecting system capabilities...</p>
            </div>
          </div>
        );
      }
      
      return (
        <>
          <WelcomeScreen 
            onConfigSelected={handleConfigSelected} 
            systemCapabilities={systemCapabilities}
          />
          <ErrorDialog error={currentError} onClose={clearError} />
        </>
      );
    
    case 'initializing':
      return (
        <>
          <div className="min-h-screen bg-background flex items-center justify-center">
            <div className="bg-card rounded-lg shadow-lg p-8 max-w-2xl w-full mx-4">
            {/* Show download progress if downloading */}
            {downloadState.isDownloading || downloadState.downloads.size > 0 ? (
              <DownloadProgressMonitor downloadState={downloadState} />
            ) : (
              /* Standard initialization UI */
              <div className="text-center">
                <Loader2 className="w-12 h-12 animate-spin mx-auto mb-6 text-primary" />
                <h2 className="text-xl font-semibold mb-2 text-foreground">Initializing Chatterley</h2>
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
          <ErrorDialog error={currentError} onClose={clearError} />
        </>
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
      return (
        <>
          <AppLayout />
          <ErrorDialog error={currentError} onClose={clearError} />
        </>
      );

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