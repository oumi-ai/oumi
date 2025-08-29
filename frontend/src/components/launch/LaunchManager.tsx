/**
 * Launch Manager - handles app initialization flow
 */

"use client";

import React from 'react';
import WelcomeScreen from '@/components/welcome/WelcomeScreen';
import AppLayout from '@/components/layout/AppLayout';
import { Loader2, AlertTriangle } from 'lucide-react';
import apiClient from '@/lib/unified-api';

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

  const handleConfigSelected = async (configId: string, systemPrompt?: string) => {
    setSelectedConfig(configId);
    setLaunchState('initializing');
    setError(null);
    
    console.log('🚀 Starting with config:', configId);
    if (systemPrompt) {
      console.log('🎭 Using system prompt:', systemPrompt.substring(0, 100) + '...');
    }

    try {
      // Step 1: Validate config selection
      setInitProgress('Validating configuration...');
      await new Promise(resolve => setTimeout(resolve, 500));

      // Step 2: Start backend server if in Electron
      if (apiClient.isElectron && apiClient.isElectron()) {
        setInitProgress('Starting backend server...');
        
        // Check if server is already running
        const serverStatus = await apiClient.getServerStatus();
        if (!serverStatus.success || !serverStatus.data?.running) {
          // Start the server
          const startResult = await apiClient.startServer();
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

      // Step 4: Initialize app state
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

  // Check if user has already completed welcome (for subsequent app launches)
  React.useEffect(() => {
    const checkPreviousSetup = async () => {
      try {
        if (apiClient.isElectron && apiClient.isElectron()) {
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
        
        // For web version or first-time users, show welcome
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
        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
          <div className="bg-white rounded-lg shadow-lg p-8 max-w-md w-full mx-4 text-center">
            <Loader2 className="w-12 h-12 animate-spin mx-auto mb-6 text-blue-600" />
            <h2 className="text-xl font-semibold mb-2">Initializing Oumi Chat</h2>
            <p className="text-gray-600 mb-4">{initProgress}</p>
            
            {/* Progress indicator */}
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-600 h-2 rounded-full transition-all duration-300 ease-out"
                style={{ 
                  width: launchState === 'initializing' ? '60%' : '0%'
                }}
              ></div>
            </div>
          </div>
        </div>
      );

    case 'error':
      return (
        <div className="min-h-screen bg-gradient-to-br from-red-50 to-pink-100 flex items-center justify-center">
          <div className="bg-white rounded-lg shadow-lg p-8 max-w-md w-full mx-4 text-center">
            <AlertTriangle className="w-12 h-12 mx-auto mb-6 text-red-500" />
            <h2 className="text-xl font-semibold mb-2 text-red-700">Initialization Failed</h2>
            <p className="text-gray-600 mb-2">{error?.message}</p>
            {error?.details && (
              <p className="text-sm text-gray-500 mb-6 bg-gray-50 p-3 rounded">
                {error.details}
              </p>
            )}
            
            <div className="flex flex-col sm:flex-row gap-3">
              {error?.canRetry && (
                <button 
                  onClick={handleRetryLaunch}
                  className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Retry
                </button>
              )}
              <button 
                onClick={handleBackToWelcome}
                className="flex-1 px-4 py-2 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300 transition-colors"
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
        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
          <div className="bg-white rounded-lg shadow-lg p-8 max-w-md w-full mx-4 text-center">
            <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4 text-blue-600" />
            <p className="text-gray-600">Loading...</p>
          </div>
        </div>
      );
  }
}