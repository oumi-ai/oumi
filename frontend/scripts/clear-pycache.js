/**
 * Script to clear Python cache files before building Electron app
 * 
 * This helps prevent stale .pyc files and __pycache__ directories from causing 
 * issues with loading updated Python modules.
 */

const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

// Get the oumi package root directory
const getOumiRootPath = () => {
  // Go up from frontend/scripts to the oumi root
  return path.resolve(__dirname, '../../');
};

// Clear Python cache
const clearPythonCache = () => {
  try {
    const oumiRoot = getOumiRootPath();
    console.log(`Clearing Python cache in: ${oumiRoot}`);
    
    // Commands to run
    const commands = [
      // Find and delete .pyc files
      `find "${oumiRoot}" -name "*.pyc" -delete`,
      // Find and delete __pycache__ directories
      `find "${oumiRoot}" -name "__pycache__" -type d -exec rm -rf {} +`
    ];
    
    // Execute commands
    commands.forEach(cmd => {
      try {
        execSync(cmd, { stdio: 'inherit' });
      } catch (error) {
        // Some find commands may fail if permissions are denied or paths don't exist
        // Just continue with the process
        console.log(`Warning: Command had non-zero exit: ${cmd}`);
      }
    });
    
    console.log('Python cache successfully cleared!');
  } catch (error) {
    console.error('Error clearing Python cache:', error);
    process.exit(1);
  }
};

// Run the script
clearPythonCache();