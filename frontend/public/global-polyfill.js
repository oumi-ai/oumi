// Enhanced polyfill for Electron renderer process
// Provides minimal require() support for remaining dependencies

console.log('🔧 Loading enhanced polyfills for Electron renderer');

// Ensure global exists for compatibility
if (typeof global === 'undefined') {
  var global = (typeof globalThis !== 'undefined') ? globalThis : window;
}

// Minimal require polyfill for remaining dependencies
if (typeof require === 'undefined') {
  var require = function(id) {
    console.log('📦 Polyfill require() called for:', id);
    
    // Return empty objects for Node.js modules
    if (id === 'buffer' || id === 'process' || id === 'util' || id === 'crypto' || 
        id === 'fs' || id === 'path' || id === 'os' || id === 'stream' || 
        id === 'events' || id === 'url' || id === 'querystring') {
      return {};
    }
    
    // For unknown modules, return empty object
    return {};
  };
  
  // Make require available globally
  if (typeof window !== 'undefined') {
    window.require = require;
  }
  if (typeof globalThis !== 'undefined') {
    globalThis.require = require;
  }
}

console.log('✅ Enhanced polyfills loaded successfully');