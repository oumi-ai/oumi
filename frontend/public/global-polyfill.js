// Global polyfills for browser compatibility
// This fixes "global is not defined" and "require is not defined" errors in production builds

// Fix global reference
if (typeof global === 'undefined') {
  var global = globalThis;
}

// Fix require reference for client-side with common Node.js modules
if (typeof require === 'undefined') {
  var require = function(module) {
    // Handle common Node.js modules that might be referenced
    switch (module) {
      case 'process':
        return process;
      case 'buffer':
        return { Buffer: globalThis.Buffer || {} };
      case 'util':
        return { 
          inherits: function(ctor, superCtor) {
            ctor.super_ = superCtor;
            ctor.prototype = Object.create(superCtor.prototype);
          }
        };
      default:
        console.warn('Module not available in browser:', module);
        return {};
    }
  };
}

// Fix process reference 
if (typeof process === 'undefined') {
  var process = {
    env: {},
    browser: true,
    version: '',
    versions: {},
    nextTick: function(fn) { setTimeout(fn, 0); }
  };
}