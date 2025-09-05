// Global Jest setup for frontend tests

// Provide minimal Electron IPC and storage stubs used by electron-api.ts
Object.defineProperty(window, 'electronAPI', {
  value: {
    events: {
      on: jest.fn(),
      off: jest.fn(),
      send: jest.fn(),
      invoke: jest.fn().mockResolvedValue(undefined),
    },
    storage: {
      get: jest.fn().mockResolvedValue(undefined),
      set: jest.fn().mockResolvedValue(undefined),
      delete: jest.fn().mockResolvedValue(undefined),
      clear: jest.fn().mockResolvedValue(undefined),
      getAllKeys: jest.fn().mockResolvedValue([]),
      resetWelcomeSettings: jest.fn().mockResolvedValue({ success: true }),
    },
    app: {
      getVersion: jest.fn().mockResolvedValue('test'),
      quit: jest.fn(),
      reload: jest.fn(),
      toggleDevTools: jest.fn(),
      toggleFullScreen: jest.fn(),
      zoom: jest.fn(),
    },
    chat: {},
    server: {},
    apiKeys: {},
    python: {},
  },
  configurable: true,
});

// Use fake timers to flush debounced saves between tests
beforeEach(() => {
  jest.useFakeTimers();
});

afterEach(() => {
  jest.runOnlyPendingTimers();
  jest.useRealTimers();
});
