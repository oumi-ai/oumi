const isDev = process.env.NODE_ENV === 'development';
const isElectron = process.env.ELECTRON_BUILD === 'true';

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable static export for Electron builds
  output: isElectron ? 'export' : undefined,
  
  // Disable server-side features for Electron
  trailingSlash: isElectron ? true : undefined,
  images: isElectron ? { unoptimized: true } : undefined,
  
  // Base path for static export (if needed)
  basePath: isElectron && !isDev ? '' : undefined,
  
  // Asset prefix for Electron
  assetPrefix: isElectron && !isDev ? './' : undefined,
  
  // Webpack configuration for Electron compatibility
  webpack: (config, { isServer, webpack }) => {
    // Electron-specific webpack configuration
    if (isElectron && !isServer) {
      // Set the correct target for Electron renderer process
      config.target = 'electron-renderer';
      
      // Fix global is not defined error for Electron
      config.plugins.push(
        new webpack.DefinePlugin({
          global: 'globalThis',
          'process.env.ELECTRON': JSON.stringify(true),
          'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV || 'production'),
        })
      );

      // Add Node.js polyfills for Electron renderer
      config.plugins.push(
        new webpack.ProvidePlugin({
          Buffer: ['buffer', 'Buffer'],
          process: 'process/browser',
        })
      );
    } else if (!isServer) {
      // Web-specific configuration (when not Electron)
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        os: false,
        crypto: false,
        stream: false,
        assert: false,
        http: false,
        https: false,
        url: false,
        zlib: false,
        buffer: false,
        util: false,
      };
      
      config.plugins.push(
        new webpack.DefinePlugin({
          global: 'globalThis',
          'process.env.ELECTRON': JSON.stringify(false),
        })
      );
    }
    
    return config;
  },
  
  // Only use rewrites in non-Electron mode (web version)
  async rewrites() {
    if (isElectron) return [];
    
    // Use dynamic backend URL from environment, fallback to default
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:9000';
    
    return [
      {
        source: '/api/:path*',
        destination: `${backendUrl}/v1/:path*`, // Proxy to dynamic FastAPI backend
      },
    ]
  },
  
  // ESLint configuration
  eslint: {
    // Allow production builds with ESLint errors (for initial testing)
    ignoreDuringBuilds: true,
  },
  
  // TypeScript configuration
  typescript: {
    // Allow production builds with TypeScript errors (for initial testing)
    ignoreBuildErrors: false,
  },
  
  // Experimental features
  experimental: {
    // Enable modern bundling optimizations
    optimizePackageImports: ['lucide-react', 'react-markdown'],
  },
};

module.exports = nextConfig;