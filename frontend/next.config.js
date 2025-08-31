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
    // Apply configuration for client-side builds only
    if (!isServer) {
      // Set target for Electron renderer process
      if (isElectron) {
        config.target = 'electron-renderer';
      }
      
      // Define environment variables
      config.plugins.push(
        new webpack.DefinePlugin({
          global: 'globalThis',
          'process.env.ELECTRON': JSON.stringify(isElectron),
          'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV || 'production'),
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
    optimizePackageImports: ['lucide-react', 'markdown-to-jsx'],
  },
};

module.exports = nextConfig;