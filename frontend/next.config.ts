import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    // Use dynamic backend URL from environment, fallback to default
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:9000';
    
    return [
      {
        source: '/api/:path*',
        destination: `${backendUrl}/v1/:path*`, // Proxy to dynamic FastAPI backend
      },
    ]
  },
};

export default nextConfig;