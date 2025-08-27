import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:9000/v1/:path*', // Proxy to FastAPI backend
      },
    ]
  },
};

export default nextConfig;