import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Chatterley: Powered by Oumi",
  description: "Advanced AI chat interface with conversation branching",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        {/* CRITICAL: This script MUST load synchronously before webpack chunks */}
        <script src="./global-polyfill.js"></script>
        <style dangerouslySetInnerHTML={{
          __html: `
            #electron-loading-screen {
              position: fixed;
              top: 0;
              left: 0;
              right: 0;
              bottom: 0;
              background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
              display: flex;
              flex-direction: column;
              justify-content: center;
              align-items: center;
              z-index: 9999;
              color: white;
              font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            
            .loading-logo {
              width: 80px;
              height: 80px;
              margin-bottom: 30px;
              border-radius: 16px;
              background: rgba(255,255,255,0.1);
              display: flex;
              align-items: center;
              justify-content: center;
              font-size: 36px;
              font-weight: bold;
            }
            
            .loading-title {
              font-size: 28px;
              font-weight: 600;
              margin-bottom: 10px;
              text-align: center;
            }
            
            .loading-subtitle {
              font-size: 16px;
              opacity: 0.8;
              margin-bottom: 40px;
              text-align: center;
            }
            
            .loading-spinner {
              width: 40px;
              height: 40px;
              border: 3px solid rgba(255,255,255,0.3);
              border-top: 3px solid white;
              border-radius: 50%;
              animation: spin 1s linear infinite;
              margin-bottom: 20px;
            }
            
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
            
            .loading-message {
              font-size: 14px;
              opacity: 0.7;
              text-align: center;
              max-width: 400px;
              line-height: 1.4;
            }
            
            .loading-tip {
              font-size: 12px;
              opacity: 0.6;
              margin-top: 30px;
              text-align: center;
              max-width: 350px;
              line-height: 1.3;
            }
          `
        }} />
      </head>
      <body className="antialiased">
        <div id="electron-loading-screen">
          <div className="loading-logo">💬</div>
          <div className="loading-title">Chatterley</div>
          <div className="loading-subtitle">Powered by Oumi AI</div>
          <div className="loading-spinner"></div>
          <div className="loading-message">Loading chat interface...</div>
          <div className="loading-tip">
            ✨ Please wait while we initialize the application.<br/>
            Menu functions will be available once loading completes.
          </div>
        </div>
        {children}
      </body>
    </html>
  );
}