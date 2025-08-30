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
        <script src="./global-polyfill.js" />
      </head>
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}