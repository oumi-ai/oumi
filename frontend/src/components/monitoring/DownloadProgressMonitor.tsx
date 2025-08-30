/**
 * Download progress monitoring component with real-time model download visibility
 */

"use client";

import React from 'react';
import { Download, AlertCircle, CheckCircle, Clock } from 'lucide-react';
import { DownloadProgress, DownloadState, DownloadErrorEvent } from '@/lib/types';

interface DownloadProgressBarProps {
  download: DownloadProgress;
}

const DownloadProgressBar: React.FC<DownloadProgressBarProps> = ({ download }) => {
  const percentage = download.progress;
  
  // Determine color based on completion status
  let barColor = 'bg-blue-500';
  if (download.isComplete) {
    barColor = 'bg-green-500';
  }

  return (
    <div className="space-y-2 p-3 bg-card border border-border rounded-lg">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <div className="flex-shrink-0">
            {download.isComplete ? (
              <CheckCircle className="w-4 h-4 text-green-500" />
            ) : (
              <Download className="w-4 h-4 text-blue-500" />
            )}
          </div>
          <span className="text-sm font-medium truncate text-foreground" title={download.filename}>
            {download.filename}
          </span>
        </div>
        <div className="text-xs text-muted-foreground ml-2 flex-shrink-0">
          {download.downloaded} / {download.total}
        </div>
      </div>
      
      <div className="w-full bg-muted rounded-full h-2">
        <div
          className={`h-2 rounded-full transition-all duration-300 ${barColor}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      
      <div className="flex justify-between items-center text-xs text-muted-foreground">
        <div className="flex items-center gap-4">
          <span className="font-medium">{percentage}%</span>
          <span>{download.speed}</span>
        </div>
        {download.estimatedTimeRemaining && !download.isComplete && (
          <div className="flex items-center gap-1">
            <Clock className="w-3 h-3" />
            <span>{download.estimatedTimeRemaining}</span>
          </div>
        )}
      </div>
    </div>
  );
};

interface DownloadProgressMonitorProps {
  downloadState: DownloadState;
  className?: string;
}

const DownloadProgressMonitor: React.FC<DownloadProgressMonitorProps> = ({ 
  downloadState, 
  className = '' 
}) => {
  if (!downloadState.isDownloading && downloadState.downloads.size === 0) {
    return null;
  }

  const downloads = Array.from(downloadState.downloads.values());
  const activeDownloads = downloads.filter(d => !d.isComplete);
  const completedDownloads = downloads.filter(d => d.isComplete);

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="flex items-center justify-center gap-2">
          {downloadState.hasError ? (
            <AlertCircle className="w-5 h-5 text-red-500" />
          ) : downloadState.isDownloading ? (
            <Download className="w-5 h-5 text-blue-500 animate-bounce" />
          ) : (
            <CheckCircle className="w-5 h-5 text-green-500" />
          )}
          <h3 className="text-lg font-semibold text-foreground">
            {downloadState.hasError 
              ? 'Download Error' 
              : downloadState.isDownloading 
                ? 'Downloading Model Files' 
                : 'Download Complete'
            }
          </h3>
        </div>
        
        {downloadState.totalFiles > 0 && (
          <p className="text-sm text-muted-foreground">
            {downloadState.completedFiles} of {downloadState.totalFiles} files complete
          </p>
        )}

        {downloadState.hasError && downloadState.errorMessage && (
          <div className="bg-red-50 border border-red-200 rounded-md p-3 text-sm text-red-800">
            {downloadState.errorMessage}
          </div>
        )}
      </div>

      {/* Overall progress */}
      {downloadState.totalFiles > 0 && (
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-foreground">Overall Progress</span>
            <span className="text-muted-foreground">{downloadState.overallProgress.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-muted rounded-full h-3">
            <div
              className={`h-3 rounded-full transition-all duration-500 ${
                downloadState.hasError 
                  ? 'bg-red-500' 
                  : downloadState.overallProgress === 100 
                    ? 'bg-green-500' 
                    : 'bg-gradient-to-r from-blue-500 to-purple-500'
              }`}
              style={{ width: `${downloadState.overallProgress}%` }}
            />
          </div>
        </div>
      )}

      {/* Individual download progress */}
      {downloads.length > 0 && (
        <div className="max-h-64 overflow-y-auto space-y-3">
          {/* Show active downloads first */}
          {activeDownloads.map(download => (
            <DownloadProgressBar key={download.filename} download={download} />
          ))}
          
          {/* Show completed downloads */}
          {completedDownloads.length > 0 && (
            <>
              {activeDownloads.length > 0 && (
                <div className="border-t border-border pt-3">
                  <p className="text-xs text-muted-foreground mb-3">Completed:</p>
                </div>
              )}
              {completedDownloads.slice(-3).map(download => (
                <DownloadProgressBar key={download.filename} download={download} />
              ))}
              {completedDownloads.length > 3 && (
                <div className="text-xs text-muted-foreground text-center py-2">
                  ... and {completedDownloads.length - 3} more completed
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Status message */}
      {!downloadState.hasError && downloadState.isDownloading && (
        <div className="text-center">
          <p className="text-sm text-muted-foreground">
            This may take several minutes for large models...
          </p>
        </div>
      )}
    </div>
  );
};

export default DownloadProgressMonitor;
export type { DownloadProgressBarProps };