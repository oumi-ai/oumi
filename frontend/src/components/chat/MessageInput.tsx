/**
 * Message input component with send functionality, image/file attachments, and fetch command
 */

"use client";

import React from 'react';
import { Send, Image, Paperclip, Loader2, Globe } from 'lucide-react';
import { isValidCommand } from '@/lib/constants';
import apiClient from '@/lib/unified-api';

interface AttachmentResult {
  success: boolean;
  files?: File[];
  error?: string;
}

interface StagedAttachment {
  id: string;
  file?: File;
  type: 'image' | 'document' | 'fetch';
  name: string;
  size?: number;
  fetchUrl?: string;
  fetchContent?: string;
}

interface MessageInputProps {
  onSendMessage: (message: string, attachments?: StagedAttachment[]) => void;
  onAttachFiles?: (files: FileList) => void;
  disabled?: boolean;
  isLoading?: boolean;
  placeholder?: string;
}

export default function MessageInput({
  onSendMessage,
  onAttachFiles,
  disabled = false,
  isLoading = false,
  placeholder = "Type your message...",
}: MessageInputProps) {
  const [message, setMessage] = React.useState('');
  const [stagedAttachments, setStagedAttachments] = React.useState<StagedAttachment[]>([]);
  const [showFetchDialog, setShowFetchDialog] = React.useState(false);
  const [fetchUrl, setFetchUrl] = React.useState('');
  const [isFetching, setIsFetching] = React.useState(false);
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);
  const imageInputRef = React.useRef<HTMLInputElement>(null);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  // Supported file types based on Oumi backend analysis
  const SUPPORTED_IMAGE_TYPES = '.jpg,.jpeg,.png,.gif,.bmp,.tiff,.webp,.svg';
  const SUPPORTED_FILE_TYPES = '.txt,.md,.rst,.log,.cfg,.ini,.conf,.py,.js,.ts,.html,.css,.java,.cpp,.c,.h,.go,.rs,.php,.rb,.swift,.kt,.scala,.sh,.yaml,.yml,.pdf,.csv,.json';
  const MAX_FILE_SIZE = 30 * 1024 * 1024; // 30MB limit

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if ((message.trim() || stagedAttachments.length > 0) && !disabled && !isLoading) {
      onSendMessage(message.trim(), stagedAttachments.length > 0 ? stagedAttachments : undefined);
      setMessage('');
      setStagedAttachments([]);
      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setMessage(e.target.value);
    
    // Auto-resize textarea
    const textarea = e.target;
    textarea.style.height = 'auto';
    const newHeight = Math.min(textarea.scrollHeight, 120); // Max height of 120px
    textarea.style.height = `${newHeight}px`;
  };

  // File validation functions
  const validateFile = (file: File, allowedTypes: string[], isImage: boolean = false): { valid: boolean; error?: string } => {
    // Check file size
    if (file.size > MAX_FILE_SIZE) {
      return { valid: false, error: `File "${file.name}" exceeds 30MB limit (${(file.size / (1024 * 1024)).toFixed(1)}MB)` };
    }

    // Check for empty files
    if (file.size === 0) {
      return { valid: false, error: `File "${file.name}" is empty` };
    }

    // Check file extension
    const extension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!allowedTypes.includes(extension)) {
      const typeLabel = isImage ? 'image' : 'document';
      return { valid: false, error: `File "${file.name}" is not a supported ${typeLabel} format` };
    }

    // Check for archives (common archive extensions)
    const archiveExtensions = ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz', '.tgz'];
    if (archiveExtensions.includes(extension)) {
      return { valid: false, error: `Archive files are not allowed: "${file.name}"` };
    }

    // Additional image validation
    if (isImage) {
      // Check MIME type if available
      if (file.type && !file.type.startsWith('image/')) {
        return { valid: false, error: `File "${file.name}" is not a valid image file` };
      }
    }

    return { valid: true };
  };

  const validateFiles = (files: FileList, allowedTypes: string, isImage: boolean = false): AttachmentResult => {
    const validFiles: File[] = [];
    const errors: string[] = [];
    const typeArray = allowedTypes.split(',');

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const validation = validateFile(file, typeArray, isImage);
      
      if (validation.valid) {
        validFiles.push(file);
      } else {
        errors.push(validation.error!);
      }
    }

    if (errors.length > 0) {
      return { success: false, error: errors.join('\n') };
    }

    return { success: true, files: validFiles };
  };

  // Check if current model supports multiple images (simplified heuristic)
  const getCurrentModelInfo = async () => {
    try {
      const response = await apiClient.getServerStatus();
      // This is a simplified check - in reality you'd get model capabilities from the config
      return {
        supportsMultipleImages: false, // Default to single image for now
        isVisionModel: false // Default to text-only for now
      };
    } catch (error) {
      return {
        supportsMultipleImages: false,
        isVisionModel: false
      };
    }
  };

  const handleImageSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;

    const validation = validateFiles(e.target.files, SUPPORTED_IMAGE_TYPES, true);
    
    if (!validation.success) {
      alert(`❌ Image Attachment Error:\n\n${validation.error}`);
      e.target.value = '';
      return;
    }

    const modelInfo = await getCurrentModelInfo();

    // For single-image models, warn about chat reset
    if (!modelInfo.supportsMultipleImages && (stagedAttachments.some(a => a.type === 'image') || e.target.files.length > 1)) {
      const shouldReset = confirm(
        '⚠️ Single Image Model Warning\n\n' +
        'This model only supports one image at a time. Adding images will replace existing images.\n\n' +
        'Do you want to continue?'
      );
      
      if (!shouldReset) {
        e.target.value = '';
        return;
      }
      
      // Clear existing images for single-image models
      setStagedAttachments(prev => prev.filter(a => a.type !== 'image'));
    }

    // Stage the images instead of immediately attaching
    const newAttachments: StagedAttachment[] = Array.from(e.target.files).map(file => ({
      id: `image-${Date.now()}-${Math.random()}`,
      file,
      type: 'image' as const,
      name: file.name,
      size: file.size
    }));

    setStagedAttachments(prev => [...prev, ...newAttachments]);
    e.target.value = '';
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;

    const validation = validateFiles(e.target.files, SUPPORTED_FILE_TYPES, false);
    
    if (!validation.success) {
      alert(`❌ File Attachment Error:\n\n${validation.error}`);
      e.target.value = '';
      return;
    }

    // Warn about large files that might exceed token limits
    const largeFiles = Array.from(e.target.files).filter(f => f.size > 1024 * 1024); // > 1MB
    if (largeFiles.length > 0) {
      const fileNames = largeFiles.map(f => `${f.name} (${(f.size / (1024 * 1024)).toFixed(1)}MB)`).join('\n');
      const shouldContinue = confirm(
        '⚠️ Large File Warning\n\n' +
        'The following files are large and may exceed token limits:\n\n' +
        fileNames + '\n\n' +
        'Large files will be truncated or summarized. Continue?'
      );
      
      if (!shouldContinue) {
        e.target.value = '';
        return;
      }
    }

    // Stage the documents instead of immediately attaching
    const newAttachments: StagedAttachment[] = Array.from(e.target.files).map(file => ({
      id: `document-${Date.now()}-${Math.random()}`,
      file,
      type: 'document' as const,
      name: file.name,
      size: file.size
    }));

    setStagedAttachments(prev => [...prev, ...newAttachments]);
    e.target.value = '';
  };

  const handleFetch = async () => {
    if (!fetchUrl.trim()) {
      alert('Please enter a valid URL');
      return;
    }

    // Basic URL validation
    try {
      const url = new URL(fetchUrl.trim());
      if (!['http:', 'https:'].includes(url.protocol)) {
        throw new Error('Only HTTP and HTTPS URLs are supported');
      }
    } catch (error) {
      alert('❌ Invalid URL. Please enter a valid HTTP or HTTPS URL.');
      return;
    }

    const shouldContinue = confirm(
      '🌐 Fetch Website Content\n\n' +
      `URL: ${fetchUrl}\n\n` +
      '⚠️ Note: Only text content will be retrieved. Images, videos, and interactive elements will be ignored.\n\n' +
      'Continue?'
    );

    if (!shouldContinue) return;

    setIsFetching(true);
    try {
      // Call the backend fetch command
      const response = await apiClient.executeCommand('fetch', [fetchUrl.trim()]);
      
      if (response.success) {
        // Stage the fetched content instead of immediately sending
        const fetchAttachment: StagedAttachment = {
          id: `fetch-${Date.now()}`,
          type: 'fetch',
          name: `Website: ${fetchUrl}`,
          fetchUrl: fetchUrl.trim(),
          fetchContent: response.data || 'Content retrieved successfully.'
        };

        setStagedAttachments(prev => [...prev, fetchAttachment]);
        setShowFetchDialog(false);
        setFetchUrl('');
      } else {
        alert(`❌ Failed to fetch content:\n${response.message || 'Unknown error occurred'}`);
      }
    } catch (error) {
      console.error('Fetch error:', error);
      alert(`❌ Error fetching content:\n${error instanceof Error ? error.message : 'Unknown error occurred'}`);
    } finally {
      setIsFetching(false);
    }
  };

  // Function to remove staged attachment
  const removeStagedAttachment = (id: string) => {
    setStagedAttachments(prev => prev.filter(a => a.id !== id));
  };

  const isCommand = isValidCommand(message.trim());

  return (
    <>
      <div className="border-t border-border bg-card p-4">
        <form onSubmit={handleSubmit} className="flex items-end gap-3">
          {/* Attachment buttons */}
          {onAttachFiles && (
            <div className="flex-shrink-0 flex gap-2">
              {/* Image attachment */}
              <>
                <input
                  ref={imageInputRef}
                  type="file"
                  multiple
                  className="hidden"
                  onChange={handleImageSelect}
                  accept={SUPPORTED_IMAGE_TYPES}
                />
                <button
                  type="button"
                  onClick={() => imageInputRef.current?.click()}
                  disabled={disabled || isLoading}
                  className="p-2 rounded-md border border-border hover:bg-accent disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  title="Attach images (jpg, png, gif, etc.)"
                >
                  <Image size={20} className="text-blue-600" />
                </button>
              </>

              {/* File attachment */}
              <>
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  className="hidden"
                  onChange={handleFileSelect}
                  accept={SUPPORTED_FILE_TYPES}
                />
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={disabled || isLoading}
                  className="p-2 rounded-md border border-border hover:bg-accent disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  title="Attach documents (pdf, txt, code files, etc.)"
                >
                  <Paperclip size={20} className="text-green-600" />
                </button>
              </>

              {/* Fetch button */}
              <button
                type="button"
                onClick={() => setShowFetchDialog(true)}
                disabled={disabled || isLoading || isFetching}
                className="p-2 rounded-md border border-border hover:bg-accent disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                title="Fetch content from a website"
              >
                {isFetching ? (
                  <Loader2 size={20} className="text-purple-600 animate-spin" />
                ) : (
                  <Globe size={20} className="text-purple-600" />
                )}
              </button>
            </div>
          )}

        {/* Message input */}
        <div className="flex-1 relative">
          {/* Staged attachments display */}
          {stagedAttachments.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-2">
              {stagedAttachments.map((attachment) => (
                <div
                  key={attachment.id}
                  className="flex items-center gap-2 px-3 py-2 bg-muted border border-border rounded-md text-sm"
                >
                  <div className="flex items-center gap-1">
                    {attachment.type === 'image' && <Image size={14} className="text-blue-600" />}
                    {attachment.type === 'document' && <Paperclip size={14} className="text-green-600" />}
                    {attachment.type === 'fetch' && <Globe size={14} className="text-purple-600" />}
                    <span className="font-medium truncate max-w-32">
                      {attachment.name}
                    </span>
                    {attachment.size && (
                      <span className="text-muted-foreground text-xs">
                        ({(attachment.size / 1024).toFixed(1)}KB)
                      </span>
                    )}
                  </div>
                  <button
                    onClick={() => removeStagedAttachment(attachment.id)}
                    className="text-muted-foreground hover:text-red-600 transition-colors"
                    title="Remove attachment"
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          )}

          <textarea
            ref={textareaRef}
            value={message}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder={stagedAttachments.length > 0 ? "Add a message (optional)..." : placeholder}
            disabled={disabled || isLoading}
            rows={1}
            className={`w-full resize-none border rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed transition-all text-input bg-input placeholder:text-muted-foreground ${
              isCommand 
                ? 'border-red-500 bg-red-900/20' 
                : 'border-border'
            }`}
            style={{ minHeight: '40px' }}
          />
          
          {/* Command indicator */}
          {isCommand && (
            <div className="absolute -top-6 left-0 text-xs text-red-400 font-medium">
              Command blocked
            </div>
          )}
        </div>

        {/* Send button */}
        <button
          type="submit"
          disabled={(!message.trim() && stagedAttachments.length === 0) || disabled || isLoading}
          className="flex-shrink-0 bg-primary hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed text-primary-foreground p-2 rounded-md transition-colors flex items-center justify-center min-w-[40px] min-h-[40px]"
        >
          {isLoading ? (
            <Loader2 size={20} className="animate-spin" />
          ) : (
            <Send size={20} />
          )}
        </button>
      </form>
      
        {/* Helper text */}
        <div className="mt-2 text-xs text-muted-foreground flex items-center justify-between">
          <span>
            Press Enter to send, Shift+Enter for new line
          </span>
          {isCommand && (
            <span className="text-red-400">
              Commands cannot be executed here - use UI controls instead
            </span>
          )}
        </div>
      </div>

      {/* Fetch Dialog */}
      {showFetchDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="w-full max-w-md bg-background border border-border rounded-lg shadow-2xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold flex items-center gap-2">
                <Globe size={20} className="text-purple-600" />
                Fetch Website Content
              </h3>
              <button
                onClick={() => {
                  setShowFetchDialog(false);
                  setFetchUrl('');
                }}
                className="p-1 rounded-md hover:bg-accent text-muted-foreground hover:text-foreground"
              >
                ✕
              </button>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">
                  Website URL
                </label>
                <input
                  type="url"
                  value={fetchUrl}
                  onChange={(e) => setFetchUrl(e.target.value)}
                  placeholder="https://example.com"
                  className="w-full px-3 py-2 border border-border bg-input text-foreground placeholder:text-muted-foreground rounded-md focus:ring-2 focus:ring-primary focus:border-primary"
                  disabled={isFetching}
                />
              </div>
              
              <div className="bg-muted/50 border border-border rounded-md p-3">
                <p className="text-sm text-muted-foreground">
                  ⚠️ <strong>Note:</strong> Only text content will be retrieved. Images, videos, and interactive elements will be ignored.
                </p>
              </div>
              
              <div className="flex gap-3 pt-2">
                <button
                  onClick={() => {
                    setShowFetchDialog(false);
                    setFetchUrl('');
                  }}
                  className="flex-1 px-4 py-2 text-sm border border-border rounded-md hover:bg-accent transition-colors"
                  disabled={isFetching}
                >
                  Cancel
                </button>
                <button
                  onClick={handleFetch}
                  disabled={!fetchUrl.trim() || isFetching}
                  className="flex-1 px-4 py-2 text-sm bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
                >
                  {isFetching ? (
                    <>
                      <Loader2 size={16} className="animate-spin" />
                      Fetching...
                    </>
                  ) : (
                    <>
                      <Globe size={16} />
                      Fetch Content
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}