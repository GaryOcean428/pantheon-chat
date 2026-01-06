import { useState, useCallback, useRef } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Button, Badge, Checkbox, Progress } from '@/components/ui';
import { Upload, FileText, CheckCircle2, XCircle, Loader2, X, BookOpen } from 'lucide-react';
import { apiRequest, queryClient } from '@/lib/queryClient';
import type { ChatUploadResult } from '@shared/schema';

interface ChatFileUploadProps {
  onContentReady?: (content: string, filename: string) => void;
  compact?: boolean;
}

export function ChatFileUpload({ onContentReady, compact = false }: ChatFileUploadProps) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [addToCurriculum, setAddToCurriculum] = useState(false);
  const [result, setResult] = useState<ChatUploadResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const uploadMutation = useMutation({
    mutationFn: async (file: File): Promise<ChatUploadResult> => {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('add_to_curriculum', addToCurriculum.toString());

      const response = await fetch('/api/uploads/chat', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Upload failed');
      }

      return response.json();
    },
    onSuccess: (data) => {
      setResult(data);
      if (data.success && data.rag_content && onContentReady && selectedFile) {
        onContentReady(data.rag_content, selectedFile.name);
      }
      setSelectedFile(null);
      if (data.curriculum_added) {
        queryClient.invalidateQueries({ queryKey: ['/api/learning/stats'] });
      }
    },
    onError: (error: Error) => {
      setResult({
        success: false,
        error: error.message,
      });
      setSelectedFile(null);
    },
  });

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const ACCEPTED_EXTENSIONS = ['.md', '.txt', '.markdown'];

  const isAcceptedFile = (file: File) => {
    const name = file.name.toLowerCase();
    return ACCEPTED_EXTENSIONS.some(ext => name.endsWith(ext));
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (isAcceptedFile(file)) {
        setSelectedFile(file);
        setResult(null);
      }
    }
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      if (isAcceptedFile(file)) {
        setSelectedFile(file);
        setResult(null);
      }
    }
  }, []);

  const handleUpload = () => {
    if (selectedFile) {
      uploadMutation.mutate(selectedFile);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    setResult(null);
  };

  if (compact) {
    return (
      <div className="space-y-2">
        <input
          ref={fileInputRef}
          type="file"
          accept=".md,.txt,.markdown"
          className="hidden"
          onChange={handleFileSelect}
          data-testid="input-chat-file"
        />
        
        {!selectedFile && !result && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => fileInputRef.current?.click()}
            data-testid="button-select-chat-file"
          >
            <Upload className="h-4 w-4 mr-1" />
            Upload for Discussion
          </Button>
        )}

        {selectedFile && (
          <div className="flex items-center gap-2 p-2 bg-muted/30 rounded-md">
            <FileText className="h-4 w-4 text-cyan-400 flex-shrink-0" />
            <span className="text-sm font-mono truncate flex-1">{selectedFile.name}</span>
            <div className="flex items-center gap-1">
              <Checkbox
                id="curriculum-toggle-compact"
                checked={addToCurriculum}
                onCheckedChange={(checked) => setAddToCurriculum(checked === true)}
                data-testid="checkbox-add-curriculum-compact"
              />
              <label htmlFor="curriculum-toggle-compact" className="text-xs text-muted-foreground">
                Learn
              </label>
            </div>
            <Button
              size="sm"
              onClick={handleUpload}
              disabled={uploadMutation.isPending}
              data-testid="button-upload-chat-file"
            >
              {uploadMutation.isPending ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                'Upload'
              )}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={clearFile}
              data-testid="button-clear-chat-file"
            >
              <X className="h-3 w-3" />
            </Button>
          </div>
        )}

        {result && (
          <div className={`flex items-center gap-2 text-sm ${result.success ? 'text-green-400' : 'text-red-400'}`}>
            {result.success ? (
              <>
                <CheckCircle2 className="h-4 w-4" />
                <span>Ready ({result.word_count} words)</span>
                {result.curriculum_added && (
                  <Badge variant="outline" className="text-xs">
                    <BookOpen className="h-3 w-3 mr-1" />
                    Added to curriculum
                  </Badge>
                )}
              </>
            ) : (
              <>
                <XCircle className="h-4 w-4" />
                <span>{result.error}</span>
              </>
            )}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div
        className={`border-2 border-dashed rounded-md p-4 text-center transition-colors ${
          dragActive
            ? 'border-cyan-400 bg-cyan-400/10'
            : 'border-muted-foreground/30 hover:border-muted-foreground/50'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        data-testid="dropzone-chat"
      >
        <Upload className="h-6 w-6 mx-auto mb-2 text-muted-foreground" />
        <p className="text-sm text-muted-foreground font-mono">
          Drop a file here for immediate discussion, or
        </p>
        <label className="cursor-pointer">
          <input
            type="file"
            accept=".md,.txt,.markdown"
            className="hidden"
            onChange={handleFileSelect}
            data-testid="input-chat-file-full"
          />
          <span className="text-sm text-cyan-400 hover:underline font-mono">
            click to browse
          </span>
        </label>
      </div>

      {selectedFile && (
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-2 p-2 bg-muted/30 rounded-md">
            <div className="flex items-center gap-2 min-w-0">
              <FileText className="h-4 w-4 text-cyan-400 flex-shrink-0" />
              <span className="text-sm font-mono truncate">{selectedFile.name}</span>
              <span className="text-xs text-muted-foreground flex-shrink-0">
                ({(selectedFile.size / 1024).toFixed(1)} KB)
              </span>
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 flex-shrink-0"
              onClick={clearFile}
              data-testid="button-remove-chat-file"
            >
              <X className="h-3 w-3" />
            </Button>
          </div>

          <div className="flex items-center gap-2">
            <Checkbox
              id="curriculum-toggle"
              checked={addToCurriculum}
              onCheckedChange={(checked) => setAddToCurriculum(checked === true)}
              data-testid="checkbox-add-curriculum"
            />
            <label htmlFor="curriculum-toggle" className="text-sm text-muted-foreground flex items-center gap-1">
              <BookOpen className="h-3 w-3" />
              Also add to curriculum for long-term learning
            </label>
          </div>

          <Button
            onClick={handleUpload}
            disabled={uploadMutation.isPending}
            className="w-full"
            data-testid="button-upload-chat"
          >
            {uploadMutation.isPending ? (
              <>
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Upload className="h-4 w-4 mr-1" />
                Upload & Discuss
              </>
            )}
          </Button>
        </div>
      )}

      {uploadMutation.isPending && (
        <div className="space-y-2">
          <Progress value={undefined} className="h-2" />
          <p className="text-xs text-muted-foreground font-mono text-center">
            Processing file for discussion...
          </p>
        </div>
      )}

      {result && (
        <div
          className={`p-3 rounded-md border ${
            result.success
              ? 'bg-green-500/10 border-green-500/30'
              : 'bg-red-500/10 border-red-500/30'
          }`}
          data-testid="result-chat-upload"
        >
          <div className="flex items-center gap-2">
            {result.success ? (
              <>
                <CheckCircle2 className="h-4 w-4 text-green-400" />
                <span className="font-mono text-sm">
                  Ready for discussion ({result.word_count} words)
                </span>
                {result.curriculum_added && (
                  <Badge variant="outline" className="text-xs ml-auto">
                    <BookOpen className="h-3 w-3 mr-1" />
                    Added to curriculum
                  </Badge>
                )}
              </>
            ) : (
              <>
                <XCircle className="h-4 w-4 text-red-400" />
                <span className="text-sm text-red-400 font-mono">{result.error}</span>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
