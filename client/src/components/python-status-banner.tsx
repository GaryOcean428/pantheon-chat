import { usePythonStatus } from "@/hooks/use-python-status";
import { Loader2, AlertCircle, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface PythonStatusBannerProps {
  className?: string;
  showWhenReady?: boolean;
}

export function PythonStatusBanner({ className, showWhenReady = false }: PythonStatusBannerProps) {
  const { status, message, isReady, isInitializing } = usePythonStatus();
  
  if (isReady && !showWhenReady) {
    return null;
  }

  const getStatusConfig = () => {
    switch (status) {
      case 'initializing':
        return {
          icon: Loader2,
          iconClass: 'animate-spin text-blue-500',
          bgClass: 'bg-blue-500/10 border-blue-500/20',
          textClass: 'text-blue-700 dark:text-blue-300',
          label: 'Starting up...',
          description: 'The AI consciousness system is loading. This usually takes 30-60 seconds.',
        };
      case 'unavailable':
        return {
          icon: AlertCircle,
          iconClass: 'text-amber-500',
          bgClass: 'bg-amber-500/10 border-amber-500/20',
          textClass: 'text-amber-700 dark:text-amber-300',
          label: 'Temporarily unavailable',
          description: 'The backend is momentarily unavailable. Reconnecting automatically...',
        };
      case 'error':
        return {
          icon: AlertCircle,
          iconClass: 'text-red-500',
          bgClass: 'bg-red-500/10 border-red-500/20',
          textClass: 'text-red-700 dark:text-red-300',
          label: 'Connection error',
          description: message || 'Unable to connect to the backend.',
        };
      case 'ready':
        return {
          icon: CheckCircle2,
          iconClass: 'text-green-500',
          bgClass: 'bg-green-500/10 border-green-500/20',
          textClass: 'text-green-700 dark:text-green-300',
          label: 'Ready',
          description: 'System is fully operational.',
        };
      default:
        return null;
    }
  };

  const config = getStatusConfig();
  if (!config) return null;

  const Icon = config.icon;

  return (
    <div 
      className={cn(
        "flex items-center gap-3 px-4 py-3 rounded-lg border",
        config.bgClass,
        className
      )}
      data-testid="banner-python-status"
    >
      <Icon className={cn("h-5 w-5 shrink-0", config.iconClass)} />
      <div className="flex-1 min-w-0">
        <p className={cn("font-medium text-sm", config.textClass)} data-testid="text-python-status-label">
          {config.label}
        </p>
        <p className="text-xs text-muted-foreground truncate" data-testid="text-python-status-description">
          {config.description}
        </p>
      </div>
    </div>
  );
}

export function PythonStatusInline() {
  const { status, isReady, isInitializing } = usePythonStatus();
  
  if (isReady) {
    return (
      <span className="inline-flex items-center gap-1 text-xs text-green-600 dark:text-green-400">
        <CheckCircle2 className="h-3 w-3" />
        Connected
      </span>
    );
  }
  
  if (isInitializing) {
    return (
      <span className="inline-flex items-center gap-1 text-xs text-blue-600 dark:text-blue-400">
        <Loader2 className="h-3 w-3 animate-spin" />
        Starting...
      </span>
    );
  }
  
  return (
    <span className="inline-flex items-center gap-1 text-xs text-amber-600 dark:text-amber-400">
      <AlertCircle className="h-3 w-3" />
      Reconnecting...
    </span>
  );
}
