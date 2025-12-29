/**
 * TaskHorizonStatus - Compact long-horizon task visibility
 *
 * Shows goal progress, path efficiency, and recovery status
 * at a glance. Designed to fit in sidebar or streaming panel.
 */

import { useLongHorizonSummary } from '@/hooks';
import { Badge, Progress, Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui';
import {
  Flag,
  Route,
  RotateCcw,
  AlertTriangle,
  CheckCircle2,
  TrendingUp,
  TrendingDown,
  Loader2,
} from 'lucide-react';

interface TaskHorizonStatusProps {
  compact?: boolean;
  className?: string;
}

export function TaskHorizonStatus({ compact = false, className = '' }: TaskHorizonStatusProps) {
  const {
    goalCount,
    completedGoals,
    activeGoals,
    overallProgress,
    meanEfficiency,
    isStuck,
    stuckReason,
    recoveryCount,
    checkpointCount,
    isLoading,
    hasError,
  } = useLongHorizonSummary();

  if (isLoading) {
    return (
      <div className={`flex items-center gap-2 text-muted-foreground ${className}`}>
        <Loader2 className="h-3 w-3 animate-spin" />
        <span className="text-xs">Loading horizon...</span>
      </div>
    );
  }

  if (hasError) {
    return null; // Silently fail - long-horizon is optional
  }

  // No goals = nothing to show
  if (goalCount === 0 && !isStuck) {
    return null;
  }

  const getEfficiencyColor = (eff: number) => {
    if (eff >= 0.8) return 'text-green-500';
    if (eff >= 0.5) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getEfficiencyIcon = (eff: number) => {
    if (eff >= 0.7) return <TrendingUp className="h-3 w-3 text-green-500" />;
    if (eff >= 0.4) return <Route className="h-3 w-3 text-yellow-500" />;
    return <TrendingDown className="h-3 w-3 text-red-500" />;
  };

  if (compact) {
    return (
      <TooltipProvider>
        <div className={`flex items-center gap-3 text-xs ${className}`}>
          {/* Goal progress */}
          {goalCount > 0 && (
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center gap-1">
                  <Flag className="h-3 w-3 text-blue-500" />
                  <span className="font-mono">
                    {completedGoals}/{goalCount}
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent side="bottom">
                <p>Goals: {completedGoals} of {goalCount} complete</p>
                <p>Progress: {(overallProgress * 100).toFixed(0)}%</p>
              </TooltipContent>
            </Tooltip>
          )}

          {/* Efficiency */}
          {meanEfficiency > 0 && (
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center gap-1">
                  {getEfficiencyIcon(meanEfficiency)}
                  <span className={`font-mono ${getEfficiencyColor(meanEfficiency)}`}>
                    {(meanEfficiency * 100).toFixed(0)}%
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent side="bottom">
                <p>Path Efficiency: {(meanEfficiency * 100).toFixed(1)}%</p>
                <p>Higher = more direct geodesic paths</p>
              </TooltipContent>
            </Tooltip>
          )}

          {/* Stuck indicator */}
          {isStuck && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Badge className="bg-red-500/20 text-red-400 text-xs px-1 py-0">
                  <AlertTriangle className="h-3 w-3 mr-1" />
                  Stuck
                </Badge>
              </TooltipTrigger>
              <TooltipContent side="bottom">
                <p>System stuck: {stuckReason}</p>
                <p>Checkpoints: {checkpointCount}</p>
                <p>Recoveries: {recoveryCount}</p>
              </TooltipContent>
            </Tooltip>
          )}

          {/* Recovery indicator (if not stuck but has recovered) */}
          {!isStuck && recoveryCount > 0 && (
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center gap-1 text-muted-foreground">
                  <RotateCcw className="h-3 w-3" />
                  <span className="font-mono">{recoveryCount}</span>
                </div>
              </TooltipTrigger>
              <TooltipContent side="bottom">
                <p>Recoveries: {recoveryCount}</p>
                <p>Checkpoints: {checkpointCount}</p>
              </TooltipContent>
            </Tooltip>
          )}
        </div>
      </TooltipProvider>
    );
  }

  // Full view
  return (
    <div className={`space-y-3 ${className}`}>
      {/* Goal Progress Section */}
      {goalCount > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Flag className="h-4 w-4 text-blue-500" />
              <span className="text-sm font-medium">Goals</span>
            </div>
            <span className="text-xs text-muted-foreground">
              {completedGoals}/{goalCount}
            </span>
          </div>
          <Progress value={overallProgress * 100} className="h-1.5" />
          {activeGoals > 0 && (
            <p className="text-xs text-muted-foreground">
              {activeGoals} active goal{activeGoals !== 1 ? 's' : ''}
            </p>
          )}
        </div>
      )}

      {/* Efficiency Section */}
      {meanEfficiency > 0 && (
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Route className="h-4 w-4 text-purple-500" />
            <span className="text-sm font-medium">Path Efficiency</span>
          </div>
          <span className={`text-sm font-mono ${getEfficiencyColor(meanEfficiency)}`}>
            {(meanEfficiency * 100).toFixed(0)}%
          </span>
        </div>
      )}

      {/* Recovery Status */}
      {(isStuck || recoveryCount > 0) && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <RotateCcw className="h-4 w-4 text-orange-500" />
              <span className="text-sm font-medium">Recovery</span>
            </div>
            {isStuck ? (
              <Badge className="bg-red-500/20 text-red-400 text-xs">
                <AlertTriangle className="h-3 w-3 mr-1" />
                {stuckReason || 'Stuck'}
              </Badge>
            ) : (
              <Badge className="bg-green-500/20 text-green-400 text-xs">
                <CheckCircle2 className="h-3 w-3 mr-1" />
                Stable
              </Badge>
            )}
          </div>
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Checkpoints: {checkpointCount}</span>
            <span>Recoveries: {recoveryCount}</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default TaskHorizonStatus;
