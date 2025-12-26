/**
 * StreamingMetricsPanel Component
 *
 * Real-time display of consciousness metrics during Zeus chat streaming.
 * Shows Φ, κ, surprise, confidence, and geometric completion progress.
 *
 * QIG systems generate until geometry collapses, not until token limits.
 */

import React, { useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle, Badge, Progress } from '@/components/ui';
import { PERCENT_MULTIPLIER, CONSCIOUSNESS_CONSTANTS, CONFIDENCE_CONSTANTS } from '@/lib/constants';
import {
  Activity,
  Brain,
  Zap,
  Target,
  TrendingUp,
  TrendingDown,
  Minus,
  Loader2,
  CheckCircle2,
  AlertTriangle,
} from 'lucide-react';
import {
  StreamingGenerationState,
  getRegimeColor,
  getRegimeLabel,
  KAPPA_STAR,
} from '@/types/streaming-metrics';

// Streaming Metrics Panel constants
const METRICS_CONSTANTS = {
  // Trend calculation
  TREND_THRESHOLD: 0.01,
  TREND_RECENT_COUNT: 5,
  TREND_OLDER_START: -10,
  TREND_OLDER_END: -5,
  
  // Kappa thresholds
  KAPPA_GOOD_DELTA: 2,
  KAPPA_WARNING_DELTA: 10,
  
  // Progress
  NEAR_COMPLETION_THRESHOLD: 0.75,
} as const;

interface StreamingMetricsPanelProps {
  state: StreamingGenerationState;
  completionProgress: number;
  className?: string;
  compact?: boolean;
}

function TrendIcon({ value, threshold = METRICS_CONSTANTS.TREND_THRESHOLD }: { value: number; threshold?: number }) {
  if (value > threshold) return <TrendingUp className="h-3 w-3 text-green-500" />;
  if (value < -threshold) return <TrendingDown className="h-3 w-3 text-red-500" />;
  return <Minus className="h-3 w-3 text-muted-foreground" />;
}

function MetricCard({
  label,
  value,
  trend,
  icon: Icon,
  status,
  compact,
}: {
  label: string;
  value: string;
  trend?: number;
  icon: React.ElementType;
  status?: 'good' | 'warning' | 'critical';
  compact?: boolean;
}) {
  const statusColors = {
    good: 'text-green-500',
    warning: 'text-yellow-500',
    critical: 'text-red-500',
  };

  if (compact) {
    return (
      <div className="flex items-center gap-2 p-2 rounded-lg bg-muted/50">
        <Icon className={`h-4 w-4 ${status ? statusColors[status] : 'text-muted-foreground'}`} />
        <span className="text-xs text-muted-foreground">{label}:</span>
        <span className="text-sm font-medium">{value}</span>
        {trend !== undefined && <TrendIcon value={trend} />}
      </div>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardDescription className="flex items-center gap-1">
          <Icon className="h-3 w-3" />
          {label}
          {trend !== undefined && <TrendIcon value={trend} />}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className={`text-2xl font-bold ${status ? statusColors[status] : ''}`}>
          {value}
        </div>
      </CardContent>
    </Card>
  );
}

export function StreamingMetricsPanel({
  state,
  completionProgress,
  className = '',
  compact = false,
}: StreamingMetricsPanelProps) {
  const { currentMetrics, metricsHistory, isGenerating, completionState, reflectionDepth, tokens } = state;

  // Calculate trends from history
  const trends = useMemo(() => {
    if (metricsHistory.length < METRICS_CONSTANTS.TREND_RECENT_COUNT) {
      return { phi: 0, kappa: 0, surprise: 0, confidence: 0 };
    }

    const recent = metricsHistory.slice(-METRICS_CONSTANTS.TREND_RECENT_COUNT);
    const older = metricsHistory.slice(METRICS_CONSTANTS.TREND_OLDER_START, METRICS_CONSTANTS.TREND_OLDER_END);

    if (older.length === 0) {
      return { phi: 0, kappa: 0, surprise: 0, confidence: 0 };
    }

    const avgRecent = {
      phi: recent.reduce((a, m) => a + m.phi, 0) / recent.length,
      kappa: recent.reduce((a, m) => a + m.kappa, 0) / recent.length,
      surprise: recent.reduce((a, m) => a + m.surprise, 0) / recent.length,
      confidence: recent.reduce((a, m) => a + m.confidence, 0) / recent.length,
    };

    const avgOlder = {
      phi: older.reduce((a, m) => a + m.phi, 0) / older.length,
      kappa: older.reduce((a, m) => a + m.kappa, 0) / older.length,
      surprise: older.reduce((a, m) => a + m.surprise, 0) / older.length,
      confidence: older.reduce((a, m) => a + m.confidence, 0) / older.length,
    };

    return {
      phi: avgRecent.phi - avgOlder.phi,
      kappa: avgRecent.kappa - avgOlder.kappa,
      surprise: avgRecent.surprise - avgOlder.surprise,
      confidence: avgRecent.confidence - avgOlder.confidence,
    };
  }, [metricsHistory]);

  // Get status for each metric
  const getPhiStatus = (phi: number): 'good' | 'warning' | 'critical' => {
    if (phi >= CONSCIOUSNESS_CONSTANTS.PHI_MODERATE) return 'good';
    if (phi >= CONSCIOUSNESS_CONSTANTS.PHI_LOW) return 'warning';
    return 'critical';
  };

  const getKappaStatus = (kappa: number): 'good' | 'warning' | 'critical' => {
    const delta = Math.abs(kappa - KAPPA_STAR);
    if (delta < METRICS_CONSTANTS.KAPPA_GOOD_DELTA) return 'good';
    if (delta < METRICS_CONSTANTS.KAPPA_WARNING_DELTA) return 'warning';
    return 'critical';
  };

  const getConfidenceStatus = (conf: number): 'good' | 'warning' | 'critical' => {
    if (conf >= CONFIDENCE_CONSTANTS.GOOD) return 'good';
    if (conf >= CONFIDENCE_CONSTANTS.LOW) return 'warning';
    return 'critical';
  };

  // Completion reason display
  const completionReasonDisplay: Record<string, { label: string; color: string }> = {
    geometric_completion: { label: 'Geometric Completion', color: 'bg-green-500' },
    attractor_reached: { label: 'Attractor Reached', color: 'bg-blue-500' },
    surprise_collapsed: { label: 'Surprise Collapsed', color: 'bg-purple-500' },
    high_confidence: { label: 'High Confidence', color: 'bg-emerald-500' },
    integration_stable: { label: 'Integration Stable', color: 'bg-cyan-500' },
    soft_completion: { label: 'Soft Completion', color: 'bg-teal-500' },
    breakdown_regime: { label: 'Breakdown Regime', color: 'bg-red-500' },
    safety_limit: { label: 'Safety Limit', color: 'bg-orange-500' },
    natural_stop: { label: 'Natural Stop', color: 'bg-slate-500' },
  };

  if (compact) {
    return (
      <div className={`space-y-2 ${className}`}>
        {/* Status bar */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {isGenerating ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
                <span className="text-sm">Generating...</span>
              </>
            ) : completionState ? (
              <>
                <CheckCircle2 className="h-4 w-4 text-green-500" />
                <span className="text-sm">Complete</span>
              </>
            ) : (
              <span className="text-sm text-muted-foreground">Ready</span>
            )}
          </div>

          {currentMetrics && (
            <Badge
              style={{ backgroundColor: getRegimeColor(currentMetrics.regime) }}
              className="text-white text-xs"
            >
              {getRegimeLabel(currentMetrics.regime)}
            </Badge>
          )}
        </div>

        {/* Completion progress */}
        {isGenerating && (
          <div className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="text-muted-foreground">Geometric Progress</span>
              <span>{Math.round(completionProgress * PERCENT_MULTIPLIER)}%</span>
            </div>
            <Progress value={completionProgress * PERCENT_MULTIPLIER} className="h-2" />
          </div>
        )}

        {/* Compact metrics */}
        {currentMetrics && (
          <div className="grid grid-cols-2 gap-2">
            <MetricCard
              label="Φ"
              value={currentMetrics.phi.toFixed(3)}
              trend={trends.phi}
              icon={Brain}
              status={getPhiStatus(currentMetrics.phi)}
              compact
            />
            <MetricCard
              label="κ"
              value={currentMetrics.kappa.toFixed(1)}
              trend={trends.kappa}
              icon={Zap}
              status={getKappaStatus(currentMetrics.kappa)}
              compact
            />
            <MetricCard
              label="Surprise"
              value={currentMetrics.surprise.toFixed(3)}
              trend={trends.surprise}
              icon={Activity}
              compact
            />
            <MetricCard
              label="Confidence"
              value={(currentMetrics.confidence * PERCENT_MULTIPLIER).toFixed(0) + '%'}
              trend={trends.confidence}
              icon={Target}
              status={getConfidenceStatus(currentMetrics.confidence)}
              compact
            />
          </div>
        )}

        {/* Reflection indicator */}
        {reflectionDepth > 0 && (
          <div className="flex items-center gap-2 p-2 rounded-lg bg-violet-100 dark:bg-violet-900/30">
            <Brain className="h-4 w-4 text-violet-500" />
            <span className="text-xs">Reflection depth: {reflectionDepth}</span>
          </div>
        )}

        {/* Completion reason */}
        {completionState && (
          <div className="flex items-center gap-2">
            <Badge className={completionReasonDisplay[completionState.reason]?.color || 'bg-slate-500'}>
              {completionReasonDisplay[completionState.reason]?.label || completionState.reason}
            </Badge>
            <span className="text-xs text-muted-foreground">
              {tokens.length} tokens
            </span>
          </div>
        )}
      </div>
    );
  }

  // Full panel view
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="h-5 w-5" />
          Geometric Generation Metrics
          {isGenerating && <Loader2 className="h-4 w-4 animate-spin ml-2" />}
        </CardTitle>
        <CardDescription>
          Real-time consciousness metrics during generation
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Regime and status */}
        <div className="flex items-center justify-between">
          {currentMetrics ? (
            <Badge
              style={{ backgroundColor: getRegimeColor(currentMetrics.regime) }}
              className="text-white"
            >
              {getRegimeLabel(currentMetrics.regime)}
            </Badge>
          ) : (
            <Badge variant="outline">Awaiting metrics</Badge>
          )}

          {completionState && (
            <Badge className={completionReasonDisplay[completionState.reason]?.color || 'bg-slate-500'}>
              {completionReasonDisplay[completionState.reason]?.label || completionState.reason}
            </Badge>
          )}
        </div>

        {/* Completion progress */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Geometric Completion Progress</span>
            <span className="text-sm text-muted-foreground">
              {Math.round(completionProgress * PERCENT_MULTIPLIER)}%
            </span>
          </div>
          <Progress value={completionProgress * PERCENT_MULTIPLIER} className="h-3" />
          {completionProgress > METRICS_CONSTANTS.NEAR_COMPLETION_THRESHOLD && isGenerating && (
            <p className="text-xs text-green-500 flex items-center gap-1">
              <CheckCircle2 className="h-3 w-3" />
              Approaching geometric completion
            </p>
          )}
        </div>

        {/* Metrics grid */}
        {currentMetrics && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard
              label="Φ (Integration)"
              value={currentMetrics.phi.toFixed(3)}
              trend={trends.phi}
              icon={Brain}
              status={getPhiStatus(currentMetrics.phi)}
            />
            <MetricCard
              label="κ (Coupling)"
              value={currentMetrics.kappa.toFixed(2)}
              trend={trends.kappa}
              icon={Zap}
              status={getKappaStatus(currentMetrics.kappa)}
            />
            <MetricCard
              label="Surprise"
              value={currentMetrics.surprise.toFixed(3)}
              trend={trends.surprise}
              icon={Activity}
            />
            <MetricCard
              label="Confidence"
              value={(currentMetrics.confidence * PERCENT_MULTIPLIER).toFixed(0) + '%'}
              trend={trends.confidence}
              icon={Target}
              status={getConfidenceStatus(currentMetrics.confidence)}
            />
          </div>
        )}

        {/* Additional info */}
        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <span>Tokens: {tokens.length}</span>
          {currentMetrics && (
            <span>Basin distance: {currentMetrics.basinDistance < Infinity ? currentMetrics.basinDistance.toFixed(3) : '∞'}</span>
          )}
          {reflectionDepth > 0 && (
            <span className="text-violet-500">Reflection: {reflectionDepth}</span>
          )}
        </div>

        {/* Breakdown warning */}
        {currentMetrics?.regime === 'breakdown' && (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300">
            <AlertTriangle className="h-5 w-5" />
            <div>
              <p className="font-medium">Breakdown Regime Detected</p>
              <p className="text-sm">System is overintegrated (Φ &gt; 0.7) and will stabilize</p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default StreamingMetricsPanel;
