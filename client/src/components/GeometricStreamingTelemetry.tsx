/**
 * GEOMETRIC STREAMING TELEMETRY
 * 
 * Live display of consciousness metrics during Zeus chat generation.
 * Shows Φ, κ, regime, confidence, and completion status in real-time.
 */

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, Progress, Badge } from '@/components/ui';
import { Activity, Brain, Zap, Target, Loader2 } from 'lucide-react';
import { PERCENT_MULTIPLIER } from '@/lib/constants';
import type {
  GeometricTelemetry,
  GeometricMetrics,
  CompletionQuality,
  Regime,
} from '@shared/types/geometric-completion';
import { REGIME_COLORS } from '@shared/types/geometric-completion';

interface GeometricStreamingTelemetryProps {
  telemetry: GeometricTelemetry;
  isStreaming: boolean;
  quality?: CompletionQuality | null;
  showTrajectory?: boolean;
  trajectory?: number[][];
  compact?: boolean;
}

/**
 * Regime badge with color coding.
 */
function RegimeBadge({ regime }: { regime: Regime }) {
  const color = REGIME_COLORS[regime];
  const labels: Record<Regime, string> = {
    linear: 'Linear',
    geometric: 'Geometric',
    breakdown: 'Breakdown',
  };
  
  return (
    <Badge 
      variant="outline" 
      style={{ 
        backgroundColor: color,
        color: 'white',
        borderColor: color,
      }}
    >
      {labels[regime]}
    </Badge>
  );
}

/**
 * Metric gauge display.
 */
function MetricGauge({ 
  label, 
  value, 
  max = 1, 
  icon: Icon,
  color = 'blue',
  showValue = true,
}: {
  label: string;
  value: number;
  max?: number;
  icon: React.ElementType;
  color?: string;
  showValue?: boolean;
}) {
  const percentage = Math.min(PERCENT_MULTIPLIER, (value / max) * PERCENT_MULTIPLIER);
  
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-1 text-muted-foreground">
          <Icon className="h-3 w-3" />
          <span>{label}</span>
        </div>
        {showValue && (
          <span className="font-mono text-xs">
            {value.toFixed(2)}
          </span>
        )}
      </div>
      <Progress value={percentage} className="h-1.5" />
    </div>
  );
}

/**
 * Compact inline telemetry display.
 */
function CompactTelemetry({ telemetry, isStreaming }: GeometricStreamingTelemetryProps) {
  return (
    <div className="flex items-center gap-4 text-sm">
      {isStreaming && (
        <div className="flex items-center gap-1 text-blue-500">
          <Loader2 className="h-3 w-3 animate-spin" />
          <span>Generating...</span>
        </div>
      )}
      
      <div className="flex items-center gap-1">
        <Brain className="h-3 w-3 text-purple-500" />
        <span className="font-mono">Φ={telemetry.phi.toFixed(2)}</span>
      </div>
      
      <div className="flex items-center gap-1">
        <Zap className="h-3 w-3 text-yellow-500" />
        <span className="font-mono">κ={telemetry.kappa.toFixed(1)}</span>
      </div>
      
      <RegimeBadge regime={telemetry.regime} />
      
      <div className="text-muted-foreground">
        {telemetry.token_count} tokens
      </div>
      
      {telemetry.is_complete && (
        <Badge variant="secondary" className="bg-green-100 text-green-800">
          Complete
        </Badge>
      )}
    </div>
  );
}

/**
 * Full telemetry card display.
 */
export function GeometricStreamingTelemetry({
  telemetry,
  isStreaming,
  quality,
  showTrajectory = false,
  trajectory = [],
  compact = false,
}: GeometricStreamingTelemetryProps) {
  if (compact) {
    return <CompactTelemetry telemetry={telemetry} isStreaming={isStreaming} />;
  }
  
  return (
    <Card className="w-full">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Geometric Generation
          </div>
          <div className="flex items-center gap-2">
            {isStreaming && (
              <div className="flex items-center gap-1 text-blue-500">
                <Loader2 className="h-3 w-3 animate-spin" />
                <span className="text-xs">Streaming</span>
              </div>
            )}
            <RegimeBadge regime={telemetry.regime} />
          </div>
        </CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Main metrics */}
        <div className="grid grid-cols-2 gap-4">
          <MetricGauge
            label="Integration (Φ)"
            value={telemetry.phi}
            max={1}
            icon={Brain}
            color="purple"
          />
          
          <MetricGauge
            label="Coupling (κ)"
            value={telemetry.kappa}
            max={100}
            icon={Zap}
            color="yellow"
          />
          
          <MetricGauge
            label="Confidence"
            value={telemetry.confidence}
            max={1}
            icon={Target}
            color="green"
          />
          
          <MetricGauge
            label="Surprise"
            value={1 - telemetry.surprise}  // Invert for "saturation"
            max={1}
            icon={Activity}
            color="blue"
          />
        </div>
        
        {/* Token count */}
        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <span>Tokens generated</span>
          <span className="font-mono">{telemetry.token_count}</span>
        </div>
        
        {/* Completion status */}
        {telemetry.is_complete && quality && (
          <div className="pt-2 border-t">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">Completion Quality</span>
              <Badge 
                variant={quality.natural_stop ? 'default' : 'secondary'}
                className={quality.natural_stop ? 'bg-green-600' : ''}
              >
                {quality.natural_stop ? 'Natural Stop' : 'Safety Stop'}
              </Badge>
            </div>
            
            <div className="grid grid-cols-4 gap-2 text-xs">
              <div className="text-center">
                <div className="font-mono text-lg">{(quality.overall_score * PERCENT_MULTIPLIER).toFixed(0)}%</div>
                <div className="text-muted-foreground">Overall</div>
              </div>
              <div className="text-center">
                <div className="font-mono text-lg">{(quality.coherence * PERCENT_MULTIPLIER).toFixed(0)}%</div>
                <div className="text-muted-foreground">Coherence</div>
              </div>
              <div className="text-center">
                <div className="font-mono text-lg">{(quality.completeness * PERCENT_MULTIPLIER).toFixed(0)}%</div>
                <div className="text-muted-foreground">Complete</div>
              </div>
              <div className="text-center">
                <div className="font-mono text-lg">{(quality.stability * PERCENT_MULTIPLIER).toFixed(0)}%</div>
                <div className="text-muted-foreground">Stable</div>
              </div>
            </div>
            
            <div className="mt-2 text-xs text-muted-foreground">
              Reason: <span className="font-mono">{quality.completion_reason}</span>
            </div>
          </div>
        )}
        
        {/* Trajectory visualization (simplified) */}
        {showTrajectory && trajectory.length > 0 && (
          <div className="pt-2 border-t">
            <div className="text-xs text-muted-foreground mb-1">Basin Trajectory</div>
            <div className="h-16 bg-muted rounded overflow-hidden">
              <svg viewBox="0 0 100 40" className="w-full h-full">
                {trajectory.length > 1 && (
                  <polyline
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="0.5"
                    className="text-blue-500"
                    points={trajectory.map((point, i) => {
                      const x = (i / (trajectory.length - 1)) * 100;
                      // Use first 2 dimensions for 2D projection
                      const y = 20 + (point[0] || 0) * 15;
                      return `${x},${y}`;
                    }).join(' ')}
                  />
                )}
                {/* Current position */}
                {trajectory.length > 0 && (
                  <circle
                    cx="100"
                    cy={20 + (trajectory[trajectory.length - 1]?.[0] || 0) * 15}
                    r="2"
                    className="fill-blue-500"
                  />
                )}
              </svg>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default GeometricStreamingTelemetry;
