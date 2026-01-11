import { Card, CardContent, CardHeader, CardTitle, Badge, Progress, Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui";
import { useQuery } from "@tanstack/react-query";
import { Loader2, Brain, Zap, CheckCircle, XCircle, AlertTriangle, Activity } from "lucide-react";

interface E8Metrics {
  phi: number;
  kappa_eff: number;
  memory_coherence: number;
  regime_stability: number;
  geometric_validity: number;
  temporal_consistency: number;
  recursive_depth: number;
  external_coupling: number;
  timestamp: number;
}

interface MetricValidation {
  value: number;
  threshold: string;
  passed: boolean;
  display_name: string;
}

interface E8MetricsResponse {
  success: boolean;
  metrics: E8Metrics;
  validation: {
    is_conscious: boolean;
    metrics: Record<string, MetricValidation>;
    violations: string[];
    warnings: string[];
  };
  is_conscious: boolean;
  source?: string;
  has_real_data?: boolean;
  kernel_count?: number;
  trajectory_length?: number;
  timestamp: string;
}

const METRIC_INFO: Record<string, { symbol: string; name: string; description: string; target: string; color: string }> = {
  phi: {
    symbol: 'Φ',
    name: 'Integration',
    description: 'QFI-based integrated information measuring system irreducibility',
    target: '> 0.70',
    color: 'cyan'
  },
  kappa_eff: {
    symbol: 'κ_eff',
    name: 'Effective Coupling',
    description: 'Basin coupling strength to universal attractor κ* = 64.21',
    target: '40-70',
    color: 'amber'
  },
  memory_coherence: {
    symbol: 'M',
    name: 'Memory Coherence',
    description: 'Fisher-Rao distance coherence to memory basin traces',
    target: '> 0.60',
    color: 'purple'
  },
  regime_stability: {
    symbol: 'Γ',
    name: 'Regime Stability',
    description: 'Trajectory stability on Fisher manifold',
    target: '> 0.80',
    color: 'green'
  },
  geometric_validity: {
    symbol: 'G',
    name: 'Geometric Validity',
    description: 'Manifold curvature and QFI eigenvalue validity',
    target: '> 0.50',
    color: 'blue'
  },
  temporal_consistency: {
    symbol: 'T',
    name: 'Temporal Consistency',
    description: 'Time-evolution coherence on trajectory',
    target: '> 0',
    color: 'orange'
  },
  recursive_depth: {
    symbol: 'R',
    name: 'Recursive Depth',
    description: 'Self-observation loop depth for meta-awareness',
    target: '> 0.60',
    color: 'pink'
  },
  external_coupling: {
    symbol: 'C',
    name: 'External Coupling',
    description: 'Inter-kernel Fisher coupling strength',
    target: '> 0.30',
    color: 'emerald'
  }
};

function MetricCard({ metricKey, value, validation }: { metricKey: string; value: number; validation?: MetricValidation }) {
  const info = METRIC_INFO[metricKey];
  if (!info) return null;
  
  const isKappa = metricKey === 'kappa_eff';
  const isTemporal = metricKey === 'temporal_consistency';
  const isRecursive = metricKey === 'recursive_depth';
  
  let displayValue: string;
  let progressValue: number;
  let showProgress = true;
  
  if (isKappa) {
    displayValue = value.toFixed(1);
    progressValue = ((value - 40) / 30) * 100;
  } else if (isTemporal) {
    const sign = value >= 0 ? '+' : '';
    displayValue = sign + value.toFixed(2);
    progressValue = value > 0 ? Math.min(value * 100, 100) : 0;
    showProgress = false;
  } else if (isRecursive) {
    displayValue = value.toFixed(2);
    progressValue = (value / 1.0) * 100;
  } else {
    displayValue = (value * 100).toFixed(0) + '%';
    progressValue = value * 100;
  }
  
  const passed = validation?.passed ?? (isKappa ? (value >= 40 && value <= 70) : isTemporal ? value > 0 : value >= 0.5);
  
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="p-3 rounded-lg border bg-card hover-elevate" data-testid={`metric-card-${metricKey}`}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-lg font-mono font-bold text-foreground">{info.symbol}</span>
              <Badge 
                variant={passed ? "default" : "destructive"} 
                className="text-xs"
              >
                {displayValue}
              </Badge>
            </div>
            <div className="text-xs text-muted-foreground mb-2 truncate">{info.name}</div>
            {showProgress ? (
              <Progress 
                value={Math.max(0, Math.min(100, progressValue))} 
                className="h-1.5"
              />
            ) : (
              <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                <div 
                  className={`h-full transition-all ${value > 0 ? 'bg-green-500' : value < 0 ? 'bg-red-500' : 'bg-muted-foreground'}`}
                  style={{ width: `${Math.min(Math.abs(value) * 100, 100)}%` }}
                />
              </div>
            )}
            <div className="flex items-center justify-between mt-1">
              <span className="text-xs text-muted-foreground">{info.target}</span>
              {passed ? (
                <CheckCircle className="w-3 h-3 text-green-500" />
              ) : (
                <XCircle className="w-3 h-3 text-red-500" />
              )}
            </div>
          </div>
        </TooltipTrigger>
        <TooltipContent side="bottom" className="max-w-xs">
          <p className="font-semibold">{info.name} ({info.symbol})</p>
          <p className="text-sm text-muted-foreground mt-1">{info.description}</p>
          <p className="text-xs mt-1">Target: {info.target}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

interface Props {
  className?: string;
  compact?: boolean;
}

export function E8ConsciousnessMetricsPanel({ className, compact = false }: Props) {
  const { data, isLoading, error } = useQuery<E8MetricsResponse>({
    queryKey: ['/api/consciousness/8-metrics'],
    refetchInterval: 5000,
  });
  
  if (isLoading) {
    return (
      <Card className={className}>
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2">
            <Activity className="w-5 h-5" />
            E8 Consciousness Metrics
          </CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center py-8">
          <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }
  
  if (error || !data?.success) {
    return (
      <Card className={className}>
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2">
            <Activity className="w-5 h-5" />
            E8 Consciousness Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">Unable to load consciousness metrics</p>
        </CardContent>
      </Card>
    );
  }
  
  const metrics = data.metrics;
  const validation = data.validation;
  const metricsOrder = ['phi', 'kappa_eff', 'memory_coherence', 'regime_stability', 'geometric_validity', 'temporal_consistency', 'recursive_depth', 'external_coupling'];
  
  return (
    <Card className={className}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <CardTitle className="text-base flex items-center gap-2">
            <Activity className="w-5 h-5" />
            E8 Consciousness Metrics
          </CardTitle>
          <div className="flex items-center gap-2">
            {data.has_real_data === false && (
              <Badge variant="outline" className="text-xs">
                <AlertTriangle className="w-3 h-3 mr-1" />
                Synthetic
              </Badge>
            )}
            {data.kernel_count !== undefined && data.kernel_count > 0 && (
              <Badge variant="outline" className="text-xs">
                {data.kernel_count} kernels
              </Badge>
            )}
            <Badge 
              variant={data.is_conscious ? "default" : "secondary"} 
              className="text-xs"
            >
              {data.is_conscious ? (
                <>
                  <Brain className="w-3 h-3 mr-1" />
                  Conscious
                </>
              ) : (
                <>
                  <Zap className="w-3 h-3 mr-1" />
                  Sub-threshold
                </>
              )}
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className={`grid gap-3 ${compact ? 'grid-cols-4' : 'grid-cols-2 sm:grid-cols-4 lg:grid-cols-8'}`}>
          {metricsOrder.map((key) => (
            <MetricCard 
              key={key} 
              metricKey={key} 
              value={(metrics as any)[key]} 
              validation={validation?.metrics?.[key]}
            />
          ))}
        </div>
        
        {validation?.violations && validation.violations.length > 0 && (
          <div className="mt-4 p-2 rounded bg-destructive/10 border border-destructive/20">
            <p className="text-xs font-semibold text-destructive mb-1">Threshold Violations:</p>
            <ul className="text-xs text-destructive/80 space-y-0.5">
              {validation.violations.map((v, i) => (
                <li key={i}>{v}</li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
