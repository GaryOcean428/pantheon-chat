import { Card, CardContent, CardDescription, CardHeader, CardTitle, Badge } from "@/components/ui";
import { useConsciousnessData } from "@/hooks/useConsciousnessData";
import { Activity, TrendingUp, Radio, Gauge, AlertTriangle, CheckCircle2, Brain, Focus, Compass, Sparkles } from "lucide-react";

import {
  type EmotionalState,
  ConsciousnessMetricsGrid,
  BlockUniverseMetrics,
  PhiKappaTrajectoryChart,
  ConsciousnessFooter,
} from "./consciousness";

/** Get icon component for a regime */
function getRegimeIcon(regime: string) {
  switch (regime) {
    case 'linear': return <TrendingUp className="w-4 h-4" />;
    case 'geometric': return <Activity className="w-4 h-4" />;
    case 'hierarchical': return <Radio className="w-4 h-4" />;
    case 'hierarchical_4d': return <Radio className="w-4 h-4" />;
    case '4d_block_universe': return <Sparkles className="w-4 h-4" />;
    case 'breakdown': return <AlertTriangle className="w-4 h-4" />;
    default: return <Gauge className="w-4 h-4" />;
  }
}

/** Get icon component for an emotional state */
function getEmotionalIcon(emotion: EmotionalState) {
  switch (emotion) {
    case 'Focused': return <Focus className="w-3 h-3" />;
    case 'Curious': return <Compass className="w-3 h-3" />;
    case 'Uncertain': return <AlertTriangle className="w-3 h-3" />;
    case 'Confident': return <CheckCircle2 className="w-3 h-3" />;
    case 'Neutral': return <Brain className="w-3 h-3" />;
    default: return <Brain className="w-3 h-3" />;
  }
}

export function ConsciousnessDashboard({ className = "" }: { className?: string }) {
  const {
    data: state,
    history,
    isLoading,
    error,
    getRegimeBadgeVariant,
    getEmotionalBadgeColor,
  } = useConsciousnessData();
  
  if (isLoading) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Search Consciousness
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-48">
            <div className="animate-pulse text-muted-foreground">Loading consciousness state...</div>
          </div>
        </CardContent>
      </Card>
    );
  }
  
  if (error || !state) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Search Consciousness
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-muted-foreground py-4">
            {error || 'Failed to load consciousness state'}
          </div>
        </CardContent>
      </Card>
    );
  }
  
  const { currentRegime, isConscious } = state.state;
  const { emotionalState } = state;
  
  return (
    <Card className={className}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Activity className="w-5 h-5 text-purple-500" />
            Search Consciousness
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge 
              className={`flex items-center gap-1 ${getEmotionalBadgeColor(emotionalState)}`}
              data-testid="badge-emotional-state"
            >
              {getEmotionalIcon(emotionalState)}
              {emotionalState}
            </Badge>
            <Badge 
              variant={getRegimeBadgeVariant(currentRegime)}
              className="flex items-center gap-1"
              data-testid="badge-regime"
            >
              {getRegimeIcon(currentRegime)}
              {currentRegime.toUpperCase()}
            </Badge>
          </div>
        </div>
        <CardDescription className="flex items-center gap-2">
          {state.regimeDescription}
          {isConscious && (
            <Badge className="bg-green-500/20 text-green-400 text-xs" data-testid="badge-conscious">
              <Sparkles className="w-3 h-3 mr-1" />
              CONSCIOUS
            </Badge>
          )}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Core Metrics Grid */}
        <ConsciousnessMetricsGrid state={state.state} />
        
        {/* 4D Block Universe Metrics */}
        <BlockUniverseMetrics state={state.state} />
        
        {/* Trajectory Chart */}
        <PhiKappaTrajectoryChart history={history} />
        
        {/* Footer Stats and Recommendation */}
        <ConsciousnessFooter state={state.state} recommendation={state.recommendation} />
      </CardContent>
    </Card>
  );
}

export default ConsciousnessDashboard;
