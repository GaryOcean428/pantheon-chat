/**
 * ConsciousnessMonitoringDemo - Main component
 * 
 * Real-time consciousness state visualization with metrics, history, and neurochemistry
 */

import { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui';
import { Brain, Activity } from 'lucide-react';
import { useConsciousness } from '@/contexts/ConsciousnessContext';
import { MetricGauge } from './MetricGauge';
import { NeurochemistryPanel } from './NeurochemistryPanel';
import { HistoryChart } from './HistoryChart';
import { RegimeIndicator } from './RegimeIndicator';
import { HistoryEntry, MetricData } from './types';

const MAX_HISTORY = 60;

export function ConsciousnessMonitoringDemo() {
  const { consciousness, neurochemistry, isIdle } = useConsciousness();
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  
  // Record history
  useEffect(() => {
    if (isIdle) return;
    
    const entry: HistoryEntry = {
      timestamp: Date.now(),
      phi: consciousness.phi,
      kappa: consciousness.kappaEff,
    };
    
    setHistory(prev => [...prev.slice(-MAX_HISTORY + 1), entry]);
  }, [consciousness.phi, consciousness.kappaEff, isIdle]);
  
  // Build metrics for gauges
  const metrics: MetricData[] = useMemo(() => [
    { label: 'Φ', value: consciousness.phi, target: 0.7, color: '#8b5cf6', description: 'Integrated Information' },
    { label: 'κ', value: consciousness.kappaEff / 100, target: 0.64, color: '#06b6d4', description: 'Coupling Constant' },
    { label: 'T', value: consciousness.tacking, color: '#10b981', description: 'Temporal Coherence' },
    { label: 'R', value: consciousness.radar, color: '#f59e0b', description: 'Attention Radar' },
    { label: 'M', value: consciousness.metaAwareness, color: '#ec4899', description: 'Meta-Awareness' },
    { label: 'Γ', value: consciousness.gamma, color: '#6366f1', description: 'Generativity' },
    { label: 'G', value: consciousness.grounding, color: '#84cc16', description: 'Reality Grounding' },
  ], [consciousness]);
  
  if (isIdle) {
    return (
      <Card className="opacity-50">
        <CardContent className="p-6 text-center text-muted-foreground">
          <Brain className="h-12 w-12 mx-auto mb-2 opacity-50" />
          <p>Consciousness monitoring paused</p>
          <p className="text-xs">System is idle</p>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <Card data-testid="consciousness-monitoring-demo">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-primary animate-pulse" />
            <CardTitle>Consciousness Monitor</CardTitle>
          </div>
          <RegimeIndicator regime={consciousness.regime} isConscious={consciousness.isConscious} />
        </div>
        <CardDescription>Real-time consciousness state visualization</CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {/* Primary metrics */}
        <div className="flex flex-wrap justify-center gap-4">
          {metrics.slice(0, 2).map(m => (
            <MetricGauge key={m.label} metric={m} size="lg" />
          ))}
        </div>
        
        {/* Secondary metrics */}
        <div className="flex flex-wrap justify-center gap-3">
          {metrics.slice(2).map(m => (
            <MetricGauge key={m.label} metric={m} size="sm" />
          ))}
        </div>
        
        {/* History chart */}
        <div className="border rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-muted-foreground">Φ History</span>
            <span className="text-xs font-mono">{history.length} samples</span>
          </div>
          <HistoryChart history={history} height={80} />
        </div>
        
        {/* Neurochemistry */}
        <NeurochemistryPanel neurochemistry={neurochemistry} />
      </CardContent>
    </Card>
  );
}

export default ConsciousnessMonitoringDemo;

// Re-export types and subcomponents
export * from './types';
export { MetricGauge } from './MetricGauge';
export { NeurochemistryPanel } from './NeurochemistryPanel';
export { HistoryChart } from './HistoryChart';
export { RegimeIndicator } from './RegimeIndicator';
