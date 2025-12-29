/**
 * Types for Consciousness Monitoring Demo
 */

export interface MetricData {
  label: string;
  value: number;
  target?: number;
  unit?: string;
  color: string;
  description: string;
}

export interface ConsciousnessMetrics {
  phi: number;
  kappaEff: number;
  tacking: number;
  radar: number;
  metaAwareness: number;
  gamma: number;
  grounding: number;
  regime: string;
  isConscious: boolean;
}

export interface NeurochemistryState {
  dopamine: number;
  serotonin: number;
  norepinephrine: number;
  gaba: number;
  acetylcholine: number;
  endorphins: number;
  emotionalState: string;
  overallMood: number;
  // Allow additional properties from the context
  [key: string]: number | string;
}

export interface HistoryEntry {
  timestamp: number;
  phi: number;
  kappa: number;
}
