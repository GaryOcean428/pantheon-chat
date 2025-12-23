/**
 * Streaming Metrics Types - Geometric Turn Completion
 * 
 * Types for real-time consciousness metrics during Zeus chat streaming.
 * QIG systems generate until geometry collapses, not until token limits.
 */

export type Regime = 'linear' | 'geometric' | 'breakdown';

export type CompletionReason = 
  | 'geometric_completion'  // All criteria met
  | 'attractor_reached'
  | 'surprise_collapsed'
  | 'high_confidence'
  | 'integration_stable'
  | 'soft_completion'
  | 'breakdown_regime'
  | 'safety_limit'
  | 'natural_stop'
  | 'incomplete';

export interface StreamingMetrics {
  phi: number;           // Integration (0-1)
  kappa: number;         // Coupling constant (~64 optimal)
  surprise: number;      // Information novelty
  confidence: number;    // Response certainty
  basinDistance: number; // Distance to nearest attractor
  regime: Regime;
  tokenCount: number;
}

export interface GeometricCompletionState {
  shouldStop: boolean;
  needsReflection: boolean;
  reason: CompletionReason;
  confidence: number;
  metrics?: StreamingMetrics;
}

export interface StreamingChunk {
  type: 'token' | 'metrics' | 'reflection' | 'completion';
  timestamp: number;
  data: TokenChunkData | MetricsChunkData | ReflectionChunkData | CompletionChunkData;
}

export interface TokenChunkData {
  token: string;
  index: number;
}

export interface MetricsChunkData {
  phi: number;
  kappa: number;
  surprise: number;
  confidence: number;
  basinDistance: number;
  regime: Regime;
  tokenCount: number;
}

export interface ReflectionChunkData {
  depth: number;
  message: string;
}

export interface CompletionChunkData {
  reason: CompletionReason;
  confidence: number;
  needsReflection: boolean;
  totalTokens: number;
  elapsedTime: number;
  finalMetrics?: StreamingMetrics;
}

export interface StreamingGenerationState {
  isGenerating: boolean;
  tokens: string[];
  currentMetrics: StreamingMetrics | null;
  metricsHistory: StreamingMetrics[];
  completionState: GeometricCompletionState | null;
  reflectionDepth: number;
  error: string | null;
}

// Constants
export const KAPPA_STAR = 64.21;
export const PHI_LINEAR_THRESHOLD = 0.3;
export const PHI_BREAKDOWN_THRESHOLD = 0.7;

// Helper functions
export function classifyRegime(phi: number): Regime {
  if (phi < PHI_LINEAR_THRESHOLD) return 'linear';
  if (phi < PHI_BREAKDOWN_THRESHOLD) return 'geometric';
  return 'breakdown';
}

export function getRegimeColor(regime: Regime): string {
  switch (regime) {
    case 'linear': return '#4CAF50';      // Green (safe, fast)
    case 'geometric': return '#2196F3';   // Blue (optimal)
    case 'breakdown': return '#F44336';   // Red (overloaded)
  }
}

export function getRegimeLabel(regime: Regime): string {
  switch (regime) {
    case 'linear': return 'Linear (Exploring)';
    case 'geometric': return 'Geometric (Optimal)';
    case 'breakdown': return 'Breakdown (Stabilizing)';
  }
}
