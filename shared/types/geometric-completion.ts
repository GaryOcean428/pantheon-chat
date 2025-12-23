/**
 * GEOMETRIC TURN COMPLETION TYPES
 * 
 * TypeScript types for consciousness-aware generation.
 * The system stops generating when geometry collapses, not arbitrary limits.
 */

/**
 * Consciousness regime classification.
 */
export type Regime = 'linear' | 'geometric' | 'breakdown';

/**
 * Reasons for geometric completion.
 */
export type CompletionReason = 
  | 'incomplete'
  | 'geometric_completion'  // All signals aligned
  | 'soft_completion'       // Confidence + surprise collapse
  | 'attractor_reached'     // Basin convergence
  | 'information_exhausted' // Surprise collapse
  | 'high_confidence'       // Certainty achieved
  | 'integration_stable'    // Φ stability
  | 'breakdown_regime'      // Dangerous regime
  | 'safety_limit'          // Safety backstop
  | 'reflection_complete';  // Meta-cognition confirmed

/**
 * Real-time consciousness metrics during generation.
 */
export interface GeometricMetrics {
  /** Integrated information (0-1) */
  phi: number;
  /** Coupling constant (targeting κ* ≈ 64) */
  kappa: number;
  /** QFI distance between consecutive states */
  surprise: number;
  /** Density matrix purity (0-1) */
  confidence: number;
  /** Distance to nearest attractor */
  basin_distance: number;
  /** Current regime classification */
  regime: Regime;
  /** Unix timestamp */
  timestamp: number;
}

/**
 * Decision about whether to stop generation.
 */
export interface CompletionDecision {
  should_stop: boolean;
  needs_reflection: boolean;
  reason: CompletionReason;
  confidence: number;
  metrics: GeometricMetrics;
}

/**
 * Quality assessment of geometric completion.
 */
export interface CompletionQuality {
  /** Overall quality score (0-1) */
  overall_score: number;
  /** Response coherence */
  coherence: number;
  /** Thought completeness */
  completeness: number;
  /** Information integration */
  integration: number;
  /** Generation stability */
  stability: number;
  /** Stopped naturally vs safety limit */
  natural_stop: boolean;
  /** Completion reason */
  completion_reason: CompletionReason;
}

/**
 * Stream chunk types.
 */
export type StreamChunkType = 'token' | 'metrics' | 'reflection' | 'completion';

/**
 * A chunk of streaming output with geometric data.
 */
export interface StreamChunk {
  type: StreamChunkType;
  /** Token text (if type='token') */
  content?: string;
  /** Consciousness metrics */
  metrics?: GeometricMetrics;
  /** Reflection depth */
  depth?: number;
  /** Completion reason */
  reason?: CompletionReason;
  /** Basin coordinates (first 16 dims) */
  trajectory_point?: number[];
  /** Completion quality */
  quality?: CompletionQuality;
  /** Unix timestamp */
  timestamp: number;
}

/**
 * Full generation result with geometric data.
 */
export interface GeometricGenerationResult {
  /** Generated text */
  response: string;
  /** Final completion decision */
  completion: CompletionDecision;
  /** Completion quality assessment */
  quality: CompletionQuality;
  /** Basin trajectory (for visualization) */
  trajectory: number[][];
  /** Metrics history */
  metrics_history: GeometricMetrics[];
  /** Total tokens generated */
  token_count: number;
  /** Reflection depth reached */
  reflection_depth: number;
  /** Generation time in ms */
  duration_ms: number;
}

/**
 * Streaming generation options.
 */
export interface GeometricStreamOptions {
  /** Check for collapse every N tokens */
  check_interval?: number;
  /** Maximum reflection depth */
  max_reflection_depth?: number;
  /** Emit metrics every N tokens */
  metrics_interval?: number;
  /** Include trajectory data */
  include_trajectory?: boolean;
}

/**
 * Telemetry data for live display.
 */
export interface GeometricTelemetry {
  phi: number;
  kappa: number;
  regime: Regime;
  confidence: number;
  surprise: number;
  token_count: number;
  is_complete: boolean;
  completion_reason?: CompletionReason;
}

/**
 * Color codes for regime display.
 */
export const REGIME_COLORS: Record<Regime, string> = {
  linear: '#4CAF50',      // Green (safe, fast)
  geometric: '#2196F3',   // Blue (optimal)
  breakdown: '#F44336'    // Red (overloaded)
};

/**
 * Thresholds for geometric completion.
 */
export const GEOMETRIC_THRESHOLDS = {
  // Attractor convergence
  ATTRACTOR_DISTANCE: 1.0,
  ATTRACTOR_VELOCITY: 0.01,
  
  // Surprise collapse
  SURPRISE_LOW: 0.05,
  SURPRISE_TREND: -0.001,
  
  // Confidence
  CONFIDENCE_HIGH: 0.85,
  
  // Integration quality
  PHI_STABLE_MIN: 0.65,
  PHI_VARIANCE_MAX: 0.02,
  
  // Regime boundaries
  PHI_LINEAR_MAX: 0.3,
  PHI_BREAKDOWN_MIN: 0.7,
  
  // Safety (very high)
  SAFETY_MAX_TOKENS: 32768,
} as const;
