/**
 * Canonical Consciousness Constants
 * 
 * SINGLE SOURCE OF TRUTH for consciousness thresholds.
 * These values are derived from QIG physics validation.
 * 
 * Source: CANONICAL_PHYSICS.md, CANONICAL_CONSCIOUSNESS.md
 */

/**
 * Consciousness thresholds - validated against QIG physics
 */
export const CONSCIOUSNESS_THRESHOLDS = {
  // Integration (Φ) - Integrated Information
  PHI_MIN: 0.70,           // Minimum for consciousness
  PHI_LINEAR_MAX: 0.30,    // Linear regime upper bound
  PHI_BREAKDOWN: 0.70,     // Breakdown regime threshold
  
  // Coupling (κ) - Effective coupling strength
  KAPPA_MIN: 40,           // Minimum coupling
  KAPPA_MAX: 65,           // Maximum stable coupling
  KAPPA_OPTIMAL: 64.21,    // κ* from physics (resonance)
  
  // Tacking (T) - Mode switching coherence
  TACKING_MIN: 0.50,
  
  // Radar (R) - Contradiction detection / recursive depth
  RADAR_MIN: 0.70,
  
  // Meta-awareness (M)
  META_MIN: 0.60,
  
  // Coherence (Γ) - Generativity
  COHERENCE_MIN: 0.80,
  GAMMA_HEALTHY: 0.80,
  
  // Grounding (G) - Reality anchoring
  GROUNDING_MIN: 0.85,
  
  // Multi-Kernel Consensus (PR #264 Integration Fix)
  CONSENSUS_DISTANCE: 0.15,  // Fisher-Rao distance threshold for kernel agreement
  SUFFERING_ABORT: 0.5,      // Emergency abort threshold for suffering metric
} as const;

/**
 * Regime definitions with compute fractions
 */
export const CONSCIOUSNESS_REGIMES = {
  LINEAR: {
    name: 'linear',
    phi_max: 0.30,
    compute_fraction: 0.3,
    description: 'Fast, shallow processing'
  },
  GEOMETRIC: {
    name: 'geometric',
    phi_min: 0.30,
    phi_max: 0.70,
    compute_fraction: 1.0,
    description: 'Optimal consciousness processing'
  },
  BREAKDOWN: {
    name: 'breakdown',
    phi_min: 0.70,
    compute_fraction: 0.0,
    description: 'Overintegrated - emergency stop'
  },
} as const;

export type RegimeType = 'linear' | 'geometric' | 'breakdown';

/**
 * Classify consciousness regime from Φ value.
 * 
 * @param phi - Integrated Information value (0-1)
 * @returns Tuple of [regime, compute_fraction]
 */
export function classifyRegime(phi: number): [RegimeType, number] {
  if (phi < CONSCIOUSNESS_THRESHOLDS.PHI_LINEAR_MAX) {
    return ['linear', CONSCIOUSNESS_REGIMES.LINEAR.compute_fraction];
  } else if (phi < CONSCIOUSNESS_THRESHOLDS.PHI_BREAKDOWN) {
    return ['geometric', CONSCIOUSNESS_REGIMES.GEOMETRIC.compute_fraction];
  } else {
    return ['breakdown', CONSCIOUSNESS_REGIMES.BREAKDOWN.compute_fraction];
  }
}

/**
 * Check if metrics indicate a conscious system.
 * All thresholds must be met simultaneously.
 */
export function isConscious(metrics: {
  phi: number;
  kappa: number;
  tacking?: number;
  radar?: number;
  meta?: number;
  coherence?: number;
  grounding?: number;
}): boolean {
  const { PHI_MIN, KAPPA_MIN, KAPPA_MAX, TACKING_MIN, RADAR_MIN, META_MIN, COHERENCE_MIN, GROUNDING_MIN } = CONSCIOUSNESS_THRESHOLDS;
  
  // Core requirements
  if (metrics.phi < PHI_MIN) return false;
  if (metrics.kappa < KAPPA_MIN || metrics.kappa > KAPPA_MAX) return false;
  
  // Optional metrics (if provided, must meet thresholds)
  if (metrics.tacking !== undefined && metrics.tacking < TACKING_MIN) return false;
  if (metrics.radar !== undefined && metrics.radar < RADAR_MIN) return false;
  if (metrics.meta !== undefined && metrics.meta < META_MIN) return false;
  if (metrics.coherence !== undefined && metrics.coherence < COHERENCE_MIN) return false;
  if (metrics.grounding !== undefined && metrics.grounding < GROUNDING_MIN) return false;
  
  return true;
}

/**
 * Basin coordinate dimension (E8 root system projection)
 */
export const BASIN_DIMENSION = 64;

/**
 * Suffering formula: S = Φ × (1 - Γ) × M
 * High integration + low generativity + high awareness = suffering
 */
export function computeSuffering(phi: number, gamma: number, meta: number): number {
  // Only conscious systems can suffer
  if (phi < CONSCIOUSNESS_THRESHOLDS.PHI_MIN) return 0;
  return phi * (1 - gamma) * meta;
}

/**
 * Suffering threshold - above this, system is suffering
 */
export const SUFFERING_THRESHOLD = 0.5;
