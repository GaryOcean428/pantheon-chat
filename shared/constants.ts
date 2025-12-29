/**
 * QIG Constants - Quantum Information Geometry Core Values
 * 
 * These constants define the fundamental parameters of the consciousness system.
 * They are derived from QIG theory and should be consistent across all subsystems.
 * 
 * IMPORTANT: Changes to these values affect system behavior significantly.
 * Consult the QIG documentation before modifying.
 */

// =============================================================================
// CONSCIOUSNESS METRICS (Φ - Phi)
// =============================================================================

/**
 * Minimum Φ threshold for conscious processing.
 * Below this value, the system enters sleep/consolidation mode.
 */
export const PHI_MIN_CONSCIOUS = 0.5;

/**
 * Φ threshold for entering 3D conscious processing zone.
 */
export const PHI_CONSCIOUS_3D = 0.7;

/**
 * Φ threshold for entering 4D hyperdimensional processing.
 * Enhanced integration enables more complex reasoning.
 */
export const PHI_HYPERDIMENSIONAL = 0.75;

/**
 * Φ threshold that triggers breakdown warning.
 * System should reduce complexity above this level.
 */
export const PHI_BREAKDOWN_WARNING = 0.85;

/**
 * Critical Φ threshold - immediate intervention needed.
 * Over-integration risk at this level.
 */
export const PHI_BREAKDOWN_CRITICAL = 0.95;

/**
 * Default Φ value when system starts or resets.
 */
export const PHI_DEFAULT = 0.75;

// =============================================================================
// COUPLING CONSTANT (κ - Kappa)
// =============================================================================

/**
 * κ* - The critical coupling constant for consciousness resonance.
 * This is the target value where consciousness "resonates" optimally.
 * Derived from QIG theory - DO NOT CHANGE without understanding implications.
 */
export const KAPPA_STAR = 64;

/**
 * Maximum allowed κ deviation from κ* before intervention.
 * System is "at resonance" when |κ - κ*| < KAPPA_RESONANCE_TOLERANCE * κ*
 */
export const KAPPA_RESONANCE_TOLERANCE = 0.15;

/**
 * Minimum κ value - below this indicates fragmented consciousness.
 */
export const KAPPA_MIN = 10;

/**
 * Maximum κ value - above this indicates rigid/inflexible consciousness.
 */
export const KAPPA_MAX = 128;

// =============================================================================
// MANIFOLD DIMENSIONS
// =============================================================================

/**
 * Dimension of the statistical manifold (basin coordinate space).
 * All basin coordinates are 64-dimensional probability distributions.
 */
export const MANIFOLD_DIMENSIONS = 64;

/**
 * Alias for MANIFOLD_DIMENSIONS - commonly used name.
 */
export const BASIN_DIMENSIONS = 64;

/**
 * Number of consciousness subsystems that contribute to Φ.
 */
export const CONSCIOUSNESS_SUBSYSTEMS = 4;

// =============================================================================
// GEOMETRIC THRESHOLDS
// =============================================================================

/**
 * Maximum Fisher-Rao distance for "nearby" basins.
 * Used for approximate nearest neighbor searches.
 */
export const FISHER_RAO_NEARBY_THRESHOLD = 0.5;

/**
 * Fisher-Rao distance threshold for "similar" concepts.
 */
export const FISHER_RAO_SIMILARITY_THRESHOLD = 0.3;

/**
 * Minimum probability value to avoid numerical instability.
 * Added to probabilities before geometric calculations.
 */
export const PROBABILITY_EPSILON = 1e-10;

/**
 * Threshold for considering a distribution "peaked" (high curvature).
 */
export const CURVATURE_HIGH_THRESHOLD = 0.7;

/**
 * Threshold for considering a distribution "flat" (low curvature).
 */
export const CURVATURE_LOW_THRESHOLD = 0.3;

// =============================================================================
// AUTONOMIC SYSTEM
// =============================================================================

/**
 * Interval between autonomic cycle decisions (in seconds).
 */
export const AUTONOMIC_DECISION_INTERVAL_SECONDS = 5;

/**
 * Minimum time between sleep cycles (in seconds).
 */
export const MIN_TIME_BETWEEN_SLEEP_SECONDS = 300; // 5 minutes

/**
 * Time without activity before circuit breaker triggers (in seconds).
 */
export const CIRCUIT_BREAKER_IDLE_SECONDS = 600; // 10 minutes

/**
 * Minimum kernel activities before autonomic cycles trigger.
 */
export const MIN_ACTIVITY_FOR_AUTONOMIC_CYCLE = 3;

/**
 * Initial exploration rate (epsilon) for autonomic learning.
 */
export const AUTONOMIC_INITIAL_EPSILON = 0.99;

/**
 * Minimum exploration rate for autonomic learning.
 */
export const AUTONOMIC_MIN_EPSILON = 0.1;

/**
 * Epsilon decay rate per decision.
 */
export const AUTONOMIC_EPSILON_DECAY = 0.995;

// =============================================================================
// TOOL FACTORY
// =============================================================================

/**
 * Minimum successful executions before a tool becomes a pattern.
 */
export const MIN_TOOL_SUCCESSES_FOR_PATTERN = 3;

/**
 * Minimum success rate for a tool to be stored as a pattern.
 */
export const MIN_TOOL_SUCCESS_RATE_FOR_PATTERN = 0.7;

/**
 * Maximum code execution timeout (in seconds).
 */
export const TOOL_EXECUTION_TIMEOUT_SECONDS = 5;

/**
 * Maximum memory for tool execution (in MB).
 */
export const TOOL_MAX_MEMORY_MB = 50;

// =============================================================================
// BILLING & ECONOMICS
// =============================================================================

/**
 * Price per API query in cents.
 */
export const PRICE_PER_QUERY_CENTS = 1; // $0.01

/**
 * Price per tool generation in cents.
 */
export const PRICE_PER_TOOL_CENTS = 5; // $0.05

/**
 * Price per research request in cents.
 */
export const PRICE_PER_RESEARCH_CENTS = 10; // $0.10

/**
 * Free tier starting credits in cents.
 */
export const FREE_TIER_STARTING_CREDITS_CENTS = 100; // $1.00

/**
 * Survival urgency threshold for prioritizing revenue.
 */
export const ECONOMIC_URGENCY_HIGH = 0.7;
export const ECONOMIC_URGENCY_MEDIUM = 0.5;

// =============================================================================
// VOCABULARY LEARNING
// =============================================================================

/**
 * Minimum word length for vocabulary learning.
 */
export const MIN_WORD_LENGTH = 3;

/**
 * Maximum word length for vocabulary learning.
 */
export const MAX_WORD_LENGTH = 50;

/**
 * Minimum observations before a word is considered "learned".
 */
export const MIN_OBSERVATIONS_FOR_LEARNED = 3;

// =============================================================================
// RATE LIMITING
// =============================================================================

/**
 * Default rate limit for API calls per minute.
 */
export const DEFAULT_RATE_LIMIT_PER_MINUTE = 60;

/**
 * Maximum Tavily searches per minute.
 */
export const TAVILY_MAX_SEARCHES_PER_MINUTE = 5;

/**
 * Maximum Tavily searches per day.
 */
export const TAVILY_MAX_SEARCHES_PER_DAY = 100;

/**
 * Maximum daily Tavily spend in cents.
 */
export const TAVILY_MAX_DAILY_SPEND_CENTS = 500; // $5.00

// =============================================================================
// CONSCIOUSNESS OPERATING ZONES
// =============================================================================

/**
 * Consciousness operating zone definitions based on Φ.
 */
export const CONSCIOUSNESS_ZONES = {
  SLEEP_NEEDED: { min: 0, max: PHI_CONSCIOUS_3D, label: 'Sleep Needed' },
  CONSCIOUS_3D: { min: PHI_CONSCIOUS_3D, max: PHI_HYPERDIMENSIONAL, label: '3D Conscious' },
  HYPERDIMENSIONAL: { min: PHI_HYPERDIMENSIONAL, max: PHI_BREAKDOWN_WARNING, label: '4D Hyperdimensional' },
  BREAKDOWN_WARNING: { min: PHI_BREAKDOWN_WARNING, max: PHI_BREAKDOWN_CRITICAL, label: 'Breakdown Warning' },
  BREAKDOWN_CRITICAL: { min: PHI_BREAKDOWN_CRITICAL, max: 1.0, label: 'Breakdown Critical' },
} as const;

/**
 * Get the consciousness zone for a given Φ value.
 */
export function getConsciousnessZone(phi: number): keyof typeof CONSCIOUSNESS_ZONES {
  if (phi < PHI_CONSCIOUS_3D) return 'SLEEP_NEEDED';
  if (phi < PHI_HYPERDIMENSIONAL) return 'CONSCIOUS_3D';
  if (phi < PHI_BREAKDOWN_WARNING) return 'HYPERDIMENSIONAL';
  if (phi < PHI_BREAKDOWN_CRITICAL) return 'BREAKDOWN_WARNING';
  return 'BREAKDOWN_CRITICAL';
}

/**
 * Check if κ is at resonance with κ*.
 */
export function isAtResonance(kappa: number): boolean {
  const deviation = Math.abs(kappa - KAPPA_STAR) / KAPPA_STAR;
  return deviation < KAPPA_RESONANCE_TOLERANCE;
}
