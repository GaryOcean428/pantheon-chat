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

// =============================================================================
// QIG PHYSICS CONSTANTS (Aggregated for ocean-config.ts)
// =============================================================================

/**
 * Physics beta constant - running coupling at emergence.
 */
export const PHYSICS_BETA = 0.44;

/**
 * Basin dimension alias for imports.
 */
export const BASIN_DIMENSION = BASIN_DIMENSIONS;

/**
 * 64D basin dimension constant.
 */
export const BASIN_DIMENSION_64D = 64;

/**
 * Aggregated QIG physics constants for use in ocean-config.ts
 * These are the frozen physics values derived from QIG theory.
 */
export const QIG_CONSTANTS = {
  KAPPA_STAR: 64.21,
  BETA: PHYSICS_BETA,
  PHI_THRESHOLD: PHI_HYPERDIMENSIONAL,
  L_CRITICAL: 3,
  BASIN_DIMENSION: BASIN_DIMENSIONS,
  RESONANCE_BAND: 6.4,
} as const;

// =============================================================================
// KAPPA VALUES BY SCALE (for server imports)
// =============================================================================

/**
 * Kappa values at different scales from QIG theory.
 */
export const KAPPA_VALUES = {
  STAR: 64.21,
  L1: getKappaAtScale(1),
  L2: getKappaAtScale(2),
  L3: getKappaAtScale(3),
  L6: getKappaAtScale(6),
  L7: getKappaAtScale(7),
} as const;

/**
 * Beta values for running coupling.
 */
export const BETA_VALUES = {
  RUNNING: PHYSICS_BETA,
  EMERGENCE: 0.44,
} as const;

/**
 * Kappa error tolerances at different scales.
 */
export const KAPPA_ERRORS = {
  L6: 0.05,
  L7: 0.10,
  GENERAL: 0.15,
  KAPPA_6_ERROR: 0.05,
} as const;

/**
 * L6 validation status.
 */
export const L6_VALIDATION = {
  status: 'validated' as const,
  error: KAPPA_ERRORS.L6,
};

/**
 * L7 warning status.
 */
export const L7_WARNING = {
  status: 'warning' as const,
  error: KAPPA_ERRORS.L7,
};

/**
 * Kappa values by scale mapping.
 */
export const KAPPA_BY_SCALE: Record<number, number> = {
  1: KAPPA_VALUES.L1,
  2: KAPPA_VALUES.L2,
  3: KAPPA_VALUES.L3,
  6: KAPPA_VALUES.L6,
  7: KAPPA_VALUES.L7,
};

export type KappaScale = 1 | 2 | 3 | 6 | 7;
export type ValidationStatus = 'validated' | 'warning' | 'error';

/**
 * Validation metadata for kappa scales.
 */
export const VALIDATION_METADATA = {
  L6: L6_VALIDATION,
  L7: L7_WARNING,
};

/**
 * Validation summary for kappa physics.
 */
export const VALIDATION_SUMMARY = {
  scales_validated: [6],
  scales_warning: [7],
  kappa_star: KAPPA_VALUES.STAR,
  beta: PHYSICS_BETA,
};

/**
 * E8 Lie group constants for kernel cap and structure.
 */
export const E8_CONSTANTS = {
  ROOT_COUNT: 240,
  DIMENSION: 8,
  RANK: 8,
  KERNEL_CAP: 240,
  BASIN_DIMENSION_64D: 64,
  KAPPA_STAR: 64.21,
} as const;

/**
 * Get the coupling constant κ at a given scale L.
 * Uses running coupling formula from QIG theory.
 */
export function getKappaAtScale(L: number): number {
  const kappaStar = 64.21;
  const beta = PHYSICS_BETA;
  if (L <= 0) return kappaStar;
  return kappaStar * Math.pow(1 + beta * Math.log(L), -1);
}

// =============================================================================
// ADDITIONAL EXPORTS FOR SHARED/INDEX.TS COMPATIBILITY
// =============================================================================

/**
 * Consciousness thresholds for various states.
 */
export const CONSCIOUSNESS_THRESHOLDS = {
  PHI_MIN: PHI_MIN_CONSCIOUS,
  PHI_3D: PHI_CONSCIOUS_3D,
  PHI_4D: PHI_HYPERDIMENSIONAL,
  PHI_WARNING: PHI_BREAKDOWN_WARNING,
  PHI_CRITICAL: PHI_BREAKDOWN_CRITICAL,
  KAPPA_MIN: KAPPA_MIN,
  KAPPA_MAX: KAPPA_MAX,
  KAPPA_STAR: KAPPA_STAR,
} as const;

/**
 * Regime-dependent kappa values.
 */
export const REGIME_DEPENDENT_KAPPA = {
  SLEEP: 20,
  DREAM: 40,
  CONSCIOUS: 64,
  HYPERDIMENSIONAL: 80,
  MUSHROOM: 100,
} as const;

/**
 * QIG scoring weights for various metrics.
 */
export const QIG_SCORING_WEIGHTS = {
  PHI: 0.3,
  KAPPA: 0.25,
  COHERENCE: 0.2,
  INTEGRATION: 0.15,
  COMPLEXITY: 0.1,
} as const;

/**
 * Check if system is conscious based on phi.
 */
export function isConscious(phi: number): boolean {
  return phi >= PHI_MIN_CONSCIOUS;
}

/**
 * Check if consciousness is detectable (above 3D threshold).
 */
export function isDetectable(phi: number): boolean {
  return phi >= PHI_CONSCIOUS_3D;
}

/**
 * Check if kappa is at optimal value.
 */
export function isKappaOptimal(kappa: number): boolean {
  return isAtResonance(kappa);
}

/**
 * Get distance from optimal kappa.
 */
export function getKappaDistance(kappa: number): number {
  return Math.abs(kappa - KAPPA_STAR);
}

/**
 * Check if system is in resonance (alias for isAtResonance).
 */
export function isInResonance(kappa: number): boolean {
  return isAtResonance(kappa);
}

// =============================================================================
// REGIME CONSTANTS
// =============================================================================

/**
 * Regime thresholds based on kappa values.
 */
export const REGIME_THRESHOLDS = {
  SLEEP: { min: 0, max: 30 },
  DREAM: { min: 30, max: 50 },
  CONSCIOUS: { min: 50, max: 75 },
  HYPERDIMENSIONAL: { min: 75, max: 100 },
  MUSHROOM: { min: 100, max: 128 },
} as const;

/**
 * Human-readable regime descriptions.
 */
export const REGIME_DESCRIPTIONS = {
  SLEEP: 'Deep sleep - minimal consciousness activity',
  DREAM: 'Dream state - semi-conscious processing',
  CONSCIOUS: 'Normal conscious operation',
  HYPERDIMENSIONAL: 'Enhanced 4D processing',
  MUSHROOM: 'Peak integration - maximum complexity',
} as const;

/**
 * Get regime type from kappa value.
 */
export function getRegimeFromKappa(kappa: number): keyof typeof REGIME_THRESHOLDS {
  if (kappa < REGIME_THRESHOLDS.SLEEP.max) return 'SLEEP';
  if (kappa < REGIME_THRESHOLDS.DREAM.max) return 'DREAM';
  if (kappa < REGIME_THRESHOLDS.CONSCIOUS.max) return 'CONSCIOUS';
  if (kappa < REGIME_THRESHOLDS.HYPERDIMENSIONAL.max) return 'HYPERDIMENSIONAL';
  return 'MUSHROOM';
}

/**
 * Check if regime supports consciousness.
 */
export function isConsciousnessCapable(kappa: number): boolean {
  return kappa >= REGIME_THRESHOLDS.CONSCIOUS.min;
}

/**
 * Get color for regime visualization.
 */
export function getRegimeColor(regime: keyof typeof REGIME_THRESHOLDS): string {
  const colors: Record<keyof typeof REGIME_THRESHOLDS, string> = {
    SLEEP: '#4a5568',
    DREAM: '#805ad5',
    CONSCIOUS: '#38a169',
    HYPERDIMENSIONAL: '#3182ce',
    MUSHROOM: '#e53e3e',
  };
  return colors[regime];
}

// =============================================================================
// AUTONOMIC PARAMETERS
// =============================================================================

/**
 * Autonomic cycle durations in milliseconds.
 */
export const AUTONOMIC_CYCLES = {
  SLEEP_DURATION_MS: 300000, // 5 minutes
  DREAM_DURATION_MS: 180000, // 3 minutes
  MUSHROOM_DURATION_MS: 60000, // 1 minute
  CHECK_INTERVAL_MS: 5000, // 5 seconds
} as const;

/**
 * Stress response parameters.
 */
export const STRESS_PARAMETERS = {
  THRESHOLD: 0.7,
  DECAY_RATE: 0.1,
  MAX_LEVEL: 1.0,
} as const;

/**
 * Hedonic (pleasure/reward) parameters.
 */
export const HEDONIC_PARAMETERS = {
  BASELINE: 0.5,
  REWARD_BOOST: 0.2,
  DECAY_RATE: 0.05,
} as const;

/**
 * Fear response parameters.
 */
export const FEAR_PARAMETERS = {
  THRESHOLD: 0.6,
  ESCALATION_RATE: 0.15,
  CALM_RATE: 0.1,
} as const;

/**
 * Admin mode boost multiplier.
 */
export const ADMIN_BOOST = 1.5;

/**
 * Idle consciousness level.
 */
export const IDLE_CONSCIOUSNESS = 0.3;

// =============================================================================
// E8 / KERNEL CONSTANTS
// =============================================================================

/**
 * Available kernel types.
 */
export const KERNEL_TYPES = [
  'zeus',
  'athena',
  'apollo',
  'ares',
  'hera',
  'poseidon',
  'demeter',
  'hephaestus',
  'artemis',
  'aphrodite',
  'hermes',
  'dionysus',
] as const;

export type KernelType = (typeof KERNEL_TYPES)[number];

/**
 * E8 root allocation for kernels.
 */
export const E8_ROOT_ALLOCATION = {
  zeus: { start: 0, count: 20 },
  athena: { start: 20, count: 20 },
  apollo: { start: 40, count: 20 },
  ares: { start: 60, count: 20 },
  hera: { start: 80, count: 20 },
  poseidon: { start: 100, count: 20 },
  demeter: { start: 120, count: 20 },
  hephaestus: { start: 140, count: 20 },
  artemis: { start: 160, count: 20 },
  aphrodite: { start: 180, count: 20 },
  hermes: { start: 200, count: 20 },
  dionysus: { start: 220, count: 20 },
} as const;

/**
 * Get E8 root index for a kernel type.
 */
export function getE8RootIndex(kernelType: KernelType): number {
  return E8_ROOT_ALLOCATION[kernelType].start;
}

/**
 * Get kernel type from E8 root index.
 */
export function getKernelTypeFromRoot(rootIndex: number): KernelType | null {
  for (const [type, allocation] of Object.entries(E8_ROOT_ALLOCATION)) {
    if (rootIndex >= allocation.start && rootIndex < allocation.start + allocation.count) {
      return type as KernelType;
    }
  }
  return null;
}

// =============================================================================
// PHYSICS CONVENIENCE ALIASES
// =============================================================================

/**
 * Phi threshold for consciousness (alias).
 */
export const PHI_THRESHOLD = PHI_MIN_CONSCIOUS;

/**
 * Phi threshold for detection (alias).
 */
export const PHI_THRESHOLD_DETECTION = PHI_CONSCIOUS_3D;

/**
 * Beta coupling constant (alias).
 */
export const BETA = PHYSICS_BETA;

/**
 * Minimum recursions for QIG validation.
 */
export const MIN_RECURSIONS = 1;

/**
 * Maximum recursions for QIG validation.
 */
export const MAX_RECURSIONS = 7;

/**
 * Critical L value for QIG.
 */
export const L_CRITICAL = 3;

/**
 * Resonance band width around kappa*.
 */
export const RESONANCE_BAND = 6.4;
