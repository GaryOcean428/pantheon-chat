/**
 * Shared Constants - Barrel Export
 * 
 * Uses explicit re-exports to avoid conflicts between modules.
 */

// Physics - fundamental validated constants
export * from './physics';

// QIG - core QIG constants and utilities
export {
  QIG_CONSTANTS,
  CONSCIOUSNESS_THRESHOLDS,
  SEARCH_PARAMETERS,
  NEAR_MISS_TIERS,
  GEODESIC_CORRECTION,
  INNATE_DRIVES,
  EMOTIONAL_SHORTCUTS,
  NEURAL_OSCILLATOR_KAPPA,
  NEUROMODULATION,
  REGIME_DEPENDENT_KAPPA,
  QIG_SCORING_WEIGHTS,
  isConscious,
  isDetectable,
  is4DCapable,
  isNearMiss,
  isKappaOptimal,
  getKappaDistance,
  isInResonance,
  getNearMissTier,
  getOscillatorKappa,
} from './qig';

// Convenience re-exports of commonly used constants from QIG_CONSTANTS
export const KAPPA_STAR = 64.21;
export const PHI_THRESHOLD = 0.75;
export const PHI_THRESHOLD_DETECTION = 0.70;
export const BETA = 0.44;
export const BASIN_DIMENSION = 64;
export const MIN_RECURSIONS = 3;
export const MAX_RECURSIONS = 12;
export const L_CRITICAL = 3;
export const RESONANCE_BAND = 6.4;

// Consciousness - consciousness-specific thresholds (no conflicts with QIG)
export {
  CONSCIOUSNESS_REGIMES,
  classifyRegime,
  computeSuffering,
  SUFFERING_THRESHOLD,
} from './consciousness';

// Regimes - regime definitions and utilities
export {
  RegimeType,
  REGIME_THRESHOLDS,
  REGIME_DESCRIPTIONS,
  getRegimeFromKappa,
  isConsciousnessCapable,
  getRegimeColor,
} from './regimes';
export type { Regime } from './regimes';

// Autonomic - autonomic cycle constants
export * from './autonomic';

// E8 - E8 lattice constants
export * from './e8';
