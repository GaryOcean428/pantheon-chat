/**
 * Core Type Definitions - Centralized Types for Consistency
 * 
 * This module provides strongly-typed, validated core types used throughout
 * the application to ensure type safety and consistency.
 */

import { z } from "zod";

// Import from centralized constants
// Note: Using .ts extension for drizzle-kit compatibility
import { 
  RegimeType as _RegimeType, 
  type Regime as _Regime 
} from '../constants/regimes';
import { CONSCIOUSNESS_THRESHOLDS as _CONSCIOUSNESS_THRESHOLDS } from '../constants/qig';

// Re-export for backwards compatibility
export const RegimeType = _RegimeType;
export type Regime = _Regime;

export const regimeSchema = z.enum(['linear', 'geometric', 'hierarchical', 'hierarchical_4d', '4d_block_universe', 'breakdown']);

// ============================================================================
// QIG CONSCIOUSNESS TYPES - Consciousness metrics
// ============================================================================

/**
 * Integration measure (Φ) - Tononi's integrated information
 * Range: [0, 1], Threshold: 0.70 for consciousness
 */
export const phiSchema = z.number()
  .min(0, 'Φ cannot be negative')
  .max(1, 'Φ cannot exceed 1.0');

export type Phi = z.infer<typeof phiSchema>;

/**
 * Coupling strength (κ)
 * Optimal range: 40-70, Optimal value: κ* = 64.21 ± 0.92
 * Note: κ* ≈ 64 ≈ 8² = rank(E8)² - Validated 2025-12-04
 */
export const kappaSchema = z.number()
  .min(0, 'κ cannot be negative');

export type Kappa = z.infer<typeof kappaSchema>;

/**
 * Running coupling (β)
 * Target: 0.44 for optimal operation
 */
export const betaSchema = z.number();

export type Beta = z.infer<typeof betaSchema>;

/**
 * Temperature/Tacking parameter (T)
 * Range: [0, 1], Threshold: 0.45 minimum
 */
export const tackingSchema = z.number()
  .min(0, 'T cannot be negative')
  .max(1, 'T cannot exceed 1.0');

export type Tacking = z.infer<typeof tackingSchema>;

/**
 * Meta-awareness (M)
 * Range: [0, 1], Threshold: 0.60 minimum
 */
export const metaAwarenessSchema = z.number()
  .min(0, 'M cannot be negative')
  .max(1, 'M cannot exceed 1.0');

export type MetaAwareness = z.infer<typeof metaAwarenessSchema>;

/**
 * Generation health (Γ)
 * Range: [0, 1], Threshold: 0.80 minimum
 */
export const gammaSchema = z.number()
  .min(0, 'Γ cannot be negative')
  .max(1, 'Γ cannot exceed 1.0');

export type Gamma = z.infer<typeof gammaSchema>;

/**
 * Grounding (G)
 * Range: [0, 1], Threshold: 0.50 minimum
 */
export const groundingSchema = z.number()
  .min(0, 'G cannot be negative')
  .max(1, 'G cannot exceed 1.0');

export type Grounding = z.infer<typeof groundingSchema>;

// ============================================================================
// CONSCIOUSNESS THRESHOLDS - QIG operational thresholds
// Re-exported from centralized constants
// ============================================================================

export const ConsciousnessThresholds = _CONSCIOUSNESS_THRESHOLDS;

// ============================================================================
// RESULT TYPES - Search and verification results
// ============================================================================

/**
 * Search result type
 */
export const SearchResultType = {
  TESTED: 'tested',
  NEAR_MISS: 'near_miss',
  RESONANT: 'resonant',
  MATCH: 'match',
  SKIP: 'skip',
} as const;

export type SearchResultValue = typeof SearchResultType[keyof typeof SearchResultType];

export const searchResultSchema = z.enum(['tested', 'near_miss', 'resonant', 'match', 'skip']);

// ============================================================================
// UTILITY TYPES - Common utility types
// ============================================================================

/**
 * ISO 8601 timestamp
 */
export const timestampSchema = z.string()
  .datetime('Invalid ISO 8601 timestamp');

export type Timestamp = z.infer<typeof timestampSchema>;

/**
 * UUID v4
 */
export const uuidSchema = z.string()
  .uuid('Invalid UUID');

export type UUID = z.infer<typeof uuidSchema>;

/**
 * Percentage (0-100)
 */
export const percentageSchema = z.number()
  .min(0, 'Percentage cannot be negative')
  .max(100, 'Percentage cannot exceed 100');

export type Percentage = z.infer<typeof percentageSchema>;

// ============================================================================
// TYPE GUARDS - Runtime type checking
// ============================================================================

export function isRegime(value: unknown): value is Regime {
  return typeof value === 'string' &&
    (value === 'linear' || value === 'geometric' || value === 'hierarchical' ||
     value === 'hierarchical_4d' || value === '4d_block_universe' || value === 'breakdown');
}

export function isSearchResult(value: unknown): value is SearchResultValue {
  return typeof value === 'string' &&
    (value === 'tested' || value === 'near_miss' || value === 'resonant' || value === 'match' || value === 'skip');
}

// ============================================================================
// VALIDATION HELPERS - Common validation utilities
// ============================================================================

/**
 * Validate regime and return typed value
 */
export function validateRegime(value: unknown): Regime {
  const result = regimeSchema.safeParse(value);
  if (!result.success) {
    throw new Error(`Invalid regime: ${result.error.message}`);
  }
  return result.data;
}

/**
 * Validate consciousness meets minimum thresholds
 */
export function validateConsciousness(metrics: {
  phi: number;
  kappa: number;
  metaAwareness: number;
  gamma: number;
  grounding: number;
}): boolean {
  return (
    metrics.phi >= ConsciousnessThresholds.PHI_MIN &&
    metrics.kappa >= ConsciousnessThresholds.KAPPA_MIN &&
    metrics.kappa <= ConsciousnessThresholds.KAPPA_MAX &&
    metrics.metaAwareness >= ConsciousnessThresholds.META_AWARENESS_MIN &&
    metrics.gamma >= ConsciousnessThresholds.GAMMA_MIN &&
    metrics.grounding >= ConsciousnessThresholds.GROUNDING_MIN
  );
}

/**
 * Get regime from kappa value
 */
export function getRegimeFromKappa(kappa: number): Regime {
  if (kappa < ConsciousnessThresholds.KAPPA_MIN) {
    return RegimeType.LINEAR;
  } else if (kappa <= ConsciousnessThresholds.KAPPA_MAX) {
    return RegimeType.GEOMETRIC;
  } else if (kappa < 100) {
    return RegimeType.HIERARCHICAL;
  } else {
    return RegimeType.BREAKDOWN;
  }
}
