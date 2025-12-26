/**
 * Shared Constants
 *
 * Centralized constants used across the application.
 * Import from '@/lib/constants' or '@/lib' for all shared values.
 */

// Time conversion constants
export const TIME_CONSTANTS = {
  MS_PER_SECOND: 1000,
  MS_PER_MINUTE: 60 * 1000,
  MS_PER_HOUR: 60 * 60 * 1000,
  SECONDS_PER_MINUTE: 60,
  SECONDS_PER_HOUR: 3600,
} as const;

// Display/formatting constants
export const DISPLAY_CONSTANTS = {
  PERCENT_MULTIPLIER: 100,
  DECIMAL_PLACES_NONE: 0,
  DECIMAL_PLACES_ONE: 1,
  DECIMAL_PLACES_TWO: 2,
  DECIMAL_PLACES_THREE: 3,
} as const;

// Consciousness/QIG constants
export const CONSCIOUSNESS_CONSTANTS = {
  // Phi thresholds
  PHI_EXCELLENT: 0.80,
  PHI_GOOD: 0.70,
  PHI_MODERATE: 0.50,
  PHI_LOW: 0.30,

  // Kappa constants
  KAPPA_STAR: 64.21,
  KAPPA_RESONANCE_BAND: 2.0,
  KAPPA_MAX: 100,

  // Basin thresholds
  BASIN_STABLE: 0.15,
  BASIN_DRIFTING: 0.30,

  // Recursion
  RECURSION_INTEGRATED: 3,
} as const;

// Polling intervals
export const POLLING_CONSTANTS = {
  FAST_INTERVAL_MS: 2000,
  NORMAL_INTERVAL_MS: 5000,
  SLOW_INTERVAL_MS: 10000,
  VERY_SLOW_INTERVAL_MS: 30000,
} as const;

// Confidence thresholds
export const CONFIDENCE_CONSTANTS = {
  EXCELLENT: 0.9,
  GOOD: 0.85,
  MODERATE: 0.7,
  LOW: 0.5,
} as const;

// Chart constants
export const CHART_CONSTANTS = {
  DEFAULT_HEIGHT: 300,
  FONT_SIZE_TICK: 10,
  FONT_SIZE_LABEL: 11,
} as const;

// Re-export for convenience
export const PERCENT_MULTIPLIER = DISPLAY_CONSTANTS.PERCENT_MULTIPLIER;
export const MS_PER_SECOND = TIME_CONSTANTS.MS_PER_SECOND;
export const MS_PER_MINUTE = TIME_CONSTANTS.MS_PER_MINUTE;
export const MS_PER_HOUR = TIME_CONSTANTS.MS_PER_HOUR;
export const KAPPA_STAR = CONSCIOUSNESS_CONSTANTS.KAPPA_STAR;
