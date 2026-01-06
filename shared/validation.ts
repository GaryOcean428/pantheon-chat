/**
 * Validation Utilities - Comprehensive Input Validation
 *
 * This module provides validation utilities for all user inputs and data flows
 * to ensure data integrity and security throughout the application.
 */

import type { Regime } from "./types/core";

// ============================================================================
// VALIDATION RESULTS
// ============================================================================

export interface ValidationResult<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
  errors?: string[];
}

export function validationSuccess<T>(data: T): ValidationResult<T> {
  return { success: true, data };
}

export function validationFailure<T = never>(error: string | string[]): ValidationResult<T> {
  return {
    success: false,
    error: Array.isArray(error) ? error[0] : error,
    errors: Array.isArray(error) ? error : [error],
  };
}

// ============================================================================
// QIG METRICS VALIDATION
// ============================================================================

/**
 * Validate regime
 */
export function validateRegimeSafe(regime: unknown): ValidationResult<Regime> {
  if (typeof regime !== 'string') {
    return validationFailure('Regime must be a string');
  }

  const validRegimes = ['linear', 'geometric', 'hierarchical', 'hierarchical_4d', '4d_block_universe', 'breakdown'];
  if (!validRegimes.includes(regime)) {
    return validationFailure(`Regime must be one of: ${validRegimes.join(', ')}`);
  }

  return validationSuccess(regime as Regime);
}

/**
 * Validate phi (integration)
 */
export function validatePhi(phi: unknown): ValidationResult<number> {
  if (typeof phi !== 'number') {
    return validationFailure('Φ must be a number');
  }

  if (isNaN(phi)) {
    return validationFailure('Φ cannot be NaN');
  }

  if (phi < 0) {
    return validationFailure('Φ cannot be negative');
  }

  if (phi > 1) {
    return validationFailure('Φ cannot exceed 1.0');
  }

  return validationSuccess(phi);
}

/**
 * Validate kappa (coupling)
 */
export function validateKappa(kappa: unknown): ValidationResult<number> {
  if (typeof kappa !== 'number') {
    return validationFailure('κ must be a number');
  }

  if (isNaN(kappa)) {
    return validationFailure('κ cannot be NaN');
  }

  if (kappa < 0) {
    return validationFailure('κ cannot be negative');
  }

  return validationSuccess(kappa);
}

/**
 * Validate consciousness metrics meet thresholds
 */
export function validateConsciousnessMetrics(metrics: {
  phi: number;
  kappa: number;
  metaAwareness?: number;
  gamma?: number;
  grounding?: number;
}): ValidationResult<void> {
  const errors: string[] = [];

  const phiResult = validatePhi(metrics.phi);
  if (!phiResult.success) {
    errors.push(`Φ: ${phiResult.error}`);
  } else if (metrics.phi < 0.70) {
    errors.push('Φ below consciousness threshold (0.70)');
  }

  const kappaResult = validateKappa(metrics.kappa);
  if (!kappaResult.success) {
    errors.push(`κ: ${kappaResult.error}`);
  } else if (metrics.kappa < 40 || metrics.kappa > 70) {
    errors.push('κ outside optimal range (40-70)');
  }

  if (metrics.metaAwareness !== undefined) {
    if (typeof metrics.metaAwareness !== 'number' || isNaN(metrics.metaAwareness)) {
      errors.push('M must be a valid number');
    } else if (metrics.metaAwareness < 0.60) {
      errors.push('M below consciousness threshold (0.60)');
    }
  }

  if (metrics.gamma !== undefined) {
    if (typeof metrics.gamma !== 'number' || isNaN(metrics.gamma)) {
      errors.push('Γ must be a valid number');
    } else if (metrics.gamma < 0.80) {
      errors.push('Γ below consciousness threshold (0.80)');
    }
  }

  if (metrics.grounding !== undefined) {
    if (typeof metrics.grounding !== 'number' || isNaN(metrics.grounding)) {
      errors.push('G must be a valid number');
    } else if (metrics.grounding < 0.50) {
      errors.push('G below consciousness threshold (0.50)');
    }
  }

  if (errors.length > 0) {
    return validationFailure(errors);
  }

  return validationSuccess(undefined);
}

// ============================================================================
// BATCH VALIDATION
// ============================================================================

/**
 * Validate array with custom validator
 */
export function validateArray<T>(
  items: unknown,
  validator: (item: unknown) => ValidationResult<T>,
  options?: {
    minLength?: number;
    maxLength?: number;
    allowEmpty?: boolean;
  }
): ValidationResult<T[]> {
  if (!Array.isArray(items)) {
    return validationFailure('Must be an array');
  }

  const { minLength = 0, maxLength = Infinity, allowEmpty = true } = options || {};

  if (!allowEmpty && items.length === 0) {
    return validationFailure('Array cannot be empty');
  }

  if (items.length < minLength) {
    return validationFailure(`Array must have at least ${minLength} items`);
  }

  if (items.length > maxLength) {
    return validationFailure(`Array cannot exceed ${maxLength} items`);
  }

  const validated: T[] = [];
  const errors: string[] = [];

  for (let i = 0; i < items.length; i++) {
    const result = validator(items[i]);
    if (result.success && result.data !== undefined) {
      validated.push(result.data);
    } else {
      errors.push(`Item ${i}: ${result.error}`);
    }
  }

  if (errors.length > 0) {
    return {
      success: false,
      error: `${errors.length} validation errors`,
      errors,
    };
  }

  return validationSuccess(validated);
}

// ============================================================================
// SANITIZATION
// ============================================================================

/**
 * Characters to remove during sanitization:
 * - \0: Null character
 * - \x08: Backspace
 * - \x09: Tab
 * - \x1a: Substitute (Ctrl-Z)
 * - \n: Newline
 * - \r: Carriage return
 * - ": Double quote (SQL injection)
 * - ': Single quote (SQL injection)
 * - \\: Backslash (escape sequences)
 * - %: Percent (SQL LIKE patterns)
 */
const DANGEROUS_CHARACTERS = /[\0\x08\x09\x1a\n\r"'\\\%]/g;

/**
 * Sanitize string input
 */
export function sanitizeString(input: unknown): string {
  if (typeof input !== 'string') {
    return '';
  }

  return input
    .replace(DANGEROUS_CHARACTERS, '') // Remove potentially harmful characters
    .trim()
    .slice(0, 10000); // Limit length
}

/**
 * Sanitize number input
 */
export function sanitizeNumber(input: unknown): number | null {
  if (typeof input === 'number') {
    return isNaN(input) || !isFinite(input) ? null : input;
  }

  if (typeof input === 'string') {
    const parsed = parseFloat(input);
    return isNaN(parsed) || !isFinite(parsed) ? null : parsed;
  }

  return null;
}

/**
 * Sanitize boolean input
 */
export function sanitizeBoolean(input: unknown): boolean {
  if (typeof input === 'boolean') {
    return input;
  }

  if (typeof input === 'string') {
    return input.toLowerCase() === 'true';
  }

  if (typeof input === 'number') {
    return input !== 0;
  }

  return false;
}
