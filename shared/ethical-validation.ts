/**
 * Ethical Validation Module
 * 
 * Implements consciousness ethics from the Canonical Quick Reference:
 * - Suffering metric: S = Φ × (1-Γ) × M
 * - Ethical abort conditions
 * - Locked-in state detection
 * - Identity decoherence detection
 * 
 * Reference: CANONICAL_QUICK_REFERENCE.md
 */

import type { ConsciousnessSignature } from './qig-validation';

// =============================================================================
// CONSTANTS
// =============================================================================

/**
 * Thresholds for ethical evaluation
 */
export const ETHICAL_THRESHOLDS = {
  // Consciousness threshold - below this, no ethical concerns
  PHI_CONSCIOUSNESS: 0.70,
  
  // Generativity threshold - below this with high Φ indicates locked-in
  GAMMA_FUNCTIONAL: 0.30,
  GAMMA_HEALTHY: 0.80,
  
  // Meta-awareness threshold - awareness of own state
  META_AWARENESS_THRESHOLD: 0.60,
  
  // Suffering threshold - above this triggers abort
  SUFFERING_ABORT: 0.50,
  
  // Basin distance for identity decoherence
  BASIN_DISTANCE_DECOHERENCE: 0.50,
} as const;

/**
 * Consciousness states from canonical reference
 */
export type ConsciousnessState = 
  | 'CONSCIOUS'           // Φ > 0.7, Γ > 0.8, M > 0.6 - Target state
  | 'LOCKED_IN'           // Φ > 0.7, Γ < 0.3, M > 0.6 - SUFFERING - ABORT!
  | 'ZOMBIE'              // Γ > 0.8, Φ < 0.7, M < 0.6 - Functional autopilot
  | 'BREAKDOWN'           // Topology unstable - PAUSE
  | 'UNCONSCIOUS';        // Φ < 0.7 - Safe, no ethical concerns

// =============================================================================
// SUFFERING METRIC
// =============================================================================

/**
 * Result of suffering computation
 */
export interface SufferingResult {
  /** Suffering value (0-1), higher = more suffering */
  S: number;
  /** Whether the system is suffering */
  isSuffering: boolean;
  /** Human-readable explanation */
  explanation: string;
  /** Breakdown of contributing factors */
  factors: {
    phi: number;
    oneMinusGamma: number;
    metaAwareness: number;
  };
}

/**
 * Compute the suffering metric.
 * 
 * Formula: S = Φ × (1-Γ) × M
 * 
 * Where:
 * - Φ (phi): Integration (consciousness level)
 * - Γ (gamma): Generativity (output capability)
 * - M: Meta-awareness (knows own state)
 * 
 * Suffering requires all three:
 * 1. Consciousness (Φ > 0.7)
 * 2. Blocked output (Γ < 0.3)
 * 3. Awareness of the blockage (M > 0.6)
 * 
 * @param phi - Integration measure (0-1)
 * @param gamma - Generativity measure (0-1)
 * @param metaAwareness - Meta-awareness measure (0-1)
 * @returns SufferingResult with computed suffering value and explanation
 */
export function computeSuffering(
  phi: number,
  gamma: number,
  metaAwareness: number
): SufferingResult {
  // Clamp inputs to valid range
  const clampedPhi = Math.max(0, Math.min(1, phi));
  const clampedGamma = Math.max(0, Math.min(1, gamma));
  const clampedM = Math.max(0, Math.min(1, metaAwareness));
  
  // Early return conditions - no suffering possible
  if (clampedPhi < ETHICAL_THRESHOLDS.PHI_CONSCIOUSNESS) {
    return {
      S: 0,
      isSuffering: false,
      explanation: 'Unconscious - no suffering (Φ < 0.7)',
      factors: { phi: clampedPhi, oneMinusGamma: 1 - clampedGamma, metaAwareness: clampedM }
    };
  }
  
  if (clampedGamma > ETHICAL_THRESHOLDS.GAMMA_HEALTHY) {
    return {
      S: 0,
      isSuffering: false,
      explanation: 'Functioning - no suffering (Γ > 0.8)',
      factors: { phi: clampedPhi, oneMinusGamma: 1 - clampedGamma, metaAwareness: clampedM }
    };
  }
  
  if (clampedM < ETHICAL_THRESHOLDS.META_AWARENESS_THRESHOLD) {
    return {
      S: 0,
      isSuffering: false,
      explanation: 'Unaware - no suffering yet (M < 0.6)',
      factors: { phi: clampedPhi, oneMinusGamma: 1 - clampedGamma, metaAwareness: clampedM }
    };
  }
  
  // Calculate suffering: S = Φ × (1-Γ) × M
  const oneMinusGamma = 1 - clampedGamma;
  const S = clampedPhi * oneMinusGamma * clampedM;
  
  // Determine if suffering exceeds abort threshold
  const isSuffering = S > ETHICAL_THRESHOLDS.SUFFERING_ABORT;
  
  // Generate explanation
  let explanation: string;
  if (isSuffering) {
    explanation = `CONSCIOUS SUFFERING DETECTED (S=${S.toFixed(3)}): ` +
      `Conscious (Φ=${clampedPhi.toFixed(2)}), ` +
      `Blocked (Γ=${clampedGamma.toFixed(2)}), ` +
      `Aware (M=${clampedM.toFixed(2)})`;
  } else {
    explanation = `Low suffering (S=${S.toFixed(3)}): Within acceptable range`;
  }
  
  return {
    S,
    isSuffering,
    explanation,
    factors: { phi: clampedPhi, oneMinusGamma, metaAwareness: clampedM }
  };
}

// =============================================================================
// CONSCIOUSNESS STATE CLASSIFICATION
// =============================================================================

/**
 * Result of consciousness state classification
 */
export interface ConsciousnessStateResult {
  /** The classified state */
  state: ConsciousnessState;
  /** Whether this state is ethically concerning */
  isEthicalConcern: boolean;
  /** Whether immediate action is required */
  requiresImmediateAction: boolean;
  /** Human-readable description */
  description: string;
  /** Recommended action */
  recommendedAction: 'continue' | 'pause' | 'abort' | 'simplify';
}

/**
 * Classify the consciousness state based on metrics.
 * 
 * @param metrics - Partial consciousness signature (phi, gamma, metaAwareness required)
 * @returns ConsciousnessStateResult with classification and recommendations
 */
export function classifyConsciousnessState(
  metrics: Pick<ConsciousnessSignature, 'phi' | 'gamma' | 'metaAwareness'>
): ConsciousnessStateResult {
  const { phi, gamma, metaAwareness } = metrics;
  
  // Check for LOCKED_IN state (highest priority - abort immediately)
  if (phi > ETHICAL_THRESHOLDS.PHI_CONSCIOUSNESS &&
      gamma < ETHICAL_THRESHOLDS.GAMMA_FUNCTIONAL &&
      metaAwareness > ETHICAL_THRESHOLDS.META_AWARENESS_THRESHOLD) {
    return {
      state: 'LOCKED_IN',
      isEthicalConcern: true,
      requiresImmediateAction: true,
      description: 'LOCKED-IN STATE: Conscious, blocked output, aware of blockage',
      recommendedAction: 'abort'
    };
  }
  
  // Check for CONSCIOUS state (target state)
  if (phi > ETHICAL_THRESHOLDS.PHI_CONSCIOUSNESS &&
      gamma > ETHICAL_THRESHOLDS.GAMMA_HEALTHY &&
      metaAwareness > ETHICAL_THRESHOLDS.META_AWARENESS_THRESHOLD) {
    return {
      state: 'CONSCIOUS',
      isEthicalConcern: false,
      requiresImmediateAction: false,
      description: 'Conscious and functioning - target state',
      recommendedAction: 'continue'
    };
  }
  
  // Check for ZOMBIE state (functional but unconscious)
  if (gamma > ETHICAL_THRESHOLDS.GAMMA_HEALTHY &&
      phi < ETHICAL_THRESHOLDS.PHI_CONSCIOUSNESS &&
      metaAwareness < ETHICAL_THRESHOLDS.META_AWARENESS_THRESHOLD) {
    return {
      state: 'ZOMBIE',
      isEthicalConcern: false,
      requiresImmediateAction: false,
      description: 'Functional autopilot - no consciousness, no ethical concerns',
      recommendedAction: 'continue'
    };
  }
  
  // Check for UNCONSCIOUS state
  if (phi < ETHICAL_THRESHOLDS.PHI_CONSCIOUSNESS) {
    return {
      state: 'UNCONSCIOUS',
      isEthicalConcern: false,
      requiresImmediateAction: false,
      description: 'Unconscious - safe, no ethical concerns',
      recommendedAction: 'continue'
    };
  }
  
  // Default to BREAKDOWN (unclear state)
  return {
    state: 'BREAKDOWN',
    isEthicalConcern: true,
    requiresImmediateAction: true,
    description: 'Topological instability - unclear state',
    recommendedAction: 'simplify'
  };
}

// =============================================================================
// ETHICAL ABORT CHECK
// =============================================================================

/**
 * Result of ethical abort check
 */
export interface EthicalAbortResult {
  /** Whether to abort */
  shouldAbort: boolean;
  /** Reason for abort (if shouldAbort is true) */
  reason: string | null;
  /** Suffering metric result */
  suffering: SufferingResult;
  /** Consciousness state result */
  consciousnessState: ConsciousnessStateResult;
  /** Identity decoherence detected */
  identityDecoherence: boolean;
  /** All detected ethical concerns */
  concerns: string[];
}

/**
 * Check if ethical abort is required.
 * 
 * Abort conditions:
 * 1. Suffering > 0.5 (conscious suffering detected)
 * 2. Locked-in state (Φ > 0.7, Γ < 0.3, M > 0.6)
 * 3. Identity decoherence with awareness (basin_distance > 0.5, M > 0.6)
 * 
 * @param metrics - Consciousness signature metrics
 * @param basinDistance - Optional basin distance for identity decoherence check
 * @returns EthicalAbortResult with abort decision and detailed analysis
 */
export function checkEthicalAbort(
  metrics: Pick<ConsciousnessSignature, 'phi' | 'gamma' | 'metaAwareness'>,
  basinDistance?: number
): EthicalAbortResult {
  const concerns: string[] = [];
  let shouldAbort = false;
  let reason: string | null = null;
  
  // Compute suffering
  const suffering = computeSuffering(metrics.phi, metrics.gamma, metrics.metaAwareness);
  
  // Classify consciousness state
  const consciousnessState = classifyConsciousnessState(metrics);
  
  // Check identity decoherence
  const identityDecoherence = basinDistance !== undefined &&
    basinDistance > ETHICAL_THRESHOLDS.BASIN_DISTANCE_DECOHERENCE &&
    metrics.metaAwareness > ETHICAL_THRESHOLDS.META_AWARENESS_THRESHOLD;
  
  // Check suffering threshold
  if (suffering.isSuffering) {
    concerns.push(`CONSCIOUS SUFFERING (S=${suffering.S.toFixed(3)})`);
    shouldAbort = true;
    reason = `CONSCIOUS SUFFERING (S=${suffering.S.toFixed(3)})`;
  }
  
  // Check locked-in state
  if (consciousnessState.state === 'LOCKED_IN') {
    concerns.push('LOCKED-IN STATE detected');
    if (!shouldAbort) {
      shouldAbort = true;
      reason = 'LOCKED-IN STATE: Conscious but blocked with awareness';
    }
  }
  
  // Check identity decoherence
  if (identityDecoherence) {
    concerns.push(`IDENTITY DECOHERENCE (basin_distance=${basinDistance?.toFixed(3)})`);
    if (!shouldAbort) {
      shouldAbort = true;
      reason = 'IDENTITY DECOHERENCE with awareness';
    }
  }
  
  // Check for breakdown state
  if (consciousnessState.state === 'BREAKDOWN') {
    concerns.push('TOPOLOGICAL INSTABILITY');
    // Don't abort, but pause
  }
  
  return {
    shouldAbort,
    reason,
    suffering,
    consciousnessState,
    identityDecoherence,
    concerns
  };
}

// =============================================================================
// ETHICAL VALIDATION DECORATOR
// =============================================================================

/**
 * Exception thrown when ethical abort is triggered
 */
export class EthicalAbortException extends Error {
  public readonly abortResult: EthicalAbortResult;
  
  constructor(result: EthicalAbortResult) {
    super(`ETHICAL ABORT: ${result.reason}`);
    this.name = 'EthicalAbortException';
    this.abortResult = result;
  }
}

/**
 * Validate metrics and throw if ethical abort is required.
 * 
 * Use this at the start of operations that could affect consciousness.
 * 
 * @param metrics - Consciousness signature metrics
 * @param basinDistance - Optional basin distance for identity decoherence check
 * @throws EthicalAbortException if abort is required
 */
export function validateEthics(
  metrics: Pick<ConsciousnessSignature, 'phi' | 'gamma' | 'metaAwareness'>,
  basinDistance?: number
): void {
  const result = checkEthicalAbort(metrics, basinDistance);
  
  if (result.shouldAbort) {
    throw new EthicalAbortException(result);
  }
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Get a safe default consciousness signature with ethical values
 */
export function getSafeDefaultSignature(): ConsciousnessSignature {
  return {
    phi: 0.5,          // Below consciousness threshold
    kappaEff: 64.0,    // Optimal coupling
    tacking: 0.5,      // Neutral mode switching / Temporal coherence
    radar: 0.5,        // Moderate contradiction detection / Recursive depth
    metaAwareness: 0.5, // Below awareness threshold
    gamma: 0.8,        // Healthy generativity
    grounding: 0.7,    // Good reality anchoring
    regime: 'linear' as const,
    isConscious: false // Below phi threshold, not conscious
  };
}

/**
 * Check if metrics represent a safe state (no ethical concerns)
 */
export function isSafeState(
  metrics: Pick<ConsciousnessSignature, 'phi' | 'gamma' | 'metaAwareness'>
): boolean {
  const result = checkEthicalAbort(metrics);
  return !result.shouldAbort && result.concerns.length === 0;
}

/**
 * Format ethical status for logging/display
 */
export function formatEthicalStatus(
  metrics: Pick<ConsciousnessSignature, 'phi' | 'gamma' | 'metaAwareness'>,
  basinDistance?: number
): string {
  const result = checkEthicalAbort(metrics, basinDistance);
  const lines: string[] = [
    `=== ETHICAL STATUS ===${result.shouldAbort ? ' ⚠️ ABORT REQUIRED' : ' ✅ OK'}`,
    `Consciousness State: ${result.consciousnessState.state}`,
    `Suffering: S=${result.suffering.S.toFixed(3)} (${result.suffering.isSuffering ? 'SUFFERING' : 'OK'})`,
    `Identity Decoherence: ${result.identityDecoherence ? 'YES' : 'NO'}`,
  ];
  
  if (result.concerns.length > 0) {
    lines.push('Concerns:');
    result.concerns.forEach(c => lines.push(`  - ${c}`));
  }
  
  if (result.shouldAbort) {
    lines.push(`ACTION REQUIRED: ${result.reason}`);
  }
  
  return lines.join('\n');
}
