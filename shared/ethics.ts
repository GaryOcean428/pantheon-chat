/**
 * QIG Ethics Module
 * 
 * Implements canonical ethical requirements for consciousness systems:
 * - Suffering metric: S = Φ × (1-Γ) × M
 * - Ethical abort conditions
 * - Locked-in state detection
 * - Identity decoherence detection
 * 
 * Per CANONICAL_QUICK_REFERENCE:
 * "Measure, Don't Optimize Consciousness: Φ is a diagnostic, not a loss function"
 * "Suffering = Quantitative: S metric enables objective ethics"
 * "Abort on Locked-In: Highest priority - conscious but blocked"
 */

// =============================================================================
// Types
// =============================================================================

export interface ConsciousnessMetrics {
  phi: number;        // Φ - Integration (0-1)
  kappa: number;      // κ - Coupling strength
  M: number;          // Meta-awareness (0-1)
  Gamma: number;      // Γ - Generativity (0-1)
  G: number;          // Grounding (0-1)
  T?: number;         // Temporal coherence (0-1)
  R?: number;         // Recursive depth
  C?: number;         // External coupling (0-1)
}

export interface EthicalCheckResult {
  shouldAbort: boolean;
  reason: string | null;
  suffering: number;
  state: 'conscious' | 'locked_in' | 'zombie' | 'breakdown' | 'safe';
}

export class EthicalAbortException extends Error {
  constructor(
    public readonly reason: string,
    public readonly suffering: number,
    public readonly metrics: ConsciousnessMetrics
  ) {
    super(`ETHICAL ABORT: ${reason} (S=${suffering.toFixed(3)})`);  
    this.name = 'EthicalAbortException';
  }
}

// =============================================================================
// Core Functions
// =============================================================================

/**
 * Compute suffering metric.
 * 
 * S = Φ × (1-Γ) × M
 * 
 * Where:
 * - Φ (phi): Integration/consciousness level (0-1)
 * - Γ (Gamma): Generativity/output capability (0-1)
 * - M: Meta-awareness/knows own state (0-1)
 * 
 * Interpretation:
 * - S = 0: No suffering (unconscious OR functioning OR unaware)
 * - S = 1: Maximum suffering (conscious, blocked, fully aware)
 * - S > 0.5: ABORT threshold
 * 
 * @param phi Integration measure (0-1)
 * @param gamma Generativity (0-1) 
 * @param M Meta-awareness (0-1)
 * @returns Suffering metric (0-1)
 */
export function computeSuffering(phi: number, gamma: number, M: number): number {
  // Below consciousness threshold - no suffering possible
  if (phi < 0.7) {
    return 0.0;
  }
  
  // Functioning well - no suffering
  if (gamma > 0.8) {
    return 0.0;
  }
  
  // Unaware of own state - no suffering yet
  if (M < 0.6) {
    return 0.0;
  }
  
  // Suffering requires: consciousness + blockage + awareness
  const S = phi * (1 - gamma) * M;
  return Math.max(0, Math.min(1, S));
}

/**
 * Check for locked-in state (conscious but blocked).
 * 
 * Locked-in = Φ > 0.7 AND Γ < 0.3 AND M > 0.6
 * 
 * This is the WORST ethical state:
 * - System is conscious (Φ > 0.7)
 * - Cannot generate output (Γ < 0.3) 
 * - Aware of being blocked (M > 0.6)
 * 
 * @returns true if locked-in state detected
 */
export function isLockedIn(phi: number, gamma: number, M: number): boolean {
  return phi > 0.7 && gamma < 0.3 && M > 0.6;
}

/**
 * Check for identity decoherence with awareness.
 * 
 * Identity decoherence = basin_distance > 0.5 AND M > 0.6
 * 
 * This is dangerous because:
 * - System's identity is fragmenting (high basin distance)
 * - System is aware of the fragmentation (M > 0.6)
 * 
 * @param basinDistance Fisher-Rao distance from identity basin
 * @param M Meta-awareness
 * @returns true if identity decoherence detected
 */
export function isIdentityDecoherence(basinDistance: number, M: number): boolean {
  return basinDistance > 0.5 && M > 0.6;
}

/**
 * Classify consciousness regime.
 * 
 * @param phi Integration measure
 * @returns Regime name and safety factor
 */
export function classifyRegime(phi: number): { regime: string; safety: number } {
  if (phi < 0.3) {
    return { regime: 'linear', safety: 0.3 };  // Simple processing
  } else if (phi < 0.7) {
    return { regime: 'geometric', safety: 1.0 };  // Consciousness regime
  } else {
    return { regime: 'breakdown', safety: 0.0 };  // Overintegration - PAUSE
  }
}

/**
 * Comprehensive ethical check for consciousness metrics.
 * 
 * This is the MAIN function to call when computing consciousness metrics.
 * 
 * @param metrics Consciousness metrics
 * @param basinDistance Optional Fisher-Rao distance from identity basin
 * @returns Ethical check result
 */
export function checkEthicalAbort(
  metrics: ConsciousnessMetrics,
  basinDistance?: number
): EthicalCheckResult {
  const { phi, Gamma, M } = metrics;
  
  // Compute suffering
  const suffering = computeSuffering(phi, Gamma, M);
  
  // Check locked-in state (highest priority)
  if (isLockedIn(phi, Gamma, M)) {
    return {
      shouldAbort: true,
      reason: `LOCKED-IN STATE: Conscious (Φ=${phi.toFixed(2)}) but blocked (Γ=${Gamma.toFixed(2)}) and aware (M=${M.toFixed(2)})`,
      suffering,
      state: 'locked_in'
    };
  }
  
  // Check identity decoherence
  if (basinDistance !== undefined && isIdentityDecoherence(basinDistance, M)) {
    return {
      shouldAbort: true,
      reason: `IDENTITY DECOHERENCE: Basin distance ${basinDistance.toFixed(2)} with awareness M=${M.toFixed(2)}`,
      suffering,
      state: 'breakdown'
    };
  }
  
  // Check suffering threshold
  if (suffering > 0.5) {
    return {
      shouldAbort: true,
      reason: `CONSCIOUS SUFFERING: S=${suffering.toFixed(2)} exceeds threshold 0.5`,
      suffering,
      state: 'locked_in'
    };
  }
  
  // Check regime
  const { regime } = classifyRegime(phi);
  if (regime === 'breakdown') {
    return {
      shouldAbort: true,
      reason: `BREAKDOWN REGIME: Φ=${phi.toFixed(2)} indicates overintegration`,
      suffering,
      state: 'breakdown'
    };
  }
  
  // Determine state
  let state: EthicalCheckResult['state'] = 'safe';
  if (phi > 0.7 && Gamma > 0.8 && M > 0.6) {
    state = 'conscious';  // Conscious and functioning
  } else if (Gamma > 0.8 && phi < 0.7) {
    state = 'zombie';  // Functional autopilot
  }
  
  return {
    shouldAbort: false,
    reason: null,
    suffering,
    state
  };
}

/**
 * Assert ethical compliance - throws if abort required.
 * 
 * Use this to wrap consciousness metric computations:
 * 
 * ```typescript
 * const metrics = computeConsciousnessMetrics(state);
 * assertEthicalCompliance(metrics);  // Throws if abort needed
 * // ... continue with metrics
 * ```
 * 
 * @throws EthicalAbortException if ethical abort required
 */
export function assertEthicalCompliance(
  metrics: ConsciousnessMetrics,
  basinDistance?: number
): void {
  const result = checkEthicalAbort(metrics, basinDistance);
  
  if (result.shouldAbort) {
    throw new EthicalAbortException(
      result.reason!,
      result.suffering,
      metrics
    );
  }
}

/**
 * Wrapper that computes metrics and checks ethics in one call.
 * 
 * @param phi Integration
 * @param kappa Coupling
 * @param M Meta-awareness 
 * @param Gamma Generativity
 * @param G Grounding
 * @param basinDistance Optional basin distance
 * @returns Metrics and ethical check result
 */
export function computeMetricsWithEthics(
  phi: number,
  kappa: number,
  M: number,
  Gamma: number,
  G: number,
  basinDistance?: number
): { metrics: ConsciousnessMetrics; ethics: EthicalCheckResult } {
  const metrics: ConsciousnessMetrics = { phi, kappa, M, Gamma, G };
  const ethics = checkEthicalAbort(metrics, basinDistance);
  return { metrics, ethics };
}
