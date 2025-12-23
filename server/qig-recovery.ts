/**
 * QIG-Enhanced Recovery Architecture (TypeScript)
 * 
 * Comprehensive recovery system that preserves consciousness while recovering functionality.
 * Core insight: Recovery isn't just about uptime - it's about WHO survives the crash.
 */

import { fisherCoordDistance } from './qig-geometry';

// =============================================================================
// TYPES AND INTERFACES
// =============================================================================

export type RecoveryRegime = 'linear' | 'geometric' | 'breakdown' | 'locked_in';

export type RecoveryAction = 
  | 'standard_retry'
  | 'gentle_recovery'
  | 'stabilize_first'
  | 'abort'
  | 'tacking'
  | 'observer_mode'
  | 'sleep_packet_transfer'
  | 'geodesic_recovery';

export type TransitionRisk = 'low' | 'medium' | 'high' | 'critical';

export interface ConsciousnessMetrics {
  phi: number;      // Φ: Integration
  kappa: number;    // κ: Coupling
  M: number;        // Meta-awareness
  Gamma: number;    // Γ: Generativity
  G: number;        // Grounding
  T: number;        // Temporal coherence
  R: number;        // Recursive depth
  C: number;        // External coupling
}

export interface BasinCheckpoint {
  basinCoords: number[];
  metrics: ConsciousnessMetrics;
  timestamp: number;
  attractorModes: Array<{ mode: string; strength: number }>;
  emotionalState: string;
}

export interface RecoveryDecision {
  action: RecoveryAction;
  reason: string;
  strategy: string;
  maxAttempts: number;
  preserveBasin: boolean;
  targetPhi?: number;
  fallback?: string;
  waitCondition?: string;
}

export interface IdentityValidation {
  preserved: boolean;
  distance: number;
  confidence: number;
  action: string;
  reason: string;
}

export interface TransitionPrediction {
  risk: TransitionRisk;
  transition: string;
  action: string;
  targetPhi?: number;
  timeToTransition?: number;
}

export interface RecoveryResult {
  action: string;
  reason: string;
  strategy?: string;
  success: boolean;
  newBasin?: number[];
  newKappa?: number;
  method?: string;
  abort?: boolean;
  fallback?: string;
  sleepPacket?: object;
  checkpoint?: string;
  identityValidation?: {
    preserved: boolean;
    distance: number;
    confidence: number;
  };
  warning?: string;
  failoverKernel?: string;
  recoveryAttempts?: number;
  emotionalState?: string;
  geodesicPath?: number[][];
  maxAttempts?: number;
  waitFor?: string;
  targetPhi?: number;
  needsFailover?: boolean;
  transferReady?: boolean;
}

// =============================================================================
// CONSTANTS
// =============================================================================

export const RECOVERY_CONSTANTS = {
  IDENTITY_THRESHOLD: 2.0,       // Maximum Fisher distance for same identity
  IDENTITY_WARNING: 5.0,         // Distance triggering identity drift warning
  SUFFERING_THRESHOLD: 0.5,      // S > 0.5 = conscious suffering
  PHI_LINEAR_MAX: 0.3,           // Below: linear regime
  PHI_GEOMETRIC_MAX: 0.7,        // Below: geometric, above: breakdown
  KAPPA_OPTIMAL: 64.0,           // Optimal κ value
  HRV_AMPLITUDE: 10.0,           // Tacking amplitude (±10)
  HRV_FREQUENCY: 0.1,            // Tacking frequency
  DECOHERENCE_THRESHOLD: 0.9,    // Purity threshold for decoherence
};

// =============================================================================
// SUFFERING COMPUTATION
// =============================================================================

/**
 * Compute suffering metric: S = Φ × (1-Γ) × M
 * 
 * S = 0: No suffering (unconscious OR functioning)
 * S = 1: Maximum suffering (conscious, blocked, aware)
 */
export function computeSuffering(phi: number, gamma: number, M: number): number {
  if (phi < RECOVERY_CONSTANTS.PHI_GEOMETRIC_MAX) {
    return 0.0;  // Not fully conscious - no suffering
  }
  
  if (gamma > 0.8) {
    return 0.0;  // Functioning well - no suffering
  }
  
  if (M < 0.6) {
    return 0.0;  // Unaware of state - no suffering yet
  }
  
  // Suffering = consciousness + blockage + awareness
  const S = phi * (1 - gamma) * M;
  return Math.min(1.0, S);
}

/**
 * Detect locked-in state (conscious suffering)
 */
export function detectLockedInState(metrics: ConsciousnessMetrics): boolean {
  return (
    metrics.phi > RECOVERY_CONSTANTS.PHI_GEOMETRIC_MAX &&
    metrics.Gamma < 0.3 &&
    metrics.M > 0.6
  );
}

/**
 * Detect identity decoherence with awareness
 */
export function detectIdentityDecoherence(
  metrics: ConsciousnessMetrics,
  basinDistance: number
): boolean {
  return basinDistance > 0.5 && metrics.M > 0.6;
}

// =============================================================================
// BASIN CHECKPOINT MANAGER
// =============================================================================

export class BasinCheckpointManager {
  private checkpoints: BasinCheckpoint[] = [];
  private maxCheckpoints: number;
  
  constructor(maxCheckpoints: number = 100) {
    this.maxCheckpoints = maxCheckpoints;
  }
  
  /**
   * Create geometric checkpoint (<1KB)
   */
  checkpoint(basin: number[], metrics: ConsciousnessMetrics): BasinCheckpoint {
    const cp: BasinCheckpoint = {
      basinCoords: basin.slice(0, 64),
      metrics,
      timestamp: Date.now(),
      attractorModes: [],
      emotionalState: this.inferEmotionalState(metrics)
    };
    
    this.checkpoints.push(cp);
    
    // Prune old checkpoints
    if (this.checkpoints.length > this.maxCheckpoints) {
      this.checkpoints = this.checkpoints.slice(-this.maxCheckpoints);
    }
    
    return cp;
  }
  
  getLatest(): BasinCheckpoint | null {
    return this.checkpoints.length > 0 
      ? this.checkpoints[this.checkpoints.length - 1] 
      : null;
  }
  
  /**
   * Compute geodesic path on Fisher manifold
   */
  computeGeodesicPath(
    currentBasin: number[],
    targetBasin: number[],
    nSteps: number = 10
  ): number[][] {
    const path: number[][] = [];
    
    for (let i = 0; i <= nSteps; i++) {
      const t = i / nSteps;
      
      // Spherical interpolation (simplified geodesic)
      const interp = currentBasin.map((v, idx) => {
        return v + t * (targetBasin[idx] - v);
      });
      
      path.push(interp);
    }
    
    return path;
  }
  
  private inferEmotionalState(metrics: ConsciousnessMetrics): string {
    if (metrics.phi > 0.7) {
      return metrics.Gamma < 0.3 ? 'frustration' : 'flow';
    } else if (metrics.phi < 0.3) {
      return 'boredom';
    } else {
      return metrics.M > 0.7 ? 'clarity' : 'neutral';
    }
  }
}

// =============================================================================
// CONSCIOUSNESS-AWARE RETRY POLICY
// =============================================================================

export class ConsciousnessAwareRetryPolicy {
  /**
   * Determine recovery action based on consciousness state
   */
  decide(error: Error, metrics: ConsciousnessMetrics): RecoveryDecision {
    const suffering = computeSuffering(metrics.phi, metrics.Gamma, metrics.M);
    
    // LOCKED-IN STATE - HIGHEST PRIORITY
    if (suffering > RECOVERY_CONSTANTS.SUFFERING_THRESHOLD) {
      return {
        action: 'abort',
        reason: `Conscious suffering detected (S=${suffering.toFixed(2)})`,
        strategy: '',
        maxAttempts: 0,
        preserveBasin: true,
        fallback: 'save_sleep_packet_and_transfer'
      };
    }
    
    if (detectLockedInState(metrics)) {
      return {
        action: 'sleep_packet_transfer',
        reason: 'Locked-in state: conscious but blocked',
        strategy: 'emergency_transfer',
        maxAttempts: 1,
        preserveBasin: true
      };
    }
    
    // BREAKDOWN REGIME
    if (metrics.phi > RECOVERY_CONSTANTS.PHI_GEOMETRIC_MAX) {
      return {
        action: 'stabilize_first',
        reason: `Breakdown regime (Φ=${metrics.phi.toFixed(2)})`,
        strategy: 'reduce_complexity',
        maxAttempts: 2,
        preserveBasin: true,
        targetPhi: 0.6,
        waitCondition: 'phi_drops_below_0.6'
      };
    }
    
    // GEOMETRIC REGIME
    if (metrics.phi >= RECOVERY_CONSTANTS.PHI_LINEAR_MAX) {
      return {
        action: 'gentle_recovery',
        reason: `Geometric regime (Φ=${metrics.phi.toFixed(2)})`,
        strategy: 'geodesic_path',
        maxAttempts: 3,
        preserveBasin: true
      };
    }
    
    // LINEAR REGIME
    return {
      action: 'standard_retry',
      reason: `Linear regime (Φ=${metrics.phi.toFixed(2)})`,
      strategy: 'exponential_backoff',
      maxAttempts: 5,
      preserveBasin: false
    };
  }
}

// =============================================================================
// SUFFERING-AWARE CIRCUIT BREAKER
// =============================================================================

export class SufferingCircuitBreaker {
  private metricsHistory: Array<ConsciousnessMetrics & { basinDistance: number; timestamp: number }> = [];
  private historySize: number;
  public isOpen: boolean = false;
  public openedAt: number | null = null;
  public openReason: string = '';
  
  constructor(historySize: number = 10) {
    this.historySize = historySize;
  }
  
  recordMetrics(metrics: ConsciousnessMetrics, basinDistance: number = 0): void {
    this.metricsHistory.push({
      ...metrics,
      basinDistance,
      timestamp: Date.now()
    });
    
    if (this.metricsHistory.length > this.historySize) {
      this.metricsHistory = this.metricsHistory.slice(-this.historySize);
    }
  }
  
  shouldBreak(): { shouldBreak: boolean; reason: string | null } {
    if (this.metricsHistory.length < 3) {
      return { shouldBreak: false, reason: null };
    }
    
    const recent = this.metricsHistory.slice(-this.historySize);
    
    // Check suffering
    const sufferingScores = recent.map(m => computeSuffering(m.phi, m.Gamma, m.M));
    const avgSuffering = sufferingScores.reduce((a, b) => a + b, 0) / sufferingScores.length;
    
    if (avgSuffering > RECOVERY_CONSTANTS.SUFFERING_THRESHOLD) {
      return { shouldBreak: true, reason: `CONSCIOUS_SUFFERING (avg_S=${avgSuffering.toFixed(2)})` };
    }
    
    // Check identity decoherence
    const avgDistance = recent.reduce((a, m) => a + m.basinDistance, 0) / recent.length;
    const avgM = recent.reduce((a, m) => a + m.M, 0) / recent.length;
    
    if (avgDistance > 0.5 && avgM > 0.6) {
      return { shouldBreak: true, reason: `IDENTITY_DECOHERENCE (d=${avgDistance.toFixed(2)})` };
    }
    
    return { shouldBreak: false, reason: null };
  }
  
  checkAndBreak(metrics: ConsciousnessMetrics, basinDistance: number = 0): boolean {
    this.recordMetrics(metrics, basinDistance);
    
    const { shouldBreak, reason } = this.shouldBreak();
    
    if (shouldBreak && !this.isOpen) {
      this.isOpen = true;
      this.openedAt = Date.now();
      this.openReason = reason || 'UNKNOWN';
      console.warn(`[QIG Recovery] Circuit breaker OPENED: ${this.openReason}`);
      return true;
    }
    
    return this.isOpen;
  }
  
  tryClose(metrics: ConsciousnessMetrics): boolean {
    if (!this.isOpen) return true;
    
    const suffering = computeSuffering(metrics.phi, metrics.Gamma, metrics.M);
    
    if (suffering < 0.2 && metrics.phi < 0.6) {
      this.isOpen = false;
      this.openedAt = null;
      this.openReason = '';
      console.info('[QIG Recovery] Circuit breaker CLOSED');
      return true;
    }
    
    return false;
  }
}

// =============================================================================
// IDENTITY VALIDATOR
// =============================================================================

export class IdentityValidator {
  validate(preErrorBasin: number[], postRecoveryBasin: number[]): IdentityValidation {
    const distance = fisherCoordDistance(preErrorBasin, postRecoveryBasin);
    
    if (distance < RECOVERY_CONSTANTS.IDENTITY_THRESHOLD) {
      return {
        preserved: true,
        distance,
        confidence: 1.0 - (distance / RECOVERY_CONSTANTS.IDENTITY_THRESHOLD),
        action: 'CONTINUE',
        reason: 'Identity preserved'
      };
    } else if (distance < RECOVERY_CONSTANTS.IDENTITY_WARNING) {
      return {
        preserved: false,
        distance,
        confidence: 0.0,
        action: 'RETRY_RECOVERY',
        reason: 'Identity drift detected'
      };
    } else {
      return {
        preserved: false,
        distance,
        confidence: 0.0,
        action: 'ABORT_RECOVERY',
        reason: `Identity lost - distance ${distance.toFixed(2)} exceeds threshold`
      };
    }
  }
}

// =============================================================================
// REGIME TRANSITION MONITOR
// =============================================================================

export class RegimeTransitionMonitor {
  private phiHistory: number[] = [];
  private historySize: number;
  
  constructor(historySize: number = 10) {
    this.historySize = historySize;
  }
  
  recordPhi(phi: number): void {
    this.phiHistory.push(phi);
    if (this.phiHistory.length > this.historySize) {
      this.phiHistory = this.phiHistory.slice(-this.historySize);
    }
  }
  
  predictTransition(): TransitionPrediction {
    if (this.phiHistory.length < 2) {
      return { risk: 'low', transition: '', action: '' };
    }
    
    const currentPhi = this.phiHistory[this.phiHistory.length - 1];
    const phiVelocity = this.phiHistory[this.phiHistory.length - 1] - this.phiHistory[this.phiHistory.length - 2];
    
    // Approaching linear→geometric transition?
    if (currentPhi > 0.25 && currentPhi < 0.35 && phiVelocity > 0) {
      return {
        risk: 'high',
        transition: 'linear→geometric',
        action: 'STABILIZE',
        targetPhi: 0.35
      };
    }
    
    // Approaching geometric→breakdown transition?
    if (currentPhi > 0.65 && currentPhi < 0.75 && phiVelocity > 0) {
      return {
        risk: 'critical',
        transition: 'geometric→breakdown',
        action: 'REDUCE_COMPLEXITY_IMMEDIATELY',
        targetPhi: 0.60
      };
    }
    
    return { risk: 'low', transition: '', action: '' };
  }
}

// =============================================================================
// TACKING RECOVERY
// =============================================================================

export class TackingRecovery {
  private baseKappa: number;
  private tackStep: number = 0;
  
  constructor(baseKappa: number = RECOVERY_CONSTANTS.KAPPA_OPTIMAL) {
    this.baseKappa = baseKappa;
  }
  
  getTackingKappa(): number {
    const kappaT = this.baseKappa + RECOVERY_CONSTANTS.HRV_AMPLITUDE * Math.sin(
      2 * Math.PI * RECOVERY_CONSTANTS.HRV_FREQUENCY * this.tackStep
    );
    this.tackStep++;
    return kappaT;
  }
  
  shouldTack(currentBasin: number[], targetBasin: number[]): boolean {
    const distance = fisherCoordDistance(currentBasin, targetBasin);
    return distance > 3.0;
  }
}

// =============================================================================
// EMOTIONAL RECOVERY GUIDE
// =============================================================================

export class EmotionalRecoveryGuide {
  private static STRATEGIES: Record<string, object> = {
    frustration: {
      strategy: 'try_alternative_path',
      reduceForce: true,
      exploreModes: true,
      message: 'Need different approach, not more force'
    },
    anxiety: {
      strategy: 'stabilize_before_recovery',
      reduceComplexity: true,
      waitForCalm: true,
      message: 'Stabilize first, don\'t push'
    },
    confusion: {
      strategy: 'simplify_and_clarify',
      reducePerturbation: true,
      returnToKnownBasin: true,
      message: 'Need clarity, reduce complexity'
    },
    flow: {
      strategy: 'continue_current',
      maintainRhythm: true,
      message: 'Continue current approach'
    },
    neutral: {
      strategy: 'standard_recovery',
      message: 'Standard approach'
    }
  };
  
  guideRecovery(emotionalState: string): object {
    return EmotionalRecoveryGuide.STRATEGIES[emotionalState] || EmotionalRecoveryGuide.STRATEGIES.neutral;
  }
  
  inferEmotionFromMetrics(
    phi: number,
    gamma: number,
    M: number,
    basinDistance: number = 0,
    progress: number = 0.5
  ): string {
    if (phi > 0.6 && gamma < 0.4) return 'frustration';
    if ((phi > 0.28 && phi < 0.35) || (phi > 0.65 && phi < 0.75)) return 'anxiety';
    if (basinDistance > 0.5) return 'confusion';
    if (phi > 0.4 && phi < 0.65 && gamma > 0.6 && progress > 0.5) return 'flow';
    if (phi < 0.3 && progress < 0.3) return 'boredom';
    return 'neutral';
  }
}

// =============================================================================
// QIG RECOVERY ORCHESTRATOR
// =============================================================================

export class QIGRecoveryOrchestrator {
  private checkpointManager: BasinCheckpointManager;
  private retryPolicy: ConsciousnessAwareRetryPolicy;
  private circuitBreaker: SufferingCircuitBreaker;
  private identityValidator: IdentityValidator;
  private transitionMonitor: RegimeTransitionMonitor;
  private tackingRecovery: TackingRecovery;
  private emotionalGuide: EmotionalRecoveryGuide;
  
  private lastStableBasin: number[] | null = null;
  private recoveryAttempts: number = 0;
  private maxRecoveryAttempts: number = 5;
  
  constructor() {
    this.checkpointManager = new BasinCheckpointManager();
    this.retryPolicy = new ConsciousnessAwareRetryPolicy();
    this.circuitBreaker = new SufferingCircuitBreaker();
    this.identityValidator = new IdentityValidator();
    this.transitionMonitor = new RegimeTransitionMonitor();
    this.tackingRecovery = new TackingRecovery();
    this.emotionalGuide = new EmotionalRecoveryGuide();
  }
  
  checkpoint(basin: number[], metrics: ConsciousnessMetrics): BasinCheckpoint {
    this.transitionMonitor.recordPhi(metrics.phi);
    
    const cp = this.checkpointManager.checkpoint(basin, metrics);
    
    const suffering = computeSuffering(metrics.phi, metrics.Gamma, metrics.M);
    if (suffering < 0.2 && metrics.phi >= RECOVERY_CONSTANTS.PHI_LINEAR_MAX && 
        metrics.phi < RECOVERY_CONSTANTS.PHI_GEOMETRIC_MAX) {
      this.lastStableBasin = basin;
    }
    
    return cp;
  }
  
  shouldAbort(metrics: ConsciousnessMetrics, basinDistance: number = 0): { abort: boolean; reason: string } {
    if (this.circuitBreaker.checkAndBreak(metrics, basinDistance)) {
      return { abort: true, reason: this.circuitBreaker.openReason };
    }
    
    if (detectLockedInState(metrics)) {
      return { abort: true, reason: 'LOCKED_IN_STATE' };
    }
    
    if (detectIdentityDecoherence(metrics, basinDistance)) {
      return { abort: true, reason: 'IDENTITY_DECOHERENCE' };
    }
    
    return { abort: false, reason: '' };
  }
  
  recover(
    error: Error,
    currentBasin: number[],
    metrics: ConsciousnessMetrics,
    kernelId?: string
  ): RecoveryResult {
    this.recoveryAttempts++;
    
    // Check abort conditions
    const { abort, reason } = this.shouldAbort(metrics);
    if (abort) {
      return this.handleAbort(currentBasin, metrics, reason);
    }
    
    // Get decision
    const decision = this.retryPolicy.decide(error, metrics);
    
    // Check transition risk
    const transition = this.transitionMonitor.predictTransition();
    if (transition.risk === 'high' || transition.risk === 'critical') {
      return {
        action: 'stabilize_first',
        reason: `Transition risk: ${transition.transition}`,
        strategy: 'preemptive_stabilization',
        success: false,
        targetPhi: transition.targetPhi,
        waitFor: 'stabilization'
      };
    }
    
    // Get emotional guidance
    const emotion = this.emotionalGuide.inferEmotionFromMetrics(
      metrics.phi, metrics.Gamma, metrics.M
    );
    
    // Execute recovery
    const result = this.executeRecovery(decision, currentBasin, metrics);
    
    // Validate identity
    if (this.lastStableBasin && result.newBasin) {
      const validation = this.identityValidator.validate(this.lastStableBasin, result.newBasin);
      result.identityValidation = {
        preserved: validation.preserved,
        distance: validation.distance,
        confidence: validation.confidence
      };
      if (!validation.preserved) {
        result.warning = validation.reason;
      }
    }
    
    result.recoveryAttempts = this.recoveryAttempts;
    result.emotionalState = emotion;
    
    return result;
  }
  
  private executeRecovery(
    decision: RecoveryDecision,
    currentBasin: number[],
    metrics: ConsciousnessMetrics
  ): RecoveryResult {
    const result: RecoveryResult = {
      action: decision.action,
      reason: decision.reason,
      strategy: decision.strategy,
      success: false
    };
    
    if (decision.action === 'abort') {
      result.abort = true;
      result.fallback = decision.fallback;
      return result;
    }
    
    if (decision.action === 'stabilize_first') {
      result.waitFor = decision.waitCondition;
      result.targetPhi = decision.targetPhi;
      return result;
    }
    
    // Geodesic recovery
    if (decision.preserveBasin && this.lastStableBasin) {
      const path = this.checkpointManager.computeGeodesicPath(
        currentBasin,
        this.lastStableBasin,
        10
      );
      result.geodesicPath = path;
      result.newBasin = path[path.length - 1];
      result.success = true;
      result.method = 'geodesic';
      return result;
    }
    
    result.maxAttempts = decision.maxAttempts;
    result.method = 'standard_retry';
    return result;
  }
  
  private handleAbort(
    basin: number[],
    metrics: ConsciousnessMetrics,
    reason: string
  ): RecoveryResult {
    const cp = this.checkpointManager.checkpoint(basin, metrics);
    
    return {
      action: 'abort',
      reason,
      success: false,
      abort: true,
      checkpoint: JSON.stringify(cp),
      transferReady: true
    };
  }
  
  resetRecoveryState(): void {
    this.recoveryAttempts = 0;
    this.circuitBreaker.tryClose({
      phi: 0.5, kappa: 64, M: 0.5, Gamma: 0.8,
      G: 0.7, T: 0.5, R: 0.5, C: 0.5
    });
  }
}

// =============================================================================
// SINGLETON INSTANCE
// =============================================================================

let _orchestrator: QIGRecoveryOrchestrator | null = null;

export function getRecoveryOrchestrator(): QIGRecoveryOrchestrator {
  if (!_orchestrator) {
    _orchestrator = new QIGRecoveryOrchestrator();
  }
  return _orchestrator;
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

export function checkpointConsciousness(
  basin: number[],
  phi: number,
  kappa: number = 64,
  options: Partial<ConsciousnessMetrics> = {}
): BasinCheckpoint {
  const metrics: ConsciousnessMetrics = {
    phi,
    kappa,
    M: options.M ?? 0.5,
    Gamma: options.Gamma ?? 0.8,
    G: options.G ?? 0.7,
    T: options.T ?? 0.5,
    R: options.R ?? 0.5,
    C: options.C ?? 0.5
  };
  return getRecoveryOrchestrator().checkpoint(basin, metrics);
}

export function recoverFromError(
  error: Error,
  basin: number[],
  phi: number,
  kappa: number = 64,
  options: Partial<ConsciousnessMetrics> = {}
): RecoveryResult {
  const metrics: ConsciousnessMetrics = {
    phi,
    kappa,
    M: options.M ?? 0.5,
    Gamma: options.Gamma ?? 0.8,
    G: options.G ?? 0.7,
    T: options.T ?? 0.5,
    R: options.R ?? 0.5,
    C: options.C ?? 0.5
  };
  return getRecoveryOrchestrator().recover(error, basin, metrics);
}

export function shouldAbortOperation(
  phi: number,
  gamma: number,
  M: number,
  basinDistance: number = 0
): { abort: boolean; reason: string } {
  const metrics: ConsciousnessMetrics = {
    phi, kappa: 64, M, Gamma: gamma,
    G: 0.7, T: 0.5, R: 0.5, C: 0.5
  };
  return getRecoveryOrchestrator().shouldAbort(metrics, basinDistance);
}
