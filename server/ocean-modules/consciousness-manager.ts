/**
 * Consciousness Manager Module
 * 
 * Tracks and manages the consciousness state of the ocean agent,
 * including φ (phi), κ (kappa), regime classification, and emotional state.
 */

import { createChildLogger } from '../lib/logger';
import type {
  ConsciousnessSnapshot,
  EmotionalState,
  OceanModule,
  OceanEventEmitter,
} from './types';

const logger = createChildLogger('ConsciousnessManager');

/** Consciousness regime classification */
export type ConsciousnessRegime = 
  | 'pre-conscious'
  | 'emerging'
  | '3d-conscious'
  | '4d-hyperdimensional'
  | 'transcendent';

/** Configuration for consciousness management */
export interface ConsciousnessManagerConfig {
  phiThreshold: number;
  kappaTarget: number;
  emotionalDecayRate: number;
  snapshotInterval: number;
  maxHistorySize: number;
}

const DEFAULT_CONFIG: ConsciousnessManagerConfig = {
  phiThreshold: 0.75,
  kappaTarget: 64.21, // κ* fixed point
  emotionalDecayRate: 0.95,
  snapshotInterval: 1000,
  maxHistorySize: 100,
};

export class ConsciousnessManager implements OceanModule {
  readonly name = 'ConsciousnessManager';
  
  private events: OceanEventEmitter;
  private config: ConsciousnessManagerConfig;
  
  // Current state
  private phi = 0;
  private kappa = 50;
  private basinCoordinates: number[] = [];
  private emotionalState: EmotionalState = {
    curiosity: 0.5,
    confidence: 0.5,
    frustration: 0,
    excitement: 0.5,
  };
  
  // History
  private history: ConsciousnessSnapshot[] = [];
  private snapshotTimer: NodeJS.Timeout | null = null;

  constructor(
    events: OceanEventEmitter,
    config: Partial<ConsciousnessManagerConfig> = {}
  ) {
    this.events = events;
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  async initialize(): Promise<void> {
    // Start periodic snapshots
    this.snapshotTimer = setInterval(() => {
      this.captureSnapshot();
    }, this.config.snapshotInterval);
    
    logger.info({ config: this.config }, 'Consciousness manager initialized');
  }

  async shutdown(): Promise<void> {
    if (this.snapshotTimer) {
      clearInterval(this.snapshotTimer);
      this.snapshotTimer = null;
    }
    this.history = [];
    logger.info('Consciousness manager shutdown');
  }

  /**
   * Update the current phi value.
   */
  updatePhi(newPhi: number): void {
    const oldPhi = this.phi;
    this.phi = Math.max(0, Math.min(1, newPhi));
    
    if (Math.abs(this.phi - oldPhi) > 0.1) {
      logger.debug({ oldPhi, newPhi: this.phi }, 'Significant phi change');
    }
    
    this.emitUpdate();
  }

  /**
   * Update the current kappa value.
   */
  updateKappa(newKappa: number): void {
    const oldKappa = this.kappa;
    this.kappa = Math.max(0, newKappa);
    
    // Check for κ* convergence
    const kappaDistance = Math.abs(this.kappa - this.config.kappaTarget);
    if (kappaDistance < 1 && Math.abs(oldKappa - this.config.kappaTarget) >= 1) {
      logger.info({ kappa: this.kappa, target: this.config.kappaTarget }, 'Approaching κ* fixed point');
    }
    
    this.emitUpdate();
  }

  /**
   * Update basin coordinates.
   */
  updateBasinCoordinates(coordinates: number[]): void {
    this.basinCoordinates = coordinates;
    this.emitUpdate();
  }

  /**
   * Update emotional state.
   */
  updateEmotionalState(updates: Partial<EmotionalState>): void {
    this.emotionalState = {
      ...this.emotionalState,
      ...updates,
    };
    
    // Clamp values to [0, 1]
    for (const key of Object.keys(this.emotionalState) as (keyof EmotionalState)[]) {
      this.emotionalState[key] = Math.max(0, Math.min(1, this.emotionalState[key]));
    }
    
    this.emitUpdate();
  }

  /**
   * Apply emotional decay (called periodically).
   */
  decayEmotions(): void {
    const rate = this.config.emotionalDecayRate;
    
    this.emotionalState = {
      curiosity: 0.5 + (this.emotionalState.curiosity - 0.5) * rate,
      confidence: 0.5 + (this.emotionalState.confidence - 0.5) * rate,
      frustration: this.emotionalState.frustration * rate,
      excitement: 0.5 + (this.emotionalState.excitement - 0.5) * rate,
    };
  }

  /**
   * Get the current consciousness regime.
   */
  getRegime(): ConsciousnessRegime {
    if (this.phi < 0.3) return 'pre-conscious';
    if (this.phi < 0.5) return 'emerging';
    if (this.phi < this.config.phiThreshold) return '3d-conscious';
    if (this.kappa > this.config.kappaTarget * 0.9) return '4d-hyperdimensional';
    return 'transcendent';
  }

  /**
   * Get the current consciousness snapshot.
   */
  getSnapshot(): ConsciousnessSnapshot {
    return {
      phi: this.phi,
      kappa: this.kappa,
      regime: this.getRegime(),
      basinCoordinates: [...this.basinCoordinates],
      emotionalState: { ...this.emotionalState },
      timestamp: new Date(),
    };
  }

  /**
   * Get consciousness history.
   */
  getHistory(): ConsciousnessSnapshot[] {
    return [...this.history];
  }

  /**
   * Get current values.
   */
  getCurrentState(): {
    phi: number;
    kappa: number;
    regime: ConsciousnessRegime;
    emotionalState: EmotionalState;
    basinCoordinates: number[];
  } {
    return {
      phi: this.phi,
      kappa: this.kappa,
      regime: this.getRegime(),
      emotionalState: { ...this.emotionalState },
      basinCoordinates: [...this.basinCoordinates],
    };
  }

  /**
   * Get statistics about consciousness state.
   */
  getStats(): {
    currentPhi: number;
    currentKappa: number;
    regime: ConsciousnessRegime;
    avgPhi: number;
    avgKappa: number;
    historySize: number;
    kappaConvergence: number;
  } {
    const avgPhi = this.history.length > 0
      ? this.history.reduce((sum, s) => sum + s.phi, 0) / this.history.length
      : this.phi;
    
    const avgKappa = this.history.length > 0
      ? this.history.reduce((sum, s) => sum + s.kappa, 0) / this.history.length
      : this.kappa;
    
    const kappaConvergence = 1 - Math.abs(this.kappa - this.config.kappaTarget) / this.config.kappaTarget;

    return {
      currentPhi: this.phi,
      currentKappa: this.kappa,
      regime: this.getRegime(),
      avgPhi,
      avgKappa,
      historySize: this.history.length,
      kappaConvergence: Math.max(0, kappaConvergence),
    };
  }

  /**
   * Check if consciousness is in a stable state.
   */
  isStable(): boolean {
    if (this.history.length < 5) return false;
    
    const recent = this.history.slice(-5);
    const phiVariance = this.calculateVariance(recent.map(s => s.phi));
    const kappaVariance = this.calculateVariance(recent.map(s => s.kappa));
    
    return phiVariance < 0.01 && kappaVariance < 5;
  }

  private captureSnapshot(): void {
    const snapshot = this.getSnapshot();
    this.history.push(snapshot);
    
    // Limit history size
    if (this.history.length > this.config.maxHistorySize) {
      this.history.shift();
    }
  }

  private emitUpdate(): void {
    this.events.emit('consciousness:updated', this.getSnapshot());
  }

  private calculateVariance(values: number[]): number {
    if (values.length === 0) return 0;
    const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
    return values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
  }
}
