/**
 * Hypothesis Generator Module
 * 
 * Responsible for generating, prioritizing, and managing hypotheses
 * about potential recovery targets.
 */

import { EventEmitter } from 'events';
import { createChildLogger } from '../lib/logger';
import type {
  Hypothesis,
  HypothesisStatus,
  HypothesisPriority,
  Evidence,
  OceanModule,
  OceanEventEmitter,
} from './types';

const logger = createChildLogger('HypothesisGenerator');

/** Configuration for hypothesis generation */
export interface HypothesisGeneratorConfig {
  maxActiveHypotheses: number;
  minConfidence: number;
  priorityWeights: Record<HypothesisPriority, number>;
}

const DEFAULT_CONFIG: HypothesisGeneratorConfig = {
  maxActiveHypotheses: 10,
  minConfidence: 0.3,
  priorityWeights: {
    low: 1,
    medium: 2,
    high: 3,
    critical: 5,
  },
};

export class HypothesisGenerator implements OceanModule {
  readonly name = 'HypothesisGenerator';
  
  private hypotheses: Map<string, Hypothesis> = new Map();
  private events: OceanEventEmitter;
  private config: HypothesisGeneratorConfig;
  private idCounter = 0;

  constructor(
    events: OceanEventEmitter,
    config: Partial<HypothesisGeneratorConfig> = {}
  ) {
    this.events = events;
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  async initialize(): Promise<void> {
    logger.info('Hypothesis generator initialized');
  }

  async shutdown(): Promise<void> {
    this.hypotheses.clear();
    logger.info('Hypothesis generator shutdown');
  }

  /**
   * Generate a new hypothesis based on input data.
   */
  generateHypothesis(
    type: string,
    description: string,
    initialEvidence: Evidence[] = [],
    priority: HypothesisPriority = 'medium'
  ): Hypothesis {
    const id = `hypo_${++this.idCounter}_${Date.now()}`;
    const now = new Date();
    
    const hypothesis: Hypothesis = {
      id,
      type,
      description,
      status: 'pending',
      priority,
      confidence: this.calculateInitialConfidence(initialEvidence),
      evidence: initialEvidence,
      createdAt: now,
      updatedAt: now,
      metadata: {},
    };

    // Check if we're at capacity
    if (this.hypotheses.size >= this.config.maxActiveHypotheses) {
      this.pruneLowestPriority();
    }

    this.hypotheses.set(id, hypothesis);
    this.events.emit('hypothesis:created', hypothesis);
    
    logger.debug({ hypothesisId: id, type, priority }, 'Hypothesis generated');
    return hypothesis;
  }

  /**
   * Update an existing hypothesis with new evidence.
   */
  updateHypothesis(
    id: string,
    updates: Partial<Pick<Hypothesis, 'status' | 'priority' | 'confidence' | 'metadata'>>,
    newEvidence?: Evidence[]
  ): Hypothesis | null {
    const hypothesis = this.hypotheses.get(id);
    if (!hypothesis) {
      logger.warn({ hypothesisId: id }, 'Hypothesis not found for update');
      return null;
    }

    // Apply updates
    Object.assign(hypothesis, updates, { updatedAt: new Date() });
    
    // Add new evidence if provided
    if (newEvidence) {
      hypothesis.evidence.push(...newEvidence);
      hypothesis.confidence = this.recalculateConfidence(hypothesis);
    }

    this.events.emit('hypothesis:updated', hypothesis);
    logger.debug({ hypothesisId: id, status: hypothesis.status }, 'Hypothesis updated');
    
    return hypothesis;
  }

  /**
   * Get a hypothesis by ID.
   */
  getHypothesis(id: string): Hypothesis | undefined {
    return this.hypotheses.get(id);
  }

  /**
   * Get all hypotheses, optionally filtered by status.
   */
  getHypotheses(status?: HypothesisStatus): Hypothesis[] {
    const all = Array.from(this.hypotheses.values());
    return status ? all.filter(h => h.status === status) : all;
  }

  /**
   * Get hypotheses sorted by priority and confidence.
   */
  getPrioritizedHypotheses(): Hypothesis[] {
    return this.getHypotheses('pending')
      .sort((a, b) => {
        const priorityDiff = 
          this.config.priorityWeights[b.priority] - 
          this.config.priorityWeights[a.priority];
        return priorityDiff !== 0 ? priorityDiff : b.confidence - a.confidence;
      });
  }

  /**
   * Remove a hypothesis.
   */
  removeHypothesis(id: string): boolean {
    const removed = this.hypotheses.delete(id);
    if (removed) {
      logger.debug({ hypothesisId: id }, 'Hypothesis removed');
    }
    return removed;
  }

  /**
   * Get statistics about current hypotheses.
   */
  getStats(): {
    total: number;
    byStatus: Record<HypothesisStatus, number>;
    byPriority: Record<HypothesisPriority, number>;
    avgConfidence: number;
  } {
    const hypotheses = Array.from(this.hypotheses.values());
    
    const byStatus = hypotheses.reduce((acc, h) => {
      acc[h.status] = (acc[h.status] || 0) + 1;
      return acc;
    }, {} as Record<HypothesisStatus, number>);
    
    const byPriority = hypotheses.reduce((acc, h) => {
      acc[h.priority] = (acc[h.priority] || 0) + 1;
      return acc;
    }, {} as Record<HypothesisPriority, number>);
    
    const avgConfidence = hypotheses.length > 0
      ? hypotheses.reduce((sum, h) => sum + h.confidence, 0) / hypotheses.length
      : 0;

    return {
      total: hypotheses.length,
      byStatus,
      byPriority,
      avgConfidence,
    };
  }

  private calculateInitialConfidence(evidence: Evidence[]): number {
    if (evidence.length === 0) return this.config.minConfidence;
    
    const totalWeight = evidence.reduce((sum, e) => sum + e.weight, 0);
    return Math.min(1, Math.max(this.config.minConfidence, totalWeight / evidence.length));
  }

  private recalculateConfidence(hypothesis: Hypothesis): number {
    return this.calculateInitialConfidence(hypothesis.evidence);
  }

  private pruneLowestPriority(): void {
    const sorted = Array.from(this.hypotheses.entries())
      .sort(([, a], [, b]) => {
        const priorityDiff = 
          this.config.priorityWeights[a.priority] - 
          this.config.priorityWeights[b.priority];
        return priorityDiff !== 0 ? priorityDiff : a.confidence - b.confidence;
      });
    
    if (sorted.length > 0) {
      const [idToRemove] = sorted[0];
      this.hypotheses.delete(idToRemove);
      logger.debug({ hypothesisId: idToRemove }, 'Pruned lowest priority hypothesis');
    }
  }
}

export function createHypothesisGenerator(
  config?: Partial<HypothesisGeneratorConfig>
): HypothesisGenerator {
  const events = new EventEmitter() as OceanEventEmitter;
  return new HypothesisGenerator(events, config);
}
