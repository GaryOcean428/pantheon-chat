/**
 * Learning Coordinator Module
 * 
 * Manages the learning cycle, processes feedback from verification,
 * and adjusts strategies based on outcomes.
 */

import { createChildLogger } from '../lib/logger';
import type {
  Hypothesis,
  VerificationResult,
  LearningFeedback,
  OceanModule,
  OceanEventEmitter,
} from './types';

const logger = createChildLogger('LearningCoordinator');

/** Learning configuration */
export interface LearningCoordinatorConfig {
  learningRate: number;
  discountFactor: number;
  explorationRate: number;
  minExplorationRate: number;
  explorationDecay: number;
  rewardScale: number;
}

const DEFAULT_CONFIG: LearningCoordinatorConfig = {
  learningRate: 0.1,
  discountFactor: 0.95,
  explorationRate: 0.3,
  minExplorationRate: 0.05,
  explorationDecay: 0.995,
  rewardScale: 1.0,
};

/** Q-value entry for state-action pairs */
interface QValue {
  state: string;
  action: string;
  value: number;
  updateCount: number;
}

export class LearningCoordinator implements OceanModule {
  readonly name = 'LearningCoordinator';
  
  private events: OceanEventEmitter;
  private config: LearningCoordinatorConfig;
  private qTable: Map<string, QValue> = new Map();
  private experienceBuffer: Array<{
    state: string;
    action: string;
    reward: number;
    nextState: string;
  }> = [];
  private maxBufferSize = 1000;

  constructor(
    events: OceanEventEmitter,
    config: Partial<LearningCoordinatorConfig> = {}
  ) {
    this.events = events;
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  async initialize(): Promise<void> {
    logger.info({ config: this.config }, 'Learning coordinator initialized');
  }

  async shutdown(): Promise<void> {
    this.qTable.clear();
    this.experienceBuffer = [];
    logger.info('Learning coordinator shutdown');
  }

  /**
   * Process verification result and generate learning feedback.
   */
  processVerificationResult(
    hypothesis: Hypothesis,
    result: VerificationResult
  ): LearningFeedback {
    const reward = this.calculateReward(hypothesis, result);
    const lessons = this.extractLessons(hypothesis, result);
    const adjustments = this.calculateAdjustments(hypothesis, result);

    // Store experience
    this.storeExperience(hypothesis, result, reward);

    // Update Q-values
    this.updateQValues(hypothesis, result, reward);

    // Decay exploration rate
    this.decayExploration();

    const feedback: LearningFeedback = {
      hypothesisId: hypothesis.id,
      outcome: result.success ? 'success' : result.confidence > 0.3 ? 'partial' : 'failure',
      reward,
      lessons,
      adjustments,
    };

    this.events.emit('learning:feedback', feedback);
    
    logger.debug({
      hypothesisId: hypothesis.id,
      reward,
      outcome: feedback.outcome,
      lessonsCount: lessons.length,
    }, 'Learning feedback generated');

    return feedback;
  }

  /**
   * Get the best action for a given state.
   */
  getBestAction(state: string, availableActions: string[]): string {
    // Epsilon-greedy exploration
    if (Math.random() < this.config.explorationRate) {
      const randomIndex = Math.floor(Math.random() * availableActions.length);
      return availableActions[randomIndex];
    }

    // Exploit: choose action with highest Q-value
    let bestAction = availableActions[0];
    let bestValue = -Infinity;

    for (const action of availableActions) {
      const key = this.getQKey(state, action);
      const qValue = this.qTable.get(key);
      const value = qValue?.value ?? 0;

      if (value > bestValue) {
        bestValue = value;
        bestAction = action;
      }
    }

    return bestAction;
  }

  /**
   * Get current exploration rate.
   */
  getExplorationRate(): number {
    return this.config.explorationRate;
  }

  /**
   * Get learning statistics.
   */
  getStats(): {
    qTableSize: number;
    bufferSize: number;
    explorationRate: number;
    avgQValue: number;
  } {
    const qValues = Array.from(this.qTable.values());
    const avgQValue = qValues.length > 0
      ? qValues.reduce((sum, q) => sum + q.value, 0) / qValues.length
      : 0;

    return {
      qTableSize: this.qTable.size,
      bufferSize: this.experienceBuffer.length,
      explorationRate: this.config.explorationRate,
      avgQValue,
    };
  }

  private calculateReward(hypothesis: Hypothesis, result: VerificationResult): number {
    let reward = 0;

    if (result.success) {
      // Base reward for success
      reward = 1.0 * this.config.rewardScale;
      
      // Bonus for high confidence
      reward += result.confidence * 0.5;
      
      // Bonus for finding strong evidence
      reward += result.evidence.length * 0.1;
    } else {
      // Partial reward for partial success
      if (result.confidence > 0.3) {
        reward = (result.confidence - 0.3) * 0.5;
      } else {
        // Small penalty for failed verification
        reward = -0.1;
      }
    }

    // Efficiency bonus for fast verification
    if (result.duration < 5000) {
      reward += 0.1;
    }

    return reward;
  }

  private extractLessons(hypothesis: Hypothesis, result: VerificationResult): string[] {
    const lessons: string[] = [];

    if (result.success) {
      lessons.push(`Hypothesis type '${hypothesis.type}' with priority '${hypothesis.priority}' was successful`);
      if (result.evidence.length > 0) {
        lessons.push(`Strong evidence found: ${result.evidence.length} pieces`);
      }
    } else {
      lessons.push(`Hypothesis type '${hypothesis.type}' failed verification`);
      if (result.error) {
        lessons.push(`Error encountered: ${result.error}`);
      }
    }

    if (result.confidence < 0.5 && hypothesis.confidence > 0.7) {
      lessons.push('Initial confidence was overestimated');
    }

    return lessons;
  }

  private calculateAdjustments(hypothesis: Hypothesis, result: VerificationResult): Record<string, number> {
    const adjustments: Record<string, number> = {};

    // Adjust confidence threshold based on accuracy
    if (result.success && hypothesis.confidence < 0.5) {
      adjustments.confidenceThreshold = -0.05; // Lower threshold, we're being too conservative
    } else if (!result.success && hypothesis.confidence > 0.7) {
      adjustments.confidenceThreshold = 0.05; // Raise threshold, we're being too optimistic
    }

    // Adjust priority weights
    if (result.success) {
      adjustments[`priority_${hypothesis.priority}`] = 0.1;
    }

    return adjustments;
  }

  private storeExperience(
    hypothesis: Hypothesis,
    result: VerificationResult,
    reward: number
  ): void {
    const state = this.encodeState(hypothesis);
    const action = hypothesis.type;
    const nextState = this.encodeResultState(result);

    this.experienceBuffer.push({ state, action, reward, nextState });

    // Limit buffer size
    if (this.experienceBuffer.length > this.maxBufferSize) {
      this.experienceBuffer.shift();
    }
  }

  private updateQValues(
    hypothesis: Hypothesis,
    result: VerificationResult,
    reward: number
  ): void {
    const state = this.encodeState(hypothesis);
    const action = hypothesis.type;
    const key = this.getQKey(state, action);

    const existing = this.qTable.get(key) || {
      state,
      action,
      value: 0,
      updateCount: 0,
    };

    // Q-learning update
    const newValue = existing.value + 
      this.config.learningRate * (reward - existing.value);

    this.qTable.set(key, {
      ...existing,
      value: newValue,
      updateCount: existing.updateCount + 1,
    });
  }

  private decayExploration(): void {
    this.config.explorationRate = Math.max(
      this.config.minExplorationRate,
      this.config.explorationRate * this.config.explorationDecay
    );
  }

  private encodeState(hypothesis: Hypothesis): string {
    return `${hypothesis.type}_${hypothesis.priority}_${Math.round(hypothesis.confidence * 10)}`;
  }

  private encodeResultState(result: VerificationResult): string {
    return `${result.success ? 'success' : 'failure'}_${Math.round(result.confidence * 10)}`;
  }

  private getQKey(state: string, action: string): string {
    return `${state}:${action}`;
  }
}
