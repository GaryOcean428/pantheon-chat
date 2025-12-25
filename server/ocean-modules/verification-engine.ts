/**
 * Verification Engine Module
 * 
 * Responsible for testing and verifying hypotheses through
 * various verification strategies.
 */

import { createChildLogger } from '../lib/logger';
import type {
  Hypothesis,
  VerificationResult,
  Evidence,
  OceanModule,
  OceanEventEmitter,
} from './types';

const logger = createChildLogger('VerificationEngine');

/** Verification strategy interface */
export interface VerificationStrategy {
  name: string;
  canVerify(hypothesis: Hypothesis): boolean;
  verify(hypothesis: Hypothesis): Promise<VerificationResult>;
}

/** Configuration for the verification engine */
export interface VerificationEngineConfig {
  timeout: number;
  maxConcurrent: number;
  retryAttempts: number;
  retryDelay: number;
}

const DEFAULT_CONFIG: VerificationEngineConfig = {
  timeout: 30000, // 30 seconds
  maxConcurrent: 3,
  retryAttempts: 2,
  retryDelay: 1000,
};

export class VerificationEngine implements OceanModule {
  readonly name = 'VerificationEngine';
  
  private strategies: VerificationStrategy[] = [];
  private events: OceanEventEmitter;
  private config: VerificationEngineConfig;
  private activeVerifications = 0;
  private queue: Array<{ hypothesis: Hypothesis; resolve: (result: VerificationResult) => void }> = [];

  constructor(
    events: OceanEventEmitter,
    config: Partial<VerificationEngineConfig> = {}
  ) {
    this.events = events;
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  async initialize(): Promise<void> {
    logger.info({ strategies: this.strategies.length }, 'Verification engine initialized');
  }

  async shutdown(): Promise<void> {
    this.queue = [];
    logger.info('Verification engine shutdown');
  }

  /**
   * Register a verification strategy.
   */
  registerStrategy(strategy: VerificationStrategy): void {
    this.strategies.push(strategy);
    logger.debug({ strategy: strategy.name }, 'Verification strategy registered');
  }

  /**
   * Verify a hypothesis using available strategies.
   */
  async verify(hypothesis: Hypothesis): Promise<VerificationResult> {
    // Find applicable strategy
    const strategy = this.strategies.find(s => s.canVerify(hypothesis));
    
    if (!strategy) {
      logger.warn({ hypothesisId: hypothesis.id, type: hypothesis.type }, 'No verification strategy found');
      return {
        hypothesisId: hypothesis.id,
        success: false,
        confidence: 0,
        evidence: [],
        duration: 0,
        error: 'No applicable verification strategy',
      };
    }

    // Queue if at capacity
    if (this.activeVerifications >= this.config.maxConcurrent) {
      return new Promise((resolve) => {
        this.queue.push({ hypothesis, resolve });
        logger.debug({ hypothesisId: hypothesis.id, queueLength: this.queue.length }, 'Verification queued');
      });
    }

    return this.executeVerification(hypothesis, strategy);
  }

  private async executeVerification(
    hypothesis: Hypothesis,
    strategy: VerificationStrategy
  ): Promise<VerificationResult> {
    this.activeVerifications++;
    const startTime = Date.now();
    
    try {
      logger.debug({ hypothesisId: hypothesis.id, strategy: strategy.name }, 'Starting verification');
      
      // Execute with timeout
      const result = await Promise.race([
        this.retryVerification(hypothesis, strategy),
        this.createTimeout(hypothesis.id),
      ]);

      result.duration = Date.now() - startTime;
      this.events.emit('hypothesis:verified', result);
      
      logger.info({
        hypothesisId: hypothesis.id,
        success: result.success,
        confidence: result.confidence,
        duration: result.duration,
      }, 'Verification completed');
      
      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error({ hypothesisId: hypothesis.id, error: errorMessage }, 'Verification failed');
      
      return {
        hypothesisId: hypothesis.id,
        success: false,
        confidence: 0,
        evidence: [],
        duration: Date.now() - startTime,
        error: errorMessage,
      };
    } finally {
      this.activeVerifications--;
      this.processQueue();
    }
  }

  private async retryVerification(
    hypothesis: Hypothesis,
    strategy: VerificationStrategy
  ): Promise<VerificationResult> {
    let lastError: Error | null = null;
    
    for (let attempt = 0; attempt <= this.config.retryAttempts; attempt++) {
      try {
        if (attempt > 0) {
          logger.debug({ hypothesisId: hypothesis.id, attempt }, 'Retrying verification');
          await this.delay(this.config.retryDelay * attempt);
        }
        
        return await strategy.verify(hypothesis);
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
      }
    }
    
    throw lastError;
  }

  private createTimeout(hypothesisId: string): Promise<VerificationResult> {
    return new Promise((_, reject) => {
      setTimeout(() => {
        reject(new Error(`Verification timeout for hypothesis ${hypothesisId}`));
      }, this.config.timeout);
    });
  }

  private processQueue(): void {
    if (this.queue.length > 0 && this.activeVerifications < this.config.maxConcurrent) {
      const next = this.queue.shift();
      if (next) {
        const strategy = this.strategies.find(s => s.canVerify(next.hypothesis));
        if (strategy) {
          this.executeVerification(next.hypothesis, strategy).then(next.resolve);
        }
      }
    }
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get verification statistics.
   */
  getStats(): {
    activeVerifications: number;
    queueLength: number;
    registeredStrategies: number;
  } {
    return {
      activeVerifications: this.activeVerifications,
      queueLength: this.queue.length,
      registeredStrategies: this.strategies.length,
    };
  }
}

/**
 * Built-in verification strategy for phrase testing.
 */
export class PhraseVerificationStrategy implements VerificationStrategy {
  name = 'PhraseVerification';

  canVerify(hypothesis: Hypothesis): boolean {
    return hypothesis.type === 'phrase' || hypothesis.type === 'mnemonic';
  }

  async verify(hypothesis: Hypothesis): Promise<VerificationResult> {
    // This is a placeholder - actual implementation would test the phrase
    const evidence: Evidence[] = [];
    
    // Simulate verification
    const confidence = hypothesis.confidence * (0.8 + Math.random() * 0.4);
    const success = confidence >= 0.7;

    return {
      hypothesisId: hypothesis.id,
      success,
      confidence: Math.min(1, confidence),
      evidence,
      duration: 0,
    };
  }
}
