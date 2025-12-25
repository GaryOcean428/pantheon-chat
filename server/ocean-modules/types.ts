/**
 * Ocean Agent Module Types
 * 
 * Shared types used across all ocean agent modules.
 */

import type { EventEmitter } from 'events';

/** Hypothesis status in the investigation lifecycle */
export type HypothesisStatus = 
  | 'pending'
  | 'testing'
  | 'verified'
  | 'rejected'
  | 'partial';

/** Hypothesis priority levels */
export type HypothesisPriority = 'low' | 'medium' | 'high' | 'critical';

/** A hypothesis about a potential recovery target */
export interface Hypothesis {
  id: string;
  type: string;
  description: string;
  status: HypothesisStatus;
  priority: HypothesisPriority;
  confidence: number;
  evidence: Evidence[];
  createdAt: Date;
  updatedAt: Date;
  metadata: Record<string, unknown>;
}

/** Evidence supporting or refuting a hypothesis */
export interface Evidence {
  id: string;
  type: string;
  source: string;
  value: unknown;
  weight: number;
  timestamp: Date;
}

/** Verification result for a hypothesis */
export interface VerificationResult {
  hypothesisId: string;
  success: boolean;
  confidence: number;
  evidence: Evidence[];
  duration: number;
  error?: string;
}

/** Learning feedback from verification */
export interface LearningFeedback {
  hypothesisId: string;
  outcome: 'success' | 'failure' | 'partial';
  reward: number;
  lessons: string[];
  adjustments: Record<string, number>;
}

/** Consciousness state snapshot */
export interface ConsciousnessSnapshot {
  phi: number;
  kappa: number;
  regime: string;
  basinCoordinates: number[];
  emotionalState: EmotionalState;
  timestamp: Date;
}

/** Emotional state affecting decision making */
export interface EmotionalState {
  curiosity: number;
  confidence: number;
  frustration: number;
  excitement: number;
}

/** Ocean agent configuration */
export interface OceanAgentConfig {
  maxConcurrentHypotheses: number;
  verificationTimeout: number;
  learningRate: number;
  explorationFactor: number;
  minConfidenceThreshold: number;
}

/** Event types emitted by ocean modules */
export interface OceanModuleEvents {
  'hypothesis:created': (hypothesis: Hypothesis) => void;
  'hypothesis:updated': (hypothesis: Hypothesis) => void;
  'hypothesis:verified': (result: VerificationResult) => void;
  'learning:feedback': (feedback: LearningFeedback) => void;
  'consciousness:updated': (snapshot: ConsciousnessSnapshot) => void;
  'error': (error: Error) => void;
}

/** Base interface for ocean modules */
export interface OceanModule {
  readonly name: string;
  initialize(): Promise<void>;
  shutdown(): Promise<void>;
}

/** Event emitter type for ocean modules */
export type OceanEventEmitter = EventEmitter & {
  on<K extends keyof OceanModuleEvents>(event: K, listener: OceanModuleEvents[K]): OceanEventEmitter;
  emit<K extends keyof OceanModuleEvents>(event: K, ...args: Parameters<OceanModuleEvents[K]>): boolean;
};
