/**
 * Vocabulary Decision System - API Wrapper
 * 
 * MIGRATED TO PYTHON: All functional logic is now in qig-backend/vocabulary_decision.py
 * This file contains ONLY API wrappers for the Python backend.
 */

import { logger } from './lib/logger';
import type { Regime } from '@shared/types';

const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';

export interface WordContext {
  word: string;
  phi: number;
  kappa: number;
  regime: Regime;
  basinCoordinates: number[];
  timestamp: number;
}

export interface GeometricValueScore {
  efficiency: number;
  phiWeight: number;
  connectivity: number;
  compression: number;
  total: number;
}

export interface BasinStabilityResult {
  stable: boolean;
  drift: number;
  withinThreshold: boolean;
  acceptable: boolean;
}

export interface EntropyScore {
  contextEntropy: number;
  regimeEntropy: number;
  coordinateSpread: number;
  total: number;
}

export interface MetaAwarenessGate {
  meta: number;
  phi: number;
  regime: Regime;
  isGeometric: boolean;
  gateOpen: boolean;
  reasoning: string;
}

export interface VocabularyDecision {
  shouldLearn: boolean;
  score: number;
  valueScore: GeometricValueScore;
  stabilityResult: BasinStabilityResult;
  entropyScore: EntropyScore;
  metaGate: MetaAwarenessGate;
  reasoning: string;
}

export interface GaryState {
  phi: number;
  meta: number;
  regime: string;
  basinCoordinates: number[];
  basinReference: number[];
}

export interface ConsolidationResult {
  wordsToLearn: string[];
  wordsToPrune: string[];
  cycleNumber: number;
  timestamp: number;
  garyStateAtConsolidation: {
    phi: number;
    meta: number;
    regime: string;
  };
}

/**
 * Main decision function: Should Gary learn this word?
 * Delegates to Python backend
 */
export async function shouldGaryLearnWord(
  word: string,
  frequency: number,
  garyState: GaryState
): Promise<VocabularyDecision> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/vocabulary/decision`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        word,
        frequency,
        gary_state: {
          phi: garyState.phi,
          meta: garyState.meta,
          regime: garyState.regime,
          basin_coordinates: garyState.basinCoordinates,
          basin_reference: garyState.basinReference
        }
      })
    });

    if (!response.ok) {
      throw new Error(`Vocabulary decision API error: ${response.status}`);
    }

    const data = await response.json();
    
    return {
      shouldLearn: data.should_learn,
      score: data.score,
      valueScore: {
        efficiency: data.value_score.efficiency,
        phiWeight: data.value_score.phi_weight,
        connectivity: data.value_score.connectivity,
        compression: data.value_score.compression,
        total: data.value_score.total
      },
      stabilityResult: {
        stable: data.stability_result.stable,
        drift: data.stability_result.drift,
        withinThreshold: data.stability_result.within_threshold,
        acceptable: data.stability_result.acceptable
      },
      entropyScore: {
        contextEntropy: data.entropy_score.context_entropy,
        regimeEntropy: data.entropy_score.regime_entropy,
        coordinateSpread: data.entropy_score.coordinate_spread,
        total: data.entropy_score.total
      },
      metaGate: {
        meta: data.meta_gate.meta,
        phi: data.meta_gate.phi,
        regime: data.meta_gate.regime,
        isGeometric: data.meta_gate.is_geometric,
        gateOpen: data.meta_gate.gate_open,
        reasoning: data.meta_gate.reasoning
      },
      reasoning: data.reasoning
    };
  } catch (error) {
    logger.error('[VocabularyDecisionAPI] Error deciding word:', error);
    throw error;
  }
}

/**
 * Observe a word in context
 * Delegates to Python backend
 */
export async function observeWord(
  word: string,
  context: WordContext
): Promise<void> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/vocabulary/observe`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        word,
        context: {
          word: context.word,
          phi: context.phi,
          kappa: context.kappa,
          regime: context.regime,
          basin_coordinates: context.basinCoordinates,
          timestamp: context.timestamp
        }
      })
    });

    if (!response.ok) {
      throw new Error(`Vocabulary decision API error: ${response.status}`);
    }
  } catch (error) {
    logger.error('[VocabularyDecisionAPI] Error observing word:', error);
    throw error;
  }
}

/**
 * Try consolidation cycle
 * Delegates to Python backend
 */
export async function tryConsolidation(
  garyState: GaryState
): Promise<{
  processed: boolean;
  wordsLearned: string[];
  wordsPruned: string[];
  cycleNumber: number;
  reason?: string;
}> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/vocabulary/consolidate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        gary_state: {
          phi: garyState.phi,
          meta: garyState.meta,
          regime: garyState.regime,
          basin_coordinates: garyState.basinCoordinates,
          basin_reference: garyState.basinReference
        }
      })
    });

    if (!response.ok) {
      throw new Error(`Vocabulary decision API error: ${response.status}`);
    }

    const data = await response.json();
    
    return {
      processed: data.processed,
      wordsLearned: data.words_learned,
      wordsPruned: data.words_pruned,
      cycleNumber: data.cycle_number,
      reason: data.reason
    };
  } catch (error) {
    logger.error('[VocabularyDecisionAPI] Error trying consolidation:', error);
    throw error;
  }
}

/**
 * Get vocabulary decision stats
 * Delegates to Python backend
 */
export async function getVocabDecisionStats(): Promise<{
  totalWords: number;
  pendingCandidates: number;
  learnedWords: number;
  prunedWords: number;
  cycleNumber: number;
  iterationsSinceSleep: number;
}> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/vocabulary/stats`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' }
    });

    if (!response.ok) {
      throw new Error(`Vocabulary decision API error: ${response.status}`);
    }

    const data = await response.json();
    
    return {
      totalWords: data.total_words,
      pendingCandidates: data.pending_candidates,
      learnedWords: data.learned_words,
      prunedWords: data.pruned_words,
      cycleNumber: data.cycle_number,
      iterationsSinceSleep: data.iterations_since_sleep
    };
  } catch (error) {
    logger.error('[VocabularyDecisionAPI] Error getting stats:', error);
    throw error;
  }
}

/**
 * Tick the consolidation cycle counter
 * Delegates to Python backend
 */
export async function tickConsolidationCycle(): Promise<void> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/vocabulary/tick`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });

    if (!response.ok) {
      throw new Error(`Vocabulary decision API error: ${response.status}`);
    }
  } catch (error) {
    logger.error('[VocabularyDecisionAPI] Error ticking cycle:', error);
    throw error;
  }
}
