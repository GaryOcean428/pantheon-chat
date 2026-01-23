/**
 * Vocabulary Expander - API Wrapper
 * 
 * MIGRATED TO PYTHON: All functional logic is now in qig-backend/vocabulary_expander.py
 * This file contains ONLY API wrappers for the Python backend.
 */

import { logger } from './lib/logger';
import type { Regime } from '@shared/types';

const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';

export interface ManifoldWord {
  text: string;
  coordinates: number[];
  phi: number;
  kappa: number;
  frequency: number;
  components?: string[];
  geodesicOrigin?: string;
}

export interface ExpansionEvent {
  timestamp: string;
  word: string;
  type: 'learned' | 'compound' | 'pattern';
  components?: string[];
  phi: number;
  reasoning: string;
}

export interface QIGScore {
  keyType: string;
  phi: number;
  kappa: number;
  beta: number;
  basinCoordinates: number[];
  regime: Regime;
  patternScore: number;
  quality: number;
}

/**
 * Add a new word to the Fisher manifold
 * Delegates to Python backend
 */
export async function addWord(
  text: string,
  qigScore: QIGScore,
  options: {
    components?: string[];
    source?: string;
  } = {}
): Promise<ManifoldWord> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/vocabulary/expander/add`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        qig_score: {
          key_type: qigScore.keyType,
          phi: qigScore.phi,
          kappa: qigScore.kappa,
          beta: qigScore.beta,
          basin_coordinates: qigScore.basinCoordinates,
          regime: qigScore.regime,
          pattern_score: qigScore.patternScore,
          quality: qigScore.quality
        },
        components: options.components,
        source: options.source
      })
    });

    if (!response.ok) {
      throw new Error(`Vocabulary expander API error: ${response.status}`);
    }

    const data = await response.json();
    
    return {
      text: data.text,
      coordinates: data.coordinates,
      phi: data.phi,
      kappa: data.kappa,
      frequency: data.frequency,
      components: data.components,
      geodesicOrigin: data.geodesic_origin
    };
  } catch (error) {
    logger.error('[VocabularyExpanderAPI] Error adding word:', error);
    throw error;
  }
}

/**
 * Check and execute automatic vocabulary expansion
 * Delegates to Python backend
 */
export async function checkAutoExpansion(): Promise<ExpansionEvent[]> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/vocabulary/expander/auto-expand`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });

    if (!response.ok) {
      throw new Error(`Vocabulary expander API error: ${response.status}`);
    }

    const data = await response.json();
    
    return data.expansions.map((event: any) => ({
      timestamp: event.timestamp,
      word: event.word,
      type: event.type,
      components: event.components,
      phi: event.phi,
      reasoning: event.reasoning
    }));
  } catch (error) {
    logger.error('[VocabularyExpanderAPI] Error checking auto expansion:', error);
    throw error;
  }
}

/**
 * Get word from manifold
 * Delegates to Python backend
 */
export async function getWord(text: string): Promise<ManifoldWord | null> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/vocabulary/expander/word/${encodeURIComponent(text)}`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' }
    });

    if (response.status === 404) {
      return null;
    }

    if (!response.ok) {
      throw new Error(`Vocabulary expander API error: ${response.status}`);
    }

    const data = await response.json();
    
    return {
      text: data.text,
      coordinates: data.coordinates,
      phi: data.phi,
      kappa: data.kappa,
      frequency: data.frequency,
      components: data.components,
      geodesicOrigin: data.geodesic_origin
    };
  } catch (error) {
    logger.error('[VocabularyExpanderAPI] Error getting word:', error);
    throw error;
  }
}

/**
 * Find words near a point on the manifold
 * Delegates to Python backend
 */
export async function findNearbyWords(
  coordinates: number[],
  maxDistance: number = 2.0
): Promise<ManifoldWord[]> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/vocabulary/expander/nearby`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        coordinates,
        max_distance: maxDistance
      })
    });

    if (!response.ok) {
      throw new Error(`Vocabulary expander API error: ${response.status}`);
    }

    const data = await response.json();
    
    return data.words.map((word: any) => ({
      text: word.text,
      coordinates: word.coordinates,
      phi: word.phi,
      kappa: word.kappa,
      frequency: word.frequency,
      components: word.components,
      geodesicOrigin: word.geodesic_origin
    }));
  } catch (error) {
    logger.error('[VocabularyExpanderAPI] Error finding nearby words:', error);
    throw error;
  }
}

/**
 * Generate hypotheses from vocabulary manifold
 * Delegates to Python backend
 */
export async function generateManifoldHypotheses(count: number = 20): Promise<string[]> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/vocabulary/expander/hypotheses`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ count })
    });

    if (!response.ok) {
      throw new Error(`Vocabulary expander API error: ${response.status}`);
    }

    const data = await response.json();
    return data.hypotheses;
  } catch (error) {
    logger.error('[VocabularyExpanderAPI] Error generating hypotheses:', error);
    throw error;
  }
}

/**
 * Get vocabulary manifold statistics
 * Delegates to Python backend
 */
export async function getExpanderStats(): Promise<{
  totalWords: number;
  totalExpansions: number;
  highPhiWords: number;
  avgPhi: number;
  recentExpansions: ExpansionEvent[];
  topWords: Array<{text: string; phi: number; frequency: number}>;
}> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/vocabulary/expander/stats`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' }
    });

    if (!response.ok) {
      throw new Error(`Vocabulary expander API error: ${response.status}`);
    }

    const data = await response.json();
    
    return {
      totalWords: data.total_words,
      totalExpansions: data.total_expansions,
      highPhiWords: data.high_phi_words,
      avgPhi: data.avg_phi,
      recentExpansions: data.recent_expansions.map((event: any) => ({
        timestamp: event.timestamp,
        word: event.word,
        type: event.type,
        components: event.components,
        phi: event.phi,
        reasoning: event.reasoning
      })),
      topWords: data.top_words
    };
  } catch (error) {
    logger.error('[VocabularyExpanderAPI] Error getting stats:', error);
    throw error;
  }
}
