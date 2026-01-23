/**
 * Vocabulary Tracker - API Wrapper
 * 
 * MIGRATED TO PYTHON: All functional logic is now in qig-backend/vocabulary_tracker.py
 * This file contains ONLY API wrappers for the Python backend.
 */

import { logger } from './lib/logger';

const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';

export type PhraseCategory = 'bip39_seed' | 'passphrase' | 'mutation' | 'bip39_word' | 'unknown';

export interface VocabularyCandidate {
  text: string;
  type: 'word' | 'phrase' | 'sequence';
  frequency: number;
  avgPhi: number;
  maxPhi: number;
  efficiencyGain: number;
  reasoning: string;
  isRealWord: boolean;
  components?: string[];
}

/**
 * Observe a phrase from search results
 * Delegates to Python backend
 */
export async function observe(
  phrase: string,
  phi: number,
  kappa?: number,
  regime?: string,
  basinCoordinates?: number[]
): Promise<void> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/vocabulary/tracker/observe`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        phrase,
        phi,
        kappa,
        regime,
        basin_coordinates: basinCoordinates
      })
    });

    if (!response.ok) {
      throw new Error(`Vocabulary tracker API error: ${response.status}`);
    }
  } catch (error) {
    logger.error('[VocabularyTrackerAPI] Error observing phrase:', { 
      error: error instanceof Error ? error.message : String(error) 
    });
    throw error;
  }
}

/**
 * Classify phrase into categories
 * Delegates to Python backend
 */
export async function classifyPhraseCategory(text: string): Promise<PhraseCategory> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/vocabulary/tracker/classify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    if (!response.ok) {
      throw new Error(`Vocabulary tracker API error: ${response.status}`);
    }

    const data = await response.json();
    return data.category;
  } catch (error) {
    logger.error('[VocabularyTrackerAPI] Error classifying phrase:', { 
      error: error instanceof Error ? error.message : String(error) 
    });
    throw error;
  }
}

/**
 * Get vocabulary expansion candidates
 * Delegates to Python backend
 */
export async function getCandidates(topK: number = 20): Promise<VocabularyCandidate[]> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/vocabulary/tracker/candidates`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ top_k: topK })
    });

    if (!response.ok) {
      throw new Error(`Vocabulary tracker API error: ${response.status}`);
    }

    const data = await response.json();
    
    return data.candidates.map((candidate: any) => ({
      text: candidate.text,
      type: candidate.type,
      frequency: candidate.frequency,
      avgPhi: candidate.avg_phi,
      maxPhi: candidate.max_phi,
      efficiencyGain: candidate.efficiency_gain,
      reasoning: candidate.reasoning,
      isRealWord: candidate.is_real_word,
      components: candidate.components
    }));
  } catch (error) {
    logger.error('[VocabularyTrackerAPI] Error getting candidates:', { 
      error: error instanceof Error ? error.message : String(error) 
    });
    throw error;
  }
}

/**
 * Get vocabulary tracker statistics
 * Delegates to Python backend
 */
export async function getTrackerStats(): Promise<{
  totalWords: number;
  totalPhrases: number;
  totalSequences: number;
  topWords: Array<{text: string; frequency: number; avgPhi: number; isRealWord: boolean}>;
  topPhrases: Array<{text: string; frequency: number; avgPhi: number}>;
  topSequences: Array<{sequence: string; frequency: number; avgPhi: number}>;
  candidatesReady: number;
}> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/vocabulary/tracker/stats`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' }
    });

    if (!response.ok) {
      throw new Error(`Vocabulary tracker API error: ${response.status}`);
    }

    const data = await response.json();
    
    return {
      totalWords: data.total_words,
      totalPhrases: data.total_phrases,
      totalSequences: data.total_sequences,
      topWords: data.top_words.map((word: any) => ({
        text: word.text,
        frequency: word.frequency,
        avgPhi: word.avg_phi,
        isRealWord: word.is_real_word
      })),
      topPhrases: data.top_phrases.map((phrase: any) => ({
        text: phrase.text,
        frequency: phrase.frequency,
        avgPhi: phrase.avg_phi
      })),
      topSequences: data.top_sequences.map((seq: any) => ({
        sequence: seq.sequence,
        frequency: seq.frequency,
        avgPhi: seq.avg_phi
      })),
      candidatesReady: data.candidates_ready
    };
  } catch (error) {
    logger.error('[VocabularyTrackerAPI] Error getting stats:', { 
      error: error instanceof Error ? error.message : String(error) 
    });
    throw error;
  }
}

/**
 * Get category statistics for kernel learning
 * Delegates to Python backend
 */
export async function getCategoryStats(): Promise<{
  categories: Record<string, { count: number; avgPhi: number; examples: string[] }>;
  bip39Coverage: number;
  totalObservations: number;
}> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/vocabulary/tracker/category-stats`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' }
    });

    if (!response.ok) {
      throw new Error(`Vocabulary tracker API error: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    logger.error('[VocabularyTrackerAPI] Error getting category stats:', { 
      error: error instanceof Error ? error.message : String(error) 
    });
    throw error;
  }
}

/**
 * Export observations for Python tokenizer
 * Delegates to Python backend
 */
export async function exportForTokenizer(): Promise<Array<{
  text: string;
  frequency: number;
  avgPhi: number;
  maxPhi: number;
  type: 'word' | 'phrase' | 'sequence';
  isRealWord: boolean;
  isBip39Word: boolean;
  phraseCategory: PhraseCategory;
}>> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/vocabulary/tracker/export`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' }
    });

    if (!response.ok) {
      throw new Error(`Vocabulary tracker API error: ${response.status}`);
    }

    const data = await response.json();
    
    return data.observations.map((obs: any) => ({
      text: obs.text,
      frequency: obs.frequency,
      avgPhi: obs.avg_phi,
      maxPhi: obs.max_phi,
      type: obs.type,
      isRealWord: obs.is_real_word,
      isBip39Word: obs.is_bip39_word,
      phraseCategory: obs.phrase_category
    }));
  } catch (error) {
    logger.error('[VocabularyTrackerAPI] Error exporting for tokenizer:', { 
      error: error instanceof Error ? error.message : String(error) 
    });
    throw error;
  }
}

/**
 * Wait for PostgreSQL data to finish loading
 * Delegates to Python backend
 */
export async function waitForData(): Promise<void> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/vocabulary/tracker/wait`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' }
    });

    if (!response.ok) {
      throw new Error(`Vocabulary tracker API error: ${response.status}`);
    }
  } catch (error) {
    logger.error('[VocabularyTrackerAPI] Error waiting for data:', { 
      error: error instanceof Error ? error.message : String(error) 
    });
    throw error;
  }
}

/**
 * Check if data has finished loading
 * Delegates to Python backend
 */
export async function isDataLoaded(): Promise<boolean> {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/vocabulary/tracker/loaded`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' }
    });

    if (!response.ok) {
      throw new Error(`Vocabulary tracker API error: ${response.status}`);
    }

    const data = await response.json();
    return data.loaded;
  } catch (error) {
    logger.error('[VocabularyTrackerAPI] Error checking if data loaded:', { 
      error: error instanceof Error ? error.message : String(error) 
    });
    throw error;
  }
}
