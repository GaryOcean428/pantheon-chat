/**
 * Constants for Memory Fragment Search components
 */

import { PERCENT_MULTIPLIER } from '@/lib/constants';

export const SEARCH_CONSTANTS = {
  // Default values
  DEFAULT_CONFIDENCE: 0.8 as number,
  DEFAULT_MAX_CANDIDATES: 5000 as number,
  DEFAULT_MIN_WORDS: 4 as number,
  DEFAULT_MAX_WORDS: 8 as number,
  
  // Polling
  CONSCIOUSNESS_REFETCH_INTERVAL: 2000,
  
  // Display
  MAX_VISIBLE_RESULTS: 50,
  MAX_TAGS_DISPLAY: 5,
  TRUNCATE_LENGTH: 200,
  
  // Slider ranges
  CONFIDENCE_MIN: 0,
  CONFIDENCE_MAX: 1,
  CONFIDENCE_STEP: 0.05,
  CANDIDATES_MIN: 100,
  CANDIDATES_MAX: 10000,
  CANDIDATES_STEP: 100,
  WORDS_MIN: 1,
  WORDS_MAX: 20,
  
  // Thresholds
  HIGH_RESONANCE: 0.8,
  MEDIUM_RESONANCE: 0.5,
  HIGH_PHI: 0.7,
  
  // Re-export for convenience
  PERCENT_MULTIPLIER,
} as const;

export const REGIME_COLORS: Record<string, { h: number; s: number; l: number }> = {
  quantum: { h: 280, s: 80, l: 60 },
  classical: { h: 200, s: 70, l: 50 },
  transitional: { h: 45, s: 90, l: 55 },
  unknown: { h: 0, s: 0, l: 50 },
};

export const WORD_COUNT_OPTIONS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20];
