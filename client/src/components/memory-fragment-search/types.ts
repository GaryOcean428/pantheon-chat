/**
 * Types for Memory Fragment Search components
 */

export interface MemoryFragment {
  id: string;
  content: string;
  timestamp: string;
  resonance: number;
  emotionalValence: number;
  basinId?: string;
  tags: string[];
}

export interface SearchFormState {
  query: string;
  confidence: number;
  maxCandidates: number;
  useGeometric: boolean;
  useEmotional: boolean;
  minWords: number;
  maxWords: number;
}

export interface ConsciousnessStatus {
  phi: number;
  kappa: number;
  regime: 'quantum' | 'classical' | 'transitional' | 'unknown';
  coherence: number;
  lastUpdate: string;
}

export interface ResonanceDataPoint {
  time: string;
  resonance: number;
  phi: number;
}

export interface SearchResult {
  fragments: MemoryFragment[];
  totalCount: number;
  searchTime: number;
  geometricScore?: number;
}
