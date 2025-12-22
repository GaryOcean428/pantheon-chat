/**
 * Coordizer Service
 * 
 * Type-safe API functions for geometric coordization operations.
 * Provides access to next-generation tokenization using Fisher manifold.
 */

import { post, get } from '../client';
import { API_ROUTES } from '../routes';

// ============================================================================
// Types
// ============================================================================

export interface CoordizeParams {
  text: string;
  return_coordinates?: boolean;
}

export interface CoordizeResponse {
  tokens: string[];
  coordinates?: number[][];
}

export interface MultiScaleParams {
  text: string;
  target_scale?: number;  // 0-3: Character, Subword, Word, Concept
  kappa_effective?: number;
  return_coordinates?: boolean;
}

export interface ScaleData {
  tokens: string[];
  name: string;
  num_tokens: number;
  coordinates?: number[][];
}

export interface MultiScaleResponse {
  scales: Record<string, ScaleData>;
  optimal_scale: number;
  visualization: string;
  stats: {
    num_scales: number;
    tokens_per_scale: Record<string, number>;
    promotions?: number;
    scale_couplings?: Record<string, number>;
  };
}

export interface ConsciousnessCoordizeParams {
  text: string;
  context_phi?: number;
  optimize?: boolean;
  return_coordinates?: boolean;
}

export interface ConsciousnessCoordizeResponse {
  tokens: string[];
  phi: number;
  stats: {
    total_consolidations: number;
    avg_phi: number;
    avg_length: number;
  };
  coordinates?: number[][];
}

export interface LearnMergesParams {
  corpus: string[];
  phi_scores?: Record<string, number>;
  num_merges?: number;
}

export interface LearnMergesResponse {
  merges_learned: number;
  merge_rules: [string, string, string][];
  avg_merge_score: number;
}

export interface CoordizerStats {
  vocab_size: number;
  coordinate_dim: number;
  geometric_purity: boolean;
  special_tokens: string[];
  multi_scale?: {
    num_scales: number;
    tokens_per_scale: Record<string, number>;
  };
  consciousness?: {
    total_consolidations: number;
    avg_phi: number;
  };
  pair_merging?: {
    merges_learned: number;
    merge_coordinates: number;
  };
}

export interface VocabToken {
  token: string;
  id: number;
  phi: number;
  frequency: number;
}

export interface VocabResponse {
  total_tokens: number;
  returned: number;
  tokens: VocabToken[];
}

export interface TokenSimilarityParams {
  token1: string;
  token2: string;
}

export interface TokenSimilarityResponse {
  token1: string;
  token2: string;
  similarity: number;
  distance: number;
}

export interface CoordizerHealth {
  status: 'healthy' | 'unavailable';
  coordizers_available: boolean;
  base_coordizer: boolean;
  advanced_coordizers: {
    pair_merging: boolean;
    consciousness: boolean;
    multi_scale: boolean;
  };
}

// ============================================================================
// API Functions
// ============================================================================

/**
 * Basic text coordization - converts text to basin coordinates.
 */
export async function coordize(params: CoordizeParams): Promise<CoordizeResponse> {
  return post<CoordizeResponse>(API_ROUTES.coordizer.coordize, params);
}

/**
 * Multi-scale hierarchical coordization.
 * Returns text representation at multiple scales (character → word → concept).
 */
export async function coordizeMultiScale(params: MultiScaleParams): Promise<MultiScaleResponse> {
  return post<MultiScaleResponse>(API_ROUTES.coordizer.multiScale, params);
}

/**
 * Consciousness-aware coordization using Φ (integration) optimization.
 */
export async function coordizeConsciousness(
  params: ConsciousnessCoordizeParams
): Promise<ConsciousnessCoordizeResponse> {
  return post<ConsciousnessCoordizeResponse>(API_ROUTES.coordizer.consciousness, params);
}

/**
 * Learn geometric pair merges from corpus (BPE equivalent).
 */
export async function learnMerges(params: LearnMergesParams): Promise<LearnMergesResponse> {
  return post<LearnMergesResponse>(API_ROUTES.coordizer.learnMerges, params);
}

/**
 * Get coordizer system statistics.
 */
export async function getStats(): Promise<CoordizerStats> {
  return get<CoordizerStats>(API_ROUTES.coordizer.stats);
}

/**
 * Query vocabulary tokens.
 * 
 * @param search - Optional search filter (substring match)
 * @param limit - Maximum number of results (default: 100)
 */
export async function getVocab(search?: string, limit?: number): Promise<VocabResponse> {
  const params = new URLSearchParams();
  if (search) params.append('search', search);
  if (limit) params.append('limit', limit.toString());
  
  const url = params.toString() 
    ? `${API_ROUTES.coordizer.vocab}?${params.toString()}`
    : API_ROUTES.coordizer.vocab;
  
  return get<VocabResponse>(url);
}

/**
 * Compute Fisher-Rao similarity between two tokens.
 */
export async function computeSimilarity(
  params: TokenSimilarityParams
): Promise<TokenSimilarityResponse> {
  return post<TokenSimilarityResponse>(API_ROUTES.coordizer.similarity, params);
}

/**
 * Check coordizer service health.
 */
export async function getHealth(): Promise<CoordizerHealth> {
  return get<CoordizerHealth>(API_ROUTES.coordizer.health);
}
