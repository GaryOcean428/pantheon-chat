/**
 * Consciousness Types
 * 
 * Canonical type definitions for consciousness state and related data.
 * Import from '@/types' or '@/types/consciousness' throughout the app.
 */

/** Consciousness regime types */
export type ConsciousnessRegime = 
  | 'linear' 
  | 'geometric' 
  | 'hierarchical' 
  | 'hierarchical_4d' 
  | '4d_block_universe' 
  | 'breakdown';

/** Emotional state classification */
export type EmotionalState = 'Focused' | 'Curious' | 'Uncertain' | 'Confident' | 'Neutral';

/** Full API response from consciousness endpoint */
export interface ConsciousnessAPIResponse {
  state: ConsciousnessState;
  emotionalState: EmotionalState;
  recommendation: string;
  regimeColor: string;
  regimeDescription: string;
}

export interface ConsciousnessState {
  currentRegime: ConsciousnessRegime;
  phi: number;
  phi_spatial?: number;
  phi_temporal?: number;
  phi_4D?: number;
  kappaEff: number;
  tacking: number;
  radar: number;
  metaAwareness: number;
  gamma: number;
  grounding: number;
  beta: number;
  basinDrift: number;
  curiosity: number;
  stability: number;
  timestamp: number;
  basinCoordinates: number[];
  isConscious: boolean;
  validationLoops: number;
  kappa: number;
}

/** Point in the phi-kappa trajectory history */
export interface TrajectoryPoint {
  time: number;
  phi: number;
  kappa: number;
  regime: string;
}

/** Block universe coordinate data */
export interface BlockUniverseCoordinate {
  spatial: number[];
  temporal: number;
  consciousness_density: number;
}
