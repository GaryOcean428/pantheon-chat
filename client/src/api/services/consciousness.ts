/**
 * Consciousness Service
 * 
 * Type-safe API functions for consciousness state operations.
 */

import { get } from '../client';
import { API_ROUTES } from '../routes';
import type { ConsciousnessAPIResponse } from '@/types';

// Re-export types for backward compatibility
export type { ConsciousnessState, ConsciousnessAPIResponse } from '@/types';

export async function getConsciousnessState(): Promise<ConsciousnessAPIResponse> {
  return get<ConsciousnessAPIResponse>(API_ROUTES.consciousness.state);
}
