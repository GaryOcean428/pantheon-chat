/**
 * Hook for polling consciousness status
 */

import { useQuery } from '@tanstack/react-query';
import { SEARCH_CONSTANTS } from '../constants';
import type { ConsciousnessStatus } from '../types';

const fetchConsciousnessStatus = async (): Promise<ConsciousnessStatus> => {
  const response = await fetch('/api/consciousness/status');
  if (!response.ok) {
    throw new Error('Failed to fetch consciousness status');
  }
  return response.json();
};

export function useConsciousnessStatus() {
  return useQuery<ConsciousnessStatus>({
    queryKey: ['consciousness-status'],
    queryFn: fetchConsciousnessStatus,
    refetchInterval: SEARCH_CONSTANTS.CONSCIOUSNESS_REFETCH_INTERVAL,
    staleTime: SEARCH_CONSTANTS.CONSCIOUSNESS_REFETCH_INTERVAL,
  });
}
