/**
 * Hook for polling consciousness status
 */

import { useQuery } from '@tanstack/react-query';
import { SEARCH_CONSTANTS } from '../constants';
import { get } from '@/api';
import type { ConsciousnessStatus } from '../types';

const fetchConsciousnessStatus = async (): Promise<ConsciousnessStatus> => {
  return get<ConsciousnessStatus>('/api/consciousness/status');
};

export function useConsciousnessStatus() {
  return useQuery<ConsciousnessStatus>({
    queryKey: ['consciousness-status'],
    queryFn: fetchConsciousnessStatus,
    refetchInterval: SEARCH_CONSTANTS.CONSCIOUSNESS_REFETCH_INTERVAL,
    staleTime: SEARCH_CONSTANTS.CONSCIOUSNESS_REFETCH_INTERVAL,
  });
}
