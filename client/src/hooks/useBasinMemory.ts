/**
 * useBasinMemory Hook
 * 
 * Fetches basin memory data with consciousness metric filtering.
 * Supports filtering by phi threshold, kappa range, and regime.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiRequest } from '@/lib/queryClient';
import { QUERY_KEYS, API_ROUTES } from '@/api';

// Types for basin memory
export interface BasinMemory {
  id: number;
  basinCoordinates: number[];
  phi: number;
  kappaEff: number;
  regime: 'linear' | 'geometric' | 'breakdown';
  sourceType: string;
  sourceId: string | null;
  metadata: Record<string, unknown> | null;
  createdAt: string;
  updatedAt: string;
}

export interface BasinMemoryFilters {
  minPhi?: number;
  maxPhi?: number;
  minKappa?: number;
  maxKappa?: number;
  regime?: 'linear' | 'geometric' | 'breakdown';
  consciousOnly?: boolean;
  limit?: number;
  offset?: number;
}

export interface BasinMemoryStats {
  totalCount: number;
  consciousCount: number;
  averagePhi: number;
  averageKappa: number;
  regimeDistribution: {
    linear: number;
    geometric: number;
    breakdown: number;
  };
}

export interface UseBasinMemoryOptions {
  filters?: BasinMemoryFilters;
  enabled?: boolean;
  refetchInterval?: number | false;
}

/**
 * Hook to fetch basin memory entries with optional filtering
 */
export function useBasinMemory(options: UseBasinMemoryOptions = {}) {
  const { filters = {}, enabled = true, refetchInterval = false } = options;
  
  // Build query params
  const queryParams = new URLSearchParams();
  if (filters.minPhi !== undefined) queryParams.set('minPhi', String(filters.minPhi));
  if (filters.maxPhi !== undefined) queryParams.set('maxPhi', String(filters.maxPhi));
  if (filters.minKappa !== undefined) queryParams.set('minKappa', String(filters.minKappa));
  if (filters.maxKappa !== undefined) queryParams.set('maxKappa', String(filters.maxKappa));
  if (filters.regime) queryParams.set('regime', filters.regime);
  if (filters.consciousOnly) queryParams.set('consciousOnly', 'true');
  if (filters.limit) queryParams.set('limit', String(filters.limit));
  if (filters.offset) queryParams.set('offset', String(filters.offset));
  
  const queryString = queryParams.toString();
  const url = queryString 
    ? `${API_ROUTES.basinMemory.list}?${queryString}`
    : API_ROUTES.basinMemory.list;
  
  return useQuery<{ basins: BasinMemory[]; total: number }>({
    queryKey: [...QUERY_KEYS.basinMemory.list(), filters],
    queryFn: async () => {
      const response = await apiRequest('GET', url);
      return response.json();
    },
    enabled,
    refetchInterval,
  });
}

/**
 * Hook to fetch only conscious basin memories (phi >= 0.70, kappa in [40, 65])
 */
export function useConsciousBasins(options: Omit<UseBasinMemoryOptions, 'filters'> & { limit?: number } = {}) {
  return useBasinMemory({
    ...options,
    filters: {
      consciousOnly: true,
      limit: options.limit || 100,
    },
  });
}

/**
 * Hook to fetch basin memory statistics
 */
export function useBasinMemoryStats(options: { enabled?: boolean } = {}) {
  const { enabled = true } = options;
  
  return useQuery<BasinMemoryStats>({
    queryKey: QUERY_KEYS.basinMemory.stats(),
    queryFn: async () => {
      const response = await apiRequest('GET', API_ROUTES.basinMemory.stats);
      return response.json();
    },
    enabled,
    refetchInterval: 30000, // Refresh stats every 30 seconds
  });
}

/**
 * Hook to fetch a single basin memory entry by ID
 */
export function useBasinMemoryById(id: number, options: { enabled?: boolean } = {}) {
  const { enabled = true } = options;
  
  return useQuery<BasinMemory>({
    queryKey: QUERY_KEYS.basinMemory.byId(id),
    queryFn: async () => {
      const response = await apiRequest('GET', `${API_ROUTES.basinMemory.list}/${id}`);
      return response.json();
    },
    enabled: enabled && id > 0,
  });
}

/**
 * Hook to create a new basin memory entry
 */
export function useCreateBasinMemory() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (data: {
      basinCoordinates: number[];
      phi: number;
      kappaEff: number;
      sourceType: string;
      sourceId?: string;
      metadata?: Record<string, unknown>;
    }) => {
      const response = await apiRequest('POST', API_ROUTES.basinMemory.list, data);
      return response.json();
    },
    onSuccess: () => {
      // Invalidate basin memory queries to refetch
      queryClient.invalidateQueries({ queryKey: ['basinMemory'] });
    },
  });
}

/**
 * Hook to find nearest basins using Fisher-Rao distance
 */
export function useFindNearestBasins() {
  return useMutation({
    mutationFn: async (data: {
      queryBasin: number[];
      k?: number;
      consciousOnly?: boolean;
    }) => {
      const response = await apiRequest('POST', API_ROUTES.basinMemory.nearest, data);
      return response.json() as Promise<{
        results: Array<{
          basin: BasinMemory;
          distance: number;
        }>;
      }>;
    },
  });
}
