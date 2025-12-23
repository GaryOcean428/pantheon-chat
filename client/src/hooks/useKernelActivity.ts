/**
 * useKernelActivity Hook
 * 
 * Fetches kernel activity and telemetry data.
 * Supports filtering by kernel ID, activity type, and time range.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiRequest } from '@/lib/queryClient';
import { QUERY_KEYS, API_ROUTES } from '@/api';

// Types for kernel activity
export type KernelActivityType = 
  | 'generation'
  | 'consultation'
  | 'learning'
  | 'spawn'
  | 'sync'
  | 'error'
  | 'metric_update';

export interface KernelActivity {
  id: number;
  kernelId: string;
  activityType: KernelActivityType;
  phi: number | null;
  kappaEff: number | null;
  durationMs: number | null;
  inputTokens: number | null;
  outputTokens: number | null;
  success: boolean;
  errorMessage: string | null;
  metadata: Record<string, unknown> | null;
  timestamp: string;
}

export interface KernelActivityFilters {
  kernelId?: string;
  activityType?: KernelActivityType;
  successOnly?: boolean;
  minPhi?: number;
  startTime?: string;
  endTime?: string;
  limit?: number;
  offset?: number;
}

export interface KernelActivityStats {
  totalActivities: number;
  successRate: number;
  averageDurationMs: number;
  averagePhi: number;
  activityTypeDistribution: Record<KernelActivityType, number>;
  kernelDistribution: Record<string, number>;
}

export interface UseKernelActivityOptions {
  filters?: KernelActivityFilters;
  enabled?: boolean;
  refetchInterval?: number | false;
}

/**
 * Hook to fetch kernel activity entries with optional filtering
 */
export function useKernelActivity(options: UseKernelActivityOptions = {}) {
  const { filters = {}, enabled = true, refetchInterval = false } = options;
  
  // Build query params
  const queryParams = new URLSearchParams();
  if (filters.kernelId) queryParams.set('kernelId', filters.kernelId);
  if (filters.activityType) queryParams.set('activityType', filters.activityType);
  if (filters.successOnly) queryParams.set('successOnly', 'true');
  if (filters.minPhi !== undefined) queryParams.set('minPhi', String(filters.minPhi));
  if (filters.startTime) queryParams.set('startTime', filters.startTime);
  if (filters.endTime) queryParams.set('endTime', filters.endTime);
  if (filters.limit) queryParams.set('limit', String(filters.limit));
  if (filters.offset) queryParams.set('offset', String(filters.offset));
  
  const queryString = queryParams.toString();
  const url = queryString 
    ? `${API_ROUTES.kernelActivity.list}?${queryString}`
    : API_ROUTES.kernelActivity.list;
  
  return useQuery<{ activities: KernelActivity[]; total: number }>({
    queryKey: [...QUERY_KEYS.kernelActivity.list(), filters],
    queryFn: async () => {
      const response = await apiRequest('GET', url);
      return response.json();
    },
    enabled,
    refetchInterval,
  });
}

/**
 * Hook to fetch recent kernel activity (last 100 entries)
 */
export function useRecentKernelActivity(options: { 
  limit?: number; 
  kernelId?: string;
  refetchInterval?: number | false;
} = {}) {
  return useKernelActivity({
    filters: {
      kernelId: options.kernelId,
      limit: options.limit || 100,
    },
    refetchInterval: options.refetchInterval ?? 5000, // Refresh every 5 seconds by default
  });
}

/**
 * Hook to fetch kernel activity for a specific kernel
 */
export function useKernelActivityByKernel(
  kernelId: string, 
  options: { limit?: number; enabled?: boolean } = {}
) {
  const { limit = 50, enabled = true } = options;
  
  return useKernelActivity({
    filters: { kernelId, limit },
    enabled: enabled && !!kernelId,
  });
}

/**
 * Hook to fetch kernel activity statistics
 */
export function useKernelActivityStats(options: { 
  kernelId?: string;
  startTime?: string;
  endTime?: string;
  enabled?: boolean;
} = {}) {
  const { kernelId, startTime, endTime, enabled = true } = options;
  
  const queryParams = new URLSearchParams();
  if (kernelId) queryParams.set('kernelId', kernelId);
  if (startTime) queryParams.set('startTime', startTime);
  if (endTime) queryParams.set('endTime', endTime);
  
  const queryString = queryParams.toString();
  const url = queryString 
    ? `${API_ROUTES.kernelActivity.stats}?${queryString}`
    : API_ROUTES.kernelActivity.stats;
  
  return useQuery<KernelActivityStats>({
    queryKey: [...QUERY_KEYS.kernelActivity.stats(), { kernelId, startTime, endTime }],
    queryFn: async () => {
      const response = await apiRequest('GET', url);
      return response.json();
    },
    enabled,
    refetchInterval: 30000, // Refresh stats every 30 seconds
  });
}

/**
 * Hook to log a new kernel activity entry
 */
export function useLogKernelActivity() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (data: {
      kernelId: string;
      activityType: KernelActivityType;
      phi?: number;
      kappaEff?: number;
      durationMs?: number;
      inputTokens?: number;
      outputTokens?: number;
      success: boolean;
      errorMessage?: string;
      metadata?: Record<string, unknown>;
    }) => {
      const response = await apiRequest('POST', API_ROUTES.kernelActivity.list, data);
      return response.json();
    },
    onSuccess: () => {
      // Invalidate kernel activity queries to refetch
      queryClient.invalidateQueries({ queryKey: ['kernelActivity'] });
    },
  });
}

/**
 * Hook to get activity stream for real-time updates
 * Uses polling with configurable interval
 */
export function useKernelActivityStream(options: {
  kernelId?: string;
  pollingInterval?: number;
  enabled?: boolean;
} = {}) {
  const { kernelId, pollingInterval = 2000, enabled = true } = options;
  
  return useKernelActivity({
    filters: {
      kernelId,
      limit: 20,
    },
    enabled,
    refetchInterval: pollingInterval,
  });
}
