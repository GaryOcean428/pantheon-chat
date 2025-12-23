/**
 * React Hook for Kernel Activity Stream
 * 
 * Fetches all inter-god communications, debates, discoveries,
 * and kernel activity from the Pantheon chat system.
 * 
 * Provides both polling-based (useKernelActivity) and 
 * WebSocket-based (useKernelActivityWebSocket) options.
 */

import { useQuery } from '@tanstack/react-query';
import { QUERY_KEYS } from '@/api';

// Re-export WebSocket hook
export { useKernelActivityWebSocket } from './useKernelActivityWebSocket';

export type ActivityType = 
  | 'insight'
  | 'praise'
  | 'challenge'
  | 'question'
  | 'warning'
  | 'discovery'
  | 'challenge_response'
  | 'spawn_proposal'
  | 'spawn_vote'
  | 'debate_start'
  | 'debate_resolved';

export interface KernelActivityItem {
  id: string;
  type: ActivityType;
  from: string;
  to: string;
  content: string;
  timestamp: string;
  read: boolean;
  responded: boolean;
  metadata?: {
    debate_id?: string;
    event_type?: string;
    autonomic?: boolean;
    metrics?: Record<string, number>;
    knowledge?: Record<string, unknown>;
    resolution?: {
      winner?: string;
      reasoning?: string;
    };
    [key: string]: unknown;
  };
}

export interface KernelActivityResponse {
  activity: KernelActivityItem[];
  debates: {
    active: Array<{
      id: string;
      topic: string;
      initiator: string;
      opponent: string;
      arguments: Array<{
        god: string;
        argument: string;
        timestamp: string;
      }>;
      status: string;
      started_at: string;
    }>;
    resolved: Array<{
      id: string;
      topic: string;
      winner: string;
      resolution: {
        reasoning: string;
      };
    }>;
  };
  status: {
    total_messages: number;
    active_debates: number;
    resolved_debates: number;
    knowledge_transfers: number;
  };
}

/**
 * Polling-based kernel activity hook.
 * Use this for simple cases or as fallback.
 * 
 * @deprecated Prefer useKernelActivityWebSocket for real-time updates
 */
export function useKernelActivity(limit: number = 50) {
  return useQuery<KernelActivityResponse>({
    queryKey: [...QUERY_KEYS.olympus.activity(), limit],
    staleTime: 5000,
    refetchInterval: 5000, // Poll every 5 seconds for live updates
  });
}

/**
 * Polling-based kernel activity stream hook.
 * 
 * @deprecated Prefer useKernelActivityWebSocket for real-time updates
 */
export function useKernelActivityStream(limit: number = 100) {
  return useQuery<KernelActivityItem[]>({
    queryKey: [...QUERY_KEYS.olympus.activity(), 'stream', limit],
    queryFn: async () => {
      const res = await fetch(`/api/olympus/pantheon/activity?limit=${limit}`, {
        credentials: 'include',
      });
      if (!res.ok) throw new Error('Failed to fetch activity');
      const data = await res.json();
      return data.activity || [];
    },
    staleTime: 5000,
    refetchInterval: 5000,
  });
}
