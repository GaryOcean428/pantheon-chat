/**
 * Long-Horizon Task Management Hooks
 *
 * Provides React hooks for goal tracking, geodesic efficiency,
 * and geometric error recovery status.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { API_ROUTES, QUERY_KEYS } from '@/api';

// ============================================================================
// Types
// ============================================================================

export interface Goal {
  goal_id: string;
  description: string;
  completed: boolean;
  completion_threshold: number;
  parent_goal_id: string | null;
  subgoal_ids: string[];
  initial_distance: number | null;
  steps_taken: number;
}

export interface GoalProgress {
  progress: number;
  distance_remaining: number;
  stuck: boolean;
  completed: boolean;
  steps_taken: number;
}

export interface GoalSummary {
  total_goals: number;
  completed_goals: number;
  active_goals: number;
  overall_progress: number;
  root_goals: number;
}

export interface EfficiencyStats {
  count: number;
  mean_efficiency: number;
  min_efficiency: number;
  max_efficiency: number;
  std_efficiency: number;
}

export interface EfficiencyByType {
  [operationType: string]: {
    count: number;
    mean_efficiency: number;
  };
}

export interface RecoveryStats {
  total_steps: number;
  trajectory_length: number;
  checkpoint_count: number;
  recovery_count: number;
  avg_checkpoint_score: number;
}

export interface StuckStatus {
  is_stuck: boolean;
  reason: string;
  diagnostics: Record<string, number | string | boolean>;
}

// ============================================================================
// Goal Tracking Hooks
// ============================================================================

export function useGoals() {
  return useQuery({
    queryKey: QUERY_KEYS.longHorizon.goals.list(),
    queryFn: async () => {
      const res = await fetch(API_ROUTES.longHorizon.goals.list);
      if (!res.ok) throw new Error('Failed to fetch goals');
      const data = await res.json();
      return data as {
        success: boolean;
        goals: Goal[];
        summary: GoalSummary;
      };
    },
    refetchInterval: 5000,
  });
}

export function useCreateGoal() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (params: {
      description: string;
      basin_coords: number[];
      parent_id?: string;
      completion_threshold?: number;
    }) => {
      const res = await fetch(API_ROUTES.longHorizon.goals.create, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      if (!res.ok) throw new Error('Failed to create goal');
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.longHorizon.goals.list() });
    },
  });
}

export function useTrackProgress() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (current_basin: number[]) => {
      const res = await fetch(API_ROUTES.longHorizon.goals.progress, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ current_basin }),
      });
      if (!res.ok) throw new Error('Failed to track progress');
      return res.json() as Promise<{
        success: boolean;
        progress: Record<string, GoalProgress>;
        summary: GoalSummary;
      }>;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.longHorizon.goals.list() });
    },
  });
}

// ============================================================================
// Efficiency Tracking Hooks
// ============================================================================

export function useEfficiencyStats() {
  return useQuery({
    queryKey: QUERY_KEYS.longHorizon.efficiency.stats(),
    queryFn: async () => {
      const res = await fetch(API_ROUTES.longHorizon.efficiency.stats);
      if (!res.ok) throw new Error('Failed to fetch efficiency stats');
      const data = await res.json();
      return data as {
        success: boolean;
        stats: EfficiencyStats;
        by_operation_type: EfficiencyByType;
      };
    },
    refetchInterval: 10000,
  });
}

export function useEfficiencyDegradation() {
  return useQuery({
    queryKey: QUERY_KEYS.longHorizon.efficiency.degradation(),
    queryFn: async () => {
      const res = await fetch(API_ROUTES.longHorizon.efficiency.degradation);
      if (!res.ok) throw new Error('Failed to check degradation');
      const data = await res.json();
      return data as {
        success: boolean;
        degradation: {
          is_degraded: boolean;
          current_efficiency: number;
          baseline_efficiency: number;
          drop_percentage: number;
        } | null;
      };
    },
    refetchInterval: 15000,
  });
}

// ============================================================================
// Recovery Status Hooks
// ============================================================================

export function useRecoveryStatus() {
  return useQuery({
    queryKey: QUERY_KEYS.longHorizon.recovery.status(),
    queryFn: async () => {
      const res = await fetch(API_ROUTES.longHorizon.recovery.status);
      if (!res.ok) throw new Error('Failed to fetch recovery status');
      const data = await res.json();
      return data as {
        success: boolean;
        stats: RecoveryStats;
        recent_recoveries: Array<{
          stuck_reason: string;
          steps_back: number;
          timestamp: number;
        }>;
      };
    },
    refetchInterval: 5000,
  });
}

export function useStuckCheck() {
  return useQuery({
    queryKey: QUERY_KEYS.longHorizon.recovery.check(),
    queryFn: async () => {
      const res = await fetch(API_ROUTES.longHorizon.recovery.check);
      if (!res.ok) throw new Error('Failed to check stuck status');
      const data = await res.json();
      return data as {
        success: boolean;
        is_stuck: boolean;
        reason: string;
        diagnostics: Record<string, number | string | boolean>;
      };
    },
    refetchInterval: 3000,
  });
}

export function useTriggerRecovery() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async () => {
      const res = await fetch(API_ROUTES.longHorizon.recovery.recover, {
        method: 'POST',
      });
      if (!res.ok) throw new Error('Failed to trigger recovery');
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.longHorizon.recovery.status() });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.longHorizon.recovery.check() });
    },
  });
}

// ============================================================================
// Combined Hook for Dashboard Display
// ============================================================================

export function useLongHorizonSummary() {
  const goals = useGoals();
  const efficiency = useEfficiencyStats();
  const recovery = useRecoveryStatus();
  const stuckCheck = useStuckCheck();

  const isLoading = goals.isLoading || efficiency.isLoading || recovery.isLoading;
  const hasError = goals.isError || efficiency.isError || recovery.isError;

  return {
    // Goal summary
    goalCount: goals.data?.summary?.total_goals ?? 0,
    completedGoals: goals.data?.summary?.completed_goals ?? 0,
    activeGoals: goals.data?.summary?.active_goals ?? 0,
    overallProgress: goals.data?.summary?.overall_progress ?? 0,

    // Efficiency summary
    meanEfficiency: efficiency.data?.stats?.mean_efficiency ?? 0,
    efficiencyCount: efficiency.data?.stats?.count ?? 0,

    // Recovery summary
    isStuck: stuckCheck.data?.is_stuck ?? false,
    stuckReason: stuckCheck.data?.reason ?? '',
    recoveryCount: recovery.data?.stats?.recovery_count ?? 0,
    checkpointCount: recovery.data?.stats?.checkpoint_count ?? 0,

    // Meta
    isLoading,
    hasError,

    // Raw data for detailed views
    goals: goals.data,
    efficiency: efficiency.data,
    recovery: recovery.data,
    stuckCheck: stuckCheck.data,
  };
}
