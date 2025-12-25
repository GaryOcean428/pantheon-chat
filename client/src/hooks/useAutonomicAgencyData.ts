/**
 * useAutonomicAgencyData
 * 
 * Custom hook for fetching autonomic agency status and managing mutations.
 * Extracts data fetching logic from AutonomicAgencyPanel component.
 */

import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { API_ROUTES, QUERY_KEYS } from "@/api";
import { useToast } from "@/hooks/use-toast";
import type { AgencyStatus } from "@/types";
import { ACTION_LABELS } from "@/types";

export interface UseAutonomicAgencyDataReturn {
  /** Current agency status */
  status: AgencyStatus | undefined;
  /** Whether the initial fetch is loading */
  isLoading: boolean;
  /** Whether there was an error fetching */
  isError: boolean;
  /** Refetch the status */
  refetch: () => void;
  /** Start the agency controller */
  start: () => void;
  /** Stop the agency controller */
  stop: () => void;
  /** Force a specific intervention action */
  forceIntervention: (action: string) => void;
  /** Whether start mutation is pending */
  isStartPending: boolean;
  /** Whether stop mutation is pending */
  isStopPending: boolean;
  /** Whether force mutation is pending */
  isForcePending: boolean;
  /** Calculated exploration percentage */
  explorationPercent: number;
}

export function useAutonomicAgencyData(): UseAutonomicAgencyDataReturn {
  const { toast } = useToast();

  const { data: status, isLoading, isError, refetch } = useQuery<AgencyStatus>({
    queryKey: QUERY_KEYS.qig.autonomicAgencyStatus(),
    queryFn: async () => {
      const res = await fetch(API_ROUTES.qig.autonomic.agencyStatus);
      if (!res.ok) throw new Error("Failed to fetch agency status");
      return res.json();
    },
    refetchInterval: 5000,
    retry: 3,
    retryDelay: 2000,
  });

  const startMutation = useMutation({
    mutationFn: async () => {
      return apiRequest("POST", API_ROUTES.qig.autonomic.agencyStart);
    },
    onSuccess: () => {
      toast({ title: "Agency Started", description: "Autonomous controller is now running" });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.qig.autonomicAgencyStatus() });
    },
    onError: (error: Error) => {
      toast({ title: "Failed to start", description: error.message, variant: "destructive" });
    },
  });

  const stopMutation = useMutation({
    mutationFn: async () => {
      return apiRequest("POST", API_ROUTES.qig.autonomic.agencyStop);
    },
    onSuccess: () => {
      toast({ title: "Agency Stopped", description: "Autonomous controller has been paused" });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.qig.autonomicAgencyStatus() });
    },
    onError: (error: Error) => {
      toast({ title: "Failed to stop", description: error.message, variant: "destructive" });
    },
  });

  const forceMutation = useMutation({
    mutationFn: async (action: string) => {
      return apiRequest("POST", API_ROUTES.qig.autonomic.agencyForce, { action });
    },
    onSuccess: (_, action) => {
      toast({ 
        title: "Intervention Triggered", 
        description: `Forced ${ACTION_LABELS[action] || action}` 
      });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.qig.autonomicAgencyStatus() });
    },
    onError: (error: Error) => {
      toast({ title: "Intervention failed", description: error.message, variant: "destructive" });
    },
  });

  const explorationPercent = Math.round((1 - (status?.epsilon ?? 1)) * 100);

  return {
    status,
    isLoading,
    isError,
    refetch,
    start: () => startMutation.mutate(),
    stop: () => stopMutation.mutate(),
    forceIntervention: (action: string) => forceMutation.mutate(action),
    isStartPending: startMutation.isPending,
    isStopPending: stopMutation.isPending,
    isForcePending: forceMutation.isPending,
    explorationPercent,
  };
}
