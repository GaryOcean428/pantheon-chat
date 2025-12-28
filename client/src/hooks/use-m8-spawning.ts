/**
 * React Hooks for M8 Kernel Spawning Protocol
 * 
 * Enables dynamic spawning of new specialized god-kernels through
 * geometric consensus voting system.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { API_ROUTES, QUERY_KEYS, deleteKernel, cannibalizeKernel, mergeKernels, autoCannibalize, autoMerge } from '@/api';
import { useAuth } from '@/hooks/useAuth';
import {
  getM8Client,
  type M8Status,
  type SpawnProposal,
  type SpawnedKernel,
  type CreateProposalRequest,
  type CreateProposalResponse,
  type VoteResponse,
  type SpawnResponse,
  type SpawnDirectRequest,
  type SpawnDirectResponse,
  type ListProposalsResponse,
  type ListKernelsResponse,
  type ProposalStatus,
} from '@/lib/m8-kernel-spawning';

export type KernelStatus = 'active' | 'idle' | 'breeding' | 'dormant' | 'dead' | 'shadow';

export interface PostgresKernel {
  kernel_id: string;
  god_name: string;
  domain: string;
  status: KernelStatus;
  primitive_root: number | null;
  basin_coordinates: number[] | null;
  parent_kernels: string[];
  spawned_by: string;
  spawn_reason: string;
  spawn_rationale: string;
  position_rationale: string | null;
  affinity_strength: number;
  entropy_threshold: number;
  spawned_at: string;
  last_active_at: string | null;
  spawned_during_war_id: string | null;
  phi: number;
  kappa: number;
  regime: string | null;
  generation: number;
  success_count: number;
  failure_count: number;
  reputation: string;
  element_group: string | null;
  ecological_niche: string | null;
  target_function: string | null;
  valence: number | null;
  breeding_target: string | null;
  merge_candidate: boolean;
  split_candidate: boolean;
  metadata: Record<string, unknown> | null;
}

export interface PostgresKernelsResponse {
  kernels: PostgresKernel[];
  total: number;
  live_count: number;
  cap: number;
  available: number;
  status_filter?: string[];
}

const M8_KEYS = {
  status: ['m8', 'status'] as const,
  proposals: ['m8', 'proposals'] as const,
  proposal: ['m8', 'proposal'] as const,
  kernels: ['m8', 'kernels'] as const,
  kernel: ['m8', 'kernel'] as const,
  warHistory: ['m8', 'warHistory'] as const,
  activeWar: ['m8', 'activeWar'] as const,
};

export type WarMode = 'FLOW' | 'DEEP_FOCUS' | 'INSIGHT_HUNT' | 'BLITZKRIEG' | 'SIEGE' | 'HUNT';
export type WarOutcome = 'success' | 'partial_success' | 'failure' | 'aborted';
export type WarStatus = 'active' | 'completed' | 'aborted';

export interface WarHistoryRecord {
  id: string;
  mode: WarMode;
  target: string;
  status: WarStatus;
  outcome?: WarOutcome | null;
  strategy?: string | null;
  godsEngaged?: string[] | null;
  declaredAt: string;
  endedAt?: string | null;
  convergenceScore?: number | null;
  phrasesTestedDuringWar?: number;
  discoveriesDuringWar?: number;
  kernelsSpawnedDuringWar?: number;
  metadata?: Record<string, unknown> | null;
}

export function useM8Status() {
  const { isAuthenticated } = useAuth();
  return useQuery<M8Status>({
    queryKey: M8_KEYS.status,
    queryFn: () => getM8Client().getStatus(),
    staleTime: 30000,
    refetchInterval: 60000,
    enabled: isAuthenticated,
  });
}

export function useListProposals(status?: ProposalStatus) {
  const { isAuthenticated } = useAuth();
  return useQuery<ListProposalsResponse>({
    queryKey: [...M8_KEYS.proposals, status],
    queryFn: () => getM8Client().listProposals(status),
    staleTime: 30000,
    enabled: isAuthenticated,
  });
}

export function useGetProposal(proposalId: string | null) {
  return useQuery<SpawnProposal>({
    queryKey: [...M8_KEYS.proposal, proposalId],
    queryFn: () => {
      if (!proposalId) throw new Error('Proposal ID required');
      return getM8Client().getProposal(proposalId);
    },
    enabled: !!proposalId,
    staleTime: 30000,
  });
}

export function useListSpawnedKernels() {
  const { isAuthenticated } = useAuth();
  return useQuery<PostgresKernelsResponse>({
    queryKey: QUERY_KEYS.olympus.kernels(),
    queryFn: async () => {
      const response = await fetch(API_ROUTES.olympus.kernels, { credentials: 'include' });
      if (!response.ok) {
        throw new Error('Failed to fetch kernels');
      }
      return response.json();
    },
    staleTime: 30000,
    refetchInterval: 60000,
    enabled: isAuthenticated,
  });
}

export function useGetKernel(kernelId: string | null) {
  return useQuery<SpawnedKernel>({
    queryKey: [...M8_KEYS.kernel, kernelId],
    queryFn: () => {
      if (!kernelId) throw new Error('Kernel ID required');
      return getM8Client().getKernel(kernelId);
    },
    enabled: !!kernelId,
    staleTime: 30000,
  });
}

export function useCreateProposal() {
  const queryClient = useQueryClient();

  return useMutation<CreateProposalResponse, Error, CreateProposalRequest>({
    mutationFn: (request) => getM8Client().createProposal(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: M8_KEYS.proposals });
      queryClient.invalidateQueries({ queryKey: M8_KEYS.status });
    },
  });
}

export function useVoteOnProposal() {
  const queryClient = useQueryClient();

  return useMutation<VoteResponse, Error, { proposalId: string; autoVote?: boolean }>({
    mutationFn: ({ proposalId, autoVote }) => 
      getM8Client().vote(proposalId, { auto_vote: autoVote }),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: M8_KEYS.proposals });
      queryClient.invalidateQueries({ queryKey: [...M8_KEYS.proposal, data.proposal_id] });
      queryClient.invalidateQueries({ queryKey: M8_KEYS.status });
    },
  });
}

export function useSpawnKernel() {
  const queryClient = useQueryClient();

  return useMutation<SpawnResponse, Error, { proposalId: string; force?: boolean }>({
    mutationFn: ({ proposalId, force }) => 
      getM8Client().spawn(proposalId, { force }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: M8_KEYS.proposals });
      queryClient.invalidateQueries({ queryKey: M8_KEYS.kernels });
      queryClient.invalidateQueries({ queryKey: M8_KEYS.status });
    },
  });
}

export function useSpawnDirect() {
  const queryClient = useQueryClient();

  return useMutation<SpawnDirectResponse, Error, SpawnDirectRequest>({
    mutationFn: (request) => getM8Client().spawnDirect(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: M8_KEYS.proposals });
      queryClient.invalidateQueries({ queryKey: M8_KEYS.kernels });
      queryClient.invalidateQueries({ queryKey: M8_KEYS.status });
    },
  });
}

export interface M8SpawningHook {
  status: ReturnType<typeof useM8Status>;
  proposals: ReturnType<typeof useListProposals>;
  kernels: ReturnType<typeof useListSpawnedKernels>;
  createProposal: ReturnType<typeof useCreateProposal>;
  vote: ReturnType<typeof useVoteOnProposal>;
  spawn: ReturnType<typeof useSpawnKernel>;
  spawnDirect: ReturnType<typeof useSpawnDirect>;
}

export function useM8Spawning(): M8SpawningHook {
  const status = useM8Status();
  const proposals = useListProposals();
  const kernels = useListSpawnedKernels();
  const createProposal = useCreateProposal();
  const vote = useVoteOnProposal();
  const spawn = useSpawnKernel();
  const spawnDirect = useSpawnDirect();

  return {
    status,
    proposals,
    kernels,
    createProposal,
    vote,
    spawn,
    spawnDirect,
  };
}

export function useWarHistory(limit: number = 50) {
  const { isAuthenticated } = useAuth();
  return useQuery<WarHistoryRecord[]>({
    queryKey: [...M8_KEYS.warHistory, limit],
    queryFn: async () => {
      const response = await fetch(API_ROUTES.olympus.warHistory(limit), { credentials: 'include' });
      if (!response.ok) {
        throw new Error('Failed to fetch war history');
      }
      const data = await response.json();
      return data.history || [];
    },
    staleTime: 30000,
    refetchInterval: 60000,
    enabled: isAuthenticated,
  });
}

export function useActiveWar() {
  const { isAuthenticated } = useAuth();
  return useQuery<WarHistoryRecord | null>({
    queryKey: M8_KEYS.activeWar,
    queryFn: async () => {
      const response = await fetch(API_ROUTES.olympus.warActive, { credentials: 'include' });
      if (!response.ok) {
        throw new Error('Failed to fetch active war');
      }
      const data = await response.json();
      return data.war || null;
    },
    staleTime: 10000,
    refetchInterval: 15000,
    enabled: isAuthenticated,
  });
}

export interface IdleKernel extends PostgresKernel {
  idle_duration_seconds: number;
  idle_since: string;
}

export interface IdleKernelsResponse {
  kernels: IdleKernel[];
  total: number;
}

export interface DeleteKernelResponse {
  success: boolean;
  kernel_id: string;
  message: string;
}

// Import types from API service
import type {
  CannibalizeResponse,
  MergeKernelsResponse,
  AutoCannibalizeRequest,
  AutoCannibalizeResponse,
  AutoMergeRequest,
  AutoMergeResponse,
} from '@/api/services/olympus';

export function useIdleKernels(threshold_seconds: number = 300) {
  const { isAuthenticated } = useAuth();
  return useQuery<IdleKernelsResponse>({
    queryKey: ['m8', 'idleKernels', threshold_seconds],
    queryFn: async () => {
      const response = await fetch(`${API_ROUTES.olympus.m8.idleKernels}?threshold=${threshold_seconds}`, {
        credentials: 'include',
      });
      if (!response.ok) {
        throw new Error('Failed to fetch idle kernels');
      }
      return response.json();
    },
    staleTime: 30000,
    refetchInterval: 30000,
    enabled: isAuthenticated,
  });
}

export function useDeleteKernel() {
  const queryClient = useQueryClient();

  return useMutation<DeleteKernelResponse, Error, { kernelId: string }>({
    mutationFn: ({ kernelId }) => deleteKernel(kernelId) as Promise<DeleteKernelResponse>,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: M8_KEYS.kernels });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.olympus.kernels() });
      queryClient.invalidateQueries({ queryKey: ['m8', 'idleKernels'] });
      queryClient.invalidateQueries({ queryKey: M8_KEYS.status });
    },
  });
}

export function useCannibalizeKernel() {
  const queryClient = useQueryClient();

  return useMutation<CannibalizeResponse, Error, { source_kernel_id: string; target_kernel_id: string }>({
    mutationFn: (request) => cannibalizeKernel({ source_id: request.source_kernel_id, target_id: request.target_kernel_id }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: M8_KEYS.kernels });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.olympus.kernels() });
      queryClient.invalidateQueries({ queryKey: ['m8', 'idleKernels'] });
      queryClient.invalidateQueries({ queryKey: M8_KEYS.status });
    },
  });
}

export function useMergeKernels() {
  const queryClient = useQueryClient();

  return useMutation<MergeKernelsResponse, Error, { kernel_ids: string[]; new_name?: string }>({
    mutationFn: (request) => mergeKernels({ kernel_ids: request.kernel_ids }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: M8_KEYS.kernels });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.olympus.kernels() });
      queryClient.invalidateQueries({ queryKey: ['m8', 'idleKernels'] });
      queryClient.invalidateQueries({ queryKey: M8_KEYS.status });
    },
  });
}

export function useAutoCannibalize() {
  const queryClient = useQueryClient();

  return useMutation<AutoCannibalizeResponse, Error, AutoCannibalizeRequest>({
    mutationFn: (request) => autoCannibalize(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: M8_KEYS.kernels });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.olympus.kernels() });
      queryClient.invalidateQueries({ queryKey: ['m8', 'idleKernels'] });
      queryClient.invalidateQueries({ queryKey: M8_KEYS.status });
    },
  });
}

export function useAutoMerge() {
  const queryClient = useQueryClient();

  return useMutation<AutoMergeResponse, Error, AutoMergeRequest>({
    mutationFn: (request) => autoMerge(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: M8_KEYS.kernels });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.olympus.kernels() });
      queryClient.invalidateQueries({ queryKey: ['m8', 'idleKernels'] });
      queryClient.invalidateQueries({ queryKey: M8_KEYS.status });
    },
  });
}

// ============= Governance Hooks (God Oversight on Lifecycle Decisions) =============

export type LifecycleAction = 'spawn' | 'merge' | 'cannibalize' | 'evolve' | 'hibernate' | 'awaken';
export type VoteDecision = 'approve' | 'reject' | 'abstain' | 'defer';
export type GovernanceProposalStatus = 'pending' | 'debating' | 'approved' | 'rejected' | 'executed' | 'expired';

export interface GovernanceStats {
  e8_cap: number;
  current_kernels: number;
  available_slots: number;
  at_capacity: boolean;
  active_proposals: number;
  approved_pending: number;
  protected_gods: number;
  registered_kernels: number;
  primitives_tracked: number;
}

export interface GovernanceProposal {
  proposal_id: string;
  action: LifecycleAction;
  target_kernel_id: string | null;
  proposed_by: string;
  reason: string;
  timestamp: number;
  votes: Record<string, VoteDecision>;
  arguments: Array<{ god: string; argument: string; vote: string; timestamp: number }>;
  domain: string | null;
  basin_coordinates: number[] | null;
  parent_kernels: string[] | null;
  merge_target_id: string | null;
  kernel_type: string;
  primitive_function: string | null;
  status: GovernanceProposalStatus;
  execution_result: Record<string, unknown> | null;
}

export interface E8Capacity {
  current: number;
  cap: number;
  available: number;
  at_capacity: boolean;
}

const GOV_KEYS = {
  stats: ['governance', 'stats'] as const,
  proposals: ['governance', 'proposals'] as const,
  capacity: ['governance', 'capacity'] as const,
};

export function useGovernanceStats() {
  const { isAuthenticated } = useAuth();
  return useQuery<{ success: boolean; data: GovernanceStats }>({
    queryKey: GOV_KEYS.stats,
    queryFn: async () => {
      const res = await fetch(`${API_ROUTES.olympus}/m8/governance/stats`);
      if (!res.ok) throw new Error('Failed to fetch governance stats');
      return res.json();
    },
    enabled: isAuthenticated,
    staleTime: 10000,
  });
}

export function useGovernanceProposals() {
  const { isAuthenticated } = useAuth();
  return useQuery<{ proposals: GovernanceProposal[]; total: number }>({
    queryKey: GOV_KEYS.proposals,
    queryFn: async () => {
      const res = await fetch(`${API_ROUTES.olympus}/m8/governance/proposals`);
      if (!res.ok) throw new Error('Failed to fetch proposals');
      return res.json();
    },
    enabled: isAuthenticated,
    staleTime: 5000,
  });
}

export function useE8Capacity() {
  const { isAuthenticated } = useAuth();
  return useQuery<E8Capacity>({
    queryKey: GOV_KEYS.capacity,
    queryFn: async () => {
      const res = await fetch(`${API_ROUTES.olympus}/m8/governance/capacity`);
      if (!res.ok) throw new Error('Failed to fetch capacity');
      return res.json();
    },
    enabled: isAuthenticated,
    staleTime: 15000,
  });
}

export interface ProposeActionRequest {
  action: LifecycleAction;
  proposed_by: string;
  reason: string;
  target_kernel_id?: string;
  domain?: string;
  basin_coordinates?: number[];
  parent_kernels?: string[];
  merge_target_id?: string;
  primitive_function?: string;
}

export function useProposeLifecycleAction() {
  const queryClient = useQueryClient();

  return useMutation<{ success: boolean; message: string; proposal_id?: string }, Error, ProposeActionRequest>({
    mutationFn: async (request) => {
      const res = await fetch(`${API_ROUTES.olympus}/m8/governance/propose`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });
      if (!res.ok) throw new Error('Failed to create proposal');
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: GOV_KEYS.proposals });
      queryClient.invalidateQueries({ queryKey: GOV_KEYS.stats });
    },
  });
}

export interface VoteRequest {
  proposalId: string;
  god_name: string;
  vote: VoteDecision;
  argument?: string;
}

export function useVoteOnLifecycleProposal() {
  const queryClient = useQueryClient();

  return useMutation<{ success: boolean; message: string }, Error, VoteRequest>({
    mutationFn: async ({ proposalId, ...body }) => {
      const res = await fetch(`${API_ROUTES.olympus}/m8/governance/vote/${proposalId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error('Failed to submit vote');
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: GOV_KEYS.proposals });
      queryClient.invalidateQueries({ queryKey: GOV_KEYS.stats });
    },
  });
}

export function useExecuteLifecycleProposal() {
  const queryClient = useQueryClient();

  return useMutation<{ success: boolean; message: string; result?: Record<string, unknown> }, Error, string>({
    mutationFn: async (proposalId) => {
      const res = await fetch(`${API_ROUTES.olympus}/m8/governance/execute/${proposalId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      if (!res.ok) throw new Error('Failed to execute proposal');
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: GOV_KEYS.proposals });
      queryClient.invalidateQueries({ queryKey: GOV_KEYS.stats });
      queryClient.invalidateQueries({ queryKey: GOV_KEYS.capacity });
      queryClient.invalidateQueries({ queryKey: M8_KEYS.kernels });
      queryClient.invalidateQueries({ queryKey: M8_KEYS.status });
    },
  });
}
