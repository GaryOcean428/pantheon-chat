/**
 * Use Pantheon Registry Hook
 * ==========================
 * 
 * React hook for accessing formal pantheon registry and kernel spawner.
 * Provides god contracts, chaos kernel rules, and spawn selection.
 * 
 * Authority: E8 Protocol v4.0, WP5.1
 * Status: ACTIVE
 * Created: 2026-01-20
 */

import { useQuery, useMutation, type UseQueryResult } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type {
  PantheonRegistry,
  GodContract,
  ChaosKernelRules,
  RoleSpec,
  KernelSelection,
  ChaosKernelIdentity,
} from '@/shared/pantheon-registry-schema';

// =============================================================================
// API CLIENT FUNCTIONS
// =============================================================================

/**
 * Fetch full pantheon registry
 */
export async function fetchRegistry(): Promise<PantheonRegistry> {
  const response = await api.get('/pantheon/registry');
  return response.data.data;
}

/**
 * Fetch registry metadata
 */
export async function fetchRegistryMetadata() {
  const response = await api.get('/pantheon/registry/metadata');
  return response.data.data;
}

/**
 * Fetch all god contracts
 */
export async function fetchGods(): Promise<Record<string, GodContract>> {
  const response = await api.get('/pantheon/registry/gods');
  return response.data.data;
}

/**
 * Fetch specific god contract
 */
export async function fetchGod(name: string): Promise<GodContract> {
  const response = await api.get(`/pantheon/registry/gods/${name}`);
  return response.data.data;
}

/**
 * Fetch gods by tier
 */
export async function fetchGodsByTier(tier: 'essential' | 'specialized') {
  const response = await api.get(`/pantheon/registry/gods/by-tier/${tier}`);
  return response.data.data;
}

/**
 * Find gods by domain
 */
export async function findGodsByDomain(domain: string) {
  const response = await api.get(`/pantheon/registry/gods/by-domain/${domain}`);
  return response.data.data;
}

/**
 * Fetch chaos kernel rules
 */
export async function fetchChaosRules(): Promise<ChaosKernelRules> {
  const response = await api.get('/pantheon/registry/chaos-rules');
  return response.data.data;
}

/**
 * Select god or chaos kernel for role
 */
export async function selectKernel(role: RoleSpec): Promise<KernelSelection> {
  const response = await api.post('/pantheon/spawner/select', role);
  return response.data.data;
}

/**
 * Validate spawn request
 */
export async function validateSpawn(name: string): Promise<{ valid: boolean; reason: string }> {
  const response = await api.post('/pantheon/spawner/validate', { name });
  return response.data.data;
}

/**
 * Parse chaos kernel name
 */
export async function parseChaosName(name: string): Promise<ChaosKernelIdentity> {
  const response = await api.get(`/pantheon/spawner/chaos/parse/${name}`);
  return response.data.data;
}

/**
 * Fetch spawner status
 */
export async function fetchSpawnerStatus() {
  const response = await api.get('/pantheon/spawner/status');
  return response.data.data;
}

/**
 * Health check
 */
export async function fetchRegistryHealth() {
  const response = await api.get('/pantheon/health');
  return response.data.data;
}

// =============================================================================
// REACT HOOKS
// =============================================================================

/**
 * Hook to fetch full pantheon registry
 */
export function usePantheonRegistry() {
  return useQuery({
    queryKey: ['pantheon', 'registry'],
    queryFn: fetchRegistry,
    staleTime: 5 * 60 * 1000, // 5 minutes - registry rarely changes
    gcTime: 10 * 60 * 1000, // 10 minutes
  });
}

/**
 * Hook to fetch registry metadata
 */
export function useRegistryMetadata() {
  return useQuery({
    queryKey: ['pantheon', 'metadata'],
    queryFn: fetchRegistryMetadata,
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Hook to fetch all gods
 */
export function useGods() {
  return useQuery({
    queryKey: ['pantheon', 'gods'],
    queryFn: fetchGods,
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Hook to fetch specific god
 */
export function useGod(name: string | null) {
  return useQuery({
    queryKey: ['pantheon', 'god', name],
    queryFn: () => fetchGod(name!),
    enabled: !!name,
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Hook to fetch gods by tier
 */
export function useGodsByTier(tier: 'essential' | 'specialized') {
  return useQuery({
    queryKey: ['pantheon', 'gods-by-tier', tier],
    queryFn: () => fetchGodsByTier(tier),
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Hook to find gods by domain
 */
export function useGodsByDomain(domain: string | null) {
  return useQuery({
    queryKey: ['pantheon', 'gods-by-domain', domain],
    queryFn: () => findGodsByDomain(domain!),
    enabled: !!domain,
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Hook to fetch chaos kernel rules
 */
export function useChaosRules() {
  return useQuery({
    queryKey: ['pantheon', 'chaos-rules'],
    queryFn: fetchChaosRules,
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Hook to fetch spawner status
 */
export function useSpawnerStatus() {
  return useQuery({
    queryKey: ['pantheon', 'spawner-status'],
    queryFn: fetchSpawnerStatus,
    refetchInterval: 10000, // Refetch every 10 seconds for live counts
  });
}

/**
 * Hook to fetch registry health
 */
export function useRegistryHealth() {
  return useQuery({
    queryKey: ['pantheon', 'health'],
    queryFn: fetchRegistryHealth,
    refetchInterval: 30000, // Refetch every 30 seconds
  });
}

/**
 * Mutation hook to select kernel for role
 */
export function useSelectKernel() {
  return useMutation({
    mutationFn: selectKernel,
  });
}

/**
 * Mutation hook to validate spawn
 */
export function useValidateSpawn() {
  return useMutation({
    mutationFn: validateSpawn,
  });
}

/**
 * Mutation hook to parse chaos kernel name
 */
export function useParseChaosName() {
  return useMutation({
    mutationFn: parseChaosName,
  });
}

// =============================================================================
// COMPOSITE HOOKS
// =============================================================================

/**
 * Hook to get all registry data at once
 */
export function useFullRegistry() {
  const registry = usePantheonRegistry();
  const metadata = useRegistryMetadata();
  const gods = useGods();
  const chaosRules = useChaosRules();
  const spawnerStatus = useSpawnerStatus();
  const health = useRegistryHealth();

  return {
    registry,
    metadata,
    gods,
    chaosRules,
    spawnerStatus,
    health,
    isLoading:
      registry.isLoading ||
      metadata.isLoading ||
      gods.isLoading ||
      chaosRules.isLoading,
    isError:
      registry.isError ||
      metadata.isError ||
      gods.isError ||
      chaosRules.isError,
  };
}

/**
 * Hook for god selection workflow
 * 
 * Example:
 *   const { selectForRole, selection, isSelecting } = useGodSelection();
 *   
 *   await selectForRole({
 *     domain: ['synthesis', 'foresight'],
 *     required_capabilities: ['prediction'],
 *   });
 *   
 *   if (selection?.selected_type === 'god') {
 *     console.log(`Selected: ${selection.god_name} ${selection.epithet}`);
 *   }
 */
export function useGodSelection() {
  const { mutate, data, isPending, isError, error } = useSelectKernel();

  return {
    selectForRole: mutate,
    selection: data,
    isSelecting: isPending,
    isError,
    error,
  };
}

/**
 * Hook for spawn validation workflow
 */
export function useSpawnValidation() {
  const { mutate, data, isPending, isError, error } = useValidateSpawn();

  return {
    validateSpawn: mutate,
    validation: data,
    isValidating: isPending,
    isError,
    error,
  };
}
