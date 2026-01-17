/**
 * PANTHEON REGISTRY SCHEMA
 * ========================
 * 
 * Zod schemas and TypeScript types for the formal Pantheon Registry.
 * Defines god contracts, chaos kernel rules, and validation logic.
 * 
 * Authority: E8 Protocol v4.0, WP5.1
 * Status: ACTIVE
 * Created: 2026-01-17
 */

import { z } from 'zod';

// =============================================================================
// REST POLICY SCHEMAS
// =============================================================================

export const RestPolicyNeverSchema = z.object({
  type: z.literal('never'),
  reason: z.string(),
});

export const RestPolicyMinimalRotatingSchema = z.object({
  type: z.literal('minimal_rotating'),
  partner: z.string(),
  duty_cycle: z.number().min(0).max(1),
  reason: z.string(),
});

export const RestPolicyCoordinatedAlternatingSchema = z.object({
  type: z.literal('coordinated_alternating'),
  partner: z.string(),
  duty_cycle: z.number().min(0).max(1),
  reason: z.string(),
});

export const RestPolicyScheduledSchema = z.object({
  type: z.literal('scheduled'),
  duty_cycle: z.number().min(0).max(1),
  rest_duration: z.number().min(0).max(1),
  reason: z.string(),
});

export const RestPolicySeasonalSchema = z.object({
  type: z.literal('seasonal'),
  active_season: z.enum(['wake', 'dream', 'mushroom']),
  rest_season: z.enum(['wake', 'dream', 'mushroom']),
  duty_cycle: z.number().min(0).max(1),
  reason: z.string(),
});

export const RestPolicySchema = z.discriminatedUnion('type', [
  RestPolicyNeverSchema,
  RestPolicyMinimalRotatingSchema,
  RestPolicyCoordinatedAlternatingSchema,
  RestPolicyScheduledSchema,
  RestPolicySeasonalSchema,
]);

export type RestPolicy = z.infer<typeof RestPolicySchema>;

// =============================================================================
// SPAWN CONSTRAINTS SCHEMA
// =============================================================================

export const SpawnConstraintsSchema = z.object({
  max_instances: z.number().int().min(1),
  when_allowed: z.enum(['never', 'always', 'conditional']),
  rationale: z.string(),
});

export type SpawnConstraints = z.infer<typeof SpawnConstraintsSchema>;

// =============================================================================
// E8 ALIGNMENT SCHEMA
// =============================================================================

export const E8AlignmentSchema = z.object({
  simple_root: z.string().nullable(), // "α₁", "α₂", etc., or null for composite
  layer: z.enum(['0/1', '4', '8', '64', '240']),
});

export type E8Alignment = z.infer<typeof E8AlignmentSchema>;

// =============================================================================
// GOD CONTRACT SCHEMA
// =============================================================================

export const GodContractSchema = z.object({
  tier: z.enum(['essential', 'specialized']),
  domain: z.array(z.string()).min(1),
  description: z.string(),
  octant: z.number().int().min(0).max(7).nullable(),
  epithets: z.array(z.string()),
  coupling_affinity: z.array(z.string()),
  rest_policy: RestPolicySchema,
  spawn_constraints: SpawnConstraintsSchema,
  promotion_from: z.string().nullable(),
  e8_alignment: E8AlignmentSchema,
});

export type GodContract = z.infer<typeof GodContractSchema>;

// =============================================================================
// CHAOS KERNEL LIFECYCLE SCHEMAS
// =============================================================================

export const ChaosSpawnSchema = z.object({
  trigger: z.string(),
  approval: z.string(),
  initial_state: z.string(),
  naming: z.string(),
});

export const ChaosProtectSchema = z.object({
  duration_cycles: z.number().int().positive(),
  graduated_metrics: z.boolean(),
  protection_rules: z.array(z.string()),
});

export const ChaosLearnSchema = z.object({
  mentor_assignment: z.string(),
  learning_mode: z.string(),
  transfer_knowledge_from: z.string(),
});

export const ChaosWorkSchema = z.object({
  adult_standards_apply: z.boolean(),
  phi_threshold: z.number().min(0).max(1),
  pruning_eligible: z.boolean(),
});

export const ChaosCandidateSchema = z.object({
  phi_threshold: z.number().min(0).max(1),
  duration_cycles: z.number().int().positive(),
  requirements: z.array(z.string()),
});

export const ChaosPromoteSchema = z.object({
  process: z.string(),
  requirements: z.array(z.string()),
  ascension: z.string(),
});

export const ChaosPruningSchema = z.object({
  destination: z.string(),
  manager: z.string(),
  criteria: z.array(z.string()),
  process: z.array(z.string()),
});

export const ChaosSpawningLimitsSchema = z.object({
  max_chaos_kernels: z.number().int().positive(),
  per_domain_limit: z.number().int().positive(),
  total_active_limit: z.number().int().positive(),
});

export const ChaosGeneticLineageSchema = z.object({
  parent_tracking: z.boolean(),
  mutation_allowed: z.boolean(),
  breeding_enabled: z.boolean(),
  lineage_recording: z.string(),
});

export const ChaosLifecycleSchema = z.object({
  spawn: ChaosSpawnSchema,
  protect: ChaosProtectSchema,
  learn: ChaosLearnSchema,
  work: ChaosWorkSchema,
  candidate: ChaosCandidateSchema,
  promote: ChaosPromoteSchema,
});

export const ChaosKernelRulesSchema = z.object({
  naming_pattern: z.string(),
  description: z.string(),
  lifecycle: ChaosLifecycleSchema,
  pruning: ChaosPruningSchema,
  spawning_limits: ChaosSpawningLimitsSchema,
  genetic_lineage: ChaosGeneticLineageSchema,
});

export type ChaosKernelRules = z.infer<typeof ChaosKernelRulesSchema>;

// =============================================================================
// REGISTRY METADATA SCHEMA
// =============================================================================

export const RegistryMetadataSchema = z.object({
  version: z.string(),
  status: z.string(),
  created: z.string(),
  authority: z.string(),
  validation_required: z.boolean(),
});

export const RegistryCompatibilitySchema = z.object({
  e8_protocol: z.string(),
  qig_backend: z.string(),
});

export type RegistryMetadata = z.infer<typeof RegistryMetadataSchema>;

// =============================================================================
// FULL PANTHEON REGISTRY SCHEMA
// =============================================================================

export const PantheonRegistrySchema = z.object({
  gods: z.record(z.string(), GodContractSchema),
  chaos_kernel_rules: ChaosKernelRulesSchema,
  metadata: RegistryMetadataSchema,
  schema_version: z.string(),
  compatibility: RegistryCompatibilitySchema,
  validation_rules: z.array(z.string()),
});

export type PantheonRegistry = z.infer<typeof PantheonRegistrySchema>;

// =============================================================================
// GOD LOOKUP TYPES
// =============================================================================

export interface GodLookup {
  name: string;
  contract: GodContract;
}

export interface GodsByTier {
  essential: GodLookup[];
  specialized: GodLookup[];
}

export interface GodsByDomain {
  [domain: string]: GodLookup[];
}

// =============================================================================
// CHAOS KERNEL TYPES
// =============================================================================

export interface ChaosKernelIdentity {
  id: string;
  name: string; // Format: chaos_{domain}_{id}
  domain: string;
  sequential_id: number;
}

export interface ChaosKernelState {
  identity: ChaosKernelIdentity;
  lifecycle_stage: 'protected' | 'learning' | 'working' | 'candidate' | 'promoted' | 'pruned';
  cycles_in_stage: number;
  phi_score: number;
  mentor_id?: string;
  parent_lineage: string[];
  unique_capability?: string;
}

// =============================================================================
// KERNEL SPAWNER TYPES
// =============================================================================

export interface RoleSpec {
  domain: string[];
  required_capabilities: string[];
  preferred_god?: string;
  allow_chaos_spawn?: boolean;
}

export interface KernelSelection {
  selected_type: 'god' | 'chaos';
  god_name?: string;
  epithet?: string;
  chaos_name?: string;
  rationale: string;
  spawn_approved: boolean;
}

// =============================================================================
// VALIDATION FUNCTIONS
// =============================================================================

/**
 * Validates a pantheon registry against the schema and business rules.
 */
export function validatePantheonRegistry(registry: unknown): {
  valid: boolean;
  errors: string[];
  warnings: string[];
} {
  const errors: string[] = [];
  const warnings: string[] = [];

  // Parse with Zod
  const parseResult = PantheonRegistrySchema.safeParse(registry);
  if (!parseResult.success) {
    errors.push(...parseResult.error.errors.map(e => `${e.path.join('.')}: ${e.message}`));
    return { valid: false, errors, warnings };
  }

  const data = parseResult.data;

  // Business rule validations
  
  // 1. All gods must have unique names
  const godNames = Object.keys(data.gods);
  const uniqueNames = new Set(godNames);
  if (godNames.length !== uniqueNames.size) {
    errors.push('Duplicate god names found');
  }

  // 2. All gods must have max_instances: 1
  for (const [name, god] of Object.entries(data.gods)) {
    if (god.spawn_constraints.max_instances !== 1) {
      errors.push(`God ${name} must have max_instances: 1 (gods are singular)`);
    }
  }

  // 3. Essential tier gods must never sleep
  for (const [name, god] of Object.entries(data.gods)) {
    if (god.tier === 'essential' && god.rest_policy.type !== 'never') {
      errors.push(`Essential god ${name} must have rest_policy.type: never`);
    }
  }

  // 4. Specialized gods should have e8_alignment (warning, not error)
  for (const [name, god] of Object.entries(data.gods)) {
    if (god.tier === 'specialized' && !god.e8_alignment) {
      warnings.push(`Specialized god ${name} should have e8_alignment defined`);
    }
  }

  // 5. Check coupling affinity references valid gods
  for (const [name, god] of Object.entries(data.gods)) {
    for (const affinity of god.coupling_affinity) {
      if (!data.gods[affinity]) {
        errors.push(`God ${name} references non-existent coupling affinity: ${affinity}`);
      }
    }
  }

  // 6. Check rest policy partners exist
  for (const [name, god] of Object.entries(data.gods)) {
    if ('partner' in god.rest_policy) {
      const partner = (god.rest_policy as any).partner;
      if (!data.gods[partner]) {
        errors.push(`God ${name} references non-existent rest partner: ${partner}`);
      }
    }
  }

  // 7. Validate chaos kernel naming pattern (check template string, not regex)
  const expectedPattern = 'chaos_{domain}_{id}';
  if (data.chaos_kernel_rules.naming_pattern !== expectedPattern) {
    warnings.push(`Chaos kernel naming pattern should be: ${expectedPattern}`);
  }

  // 8. Check total active limit <= E8 roots (240)
  if (data.chaos_kernel_rules.spawning_limits.total_active_limit > 240) {
    warnings.push('Total active limit should not exceed 240 (E8 root system)');
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Parses a chaos kernel name and extracts domain and ID.
 */
export function parseChaosKernelName(name: string): ChaosKernelIdentity | null {
  const match = name.match(/^chaos_([a-z_]+)_(\d+)$/);
  if (!match) {
    return null;
  }

  return {
    id: name,
    name,
    domain: match[1],
    sequential_id: parseInt(match[2], 10),
  };
}

/**
 * Validates a chaos kernel name follows the naming pattern.
 */
export function isValidChaosKernelName(name: string): boolean {
  return parseChaosKernelName(name) !== null;
}

/**
 * Checks if a name is a god (not chaos kernel).
 */
export function isGodName(name: string, registry: PantheonRegistry): boolean {
  return name in registry.gods;
}

/**
 * Gets god contract by name.
 */
export function getGodContract(
  name: string,
  registry: PantheonRegistry
): GodContract | null {
  return registry.gods[name] || null;
}

/**
 * Gets all gods by tier.
 */
export function getGodsByTier(registry: PantheonRegistry): GodsByTier {
  const essential: GodLookup[] = [];
  const specialized: GodLookup[] = [];

  for (const [name, contract] of Object.entries(registry.gods)) {
    const lookup: GodLookup = { name, contract };
    if (contract.tier === 'essential') {
      essential.push(lookup);
    } else {
      specialized.push(lookup);
    }
  }

  return { essential, specialized };
}

/**
 * Gets all gods by domain.
 */
export function getGodsByDomain(registry: PantheonRegistry): GodsByDomain {
  const byDomain: GodsByDomain = {};

  for (const [name, contract] of Object.entries(registry.gods)) {
    for (const domain of contract.domain) {
      if (!byDomain[domain]) {
        byDomain[domain] = [];
      }
      byDomain[domain].push({ name, contract });
    }
  }

  return byDomain;
}

/**
 * Finds gods matching a domain.
 */
export function findGodsByDomain(
  domain: string,
  registry: PantheonRegistry
): GodLookup[] {
  const results: GodLookup[] = [];

  for (const [name, contract] of Object.entries(registry.gods)) {
    if (contract.domain.includes(domain)) {
      results.push({ name, contract });
    }
  }

  return results;
}
