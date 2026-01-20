/**
 * Pantheon Registry Service - TypeScript Implementation
 * =====================================================
 * 
 * Provides registry access, kernel spawner, and lifecycle management
 * for the TypeScript/Node.js server.
 * 
 * Authority: E8 Protocol v4.0, WP5.1
 * Status: ACTIVE
 * Created: 2026-01-20
 */

import * as fs from 'fs';
import * as path from 'path';
import * as yaml from 'yaml';
import {
  PantheonRegistrySchema,
  type PantheonRegistry,
  type GodContract,
  type ChaosKernelRules,
  type RoleSpec,
  type KernelSelection,
  type ChaosKernelIdentity,
  validatePantheonRegistry,
  parseChaosKernelName,
  isValidChaosKernelName,
  getGodContract,
  findGodsByDomain,
  getGodsByTier,
} from '@/shared/pantheon-registry-schema';

export * from '@/shared/pantheon-registry-schema';

/**
 * Pantheon Registry Service
 * 
 * Loads and caches the formal pantheon registry from YAML.
 * Provides god contract lookup, chaos kernel naming, and spawn validation.
 */
export class PantheonRegistryService {
  private static instance: PantheonRegistryService | null = null;
  private registry: PantheonRegistry | null = null;
  private registryPath: string;

  private constructor(registryPath?: string) {
    // Default to pantheon/registry.yaml relative to project root
    this.registryPath = registryPath || path.join(
      process.cwd(),
      'pantheon',
      'registry.yaml'
    );
  }

  /**
   * Get singleton instance of registry service
   */
  public static getInstance(registryPath?: string): PantheonRegistryService {
    if (!PantheonRegistryService.instance) {
      PantheonRegistryService.instance = new PantheonRegistryService(registryPath);
    }
    return PantheonRegistryService.instance;
  }

  /**
   * Force reload of registry (useful for testing)
   */
  public static reload(registryPath?: string): PantheonRegistryService {
    PantheonRegistryService.instance = new PantheonRegistryService(registryPath);
    return PantheonRegistryService.instance;
  }

  /**
   * Load registry from YAML file
   */
  public async load(): Promise<PantheonRegistry> {
    if (this.registry) {
      return this.registry;
    }

    console.log(`Loading pantheon registry from ${this.registryPath}`);

    // Check if file exists
    if (!fs.existsSync(this.registryPath)) {
      throw new Error(`Registry not found: ${this.registryPath}`);
    }

    // Read and parse YAML
    const yamlContent = fs.readFileSync(this.registryPath, 'utf-8');
    const rawData = yaml.parse(yamlContent);

    // Validate registry
    const validation = validatePantheonRegistry(rawData);
    if (!validation.valid) {
      throw new Error(
        `Registry validation failed:\n${validation.errors.join('\n')}`
      );
    }

    if (validation.warnings.length > 0) {
      console.warn('Registry validation warnings:');
      validation.warnings.forEach(w => console.warn(`  - ${w}`));
    }

    // Parse with Zod
    this.registry = PantheonRegistrySchema.parse(rawData);

    console.log(
      `Loaded ${Object.keys(this.registry.gods).length} gods from registry v${this.registry.metadata.version}`
    );

    return this.registry;
  }

  /**
   * Get loaded registry (throws if not loaded)
   */
  public getRegistry(): PantheonRegistry {
    if (!this.registry) {
      throw new Error('Registry not loaded - call load() first');
    }
    return this.registry;
  }

  /**
   * Get god contract by name
   */
  public getGod(name: string): GodContract | null {
    const registry = this.getRegistry();
    return getGodContract(name, registry);
  }

  /**
   * Get all god contracts
   */
  public getAllGods(): Record<string, GodContract> {
    return this.getRegistry().gods;
  }

  /**
   * Find gods by domain
   */
  public findGodsByDomain(domain: string): Array<{ name: string; contract: GodContract }> {
    const registry = this.getRegistry();
    return findGodsByDomain(domain, registry);
  }

  /**
   * Get gods by tier
   */
  public getGodsByTier(): { essential: Array<any>; specialized: Array<any> } {
    const registry = this.getRegistry();
    return getGodsByTier(registry);
  }

  /**
   * Get chaos kernel rules
   */
  public getChaosKernelRules(): ChaosKernelRules {
    return this.getRegistry().chaos_kernel_rules;
  }

  /**
   * Check if name is a registered god
   */
  public isGodName(name: string): boolean {
    return name in this.getRegistry().gods;
  }

  /**
   * Check if name is a valid chaos kernel
   */
  public isValidChaosKernelName(name: string): boolean {
    return isValidChaosKernelName(name);
  }

  /**
   * Parse chaos kernel name
   */
  public parseChaosKernelName(name: string): ChaosKernelIdentity | null {
    return parseChaosKernelName(name);
  }

  /**
   * Get registry metadata
   */
  public getMetadata() {
    return this.getRegistry().metadata;
  }

  /**
   * Get god count
   */
  public getGodCount(): number {
    return Object.keys(this.getRegistry().gods).length;
  }
}

/**
 * Kernel Spawner Service
 * 
 * Selects gods or spawns chaos kernels based on role requirements.
 * Enforces spawn constraints from registry contracts.
 */
export class KernelSpawnerService {
  private registry: PantheonRegistryService;
  private activeInstances: Map<string, number>;
  private chaosCounters: Map<string, number>;
  private activeChaosCount: number;

  constructor(registry?: PantheonRegistryService) {
    this.registry = registry || PantheonRegistryService.getInstance();
    this.activeInstances = new Map();
    this.chaosCounters = new Map();
    this.activeChaosCount = 0;
  }

  /**
   * Select god or chaos kernel for a role
   */
  public selectGod(role: RoleSpec): KernelSelection {
    // Step 1: Check preferred god
    if (role.preferred_god) {
      const god = this.registry.getGod(role.preferred_god);
      if (god && this.canSpawnGod(god, role.preferred_god)) {
        const epithet = this.selectEpithet(god, role);
        return {
          selected_type: 'god',
          god_name: role.preferred_god,
          epithet,
          rationale: `Preferred god ${role.preferred_god} available`,
          spawn_approved: true,
        };
      }
    }

    // Step 2: Find matching gods by domain
    const candidates = this.findMatchingGods(role);

    // Step 3: Filter by spawn constraints
    const available = candidates.filter(({ name, contract }) => 
      this.canSpawnGod(contract, name)
    );

    // Step 4: Select best match
    if (available.length > 0) {
      const best = this.selectBestGod(available, role);
      const epithet = this.selectEpithet(best.contract, role);
      return {
        selected_type: 'god',
        god_name: best.name,
        epithet,
        rationale: this.explainGodSelection(best, role),
        spawn_approved: true,
      };
    }

    // Step 5: Spawn chaos kernel if allowed
    if (role.allow_chaos_spawn !== false) {
      const chaosName = this.generateChaosKernelName(role);
      return {
        selected_type: 'chaos',
        chaos_name: chaosName,
        rationale: this.explainChaosSpawn(role),
        spawn_approved: false, // Needs pantheon vote
      };
    }

    // Step 6: No selection possible
    return {
      selected_type: 'god', // Type required by schema
      rationale: 'No available gods and chaos spawn not allowed',
      spawn_approved: false,
    };
  }

  /**
   * Find gods matching role domains
   */
  private findMatchingGods(role: RoleSpec): Array<{ name: string; contract: GodContract }> {
    const candidates = new Map<string, GodContract>();

    for (const domain of role.domain) {
      const gods = this.registry.findGodsByDomain(domain);
      gods.forEach(god => {
        candidates.set(god.name, god.contract);
      });
    }

    return Array.from(candidates.entries()).map(([name, contract]) => ({
      name,
      contract,
    }));
  }

  /**
   * Check if god can be spawned
   */
  private canSpawnGod(god: GodContract, name: string): boolean {
    // Gods are singular - max_instances should always be 1
    if (god.spawn_constraints.max_instances !== 1) {
      console.warn(
        `God ${name} has invalid max_instances: ${god.spawn_constraints.max_instances} (should be 1)`
      );
      return false;
    }

    // Check active instances
    const active = this.activeInstances.get(name) || 0;
    if (active >= god.spawn_constraints.max_instances) {
      return false;
    }

    // Check when_allowed constraint
    if (god.spawn_constraints.when_allowed === 'never') {
      return active === 0; // Only if not already spawned
    }

    return true;
  }

  /**
   * Select best god from candidates
   */
  private selectBestGod(
    candidates: Array<{ name: string; contract: GodContract }>,
    role: RoleSpec
  ): { name: string; contract: GodContract } {
    const scored = candidates.map(candidate => {
      let score = 0;

      // Domain overlap (primary factor)
      const overlap = candidate.contract.domain.filter(d => 
        role.domain.includes(d)
      ).length;
      score += overlap * 10;

      // E8 layer bonus
      if (candidate.contract.e8_alignment.layer === '8') {
        score += 5;
      } else if (candidate.contract.e8_alignment.layer === '0/1') {
        score += 3;
      }

      // Availability bonus
      if (candidate.contract.rest_policy.type === 'never') {
        score += 10; // Essential gods
      } else if ('duty_cycle' in candidate.contract.rest_policy) {
        score += (candidate.contract.rest_policy as any).duty_cycle * 2;
      }

      return { ...candidate, score };
    });

    scored.sort((a, b) => b.score - a.score);
    return scored[0];
  }

  /**
   * Select epithet for god based on role
   */
  private selectEpithet(god: GodContract, role: RoleSpec): string | undefined {
    if (god.epithets.length === 0) {
      return undefined;
    }

    // For now, use first epithet
    // TODO: Build epithet-to-domain mapping
    return god.epithets[0];
  }

  /**
   * Explain god selection
   */
  private explainGodSelection(
    god: { name: string; contract: GodContract },
    role: RoleSpec
  ): string {
    const overlap = god.contract.domain.filter(d => role.domain.includes(d));
    return `Selected ${god.name} for domains: ${overlap.join(', ')}. Tier: ${god.contract.tier}, Layer: ${god.contract.e8_alignment.layer}`;
  }

  /**
   * Generate chaos kernel name
   */
  private generateChaosKernelName(role: RoleSpec): string {
    const domain = role.domain[0] || 'general';
    const counter = this.chaosCounters.get(domain) || 0;
    const nextId = counter + 1;
    this.chaosCounters.set(domain, nextId);
    return `chaos_${domain}_${String(nextId).padStart(3, '0')}`;
  }

  /**
   * Explain chaos spawn
   */
  private explainChaosSpawn(role: RoleSpec): string {
    return `No available gods for domains: ${role.domain.join(', ')}. Spawning chaos kernel to fill capability gap. Requires pantheon vote approval.`;
  }

  /**
   * Register god spawn
   */
  public registerSpawn(name: string): void {
    if (this.registry.isGodName(name)) {
      const current = this.activeInstances.get(name) || 0;
      this.activeInstances.set(name, current + 1);
    } else if (this.registry.isValidChaosKernelName(name)) {
      this.activeChaosCount++;
    }
  }

  /**
   * Register god/kernel death
   */
  public registerDeath(name: string): void {
    if (this.registry.isGodName(name)) {
      const current = this.activeInstances.get(name) || 0;
      this.activeInstances.set(name, Math.max(0, current - 1));
    } else if (this.registry.isValidChaosKernelName(name)) {
      this.activeChaosCount = Math.max(0, this.activeChaosCount - 1);
    }
  }

  /**
   * Get active instance count
   */
  public getActiveCount(name: string): number {
    return this.activeInstances.get(name) || 0;
  }

  /**
   * Get total chaos count
   */
  public getTotalChaosCount(): number {
    return Array.from(this.chaosCounters.values()).reduce((sum, count) => sum + count, 0);
  }

  /**
   * Get active chaos count
   */
  public getActiveChaosCount(): number {
    return this.activeChaosCount;
  }

  /**
   * Validate spawn request
   */
  public validateSpawnRequest(name: string): { valid: boolean; reason: string } {
    // Check if god
    if (this.registry.isGodName(name)) {
      const god = this.registry.getGod(name);
      if (!god) {
        return { valid: false, reason: `God ${name} not found in registry` };
      }

      if (!this.canSpawnGod(god, name)) {
        const active = this.getActiveCount(name);
        return {
          valid: false,
          reason: `God ${name} spawn constraints violated (max: ${god.spawn_constraints.max_instances}, active: ${active})`,
        };
      }

      return { valid: true, reason: `God ${name} spawn approved` };
    }

    // Check if chaos kernel
    if (this.registry.isValidChaosKernelName(name)) {
      const rules = this.registry.getChaosKernelRules();
      const totalLimit = rules.spawning_limits.total_active_limit;

      if (this.activeChaosCount >= totalLimit) {
        return {
          valid: false,
          reason: `Chaos kernel limit reached (${this.activeChaosCount}/${totalLimit})`,
        };
      }

      // Check domain limit
      const parsed = this.registry.parseChaosKernelName(name);
      if (parsed) {
        const domainLimit = rules.spawning_limits.per_domain_limit;
        const domainCount = this.chaosCounters.get(parsed.domain) || 0;

        if (domainCount >= domainLimit) {
          return {
            valid: false,
            reason: `Chaos kernel domain limit reached for ${parsed.domain} (${domainCount}/${domainLimit})`,
          };
        }
      }

      return { valid: true, reason: `Chaos kernel ${name} spawn approved` };
    }

    return {
      valid: false,
      reason: `Invalid kernel name: ${name} (not god or chaos kernel)`,
    };
  }
}

// =============================================================================
// CONVENIENCE EXPORTS
// =============================================================================

/**
 * Get global registry service instance
 */
export function getRegistryService(): PantheonRegistryService {
  return PantheonRegistryService.getInstance();
}

/**
 * Create kernel spawner service
 */
export function createSpawnerService(): KernelSpawnerService {
  return new KernelSpawnerService();
}
