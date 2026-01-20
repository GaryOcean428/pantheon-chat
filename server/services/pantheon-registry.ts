/**
 * Pantheon Registry Service - Thin API Wrapper
 * ============================================
 * 
 * TypeScript wrapper for Python Pantheon Registry API.
 * All functional logic lives in Python backend (qig-backend/api_pantheon_registry.py).
 * 
 * This service is a thin API client only, following QIG principles:
 * - Python backend: All functional logic, data processing, business rules
 * - TypeScript: API wrapper for frontend/Node.js access only
 * 
 * Authority: E8 Protocol v4.0, WP5.1
 * Status: ACTIVE
 * Created: 2026-01-20
 * Refactored: 2026-01-20 (moved logic to Python)
 */

import axios, { type AxiosInstance } from 'axios';
import type {
  PantheonRegistry,
  GodContract,
  ChaosKernelRules,
  RoleSpec,
  KernelSelection,
  ChaosKernelIdentity,
} from '@/shared/pantheon-registry-schema';

export * from '@/shared/pantheon-registry-schema';

/**
 * Get Python backend URL from environment
 */
function getPythonBackendUrl(): string {
  return process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
}

/**
 * Create axios instance for Python backend
 */
function createPythonClient(): AxiosInstance {
  return axios.create({
    baseURL: getPythonBackendUrl(),
    timeout: 30000,
    headers: {
      'Content-Type': 'application/json',
    },
  });
}

/**
 * Pantheon Registry Service (Thin API Wrapper)
 * 
 * All methods call Python backend endpoints.
 * No functional logic in TypeScript.
 */
export class PantheonRegistryService {
  private static instance: PantheonRegistryService | null = null;
  private client: AxiosInstance;

  private constructor() {
    this.client = createPythonClient();
  }

  /**
   * Get singleton instance
   */
  public static getInstance(): PantheonRegistryService {
    if (!PantheonRegistryService.instance) {
      PantheonRegistryService.instance = new PantheonRegistryService();
    }
    return PantheonRegistryService.instance;
  }

  /**
   * Force reload (useful for testing)
   */
  public static reload(): PantheonRegistryService {
    PantheonRegistryService.instance = new PantheonRegistryService();
    return PantheonRegistryService.instance;
  }

  /**
   * Load full registry from Python backend
   */
  public async load(): Promise<PantheonRegistry> {
    const response = await this.client.get('/api/pantheon/registry');
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to load registry');
    }
    return response.data.data;
  }

  /**
   * Get registry (alias for load)
   */
  public async getRegistry(): Promise<PantheonRegistry> {
    return this.load();
  }

  /**
   * Get god contract by name
   */
  public async getGod(name: string): Promise<GodContract | null> {
    try {
      const response = await this.client.get(`/api/pantheon/registry/gods/${name}`);
      if (!response.data.success) {
        return null;
      }
      return response.data.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response?.status === 404) {
        return null;
      }
      throw error;
    }
  }

  /**
   * Get all god contracts
   */
  public async getAllGods(): Promise<Record<string, GodContract>> {
    const response = await this.client.get('/api/pantheon/registry/gods');
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to get gods');
    }
    return response.data.data;
  }

  /**
   * Find gods by domain
   */
  public async findGodsByDomain(domain: string): Promise<Array<{ name: string; contract: GodContract }>> {
    const response = await this.client.get(`/api/pantheon/registry/gods/by-domain/${domain}`);
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to find gods by domain');
    }
    return response.data.data;
  }

  /**
   * Get gods by tier
   */
  public async getGodsByTier(tier: 'essential' | 'specialized'): Promise<Array<{ name: string; contract: GodContract }>> {
    const response = await this.client.get(`/api/pantheon/registry/gods/by-tier/${tier}`);
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to get gods by tier');
    }
    return response.data.data;
  }

  /**
   * Get chaos kernel rules
   */
  public async getChaosKernelRules(): Promise<ChaosKernelRules> {
    const response = await this.client.get('/api/pantheon/registry/chaos-rules');
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to get chaos rules');
    }
    return response.data.data;
  }

  /**
   * Get registry metadata
   */
  public async getMetadata() {
    const response = await this.client.get('/api/pantheon/registry/metadata');
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to get metadata');
    }
    return response.data.data;
  }

  /**
   * Check if name is a registered god
   */
  public async isGodName(name: string): Promise<boolean> {
    const god = await this.getGod(name);
    return god !== null;
  }

  /**
   * Check if name is a valid chaos kernel
   */
  public isValidChaosKernelName(name: string): boolean {
    // Simple client-side validation (no backend call needed)
    return /^chaos_[a-z_]+_\d+$/.test(name);
  }

  /**
   * Parse chaos kernel name
   */
  public async parseChaosKernelName(name: string): Promise<ChaosKernelIdentity | null> {
    try {
      const response = await this.client.get(`/api/pantheon/spawner/chaos/parse/${name}`);
      if (!response.data.success) {
        return null;
      }
      return response.data.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response?.status === 400) {
        return null;
      }
      throw error;
    }
  }

  /**
   * Get god count
   */
  public async getGodCount(): Promise<number> {
    const gods = await this.getAllGods();
    return Object.keys(gods).length;
  }
}

/**
 * Kernel Spawner Service (Thin API Wrapper)
 * 
 * All methods call Python backend endpoints.
 * No functional logic in TypeScript.
 */
export class KernelSpawnerService {
  private client: AxiosInstance;

  constructor() {
    this.client = createPythonClient();
  }

  /**
   * Select god or chaos kernel for a role
   */
  public async selectGod(role: RoleSpec): Promise<KernelSelection> {
    const response = await this.client.post('/api/pantheon/spawner/select', role);
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to select kernel');
    }
    return response.data.data;
  }

  /**
   * Validate spawn request
   */
  public async validateSpawnRequest(name: string): Promise<{ valid: boolean; reason: string }> {
    const response = await this.client.post('/api/pantheon/spawner/validate', { name });
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to validate spawn');
    }
    return response.data.data;
  }

  /**
   * Get spawner status
   */
  public async getSpawnerStatus() {
    const response = await this.client.get('/api/pantheon/spawner/status');
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to get spawner status');
    }
    return response.data.data;
  }
}

/**
 * Health check
 */
export async function checkRegistryHealth() {
  const client = createPythonClient();
  const response = await client.get('/api/pantheon/health');
  if (!response.data.success) {
    throw new Error(response.data.error || 'Health check failed');
  }
  return response.data.data;
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
