/**
 * Pantheon Registry API Routes
 * ============================
 * 
 * REST API endpoints for accessing pantheon registry and kernel spawner.
 * 
 * Authority: E8 Protocol v4.0, WP5.1
 * Status: ACTIVE
 * Created: 2026-01-20
 */

import { Router, type Request, type Response } from 'express';
import { z } from 'zod';
import { getRegistryService, createSpawnerService } from '../services/pantheon-registry';
import type { RoleSpec } from '@/shared/pantheon-registry-schema';

const router = Router();

// =============================================================================
// REGISTRY ENDPOINTS
// =============================================================================

/**
 * GET /api/pantheon/registry
 * Get full pantheon registry
 */
router.get('/registry', async (req: Request, res: Response) => {
  try {
    const service = getRegistryService();
    const registry = await service.getRegistry();
    
    res.json({
      success: true,
      data: registry,
    });
  } catch (error) {
    console.error('Error loading registry:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * GET /api/pantheon/registry/metadata
 * Get registry metadata
 */
router.get('/registry/metadata', async (req: Request, res: Response) => {
  try {
    const service = getRegistryService();
    const metadata = await service.getMetadata();
    
    res.json({
      success: true,
      data: metadata,
    });
  } catch (error) {
    console.error('Error getting metadata:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * GET /api/pantheon/registry/gods
 * Get all god contracts
 */
router.get('/registry/gods', async (req: Request, res: Response) => {
  try {
    const service = getRegistryService();
    const gods = await service.getAllGods();
    
    res.json({
      success: true,
      data: gods,
      count: Object.keys(gods).length,
    });
  } catch (error) {
    console.error('Error getting gods:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * GET /api/pantheon/registry/gods/:name
 * Get specific god contract
 */
router.get('/registry/gods/:name', async (req: Request, res: Response) => {
  try {
    const service = getRegistryService();
    const god = await service.getGod(req.params.name);
    
    if (!god) {
      res.status(404).json({
        success: false,
        error: `God ${req.params.name} not found`,
      });
      return;
    }
    
    res.json({
      success: true,
      data: god,
    });
  } catch (error) {
    console.error('Error getting god:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * GET /api/pantheon/registry/gods/by-tier/:tier
 * Get gods by tier (essential or specialized)
 */
router.get('/registry/gods/by-tier/:tier', async (req: Request, res: Response) => {
  try {
    const tier = req.params.tier;
    if (tier !== 'essential' && tier !== 'specialized') {
      res.status(400).json({
        success: false,
        error: 'Invalid tier - must be "essential" or "specialized"',
      });
      return;
    }
    
    const service = getRegistryService();
    const gods = await service.getGodsByTier(tier as 'essential' | 'specialized');
    
    res.json({
      success: true,
      data: gods,
      count: gods.length,
    });
  } catch (error) {
    console.error('Error getting gods by tier:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * GET /api/pantheon/registry/gods/by-domain/:domain
 * Find gods by domain
 */
router.get('/registry/gods/by-domain/:domain', async (req: Request, res: Response) => {
  try {
    const service = getRegistryService();
    const gods = await service.findGodsByDomain(req.params.domain);
    
    res.json({
      success: true,
      data: gods,
      count: gods.length,
    });
  } catch (error) {
    console.error('Error finding gods by domain:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * GET /api/pantheon/registry/chaos-rules
 * Get chaos kernel lifecycle rules
 */
router.get('/registry/chaos-rules', async (req: Request, res: Response) => {
  try {
    const service = getRegistryService();
    const rules = await service.getChaosKernelRules();
    
    res.json({
      success: true,
      data: rules,
    });
  } catch (error) {
    console.error('Error getting chaos rules:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

// =============================================================================
// KERNEL SPAWNER ENDPOINTS
// =============================================================================

/**
 * POST /api/pantheon/spawner/select
 * Select god or chaos kernel for a role
 * 
 * Body: RoleSpec
 */
router.post('/spawner/select', async (req: Request, res: Response) => {
  try {
    const spawner = createSpawnerService();
    
    // Validate role spec using Zod schema (imported from shared)
    const roleSchema = z.object({
      domain: z.array(z.string()).min(1),
      required_capabilities: z.array(z.string()),
      preferred_god: z.string().optional(),
      allow_chaos_spawn: z.boolean().optional(),
    });
    
    const parseResult = roleSchema.safeParse(req.body);
    if (!parseResult.success) {
      res.status(400).json({
        success: false,
        error: 'Invalid role spec',
        details: parseResult.error.errors,
      });
      return;
    }
    
    const role: RoleSpec = parseResult.data;
    const selection = await spawner.selectGod(role);
    
    res.json({
      success: true,
      data: selection,
    });
  } catch (error) {
    console.error('Error selecting kernel:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * POST /api/pantheon/spawner/validate
 * Validate spawn request
 * 
 * Body: { name: string }
 */
router.post('/spawner/validate', async (req: Request, res: Response) => {
  try {
    const spawner = createSpawnerService();
    const { name } = req.body;
    
    if (!name || typeof name !== 'string') {
      res.status(400).json({
        success: false,
        error: 'Invalid request - name string required',
      });
      return;
    }
    
    const validation = await spawner.validateSpawnRequest(name);
    
    res.json({
      success: true,
      data: validation,
    });
  } catch (error) {
    console.error('Error validating spawn:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * GET /api/pantheon/spawner/chaos/parse/:name
 * Parse chaos kernel name
 */
router.get('/spawner/chaos/parse/:name', async (req: Request, res: Response) => {
  try {
    const service = getRegistryService();
    const parsed = await service.parseChaosKernelName(req.params.name);
    
    if (!parsed) {
      res.status(400).json({
        success: false,
        error: `Invalid chaos kernel name: ${req.params.name}`,
      });
      return;
    }
    
    res.json({
      success: true,
      data: parsed,
    });
  } catch (error) {
    console.error('Error parsing chaos kernel name:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * GET /api/pantheon/spawner/status
 * Get spawner status (active counts, limits)
 */
router.get('/spawner/status', async (req: Request, res: Response) => {
  try {
    const spawner = createSpawnerService();
    const status = await spawner.getSpawnerStatus();
    
    res.json({
      success: true,
      data: status,
    });
  } catch (error) {
    console.error('Error getting spawner status:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

// =============================================================================
// HEALTH CHECK
// =============================================================================

/**
 * GET /api/pantheon/health
 * Health check for registry service
 */
router.get('/health', async (req: Request, res: Response) => {
  try {
    const service = getRegistryService();
    const metadata = await service.getMetadata();
    const godCount = await service.getGodCount();
    
    res.json({
      success: true,
      data: {
        status: 'healthy',
        registry_version: metadata.version,
        registry_status: metadata.status,
        god_count: godCount,
        loaded_at: new Date().toISOString(),
      },
    });
  } catch (error) {
    console.error('Registry health check failed:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
      data: {
        status: 'unhealthy',
      },
    });
  }
});

export default router;
