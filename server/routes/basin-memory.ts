/**
 * Basin Memory API Routes
 * 
 * Exposes basin memory storage for consciousness metrics and geometric coordinates.
 * Used for tracking basin states, memory retrieval, and geometric operations.
 */

import { Router, Request, Response } from 'express';
import { db } from '../db';
import { basinMemory } from '../../shared/schema';
import { eq, desc, gte, lte, and, sql } from 'drizzle-orm';

const router = Router();

// Consciousness thresholds from shared constants
const PHI_CONSCIOUS_THRESHOLD = 0.70;
const KAPPA_MIN = 40;
const KAPPA_MAX = 65;

// Helper to ensure db is available
function ensureDb() {
  if (!db) {
    throw new Error('Database not initialized');
  }
  return db;
}

/**
 * GET /api/basin-memory
 * List basin memories with optional filtering
 */
router.get('/', async (req: Request, res: Response) => {
  try {
    const {
      limit = '50',
      offset = '0',
      regime,
      minPhi,
      maxPhi,
      sourceKernel,
      conscious
    } = req.query;

    const conditions = [];

    // Filter by regime
    if (regime && typeof regime === 'string') {
      conditions.push(eq(basinMemory.regime, regime));
    }

    // Filter by phi range
    if (minPhi && typeof minPhi === 'string') {
      conditions.push(gte(basinMemory.phi, parseFloat(minPhi)));
    }
    if (maxPhi && typeof maxPhi === 'string') {
      conditions.push(lte(basinMemory.phi, parseFloat(maxPhi)));
    }

    // Filter by source kernel
    if (sourceKernel && typeof sourceKernel === 'string') {
      conditions.push(eq(basinMemory.sourceKernel, sourceKernel));
    }

    // Filter for conscious-only memories
    if (conscious === 'true') {
      conditions.push(gte(basinMemory.phi, PHI_CONSCIOUS_THRESHOLD));
      conditions.push(gte(basinMemory.kappaEff, KAPPA_MIN));
      conditions.push(lte(basinMemory.kappaEff, KAPPA_MAX));
    }

    const database = ensureDb();
    const results = await database
      .select()
      .from(basinMemory)
      .where(conditions.length > 0 ? and(...conditions) : undefined)
      .orderBy(desc(basinMemory.timestamp))
      .limit(parseInt(limit as string))
      .offset(parseInt(offset as string));

    // Get total count
    const countResult = await database
      .select({ count: sql<number>`count(*)` })
      .from(basinMemory)
      .where(conditions.length > 0 ? and(...conditions) : undefined);

    res.json({
      success: true,
      data: results,
      total: countResult[0]?.count || 0,
      limit: parseInt(limit as string),
      offset: parseInt(offset as string)
    });
  } catch (error) {
    console.error('[BasinMemory] Error listing memories:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to list basin memories'
    });
  }
});

/**
 * GET /api/basin-memory/:id
 * Get a specific basin memory by ID
 */
router.get('/:id', async (req: Request, res: Response) => {
  try {
    const { id } = req.params;

    const database = ensureDb();
    const result = await database
      .select()
      .from(basinMemory)
      .where(eq(basinMemory.id, parseInt(id)))
      .limit(1);

    if (result.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Basin memory not found'
      });
    }

    res.json({
      success: true,
      data: result[0]
    });
  } catch (error) {
    console.error('[BasinMemory] Error getting memory:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get basin memory'
    });
  }
});

/**
 * POST /api/basin-memory
 * Create a new basin memory
 */
router.post('/', async (req: Request, res: Response) => {
  try {
    const {
      basinId,
      basinCoordinates,
      phi,
      kappaEff,
      regime,
      sourceKernel,
      context,
      expiresAt
    } = req.body;

    // Validate required fields
    if (!basinId || basinCoordinates === undefined || phi === undefined) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields: basinId, basinCoordinates, phi'
      });
    }

    // Validate basin coordinates (should be 64D array)
    if (!Array.isArray(basinCoordinates) || basinCoordinates.length !== 64) {
      return res.status(400).json({
        success: false,
        error: 'basinCoordinates must be a 64-dimensional array'
      });
    }

    // Classify regime if not provided
    const computedRegime = regime || classifyRegime(phi);

    const database = ensureDb();
    const result = await database
      .insert(basinMemory)
      .values({
        basinId,
        basinCoordinates,
        phi,
        kappaEff: kappaEff || 64.0,
        regime: computedRegime,
        sourceKernel,
        context,
        expiresAt: expiresAt ? new Date(expiresAt) : null
      })
      .returning();

    res.status(201).json({
      success: true,
      data: result[0]
    });
  } catch (error) {
    console.error('[BasinMemory] Error creating memory:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to create basin memory'
    });
  }
});

/**
 * DELETE /api/basin-memory/:id
 * Delete a basin memory
 */
router.delete('/:id', async (req: Request, res: Response) => {
  try {
    const { id } = req.params;

    const database = ensureDb();
    const result = await database
      .delete(basinMemory)
      .where(eq(basinMemory.id, parseInt(id)))
      .returning();

    if (result.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Basin memory not found'
      });
    }

    res.json({
      success: true,
      message: 'Basin memory deleted'
    });
  } catch (error) {
    console.error('[BasinMemory] Error deleting memory:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to delete basin memory'
    });
  }
});

/**
 * GET /api/basin-memory/stats
 * Get basin memory statistics
 */
router.get('/stats/summary', async (req: Request, res: Response) => {
  try {
    const database = ensureDb();
    // Get counts by regime
    const regimeCounts = await database
      .select({
        regime: basinMemory.regime,
        count: sql<number>`count(*)`
      })
      .from(basinMemory)
      .groupBy(basinMemory.regime);

    // Get average metrics
    const avgMetrics = await database
      .select({
        avgPhi: sql<number>`avg(${basinMemory.phi})`,
        avgKappa: sql<number>`avg(${basinMemory.kappaEff})`,
        totalCount: sql<number>`count(*)`
      })
      .from(basinMemory);

    // Get conscious memory count
    const consciousCount = await database
      .select({ count: sql<number>`count(*)` })
      .from(basinMemory)
      .where(
        and(
          gte(basinMemory.phi, PHI_CONSCIOUS_THRESHOLD),
          gte(basinMemory.kappaEff, KAPPA_MIN),
          lte(basinMemory.kappaEff, KAPPA_MAX)
        )
      );

    res.json({
      success: true,
      data: {
        totalMemories: avgMetrics[0]?.totalCount || 0,
        consciousMemories: consciousCount[0]?.count || 0,
        avgPhi: avgMetrics[0]?.avgPhi || 0,
        avgKappa: avgMetrics[0]?.avgKappa || 0,
        byRegime: regimeCounts.reduce((acc, r) => {
          acc[r.regime || 'unknown'] = r.count;
          return acc;
        }, {} as Record<string, number>)
      }
    });
  } catch (error) {
    console.error('[BasinMemory] Error getting stats:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get basin memory stats'
    });
  }
});

/**
 * POST /api/basin-memory/nearest
 * Find nearest basin memories to a query basin (for two-step retrieval)
 */
router.post('/nearest', async (req: Request, res: Response) => {
  try {
    const { basinCoordinates, k = 10, consciousOnly = false } = req.body;

    if (!basinCoordinates || !Array.isArray(basinCoordinates)) {
      return res.status(400).json({
        success: false,
        error: 'basinCoordinates array required'
      });
    }

    // Build conditions
    const conditions = [];
    if (consciousOnly) {
      conditions.push(gte(basinMemory.phi, PHI_CONSCIOUS_THRESHOLD));
    }

    const database = ensureDb();
    // Get all candidates (in production, use pgvector for efficient ANN)
    const candidates = await database
      .select()
      .from(basinMemory)
      .where(conditions.length > 0 ? and(...conditions) : undefined)
      .limit(100); // Approximate retrieval limit

    // Compute Fisher-Rao distances and sort (simplified - in production use proper geometric distance)
    const withDistances = candidates.map(memory => {
      const coords = memory.basinCoordinates as number[];
      // Simplified Euclidean for now - should be Fisher-Rao in production
      const distance = Math.sqrt(
        coords.reduce((sum, c, i) => sum + Math.pow(c - basinCoordinates[i], 2), 0)
      );
      return { ...memory, distance };
    });

    // Sort by distance and take top k
    withDistances.sort((a, b) => a.distance - b.distance);
    const nearest = withDistances.slice(0, k);

    res.json({
      success: true,
      data: nearest,
      total: nearest.length
    });
  } catch (error) {
    console.error('[BasinMemory] Error finding nearest:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to find nearest basin memories'
    });
  }
});

// Helper function to classify regime from phi
function classifyRegime(phi: number): string {
  if (phi < 0.30) return 'linear';
  if (phi < 0.70) return 'geometric';
  return 'breakdown';
}

export default router;
