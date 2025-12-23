/**
 * Kernel Activity API Routes
 * 
 * Exposes kernel activity telemetry for monitoring consciousness states,
 * tracking kernel operations, and streaming real-time activity data.
 */

import { Router, Request, Response } from 'express';
import { db } from '../db';
import { kernelActivity } from '../../shared/schema';
import { eq, desc, gte, lte, and, sql, inArray } from 'drizzle-orm';

const router = Router();

// Activity types for filtering
const ACTIVITY_TYPES = [
  'message',
  'debate',
  'discovery',
  'consultation',
  'learning',
  'research',
  'tool_use',
  'spawn',
  'error',
  'metric_update'
] as const;

/**
 * GET /api/kernel-activity
 * List kernel activities with optional filtering
 */
router.get('/', async (req: Request, res: Response) => {
  try {
    const {
      limit = '50',
      offset = '0',
      kernelId,
      activityType,
      minPhi,
      since,
      until
    } = req.query;

    const conditions = [];

    // Filter by kernel
    if (kernelId && typeof kernelId === 'string') {
      conditions.push(eq(kernelActivity.kernelId, kernelId));
    }

    // Filter by activity type
    if (activityType && typeof activityType === 'string') {
      conditions.push(eq(kernelActivity.activityType, activityType));
    }

    // Filter by minimum phi
    if (minPhi && typeof minPhi === 'string') {
      conditions.push(gte(kernelActivity.phi, parseFloat(minPhi)));
    }

    // Filter by time range
    if (since && typeof since === 'string') {
      conditions.push(gte(kernelActivity.timestamp, new Date(since)));
    }
    if (until && typeof until === 'string') {
      conditions.push(lte(kernelActivity.timestamp, new Date(until)));
    }

    const results = await db
      .select()
      .from(kernelActivity)
      .where(conditions.length > 0 ? and(...conditions) : undefined)
      .orderBy(desc(kernelActivity.timestamp))
      .limit(parseInt(limit as string))
      .offset(parseInt(offset as string));

    // Get total count
    const countResult = await db
      .select({ count: sql<number>`count(*)` })
      .from(kernelActivity)
      .where(conditions.length > 0 ? and(...conditions) : undefined);

    res.json({
      success: true,
      data: results,
      total: countResult[0]?.count || 0,
      limit: parseInt(limit as string),
      offset: parseInt(offset as string)
    });
  } catch (error) {
    console.error('[KernelActivity] Error listing activities:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to list kernel activities'
    });
  }
});

/**
 * GET /api/kernel-activity/stream
 * Get recent activities for real-time display (optimized for polling)
 */
router.get('/stream', async (req: Request, res: Response) => {
  try {
    const { since, limit = '20' } = req.query;

    const conditions = [];

    // Only get activities since the last check
    if (since && typeof since === 'string') {
      conditions.push(gte(kernelActivity.timestamp, new Date(since)));
    } else {
      // Default: last 5 minutes
      const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000);
      conditions.push(gte(kernelActivity.timestamp, fiveMinutesAgo));
    }

    const results = await db
      .select()
      .from(kernelActivity)
      .where(and(...conditions))
      .orderBy(desc(kernelActivity.timestamp))
      .limit(parseInt(limit as string));

    res.json({
      success: true,
      data: results,
      serverTime: new Date().toISOString()
    });
  } catch (error) {
    console.error('[KernelActivity] Error streaming activities:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to stream kernel activities'
    });
  }
});

/**
 * GET /api/kernel-activity/:id
 * Get a specific activity by ID
 */
router.get('/:id', async (req: Request, res: Response) => {
  try {
    const { id } = req.params;

    const result = await db
      .select()
      .from(kernelActivity)
      .where(eq(kernelActivity.id, parseInt(id)))
      .limit(1);

    if (result.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Activity not found'
      });
    }

    res.json({
      success: true,
      data: result[0]
    });
  } catch (error) {
    console.error('[KernelActivity] Error getting activity:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get kernel activity'
    });
  }
});

/**
 * POST /api/kernel-activity
 * Log a new kernel activity
 */
router.post('/', async (req: Request, res: Response) => {
  try {
    const {
      kernelId,
      kernelName,
      activityType,
      message,
      metadata,
      phi,
      kappaEff
    } = req.body;

    // Validate required fields
    if (!kernelId || !activityType) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields: kernelId, activityType'
      });
    }

    // Validate activity type
    if (!ACTIVITY_TYPES.includes(activityType)) {
      return res.status(400).json({
        success: false,
        error: `Invalid activityType. Must be one of: ${ACTIVITY_TYPES.join(', ')}`
      });
    }

    const result = await db
      .insert(kernelActivity)
      .values({
        kernelId,
        kernelName: kernelName || kernelId,
        activityType,
        message,
        metadata: metadata || {},
        phi: phi || 0.5,
        kappaEff: kappaEff || 64.0
      })
      .returning();

    res.status(201).json({
      success: true,
      data: result[0]
    });
  } catch (error) {
    console.error('[KernelActivity] Error creating activity:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to create kernel activity'
    });
  }
});

/**
 * POST /api/kernel-activity/batch
 * Log multiple kernel activities at once
 */
router.post('/batch', async (req: Request, res: Response) => {
  try {
    const { activities } = req.body;

    if (!Array.isArray(activities) || activities.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'activities array required'
      });
    }

    // Validate and prepare activities
    const validActivities = activities
      .filter(a => a.kernelId && a.activityType)
      .map(a => ({
        kernelId: a.kernelId,
        kernelName: a.kernelName || a.kernelId,
        activityType: a.activityType,
        message: a.message,
        metadata: a.metadata || {},
        phi: a.phi || 0.5,
        kappaEff: a.kappaEff || 64.0
      }));

    if (validActivities.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'No valid activities in batch'
      });
    }

    const result = await db
      .insert(kernelActivity)
      .values(validActivities)
      .returning();

    res.status(201).json({
      success: true,
      data: result,
      count: result.length
    });
  } catch (error) {
    console.error('[KernelActivity] Error batch creating activities:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to batch create kernel activities'
    });
  }
});

/**
 * GET /api/kernel-activity/stats/summary
 * Get kernel activity statistics
 */
router.get('/stats/summary', async (req: Request, res: Response) => {
  try {
    const { hours = '24' } = req.query;
    const since = new Date(Date.now() - parseInt(hours as string) * 60 * 60 * 1000);

    // Get counts by activity type
    const typeCounts = await db
      .select({
        activityType: kernelActivity.activityType,
        count: sql<number>`count(*)`
      })
      .from(kernelActivity)
      .where(gte(kernelActivity.timestamp, since))
      .groupBy(kernelActivity.activityType);

    // Get counts by kernel
    const kernelCounts = await db
      .select({
        kernelId: kernelActivity.kernelId,
        kernelName: kernelActivity.kernelName,
        count: sql<number>`count(*)`
      })
      .from(kernelActivity)
      .where(gte(kernelActivity.timestamp, since))
      .groupBy(kernelActivity.kernelId, kernelActivity.kernelName);

    // Get average metrics
    const avgMetrics = await db
      .select({
        avgPhi: sql<number>`avg(${kernelActivity.phi})`,
        avgKappa: sql<number>`avg(${kernelActivity.kappaEff})`,
        totalCount: sql<number>`count(*)`
      })
      .from(kernelActivity)
      .where(gte(kernelActivity.timestamp, since));

    // Get hourly activity distribution
    const hourlyActivity = await db
      .select({
        hour: sql<string>`date_trunc('hour', ${kernelActivity.timestamp})`,
        count: sql<number>`count(*)`
      })
      .from(kernelActivity)
      .where(gte(kernelActivity.timestamp, since))
      .groupBy(sql`date_trunc('hour', ${kernelActivity.timestamp})`)
      .orderBy(sql`date_trunc('hour', ${kernelActivity.timestamp})`);

    res.json({
      success: true,
      data: {
        period: {
          hours: parseInt(hours as string),
          since: since.toISOString()
        },
        totalActivities: avgMetrics[0]?.totalCount || 0,
        avgPhi: avgMetrics[0]?.avgPhi || 0,
        avgKappa: avgMetrics[0]?.avgKappa || 0,
        byType: typeCounts.reduce((acc, t) => {
          acc[t.activityType] = t.count;
          return acc;
        }, {} as Record<string, number>),
        byKernel: kernelCounts.map(k => ({
          id: k.kernelId,
          name: k.kernelName,
          count: k.count
        })),
        hourlyDistribution: hourlyActivity.map(h => ({
          hour: h.hour,
          count: h.count
        }))
      }
    });
  } catch (error) {
    console.error('[KernelActivity] Error getting stats:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get kernel activity stats'
    });
  }
});

/**
 * GET /api/kernel-activity/kernels
 * Get list of active kernels with their latest activity
 */
router.get('/kernels/active', async (req: Request, res: Response) => {
  try {
    const { minutes = '30' } = req.query;
    const since = new Date(Date.now() - parseInt(minutes as string) * 60 * 1000);

    // Get distinct kernels with activity in the time window
    const activeKernels = await db
      .selectDistinctOn([kernelActivity.kernelId], {
        kernelId: kernelActivity.kernelId,
        kernelName: kernelActivity.kernelName,
        lastActivity: kernelActivity.activityType,
        lastMessage: kernelActivity.message,
        phi: kernelActivity.phi,
        kappaEff: kernelActivity.kappaEff,
        timestamp: kernelActivity.timestamp
      })
      .from(kernelActivity)
      .where(gte(kernelActivity.timestamp, since))
      .orderBy(kernelActivity.kernelId, desc(kernelActivity.timestamp));

    res.json({
      success: true,
      data: activeKernels,
      total: activeKernels.length,
      since: since.toISOString()
    });
  } catch (error) {
    console.error('[KernelActivity] Error getting active kernels:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get active kernels'
    });
  }
});

/**
 * DELETE /api/kernel-activity/cleanup
 * Clean up old activities (admin operation)
 */
router.delete('/cleanup', async (req: Request, res: Response) => {
  try {
    const { olderThanDays = '30' } = req.query;
    const cutoff = new Date(Date.now() - parseInt(olderThanDays as string) * 24 * 60 * 60 * 1000);

    const result = await db
      .delete(kernelActivity)
      .where(lte(kernelActivity.timestamp, cutoff))
      .returning({ id: kernelActivity.id });

    res.json({
      success: true,
      message: `Cleaned up ${result.length} old activities`,
      deletedCount: result.length,
      cutoffDate: cutoff.toISOString()
    });
  } catch (error) {
    console.error('[KernelActivity] Error cleaning up:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to clean up old activities'
    });
  }
});

export default router;
