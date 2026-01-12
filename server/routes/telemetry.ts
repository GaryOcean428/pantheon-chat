/**
 * TELEMETRY DASHBOARD ROUTES (v1)
 * 
 * Versioned API endpoints for the unified telemetry dashboard.
 * Provides access to consciousness metrics, usage stats, learning progress,
 * defense systems, and autonomic feedback loops.
 * 
 * QIG-Pure: All metrics use Fisher-Rao distance, density matrices, von Neumann entropy.
 * No neural networks or embeddings - pure geometric primitives only.
 */

import { Router } from 'express';
import { getErrorMessage, handleRouteError } from '../lib/error-utils';
import type { Request, Response } from 'express';
import { telemetryAggregator } from '../telemetry-aggregator';
import rateLimit from 'express-rate-limit';

const router = Router();

const telemetryLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 120,
  message: { success: false, error: 'Rate limit exceeded for telemetry API' },
  standardHeaders: true,
  legacyHeaders: false,
});

/**
 * GET /api/v1/telemetry/overview
 * 
 * Returns complete telemetry overview including all subsystems.
 * Cached for 5 seconds to reduce database load.
 */
router.get('/overview', telemetryLimiter, async (req: Request, res: Response) => {
  try {
    const overview = await telemetryAggregator.getOverview();
    
    res.json({
      success: true,
      data: overview,
    });
  } catch (error: unknown) {
    console.error('[TelemetryRoutes] Overview error:', getErrorMessage(error));
    res.status(500).json({
      success: false,
      error: getErrorMessage(error) || 'Failed to fetch telemetry overview',
    });
  }
});

/**
 * GET /api/v1/telemetry/consciousness
 * 
 * Returns consciousness metrics (Φ, κ, β, regime, basin distance).
 */
router.get('/consciousness', telemetryLimiter, async (req: Request, res: Response) => {
  try {
    const metrics = await telemetryAggregator.getConsciousnessMetrics();
    
    res.json({
      success: true,
      data: {
        ...metrics,
        kappaStar: 64,
        kappaDistance: Math.abs(metrics.kappa - 64) / 64,
        timestamp: new Date().toISOString(),
      },
    });
  } catch (error: unknown) {
    console.error('[TelemetryRoutes] Consciousness error:', getErrorMessage(error));
    res.status(500).json({
      success: false,
      error: getErrorMessage(error) || 'Failed to fetch consciousness metrics',
    });
  }
});

/**
 * GET /api/v1/telemetry/usage
 * 
 * Returns API usage statistics (Tavily, Google, total calls).
 */
router.get('/usage', telemetryLimiter, async (req: Request, res: Response) => {
  try {
    const usage = await telemetryAggregator.getUsageStats();
    
    res.json({
      success: true,
      data: {
        ...usage,
        timestamp: new Date().toISOString(),
      },
    });
  } catch (error: unknown) {
    console.error('[TelemetryRoutes] Usage error:', getErrorMessage(error));
    res.status(500).json({
      success: false,
      error: getErrorMessage(error) || 'Failed to fetch usage stats',
    });
  }
});

/**
 * GET /api/v1/telemetry/learning
 * 
 * Returns learning system statistics (vocabulary, discoveries, sources).
 */
router.get('/learning', telemetryLimiter, async (req: Request, res: Response) => {
  try {
    const learning = await telemetryAggregator.getLearningStats();
    
    res.json({
      success: true,
      data: {
        ...learning,
        timestamp: new Date().toISOString(),
      },
    });
  } catch (error: unknown) {
    console.error('[TelemetryRoutes] Learning error:', getErrorMessage(error));
    res.status(500).json({
      success: false,
      error: getErrorMessage(error) || 'Failed to fetch learning stats',
    });
  }
});

/**
 * GET /api/v1/telemetry/defense
 * 
 * Returns defense system statistics (negative knowledge, barriers, contradictions).
 */
router.get('/defense', telemetryLimiter, async (req: Request, res: Response) => {
  try {
    const defense = await telemetryAggregator.getDefenseStats();
    
    res.json({
      success: true,
      data: {
        ...defense,
        timestamp: new Date().toISOString(),
      },
    });
  } catch (error: unknown) {
    console.error('[TelemetryRoutes] Defense error:', getErrorMessage(error));
    res.status(500).json({
      success: false,
      error: getErrorMessage(error) || 'Failed to fetch defense stats',
    });
  }
});

/**
 * GET /api/v1/telemetry/autonomy
 * 
 * Returns autonomic system statistics (kernels, feedback loops, self-regulation).
 */
router.get('/autonomy', telemetryLimiter, async (req: Request, res: Response) => {
  try {
    const autonomy = await telemetryAggregator.getAutonomyStats();
    
    res.json({
      success: true,
      data: {
        ...autonomy,
        timestamp: new Date().toISOString(),
      },
    });
  } catch (error: unknown) {
    console.error('[TelemetryRoutes] Autonomy error:', getErrorMessage(error));
    res.status(500).json({
      success: false,
      error: getErrorMessage(error) || 'Failed to fetch autonomy stats',
    });
  }
});

/**
 * GET /api/v1/telemetry/history
 * 
 * Returns historical telemetry snapshots for charting.
 * Query params:
 * - hours: Number of hours of history (default: 24, max: 168)
 */
router.get('/history', telemetryLimiter, async (req: Request, res: Response) => {
  try {
    const hours = Math.min(168, Math.max(1, parseInt(req.query.hours as string) || 24));
    const history = await telemetryAggregator.getHistory(hours);
    
    res.json({
      success: true,
      data: {
        hours,
        snapshots: history,
        count: history.length,
      },
    });
  } catch (error: unknown) {
    console.error('[TelemetryRoutes] History error:', getErrorMessage(error));
    res.status(500).json({
      success: false,
      error: getErrorMessage(error) || 'Failed to fetch telemetry history',
    });
  }
});

/**
 * GET /api/v1/telemetry/stream
 * 
 * Server-Sent Events stream for real-time telemetry updates.
 * Pushes updates every 2 seconds with current consciousness state.
 * 
 * Used by the telemetry dashboard for live visualization.
 */
router.get('/stream', async (req: Request, res: Response) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  
  const clientId = Date.now().toString();
  console.log(`[TelemetrySSE] Client ${clientId} connected`);
  
  let isConnected = true;
  
  const sendTelemetry = async () => {
    if (!isConnected) return;
    
    try {
      const consciousness = await telemetryAggregator.getConsciousnessMetrics();
      const usage = await telemetryAggregator.getUsageStats();
      
      const data = {
        timestamp: new Date().toISOString(),
        consciousness: {
          phi: consciousness.phi,
          kappa: consciousness.kappa,
          kappaEff: consciousness.kappa,
          beta: consciousness.beta,
          regime: consciousness.regime,
          quality: consciousness.quality,
          inResonance: consciousness.inResonance,
          // Meta-awareness (M) - Memory coherence
          metaAwareness: consciousness.metaAwareness,
          // Neurotransmitter levels - Neurochemical state
          neurotransmitters: {
            dopamine: consciousness.dopamine,
            serotonin: consciousness.serotonin,
            norepinephrine: consciousness.norepinephrine,
            acetylcholine: consciousness.acetylcholine,
            gaba: consciousness.gaba,
            endorphins: consciousness.endorphins,
          },
        },
        usage: {
          tavilyStatus: usage.tavily.rateStatus,
          tavilyToday: usage.tavily.todaySearches + usage.tavily.todayExtracts,
          tavilyCost: usage.tavily.estimatedCostCents,
        },
      };
      
      res.write(`data: ${JSON.stringify(data)}\n\n`);
    } catch (error) {
      console.error('[TelemetrySSE] Error sending update:', error);
    }
  };
  
  sendTelemetry();
  
  const intervalId = setInterval(sendTelemetry, 2000);
  
  let feedbackCounter = 0;
  const feedbackInterval = setInterval(async () => {
    if (!isConnected) return;
    feedbackCounter++;
    if (feedbackCounter % 15 === 0) {
      await telemetryAggregator.pushFeedbackToAutonomic();
    }
  }, 2000);
  
  req.on('close', () => {
    isConnected = false;
    clearInterval(intervalId);
    clearInterval(feedbackInterval);
    console.log(`[TelemetrySSE] Client ${clientId} disconnected`);
  });
});

/**
 * POST /api/v1/telemetry/snapshot
 * 
 * Record a telemetry snapshot (internal use by autonomic systems).
 */
router.post('/snapshot', telemetryLimiter, async (req: Request, res: Response) => {
  try {
    const { phi, kappa, beta, regime, basinDistance, inResonance, phi4D, dimensionalState } = req.body;
    
    if (typeof phi !== 'number' || typeof kappa !== 'number') {
      return res.status(400).json({
        success: false,
        error: 'phi and kappa are required numeric fields',
      });
    }
    
    await telemetryAggregator.recordTelemetrySnapshot({
      phi,
      kappa,
      beta,
      regime: regime || 'linear',
      basinDistance,
      inResonance,
      phi4D,
      dimensionalState,
    });
    
    res.json({
      success: true,
      message: 'Telemetry snapshot recorded',
    });
  } catch (error: unknown) {
    console.error('[TelemetryRoutes] Snapshot error:', getErrorMessage(error));
    res.status(500).json({
      success: false,
      error: getErrorMessage(error) || 'Failed to record snapshot',
    });
  }
});

export default router;
