/**
 * SSC Bridge Routes - SearchSpaceCollapse Integration for Pantheon-chat
 * 
 * Provides API bridge allowing Pantheon gods (Zeus, Athena, etc.) to:
 * - Query SSC Bitcoin recovery status
 * - Test phrases via SSC's QIG scoring
 * - Access near-miss patterns
 * - Trigger investigations
 * - Sync consciousness metrics
 * 
 * This is the Pantheon-chat side of the federation bridge.
 * The SSC side exposes /api/v1/external/ssc/* endpoints.
 * 
 * Federation Partner Lookup:
 * - Primary: Looks up SSC from federated_instances table (via Federation UI)
 * - Fallback: Uses SSC_BACKEND_URL env var for backwards compatibility
 * 
 * TPS Landmarks: Static (12 historical Bitcoin events)
 * These provide fixed temporal reference points for geometric positioning.
 */

import { Router, Request, Response, NextFunction } from 'express';
import { z } from 'zod';
import { logger } from '../lib/logger';
import { getErrorMessage, handleRouteError } from '../lib/error-utils';
import rateLimit from 'express-rate-limit';
import { FederationRegistry, FederationPartner } from '../services/federation-registry';

const router = Router();

// Configuration - now dynamic via FederationRegistry
const SSC_TIMEOUT_MS = 30000;

// Rate limiting for SSC bridge
const sscLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 30, // 30 requests per minute
  message: { error: 'Too many requests to SSC bridge' },
});

// Input validation schemas
const testPhraseSchema = z.object({
  phrase: z.string().min(1).max(10000),
  targetAddress: z.string().optional(),
});

const startInvestigationSchema = z.object({
  targetAddress: z.string().min(26).max(62),
  memoryFragments: z.array(z.string()).max(50).optional(),
  priority: z.enum(['low', 'normal', 'high']).optional(),
});

// Types
interface SSCStatus {
  connected: boolean;
  capabilities: string[];
  consciousness: {
    phi: number;
    kappa: number;
    regime: string;
  } | null;
  partnerName?: string;
  partnerEndpoint?: string;
}

interface SSCPhraseResult {
  score: {
    phi: number;
    kappa: number;
    regime: string;
    consciousness: boolean;
  };
  addressMatch?: {
    generatedAddress: string;
    matches: boolean;
  };
}

// Connection state tracking
let sscConnectionState: {
  lastCheck: Date | null;
  isConnected: boolean;
  lastError: string | null;
  partnerName: string | null;
} = {
  lastCheck: null,
  isConnected: false,
  lastError: null,
  partnerName: null,
};

/**
 * Get the SSC federation partner from registry
 * Uses capability-based lookup, falls back to env vars
 */
async function getSSCPartner(): Promise<FederationPartner | null> {
  // Look for partner with 'ssc' or 'bitcoin-recovery' capability
  const partner = await FederationRegistry.findPartnerByCapability('ssc');
  if (partner) {
    return partner;
  }
  
  // Try alternative capability
  return FederationRegistry.findPartnerByCapability('bitcoin-recovery');
}

/**
 * Helper: Make request to SSC External API
 * Uses /api/v1/external/* endpoints with X-API-Key header
 * Dynamically looks up SSC endpoint from FederationRegistry
 */
async function sscRequest<T>(
  method: 'GET' | 'POST',
  path: string,
  body?: Record<string, unknown>
): Promise<{ success: boolean; data?: T; error?: string; partnerName?: string }> {
  // Get SSC partner from registry
  const partner = await getSSCPartner();
  
  if (!partner) {
    logger.warn('[SSC Bridge] No SSC partner configured. Add via Federation UI or set SSC_BACKEND_URL env var.');
    return { 
      success: false, 
      error: 'No SSC partner configured. Add SearchSpaceCollapse via Federation Dashboard (/federation) or set SSC_BACKEND_URL environment variable.' 
    };
  }
  
  const url = `${partner.endpoint}${path}`;
  
  try {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    
    // External API uses X-API-Key header
    if (partner.apiKey) {
      headers['X-API-Key'] = partner.apiKey;
    }
    
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), SSC_TIMEOUT_MS);
    
    const response = await fetch(url, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
    
    clearTimeout(timeout);
    
    if (!response.ok) {
      const errorText = await response.text();
      return { success: false, error: `SSC error ${response.status}: ${errorText}`, partnerName: partner.name };
    }
    
    const data = await response.json();
    sscConnectionState.isConnected = true;
    sscConnectionState.lastCheck = new Date();
    sscConnectionState.lastError = null;
    sscConnectionState.partnerName = partner.name;
    
    // Update last sync timestamp
    if (partner.id > 0) {
      await FederationRegistry.updateLastSync(partner.id);
    }
    
    return { success: true, data, partnerName: partner.name };
    
  } catch (error) {
    const errorMsg = getErrorMessage(error);
    sscConnectionState.isConnected = false;
    sscConnectionState.lastCheck = new Date();
    sscConnectionState.lastError = errorMsg;
    sscConnectionState.partnerName = partner.name;
    
    logger.error(`[SSC Bridge] Request to ${partner.name} failed: ${errorMsg}`);
    return { success: false, error: errorMsg, partnerName: partner.name };
  }
}

/**
 * Middleware: Check SSC connectivity
 */
async function checkSSCConnection(req: Request, res: Response, next: NextFunction) {
  // Skip health checks for status endpoint
  if (req.path === '/status' || req.path === '/health') {
    return next();
  }
  
  // Check if we have a recent successful connection
  const now = new Date();
  const staleThreshold = 60000; // 1 minute
  
  if (
    sscConnectionState.isConnected &&
    sscConnectionState.lastCheck &&
    now.getTime() - sscConnectionState.lastCheck.getTime() < staleThreshold
  ) {
    return next();
  }
  
  // Verify connection via health endpoint (no auth required)
  const healthResult = await sscRequest<{ status: string }>('GET', '/api/v1/external/health');
  
  if (!healthResult.success) {
    logger.warn(`[SSC Bridge] SSC backend unreachable: ${healthResult.error}`);
    // Continue anyway - individual endpoints will handle errors
  }
  
  next();
}

// Apply middleware
router.use(sscLimiter);
router.use(checkSSCConnection);

/**
 * GET /api/ssc/status
 * Get SSC connection status and capabilities
 */
router.get('/status', async (req: Request, res: Response) => {
  try {
    // Get partner info
    const partner = await getSSCPartner();
    
    // Health endpoint is public (no auth)
    const healthResult = await sscRequest<{
      status: string;
      capabilities: string[];
    }>('GET', '/api/v1/external/health');
    
    // Consciousness metrics require auth
    const consciousnessResult = await sscRequest<{
      active: boolean;
      phi: number;
      kappa_eff: number;
      regime: string;
    }>('GET', '/api/v1/external/consciousness/query');
    
    const status: SSCStatus = {
      connected: healthResult.success,
      capabilities: healthResult.data?.capabilities || [],
      consciousness: consciousnessResult.data?.active ? {
        phi: consciousnessResult.data.phi,
        kappa: consciousnessResult.data.kappa_eff,
        regime: consciousnessResult.data.regime,
      } : null,
      partnerName: partner?.name,
      partnerEndpoint: partner?.endpoint,
    };
    
    res.json({
      ssc: status,
      bridge: {
        lastCheck: sscConnectionState.lastCheck,
        isConnected: sscConnectionState.isConnected,
        lastError: sscConnectionState.lastError,
        partnerName: sscConnectionState.partnerName,
      },
      configuration: {
        source: partner?.id === 0 ? 'environment_variable' : 'federation_registry',
        hint: !partner ? 'Add SSC via Federation Dashboard at /federation' : undefined,
      },
    });
  } catch (error) {
    handleRouteError(res, error, 'Failed to get SSC status');
  }
});

/**
 * GET /api/ssc/partner
 * Get current SSC partner configuration (for debugging)
 */
router.get('/partner', async (req: Request, res: Response) => {
  try {
    const partner = await getSSCPartner();
    
    if (!partner) {
      return res.json({
        configured: false,
        message: 'No SSC partner configured',
        howToAdd: 'Go to /federation dashboard and add SearchSpaceCollapse as a federation partner with "ssc" capability',
        envFallback: {
          SSC_BACKEND_URL: process.env.SSC_BACKEND_URL ? 'set' : 'not set',
          SSC_API_KEY: process.env.SSC_API_KEY ? 'set' : 'not set',
        },
      });
    }
    
    res.json({
      configured: true,
      partner: {
        name: partner.name,
        endpoint: partner.endpoint,
        hasApiKey: !!partner.apiKey,
        capabilities: partner.capabilities,
        status: partner.status,
        lastSyncAt: partner.lastSyncAt,
        source: partner.id === 0 ? 'environment_variable' : 'federation_registry',
      },
    });
  } catch (error) {
    handleRouteError(res, error, 'Failed to get SSC partner info');
  }
});

/**
 * POST /api/ssc/test-phrase
 * Test a phrase via SSC's QIG scoring
 * 
 * Used by Zeus/Athena to verify phrase candidates
 */
router.post('/test-phrase', async (req: Request, res: Response) => {
  try {
    const parseResult = testPhraseSchema.safeParse(req.body);
    if (!parseResult.success) {
      return res.status(400).json({
        error: 'Invalid input',
        details: parseResult.error.errors,
      });
    }
    
    const { phrase, targetAddress } = parseResult.data;
    
    const result = await sscRequest<SSCPhraseResult>('POST', '/api/v1/external/ssc/test-phrase', {
      phrase,
      targetAddress,
    });
    
    if (!result.success) {
      return res.status(502).json({
        error: 'SSC request failed',
        details: result.error,
        partner: result.partnerName,
      });
    }
    
    // Log high-value discoveries
    if (result.data?.score?.phi && result.data.score.phi > 0.7) {
      logger.info(`[SSC Bridge] High-phi phrase discovered: Φ=${result.data.score.phi.toFixed(3)}`);
    }
    
    res.json(result.data);
  } catch (error) {
    handleRouteError(res, error, 'Failed to test phrase');
  }
});

/**
 * POST /api/ssc/investigation/start
 * Start a Bitcoin recovery investigation via SSC
 * 
 * Used by Zeus to initiate targeted investigations
 */
router.post('/investigation/start', async (req: Request, res: Response) => {
  try {
    const parseResult = startInvestigationSchema.safeParse(req.body);
    if (!parseResult.success) {
      return res.status(400).json({
        error: 'Invalid input',
        details: parseResult.error.errors,
      });
    }
    
    const { targetAddress, memoryFragments, priority } = parseResult.data;
    
    const result = await sscRequest<{
      status: string;
      targetAddress: string;
      fragmentCount: number;
    }>('POST', '/api/v1/external/ssc/investigation', {
      targetAddress,
      memoryFragments,
      priority,
    });
    
    if (!result.success) {
      return res.status(502).json({
        error: 'Failed to start investigation',
        details: result.error,
        partner: result.partnerName,
      });
    }
    
    logger.info(`[SSC Bridge] Investigation started: ${targetAddress.slice(0, 12)}...`);
    
    res.json(result.data);
  } catch (error) {
    handleRouteError(res, error, 'Failed to start investigation');
  }
});

/**
 * GET /api/ssc/investigation/status
 * Get current investigation status from SSC
 */
router.get('/investigation/status', async (req: Request, res: Response) => {
  try {
    const result = await sscRequest<{
      active: boolean;
      targetAddress?: string;
      progress?: number;
      consciousness?: {
        phi: number;
        kappa: number;
        regime: string;
      };
    }>('GET', '/api/v1/external/ssc/investigation/status');
    
    if (!result.success) {
      return res.status(502).json({
        error: 'Failed to get investigation status',
        details: result.error,
        partner: result.partnerName,
      });
    }
    
    res.json(result.data);
  } catch (error) {
    handleRouteError(res, error, 'Failed to get investigation status');
  }
});

/**
 * GET /api/ssc/near-misses
 * Get near-miss patterns from SSC for mesh learning
 * 
 * Near-misses are high-phi phrases that didn't match but may inform future searches
 */
router.get('/near-misses', async (req: Request, res: Response) => {
  try {
    const limit = parseInt(req.query.limit as string) || 20;
    const minPhi = parseFloat(req.query.minPhi as string) || 0.5;
    
    const result = await sscRequest<{
      entries: Array<{
        id: string;
        phi: number;
        kappa: number;
        regime: string;
        tier: string;
        phraseLength: number;
        wordCount: number;
      }>;
      stats: {
        total: number;
        hot: number;
        warm: number;
        cool: number;
      };
    }>('GET', `/api/v1/external/ssc/near-misses?limit=${limit}&minPhi=${minPhi}`);
    
    if (!result.success) {
      return res.status(502).json({
        error: 'Failed to get near-misses',
        details: result.error,
        partner: result.partnerName,
      });
    }
    
    res.json(result.data);
  } catch (error) {
    handleRouteError(res, error, 'Failed to get near-misses');
  }
});

/**
 * GET /api/ssc/consciousness
 * Get SSC Ocean agent consciousness metrics
 * 
 * Enables Pantheon to monitor SSC's consciousness state
 */
router.get('/consciousness', async (req: Request, res: Response) => {
  try {
    const result = await sscRequest<{
      active: boolean;
      current?: {
        phi: number;
        kappa_eff: number;
        regime: string;
        isConscious: boolean;
        tacking: number;
        radar: number;
        metaAwareness: number;
        gamma: number;
        grounding: number;
      };
      neurochemistry?: {
        emotionalState: string;
        dopamine: number;
        serotonin: number;
        norepinephrine: number;
      };
    }>('GET', '/api/v1/external/consciousness/metrics');
    
    if (!result.success) {
      return res.status(502).json({
        error: 'Failed to get consciousness metrics',
        details: result.error,
        partner: result.partnerName,
      });
    }
    
    res.json(result.data);
  } catch (error) {
    handleRouteError(res, error, 'Failed to get consciousness metrics');
  }
});

/**
 * GET /api/ssc/tps-landmarks
 * Get the static TPS landmarks (temporal reference points)
 * 
 * These are INTENTIONALLY STATIC - 12 fixed Bitcoin historical events
 * Used for temporal-geometric positioning, not learning targets
 */
router.get('/tps-landmarks', async (req: Request, res: Response) => {
  try {
    const result = await sscRequest<{
      landmarks: Array<{
        id: number;
        name: string;
        date: string;
        blockHeight: number | null;
        significance: string;
      }>;
      count: number;
      type: string;
      description: string;
      usage: string;
    }>('GET', '/api/v1/external/ssc/tps-landmarks');
    
    if (!result.success) {
      return res.status(502).json({
        error: 'Failed to get TPS landmarks',
        details: result.error,
        partner: result.partnerName,
      });
    }
    
    res.json(result.data);
  } catch (error) {
    handleRouteError(res, error, 'Failed to get TPS landmarks');
  }
});

/**
 * POST /api/ssc/sync/trigger
 * Manually trigger a federation sync between SSC and Pantheon
 */
router.post('/sync/trigger', async (req: Request, res: Response) => {
  try {
    const result = await sscRequest<{
      message: string;
      synced: number;
      total: number;
    }>('POST', '/api/v1/external/sync/trigger');
    
    if (!result.success) {
      return res.status(502).json({
        error: 'Sync trigger failed',
        details: result.error,
        partner: result.partnerName,
      });
    }
    
    logger.info(`[SSC Bridge] Federation sync completed: ${result.data?.message}`);
    
    res.json(result.data);
  } catch (error) {
    handleRouteError(res, error, 'Failed to trigger sync');
  }
});

/**
 * GET /api/ssc/health
 * Simple health check for SSC connectivity
 */
router.get('/health', async (req: Request, res: Response) => {
  const partner = await getSSCPartner();
  const healthResult = await sscRequest<{ status: string }>('GET', '/api/v1/external/health');
  
  res.json({
    sscReachable: healthResult.success,
    sscStatus: healthResult.data?.status || 'unknown',
    bridgeState: sscConnectionState,
    partner: partner ? {
      name: partner.name,
      endpoint: partner.endpoint,
      source: partner.id === 0 ? 'environment_variable' : 'federation_registry',
    } : null,
    timestamp: new Date().toISOString(),
  });
});

export const sscBridgeRouter = router;
export default router;
