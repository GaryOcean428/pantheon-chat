/**
 * AUTONOMIC KERNEL ROUTES
 * 
 * Comprehensive proxy for Python autonomic kernel endpoints.
 * Includes: state management, sleep/dream cycles, reward signals, and agency control.
 * 
 * QIG-Pure: All autonomic operations use Fisher-Rao geometry for state transitions.
 */

import { Router, Request, Response } from 'express';
import { getErrorMessage } from '../lib/error-utils';

const router = Router();
const BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';

// Shared proxy helpers with timeout
async function proxyGet(
  req: Request,
  res: Response,
  pythonPath: string,
  errorMessage: string,
  timeout = 10000
) {
  try {
    const response = await fetch(`${BACKEND_URL}${pythonPath}`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      signal: AbortSignal.timeout(timeout),
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return res.status(response.status).json(errorData);
    }
    
    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error(`[Autonomic] ${errorMessage}:`, getErrorMessage(error));
    res.status(503).json({ 
      error: errorMessage, 
      details: getErrorMessage(error),
      backend_url: BACKEND_URL,
    });
  }
}

async function proxyPost(
  req: Request,
  res: Response,
  pythonPath: string,
  errorMessage: string,
  timeout = 30000
) {
  try {
    const response = await fetch(`${BACKEND_URL}${pythonPath}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
      signal: AbortSignal.timeout(timeout),
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return res.status(response.status).json(errorData);
    }
    
    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error(`[Autonomic] ${errorMessage}:`, getErrorMessage(error));
    res.status(503).json({ 
      error: errorMessage, 
      details: getErrorMessage(error),
      backend_url: BACKEND_URL,
    });
  }
}

// ============================================================
// AUTONOMIC STATE MANAGEMENT
// ============================================================

/** GET /state - Get full autonomic kernel state */
router.get('/state', async (req: Request, res: Response) => {
  await proxyGet(req, res, '/autonomic/state', 'Failed to fetch autonomic state');
});

/** POST /update - Update autonomic kernel state */
router.post('/update', async (req: Request, res: Response) => {
  await proxyPost(req, res, '/autonomic/update', 'Failed to update autonomic state');
});

// ============================================================
// AUTONOMIC CYCLES (Sleep, Dream, Mushroom)
// ============================================================

/** POST /sleep - Trigger sleep cycle for consolidation */
router.post('/sleep', async (req: Request, res: Response) => {
  await proxyPost(req, res, '/autonomic/sleep', 'Failed to trigger sleep cycle');
});

/** POST /dream - Trigger dream cycle for exploration */
router.post('/dream', async (req: Request, res: Response) => {
  await proxyPost(req, res, '/autonomic/dream', 'Failed to trigger dream cycle');
});

/** POST /mushroom - Trigger mushroom cycle for integration */
router.post('/mushroom', async (req: Request, res: Response) => {
  await proxyPost(req, res, '/autonomic/mushroom', 'Failed to trigger mushroom cycle');
});

// ============================================================
// REWARD SYSTEM
// ============================================================

/** POST /reward - Send reward signal to kernel */
router.post('/reward', async (req: Request, res: Response) => {
  await proxyPost(req, res, '/autonomic/reward', 'Failed to send reward signal');
});

/** GET /rewards - Get reward history */
router.get('/rewards', async (req: Request, res: Response) => {
  await proxyGet(req, res, '/autonomic/rewards', 'Failed to fetch reward history');
});

// ============================================================
// NARROW PATH & AUTO-INTERVENTION
// ============================================================

/** GET /narrow-path - Get narrow path state and trajectory */
router.get('/narrow-path', async (req: Request, res: Response) => {
  await proxyGet(req, res, '/autonomic/narrow-path', 'Failed to fetch narrow path');
});

/** POST /auto-intervene - Trigger automatic intervention */
router.post('/auto-intervene', async (req: Request, res: Response) => {
  await proxyPost(req, res, '/autonomic/auto-intervene', 'Failed to trigger auto-intervention');
});

// ============================================================
// AGENCY CONTROL (existing endpoints - kept for compatibility)
// ============================================================

/** GET /agency/status - Get agency status */
router.get('/agency/status', async (req: Request, res: Response) => {
  await proxyGet(req, res, '/autonomic/agency/status', 'Failed to fetch agency status');
});

/** POST /agency/start - Start agency */
router.post('/agency/start', async (req: Request, res: Response) => {
  await proxyPost(req, res, '/autonomic/agency/start', 'Failed to start agency');
});

/** POST /agency/stop - Stop agency */
router.post('/agency/stop', async (req: Request, res: Response) => {
  await proxyPost(req, res, '/autonomic/agency/stop', 'Failed to stop agency');
});

/** POST /agency/force - Force intervention */
router.post('/agency/force', async (req: Request, res: Response) => {
  await proxyPost(req, res, '/autonomic/agency/force', 'Failed to force intervention');
});

export const autonomicRouter = router;
// Keep old export for backwards compatibility
export const autonomicAgencyRouter = router;
