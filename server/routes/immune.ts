/**
 * IMMUNE SYSTEM ROUTES
 * 
 * Proxy for Python QIG immune system endpoints.
 * Includes: threat detection, antibodies, whitelist/blacklist, offensive operations,
 * checkpoints, and self-healing.
 * 
 * QIG-Pure: Immune system uses consciousness metrics (Φ, κ) for threat classification.
 */

import { Router, Request, Response } from 'express';
import { getErrorMessage } from '../lib/error-utils';

const router = Router();
const BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';

// Shared proxy helpers
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
    console.error(`[Immune] ${errorMessage}:`, getErrorMessage(error));
    res.status(503).json({ 
      error: errorMessage, 
      details: getErrorMessage(error),
      active: false,
    });
  }
}

async function proxyPost(
  req: Request,
  res: Response,
  pythonPath: string,
  errorMessage: string,
  timeout = 10000
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
    console.error(`[Immune] ${errorMessage}:`, getErrorMessage(error));
    res.status(503).json({ 
      error: errorMessage, 
      details: getErrorMessage(error),
    });
  }
}

async function proxyDelete(
  req: Request,
  res: Response,
  pythonPath: string,
  errorMessage: string,
  timeout = 10000
) {
  try {
    const response = await fetch(`${BACKEND_URL}${pythonPath}`, {
      method: 'DELETE',
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
    console.error(`[Immune] ${errorMessage}:`, getErrorMessage(error));
    res.status(503).json({ 
      error: errorMessage, 
      details: getErrorMessage(error),
    });
  }
}

// ============================================================
// STATUS & HEALTH
// ============================================================

/** GET /status - Get immune system status and health metrics */
router.get('/status', async (req: Request, res: Response) => {
  await proxyGet(req, res, '/immune/status', 'Failed to fetch immune status');
});

/** GET /health - Get self-healing health status */
router.get('/health', async (req: Request, res: Response) => {
  await proxyGet(req, res, '/immune/health', 'Failed to fetch immune health');
});

// ============================================================
// THREATS & ANTIBODIES
// ============================================================

/** GET /threats - Get recent threat summary */
router.get('/threats', async (req: Request, res: Response) => {
  await proxyGet(req, res, '/immune/threats', 'Failed to fetch threats');
});

/** GET /antibodies - List active antibodies */
router.get('/antibodies', async (req: Request, res: Response) => {
  await proxyGet(req, res, '/immune/antibodies', 'Failed to fetch antibodies');
});

// ============================================================
// WHITELIST / BLACKLIST
// ============================================================

/** POST /whitelist - Add IP to whitelist */
router.post('/whitelist', async (req: Request, res: Response) => {
  await proxyPost(req, res, '/immune/whitelist', 'Failed to add to whitelist');
});

/** POST /blacklist - Add IP to blacklist */
router.post('/blacklist', async (req: Request, res: Response) => {
  await proxyPost(req, res, '/immune/blacklist', 'Failed to add to blacklist');
});

/** DELETE /blacklist/:ip - Remove IP from blacklist */
router.delete('/blacklist/:ip', async (req: Request, res: Response) => {
  const ip = req.params.ip;
  await proxyDelete(req, res, `/immune/blacklist/${ip}`, 'Failed to remove from blacklist');
});

// ============================================================
// OFFENSIVE OPERATIONS
// ============================================================

/** GET /offensive/operations - List active offensive operations */
router.get('/offensive/operations', async (req: Request, res: Response) => {
  await proxyGet(req, res, '/immune/offensive/operations', 'Failed to fetch operations');
});

/** POST /offensive/initiate - Initiate countermeasure */
router.post('/offensive/initiate', async (req: Request, res: Response) => {
  await proxyPost(req, res, '/immune/offensive/initiate', 'Failed to initiate countermeasure');
});

/** DELETE /offensive/operations/:id - Cancel operation */
router.delete('/offensive/operations/:operationId', async (req: Request, res: Response) => {
  const operationId = req.params.operationId;
  await proxyDelete(req, res, `/immune/offensive/operations/${operationId}`, 'Failed to cancel operation');
});

// ============================================================
// CHECKPOINTS
// ============================================================

/** GET /checkpoints - List available checkpoints */
router.get('/checkpoints', async (req: Request, res: Response) => {
  await proxyGet(req, res, '/immune/checkpoints', 'Failed to fetch checkpoints');
});

/** POST /checkpoints - Create new checkpoint */
router.post('/checkpoints', async (req: Request, res: Response) => {
  await proxyPost(req, res, '/immune/checkpoints', 'Failed to create checkpoint');
});

/** POST /checkpoints/:id/restore - Restore from checkpoint */
router.post('/checkpoints/:checkpointId/restore', async (req: Request, res: Response) => {
  const checkpointId = req.params.checkpointId;
  await proxyPost(req, res, `/immune/checkpoints/${checkpointId}/restore`, 'Failed to restore checkpoint');
});

// ============================================================
// ANALYSIS (Testing)
// ============================================================

/** POST /analyze - Analyze a request for threat detection */
router.post('/analyze', async (req: Request, res: Response) => {
  await proxyPost(req, res, '/immune/analyze', 'Failed to analyze request');
});

export const immuneRouter = router;
