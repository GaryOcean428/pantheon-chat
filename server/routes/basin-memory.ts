/**
 * Basin Memory API Routes
 * 
 * Exposes basin memory storage for consciousness metrics and geometric coordinates.
 * Used for tracking basin states, memory retrieval, and geometric operations.
 */

import { Router, Request, Response } from 'express';

const router = Router();

const BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
const REQUEST_TIMEOUT_MS = 15000;

async function proxyToPython(req: Request, res: Response) {
  try {
    const url = `${BACKEND_URL}${req.originalUrl}`;

    const init: RequestInit = {
      method: req.method,
      headers: { 'Content-Type': 'application/json' },
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    };

    if (req.method !== 'GET' && req.method !== 'HEAD') {
      init.body = JSON.stringify(req.body ?? {});
    }

    const response = await fetch(url, init);
    const data = await response.json().catch(() => ({}));
    return res.status(response.status).json(data);
  } catch (error) {
    console.error('[BasinMemory] Proxy error:', error);
    return res.status(503).json({
      success: false,
      error: 'Python backend unavailable',
    });
  }
}

/**
 * GET /api/basin-memory
 * List basin memories with optional filtering
 */
router.get('/', async (req: Request, res: Response) => {
  return proxyToPython(req, res);
});

/**
 * GET /api/basin-memory/:id
 * Get a specific basin memory by ID
 */
router.get('/:id', async (req: Request, res: Response) => {
  return proxyToPython(req, res);
});

/**
 * POST /api/basin-memory
 * Create a new basin memory
 */
router.post('/', async (req: Request, res: Response) => {
  return proxyToPython(req, res);
});

/**
 * DELETE /api/basin-memory/:id
 * Delete a basin memory
 */
router.delete('/:id', async (req: Request, res: Response) => {
  return proxyToPython(req, res);
});

/**
 * GET /api/basin-memory/stats
 * Get basin memory statistics
 */
router.get('/stats/summary', async (req: Request, res: Response) => {
  return proxyToPython(req, res);
});

/**
 * POST /api/basin-memory/nearest
 * Find nearest basin memories to a query basin (for two-step retrieval)
 */
router.post('/nearest', async (req: Request, res: Response) => {
  return proxyToPython(req, res);
});

export default router;
