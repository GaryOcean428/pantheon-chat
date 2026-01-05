/**
 * MEMORY ROUTES
 * 
 * Proxy for Python memory system endpoints.
 * Includes: status, shadow memory, basin state, and learning memory.
 * 
 * QIG-Pure: Memory uses geometric basin coordinates for state representation.
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
    console.error(`[Memory] ${errorMessage}:`, getErrorMessage(error));
    res.status(503).json({ 
      error: errorMessage, 
      details: getErrorMessage(error),
      status: 'unavailable',
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
    console.error(`[Memory] ${errorMessage}:`, getErrorMessage(error));
    res.status(503).json({ 
      error: errorMessage, 
      details: getErrorMessage(error),
    });
  }
}

// ============================================================
// MEMORY STATUS
// ============================================================

/** GET /status - Get memory system status */
router.get('/status', async (req: Request, res: Response) => {
  await proxyGet(req, res, '/memory/status', 'Failed to fetch memory status');
});

/** GET /shadow - Get shadow memory state */
router.get('/shadow', async (req: Request, res: Response) => {
  await proxyGet(req, res, '/memory/shadow', 'Failed to fetch shadow memory');
});

/** GET /basin - Get basin state */
router.get('/basin', async (req: Request, res: Response) => {
  await proxyGet(req, res, '/memory/basin', 'Failed to fetch basin state');
});

/** GET /learning - Get learning memory */
router.get('/learning', async (req: Request, res: Response) => {
  // Support query params like limit
  const limit = req.query.limit || 50;
  await proxyGet(req, res, `/memory/learning?limit=${limit}`, 'Failed to fetch learning memory');
});

/** POST /record - Record memory event */
router.post('/record', async (req: Request, res: Response) => {
  await proxyPost(req, res, '/memory/record', 'Failed to record memory');
});

export const memoryRouter = router;
