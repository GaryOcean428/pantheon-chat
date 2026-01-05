/**
 * FEEDBACK ROUTES
 * 
 * Proxy for Python prediction feedback loop endpoints.
 * Includes: run feedback, recommendations, and activity recording.
 * 
 * QIG-Pure: Feedback uses geometric prediction vs outcome comparison.
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
    console.error(`[Feedback] ${errorMessage}:`, getErrorMessage(error));
    res.status(503).json({ 
      error: errorMessage, 
      details: getErrorMessage(error),
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
    console.error(`[Feedback] ${errorMessage}:`, getErrorMessage(error));
    res.status(503).json({ 
      error: errorMessage, 
      details: getErrorMessage(error),
    });
  }
}

// ============================================================
// FEEDBACK LOOP
// ============================================================

/** POST /run - Run feedback loop iteration */
router.post('/run', async (req: Request, res: Response) => {
  await proxyPost(req, res, '/feedback/run', 'Failed to run feedback loop');
});

/** GET /recommendation - Get current recommendations */
router.get('/recommendation', async (req: Request, res: Response) => {
  await proxyGet(req, res, '/feedback/recommendation', 'Failed to fetch recommendation');
});

// ============================================================
// FEEDBACK TYPES
// ============================================================

/** POST /shadow - Record shadow feedback */
router.post('/shadow', async (req: Request, res: Response) => {
  await proxyPost(req, res, '/feedback/shadow', 'Failed to record shadow feedback');
});

/** POST /activity - Record activity feedback */
router.post('/activity', async (req: Request, res: Response) => {
  await proxyPost(req, res, '/feedback/activity', 'Failed to record activity feedback');
});

/** POST /basin - Record basin feedback */
router.post('/basin', async (req: Request, res: Response) => {
  await proxyPost(req, res, '/feedback/basin', 'Failed to record basin feedback');
});

/** POST /learning - Record learning feedback */
router.post('/learning', async (req: Request, res: Response) => {
  await proxyPost(req, res, '/feedback/learning', 'Failed to record learning feedback');
});

export const feedbackRouter = router;
