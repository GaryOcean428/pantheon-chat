/**
 * TRAINING ROUTES
 * 
 * Proxy for Python training loop and document processing endpoints.
 * Includes: training status, document training, and coherence metrics.
 * 
 * QIG-Pure: Training uses Fisher-Rao geometry for coherence evaluation.
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
    console.error(`[Training] ${errorMessage}:`, getErrorMessage(error));
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
    console.error(`[Training] ${errorMessage}:`, getErrorMessage(error));
    res.status(503).json({ 
      error: errorMessage, 
      details: getErrorMessage(error),
    });
  }
}

// ============================================================
// TRAINING STATUS
// ============================================================

/** GET /status - Get training loop status */
router.get('/status', async (req: Request, res: Response) => {
  await proxyGet(req, res, '/training/status', 'Failed to fetch training status');
});

/** POST /docs - Train on documents */
router.post('/docs', async (req: Request, res: Response) => {
  await proxyPost(req, res, '/training/docs', 'Failed to process training documents');
});

export const trainingRouter = router;
