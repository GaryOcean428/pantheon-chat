/**
 * COORDIZER ROUTES
 * 
 * Proxy for Python coordizer API endpoints.
 * Includes: basic coordization, multi-scale, consciousness-aware, and vocabulary.
 * 
 * QIG-Pure: All coordization uses Fisher-Rao geometry on the information manifold.
 */

import { Router, Request, Response } from 'express';
import { getErrorMessage } from '../lib/error-utils';

const router = Router();
const BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';

// Cache for vocab size (prevents 0 returns when backend is temporarily unavailable)
let cachedVocabSize = 0;
let lastVocabFetch = 0;

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
    console.error(`[Coordizer] ${errorMessage}:`, getErrorMessage(error));
    res.status(503).json({ 
      error: errorMessage, 
      details: getErrorMessage(error),
      coordizers_available: false,
    });
  }
}

async function proxyPost(
  req: Request,
  res: Response,
  pythonPath: string,
  errorMessage: string,
  timeout = 15000
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
    console.error(`[Coordizer] ${errorMessage}:`, getErrorMessage(error));
    res.status(503).json({ 
      error: errorMessage, 
      details: getErrorMessage(error),
    });
  }
}

// ============================================================
// BASIC COORDIZATION
// ============================================================

/** POST / - Basic text coordization */
router.post('/', async (req: Request, res: Response) => {
  await proxyPost(req, res, '/api/coordize', 'Failed to coordize text');
});

// ============================================================
// ADVANCED COORDIZATION
// ============================================================

/** POST /multi-scale - Multi-scale coordization */
router.post('/multi-scale', async (req: Request, res: Response) => {
  await proxyPost(req, res, '/api/coordize/multi-scale', 'Failed multi-scale coordization');
});

/** POST /consciousness - Φ-optimized coordization */
router.post('/consciousness', async (req: Request, res: Response) => {
  await proxyPost(req, res, '/api/coordize/consciousness', 'Failed consciousness coordization');
});

/** POST /merge/learn - Learn geometric pair merges */
router.post('/merge/learn', async (req: Request, res: Response) => {
  await proxyPost(req, res, '/api/coordize/merge/learn', 'Failed to learn merges');
});

// ============================================================
// STATISTICS & VOCABULARY
// ============================================================

/** GET /stats - Get coordizer statistics with caching */
router.get('/stats', async (req: Request, res: Response) => {
  try {
    const response = await fetch(`${BACKEND_URL}/api/coordize/stats`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      signal: AbortSignal.timeout(10000),
    });
    
    if (response.ok) {
      const data = await response.json();
      // Cache vocab size for fallback
      if (data.vocab_size && data.vocab_size > 0) {
        cachedVocabSize = data.vocab_size;
        lastVocabFetch = Date.now();
      }
      return res.json(data);
    }
    
    // Fallback with cached value
    const errorData = await response.json().catch(() => ({}));
    return res.status(response.status).json({
      ...errorData,
      vocab_size: cachedVocabSize || 0,
      cached: true,
    });
  } catch (error) {
    console.error('[Coordizer] Stats error:', getErrorMessage(error));
    // Return cached value on error
    res.json({
      vocab_size: cachedVocabSize || 0,
      coordinate_dim: 64,
      geometric_purity: true,
      special_tokens: ['[PAD]', '[UNK]', '[BOS]', '[EOS]'],
      status: 'backend_unavailable',
      cached: true,
      error: getErrorMessage(error),
    });
  }
});

/** GET /vocab - Get vocabulary info */
router.get('/vocab', async (req: Request, res: Response) => {
  // Forward query params
  const queryString = new URLSearchParams(req.query as Record<string, string>).toString();
  const path = `/api/coordize/vocab${queryString ? '?' + queryString : ''}`;
  await proxyGet(req, res, path, 'Failed to fetch vocabulary');
});

/** POST /similarity - Compute token similarity */
router.post('/similarity', async (req: Request, res: Response) => {
  await proxyPost(req, res, '/api/coordize/similarity', 'Failed to compute similarity');
});

/** GET /health - Coordizer health check */
router.get('/health', async (req: Request, res: Response) => {
  await proxyGet(req, res, '/api/coordize/health', 'Coordizer health check failed');
});

export const coordizerRouter = router;
