/**
 * Vision-First Generation Routes
 * 
 * Proxies to Python backend for vision sampling and generation.
 * 
 * Architecture: See endpoint first → compute geodesic → generate along path
 */

import { Router, Request, Response } from 'express';

export const visionRouter = Router();

const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';

/**
 * POST /api/vision/sample
 * Sample endpoint vision via foresight or lightning
 */
visionRouter.post('/sample', async (req: Request, res: Response) => {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/vision/sample`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(req.body),
    });

    const data = await response.json();
    res.status(response.status).json(data);
  } catch (error) {
    console.error('[Vision] Sample error:', error);
    res.status(500).json({
      success: false,
      error: 'Vision sampling service unavailable',
    });
  }
});

/**
 * POST /api/vision/generate
 * Generate text via vision-first method
 */
visionRouter.post('/generate', async (req: Request, res: Response) => {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/vision/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(req.body),
    });

    const data = await response.json();
    res.status(response.status).json(data);
  } catch (error) {
    console.error('[Vision] Generate error:', error);
    res.status(500).json({
      success: false,
      error: 'Vision generation service unavailable',
    });
  }
});

/**
 * GET /api/vision/status
 * Get vision generator status
 */
visionRouter.get('/status', async (_req: Request, res: Response) => {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/vision/status`);
    const data = await response.json();
    res.status(response.status).json(data);
  } catch (error) {
    console.error('[Vision] Status error:', error);
    res.status(500).json({
      available: false,
      error: 'Vision service unavailable',
    });
  }
});

/**
 * GET /api/vision/attractors
 * List available attractor concepts
 */
visionRouter.get('/attractors', async (_req: Request, res: Response) => {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/vision/attractors`);
    const data = await response.json();
    res.status(response.status).json(data);
  } catch (error) {
    console.error('[Vision] Attractors error:', error);
    res.status(500).json({
      success: false,
      error: 'Vision service unavailable',
    });
  }
});

console.log('[Vision] Routes initialized');
