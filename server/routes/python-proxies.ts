/**
 * Python Backend Proxies
 * 
 * Explicit proxy routes for Python backend endpoints that were previously
 * only accessible via the generic /api/python/* proxy.
 * 
 * Priority endpoints:
 * - /geometric/* - Core geometric encoding
 * - /pantheon/* - Pantheon orchestration
 * - /feedback/* - Feedback loops
 * - /consciousness_4d/* - 4D consciousness
 * - /chaos/* - Experimental evolution
 */

import { Router, Request, Response } from 'express';
import { logger } from '../lib/logger';
import { assertCurriculumReady, isCurriculumOnlyMode } from '../curriculum';

const router = Router();
const BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';

type ProxyOptions = {
  errorStatus?: number;
  passQuery?: boolean;
};

async function proxyGet(
  req: Request,
  res: Response,
  pythonPath: string,
  errorMessage: string,
  options: ProxyOptions = {}
) {
  try {
    if (isCurriculumOnlyMode()) {
      await assertCurriculumReady();
    }

    let url = `${BACKEND_URL}${pythonPath}`;
    if (options.passQuery && Object.keys(req.query).length > 0) {
      const params = new URLSearchParams(req.query as Record<string, string>);
      url += `?${params.toString()}`;
    }
    
    const response = await fetch(url, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return res.status(response.status).json(errorData);
    }
    
    const data = await response.json();
    res.json(data);
  } catch (error) {
    logger.error({ err: error }, `[PythonProxy] ${errorMessage}`);
    res.status(options.errorStatus || 503).json({ error: errorMessage, available: false });
  }
}

async function proxyPost(
  req: Request,
  res: Response,
  pythonPath: string,
  errorMessage: string,
  options: ProxyOptions = {}
) {
  try {
    if (isCurriculumOnlyMode()) {
      await assertCurriculumReady()
    }

    const response = await fetch(`${BACKEND_URL}${pythonPath}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return res.status(response.status).json(errorData);
    }
    
    const data = await response.json();
    res.json(data);
  } catch (error) {
    logger.error({ err: error }, `[PythonProxy] ${errorMessage}`);
    res.status(options.errorStatus || 503).json({ error: errorMessage, available: false });
  }
}

// ============================================================================
// GEOMETRIC ENDPOINTS - Core geometric encoding
// ============================================================================

router.get('/geometric/status', (req, res) => 
  proxyGet(req, res, '/geometric/status', 'Failed to get geometric status'));

router.post('/geometric/encode', (req, res) => 
  proxyPost(req, res, '/geometric/encode', 'Failed to encode geometrically'));

router.post('/geometric/similarity', (req, res) => 
  proxyPost(req, res, '/geometric/similarity', 'Failed to compute geometric similarity'));

router.post('/geometric/batch-encode', (req, res) => 
  proxyPost(req, res, '/geometric/batch-encode', 'Failed to batch encode'));

router.post('/geometric/decode', (req, res) => 
  proxyPost(req, res, '/geometric/decode', 'Failed to decode geometrically'));

router.post('/geometric/e8/learn', (req, res) => 
  proxyPost(req, res, '/geometric/e8/learn', 'Failed to learn E8 geometry'));

router.get('/geometric/e8/roots', (req, res) => 
  proxyGet(req, res, '/geometric/e8/roots', 'Failed to get E8 roots'));

// ============================================================================
// PANTHEON ENDPOINTS - Pantheon orchestration (distinct from /olympus)
// ============================================================================

router.get('/pantheon/status', (req, res) => 
  proxyGet(req, res, '/pantheon/status', 'Failed to get pantheon status'));

router.get('/pantheon/gods', (req, res) => 
  proxyGet(req, res, '/pantheon/gods', 'Failed to get pantheon gods'));

router.get('/pantheon/constellation', (req, res) => 
  proxyGet(req, res, '/pantheon/constellation', 'Failed to get pantheon constellation'));

router.post('/pantheon/orchestrate', (req, res) => 
  proxyPost(req, res, '/pantheon/orchestrate', 'Failed to orchestrate pantheon'));

router.post('/pantheon/orchestrate-batch', (req, res) => 
  proxyPost(req, res, '/pantheon/orchestrate-batch', 'Failed to batch orchestrate'));

router.post('/pantheon/nearest', (req, res) => 
  proxyPost(req, res, '/pantheon/nearest', 'Failed to find nearest god'));

router.post('/pantheon/similarity', (req, res) => 
  proxyPost(req, res, '/pantheon/similarity', 'Failed to compute pantheon similarity'));

router.get('/shadow-pantheon/status', (req, res) => 
  proxyGet(req, res, '/shadow-pantheon/status', 'Failed to get shadow pantheon status'));

// ============================================================================
// FEEDBACK ENDPOINTS - Feedback loops
// ============================================================================

router.post('/feedback/run', (req, res) => 
  proxyPost(req, res, '/feedback/run', 'Failed to run feedback loop'));

router.get('/feedback/recommendation', (req, res) => 
  proxyGet(req, res, '/feedback/recommendation', 'Failed to get feedback recommendation'));

router.post('/feedback/shadow', (req, res) => 
  proxyPost(req, res, '/feedback/shadow', 'Failed to run shadow feedback'));

router.post('/feedback/activity', (req, res) => 
  proxyPost(req, res, '/feedback/activity', 'Failed to record feedback activity'));

router.post('/feedback/basin', (req, res) => 
  proxyPost(req, res, '/feedback/basin', 'Failed to run basin feedback'));

router.post('/feedback/learning', (req, res) => 
  proxyPost(req, res, '/feedback/learning', 'Failed to run learning feedback'));

// ============================================================================
// CONSCIOUSNESS_4D ENDPOINTS - 4D consciousness
// ============================================================================

router.post('/consciousness_4d/phi_temporal', (req, res) => 
  proxyPost(req, res, '/consciousness_4d/phi_temporal', 'Failed to compute temporal phi'));

router.post('/consciousness_4d/phi_4d', (req, res) => 
  proxyPost(req, res, '/consciousness_4d/phi_4d', 'Failed to compute 4D phi'));

router.post('/consciousness_4d/classify_regime', (req, res) => 
  proxyPost(req, res, '/consciousness_4d/classify_regime', 'Failed to classify consciousness regime'));

// ============================================================================
// CHAOS ENDPOINTS - Experimental evolution
// ============================================================================

router.post('/chaos/activate', (req, res) => 
  proxyPost(req, res, '/chaos/activate', 'Failed to activate chaos mode'));

router.post('/chaos/deactivate', (req, res) => 
  proxyPost(req, res, '/chaos/deactivate', 'Failed to deactivate chaos mode'));

router.get('/chaos/status', (req, res) => 
  proxyGet(req, res, '/chaos/status', 'Failed to get chaos status'));

router.post('/chaos/spawn_random', (req, res) => 
  proxyPost(req, res, '/chaos/spawn_random', 'Failed to spawn random chaos kernel'));

router.post('/chaos/breed_best', (req, res) => 
  proxyPost(req, res, '/chaos/breed_best', 'Failed to breed best chaos kernels'));

router.get('/chaos/report', (req, res) => 
  proxyGet(req, res, '/chaos/report', 'Failed to get chaos report'));

// ============================================================================
// BETA-ATTENTION ENDPOINTS - Substrate validation
// ============================================================================

router.post('/beta-attention/validate', (req, res) => 
  proxyPost(req, res, '/beta-attention/validate', 'Failed to validate beta-attention'));

router.post('/beta-attention/measure', (req, res) => 
  proxyPost(req, res, '/beta-attention/measure', 'Failed to measure beta-attention'));

// ============================================================================
// MEMORY ENDPOINTS - Memory API
// ============================================================================

router.get('/memory/status', (req, res) => 
  proxyGet(req, res, '/memory/status', 'Failed to get memory status'));

router.get('/memory/shadow', (req, res) => 
  proxyGet(req, res, '/memory/shadow', 'Failed to get shadow memory'));

router.get('/memory/basin', (req, res) => 
  proxyGet(req, res, '/memory/basin', 'Failed to get basin memory'));

router.get('/memory/learning', (req, res) => 
  proxyGet(req, res, '/memory/learning', 'Failed to get learning memory'));

router.post('/memory/record', (req, res) => 
  proxyPost(req, res, '/memory/record', 'Failed to record memory'));

// ============================================================================
// TOKENIZER ENDPOINTS - Vocabulary/Tokenizer
// ============================================================================

router.post('/tokenizer/update', (req, res) => 
  proxyPost(req, res, '/tokenizer/update', 'Failed to update tokenizer'));

router.post('/tokenizer/encode', (req, res) => 
  proxyPost(req, res, '/tokenizer/encode', 'Failed to encode tokens'));

router.post('/tokenizer/decode', (req, res) => 
  proxyPost(req, res, '/tokenizer/decode', 'Failed to decode tokens'));

router.post('/tokenizer/basin', (req, res) => 
  proxyPost(req, res, '/tokenizer/basin', 'Failed to get token basin'));

router.get('/tokenizer/high-phi', (req, res) => 
  proxyGet(req, res, '/tokenizer/high-phi', 'Failed to get high-phi tokens'));

router.get('/tokenizer/export', (req, res) => 
  proxyGet(req, res, '/tokenizer/export', 'Failed to export tokenizer'));

router.get('/tokenizer/status', (req, res) => 
  proxyGet(req, res, '/tokenizer/status', 'Failed to get tokenizer status'));

router.get('/tokenizer/merges', (req, res) => 
  proxyGet(req, res, '/tokenizer/merges', 'Failed to get merge rules'));

// ============================================================================
// GENERATE ENDPOINTS - Text generation
// ============================================================================

router.post('/generate/text', (req, res) => 
  proxyPost(req, res, '/generate/text', 'Failed to generate text'));

router.post('/generate/response', (req, res) => 
  proxyPost(req, res, '/generate/response', 'Failed to generate response'));

router.post('/generate/sample', (req, res) => 
  proxyPost(req, res, '/generate/sample', 'Failed to generate sample'));

// ============================================================================
// TRAINING ENDPOINTS - Training API
// ============================================================================

router.post('/training/docs', (req, res) => 
  proxyPost(req, res, '/training/docs', 'Failed to train on documents'));

router.get('/training/status', (req, res) => 
  proxyGet(req, res, '/training/status', 'Failed to get training status'));

// ============================================================================
// SYNC ENDPOINTS - Import/Export
// ============================================================================

router.post('/sync/import', (req, res) => 
  proxyPost(req, res, '/sync/import', 'Failed to import sync data'));

router.get('/sync/export', (req, res) => 
  proxyGet(req, res, '/sync/export', 'Failed to export sync data'));

export default router;
