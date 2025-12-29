import { Router, Request, Response } from 'express';

const router = Router();

const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';

/**
 * GET /api/olympus/tools/list
 * List all available tools from the tool factory
 */
router.get('/list', async (_req: Request, res: Response) => {
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/tools/list`, {
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      // Return empty tools list if Python backend unavailable
      return res.json({
        tools: [],
        total: 0,
        message: 'Tool factory not available',
      });
    }

    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('[Tools] Error fetching tools:', error);
    res.json({
      tools: [],
      total: 0,
      message: 'Tool factory not available',
    });
  }
});

/**
 * POST /api/olympus/tools/execute
 * Execute a tool with given arguments
 */
router.post('/execute', async (req: Request, res: Response) => {
  try {
    const { tool_id, args } = req.body;

    if (!tool_id) {
      return res.status(400).json({
        success: false,
        error: 'tool_id is required',
      });
    }

    const response = await fetch(`${PYTHON_BACKEND_URL}/api/tools/execute`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        tool_id,
        args: args || {},
      }),
    });

    if (!response.ok) {
      return res.status(response.status).json({
        success: false,
        error: 'Tool execution failed',
      });
    }

    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('[Tools] Error executing tool:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to execute tool',
    });
  }
});

/**
 * GET /api/olympus/tools/:toolId
 * Get details for a specific tool
 */
router.get('/:toolId', async (req: Request, res: Response) => {
  try {
    const { toolId } = req.params;

    const response = await fetch(`${PYTHON_BACKEND_URL}/api/tools/${toolId}`, {
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      return res.status(404).json({
        error: 'Tool not found',
      });
    }

    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('[Tools] Error fetching tool:', error);
    res.status(500).json({
      error: 'Failed to fetch tool',
    });
  }
});

export { router as toolsRouter };
