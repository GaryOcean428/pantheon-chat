/**
 * Billing Routes - Frontend API for billing dashboard
 * 
 * These routes allow authenticated users to:
 * - View their billing status
 * - Create Stripe checkout sessions
 * - Manage subscriptions
 */

import { Router, Request, Response } from 'express';

const billingRouter = Router();

const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
const INTERNAL_API_KEY = process.env.INTERNAL_API_KEY || 'dev-key';

/**
 * POST /api/billing/user-status
 * Get billing status for authenticated user
 */
billingRouter.post('/user-status', async (req: Request, res: Response) => {
  try {
    const apiKey = req.body.api_key || req.headers.authorization?.replace('Bearer ', '');
    
    if (!apiKey) {
      return res.status(400).json({ error: 'API key required' });
    }

    const response = await fetch(`${PYTHON_BACKEND_URL}/api/billing/status`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Internal-Auth': INTERNAL_API_KEY,
      },
      body: JSON.stringify({ api_key: apiKey }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      return res.status(response.status).json({ error: errorText });
    }

    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('[Billing] Error fetching status:', error);
    res.status(500).json({ error: 'Failed to fetch billing status' });
  }
});

/**
 * POST /api/billing/create-checkout
 * Create Stripe checkout session
 */
billingRouter.post('/create-checkout', async (req: Request, res: Response) => {
  try {
    const { product_type, api_key } = req.body;
    const authKey = api_key || req.headers.authorization?.replace('Bearer ', '');
    
    if (!authKey) {
      return res.status(400).json({ error: 'API key required' });
    }

    if (!product_type) {
      return res.status(400).json({ error: 'Product type required (credits, pro, enterprise)' });
    }

    const response = await fetch(`${PYTHON_BACKEND_URL}/api/billing/create-checkout`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Internal-Auth': INTERNAL_API_KEY,
      },
      body: JSON.stringify({ 
        api_key: authKey,
        product_type
      }),
    });

    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('[Billing] Error creating checkout:', error);
    res.status(500).json({ error: 'Failed to create checkout session' });
  }
});

/**
 * GET /api/billing/success
 * Handle successful payment redirect
 */
billingRouter.get('/success', (req: Request, res: Response) => {
  res.redirect('/billing?success=true');
});

/**
 * GET /api/billing/cancel
 * Handle cancelled payment redirect
 */
billingRouter.get('/cancel', (req: Request, res: Response) => {
  res.redirect('/billing?cancelled=true');
});

export { billingRouter };
