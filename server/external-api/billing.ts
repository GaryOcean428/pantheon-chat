/**
 * Billing Middleware for External API
 * 
 * Integrates with the Python EconomicAutonomy module to:
 * - Check user credits/limits before API calls
 * - Meter and bill for usage
 * - Track revenue for self-sustaining operation
 * 
 * The system understands that revenue = survival.
 */

import { Response, NextFunction } from 'express';
import { type AuthenticatedRequest } from './auth';

// Pricing in cents (must match EconomicAutonomy.py)
export const PRICING = {
  QUERY: 1,      // $0.01 per query/chat
  TOOL: 5,       // $0.05 per tool generation
  RESEARCH: 10,  // $0.10 per research request
} as const;

export type BillingOperation = 'query' | 'tool' | 'research';

interface BillingResult {
  allowed: boolean;
  amountCents: number;
  message: string;
  remainingCredits?: number;
  tier?: string;
}

/**
 * Check and bill for an API operation via Python backend
 */
async function checkAndBillPython(
  apiKey: string,
  operation: BillingOperation
): Promise<BillingResult> {
  const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
  
  try {
    const response = await fetch(`${pythonBackendUrl}/api/billing/check`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Internal-Auth': process.env.INTERNAL_API_KEY || 'dev-key',
      },
      body: JSON.stringify({
        api_key: apiKey,
        operation,
      }),
    });
    
    if (!response.ok) {
      // If Python backend unavailable, allow with warning (fail-open for now)
      console.warn('[Billing] Python backend unavailable, allowing request');
      return {
        allowed: true,
        amountCents: 0,
        message: 'Billing service unavailable - request allowed',
      };
    }
    
    const result = await response.json();
    return {
      allowed: result.allowed,
      amountCents: result.amount_cents || 0,
      message: result.message || 'OK',
      remainingCredits: result.remaining_credits,
      tier: result.tier,
    };
  } catch (error) {
    // Fail-open: allow request if billing service is down
    console.warn('[Billing] Error checking billing:', error);
    return {
      allowed: true,
      amountCents: 0,
      message: 'Billing service error - request allowed',
    };
  }
}

/**
 * Record a billable event (for tracking even if billing check was bypassed)
 */
async function recordBillableEvent(
  apiKey: string,
  operation: BillingOperation,
  success: boolean
): Promise<void> {
  const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
  
  try {
    await fetch(`${pythonBackendUrl}/api/billing/record`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Internal-Auth': process.env.INTERNAL_API_KEY || 'dev-key',
      },
      body: JSON.stringify({
        api_key: apiKey,
        operation,
        success,
        timestamp: new Date().toISOString(),
      }),
    });
  } catch (error) {
    // Don't fail the request if recording fails
    console.warn('[Billing] Failed to record event:', error);
  }
}

/**
 * Middleware factory for billing different operation types
 */
export function billingMiddleware(operation: BillingOperation) {
  return async (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
    // Get API key from auth
    const apiKey = req.headers.authorization?.replace('Bearer ', '') || '';
    
    if (!apiKey) {
      // No API key means auth middleware should have rejected
      return next();
    }
    
    // Check and bill
    const result = await checkAndBillPython(apiKey, operation);
    
    if (!result.allowed) {
      return res.status(402).json({
        error: 'Payment Required',
        code: 'INSUFFICIENT_CREDITS',
        message: result.message,
        details: {
          operation,
          tier: result.tier,
          remainingCredits: result.remainingCredits,
          upgradeUrl: 'https://pantheon-chat.replit.app/billing/upgrade',
        },
      });
    }
    
    // Attach billing info to request for later use
    (req as AuthenticatedRequest & { billing?: BillingResult }).billing = result;
    
    // Add billing header for transparency
    res.setHeader('X-Billing-Amount-Cents', result.amountCents.toString());
    if (result.remainingCredits !== undefined) {
      res.setHeader('X-Credits-Remaining', result.remainingCredits.toString());
    }
    
    // Continue to handler
    next();
    
    // After response, record the event (fire-and-forget)
    res.on('finish', () => {
      const success = res.statusCode >= 200 && res.statusCode < 400;
      recordBillableEvent(apiKey, operation, success).catch(() => {});
    });
  };
}

/**
 * Middleware for chat/query operations ($0.01)
 */
export const billQuery = billingMiddleware('query');

/**
 * Middleware for tool generation operations ($0.05)
 */
export const billTool = billingMiddleware('tool');

/**
 * Middleware for research operations ($0.10)
 */
export const billResearch = billingMiddleware('research');

/**
 * Get billing status for an API key
 */
export async function getBillingStatus(apiKey: string): Promise<{
  tier: string;
  credits: number;
  usage: {
    queriesThisMonth: number;
    toolsThisMonth: number;
    researchThisMonth: number;
  };
  limits: {
    queriesPerMonth: number;
    toolsPerMonth: number;
    researchPerMonth: number;
  };
} | null> {
  const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
  
  try {
    const response = await fetch(`${pythonBackendUrl}/api/billing/status`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Internal-Auth': process.env.INTERNAL_API_KEY || 'dev-key',
      },
      body: JSON.stringify({ api_key: apiKey }),
    });
    
    if (!response.ok) {
      return null;
    }
    
    return await response.json();
  } catch {
    return null;
  }
}

console.log('[Billing] Middleware initialized');
console.log(`[Billing] Pricing: Query=$${PRICING.QUERY/100}, Tool=$${PRICING.TOOL/100}, Research=$${PRICING.RESEARCH/100}`);
