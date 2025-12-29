/**
 * Stripe Webhook Handler
 * 
 * Handles Stripe webhook events for billing:
 * - checkout.session.completed (credit purchases, subscription starts)
 * - customer.subscription.updated (tier changes)
 * - customer.subscription.deleted (cancellations)
 * - invoice.paid (recurring subscription payments)
 * 
 * IMPORTANT: This route must use express.raw() for body parsing
 * because Stripe signature verification requires the raw body.
 */

import { Router, Request, Response } from 'express';
import Stripe from 'stripe';

const stripeWebhookRouter = Router();

// Initialize Stripe (will be null if not configured)
const stripeSecretKey = process.env.STRIPE_SECRET_KEY;
const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;

let stripe: Stripe | null = null;
if (stripeSecretKey) {
  stripe = new Stripe(stripeSecretKey, {
    apiVersion: '2024-12-18.acacia',
  });
  console.log('[StripeWebhook] Stripe initialized');
} else {
  console.log('[StripeWebhook] Stripe not configured (missing STRIPE_SECRET_KEY)');
}

/**
 * Forward webhook event to Python backend for processing
 */
async function forwardToPython(event: Stripe.Event): Promise<boolean> {
  const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
  
  try {
    const response = await fetch(`${pythonBackendUrl}/api/stripe/webhook`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Internal-Auth': process.env.INTERNAL_API_KEY || 'dev-key',
      },
      body: JSON.stringify(event),
    });
    
    if (!response.ok) {
      console.error('[StripeWebhook] Python backend returned error:', response.status);
      return false;
    }
    
    const result = await response.json();
    console.log('[StripeWebhook] Python processed event:', result);
    return true;
  } catch (error) {
    console.error('[StripeWebhook] Failed to forward to Python:', error);
    return false;
  }
}

/**
 * Handle checkout.session.completed
 * - Credit purchases (one-time payments)
 * - Subscription starts
 */
async function handleCheckoutCompleted(session: Stripe.Checkout.Session): Promise<void> {
  console.log('[StripeWebhook] Checkout completed:', {
    mode: session.mode,
    customerId: session.customer,
    amountTotal: session.amount_total,
    metadata: session.metadata,
  });
  
  // The Python backend handles the actual credit/subscription updates
  // This function is for logging and any TypeScript-specific handling
}

/**
 * Handle subscription updates
 */
async function handleSubscriptionUpdated(subscription: Stripe.Subscription): Promise<void> {
  console.log('[StripeWebhook] Subscription updated:', {
    id: subscription.id,
    status: subscription.status,
    customerId: subscription.customer,
  });
}

/**
 * Handle subscription deletions/cancellations
 */
async function handleSubscriptionDeleted(subscription: Stripe.Subscription): Promise<void> {
  console.log('[StripeWebhook] Subscription deleted:', {
    id: subscription.id,
    customerId: subscription.customer,
  });
}

/**
 * Handle successful invoice payments
 */
async function handleInvoicePaid(invoice: Stripe.Invoice): Promise<void> {
  console.log('[StripeWebhook] Invoice paid:', {
    id: invoice.id,
    customerId: invoice.customer,
    amountPaid: invoice.amount_paid,
    subscriptionId: invoice.subscription,
  });
}

/**
 * POST /api/stripe/webhook
 * 
 * Stripe webhook endpoint - receives events from Stripe
 * Must be called with raw body for signature verification
 */
stripeWebhookRouter.post('/', async (req: Request, res: Response) => {
  if (!stripe || !webhookSecret) {
    console.error('[StripeWebhook] Stripe not configured');
    return res.status(500).json({ error: 'Stripe not configured' });
  }
  
  const signature = req.headers['stripe-signature'] as string;
  
  if (!signature) {
    console.error('[StripeWebhook] Missing stripe-signature header');
    return res.status(400).json({ error: 'Missing signature' });
  }
  
  let event: Stripe.Event;
  
  try {
    // Verify webhook signature
    // req.body must be raw Buffer for this to work
    event = stripe.webhooks.constructEvent(
      req.body,
      signature,
      webhookSecret
    );
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    console.error('[StripeWebhook] Signature verification failed:', message);
    return res.status(400).json({ error: `Webhook signature verification failed: ${message}` });
  }
  
  console.log('[StripeWebhook] Received event:', event.type);
  
  try {
    // Forward to Python backend for processing
    const forwarded = await forwardToPython(event);
    
    // Also handle locally for logging
    switch (event.type) {
      case 'checkout.session.completed':
        await handleCheckoutCompleted(event.data.object as Stripe.Checkout.Session);
        break;
        
      case 'customer.subscription.created':
      case 'customer.subscription.updated':
        await handleSubscriptionUpdated(event.data.object as Stripe.Subscription);
        break;
        
      case 'customer.subscription.deleted':
        await handleSubscriptionDeleted(event.data.object as Stripe.Subscription);
        break;
        
      case 'invoice.paid':
        await handleInvoicePaid(event.data.object as Stripe.Invoice);
        break;
        
      default:
        console.log('[StripeWebhook] Unhandled event type:', event.type);
    }
    
    // Return success to Stripe
    res.json({ 
      received: true, 
      event_type: event.type,
      forwarded_to_python: forwarded 
    });
    
  } catch (error) {
    console.error('[StripeWebhook] Error processing event:', error);
    // Still return 200 to Stripe to prevent retries for processing errors
    // Stripe will retry on 4xx/5xx responses
    res.json({ 
      received: true, 
      event_type: event.type,
      error: 'Processing error (logged)' 
    });
  }
});

/**
 * GET /api/stripe/webhook
 * Health check for the webhook endpoint
 */
stripeWebhookRouter.get('/', (_req: Request, res: Response) => {
  res.json({
    status: 'ok',
    stripe_configured: !!stripe,
    webhook_secret_configured: !!webhookSecret,
    message: 'Stripe webhook endpoint is ready. POST events to this URL.'
  });
});

export default stripeWebhookRouter;
