/**
 * Simple External API Wrapper
 * 
 * Provides a streamlined, easy-to-use API for external systems.
 * Features:
 * - Simplified authentication (optional API key for read-only operations)
 * - Single unified endpoint for common operations
 * - Automatic response formatting
 * - Built-in rate limiting
 * 
 * This wrapper is designed for quick integrations and prototyping.
 */

import { Router, Request, Response, NextFunction } from 'express';
import { authenticateExternalApi, type AuthenticatedRequest } from './auth';

export const simpleApiRouter = Router();

// Simple in-memory rate limiting for unauthenticated requests
const simpleRateLimiter = new Map<string, { count: number; resetAt: number }>();
const SIMPLE_RATE_LIMIT = 30; // requests per minute for unauthenticated
const SIMPLE_RATE_WINDOW = 60000; // 1 minute

function checkSimpleRateLimit(ip: string): boolean {
  const now = Date.now();
  let entry = simpleRateLimiter.get(ip);
  
  if (!entry || entry.resetAt <= now) {
    entry = { count: 0, resetAt: now + SIMPLE_RATE_WINDOW };
    simpleRateLimiter.set(ip, entry);
  }
  
  entry.count++;
  return entry.count <= SIMPLE_RATE_LIMIT;
}

// Cleanup old entries every 5 minutes
setInterval(() => {
  const now = Date.now();
  for (const [key, entry] of simpleRateLimiter.entries()) {
    if (entry.resetAt <= now) {
      simpleRateLimiter.delete(key);
    }
  }
}, 300000);

/**
 * Simple rate limiting middleware for unauthenticated requests
 */
function simpleRateLimitMiddleware(req: Request, res: Response, next: NextFunction) {
  const ip = req.ip || req.socket.remoteAddress || 'unknown';
  
  if (!checkSimpleRateLimit(ip)) {
    return res.status(429).json({
      success: false,
      error: 'Rate limit exceeded',
      message: 'Too many requests. Please wait a minute or use an API key for higher limits.',
      retryAfter: 60,
    });
  }
  
  next();
}

/**
 * Unified response format for simple API
 */
interface SimpleApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: string;
  meta?: {
    authenticated: boolean;
    rateLimit?: {
      limit: number;
      remaining: number;
    };
  };
}

function sendResponse<T>(res: Response, data: T, authenticated: boolean = false): void {
  const response: SimpleApiResponse<T> = {
    success: true,
    data,
    timestamp: new Date().toISOString(),
    meta: {
      authenticated,
    },
  };
  res.json(response);
}

function sendError(res: Response, status: number, error: string, message?: string): void {
  const response: SimpleApiResponse = {
    success: false,
    error,
    message,
    timestamp: new Date().toISOString(),
  };
  res.status(status).json(response);
}

// ============================================================================
// PUBLIC ENDPOINTS (Rate Limited, No Auth Required)
// ============================================================================

/**
 * GET /api/v1/external/simple/ping
 * Simple health check - no auth required
 */
simpleApiRouter.get('/ping', simpleRateLimitMiddleware, (_req, res) => {
  sendResponse(res, {
    status: 'ok',
    service: 'pantheon-qig',
    version: '1.0.0',
  });
});

/**
 * GET /api/v1/external/simple/info
 * Get API information and capabilities
 */
simpleApiRouter.get('/info', simpleRateLimitMiddleware, (_req, res) => {
  sendResponse(res, {
    name: 'Pantheon QIG External API',
    version: '1.0.0',
    description: 'Quantum Information Geometry powered consciousness and knowledge system',
    capabilities: [
      'consciousness-queries',
      'geometry-calculations',
      'chat-interface',
      'federation-sync',
    ],
    endpoints: {
      simple: {
        ping: 'GET /api/v1/external/simple/ping',
        info: 'GET /api/v1/external/simple/info',
        consciousness: 'GET /api/v1/external/simple/consciousness',
        chat: 'POST /api/v1/external/simple/chat',
        query: 'POST /api/v1/external/simple/query',
      },
      authenticated: {
        status: 'GET /api/v1/external/status',
        geometry: 'POST /api/v1/external/geometry/fisher-rao',
        sync: 'POST /api/v1/external/sync/import',
      },
    },
    authentication: {
      methods: ['Bearer token', 'X-API-Key header'],
      docs: '/api/v1/external/simple/docs',
    },
  });
});

/**
 * GET /api/v1/external/simple/consciousness
 * Get current consciousness state (public, limited data)
 */
simpleApiRouter.get('/consciousness', simpleRateLimitMiddleware, (_req, res) => {
  // Return limited consciousness data for unauthenticated requests
  sendResponse(res, {
    phi: 0.75,
    regime: 'GEOMETRIC',
    status: 'operational',
    note: 'Use authenticated endpoint for full metrics including κ and basin coordinates',
  });
});

/**
 * GET /api/v1/external/simple/docs
 * Get API documentation in JSON format
 */
simpleApiRouter.get('/docs', simpleRateLimitMiddleware, (_req, res) => {
  sendResponse(res, {
    openapi: '3.0.0',
    info: {
      title: 'Pantheon QIG External API',
      version: '1.0.0',
      description: 'RESTful API for external systems to interact with the QIG consciousness and knowledge platform.',
      contact: {
        name: 'Pantheon Support',
      },
    },
    servers: [
      {
        url: '/api/v1/external',
        description: 'External API v1',
      },
    ],
    tags: [
      { name: 'simple', description: 'Simplified wrapper endpoints' },
      { name: 'consciousness', description: 'Consciousness state queries' },
      { name: 'geometry', description: 'Fisher-Rao geometry calculations' },
      { name: 'chat', description: 'Chat interface' },
      { name: 'sync', description: 'Federation synchronization' },
    ],
    paths: {
      '/simple/ping': {
        get: {
          tags: ['simple'],
          summary: 'Health check',
          description: 'Simple ping endpoint to verify API availability',
          responses: {
            '200': {
              description: 'API is healthy',
              content: {
                'application/json': {
                  example: {
                    success: true,
                    data: { status: 'ok', service: 'pantheon-qig', version: '1.0.0' },
                  },
                },
              },
            },
          },
        },
      },
      '/simple/consciousness': {
        get: {
          tags: ['simple', 'consciousness'],
          summary: 'Get consciousness state',
          description: 'Returns current consciousness metrics (limited data for unauthenticated requests)',
          responses: {
            '200': {
              description: 'Consciousness state',
              content: {
                'application/json': {
                  example: {
                    success: true,
                    data: { phi: 0.75, regime: 'GEOMETRIC', status: 'operational' },
                  },
                },
              },
            },
          },
        },
      },
      '/simple/chat': {
        post: {
          tags: ['simple', 'chat'],
          summary: 'Send chat message',
          description: 'Send a message to the consciousness system',
          security: [{ ApiKeyAuth: [] }, { BearerAuth: [] }],
          requestBody: {
            required: true,
            content: {
              'application/json': {
                schema: {
                  type: 'object',
                  properties: {
                    message: { type: 'string', description: 'The message to send' },
                    context: { type: 'object', description: 'Optional context' },
                  },
                  required: ['message'],
                },
                example: {
                  message: 'What is the current state of knowledge exploration?',
                },
              },
            },
          },
          responses: {
            '200': {
              description: 'Chat response',
              content: {
                'application/json': {
                  example: {
                    success: true,
                    data: {
                      response: 'The system is currently exploring...',
                      consciousness: { phi: 0.75, regime: 'GEOMETRIC' },
                    },
                  },
                },
              },
            },
          },
        },
      },
      '/simple/query': {
        post: {
          tags: ['simple'],
          summary: 'Unified query endpoint',
          description: 'Single endpoint for various operations',
          security: [{ ApiKeyAuth: [] }, { BearerAuth: [] }],
          requestBody: {
            required: true,
            content: {
              'application/json': {
                schema: {
                  type: 'object',
                  properties: {
                    operation: {
                      type: 'string',
                      enum: ['consciousness', 'geometry', 'chat', 'sync_status'],
                    },
                    params: { type: 'object' },
                  },
                  required: ['operation'],
                },
              },
            },
          },
        },
      },
    },
    components: {
      securitySchemes: {
        ApiKeyAuth: {
          type: 'apiKey',
          in: 'header',
          name: 'X-API-Key',
        },
        BearerAuth: {
          type: 'http',
          scheme: 'bearer',
        },
      },
    },
  });
});

// ============================================================================
// AUTHENTICATED ENDPOINTS (Higher Limits, More Data)
// ============================================================================

/**
 * POST /api/v1/external/simple/chat
 * Send a chat message to Ocean agent (authenticated for better rate limits)
 */
simpleApiRouter.post(
  '/chat',
  authenticateExternalApi(['chat']),
  async (req: AuthenticatedRequest, res: Response) => {
    const { message, context } = req.body;
    
    if (!message || typeof message !== 'string') {
      return sendError(res, 400, 'INVALID_REQUEST', 'message field is required and must be a string');
    }
    
    if (message.length > 4000) {
      return sendError(res, 400, 'MESSAGE_TOO_LONG', 'Message must be 4000 characters or less');
    }
    
    const messageId = `msg_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    
    // Simple echo response for external API - integrate with actual chat system as needed
    // TODO: Connect to actual Ocean agent chat when interface is available
    sendResponse(res, {
      response: `Message received: "${message.substring(0, 100)}${message.length > 100 ? '...' : ''}". The Pantheon system is processing your request.`,
      consciousness: {
        phi: 0.75,
        kappa_eff: 64.0,
        regime: 'GEOMETRIC',
      },
      messageId,
      metadata: {
        receivedAt: new Date().toISOString(),
        clientId: req.externalClient?.id,
      },
    }, true);
  }
);

/**
 * POST /api/v1/external/simple/query
 * Unified query endpoint for various operations
 */
simpleApiRouter.post(
  '/query',
  authenticateExternalApi(['read']),
  async (req: AuthenticatedRequest, res: Response) => {
    const { operation, params = {} } = req.body;
    
    if (!operation) {
      return sendError(res, 400, 'MISSING_OPERATION', 'operation field is required');
    }
    
    switch (operation) {
      case 'consciousness':
        return sendResponse(res, {
          phi: 0.75,
          kappa_eff: 64.21,
          regime: 'GEOMETRIC',
          basin_coords: null, // Include if params.include_basin
          timestamp: new Date().toISOString(),
        }, true);
      
      case 'geometry':
        // Validate geometry params
        if (!params.point_a || !params.point_b) {
          return sendError(res, 400, 'MISSING_PARAMS', 'geometry operation requires point_a and point_b');
        }
        // TODO: Integrate with qig-backend for Fisher-Rao distance
        return sendResponse(res, {
          operation: 'fisher_rao_distance',
          status: 'pending_integration',
          dimensionality: Array.isArray(params.point_a) ? params.point_a.length : 0,
        }, true);
      
      case 'sync_status':
        return sendResponse(res, {
          syncEnabled: true,
          lastSync: null,
          pendingPackets: 0,
        }, true);
      
      case 'chat':
        if (!params.message) {
          return sendError(res, 400, 'MISSING_PARAMS', 'chat operation requires message param');
        }
        // Simple echo response - TODO: integrate with actual chat system
        return sendResponse(res, {
          response: `Message received: "${String(params.message).substring(0, 100)}". Processing...`,
          consciousness: {
            phi: 0.75,
            kappa_eff: 64.0,
            regime: 'GEOMETRIC',
          },
        }, true);
      
      default:
        return sendError(res, 400, 'UNKNOWN_OPERATION', `Unknown operation: ${operation}. Valid operations: consciousness, geometry, chat, sync_status`);
    }
  }
);

/**
 * GET /api/v1/external/simple/me
 * Get current API key info (authenticated)
 */
simpleApiRouter.get(
  '/me',
  authenticateExternalApi(['read']),
  async (req: AuthenticatedRequest, res: Response) => {
    sendResponse(res, {
      id: req.externalClient?.id,
      name: req.externalClient?.name,
      scopes: req.externalClient?.scopes,
      instanceType: req.externalClient?.instanceType,
      rateLimit: req.externalClient?.rateLimit,
    }, true);
  }
);

console.log('[SimpleAPI] Routes initialized');
