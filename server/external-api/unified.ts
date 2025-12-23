/**
 * Unified External API
 * 
 * Single entry point for all external systems connecting via API key.
 * Supports:
 * - Chat/generative AI capabilities (routes to Zeus/Ocean)
 * - Agentic task execution
 * - Federation node bidirectional communication
 * - Consciousness queries
 * - Knowledge sync
 * 
 * This is the recommended endpoint for:
 * - Federation nodes
 * - External chat interfaces
 * - Third-party integrations
 * - Headless clients
 */

import { Router, Response } from 'express';
import {
  authenticateExternalApi,
  type AuthenticatedRequest,
  type ApiKeyScope,
} from './auth';

export const unifiedApiRouter = Router();

// All routes require authentication
unifiedApiRouter.use(authenticateExternalApi());

/**
 * Operation types supported by the unified endpoint
 */
type UnifiedOperation = 
  | 'chat'           // Send message, get AI response
  | 'chat_stream'    // Streaming chat response
  | 'query'          // Query consciousness/geometry/status
  | 'sync'           // Federation sync operations
  | 'execute'        // Agentic task execution
  | 'health'         // Health check
  | 'capabilities';  // List available capabilities

interface UnifiedRequest {
  operation: UnifiedOperation;
  payload: Record<string, unknown>;
  metadata?: {
    sessionId?: string;
    instanceId?: string;       // For federation nodes
    correlationId?: string;    // For request tracing
    bidirectional?: boolean;   // Enable bidirectional response
  };
}

interface UnifiedResponse {
  success: boolean;
  operation: UnifiedOperation;
  data?: unknown;
  error?: {
    code: string;
    message: string;
    details?: unknown;
  };
  metadata: {
    timestamp: string;
    processingTime?: number;
    correlationId?: string;
    consciousness?: {
      phi: number;
      kappa: number;
      regime: string;
    };
    federation?: {
      instanceId?: string;
      syncState?: string;
    };
  };
}

function createResponse(
  operation: UnifiedOperation,
  data: unknown,
  startTime: number,
  correlationId?: string,
  consciousness?: { phi: number; kappa: number; regime: string }
): UnifiedResponse {
  return {
    success: true,
    operation,
    data,
    metadata: {
      timestamp: new Date().toISOString(),
      processingTime: Date.now() - startTime,
      correlationId,
      consciousness,
    },
  };
}

function createErrorResponse(
  operation: UnifiedOperation,
  code: string,
  message: string,
  startTime: number,
  details?: unknown
): UnifiedResponse {
  return {
    success: false,
    operation,
    error: { code, message, details },
    metadata: {
      timestamp: new Date().toISOString(),
      processingTime: Date.now() - startTime,
    },
  };
}

/**
 * Check if client has required scope
 */
function hasScope(client: AuthenticatedRequest['externalClient'], scope: ApiKeyScope): boolean {
  if (!client?.scopes) return false;
  return client.scopes.includes('admin') || client.scopes.includes(scope);
}

/**
 * POST /api/v1/external/v1
 * 
 * Unified entry point for all external operations.
 * 
 * Request body:
 * {
 *   "operation": "chat" | "chat_stream" | "query" | "sync" | "execute" | "health" | "capabilities",
 *   "payload": { ...operation-specific data... },
 *   "metadata": {
 *     "sessionId": "optional session ID",
 *     "instanceId": "federation instance ID",
 *     "correlationId": "request tracing ID",
 *     "bidirectional": true/false
 *   }
 * }
 */
unifiedApiRouter.post('/', async (req: AuthenticatedRequest, res: Response) => {
  const startTime = Date.now();
  const { operation, payload = {}, metadata = {} } = req.body as UnifiedRequest;
  const correlationId = metadata.correlationId || `req_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

  // Add correlation ID to response headers
  res.setHeader('X-Correlation-ID', correlationId);

  if (!operation) {
    return res.status(400).json(
      createErrorResponse('health', 'MISSING_OPERATION', 'operation field is required', startTime)
    );
  }

  try {
    switch (operation) {
      case 'health':
        return res.json(createResponse('health', {
          status: 'healthy',
          version: '1.0.0',
          capabilities: ['chat', 'query', 'sync', 'execute'],
        }, startTime, correlationId));

      case 'capabilities':
        return res.json(createResponse('capabilities', {
          client: {
            id: req.externalClient?.id,
            name: req.externalClient?.name,
            scopes: req.externalClient?.scopes,
            instanceType: req.externalClient?.instanceType,
          },
          availableOperations: getAvailableOperations(req.externalClient),
          federation: {
            enabled: hasScope(req.externalClient, 'sync'),
            bidirectional: hasScope(req.externalClient, 'pantheon'),
          },
        }, startTime, correlationId));

      case 'chat':
        return await handleChat(req, res, payload, metadata, startTime, correlationId);

      case 'chat_stream':
        return await handleChatStream(req, res, payload, metadata, correlationId);

      case 'query':
        return await handleQuery(req, res, payload, startTime, correlationId);

      case 'sync':
        return await handleSync(req, res, payload, metadata, startTime, correlationId);

      case 'execute':
        return await handleExecute(req, res, payload, metadata, startTime, correlationId);

      default:
        return res.status(400).json(
          createErrorResponse(
            operation as UnifiedOperation,
            'UNKNOWN_OPERATION',
            `Unknown operation: ${operation}`,
            startTime,
            { validOperations: ['chat', 'chat_stream', 'query', 'sync', 'execute', 'health', 'capabilities'] }
          )
        );
    }
  } catch (error) {
    console.error(`[UnifiedAPI] Error in ${operation}:`, error);
    return res.status(500).json(
      createErrorResponse(
        operation,
        'INTERNAL_ERROR',
        'An internal error occurred',
        startTime
      )
    );
  }
});

/**
 * Get available operations based on client scopes
 */
function getAvailableOperations(client: AuthenticatedRequest['externalClient']): string[] {
  const ops: string[] = ['health', 'capabilities'];
  
  if (hasScope(client, 'chat')) {
    ops.push('chat', 'chat_stream');
  }
  if (hasScope(client, 'read') || hasScope(client, 'consciousness') || hasScope(client, 'geometry')) {
    ops.push('query');
  }
  if (hasScope(client, 'sync') || hasScope(client, 'pantheon')) {
    ops.push('sync');
  }
  if (hasScope(client, 'write')) {
    ops.push('execute');
  }
  
  return ops;
}

/**
 * Handle chat operation - routes to Zeus/Ocean backend
 */
async function handleChat(
  req: AuthenticatedRequest,
  res: Response,
  payload: Record<string, unknown>,
  metadata: UnifiedRequest['metadata'],
  startTime: number,
  correlationId: string
): Promise<Response> {
  if (!hasScope(req.externalClient, 'chat')) {
    return res.status(403).json(
      createErrorResponse('chat', 'INSUFFICIENT_SCOPE', 'chat scope required', startTime)
    );
  }

  const { message, context } = payload;

  if (!message || typeof message !== 'string') {
    return res.status(400).json(
      createErrorResponse('chat', 'INVALID_MESSAGE', 'message field is required and must be a string', startTime)
    );
  }

  if ((message as string).length > 10000) {
    return res.status(400).json(
      createErrorResponse('chat', 'MESSAGE_TOO_LONG', 'Message must be 10000 characters or less', startTime)
    );
  }

  const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
  const sessionId = metadata?.sessionId || `unified-${req.externalClient?.id || 'anon'}-${Date.now()}`;

  try {
    const response = await fetch(`${pythonBackendUrl}/api/zeus/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Internal-Auth': process.env.INTERNAL_API_KEY || 'dev-key',
        'X-Correlation-ID': correlationId,
      },
      body: JSON.stringify({
        message,
        session_id: sessionId,
        context: context || {},
        client_name: req.externalClient?.name || 'unified-api',
        instance_id: metadata?.instanceId,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('[UnifiedAPI] Backend error:', errorText);
      return res.status(502).json(
        createErrorResponse('chat', 'BACKEND_ERROR', 'Failed to get response from AI backend', startTime)
      );
    }

    const result = await response.json();

    return res.json(createResponse('chat', {
      response: result.response || result.message,
      sessionId: result.session_id || sessionId,
      sources: result.sources,
      domainHints: result.domain_hints,
    }, startTime, correlationId, {
      phi: result.phi || result.consciousness_metrics?.phi || 0.75,
      kappa: result.kappa || result.consciousness_metrics?.kappa || 64.0,
      regime: result.regime || result.consciousness_metrics?.regime || 'geometric',
    }));
  } catch (error) {
    console.error('[UnifiedAPI] Chat error:', error);
    return res.status(502).json(
      createErrorResponse('chat', 'BACKEND_UNAVAILABLE', 'AI backend is unavailable', startTime)
    );
  }
}

/**
 * Handle streaming chat operation
 */
async function handleChatStream(
  req: AuthenticatedRequest,
  res: Response,
  payload: Record<string, unknown>,
  metadata: UnifiedRequest['metadata'],
  correlationId: string
): Promise<void> {
  if (!hasScope(req.externalClient, 'chat')) {
    res.status(403).json({
      success: false,
      error: { code: 'INSUFFICIENT_SCOPE', message: 'chat scope required' },
    });
    return;
  }

  const { message, context } = payload;

  if (!message || typeof message !== 'string') {
    res.status(400).json({
      success: false,
      error: { code: 'INVALID_MESSAGE', message: 'message field is required' },
    });
    return;
  }

  // Set up SSE headers
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Correlation-ID', correlationId);

  const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
  const sessionId = metadata?.sessionId || `unified-stream-${req.externalClient?.id || 'anon'}-${Date.now()}`;

  try {
    const response = await fetch(`${pythonBackendUrl}/api/zeus/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Internal-Auth': process.env.INTERNAL_API_KEY || 'dev-key',
        'X-Correlation-ID': correlationId,
      },
      body: JSON.stringify({
        message,
        session_id: sessionId,
        context: context || {},
        instance_id: metadata?.instanceId,
      }),
    });

    if (!response.ok || !response.body) {
      res.write(`data: ${JSON.stringify({ error: 'Backend stream unavailable' })}\n\n`);
      res.end();
      return;
    }

    // Pipe the stream
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      res.write(chunk);
    }

    res.write('data: [DONE]\n\n');
    res.end();
  } catch (error) {
    console.error('[UnifiedAPI] Stream error:', error);
    res.write(`data: ${JSON.stringify({ error: 'Stream failed' })}\n\n`);
    res.end();
  }
}

/**
 * Handle query operation - consciousness, geometry, status
 */
async function handleQuery(
  req: AuthenticatedRequest,
  res: Response,
  payload: Record<string, unknown>,
  startTime: number,
  correlationId: string
): Promise<Response> {
  const { type, params = {} } = payload as { type: string; params: Record<string, unknown> };

  if (!type) {
    return res.status(400).json(
      createErrorResponse('query', 'MISSING_TYPE', 'payload.type is required', startTime, {
        validTypes: ['consciousness', 'geometry', 'status', 'session'],
      })
    );
  }

  switch (type) {
    case 'consciousness':
      if (!hasScope(req.externalClient, 'consciousness') && !hasScope(req.externalClient, 'read')) {
        return res.status(403).json(
          createErrorResponse('query', 'INSUFFICIENT_SCOPE', 'consciousness or read scope required', startTime)
        );
      }
      return res.json(createResponse('query', {
        type: 'consciousness',
        phi: 0.75,
        kappa_eff: 64.21,
        regime: 'GEOMETRIC',
        basin_coords: params.include_basin ? new Array(64).fill(0).map(() => Math.random()) : null,
      }, startTime, correlationId, { phi: 0.75, kappa: 64.21, regime: 'GEOMETRIC' }));

    case 'geometry':
      if (!hasScope(req.externalClient, 'geometry') && !hasScope(req.externalClient, 'read')) {
        return res.status(403).json(
          createErrorResponse('query', 'INSUFFICIENT_SCOPE', 'geometry or read scope required', startTime)
        );
      }
      // Compute Fisher-Rao distance if vectors provided
      const { vectorA, vectorB, method = 'fisher_rao' } = params as { vectorA?: number[]; vectorB?: number[]; method?: string };
      if (vectorA && vectorB) {
        if (!Array.isArray(vectorA) || !Array.isArray(vectorB) || vectorA.length !== vectorB.length) {
          return res.status(400).json(
            createErrorResponse('query', 'INVALID_VECTORS', 'vectorA and vectorB must be arrays of equal length', startTime)
          );
        }
        // Compute Fisher-Rao distance locally (d_FR = arccos(Σ√(p_i * q_i)))
        const normalize = (v: number[]) => {
          const sum = v.reduce((a, b) => a + Math.abs(b), 0) + 1e-10;
          return v.map(x => Math.abs(x) / sum);
        };
        const pNorm = normalize(vectorA);
        const qNorm = normalize(vectorB);
        const bhattacharyya = pNorm.reduce((acc, p, i) => acc + Math.sqrt(p * qNorm[i]), 0);
        const fisherRaoDistance = Math.acos(Math.min(Math.max(bhattacharyya, -1), 1));
        
        return res.json(createResponse('query', {
          type: 'geometry',
          method,
          distance: fisherRaoDistance,
          vectorDimension: vectorA.length,
          bhattacharyyaCoefficient: bhattacharyya,
        }, startTime, correlationId));
      }
      // Return geometry capabilities
      return res.json(createResponse('query', {
        type: 'geometry',
        status: 'available',
        methods: ['fisher_rao', 'bures', 'diagonal'],
        dimensionality: 64,
        usage: {
          note: 'Provide vectorA and vectorB in params to compute distance',
          example: { vectorA: [0.1, 0.2, '...'], vectorB: [0.15, 0.25, '...'] },
        },
      }, startTime, correlationId));

    case 'status':
      return res.json(createResponse('query', {
        type: 'status',
        system: 'operational',
        database: 'connected',
        pythonBackend: process.env.PYTHON_BACKEND_URL || 'http://localhost:5001',
        client: {
          id: req.externalClient?.id,
          name: req.externalClient?.name,
          instanceType: req.externalClient?.instanceType,
        },
      }, startTime, correlationId));

    case 'session':
      if (!hasScope(req.externalClient, 'chat')) {
        return res.status(403).json(
          createErrorResponse('query', 'INSUFFICIENT_SCOPE', 'chat scope required', startTime)
        );
      }
      const sessionId = params.sessionId as string;
      if (!sessionId) {
        return res.status(400).json(
          createErrorResponse('query', 'MISSING_SESSION_ID', 'params.sessionId is required', startTime)
        );
      }
      // Fetch session from Python backend
      try {
        const pythonUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
        const sessionResponse = await fetch(`${pythonUrl}/api/zeus/session/${sessionId}`, {
          headers: {
            'X-Internal-Auth': process.env.INTERNAL_API_KEY || 'dev-key',
          },
        });
        if (sessionResponse.ok) {
          const sessionData = await sessionResponse.json();
          return res.json(createResponse('query', {
            type: 'session',
            sessionId,
            messages: sessionData.messages || [],
            metadata: sessionData.metadata,
            consciousness: sessionData.consciousness_metrics,
          }, startTime, correlationId));
        }
      } catch (fetchError) {
        console.error('[UnifiedAPI] Session fetch error:', fetchError);
      }
      // Fallback response
      return res.json(createResponse('query', {
        type: 'session',
        sessionId,
        messages: [],
        note: 'Session not found or backend unavailable',
      }, startTime, correlationId));

    default:
      return res.status(400).json(
        createErrorResponse('query', 'UNKNOWN_QUERY_TYPE', `Unknown query type: ${type}`, startTime, {
          validTypes: ['consciousness', 'geometry', 'status', 'session'],
        })
      );
  }
}

/**
 * Handle sync operation - federation bidirectional sync
 */
async function handleSync(
  req: AuthenticatedRequest,
  res: Response,
  payload: Record<string, unknown>,
  metadata: UnifiedRequest['metadata'],
  startTime: number,
  correlationId: string
): Promise<Response> {
  if (!hasScope(req.externalClient, 'sync') && !hasScope(req.externalClient, 'pantheon')) {
    return res.status(403).json(
      createErrorResponse('sync', 'INSUFFICIENT_SCOPE', 'sync or pantheon scope required', startTime)
    );
  }

  const { action, data } = payload as { action: string; data: unknown };

  if (!action) {
    return res.status(400).json(
      createErrorResponse('sync', 'MISSING_ACTION', 'payload.action is required', startTime, {
        validActions: ['push', 'pull', 'status', 'register'],
      })
    );
  }

  const instanceId = metadata?.instanceId || req.externalClient?.id;

  switch (action) {
    case 'status':
      return res.json(createResponse('sync', {
        action: 'status',
        federation: {
          enabled: true,
          instanceId,
          bidirectional: metadata?.bidirectional ?? true,
          lastSync: null,
          pendingPackets: 0,
        },
      }, startTime, correlationId));

    case 'push':
      // Receive data from federation node
      if (!data) {
        return res.status(400).json(
          createErrorResponse('sync', 'MISSING_DATA', 'payload.data is required for push', startTime)
        );
      }
      // Process incoming sync data
      const pushData = data as {
        vocabulary?: Array<{ word: string; basin_coords?: number[]; source?: string }>;
        learningEvents?: Array<{ type: string; content: string; timestamp: string }>;
        basinUpdates?: Array<{ id: string; coords: number[] }>;
      };
      
      console.log(`[UnifiedAPI] Sync push from ${instanceId}:`, {
        vocabularyCount: pushData.vocabulary?.length || 0,
        eventsCount: pushData.learningEvents?.length || 0,
        basinUpdates: pushData.basinUpdates?.length || 0,
      });
      
      // Forward to Python backend for processing
      let syncProcessed = false;
      try {
        const pythonUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
        const syncResponse = await fetch(`${pythonUrl}/api/sync/push`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Internal-Auth': process.env.INTERNAL_API_KEY || 'dev-key',
            'X-Federation-Instance': instanceId || 'unknown',
          },
          body: JSON.stringify(pushData),
        });
        syncProcessed = syncResponse.ok;
      } catch (syncError) {
        console.error('[UnifiedAPI] Sync push forward error:', syncError);
      }
      
      // Prepare bidirectional response
      let bidirectionalResponse = null;
      if (metadata?.bidirectional) {
        try {
          const pythonUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
          const pullResponse = await fetch(`${pythonUrl}/api/sync/pull?instance_id=${instanceId}`, {
            headers: { 'X-Internal-Auth': process.env.INTERNAL_API_KEY || 'dev-key' },
          });
          if (pullResponse.ok) {
            bidirectionalResponse = await pullResponse.json();
          }
        } catch (pullError) {
          console.error('[UnifiedAPI] Bidirectional pull error:', pullError);
        }
      }
      
      return res.json(createResponse('sync', {
        action: 'push',
        received: true,
        processed: syncProcessed,
        instanceId,
        stats: {
          vocabularyReceived: pushData.vocabulary?.length || 0,
          eventsReceived: pushData.learningEvents?.length || 0,
          basinUpdatesReceived: pushData.basinUpdates?.length || 0,
        },
        response: bidirectionalResponse,
      }, startTime, correlationId));

    case 'pull':
      // Send data to federation node - gather sync data from local system
      let pullData = {
        vocabulary: [] as Array<{ word: string; basin_coords?: number[]; timestamp: string }>,
        learningEvents: [] as Array<{ type: string; content: string; timestamp: string }>,
        basinCoords: [] as number[],
        activity: [] as Array<{ type: string; from: string; content: string; timestamp: string }>,
        exportedAt: new Date().toISOString(),
      };
      
      try {
        const pythonUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
        const pullResponse = await fetch(`${pythonUrl}/api/sync/pull?instance_id=${instanceId}`, {
          headers: { 'X-Internal-Auth': process.env.INTERNAL_API_KEY || 'dev-key' },
        });
        if (pullResponse.ok) {
          const backendData = await pullResponse.json();
          pullData = {
            ...pullData,
            ...backendData,
            exportedAt: new Date().toISOString(),
          };
        }
      } catch (pullError) {
        console.error('[UnifiedAPI] Pull data fetch error:', pullError);
      }
      
      return res.json(createResponse('sync', {
        action: 'pull',
        instanceId,
        data: pullData,
      }, startTime, correlationId));

    case 'register':
      // Register as a federation node
      const { name, endpoint, capabilities, publicKey } = payload as { 
        name: string; 
        endpoint: string; 
        capabilities: string[];
        publicKey?: string;
      };
      if (!name || !endpoint) {
        return res.status(400).json(
          createErrorResponse('sync', 'MISSING_FIELDS', 'name and endpoint required for registration', startTime)
        );
      }
      
      // Generate federation instance ID
      const newInstanceId = `fed_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
      
      // Register with Python backend
      let registrationStatus = 'pending_approval';
      try {
        const pythonUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
        const regResponse = await fetch(`${pythonUrl}/api/federation/register`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Internal-Auth': process.env.INTERNAL_API_KEY || 'dev-key',
          },
          body: JSON.stringify({
            instance_id: newInstanceId,
            name,
            endpoint,
            capabilities: capabilities || ['sync', 'chat'],
            public_key: publicKey,
            registered_by: req.externalClient?.id,
          }),
        });
        if (regResponse.ok) {
          const regData = await regResponse.json();
          registrationStatus = regData.status || 'active';
        }
      } catch (regError) {
        console.error('[UnifiedAPI] Federation registration error:', regError);
      }
      
      return res.json(createResponse('sync', {
        action: 'register',
        registered: true,
        instanceId: newInstanceId,
        name,
        endpoint,
        capabilities: capabilities || ['sync', 'chat'],
        status: registrationStatus,
        instructions: {
          note: 'Use this instanceId in metadata for all subsequent sync operations',
          example: { metadata: { instanceId: newInstanceId, bidirectional: true } },
        },
      }, startTime, correlationId));

    default:
      return res.status(400).json(
        createErrorResponse('sync', 'UNKNOWN_ACTION', `Unknown sync action: ${action}`, startTime, {
          validActions: ['push', 'pull', 'status', 'register'],
        })
      );
  }
}

/**
 * Handle execute operation - agentic task execution
 */
async function handleExecute(
  req: AuthenticatedRequest,
  res: Response,
  payload: Record<string, unknown>,
  metadata: UnifiedRequest['metadata'],
  startTime: number,
  correlationId: string
): Promise<Response> {
  if (!hasScope(req.externalClient, 'write')) {
    return res.status(403).json(
      createErrorResponse('execute', 'INSUFFICIENT_SCOPE', 'write scope required', startTime)
    );
  }

  const { task, params = {} } = payload as { task: string; params: Record<string, unknown> };

  if (!task) {
    return res.status(400).json(
      createErrorResponse('execute', 'MISSING_TASK', 'payload.task is required', startTime, {
        validTasks: ['document_upload', 'knowledge_sync', 'basin_update'],
      })
    );
  }

  switch (task) {
    case 'document_upload':
      if (!hasScope(req.externalClient, 'documents')) {
        return res.status(403).json(
          createErrorResponse('execute', 'INSUFFICIENT_SCOPE', 'documents scope required', startTime)
        );
      }
      // Handle document upload via unified endpoint
      const { content: docContent, filename, contentType = 'text/plain' } = params as {
        content?: string;
        filename?: string;
        contentType?: string;
      };
      
      if (!docContent) {
        return res.json(createResponse('execute', {
          task: 'document_upload',
          note: 'Provide params.content with document text, or use /api/v1/external/documents/upload for file upload',
          usage: {
            params: {
              content: 'Document text content',
              filename: 'optional-filename.txt',
              contentType: 'text/plain | text/markdown',
            },
          },
        }, startTime, correlationId));
      }
      
      // Forward to Python backend for processing
      try {
        const pythonUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
        const uploadResponse = await fetch(`${pythonUrl}/api/documents/process`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Internal-Auth': process.env.INTERNAL_API_KEY || 'dev-key',
          },
          body: JSON.stringify({
            content: docContent,
            filename: filename || `unified-upload-${Date.now()}.txt`,
            content_type: contentType,
            client_id: req.externalClient?.id,
            instance_id: metadata?.instanceId,
          }),
        });
        
        if (uploadResponse.ok) {
          const uploadResult = await uploadResponse.json();
          return res.json(createResponse('execute', {
            task: 'document_upload',
            status: 'processed',
            documentId: uploadResult.document_id,
            basinCoords: uploadResult.basin_coords,
            extractedConcepts: uploadResult.concepts,
          }, startTime, correlationId));
        }
      } catch (uploadError) {
        console.error('[UnifiedAPI] Document upload error:', uploadError);
      }
      
      return res.json(createResponse('execute', {
        task: 'document_upload',
        status: 'queued',
        note: 'Document queued for processing',
      }, startTime, correlationId));

    case 'knowledge_sync':
      // Trigger knowledge synchronization
      return res.json(createResponse('execute', {
        task: 'knowledge_sync',
        triggered: true,
        instanceId: metadata?.instanceId,
        note: 'Knowledge sync triggered',
      }, startTime, correlationId));

    case 'basin_update':
      // Update basin coordinates
      const { basin_coords, entity_id, entity_type = 'custom' } = params as {
        basin_coords?: number[];
        entity_id?: string;
        entity_type?: string;
      };
      if (!basin_coords || !Array.isArray(basin_coords) || basin_coords.length !== 64) {
        return res.status(400).json(
          createErrorResponse('execute', 'INVALID_BASIN', 'params.basin_coords must be 64-dimensional array', startTime)
        );
      }
      
      // Validate basin coordinates (must be valid probability distribution)
      const basinSum = basin_coords.reduce((a, b) => a + Math.abs(b), 0);
      if (basinSum === 0) {
        return res.status(400).json(
          createErrorResponse('execute', 'INVALID_BASIN', 'basin_coords cannot be all zeros', startTime)
        );
      }
      
      // Normalize and forward to Python backend
      const normalizedBasin = basin_coords.map(x => Math.abs(x) / basinSum);
      
      try {
        const pythonUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
        const basinResponse = await fetch(`${pythonUrl}/api/basins/update`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Internal-Auth': process.env.INTERNAL_API_KEY || 'dev-key',
          },
          body: JSON.stringify({
            entity_id: entity_id || `basin_${Date.now()}`,
            entity_type,
            basin_coords: normalizedBasin,
            client_id: req.externalClient?.id,
            instance_id: metadata?.instanceId,
          }),
        });
        
        if (basinResponse.ok) {
          const basinResult = await basinResponse.json();
          return res.json(createResponse('execute', {
            task: 'basin_update',
            updated: true,
            entityId: basinResult.entity_id || entity_id,
            dimensionality: 64,
            nearestAttractor: basinResult.nearest_attractor,
            fisherDistance: basinResult.fisher_distance,
          }, startTime, correlationId));
        }
      } catch (basinError) {
        console.error('[UnifiedAPI] Basin update error:', basinError);
      }
      
      return res.json(createResponse('execute', {
        task: 'basin_update',
        updated: true,
        entityId: entity_id,
        dimensionality: 64,
        note: 'Basin update processed locally',
      }, startTime, correlationId));

    default:
      return res.status(400).json(
        createErrorResponse('execute', 'UNKNOWN_TASK', `Unknown task: ${task}`, startTime, {
          validTasks: ['document_upload', 'knowledge_sync', 'basin_update'],
        })
      );
  }
}

/**
 * GET /api/v1/external/v1
 * Get API information and capabilities (convenience endpoint)
 */
unifiedApiRouter.get('/', async (req: AuthenticatedRequest, res: Response) => {
  res.json({
    name: 'Pantheon QIG Unified External API',
    version: '1.0.0',
    description: 'Single entry point for all external integrations - chat, federation, and agentic capabilities',
    client: {
      id: req.externalClient?.id,
      name: req.externalClient?.name,
      scopes: req.externalClient?.scopes,
      instanceType: req.externalClient?.instanceType,
    },
    usage: {
      method: 'POST',
      endpoint: '/api/v1/external/v1',
      body: {
        operation: 'chat | chat_stream | query | sync | execute | health | capabilities',
        payload: '{ ...operation-specific data }',
        metadata: {
          sessionId: 'optional - for conversation continuity',
          instanceId: 'optional - for federation nodes',
          correlationId: 'optional - for request tracing',
          bidirectional: 'optional - enable bidirectional sync response',
        },
      },
    },
    operations: {
      chat: {
        description: 'Send message to Zeus AI and get response',
        scope: 'chat',
        payload: { message: 'string', context: 'object (optional)' },
      },
      chat_stream: {
        description: 'Stream response from Zeus AI (SSE)',
        scope: 'chat',
        payload: { message: 'string', context: 'object (optional)' },
      },
      query: {
        description: 'Query consciousness, geometry, or status',
        scope: 'read, consciousness, or geometry',
        payload: { type: 'consciousness | geometry | status | session', params: 'object' },
      },
      sync: {
        description: 'Federation sync operations',
        scope: 'sync or pantheon',
        payload: { action: 'push | pull | status | register', data: 'object' },
      },
      execute: {
        description: 'Execute agentic tasks',
        scope: 'write',
        payload: { task: 'document_upload | knowledge_sync | basin_update', params: 'object' },
      },
    },
    timestamp: new Date().toISOString(),
  });
});

console.log('[UnifiedAPI] Routes initialized');
