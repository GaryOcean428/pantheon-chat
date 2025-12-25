/**
 * Simple Chat API for External UIs and Federation Nodes
 * 
 * This is the primary endpoint for external systems to use Zeus chat
 * and agentic capabilities. Designed for simplicity and ease of integration.
 * 
 * Endpoint: POST /api/v1/external/chat
 * 
 * Example usage:
 * ```
 * curl -X POST https://your-instance.com/api/v1/external/chat \
 *   -H "Authorization: Bearer YOUR_API_KEY" \
 *   -H "Content-Type: application/json" \
 *   -d '{"message": "What is consciousness?"}'
 * ```
 */

import { Router, Request, Response } from 'express';
import { authenticateExternalApi } from './auth';

const router = Router();

// Types for external chat API
export interface ChatRequest {
  message: string;
  sessionId?: string;
  stream?: boolean;
  context?: {
    previousMessages?: Array<{ role: 'user' | 'assistant'; content: string }>;
    systemPrompt?: string;
    temperature?: number;
  };
  metadata?: {
    instanceId?: string;  // For federation nodes
    clientName?: string;  // For external UIs
    clientVersion?: string;
  };
}

export interface ChatResponse {
  success: boolean;
  response: string;
  sessionId: string;
  metrics?: {
    phi: number;
    kappa: number;
    regime: 'linear' | 'geometric' | 'breakdown';
    completionReason: string;
  };
  sources?: Array<{
    title: string;
    url?: string;
    relevance: number;
  }>;
  error?: string;
}

// Python backend URL
const PYTHON_BACKEND = process.env.QIG_BACKEND_URL || 'http://localhost:5001';

/**
 * Simple Chat Endpoint
 * 
 * POST /api/v1/external/chat
 * 
 * This is the primary endpoint for external chat UIs and federation nodes.
 * It provides a simple interface to Zeus chat with optional streaming.
 */
router.post('/', authenticateExternalApi, async (req: Request, res: Response) => {
  try {
    const body = req.body as ChatRequest;
    
    if (!body.message || typeof body.message !== 'string') {
      return res.status(400).json({
        success: false,
        error: 'message is required and must be a string'
      });
    }
    
    // Generate session ID if not provided
    const sessionId = body.sessionId || `ext-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    // Check if streaming is requested
    if (body.stream) {
      return handleStreamingChat(req, res, body, sessionId);
    }
    
    // Non-streaming chat
    const zeusPayload = {
      message: body.message,
      conversation_id: sessionId,
      context: body.context?.previousMessages || [],
      system_prompt: body.context?.systemPrompt,
      temperature: body.context?.temperature,
      metadata: {
        source: 'external_api',
        instance_id: body.metadata?.instanceId,
        client_name: body.metadata?.clientName,
        client_version: body.metadata?.clientVersion
      }
    };
    
    // Call Python backend Zeus chat
    const response = await fetch(`${PYTHON_BACKEND}/api/zeus/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Internal-Auth': process.env.INTERNAL_API_KEY || 'dev-key'
      },
      body: JSON.stringify(zeusPayload)
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('[ExternalChat] Zeus chat error:', errorText);
      return res.status(response.status).json({
        success: false,
        error: 'Chat service unavailable',
        sessionId
      });
    }
    
    const zeusResponse = await response.json();
    
    // Format response for external clients
    const chatResponse: ChatResponse = {
      success: true,
      response: zeusResponse.response || zeusResponse.message || '',
      sessionId,
      metrics: zeusResponse.metrics ? {
        phi: zeusResponse.metrics.phi || zeusResponse.phi || 0.5,
        kappa: zeusResponse.metrics.kappa || zeusResponse.kappa || 64,
        regime: zeusResponse.metrics.regime || zeusResponse.regime || 'geometric',
        completionReason: zeusResponse.completion_reason || 'natural'
      } : undefined,
      sources: zeusResponse.sources || []
    };
    
    return res.json(chatResponse);
    
  } catch (error) {
    console.error('[ExternalChat] Error:', error);
    return res.status(500).json({
      success: false,
      error: 'Internal server error'
    });
  }
});

/**
 * Streaming Chat Handler
 * 
 * Uses Server-Sent Events (SSE) for real-time streaming responses.
 */
async function handleStreamingChat(
  req: Request,
  res: Response,
  body: ChatRequest,
  sessionId: string
): Promise<void> {
  // Set up SSE headers
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  
  // Send session ID immediately
  res.write(`data: ${JSON.stringify({ type: 'session', sessionId })}\n\n`);
  
  try {
    const zeusPayload = {
      message: body.message,
      conversation_id: sessionId,
      stream: true,
      context: body.context?.previousMessages || [],
      system_prompt: body.context?.systemPrompt,
      temperature: body.context?.temperature,
      metadata: {
        source: 'external_api_stream',
        instance_id: body.metadata?.instanceId,
        client_name: body.metadata?.clientName
      }
    };
    
    // Call Python backend streaming endpoint
    const response = await fetch(`${PYTHON_BACKEND}/api/zeus/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Internal-Auth': process.env.INTERNAL_API_KEY || 'dev-key'
      },
      body: JSON.stringify(zeusPayload)
    });
    
    if (!response.ok || !response.body) {
      res.write(`data: ${JSON.stringify({ type: 'error', error: 'Stream unavailable' })}\n\n`);
      res.end();
      return;
    }
    
    // Forward the stream from Python backend
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    let fullResponse = '';
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') {
            // Send completion event
            res.write(`data: ${JSON.stringify({ 
              type: 'done', 
              fullResponse,
              sessionId 
            })}\n\n`);
          } else {
            try {
              const parsed = JSON.parse(data);
              
              // Forward token chunks
              if (parsed.token || parsed.content || parsed.delta) {
                const token = parsed.token || parsed.content || parsed.delta;
                fullResponse += token;
                res.write(`data: ${JSON.stringify({ type: 'token', token })}\n\n`);
              }
              
              // Forward metrics if present
              if (parsed.metrics || parsed.phi) {
                res.write(`data: ${JSON.stringify({ 
                  type: 'metrics',
                  phi: parsed.metrics?.phi || parsed.phi,
                  kappa: parsed.metrics?.kappa || parsed.kappa,
                  regime: parsed.metrics?.regime || parsed.regime
                })}\n\n`);
              }
            } catch {
              // Forward raw data if not JSON
              fullResponse += data;
              res.write(`data: ${JSON.stringify({ type: 'token', token: data })}\n\n`);
            }
          }
        }
      }
    }
    
    res.end();
    
  } catch (error) {
    console.error('[ExternalChat] Streaming error:', error);
    res.write(`data: ${JSON.stringify({ type: 'error', error: 'Stream error' })}\n\n`);
    res.end();
  }
}

/**
 * Get Session History
 * 
 * GET /api/v1/external/chat/:sessionId
 */
router.get('/:sessionId', authenticateExternalApi, async (req: Request, res: Response) => {
  try {
    const { sessionId } = req.params;
    
    // Get session from Python backend
    const response = await fetch(`${PYTHON_BACKEND}/api/zeus/session/${sessionId}`, {
      headers: {
        'X-Internal-Auth': process.env.INTERNAL_API_KEY || 'dev-key'
      }
    });
    
    if (!response.ok) {
      return res.status(404).json({
        success: false,
        error: 'Session not found'
      });
    }
    
    const session = await response.json();
    
    return res.json({
      success: true,
      sessionId,
      messages: session.messages || [],
      created: session.created,
      lastActivity: session.last_activity
    });
    
  } catch (error) {
    console.error('[ExternalChat] Session error:', error);
    return res.status(500).json({
      success: false,
      error: 'Failed to retrieve session'
    });
  }
});

/**
 * Health Check for External Clients
 * 
 * GET /api/v1/external/chat/health
 */
router.get('/health', async (_req: Request, res: Response) => {
  try {
    // Check Python backend health
    const response = await fetch(`${PYTHON_BACKEND}/health`, {
      signal: AbortSignal.timeout(5000)
    });
    
    const backendHealthy = response.ok;
    
    return res.json({
      success: true,
      status: backendHealthy ? 'healthy' : 'degraded',
      services: {
        api: 'healthy',
        zeus: backendHealthy ? 'healthy' : 'unavailable'
      },
      timestamp: new Date().toISOString()
    });
    
  } catch {
    return res.json({
      success: true,
      status: 'degraded',
      services: {
        api: 'healthy',
        zeus: 'unavailable'
      },
      timestamp: new Date().toISOString()
    });
  }
});

export const chatApiRouter = router;
export default router;
