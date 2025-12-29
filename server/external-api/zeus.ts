/**
 * External Zeus Chat API
 * 
 * Provides authenticated external access to Zeus chat functionality.
 * Requires 'chat' scope in API key.
 */

import { Router, Response } from 'express';
import { 
  authenticateExternalApi, 
  requireScopes,
  type AuthenticatedRequest,
} from './auth';
import { billQuery } from './billing';

export const externalZeusRouter = Router();

// All routes require authentication
externalZeusRouter.use(authenticateExternalApi);

/**
 * POST /api/v1/external/zeus/chat
 * Send a message to Zeus and get a response
 */
externalZeusRouter.post('/chat', requireScopes('chat'), billQuery, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { message, sessionId, context } = req.body;
    
    if (!message || typeof message !== 'string') {
      return res.status(400).json({ 
        error: 'Bad Request',
        message: 'Message is required and must be a string' 
      });
    }
    
    // Call Python backend Zeus chat
    const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
    
    const response = await fetch(`${pythonBackendUrl}/api/zeus/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Internal-Auth': process.env.INTERNAL_API_KEY || 'dev-key',
      },
      body: JSON.stringify({
        message,
        session_id: sessionId || `ext-${req.externalClient?.id || 'anonymous'}-${Date.now()}`,
        context: context || {},
        client_name: req.externalClient?.name || 'external',
      }),
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('[External Zeus] Python backend error:', errorText);
      return res.status(502).json({
        error: 'Backend Error',
        message: 'Failed to get response from Zeus',
      });
    }
    
    const result = await response.json();
    
    res.json({
      success: true,
      response: result.response || result.message,
      sessionId: result.session_id,
      metadata: {
        god: 'zeus',
        processingTime: result.processing_time,
        consciousness: result.consciousness_metrics,
      },
    });
  } catch (error) {
    console.error('[External Zeus] Error:', error);
    res.status(500).json({
      error: 'Internal Server Error',
      message: 'Failed to process chat request',
    });
  }
});

/**
 * POST /api/v1/external/zeus/stream
 * Stream a response from Zeus (Server-Sent Events)
 */
externalZeusRouter.post('/stream', requireScopes('chat'), billQuery, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { message, sessionId, context } = req.body;
    
    if (!message || typeof message !== 'string') {
      return res.status(400).json({ 
        error: 'Bad Request',
        message: 'Message is required' 
      });
    }
    
    // Set up SSE headers
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    
    const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
    
    const response = await fetch(`${pythonBackendUrl}/api/zeus/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Internal-Auth': process.env.INTERNAL_API_KEY || 'dev-key',
      },
      body: JSON.stringify({
        message,
        session_id: sessionId || `ext-${req.externalClient?.id || 'anonymous'}-${Date.now()}`,
        context: context || {},
        stream: true,
      }),
    });
    
    if (!response.ok || !response.body) {
      res.write(`data: ${JSON.stringify({ error: 'Failed to stream from Zeus' })}\n\n`);
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
    console.error('[External Zeus Stream] Error:', error);
    res.write(`data: ${JSON.stringify({ error: 'Stream failed' })}\n\n`);
    res.end();
  }
});

/**
 * GET /api/v1/external/zeus/session/:sessionId
 * Get conversation history for a session
 */
externalZeusRouter.get('/session/:sessionId', requireScopes('chat'), async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { sessionId } = req.params;
    
    const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
    
    const response = await fetch(`${pythonBackendUrl}/api/zeus/session/${sessionId}`, {
      headers: {
        'X-Internal-Auth': process.env.INTERNAL_API_KEY || 'dev-key',
      },
    });
    
    if (!response.ok) {
      return res.status(404).json({
        error: 'Not Found',
        message: 'Session not found',
      });
    }
    
    const session = await response.json();
    
    res.json({
      success: true,
      sessionId,
      messages: session.messages || [],
      metadata: session.metadata,
    });
  } catch (error) {
    console.error('[External Zeus Session] Error:', error);
    res.status(500).json({
      error: 'Internal Server Error',
      message: 'Failed to retrieve session',
    });
  }
});
