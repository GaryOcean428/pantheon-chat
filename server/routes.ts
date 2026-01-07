import type { Express, RequestHandler } from "express";
import rateLimit from "express-rate-limit";
import multer from "multer";
import { createServer, type Server } from "http";
import { z } from "zod";
import { scorePhraseQIG } from "./qig-universal.js";
import { storage } from "./storage";
import { telemetryRouter } from "./telemetry-api";
import { backendTelemetryRouter } from "./backend-telemetry-api";
import TelemetryStreamer from "./telemetry-websocket";
import KernelActivityStreamer from "./kernel-activity-websocket";
import MeshNetworkStreamer from "./mesh-network-websocket";
import telemetryDashboardRouter from "./routes/telemetry";
import { pythonReadiness, createTypedErrorResponse } from "./python-readiness";

// WebSocket message validation schema (addresses Issue 13/14 from bottleneck report)
const wsMessageSchema = z.object({
  type: z.enum(["heartbeat", "basin-delta", "set-mode"]),
  data: z.any().optional(),
  mode: z.enum(["full", "partial", "observer"]).optional(),
});

// WebSocket rate limiter: Track message counts per connection
const wsRateLimiter = new Map<string, { count: number; resetTime: number }>();
const WS_RATE_LIMIT = 100; // Max messages per window
const WS_RATE_WINDOW = 60000; // 1 minute window

function checkWsRateLimit(peerId: string): boolean {
  const now = Date.now();
  const entry = wsRateLimiter.get(peerId);

  if (!entry || now > entry.resetTime) {
    wsRateLimiter.set(peerId, { count: 1, resetTime: now + WS_RATE_WINDOW });
    return true;
  }

  if (entry.count >= WS_RATE_LIMIT) {
    return false;
  }

  entry.count++;
  return true;
}

import documentsRouter from "./routes/documents";
import { logger } from './lib/logger';
import {
  adminRouter,
  attentionMetricsRouter,
  authRouter,
  autonomicRouter,
  autonomicAgencyRouter,
  federationRouter,
  consciousnessRouter,
  formatRouter,
  nearMissRouter,
  oceanRouter,
  olympusRouter,
  pythonProxiesRouter,
  searchRouter,
  sscBridgeRouter,
  ucpRouter,
  vocabularyRouter,
  zettelkastenRouter,
  // New comprehensive routers
  immuneRouter,
  trainingRouter,
  memoryRouter,
  feedbackRouter,
  coordizerRouter,
} from "./routes/index";

import { externalRouter as externalApiRouter, documentsRouter as externalDocsRouter, initExternalWebSocket } from "./external-api";
import apiDocsRouter from "./routes/api-docs";

import type { Candidate } from "@shared/schema";
import { randomUUID } from "crypto";
import { getErrorMessage } from "./lib/error-utils";
import { autoCycleManager } from "./auto-cycle-manager";
import { oceanSessionManager } from "./ocean-session-manager";
import { isAuthenticated, setupAuth } from "./replitAuth";
import { searchCoordinator } from "./search-coordinator";

const strictLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 5,
  message: { error: "Rate limit exceeded. Please try again later." },
  standardHeaders: true,
  legacyHeaders: false,
});


// Set up auto-cycle callback to start sessions via ocean session manager
autoCycleManager.setOnCycleCallback(
  async (addressId: string, address: string) => {
    try {
      console.log(
        `[AutoCycle] Starting session for address: ${address.slice(0, 16)}...`
      );
      oceanSessionManager.setAddressIdMapping(address, addressId);
      await oceanSessionManager.startSession(address);
    } catch (error) {
      // Handle SESSION_BUSY - current session hasn't met minimum requirements
      if (error instanceof Error && error.message.startsWith('SESSION_BUSY:')) {
        // Don't log as error - this is expected behavior during minimum runtime window
        return;
      }
      throw error; // Re-throw other errors
    }
  }
);

export async function registerRoutes(app: Express): Promise<Server> {
  // Start Python backend readiness tracker
  pythonReadiness.start();
  
  // Handle favicon.ico requests - redirect to favicon.png
  app.get("/favicon.ico", (req, res) => {
    res.redirect(301, "/favicon.png");
  });

  // Python backend status endpoint for frontend initialization UI
  app.get("/api/python/status", (req, res) => {
    const state = pythonReadiness.getState();
    res.json({
      ...state,
      ready: state.status === 'ready',
    });
  });

  // SSE endpoint for Python status updates
  app.get("/api/python/status/stream", (req, res) => {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    
    const sendState = () => {
      const state = pythonReadiness.getState();
      res.write(`data: ${JSON.stringify({ ...state, ready: state.status === 'ready' })}\n\n`);
    };
    
    sendState();
    const unsubscribe = pythonReadiness.subscribe(sendState);
    
    req.on('close', () => {
      unsubscribe();
    });
  });

  // CRITICAL: Simple liveness endpoint for Autoscale deployment health checks
  app.get("/health", (req, res) => {
    res.status(200).json({
      status: "ok",
      timestamp: Date.now(),
    });
  });

  // Comprehensive health endpoint for frontend monitoring
  app.get("/api/health", async (req, res) => {
    const startTime = Date.now();
    const subsystems: Record<string, any> = {
      database: { status: 'down' as const, message: 'Not checked' },
      pythonBackend: { status: 'down' as const, message: 'Not checked' },
      storage: { status: 'healthy' as const, message: 'In-memory storage active' },
    };

    try {
      const { db } = await import("./db");
      if (db) {
        const dbStart = Date.now();
        await db.execute('SELECT 1');
        subsystems.database = {
          status: 'healthy',
          latency: Date.now() - dbStart,
          message: 'PostgreSQL connected',
        };
      } else {
        subsystems.database = { status: 'degraded', message: 'No database configured' };
      }
    } catch (error) {
      subsystems.database = { status: 'down', message: 'Database connection failed' };
    }

    try {
      const backendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
      const pyStart = Date.now();
      const response = await fetch(`${backendUrl}/health`, { 
        method: 'GET',
        signal: AbortSignal.timeout(5000),
      });
      if (response.ok) {
        subsystems.pythonBackend = {
          status: 'healthy',
          latency: Date.now() - pyStart,
          message: 'Python QIG backend running',
        };
      } else {
        subsystems.pythonBackend = { status: 'degraded', message: 'Python backend returned error' };
      }
    } catch (error) {
      subsystems.pythonBackend = { status: 'down', message: 'Python backend unreachable' };
    }

    const statuses = Object.values(subsystems).map((s: any) => s.status);
    let overallStatus: 'healthy' | 'degraded' | 'down' = 'healthy';
    if (statuses.includes('down')) {
      overallStatus = statuses.every(s => s === 'down') ? 'down' : 'degraded';
    } else if (statuses.includes('degraded')) {
      overallStatus = 'degraded';
    }

    res.status(200).json({
      status: overallStatus,
      timestamp: Date.now(),
      uptime: process.uptime(),
      subsystems,
      version: process.env.npm_package_version || '1.0.0',
    });
  });

  const { db } = await import("./db");
  let authEnabled = !!db;

  if (authEnabled) {
    const authSetupSuccess = await setupAuth(app);
    if (authSetupSuccess) {
      console.log("[Auth] Replit Auth enabled");

      app.get("/api/auth/user", isAuthenticated, async (req: any, res) => {
        try {
          const { getCachedUser } = await import("./replitAuth");
          const cachedUser = getCachedUser(req.user);

          if (cachedUser) {
            const { cachedAt: _cachedAt, ...userResponse } = cachedUser;
            return res.json(userResponse);
          }

          const userId = req.user.claims.sub;
          const user = await storage.getUser(userId);

          if (user) {
            req.user.cachedProfile = {
              ...user,
              cachedAt: Date.now(),
            };
          }

          res.json(user);
        } catch (error) {
          console.error("Error fetching user:", error);
          res.status(500).json({ message: "Failed to fetch user" });
        }
      });
    } else {
      console.error("[Auth] Replit Auth setup failed - falling back to no auth");
      authEnabled = false;  // Fall through to disabled handling below
    }
  }

  if (!authEnabled) {
    console.log("[Auth] Replit Auth disabled (no DATABASE_URL or setup failed)");

    app.get("/api/auth/user", (req, res) => {
      res.status(503).json({
        message: "Authentication unavailable - database not provisioned or SESSION_SECRET missing.",
      });
    });

    app.get("/api/login", (req, res) => {
      res.status(503).json({
        message: "Authentication unavailable - database not provisioned or SESSION_SECRET missing.",
      });
    });

    app.get("/api/logout", (req, res) => {
      res.status(503).json({
        message: "Authentication unavailable - database not provisioned or SESSION_SECRET missing.",
      });
    });
  }

  app.get("/reset", (req, res) => {
    res.clearCookie("connect.sid");
    res.send(`<!DOCTYPE html>
<html><head><title>Reset Session</title></head>
<body style="font-family:sans-serif;text-align:center;padding:50px">
<h1>Session Reset</h1>
<p>Clearing your session...</p>
<script>
document.cookie.split(';').forEach(c => {
  document.cookie = c.trim().split('=')[0] + '=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/';
});
localStorage.clear();
sessionStorage.clear();
setTimeout(() => { window.location.href = '/'; }, 1000);
</script>
</body></html>`);
  });

  app.get("/api/clear-session", (req, res) => {
    if (req.session) {
      req.session.destroy((err) => {
        if (err) {
          console.error("[Auth] Session destroy error:", err);
        }
        res.clearCookie("connect.sid");
        res.redirect("/");
      });
    } else {
      res.clearCookie("connect.sid");
      res.redirect("/");
    }
  });

  // ============================================================
  // MOUNT SUB-ROUTERS (Modular Route Organization)
  // ============================================================
  app.use("/api/auth", authRouter);
  app.use("/api/consciousness", consciousnessRouter);
  app.use("/api/near-misses", nearMissRouter);
  app.use("/api/attention-metrics", attentionMetricsRouter);
  app.use("/api/ucp", ucpRouter);
  app.use("/api/vocabulary", vocabularyRouter);
  app.use("/api", searchRouter);
  app.use("/api/search", searchRouter);
  app.use("/api/format", formatRouter);
  app.use("/api/ocean", oceanRouter);
  app.use("/api", adminRouter);
  app.use("/api/olympus", olympusRouter);
  app.use("/api/documents", externalDocsRouter);
  app.use("/api/docs", apiDocsRouter);
  app.use("/api/qig/autonomic/agency", autonomicAgencyRouter);
  app.use("/api/federation", federationRouter);
  app.use("/api/zettelkasten", zettelkastenRouter);
  app.use("/api", pythonProxiesRouter);

  // New comprehensive routers (full Python backend proxies)
  app.use("/api/autonomic", autonomicRouter);
  app.use("/api/immune", immuneRouter);
  app.use("/api/training", trainingRouter);
  app.use("/api/memory", memoryRouter);
  app.use("/api/feedback", feedbackRouter);
  app.use("/api/coordize", coordizerRouter);
  
  // SSC Bridge - Connects to SearchSpaceCollapse for Bitcoin recovery
  app.use("/api/ssc", sscBridgeRouter);

  // Mount telemetry routers
  app.use("/api/telemetry", telemetryRouter);
  app.use("/api/backend-telemetry", backendTelemetryRouter);
  app.use("/api/v1/telemetry", telemetryDashboardRouter);
  app.use("/api/v1/external", externalApiRouter);

  console.log("[Routes] All sub-routers mounted");
  console.log("[Routes] New routers: autonomic, immune, training, memory, feedback, coordize, ssc");

  // Investigation status endpoint
  app.get("/api/investigation/status", (req, res) => {
    try {
      const status = oceanSessionManager.getInvestigationStatus();
      res.json(status);
    } catch (error: unknown) {
      console.error("[API] Investigation status error:", getErrorMessage(error));
      res.status(500).json({ error: getErrorMessage(error) });
    }
  });

  // Auto-cycle endpoints
  // Status is public (read-only), but modification endpoints require authentication
  app.get("/api/auto-cycle/status", (req, res) => {
    try {
      const status = autoCycleManager.getStatus();
      const position = autoCycleManager.getPositionString();
      res.json({ success: true, ...status, positionString: position });
    } catch (error: unknown) {
      res.status(500).json({ error: getErrorMessage(error) });
    }
  });

  // Conditional auth middleware - only enforce if auth is enabled
  const requireAuthIfEnabled: RequestHandler = authEnabled
    ? isAuthenticated
    : ((_req, _res, next) => next());

  app.post("/api/auto-cycle/enable", requireAuthIfEnabled, async (req, res) => {
    try {
      const result = await autoCycleManager.enable();
      res.json({ success: result.success, message: result.message, status: autoCycleManager.getStatus() });
    } catch (error: unknown) {
      res.status(500).json({ error: getErrorMessage(error) });
    }
  });

  app.post("/api/auto-cycle/disable", requireAuthIfEnabled, (req, res) => {
    try {
      const result = autoCycleManager.disable();
      res.json({ success: result.success, message: result.message, status: autoCycleManager.getStatus() });
    } catch (error: unknown) {
      res.status(500).json({ error: getErrorMessage(error) });
    }
  });

  app.post("/api/auto-cycle/force-resume", requireAuthIfEnabled, async (req, res) => {
    try {
      const result = await autoCycleManager.forceResume();
      res.json({ success: result.success, message: result.message, status: autoCycleManager.getStatus() });
    } catch (error: unknown) {
      res.status(500).json({ error: getErrorMessage(error) });
    }
  });

  // Learning upload endpoint
  const uploadMiddleware = multer({ 
    storage: multer.memoryStorage(),
    limits: { fileSize: 5 * 1024 * 1024 },
    fileFilter: (_req, file, cb) => {
      if (file.originalname.toLowerCase().endsWith('.md')) {
        cb(null, true);
      } else {
        cb(new Error('Only .md files accepted'));
      }
    }
  });

  app.post("/api/learning/upload", requireAuthIfEnabled, uploadMiddleware.single('file'), async (req: any, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'No file provided' });
      }

      const content = req.file.buffer.toString('utf-8');
      let text = content
        .replace(/```[\s\S]*?```/g, ' ')
        .replace(/`[^`]+`/g, ' ')
        .replace(/https?:\/\/\S+/g, ' ')
        .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
        .replace(/<[^>]+>/g, ' ');
      
      const wordPattern = /[a-zA-Z][a-zA-Z'-]*[a-zA-Z]|[a-zA-Z]{3,}/g;
      const rawWords = text.toLowerCase().match(wordPattern) || [];
      const stopWords = new Set(['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'will', 'with', 'this', 'that', 'from', 'they', 'were', 'said', 'each', 'which', 'their', 'would', 'there', 'could', 'other', 'into', 'more', 'some', 'than', 'them', 'these', 'then', 'its', 'also', 'just', 'only', 'come', 'made', 'may', 'now', 'way', 'many', 'like', 'use', 'such', 'when', 'what', 'how', 'who', 'did', 'get', 'very', 'being', 'about']);
      
      const validWords = rawWords.filter((word: string) => 
        word.length >= 3 && !stopWords.has(word) && !/^\d+$/.test(word)
      );
      
      const wordCounts: Record<string, number> = {};
      for (const word of validWords) {
        wordCounts[word] = (wordCounts[word] || 0) + 1;
      }
      
      const backendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
      try {
        const formData = new FormData();
        const blob = new Blob([req.file.buffer], { type: req.file.mimetype });
        formData.append('file', blob, req.file.originalname);
        
        const response = await fetch(`${backendUrl}/api/vocabulary/upload-markdown`, {
          method: 'POST',
          body: formData,
          signal: AbortSignal.timeout(10000),
        });
        
        if (response.ok) {
          const data = await response.json();
          return res.json(data);
        }
      } catch {
        console.log("[API] Learning upload - Python backend unavailable, using fallback");
      }
      
      res.json({
        success: true,
        filename: req.file.originalname,
        words_processed: Object.keys(wordCounts).length,
        words_learned: Math.min(Object.keys(wordCounts).length, 10),
        unique_words: Object.keys(wordCounts).length,
        total_occurrences: validWords.length,
        sample_words: Object.keys(wordCounts).slice(0, 20),
        mode: 'fallback',
        timestamp: new Date().toISOString()
      });
    } catch (error: unknown) {
      res.status(500).json({ error: getErrorMessage(error) });
    }
  });

  // Search budget proxies
  app.get("/api/search/budget/status", async (req, res) => {
    try {
      const backendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
      const response = await fetch(`${backendUrl}/api/search/budget/status`, { signal: AbortSignal.timeout(10000) });
      if (response.ok) return res.json(await response.json());
      res.status(response.status).json({ error: 'Backend error' });
    } catch {
      res.status(503).json({ error: 'Search budget service unavailable' });
    }
  });

  app.get("/api/search/budget/context", async (req, res) => {
    try {
      const backendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
      const response = await fetch(`${backendUrl}/api/search/budget/context`, { signal: AbortSignal.timeout(10000) });
      if (response.ok) return res.json(await response.json());
      res.status(response.status).json({ error: 'Backend error' });
    } catch {
      res.status(503).json({ error: 'Search budget service unavailable' });
    }
  });

  // Search budget modification endpoints require authentication
  app.post("/api/search/budget/toggle", requireAuthIfEnabled, async (req, res) => {
    try {
      const backendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
      const response = await fetch(`${backendUrl}/api/search/budget/toggle`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req.body), signal: AbortSignal.timeout(10000)
      });
      if (response.ok) return res.json(await response.json());
      res.status(response.status).json({ error: 'Backend error' });
    } catch {
      res.status(503).json({ error: 'Search budget service unavailable' });
    }
  });

  app.post("/api/search/budget/limits", requireAuthIfEnabled, async (req, res) => {
    try {
      const backendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
      const response = await fetch(`${backendUrl}/api/search/budget/limits`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req.body), signal: AbortSignal.timeout(10000)
      });
      if (response.ok) return res.json(await response.json());
      res.status(response.status).json({ error: 'Backend error' });
    } catch {
      res.status(503).json({ error: 'Search budget service unavailable' });
    }
  });

  app.post("/api/search/budget/overage", requireAuthIfEnabled, async (req, res) => {
    try {
      const backendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
      const response = await fetch(`${backendUrl}/api/search/budget/overage`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req.body), signal: AbortSignal.timeout(10000)
      });
      if (response.ok) return res.json(await response.json());
      res.status(response.status).json({ error: 'Backend error' });
    } catch {
      res.status(503).json({ error: 'Search budget service unavailable' });
    }
  });

  app.get("/api/search/budget/learning", async (req, res) => {
    try {
      const backendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
      const response = await fetch(`${backendUrl}/api/search/budget/learning`, { signal: AbortSignal.timeout(10000) });
      if (response.ok) return res.json(await response.json());
      res.status(response.status).json({ error: 'Backend error' });
    } catch {
      res.status(503).json({ error: 'Search budget service unavailable' });
    }
  });

  // Research proxy with SSE support
  app.get("/api/research/activity/stream", async (req: any, res) => {
    const backendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();
    
    try {
      const controller = new AbortController();
      req.on('close', () => controller.abort());
      
      const response = await fetch(`${backendUrl}${req.originalUrl}`, { signal: controller.signal });
      if (!response.ok || !response.body) {
        res.write(`data: ${JSON.stringify({ error: 'Backend unavailable' })}\n\n`);
        res.end();
        return;
      }
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      const pump = async () => {
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            res.write(decoder.decode(value, { stream: true }));
          }
        } catch (err: unknown) {
          if (getErrorMessage(err) !== 'The operation was aborted') {
            console.error("[API] SSE stream error:", getErrorMessage(err));
          }
        } finally {
          res.end();
        }
      };
      pump();
    } catch (error: unknown) {
      res.write(`data: ${JSON.stringify({ error: 'Stream failed' })}\n\n`);
      res.end();
    }
  });

  app.use("/api/research", async (req: any, res, next) => {
    try {
      const backendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
      const fetchOptions: RequestInit = {
        method: req.method,
        headers: { 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(60000),
      };
      if (req.method !== 'GET' && req.method !== 'HEAD') {
        fetchOptions.body = JSON.stringify(req.body);
      }
      const response = await fetch(`${backendUrl}${req.originalUrl}`, fetchOptions);
      const contentType = response.headers.get('content-type') || '';
      if (contentType.includes('application/json')) {
        res.status(response.status).json(await response.json());
      } else {
        res.status(response.status).send(await response.text());
      }
    } catch (error: unknown) {
      res.status(503).json(createTypedErrorResponse(pythonReadiness.getState()));
    }
  });

  app.use("/api/curiosity", async (req: any, res, next) => {
    try {
      const backendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
      const fetchOptions: RequestInit = {
        method: req.method,
        headers: { 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(30000),
      };
      if (req.method !== 'GET' && req.method !== 'HEAD') {
        fetchOptions.body = JSON.stringify(req.body);
      }
      const response = await fetch(`${backendUrl}${req.originalUrl}`, fetchOptions);
      const contentType = response.headers.get('content-type') || '';
      if (contentType.includes('application/json')) {
        res.status(response.status).json(await response.json());
      } else {
        res.status(response.status).send(await response.text());
      }
    } catch (error: unknown) {
      res.status(503).json(createTypedErrorResponse(pythonReadiness.getState()));
    }
  });

  app.use("/api/python", async (req: any, res, next) => {
    try {
      const backendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
      const targetPath = req.originalUrl.replace('/api/python', '/api');
      const fetchOptions: RequestInit = {
        method: req.method,
        headers: { 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(30000),
      };
      if (req.method !== 'GET' && req.method !== 'HEAD') {
        fetchOptions.body = JSON.stringify(req.body);
      }
      const response = await fetch(`${backendUrl}${targetPath}`, fetchOptions);
      const contentType = response.headers.get('content-type') || '';
      if (contentType.includes('application/json')) {
        res.status(response.status).json(await response.json());
      } else {
        res.status(response.status).send(await response.text());
      }
    } catch (error: unknown) {
      const err = error as { name?: string; message?: string };
      if (err.name === 'TimeoutError' || err.message?.includes('timeout')) {
        return res.status(504).json({ error: 'Python backend timeout' });
      }
      res.status(503).json(createTypedErrorResponse(pythonReadiness.getState()));
    }
  });

  // Server initialization
  searchCoordinator.start();
  const httpServer = createServer(app);

  const { WebSocketServer } = await import("ws");
  const wss = new WebSocketServer({ server: httpServer, path: "/ws/basin-sync" });
  const wsConnections = new Map<string, { ws: any; sessionId: string | null }>();

  oceanSessionManager.onSessionChange((oldSessionId, newSessionId) => {
    let closedCount = 0;
    for (const [peerId, conn] of wsConnections.entries()) {
      if (conn.sessionId === oldSessionId && oldSessionId !== null) {
        try {
          conn.ws.close(1000, "Session ended");
          wsConnections.delete(peerId);
          closedCount++;
        } catch (err) {}
      }
    }
  });

  wss.on("connection", (ws) => {
    const peerId = `peer-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    const currentSession = oceanSessionManager.getActiveSession();
    const sessionId = currentSession?.sessionId || null;
    wsConnections.set(peerId, { ws, sessionId });

    const activeOcean = oceanSessionManager.getActiveAgent();
    if (activeOcean) {
      const coordinator = activeOcean.getBasinSyncCoordinator();
      if (coordinator) coordinator.registerPeer(peerId, "observer", ws);
    }

    ws.on("message", async (data) => {
      try {
        if (!checkWsRateLimit(peerId)) {
          ws.send(JSON.stringify({ error: "Rate limit exceeded" }));
          return;
        }
        const rawMessage = JSON.parse(data.toString());
        const parseResult = wsMessageSchema.safeParse(rawMessage);
        if (!parseResult.success) return;

        const message = parseResult.data;
        const currentOcean = oceanSessionManager.getActiveAgent();
        if (!currentOcean) return;
        const coordinator = currentOcean.getBasinSyncCoordinator();
        if (!coordinator) return;

        if (message.type === "heartbeat") {
          coordinator.updatePeerLastSeen(peerId);
        } else if (message.type === "basin-delta" && message.data) {
          await coordinator.receiveFromPeer(peerId, message.data);
        } else if (message.type === "set-mode" && message.mode) {
          coordinator.registerPeer(peerId, message.mode, ws);
        }
      } catch (err) {}
    });

    ws.on("close", () => {
      wsConnections.delete(peerId);
      wsRateLimiter.delete(peerId);
      const currentOcean = oceanSessionManager.getActiveAgent();
      if (currentOcean) {
        const coordinator = currentOcean.getBasinSyncCoordinator();
        if (coordinator) coordinator.unregisterPeer(peerId);
      }
    });

    ws.on("error", (err) => {});
  });

  console.log("[BasinSync] WebSocket server initialized on /ws/basin-sync");

  const telemetryWss = new WebSocketServer({ server: httpServer, path: "/ws/telemetry" });
  const telemetryStreamer = new TelemetryStreamer();
  telemetryWss.on("connection", (ws) => {
    const clientId = `telemetry-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    telemetryStreamer.handleConnection(ws, clientId);
  });
  console.log("[TelemetryWS] WebSocket server initialized on /ws/telemetry");

  const kernelActivityWss = new WebSocketServer({ server: httpServer, path: "/ws/kernel-activity" });
  const kernelActivityStreamer = new KernelActivityStreamer();
  kernelActivityWss.on("connection", (ws) => {
    const clientId = `kernel-activity-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    kernelActivityStreamer.handleConnection(ws, clientId);
  });
  console.log("[KernelActivityWS] WebSocket server initialized on /ws/kernel-activity");

  initExternalWebSocket(httpServer);
  console.log("[ExternalWS] WebSocket server initialized on /ws/v1/external/stream");

  const cleanup = () => {
    telemetryStreamer.destroy();
    kernelActivityStreamer.destroy();
  };
  process.on("SIGTERM", cleanup);
  process.on("SIGINT", cleanup);

  return httpServer;
}
