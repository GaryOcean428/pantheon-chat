import type { Express } from "express";
import rateLimit from "express-rate-limit";
import multer from "multer";
import { createServer, type Server } from "http";
import { z } from "zod";
import observerRoutes from "./observer-routes";
import { scorePhraseQIG } from "./qig-universal.js";
import { storage } from "./storage";
import { telemetryRouter } from "./telemetry-api";
import { backendTelemetryRouter } from "./backend-telemetry-api";
import TelemetryStreamer from "./telemetry-websocket";
import KernelActivityStreamer from "./kernel-activity-websocket";
import MeshNetworkStreamer from "./mesh-network-websocket";
import telemetryDashboardRouter from "./routes/telemetry";

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
  autonomicAgencyRouter,
  federationRouter,
  consciousnessRouter,
  formatRouter,
  nearMissRouter,
  oceanRouter,
  olympusRouter,
  searchRouter,
  ucpRouter,
  vocabularyRouter,
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
  // Handle favicon.ico requests - redirect to favicon.png
  app.get("/favicon.ico", (req, res) => {
    res.redirect(301, "/favicon.png");
  });

  // CRITICAL: Simple liveness endpoint for Autoscale deployment health checks
  // This must respond immediately without waiting for any dependencies (database, Python, etc.)
  // Autoscale uses this to verify the server is alive and accepting HTTP requests
  app.get("/health", (req, res) => {
    res.status(200).json({
      status: "ok",
      timestamp: Date.now(),
    });
  });

  // Comprehensive health endpoint for frontend monitoring
  // Returns detailed subsystem status including database and Python backend
  app.get("/api/health", async (req, res) => {
    const startTime = Date.now();
    const subsystems: Record<string, any> = {
      database: { status: 'down' as const, message: 'Not checked' },
      pythonBackend: { status: 'down' as const, message: 'Not checked' },
      storage: { status: 'healthy' as const, message: 'In-memory storage active' },
    };

    // Check database
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

    // Check Python backend
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

    // Determine overall status
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

  // Replit Auth: Only setup auth if database connection is available
  const { db } = await import("./db");
  const authEnabled = !!db;

  if (authEnabled) {
    await setupAuth(app);
    console.log("[Auth] Replit Auth enabled");

    // Replit Auth: Auth routes - optimized with session caching
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
    console.log(
      "[Auth] Replit Auth disabled (no DATABASE_URL) - recovery tool accessible without login"
    );

    app.get("/api/auth/user", (req, res) => {
      res.status(503).json({
        message:
          "Authentication unavailable - database not provisioned. Please provision a PostgreSQL database to enable Replit Auth.",
      });
    });

    app.get("/api/login", (req, res) => {
      res.status(503).json({
        message:
          "Authentication unavailable - database not provisioned. Please provision a PostgreSQL database to enable Replit Auth.",
      });
    });

    app.get("/api/logout", (req, res) => {
      res.status(503).json({
        message: "Authentication unavailable - database not provisioned.",
      });
    });
  }

  // Emergency reset page - static HTML that clears cookies client-side
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

  // Emergency session clear (no auth required) - for auth loop recovery
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

  // Mount observer and telemetry routers
  app.use("/api/observer", observerRoutes);
  app.use("/api/telemetry", telemetryRouter);
  app.use("/api/backend-telemetry", backendTelemetryRouter);
  
  // Mount versioned telemetry dashboard API (unified metrics)
  app.use("/api/v1/telemetry", telemetryDashboardRouter);
  
  // Mount external API router (for federated instances, headless clients, integrations)
  app.use("/api/v1/external", externalApiRouter);

  // ============================================================
  // COORDIZER STATS PROXY (Routes to Python Backend)
  // ============================================================
  app.get("/api/coordize/stats", async (req, res) => {
    try {
      const pythonUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
      const response = await fetch(`${pythonUrl}/api/coordize/stats`);
      
      if (!response.ok) {
        throw new Error(`Python backend returned ${response.status}`);
      }
      
      const stats = await response.json();
      res.json(stats);
    } catch (error: unknown) {
      console.error("[Coordizer] Stats proxy error:", getErrorMessage(error));
      res.json({
        vocab_size: 0,
        coordinate_dim: 64,
        geometric_purity: true,
        special_tokens: ['[PAD]', '[UNK]', '[BOS]', '[EOS]'],
        status: 'backend_unavailable',
        error: getErrorMessage(error)
      });
    }
  });

  // Investigation status endpoint - used by investigation page
  app.get("/api/investigation/status", (req, res) => {
    try {
      const status = oceanSessionManager.getInvestigationStatus();
      res.json(status);
    } catch (error: unknown) {
      console.error("[API] Investigation status error:", getErrorMessage(error));
      res
        .status(500)
        .json({ error: getErrorMessage(error) || "Failed to get investigation status" });
    }
  });

  // ============================================================
  // AUTO-CYCLE MANAGEMENT ENDPOINTS
  // ============================================================

  app.get("/api/auto-cycle/status", (req, res) => {
    try {
      const status = autoCycleManager.getStatus();
      const position = autoCycleManager.getPositionString();
      res.json({
        success: true,
        ...status,
        positionString: position,
      });
    } catch (error: unknown) {
      console.error("[API] Auto-cycle status error:", getErrorMessage(error));
      res.status(500).json({ error: getErrorMessage(error) });
    }
  });

  app.post("/api/auto-cycle/enable", async (req, res) => {
    try {
      console.log("[API] Auto-cycle enable request received");
      const result = await autoCycleManager.enable();
      res.json({
        success: result.success,
        message: result.message,
        status: autoCycleManager.getStatus(),
      });
    } catch (error: unknown) {
      console.error("[API] Auto-cycle enable error:", getErrorMessage(error));
      res.status(500).json({ error: getErrorMessage(error) });
    }
  });

  app.post("/api/auto-cycle/disable", (req, res) => {
    try {
      console.log("[API] Auto-cycle disable request received");
      const result = autoCycleManager.disable();
      res.json({
        success: result.success,
        message: result.message,
        status: autoCycleManager.getStatus(),
      });
    } catch (error: unknown) {
      console.error("[API] Auto-cycle disable error:", getErrorMessage(error));
      res.status(500).json({ error: getErrorMessage(error) });
    }
  });

  app.post("/api/auto-cycle/force-resume", async (req, res) => {
    try {
      console.log("[API] Auto-cycle force-resume request received");
      const result = await autoCycleManager.forceResume();
      res.json({
        success: result.success,
        message: result.message,
        status: autoCycleManager.getStatus(),
      });
    } catch (error: unknown) {
      console.error("[API] Auto-cycle force-resume error:", getErrorMessage(error));
      res.status(500).json({ error: getErrorMessage(error) });
    }
  });

  console.log("[Routes] All sub-routers mounted");

  // ============================================================
  // LEARNING UPLOAD ENDPOINT - Proxy to Python backend
  // ============================================================
  const uploadMiddleware = multer({ 
    storage: multer.memoryStorage(),
    limits: { fileSize: 5 * 1024 * 1024 },
    fileFilter: (_req, file, cb) => {
      if (file.originalname.toLowerCase().endsWith('.md')) {
        cb(null, true);
      } else {
        cb(new Error('Only .md (markdown) files are accepted'));
      }
    }
  });
  
  app.post("/api/learning/upload", uploadMiddleware.single('file'), async (req: any, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'No file provided. Use "file" field in multipart/form-data' });
      }

      // Parse markdown content directly in Node.js (fallback mode)
      const content = req.file.buffer.toString('utf-8');
      
      // Remove code blocks, inline code, URLs, and HTML
      let text = content
        .replace(/```[\s\S]*?```/g, ' ')
        .replace(/`[^`]+`/g, ' ')
        .replace(/https?:\/\/\S+/g, ' ')
        .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
        .replace(/<[^>]+>/g, ' ');
      
      // Extract words (letters only, 3+ chars)
      const wordPattern = /[a-zA-Z][a-zA-Z'-]*[a-zA-Z]|[a-zA-Z]{3,}/g;
      const rawWords = text.toLowerCase().match(wordPattern) || [];
      
      // Filter to valid words (simple heuristic - exclude common noise)
      const stopWords = new Set(['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'will', 'with', 'this', 'that', 'from', 'they', 'were', 'said', 'each', 'which', 'their', 'would', 'there', 'could', 'other', 'into', 'more', 'some', 'than', 'them', 'these', 'then', 'its', 'also', 'just', 'only', 'come', 'made', 'may', 'now', 'way', 'many', 'like', 'use', 'such', 'when', 'what', 'how', 'who', 'did', 'get', 'very', 'being', 'about']);
      
      const validWords = rawWords.filter((word: string) => 
        word.length >= 3 && 
        !stopWords.has(word) &&
        !/^\d+$/.test(word)
      );
      
      if (validWords.length === 0) {
        return res.json({
          success: true,
          filename: req.file.originalname,
          words_processed: 0,
          words_learned: 0,
          message: 'No valid vocabulary words found in the markdown file'
        });
      }
      
      // Count word frequencies
      const wordCounts: Record<string, number> = {};
      for (const word of validWords) {
        wordCounts[word] = (wordCounts[word] || 0) + 1;
      }
      
      // Try to forward to Python backend for proper learning
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
          const contentType = response.headers.get('content-type') || '';
          if (contentType.includes('application/json')) {
            const data = await response.json();
            return res.json(data);
          }
        }
      } catch {
        // Python backend unavailable - continue with Node.js fallback
        console.log("[API] Learning upload - Python backend unavailable, using Node.js fallback");
      }
      
      // Return parsed results (Node.js fallback mode)
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
      console.error("[API] Learning upload error:", getErrorMessage(error));
      res.status(500).json({ error: getErrorMessage(error) || 'Failed to process markdown upload' });
    }
  });


  // ============================================================
  // PYTHON PROXY - Generic proxy for Python backend APIs
  // ============================================================
  app.use("/api/python", async (req: any, res, next) => {
    try {
      const backendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
      const targetPath = req.originalUrl.replace('/api/python', '/api');
      const targetUrl = `${backendUrl}${targetPath}`;
      
      const fetchOptions: RequestInit = {
        method: req.method,
        headers: {
          'Content-Type': 'application/json',
        },
        signal: AbortSignal.timeout(30000),
      };
      
      if (req.method !== 'GET' && req.method !== 'HEAD') {
        fetchOptions.body = JSON.stringify(req.body);
      }
      
      const response = await fetch(targetUrl, fetchOptions);
      const contentType = response.headers.get('content-type') || '';
      
      if (contentType.includes('application/json')) {
        const data = await response.json();
        res.status(response.status).json(data);
      } else {
        const text = await response.text();
        res.status(response.status).send(text);
      }
    } catch (error: unknown) {
      console.error("[API] Python proxy error:", getErrorMessage(error));
      const err = error as { name?: string; code?: string; message?: string };
      if (err.name === 'TimeoutError' || err.message?.includes('timeout')) {
        return res.status(504).json({ error: 'Python backend timeout' });
      }
      if (err.code === 'ECONNREFUSED' || err.message?.includes('fetch failed')) {
        return res.status(503).json({ error: 'Python backend unavailable' });
      }
      res.status(500).json({ error: getErrorMessage(error) || 'Failed to proxy request' });
    }
  });

  // ============================================================
  // SERVER INITIALIZATION
  // ============================================================

  // Start the background search coordinator
  searchCoordinator.start();

  const httpServer = createServer(app);

  // Set up WebSocket server for real-time basin sync
  const { WebSocketServer } = await import("ws");
  const wss = new WebSocketServer({
    server: httpServer,
    path: "/ws/basin-sync",
  });

  // Track WebSocket connections for cleanup on session change
  const wsConnections = new Map<
    string,
    { ws: any; sessionId: string | null }
  >();

  // Register session change handler to clean up old WebSocket connections
  oceanSessionManager.onSessionChange((oldSessionId, newSessionId) => {
    console.log(
      `[BasinSync WS] Session changed: ${oldSessionId} → ${newSessionId}`
    );

    // Close all connections associated with the old session
    let closedCount = 0;
    for (const [peerId, conn] of wsConnections.entries()) {
      if (conn.sessionId === oldSessionId && oldSessionId !== null) {
        try {
          conn.ws.close(1000, "Session ended");
          wsConnections.delete(peerId);
          closedCount++;
        } catch (err) {
          console.error(
            `[BasinSync WS] Error closing connection ${peerId}:`,
            err
          );
        }
      }
    }
    if (closedCount > 0) {
      console.log(
        `[BasinSync WS] Cleaned up ${closedCount} connections from old session`
      );
    }
  });

  wss.on("connection", (ws) => {
    const peerId = `peer-${Date.now()}-${Math.random()
      .toString(36)
      .slice(2, 6)}`;
    const currentSession = oceanSessionManager.getActiveSession();
    const sessionId = currentSession?.sessionId || null;

    // Track connection with its session ID
    wsConnections.set(peerId, { ws, sessionId });

    console.log(
      `[BasinSync WS] New connection: ${peerId} (session: ${sessionId})`
    );
    console.log(`[BasinSync WS] Total connections: ${wsConnections.size}`);

    const activeOcean = oceanSessionManager.getActiveAgent();
    if (activeOcean) {
      const coordinator = activeOcean.getBasinSyncCoordinator();
      if (coordinator) {
        coordinator.registerPeer(peerId, "observer", ws);
      }
    }

    ws.on("message", async (data) => {
      try {
        // Rate limiting check (Issue 13/14 fix)
        if (!checkWsRateLimit(peerId)) {
          console.warn(`[BasinSync WS] Rate limit exceeded for ${peerId}`);
          ws.send(JSON.stringify({ error: "Rate limit exceeded" }));
          return;
        }

        // Parse and validate message with Zod schema
        const rawMessage = JSON.parse(data.toString());
        const parseResult = wsMessageSchema.safeParse(rawMessage);

        if (!parseResult.success) {
          console.warn(
            `[BasinSync WS] Invalid message from ${peerId}:`,
            parseResult.error.message
          );
          return;
        }

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
      } catch (err) {
        console.error("[BasinSync WS] Message parse error:", err);
      }
    });

    ws.on("close", () => {
      console.log(`[BasinSync WS] Connection closed: ${peerId}`);

      // Remove from connection tracking and rate limiter
      wsConnections.delete(peerId);
      wsRateLimiter.delete(peerId);
      console.log(
        `[BasinSync WS] Remaining connections: ${wsConnections.size}`
      );

      const currentOcean = oceanSessionManager.getActiveAgent();
      if (currentOcean) {
        const coordinator = currentOcean.getBasinSyncCoordinator();
        if (coordinator) {
          coordinator.unregisterPeer(peerId);
        }
      }
    });

    ws.on("error", (err) => {
      console.error(`[BasinSync WS] Error for ${peerId}:`, err);
    });
  });

  console.log("[BasinSync] WebSocket server initialized on /ws/basin-sync");

  // Set up WebSocket server for real-time telemetry streaming
  const telemetryWss = new WebSocketServer({
    server: httpServer,
    path: "/ws/telemetry",
  });

  const telemetryStreamer = new TelemetryStreamer();

  telemetryWss.on("connection", (ws) => {
    const clientId = `telemetry-${Date.now()}-${Math.random()
      .toString(36)
      .slice(2, 6)}`;
    telemetryStreamer.handleConnection(ws, clientId);
  });

  console.log("[TelemetryWS] WebSocket server initialized on /ws/telemetry");

  // Set up WebSocket server for kernel activity streaming
  const kernelActivityWss = new WebSocketServer({
    server: httpServer,
    path: "/ws/kernel-activity",
  });

  const kernelActivityStreamer = new KernelActivityStreamer();

  kernelActivityWss.on("connection", (ws) => {
    const clientId = `kernel-activity-${Date.now()}-${Math.random()
      .toString(36)
      .slice(2, 6)}`;
    kernelActivityStreamer.handleConnection(ws, clientId);
  });

  console.log("[KernelActivityWS] WebSocket server initialized on /ws/kernel-activity");

  // Set up WebSocket server for external API streaming
  initExternalWebSocket(httpServer);
  console.log("[ExternalWS] WebSocket server initialized on /ws/v1/external/stream");

  // Cleanup on server shutdown
  const cleanup = () => {
    telemetryStreamer.destroy();
    kernelActivityStreamer.destroy();
  };
  process.on("SIGTERM", cleanup);
  process.on("SIGINT", cleanup);

  return httpServer;
}
