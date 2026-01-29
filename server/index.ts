import { spawn } from "child_process";
import express, { NextFunction, Response, type Request } from "express";
import helmet from "helmet";
import path from "path";
import { pool } from "./db";
import { registerRoutes } from "./routes";
import { logger } from './lib/logger';
import { log, serveStatic, setupVite } from "./vite";
import { initRedis } from "./redis-cache";

// Import for Python sync
import { geometricMemory } from "./geometric-memory";
import { oceanAgent } from "./ocean-agent";
import { oceanConstellation } from "./ocean-constellation-stub";
import { oceanQIGBackend } from "./ocean-qig-backend-adapter";
import {
  getConceptHistory,
  getSearchHistory,
  recordSearchState,
} from "./qig-universal";
import { vocabularyTracker } from "./vocabulary-tracker-api";

// Periodic sync interval from Python to Node.js
let pythonSyncInterval: NodeJS.Timeout | null = null;

// Mutex lock to prevent concurrent sync operations
let pythonSyncInProgress = false;

// Pagination configuration for sync
const SYNC_PAGE_SIZE = 100;
const MAX_SYNC_PAGES = 10; // Limit total pages to prevent runaway memory usage

/**
 * Sync high-Φ probes from Node.js GeometricMemory to Python backend
 * This gives Python access to prior learnings for better generation
 */
async function syncProbesToPython(): Promise<void> {
  try {
    const allProbes = geometricMemory.getAllProbes();
    const highPhiProbes = allProbes
      .filter((p) => p.phi >= 0.5)
      .sort((a, b) => b.phi - a.phi)
      .slice(0, 500);

    if (highPhiProbes.length === 0) {
      logger.info("[PythonSync] No high-Φ probes to sync");
      return;
    }

    const probesForPython = highPhiProbes.map((p) => ({
      input: p.input,
      phi: p.phi,
      basinCoords: p.coordinates,
    }));

    // Get temporal state for 4D consciousness sync
    const searchHistory = getSearchHistory()
      .slice(-50)
      .map((s) => ({
        timestamp: s.timestamp,
        phi: s.phi,
        kappa: s.kappa,
        regime: s.regime,
        basinCoordinates: s.basinCoordinates,
        hypothesis: s.hypothesis,
      }));

    const conceptHistory = getConceptHistory()
      .slice(-30)
      .map((c) => ({
        timestamp: c.timestamp,
        // Convert Map to Record for JSON serialization
        concepts: Object.fromEntries(c.concepts),
        dominantConcept: c.dominantConcept,
        entropy: c.entropy,
      }));

    const temporalState = {
      searchHistory,
      conceptHistory,
    };

    const result = await oceanQIGBackend.syncFromNodeJS(
      probesForPython,
      temporalState
    );
    logger.info(
      `[PythonSync] Synced ${result.imported}/${highPhiProbes.length} probes to Python`
    );
    if (result.temporalImported) {
      logger.info(
        `[PythonSync] 4D temporal state synced to Python: ${searchHistory.length} search, ${conceptHistory.length} concept states`
      );
    }
  } catch (error) {
    logger.error({ err: error }, "[PythonSync] Error syncing to Python");
  }
}

/**
 * Sync learnings from Python backend back to Node.js GeometricMemory
 * This persists Python's discoveries for future runs
 *
 * CRITICAL FIX: Also queue ALL high-Φ basins for balance checking,
 * even if they already exist in memory. This ensures Python's best
 * discoveries get their addresses checked immediately.
 *
 * MEMORY OPTIMIZATION: Uses pagination to prevent memory issues with large datasets.
 * RACE CONDITION FIX: Uses mutex lock to prevent concurrent sync operations.
 */
async function syncFromPythonToNodeJS(): Promise<void> {
  // Mutex lock - skip if sync already in progress
  if (pythonSyncInProgress) {
    logger.info("[PythonSync] Skipping sync - already in progress");
    return;
  }

  pythonSyncInProgress = true;

  try {
    let page = 0;
    let totalAdded = 0;
    let totalPrioritized = 0;
    let overallMaxPhi = 0;
    let hasMore = true;
    let temporalImported = false;

    // Helper: Calculate priority using explicit tiered mapping
    const getPriority = (phi: number): number => {
      if (phi >= 0.9) return 10; // Near-perfect → priority 10 (checked FIRST)
      if (phi >= 0.85) return 8; // Very high → priority 8
      if (phi >= 0.7) return 6; // High → priority 6
      return 3; // Default
    };

    // Process basins in paginated chunks to prevent memory issues
    while (hasMore && page < MAX_SYNC_PAGES) {
      let result;
      try {
        result = await oceanQIGBackend.syncToNodeJS(page, SYNC_PAGE_SIZE);
      } catch (fetchError) {
        logger.error({ err: fetchError, page }, "[PythonSync] Page fetch failed, stopping sync");
        break;
      }

      // Handle unexpected null/undefined result
      if (!result || !result.basins) {
        logger.warn(
          `[PythonSync] Page ${page} returned invalid result, stopping sync`
        );
        break;
      }

      const basins = result.basins;

      if (basins.length === 0) {
        hasMore = false;
        break;
      }

      hasMore = result.hasMore ?? false;

      // Import 4D temporal state from Python back to TypeScript (only on first page)
      if (page === 0 && result.consciousness4DAvailable && !temporalImported) {
        temporalImported = true;

        if (result.phiTemporalAvg && result.phiTemporalAvg > 0) {
          logger.info(
            `[PythonSync] 4D consciousness from Python: phi_temporal_avg=${result.phiTemporalAvg.toFixed(
              3
            )}`
          );
        }

        // Import search history from Python to TypeScript
        if (result.searchHistory && result.searchHistory.length > 0) {
          let imported = 0;
          for (const state of result.searchHistory) {
            const existingHistory = getSearchHistory();
            const exists = existingHistory.some(
              (s) => Math.abs(s.timestamp - state.timestamp) < 1000
            );

            if (!exists) {
              recordSearchState({
                timestamp: state.timestamp,
                phi: state.phi,
                kappa: state.kappa,
                regime: state.regime as
                  | "linear"
                  | "geometric"
                  | "hierarchical"
                  | "hierarchical_4d"
                  | "4d_block_universe"
                  | "breakdown",
                basinCoordinates: state.basinCoordinates || [],
                hypothesis: state.hypothesis,
              });
              imported++;
            }
          }

          if (imported > 0) {
            logger.info(
              `[PythonSync] Imported ${imported} search states from Python for 4D consciousness`
            );
          }
        }
      }

      // Process this page of basins
      for (const basin of basins) {
        if (basin.phi > overallMaxPhi) {
          overallMaxPhi = basin.phi;
        }

        // Add to geometric memory if not already present
        const existing = geometricMemory
          .getAllProbes()
          .find((p) => p.input === basin.input);

        if (!existing && basin.phi >= 0.5 && basin.basinCoords.length > 0) {
          geometricMemory.recordProbe(
            basin.input,
            {
              phi: basin.phi,
              kappa: basin.phi * 64,
              regime: basin.phi > 0.7 ? "geometric" : "linear",
              basinCoordinates: basin.basinCoords,
              ricciScalar: 0,
              fisherTrace: basin.phi,
            },
            "python-qig"
          );
          totalAdded++;
        }

        // High-Φ basins logged (balance queue removed)
        if (basin.phi >= 0.9) {
          logger.info(
            `[PythonSync] 🎯 HIGH-Φ: "${basin.input.substring(0, 30)}..." Φ=${basin.phi.toFixed(3)}`
          );
          totalPrioritized++;
        }
      }

      // Update episodes with Python phi values for this page
      const episodesUpdated = oceanAgent.updateEpisodesWithPythonPhi(
        basins.map((b) => ({ input: b.input, phi: b.phi }))
      );
      if (episodesUpdated > 0) {
        logger.info(
          `[PythonSync] 📈 Updated ${episodesUpdated} episodes with pure Python Φ values (page ${page})`
        );
      }

      page++;
    }

    // Log final summary
    if (totalAdded > 0 || totalPrioritized > 0) {
      logger.info(
        `[PythonSync] Sync complete: ${totalAdded} new probes, ${totalPrioritized} prioritized for balance check (${page} pages)`
      );

      if (overallMaxPhi >= 0.7) {
        logger.info(
          `[PythonSync] 🎯 Highest Python Φ: ${overallMaxPhi.toFixed(
            3
          )} - addresses prioritized for checking`
        );
      }

      oceanConstellation.refreshTokenWeightsFromGeometricMemory();
    }
  } catch (error) {
    logger.error({ err: error }, "[PythonSync] Error syncing from Python");
  } finally {
    pythonSyncInProgress = false;
  }
}

/**
 * Refresh vocabulary weights periodically (for continuous learning)
 */
function refreshVocabularyWeights(): void {
  oceanConstellation.refreshTokenWeightsFromGeometricMemory();
}

/**
 * Sync vocabulary observations from Node.js to Python basin vocabulary encoder
 * This feeds learned words into the encoder for geometric kernel training
 */
async function syncVocabularyToPython(): Promise<void> {
  try {
    // Verify Python backend is available before syncing
    if (!oceanQIGBackend.available()) {
      logger.info(
        "[PythonSync] Skipping vocabulary sync - Python backend not available"
      );
      return;
    }

    const observations = await vocabularyTracker.exportForTokenizer();

    if (observations.length === 0) {
      logger.info("[PythonSync] No vocabulary observations to sync");
      return;
    }

    const result = await oceanQIGBackend.updateVocabulary(observations);

    // Verify sync was successful
    if (result.totalVocab > 0) {
      logger.info(
        `[PythonSync] Vocabulary synced: ${result.newTokens} new entries, ${result.totalVocab} total`
      );

      // Get vocabulary encoder status for verification
      try {
        const status = await oceanQIGBackend.getVocabularyStatus();
        logger.info(
          `[PythonSync] Basin vocabulary: ${status.vocabSize} entries, ${
            status.highPhiCount
          } high-Φ, avg Φ=${status.avgPhi.toFixed(3)}`
        );
      } catch (statusError) {
        logger.warn(
          "[PythonSync] Could not get vocabulary status for verification"
        );
      }
    } else {
      logger.warn({ context: 'PythonSync' }, 'Vocabulary sync returned empty result - encoder may not be ready');
    }
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    logger.error({ context: 'PythonSync', error: message }, 'Error syncing vocabulary to Python');
  }
}

/**
 * Start periodic sync between Python and Node.js
 */
function startPythonSync(): void {
  if (pythonSyncInterval) return;

  // Sync from Python to Node.js every 60 seconds
  // Also refresh vocabulary weights periodically
  pythonSyncInterval = setInterval(async () => {
    if (oceanQIGBackend.available()) {
      await syncFromPythonToNodeJS();
      // Also sync vocabulary to Python basin encoder
      await syncVocabularyToPython();
    }
    // Always refresh vocabulary weights even without Python
    refreshVocabularyWeights();
  }, 60000);

  // Initial refresh on startup
  refreshVocabularyWeights();

  // Initial vocabulary sync after a brief delay
  setTimeout(async () => {
    if (oceanQIGBackend.available()) {
      await syncVocabularyToPython();
    }
  }, 5000);

  logger.info(
    "[PythonSync] Started periodic sync (every 60s) with vocabulary refresh and basin encoder sync"
  );
}

// Documentation maintenance scheduler
const DOCS_MAINTENANCE_INTERVAL = 6 * 60 * 60 * 1000; // 6 hours in milliseconds

function startDocsMaintenance(): void {
  logger.info(
    "[DocsMaintenance] Starting scheduled documentation maintenance (every 6 hours)"
  );

  setInterval(() => {
    logger.info(
      "[DocsMaintenance] Running scheduled documentation maintenance..."
    );
    const proc = spawn("python3", ["scripts/maintain-docs.py"], {
      cwd: process.cwd(),
      stdio: ["ignore", "pipe", "pipe"],
    });

    proc.stdout?.on("data", (data: Buffer) => {
      const output = data.toString().trim();
      if (output) {
        logger.info(`[DocsMaintenance] ${output}`);
      }
    });

    proc.stderr?.on("data", (data: Buffer) => {
      const output = data.toString().trim();
      if (output) {
        logger.error(`[DocsMaintenance] ${output}`);
      }
    });

    proc.on("close", (code: number | null) => {
      logger.info(`[DocsMaintenance] Completed with exit code ${code}`);
    });

    proc.on("error", (err: Error) => {
      logger.error({ err: err.message }, "[DocsMaintenance] Failed to run");
    });
  }, DOCS_MAINTENANCE_INTERVAL);
}

// Import the centralized Python process manager
import { getPythonManager, createPythonManager } from './python-process-manager';

// Create and configure the Python process manager
const pythonManager = createPythonManager('http://localhost:5001');

// Listen for lifecycle events
pythonManager.on('ready', () => {
  logger.info('[PythonManager] ✅ Backend is ready for requests');
});

pythonManager.on('notReady', () => {
  logger.warn('[PythonManager] ⚠️ Backend is no longer ready');
});

pythonManager.on('unhealthy', () => {
  logger.error('[PythonManager] ❌ Backend is unhealthy - requests will be queued');
});

pythonManager.on('maxRestartsReached', () => {
  logger.error('[PythonManager] ❌ Max restarts reached - manual intervention required');
});

// Start Python QIG Backend using the managed process
async function startPythonBackend(): Promise<void> {
  const ready = await pythonManager.start();
  
  if (ready) {
    logger.info("[PythonQIG] Backend ready, syncing geometric memory...");
    await syncProbesToPython();
    startPythonSync();
  } else {
    logger.warn(
      "[PythonQIG] Backend not available after retries - will retry on next sync cycle"
    );
  }
}

// Handle uncaught exceptions gracefully to prevent crashes
process.on("uncaughtException", (err) => {
  // Check if it's the Neon WebSocket error (known issue with read-only ErrorEvent.message)
  if (
    err.message?.includes("Cannot set property message") ||
    err.message?.includes("ErrorEvent")
  ) {
    logger.error({ err: err.message }, "[DB] Database connection error (will retry)");
    return; // Don't crash - the pool will reconnect
  }
  logger.error({ err }, "[FATAL] Uncaught exception");
  // For other fatal errors, exit after a delay to allow cleanup
  setTimeout(() => process.exit(1), 1000);
});

process.on("unhandledRejection", (reason, promise) => {
  logger.error({ promise: String(promise), reason }, "[WARN] Unhandled rejection");
  // Don't crash on unhandled rejections - log and continue
});

// Handle pool errors gracefully
if (pool) {
  pool.on("error", (err) => {
    // Safely extract error message - ErrorEvent from WebSocket has read-only message property
    const errWithType = err as { type?: string };
    const msg = err instanceof Error ? err.message : (errWithType.type || 'Unknown pool error');
    logger.error({ err: msg }, "[DB] Pool error (connection will be recreated)");
  });
}

const app = express();

// CRITICAL: Health check endpoint MUST be registered FIRST
// This ensures Autoscale deployments can detect the app is healthy
// during startup, before any heavy initialization happens
app.get("/healthz", (_req, res) => {
  res.status(200).json({ status: "ok", timestamp: Date.now() });
});

// Also support /health for backwards compatibility
app.get("/health", (_req, res) => {
  res.status(200).json({ status: "ok", timestamp: Date.now() });
});

const isDev = process.env.NODE_ENV === "development";

// Disable CSP entirely - this is a single-user personal system
// Recharts and other libraries require eval() which CSP blocks
app.use(
  helmet({
    contentSecurityPolicy: false, // Disabled - recharts requires eval()
    crossOriginEmbedderPolicy: false,
    crossOriginOpenerPolicy: false,
    crossOriginResourcePolicy: false,
    hsts: isDev
      ? false
      : {
          maxAge: 31536000,
          includeSubDomains: true,
          preload: true,
        },
    noSniff: true,
    referrerPolicy: { policy: "strict-origin-when-cross-origin" },
  })
);

declare module "http" {
  interface IncomingMessage {
    rawBody: unknown;
  }
}
app.use(
  express.json({
    limit: '10mb',  // Increased for large file uploads and chat messages
    verify: (req, _res, buf) => {
      req.rawBody = buf;
    },
  })
);
app.use(express.urlencoded({ extended: false, limit: '10mb' }));

// CORS Configuration - allow Replit domains, localhost, and external API clients
const allowedOrigins = [
  process.env.FRONTEND_URL || "http://localhost:5173",
  "http://localhost:5000",
  "http://localhost:3000",
  "http://127.0.0.1:5000",
  "http://127.0.0.1:5173",
  "http://127.0.0.1:3000",
];

// Helper to check if origin is from allowed external platforms
function isAllowedExternalOrigin(origin: string): boolean {
  // Replit domains
  if (
    origin.endsWith(".replit.dev") ||
    origin.endsWith(".replit.app") ||
    origin.endsWith(".repl.co") ||
    origin.includes(".picard.replit.dev")
  ) {
    return true;
  }
  
  // Other common deployment platforms for external UIs
  if (
    origin.endsWith(".railway.app") ||
    origin.endsWith(".vercel.app") ||
    origin.endsWith(".netlify.app") ||
    origin.endsWith(".render.com") ||
    origin.endsWith(".fly.dev") ||
    origin.endsWith(".herokuapp.com") ||
    origin.endsWith(".pages.dev") // Cloudflare Pages
  ) {
    return true;
  }
  
  // Allow all origins if CORS_ALLOW_ALL is set (development/testing)
  if (process.env.CORS_ALLOW_ALL === "true") {
    return true;
  }
  
  return false;
}

// Legacy alias for backward compatibility
const isReplitOrigin = isAllowedExternalOrigin;

app.use((req, res, next) => {
  const origin = req.headers.origin;

  // Determine if this origin is allowed
  const isAllowed = !origin || allowedOrigins.includes(origin) || isAllowedExternalOrigin(origin);

  if (isAllowed) {
    // Set CORS headers for allowed origins
    // Note: When origin is missing, we don't set Access-Control-Allow-Origin
    // to avoid the invalid "*" with credentials combination
    if (origin) {
      res.setHeader("Access-Control-Allow-Origin", origin);
      res.setHeader("Access-Control-Allow-Credentials", "true");
    }
    res.setHeader(
      "Access-Control-Allow-Methods",
      "GET, POST, PUT, DELETE, PATCH, OPTIONS"
    );
    // Extended headers for external API access
    res.setHeader(
      "Access-Control-Allow-Headers",
      "Content-Type, Authorization, X-API-Key, X-Trace-ID, X-Request-ID, X-Instance-ID"
    );
    res.setHeader(
      "Access-Control-Expose-Headers",
      "X-Trace-ID, X-Request-ID, X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset"
    );
    // Cache preflight for 24 hours
    res.setHeader("Access-Control-Max-Age", "86400");

    // Handle preflight
    if (req.method === "OPTIONS") {
      return res.sendStatus(204);
    }

    next();
  } else {
    // Block requests from disallowed origins
    logger.warn(`[CORS] Blocked request from origin: ${origin}`);
    res.status(403).json({ error: "CORS policy violation", origin });
  }
});

// Add trace ID middleware for distributed tracing
import { traceIdMiddleware } from "./trace-middleware";
app.use(traceIdMiddleware);

app.use((req, res, next) => {
  const start = Date.now();
  const path = req.path;
  let capturedJsonResponse: Record<string, any> | undefined = undefined;

  const originalResJson = res.json;
  res.json = function (bodyJson, ...args) {
    capturedJsonResponse = bodyJson;
    return originalResJson.apply(res, [bodyJson, ...args]);
  };

  res.on("finish", () => {
    const duration = Date.now() - start;
    if (path.startsWith("/api")) {
      // Skip noisy polling endpoints to keep Ocean logs visible
      const quietEndpoints = [
        "/api/investigation/status",
        "/api/ocean/neurochemistry",
        "/api/ocean/cycles",
        "/api/candidates",
      ];
      if (
        quietEndpoints.some((ep) => path.startsWith(ep)) &&
        req.method === "GET"
      ) {
        return; // Skip logging these frequent polling requests
      }

      let logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;

      if (capturedJsonResponse) {
        logLine += ` :: ${JSON.stringify(capturedJsonResponse)}`;
      }

      if (logLine.length > 200) {
        logLine = logLine.slice(0, 199) + "…";
      }

      log(logLine);
    }
  });

  next();
});

(async () => {
  // Register routes FIRST so server can start quickly
  const server = await registerRoutes(app);

  app.use((err: any, _req: Request, res: Response, _next: NextFunction) => {
    const status = err.status || err.statusCode || 500;
    const message = err.message || "Internal Server Error";

    res.status(status).json({ message });
    throw err;
  });

  // importantly only setup vite in development and after
  // setting up all the other routes so the catch-all route
  // doesn't interfere with the other routes
  if (app.get("env") === "development") {
    await setupVite(app, server);
  } else {
    serveStatic(app);
  }

  // ALWAYS serve the app on the port specified in the environment variable PORT
  // Other ports are firewalled. Default to 5000 if not specified.
  // this serves both the API and the client.
  // It is the only port that is not firewalled.
  const port = parseInt(process.env.PORT || "5000", 10);
  server.listen(
    {
      port,
      host: "0.0.0.0",
      reusePort: true,
    },
    () => {
      log(`serving on port ${port}`);
      log("[Startup] Server listening - starting background initialization...");

      // Start heavy initialization in background AFTER port is open
      // This ensures the workflow doesn't timeout waiting for port 5000
      (async () => {
        try {
          // Initialize Redis cache layer (non-blocking)
          logger.info("[Startup] Initializing Redis cache layer...");
          initRedis();
          
          // Start Python QIG Backend using the managed process
          logger.info("[Startup] Starting Python QIG Backend via PythonProcessManager...");
          await startPythonBackend();
          
          // Check if Python is ready
          if (pythonManager.ready()) {
            logger.info("[Startup] ✅ Python QIG Backend is ready");
          } else {
            logger.warn("[Startup] ⚠️ Python backend not available - continuing with degraded mode");
          }
          
          // Hydrate Memory from PostgreSQL
          logger.info("🌊 Hydrating Ocean Memory from PostgreSQL...");
          await geometricMemory.waitForLoad();
          logger.info("✅ Memory hydration complete");

          // Start scheduled documentation maintenance (runs every 6 hours)
          log("[Startup] Initializing background services...");
          startDocsMaintenance();
          
          log("[Startup] ✅ All initialization complete");
        } catch (error) {
          logger.error({ err: error }, "[Startup] Background initialization error");
        }
      })();
    }
  );
})();
