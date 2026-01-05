import { Router, type Request, type Response } from "express";
import { logger } from '../lib/logger';
import { getErrorMessage, handleRouteError } from '../lib/error-utils';
import { randomUUID } from "crypto";
import { generousLimiter } from "../rate-limiters";
import { storage } from "../storage";
import { storageFacade } from "../persistence";
import { KNOWN_12_WORD_PHRASES } from "../known-phrases";
import { generateRandomBIP39Phrase, getBIP39Wordlist } from "../bip39-words";
import { searchCoordinator } from "../search-coordinator";
import { activityLogStore } from "../activity-log-store";
import { oceanSessionManager } from "../ocean-session-manager";
import { runMemoryFragmentSearch, type MemoryFragment } from "../memory-fragment-search";
import { 
  addAddressRequestSchema, 
  generateRandomPhrasesRequestSchema, 
  createSearchJobRequestSchema,
  type Candidate,
  type TargetAddress,
  type SearchJob 
} from "@shared/schema";
import { googleWebSearchAdapter } from "../geometric-discovery/google-web-search-adapter";
import { scoreUniversalQIGAsync } from "../qig-universal";

// Search provider state (in-memory, persists until restart)
// Exported so other modules can check provider state
// Auto-enable premium providers when API keys are available
export const searchProviderState = {
  google_free: { enabled: true },
  tavily: { enabled: !!process.env.TAVILY_API_KEY },
  perplexity: { enabled: !!process.env.PERPLEXITY_API_KEY },
  duckduckgo: { enabled: true, torEnabled: true },
};

// Log auto-enable status at startup
if (searchProviderState.tavily.enabled) {
  console.log('[SearchProviders] tavily auto-enabled (API key detected)');
}
if (searchProviderState.perplexity.enabled) {
  console.log('[SearchProviders] perplexity auto-enabled (API key detected)');
}

export function isProviderEnabled(provider: 'google_free' | 'tavily' | 'perplexity' | 'duckduckgo'): boolean {
  return searchProviderState[provider].enabled;
}

export const searchRouter = Router();

searchRouter.get("/web", generousLimiter, async (req: Request, res: Response) => {
  try {
    const query = req.query.q as string;
    const limit = Math.min(parseInt(req.query.limit as string) || 5, 10);
    
    if (!query || query.trim().length === 0) {
      return res.status(400).json({ error: 'Query parameter "q" is required' });
    }
    
    // Check if Google Free Search is enabled
    if (!searchProviderState.google_free.enabled) {
      return res.json({
        query,
        results: [],
        count: 0,
        status: 'disabled',
        message: 'Google Free Search is disabled. Enable it in Sources page.',
        source: 'google-web-search',
        timestamp: new Date().toISOString(),
      });
    }
    
    console.log(`[WebSearch API] Query: "${query}" (limit: ${limit})`);
    
    const response = await googleWebSearchAdapter.simpleSearch(query, limit);
    
    res.json({
      query,
      results: response.results,
      count: response.results.length,
      status: response.status,
      error: response.error,
      source: 'google-web-search',
      timestamp: new Date().toISOString(),
    });
  } catch (error: unknown) {
    console.error('[WebSearch API] Error:', getErrorMessage(error));
    res.status(500).json({ error: getErrorMessage(error), status: 'error' });
  }
});

searchRouter.post("/web", generousLimiter, async (req: Request, res: Response) => {
  try {
    const { query, limit = 5 } = req.body;
    
    if (!query || typeof query !== 'string' || query.trim().length === 0) {
      return res.status(400).json({ error: 'Query string is required in request body' });
    }
    
    const maxLimit = Math.min(limit, 10);
    
    // Check if Google Free Search is enabled
    if (!searchProviderState.google_free.enabled) {
      return res.json({
        query,
        results: [],
        count: 0,
        status: 'disabled',
        message: 'Google Free Search is disabled. Enable it in Sources page.',
        source: 'google-web-search',
        timestamp: new Date().toISOString(),
      });
    }
    
    console.log(`[WebSearch API] POST Query: "${query}" (limit: ${maxLimit})`);
    
    const response = await googleWebSearchAdapter.simpleSearch(query, maxLimit);
    
    res.json({
      query,
      results: response.results,
      count: response.results.length,
      status: response.status,
      error: response.error,
      source: 'google-web-search',
      timestamp: new Date().toISOString(),
    });
  } catch (error: unknown) {
    console.error('[WebSearch API] Error:', getErrorMessage(error));
    res.status(500).json({ error: getErrorMessage(error), status: 'error' });
  }
});

searchRouter.get("/providers", generousLimiter, async (req: Request, res: Response) => {
  try {
    const tavilyKey = process.env.TAVILY_API_KEY;
    const perplexityKey = process.env.PERPLEXITY_API_KEY;
    
    res.json({
      success: true,
      data: {
        google_free: {
          available: true,
          enabled: searchProviderState.google_free.enabled,
          requires_key: false,
        },
        tavily: {
          available: !!tavilyKey,
          enabled: searchProviderState.tavily.enabled,
          requires_key: true,
          has_key: !!tavilyKey,
        },
        perplexity: {
          available: !!perplexityKey,
          enabled: searchProviderState.perplexity.enabled,
          requires_key: true,
          has_key: !!perplexityKey,
        },
      },
    });
  } catch (error: unknown) {
    res.status(500).json({ success: false, error: getErrorMessage(error) });
  }
});

searchRouter.post("/providers/:provider/toggle", generousLimiter, async (req: Request, res: Response) => {
  try {
    const { provider } = req.params;
    const { enabled } = req.body;
    
    if (provider !== 'google_free' && provider !== 'tavily' && provider !== 'perplexity') {
      return res.status(400).json({ success: false, error: `Unknown provider: ${provider}` });
    }
    
    if (provider === 'tavily' && enabled && !process.env.TAVILY_API_KEY) {
      return res.status(400).json({ 
        success: false, 
        error: 'TAVILY_API_KEY not set. Please add it to your secrets.',
        error_code: 'MISSING_API_KEY'
      });
    }
    
    if (provider === 'perplexity' && enabled && !process.env.PERPLEXITY_API_KEY) {
      return res.status(400).json({ 
        success: false, 
        error: 'PERPLEXITY_API_KEY not set. Please add it to your secrets.',
        error_code: 'MISSING_API_KEY'
      });
    }
    
    searchProviderState[provider].enabled = enabled;
    
    console.log(`[SearchProviders] ${provider} ${enabled ? 'enabled' : 'disabled'}`);
    
    const tavilyKey = process.env.TAVILY_API_KEY;
    const perplexityKey = process.env.PERPLEXITY_API_KEY;
    
    res.json({
      success: true,
      message: `${provider} ${enabled ? 'enabled' : 'disabled'}`,
      data: {
        provider,
        enabled,
        status: {
          google_free: {
            available: true,
            enabled: searchProviderState.google_free.enabled,
            requires_key: false,
          },
          tavily: {
            available: !!tavilyKey,
            enabled: searchProviderState.tavily.enabled,
            requires_key: true,
            has_key: !!tavilyKey,
          },
          perplexity: {
            available: !!perplexityKey,
            enabled: searchProviderState.perplexity.enabled,
            requires_key: true,
            has_key: !!perplexityKey,
          },
        },
      },
    });
  } catch (error: unknown) {
    res.status(500).json({ success: false, error: getErrorMessage(error) });
  }
});

/**
 * Tavily Usage Statistics Endpoint
 * 
 * Returns current Tavily API usage stats including:
 * - Daily request counts
 * - Estimated costs
 * - Rate limit status
 * 
 * CRITICAL: Monitor this to prevent API cost overruns.
 */
searchRouter.get("/tavily-usage", generousLimiter, async (req: Request, res: Response) => {
  try {
    const { tavilyUsageLimiter } = await import("../tavily-usage-limiter");
    const stats = tavilyUsageLimiter.getStats();
    
    res.json({
      success: true,
      data: {
        enabled: stats.enabled,
        limits: {
          perMinute: stats.limits.perMinute,
          perDay: stats.limits.perDay,
          dailyCostLimit: `$${(stats.limits.dailyCostCents / 100).toFixed(2)}`,
        },
        today: {
          date: stats.today.date,
          searchCount: stats.today.searchCount,
          extractCount: stats.today.extractCount,
          totalRequests: stats.today.searchCount + stats.today.extractCount,
          estimatedCost: `$${(stats.today.estimatedCostCents / 100).toFixed(2)}`,
        },
        recentRequestsInLastMinute: stats.recentRequestsCount,
        rateStatus: stats.recentRequestsCount >= stats.limits.perMinute ? 'RATE_LIMITED' : 'OK',
        dailyStatus: (stats.today.searchCount + stats.today.extractCount) >= stats.limits.perDay ? 'DAILY_LIMIT_REACHED' : 'OK',
      },
    });
  } catch (error: unknown) {
    res.status(500).json({ success: false, error: getErrorMessage(error) });
  }
});

/**
 * Zeus Web Search Endpoint - QIG-Pure Integration
 * 
 * Called by Python Zeus Chat to get web search results with QIG metrics.
 * Maintains QIG purity by:
 * - Computing Fisher information geometry on results
 * - Encoding results into 64D basin coordinates via block universe mapping
 * - Returning Fisher-Rao distances (NOT Euclidean/cosine)
 * 
 * Flow: Zeus (Python) → This endpoint → GoogleWebSearchAdapter → QIG scoring
 */
searchRouter.post("/zeus-web-search", generousLimiter, async (req: Request, res: Response) => {
  try {
    const { query, max_results = 5 } = req.body;
    
    if (!query || typeof query !== 'string' || query.trim().length === 0) {
      return res.status(400).json({ 
        success: false, 
        error: 'Query string is required',
        results: [],
      });
    }
    
    // Check if Google Free Search is enabled
    if (!searchProviderState.google_free.enabled) {
      console.log('[ZeusWebSearch] Google Free Search disabled');
      return res.json({
        success: true,
        query,
        results: [],
        source: 'google-free',
        status: 'disabled',
        message: 'Google Free Search is disabled. Enable it in Sources page.',
        qig_metrics: {
          provider_enabled: false,
        },
      });
    }
    
    console.log(`[ZeusWebSearch] Query from Zeus: "${query}" (max: ${max_results})`);
    
    // Execute search with QIG integration using simpleSearch
    const limit = Math.min(max_results, 10);
    const searchResponse = await googleWebSearchAdapter.simpleSearch(query, limit);
    
    // Process results with QIG metrics
    const resultsWithQIG = [];
    for (const result of searchResponse.results) {
      // Get QIG score for each result's content
      let qigScore = { phi: 0.5, kappa: 50.0, regime: 'search' };
      try {
        const content = `${result.title} ${result.description}`;
        // Use 'arbitrary' type for general text scoring
        const scored = await scoreUniversalQIGAsync(content, 'arbitrary');
        qigScore = {
          phi: scored.phi || 0.5,
          kappa: scored.kappa || 50.0,
          regime: scored.regime || 'search',
        };
      } catch (e) {
        // Fallback to default if scoring fails
        console.log(`[ZeusWebSearch] QIG scoring fallback for: ${result.title?.slice(0, 30)}`);
      }
      
      resultsWithQIG.push({
        title: result.title || 'Untitled',
        url: result.url || '',
        description: result.description || '',
        source: 'google-free',
        qig: qigScore,
        // Fisher-Rao distance and block universe coords computed by Python
        // We provide raw content for Python's basin encoder
        content_for_encoding: `${result.title} ${result.description}`,
      });
    }
    
    console.log(`[ZeusWebSearch] Returning ${resultsWithQIG.length} results with QIG metrics`);
    
    // AUTO-ADD TO ZETTELKASTEN (non-blocking)
    if (resultsWithQIG.length > 0) {
      const { getInternalHeaders } = require('../internal-auth');
      fetch('http://localhost:5000/api/zettelkasten/add-from-search', {
        method: 'POST',
        headers: getInternalHeaders(),
        body: JSON.stringify({
          query,
          results: resultsWithQIG.map(r => ({
            content: r.content_for_encoding || `${r.title} ${r.description}`,
            url: r.url,
            title: r.title
          })),
          source: 'zeus-web-search'
        })
      }).then(r => r.ok ? console.log('[ZeusWebSearch] Zettelkasten auto-saved') : console.warn('[ZeusWebSearch] Zettelkasten save failed:', r.status))
        .catch(err => console.log('[ZeusWebSearch] Zettelkasten auto-save skipped:', err.message));
    }
    
    res.json({
      success: true,
      query,
      results: resultsWithQIG,
      source: 'google-free',
      status: searchResponse.status,
      error: searchResponse.error,
      count: resultsWithQIG.length,
      qig_metrics: {
        provider_enabled: true,
        fisher_rao_ready: true,
        block_universe_coords: 'computed_by_caller',
      },
      timestamp: new Date().toISOString(),
    });
  } catch (error: unknown) {
    console.error('[ZeusWebSearch] Error:', getErrorMessage(error));
    res.status(500).json({ 
      success: false, 
      error: getErrorMessage(error), 
      results: [],
      status: 'error',
    });
  }
});

searchRouter.get("/known-phrases", generousLimiter, (req: Request, res: Response) => {
  try {
    res.json({ phrases: KNOWN_12_WORD_PHRASES });
  } catch (error: unknown) {
    res.status(500).json({ error: getErrorMessage(error) });
  }
});

searchRouter.get("/candidates", generousLimiter, async (req: Request, res: Response) => {
  try {
    const candidates = await storageFacade.candidates.getCandidates();
    res.json(candidates);
  } catch (error: unknown) {
    res.status(500).json({ error: getErrorMessage(error) });
  }
});

searchRouter.get("/analytics", generousLimiter, async (req: Request, res: Response) => {
  try {
    const candidates = await storageFacade.candidates.getCandidates();
    
    const scores = candidates.map(c => c.score);
    const mean = scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
    const sorted = [...scores].sort((a, b) => a - b);
    const median = sorted.length > 0 
      ? sorted.length % 2 === 0 
        ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
        : sorted[Math.floor(sorted.length / 2)]
      : 0;
    
    const p75 = sorted.length > 0 ? sorted[Math.floor(sorted.length * 0.75)] : 0;
    const p90 = sorted.length > 0 ? sorted[Math.floor(sorted.length * 0.90)] : 0;
    const p95 = sorted.length > 0 ? sorted[Math.floor(sorted.length * 0.95)] : 0;
    const max = sorted.length > 0 ? sorted[sorted.length - 1] : 0;
    
    const bip39Wordlist = getBIP39Wordlist();
    const bip39WordSet = new Set(bip39Wordlist.map((w: string) => w.toLowerCase()));
    
    const wordFrequency: Record<string, number> = {};
    const highPhiCandidates = candidates.filter(c => c.score >= 75);
    
    highPhiCandidates.forEach(c => {
      const words = c.phrase.toLowerCase().split(/\s+/);
      words.forEach(word => {
        if (bip39WordSet.has(word)) {
          wordFrequency[word] = (wordFrequency[word] || 0) + 1;
        }
      });
    });
    
    const topWords = Object.entries(wordFrequency)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 20)
      .map(([word, count]) => ({ word, count, frequency: count / highPhiCandidates.length }));
    
    // QIG-pure metrics (phi, kappa) - replacing legacy contextScore/eleganceScore/typingScore
    const avgPhi = candidates.length > 0
      ? candidates.reduce((sum, c) => sum + (c.qigScore?.phi ?? 0), 0) / candidates.length
      : 0;
    const avgKappa = candidates.length > 0
      ? candidates.reduce((sum, c) => sum + (c.qigScore?.kappa ?? 0), 0) / candidates.length
      : 0;
    
    const recent = candidates.slice(-100);
    const recentMean = recent.length > 0 
      ? recent.reduce((sum, c) => sum + c.score, 0) / recent.length 
      : 0;
    const older = candidates.slice(0, -100);
    const olderMean = older.length > 0 
      ? older.reduce((sum, c) => sum + c.score, 0) / older.length 
      : 0;
    const improvement = recentMean - olderMean;
    
    res.json({
      statistics: {
        count: candidates.length,
        mean: mean.toFixed(2),
        median: median.toFixed(2),
        p75: p75.toFixed(2),
        p90: p90.toFixed(2),
        p95: p95.toFixed(2),
        max: max.toFixed(2),
      },
      qigMetrics: {
        avgPhi: avgPhi.toFixed(4),
        avgKappa: avgKappa.toFixed(2),
      },
      patterns: {
        topWords,
        highPhiCount: highPhiCandidates.length,
      },
      trajectory: {
        recentMean: recentMean.toFixed(2),
        olderMean: olderMean.toFixed(2),
        improvement: improvement.toFixed(2),
        isImproving: improvement > 0,
      },
    });
  } catch (error: unknown) {
    res.status(500).json({ error: getErrorMessage(error) });
  }
});

searchRouter.get("/target-addresses", async (req: Request, res: Response) => {
  try {
    res.set('Cache-Control', 'no-store');
    const addresses = await storage.getTargetAddresses();
    res.json(addresses);
  } catch (error: unknown) {
    res.status(500).json({ error: getErrorMessage(error) });
  }
});

searchRouter.post("/target-addresses", async (req: Request, res: Response) => {
  try {
    const validation = addAddressRequestSchema.safeParse(req.body);
    
    if (!validation.success) {
      return res.status(400).json({
        error: validation.error.errors[0].message,
      });
    }

    const { address, label } = validation.data;
    const targetAddress: TargetAddress = {
      id: randomUUID(),
      address,
      label,
      addedAt: new Date().toISOString(),
    };

    await storage.addTargetAddress(targetAddress);
    res.json(targetAddress);
  } catch (error: unknown) {
    res.status(500).json({ error: getErrorMessage(error) });
  }
});

searchRouter.delete("/target-addresses/:id", async (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    
    if (id === "default") {
      return res.status(403).json({ error: "Cannot delete the default address" });
    }
    
    await storage.removeTargetAddress(id);
    res.json({ success: true });
  } catch (error: unknown) {
    res.status(500).json({ error: getErrorMessage(error) });
  }
});

searchRouter.post("/generate-random-phrases", async (req: Request, res: Response) => {
  try {
    const validation = generateRandomPhrasesRequestSchema.safeParse(req.body);
    
    if (!validation.success) {
      return res.status(400).json({
        error: validation.error.errors[0].message,
      });
    }

    const { count } = validation.data;
    const phrases: string[] = [];
    
    for (let i = 0; i < count; i++) {
      phrases.push(generateRandomBIP39Phrase());
    }

    res.json({ phrases });
  } catch (error: unknown) {
    res.status(500).json({ error: getErrorMessage(error) });
  }
});

searchRouter.post("/search-jobs", async (req: Request, res: Response) => {
  try {
    const validation = createSearchJobRequestSchema.safeParse(req.body);
    
    if (!validation.success) {
      return res.status(400).json({
        error: validation.error.errors[0].message,
      });
    }

    const { strategy, params } = validation.data;
    
    const job: SearchJob = {
      id: randomUUID(),
      strategy,
      status: "pending",
      params,
      progress: {
        tested: 0,
        highPhiCount: 0,
        lastBatchIndex: 0,
      },
      stats: {
        startTime: undefined,
        endTime: undefined,
        rate: 0,
      },
      logs: [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };

    await storageFacade.searchJobs.addSearchJob(job);
    res.json(job);
  } catch (error: unknown) {
    res.status(500).json({ error: getErrorMessage(error) });
  }
});

searchRouter.get("/search-jobs", async (req: Request, res: Response) => {
  try {
    const jobs = await storageFacade.searchJobs.getSearchJobs();
    res.json(jobs);
  } catch (error: unknown) {
    res.status(500).json({ error: getErrorMessage(error) });
  }
});

searchRouter.get("/search-jobs/:id", async (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    const job = await storageFacade.searchJobs.getSearchJob(id);
    
    if (!job) {
      return res.status(404).json({ error: "Job not found" });
    }

    res.json(job);
  } catch (error: unknown) {
    res.status(500).json({ error: getErrorMessage(error) });
  }
});

searchRouter.get("/search-jobs/:id/logs", async (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    const limit = parseInt(req.query.limit as string) || 50;
    const job = await storageFacade.searchJobs.getSearchJob(id);
    
    if (!job) {
      return res.status(404).json({ error: "Job not found" });
    }

    const logs = job.logs.slice(-limit).reverse();
    res.json({ logs, total: job.logs.length });
  } catch (error: unknown) {
    res.status(500).json({ error: getErrorMessage(error) });
  }
});

searchRouter.post("/search-jobs/:id/stop", async (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    await searchCoordinator.stopJob(id);
    
    const job = await storageFacade.searchJobs.getSearchJob(id);
    if (!job) {
      return res.status(404).json({ error: "Job not found" });
    }

    res.json(job);
  } catch (error: unknown) {
    res.status(500).json({ error: getErrorMessage(error) });
  }
});

searchRouter.delete("/search-jobs/:id", async (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    await storageFacade.searchJobs.deleteSearchJob(id);
    res.json({ success: true });
  } catch (error: unknown) {
    res.status(500).json({ error: getErrorMessage(error) });
  }
});

searchRouter.get("/activity-stream", async (req: Request, res: Response) => {
  try {
    const limit = parseInt(req.query.limit as string) || 100;
    
    const allEvents: Array<{
      id: string;
      type: string;
      identity: string;
      details: string;
      timestamp: string;
      metadata?: Record<string, unknown>;
    }> = [];
    
    let jobs: any[] = [];
    try {
      const jobsPromise = storageFacade.searchJobs.getSearchJobs();
      const timeoutPromise = new Promise<any[]>((_, reject) => 
        setTimeout(() => reject(new Error('timeout')), 2000)
      );
      jobs = await Promise.race([jobsPromise, timeoutPromise]);
    } catch {
      console.log('[ActivityStream] Search jobs fetch timed out, using Ocean logs only');
    }
    
    for (const job of jobs) {
      for (const log of job.logs) {
        allEvents.push({
          id: `${job.id}-${log.timestamp}`,
          type: log.type || 'info',
          identity: job.strategy || 'Search',
          details: log.message,
          timestamp: log.timestamp,
          metadata: { jobId: job.id },
        });
      }
    }
    
    const oceanLogs = activityLogStore.getLogs({ limit: limit * 2 });
    for (const oceanLog of oceanLogs) {
      allEvents.push({
        id: oceanLog.id,
        type: oceanLog.type || 'info',
        identity: oceanLog.category || 'Ocean',
        details: oceanLog.message,
        timestamp: oceanLog.timestamp,
        metadata: oceanLog.metadata,
      });
    }
    
    allEvents.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
    
    const oceanStatus = oceanSessionManager.getInvestigationStatus();
    const isOceanActive = oceanStatus?.isRunning || false;
    
    res.json({ 
      events: allEvents.slice(0, limit),
      activeJobs: jobs.filter(j => j.status === "running").length + (isOceanActive ? 1 : 0),
      totalJobs: jobs.length + (oceanLogs.length > 0 ? 1 : 0),
      oceanActive: isOceanActive,
    });
  } catch (error: unknown) {
    console.error('[ActivityStream] Error:', getErrorMessage(error));
    res.json({ 
      events: [],
      activeJobs: 0,
      totalJobs: 0,
      oceanActive: false,
    });
  }
});

searchRouter.post("/memory-search", async (req: Request, res: Response) => {
  try {
    const { fragments, targetAddress, options } = req.body as {
      fragments: MemoryFragment[];
      targetAddress?: string;
      options?: { maxCandidates?: number; includeTypos?: boolean };
    };
    
    if (!fragments || !Array.isArray(fragments) || fragments.length === 0) {
      return res.status(400).json({ error: "At least one memory fragment is required" });
    }
    
    const validFragments = fragments.filter(f => 
      f && typeof f.text === 'string' && f.text.trim().length > 0
    );
    
    if (validFragments.length === 0) {
      return res.status(400).json({ error: "No valid fragments provided" });
    }
    
    const addresses = await storage.getTargetAddresses();
    const target = targetAddress || addresses[0]?.address || "";
    
    const candidates = await runMemoryFragmentSearch(validFragments, target, {
      maxCandidates: options?.maxCandidates || 5000,
      includeTypos: options?.includeTypos ?? true,
    });
    
    res.json({
      candidateCount: candidates.length,
      topCandidates: candidates.slice(0, 50).map(c => ({
        phrase: c.phrase,
        confidence: c.confidence,
        fragments: c.fragments,
        phi: c.qigScore?.phi,
        kappa: c.qigScore?.kappa,
        regime: c.qigScore?.regime,
        inResonance: c.qigScore?.inResonance,
        combinedScore: c.combinedScore,
      })),
    });
  } catch (error: unknown) {
    handleRouteError(res, error, 'MemorySearch');
  }
});

export const formatRouter = Router();

formatRouter.get("/address/:address", async (req: Request, res: Response) => {
  try {
    const { detectAddressFormat, estimateAddressEra } = await import('../format-detection');
    const { address } = req.params;
    
    const formatInfo = detectAddressFormat(address);
    const eraInfo = estimateAddressEra(address);
    
    res.json({
      address,
      ...formatInfo,
      era: eraInfo,
    });
  } catch (error: unknown) {
    res.status(500).json({ error: getErrorMessage(error) });
  }
});

formatRouter.post("/mnemonic", async (req: Request, res: Response) => {
  try {
    const { detectMnemonicFormat } = await import('../format-detection');
    const { phrase } = req.body;
    
    if (!phrase || typeof phrase !== 'string') {
      return res.status(400).json({ error: 'phrase is required' });
    }
    
    const formatInfo = detectMnemonicFormat(phrase);
    
    res.json({
      phrase: phrase.split(/\s+/).slice(0, 3).join(' ') + '...',
      ...formatInfo,
    });
  } catch (error: unknown) {
    res.status(500).json({ error: getErrorMessage(error) });
  }
});

formatRouter.post("/batch-addresses", async (req: Request, res: Response) => {
  try {
    const { detectAddressFormat, estimateAddressEra } = await import('../format-detection');
    const { addresses } = req.body;
    
    if (!Array.isArray(addresses)) {
      return res.status(400).json({ error: 'addresses array is required' });
    }
    
    const results = addresses.slice(0, 100).map((address: string) => ({
      address,
      format: detectAddressFormat(address),
      era: estimateAddressEra(address),
    }));
    
    const summary = {
      total: results.length,
      byFormat: {} as Record<string, number>,
      legacy2009Era: results.filter(r => r.format.format === 'legacy').length,
    };
    
    results.forEach(r => {
      summary.byFormat[r.format.format] = (summary.byFormat[r.format.format] || 0) + 1;
    });
    
    res.json({ results, summary });
  } catch (error: unknown) {
    res.status(500).json({ error: getErrorMessage(error) });
  }
});
