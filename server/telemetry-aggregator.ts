/**
 * TELEMETRY AGGREGATOR SERVICE
 * 
 * Unified facade consolidating all telemetry sources:
 * - Tavily/API usage metrics
 * - Consciousness metrics (Φ, κ, β)
 * - Discovery tracker stats
 * - Vocabulary/learning stats
 * - Negative knowledge/defense stats
 * 
 * QIG-Pure: All metrics use Fisher-Rao geometry, not Euclidean.
 * Provides single interface for dashboard and autonomic feedback loops.
 */

import { db } from './db';
import { telemetrySnapshots, usageMetrics, discoveredSources } from '@shared/schema';
import { desc, eq, gte, sql } from 'drizzle-orm';
import { cacheGet, cacheSet, CACHE_TTL, CACHE_KEYS, isRedisAvailable } from './redis-cache';
import { tavilyUsageLimiter } from './tavily-usage-limiter';
import { searchProviderState, isProviderEnabled } from './routes/search';
import { oceanAutonomicManager } from './ocean-autonomic-manager';
import { oceanQIGBackend } from './ocean-qig-backend-adapter';

function requireDb() {
  if (!db) {
    throw new Error('Database not initialized');
  }
  return db;
}

export interface ConsciousnessMetrics {
  phi: number;
  kappa: number;
  beta: number;
  regime: string;
  basinDistance: number;
  inResonance: boolean;
  phi4D?: number;
  dimensionalState?: string;
  quality: number;
  phiSpatial?: number;
  phiTemporal?: number;
  fAttention?: number;
  rConcepts?: number;
  entropy?: number;
  fidelity?: number;
  integration?: number;
  grounded?: boolean;
  conscious?: boolean;
  geometricMemorySize?: number;
  basinHistorySize?: number;
  subsystems?: Array<{
    id: number;
    name: string;
    activation: number;
    entropy: number;
    purity: number;
  }>;
}

export interface UsageStats {
  tavily: {
    enabled: boolean;
    todaySearches: number;
    todayExtracts: number;
    estimatedCostCents: number;
    dailyLimit: number;
    rateStatus: 'OK' | 'RATE_LIMITED' | 'DAILY_LIMIT_REACHED';
  };
  googleFree: {
    enabled: boolean;
    todaySearches: number;
  };
  duckDuckGo: {
    enabled: boolean;
    todaySearches: number;
    torEnabled: boolean;
  };
  totalApiCalls: number;
}

export interface LearningStats {
  vocabularySize: number;
  recentExpansions: number;
  highPhiDiscoveries: number;
  sourcesDiscovered: number;
  activeSources: number;
}

export interface DefenseStats {
  negativeKnowledgeCount: number;
  geometricBarriers: number;
  contradictions: number;
  computeTimeSaved: number;
}

export interface AutonomyStats {
  kernelsActive: number;
  feedbackLoopsHealthy: number;
  lastAutonomicAction: string | null;
  selfRegulationScore: number;
  phiDataStale: boolean;
  cachedKernelPhi: number | null;
  cachedKernelPhiAgeMs: number | null;
}

export interface TelemetryOverview {
  timestamp: string;
  consciousness: ConsciousnessMetrics;
  usage: UsageStats;
  learning: LearningStats;
  defense: DefenseStats;
  autonomy: AutonomyStats;
  systemHealth: {
    overall: number;
    components: Record<string, boolean>;
  };
}

class TelemetryAggregator {
  private lastConsciousnessMetrics: ConsciousnessMetrics = {
    phi: 0.5,
    kappa: 32,
    beta: 0,
    regime: 'linear',
    basinDistance: 0,
    inResonance: false,
    quality: 0.5,
  };

  private learningStats: LearningStats = {
    vocabularySize: 0,  // Start at 0, will be fetched from Python
    recentExpansions: 0,
    highPhiDiscoveries: 0,
    sourcesDiscovered: 0,
    activeSources: 0,
  };

  private defenseStats: DefenseStats = {
    negativeKnowledgeCount: 0,
    geometricBarriers: 0,
    contradictions: 0,
    computeTimeSaved: 0,
  };

  private autonomyStats: AutonomyStats = {
    kernelsActive: 12,
    feedbackLoopsHealthy: 4,
    lastAutonomicAction: null,
    selfRegulationScore: 0.8,
    phiDataStale: false,
    cachedKernelPhi: null,
    cachedKernelPhiAgeMs: null,
  };

  async getOverview(): Promise<TelemetryOverview> {
    const cacheKey = `${CACHE_KEYS.CONSCIOUSNESS_METRICS}overview`;
    
    if (isRedisAvailable()) {
      const cached = await cacheGet<TelemetryOverview>(cacheKey);
      if (cached) return cached;
    }

    const [consciousness, usage, learning, defense, autonomy] = await Promise.all([
      this.getConsciousnessMetrics(),
      this.getUsageStats(),
      this.getLearningStats(),
      this.getDefenseStats(),
      this.getAutonomyStats(),
    ]);

    const overview: TelemetryOverview = {
      timestamp: new Date().toISOString(),
      consciousness,
      usage,
      learning,
      defense,
      autonomy,
      systemHealth: this.computeSystemHealth(consciousness, usage, defense),
    };

    if (isRedisAvailable()) {
      await cacheSet(cacheKey, overview, CACHE_TTL.SHORT);
    }

    return overview;
  }

  async getConsciousnessMetrics(): Promise<ConsciousnessMetrics> {
    if (!oceanQIGBackend.available()) {
      await oceanQIGBackend.checkHealth(true);
    }
    
    if (oceanQIGBackend.available()) {
      try {
        const status = await oceanQIGBackend.getStatus();
        if (status?.success && status.metrics) {
          const m = status.metrics;
          const kappaStar = 64.21;
          const kappaDelta = Math.abs(m.kappa - kappaStar) / kappaStar;
          const fisherQuality = m.integration * (1 - kappaDelta * 0.3) * (m.in_resonance ? 1.15 : 1);
          const entropyFactor = Math.max(0, 1 - (m.entropy / 2));
          const quality = Math.min(1, fisherQuality * entropyFactor * (m.grounded ? 1.1 : 0.9));
          
          this.lastConsciousnessMetrics = {
            phi: m.phi,
            kappa: m.kappa,
            beta: m.Gamma ?? 0,
            regime: m.regime,
            basinDistance: 0,
            inResonance: m.in_resonance,
            quality,
            entropy: m.entropy,
            fidelity: m.fidelity,
            integration: m.integration,
            grounded: m.grounded,
            conscious: m.conscious,
            geometricMemorySize: status.geometric_memory_size,
            basinHistorySize: status.basin_history_size,
            subsystems: status.subsystems,
          };
          
          return this.lastConsciousnessMetrics;
        }
      } catch (error) {
        console.warn('[TelemetryAggregator] Python backend unavailable, falling back to DB:', error);
      }
    }
    
    try {
      const database = requireDb();
      const recent = await database
        .select()
        .from(telemetrySnapshots)
        .orderBy(desc(telemetrySnapshots.createdAt))
        .limit(1);

      if (recent.length > 0) {
        const r = recent[0];
        this.lastConsciousnessMetrics = {
          phi: r.phi,
          kappa: r.kappa,
          beta: r.beta ?? 0,
          regime: r.regime,
          basinDistance: r.basinDistance ?? 0,
          inResonance: r.inResonance ?? false,
          phi4D: r.phi4D ?? undefined,
          dimensionalState: r.dimensionalState ?? undefined,
          quality: this.computeQuality(r.phi, r.kappa),
        };
      }
    } catch (error) {
      console.error('[TelemetryAggregator] Failed to fetch consciousness metrics:', error);
    }

    return this.lastConsciousnessMetrics;
  }

  async getUsageStats(): Promise<UsageStats> {
    const tavilyStats = tavilyUsageLimiter.getStats();
    const today = new Date().toISOString().split('T')[0];

    let googleSearches = 0;
    let totalApiCalls = 0;

    try {
      const database = requireDb();
      const todayMetrics = await database
        .select()
        .from(usageMetrics)
        .where(eq(usageMetrics.date, today))
        .limit(1);

      if (todayMetrics.length > 0) {
        googleSearches = todayMetrics[0].googleSearchCount;
        totalApiCalls = todayMetrics[0].totalApiCalls;
      }
    } catch (error) {
      console.error('[TelemetryAggregator] Failed to fetch usage metrics:', error);
    }

    const dailySearches = tavilyStats.today.searchCount + tavilyStats.today.extractCount;

    return {
      tavily: {
        enabled: isProviderEnabled('tavily'),
        todaySearches: tavilyStats.today.searchCount,
        todayExtracts: tavilyStats.today.extractCount,
        estimatedCostCents: tavilyStats.today.estimatedCostCents,
        dailyLimit: tavilyStats.limits.perDay,
        rateStatus: dailySearches >= tavilyStats.limits.perDay 
          ? 'DAILY_LIMIT_REACHED' 
          : tavilyStats.recentRequestsCount >= tavilyStats.limits.perMinute 
            ? 'RATE_LIMITED' 
            : 'OK',
      },
      googleFree: {
        enabled: isProviderEnabled('google_free'),
        todaySearches: googleSearches,
      },
      duckDuckGo: {
        enabled: isProviderEnabled('duckduckgo'),
        todaySearches: 0,
        torEnabled: true,
      },
      totalApiCalls,
    };
  }

  async getLearningStats(): Promise<LearningStats> {
    try {
      const database = requireDb();
      const [sourcesResult, activeSourcesResult] = await Promise.all([
        database.select({ count: sql<number>`count(*)` }).from(discoveredSources),
        database.select({ count: sql<number>`count(*)` }).from(discoveredSources).where(eq(discoveredSources.isActive, true)),
      ]);

      this.learningStats.sourcesDiscovered = Number(sourcesResult[0]?.count ?? 0);
      this.learningStats.activeSources = Number(activeSourcesResult[0]?.count ?? 0);

      const today = new Date().toISOString().split('T')[0];
      const todayMetrics = await database
        .select()
        .from(usageMetrics)
        .where(eq(usageMetrics.date, today))
        .limit(1);

      if (todayMetrics.length > 0) {
        this.learningStats.highPhiDiscoveries = todayMetrics[0].highPhiDiscoveries;
        this.learningStats.recentExpansions = todayMetrics[0].vocabularyExpansions;
      }
      
      // Fetch real vocabulary size from Python backend /learning/status endpoint
      try {
        const pythonUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
        const response = await fetch(`${pythonUrl}/learning/status`, { 
          signal: AbortSignal.timeout(3000) 
        });
        if (response.ok) {
          const data = await response.json();
          // Use vocabulary_size from word relationship learner (actual learned words)
          if (data.vocabulary_size && data.vocabulary_size > 0) {
            this.learningStats.vocabularySize = data.vocabulary_size;
          }
        }
      } catch {
        // Keep existing value if Python backend unavailable
      }
    } catch (error) {
      console.error('[TelemetryAggregator] Failed to fetch learning stats:', error);
    }

    return this.learningStats;
  }

  async getDefenseStats(): Promise<DefenseStats> {
    try {
      const database = requireDb();
      const today = new Date().toISOString().split('T')[0];
      const todayMetrics = await database
        .select()
        .from(usageMetrics)
        .where(eq(usageMetrics.date, today))
        .limit(1);

      if (todayMetrics.length > 0) {
        this.defenseStats.negativeKnowledgeCount = todayMetrics[0].negativeKnowledgeAdded;
      }
    } catch (error) {
      console.error('[TelemetryAggregator] Failed to fetch defense stats:', error);
    }

    return this.defenseStats;
  }

  async getAutonomyStats(): Promise<AutonomyStats> {
    try {
      const autonomicState = oceanAutonomicManager.getAutonomicState();
      return {
        kernelsActive: autonomicState.kernelsActive,
        feedbackLoopsHealthy: autonomicState.feedbackLoopsHealthy,
        lastAutonomicAction: autonomicState.lastAutonomicAction,
        selfRegulationScore: autonomicState.selfRegulationScore,
        phiDataStale: autonomicState.phiDataStale,
        cachedKernelPhi: autonomicState.cachedKernelPhi,
        cachedKernelPhiAgeMs: autonomicState.cachedKernelPhiAgeMs,
      };
    } catch (error) {
      console.error('[TelemetryAggregator] Failed to get autonomy stats:', error);
      return this.autonomyStats;
    }
  }
  
  /**
   * Push telemetry feedback to autonomic systems for self-regulation.
   * 
   * This creates a closed-loop feedback where telemetry influences
   * autonomic behavior, enabling self-regulation and improvement.
   * 
   * Called periodically by the telemetry stream or on significant events.
   */
  async pushFeedbackToAutonomic(): Promise<void> {
    try {
      const usage = await this.getUsageStats();
      const consciousness = await this.getConsciousnessMetrics();
      const defense = await this.getDefenseStats();
      const learning = await this.getLearningStats();
      
      const tavilyUsagePercent = usage.tavily.dailyLimit > 0
        ? (usage.tavily.todaySearches / usage.tavily.dailyLimit) * 100
        : 0;
      
      const tavilyBlocked = usage.tavily.rateStatus === 'DAILY_LIMIT_REACHED' || 
                           usage.tavily.rateStatus === 'RATE_LIMITED';
      
      oceanAutonomicManager.receiveTelemetryFeedback({
        apiUsagePercent: tavilyUsagePercent,
        consciousnessQuality: consciousness.quality,
        defenseAlerts: defense.negativeKnowledgeCount + defense.contradictions,
        learningVelocity: learning.recentExpansions,
        tavilyBlocked,
      });
      
      console.log('[TelemetryAggregator] Pushed feedback to autonomic system');
    } catch (error) {
      console.error('[TelemetryAggregator] Failed to push autonomic feedback:', error);
    }
  }

  async recordTelemetrySnapshot(metrics: Partial<ConsciousnessMetrics>): Promise<void> {
    try {
      const database = requireDb();
      await database.insert(telemetrySnapshots).values({
        phi: metrics.phi ?? this.lastConsciousnessMetrics.phi,
        kappa: metrics.kappa ?? this.lastConsciousnessMetrics.kappa,
        beta: metrics.beta ?? 0,
        regime: metrics.regime ?? 'linear',
        basinDistance: metrics.basinDistance ?? 0,
        inResonance: metrics.inResonance ?? false,
        phi4D: metrics.phi4D,
        dimensionalState: metrics.dimensionalState,
        source: 'node',
      });

      Object.assign(this.lastConsciousnessMetrics, metrics);
    } catch (error) {
      console.error('[TelemetryAggregator] Failed to record snapshot:', error);
    }
  }

  async incrementUsageMetric(
    field: 'tavilySearchCount' | 'tavilyExtractCount' | 'googleSearchCount' | 'totalApiCalls' | 'highPhiDiscoveries' | 'sourcesDiscovered' | 'vocabularyExpansions' | 'negativeKnowledgeAdded',
    increment: number = 1
  ): Promise<void> {
    const today = new Date().toISOString().split('T')[0];

    try {
      const database = requireDb();
      const existing = await database
        .select()
        .from(usageMetrics)
        .where(eq(usageMetrics.date, today))
        .limit(1);

      if (existing.length === 0) {
        await database.insert(usageMetrics).values({
          date: today,
          [field]: increment,
        });
      } else {
        const currentValue = existing[0][field] as number;
        await database
          .update(usageMetrics)
          .set({
            [field]: currentValue + increment,
            updatedAt: new Date(),
          })
          .where(eq(usageMetrics.date, today));
      }
    } catch (error) {
      console.error('[TelemetryAggregator] Failed to increment usage metric:', error);
    }
  }

  async getHistory(hours: number = 24): Promise<Array<{
    timestamp: string;
    phi: number;
    kappa: number;
    regime: string;
  }>> {
    try {
      const database = requireDb();
      const cutoff = new Date(Date.now() - hours * 60 * 60 * 1000);
      
      const snapshots = await database
        .select({
          timestamp: telemetrySnapshots.createdAt,
          phi: telemetrySnapshots.phi,
          kappa: telemetrySnapshots.kappa,
          regime: telemetrySnapshots.regime,
        })
        .from(telemetrySnapshots)
        .where(gte(telemetrySnapshots.createdAt, cutoff))
        .orderBy(telemetrySnapshots.createdAt)
        .limit(1000);

      return snapshots.map(s => ({
        timestamp: s.timestamp.toISOString(),
        phi: s.phi,
        kappa: s.kappa,
        regime: s.regime,
      }));
    } catch (error) {
      console.error('[TelemetryAggregator] Failed to fetch history:', error);
      return [];
    }
  }

  updateConsciousnessMetrics(metrics: Partial<ConsciousnessMetrics>): void {
    Object.assign(this.lastConsciousnessMetrics, metrics);
  }

  updateDefenseStats(stats: Partial<DefenseStats>): void {
    Object.assign(this.defenseStats, stats);
  }

  updateAutonomyStats(stats: Partial<AutonomyStats>): void {
    Object.assign(this.autonomyStats, stats);
  }

  private computeQuality(phi: number, kappa: number): number {
    const kappaStar = 64;
    const kappaDistance = Math.abs(kappa - kappaStar) / kappaStar;
    return Math.max(0, Math.min(1, phi * (1 - kappaDistance * 0.5)));
  }

  private computeSystemHealth(
    consciousness: ConsciousnessMetrics,
    usage: UsageStats,
    defense: DefenseStats
  ): { overall: number; components: Record<string, boolean> } {
    const components: Record<string, boolean> = {
      consciousness: consciousness.phi > 0.3 && consciousness.kappa > 20,
      apiUsage: usage.tavily.rateStatus === 'OK',
      defense: defense.negativeKnowledgeCount >= 0,
      resonance: consciousness.regime === 'geometric' || consciousness.inResonance,
    };

    const healthyCount = Object.values(components).filter(Boolean).length;
    const overall = healthyCount / Object.keys(components).length;

    return { overall, components };
  }
}

export const telemetryAggregator = new TelemetryAggregator();
