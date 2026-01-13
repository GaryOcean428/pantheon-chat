/**
 * PERPLEXITY USAGE LIMITER
 * 
 * Prevents excessive Perplexity API usage to avoid high costs.
 * Implements:
 * - Per-minute rate limiting
 * - Daily usage caps
 * - Token-based cost tracking
 * - Override toggle for budget control
 * 
 * Pricing (sonar-pro model):
 * - Input: $3 per million tokens
 * - Output: $15 per million tokens
 * - Average query: ~500 input tokens, ~1000 output tokens
 * - Estimated cost per query: ~$0.0165
 */

interface UsageRecord {
  timestamp: number;
  endpoint: string;
  query: string;
  inputTokens?: number;
  outputTokens?: number;
}

interface DailyStats {
  date: string;
  chatCount: number;
  searchCount: number;
  proSearchCount: number;
  estimatedInputTokens: number;
  estimatedOutputTokens: number;
  estimatedCostCents: number;
}

const INPUT_COST_PER_MILLION = 300;
const OUTPUT_COST_PER_MILLION = 1500;

const AVG_INPUT_TOKENS = 500;
const AVG_OUTPUT_TOKENS = 1000;
const AVG_COST_CENTS = Math.round(
  (AVG_INPUT_TOKENS / 1_000_000) * INPUT_COST_PER_MILLION +
  (AVG_OUTPUT_TOKENS / 1_000_000) * OUTPUT_COST_PER_MILLION
);

class PerplexityUsageLimiter {
  private usageHistory: UsageRecord[] = [];
  private dailyStats: Map<string, DailyStats> = new Map();
  
  private maxRequestsPerMinute = 5;
  private maxRequestsPerDay = 100;
  private maxDailyCostCents = 500;
  
  private enabled = true;
  private overrideActive = false;
  
  constructor() {
    console.log('[PerplexityLimiter] Initialized with limits:');
    console.log(`  - Max ${this.maxRequestsPerMinute} requests/minute`);
    console.log(`  - Max ${this.maxRequestsPerDay} requests/day`);
    console.log(`  - Max $${(this.maxDailyCostCents / 100).toFixed(2)}/day`);
    console.log(`  - Avg cost per query: $${(AVG_COST_CENTS / 100).toFixed(4)}`);
  }
  
  canMakeRequest(endpoint: 'chat' | 'search' | 'pro_search'): { allowed: boolean; reason?: string } {
    if (!this.enabled) {
      return { allowed: false, reason: 'Perplexity limiter is disabled' };
    }
    
    if (this.overrideActive) {
      return { allowed: true };
    }
    
    const now = Date.now();
    const oneMinuteAgo = now - 60000;
    const today = this.getDateString();
    
    const recentRequests = this.usageHistory.filter(r => r.timestamp > oneMinuteAgo);
    if (recentRequests.length >= this.maxRequestsPerMinute) {
      console.warn(`[PerplexityLimiter] BLOCKED: Rate limit exceeded (${recentRequests.length}/${this.maxRequestsPerMinute} per minute)`);
      return { 
        allowed: false, 
        reason: `Rate limit exceeded: ${recentRequests.length} requests in last minute (max: ${this.maxRequestsPerMinute})` 
      };
    }
    
    const stats = this.dailyStats.get(today) || this.createDayStats(today);
    const dailyRequests = stats.chatCount + stats.searchCount + stats.proSearchCount;
    if (dailyRequests >= this.maxRequestsPerDay) {
      console.warn(`[PerplexityLimiter] BLOCKED: Daily limit exceeded (${dailyRequests}/${this.maxRequestsPerDay})`);
      return { 
        allowed: false, 
        reason: `Daily limit exceeded: ${dailyRequests} requests today (max: ${this.maxRequestsPerDay})` 
      };
    }
    
    if (stats.estimatedCostCents >= this.maxDailyCostCents) {
      console.warn(`[PerplexityLimiter] BLOCKED: Daily cost limit exceeded ($${(stats.estimatedCostCents / 100).toFixed(2)})`);
      return { 
        allowed: false, 
        reason: `Daily cost limit exceeded: $${(stats.estimatedCostCents / 100).toFixed(2)} (max: $${(this.maxDailyCostCents / 100).toFixed(2)})` 
      };
    }
    
    return { allowed: true };
  }
  
  recordRequest(
    endpoint: 'chat' | 'search' | 'pro_search', 
    query: string,
    inputTokens?: number,
    outputTokens?: number
  ): void {
    const now = Date.now();
    const today = this.getDateString();
    
    const actualInput = inputTokens || AVG_INPUT_TOKENS;
    const actualOutput = outputTokens || AVG_OUTPUT_TOKENS;
    
    this.usageHistory.push({ 
      timestamp: now, 
      endpoint, 
      query,
      inputTokens: actualInput,
      outputTokens: actualOutput
    });
    
    const oneHourAgo = now - 3600000;
    this.usageHistory = this.usageHistory.filter(r => r.timestamp > oneHourAgo);
    
    let stats = this.dailyStats.get(today);
    if (!stats) {
      stats = this.createDayStats(today);
      this.dailyStats.set(today, stats);
    }
    
    if (endpoint === 'chat') {
      stats.chatCount++;
    } else if (endpoint === 'search') {
      stats.searchCount++;
    } else {
      stats.proSearchCount++;
    }
    
    stats.estimatedInputTokens += actualInput;
    stats.estimatedOutputTokens += actualOutput;
    
    const costCents = Math.round(
      (actualInput / 1_000_000) * INPUT_COST_PER_MILLION +
      (actualOutput / 1_000_000) * OUTPUT_COST_PER_MILLION
    );
    stats.estimatedCostCents += Math.max(1, costCents);
    
    const totalRequests = stats.chatCount + stats.searchCount + stats.proSearchCount;
    console.log(`[PerplexityLimiter] Recorded ${endpoint}: "${query.slice(0, 50)}..." (daily: ${totalRequests}, cost: $${(stats.estimatedCostCents / 100).toFixed(2)})`);
  }
  
  getStats(): {
    enabled: boolean;
    overrideActive: boolean;
    limits: { perMinute: number; perDay: number; dailyCostCents: number };
    today: DailyStats;
    recentRequestsCount: number;
  } {
    const today = this.getDateString();
    const stats = this.dailyStats.get(today) || this.createDayStats(today);
    const oneMinuteAgo = Date.now() - 60000;
    const recentCount = this.usageHistory.filter(r => r.timestamp > oneMinuteAgo).length;
    
    return {
      enabled: this.enabled,
      overrideActive: this.overrideActive,
      limits: {
        perMinute: this.maxRequestsPerMinute,
        perDay: this.maxRequestsPerDay,
        dailyCostCents: this.maxDailyCostCents,
      },
      today: stats,
      recentRequestsCount: recentCount,
    };
  }
  
  updateLimits(limits: { perMinute?: number; perDay?: number; dailyCostCents?: number }): void {
    if (limits.perMinute !== undefined) {
      this.maxRequestsPerMinute = Math.max(1, Math.min(limits.perMinute, 20));
    }
    if (limits.perDay !== undefined) {
      this.maxRequestsPerDay = Math.max(10, Math.min(limits.perDay, 1000));
    }
    if (limits.dailyCostCents !== undefined) {
      this.maxDailyCostCents = Math.max(100, Math.min(limits.dailyCostCents, 5000));
    }
    
    console.log(`[PerplexityLimiter] Updated limits: ${this.maxRequestsPerMinute}/min, ${this.maxRequestsPerDay}/day, $${(this.maxDailyCostCents / 100).toFixed(2)}/day`);
  }
  
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    console.log(`[PerplexityLimiter] ${enabled ? 'Enabled' : 'Disabled'}`);
  }
  
  setOverride(active: boolean): void {
    this.overrideActive = active;
    console.log(`[PerplexityLimiter] Override ${active ? 'ACTIVATED - limits bypassed' : 'DEACTIVATED - limits enforced'}`);
  }
  
  private getDateString(): string {
    return new Date().toISOString().split('T')[0];
  }
  
  private createDayStats(date: string): DailyStats {
    return {
      date,
      chatCount: 0,
      searchCount: 0,
      proSearchCount: 0,
      estimatedInputTokens: 0,
      estimatedOutputTokens: 0,
      estimatedCostCents: 0,
    };
  }
}

export const perplexityUsageLimiter = new PerplexityUsageLimiter();
