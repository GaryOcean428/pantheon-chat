/**
 * TAVILY USAGE LIMITER
 * 
 * Prevents excessive Tavily API usage to avoid high costs.
 * Implements:
 * - Per-minute rate limiting
 * - Daily usage caps
 * - Usage logging and tracking
 * - Cost estimation
 * 
 * CRITICAL: Previous project incurred $450 in Tavily charges from uncontrolled usage.
 */

interface UsageRecord {
  timestamp: number;
  endpoint: string;
  query: string;
}

interface DailyStats {
  date: string;
  searchCount: number;
  extractCount: number;
  estimatedCostCents: number;
}

// Tavily pricing (approximate):
// - Search API: $0.01 per search (1 cent)
// - Extract API: $0.02 per extract (2 cents)
const SEARCH_COST_CENTS = 1;
const EXTRACT_COST_CENTS = 2;

class TavilyUsageLimiter {
  private usageHistory: UsageRecord[] = [];
  private dailyStats: Map<string, DailyStats> = new Map();
  
  // Limits
  private maxSearchesPerMinute = 5;
  private maxSearchesPerDay = 100;
  private maxDailyCostCents = 500; // $5 max daily
  
  private enabled = true;
  
  constructor() {
    console.log('[TavilyLimiter] Initialized with limits:');
    console.log(`  - Max ${this.maxSearchesPerMinute} searches/minute`);
    console.log(`  - Max ${this.maxSearchesPerDay} searches/day`);
    console.log(`  - Max $${(this.maxDailyCostCents / 100).toFixed(2)}/day`);
  }
  
  /**
   * Check if a Tavily API call is allowed
   */
  canMakeRequest(endpoint: 'search' | 'extract'): { allowed: boolean; reason?: string } {
    if (!this.enabled) {
      return { allowed: false, reason: 'Tavily limiter is disabled' };
    }
    
    const now = Date.now();
    const oneMinuteAgo = now - 60000;
    const today = this.getDateString();
    
    // Check per-minute rate
    const recentRequests = this.usageHistory.filter(r => r.timestamp > oneMinuteAgo);
    if (recentRequests.length >= this.maxSearchesPerMinute) {
      console.warn(`[TavilyLimiter] BLOCKED: Rate limit exceeded (${recentRequests.length}/${this.maxSearchesPerMinute} per minute)`);
      return { 
        allowed: false, 
        reason: `Rate limit exceeded: ${recentRequests.length} requests in last minute (max: ${this.maxSearchesPerMinute})` 
      };
    }
    
    // Check daily limit
    const stats = this.dailyStats.get(today) || this.createDayStats(today);
    const dailySearches = stats.searchCount + stats.extractCount;
    if (dailySearches >= this.maxSearchesPerDay) {
      console.warn(`[TavilyLimiter] BLOCKED: Daily limit exceeded (${dailySearches}/${this.maxSearchesPerDay})`);
      return { 
        allowed: false, 
        reason: `Daily limit exceeded: ${dailySearches} requests today (max: ${this.maxSearchesPerDay})` 
      };
    }
    
    // Check daily cost limit
    if (stats.estimatedCostCents >= this.maxDailyCostCents) {
      console.warn(`[TavilyLimiter] BLOCKED: Daily cost limit exceeded ($${(stats.estimatedCostCents / 100).toFixed(2)})`);
      return { 
        allowed: false, 
        reason: `Daily cost limit exceeded: $${(stats.estimatedCostCents / 100).toFixed(2)} (max: $${(this.maxDailyCostCents / 100).toFixed(2)})` 
      };
    }
    
    return { allowed: true };
  }
  
  /**
   * Record a Tavily API call
   */
  recordRequest(endpoint: 'search' | 'extract', query: string): void {
    const now = Date.now();
    const today = this.getDateString();
    
    // Add to history
    this.usageHistory.push({ timestamp: now, endpoint, query });
    
    // Clean old history (keep last hour)
    const oneHourAgo = now - 3600000;
    this.usageHistory = this.usageHistory.filter(r => r.timestamp > oneHourAgo);
    
    // Update daily stats
    let stats = this.dailyStats.get(today);
    if (!stats) {
      stats = this.createDayStats(today);
      this.dailyStats.set(today, stats);
    }
    
    if (endpoint === 'search') {
      stats.searchCount++;
      stats.estimatedCostCents += SEARCH_COST_CENTS;
    } else {
      stats.extractCount++;
      stats.estimatedCostCents += EXTRACT_COST_CENTS;
    }
    
    console.log(`[TavilyLimiter] Recorded ${endpoint}: "${query.slice(0, 50)}..." (daily: ${stats.searchCount + stats.extractCount}, cost: $${(stats.estimatedCostCents / 100).toFixed(2)})`);
  }
  
  /**
   * Get current usage statistics
   */
  getStats(): {
    enabled: boolean;
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
      limits: {
        perMinute: this.maxSearchesPerMinute,
        perDay: this.maxSearchesPerDay,
        dailyCostCents: this.maxDailyCostCents,
      },
      today: stats,
      recentRequestsCount: recentCount,
    };
  }
  
  /**
   * Update limits (admin only)
   */
  updateLimits(limits: { perMinute?: number; perDay?: number; dailyCostCents?: number }): void {
    if (limits.perMinute !== undefined) {
      this.maxSearchesPerMinute = Math.max(1, Math.min(limits.perMinute, 20));
    }
    if (limits.perDay !== undefined) {
      this.maxSearchesPerDay = Math.max(10, Math.min(limits.perDay, 1000));
    }
    if (limits.dailyCostCents !== undefined) {
      this.maxDailyCostCents = Math.max(100, Math.min(limits.dailyCostCents, 5000));
    }
    
    console.log(`[TavilyLimiter] Updated limits: ${this.maxSearchesPerMinute}/min, ${this.maxSearchesPerDay}/day, $${(this.maxDailyCostCents / 100).toFixed(2)}/day`);
  }
  
  /**
   * Enable/disable the limiter
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    console.log(`[TavilyLimiter] ${enabled ? 'Enabled' : 'Disabled'}`);
  }
  
  private getDateString(): string {
    return new Date().toISOString().split('T')[0];
  }
  
  private createDayStats(date: string): DailyStats {
    return {
      date,
      searchCount: 0,
      extractCount: 0,
      estimatedCostCents: 0,
    };
  }
}

export const tavilyUsageLimiter = new TavilyUsageLimiter();
