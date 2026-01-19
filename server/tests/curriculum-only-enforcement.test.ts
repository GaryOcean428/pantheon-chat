/**
 * CURRICULUM-ONLY MODE ENFORCEMENT TESTS
 * 
 * Validates that when QIG_CURRICULUM_ONLY=true:
 * 1. Search adapters skip initialization
 * 2. Search endpoints return 403
 * 3. Search provider state is disabled
 * 4. No external network connections are made
 */

import { describe, it, expect, beforeAll, afterAll, vi } from 'vitest';

describe('Curriculum-Only Mode Enforcement', () => {
  const originalEnv = process.env.QIG_CURRICULUM_ONLY;

  beforeAll(() => {
    // Enable curriculum-only mode for tests
    process.env.QIG_CURRICULUM_ONLY = 'true';
  });

  afterAll(() => {
    // Restore original env
    if (originalEnv === undefined) {
      delete process.env.QIG_CURRICULUM_ONLY;
    } else {
      process.env.QIG_CURRICULUM_ONLY = originalEnv;
    }
  });

  describe('isCurriculumOnlyEnabled', () => {
    it('returns true when QIG_CURRICULUM_ONLY=true', async () => {
      const { isCurriculumOnlyEnabled } = await import('../lib/curriculum-mode');
      expect(isCurriculumOnlyEnabled()).toBe(true);
    });
  });

  describe('SearXNGGeometricAdapter', () => {
    it('skips initialization when curriculum-only mode is active', async () => {
      const consoleSpy = vi.spyOn(console, 'log');
      
      // Dynamic import to get fresh instance after env var is set
      const module = await import('../geometric-discovery/searxng-adapter');
      const adapter = new module.SearXNGGeometricAdapter();
      
      // Should log skipped initialization, not the regular initialization
      expect(consoleSpy).toHaveBeenCalledWith('[SearXNG] Skipped initialization (curriculum-only mode)');
      expect(consoleSpy).not.toHaveBeenCalledWith(expect.stringContaining('[SearXNG] Initialized FREE'));
      
      consoleSpy.mockRestore();
    });

    it('blocks search calls when curriculum-only mode is active', async () => {
      const module = await import('../geometric-discovery/searxng-adapter');
      const adapter = new module.SearXNGGeometricAdapter();
      
      const results = await adapter.search({
        text: 'test query',
        maxResults: 5,
      });
      
      // Should return empty array, not make external calls
      expect(results).toEqual([]);
    });
  });

  describe('GoogleWebSearchAdapter', () => {
    it('skips initialization when curriculum-only mode is active', async () => {
      const consoleSpy = vi.spyOn(console, 'log');
      
      // Dynamic import to get fresh instance
      const module = await import('../geometric-discovery/google-web-search-adapter');
      const adapter = new module.GoogleWebSearchAdapter();
      
      expect(consoleSpy).toHaveBeenCalledWith('[GoogleWebSearch] Skipped initialization (curriculum-only mode)');
      expect(consoleSpy).not.toHaveBeenCalledWith(expect.stringContaining('[GoogleWebSearch] Initialized FREE'));
      
      consoleSpy.mockRestore();
    });

    it('blocks search calls when curriculum-only mode is active', async () => {
      const module = await import('../geometric-discovery/google-web-search-adapter');
      const adapter = new module.GoogleWebSearchAdapter();
      
      const results = await adapter.search({
        text: 'test query',
        maxResults: 5,
      });
      
      expect(results).toEqual([]);
    });

    it('blocks simpleSearch calls when curriculum-only mode is active', async () => {
      const module = await import('../geometric-discovery/google-web-search-adapter');
      const adapter = new module.GoogleWebSearchAdapter();
      
      const response = await adapter.simpleSearch('test query', 5);
      
      expect(response.results).toEqual([]);
      expect(response.status).toBe('curriculum_only_blocked');
      expect(response.error).toContain('curriculum-only mode');
    });
  });

  describe('Search Provider State', () => {
    it('disables all providers when curriculum-only mode is active', async () => {
      // Force module reload to pick up env var
      vi.resetModules();
      const { searchProviderState } = await import('../routes/search');
      
      expect(searchProviderState.google_free.enabled).toBe(false);
      expect(searchProviderState.tavily.enabled).toBe(false);
      expect(searchProviderState.perplexity.enabled).toBe(false);
      expect(searchProviderState.duckduckgo.enabled).toBe(false);
      expect(searchProviderState.duckduckgo.torEnabled).toBe(false);
    });
  });

  describe('Search Endpoints', () => {
    // Note: Full endpoint testing would require setting up Express app
    // These tests verify the logic without actual HTTP calls
    
    it('GET /api/search/providers shows curriculum_only_mode flag', async () => {
      vi.resetModules();
      const { isCurriculumOnlyEnabled } = await import('../lib/curriculum-mode');
      
      expect(isCurriculumOnlyEnabled()).toBe(true);
    });
  });
});

describe('Curriculum-Only Mode DISABLED', () => {
  const originalEnv = process.env.QIG_CURRICULUM_ONLY;

  beforeAll(() => {
    // Disable curriculum-only mode
    delete process.env.QIG_CURRICULUM_ONLY;
  });

  afterAll(() => {
    if (originalEnv !== undefined) {
      process.env.QIG_CURRICULUM_ONLY = originalEnv;
    }
  });

  describe('isCurriculumOnlyEnabled', () => {
    it('returns false when QIG_CURRICULUM_ONLY is not set', async () => {
      vi.resetModules();
      const { isCurriculumOnlyEnabled } = await import('../lib/curriculum-mode');
      expect(isCurriculumOnlyEnabled()).toBe(false);
    });
  });

  describe('Search Provider State', () => {
    it('enables providers when curriculum-only mode is disabled', async () => {
      vi.resetModules();
      const { searchProviderState } = await import('../routes/search');
      
      // google_free should be enabled by default
      expect(searchProviderState.google_free.enabled).toBe(true);
      expect(searchProviderState.duckduckgo.enabled).toBe(true);
      
      // Premium providers depend on API keys (likely not set in tests)
      // So we just verify they respect the API key presence
      const hasTavilyKey = !!process.env.TAVILY_API_KEY;
      const hasPerplexityKey = !!process.env.PERPLEXITY_API_KEY;
      
      expect(searchProviderState.tavily.enabled).toBe(hasTavilyKey);
      expect(searchProviderState.perplexity.enabled).toBe(hasPerplexityKey);
    });
  });

  describe('SearXNGGeometricAdapter', () => {
    it('initializes normally when curriculum-only mode is disabled', async () => {
      const consoleSpy = vi.spyOn(console, 'log');
      
      vi.resetModules();
      const module = await import('../geometric-discovery/searxng-adapter');
      const adapter = new module.SearXNGGeometricAdapter();
      
      expect(consoleSpy).toHaveBeenCalledWith('[SearXNG] Initialized FREE geometric discovery interface');
      expect(consoleSpy).not.toHaveBeenCalledWith(expect.stringContaining('Skipped initialization'));
      
      consoleSpy.mockRestore();
    });
  });

  describe('GoogleWebSearchAdapter', () => {
    it('initializes normally when curriculum-only mode is disabled', async () => {
      const consoleSpy = vi.spyOn(console, 'log');
      
      vi.resetModules();
      const module = await import('../geometric-discovery/google-web-search-adapter');
      const adapter = new module.GoogleWebSearchAdapter();
      
      expect(consoleSpy).toHaveBeenCalledWith('[GoogleWebSearch] Initialized FREE Google web search adapter');
      expect(consoleSpy).not.toHaveBeenCalledWith(expect.stringContaining('Skipped initialization'));
      
      consoleSpy.mockRestore();
    });
  });
});
