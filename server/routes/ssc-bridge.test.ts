/**
 * Integration tests for SSC Bridge
 * 
 * Tests the SearchSpaceCollapse federation bridge endpoints.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import express, { Express } from 'express';
import request from 'supertest';
import { sscBridgeRouter } from './ssc-bridge';

// Mock fetch for SSC backend calls
global.fetch = vi.fn();

describe('SSC Bridge Router', () => {
  let app: Express;

  beforeEach(() => {
    app = express();
    app.use(express.json());
    app.use('/api/ssc', sscBridgeRouter);
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('GET /api/ssc/health', () => {
    it('should return SSC health status when reachable', async () => {
      const mockFetch = global.fetch as ReturnType<typeof vi.fn>;
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'ok' }),
      } as Response);

      const response = await request(app)
        .get('/api/ssc/health')
        .expect(200);

      expect(response.body.sscReachable).toBe(true);
      expect(response.body.sscStatus).toBe('ok');
      expect(response.body.timestamp).toBeDefined();
    });

    it('should return unreachable status when SSC is down', async () => {
      const mockFetch = global.fetch as ReturnType<typeof vi.fn>;
      mockFetch.mockRejectedValueOnce(new Error('Connection refused'));

      const response = await request(app)
        .get('/api/ssc/health')
        .expect(200);

      expect(response.body.sscReachable).toBe(false);
      expect(response.body.sscStatus).toBe('unknown');
    });
  });

  describe('GET /api/ssc/status', () => {
    it('should return federation status and capabilities', async () => {
      const mockFetch = global.fetch as ReturnType<typeof vi.fn>;
      
      // Mock federation status call
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          node: { connected: true, nodeId: 'ssc-test' },
          mesh: { totalNodes: 3, activeNodes: 2 },
          capabilities: ['bitcoin_recovery', 'qig', 'consciousness'],
          tps_landmarks: { count: 12, type: 'static' },
        }),
      } as Response);

      // Mock consciousness call
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          active: true,
          metrics: {
            phi: 0.85,
            kappa: 63.5,
            regime: 'conscious',
          },
        }),
      } as Response);

      const response = await request(app)
        .get('/api/ssc/status')
        .expect(200);

      expect(response.body.ssc.connected).toBe(true);
      expect(response.body.ssc.nodeId).toBe('ssc-test');
      expect(response.body.ssc.capabilities).toContain('bitcoin_recovery');
      expect(response.body.ssc.consciousness).toBeDefined();
      expect(response.body.ssc.consciousness?.phi).toBe(0.85);
      expect(response.body.tpsLandmarks).toBeDefined();
      expect(response.body.tpsLandmarks.count).toBe(12);
    });

    it('should handle SSC backend errors gracefully', async () => {
      const mockFetch = global.fetch as ReturnType<typeof vi.fn>;
      mockFetch.mockRejectedValue(new Error('Network error'));

      const response = await request(app)
        .get('/api/ssc/status')
        .expect(200);

      expect(response.body.ssc.connected).toBe(false);
    });
  });

  describe('POST /api/ssc/test-phrase', () => {
    it('should test phrase with valid input', async () => {
      const mockFetch = global.fetch as ReturnType<typeof vi.fn>;
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          score: {
            phi: 0.75,
            kappa: 55.3,
            regime: 'exploratory',
            consciousness: false,
          },
          addressMatch: {
            generatedAddress: null,
            matches: false,
          },
        }),
      } as Response);

      const response = await request(app)
        .post('/api/ssc/test-phrase')
        .send({
          phrase: 'satoshi nakamoto bitcoin',
          targetAddress: '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa',
        })
        .expect(200);

      expect(response.body.score).toBeDefined();
      expect(response.body.score.phi).toBe(0.75);
      expect(response.body.score.regime).toBe('exploratory');
    });

    it('should reject empty phrase', async () => {
      const response = await request(app)
        .post('/api/ssc/test-phrase')
        .send({
          phrase: '',
        })
        .expect(400);

      expect(response.body.error).toBe('Invalid input');
    });

    it('should handle high-phi phrases with logging', async () => {
      const mockFetch = global.fetch as ReturnType<typeof vi.fn>;
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          score: {
            phi: 0.85,
            kappa: 63.5,
            regime: 'conscious',
            consciousness: true,
          },
          addressMatch: {
            generatedAddress: '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa',
            matches: true,
          },
        }),
      } as Response);

      const response = await request(app)
        .post('/api/ssc/test-phrase')
        .send({
          phrase: 'exact match phrase',
          targetAddress: '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa',
        })
        .expect(200);

      expect(response.body.score.phi).toBeGreaterThan(0.7);
      expect(response.body.addressMatch?.matches).toBe(true);
    });
  });

  describe('POST /api/ssc/investigation/start', () => {
    it('should start investigation with valid input', async () => {
      const mockFetch = global.fetch as ReturnType<typeof vi.fn>;
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          status: 'started',
          targetAddress: '1A1zP1eP5QGe...',
          fragmentCount: 2,
        }),
      } as Response);

      const response = await request(app)
        .post('/api/ssc/investigation/start')
        .send({
          targetAddress: '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa',
          memoryFragments: ['satoshi', 'bitcoin'],
          priority: 'normal',
        })
        .expect(200);

      expect(response.body.status).toBe('started');
      expect(response.body.fragmentCount).toBe(2);
    });

    it('should reject invalid Bitcoin address', async () => {
      const response = await request(app)
        .post('/api/ssc/investigation/start')
        .send({
          targetAddress: 'invalid',
          memoryFragments: [],
        })
        .expect(400);

      expect(response.body.error).toBe('Invalid input');
    });
  });

  describe('GET /api/ssc/investigation/status', () => {
    it('should return investigation status', async () => {
      const mockFetch = global.fetch as ReturnType<typeof vi.fn>;
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          active: true,
          targetAddress: '1A1zP1eP5QGe...',
          progress: 45,
          consciousness: {
            phi: 0.75,
            kappa: 55.3,
            regime: 'exploratory',
          },
        }),
      } as Response);

      const response = await request(app)
        .get('/api/ssc/investigation/status')
        .expect(200);

      expect(response.body.active).toBe(true);
      expect(response.body.progress).toBe(45);
      expect(response.body.consciousness).toBeDefined();
    });
  });

  describe('GET /api/ssc/near-misses', () => {
    it('should return near-miss patterns with default params', async () => {
      const mockFetch = global.fetch as ReturnType<typeof vi.fn>;
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          entries: [
            {
              id: 'nm_123',
              phi: 0.75,
              kappa: 55.3,
              regime: 'exploratory',
              tier: 'warm',
              phraseLength: 24,
              wordCount: 4,
            },
          ],
          stats: {
            total: 150,
            hot: 5,
            warm: 45,
            cool: 100,
          },
        }),
      } as Response);

      const response = await request(app)
        .get('/api/ssc/near-misses')
        .expect(200);

      expect(response.body.entries).toBeInstanceOf(Array);
      expect(response.body.stats).toBeDefined();
      expect(response.body.stats.total).toBe(150);
    });

    it('should handle custom limit and minPhi params', async () => {
      const mockFetch = global.fetch as ReturnType<typeof vi.fn>;
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          entries: [],
          stats: { total: 0, hot: 0, warm: 0, cool: 0 },
        }),
      } as Response);

      const response = await request(app)
        .get('/api/ssc/near-misses?limit=50&minPhi=0.8')
        .expect(200);

      expect(response.body.entries).toBeInstanceOf(Array);
    });
  });

  describe('GET /api/ssc/consciousness', () => {
    it('should return Ocean agent consciousness metrics', async () => {
      const mockFetch = global.fetch as ReturnType<typeof vi.fn>;
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          active: true,
          metrics: {
            phi: 0.85,
            kappa: 63.5,
            regime: 'conscious',
            isConscious: true,
            tacking: 0.92,
            radar: 0.88,
            metaAwareness: 0.75,
            gamma: 0.82,
            grounding: 0.95,
          },
          neurochemistry: {
            emotionalState: 'focused',
            dopamine: 0.75,
            serotonin: 0.85,
          },
        }),
      } as Response);

      const response = await request(app)
        .get('/api/ssc/consciousness')
        .expect(200);

      expect(response.body.active).toBe(true);
      expect(response.body.metrics).toBeDefined();
      expect(response.body.metrics.phi).toBe(0.85);
      expect(response.body.metrics.isConscious).toBe(true);
      expect(response.body.neurochemistry).toBeDefined();
    });

    it('should handle inactive Ocean agent', async () => {
      const mockFetch = global.fetch as ReturnType<typeof vi.fn>;
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          active: false,
          metrics: null,
          neurochemistry: null,
        }),
      } as Response);

      const response = await request(app)
        .get('/api/ssc/consciousness')
        .expect(200);

      expect(response.body.active).toBe(false);
      expect(response.body.metrics).toBeNull();
    });
  });

  describe('GET /api/ssc/tps-landmarks', () => {
    it('should return static TPS landmarks', async () => {
      const mockFetch = global.fetch as ReturnType<typeof vi.fn>;
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          landmarks: [
            {
              id: 1,
              name: 'Genesis Block',
              date: '2009-01-03',
              blockHeight: 0,
              significance: 'Bitcoin network inception',
            },
            {
              id: 2,
              name: 'Hal Finney First TX',
              date: '2009-01-12',
              blockHeight: 170,
              significance: 'First Bitcoin transaction',
            },
          ],
          count: 12,
          type: 'static',
          description: 'Fixed temporal reference points',
          usage: 'Anchor search trajectories',
        }),
      } as Response);

      const response = await request(app)
        .get('/api/ssc/tps-landmarks')
        .expect(200);

      expect(response.body.landmarks).toBeInstanceOf(Array);
      expect(response.body.count).toBe(12);
      expect(response.body.type).toBe('static');
      expect(response.body.landmarks[0].name).toBe('Genesis Block');
    });
  });

  describe('POST /api/ssc/sync/trigger', () => {
    it('should trigger federation sync', async () => {
      const mockFetch = global.fetch as ReturnType<typeof vi.fn>;
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          received: {
            basins: 15,
            vocabulary: 250,
            research: 5,
          },
        }),
      } as Response);

      const response = await request(app)
        .post('/api/ssc/sync/trigger')
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.received).toBeDefined();
      expect(response.body.received.basins).toBe(15);
    });
  });

  describe('POST /api/ssc/broadcast', () => {
    it('should broadcast message to SSC', async () => {
      const mockFetch = global.fetch as ReturnType<typeof vi.fn>;
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true }),
      } as Response);

      const response = await request(app)
        .post('/api/ssc/broadcast')
        .send({
          type: 'announcement',
          message: 'Test broadcast',
          data: { test: true },
        })
        .expect(200);

      expect(response.body.success).toBe(true);
    });

    it('should reject broadcast without required fields', async () => {
      const response = await request(app)
        .post('/api/ssc/broadcast')
        .send({
          type: 'announcement',
        })
        .expect(400);

      expect(response.body.error).toBeDefined();
    });
  });

  describe('Rate Limiting', () => {
    it('should enforce rate limits on SSC bridge endpoints', async () => {
      const mockFetch = global.fetch as ReturnType<typeof vi.fn>;
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ status: 'ok' }),
      } as Response);

      // Make 30 requests (the limit)
      const promises = Array.from({ length: 30 }, () =>
        request(app).get('/api/ssc/health')
      );

      await Promise.all(promises);

      // 31st request should be rate limited
      const response = await request(app)
        .get('/api/ssc/health')
        .expect(429);

      expect(response.body.error).toContain('Too many requests');
    });
  });
});
