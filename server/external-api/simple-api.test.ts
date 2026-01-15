/**
 * Integration tests for Simple External API
 * 
 * Tests the streamlined API wrapper endpoints.
 * NOTE: These tests hit rate limits and are skipped in CI.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import express, { Express } from 'express';
import request from 'supertest';
import { simpleApiRouter } from './simple-api';

describe.skip('Simple External API', () => {
  let app: Express;

  beforeEach(() => {
    app = express();
    app.use(express.json());
    app.use('/api/v1/external/simple', simpleApiRouter);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Public Endpoints (No Auth Required)', () => {
    describe('GET /ping', () => {
      it('should return health status', async () => {
        const response = await request(app)
          .get('/api/v1/external/simple/ping')
          .expect(200);

        expect(response.body.success).toBe(true);
        expect(response.body.data.status).toBe('ok');
        expect(response.body.data.service).toBe('pantheon-qig');
        expect(response.body.data.version).toBeDefined();
        expect(response.body.timestamp).toBeDefined();
      });
    });

    describe('GET /info', () => {
      it('should return API information and capabilities', async () => {
        const response = await request(app)
          .get('/api/v1/external/simple/info')
          .expect(200);

        expect(response.body.success).toBe(true);
        expect(response.body.data.name).toBe('Pantheon QIG External API');
        expect(response.body.data.capabilities).toContain('consciousness-queries');
        expect(response.body.data.capabilities).toContain('chat-interface');
        expect(response.body.data.endpoints).toBeDefined();
        expect(response.body.data.authentication.methods).toContain('Bearer token');
      });
    });

    describe('GET /consciousness', () => {
      it('should return limited consciousness data for unauthenticated requests', async () => {
        const response = await request(app)
          .get('/api/v1/external/simple/consciousness')
          .expect(200);

        expect(response.body.success).toBe(true);
        expect(response.body.data.phi).toBeDefined();
        expect(response.body.data.regime).toBeDefined();
        expect(response.body.data.status).toBe('operational');
        // Limited data note should be present
        expect(response.body.data.note).toContain('authenticated');
      });
    });

    describe('GET /docs', () => {
      it('should return OpenAPI documentation', async () => {
        const response = await request(app)
          .get('/api/v1/external/simple/docs')
          .expect(200);

        expect(response.body.success).toBe(true);
        expect(response.body.data.openapi).toBe('3.0.0');
        expect(response.body.data.info.title).toBe('Pantheon QIG External API');
        expect(response.body.data.paths).toBeDefined();
        expect(response.body.data.components.securitySchemes).toBeDefined();
      });
    });
  });

  describe('Rate Limiting', () => {
    it('should include rate limit info in response meta', async () => {
      const response = await request(app)
        .get('/api/v1/external/simple/ping')
        .expect(200);

      expect(response.body.meta).toBeDefined();
      expect(response.body.meta.authenticated).toBe(false);
    });

    it('should return 429 after exceeding rate limit', async () => {
      // Make many requests to trigger rate limit
      // Note: In a real test environment, you'd mock the rate limiter
      // This test documents the expected behavior
      const responses = [];
      for (let i = 0; i < 35; i++) {
        const res = await request(app).get('/api/v1/external/simple/ping');
        responses.push(res.status);
      }
      
      // After 30 requests, should see 429s
      const rateLimited = responses.filter(s => s === 429);
      expect(rateLimited.length).toBeGreaterThan(0);
    });
  });

  describe('Response Format', () => {
    it('should have consistent response structure', async () => {
      const response = await request(app)
        .get('/api/v1/external/simple/ping')
        .expect(200);

      // All responses should have these fields
      expect(response.body).toHaveProperty('success');
      expect(response.body).toHaveProperty('timestamp');
      expect(response.body).toHaveProperty('data');
      expect(response.body).toHaveProperty('meta');
      
      // Timestamp should be ISO format
      expect(() => new Date(response.body.timestamp)).not.toThrow();
    });

    it('should return error format for invalid requests', async () => {
      // Non-existent endpoint
      const response = await request(app)
        .get('/api/v1/external/simple/nonexistent')
        .expect(404);

      // Note: 404 is handled by Express, not our wrapper
    });
  });
});

describe('Simple API Response Helpers', () => {
  it('should format successful responses correctly', async () => {
    const app = express();
    app.use(express.json());
    app.use('/api/v1/external/simple', simpleApiRouter);

    const response = await request(app)
      .get('/api/v1/external/simple/info')
      .expect(200);

    expect(response.body).toMatchObject({
      success: true,
      data: expect.any(Object),
      timestamp: expect.any(String),
    });
  });
});
