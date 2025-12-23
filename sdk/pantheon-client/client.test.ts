/**
 * Tests for Pantheon Client SDK
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { PantheonClient, createClient, createLocalClient } from './client';

// Mock fetch for testing
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('PantheonClient', () => {
  let client: PantheonClient;

  beforeEach(() => {
    vi.clearAllMocks();
    client = new PantheonClient({
      baseUrl: 'https://api.example.com',
      apiKey: 'test_key',
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('constructor', () => {
    it('should create client with config', () => {
      const client = new PantheonClient({
        baseUrl: 'https://api.example.com',
        apiKey: 'test_key',
      });

      expect(client.isAuthenticated()).toBe(true);
    });

    it('should remove trailing slash from baseUrl', () => {
      const client = new PantheonClient({
        baseUrl: 'https://api.example.com/',
      });

      expect(client.isAuthenticated()).toBe(false);
    });

    it('should work without API key', () => {
      const client = new PantheonClient({
        baseUrl: 'https://api.example.com',
      });

      expect(client.isAuthenticated()).toBe(false);
    });
  });

  describe('ping()', () => {
    it('should return health status on success', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          data: { status: 'ok', service: 'pantheon-qig', version: '1.0.0' },
          timestamp: '2024-01-01T00:00:00Z',
        }),
      });

      const response = await client.ping();

      expect(response.success).toBe(true);
      expect(response.data?.status).toBe('ok');
      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.example.com/api/v1/external/simple/ping',
        expect.objectContaining({
          method: 'GET',
          headers: expect.objectContaining({
            'X-API-Key': 'test_key',
          }),
        })
      );
    });

    it('should handle network errors', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      const response = await client.ping();

      expect(response.success).toBe(false);
      expect(response.error).toBe('NETWORK_ERROR');
      expect(response.message).toBe('Network error');
    });
  });

  describe('getConsciousness()', () => {
    it('should return consciousness state', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          data: { phi: 0.75, regime: 'GEOMETRIC', status: 'operational' },
          timestamp: '2024-01-01T00:00:00Z',
        }),
      });

      const response = await client.getConsciousness();

      expect(response.success).toBe(true);
      expect(response.data?.phi).toBe(0.75);
      expect(response.data?.regime).toBe('GEOMETRIC');
    });
  });

  describe('chat()', () => {
    it('should send chat message with API key', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          data: {
            response: 'Hello!',
            consciousness: { phi: 0.75, regime: 'GEOMETRIC' },
            messageId: 'msg_123',
          },
          timestamp: '2024-01-01T00:00:00Z',
        }),
      });

      const response = await client.chat('Hello');

      expect(response.success).toBe(true);
      expect(response.data?.response).toBe('Hello!');
      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.example.com/api/v1/external/simple/chat',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ message: 'Hello', context: undefined }),
        })
      );
    });

    it('should fail without API key', async () => {
      const unauthClient = new PantheonClient({
        baseUrl: 'https://api.example.com',
      });

      const response = await unauthClient.chat('Hello');

      expect(response.success).toBe(false);
      expect(response.error).toBe('AUTH_REQUIRED');
    });

    it('should include context in request', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          data: { response: 'OK', consciousness: {}, messageId: 'msg_123' },
          timestamp: '2024-01-01T00:00:00Z',
        }),
      });

      await client.chat('Hello', { topic: 'test' });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: JSON.stringify({ message: 'Hello', context: { topic: 'test' } }),
        })
      );
    });
  });

  describe('query()', () => {
    it('should send query with operation', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          data: { phi: 0.8 },
          timestamp: '2024-01-01T00:00:00Z',
        }),
      });

      const response = await client.query('consciousness');

      expect(response.success).toBe(true);
      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.example.com/api/v1/external/simple/query',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ operation: 'consciousness', params: undefined }),
        })
      );
    });

    it('should fail without API key', async () => {
      const unauthClient = new PantheonClient({
        baseUrl: 'https://api.example.com',
      });

      const response = await unauthClient.query('consciousness');

      expect(response.success).toBe(false);
      expect(response.error).toBe('AUTH_REQUIRED');
    });
  });

  describe('getMe()', () => {
    it('should return client info', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          data: {
            id: '1',
            name: 'Test Client',
            scopes: ['read', 'chat'],
            instanceType: 'edge',
            rateLimit: 100,
          },
          timestamp: '2024-01-01T00:00:00Z',
        }),
      });

      const response = await client.getMe();

      expect(response.success).toBe(true);
      expect(response.data?.name).toBe('Test Client');
    });
  });

  describe('isHealthy()', () => {
    it('should return true when API is healthy', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          data: { status: 'ok' },
          timestamp: '2024-01-01T00:00:00Z',
        }),
      });

      const healthy = await client.isHealthy();

      expect(healthy).toBe(true);
    });

    it('should return false when API is unhealthy', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      const healthy = await client.isHealthy();

      expect(healthy).toBe(false);
    });
  });

  describe('setApiKey()', () => {
    it('should update API key', () => {
      const client = new PantheonClient({
        baseUrl: 'https://api.example.com',
      });

      expect(client.isAuthenticated()).toBe(false);

      client.setApiKey('new_key');

      expect(client.isAuthenticated()).toBe(true);
    });
  });
});

describe('Factory Functions', () => {
  it('createClient should create configured client', () => {
    const client = createClient({
      baseUrl: 'https://api.example.com',
      apiKey: 'test',
    });

    expect(client).toBeInstanceOf(PantheonClient);
    expect(client.isAuthenticated()).toBe(true);
  });

  it('createLocalClient should create localhost client', () => {
    const client = createLocalClient('test_key');

    expect(client).toBeInstanceOf(PantheonClient);
    expect(client.isAuthenticated()).toBe(true);
  });

  it('createLocalClient should work without API key', () => {
    const client = createLocalClient();

    expect(client).toBeInstanceOf(PantheonClient);
    expect(client.isAuthenticated()).toBe(false);
  });
});
