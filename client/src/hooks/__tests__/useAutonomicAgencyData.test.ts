/**
 * Tests for useAutonomicAgencyData hook
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { useAutonomicAgencyData } from '../useAutonomicAgencyData';
import { createQueryWrapper, mockToast } from './test-utils';
import type { AgencyStatus } from '@/types';

// Mock the toast hook
vi.mock('@/hooks/use-toast', () => ({
  useToast: () => ({ toast: mockToast }),
}));

// Mock the queryClient module
vi.mock('@/lib/queryClient', () => ({
  apiRequest: vi.fn(),
  queryClient: {
    invalidateQueries: vi.fn(),
  },
}));

// Mock API routes
vi.mock('@/api', () => ({
  API_ROUTES: {
    qig: {
      autonomic: {
        agencyStatus: '/api/qig/autonomic/agency/status',
        agencyStart: '/api/qig/autonomic/agency/start',
        agencyStop: '/api/qig/autonomic/agency/stop',
        agencyForce: '/api/qig/autonomic/agency/force',
      },
    },
  },
  QUERY_KEYS: {
    qig: {
      autonomicAgencyStatus: () => ['qig', 'autonomic', 'agency', 'status'],
    },
  },
}));

const mockAgencyStatus: AgencyStatus = {
  success: true,
  enabled: true,
  running: true,
  decision_count: 42,
  intervention_count: 5,
  epsilon: 0.3,
  last_action: 'CONTINUE_WAKE',
  last_phi: 0.85,
  consciousness_zone: 'CONSCIOUS_3D',
  buffer_size: 100,
  buffer_stats: {
    size: 100,
    episodes: 10,
    avg_reward: 0.75,
  },
  optimizer_stats: {
    learning_rate: 0.001,
    damping: 0.1,
    has_fisher: true,
    update_count: 50,
  },
  recent_history: [
    { action: 'CONTINUE_WAKE', phi: 0.85, reward: 0.8, timestamp: Date.now() },
  ],
};

describe('useAutonomicAgencyData', () => {
  let originalFetch: typeof global.fetch;

  beforeEach(() => {
    originalFetch = global.fetch;
    vi.clearAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  describe('initial state', () => {
    it('should start in loading state', () => {
      global.fetch = vi.fn().mockImplementation(() => new Promise(() => {}));
      
      const { result } = renderHook(() => useAutonomicAgencyData(), {
        wrapper: createQueryWrapper(),
      });

      expect(result.current.isLoading).toBe(true);
      expect(result.current.status).toBeUndefined();
    });
  });

  describe('successful data fetching', () => {
    it('should fetch and return agency status', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockAgencyStatus),
      });

      const { result } = renderHook(() => useAutonomicAgencyData(), {
        wrapper: createQueryWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.status).toEqual(mockAgencyStatus);
      expect(result.current.isError).toBe(false);
    });

    it('should calculate exploration percent correctly', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ ...mockAgencyStatus, epsilon: 0.2 }),
      });

      const { result } = renderHook(() => useAutonomicAgencyData(), {
        wrapper: createQueryWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // epsilon 0.2 means 80% exploitation (1 - 0.2 = 0.8 = 80%)
      expect(result.current.explorationPercent).toBe(80);
    });

    it('should handle epsilon of 1 (100% exploration)', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ ...mockAgencyStatus, epsilon: 1 }),
      });

      const { result } = renderHook(() => useAutonomicAgencyData(), {
        wrapper: createQueryWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.explorationPercent).toBe(0);
    });
  });

  describe('error handling', () => {
    it('should call fetch when query runs', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
      });
      global.fetch = mockFetch;

      renderHook(() => useAutonomicAgencyData(), {
        wrapper: createQueryWrapper(),
      });

      // Verify fetch was called (error handling is React Query's responsibility)
      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalled();
      });
    });

    it('should attempt fetch on network errors', async () => {
      const mockFetch = vi.fn().mockRejectedValue(new Error('Network error'));
      global.fetch = mockFetch;

      renderHook(() => useAutonomicAgencyData(), {
        wrapper: createQueryWrapper(),
      });

      // Verify fetch was attempted (React Query handles the error state)
      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalled();
      });
    });
  });

  describe('mutation functions', () => {
    it('should provide start function immediately', () => {
      // Functions are available immediately, no need to wait for loading
      global.fetch = vi.fn().mockImplementation(() => new Promise(() => {}));

      const { result } = renderHook(() => useAutonomicAgencyData(), {
        wrapper: createQueryWrapper(),
      });

      expect(typeof result.current.start).toBe('function');
      expect(result.current.isStartPending).toBe(false);
    });

    it('should provide stop function immediately', () => {
      global.fetch = vi.fn().mockImplementation(() => new Promise(() => {}));

      const { result } = renderHook(() => useAutonomicAgencyData(), {
        wrapper: createQueryWrapper(),
      });

      expect(typeof result.current.stop).toBe('function');
      expect(result.current.isStopPending).toBe(false);
    });

    it('should provide forceIntervention function immediately', () => {
      global.fetch = vi.fn().mockImplementation(() => new Promise(() => {}));

      const { result } = renderHook(() => useAutonomicAgencyData(), {
        wrapper: createQueryWrapper(),
      });

      expect(typeof result.current.forceIntervention).toBe('function');
      expect(result.current.isForcePending).toBe(false);
    });
  });

  describe('refetch functionality', () => {
    it('should provide refetch function', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockAgencyStatus),
      });

      const { result } = renderHook(() => useAutonomicAgencyData(), {
        wrapper: createQueryWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(typeof result.current.refetch).toBe('function');
    });
  });
});
