/**
 * Test utilities for React hooks
 * 
 * Provides QueryClient wrapper and mock utilities for testing hooks.
 */

import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { vi } from 'vitest';

/**
 * Create a fresh QueryClient for testing
 */
export function createTestQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: Infinity,
      },
      mutations: {
        retry: false,
      },
    },
  });
}

/**
 * Wrapper component for testing hooks that use React Query
 */
export function createQueryWrapper() {
  const queryClient = createTestQueryClient();
  return function QueryWrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    );
  };
}

/**
 * Mock toast hook
 */
export const mockToast = vi.fn();
export const mockUseToast = () => ({ toast: mockToast });

/**
 * Mock fetch response helper
 */
export function mockFetchResponse<T>(data: T, options?: { ok?: boolean; status?: number }) {
  return vi.fn().mockResolvedValue({
    ok: options?.ok ?? true,
    status: options?.status ?? 200,
    json: () => Promise.resolve(data),
  });
}

/**
 * Mock fetch error helper
 */
export function mockFetchError(message: string) {
  return vi.fn().mockRejectedValue(new Error(message));
}

/**
 * Wait for async operations to complete
 */
export async function waitForAsync(ms = 0): Promise<void> {
  await new Promise(resolve => setTimeout(resolve, ms));
}
