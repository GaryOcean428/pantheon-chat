import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
  test: {
    globals: true,
    environment: 'jsdom',
    // Set NODE_ENV=test to skip Python backend connection attempts in tests
    env: {
      NODE_ENV: 'test',
    },
    include: ['server/**/*.test.ts', 'shared/**/*.test.ts', 'client/**/*.test.ts', 'client/**/*.test.tsx'],
    exclude: ['node_modules', '.cache'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
      include: ['server/**/*.ts', 'shared/**/*.ts', 'client/src/**/*.ts', 'client/src/**/*.tsx'],
      exclude: ['**/*.test.ts', '**/*.test.tsx', '**/node_modules/**'],
    },
    testTimeout: 30000,
    // Use node for server tests, jsdom is default for all
    environmentMatchGlobs: [
      ['server/**/*.test.ts', 'node'],
      ['shared/**/*.test.ts', 'node'],
    ],
  },
  resolve: {
    alias: {
      '@shared': path.resolve(__dirname, './shared'),
      '@': path.resolve(__dirname, './client/src'),
    },
  },
});
