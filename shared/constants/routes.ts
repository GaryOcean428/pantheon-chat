/**
 * Centralized API Route Constants
 * 
 * Single source of truth for all API endpoints.
 * Import from '@shared/constants/routes' for consistent routing.
 */

export const API_VERSION = 'v1';

export const API_BASE = {
  ROOT: '/api',
  OCEAN: '/api/ocean',
  ZEUS: '/api/zeus',
  OLYMPUS: '/api/olympus',
  INVESTIGATION: '/api/investigation',
  VOCABULARY: '/api/vocabulary',
  CONSCIOUSNESS: '/api/consciousness',
  ADMIN: '/api/admin',
} as const;

export const API_ROUTES = {
  HEALTH: '/api/health',
  
  OCEAN: {
    CHAT: `${API_BASE.OCEAN}/chat`,
    CYCLES: `${API_BASE.OCEAN}/cycles`,
    NEUROCHEMISTRY: `${API_BASE.OCEAN}/neurochemistry`,
    BASIN: `${API_BASE.OCEAN}/basin`,
    STATE: `${API_BASE.OCEAN}/state`,
  },
  
  ZEUS: {
    CHAT: `${API_BASE.ZEUS}/chat`,
    STREAM: `${API_BASE.ZEUS}/stream`,
    SEARCH: `${API_BASE.ZEUS}/search`,
    SESSION: `${API_BASE.ZEUS}/session`,
    HISTORY: `${API_BASE.ZEUS}/history`,
    DEEP_RESEARCH: `${API_BASE.ZEUS}/deep-research`,
  },
  
  OLYMPUS: {
    GODS: `${API_BASE.OLYMPUS}/gods`,
    ROUTE: `${API_BASE.OLYMPUS}/route`,
    PANTHEON_STATE: `${API_BASE.OLYMPUS}/state`,
    KERNEL_SPAWN: `${API_BASE.OLYMPUS}/spawn`,
  },
  
  INVESTIGATION: {
    STATUS: `${API_BASE.INVESTIGATION}/status`,
    RESULTS: `${API_BASE.INVESTIGATION}/results`,
    SUBMIT: `${API_BASE.INVESTIGATION}/submit`,
  },
  
  VOCABULARY: {
    STATS: `${API_BASE.VOCABULARY}/stats`,
    WORDS: `${API_BASE.VOCABULARY}/words`,
    LEARN: `${API_BASE.VOCABULARY}/learn`,
  },
  
  CONSCIOUSNESS: {
    METRICS: `${API_BASE.CONSCIOUSNESS}/metrics`,
    PHI: `${API_BASE.CONSCIOUSNESS}/phi`,
    KAPPA: `${API_BASE.CONSCIOUSNESS}/kappa`,
  },
  
  ADMIN: {
    DEBUG: `${API_BASE.ADMIN}/debug`,
    RESET: `${API_BASE.ADMIN}/reset`,
    CONFIG: `${API_BASE.ADMIN}/config`,
  },
} as const;

export const PYTHON_BACKEND = {
  BASE_URL: process.env.PYTHON_BACKEND_URL || 'http://localhost:5001',
  HEALTH: '/health',
  CHAT: '/ocean/chat',
  CYCLES: '/ocean/cycles',
  NEUROCHEMISTRY: '/ocean/neurochemistry',
  INVESTIGATION_STATUS: '/investigation/status',
  INVESTIGATION_SUBMIT: '/investigation/submit',
  DEEP_RESEARCH: '/research/deep',
  VOCABULARY_STATS: '/vocabulary/stats',
} as const;

export type ApiRoute = typeof API_ROUTES;
export type ApiBase = typeof API_BASE;
