/**
 * Redis Cache Layer for TypeScript
 * 
 * DISABLED: Redis is no longer used. PostgreSQL is the primary storage.
 * All functions return stub values to maintain compatibility with existing code.
 * 
 * QIG Purity: PostgreSQL is the sole data layer.
 */

// TTLs in seconds (kept for interface compatibility)
export const CACHE_TTL = {
  SHORT: 300,
  MEDIUM: 3600,
  LONG: 86400,
  PERMANENT: 86400 * 7,
};

// Key prefixes for namespacing (kept for interface compatibility)
export const CACHE_KEYS = {
  TESTED_PHRASE: 'tested:',
  BALANCE_HIT: 'hit:',
  VOCABULARY: 'vocab:',
  KERNEL_STATE: 'kernel:',
  SESSION: 'session:',
  AUTO_CYCLE: 'autocycle:state',
  NEAR_MISS: 'nearmiss:state',
  OCEAN_MEMORY: 'ocean:memory',
};

/**
 * Initialize Redis connection - DISABLED
 */
export function initRedis(): null {
  console.log('[Redis] DISABLED - using PostgreSQL as primary storage');
  return null;
}

/**
 * Get Redis client - DISABLED
 */
export function getRedis(): null {
  return null;
}

/**
 * Check if Redis is available - always false (DISABLED)
 */
export function isRedisAvailable(): boolean {
  return false;
}

/**
 * Set a value with optional TTL - DISABLED
 */
export async function cacheSet(
  _key: string,
  _value: unknown,
  _ttlSeconds: number = CACHE_TTL.MEDIUM
): Promise<boolean> {
  return false;
}

/**
 * Get a value - DISABLED
 */
export async function cacheGet<T>(_key: string): Promise<T | null> {
  return null;
}

/**
 * Check if key exists - DISABLED
 */
export async function cacheExists(_key: string): Promise<boolean> {
  return false;
}

/**
 * Delete a key - DISABLED
 */
export async function cacheDel(_key: string): Promise<boolean> {
  return false;
}

/**
 * Set multiple values in a hash - DISABLED
 */
export async function cacheHSet(
  _hashKey: string,
  _field: string,
  _value: unknown
): Promise<boolean> {
  return false;
}

/**
 * Get a field from a hash - DISABLED
 */
export async function cacheHGet<T>(_hashKey: string, _field: string): Promise<T | null> {
  return null;
}

/**
 * Check if field exists in hash - DISABLED
 */
export async function cacheHExists(_hashKey: string, _field: string): Promise<boolean> {
  return false;
}

/**
 * Increment a counter - DISABLED
 */
export async function cacheIncr(_key: string): Promise<number> {
  return 0;
}

/**
 * Get cache stats - DISABLED
 */
export async function getCacheStats(): Promise<{
  connected: boolean;
  keys?: number;
  memory?: string;
}> {
  return { connected: false };
}

/**
 * Pipeline multiple operations - DISABLED
 */
export function getPipeline(): null {
  return null;
}
