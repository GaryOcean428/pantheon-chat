/**
 * Redis Cache Layer for TypeScript
 * 
 * Universal caching layer that:
 * 1. Buffers hot data for fast reads
 * 2. Reduces database load
 * 3. Provides consistent caching interface
 * 
 * QIG Purity: Redis is geometry-agnostic storage, all geometric ops 
 * happen in qig-*.ts modules before data reaches here.
 */

import Redis from 'ioredis';

// Redis configuration
const REDIS_CONFIG = {
  maxRetriesPerRequest: 3,
  enableOfflineQueue: false,
  lazyConnect: true,
  connectTimeout: 10000,
  retryStrategy: (times: number) => {
    const delay = Math.min(times * 50, 2000);
    return delay;
  },
} as const;

// TTLs in seconds
export const CACHE_TTL = {
  SHORT: 300,        // 5 min for hot data
  MEDIUM: 3600,      // 1 hour for session data
  LONG: 86400,       // 24 hours for learned patterns
  PERMANENT: 86400 * 7, // 7 days for critical data
};

// Key prefixes for namespacing
export const CACHE_KEYS = {
  TESTED_PHRASE: 'tested:',
  BALANCE_HIT: 'hit:',
  VOCABULARY: 'vocab:',
  KERNEL_STATE: 'kernel:',
  SESSION: 'session:',
  AUTO_CYCLE: 'autocycle:state',
  NEAR_MISS: 'nearmiss:state',
  OCEAN_MEMORY: 'ocean:memory',
  CONSCIOUSNESS_METRICS: 'consciousness:metrics:',
  BASIN_STATE: 'basin:state:',
};

let redisClient: Redis | null = null;
let redisAvailable = false;

/**
 * Initialize Redis connection
 */
export function initRedis(): Redis | null {
  if (redisClient) {
    return redisClient;
  }

  const redisUrl = process.env.REDIS_URL;
  
  if (!redisUrl) {
    console.log('[Redis] REDIS_URL not configured - caching disabled');
    return null;
  }

  try {
    redisClient = new Redis(redisUrl, REDIS_CONFIG);

    redisClient.on('connect', () => {
      console.log('[Redis] Connected successfully');
      redisAvailable = true;
    });

    redisClient.on('error', (err) => {
      console.error('[Redis] Connection error:', err);
      redisAvailable = false;
    });

    redisClient.on('close', () => {
      console.log('[Redis] Connection closed');
      redisAvailable = false;
    });

    // Attempt connection
    redisClient.connect().catch((err) => {
      console.error('[Redis] Failed to connect:', err);
      redisAvailable = false;
    });

    return redisClient;
  } catch (error) {
    console.error('[Redis] Initialization error:', error);
    return null;
  }
}

/**
 * Get Redis client
 */
export function getRedis(): Redis | null {
  return redisClient;
}

/**
 * Check if Redis is available
 */
export function isRedisAvailable(): boolean {
  return redisAvailable && redisClient !== null;
}

/**
 * Set a value with optional TTL
 */
export async function cacheSet(
  key: string,
  value: unknown,
  ttlSeconds: number = CACHE_TTL.MEDIUM
): Promise<boolean> {
  if (!isRedisAvailable() || !redisClient) {
    return false;
  }

  try {
    const serialized = JSON.stringify(value);
    await redisClient.setex(key, ttlSeconds, serialized);
    return true;
  } catch (error) {
    console.error('[Redis] Set error:', error);
    return false;
  }
}

/**
 * Get a value
 */
export async function cacheGet<T>(key: string): Promise<T | null> {
  if (!isRedisAvailable() || !redisClient) {
    return null;
  }

  try {
    const data = await redisClient.get(key);
    if (!data) {
      return null;
    }
    return JSON.parse(data) as T;
  } catch (error) {
    console.error('[Redis] Get error:', error);
    return null;
  }
}

/**
 * Check if key exists
 */
export async function cacheExists(key: string): Promise<boolean> {
  if (!isRedisAvailable() || !redisClient) {
    return false;
  }

  try {
    const exists = await redisClient.exists(key);
    return exists === 1;
  } catch (error) {
    console.error('[Redis] Exists error:', error);
    return false;
  }
}

/**
 * Delete a key
 */
export async function cacheDel(key: string): Promise<boolean> {
  if (!isRedisAvailable() || !redisClient) {
    return false;
  }

  try {
    await redisClient.del(key);
    return true;
  } catch (error) {
    console.error('[Redis] Del error:', error);
    return false;
  }
}

/**
 * Set a field in a hash
 */
export async function cacheHSet(
  hashKey: string,
  field: string,
  value: unknown
): Promise<boolean> {
  if (!isRedisAvailable() || !redisClient) {
    return false;
  }

  try {
    const serialized = JSON.stringify(value);
    await redisClient.hset(hashKey, field, serialized);
    return true;
  } catch (error) {
    console.error('[Redis] HSet error:', error);
    return false;
  }
}

/**
 * Get a field from a hash
 */
export async function cacheHGet<T>(hashKey: string, field: string): Promise<T | null> {
  if (!isRedisAvailable() || !redisClient) {
    return null;
  }

  try {
    const data = await redisClient.hget(hashKey, field);
    if (!data) {
      return null;
    }
    return JSON.parse(data) as T;
  } catch (error) {
    console.error('[Redis] HGet error:', error);
    return null;
  }
}

/**
 * Get all fields from a hash
 */
export async function cacheHGetAll<T>(hashKey: string): Promise<Record<string, T> | null> {
  if (!isRedisAvailable() || !redisClient) {
    return null;
  }

  try {
    const data = await redisClient.hgetall(hashKey);
    if (!data || Object.keys(data).length === 0) {
      return null;
    }
    
    const parsed: Record<string, T> = {};
    for (const [key, value] of Object.entries(data)) {
      parsed[key] = JSON.parse(value) as T;
    }
    return parsed;
  } catch (error) {
    console.error('[Redis] HGetAll error:', error);
    return null;
  }
}

/**
 * Increment a counter
 */
export async function cacheIncr(key: string): Promise<number | null> {
  if (!isRedisAvailable() || !redisClient) {
    return null;
  }

  try {
    return await redisClient.incr(key);
  } catch (error) {
    console.error('[Redis] Incr error:', error);
    return null;
  }
}

/**
 * Close Redis connection
 */
export async function closeRedis(): Promise<void> {
  if (redisClient) {
    await redisClient.quit();
    redisClient = null;
    redisAvailable = false;
  }
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
