/**
 * Redis Cache Layer
 * 
 * Provides caching for expensive operations like Python backend calls,
 * geometric calculations, and vocabulary lookups.
 * 
 * Usage:
 *   import { cache, withCache } from './lib/cache';
 *   
 *   // Direct cache operations
 *   await cache.set('key', { data: 'value' }, 300); // 5 min TTL
 *   const cached = await cache.get<MyType>('key');
 *   
 *   // Function wrapper
 *   const cachedFn = withCache('prefix', originalFn, 300);
 */

import { createClient, RedisClientType } from 'redis';
import { createChildLogger } from './logger';

const logger = createChildLogger('Cache');

// Default TTLs in seconds
export const CACHE_TTL = {
  SHORT: 60,          // 1 minute
  MEDIUM: 300,        // 5 minutes
  LONG: 1800,         // 30 minutes
  VERY_LONG: 3600,    // 1 hour
  PYTHON_RESPONSE: 300,  // 5 minutes for Python backend responses
  VOCABULARY: 1800,      // 30 minutes for vocabulary lookups
  GEOMETRIC: 600,        // 10 minutes for geometric calculations
} as const;

class CacheManager {
  private client: RedisClientType | null = null;
  private isConnected = false;
  private connectionPromise: Promise<void> | null = null;
  private memoryFallback: Map<string, { value: string; expiry: number }> = new Map();

  /**
   * Initialize Redis connection.
   * Falls back to in-memory cache if Redis is not available.
   */
  async connect(): Promise<void> {
    if (this.connectionPromise) {
      return this.connectionPromise;
    }

    this.connectionPromise = this.doConnect();
    return this.connectionPromise;
  }

  private async doConnect(): Promise<void> {
    const redisUrl = process.env.REDIS_URL || 'redis://localhost:6379';
    
    try {
      this.client = createClient({ url: redisUrl });
      
      this.client.on('error', (err) => {
        logger.warn({ error: err.message }, 'Redis client error, using memory fallback');
        this.isConnected = false;
      });

      this.client.on('connect', () => {
        logger.info('Redis connected');
        this.isConnected = true;
      });

      this.client.on('reconnecting', () => {
        logger.debug('Redis reconnecting...');
      });

      await this.client.connect();
      this.isConnected = true;
      logger.info({ url: redisUrl.replace(/\/\/.*@/, '//*****@') }, 'Redis cache initialized');
    } catch (error) {
      logger.warn({ error }, 'Redis not available, using in-memory fallback');
      this.isConnected = false;
    }
  }

  /**
   * Get a value from cache.
   * @param key - Cache key
   * @returns Cached value or null if not found/expired
   */
  async get<T>(key: string): Promise<T | null> {
    try {
      if (this.isConnected && this.client) {
        const value = await this.client.get(key);
        if (value) {
          logger.trace({ key }, 'Cache hit');
          return JSON.parse(value) as T;
        }
      } else {
        // Memory fallback
        const entry = this.memoryFallback.get(key);
        if (entry && entry.expiry > Date.now()) {
          logger.trace({ key }, 'Memory cache hit');
          return JSON.parse(entry.value) as T;
        }
        this.memoryFallback.delete(key);
      }
    } catch (error) {
      logger.debug({ key, error }, 'Cache get error');
    }
    
    logger.trace({ key }, 'Cache miss');
    return null;
  }

  /**
   * Set a value in cache.
   * @param key - Cache key
   * @param value - Value to cache (will be JSON serialized)
   * @param ttlSeconds - Time to live in seconds
   */
  async set<T>(key: string, value: T, ttlSeconds: number = CACHE_TTL.MEDIUM): Promise<void> {
    try {
      const serialized = JSON.stringify(value);
      
      if (this.isConnected && this.client) {
        await this.client.setEx(key, ttlSeconds, serialized);
      } else {
        // Memory fallback
        this.memoryFallback.set(key, {
          value: serialized,
          expiry: Date.now() + (ttlSeconds * 1000),
        });
        
        // Clean up old entries periodically
        if (this.memoryFallback.size > 1000) {
          this.cleanupMemoryCache();
        }
      }
      
      logger.trace({ key, ttl: ttlSeconds }, 'Cache set');
    } catch (error) {
      logger.debug({ key, error }, 'Cache set error');
    }
  }

  /**
   * Delete a value from cache.
   * @param key - Cache key
   */
  async delete(key: string): Promise<void> {
    try {
      if (this.isConnected && this.client) {
        await this.client.del(key);
      } else {
        this.memoryFallback.delete(key);
      }
      logger.trace({ key }, 'Cache delete');
    } catch (error) {
      logger.debug({ key, error }, 'Cache delete error');
    }
  }

  /**
   * Delete all keys matching a pattern.
   * @param pattern - Pattern to match (e.g., 'python:*')
   */
  async deletePattern(pattern: string): Promise<void> {
    try {
      if (this.isConnected && this.client) {
        const keys = await this.client.keys(pattern);
        if (keys.length > 0) {
          await this.client.del(keys);
          logger.debug({ pattern, count: keys.length }, 'Cache pattern delete');
        }
      } else {
        // Memory fallback - convert Redis pattern to regex
        const regex = new RegExp('^' + pattern.replace(/\*/g, '.*') + '$');
        for (const key of this.memoryFallback.keys()) {
          if (regex.test(key)) {
            this.memoryFallback.delete(key);
          }
        }
      }
    } catch (error) {
      logger.debug({ pattern, error }, 'Cache pattern delete error');
    }
  }

  /**
   * Check if cache is connected to Redis.
   */
  isRedisConnected(): boolean {
    return this.isConnected;
  }

  private cleanupMemoryCache(): void {
    const now = Date.now();
    for (const [key, entry] of this.memoryFallback.entries()) {
      if (entry.expiry <= now) {
        this.memoryFallback.delete(key);
      }
    }
  }

  /**
   * Gracefully close the cache connection.
   */
  async close(): Promise<void> {
    if (this.client && this.isConnected) {
      await this.client.quit();
      this.isConnected = false;
      logger.info('Redis connection closed');
    }
  }
}

// Singleton cache instance
export const cache = new CacheManager();

/**
 * Create a cache key from components.
 * @param prefix - Cache key prefix (e.g., 'python', 'vocab')
 * @param parts - Key parts to join
 */
export function cacheKey(prefix: string, ...parts: (string | number)[]): string {
  return `${prefix}:${parts.join(':')}`;
}

/**
 * Wrap a function with caching.
 * @param prefix - Cache key prefix
 * @param fn - Function to wrap
 * @param ttlSeconds - Cache TTL
 * @param keyFn - Function to generate cache key from arguments
 */
export function withCache<TArgs extends unknown[], TResult>(
  prefix: string,
  fn: (...args: TArgs) => Promise<TResult>,
  ttlSeconds: number = CACHE_TTL.MEDIUM,
  keyFn: (...args: TArgs) => string = (...args) => JSON.stringify(args)
): (...args: TArgs) => Promise<TResult> {
  return async (...args: TArgs): Promise<TResult> => {
    const key = cacheKey(prefix, keyFn(...args));
    
    // Try to get from cache
    const cached = await cache.get<TResult>(key);
    if (cached !== null) {
      return cached;
    }
    
    // Execute function and cache result
    const result = await fn(...args);
    await cache.set(key, result, ttlSeconds);
    return result;
  };
}

/**
 * Decorator for caching class methods.
 * Usage:
 *   @Cached('prefix', 300)
 *   async myMethod(arg: string): Promise<Result> { ... }
 */
export function Cached(prefix: string, ttlSeconds: number = CACHE_TTL.MEDIUM) {
  return function (
    target: object,
    propertyKey: string,
    descriptor: PropertyDescriptor
  ) {
    const originalMethod = descriptor.value;
    
    descriptor.value = async function (...args: unknown[]) {
      const key = cacheKey(prefix, propertyKey, JSON.stringify(args));
      
      const cached = await cache.get(key);
      if (cached !== null) {
        return cached;
      }
      
      const result = await originalMethod.apply(this, args);
      await cache.set(key, result, ttlSeconds);
      return result;
    };
    
    return descriptor;
  };
}

export default cache;
