/**
 * Structured Logger
 * 
 * Provides consistent, structured logging throughout the server.
 * Uses pino for high-performance JSON logging with log levels.
 * 
 * Usage:
 *   import { logger, createChildLogger } from './lib/logger';
 *   
 *   logger.info({ userId: '123' }, 'User logged in');
 *   
 *   const childLogger = createChildLogger('OceanAgent');
 *   childLogger.debug({ phi: 0.8 }, 'Processing hypothesis');
 */

import pino from 'pino';

// Log level from environment, defaults to 'info' in production, 'debug' in development
const level = process.env.LOG_LEVEL || (process.env.NODE_ENV === 'production' ? 'info' : 'debug');

// Base logger configuration
const baseConfig: pino.LoggerOptions = {
  level,
  // Use pretty print in development
  transport: process.env.NODE_ENV !== 'production' ? {
    target: 'pino-pretty',
    options: {
      colorize: true,
      translateTime: 'SYS:standard',
      ignore: 'pid,hostname',
    },
  } : undefined,
  // Base context included in all logs
  base: {
    env: process.env.NODE_ENV || 'development',
  },
  // Timestamp format
  timestamp: pino.stdTimeFunctions.isoTime,
  // Redact sensitive fields
  redact: {
    paths: ['password', 'secret', 'token', 'apiKey', 'authorization', '*.password', '*.secret', '*.token'],
    censor: '[REDACTED]',
  },
};

// Create the base logger
export const logger = pino(baseConfig);

/**
 * Create a child logger with a specific context name.
 * Child loggers inherit the parent's configuration but add context.
 * 
 * @param context - The context name (e.g., 'OceanAgent', 'AuthMiddleware')
 * @returns A child logger with the context attached
 */
export function createChildLogger(context: string): pino.Logger {
  return logger.child({ context });
}

/**
 * Log levels available:
 * - trace: Very detailed debugging info
 * - debug: Debugging info
 * - info: General operational info
 * - warn: Warning conditions
 * - error: Error conditions
 * - fatal: Critical errors causing shutdown
 */
export type LogLevel = 'trace' | 'debug' | 'info' | 'warn' | 'error' | 'fatal';

/**
 * Request logger middleware for Express.
 * Logs incoming requests and response times.
 */
export function createRequestLogger() {
  const requestLogger = createChildLogger('HTTP');
  
  return (req: { method: string; url: string; ip?: string }, res: { statusCode: number; on: (event: string, cb: () => void) => void }, next: () => void) => {
    const start = Date.now();
    
    res.on('finish', () => {
      const duration = Date.now() - start;
      const logData = {
        method: req.method,
        url: req.url,
        status: res.statusCode,
        duration: `${duration}ms`,
        ip: req.ip,
      };
      
      if (res.statusCode >= 500) {
        requestLogger.error(logData, 'Request failed');
      } else if (res.statusCode >= 400) {
        requestLogger.warn(logData, 'Request client error');
      } else {
        requestLogger.info(logData, 'Request completed');
      }
    });
    
    next();
  };
}

// Pre-created loggers for common contexts
export const loggers = {
  ocean: createChildLogger('Ocean'),
  auth: createChildLogger('Auth'),
  db: createChildLogger('DB'),
  api: createChildLogger('API'),
  python: createChildLogger('Python'),
  consciousness: createChildLogger('Consciousness'),
  qig: createChildLogger('QIG'),
  telemetry: createChildLogger('Telemetry'),
  federation: createChildLogger('Federation'),
  observer: createChildLogger('Observer'),
};

export default logger;
