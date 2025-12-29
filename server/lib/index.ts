/**
 * Server Library Utilities
 * 
 * Centralized exports for server-side utilities.
 */

// Error handling utilities
export {
  // Type guards
  isError,
  isErrorWithCode,
  isHttpError,
  hasMessage,
  
  // Property extractors
  getErrorMessage,
  getErrorStack,
  getErrorCode,
  getHttpStatus,
  
  // Conversion
  toError,
  wrapError,
  
  // Route handlers
  handleRouteError,
  handleRouteErrorWithData,
  
  // Async helpers
  tryCatch,
  tryCatchSync,
  
  // Types
  type Result,
} from './error-utils';

// Logger utilities
export { logger, createChildLogger } from './logger';

// Cache utilities
export { cache, withCache, CACHE_TTL } from './cache';

// Lazy wordlist utilities
export { getWordlist, isValidWord, getWordIndex } from './lazy-wordlist';
