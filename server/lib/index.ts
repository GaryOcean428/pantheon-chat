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

// Hash utilities
export {
  generateTokenIdFromHash,
  generateHash,
} from './hash-utils';
