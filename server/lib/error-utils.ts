/**
 * Typed Error Handling Utilities
 * 
 * This module provides type-safe utilities for handling errors in catch blocks,
 * eliminating the need for 'catch (error: any)' patterns.
 * 
 * Usage:
 * ```typescript
 * import { getErrorMessage, handleRouteError } from './lib/error-utils';
 * 
 * // In try/catch blocks:
 * try {
 *   // ...
 * } catch (error: unknown) {
 *   console.error('Failed:', getErrorMessage(error));
 * }
 * 
 * // In Express route handlers:
 * try {
 *   // ...
 * } catch (error: unknown) {
 *   handleRouteError(res, error, 'MyRoute');
 * }
 * ```
 */

import type { Response } from 'express';

// =============================================================================
// Type Guards
// =============================================================================

/**
 * Check if a value is an Error instance
 */
export function isError(value: unknown): value is Error {
  return value instanceof Error;
}

/**
 * Check if an error has a 'code' property (common in Node.js errors)
 */
export function isErrorWithCode(error: unknown): error is Error & { code: string } {
  return isError(error) && typeof (error as Error & { code?: unknown }).code === 'string';
}

/**
 * Check if an error has a 'status' or 'statusCode' property (common in HTTP errors)
 */
export function isHttpError(error: unknown): error is Error & { status?: number; statusCode?: number } {
  if (!isError(error)) return false;
  const e = error as Error & { status?: unknown; statusCode?: unknown };
  return typeof e.status === 'number' || typeof e.statusCode === 'number';
}

/**
 * Check if a value has a message property
 */
export function hasMessage(value: unknown): value is { message: string } {
  return (
    typeof value === 'object' &&
    value !== null &&
    'message' in value &&
    typeof (value as { message: unknown }).message === 'string'
  );
}

// =============================================================================
// Error Property Extractors
// =============================================================================

/**
 * Safely extract an error message from any value.
 * Works with Error instances, objects with message property, strings, etc.
 */
export function getErrorMessage(error: unknown): string {
  // Error instance
  if (isError(error)) {
    return error.message;
  }
  
  // Object with message property
  if (hasMessage(error)) {
    return error.message;
  }
  
  // String error
  if (typeof error === 'string') {
    return error;
  }
  
  // Try to stringify
  try {
    return JSON.stringify(error);
  } catch {
    return 'Unknown error';
  }
}

/**
 * Safely extract a stack trace from an error.
 * Returns undefined if no stack is available.
 */
export function getErrorStack(error: unknown): string | undefined {
  if (isError(error)) {
    return error.stack;
  }
  return undefined;
}

/**
 * Safely extract an error code from an error.
 * Returns undefined if no code is available.
 */
export function getErrorCode(error: unknown): string | undefined {
  if (isErrorWithCode(error)) {
    return error.code;
  }
  return undefined;
}

/**
 * Get HTTP status code from an error, with a default fallback.
 */
export function getHttpStatus(error: unknown, defaultStatus: number = 500): number {
  if (isHttpError(error)) {
    return error.status ?? error.statusCode ?? defaultStatus;
  }
  return defaultStatus;
}

// =============================================================================
// Error Conversion
// =============================================================================

/**
 * Convert any value to an Error instance.
 * Useful when you need to rethrow or log with full Error semantics.
 */
export function toError(error: unknown): Error {
  if (isError(error)) {
    return error;
  }
  
  const message = getErrorMessage(error);
  const newError = new Error(message);
  
  // Preserve the original value for debugging
  (newError as Error & { originalError: unknown }).originalError = error;
  
  return newError;
}

/**
 * Wrap an error with additional context.
 */
export function wrapError(error: unknown, context: string): Error {
  const originalMessage = getErrorMessage(error);
  const wrappedError = new Error(`${context}: ${originalMessage}`);
  
  // Preserve original stack if available
  const originalStack = getErrorStack(error);
  if (originalStack) {
    wrappedError.stack = `${wrappedError.stack}\nCaused by: ${originalStack}`;
  }
  
  // Preserve original error
  (wrappedError as Error & { cause: unknown }).cause = error;
  
  return wrappedError;
}

// =============================================================================
// Express Route Error Handlers
// =============================================================================

/**
 * Standard error response handler for Express routes.
 * Sends a JSON error response with appropriate status code.
 * 
 * @param res - Express Response object
 * @param error - The caught error (unknown type)
 * @param context - Optional context string for logging (e.g., route name)
 * @param defaultStatus - Default HTTP status code (default: 500)
 */
export function handleRouteError(
  res: Response,
  error: unknown,
  context?: string,
  defaultStatus: number = 500
): void {
  const message = getErrorMessage(error);
  const status = getHttpStatus(error, defaultStatus);
  
  // Log the error with context
  if (context) {
    console.error(`[${context}] Error:`, message);
  } else {
    console.error('Route error:', message);
  }
  
  // Log stack in development
  if (process.env.NODE_ENV !== 'production') {
    const stack = getErrorStack(error);
    if (stack) {
      console.error(stack);
    }
  }
  
  // Send error response
  res.status(status).json({
    error: message,
    ...(context && { context }),
  });
}

/**
 * Handle route errors that should include additional data in the response.
 */
export function handleRouteErrorWithData(
  res: Response,
  error: unknown,
  data: Record<string, unknown>,
  context?: string,
  defaultStatus: number = 500
): void {
  const message = getErrorMessage(error);
  const status = getHttpStatus(error, defaultStatus);
  
  if (context) {
    console.error(`[${context}] Error:`, message);
  }
  
  res.status(status).json({
    error: message,
    ...data,
  });
}

// =============================================================================
// Async Error Wrapper
// =============================================================================

/**
 * Wrap an async function to catch errors and return a Result type.
 * Useful for avoiding try/catch in simple cases.
 */
export type Result<T, E = Error> = 
  | { success: true; data: T }
  | { success: false; error: E };

export async function tryCatch<T>(
  fn: () => Promise<T>
): Promise<Result<T, Error>> {
  try {
    const data = await fn();
    return { success: true, data };
  } catch (error: unknown) {
    return { success: false, error: toError(error) };
  }
}

/**
 * Synchronous version of tryCatch.
 */
export function tryCatchSync<T>(
  fn: () => T
): Result<T, Error> {
  try {
    const data = fn();
    return { success: true, data };
  } catch (error: unknown) {
    return { success: false, error: toError(error) };
  }
}
