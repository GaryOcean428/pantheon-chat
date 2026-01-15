/**
 * Hash Utilities
 * 
 * Shared utilities for cryptographic hashing operations.
 */

import { createHash } from 'crypto';

/**
 * Generate a deterministic token ID from a string using SHA256 hash.
 * 
 * @param input - The input string to hash (typically a token/word)
 * @returns A positive integer token ID derived from the first 8 hex characters of the hash
 * 
 * @example
 * ```typescript
 * const tokenId = generateTokenIdFromHash('hello'); // e.g., 1335922160
 * ```
 */
export function generateTokenIdFromHash(input: string): number {
  return parseInt(
    createHash('sha256').update(input).digest('hex').slice(0, 8),
    16
  );
}

/**
 * Generate a deterministic hash string from input.
 * 
 * @param input - The input string to hash
 * @param algorithm - Hash algorithm to use (default: 'sha256')
 * @returns Hex-encoded hash string
 * 
 * @example
 * ```typescript
 * const hash = generateHash('hello'); // full SHA256 hash in hex
 * const shortHash = generateHash('hello').slice(0, 16); // first 16 chars
 * ```
 */
export function generateHash(input: string, algorithm: 'sha256' | 'md5' = 'sha256'): string {
  return createHash(algorithm).update(input).digest('hex');
}
