/**
 * Lazy-loaded BIP39 Wordlist
 * 
 * Provides lazy initialization of the BIP39 wordlist to avoid loading
 * the entire wordlist on application startup.
 * 
 * Usage:
 *   import { getWordlist, isValidWord, getWordIndex } from './lib/lazy-wordlist';
 *   
 *   const wordlist = await getWordlist();
 *   const isValid = await isValidWord('abandon');
 */

import { createChildLogger } from './logger';

const logger = createChildLogger('Wordlist');

// Cached wordlist and set for O(1) lookups
let cachedWordlist: string[] | null = null;
let cachedWordSet: Set<string> | null = null;
let loadPromise: Promise<string[]> | null = null;

/**
 * Lazily load and cache the BIP39 wordlist.
 * Only loads from disk on first access.
 */
export async function getWordlist(): Promise<string[]> {
  if (cachedWordlist) {
    return cachedWordlist;
  }
  
  if (loadPromise) {
    return loadPromise;
  }
  
  loadPromise = loadWordlist();
  return loadPromise;
}

async function loadWordlist(): Promise<string[]> {
  const startTime = Date.now();
  
  try {
    // Dynamic import to avoid loading at startup
    const bip39 = await import('bip39');
    cachedWordlist = bip39.wordlists.english;
    cachedWordSet = new Set(cachedWordlist);
    
    logger.debug({ 
      count: cachedWordlist.length,
      loadTime: `${Date.now() - startTime}ms` 
    }, 'BIP39 wordlist loaded');
    
    return cachedWordlist;
  } catch (error) {
    logger.error({ error }, 'Failed to load BIP39 wordlist');
    throw error;
  }
}

/**
 * Check if a word is in the BIP39 wordlist.
 * Uses O(1) Set lookup for performance.
 */
export async function isValidWord(word: string): Promise<boolean> {
  if (!cachedWordSet) {
    await getWordlist();
  }
  return cachedWordSet!.has(word.toLowerCase());
}

/**
 * Get the index of a word in the BIP39 wordlist.
 * Returns -1 if not found.
 */
export async function getWordIndex(word: string): Promise<number> {
  const wordlist = await getWordlist();
  return wordlist.indexOf(word.toLowerCase());
}

/**
 * Get a word by its index in the BIP39 wordlist.
 * Returns undefined if index is out of range.
 */
export async function getWordByIndex(index: number): Promise<string | undefined> {
  const wordlist = await getWordlist();
  return wordlist[index];
}

/**
 * Check if the wordlist has been loaded.
 */
export function isWordlistLoaded(): boolean {
  return cachedWordlist !== null;
}

/**
 * Get wordlist size without loading it.
 * Returns 0 if not loaded, otherwise returns the size.
 */
export function getWordlistSize(): number {
  return cachedWordlist?.length ?? 0;
}

/**
 * Preload the wordlist in the background.
 * Call this during application startup if you want to warm the cache.
 */
export function preloadWordlist(): void {
  getWordlist().catch((error) => {
    logger.warn({ error }, 'Failed to preload wordlist');
  });
}
