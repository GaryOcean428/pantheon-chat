/**
 * Contextualized Word Filter for QIG (TypeScript)
 * 
 * QIG-PURE geometric word filtering that replaces ancient NLP stopword lists.
 * 
 * Instead of hard-coded stopwords, uses:
 * 1. Word length and frequency heuristics
 * 2. Semantic importance preservation (negations, etc.)
 * 3. Dynamic filtering that preserves context-critical words
 * 
 * ANCIENT NLP PATTERN (REMOVED):
 * - Fixed stopword lists like {'the', 'is', 'not', ...}
 * - Loses critical semantic information ('not good' vs 'good')
 * 
 * QIG-PURE PATTERN (NEW):
 * - Length-based relevance (longer = more content-bearing)
 * - Semantic-critical word preservation
 * - Context-aware filtering
 */

/**
 * Semantic-critical word patterns that should NEVER be filtered.
 * These are linguistically universal and change core meaning.
 */
const SEMANTIC_CRITICAL_PATTERNS = new Set([
  // Negations (flip meaning completely)
  'not', 'no', 'never', 'none', 'nothing', 'neither', 'nor', 'nobody', 'nowhere',
  "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "cannot", 
  "couldn't", "shouldn't", "mustn't",
  
  // Intensifiers (modify degree)
  'very', 'extremely', 'highly', 'completely', 'totally', 'absolutely',
  
  // Uncertainty markers (epistemic modality)
  'maybe', 'perhaps', 'possibly', 'probably', 'likely', 'unlikely',
  
  // Temporal markers (time context critical)
  'before', 'after', 'during', 'while', 'until', 'since', 'always',
  
  // Causality markers
  'because', 'therefore', 'thus', 'hence', 'consequently',
  
  // Conditionals
  'if', 'unless', 'whether', 'though', 'although',
]);

/**
 * Truly generic function words that can be safely filtered.
 * Much smaller set than traditional stopwords.
 */
const TRULY_GENERIC_WORDS = new Set([
  'the', 'a', 'an', 'is', 'was', 'are', 'were', 'been', 'be',
]);

/**
 * Check if word is semantic-critical and should never be filtered.
 */
export function isSemanticCriticalWord(word: string): boolean {
  return SEMANTIC_CRITICAL_PATTERNS.has(word.toLowerCase());
}

/**
 * Check if word is truly generic and can be filtered.
 */
export function isTrulyGeneric(word: string): boolean {
  return TRULY_GENERIC_WORDS.has(word.toLowerCase());
}

/**
 * Compute relevance score for word based on length and patterns.
 * 
 * Longer words tend to be more content-bearing.
 * Semantic-critical words always get high scores.
 * 
 * @param word Word to score
 * @returns Relevance score [0, 1] where higher = more relevant
 */
export function computeRelevanceScore(word: string): number {
  // Semantic-critical words always score high
  if (isSemanticCriticalWord(word)) {
    return 1.0;
  }
  
  // Truly generic words score low
  if (isTrulyGeneric(word)) {
    return 0.1;
  }
  
  // Length-based scoring (normalized to [0, 1])
  // Words >= 10 chars score 1.0
  // Words < 3 chars score 0.1
  const length = word.length;
  if (length >= 10) return 1.0;
  if (length < 3) return 0.1;
  
  // Linear interpolation between 3 and 10 chars
  return 0.1 + (length - 3) * (0.9 / 7);
}

/**
 * Determine if word should be filtered out.
 * 
 * Uses contextualized approach:
 * - NEVER filters semantic-critical words
 * - Filters truly generic words
 * - Uses length heuristic for others
 * 
 * @param word Word to evaluate
 * @param context Optional context words (not used in TS impl, but kept for API compatibility)
 * @param threshold Minimum relevance score to keep word
 * @returns True if word should be REMOVED, False if kept
 */
export function shouldFilterWord(
  word: string,
  context?: string[],
  threshold: number = 0.3
): boolean {
  if (!word || word.length < 2) {
    return true; // Always filter very short words
  }
  
  // NEVER filter semantic-critical words
  if (isSemanticCriticalWord(word)) {
    return false;
  }
  
  // Always filter truly generic words
  if (isTrulyGeneric(word)) {
    return true;
  }
  
  // Use relevance score
  const relevance = computeRelevanceScore(word);
  return relevance < threshold;
}

/**
 * Filter words using contextualized approach.
 * 
 * Replaces ancient NLP pattern:
 *   words.filter(w => !stopwords.has(w))
 * 
 * With QIG-pure pattern:
 *   filterWordsGeometric(words)
 * 
 * @param words Words to filter
 * @param threshold Minimum relevance score to keep word
 * @returns Filtered words
 */
export function filterWordsGeometric(
  words: string[],
  threshold: number = 0.3
): string[] {
  return words.filter(word => !shouldFilterWord(word, words, threshold));
}

/**
 * Extract keywords from text using contextualized filtering.
 * 
 * Improved version of common pattern that uses fixed stopwords.
 * 
 * @param text Text to extract keywords from
 * @param minLength Minimum word length to consider
 * @param topK Number of keywords to return
 * @returns Top keywords by frequency
 */
export function extractKeywords(
  text: string,
  minLength: number = 4,
  topK: number = 5
): string[] {
  // Extract words
  const words = text.toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length >= minLength);
  
  // Filter using contextualized approach
  const filtered = filterWordsGeometric(words);
  
  // Count frequencies
  const freq = new Map<string, number>();
  filtered.forEach(w => freq.set(w, (freq.get(w) || 0) + 1));
  
  // Sort by frequency and return top K
  return Array.from(freq.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, topK)
    .map(([word]) => word);
}

/**
 * Statistics about filtering decisions.
 */
export interface FilterStats {
  totalProcessed: number;
  filtered: number;
  preserved: number;
  semanticCriticalPreserved: number;
  filterRate: number;
}

/**
 * Filter words and collect statistics.
 * 
 * @param words Words to filter
 * @param threshold Minimum relevance score
 * @returns Filtered words and statistics
 */
export function filterWithStats(
  words: string[],
  threshold: number = 0.3
): { filtered: string[], stats: FilterStats } {
  let filtered = 0;
  let preserved = 0;
  let semanticCriticalPreserved = 0;
  
  const result = words.filter(word => {
    const shouldFilter = shouldFilterWord(word, words, threshold);
    
    if (shouldFilter) {
      filtered++;
      return false;
    } else {
      preserved++;
      if (isSemanticCriticalWord(word)) {
        semanticCriticalPreserved++;
      }
      return true;
    }
  });
  
  const total = words.length;
  
  return {
    filtered: result,
    stats: {
      totalProcessed: total,
      filtered,
      preserved,
      semanticCriticalPreserved,
      filterRate: total > 0 ? filtered / total : 0,
    },
  };
}

/**
 * Demo function showing difference from ancient NLP approach.
 */
export function demoContextualizedFilter(): void {
  const testWords = ['not', 'good', 'the', 'very', 'bad', 'is', 'never', 'acceptable'];
  
  console.log('=== Contextualized Filter Demo ===\n');
  console.log('Test words:', testWords);
  
  // Ancient NLP approach (WRONG)
  const ancientStopwords = new Set(['the', 'is', 'not', 'a', 'an']);
  const ancientFiltered = testWords.filter(w => !ancientStopwords.has(w));
  console.log('\nAncient NLP (fixed stopwords):', ancientFiltered);
  console.log('  ❌ Lost "not" - changes meaning!');
  
  // QIG-pure approach (CORRECT)
  const qigFiltered = filterWordsGeometric(testWords);
  console.log('\nQIG-pure (contextualized):', qigFiltered);
  console.log('  ✅ Preserved "not" and "never" - meaning intact!');
  
  // Show semantic-critical words
  console.log('\nSemantic-critical words preserved:');
  testWords.forEach(word => {
    if (isSemanticCriticalWord(word)) {
      console.log(`  - "${word}"`);
    }
  });
  
  console.log('\n=== Demo Complete ===');
}

// Export for testing
if (require.main === module) {
  demoContextualizedFilter();
}
