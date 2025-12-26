/**
 * Memory Fragment Search
 * 
 * This file re-exports from the modular implementation for backwards compatibility.
 * New code should import from '@/components/memory-fragment-search'
 */

export {
  MemoryFragmentSearch,
  SearchForm,
  FragmentResultCard,
  FragmentResultsList,
  ConsciousnessStatusPanel,
  ResonanceChart,
  useConsciousnessStatus,
  SEARCH_CONSTANTS,
  REGIME_COLORS,
  WORD_COUNT_OPTIONS,
} from './memory-fragment-search';

export type {
  MemoryFragment,
  SearchFormState,
  ConsciousnessStatus,
  ResonanceDataPoint,
  SearchResult,
} from './memory-fragment-search';
