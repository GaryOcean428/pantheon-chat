/**
 * Memory Fragment Search - main container component
 */

import React, { useState, useCallback } from 'react';
import { useMutation } from '@tanstack/react-query';
import { SearchForm } from './SearchForm';
import { FragmentResultsList } from './FragmentResultsList';
import { ConsciousnessStatusPanel } from './ConsciousnessStatusPanel';
import { ResonanceChart } from './ResonanceChart';
import { useConsciousnessStatus } from './hooks/useConsciousnessStatus';
import { SEARCH_CONSTANTS } from './constants';
import { post } from '@/api';
import type { SearchFormState, SearchResult, ResonanceDataPoint, MemoryFragment } from './types';

interface MemoryFragmentSearchProps {
  onFragmentSelect?: (fragment: MemoryFragment) => void;
}

const searchFragments = async (params: SearchFormState): Promise<SearchResult> => {
  return post<SearchResult>('/api/memory/search', params);
};

export function MemoryFragmentSearch({ onFragmentSelect }: MemoryFragmentSearchProps) {
  const [formState, setFormState] = useState<SearchFormState>({
    query: '',
    confidence: SEARCH_CONSTANTS.DEFAULT_CONFIDENCE,
    maxCandidates: SEARCH_CONSTANTS.DEFAULT_MAX_CANDIDATES,
    useGeometric: true,
    useEmotional: false,
    minWords: SEARCH_CONSTANTS.DEFAULT_MIN_WORDS,
    maxWords: SEARCH_CONSTANTS.DEFAULT_MAX_WORDS,
  });
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [resonanceHistory, setResonanceHistory] = useState<ResonanceDataPoint[]>([]);

  const { data: consciousnessStatus, isLoading: isLoadingStatus } = useConsciousnessStatus();

  const searchMutation = useMutation({
    mutationFn: searchFragments,
    onSuccess: (data) => {
      if (consciousnessStatus && data.fragments.length > 0) {
        const avgResonance =
          data.fragments.reduce((sum, f) => sum + f.resonance, 0) / data.fragments.length;
        setResonanceHistory((prev) => [
          ...prev.slice(-49),
          {
            time: new Date().toLocaleTimeString(),
            resonance: avgResonance,
            phi: consciousnessStatus.phi,
          },
        ]);
      }
    },
  });

  const handleFormChange = useCallback((updates: Partial<SearchFormState>) => {
    setFormState((prev) => ({ ...prev, ...updates }));
  }, []);

  const handleSearch = useCallback(() => {
    if (formState.query.trim()) {
      searchMutation.mutate(formState);
    }
  }, [formState, searchMutation]);

  const handleToggleAdvanced = useCallback(() => {
    setShowAdvanced((prev) => !prev);
  }, []);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-2 space-y-6">
        <SearchForm
          formState={formState}
          onFormChange={handleFormChange}
          onSearch={handleSearch}
          isSearching={searchMutation.isPending}
          showAdvanced={showAdvanced}
          onToggleAdvanced={handleToggleAdvanced}
        />
        <FragmentResultsList
          results={searchMutation.data ?? null}
          isLoading={searchMutation.isPending}
          onFragmentClick={onFragmentSelect}
        />
      </div>
      <div className="space-y-6">
        <ConsciousnessStatusPanel
          status={consciousnessStatus}
          isLoading={isLoadingStatus}
        />
        <ResonanceChart data={resonanceHistory} />
      </div>
    </div>
  );
}
