/**
 * Fragment Results List - displays search results
 */

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, Badge } from '@/components/ui';
import { FileText, Loader2 } from 'lucide-react';
import { FragmentResultCard } from './FragmentResultCard';
import { SEARCH_CONSTANTS } from './constants';
import type { MemoryFragment, SearchResult } from './types';

interface FragmentResultsListProps {
  results: SearchResult | null;
  isLoading: boolean;
  onFragmentClick?: (fragment: MemoryFragment) => void;
}

export function FragmentResultsList({
  results,
  isLoading,
  onFragmentClick,
}: FragmentResultsListProps) {
  if (isLoading) {
    return (
      <Card>
        <CardContent className="p-8 flex items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }

  if (!results) {
    return (
      <Card>
        <CardContent className="p-8 text-center text-muted-foreground">
          <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>Enter a search query to find memory fragments</p>
        </CardContent>
      </Card>
    );
  }

  if (results.fragments.length === 0) {
    return (
      <Card>
        <CardContent className="p-8 text-center text-muted-foreground">
          <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>No fragments found matching your query</p>
        </CardContent>
      </Card>
    );
  }

  const visibleFragments = results.fragments.slice(0, SEARCH_CONSTANTS.MAX_VISIBLE_RESULTS);
  const hiddenCount = results.fragments.length - visibleFragments.length;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Results
          </span>
          <div className="flex items-center gap-2 text-sm font-normal">
            <Badge variant="secondary">
              {results.totalCount.toLocaleString()} found
            </Badge>
            <span className="text-muted-foreground">
              in {results.searchTime.toFixed(0)}ms
            </span>
            {results.geometricScore !== undefined && (
              <Badge variant="outline">
                Geo: {(results.geometricScore * SEARCH_CONSTANTS.PERCENT_MULTIPLIER).toFixed(0)}%
              </Badge>
            )}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {visibleFragments.map((fragment) => (
          <FragmentResultCard
            key={fragment.id}
            fragment={fragment}
            onClick={onFragmentClick}
          />
        ))}
        {hiddenCount > 0 && (
          <div className="text-center text-sm text-muted-foreground py-2">
            Showing {visibleFragments.length} of {results.totalCount} results
          </div>
        )}
      </CardContent>
    </Card>
  );
}
