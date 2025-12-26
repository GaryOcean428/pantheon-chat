/**
 * Fragment Result Card - displays a single memory fragment
 */

import React from 'react';
import { Card, CardContent, Badge } from '@/components/ui';
import { Clock, Sparkles, Heart } from 'lucide-react';
import { SEARCH_CONSTANTS } from './constants';
import type { MemoryFragment } from './types';

interface FragmentResultCardProps {
  fragment: MemoryFragment;
  onClick?: (fragment: MemoryFragment) => void;
}

export function FragmentResultCard({ fragment, onClick }: FragmentResultCardProps) {
  const resonancePercent = fragment.resonance * SEARCH_CONSTANTS.PERCENT_MULTIPLIER;
  const isHighResonance = fragment.resonance >= SEARCH_CONSTANTS.HIGH_RESONANCE;
  const isMediumResonance = fragment.resonance >= SEARCH_CONSTANTS.MEDIUM_RESONANCE;

  const truncatedContent =
    fragment.content.length > SEARCH_CONSTANTS.TRUNCATE_LENGTH
      ? `${fragment.content.slice(0, SEARCH_CONSTANTS.TRUNCATE_LENGTH)}...`
      : fragment.content;

  const visibleTags = fragment.tags.slice(0, SEARCH_CONSTANTS.MAX_TAGS_DISPLAY);
  const hiddenTagCount = fragment.tags.length - visibleTags.length;

  return (
    <Card
      className={`cursor-pointer transition-all hover:shadow-md ${
        isHighResonance ? 'border-primary/50' : ''
      }`}
      onClick={() => onClick?.(fragment)}
    >
      <CardContent className="p-4 space-y-3">
        <div className="flex items-start justify-between gap-2">
          <p className="text-sm flex-1">{truncatedContent}</p>
          <div className="flex items-center gap-1 shrink-0">
            <Sparkles
              className={`h-4 w-4 ${
                isHighResonance
                  ? 'text-yellow-500'
                  : isMediumResonance
                  ? 'text-blue-500'
                  : 'text-muted-foreground'
              }`}
            />
            <span className="text-xs font-mono">{resonancePercent.toFixed(0)}%</span>
          </div>
        </div>

        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <div className="flex items-center gap-1">
            <Clock className="h-3 w-3" />
            {new Date(fragment.timestamp).toLocaleString()}
          </div>
          {fragment.emotionalValence !== 0 && (
            <div className="flex items-center gap-1">
              <Heart
                className={`h-3 w-3 ${
                  fragment.emotionalValence > 0 ? 'text-red-400' : 'text-blue-400'
                }`}
              />
              <span>{fragment.emotionalValence > 0 ? '+' : ''}{fragment.emotionalValence.toFixed(2)}</span>
            </div>
          )}
        </div>

        {fragment.tags.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {visibleTags.map((tag) => (
              <Badge key={tag} variant="secondary" className="text-xs">
                {tag}
              </Badge>
            ))}
            {hiddenTagCount > 0 && (
              <Badge variant="outline" className="text-xs">
                +{hiddenTagCount} more
              </Badge>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
