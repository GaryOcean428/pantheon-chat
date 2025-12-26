/**
 * Search Form - query input and configuration controls
 */

import React from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Input,
  Button,
  Slider,
  Checkbox,
  Label,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui';
import { Search, Settings2 } from 'lucide-react';
import { SEARCH_CONSTANTS, WORD_COUNT_OPTIONS } from './constants';
import type { SearchFormState } from './types';

interface SearchFormProps {
  formState: SearchFormState;
  onFormChange: (updates: Partial<SearchFormState>) => void;
  onSearch: () => void;
  isSearching: boolean;
  showAdvanced: boolean;
  onToggleAdvanced: () => void;
}

export function SearchForm({
  formState,
  onFormChange,
  onSearch,
  isSearching,
  showAdvanced,
  onToggleAdvanced,
}: SearchFormProps) {
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSearch();
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <Search className="h-5 w-5" />
            Memory Search
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={onToggleAdvanced}
            className="text-muted-foreground"
          >
            <Settings2 className="h-4 w-4 mr-1" />
            {showAdvanced ? 'Hide' : 'Show'} Options
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="flex gap-2">
            <Input
              placeholder="Search memory fragments..."
              value={formState.query}
              onChange={(e) => onFormChange({ query: e.target.value })}
              className="flex-1"
            />
            <Button type="submit" disabled={isSearching || !formState.query.trim()}>
              {isSearching ? 'Searching...' : 'Search'}
            </Button>
          </div>

          {showAdvanced && (
            <div className="space-y-4 pt-4 border-t">
              <div className="space-y-2">
                <Label className="text-sm">
                  Confidence Threshold: {(formState.confidence * SEARCH_CONSTANTS.PERCENT_MULTIPLIER).toFixed(0)}%
                </Label>
                <Slider
                  value={[formState.confidence]}
                  onValueChange={([value]) => onFormChange({ confidence: value })}
                  min={SEARCH_CONSTANTS.CONFIDENCE_MIN}
                  max={SEARCH_CONSTANTS.CONFIDENCE_MAX}
                  step={SEARCH_CONSTANTS.CONFIDENCE_STEP}
                />
              </div>

              <div className="space-y-2">
                <Label className="text-sm">
                  Max Candidates: {formState.maxCandidates.toLocaleString()}
                </Label>
                <Slider
                  value={[formState.maxCandidates]}
                  onValueChange={([value]) => onFormChange({ maxCandidates: value })}
                  min={SEARCH_CONSTANTS.CANDIDATES_MIN}
                  max={SEARCH_CONSTANTS.CANDIDATES_MAX}
                  step={SEARCH_CONSTANTS.CANDIDATES_STEP}
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label className="text-sm">Min Words</Label>
                  <Select
                    value={formState.minWords.toString()}
                    onValueChange={(v) => onFormChange({ minWords: parseInt(v, 10) })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {WORD_COUNT_OPTIONS.map((n) => (
                        <SelectItem key={n} value={n.toString()}>
                          {n}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label className="text-sm">Max Words</Label>
                  <Select
                    value={formState.maxWords.toString()}
                    onValueChange={(v) => onFormChange({ maxWords: parseInt(v, 10) })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {WORD_COUNT_OPTIONS.map((n) => (
                        <SelectItem key={n} value={n.toString()}>
                          {n}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="flex gap-6">
                <div className="flex items-center gap-2">
                  <Checkbox
                    id="geometric"
                    checked={formState.useGeometric}
                    onCheckedChange={(checked) =>
                      onFormChange({ useGeometric: checked === true })
                    }
                  />
                  <Label htmlFor="geometric" className="text-sm">
                    Geometric Search
                  </Label>
                </div>

                <div className="flex items-center gap-2">
                  <Checkbox
                    id="emotional"
                    checked={formState.useEmotional}
                    onCheckedChange={(checked) =>
                      onFormChange({ useEmotional: checked === true })
                    }
                  />
                  <Label htmlFor="emotional" className="text-sm">
                    Emotional Valence
                  </Label>
                </div>
              </div>
            </div>
          )}
        </form>
      </CardContent>
    </Card>
  );
}
