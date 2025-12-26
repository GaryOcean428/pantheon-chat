/**
 * Tool Filters - search, category, and status filters
 */

import React from 'react';
import {
  Input,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Button,
} from '@/components/ui';
import { Search, Grid, List, Plus } from 'lucide-react';
import { TOOL_CATEGORIES, TOOL_STATUSES } from '../constants';
import type { ViewMode } from '../types';

interface ToolFiltersProps {
  searchQuery: string;
  selectedCategory: string | null;
  selectedStatus: string | null;
  viewMode: ViewMode;
  onSearchChange: (query: string) => void;
  onCategoryChange: (category: string | null) => void;
  onStatusChange: (status: string | null) => void;
  onViewModeChange: (mode: ViewMode) => void;
  onCreateClick: () => void;
}

export function ToolFilters({
  searchQuery,
  selectedCategory,
  selectedStatus,
  viewMode,
  onSearchChange,
  onCategoryChange,
  onStatusChange,
  onViewModeChange,
  onCreateClick,
}: ToolFiltersProps) {
  return (
    <div className="flex flex-col sm:flex-row gap-4">
      <div className="relative flex-1">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          placeholder="Search tools..."
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          className="pl-9"
        />
      </div>

      <Select
        value={selectedCategory ?? 'all'}
        onValueChange={(v) => onCategoryChange(v === 'all' ? null : v)}
      >
        <SelectTrigger className="w-[160px]">
          <SelectValue placeholder="Category" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All Categories</SelectItem>
          {TOOL_CATEGORIES.map((cat) => (
            <SelectItem key={cat.value} value={cat.value}>
              {cat.icon} {cat.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      <Select
        value={selectedStatus ?? 'all'}
        onValueChange={(v) => onStatusChange(v === 'all' ? null : v)}
      >
        <SelectTrigger className="w-[140px]">
          <SelectValue placeholder="Status" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All Statuses</SelectItem>
          {TOOL_STATUSES.map((status) => (
            <SelectItem key={status.value} value={status.value}>
              {status.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      <div className="flex gap-2">
        <Button
          variant={viewMode === 'grid' ? 'default' : 'outline'}
          size="icon"
          onClick={() => onViewModeChange('grid')}
        >
          <Grid className="h-4 w-4" />
        </Button>
        <Button
          variant={viewMode === 'list' ? 'default' : 'outline'}
          size="icon"
          onClick={() => onViewModeChange('list')}
        >
          <List className="h-4 w-4" />
        </Button>
      </div>

      <Button onClick={onCreateClick}>
        <Plus className="h-4 w-4 mr-2" />
        New Tool
      </Button>
    </div>
  );
}
