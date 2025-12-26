/**
 * Constants for Tool Factory Dashboard
 */

import type { ToolCategory, ToolStatus } from './types';

export const TOOL_CATEGORIES: { value: ToolCategory; label: string; icon: string }[] = [
  { value: 'search', label: 'Search', icon: '🔍' },
  { value: 'analysis', label: 'Analysis', icon: '📊' },
  { value: 'generation', label: 'Generation', icon: '✨' },
  { value: 'transformation', label: 'Transformation', icon: '🔄' },
  { value: 'integration', label: 'Integration', icon: '🔗' },
  { value: 'utility', label: 'Utility', icon: '🛠️' },
];

export const TOOL_STATUSES: { value: ToolStatus; label: string; color: string }[] = [
  { value: 'active', label: 'Active', color: 'bg-green-500' },
  { value: 'inactive', label: 'Inactive', color: 'bg-gray-500' },
  { value: 'draft', label: 'Draft', color: 'bg-yellow-500' },
  { value: 'deprecated', label: 'Deprecated', color: 'bg-red-500' },
];

export const PARAMETER_TYPES = [
  { value: 'string', label: 'String' },
  { value: 'number', label: 'Number' },
  { value: 'boolean', label: 'Boolean' },
  { value: 'array', label: 'Array' },
  { value: 'object', label: 'Object' },
] as const;

export const DASHBOARD_CONSTANTS = {
  PERCENT_MULTIPLIER: 100,
  DECIMAL_PLACES: 1,
  DEFAULT_PAGE_SIZE: 12,
  MAX_TAGS_DISPLAY: 3,
  DEBOUNCE_MS: 300,
  REFETCH_INTERVAL: 30000,
} as const;
