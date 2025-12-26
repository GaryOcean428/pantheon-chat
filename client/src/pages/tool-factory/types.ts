/**
 * Types for Tool Factory Dashboard
 */

export interface Tool {
  id: string;
  name: string;
  description: string;
  category: ToolCategory;
  status: ToolStatus;
  version: string;
  createdAt: string;
  updatedAt: string;
  usageCount: number;
  successRate: number;
  avgExecutionTime: number;
  parameters: ToolParameter[];
  tags: string[];
  author: string;
}

export type ToolCategory =
  | 'search'
  | 'analysis'
  | 'generation'
  | 'transformation'
  | 'integration'
  | 'utility';

export type ToolStatus = 'active' | 'inactive' | 'draft' | 'deprecated';

export interface ToolParameter {
  name: string;
  type: 'string' | 'number' | 'boolean' | 'array' | 'object';
  required: boolean;
  description: string;
  defaultValue?: unknown;
}

export interface ToolFormData {
  name: string;
  description: string;
  category: ToolCategory;
  parameters: ToolParameter[];
  tags: string[];
}

export interface ToolStats {
  totalTools: number;
  activeTools: number;
  totalUsage: number;
  avgSuccessRate: number;
}

export type ViewMode = 'grid' | 'list';
export type SortField = 'name' | 'usageCount' | 'successRate' | 'updatedAt';
export type SortDirection = 'asc' | 'desc';
