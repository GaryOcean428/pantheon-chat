/**
 * Tool List - displays tools in grid or list view
 */

import React from 'react';
import {
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui';
import { Loader2, FileX, ArrowUpDown } from 'lucide-react';
import { ToolCard } from './ToolCard';
import { TOOL_CATEGORIES, TOOL_STATUSES, DASHBOARD_CONSTANTS } from '../constants';
import type { Tool, ViewMode, SortField, SortDirection } from '../types';

interface ToolListProps {
  tools: Tool[];
  viewMode: ViewMode;
  isLoading: boolean;
  sortField: SortField;
  sortDirection: SortDirection;
  onToolClick: (tool: Tool) => void;
  onToolEdit: (tool: Tool) => void;
  onSort: (field: SortField) => void;
}

export function ToolList({
  tools,
  viewMode,
  isLoading,
  sortField,
  sortDirection,
  onToolClick,
  onToolEdit,
  onSort,
}: ToolListProps) {
  if (isLoading) {
    return (
      <Card>
        <CardContent className="p-12 flex items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }

  if (tools.length === 0) {
    return (
      <Card>
        <CardContent className="p-12 text-center text-muted-foreground">
          <FileX className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>No tools found matching your criteria</p>
        </CardContent>
      </Card>
    );
  }

  if (viewMode === 'grid') {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {tools.map((tool) => (
          <ToolCard
            key={tool.id}
            tool={tool}
            onClick={onToolClick}
            onEdit={onToolEdit}
          />
        ))}
      </div>
    );
  }

  const SortHeader = ({ field, children }: { field: SortField; children: React.ReactNode }) => (
    <TableHead
      className="cursor-pointer hover:bg-muted/50"
      onClick={() => onSort(field)}
    >
      <div className="flex items-center gap-1">
        {children}
        {sortField === field && (
          <ArrowUpDown
            className={`h-3 w-3 ${sortDirection === 'desc' ? 'rotate-180' : ''}`}
          />
        )}
      </div>
    </TableHead>
  );

  return (
    <Card>
      <Table>
        <TableHeader>
          <TableRow>
            <SortHeader field="name">Name</SortHeader>
            <TableHead>Category</TableHead>
            <TableHead>Status</TableHead>
            <SortHeader field="usageCount">Usage</SortHeader>
            <SortHeader field="successRate">Success Rate</SortHeader>
            <SortHeader field="updatedAt">Updated</SortHeader>
            <TableHead></TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {tools.map((tool) => {
            const category = TOOL_CATEGORIES.find((c) => c.value === tool.category);
            const status = TOOL_STATUSES.find((s) => s.value === tool.status);
            return (
              <TableRow
                key={tool.id}
                className="cursor-pointer"
                onClick={() => onToolClick(tool)}
              >
                <TableCell className="font-medium">
                  <span className="mr-2">{category?.icon}</span>
                  {tool.name}
                </TableCell>
                <TableCell>{category?.label}</TableCell>
                <TableCell>
                  <span
                    className={`inline-block w-2 h-2 rounded-full mr-2 ${status?.color}`}
                  />
                  {status?.label}
                </TableCell>
                <TableCell>{tool.usageCount.toLocaleString()}</TableCell>
                <TableCell>
                  {(tool.successRate * DASHBOARD_CONSTANTS.PERCENT_MULTIPLIER).toFixed(
                    DASHBOARD_CONSTANTS.DECIMAL_PLACES
                  )}%
                </TableCell>
                <TableCell className="text-muted-foreground">
                  {new Date(tool.updatedAt).toLocaleDateString()}
                </TableCell>
                <TableCell>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onToolEdit(tool);
                    }}
                    className="text-sm text-primary hover:underline"
                  >
                    Edit
                  </button>
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </Card>
  );
}
