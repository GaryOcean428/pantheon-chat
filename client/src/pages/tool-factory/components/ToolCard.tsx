/**
 * Tool Card - displays a single tool in grid view
 */

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, Badge } from '@/components/ui';
import { Clock, Activity, Zap } from 'lucide-react';
import { TOOL_CATEGORIES, TOOL_STATUSES, DASHBOARD_CONSTANTS } from '../constants';
import type { Tool } from '../types';

interface ToolCardProps {
  tool: Tool;
  onClick: (tool: Tool) => void;
  onEdit: (tool: Tool) => void;
}

export function ToolCard({ tool, onClick, onEdit }: ToolCardProps) {
  const category = TOOL_CATEGORIES.find((c) => c.value === tool.category);
  const status = TOOL_STATUSES.find((s) => s.value === tool.status);
  const successPercent = tool.successRate * DASHBOARD_CONSTANTS.PERCENT_MULTIPLIER;

  return (
    <Card
      className="cursor-pointer transition-all hover:shadow-md hover:border-primary/50"
      onClick={() => onClick(tool)}
    >
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between">
          <CardTitle className="text-base flex items-center gap-2">
            <span>{category?.icon}</span>
            {tool.name}
          </CardTitle>
          <Badge
            variant="secondary"
            className={`${status?.color} text-white text-xs`}
          >
            {status?.label}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-sm text-muted-foreground line-clamp-2">
          {tool.description}
        </p>

        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <Activity className="h-3 w-3" />
            {tool.usageCount.toLocaleString()} uses
          </span>
          <span className="flex items-center gap-1">
            <Zap className="h-3 w-3" />
            {successPercent.toFixed(DASHBOARD_CONSTANTS.DECIMAL_PLACES)}%
          </span>
          <span className="flex items-center gap-1">
            <Clock className="h-3 w-3" />
            {tool.avgExecutionTime}ms
          </span>
        </div>

        {tool.tags.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {tool.tags.slice(0, DASHBOARD_CONSTANTS.MAX_TAGS_DISPLAY).map((tag) => (
              <Badge key={tag} variant="outline" className="text-xs">
                {tag}
              </Badge>
            ))}
            {tool.tags.length > DASHBOARD_CONSTANTS.MAX_TAGS_DISPLAY && (
              <Badge variant="outline" className="text-xs">
                +{tool.tags.length - DASHBOARD_CONSTANTS.MAX_TAGS_DISPLAY}
              </Badge>
            )}
          </div>
        )}

        <div className="flex justify-between items-center pt-2 border-t">
          <span className="text-xs text-muted-foreground">v{tool.version}</span>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onEdit(tool);
            }}
            className="text-xs text-primary hover:underline"
          >
            Edit
          </button>
        </div>
      </CardContent>
    </Card>
  );
}
