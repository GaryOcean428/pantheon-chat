/**
 * Tool Detail Panel - displays detailed tool information
 */

import React from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Badge,
  Button,
  Separator,
} from '@/components/ui';
import { X, Edit, Trash2, Clock, User, Activity, Zap, Code } from 'lucide-react';
import { TOOL_CATEGORIES, TOOL_STATUSES, DASHBOARD_CONSTANTS } from '../constants';
import type { Tool } from '../types';

interface ToolDetailPanelProps {
  tool: Tool;
  onClose: () => void;
  onEdit: (tool: Tool) => void;
  onDelete: (tool: Tool) => void;
  isDeleting: boolean;
}

export function ToolDetailPanel({
  tool,
  onClose,
  onEdit,
  onDelete,
  isDeleting,
}: ToolDetailPanelProps) {
  const category = TOOL_CATEGORIES.find((c) => c.value === tool.category);
  const status = TOOL_STATUSES.find((s) => s.value === tool.status);
  const successPercent = tool.successRate * DASHBOARD_CONSTANTS.PERCENT_MULTIPLIER;

  return (
    <Card className="h-full">
      <CardHeader className="flex flex-row items-start justify-between space-y-0">
        <div>
          <CardTitle className="flex items-center gap-2">
            <span>{category?.icon}</span>
            {tool.name}
          </CardTitle>
          <p className="text-sm text-muted-foreground mt-1">v{tool.version}</p>
        </div>
        <Button variant="ghost" size="icon" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="flex gap-2">
          <Badge variant="secondary">{category?.label}</Badge>
          <Badge className={`${status?.color} text-white`}>{status?.label}</Badge>
        </div>

        <p className="text-sm">{tool.description}</p>

        <Separator />

        <div className="grid grid-cols-2 gap-4 text-sm">
          <div className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground">Usage:</span>
            <span className="font-medium">{tool.usageCount.toLocaleString()}</span>
          </div>
          <div className="flex items-center gap-2">
            <Zap className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground">Success:</span>
            <span className="font-medium">
              {successPercent.toFixed(DASHBOARD_CONSTANTS.DECIMAL_PLACES)}%
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Clock className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground">Avg Time:</span>
            <span className="font-medium">{tool.avgExecutionTime}ms</span>
          </div>
          <div className="flex items-center gap-2">
            <User className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground">Author:</span>
            <span className="font-medium">{tool.author}</span>
          </div>
        </div>

        {tool.tags.length > 0 && (
          <>
            <Separator />
            <div>
              <p className="text-sm font-medium mb-2">Tags</p>
              <div className="flex flex-wrap gap-1">
                {tool.tags.map((tag) => (
                  <Badge key={tag} variant="outline">
                    {tag}
                  </Badge>
                ))}
              </div>
            </div>
          </>
        )}

        {tool.parameters.length > 0 && (
          <>
            <Separator />
            <div>
              <p className="text-sm font-medium mb-2 flex items-center gap-1">
                <Code className="h-4 w-4" />
                Parameters ({tool.parameters.length})
              </p>
              <div className="space-y-2">
                {tool.parameters.map((param) => (
                  <div
                    key={param.name}
                    className="text-xs p-2 rounded bg-muted"
                  >
                    <div className="flex items-center gap-2">
                      <span className="font-mono font-medium">{param.name}</span>
                      <Badge variant="outline" className="text-xs">
                        {param.type}
                      </Badge>
                      {param.required && (
                        <Badge variant="destructive" className="text-xs">
                          required
                        </Badge>
                      )}
                    </div>
                    <p className="text-muted-foreground mt-1">{param.description}</p>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}

        <Separator />

        <div className="text-xs text-muted-foreground space-y-1">
          <p>Created: {new Date(tool.createdAt).toLocaleString()}</p>
          <p>Updated: {new Date(tool.updatedAt).toLocaleString()}</p>
        </div>

        <div className="flex gap-2">
          <Button className="flex-1" onClick={() => onEdit(tool)}>
            <Edit className="h-4 w-4 mr-2" />
            Edit
          </Button>
          <Button
            variant="destructive"
            onClick={() => onDelete(tool)}
            disabled={isDeleting}
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
