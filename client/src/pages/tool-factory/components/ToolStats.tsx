/**
 * Tool Stats - displays aggregate statistics
 */

import React from 'react';
import { Card, CardContent } from '@/components/ui';
import { Wrench, CheckCircle, Activity, TrendingUp } from 'lucide-react';
import { DASHBOARD_CONSTANTS } from '../constants';
import type { Tool } from '../types';

interface ToolStatsProps {
  tools: Tool[];
}

export function ToolStats({ tools }: ToolStatsProps) {
  const totalTools = tools.length;
  const activeTools = tools.filter((t) => t.status === 'active').length;
  const totalUsage = tools.reduce((sum, t) => sum + t.usageCount, 0);
  const avgSuccessRate =
    tools.length > 0
      ? tools.reduce((sum, t) => sum + t.successRate, 0) / tools.length
      : 0;

  const stats = [
    {
      label: 'Total Tools',
      value: totalTools,
      icon: Wrench,
      color: 'text-blue-500',
    },
    {
      label: 'Active',
      value: activeTools,
      icon: CheckCircle,
      color: 'text-green-500',
    },
    {
      label: 'Total Usage',
      value: totalUsage.toLocaleString(),
      icon: Activity,
      color: 'text-purple-500',
    },
    {
      label: 'Avg Success',
      value: `${(avgSuccessRate * DASHBOARD_CONSTANTS.PERCENT_MULTIPLIER).toFixed(
        DASHBOARD_CONSTANTS.DECIMAL_PLACES
      )}%`,
      icon: TrendingUp,
      color: 'text-orange-500',
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {stats.map((stat) => (
        <Card key={stat.label}>
          <CardContent className="p-4 flex items-center gap-3">
            <stat.icon className={`h-8 w-8 ${stat.color}`} />
            <div>
              <p className="text-2xl font-bold">{stat.value}</p>
              <p className="text-xs text-muted-foreground">{stat.label}</p>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
