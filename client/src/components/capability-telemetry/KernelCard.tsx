/**
 * KernelCard - Kernel summary card
 */

import { Card, CardContent, Badge } from '@/components/ui';
import { PERCENT_MULTIPLIER } from '@/lib/constants';
import { KernelSummary } from './types';
import { OLYMPUS_KERNEL_IDS, SHADOW_KERNEL_IDS } from './constants';

interface KernelCardProps {
  summary: KernelSummary;
  onSelect: () => void;
}

export function KernelCard({ summary, onSelect }: KernelCardProps) {
  const isOlympus = (OLYMPUS_KERNEL_IDS as readonly string[]).includes(summary.kernel_id);
  const isShadow = (SHADOW_KERNEL_IDS as readonly string[]).includes(summary.kernel_id);
  
  return (
    <Card 
      className="cursor-pointer hover-elevate transition-colors"
      onClick={onSelect}
      data-testid={`kernel-card-${summary.kernel_id}`}
    >
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <div className={`h-2 w-2 rounded-full ${
              isShadow ? 'bg-gray-500' : isOlympus ? 'bg-yellow-500' : 'bg-blue-500'
            }`} />
            <span className="font-medium">{summary.kernel_name}</span>
          </div>
          <Badge variant="outline" className="text-xs">
            {summary.enabled}/{summary.total_capabilities}
          </Badge>
        </div>
        
        <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
          <div>
            <span>Invocations: </span>
            <span className="text-foreground">{summary.total_invocations}</span>
          </div>
          <div>
            <span>Success: </span>
            <span className="text-foreground">
              {Math.round(summary.success_rate * PERCENT_MULTIPLIER)}%
            </span>
          </div>
          {summary.strongest && (
            <div className="col-span-2">
              <span>Strongest: </span>
              <Badge variant="secondary" className="text-xs">{summary.strongest}</Badge>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
