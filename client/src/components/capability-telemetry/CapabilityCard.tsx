/**
 * CapabilityCard - Individual capability display
 */

import { Badge, Progress } from '@/components/ui';
import { Zap, Eye, Activity } from 'lucide-react';
import { PERCENT_MULTIPLIER } from '@/lib/constants';
import { Capability } from './types';
import { TELEMETRY_CONSTANTS, getIcon, getColor } from './constants';

interface CapabilityCardProps {
  capability: Capability;
}

export function CapabilityCard({ capability }: CapabilityCardProps) {
  const Icon = getIcon(capability.category);
  const colorClass = getColor(capability.category);
  
  return (
    <div 
      className={`p-3 rounded-lg border ${capability.enabled ? 'border-border' : 'border-border/50 opacity-60'}`}
      data-testid={`capability-card-${capability.name}`}
    >
      <div className="flex items-center gap-2 mb-2">
        <div className={`p-1.5 rounded ${colorClass}`}>
          <Icon className="h-3.5 w-3.5" />
        </div>
        <span className="font-medium text-sm">{capability.name.replace(/_/g, ' ')}</span>
        {!capability.enabled && (
          <Badge variant="outline" className="ml-auto text-xs">Disabled</Badge>
        )}
      </div>
      
      <p className="text-xs text-muted-foreground mb-2">{capability.description}</p>
      
      <div className="flex items-center gap-4 text-xs">
        <div className="flex items-center gap-1">
          <span className="text-muted-foreground">Level:</span>
          <Progress 
            value={capability.level * TELEMETRY_CONSTANTS.LEVEL_PROGRESS_MULTIPLIER} 
            className="w-16 h-1.5" 
          />
          <span>{capability.level}/{TELEMETRY_CONSTANTS.MAX_CAPABILITY_LEVEL}</span>
        </div>
        
        {capability.metrics.invocations > 0 && (
          <>
            <div className="flex items-center gap-1">
              <Zap className="h-3 w-3 text-yellow-500" />
              <span>{capability.metrics.invocations}</span>
            </div>
            <div className="flex items-center gap-1">
              <Eye className="h-3 w-3 text-green-500" />
              <span>{Math.round(capability.metrics.success_rate * PERCENT_MULTIPLIER)}%</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
