/**
 * NeurochemistryPanel - Neurochemistry state display
 */

import { Card, CardContent, CardHeader, CardTitle, Badge } from '@/components/ui';
import { Heart, Smile, Zap, AlertTriangle } from 'lucide-react';

interface NeurochemistryPanelProps {
  neurochemistry: {
    dopamine: number;
    serotonin: number;
    norepinephrine: number;
    cortisol?: number;
    emotionalState: string;
  };
}

const NEUROTRANSMITTER_CONFIG = [
  { key: 'dopamine', label: 'Dopamine', icon: Smile, color: 'bg-yellow-500' },
  { key: 'serotonin', label: 'Serotonin', icon: Heart, color: 'bg-pink-500' },
  { key: 'norepinephrine', label: 'Norepinephrine', icon: Zap, color: 'bg-blue-500' },
  { key: 'cortisol', label: 'Cortisol', icon: AlertTriangle, color: 'bg-red-500' },
] as const;

export function NeurochemistryPanel({ neurochemistry }: NeurochemistryPanelProps) {
  return (
    <Card data-testid="neurochemistry-panel">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">Neurochemistry</CardTitle>
          <Badge variant="outline" className="text-xs">
            {neurochemistry.emotionalState}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {NEUROTRANSMITTER_CONFIG.map(({ key, label, icon: Icon, color }) => (
          <div key={key} className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-1.5">
                <Icon className="h-3 w-3" />
                <span>{label}</span>
              </div>
              <span className="text-muted-foreground">
                {Math.round(((neurochemistry as Record<string, number | string>)[key] as number || 0) * 100)}%
              </span>
            </div>
            <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
              <div 
                className={`h-full rounded-full ${color}`} 
                style={{ width: `${((neurochemistry as Record<string, number | string>)[key] as number || 0) * 100}%` }} 
              />
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
