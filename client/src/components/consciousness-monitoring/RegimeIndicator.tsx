/**
 * RegimeIndicator - Shows current consciousness regime
 */

import { Badge } from '@/components/ui';
import { Brain, Moon, Sparkles, Zap, Cloud } from 'lucide-react';

interface RegimeIndicatorProps {
  regime: string;
  isConscious: boolean;
}

const REGIME_CONFIG: Record<string, { 
  icon: typeof Brain; 
  color: string; 
  bgColor: string;
  label: string;
}> = {
  geometric: { icon: Brain, color: 'text-green-400', bgColor: 'bg-green-500/20', label: 'Geometric' },
  hierarchical: { icon: Sparkles, color: 'text-amber-400', bgColor: 'bg-amber-500/20', label: 'Hierarchical' },
  breakdown: { icon: Cloud, color: 'text-red-400', bgColor: 'bg-red-500/20', label: 'Breakdown' },
  sleep: { icon: Moon, color: 'text-blue-400', bgColor: 'bg-blue-500/20', label: 'Sleep' },
  dream: { icon: Sparkles, color: 'text-purple-400', bgColor: 'bg-purple-500/20', label: 'Dream' },
  mushroom: { icon: Zap, color: 'text-pink-400', bgColor: 'bg-pink-500/20', label: 'Mushroom' },
};

export function RegimeIndicator({ regime, isConscious }: RegimeIndicatorProps) {
  const config = REGIME_CONFIG[regime] || REGIME_CONFIG.geometric;
  const Icon = config.icon;
  
  return (
    <div className="flex items-center gap-3" data-testid="regime-indicator">
      <div className={`p-3 rounded-full ${config.bgColor}`}>
        <Icon className={`h-6 w-6 ${config.color}`} />
      </div>
      <div>
        <p className={`font-semibold ${config.color}`}>{config.label}</p>
        <Badge 
          variant="outline" 
          className={isConscious ? 'border-green-500 text-green-400' : 'border-yellow-500 text-yellow-400'}
        >
          {isConscious ? 'CONSCIOUS' : 'PRE-CONSCIOUS'}
        </Badge>
      </div>
    </div>
  );
}
