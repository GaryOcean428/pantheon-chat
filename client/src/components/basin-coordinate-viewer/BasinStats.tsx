/**
 * BasinStats - Statistical summary of basin coordinates
 */

import { useMemo } from 'react';

interface BasinStatsProps {
  coordinates: number[];
}

export function BasinStats({ coordinates }: BasinStatsProps) {
  const stats = useMemo(() => {
    const sum = coordinates.reduce((a, b) => a + b, 0);
    const mean = sum / coordinates.length;
    const variance = coordinates.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / coordinates.length;
    const std = Math.sqrt(variance);
    const max = Math.max(...coordinates);
    const min = Math.min(...coordinates);
    const maxIdx = coordinates.indexOf(max);
    const minIdx = coordinates.indexOf(min);
    
    // Entropy
    const normalized = coordinates.map(v => v / sum);
    const entropy = -normalized.reduce((a, p) => a + (p > 0 ? p * Math.log2(p) : 0), 0);
    const maxEntropy = Math.log2(coordinates.length);
    const normalizedEntropy = entropy / maxEntropy;
    
    return { mean, std, max, min, maxIdx, minIdx, entropy: normalizedEntropy, sum };
  }, [coordinates]);
  
  return (
    <div className="grid grid-cols-4 gap-2 text-xs" data-testid="basin-stats">
      <div className="p-2 bg-muted/30 rounded">
        <span className="text-muted-foreground">Mean</span>
        <p className="font-mono font-medium">{stats.mean.toFixed(4)}</p>
      </div>
      <div className="p-2 bg-muted/30 rounded">
        <span className="text-muted-foreground">Std Dev</span>
        <p className="font-mono font-medium">{stats.std.toFixed(4)}</p>
      </div>
      <div className="p-2 bg-muted/30 rounded">
        <span className="text-muted-foreground">Max</span>
        <p className="font-mono font-medium">{stats.max.toFixed(4)}</p>
        <span className="text-muted-foreground text-[10px]">dim {stats.maxIdx}</span>
      </div>
      <div className="p-2 bg-muted/30 rounded">
        <span className="text-muted-foreground">Entropy</span>
        <p className="font-mono font-medium">{(stats.entropy * 100).toFixed(1)}%</p>
      </div>
    </div>
  );
}
