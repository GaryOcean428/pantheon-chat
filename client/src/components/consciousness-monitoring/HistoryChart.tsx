/**
 * HistoryChart - Mini chart showing phi/kappa history
 */

import { HistoryEntry } from './types';

interface HistoryChartProps {
  history: HistoryEntry[];
  height?: number;
}

export function HistoryChart({ history, height = 60 }: HistoryChartProps) {
  if (history.length < 2) {
    return (
      <div 
        className="flex items-center justify-center text-muted-foreground text-xs"
        style={{ height }}
      >
        Collecting data...
      </div>
    );
  }
  
  const maxPhi = Math.max(...history.map(h => h.phi), 1);
  const minPhi = Math.min(...history.map(h => h.phi), 0);
  const range = maxPhi - minPhi || 1;
  
  const points = history.map((entry, i) => {
    const x = (i / (history.length - 1)) * 100;
    const y = height - ((entry.phi - minPhi) / range) * height;
    return `${x},${y}`;
  }).join(' ');
  
  return (
    <div className="relative" style={{ height }} data-testid="history-chart">
      <svg className="w-full h-full" preserveAspectRatio="none">
        {/* Grid lines */}
        <line x1="0" y1="50%" x2="100%" y2="50%" stroke="currentColor" strokeOpacity="0.1" />
        <line x1="0" y1="25%" x2="100%" y2="25%" stroke="currentColor" strokeOpacity="0.05" />
        <line x1="0" y1="75%" x2="100%" y2="75%" stroke="currentColor" strokeOpacity="0.05" />
        
        {/* Phi line */}
        <polyline
          points={points}
          fill="none"
          stroke="#8b5cf6"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />
        
        {/* Current point */}
        {history.length > 0 && (
          <circle
            cx="100%"
            cy={height - ((history[history.length - 1].phi - minPhi) / range) * height}
            r="3"
            fill="#8b5cf6"
          />
        )}
      </svg>
      
      {/* Labels */}
      <div className="absolute top-0 right-0 text-xs text-muted-foreground">
        {maxPhi.toFixed(2)}
      </div>
      <div className="absolute bottom-0 right-0 text-xs text-muted-foreground">
        {minPhi.toFixed(2)}
      </div>
    </div>
  );
}
