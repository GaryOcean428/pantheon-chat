/**
 * BarChartView - Bar chart visualization of basin coordinates
 */

import { getColorForValue } from './constants';

interface BarChartViewProps {
  coordinates: number[];
  height?: number;
}

export function BarChartView({ coordinates, height = 120 }: BarChartViewProps) {
  const maxValue = Math.max(...coordinates, 0.01);
  
  return (
    <div 
      className="flex items-end gap-px bg-muted/30 rounded p-2"
      style={{ height }}
      data-testid="basin-bar-chart"
    >
      {coordinates.map((value, i) => {
        const normalizedHeight = (value / maxValue) * 100;
        const color = getColorForValue(value / maxValue);
        
        return (
          <div
            key={i}
            className="flex-1 transition-all duration-200 rounded-t hover:opacity-80"
            style={{
              height: `${normalizedHeight}%`,
              backgroundColor: color,
              minWidth: 2,
            }}
            title={`Dim ${i}: ${value.toFixed(4)}`}
          />
        );
      })}
    </div>
  );
}
