/**
 * LineChartView - Line chart visualization of basin coordinates
 */

interface LineChartViewProps {
  coordinates: number[];
  height?: number;
}

export function LineChartView({ coordinates, height = 120 }: LineChartViewProps) {
  const maxValue = Math.max(...coordinates, 0.01);
  const minValue = Math.min(...coordinates, 0);
  const range = maxValue - minValue || 1;
  
  const points = coordinates.map((value, i) => {
    const x = (i / (coordinates.length - 1)) * 100;
    const y = height - ((value - minValue) / range) * height;
    return `${x}%,${y}`;
  }).join(' ');
  
  return (
    <div style={{ height }} className="relative" data-testid="basin-line-chart">
      <svg className="w-full h-full" preserveAspectRatio="none">
        {/* Grid lines */}
        {[0, 25, 50, 75, 100].map(y => (
          <line
            key={y}
            x1="0"
            y1={`${y}%`}
            x2="100%"
            y2={`${y}%`}
            stroke="currentColor"
            strokeOpacity={0.1}
          />
        ))}
        
        {/* Data line */}
        <polyline
          points={points}
          fill="none"
          stroke="#8b5cf6"
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />
        
        {/* Area fill */}
        <polygon
          points={`0%,${height} ${points} 100%,${height}`}
          fill="url(#lineGradient)"
          fillOpacity={0.3}
        />
        
        <defs>
          <linearGradient id="lineGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#8b5cf6" />
            <stop offset="100%" stopColor="transparent" />
          </linearGradient>
        </defs>
      </svg>
      
      {/* Y-axis labels */}
      <div className="absolute top-0 right-1 text-[10px] text-muted-foreground">
        {maxValue.toFixed(3)}
      </div>
      <div className="absolute bottom-0 right-1 text-[10px] text-muted-foreground">
        {minValue.toFixed(3)}
      </div>
    </div>
  );
}
