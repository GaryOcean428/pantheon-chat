/**
 * RadarView - Radial visualization of basin coordinates
 */



interface RadarViewProps {
  coordinates: number[];
  size?: number;
}

export function RadarView({ coordinates, size = 250 }: RadarViewProps) {
  const maxValue = Math.max(...coordinates, 0.01);
  const centerX = size / 2;
  const centerY = size / 2;
  const maxRadius = size / 2 - 20;
  
  // Generate polygon points
  const points = coordinates.map((value, i) => {
    const angle = (i / coordinates.length) * 2 * Math.PI - Math.PI / 2;
    const radius = (value / maxValue) * maxRadius;
    const x = centerX + radius * Math.cos(angle);
    const y = centerY + radius * Math.sin(angle);
    return `${x},${y}`;
  }).join(' ');
  
  // Generate axis lines (every 8 dimensions)
  const axisLines = Array.from({ length: 8 }, (_, i) => {
    const angle = (i / 8) * 2 * Math.PI - Math.PI / 2;
    const x2 = centerX + maxRadius * Math.cos(angle);
    const y2 = centerY + maxRadius * Math.sin(angle);
    return { x1: centerX, y1: centerY, x2, y2, label: i * 8 };
  });
  
  return (
    <div className="flex justify-center" data-testid="basin-radar">
      <svg width={size} height={size} className="overflow-visible">
        {/* Background circles */}
        {[0.25, 0.5, 0.75, 1].map((scale) => (
          <circle
            key={scale}
            cx={centerX}
            cy={centerY}
            r={maxRadius * scale}
            fill="none"
            stroke="currentColor"
            strokeOpacity={0.1}
          />
        ))}
        
        {/* Axis lines */}
        {axisLines.map((axis, i) => (
          <g key={i}>
            <line
              x1={axis.x1}
              y1={axis.y1}
              x2={axis.x2}
              y2={axis.y2}
              stroke="currentColor"
              strokeOpacity={0.2}
            />
            <text
              x={axis.x2 + (axis.x2 > centerX ? 5 : -15)}
              y={axis.y2 + (axis.y2 > centerY ? 12 : -5)}
              className="text-[10px] fill-muted-foreground"
            >
              {axis.label}
            </text>
          </g>
        ))}
        
        {/* Data polygon */}
        <polygon
          points={points}
          fill="url(#radarGradient)"
          fillOpacity={0.4}
          stroke="#8b5cf6"
          strokeWidth={2}
        />
        
        {/* Gradient definition */}
        <defs>
          <radialGradient id="radarGradient">
            <stop offset="0%" stopColor="#8b5cf6" />
            <stop offset="100%" stopColor="#4c1d95" />
          </radialGradient>
        </defs>
      </svg>
    </div>
  );
}
