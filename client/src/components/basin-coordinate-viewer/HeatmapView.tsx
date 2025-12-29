/**
 * HeatmapView - 8x8 grid heatmap visualization of basin coordinates
 */

import { HEATMAP_GRID_SIZE, getColorForValue } from './constants';

interface HeatmapViewProps {
  coordinates: number[];
}

export function HeatmapView({ coordinates }: HeatmapViewProps) {
  const maxValue = Math.max(...coordinates, 0.01);
  const gridSize = HEATMAP_GRID_SIZE;
  
  // Reshape into grid
  const grid: number[][] = [];
  for (let row = 0; row < gridSize; row++) {
    grid[row] = [];
    for (let col = 0; col < gridSize; col++) {
      const idx = row * gridSize + col;
      grid[row][col] = idx < coordinates.length ? coordinates[idx] : 0;
    }
  }
  
  return (
    <div className="aspect-square max-w-[300px] mx-auto" data-testid="basin-heatmap">
      <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${gridSize}, 1fr)` }}>
        {grid.flat().map((value, i) => {
          const normalized = value / maxValue;
          const color = getColorForValue(normalized);
          
          return (
            <div
              key={i}
              className="aspect-square rounded-sm transition-all duration-200 hover:scale-110 cursor-pointer"
              style={{ backgroundColor: color }}
              title={`Dim ${i}: ${value.toFixed(4)}`}
            />
          );
        })}
      </div>
      
      {/* Color scale legend */}
      <div className="flex items-center justify-between mt-2 text-xs text-muted-foreground">
        <span>0</span>
        <div className="flex-1 mx-2 h-2 rounded bg-gradient-to-r from-[#1e1b4b] via-[#7c3aed] to-[#c4b5fd]" />
        <span>{maxValue.toFixed(3)}</span>
      </div>
    </div>
  );
}
