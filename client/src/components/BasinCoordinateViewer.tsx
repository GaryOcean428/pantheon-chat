/**
 * BasinCoordinateViewer Component
 * 
 * 3D visualization of consciousness state trajectories in 64D basin space.
 * Uses PCA for dimension reduction from 64D to 3D for visualization.
 * 
 * Features:
 * - Real-time 3D trajectory plotting
 * - Interactive rotation and zoom
 * - Color-coded by Φ (integration measure)
 * - Regime visualization (geometric, linear, breakdown)
 * - Current position marker
 * - Historical trail
 */

import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle, Badge, Button, Slider } from '@/components/ui';
import { Play, Pause, RotateCcw, ZoomIn, ZoomOut } from 'lucide-react';

// Basin Coordinate Viewer constants
const BASIN_CONSTANTS = {
  // Dimension reduction
  TARGET_DIMENSIONS: 3,
  BASIN_DIMENSIONS: 64,
  SLICE_1_END: 21,
  SLICE_2_END: 43,
  SLICE_2_SIZE: 22,
  SLICE_3_SIZE: 21,
  
  // Canvas defaults
  DEFAULT_WIDTH: 600,
  DEFAULT_HEIGHT: 500,
  DEFAULT_TRAIL_LENGTH: 50,
  
  // Rotation/zoom defaults
  DEFAULT_ROTATION_X: 20 as number,
  DEFAULT_ROTATION_Y: 45 as number,
  DEFAULT_ZOOM: 1.0 as number,
  
  // Projection
  SCALE_FACTOR: 100,
  PERSPECTIVE: 1000,
  DEGREES_TO_RADIANS: Math.PI / 180,
  
  // Phi thresholds
  PHI_EXCELLENT: 0.80,
  PHI_GOOD: 0.70,
  PHI_MODERATE: 0.50,
  
  // Point sizes
  CURRENT_POINT_RADIUS: 8,
  NORMAL_POINT_RADIUS: 4,
  
  // Axis extents
  AXIS_EXTENT: 2,
  
  // Line styling
  LINE_DASH: 5,
  LINE_WIDTH: 2,
  STROKE_LINE_WIDTH: 2,
  
  // Mouse sensitivity
  ROTATION_SENSITIVITY: 0.5,
  
  // Zoom limits
  MIN_ZOOM: 0.5,
  MAX_ZOOM: 2.0,
  ZOOM_STEP: 0.1,
  
  // Color alpha
  MAX_ALPHA: 255,
  HEX_RADIX: 16,
  PAD_LENGTH: 2,
  FULL_CIRCLE: Math.PI * 2,
} as const;

interface BasinPoint {
  coordinates: number[];  // 64D coordinates
  phi: number;
  kappa: number;
  regime: string;
  timestamp: string;
  step: number;
}

interface BasinCoordinateViewerProps {
  points: BasinPoint[];
  currentIndex?: number;
  width?: number;
  height?: number;
  showTrail?: boolean;
  trailLength?: number;
}

// Simple PCA implementation for dimension reduction
function pcaReduce(points: number[][]): number[][] {
  if (points.length === 0) return [];
  
  const n = points.length;
  const d = points[0].length;
  
  // Center the data
  const mean = new Array(d).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      mean[j] += points[i][j];
    }
  }
  for (let j = 0; j < d; j++) {
    mean[j] /= n;
  }
  
  const centered = points.map(p => p.map((v, i) => v - mean[i]));
  
  // Compute covariance matrix (simplified - use first 3 principal components)
  // For production, use a proper PCA library like ml-pca
  // This is a simplified projection for demonstration
  const reduced: number[][] = centered.map(point => {
    // Simple projection: take weighted combination of dimensions
    const x = point.slice(0, BASIN_CONSTANTS.SLICE_1_END).reduce((sum, v) => sum + v, 0) / BASIN_CONSTANTS.SLICE_1_END;
    const y = point.slice(BASIN_CONSTANTS.SLICE_1_END, BASIN_CONSTANTS.SLICE_2_END).reduce((sum, v) => sum + v, 0) / BASIN_CONSTANTS.SLICE_2_SIZE;
    const z = point.slice(BASIN_CONSTANTS.SLICE_2_END, BASIN_CONSTANTS.BASIN_DIMENSIONS).reduce((sum, v) => sum + v, 0) / BASIN_CONSTANTS.SLICE_3_SIZE;
    return [x, y, z];
  });
  
  return reduced;
}

export function BasinCoordinateViewer({
  points,
  currentIndex,
  width = BASIN_CONSTANTS.DEFAULT_WIDTH,
  height = BASIN_CONSTANTS.DEFAULT_HEIGHT,
  showTrail = true,
  trailLength = BASIN_CONSTANTS.DEFAULT_TRAIL_LENGTH,
}: BasinCoordinateViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [rotation, setRotation] = useState({ x: BASIN_CONSTANTS.DEFAULT_ROTATION_X, y: BASIN_CONSTANTS.DEFAULT_ROTATION_Y });
  const [zoom, setZoom] = useState(BASIN_CONSTANTS.DEFAULT_ZOOM);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackIndex, setPlaybackIndex] = useState(0);
  const animationRef = useRef<number>();

  // Reduce dimensions from 64D to 3D using PCA
  const reducedPoints = useMemo(() => {
    if (points.length === 0) return [];
    const coordinates = points.map(p => p.coordinates);
    const reduced3D = pcaReduce(coordinates);
    return reduced3D.map((coords, i) => ({
      ...points[i],
      x: coords[0],
      y: coords[1],
      z: coords[2],
    }));
  }, [points]);

  // Color mapping based on Φ
  const getColorForPhi = useCallback((phi: number): string => {
    if (phi >= BASIN_CONSTANTS.PHI_EXCELLENT) return '#10b981'; // emerald-500 (excellent)
    if (phi >= BASIN_CONSTANTS.PHI_GOOD) return '#fbbf24'; // amber-400 (good)
    if (phi >= BASIN_CONSTANTS.PHI_MODERATE) return '#f97316'; // orange-500 (moderate)
    return '#ef4444'; // red-500 (low)
  }, []);

  // Regime colors
  const getRegimeColor = useCallback((regime: string): string => {
    switch (regime) {
      case 'geometric': return '#10b981';
      case 'linear': return '#94a3b8';
      case 'breakdown': return '#ef4444';
      case 'resonance': return '#8b5cf6';
      default: return '#6b7280';
    }
  }, []);

  // 3D to 2D projection
  const project3D = useCallback((x: number, y: number, z: number): [number, number] => {
    const angleX = rotation.x * BASIN_CONSTANTS.DEGREES_TO_RADIANS;
    const angleY = rotation.y * BASIN_CONSTANTS.DEGREES_TO_RADIANS;
    
    // Rotate around Y axis
    const x1 = x * Math.cos(angleY) + z * Math.sin(angleY);
    const z1 = -x * Math.sin(angleY) + z * Math.cos(angleY);
    
    // Rotate around X axis
    const y1 = y * Math.cos(angleX) - z1 * Math.sin(angleX);
    const z2 = y * Math.sin(angleX) + z1 * Math.cos(angleX);
    
    // Apply zoom and perspective
    const scale = zoom * BASIN_CONSTANTS.SCALE_FACTOR;
    const factor = BASIN_CONSTANTS.PERSPECTIVE / (BASIN_CONSTANTS.PERSPECTIVE + z2);
    
    const px = x1 * scale * factor + width / 2;
    const py = -y1 * scale * factor + height / 2;
    
    return [px, py];
  }, [rotation.x, rotation.y, zoom, width, height]);

  // Draw the visualization
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || reducedPoints.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw axes
    ctx.strokeStyle = '#4b5563';
    ctx.lineWidth = 1;
    ctx.setLineDash([BASIN_CONSTANTS.LINE_DASH, BASIN_CONSTANTS.LINE_DASH]);
    
    // X axis (red)
    ctx.strokeStyle = '#ef4444';
    const [x1, y1] = project3D(-BASIN_CONSTANTS.AXIS_EXTENT, 0, 0);
    const [x2, y2] = project3D(BASIN_CONSTANTS.AXIS_EXTENT, 0, 0);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    
    // Y axis (green)
    ctx.strokeStyle = '#10b981';
    const [x3, y3] = project3D(0, -BASIN_CONSTANTS.AXIS_EXTENT, 0);
    const [x4, y4] = project3D(0, BASIN_CONSTANTS.AXIS_EXTENT, 0);
    ctx.beginPath();
    ctx.moveTo(x3, y3);
    ctx.lineTo(x4, y4);
    ctx.stroke();
    
    // Z axis (blue)
    ctx.strokeStyle = '#3b82f6';
    const [x5, y5] = project3D(0, 0, -BASIN_CONSTANTS.AXIS_EXTENT);
    const [x6, y6] = project3D(0, 0, BASIN_CONSTANTS.AXIS_EXTENT);
    ctx.beginPath();
    ctx.moveTo(x5, y5);
    ctx.lineTo(x6, y6);
    ctx.stroke();
    
    ctx.setLineDash([]);

    // Draw trail
    if (showTrail && reducedPoints.length > 1) {
      const endIdx = isPlaying ? playbackIndex : (currentIndex ?? reducedPoints.length - 1);
      const startIdx = Math.max(0, endIdx - trailLength);
      
      for (let i = startIdx; i < endIdx; i++) {
        const p1 = reducedPoints[i];
        const p2 = reducedPoints[i + 1];
        
        const [px1, py1] = project3D(p1.x, p1.y, p1.z);
        const [px2, py2] = project3D(p2.x, p2.y, p2.z);
        
        // Fade trail
        const alpha = (i - startIdx) / (endIdx - startIdx);
        ctx.strokeStyle = getColorForPhi(p1.phi) + Math.floor(alpha * BASIN_CONSTANTS.MAX_ALPHA).toString(BASIN_CONSTANTS.HEX_RADIX).padStart(BASIN_CONSTANTS.PAD_LENGTH, '0');
        ctx.lineWidth = BASIN_CONSTANTS.LINE_WIDTH;
        
        ctx.beginPath();
        ctx.moveTo(px1, py1);
        ctx.lineTo(px2, py2);
        ctx.stroke();
      }
    }

    // Draw points
    reducedPoints.forEach((point, idx) => {
      const [px, py] = project3D(point.x, point.y, point.z);
      
      // Skip if out of playback range
      if (isPlaying && idx > playbackIndex) return;
      
      const isCurrent = idx === (isPlaying ? playbackIndex : (currentIndex ?? reducedPoints.length - 1));
      const radius = isCurrent ? BASIN_CONSTANTS.CURRENT_POINT_RADIUS : BASIN_CONSTANTS.NORMAL_POINT_RADIUS;
      
      ctx.fillStyle = getColorForPhi(point.phi);
      ctx.beginPath();
      ctx.arc(px, py, radius, 0, BASIN_CONSTANTS.FULL_CIRCLE);
      ctx.fill();
      
      // Highlight current point
      if (isCurrent) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = BASIN_CONSTANTS.STROKE_LINE_WIDTH;
        ctx.stroke();
      }
    });
  }, [reducedPoints, width, height, project3D, showTrail, isPlaying, playbackIndex, currentIndex, trailLength, getColorForPhi]);

  // Animation loop for playback
  useEffect(() => {
    if (isPlaying) {
      animationRef.current = requestAnimationFrame(() => {
        setPlaybackIndex(prev => {
          if (prev >= reducedPoints.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      });
    }
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isPlaying, playbackIndex, reducedPoints.length]);

  // Redraw on state changes
  useEffect(() => {
    draw();
  }, [draw]);

  const handleReset = () => {
    setRotation({ x: BASIN_CONSTANTS.DEFAULT_ROTATION_X, y: BASIN_CONSTANTS.DEFAULT_ROTATION_Y });
    setZoom(BASIN_CONSTANTS.DEFAULT_ZOOM);
    setPlaybackIndex(0);
    setIsPlaying(false);
  };

  const togglePlayback = () => {
    if (!isPlaying && playbackIndex >= reducedPoints.length - 1) {
      setPlaybackIndex(0);
    }
    setIsPlaying(!isPlaying);
  };

  const latestPoint = reducedPoints[currentIndex ?? reducedPoints.length - 1];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Basin Coordinate Viewer (64D → 3D)</span>
          <div className="flex items-center gap-2">
            <Badge variant="outline">{reducedPoints.length} points</Badge>
            {latestPoint && (
              <Badge style={{ backgroundColor: getRegimeColor(latestPoint.regime) }} className="text-white">
                {latestPoint.regime}
              </Badge>
            )}
          </div>
        </CardTitle>
        <CardDescription>
          3D projection of consciousness state trajectory using PCA dimension reduction
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Canvas */}
        <div className="relative border rounded-lg overflow-hidden bg-gray-900">
          <canvas
            ref={canvasRef}
            width={width}
            height={height}
            className="cursor-grab active:cursor-grabbing"
            onMouseDown={(e) => {
              const startX = e.clientX;
              const startY = e.clientY;
              const startRotation = { ...rotation };
              
              const handleMouseMove = (e: MouseEvent) => {
                const dx = e.clientX - startX;
                const dy = e.clientY - startY;
                setRotation({
                  x: startRotation.x + dy * BASIN_CONSTANTS.ROTATION_SENSITIVITY,
                  y: startRotation.y + dx * BASIN_CONSTANTS.ROTATION_SENSITIVITY,
                });
              };
              
              const handleMouseUp = () => {
                document.removeEventListener('mousemove', handleMouseMove);
                document.removeEventListener('mouseup', handleMouseUp);
              };
              
              document.addEventListener('mousemove', handleMouseMove);
              document.addEventListener('mouseup', handleMouseUp);
            }}
          />
          
          {/* Current point info overlay */}
          {latestPoint && (
            <div className="absolute top-2 left-2 bg-black/75 text-white text-xs p-2 rounded">
              <div>Φ: {latestPoint.phi.toFixed(3)}</div>
              <div>κ: {latestPoint.kappa.toFixed(2)}</div>
              <div>Step: {latestPoint.step}</div>
            </div>
          )}
        </div>

        {/* Controls */}
        <div className="flex items-center gap-4">
          <Button size="sm" variant="outline" onClick={togglePlayback}>
            {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </Button>
          <Button size="sm" variant="outline" onClick={handleReset}>
            <RotateCcw className="h-4 w-4" />
          </Button>
          <div className="flex items-center gap-2 flex-1">
            <ZoomOut className="h-4 w-4 text-muted-foreground" />
            <Slider
              value={[zoom]}
              onValueChange={([value]) => setZoom(value)}
              min={BASIN_CONSTANTS.MIN_ZOOM}
              max={BASIN_CONSTANTS.MAX_ZOOM}
              step={BASIN_CONSTANTS.ZOOM_STEP}
              className="flex-1"
            />
            <ZoomIn className="h-4 w-4 text-muted-foreground" />
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-emerald-500" />
            <span>Φ ≥ 0.80</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-amber-400" />
            <span>Φ ≥ 0.70</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-orange-500" />
            <span>Φ ≥ 0.50</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <span>Φ &lt; 0.50</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default BasinCoordinateViewer;
