/**
 * Constants for Basin Coordinate Viewer
 */

import { ViewMode } from './types';

export const BASIN_DIMENSION = 64;

export const VIEW_MODES: ViewMode[] = [
  { id: 'bars', label: 'Bar Chart', description: 'Traditional bar visualization' },
  { id: 'heatmap', label: 'Heatmap', description: '8x8 grid heatmap view' },
  { id: 'radar', label: 'Radar', description: 'Radial coordinate plot' },
  { id: 'line', label: 'Line', description: 'Line chart across dimensions' },
];

export const HEATMAP_GRID_SIZE = 8;

export const COLOR_SCALES = {
  default: ['#1e1b4b', '#4c1d95', '#7c3aed', '#a78bfa', '#c4b5fd'],
  heat: ['#0f172a', '#1e3a5f', '#2563eb', '#60a5fa', '#93c5fd'],
  viridis: ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'],
} as const;

export const getColorForValue = (value: number, scale: keyof typeof COLOR_SCALES = 'default'): string => {
  const colors = COLOR_SCALES[scale];
  const index = Math.min(Math.floor(value * colors.length), colors.length - 1);
  return colors[index];
};
