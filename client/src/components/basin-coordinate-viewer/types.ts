/**
 * Types for Basin Coordinate Viewer
 */

export interface BasinCoordinate {
  dimension: number;
  value: number;
  label?: string;
}

export interface BasinData {
  coordinates: number[];
  label?: string;
  color?: string;
}

export interface ViewMode {
  id: string;
  label: string;
  description: string;
}

export interface HeatmapCell {
  x: number;
  y: number;
  value: number;
}
