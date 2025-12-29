/**
 * BasinCoordinateViewer - Main component
 * 
 * Multi-view visualization of 64D basin coordinates with statistics
 */

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui';
import { Layers, BarChart3, Grid3X3, Radar, TrendingUp } from 'lucide-react';
import { VIEW_MODES, BASIN_DIMENSION } from './constants';
import { BarChartView } from './BarChartView';
import { HeatmapView } from './HeatmapView';
import { RadarView } from './RadarView';
import { LineChartView } from './LineChartView';
import { BasinStats } from './BasinStats';

interface BasinCoordinateViewerProps {
  coordinates?: number[];
  title?: string;
  description?: string;
  showStats?: boolean;
}

const VIEW_ICONS = {
  bars: BarChart3,
  heatmap: Grid3X3,
  radar: Radar,
  line: TrendingUp,
};

export function BasinCoordinateViewer({
  coordinates,
  title = 'Basin Coordinates',
  description = '64-dimensional manifold position',
  showStats = true,
}: BasinCoordinateViewerProps) {
  const [viewMode, setViewMode] = useState('bars');
  
  // Generate sample data if none provided
  const data = coordinates || Array.from({ length: BASIN_DIMENSION }, () => Math.random());
  
  if (data.length === 0) {
    return (
      <Card>
        <CardContent className="p-6 text-center text-muted-foreground">
          <Layers className="h-12 w-12 mx-auto mb-2 opacity-50" />
          <p>No basin coordinates available</p>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <Card data-testid="basin-coordinate-viewer">
      <CardHeader className="pb-2">
        <div className="flex items-center gap-2">
          <Layers className="h-5 w-5 text-primary" />
          <CardTitle className="text-base">{title}</CardTitle>
        </div>
        <CardDescription className="text-xs">{description}</CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-4">
        <Tabs value={viewMode} onValueChange={setViewMode}>
          <TabsList className="grid grid-cols-4 h-8">
            {VIEW_MODES.map(mode => {
              const Icon = VIEW_ICONS[mode.id as keyof typeof VIEW_ICONS];
              return (
                <TabsTrigger 
                  key={mode.id} 
                  value={mode.id} 
                  className="text-xs gap-1"
                  title={mode.description}
                >
                  <Icon className="h-3 w-3" />
                  {mode.label}
                </TabsTrigger>
              );
            })}
          </TabsList>
          
          <TabsContent value="bars" className="mt-3">
            <BarChartView coordinates={data} />
          </TabsContent>
          
          <TabsContent value="heatmap" className="mt-3">
            <HeatmapView coordinates={data} />
          </TabsContent>
          
          <TabsContent value="radar" className="mt-3">
            <RadarView coordinates={data} />
          </TabsContent>
          
          <TabsContent value="line" className="mt-3">
            <LineChartView coordinates={data} />
          </TabsContent>
        </Tabs>
        
        {showStats && <BasinStats coordinates={data} />}
      </CardContent>
    </Card>
  );
}

export default BasinCoordinateViewer;

// Re-export types and subcomponents
export * from './types';
export { BarChartView } from './BarChartView';
export { HeatmapView } from './HeatmapView';
export { RadarView } from './RadarView';
export { LineChartView } from './LineChartView';
export { BasinStats } from './BasinStats';
export { VIEW_MODES, BASIN_DIMENSION, getColorForValue } from './constants';
