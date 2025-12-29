/**
 * CapabilityTelemetryPanel - Main component
 * 
 * Self-awareness metrics for all kernels - what they can do and how well they do it
 */

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui';
import { Brain } from 'lucide-react';
import { QUERY_KEYS } from '@/api';
import { POLLING_CONSTANTS } from '@/lib/constants';
import { FleetTelemetry } from './types';
import { FleetOverview } from './FleetOverview';
import { KernelCard } from './KernelCard';
import { KernelDetail } from './KernelDetail';

export function CapabilityTelemetryPanel() {
  const [selectedKernel, setSelectedKernel] = useState<string | null>(null);
  
  const { data: fleetData, isLoading, error } = useQuery<{ success: boolean; data: FleetTelemetry }>({
    queryKey: QUERY_KEYS.olympus.telemetryFleet(),
    refetchInterval: POLLING_CONSTANTS.VERY_SLOW_INTERVAL_MS,
  });
  
  if (isLoading) {
    return (
      <Card>
        <CardContent className="p-6 text-center text-muted-foreground">
          Loading capability telemetry...
        </CardContent>
      </Card>
    );
  }
  
  if (error || !fleetData?.data) {
    return (
      <Card>
        <CardContent className="p-6 text-center text-destructive">
          Failed to load capability telemetry
        </CardContent>
      </Card>
    );
  }
  
  const fleet = fleetData.data;
  
  return (
    <Card data-testid="capability-telemetry-panel">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Brain className="h-5 w-5 text-primary" />
          <CardTitle>Kernel Capability Telemetry</CardTitle>
        </div>
        <CardDescription>
          Self-awareness metrics for all kernels - what they can do and how well they do it
        </CardDescription>
      </CardHeader>
      
      <CardContent>
        <Tabs defaultValue="overview">
          <TabsList>
            <TabsTrigger value="overview" data-testid="tab-overview">Overview</TabsTrigger>
            <TabsTrigger value="kernels" data-testid="tab-kernels">Kernels</TabsTrigger>
            {selectedKernel && (
              <TabsTrigger value="detail" data-testid="tab-detail">
                {selectedKernel.charAt(0).toUpperCase() + selectedKernel.slice(1)}
              </TabsTrigger>
            )}
          </TabsList>
          
          <TabsContent value="overview" className="mt-4">
            <FleetOverview data={fleet} />
          </TabsContent>
          
          <TabsContent value="kernels" className="mt-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {fleet.kernel_summaries.map(summary => (
                <KernelCard 
                  key={summary.kernel_id} 
                  summary={summary}
                  onSelect={() => setSelectedKernel(summary.kernel_id)}
                />
              ))}
            </div>
          </TabsContent>
          
          {selectedKernel && (
            <TabsContent value="detail" className="mt-4">
              <KernelDetail kernelId={selectedKernel} />
            </TabsContent>
          )}
        </Tabs>
      </CardContent>
    </Card>
  );
}

export default CapabilityTelemetryPanel;

// Re-export types and subcomponents for flexibility
export * from './types';
export { CapabilityCard } from './CapabilityCard';
export { KernelCard } from './KernelCard';
export { FleetOverview } from './FleetOverview';
export { KernelDetail } from './KernelDetail';
