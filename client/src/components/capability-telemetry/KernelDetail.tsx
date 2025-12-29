/**
 * KernelDetail - Detailed kernel capabilities view
 */

import { useQuery } from '@tanstack/react-query';
import { Badge, Tabs, TabsContent, TabsList, TabsTrigger, ScrollArea } from '@/components/ui';
import { QUERY_KEYS } from '@/api';
import { Capability } from './types';
import { CapabilityCard } from './CapabilityCard';
import { getIcon } from './constants';

interface KernelDetailProps {
  kernelId: string;
}

interface KernelCapabilitiesResponse {
  success: boolean;
  data: {
    kernel: string;
    capabilities: Capability[];
    count: number;
  };
}

export function KernelDetail({ kernelId }: KernelDetailProps) {
  const { data, isLoading, error } = useQuery<KernelCapabilitiesResponse>({
    queryKey: QUERY_KEYS.olympus.telemetryKernelCapabilities(kernelId),
  });
  
  if (isLoading) {
    return (
      <div className="p-4 text-center text-muted-foreground">
        Loading capabilities...
      </div>
    );
  }
  
  if (error || !data?.data) {
    return (
      <div className="p-4 text-center text-destructive">
        Failed to load capabilities
      </div>
    );
  }
  
  const capabilities = data.data.capabilities;
  const categories = [...new Set(capabilities.map(c => c.category))];
  
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold">
          {kernelId.charAt(0).toUpperCase() + kernelId.slice(1)} Capabilities
        </h3>
        <Badge variant="outline">{capabilities.length} total</Badge>
      </div>
      
      <Tabs defaultValue={categories[0] || 'all'}>
        <TabsList className="flex flex-wrap h-auto gap-1">
          {categories.map(cat => {
            const Icon = getIcon(cat);
            return (
              <TabsTrigger key={cat} value={cat} className="gap-1 text-xs">
                <Icon className="h-3 w-3" />
                {cat}
              </TabsTrigger>
            );
          })}
        </TabsList>
        
        {categories.map(cat => (
          <TabsContent key={cat} value={cat}>
            <ScrollArea className="h-[300px]">
              <div className="space-y-2 pr-4">
                {capabilities
                  .filter(c => c.category === cat)
                  .map(cap => (
                    <CapabilityCard key={cap.name} capability={cap} />
                  ))}
              </div>
            </ScrollArea>
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}
