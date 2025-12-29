/**
 * FleetOverview - Fleet stats and category distribution
 */

import { Card, CardContent, CardHeader, CardTitle, Badge } from '@/components/ui';
import { Users, Cpu, Zap, Activity } from 'lucide-react';
import { PERCENT_MULTIPLIER } from '@/lib/constants';
import { FleetTelemetry } from './types';
import { getIcon, getColor } from './constants';

interface FleetOverviewProps {
  data: FleetTelemetry;
}

export function FleetOverview({ data }: FleetOverviewProps) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Users className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">Kernels</span>
            </div>
            <p className="text-2xl font-bold mt-1" data-testid="text-kernel-count">
              {data.kernels}
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Cpu className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">Capabilities</span>
            </div>
            <p className="text-2xl font-bold mt-1" data-testid="text-capability-count">
              {data.total_capabilities}
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">Invocations</span>
            </div>
            <p className="text-2xl font-bold mt-1" data-testid="text-invocation-count">
              {data.total_invocations}
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">Success Rate</span>
            </div>
            <p className="text-2xl font-bold mt-1" data-testid="text-success-rate">
              {Math.round(data.fleet_success_rate * PERCENT_MULTIPLIER)}%
            </p>
          </CardContent>
        </Card>
      </div>
      
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Category Distribution</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {Object.entries(data.category_distribution).map(([cat, count]) => {
              const Icon = getIcon(cat);
              const colorClass = getColor(cat);
              return (
                <Badge 
                  key={cat} 
                  variant="secondary" 
                  className={`${colorClass} gap-1`}
                  data-testid={`badge-category-${cat}`}
                >
                  <Icon className="h-3 w-3" />
                  {cat}: {count}
                </Badge>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
