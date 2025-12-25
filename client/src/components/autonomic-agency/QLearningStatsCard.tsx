import { Card, CardContent, CardHeader, CardTitle, Badge } from "@/components/ui";
import { BarChart3 } from "lucide-react";
import type { AgencyStatus } from "./types";

interface QLearningStatsCardProps {
  status: AgencyStatus;
}

export function QLearningStatsCard({ status }: QLearningStatsCardProps) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <BarChart3 className="h-4 w-4" />
          Q-Learning Stats
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Learning Rate</p>
            <p className="font-mono text-sm" data-testid="text-learning-rate">
              {status?.optimizer_stats?.learning_rate ?? 0.001}
            </p>
          </div>
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Damping</p>
            <p className="font-mono text-sm" data-testid="text-damping">
              {status?.optimizer_stats?.damping ?? 0.0001}
            </p>
          </div>
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Update Count</p>
            <p className="font-mono text-sm" data-testid="text-update-count">
              {status?.optimizer_stats?.update_count ?? 0}
            </p>
          </div>
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Buffer Size</p>
            <p className="font-mono text-sm" data-testid="text-buffer-size">
              {status?.buffer_size ?? 0}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Fisher Matrix:</span>
          {status?.optimizer_stats?.has_fisher ? (
            <Badge className="bg-green-500/20 text-green-400">Computed</Badge>
          ) : (
            <Badge variant="secondary">Pending</Badge>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
