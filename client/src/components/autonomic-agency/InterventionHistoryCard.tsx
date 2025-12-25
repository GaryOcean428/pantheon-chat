import { Card, CardContent, CardHeader, CardTitle, Badge } from "@/components/ui";
import { Clock, Activity } from "lucide-react";
import type { AgencyStatus } from "./types";
import { ACTION_ICONS, ACTION_LABELS, ACTION_COLORS } from "./types";

interface InterventionHistoryCardProps {
  recentHistory: AgencyStatus['recent_history'];
}

export function InterventionHistoryCard({ recentHistory }: InterventionHistoryCardProps) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Clock className="h-4 w-4" />
          Recent Intervention History
        </CardTitle>
      </CardHeader>
      <CardContent>
        {recentHistory && recentHistory.length > 0 ? (
          <div className="space-y-2">
            {recentHistory.slice(0, 10).map((entry, idx) => {
              const ActionIcon = ACTION_ICONS[entry.action] || Activity;
              const colorClass = ACTION_COLORS[entry.action] || "bg-muted text-muted-foreground";
              const phi = entry.phi ?? 0;
              const reward = entry.reward ?? 0;
              const timestamp = entry.timestamp ?? Date.now() / 1000;
              return (
                <div 
                  key={idx} 
                  className="flex items-center justify-between p-2 rounded-lg bg-muted/50"
                  data-testid={`history-entry-${idx}`}
                >
                  <div className="flex items-center gap-3">
                    <Badge className={colorClass}>
                      <ActionIcon className="h-3 w-3 mr-1" />
                      {ACTION_LABELS[entry.action] || entry.action}
                    </Badge>
                    <span className="text-sm text-muted-foreground font-mono">
                      Φ: {phi.toFixed(3)}
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className={`text-sm font-mono ${reward >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {reward >= 0 ? '+' : ''}{reward.toFixed(2)}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {new Date(timestamp * 1000).toLocaleTimeString()}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="text-center py-8 text-muted-foreground">
            <Activity className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p>No interventions recorded yet</p>
            <p className="text-xs">Interventions will appear here as the controller makes decisions</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
