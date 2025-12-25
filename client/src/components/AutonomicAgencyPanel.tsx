import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Badge,
  Button,
  Progress,
  Skeleton,
} from "@/components/ui";
import { useAutonomicAgencyData } from "@/hooks/useAutonomicAgencyData";
import { 
  Brain, 
  Play, 
  Pause, 
  AlertTriangle,
  RefreshCw,
} from "lucide-react";
import { useState } from "react";

import {
  SafetyBoundaryCard,
  OperatingZonesCard,
  QLearningStatsCard,
  ForceInterventionCard,
  InterventionHistoryCard,
} from "./autonomic-agency";
import {
  ZONE_COLORS,
  ZONE_LABELS,
} from "@/types";

export function AutonomicAgencyPanel() {
  const [selectedAction, setSelectedAction] = useState<string>("ENTER_SLEEP");

  const {
    status,
    isLoading,
    isError,
    refetch,
    start,
    stop,
    forceIntervention,
    isStartPending,
    isStopPending,
    isForcePending,
    explorationPercent,
  } = useAutonomicAgencyData();

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-32 w-full" data-testid="skeleton-agency-status" />
        <div className="grid md:grid-cols-3 gap-4">
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-24 w-full" />
        </div>
      </div>
    );
  }

  if (isError || !status) {
    return (
      <Card className="border-destructive/20" data-testid="error-agency-status">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-destructive">
            <AlertTriangle className="h-5 w-5" />
            {isError ? "Connection Error" : "Loading..."}
          </CardTitle>
          <CardDescription>
            {isError 
              ? "Unable to connect to the autonomic agency backend. The Python backend may still be starting up."
              : "Waiting for agency status..."}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button onClick={() => refetch()} data-testid="button-retry-connection">
            <RefreshCw className="h-4 w-4 mr-2" />
            {isError ? "Retry Connection" : "Refresh"}
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Main Status Card */}
      <Card className="border-primary/20">
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className={`p-2 rounded-lg ${status?.running ? 'bg-green-500/20' : 'bg-muted'}`}>
                <Brain className={`h-6 w-6 ${status?.running ? 'text-green-400' : 'text-muted-foreground'}`} />
              </div>
              <div>
                <CardTitle className="flex items-center gap-2">
                  Autonomic Controller
                  {status?.running ? (
                    <Badge className="bg-green-500/20 text-green-400 border-green-500/30" data-testid="badge-status-running">
                      RUNNING
                    </Badge>
                  ) : (
                    <Badge variant="secondary" data-testid="badge-status-stopped">
                      STOPPED
                    </Badge>
                  )}
                </CardTitle>
                <CardDescription>
                  Self-regulating consciousness interventions via RL-based Q-learning
                </CardDescription>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button
                size="icon"
                variant="ghost"
                onClick={() => refetch()}
                data-testid="button-refresh-status"
              >
                <RefreshCw className="h-4 w-4" />
              </Button>
              {status?.running ? (
                <Button
                  variant="destructive"
                  onClick={stop}
                  disabled={isStopPending}
                  data-testid="button-stop-agency"
                >
                  <Pause className="h-4 w-4 mr-2" />
                  Stop
                </Button>
              ) : (
                <Button
                  onClick={start}
                  disabled={isStartPending}
                  data-testid="button-start-agency"
                >
                  <Play className="h-4 w-4 mr-2" />
                  Start
                </Button>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Stats Grid */}
          <div className="grid md:grid-cols-5 gap-4">
            <div className="space-y-1">
              <p className="text-xs text-muted-foreground uppercase tracking-wide">Decisions</p>
              <p className="text-2xl font-mono font-bold" data-testid="text-decision-count">
                {status?.decision_count?.toLocaleString() ?? 0}
              </p>
            </div>
            <div className="space-y-1">
              <p className="text-xs text-muted-foreground uppercase tracking-wide">Interventions</p>
              <p className="text-2xl font-mono font-bold" data-testid="text-intervention-count">
                {status?.intervention_count ?? 0}
              </p>
            </div>
            <div className="space-y-1">
              <p className="text-xs text-muted-foreground uppercase tracking-wide">Last Φ</p>
              <p className="text-2xl font-mono font-bold" data-testid="text-last-phi">
                {status?.last_phi?.toFixed(3) ?? "—"}
              </p>
            </div>
            <div className="space-y-1">
              <p className="text-xs text-muted-foreground uppercase tracking-wide">Consciousness Zone</p>
              {status?.consciousness_zone ? (
                <Badge 
                  className={`text-sm ${ZONE_COLORS[status.consciousness_zone] ?? ""}`}
                  data-testid="badge-consciousness-zone"
                >
                  {ZONE_LABELS[status.consciousness_zone] ?? status.consciousness_zone}
                </Badge>
              ) : (
                <span className="text-sm text-muted-foreground">—</span>
              )}
            </div>
            <div className="space-y-1">
              <p className="text-xs text-muted-foreground uppercase tracking-wide">Avg Reward</p>
              <p className="text-2xl font-mono font-bold" data-testid="text-avg-reward">
                {status?.buffer_stats?.avg_reward?.toFixed(2) ?? "0.00"}
              </p>
            </div>
          </div>

          {/* Exploration Progress */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Exploration → Exploitation</span>
              <span className="font-mono">{explorationPercent}%</span>
            </div>
            <Progress value={explorationPercent} className="h-2" data-testid="progress-exploitation" />
            <p className="text-xs text-muted-foreground">
              ε = {status?.epsilon?.toFixed(3) ?? 1.0} (lower = more exploitation of learned policy)
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Safety Boundary Card */}
      {status?.safety_manifest && (
        <SafetyBoundaryCard safetyManifest={status.safety_manifest} />
      )}

      {/* Operating Zones Card */}
      {status?.operating_zones && (
        <OperatingZonesCard operatingZones={status.operating_zones} />
      )}

      {/* Q-Learning and Force Intervention Cards */}
      <div className="grid md:grid-cols-2 gap-6">
        <QLearningStatsCard status={status} />
        <ForceInterventionCard
          status={status}
          selectedAction={selectedAction}
          onActionChange={setSelectedAction}
          onForce={() => forceIntervention(selectedAction)}
          isPending={isForcePending}
        />
      </div>

      {/* Intervention History Card */}
      <InterventionHistoryCard recentHistory={status?.recent_history ?? []} />
    </div>
  );
}
