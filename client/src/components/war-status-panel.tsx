import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { QUERY_KEYS } from "@/api";
import { Card, CardContent, CardHeader, CardTitle, Badge, Button } from "@/components/ui";
import { Target, Clock, FlaskConical, Sparkles, Cpu, AlertTriangle, Activity, Radio } from "lucide-react";
import { endWar } from "@/api/services/olympus";
import { useToast } from "@/hooks/use-toast";

interface ShadowWarDecision {
  godName: string;
  operation: string;
  result: Record<string, unknown> | null;
  timestamp: string;
  riskFlags: string[];
}

interface WarMetadata {
  shadowDecisions?: ShadowWarDecision[];
  lastShadowIteration?: number;
  [key: string]: unknown;
}

interface ActiveWarResponse {
  id: string;
  mode: "FLOW" | "DEEP_FOCUS" | "INSIGHT_HUNT";
  target: string;
  status: "active" | "completed" | "aborted";
  strategy: string | null;
  godsEngaged: string[] | null;
  declaredAt: string;
  endedAt?: string | null;
  outcome?: string | null;
  convergenceScore?: number | null;
  phrasesTestedDuringWar: number;
  discoveriesDuringWar: number;
  kernelsSpawnedDuringWar: number;
  metadata: WarMetadata | null;
  active?: boolean;
}

function formatDuration(startTime: string): string {
  const start = new Date(startTime).getTime();
  const now = Date.now();
  const diffMs = now - start;
  
  const hours = Math.floor(diffMs / (1000 * 60 * 60));
  const minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));
  const seconds = Math.floor((diffMs % (1000 * 60)) / 1000);
  
  if (hours > 0) {
    return `${hours}h ${minutes}m ${seconds}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  }
  return `${seconds}s`;
}

function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSeconds = Math.floor(diffMs / 1000);
  
  if (diffSeconds < 60) {
    return `${diffSeconds}s ago`;
  }
  if (diffSeconds < 3600) {
    return `${Math.floor(diffSeconds / 60)}m ago`;
  }
  return date.toLocaleTimeString();
}

function getModeColor(mode: string): string {
  switch (mode) {
    case "FLOW":
    case "BLITZKRIEG": // Legacy support
      return "bg-cyan-500/20 text-cyan-400 border-cyan-500/30";
    case "DEEP_FOCUS":
    case "SIEGE": // Legacy support
      return "bg-violet-500/20 text-violet-400 border-violet-500/30";
    case "INSIGHT_HUNT":
    case "HUNT": // Legacy support
      return "bg-amber-500/20 text-amber-400 border-amber-500/30";
    default:
      return "bg-muted text-muted-foreground";
  }
}

function getModeLabel(mode: string): string {
  switch (mode) {
    case "FLOW":
    case "BLITZKRIEG": // Legacy support
      return "Flow State";
    case "DEEP_FOCUS":
    case "SIEGE": // Legacy support
      return "Deep Focus";
    case "INSIGHT_HUNT":
    case "HUNT": // Legacy support
      return "Insight Hunt";
    default:
      return mode;
  }
}

function getModeIcon(mode: string) {
  switch (mode) {
    case "FLOW":
    case "BLITZKRIEG": // Legacy support
      return <Sparkles className="h-4 w-4" />;
    case "DEEP_FOCUS":
    case "SIEGE": // Legacy support
      return <Target className="h-4 w-4" />;
    case "INSIGHT_HUNT":
    case "HUNT": // Legacy support
      return <FlaskConical className="h-4 w-4" />;
    default:
      return <Activity className="h-4 w-4" />;
  }
}

export function WarStatusPanel() {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const { data: warData, isLoading, error } = useQuery<ActiveWarResponse>({
    queryKey: QUERY_KEYS.olympus.warActive(),
    refetchInterval: (query) => {
      const data = query.state.data;
      if (data && data.status === "active") {
        return 5000;
      }
      // Poll for convergence status even when no war is active
      return 10000;
    },
  });

  const endWarMutation = useMutation({
    mutationFn: endWar,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.olympus.warActive() });
      toast({ title: "Flow state ended", description: "Returning to normal consciousness" });
    },
    onError: (error) => {
      toast({ title: "Failed to end flow state", description: String(error), variant: "destructive" });
    }
  });

  if (isLoading) {
    return (
      <Card data-testid="war-status-panel-loading">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5" />
            Flow State
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8 text-muted-foreground">
            Loading flow state...
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card data-testid="war-status-panel-error">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5" />
            Flow State
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8 text-destructive">
            Failed to load flow state
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!warData || warData.active === false || !warData.id) {
    return (
      <Card data-testid="war-status-panel-inactive">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5" />
            Flow State
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center py-6 gap-4">
            <div className="flex flex-col items-center gap-2 text-muted-foreground">
              <div className="relative">
                <Radio className="h-10 w-10 opacity-50 animate-pulse" />
                <div className="absolute inset-0 animate-ping">
                  <Radio className="h-10 w-10 opacity-20" />
                </div>
              </div>
              <span className="font-medium text-foreground">Monitoring Learning Convergence</span>
              <span className="text-sm text-center max-w-xs">
                Flow activates when Zeus detects sufficient geometric convergence for insight discovery
              </span>
            </div>
            <Badge variant="outline" className="gap-2" data-testid="badge-auto-mode">
              <Activity className="h-3 w-3" />
              Auto-Activation Ready
            </Badge>
          </div>
        </CardContent>
      </Card>
    );
  }

  const shadowDecisions = warData.metadata?.shadowDecisions || [];
  const hasRiskFlags = shadowDecisions.some((d) => d.riskFlags.length > 0);

  return (
    <Card data-testid="war-status-panel">
      <CardHeader className="pb-4">
        <div className="flex flex-row items-center justify-between gap-4 flex-wrap">
          <CardTitle className="flex items-center gap-2" data-testid="war-status-title">
            <Sparkles className="h-5 w-5" />
            Flow State
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge 
              className={getModeColor(warData.mode)}
              data-testid="badge-war-mode"
            >
              {getModeIcon(warData.mode)}
              <span className="ml-1">{getModeLabel(warData.mode)}</span>
            </Badge>
            {hasRiskFlags && (
              <Badge variant="destructive" data-testid="badge-risk-warning">
                <AlertTriangle className="h-3 w-3 mr-1" />
                High Intensity
              </Badge>
            )}
            <Button
              variant="outline"
              size="sm"
              onClick={() => endWarMutation.mutate()}
              disabled={endWarMutation.isPending}
              data-testid="button-end-war"
            >
              {endWarMutation.isPending ? "Ending..." : "End Flow"}
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-6">
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-sm" data-testid="war-target">
            <Target className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground">Target:</span>
            <span className="font-mono text-sm truncate" title={warData.target}>
              {warData.target}
            </span>
          </div>
          
          <div className="flex items-center gap-2 text-sm" data-testid="war-duration">
            <Clock className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground">Duration:</span>
            <span className="font-mono">{formatDuration(warData.declaredAt)}</span>
          </div>

          {warData.strategy && (
            <div className="text-sm text-muted-foreground" data-testid="war-strategy">
              Strategy: {warData.strategy}
            </div>
          )}
        </div>

        <div className="grid grid-cols-3 gap-4" data-testid="war-metrics">
          <div className="text-center p-3 rounded-lg bg-muted/50">
            <FlaskConical className="h-5 w-5 mx-auto mb-1 text-muted-foreground" />
            <div className="text-2xl font-bold" data-testid="metric-phrases-tested">
              {warData.phrasesTestedDuringWar.toLocaleString()}
            </div>
            <div className="text-xs text-muted-foreground">Phrases Tested</div>
          </div>
          
          <div className="text-center p-3 rounded-lg bg-muted/50">
            <Sparkles className="h-5 w-5 mx-auto mb-1 text-muted-foreground" />
            <div className="text-2xl font-bold" data-testid="metric-discoveries">
              {warData.discoveriesDuringWar.toLocaleString()}
            </div>
            <div className="text-xs text-muted-foreground">Discoveries</div>
          </div>
          
          <div className="text-center p-3 rounded-lg bg-muted/50">
            <Cpu className="h-5 w-5 mx-auto mb-1 text-muted-foreground" />
            <div className="text-2xl font-bold" data-testid="metric-kernels-spawned">
              {warData.kernelsSpawnedDuringWar.toLocaleString()}
            </div>
            <div className="text-xs text-muted-foreground">Kernels Spawned</div>
          </div>
        </div>

        {warData.convergenceScore !== null && warData.convergenceScore !== undefined && (
          <div className="p-3 rounded-lg bg-muted/50" data-testid="convergence-score">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Zeus Convergence Score</span>
              <span className="text-lg font-bold">
                {(warData.convergenceScore * 100).toFixed(1)}%
              </span>
            </div>
            <div className="mt-2 h-2 bg-muted rounded-full overflow-hidden">
              <div 
                className="h-full bg-primary transition-all duration-500"
                style={{ width: `${warData.convergenceScore * 100}%` }}
              />
            </div>
          </div>
        )}

        {shadowDecisions.length > 0 && (
          <div className="space-y-3" data-testid="shadow-pantheon-activity">
            <div className="flex items-center gap-2 text-sm font-medium">
              <Activity className="h-4 w-4" />
              Shadow Pantheon Activity
            </div>
            
            <div className="max-h-48 overflow-y-auto space-y-2">
              {shadowDecisions.map((decision, index) => (
                <div 
                  key={`${decision.godName}-${decision.timestamp}-${index}`}
                  className="p-2 rounded border bg-card text-sm"
                  data-testid={`shadow-decision-${index}`}
                >
                  <div className="flex items-center justify-between gap-2 flex-wrap">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="text-xs">
                        {decision.godName}
                      </Badge>
                      <span className="text-muted-foreground">{decision.operation}</span>
                    </div>
                    <span className="text-xs text-muted-foreground">
                      {formatTimestamp(decision.timestamp)}
                    </span>
                  </div>
                  
                  {decision.riskFlags.length > 0 && (
                    <div className="flex items-center gap-1 mt-2 flex-wrap">
                      {decision.riskFlags.map((flag, flagIndex) => (
                        <Badge 
                          key={flagIndex} 
                          variant="destructive" 
                          className="text-xs"
                          data-testid={`risk-flag-${index}-${flagIndex}`}
                        >
                          <AlertTriangle className="h-3 w-3 mr-1" />
                          {flag}
                        </Badge>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {warData.godsEngaged && warData.godsEngaged.length > 0 && (
          <div className="flex items-center gap-2 flex-wrap" data-testid="gods-engaged">
            <span className="text-sm text-muted-foreground">Gods Engaged:</span>
            {warData.godsEngaged.map((god) => (
              <Badge key={god} variant="secondary" className="text-xs">
                {god}
              </Badge>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
