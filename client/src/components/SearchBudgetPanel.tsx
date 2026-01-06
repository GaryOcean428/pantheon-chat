import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { API_ROUTES } from "@/api/routes";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Switch,
  Label,
  Input,
  Button,
  Badge,
  Progress,
  Skeleton,
} from "@/components/ui";
import { useToast } from "@/hooks/use-toast";
import { Search, AlertTriangle, CheckCircle2, DollarSign, Zap, Globe, Brain, RefreshCw } from "lucide-react";
import { useState } from "react";

interface ProviderStatus {
  enabled: boolean;
  daily_limit: number;
  used_today: number;
  remaining: number;
  cost_per_query: number;
  has_api_key: boolean;
}

interface BudgetStatus {
  providers: Record<string, ProviderStatus>;
  allow_overage: boolean;
  recommendation: string;
  date: string;
}

interface LearningMetrics {
  total_outcomes: number;
  efficacy_scores: Record<string, number>;
  average_relevance: number;
  budget_efficiency: number;
}

export function SearchBudgetPanel() {
  const { toast } = useToast();
  const [editingLimits, setEditingLimits] = useState<Record<string, number>>({});

  const { data: budgetStatus, isLoading: statusLoading, refetch: refetchStatus } = useQuery<BudgetStatus>({
    queryKey: [API_ROUTES.searchBudget.status],
    refetchInterval: 30000,
  });

  const { data: learningMetrics } = useQuery<LearningMetrics>({
    queryKey: [API_ROUTES.searchBudget.learning],
    refetchInterval: 60000,
  });

  const toggleMutation = useMutation({
    mutationFn: async ({ provider, enabled }: { provider: string; enabled: boolean }) => {
      return apiRequest("POST", API_ROUTES.searchBudget.toggle, { provider, enabled });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [API_ROUTES.searchBudget.status] });
      toast({ title: "Provider Updated", description: "Search provider toggled successfully." });
    },
    onError: (error: Error) => {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    },
  });

  const limitsMutation = useMutation({
    mutationFn: async (limits: Record<string, number>) => {
      return apiRequest("POST", API_ROUTES.searchBudget.limits, limits);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [API_ROUTES.searchBudget.status] });
      setEditingLimits({});
      toast({ title: "Limits Updated", description: "Daily limits saved successfully." });
    },
    onError: (error: Error) => {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    },
  });

  const overageMutation = useMutation({
    mutationFn: async (allow: boolean) => {
      return apiRequest("POST", API_ROUTES.searchBudget.overage, { allow });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [API_ROUTES.searchBudget.status] });
      toast({ title: "Overage Setting Updated" });
    },
    onError: (error: Error) => {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    },
  });

  if (statusLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="h-5 w-5" />
            Search Budget
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-40 w-full" />
        </CardContent>
      </Card>
    );
  }

  const providers = budgetStatus?.providers || {};
  const providerIcons: Record<string, typeof Globe> = {
    duckduckgo: Search,
    google: Globe,
    perplexity: Brain,
    tavily: Zap,
  };

  const providerLabels: Record<string, string> = {
    duckduckgo: "DuckDuckGo (Free)",
    google: "Google Search",
    perplexity: "Perplexity",
    tavily: "Tavily",
  };

  const getUsagePercentage = (used: number, limit: number) => {
    if (limit <= 0) return 0;
    return Math.min(100, (used / limit) * 100);
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Search className="h-5 w-5" />
            <CardTitle>Search Budget</CardTitle>
          </div>
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={() => refetchStatus()}
            data-testid="button-refresh-budget"
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
        <CardDescription>
          Manage search provider limits and track usage. Date: {budgetStatus?.date || "N/A"}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {budgetStatus?.recommendation && (
          <div className="flex items-center gap-2 p-3 rounded-md bg-muted">
            <Brain className="h-4 w-4 text-primary" />
            <span className="text-sm">{budgetStatus.recommendation}</span>
          </div>
        )}

        <div className="space-y-4">
          {Object.entries(providers).map(([provider, status]) => {
            const Icon = providerIcons[provider] || Search;
            const label = providerLabels[provider] || provider;
            const usagePercent = getUsagePercentage(status.used_today, status.daily_limit);
            const isEditing = provider in editingLimits;

            return (
              <div 
                key={provider} 
                className="p-4 border rounded-lg space-y-3"
                data-testid={`provider-card-${provider}`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Icon className="h-5 w-5 text-muted-foreground" />
                    <div>
                      <div className="font-medium">{label}</div>
                      {status.cost_per_query > 0 && (
                        <div className="text-xs text-muted-foreground flex items-center gap-1">
                          <DollarSign className="h-3 w-3" />
                          ${(status.cost_per_query / 100).toFixed(3)}/query
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    {!status.has_api_key && provider !== "duckduckgo" && (
                      <Badge variant="outline" className="text-yellow-600">
                        <AlertTriangle className="h-3 w-3 mr-1" />
                        No API Key
                      </Badge>
                    )}
                    <Switch
                      checked={status.enabled}
                      disabled={!status.has_api_key && provider !== "duckduckgo"}
                      onCheckedChange={(checked) => 
                        toggleMutation.mutate({ provider, enabled: checked })
                      }
                      data-testid={`toggle-${provider}`}
                    />
                  </div>
                </div>

                {status.enabled && (
                  <>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">
                          {status.used_today} / {status.daily_limit === -1 ? "∞" : status.daily_limit} today
                        </span>
                        <span className="text-muted-foreground">
                          {status.remaining === -1 ? "Unlimited" : `${status.remaining} remaining`}
                        </span>
                      </div>
                      {status.daily_limit > 0 && (
                        <Progress 
                          value={usagePercent} 
                          className={usagePercent > 80 ? "bg-yellow-200" : ""}
                        />
                      )}
                    </div>

                    {provider !== "duckduckgo" && (
                      <div className="flex items-center gap-2">
                        <Label className="text-sm text-muted-foreground">Daily Limit:</Label>
                        <Input
                          type="number"
                          className="w-24 h-8"
                          value={isEditing ? editingLimits[provider] : status.daily_limit}
                          onChange={(e) => setEditingLimits({
                            ...editingLimits,
                            [provider]: parseInt(e.target.value) || 0
                          })}
                          min={0}
                          data-testid={`input-limit-${provider}`}
                        />
                        {isEditing && (
                          <Button
                            size="sm"
                            onClick={() => limitsMutation.mutate({ [provider]: editingLimits[provider] })}
                            disabled={limitsMutation.isPending}
                            data-testid={`button-save-limit-${provider}`}
                          >
                            Save
                          </Button>
                        )}
                      </div>
                    )}
                  </>
                )}
              </div>
            );
          })}
        </div>

        <div className="flex items-center justify-between p-4 border rounded-lg">
          <div>
            <Label className="font-medium">Allow Budget Overage</Label>
            <p className="text-sm text-muted-foreground">
              Continue using paid providers after daily limit is reached
            </p>
          </div>
          <Switch
            checked={budgetStatus?.allow_overage || false}
            onCheckedChange={(checked) => overageMutation.mutate(checked)}
            data-testid="toggle-overage"
          />
        </div>

        {learningMetrics && (
          <div className="pt-4 border-t space-y-2">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-green-500" />
              <span className="text-sm font-medium">Learning Metrics</span>
            </div>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Total Outcomes:</span>{" "}
                {learningMetrics.total_outcomes}
              </div>
              <div>
                <span className="text-muted-foreground">Avg Relevance:</span>{" "}
                {(learningMetrics.average_relevance * 100).toFixed(1)}%
              </div>
              <div>
                <span className="text-muted-foreground">Budget Efficiency:</span>{" "}
                {(learningMetrics.budget_efficiency * 100).toFixed(1)}%
              </div>
              <div>
                <span className="text-muted-foreground">Efficacy Scores:</span>
                <div className="flex gap-1 flex-wrap mt-1">
                  {Object.entries(learningMetrics.efficacy_scores || {}).map(([p, score]) => (
                    <Badge key={p} variant="secondary" className="text-xs">
                      {p}: {(score * 100).toFixed(0)}%
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default SearchBudgetPanel;
