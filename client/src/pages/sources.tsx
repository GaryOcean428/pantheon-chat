import { useState } from "react";


import { useQuery, useMutation } from "@tanstack/react-query";
import {
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Badge,
  Input,
  Skeleton,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Switch,
} from "@/components/ui";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { Globe, Plus, Trash2, RefreshCw, Activity, TrendingUp, Clock, ExternalLink, Search, Key, AlertTriangle } from "lucide-react";

interface Source {
  url: string;
  category: string;
  phi_avg: number;
  hit_count: number;
  origin: string;
  discovered_at: number;
  success_count: number;
  failure_count: number;
  last_used: number;
}

interface SourcesResponse {
  success: boolean;
  sources: Source[];
  total: number;
}

interface ProviderInfo {
  available: boolean;
  enabled: boolean;
  requires_key: boolean;
  has_key?: boolean;
}

interface ProvidersResponse {
  success: boolean;
  data: {
    google_free: ProviderInfo;
    tavily: ProviderInfo;
    perplexity: ProviderInfo;
    duckduckgo: ProviderInfo;
  };
}

const CATEGORIES = [
  { value: "research", label: "Research" },
  { value: "documentation", label: "Documentation" },
  { value: "forum", label: "Forum" },
  { value: "news", label: "News" },
  { value: "blog", label: "Blog" },
  { value: "api", label: "API" },
  { value: "manual", label: "Manual" },
];

function formatTimestamp(ts: number): string {
  if (!ts) return "Never";
  const date = new Date(ts * 1000);
  return date.toLocaleDateString() + " " + date.toLocaleTimeString();
}

function formatTimeAgo(ts: number): string {
  if (!ts) return "Never";
  const now = Date.now() / 1000;
  const diff = now - ts;
  if (diff < 60) return "Just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

export default function Sources() {
  const { toast } = useToast();
  const [newUrl, setNewUrl] = useState("");
  const [newCategory, setNewCategory] = useState("research");

  const { data, isLoading, refetch } = useQuery<SourcesResponse>({
    queryKey: ["/api/python/research/sources"],
  });

  const { data: providersData, isLoading: providersLoading, refetch: refetchProviders } = useQuery<ProvidersResponse>({
    queryKey: ["/api/search/providers"],
  });

  const toggleProviderMutation = useMutation({
    mutationFn: async ({ provider, enabled }: { provider: string; enabled: boolean }) => {
      const res = await apiRequest("POST", `/api/search/providers/${provider}/toggle`, { enabled });
      return res.json();
    },
    onSuccess: (data) => {
      if (data.success) {
        toast({ title: "Provider Updated", description: data.message });
        queryClient.invalidateQueries({ queryKey: ["/api/search/providers"] });
      } else {
        toast({ title: "Error", description: data.error, variant: "destructive" });
      }
    },
    onError: (err: Error) => {
      toast({ title: "Error", description: err.message, variant: "destructive" });
    },
  });

  const addSourceMutation = useMutation({
    mutationFn: async ({ url, category }: { url: string; category: string }) => {
      const res = await apiRequest("POST", "/api/python/research/sources", { url, category });
      return res.json();
    },
    onSuccess: (data) => {
      if (data.success) {
        toast({ title: "Source Added", description: `Added ${newUrl}` });
        setNewUrl("");
        queryClient.invalidateQueries({ queryKey: ["/api/python/research/sources"] });
      } else {
        toast({ title: "Error", description: data.error, variant: "destructive" });
      }
    },
    onError: (err: Error) => {
      toast({ title: "Error", description: err.message, variant: "destructive" });
    },
  });

  const deleteSourceMutation = useMutation({
    mutationFn: async (url: string) => {
      const encodedUrl = encodeURIComponent(url);
      const res = await apiRequest("DELETE", `/api/python/research/sources/${encodedUrl}`);
      return res.json();
    },
    onSuccess: (data) => {
      if (data.success) {
        toast({ title: "Source Removed" });
        queryClient.invalidateQueries({ queryKey: ["/api/python/research/sources"] });
      } else {
        toast({ title: "Error", description: data.error, variant: "destructive" });
      }
    },
    onError: (err: Error) => {
      toast({ title: "Error", description: err.message, variant: "destructive" });
    },
  });

  const handleAddSource = () => {
    if (!newUrl.trim()) {
      toast({ title: "Error", description: "URL is required", variant: "destructive" });
      return;
    }
    if (!newUrl.startsWith("http")) {
      toast({ title: "Error", description: "URL must start with http:// or https://", variant: "destructive" });
      return;
    }
    addSourceMutation.mutate({ url: newUrl.trim(), category: newCategory });
  };

  const sources = data?.sources ?? [];

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-6xl mx-auto space-y-8">
        <div className="flex items-center justify-between">
          <div className="space-y-2">
            <h1 className="text-4xl font-bold flex items-center gap-3" data-testid="text-sources-title">
              <Globe className="h-8 w-8 text-primary" />
              Indexing Sources
            </h1>
            <p className="text-muted-foreground">
              Manage sources for Shadow Search and research indexing
            </p>
          </div>
          <Button
            variant="outline"
            size="icon"
            onClick={() => { refetch(); refetchProviders(); }}
            data-testid="button-refresh-sources"
          >
            <RefreshCw className={`h-4 w-4 ${isLoading || providersLoading ? "animate-spin" : ""}`} />
          </Button>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Search className="h-5 w-5" />
              Search Providers
            </CardTitle>
            <CardDescription>Configure which search engines to use for Shadow Search</CardDescription>
          </CardHeader>
          <CardContent>
            {providersLoading ? (
              <div className="space-y-4">
                <Skeleton className="h-16 w-full" />
                <Skeleton className="h-16 w-full" />
              </div>
            ) : (
              <div className="grid gap-4 md:grid-cols-2">
                <div className="flex items-center justify-between p-4 rounded-lg border">
                  <div className="flex items-start gap-3">
                    <Search className="h-5 w-5 mt-0.5 text-orange-500" />
                    <div>
                      <div className="font-medium">DuckDuckGo</div>
                      <p className="text-sm text-muted-foreground">
                        Free privacy-focused search (always available)
                      </p>
                      <Badge variant="outline" className="mt-1 text-green-500 border-green-500">
                        Free
                      </Badge>
                    </div>
                  </div>
                  <Switch
                    checked={providersData?.data?.duckduckgo?.enabled ?? true}
                    disabled={toggleProviderMutation.isPending}
                    onCheckedChange={(checked) => 
                      toggleProviderMutation.mutate({ provider: 'duckduckgo', enabled: checked })
                    }
                    data-testid="switch-duckduckgo"
                  />
                </div>

                <div className="flex items-center justify-between p-4 rounded-lg border">
                  <div className="flex items-start gap-3">
                    <Globe className="h-5 w-5 mt-0.5 text-blue-500" />
                    <div>
                      <div className="font-medium">Google Free Search</div>
                      <p className="text-sm text-muted-foreground">
                        Web search via scraping (no API key needed)
                      </p>
                      {providersData?.data?.google_free?.available === false && (
                        <Badge variant="outline" className="mt-1 text-amber-500 border-amber-500">
                          <AlertTriangle className="h-3 w-3 mr-1" />
                          Unavailable
                        </Badge>
                      )}
                    </div>
                  </div>
                  <Switch
                    checked={providersData?.data?.google_free?.enabled ?? true}
                    disabled={!providersData?.data?.google_free?.available || toggleProviderMutation.isPending}
                    onCheckedChange={(checked) => 
                      toggleProviderMutation.mutate({ provider: 'google_free', enabled: checked })
                    }
                    data-testid="switch-google-free"
                  />
                </div>
                
                <div className="flex items-center justify-between p-4 rounded-lg border">
                  <div className="flex items-start gap-3">
                    <Key className="h-5 w-5 mt-0.5 text-purple-500" />
                    <div>
                      <div className="font-medium">Tavily Search</div>
                      <p className="text-sm text-muted-foreground">
                        AI-powered web search (requires API key)
                      </p>
                      {!providersData?.data?.tavily?.has_key && (
                        <Badge variant="outline" className="mt-1 text-amber-500 border-amber-500">
                          <Key className="h-3 w-3 mr-1" />
                          API Key Not Set
                        </Badge>
                      )}
                      {providersData?.data?.tavily?.has_key && providersData?.data?.tavily?.available && (
                        <Badge variant="outline" className="mt-1 text-green-500 border-green-500">
                          Ready
                        </Badge>
                      )}
                    </div>
                  </div>
                  <Switch
                    checked={providersData?.data?.tavily?.enabled ?? false}
                    disabled={!providersData?.data?.tavily?.has_key || toggleProviderMutation.isPending}
                    onCheckedChange={(checked) => 
                      toggleProviderMutation.mutate({ provider: 'tavily', enabled: checked })
                    }
                    data-testid="switch-tavily"
                  />
                </div>

                <div className="flex items-center justify-between p-4 rounded-lg border">
                  <div className="flex items-start gap-3">
                    <Key className="h-5 w-5 mt-0.5 text-cyan-500" />
                    <div>
                      <div className="font-medium">Perplexity</div>
                      <p className="text-sm text-muted-foreground">
                        AI-enhanced deep search (requires API key)
                      </p>
                      {!providersData?.data?.perplexity?.has_key && (
                        <Badge variant="outline" className="mt-1 text-amber-500 border-amber-500">
                          <Key className="h-3 w-3 mr-1" />
                          API Key Not Set
                        </Badge>
                      )}
                      {providersData?.data?.perplexity?.has_key && providersData?.data?.perplexity?.available && (
                        <Badge variant="outline" className="mt-1 text-green-500 border-green-500">
                          Ready
                        </Badge>
                      )}
                    </div>
                  </div>
                  <Switch
                    checked={providersData?.data?.perplexity?.enabled ?? false}
                    disabled={!providersData?.data?.perplexity?.has_key || toggleProviderMutation.isPending}
                    onCheckedChange={(checked) => 
                      toggleProviderMutation.mutate({ provider: 'perplexity', enabled: checked })
                    }
                    data-testid="switch-perplexity"
                  />
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Add New Source</CardTitle>
            <CardDescription>Add a URL to be indexed for research and knowledge discovery</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col sm:flex-row gap-4">
              <Input
                placeholder="https://example.com/research"
                value={newUrl}
                onChange={(e) => setNewUrl(e.target.value)}
                className="flex-1"
                data-testid="input-source-url"
              />
              <Select value={newCategory} onValueChange={setNewCategory}>
                <SelectTrigger className="w-40" data-testid="select-category">
                  <SelectValue placeholder="Category" />
                </SelectTrigger>
                <SelectContent>
                  {CATEGORIES.map((cat) => (
                    <SelectItem key={cat.value} value={cat.value}>
                      {cat.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button
                onClick={handleAddSource}
                disabled={addSourceMutation.isPending}
                data-testid="button-add-source"
              >
                <Plus className="h-4 w-4 mr-2" />
                Add Source
              </Button>
            </div>
          </CardContent>
        </Card>

        <div className="grid gap-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold">Active Sources ({sources.length})</h2>
          </div>

          {isLoading ? (
            <div className="grid gap-4">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-24 w-full" data-testid={`skeleton-source-${i}`} />
              ))}
            </div>
          ) : sources.length === 0 ? (
            <Card>
              <CardContent className="py-12 text-center">
                <Globe className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-medium mb-2">No Sources Yet</h3>
                <p className="text-muted-foreground">
                  Add your first indexing source to enable Shadow Search
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-4">
              {sources.map((source) => (
                <Card key={source.url} className="hover-elevate" data-testid={`card-source-${source.url}`}>
                  <CardContent className="py-4">
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-2">
                          <a
                            href={source.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="font-medium text-primary hover:underline truncate flex items-center gap-1"
                            data-testid={`link-source-${source.url}`}
                          >
                            {source.url}
                            <ExternalLink className="h-3 w-3 shrink-0" />
                          </a>
                        </div>
                        <div className="flex flex-wrap items-center gap-3 text-sm text-muted-foreground">
                          <Badge variant="secondary">{source.category}</Badge>
                          <span className="flex items-center gap-1">
                            <Activity className="h-3 w-3" />
                            {source.phi_avg.toFixed(3)}
                          </span>
                          <span className="flex items-center gap-1">
                            <TrendingUp className="h-3 w-3" />
                            {source.success_count} hits
                          </span>
                          <span className="flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {formatTimeAgo(source.last_used)}
                          </span>
                          <Badge variant="outline" className="text-xs">
                            {source.origin}
                          </Badge>
                        </div>
                      </div>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => deleteSourceMutation.mutate(source.url)}
                        disabled={deleteSourceMutation.isPending}
                        data-testid={`button-delete-${source.url}`}
                      >
                        <Trash2 className="h-4 w-4 text-destructive" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
