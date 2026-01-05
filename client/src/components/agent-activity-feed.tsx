import { useQuery } from "@tanstack/react-query";
import { useEffect, useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { 
  Search, 
  Globe, 
  BookOpen, 
  Sparkles, 
  RefreshCw,
  Activity,
  Clock,
  ExternalLink
} from "lucide-react";
import { cn } from "@/lib/utils";
import { usePythonStatus } from "@/hooks/use-python-status";
import { PythonStatusBanner } from "./python-status-banner";

interface AgentActivity {
  id: number;
  activity_type: string;
  agent_id?: string;
  agent_name?: string;
  title: string;
  description?: string;
  source_url?: string;
  search_query?: string;
  provider?: string;
  result_count?: number;
  phi?: number;
  metadata?: Record<string, unknown>;
  created_at: string;
}

interface AgentActivityResponse {
  success: boolean;
  activities: AgentActivity[];
  count: number;
}

const ACTIVITY_ICONS: Record<string, typeof Search> = {
  search_started: Search,
  search_completed: Search,
  source_discovered: Globe,
  source_scraped: Globe,
  content_learned: BookOpen,
  curriculum_loaded: BookOpen,
  kernel_spawned: Sparkles,
  kernel_activated: Sparkles,
  research_initiated: Activity,
  pattern_recognized: Sparkles,
};

const ACTIVITY_COLORS: Record<string, string> = {
  search_started: "bg-blue-500/10 text-blue-600 dark:text-blue-400",
  search_completed: "bg-green-500/10 text-green-600 dark:text-green-400",
  source_discovered: "bg-purple-500/10 text-purple-600 dark:text-purple-400",
  source_scraped: "bg-indigo-500/10 text-indigo-600 dark:text-indigo-400",
  content_learned: "bg-amber-500/10 text-amber-600 dark:text-amber-400",
  curriculum_loaded: "bg-cyan-500/10 text-cyan-600 dark:text-cyan-400",
  kernel_spawned: "bg-pink-500/10 text-pink-600 dark:text-pink-400",
  kernel_activated: "bg-rose-500/10 text-rose-600 dark:text-rose-400",
  research_initiated: "bg-teal-500/10 text-teal-600 dark:text-teal-400",
  pattern_recognized: "bg-violet-500/10 text-violet-600 dark:text-violet-400",
};

function formatTimeAgo(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);
  
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

function ActivityCard({ activity }: { activity: AgentActivity }) {
  const Icon = ACTIVITY_ICONS[activity.activity_type] || Activity;
  const colorClass = ACTIVITY_COLORS[activity.activity_type] || "bg-gray-500/10 text-gray-600";
  
  return (
    <div 
      className="flex gap-3 p-3 rounded-lg hover-elevate border border-transparent hover:border-border/50 transition-colors"
      data-testid={`activity-card-${activity.id}`}
    >
      <div className={cn("p-2 rounded-lg shrink-0", colorClass)}>
        <Icon className="h-4 w-4" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-start justify-between gap-2">
          <p className="text-sm font-medium truncate" data-testid={`text-activity-title-${activity.id}`}>
            {activity.title}
          </p>
          <span className="text-xs text-muted-foreground shrink-0 flex items-center gap-1">
            <Clock className="h-3 w-3" />
            {formatTimeAgo(activity.created_at)}
          </span>
        </div>
        {activity.description && (
          <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
            {activity.description}
          </p>
        )}
        <div className="flex flex-wrap items-center gap-2 mt-2">
          {activity.agent_name && (
            <Badge variant="outline" className="text-xs">
              {activity.agent_name}
            </Badge>
          )}
          {activity.provider && (
            <Badge variant="secondary" className="text-xs">
              {activity.provider}
            </Badge>
          )}
          {activity.result_count !== undefined && activity.result_count !== null && (
            <span className="text-xs text-muted-foreground">
              {activity.result_count} {activity.activity_type.includes('search') ? 'results' : 'items'}
            </span>
          )}
          {activity.source_url && (
            <a 
              href={activity.source_url} 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-xs text-primary flex items-center gap-1 hover:underline"
              data-testid={`link-activity-source-${activity.id}`}
            >
              <ExternalLink className="h-3 w-3" />
              View
            </a>
          )}
        </div>
      </div>
    </div>
  );
}

interface AgentActivityFeedProps {
  limit?: number;
  showHeader?: boolean;
  className?: string;
}

export function AgentActivityFeed({ limit = 50, showHeader = true, className }: AgentActivityFeedProps) {
  const { isReady, isInitializing } = usePythonStatus();
  const [useSSE, setUseSSE] = useState(true);
  const [sseActivities, setSSEActivities] = useState<AgentActivity[]>([]);
  
  const { data, isLoading, refetch } = useQuery<AgentActivityResponse>({
    queryKey: ['/api/research/activity', { limit }],
    enabled: isReady,
    refetchInterval: useSSE ? false : 10000,
    staleTime: 5000,
  });
  
  useEffect(() => {
    if (!isReady || !useSSE) return;
    
    let eventSource: EventSource | null = null;
    
    const connect = () => {
      eventSource = new EventSource('/api/research/activity/stream');
      
      eventSource.onmessage = (event) => {
        try {
          const activity = JSON.parse(event.data) as AgentActivity;
          if (activity.id) {
            setSSEActivities(prev => {
              const exists = prev.some(a => a.id === activity.id);
              if (exists) return prev;
              return [activity, ...prev].slice(0, limit);
            });
          }
        } catch (e) {
          console.error('[ActivityFeed] Failed to parse SSE:', e);
        }
      };
      
      eventSource.onerror = () => {
        eventSource?.close();
        setTimeout(connect, 5000);
      };
    };
    
    connect();
    
    return () => {
      eventSource?.close();
    };
  }, [isReady, useSSE, limit]);
  
  const allActivities = [
    ...sseActivities,
    ...(data?.activities || []).filter(a => !sseActivities.some(s => s.id === a.id))
  ].slice(0, limit);
  
  const handleRefresh = useCallback(() => {
    setSSEActivities([]);
    refetch();
  }, [refetch]);
  
  if (isInitializing) {
    return (
      <Card className={className}>
        {showHeader && (
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Agent Discovery Activity
            </CardTitle>
          </CardHeader>
        )}
        <CardContent>
          <PythonStatusBanner />
        </CardContent>
      </Card>
    );
  }
  
  return (
    <Card className={className} data-testid="card-agent-activity">
      {showHeader && (
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between gap-2">
            <CardTitle className="text-lg flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Agent Discovery Activity
            </CardTitle>
            <Button 
              variant="ghost" 
              size="icon" 
              onClick={handleRefresh}
              disabled={isLoading}
              data-testid="button-refresh-activity"
            >
              <RefreshCw className={cn("h-4 w-4", isLoading && "animate-spin")} />
            </Button>
          </div>
        </CardHeader>
      )}
      <CardContent className="p-0">
        {allActivities.length === 0 ? (
          <div className="p-6 text-center text-muted-foreground">
            <Activity className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No recent activity</p>
            <p className="text-xs mt-1">Agent discoveries will appear here</p>
          </div>
        ) : (
          <ScrollArea className="h-[400px]">
            <div className="divide-y divide-border/50 px-4">
              {allActivities.map((activity) => (
                <ActivityCard key={activity.id} activity={activity} />
              ))}
            </div>
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  );
}
