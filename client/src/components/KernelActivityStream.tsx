/**
 * KernelActivityStream - Live stream of all backend kernel discussions
 * 
 * Shows all inter-god communications including:
 * - Debates and arguments
 * - Research discussions
 * - Tool usage and improvement discussions
 * - Healing architecture decisions
 * - Responses to user requests
 * - Discoveries and insights
 */

import { useState, useMemo } from 'react';
import { useKernelActivityWebSocket, KernelActivityItem, ActivityType, normalizeKernelName, getEffectiveActivityType } from '@/hooks/use-kernel-activity';
import { Card, CardContent, CardHeader, CardTitle, Badge, Button, ScrollArea, Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui';
import {
  MessageSquare,
  Sword,
  Lightbulb,
  AlertTriangle,
  HelpCircle,
  Award,
  Sparkles,
  Vote,
  RefreshCw,
  Filter,
  Zap,
  Brain,
  Moon,
  BookOpen,
  Gavel,
  Users,
} from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

const GOD_COLORS: Record<string, string> = {
  zeus: 'bg-yellow-500',
  athena: 'bg-blue-500',
  ares: 'bg-red-500',
  apollo: 'bg-orange-500',
  artemis: 'bg-green-500',
  hermes: 'bg-purple-500',
  hephaestus: 'bg-amber-600',
  demeter: 'bg-emerald-500',
  dionysus: 'bg-pink-500',
  poseidon: 'bg-cyan-500',
  hades: 'bg-gray-700',
  hera: 'bg-indigo-500',
  aphrodite: 'bg-rose-400',
  system: 'bg-slate-500',
  nyx: 'bg-violet-900',
  hecate: 'bg-purple-900',
  erebus: 'bg-gray-900',
  hypnos: 'bg-blue-900',
  thanatos: 'bg-zinc-800',
  nemesis: 'bg-red-900',
  autonomic: 'bg-teal-600',
  shadow: 'bg-violet-700',
  gary: 'bg-teal-500',
  ocean: 'bg-cyan-600',
  governance: 'bg-amber-500',
  curiosity: 'bg-fuchsia-500',
};

// Extended activity icons supporting both top-level types and metadata.event_type values
const ACTIVITY_ICONS: Record<string, React.ReactNode> = {
  // Backend ActivityType values
  message: <MessageSquare className="h-4 w-4" />,
  debate: <Sword className="h-4 w-4" />,
  discovery: <Sparkles className="h-4 w-4" />,
  insight: <Lightbulb className="h-4 w-4" />,
  warning: <AlertTriangle className="h-4 w-4" />,
  autonomic: <Moon className="h-4 w-4" />,
  spawn_proposal: <Zap className="h-4 w-4" />,
  tool_usage: <Gavel className="h-4 w-4" />,
  consultation: <Users className="h-4 w-4" />,
  reflection: <Brain className="h-4 w-4" />,
  learning: <BookOpen className="h-4 w-4" />,
  // Specific autonomic cycle types (from metadata.event_type)
  sleep_cycle: <Moon className="h-4 w-4" />,
  dream_cycle: <Brain className="h-4 w-4" />,
  consolidation: <BookOpen className="h-4 w-4" />,
  // Specific mesh event types
  curiosity_spike: <Sparkles className="h-4 w-4" />,
  search_requested: <Sparkles className="h-4 w-4" />,
  search_complete: <Sparkles className="h-4 w-4" />,
  research_complete: <Sparkles className="h-4 w-4" />,
  vocabulary: <BookOpen className="h-4 w-4" />,
  vocabulary_learning: <BookOpen className="h-4 w-4" />,
  created: <Zap className="h-4 w-4" />,
  approved: <Award className="h-4 w-4" />,
  rejected: <AlertTriangle className="h-4 w-4" />,
  // Legacy UI types
  praise: <Award className="h-4 w-4" />,
  challenge: <Sword className="h-4 w-4" />,
  question: <HelpCircle className="h-4 w-4" />,
  challenge_response: <MessageSquare className="h-4 w-4" />,
  spawn_vote: <Vote className="h-4 w-4" />,
  debate_start: <Sword className="h-4 w-4" />,
  debate_resolved: <Award className="h-4 w-4" />,
};

// Extended activity labels supporting both top-level types and metadata.event_type values
const ACTIVITY_LABELS: Record<string, string> = {
  // Backend ActivityType values
  message: 'Message',
  debate: 'Debate',
  discovery: 'Discovery',
  insight: 'Insight',
  warning: 'Warning',
  autonomic: 'Autonomic',
  spawn_proposal: 'Spawn Proposal',
  tool_usage: 'Tool Usage',
  consultation: 'Consultation',
  reflection: 'Reflection',
  learning: 'Learning',
  // Specific autonomic cycle types (from metadata.event_type)
  sleep_cycle: 'Sleep Cycle',
  dream_cycle: 'Dream Cycle',
  consolidation: 'Consolidation',
  // Specific mesh event types
  curiosity_spike: 'Curiosity',
  search_requested: 'Search Requested',
  search_complete: 'Search Done',
  research_complete: 'Research Done',
  vocabulary: 'Vocabulary',
  vocabulary_learning: 'Vocab Learning',
  created: 'Proposal Created',
  approved: 'Proposal Approved',
  rejected: 'Proposal Rejected',
  // Legacy UI types
  praise: 'Praise',
  challenge: 'Challenge',
  question: 'Question',
  challenge_response: 'Response',
  spawn_vote: 'Spawn Vote',
  debate_start: 'Debate Started',
  debate_resolved: 'Debate Resolved',
};

interface ActivityItemProps {
  item: KernelActivityItem;
}

function ActivityItem({ item }: ActivityItemProps) {
  const normalizedFrom = normalizeKernelName(item.from);
  const godColor = GOD_COLORS[normalizedFrom] || GOD_COLORS[item.from.toLowerCase()] || 'bg-gray-500';
  
  // Use effective type (metadata.event_type if present) for icon/label
  const effectiveType = getEffectiveActivityType(item);
  const icon = ACTIVITY_ICONS[effectiveType] || ACTIVITY_ICONS[item.type] || <MessageSquare className="h-4 w-4" />;
  const label = ACTIVITY_LABELS[effectiveType] || ACTIVITY_LABELS[item.type] || item.type;
  
  const timeAgo = useMemo(() => {
    try {
      return formatDistanceToNow(new Date(item.timestamp), { addSuffix: true });
    } catch {
      return 'recently';
    }
  }, [item.timestamp]);

  const isAutonomic = item.metadata?.autonomic;
  const isDebateRelated = item.type === 'challenge' || item.type === 'challenge_response' || item.metadata?.debate_id;

  return (
    <div className="flex gap-3 p-3 hover:bg-muted/50 rounded-lg transition-colors border-b border-border/50 last:border-0">
      {/* God Avatar */}
      <div className={`w-8 h-8 rounded-full ${godColor} flex items-center justify-center text-white text-xs font-bold shrink-0`}>
        {item.from.charAt(0).toUpperCase()}
      </div>
      
      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-semibold text-sm">{item.from}</span>
          <span className="text-muted-foreground text-xs">→</span>
          <span className="text-sm text-muted-foreground">
            {item.to === 'pantheon' ? 'All Gods' : item.to}
          </span>
          <Badge variant="outline" className="text-xs flex items-center gap-1">
            {icon}
            {label}
          </Badge>
          {isAutonomic && (
            <Badge variant="secondary" className="text-xs">
              <Brain className="h-3 w-3 mr-1" />
              Autonomic
            </Badge>
          )}
          {isDebateRelated && (
            <Badge variant="destructive" className="text-xs">
              <Sword className="h-3 w-3 mr-1" />
              Debate
            </Badge>
          )}
        </div>
        
        <p className="text-sm mt-1 text-foreground/90 break-words">
          {item.content}
        </p>
        
        {/* Metadata display */}
        {item.metadata?.metrics && (
          <div className="flex gap-2 mt-2 flex-wrap">
            {Object.entries(item.metadata.metrics).map(([key, value]) => (
              <Badge key={key} variant="outline" className="text-xs">
                {key}: {typeof value === 'number' ? value.toFixed(2) : value}
              </Badge>
            ))}
          </div>
        )}
        
        {item.metadata?.resolution && (
          <div className="mt-2 p-2 bg-muted rounded text-xs">
            <span className="font-medium">Resolution:</span> {item.metadata.resolution.reasoning}
            {item.metadata.resolution.winner && (
              <Badge variant="default" className="ml-2">
                Winner: {item.metadata.resolution.winner}
              </Badge>
            )}
          </div>
        )}
        
        <span className="text-xs text-muted-foreground mt-1 block">{timeAgo}</span>
      </div>
    </div>
  );
}

interface KernelActivityStreamProps {
  limit?: number;
  showFilters?: boolean;
  maxHeight?: string;
}

export function KernelActivityStream({ 
  limit = 50, 
  showFilters = true,
  maxHeight = '600px'
}: KernelActivityStreamProps) {
  const { 
    data, 
    isLoading, 
    isConnected,
    error: wsError,
    refetch, 
    isFetching,
    clearActivities 
  } = useKernelActivityWebSocket({ maxItems: limit });
  
  const error = wsError ? new Error(wsError) : null;
  const [activeFilter, setActiveFilter] = useState<ActivityType | 'all'>('all');
  const [activeTab, setActiveTab] = useState('all');

  const filteredActivity = useMemo(() => {
    if (!data?.activity) return [];
    
    let items = [...data.activity];
    
    // Sort by timestamp descending (newest first)
    items.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
    
    // Apply type filter
    if (activeFilter !== 'all') {
      items = items.filter(item => item.type === activeFilter);
    }
    
    // Apply tab filter (considering both item.type and metadata.event_type)
    if (activeTab === 'debates') {
      items = items.filter(item => {
        const effectiveType = getEffectiveActivityType(item);
        return item.type === 'debate' || 
          item.type === 'challenge' || 
          item.type === 'challenge_response' ||
          effectiveType === 'debate_start' ||
          effectiveType === 'debate_resolved' ||
          item.metadata?.debate_id;
      });
    } else if (activeTab === 'discoveries') {
      items = items.filter(item => 
        item.type === 'discovery' || 
        item.type === 'insight' ||
        item.type === 'reflection'
      );
    } else if (activeTab === 'autonomic') {
      items = items.filter(item => {
        const effectiveType = getEffectiveActivityType(item);
        return item.type === 'autonomic' ||
          item.metadata?.autonomic ||
          item.metadata?.cycle_type ||
          effectiveType === 'sleep_cycle' ||
          effectiveType === 'dream_cycle' ||
          effectiveType === 'consolidation';
      });
    } else if (activeTab === 'learning') {
      items = items.filter(item => 
        item.type === 'learning' ||
        item.type === 'consultation' ||
        item.type === 'tool_usage'
      );
    } else if (activeTab === 'governance') {
      items = items.filter(item => 
        item.type === 'spawn_proposal' ||
        item.type === 'spawn_vote'
      );
    }
    
    return items;
  }, [data?.activity, activeFilter, activeTab]);

  if (error) {
    return (
      <Card>
        <CardContent className="py-8 text-center">
          <AlertTriangle className="h-8 w-8 mx-auto mb-2 text-destructive" />
          <p className="text-sm text-muted-foreground">Failed to load kernel activity</p>
          <Button variant="outline" size="sm" className="mt-2" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Kernel Activity Stream
          </CardTitle>
          <div className="flex items-center gap-2">
            {/* Connection status indicator */}
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} 
                 title={isConnected ? 'Connected (real-time)' : 'Disconnected'} />
            {data?.status && (
              <div className="text-xs text-muted-foreground">
                {data.status.total_messages} messages • 
                {data.status.active_debates} active debates
              </div>
            )}
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={() => refetch()}
              disabled={isFetching}
              title={isConnected ? 'Reconnect' : 'Connect'}
            >
              <RefreshCw className={`h-4 w-4 ${isFetching ? 'animate-spin' : ''}`} />
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="pt-0">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-6 mb-4">
            <TabsTrigger value="all" className="text-xs" data-testid="tab-all">
              All
            </TabsTrigger>
            <TabsTrigger value="debates" className="text-xs" data-testid="tab-debates">
              <Sword className="h-3 w-3 mr-1" />
              Debates
            </TabsTrigger>
            <TabsTrigger value="discoveries" className="text-xs" data-testid="tab-discoveries">
              <Sparkles className="h-3 w-3 mr-1" />
              Research
            </TabsTrigger>
            <TabsTrigger value="autonomic" className="text-xs" data-testid="tab-autonomic">
              <Moon className="h-3 w-3 mr-1" />
              Cycles
            </TabsTrigger>
            <TabsTrigger value="learning" className="text-xs" data-testid="tab-learning">
              <BookOpen className="h-3 w-3 mr-1" />
              Learning
            </TabsTrigger>
            <TabsTrigger value="governance" className="text-xs" data-testid="tab-governance">
              <Gavel className="h-3 w-3 mr-1" />
              Spawn
            </TabsTrigger>
          </TabsList>
          
          {showFilters && (
            <div className="flex gap-1 mb-3 flex-wrap">
              <Button
                variant={activeFilter === 'all' ? 'default' : 'outline'}
                size="sm"
                className="text-xs h-7"
                onClick={() => setActiveFilter('all')}
              >
                <Filter className="h-3 w-3 mr-1" />
                All
              </Button>
              {(['discovery', 'debate', 'autonomic', 'learning', 'spawn_proposal', 'warning', 'consultation'] as string[]).map(type => (
                <Button
                  key={type}
                  variant={activeFilter === type ? 'default' : 'outline'}
                  size="sm"
                  className="text-xs h-7"
                  onClick={() => setActiveFilter(type)}
                >
                  {ACTIVITY_ICONS[type]}
                  <span className="ml-1">{ACTIVITY_LABELS[type]}</span>
                </Button>
              ))}
            </div>
          )}
          
          <TabsContent value={activeTab} className="mt-0">
            <ScrollArea style={{ height: maxHeight }}>
              {isLoading ? (
                <div className="flex items-center justify-center py-8">
                  <RefreshCw className="h-6 w-6 animate-spin text-muted-foreground" />
                </div>
              ) : filteredActivity.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <MessageSquare className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">No activity yet</p>
                  <p className="text-xs">Kernel discussions will appear here as gods communicate</p>
                </div>
              ) : (
                <div className="space-y-0">
                  {filteredActivity.map((item) => (
                    <ActivityItem key={item.id} item={item} />
                  ))}
                </div>
              )}
            </ScrollArea>
          </TabsContent>
        </Tabs>
        
        {/* Active Debates Summary */}
        {data?.debates?.active && data.debates.active.length > 0 && (
          <div className="mt-4 pt-4 border-t">
            <h4 className="text-sm font-semibold mb-2 flex items-center gap-2">
              <Sword className="h-4 w-4 text-destructive" />
              Active Debates ({data.debates.active.length})
            </h4>
            <div className="space-y-2">
              {data.debates.active.map(debate => (
                <div key={debate.id} className="p-2 bg-muted rounded text-sm">
                  <div className="font-medium">{debate.topic}</div>
                  <div className="text-xs text-muted-foreground">
                    {debate.initiator} vs {debate.opponent} • 
                    {debate.arguments.length} arguments
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default KernelActivityStream;
