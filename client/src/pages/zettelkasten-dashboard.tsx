/**
 * Zettelkasten Dashboard - Knowledge Graph Visualization
 * 
 * Interactive visualization of the Zettelkasten memory system with:
 * - Force-directed graph visualization
 * - Stats overview (total zettels, links, keywords)
 * - Search functionality
 * - Zettel detail panel
 * - Add new zettel form
 */

import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Input,
  Button,
  Badge,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  Textarea,
  ScrollArea,
  Separator,
  useToast,
} from '@/components/ui';
import {
  Search,
  Plus,
  Network,
  FileText,
  Link2,
  Tag,
  ZoomIn,
  ZoomOut,
  Maximize2,
  RefreshCw,
  Brain,
  X,
  ChevronRight,
  Sparkles,
} from 'lucide-react';

// =============================================================================
// TYPES
// =============================================================================

interface ZettelNode {
  id: string;
  content: string;
  keywords: string[];
  source: string;
  link_count: number;
  access_count: number;
  created_at: number;
}

interface ZettelEdge {
  source: string;
  target: string;
  strength: number;
  link_type: string;
}

interface GraphData {
  nodes: ZettelNode[];
  edges: ZettelEdge[];
  stats: {
    total_zettels: number;
    total_links: number;
    total_keywords: number;
  };
}

interface ZettelStats {
  total_zettels: number;
  total_links: number;
  total_keywords: number;
  avg_links_per_zettel: number;
  sources: Record<string, number>;
}

interface SimulationNode extends ZettelNode {
  x: number;
  y: number;
  vx: number;
  vy: number;
  fx?: number | null;
  fy?: number | null;
}

// =============================================================================
// API FUNCTIONS
// =============================================================================

const API_BASE = '/api/zettelkasten';

async function fetchGraph(maxNodes = 100): Promise<GraphData> {
  const res = await fetch(`${API_BASE}/graph?max_nodes=${maxNodes}`);
  if (!res.ok) throw new Error('Failed to fetch graph');
  const data = await res.json();
  return data;
}

async function fetchStats(): Promise<ZettelStats> {
  const res = await fetch(`${API_BASE}/stats`);
  if (!res.ok) throw new Error('Failed to fetch stats');
  const data = await res.json();
  return data;
}

async function searchZettels(query: string): Promise<{ zettels: ZettelNode[]; count: number }> {
  const res = await fetch(`${API_BASE}/retrieve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, max_results: 20 }),
  });
  if (!res.ok) throw new Error('Failed to search');
  const data = await res.json();
  return { zettels: data.zettels || [], count: data.count || 0 };
}

async function addZettel(content: string, source: string): Promise<{ zettel_id: string }> {
  const res = await fetch(`${API_BASE}/add`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content, source }),
  });
  if (!res.ok) throw new Error('Failed to add zettel');
  return res.json();
}

async function fetchZettel(id: string): Promise<ZettelNode & { links: Array<{ target_id: string; strength: number; link_type: string }> }> {
  const res = await fetch(`${API_BASE}/zettel/${id}`);
  if (!res.ok) throw new Error('Failed to fetch zettel');
  const data = await res.json();
  return data.zettel;
}

// =============================================================================
// FORCE SIMULATION HOOK
// =============================================================================

function useForceSimulation(
  nodes: ZettelNode[],
  edges: ZettelEdge[],
  width: number,
  height: number
) {
  const [simulationNodes, setSimulationNodes] = useState<SimulationNode[]>([]);
  const [isStable, setIsStable] = useState(false);
  const animationRef = useRef<number>();
  const nodesRef = useRef<SimulationNode[]>([]);

  useEffect(() => {
    if (nodes.length === 0) {
      setSimulationNodes([]);
      return;
    }

    // Initialize nodes with random positions
    const simNodes: SimulationNode[] = nodes.map((node, i) => ({
      ...node,
      x: width / 2 + (Math.random() - 0.5) * width * 0.8,
      y: height / 2 + (Math.random() - 0.5) * height * 0.8,
      vx: 0,
      vy: 0,
    }));

    nodesRef.current = simNodes;
    setIsStable(false);

    // Build adjacency map for edge lookup
    const edgeMap = new Map<string, Set<string>>();
    edges.forEach(e => {
      if (!edgeMap.has(e.source)) edgeMap.set(e.source, new Set());
      if (!edgeMap.has(e.target)) edgeMap.set(e.target, new Set());
      edgeMap.get(e.source)!.add(e.target);
      edgeMap.get(e.target)!.add(e.source);
    });

    let iterations = 0;
    const maxIterations = 300;
    const alpha = 0.3;
    const alphaDecay = 0.02;
    let currentAlpha = alpha;

    const simulate = () => {
      if (iterations >= maxIterations || currentAlpha < 0.001) {
        setIsStable(true);
        return;
      }

      const nodes = nodesRef.current;

      // Center force
      const centerX = width / 2;
      const centerY = height / 2;

      // Apply forces
      for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i];
        if (node.fx !== undefined && node.fx !== null) continue;

        // Center gravity
        node.vx += (centerX - node.x) * 0.01 * currentAlpha;
        node.vy += (centerY - node.y) * 0.01 * currentAlpha;

        // Repulsion from other nodes
        for (let j = 0; j < nodes.length; j++) {
          if (i === j) continue;
          const other = nodes[j];
          const dx = node.x - other.x;
          const dy = node.y - other.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const minDist = 60;
          if (dist < minDist * 3) {
            const force = (minDist * minDist) / (dist * dist) * currentAlpha * 0.5;
            node.vx += (dx / dist) * force;
            node.vy += (dy / dist) * force;
          }
        }

        // Attraction along edges
        const connected = edgeMap.get(node.id);
        if (connected) {
          connected.forEach(targetId => {
            const target = nodes.find(n => n.id === targetId);
            if (target) {
              const dx = target.x - node.x;
              const dy = target.y - node.y;
              const dist = Math.sqrt(dx * dx + dy * dy) || 1;
              const force = dist * 0.001 * currentAlpha;
              node.vx += dx * force;
              node.vy += dy * force;
            }
          });
        }
      }

      // Update positions with velocity damping
      for (const node of nodes) {
        if (node.fx !== undefined && node.fx !== null) {
          node.x = node.fx;
          node.y = node.fy!;
          continue;
        }
        node.vx *= 0.6;
        node.vy *= 0.6;
        node.x += node.vx;
        node.y += node.vy;

        // Boundary constraints
        const padding = 30;
        node.x = Math.max(padding, Math.min(width - padding, node.x));
        node.y = Math.max(padding, Math.min(height - padding, node.y));
      }

      currentAlpha *= (1 - alphaDecay);
      iterations++;

      setSimulationNodes([...nodes]);
      animationRef.current = requestAnimationFrame(simulate);
    };

    animationRef.current = requestAnimationFrame(simulate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [nodes, edges, width, height]);

  return { simulationNodes, isStable };
}

// =============================================================================
// GRAPH COMPONENT
// =============================================================================

interface GraphViewProps {
  nodes: SimulationNode[];
  edges: ZettelEdge[];
  selectedNode: string | null;
  highlightedNodes: Set<string>;
  onSelectNode: (id: string | null) => void;
  zoom: number;
  onZoomChange: (zoom: number) => void;
}

function GraphView({
  nodes,
  edges,
  selectedNode,
  highlightedNodes,
  onSelectNode,
  zoom,
  onZoomChange,
}: GraphViewProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  const nodeMap = useMemo(() => {
    const map = new Map<string, SimulationNode>();
    nodes.forEach(n => map.set(n.id, n));
    return map;
  }, [nodes]);

  const getNodeColor = (node: SimulationNode) => {
    if (node.id === selectedNode) return '#8b5cf6'; // Purple for selected
    if (highlightedNodes.has(node.id)) return '#f59e0b'; // Amber for highlighted
    
    // Color by source
    const sourceColors: Record<string, string> = {
      'zeus_chat': '#3b82f6',
      'user': '#10b981',
      'learned': '#ec4899',
      'test': '#6366f1',
    };
    return sourceColors[node.source] || '#6b7280';
  };

  const getNodeSize = (node: SimulationNode) => {
    const minSize = 6;
    const maxSize = 20;
    const linkScale = Math.min(node.link_count / 10, 1);
    return minSize + linkScale * (maxSize - minSize);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0) {
      setIsPanning(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isPanning) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }
  };

  const handleMouseUp = () => {
    setIsPanning(false);
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    onZoomChange(Math.max(0.2, Math.min(3, zoom + delta)));
  };

  return (
    <div className="relative w-full h-full bg-muted/30 rounded-lg overflow-hidden">
      <svg
        ref={svgRef}
        className="w-full h-full cursor-grab active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      >
        <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
          {/* Edges */}
          {edges.map((edge, i) => {
            const source = nodeMap.get(edge.source);
            const target = nodeMap.get(edge.target);
            if (!source || !target) return null;

            const isHighlighted =
              selectedNode === edge.source ||
              selectedNode === edge.target ||
              highlightedNodes.has(edge.source) ||
              highlightedNodes.has(edge.target);

            return (
              <line
                key={`edge-${i}`}
                x1={source.x}
                y1={source.y}
                x2={target.x}
                y2={target.y}
                stroke={isHighlighted ? '#8b5cf6' : '#374151'}
                strokeWidth={Math.max(1, edge.strength * 3)}
                strokeOpacity={isHighlighted ? 0.8 : 0.3}
              />
            );
          })}

          {/* Nodes */}
          {nodes.map(node => {
            const size = getNodeSize(node);
            const color = getNodeColor(node);
            const isSelected = node.id === selectedNode;

            return (
              <g
                key={node.id}
                transform={`translate(${node.x}, ${node.y})`}
                onClick={(e) => {
                  e.stopPropagation();
                  onSelectNode(isSelected ? null : node.id);
                }}
                className="cursor-pointer"
              >
                {/* Glow effect for selected/highlighted */}
                {(isSelected || highlightedNodes.has(node.id)) && (
                  <circle
                    r={size + 4}
                    fill="none"
                    stroke={color}
                    strokeWidth={2}
                    strokeOpacity={0.4}
                  />
                )}
                <circle
                  r={size}
                  fill={color}
                  className="transition-all duration-200 hover:brightness-125"
                />
                {/* Show label for larger/selected nodes */}
                {(size > 12 || isSelected) && (
                  <text
                    y={size + 12}
                    textAnchor="middle"
                    className="text-[10px] fill-foreground pointer-events-none"
                  >
                    {node.content.slice(0, 20)}...
                  </text>
                )}
              </g>
            );
          })}
        </g>
      </svg>

      {/* Zoom controls */}
      <div className="absolute bottom-4 right-4 flex flex-col gap-2">
        <Button
          variant="secondary"
          size="icon"
          onClick={() => onZoomChange(Math.min(3, zoom + 0.2))}
        >
          <ZoomIn className="h-4 w-4" />
        </Button>
        <Button
          variant="secondary"
          size="icon"
          onClick={() => onZoomChange(Math.max(0.2, zoom - 0.2))}
        >
          <ZoomOut className="h-4 w-4" />
        </Button>
        <Button
          variant="secondary"
          size="icon"
          onClick={() => {
            setPan({ x: 0, y: 0 });
            onZoomChange(1);
          }}
        >
          <Maximize2 className="h-4 w-4" />
        </Button>
      </div>

      {/* Legend */}
      <div className="absolute top-4 left-4 bg-background/80 backdrop-blur-sm rounded-lg p-3 text-xs space-y-1">
        <div className="font-medium mb-2">Sources</div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-blue-500" />
          <span>Zeus Chat</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-emerald-500" />
          <span>User</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-pink-500" />
          <span>Learned</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-gray-500" />
          <span>Other</span>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// ZETTEL DETAIL PANEL
// =============================================================================

interface ZettelDetailProps {
  zettelId: string;
  onClose: () => void;
  onNavigate: (id: string) => void;
}

function ZettelDetail({ zettelId, onClose, onNavigate }: ZettelDetailProps) {
  const { data: zettel, isLoading } = useQuery({
    queryKey: ['zettel', zettelId],
    queryFn: () => fetchZettel(zettelId),
    enabled: !!zettelId,
  });

  if (isLoading) {
    return (
      <div className="p-4 animate-pulse">
        <div className="h-4 bg-muted rounded w-3/4 mb-2" />
        <div className="h-20 bg-muted rounded" />
      </div>
    );
  }

  if (!zettel) {
    return (
      <div className="p-4 text-muted-foreground">
        Zettel not found
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between p-4 border-b">
        <h3 className="font-semibold flex items-center gap-2">
          <FileText className="h-4 w-4" />
          Zettel Details
        </h3>
        <Button variant="ghost" size="icon" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4 space-y-4">
          {/* Content */}
          <div>
            <label className="text-sm text-muted-foreground">Content</label>
            <p className="mt-1 text-sm leading-relaxed">{zettel.content}</p>
          </div>

          <Separator />

          {/* Keywords */}
          <div>
            <label className="text-sm text-muted-foreground flex items-center gap-1">
              <Tag className="h-3 w-3" /> Keywords
            </label>
            <div className="flex flex-wrap gap-1 mt-2">
              {zettel.keywords?.map((kw, i) => (
                <Badge key={i} variant="secondary" className="text-xs">
                  {kw}
                </Badge>
              ))}
              {(!zettel.keywords || zettel.keywords.length === 0) && (
                <span className="text-xs text-muted-foreground">No keywords</span>
              )}
            </div>
          </div>

          <Separator />

          {/* Links */}
          <div>
            <label className="text-sm text-muted-foreground flex items-center gap-1">
              <Link2 className="h-3 w-3" /> Links ({zettel.links?.length || 0})
            </label>
            <div className="mt-2 space-y-1">
              {zettel.links?.map((link, i) => (
                <button
                  key={i}
                  onClick={() => onNavigate(link.target_id)}
                  className="w-full text-left text-xs p-2 rounded hover:bg-muted flex items-center justify-between group"
                >
                  <span className="truncate font-mono">{link.target_id}</span>
                  <ChevronRight className="h-3 w-3 opacity-0 group-hover:opacity-100 transition-opacity" />
                </button>
              ))}
              {(!zettel.links || zettel.links.length === 0) && (
                <span className="text-xs text-muted-foreground">No links</span>
              )}
            </div>
          </div>

          <Separator />

          {/* Metadata */}
          <div className="space-y-2 text-xs">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Source</span>
              <Badge variant="outline">{zettel.source}</Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Access Count</span>
              <span>{zettel.access_count}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">ID</span>
              <span className="font-mono truncate max-w-[150px]">{zettel.id}</span>
            </div>
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}

// =============================================================================
// ADD ZETTEL FORM
// =============================================================================

function AddZettelForm({ onSuccess }: { onSuccess: () => void }) {
  const [content, setContent] = useState('');
  const [source, setSource] = useState('user');
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: () => addZettel(content, source),
    onSuccess: () => {
      toast({
        title: 'Zettel added',
        description: 'New knowledge has been stored in the graph.',
      });
      setContent('');
      queryClient.invalidateQueries({ queryKey: ['zettelkasten'] });
      onSuccess();
    },
    onError: (error) => {
      toast({
        title: 'Error',
        description: `Failed to add zettel: ${error.message}`,
        variant: 'destructive',
      });
    },
  });

  return (
    <div className="space-y-4">
      <div>
        <label className="text-sm font-medium">Content</label>
        <Textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="Enter atomic idea or knowledge..."
          className="mt-1 min-h-[100px]"
        />
      </div>
      <div>
        <label className="text-sm font-medium">Source</label>
        <Input
          value={source}
          onChange={(e) => setSource(e.target.value)}
          placeholder="user"
          className="mt-1"
        />
      </div>
      <Button
        onClick={() => mutation.mutate()}
        disabled={!content.trim() || mutation.isPending}
        className="w-full"
      >
        {mutation.isPending ? (
          <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
        ) : (
          <Plus className="h-4 w-4 mr-2" />
        )}
        Add Zettel
      </Button>
    </div>
  );
}

// =============================================================================
// MAIN DASHBOARD
// =============================================================================

export default function ZettelkastenDashboard() {
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [highlightedNodes, setHighlightedNodes] = useState<Set<string>>(new Set());
  const [zoom, setZoom] = useState(1);
  const [graphDimensions] = useState({ width: 800, height: 600 });
  const { toast } = useToast();
  const queryClient = useQueryClient();

  // Fetch graph data
  const {
    data: graphData,
    isLoading: isLoadingGraph,
    refetch: refetchGraph,
  } = useQuery({
    queryKey: ['zettelkasten', 'graph'],
    queryFn: () => fetchGraph(100),
    staleTime: 30000,
  });

  // Fetch stats
  const { data: stats, isLoading: isLoadingStats } = useQuery({
    queryKey: ['zettelkasten', 'stats'],
    queryFn: fetchStats,
    staleTime: 30000,
  });

  // Force simulation
  const { simulationNodes, isStable } = useForceSimulation(
    graphData?.nodes || [],
    graphData?.edges || [],
    graphDimensions.width,
    graphDimensions.height
  );

  // Search handler
  const handleSearch = useCallback(async () => {
    if (!searchQuery.trim()) {
      setHighlightedNodes(new Set());
      return;
    }

    try {
      const results = await searchZettels(searchQuery);
      const ids = new Set(results.zettels.map(z => z.id));
      setHighlightedNodes(ids);
      toast({
        title: 'Search complete',
        description: `Found ${results.count} matching zettels`,
      });
    } catch (error) {
      toast({
        title: 'Search failed',
        description: String(error),
        variant: 'destructive',
      });
    }
  }, [searchQuery, toast]);

  // Stats cards
  const statCards = [
    {
      title: 'Total Zettels',
      value: stats?.total_zettels ?? 0,
      icon: FileText,
      color: 'text-blue-500',
    },
    {
      title: 'Total Links',
      value: stats?.total_links ?? 0,
      icon: Link2,
      color: 'text-purple-500',
    },
    {
      title: 'Keywords',
      value: stats?.total_keywords ?? 0,
      icon: Tag,
      color: 'text-amber-500',
    },
    {
      title: 'Avg Links/Zettel',
      value: stats?.avg_links_per_zettel?.toFixed(1) ?? '0',
      icon: Network,
      color: 'text-emerald-500',
    },
  ];

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <Brain className="h-8 w-8 text-primary" />
            Zettelkasten Memory
          </h1>
          <p className="text-muted-foreground mt-1">
            Interactive knowledge graph visualization
          </p>
        </div>
        <Button
          onClick={() => {
            refetchGraph();
            queryClient.invalidateQueries({ queryKey: ['zettelkasten', 'stats'] });
          }}
          variant="outline"
        >
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {statCards.map((stat, i) => (
          <Card key={i}>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">{stat.title}</p>
                  <p className="text-2xl font-bold">
                    {isLoadingStats ? '...' : stat.value}
                  </p>
                </div>
                <stat.icon className={`h-8 w-8 ${stat.color} opacity-50`} />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Search Bar */}
      <Card>
        <CardContent className="p-4">
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="Search zettels by content or keywords..."
                className="pl-10"
              />
            </div>
            <Button onClick={handleSearch}>
              <Search className="h-4 w-4 mr-2" />
              Search
            </Button>
            {highlightedNodes.size > 0 && (
              <Button
                variant="outline"
                onClick={() => {
                  setHighlightedNodes(new Set());
                  setSearchQuery('');
                }}
              >
                Clear ({highlightedNodes.size})
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Graph Visualization */}
        <div className="lg:col-span-2">
          <Card className="h-[600px]">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2">
                <Network className="h-5 w-5" />
                Knowledge Graph
                {!isStable && (
                  <Badge variant="secondary" className="ml-2">
                    <Sparkles className="h-3 w-3 mr-1 animate-pulse" />
                    Stabilizing...
                  </Badge>
                )}
              </CardTitle>
              <CardDescription>
                {graphData?.nodes.length ?? 0} nodes, {graphData?.edges.length ?? 0} edges
              </CardDescription>
            </CardHeader>
            <CardContent className="p-2 h-[calc(100%-80px)]">
              {isLoadingGraph ? (
                <div className="flex items-center justify-center h-full">
                  <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
              ) : simulationNodes.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
                  <Network className="h-16 w-16 mb-4 opacity-50" />
                  <p>No zettels yet</p>
                  <p className="text-sm">Add knowledge to see the graph</p>
                </div>
              ) : (
                <GraphView
                  nodes={simulationNodes}
                  edges={graphData?.edges || []}
                  selectedNode={selectedNode}
                  highlightedNodes={highlightedNodes}
                  onSelectNode={setSelectedNode}
                  zoom={zoom}
                  onZoomChange={setZoom}
                />
              )}
            </CardContent>
          </Card>
        </div>

        {/* Side Panel */}
        <div className="space-y-6">
          <Tabs defaultValue="details">
            <TabsList className="w-full">
              <TabsTrigger value="details" className="flex-1">Details</TabsTrigger>
              <TabsTrigger value="add" className="flex-1">Add New</TabsTrigger>
            </TabsList>

            <TabsContent value="details" className="mt-4">
              <Card className="h-[520px]">
                {selectedNode ? (
                  <ZettelDetail
                    zettelId={selectedNode}
                    onClose={() => setSelectedNode(null)}
                    onNavigate={setSelectedNode}
                  />
                ) : (
                  <div className="flex flex-col items-center justify-center h-full text-muted-foreground p-4">
                    <FileText className="h-12 w-12 mb-4 opacity-50" />
                    <p className="text-center">
                      Click on a node in the graph to view its details
                    </p>
                  </div>
                )}
              </Card>
            </TabsContent>

            <TabsContent value="add" className="mt-4">
              <Card className="h-[520px]">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Plus className="h-5 w-5" />
                    Add New Zettel
                  </CardTitle>
                  <CardDescription>
                    Store new knowledge in the graph
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <AddZettelForm
                    onSuccess={() => {
                      refetchGraph();
                      queryClient.invalidateQueries({ queryKey: ['zettelkasten', 'stats'] });
                    }}
                  />
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}
