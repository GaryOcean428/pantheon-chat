/**
 * Zettelkasten Knowledge Graph Dashboard
 * 
 * Visualizes the Zettelkasten memory network with:
 * - Force-directed graph visualization
 * - Stats overview
 * - Search and retrieval
 * - Hub and cluster analysis
 * - Add new memories
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { Brain, Search, Plus, Network, Sparkles, Link2, Tag, Eye, RefreshCw } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useToast } from '@/hooks/use-toast';

// Types
interface ZettelStats {
  total_zettels: number;
  total_links: number;
  total_keywords: number;
  total_evolutions: number;
  avg_links_per_zettel: number;
}

interface GraphNode {
  id: string;
  label: string;
  keywords: string[];
  access_count?: number;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
}

interface GraphEdge {
  source: string;
  target: string;
  strength: number;
  link_type?: string;
}

interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: ZettelStats;
}

interface SearchResult {
  zettel_id: string;
  content: string;
  relevance: number;
  keywords: string[];
  contextual_description: string;
  links_count: number;
  access_count: number;
}

interface HubZettel {
  zettel_id: string;
  content: string;
  keywords: string[];
  outgoing_links: number;
  access_count: number;
  evolution_count: number;
}

interface Cluster {
  cluster_id: number;
  size: number;
  top_keywords: string[];
  zettels: {
    zettel_id: string;
    content: string;
    keywords: string[];
  }[];
}

interface ZettelDetail {
  zettel_id: string;
  content: string;
  contextual_description: string;
  keywords: string[];
  links: {
    target_id: string;
    link_type: string;
    strength: number;
    context: string;
  }[];
  access_count: number;
  evolution_count: number;
  source: string;
  created_at: number;
}

// Force-directed graph component
function ForceGraph({ 
  nodes, 
  edges, 
  onNodeClick,
  selectedNodeId
}: { 
  nodes: GraphNode[];
  edges: GraphEdge[];
  onNodeClick: (nodeId: string) => void;
  selectedNodeId: string | null;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [positions, setPositions] = useState<Map<string, { x: number; y: number }>>(new Map());
  const [dragging, setDragging] = useState<string | null>(null);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const animationRef = useRef<number | null>(null);

  // Initialize positions and run force simulation
  useEffect(() => {
    if (nodes.length === 0) return;

    const width = 800;
    const height = 600;
    const centerX = width / 2;
    const centerY = height / 2;

    // Initialize positions in a circle
    const newPositions = new Map<string, { x: number; y: number; vx: number; vy: number }>();
    nodes.forEach((node, i) => {
      const angle = (2 * Math.PI * i) / nodes.length;
      const radius = Math.min(width, height) / 3;
      newPositions.set(node.id, {
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
        vx: 0,
        vy: 0
      });
    });

    // Create edge lookup
    const edgeMap = new Map<string, Set<string>>();
    edges.forEach(edge => {
      if (!edgeMap.has(edge.source)) edgeMap.set(edge.source, new Set());
      if (!edgeMap.has(edge.target)) edgeMap.set(edge.target, new Set());
      edgeMap.get(edge.source)?.add(edge.target);
      edgeMap.get(edge.target)?.add(edge.source);
    });

    // Force simulation
    let iteration = 0;
    const maxIterations = 150;
    const alpha = 0.3;
    const repulsion = 5000;
    const attraction = 0.01;
    const damping = 0.9;

    const simulate = () => {
      if (iteration >= maxIterations) {
        setPositions(new Map([...newPositions].map(([k, v]) => [k, { x: v.x, y: v.y }])));
        return;
      }

      const currentAlpha = alpha * (1 - iteration / maxIterations);

      // Apply forces
      const nodeArray = [...newPositions.entries()];
      
      // Repulsion between all nodes
      for (let i = 0; i < nodeArray.length; i++) {
        for (let j = i + 1; j < nodeArray.length; j++) {
          const [id1, pos1] = nodeArray[i];
          const [id2, pos2] = nodeArray[j];
          
          const dx = pos2.x - pos1.x;
          const dy = pos2.y - pos1.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const force = repulsion / (dist * dist);
          
          const fx = (dx / dist) * force * currentAlpha;
          const fy = (dy / dist) * force * currentAlpha;
          
          pos1.vx -= fx;
          pos1.vy -= fy;
          pos2.vx += fx;
          pos2.vy += fy;
        }
      }

      // Attraction along edges
      edges.forEach(edge => {
        const pos1 = newPositions.get(edge.source);
        const pos2 = newPositions.get(edge.target);
        if (!pos1 || !pos2) return;

        const dx = pos2.x - pos1.x;
        const dy = pos2.y - pos1.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = dist * attraction * edge.strength * currentAlpha;

        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;

        pos1.vx += fx;
        pos1.vy += fy;
        pos2.vx -= fx;
        pos2.vy -= fy;
      });

      // Center gravity
      nodeArray.forEach(([_, pos]) => {
        pos.vx += (centerX - pos.x) * 0.01 * currentAlpha;
        pos.vy += (centerY - pos.y) * 0.01 * currentAlpha;
      });

      // Apply velocities with damping
      nodeArray.forEach(([_, pos]) => {
        pos.vx *= damping;
        pos.vy *= damping;
        pos.x += pos.vx;
        pos.y += pos.vy;
        
        // Bounds
        pos.x = Math.max(50, Math.min(width - 50, pos.x));
        pos.y = Math.max(50, Math.min(height - 50, pos.y));
      });

      iteration++;
      setPositions(new Map([...newPositions].map(([k, v]) => [k, { x: v.x, y: v.y }])));
      animationRef.current = requestAnimationFrame(simulate);
    };

    simulate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [nodes, edges]);

  const handleMouseDown = useCallback((e: React.MouseEvent, nodeId: string) => {
    e.stopPropagation();
    const pos = positions.get(nodeId);
    if (pos) {
      setDragging(nodeId);
      const rect = svgRef.current?.getBoundingClientRect();
      if (rect) {
        setOffset({
          x: (e.clientX - rect.left) / zoom - pan.x - pos.x,
          y: (e.clientY - rect.top) / zoom - pan.y - pos.y
        });
      }
    }
  }, [positions, zoom, pan]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (dragging) {
      const rect = svgRef.current?.getBoundingClientRect();
      if (rect) {
        const x = (e.clientX - rect.left) / zoom - pan.x - offset.x;
        const y = (e.clientY - rect.top) / zoom - pan.y - offset.y;
        setPositions(prev => {
          const newMap = new Map(prev);
          newMap.set(dragging, { x, y });
          return newMap;
        });
      }
    }
  }, [dragging, offset, zoom, pan]);

  const handleMouseUp = useCallback(() => {
    setDragging(null);
  }, []);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(z => Math.max(0.5, Math.min(2, z * delta)));
  }, []);

  // Assign colors based on node index (simple cluster coloring)
  const getNodeColor = (nodeId: string, index: number) => {
    const colors = [
      '#8b5cf6', // purple
      '#6366f1', // indigo
      '#3b82f6', // blue
      '#14b8a6', // teal
      '#22c55e', // green
      '#eab308', // yellow
      '#f97316', // orange
      '#ef4444', // red
    ];
    return colors[index % colors.length];
  };

  return (
    <svg
      ref={svgRef}
      width="100%"
      height="100%"
      viewBox="0 0 800 600"
      className="bg-gradient-to-br from-slate-900 via-purple-900/20 to-slate-900 rounded-lg"
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onWheel={handleWheel}
    >
      <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
        {/* Edges */}
        {edges.map((edge, i) => {
          const source = positions.get(edge.source);
          const target = positions.get(edge.target);
          if (!source || !target) return null;
          return (
            <line
              key={`edge-${i}`}
              x1={source.x}
              y1={source.y}
              x2={target.x}
              y2={target.y}
              stroke="rgba(139, 92, 246, 0.3)"
              strokeWidth={Math.max(1, edge.strength * 3)}
            />
          );
        })}

        {/* Nodes */}
        {nodes.map((node, i) => {
          const pos = positions.get(node.id);
          if (!pos) return null;
          const isSelected = selectedNodeId === node.id;
          const color = getNodeColor(node.id, i);
          
          return (
            <g
              key={node.id}
              transform={`translate(${pos.x}, ${pos.y})`}
              onMouseDown={(e) => handleMouseDown(e, node.id)}
              onClick={() => onNodeClick(node.id)}
              style={{ cursor: 'pointer' }}
            >
              {/* Glow effect for selected */}
              {isSelected && (
                <circle
                  r={20}
                  fill="none"
                  stroke={color}
                  strokeWidth={3}
                  opacity={0.5}
                  className="animate-pulse"
                />
              )}
              {/* Node circle */}
              <circle
                r={12}
                fill={color}
                stroke={isSelected ? '#fff' : 'rgba(255,255,255,0.3)'}
                strokeWidth={isSelected ? 3 : 1}
              />
              {/* Label */}
              <text
                y={25}
                textAnchor="middle"
                fill="rgba(255,255,255,0.8)"
                fontSize={10}
                className="pointer-events-none"
              >
                {node.label.slice(0, 20)}{node.label.length > 20 ? '...' : ''}
              </text>
            </g>
          );
        })}
      </g>

      {/* Zoom controls */}
      <g transform="translate(20, 20)">
        <rect
          x={0}
          y={0}
          width={30}
          height={60}
          rx={5}
          fill="rgba(0,0,0,0.5)"
        />
        <text
          x={15}
          y={22}
          textAnchor="middle"
          fill="white"
          fontSize={18}
          className="cursor-pointer"
          onClick={() => setZoom(z => Math.min(2, z * 1.2))}
        >
          +
        </text>
        <text
          x={15}
          y={50}
          textAnchor="middle"
          fill="white"
          fontSize={18}
          className="cursor-pointer"
          onClick={() => setZoom(z => Math.max(0.5, z * 0.8))}
        >
          −
        </text>
      </g>
    </svg>
  );
}

// Main dashboard component
export default function ZettelkastenDashboard() {
  const { toast } = useToast();
  const [stats, setStats] = useState<ZettelStats | null>(null);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [selectedTab, setSelectedTab] = useState('graph');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [hubs, setHubs] = useState<HubZettel[]>([]);
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [selectedZettel, setSelectedZettel] = useState<ZettelDetail | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSearching, setIsSearching] = useState(false);

  // New zettel form state
  const [newContent, setNewContent] = useState('');
  const [newSource, setNewSource] = useState('');
  const [isAdding, setIsAdding] = useState(false);

  // Fetch stats
  const fetchStats = async () => {
    try {
      const response = await fetch('/api/zettelkasten/stats');
      const data = await response.json();
      if (data.success) {
        setStats(data);
      }
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  // Fetch graph data
  const fetchGraph = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/zettelkasten/graph?max_nodes=50');
      const data = await response.json();
      if (data.success) {
        setGraphData(data);
      }
    } catch (error) {
      console.error('Error fetching graph:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch hubs
  const fetchHubs = async () => {
    try {
      const response = await fetch('/api/zettelkasten/hubs?top_n=10');
      const data = await response.json();
      if (data.success) {
        setHubs(data.hubs);
      }
    } catch (error) {
      console.error('Error fetching hubs:', error);
    }
  };

  // Fetch clusters
  const fetchClusters = async () => {
    try {
      const response = await fetch('/api/zettelkasten/clusters?min_size=2');
      const data = await response.json();
      if (data.success) {
        setClusters(data.clusters);
      }
    } catch (error) {
      console.error('Error fetching clusters:', error);
    }
  };

  // Search
  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    setIsSearching(true);
    try {
      const response = await fetch('/api/zettelkasten/retrieve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: searchQuery, max_results: 20 })
      });
      const data = await response.json();
      if (data.success) {
        setSearchResults(data.results);
      }
    } catch (error) {
      console.error('Error searching:', error);
      toast({
        title: 'Search Failed',
        description: 'Could not search the knowledge base',
        variant: 'destructive'
      });
    } finally {
      setIsSearching(false);
    }
  };

  // Fetch zettel details
  const fetchZettelDetails = async (zettelId: string) => {
    try {
      const response = await fetch(`/api/zettelkasten/zettel/${zettelId}`);
      const data = await response.json();
      if (data.success) {
        setSelectedZettel(data.zettel);
        setSelectedNodeId(zettelId);
      }
    } catch (error) {
      console.error('Error fetching zettel:', error);
    }
  };

  // Add new zettel
  const handleAddZettel = async () => {
    if (!newContent.trim()) {
      toast({
        title: 'Content Required',
        description: 'Please enter some content for the new memory',
        variant: 'destructive'
      });
      return;
    }

    setIsAdding(true);
    try {
      const response = await fetch('/api/zettelkasten/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: newContent, source: newSource || 'manual' })
      });
      const data = await response.json();
      
      if (data.success) {
        toast({
          title: 'Memory Added',
          description: `Created zettel with ${data.links_created} links`
        });
        setNewContent('');
        setNewSource('');
        // Refresh data
        fetchStats();
        fetchGraph();
      } else {
        throw new Error(data.error || 'Failed to add memory');
      }
    } catch (error) {
      console.error('Error adding zettel:', error);
      toast({
        title: 'Failed to Add',
        description: String(error),
        variant: 'destructive'
      });
    } finally {
      setIsAdding(false);
    }
  };

  // Handle node click
  const handleNodeClick = (nodeId: string) => {
    fetchZettelDetails(nodeId);
  };

  // Initial load
  useEffect(() => {
    fetchStats();
    fetchGraph();
    fetchHubs();
    fetchClusters();
  }, []);

  // Refresh data
  const handleRefresh = () => {
    fetchStats();
    fetchGraph();
    fetchHubs();
    fetchClusters();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-purple-950/30 to-slate-950 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-3 rounded-xl bg-gradient-to-br from-purple-500 to-indigo-600 shadow-lg shadow-purple-500/20">
              <Brain className="h-8 w-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-indigo-400 bg-clip-text text-transparent">
                Zettelkasten Knowledge Graph
              </h1>
              <p className="text-slate-400">Self-organizing memory network with geometric linking</p>
            </div>
          </div>
          <Button 
            onClick={handleRefresh}
            variant="outline"
            className="border-purple-500/30 hover:bg-purple-500/10"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className="bg-slate-900/50 border-purple-500/20">
            <CardHeader className="pb-2">
              <CardDescription className="text-slate-400">Total Zettels</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2">
                <Brain className="h-5 w-5 text-purple-400" />
                <span className="text-2xl font-bold text-white">
                  {stats?.total_zettels ?? 0}
                </span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-900/50 border-purple-500/20">
            <CardHeader className="pb-2">
              <CardDescription className="text-slate-400">Total Links</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2">
                <Link2 className="h-5 w-5 text-indigo-400" />
                <span className="text-2xl font-bold text-white">
                  {stats?.total_links ?? 0}
                </span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-900/50 border-purple-500/20">
            <CardHeader className="pb-2">
              <CardDescription className="text-slate-400">Keywords</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2">
                <Tag className="h-5 w-5 text-blue-400" />
                <span className="text-2xl font-bold text-white">
                  {stats?.total_keywords ?? 0}
                </span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-900/50 border-purple-500/20">
            <CardHeader className="pb-2">
              <CardDescription className="text-slate-400">Avg Links/Zettel</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2">
                <Network className="h-5 w-5 text-teal-400" />
                <span className="text-2xl font-bold text-white">
                  {stats?.avg_links_per_zettel?.toFixed(1) ?? '0.0'}
                </span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left: Tabs with Graph/Hubs/Clusters/Search */}
          <div className="lg:col-span-2">
            <Card className="bg-slate-900/50 border-purple-500/20">
              <Tabs value={selectedTab} onValueChange={setSelectedTab}>
                <CardHeader className="pb-0">
                  <TabsList className="bg-slate-800/50">
                    <TabsTrigger value="graph" className="data-[state=active]:bg-purple-600">
                      <Network className="h-4 w-4 mr-2" />
                      Graph
                    </TabsTrigger>
                    <TabsTrigger value="hubs" className="data-[state=active]:bg-purple-600">
                      <Sparkles className="h-4 w-4 mr-2" />
                      Hubs
                    </TabsTrigger>
                    <TabsTrigger value="clusters" className="data-[state=active]:bg-purple-600">
                      <Brain className="h-4 w-4 mr-2" />
                      Clusters
                    </TabsTrigger>
                    <TabsTrigger value="search" className="data-[state=active]:bg-purple-600">
                      <Search className="h-4 w-4 mr-2" />
                      Search
                    </TabsTrigger>
                    <TabsTrigger value="add" className="data-[state=active]:bg-purple-600">
                      <Plus className="h-4 w-4 mr-2" />
                      Add
                    </TabsTrigger>
                  </TabsList>
                </CardHeader>

                <CardContent className="pt-4">
                  {/* Graph View */}
                  <TabsContent value="graph" className="m-0">
                    <div className="h-[500px] rounded-lg overflow-hidden">
                      {isLoading ? (
                        <div className="h-full flex items-center justify-center text-slate-400">
                          <RefreshCw className="h-8 w-8 animate-spin mr-2" />
                          Loading graph...
                        </div>
                      ) : graphData && graphData.nodes.length > 0 ? (
                        <ForceGraph
                          nodes={graphData.nodes}
                          edges={graphData.edges}
                          onNodeClick={handleNodeClick}
                          selectedNodeId={selectedNodeId}
                        />
                      ) : (
                        <div className="h-full flex flex-col items-center justify-center text-slate-400">
                          <Brain className="h-12 w-12 mb-4 opacity-50" />
                          <p>Building knowledge graph...</p>
                          <p className="text-sm">Memories are added automatically from conversations and searches. Chat with the gods to populate your graph!</p>
                        </div>
                      )}
                    </div>
                  </TabsContent>

                  {/* Hubs View */}
                  <TabsContent value="hubs" className="m-0">
                    <ScrollArea className="h-[500px]">
                      <div className="space-y-3">
                        {hubs.length > 0 ? (
                          hubs.map((hub) => (
                            <Card 
                              key={hub.zettel_id} 
                              className="bg-slate-800/50 border-purple-500/10 cursor-pointer hover:border-purple-500/30 transition-colors"
                              onClick={() => fetchZettelDetails(hub.zettel_id)}
                            >
                              <CardContent className="p-4">
                                <div className="flex items-start justify-between">
                                  <div className="flex-1">
                                    <p className="text-white text-sm line-clamp-2">{hub.content}</p>
                                    <div className="flex flex-wrap gap-1 mt-2">
                                      {hub.keywords.slice(0, 5).map((kw, i) => (
                                        <Badge key={i} variant="secondary" className="bg-purple-500/20 text-purple-300 text-xs">
                                          {kw}
                                        </Badge>
                                      ))}
                                    </div>
                                  </div>
                                  <div className="text-right ml-4">
                                    <Badge className="bg-indigo-500/20 text-indigo-300">
                                      {hub.outgoing_links} links
                                    </Badge>
                                  </div>
                                </div>
                              </CardContent>
                            </Card>
                          ))
                        ) : (
                          <div className="text-center text-slate-400 py-8">
                            <Sparkles className="h-12 w-12 mx-auto mb-4 opacity-50" />
                            <p>No hub zettels found</p>
                          </div>
                        )}
                      </div>
                    </ScrollArea>
                  </TabsContent>

                  {/* Clusters View */}
                  <TabsContent value="clusters" className="m-0">
                    <ScrollArea className="h-[500px]">
                      <div className="space-y-4">
                        {clusters.length > 0 ? (
                          clusters.map((cluster) => (
                            <Card key={cluster.cluster_id} className="bg-slate-800/50 border-purple-500/10">
                              <CardHeader className="pb-2">
                                <CardTitle className="text-lg text-white flex items-center gap-2">
                                  <div className="w-3 h-3 rounded-full" style={{ 
                                    backgroundColor: ['#8b5cf6', '#6366f1', '#3b82f6', '#14b8a6'][cluster.cluster_id % 4]
                                  }} />
                                  Cluster {cluster.cluster_id + 1}
                                  <Badge variant="outline" className="ml-2">{cluster.size} zettels</Badge>
                                </CardTitle>
                                <div className="flex flex-wrap gap-1">
                                  {cluster.top_keywords.map((kw, i) => (
                                    <Badge key={i} variant="secondary" className="bg-purple-500/20 text-purple-300 text-xs">
                                      {kw}
                                    </Badge>
                                  ))}
                                </div>
                              </CardHeader>
                              <CardContent className="pt-0">
                                <div className="space-y-2">
                                  {cluster.zettels.slice(0, 3).map((zettel) => (
                                    <div 
                                      key={zettel.zettel_id}
                                      className="p-2 rounded bg-slate-900/50 cursor-pointer hover:bg-slate-900/80 transition-colors"
                                      onClick={() => fetchZettelDetails(zettel.zettel_id)}
                                    >
                                      <p className="text-slate-300 text-sm line-clamp-1">{zettel.content}</p>
                                    </div>
                                  ))}
                                  {cluster.size > 3 && (
                                    <p className="text-slate-500 text-xs">+{cluster.size - 3} more</p>
                                  )}
                                </div>
                              </CardContent>
                            </Card>
                          ))
                        ) : (
                          <div className="text-center text-slate-400 py-8">
                            <Brain className="h-12 w-12 mx-auto mb-4 opacity-50" />
                            <p>No clusters detected yet</p>
                            <p className="text-sm">Add more connected zettels to form clusters</p>
                          </div>
                        )}
                      </div>
                    </ScrollArea>
                  </TabsContent>

                  {/* Search View */}
                  <TabsContent value="search" className="m-0">
                    <div className="space-y-4">
                      <div className="flex gap-2">
                        <Input
                          placeholder="Search the knowledge base..."
                          value={searchQuery}
                          onChange={(e) => setSearchQuery(e.target.value)}
                          onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                          className="bg-slate-800/50 border-purple-500/20"
                        />
                        <Button 
                          onClick={handleSearch}
                          disabled={isSearching}
                          className="bg-purple-600 hover:bg-purple-700"
                        >
                          {isSearching ? (
                            <RefreshCw className="h-4 w-4 animate-spin" />
                          ) : (
                            <Search className="h-4 w-4" />
                          )}
                        </Button>
                      </div>

                      <ScrollArea className="h-[440px]">
                        <div className="space-y-3">
                          {searchResults.length > 0 ? (
                            searchResults.map((result) => (
                              <Card 
                                key={result.zettel_id}
                                className="bg-slate-800/50 border-purple-500/10 cursor-pointer hover:border-purple-500/30 transition-colors"
                                onClick={() => fetchZettelDetails(result.zettel_id)}
                              >
                                <CardContent className="p-4">
                                  <div className="flex items-start justify-between">
                                    <div className="flex-1">
                                      <p className="text-white text-sm">{result.content}</p>
                                      <p className="text-slate-400 text-xs mt-1">{result.contextual_description}</p>
                                      <div className="flex flex-wrap gap-1 mt-2">
                                        {result.keywords.slice(0, 4).map((kw, i) => (
                                          <Badge key={i} variant="secondary" className="bg-purple-500/20 text-purple-300 text-xs">
                                            {kw}
                                          </Badge>
                                        ))}
                                      </div>
                                    </div>
                                    <Badge className="bg-green-500/20 text-green-300 ml-4">
                                      {(result.relevance * 100).toFixed(0)}%
                                    </Badge>
                                  </div>
                                </CardContent>
                              </Card>
                            ))
                          ) : searchQuery ? (
                            <div className="text-center text-slate-400 py-8">
                              <Search className="h-12 w-12 mx-auto mb-4 opacity-50" />
                              <p>No results found</p>
                            </div>
                          ) : (
                            <div className="text-center text-slate-400 py-8">
                              <Search className="h-12 w-12 mx-auto mb-4 opacity-50" />
                              <p>Enter a query to search</p>
                            </div>
                          )}
                        </div>
                      </ScrollArea>
                    </div>
                  </TabsContent>

                  {/* Add New Zettel */}
                  <TabsContent value="add" className="m-0">
                    <div className="space-y-4">
                      <div>
                        <label htmlFor="zettel-content" className="text-sm text-slate-400 mb-2 block">Content</label>
                        <Textarea
                          id="zettel-content"
                          placeholder="Enter an atomic idea or piece of knowledge..."
                          value={newContent}
                          onChange={(e) => setNewContent(e.target.value)}
                          className="bg-slate-800/50 border-purple-500/20 min-h-[200px]"
                        />
                      </div>
                      <div>
                        <label htmlFor="zettel-source" className="text-sm text-slate-400 mb-2 block">Source (optional)</label>
                        <Input
                          id="zettel-source"
                          placeholder="Where did this knowledge come from?"
                          value={newSource}
                          onChange={(e) => setNewSource(e.target.value)}
                          className="bg-slate-800/50 border-purple-500/20"
                        />
                      </div>
                      <Button
                        onClick={handleAddZettel}
                        disabled={isAdding || !newContent.trim()}
                        className="w-full bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700"
                      >
                        {isAdding ? (
                          <>
                            <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                            Adding...
                          </>
                        ) : (
                          <>
                            <Plus className="h-4 w-4 mr-2" />
                            Add to Knowledge Graph
                          </>
                        )}
                      </Button>
                    </div>
                  </TabsContent>
                </CardContent>
              </Tabs>
            </Card>
          </div>

          {/* Right: Selected Zettel Details */}
          <div className="lg:col-span-1">
            <Card className="bg-slate-900/50 border-purple-500/20 sticky top-6">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Eye className="h-5 w-5 text-purple-400" />
                  Zettel Details
                </CardTitle>
              </CardHeader>
              <CardContent>
                {selectedZettel ? (
                  <div className="space-y-4">
                    <div>
                      <h4 className="text-xs text-slate-500 uppercase tracking-wide mb-1">Content</h4>
                      <p className="text-white text-sm">{selectedZettel.content}</p>
                    </div>

                    {selectedZettel.contextual_description && (
                      <div>
                        <h4 className="text-xs text-slate-500 uppercase tracking-wide mb-1">Context</h4>
                        <p className="text-slate-300 text-sm">{selectedZettel.contextual_description}</p>
                      </div>
                    )}

                    <div>
                      <h4 className="text-xs text-slate-500 uppercase tracking-wide mb-1">Keywords</h4>
                      <div className="flex flex-wrap gap-1">
                        {selectedZettel.keywords.map((kw, i) => (
                          <Badge key={i} variant="secondary" className="bg-purple-500/20 text-purple-300 text-xs">
                            {kw}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                      <div className="p-2 rounded bg-slate-800/50">
                        <p className="text-xs text-slate-500">Access Count</p>
                        <p className="text-lg font-bold text-purple-400">{selectedZettel.access_count}</p>
                      </div>
                      <div className="p-2 rounded bg-slate-800/50">
                        <p className="text-xs text-slate-500">Evolutions</p>
                        <p className="text-lg font-bold text-indigo-400">{selectedZettel.evolution_count}</p>
                      </div>
                    </div>

                    {selectedZettel.links.length > 0 && (
                      <div>
                        <h4 className="text-xs text-slate-500 uppercase tracking-wide mb-2">
                          Links ({selectedZettel.links.length})
                        </h4>
                        <ScrollArea className="h-[150px]">
                          <div className="space-y-2">
                            {selectedZettel.links.map((link, i) => (
                              <div 
                                key={i}
                                className="p-2 rounded bg-slate-800/50 cursor-pointer hover:bg-slate-800/80 transition-colors"
                                onClick={() => fetchZettelDetails(link.target_id)}
                              >
                                <div className="flex items-center justify-between">
                                  <span className="text-xs text-slate-400 truncate">
                                    {link.target_id.slice(0, 12)}...
                                  </span>
                                  <Badge variant="outline" className="text-xs">
                                    {link.link_type}
                                  </Badge>
                                </div>
                                <div className="mt-1 h-1 bg-slate-700 rounded-full overflow-hidden">
                                  <div 
                                    className="h-full bg-gradient-to-r from-purple-500 to-indigo-500"
                                    style={{ width: `${link.strength * 100}%` }}
                                  />
                                </div>
                              </div>
                            ))}
                          </div>
                        </ScrollArea>
                      </div>
                    )}

                    {selectedZettel.source && (
                      <div>
                        <h4 className="text-xs text-slate-500 uppercase tracking-wide mb-1">Source</h4>
                        <p className="text-slate-400 text-sm">{selectedZettel.source}</p>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center text-slate-400 py-8">
                    <Eye className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>Select a zettel to view details</p>
                    <p className="text-sm mt-1">Click on a node in the graph or a search result</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
