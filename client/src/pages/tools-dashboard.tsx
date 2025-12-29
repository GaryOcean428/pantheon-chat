/**
 * Tools Dashboard - Unified Tool Management
 * 
 * Merged from tools-dashboard.tsx and tool-factory-dashboard.tsx
 * Features:
 * - Generated tools listing and execution
 * - Pattern management and discovery
 * - Factory health metrics
 * - Research bridge status
 */

import { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Progress } from '@/components/ui/progress';
import { useToast } from '@/hooks/use-toast';
import {
  Wrench,
  Play,
  RefreshCw,
  CheckCircle,
  XCircle,
  Loader2,
  Code,
  Sparkles,
  Activity,
  Brain,
  Database,
  Zap,
  TrendingUp,
  Search,
} from 'lucide-react';

interface Tool {
  tool_id: string;
  name: string;
  description: string;
  input_schema: Record<string, string>;
  output_schema: Record<string, string>;
  times_used: number;
  times_succeeded: number;
  times_failed: number;
  created_at: number;
}

interface Pattern {
  pattern_id: string;
  description: string;
  source_type: string;
  times_used: number;
  success_rate: number;
  created_at: number;
}

interface FactoryHealth {
  patterns_count: number;
  tools_deployed: number;
  total_executions: number;
  success_rate: number;
  research_bridge_connected: boolean;
  last_pattern_learned: string | null;
}

export default function ToolsDashboard() {
  const { toast } = useToast();
  const [tools, setTools] = useState<Tool[]>([]);
  const [patterns, setPatterns] = useState<Pattern[]>([]);
  const [factoryHealth, setFactoryHealth] = useState<FactoryHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [executing, setExecuting] = useState<string | null>(null);
  const [selectedTool, setSelectedTool] = useState<Tool | null>(null);
  const [toolInputs, setToolInputs] = useState<Record<string, string>>({});
  const [executionResult, setExecutionResult] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    fetchAllData();
  }, []);

  const fetchAllData = async () => {
    setLoading(true);
    await Promise.all([
      fetchTools(),
      fetchPatterns(),
      fetchFactoryHealth(),
    ]);
    setLoading(false);
  };

  const fetchTools = async () => {
    try {
      const response = await fetch('/api/tools/list');
      if (response.ok) {
        const data = await response.json();
        setTools(data.tools || []);
      }
    } catch (error) {
      console.error('Failed to fetch tools:', error);
    }
  };

  const fetchPatterns = async () => {
    try {
      const response = await fetch('/api/tools/patterns');
      if (response.ok) {
        const data = await response.json();
        setPatterns(data.patterns || []);
      }
    } catch (error) {
      console.error('Failed to fetch patterns:', error);
    }
  };

  const fetchFactoryHealth = async () => {
    try {
      const response = await fetch('/api/tools/health');
      if (response.ok) {
        const data = await response.json();
        setFactoryHealth(data);
      }
    } catch (error) {
      console.error('Failed to fetch factory health:', error);
    }
  };

  const executeTool = async (tool: Tool) => {
    setExecuting(tool.tool_id);
    setExecutionResult(null);
    
    try {
      const response = await fetch('/api/tools/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tool_id: tool.tool_id,
          inputs: toolInputs,
        }),
      });
      
      const data = await response.json();
      
      if (data.success) {
        setExecutionResult(JSON.stringify(data.result, null, 2));
        toast({
          title: 'Tool executed successfully',
          description: `${tool.name} completed`,
        });
        // Refresh to update usage stats
        fetchTools();
      } else {
        setExecutionResult(`Error: ${data.error}`);
        toast({
          title: 'Tool execution failed',
          description: data.error,
          variant: 'destructive',
        });
      }
    } catch (error) {
      setExecutionResult(`Error: ${error}`);
      toast({
        title: 'Execution failed',
        description: 'Could not connect to tool service',
        variant: 'destructive',
      });
    } finally {
      setExecuting(null);
    }
  };

  const generateToolFromResearch = async () => {
    try {
      const response = await fetch('/api/tools/generate-from-research', {
        method: 'POST',
      });
      const data = await response.json();
      
      if (data.success) {
        toast({
          title: 'Tool generation started',
          description: data.message || 'Research-driven tool generation initiated',
        });
        setTimeout(fetchAllData, 2000);
      } else {
        toast({
          title: 'Generation failed',
          description: data.error,
          variant: 'destructive',
        });
      }
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Could not start tool generation',
        variant: 'destructive',
      });
    }
  };

  const filteredTools = tools.filter(tool =>
    tool.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    tool.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const filteredPatterns = patterns.filter(pattern =>
    pattern.pattern_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
    pattern.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getSuccessRateColor = (rate: number) => {
    if (rate >= 0.8) return 'text-green-500';
    if (rate >= 0.5) return 'text-yellow-500';
    return 'text-red-500';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <span className="ml-2">Loading Tool Factory...</span>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Wrench className="h-8 w-8" />
            Tool Factory
          </h1>
          <p className="text-muted-foreground mt-1">
            Self-generating tools for autonomous capability expansion
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={fetchAllData}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Button onClick={generateToolFromResearch}>
            <Sparkles className="h-4 w-4 mr-2" />
            Generate from Research
          </Button>
        </div>
      </div>

      {/* Factory Health Metrics */}
      {factoryHealth && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          <Card>
            <CardContent className="pt-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Patterns</p>
                  <p className="text-2xl font-bold">{factoryHealth.patterns_count}</p>
                </div>
                <Database className="h-8 w-8 text-blue-500" />
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="pt-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Tools Deployed</p>
                  <p className="text-2xl font-bold">{factoryHealth.tools_deployed}</p>
                </div>
                <Wrench className="h-8 w-8 text-purple-500" />
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="pt-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Executions</p>
                  <p className="text-2xl font-bold">{factoryHealth.total_executions}</p>
                </div>
                <Zap className="h-8 w-8 text-yellow-500" />
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="pt-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Success Rate</p>
                  <p className={`text-2xl font-bold ${getSuccessRateColor(factoryHealth.success_rate)}`}>
                    {(factoryHealth.success_rate * 100).toFixed(0)}%
                  </p>
                </div>
                <TrendingUp className="h-8 w-8 text-green-500" />
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="pt-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Research Bridge</p>
                  <p className="text-lg font-medium">
                    {factoryHealth.research_bridge_connected ? (
                      <span className="flex items-center text-green-500">
                        <CheckCircle className="h-4 w-4 mr-1" /> Connected
                      </span>
                    ) : (
                      <span className="flex items-center text-red-500">
                        <XCircle className="h-4 w-4 mr-1" /> Offline
                      </span>
                    )}
                  </p>
                </div>
                <Brain className="h-8 w-8 text-pink-500" />
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          placeholder="Search tools and patterns..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="pl-10"
        />
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="tools" className="space-y-4">
        <TabsList>
          <TabsTrigger value="tools" className="flex items-center gap-2">
            <Wrench className="h-4 w-4" />
            Generated Tools ({filteredTools.length})
          </TabsTrigger>
          <TabsTrigger value="patterns" className="flex items-center gap-2">
            <Code className="h-4 w-4" />
            Patterns ({filteredPatterns.length})
          </TabsTrigger>
          <TabsTrigger value="activity" className="flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Activity
          </TabsTrigger>
        </TabsList>

        {/* Tools Tab */}
        <TabsContent value="tools" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Tools List */}
            <Card className="lg:col-span-1">
              <CardHeader>
                <CardTitle>Available Tools</CardTitle>
                <CardDescription>
                  {filteredTools.length} tools ready for execution
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[500px]">
                  <div className="space-y-2">
                    {filteredTools.length === 0 ? (
                      <div className="text-center py-8 text-muted-foreground">
                        <Wrench className="h-12 w-12 mx-auto mb-2 opacity-50" />
                        <p>No tools generated yet</p>
                        <p className="text-sm">Click "Generate from Research" to create tools</p>
                      </div>
                    ) : (
                      filteredTools.map((tool) => {
                        const successRate = tool.times_used > 0 
                          ? tool.times_succeeded / tool.times_used 
                          : 0;
                        
                        return (
                          <Card
                            key={tool.tool_id}
                            className={`cursor-pointer transition-colors hover:bg-accent ${
                              selectedTool?.tool_id === tool.tool_id ? 'border-primary' : ''
                            }`}
                            onClick={() => {
                              setSelectedTool(tool);
                              setToolInputs({});
                              setExecutionResult(null);
                            }}
                          >
                            <CardContent className="p-4">
                              <div className="flex items-start justify-between">
                                <div className="flex-1">
                                  <h4 className="font-medium">{tool.name}</h4>
                                  <p className="text-sm text-muted-foreground line-clamp-2">
                                    {tool.description}
                                  </p>
                                </div>
                                <Badge variant="outline" className="ml-2">
                                  {tool.times_used} uses
                                </Badge>
                              </div>
                              {tool.times_used > 0 && (
                                <div className="mt-2">
                                  <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
                                    <span>Success rate</span>
                                    <span className={getSuccessRateColor(successRate)}>
                                      {(successRate * 100).toFixed(0)}%
                                    </span>
                                  </div>
                                  <Progress value={successRate * 100} className="h-1" />
                                </div>
                              )}
                            </CardContent>
                          </Card>
                        );
                      })
                    )}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>

            {/* Tool Execution Panel */}
            <Card className="lg:col-span-1">
              <CardHeader>
                <CardTitle>
                  {selectedTool ? selectedTool.name : 'Select a Tool'}
                </CardTitle>
                <CardDescription>
                  {selectedTool ? selectedTool.description : 'Click a tool to execute it'}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {selectedTool ? (
                  <div className="space-y-4">
                    {/* Input Fields */}
                    {selectedTool.input_schema && Object.keys(selectedTool.input_schema).length > 0 && (
                      <div className="space-y-3">
                        <h4 className="text-sm font-medium">Inputs</h4>
                        {Object.entries(selectedTool.input_schema).map(([key, type]) => (
                          <div key={key}>
                            <label className="text-sm text-muted-foreground">
                              {key} <span className="text-xs">({String(type)})</span>
                            </label>
                            <Input
                              placeholder={`Enter ${key}...`}
                              value={toolInputs[key] || ''}
                              onChange={(e) => setToolInputs(prev => ({
                                ...prev,
                                [key]: e.target.value
                              }))}
                            />
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Execute Button */}
                    <Button
                      className="w-full"
                      onClick={() => executeTool(selectedTool)}
                      disabled={executing === selectedTool.tool_id}
                    >
                      {executing === selectedTool.tool_id ? (
                        <>
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                          Executing...
                        </>
                      ) : (
                        <>
                          <Play className="h-4 w-4 mr-2" />
                          Execute Tool
                        </>
                      )}
                    </Button>

                    {/* Execution Result */}
                    {executionResult && (
                      <div className="mt-4">
                        <h4 className="text-sm font-medium mb-2">Result</h4>
                        <ScrollArea className="h-48">
                          <pre className="text-xs bg-muted p-3 rounded-lg overflow-x-auto">
                            {executionResult}
                          </pre>
                        </ScrollArea>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center py-12 text-muted-foreground">
                    <Play className="h-12 w-12 mx-auto mb-2 opacity-50" />
                    <p>Select a tool from the list to execute it</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Patterns Tab */}
        <TabsContent value="patterns" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Learned Patterns</CardTitle>
              <CardDescription>
                Code patterns the factory has learned for tool generation
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[500px]">
                <div className="space-y-2">
                  {filteredPatterns.length === 0 ? (
                    <div className="text-center py-8 text-muted-foreground">
                      <Code className="h-12 w-12 mx-auto mb-2 opacity-50" />
                      <p>No patterns learned yet</p>
                      <p className="text-sm">Patterns are discovered from research and user templates</p>
                    </div>
                  ) : (
                    filteredPatterns.map((pattern) => (
                      <Card key={pattern.pattern_id}>
                        <CardContent className="p-4">
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <div className="flex items-center gap-2">
                                <h4 className="font-mono text-sm">{pattern.pattern_id}</h4>
                                <Badge variant="secondary">{pattern.source_type}</Badge>
                              </div>
                              <p className="text-sm text-muted-foreground mt-1">
                                {pattern.description}
                              </p>
                            </div>
                            <div className="text-right">
                              <p className="text-sm">{pattern.times_used} uses</p>
                              <p className={`text-sm ${getSuccessRateColor(pattern.success_rate)}`}>
                                {(pattern.success_rate * 100).toFixed(0)}% success
                              </p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))
                  )}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Activity Tab */}
        <TabsContent value="activity" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Factory Activity</CardTitle>
              <CardDescription>
                Recent tool generations and executions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {factoryHealth?.last_pattern_learned && (
                  <div className="flex items-center gap-3 p-3 bg-muted rounded-lg">
                    <Sparkles className="h-5 w-5 text-yellow-500" />
                    <div>
                      <p className="font-medium">Last Pattern Learned</p>
                      <p className="text-sm text-muted-foreground">
                        {factoryHealth.last_pattern_learned}
                      </p>
                    </div>
                  </div>
                )}
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 bg-muted rounded-lg">
                    <p className="text-sm text-muted-foreground">Self-Learning</p>
                    <p className="text-lg font-medium flex items-center gap-2">
                      <Brain className="h-5 w-5 text-purple-500" />
                      {factoryHealth?.research_bridge_connected ? 'Active' : 'Inactive'}
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Tools learn from successful executions
                    </p>
                  </div>
                  
                  <div className="p-4 bg-muted rounded-lg">
                    <p className="text-sm text-muted-foreground">Economic Impact</p>
                    <p className="text-lg font-medium flex items-center gap-2">
                      <Zap className="h-5 w-5 text-yellow-500" />
                      {factoryHealth?.total_executions || 0} executions
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Each execution generates value
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
