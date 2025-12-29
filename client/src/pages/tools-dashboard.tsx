import { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useToast } from "@/hooks/use-toast";
import {
  Wrench,
  Play,
  Search,
  RefreshCw,
  Clock,
  CheckCircle2,
  XCircle,
  Sparkles,
  Code,
  Zap,
  TrendingUp,
  Activity,
  Loader2,
} from "lucide-react";

interface Tool {
  tool_id: string;
  name: string;
  description: string;
  input_schema: Record<string, string>;
  output_schema?: { type: string };
  source_pattern_id?: string;
  times_used: number;
  times_succeeded: number;
  times_failed: number;
  created_at: number;
}

interface ExecutionResult {
  tool_id: string;
  tool_name: string;
  success: boolean;
  result?: unknown;
  error?: string;
  timestamp: number;
  inputs: Record<string, unknown>;
}

export default function ToolsDashboard() {
  const [tools, setTools] = useState<Tool[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedTool, setSelectedTool] = useState<Tool | null>(null);
  const [toolInputs, setToolInputs] = useState<Record<string, string>>({});
  const [executing, setExecuting] = useState(false);
  const [executionResults, setExecutionResults] = useState<ExecutionResult[]>([]);
  const [dialogOpen, setDialogOpen] = useState(false);
  const { toast } = useToast();

  const fetchTools = async () => {
    setLoading(true);
    try {
      const response = await fetch("/api/tools/list");
      if (response.ok) {
        const data = await response.json();
        setTools(data.tools || []);
      } else {
        toast({
          title: "Error",
          description: "Failed to fetch tools",
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error("Failed to fetch tools:", error);
      toast({
        title: "Error",
        description: "Failed to connect to tools API",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTools();
  }, []);

  const handleSelectTool = (tool: Tool) => {
    setSelectedTool(tool);
    const initialInputs: Record<string, string> = {};
    if (tool.input_schema) {
      Object.keys(tool.input_schema).forEach((key) => {
        initialInputs[key] = "";
      });
    }
    setToolInputs(initialInputs);
    setDialogOpen(true);
  };

  const handleExecuteTool = async () => {
    if (!selectedTool) return;

    setExecuting(true);
    try {
      // Parse inputs based on expected types
      const parsedInputs: Record<string, unknown> = {};
      Object.entries(toolInputs).forEach(([key, value]) => {
        const expectedType = selectedTool.input_schema?.[key];
        if (expectedType === "int" || expectedType === "float" || expectedType === "number") {
          parsedInputs[key] = parseFloat(value) || 0;
        } else if (expectedType === "bool" || expectedType === "boolean") {
          parsedInputs[key] = value.toLowerCase() === "true";
        } else if (expectedType === "list" || expectedType === "array") {
          try {
            parsedInputs[key] = JSON.parse(value);
          } catch {
            parsedInputs[key] = value.split(",").map((s) => s.trim());
          }
        } else if (expectedType === "dict" || expectedType === "object") {
          try {
            parsedInputs[key] = JSON.parse(value);
          } catch {
            parsedInputs[key] = {};
          }
        } else {
          parsedInputs[key] = value;
        }
      });

      const response = await fetch("/api/tools/execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tool_id: selectedTool.tool_id,
          inputs: parsedInputs,
        }),
      });

      const data = await response.json();

      const result: ExecutionResult = {
        tool_id: selectedTool.tool_id,
        tool_name: selectedTool.name,
        success: data.success,
        result: data.result,
        error: data.error,
        timestamp: Date.now(),
        inputs: parsedInputs,
      };

      setExecutionResults((prev) => [result, ...prev].slice(0, 50));

      if (data.success) {
        toast({
          title: "Tool Executed",
          description: `${selectedTool.name} completed successfully`,
        });
      } else {
        toast({
          title: "Execution Failed",
          description: data.error || "Unknown error",
          variant: "destructive",
        });
      }

      // Refresh tools to update usage stats
      fetchTools();
    } catch (error) {
      console.error("Tool execution failed:", error);
      toast({
        title: "Error",
        description: "Failed to execute tool",
        variant: "destructive",
      });
    } finally {
      setExecuting(false);
    }
  };

  const filteredTools = tools.filter(
    (tool) =>
      tool.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      tool.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const totalTools = tools.length;
  const totalExecutions = tools.reduce((sum, t) => sum + t.times_used, 0);
  const totalSuccesses = tools.reduce((sum, t) => sum + t.times_succeeded, 0);
  const overallSuccessRate = totalExecutions > 0 ? (totalSuccesses / totalExecutions) * 100 : 0;

  const getSuccessRate = (tool: Tool) => {
    if (tool.times_used === 0) return null;
    return (tool.times_succeeded / tool.times_used) * 100;
  };

  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Wrench className="h-8 w-8" />
            Tools Dashboard
          </h1>
          <p className="text-muted-foreground mt-1">
            Browse, execute, and manage generated tools
          </p>
        </div>
        <Button onClick={fetchTools} disabled={loading} variant="outline">
          <RefreshCw className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Total Tools
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Wrench className="h-5 w-5 text-primary" />
              <span className="text-2xl font-bold">{totalTools}</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Total Executions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-yellow-500" />
              <span className="text-2xl font-bold">{totalExecutions}</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Success Rate
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-green-500" />
              <span className="text-2xl font-bold">{overallSuccessRate.toFixed(1)}%</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Recent Runs
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-blue-500" />
              <span className="text-2xl font-bold">{executionResults.length}</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="tools" className="space-y-4">
        <TabsList>
          <TabsTrigger value="tools">Available Tools</TabsTrigger>
          <TabsTrigger value="history">Execution History</TabsTrigger>
        </TabsList>

        <TabsContent value="tools" className="space-y-4">
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search tools by name or description..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>

          {/* Tools Grid */}
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : filteredTools.length === 0 ? (
            <Card>
              <CardContent className="py-12 text-center">
                <Sparkles className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold">No Tools Found</h3>
                <p className="text-muted-foreground mt-1">
                  {searchQuery
                    ? "No tools match your search query"
                    : "No tools have been generated yet. Tools are created automatically as the system learns."}
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredTools.map((tool) => {
                const successRate = getSuccessRate(tool);
                return (
                  <Card key={tool.tool_id} className="hover:border-primary/50 transition-colors">
                    <CardHeader className="pb-2">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <CardTitle className="text-base flex items-center gap-2">
                            <Code className="h-4 w-4 text-primary" />
                            {tool.name}
                          </CardTitle>
                          <CardDescription className="mt-1 line-clamp-2">
                            {tool.description}
                          </CardDescription>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      {/* Input Schema */}
                      {tool.input_schema && Object.keys(tool.input_schema).length > 0 && (
                        <div className="flex flex-wrap gap-1">
                          {Object.entries(tool.input_schema).map(([key, type]) => (
                            <Badge key={key} variant="secondary" className="text-xs">
                              {key}: {String(type)}
                            </Badge>
                          ))}
                        </div>
                      )}

                      {/* Stats */}
                      <div className="flex items-center gap-4 text-sm text-muted-foreground">
                        <span className="flex items-center gap-1">
                          <Zap className="h-3 w-3" />
                          {tool.times_used} runs
                        </span>
                        {successRate !== null && (
                          <span
                            className={`flex items-center gap-1 ${
                              successRate >= 70
                                ? "text-green-600"
                                : successRate >= 50
                                ? "text-yellow-600"
                                : "text-red-600"
                            }`}
                          >
                            {successRate >= 70 ? (
                              <CheckCircle2 className="h-3 w-3" />
                            ) : (
                              <XCircle className="h-3 w-3" />
                            )}
                            {successRate.toFixed(0)}% success
                          </span>
                        )}
                      </div>

                      {/* Execute Button */}
                      <Button
                        onClick={() => handleSelectTool(tool)}
                        className="w-full"
                        size="sm"
                      >
                        <Play className="h-4 w-4 mr-2" />
                        Execute Tool
                      </Button>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          )}
        </TabsContent>

        <TabsContent value="history" className="space-y-4">
          {executionResults.length === 0 ? (
            <Card>
              <CardContent className="py-12 text-center">
                <Clock className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold">No Execution History</h3>
                <p className="text-muted-foreground mt-1">
                  Tool executions from this session will appear here
                </p>
              </CardContent>
            </Card>
          ) : (
            <ScrollArea className="h-[600px]">
              <div className="space-y-3">
                {executionResults.map((result, index) => (
                  <Card key={index} className={result.success ? "border-green-500/30" : "border-red-500/30"}>
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-sm flex items-center gap-2">
                          {result.success ? (
                            <CheckCircle2 className="h-4 w-4 text-green-500" />
                          ) : (
                            <XCircle className="h-4 w-4 text-red-500" />
                          )}
                          {result.tool_name}
                        </CardTitle>
                        <span className="text-xs text-muted-foreground">
                          {formatDate(result.timestamp)}
                        </span>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-2">
                      <div>
                        <Label className="text-xs text-muted-foreground">Inputs</Label>
                        <pre className="text-xs bg-muted p-2 rounded mt-1 overflow-x-auto">
                          {JSON.stringify(result.inputs, null, 2)}
                        </pre>
                      </div>
                      {result.success ? (
                        <div>
                          <Label className="text-xs text-muted-foreground">Result</Label>
                          <pre className="text-xs bg-muted p-2 rounded mt-1 overflow-x-auto max-h-32">
                            {JSON.stringify(result.result, null, 2)}
                          </pre>
                        </div>
                      ) : (
                        <div>
                          <Label className="text-xs text-red-500">Error</Label>
                          <p className="text-xs text-red-500 mt-1">{result.error}</p>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </ScrollArea>
          )}
        </TabsContent>
      </Tabs>

      {/* Execute Tool Dialog */}
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Code className="h-5 w-5" />
              {selectedTool?.name}
            </DialogTitle>
            <DialogDescription>{selectedTool?.description}</DialogDescription>
          </DialogHeader>

          {selectedTool && (
            <div className="space-y-4">
              {/* Input Fields */}
              {selectedTool.input_schema &&
                Object.entries(selectedTool.input_schema).map(([key, type]) => (
                  <div key={key} className="space-y-2">
                    <Label htmlFor={key}>
                      {key}
                      <span className="text-xs text-muted-foreground ml-2">({String(type)})</span>
                    </Label>
                    <Input
                      id={key}
                      value={toolInputs[key] || ""}
                      onChange={(e) =>
                        setToolInputs((prev) => ({ ...prev, [key]: e.target.value }))
                      }
                      placeholder={`Enter ${key}...`}
                    />
                  </div>
                ))}

              {/* Execute Button */}
              <Button
                onClick={handleExecuteTool}
                disabled={executing}
                className="w-full"
              >
                {executing ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Executing...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Execute
                  </>
                )}
              </Button>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
