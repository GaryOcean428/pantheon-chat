import { useState, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Wrench, Play, Loader2, Search, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

interface Tool {
  tool_id: string;
  name: string;
  description: string;
  input_schema: Record<string, string>;
  times_used: number;
  success_rate: number;
  is_validated: boolean;
}

interface ToolPickerProps {
  onToolExecuted: (result: {
    toolName: string;
    inputs: Record<string, string>;
    output: string;
    success: boolean;
  }) => void;
}

export function ToolPicker({ onToolExecuted }: ToolPickerProps) {
  const [open, setOpen] = useState(false);
  const [tools, setTools] = useState<Tool[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedTool, setSelectedTool] = useState<Tool | null>(null);
  const [inputs, setInputs] = useState<Record<string, string>>({});
  const [executing, setExecuting] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [error, setError] = useState<string | null>(null);

  // Fetch available tools when dialog opens
  useEffect(() => {
    if (open) {
      fetchTools();
    }
  }, [open]);

  const fetchTools = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/olympus/tools/list");
      if (!response.ok) {
        throw new Error("Failed to fetch tools");
      }
      const data = await response.json();
      setTools(data.tools || []);
    } catch (err) {
      setError("Failed to load tools. The tool factory may not be available.");
      setTools([]);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectTool = (tool: Tool) => {
    setSelectedTool(tool);
    // Initialize inputs with empty values
    const initialInputs: Record<string, string> = {};
    if (tool.input_schema) {
      Object.keys(tool.input_schema).forEach((key) => {
        initialInputs[key] = "";
      });
    }
    setInputs(initialInputs);
  };

  const handleExecute = async () => {
    if (!selectedTool) return;

    setExecuting(true);
    setError(null);

    try {
      const response = await fetch("/api/olympus/tools/execute", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          tool_id: selectedTool.tool_id,
          args: inputs,
        }),
      });

      const data = await response.json();

      onToolExecuted({
        toolName: selectedTool.name,
        inputs,
        output: data.success ? JSON.stringify(data.result, null, 2) : data.error,
        success: data.success,
      });

      // Close dialog on success
      if (data.success) {
        setOpen(false);
        setSelectedTool(null);
        setInputs({});
      } else {
        setError(data.error || "Tool execution failed");
      }
    } catch (err) {
      setError("Failed to execute tool");
    } finally {
      setExecuting(false);
    }
  };

  const handleBack = () => {
    setSelectedTool(null);
    setInputs({});
    setError(null);
  };

  const filteredTools = tools.filter(
    (tool) =>
      tool.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      tool.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 text-muted-foreground hover:text-primary"
          title="Use Tool"
        >
          <Wrench className="h-4 w-4" />
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            {selectedTool ? selectedTool.name : "Available Tools"}
          </DialogTitle>
          <DialogDescription>
            {selectedTool
              ? selectedTool.description
              : "Select a tool to use in your conversation"}
          </DialogDescription>
        </DialogHeader>

        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            <span className="ml-2 text-muted-foreground">Loading tools...</span>
          </div>
        ) : selectedTool ? (
          // Tool input form
          <div className="space-y-4">
            {Object.entries(selectedTool.input_schema || {}).map(([key, type]) => (
              <div key={key} className="space-y-2">
                <Label htmlFor={key} className="capitalize">
                  {key.replace(/_/g, " ")}
                  <span className="ml-2 text-xs text-muted-foreground">({type})</span>
                </Label>
                <Input
                  id={key}
                  value={inputs[key] || ""}
                  onChange={(e) =>
                    setInputs((prev) => ({ ...prev, [key]: e.target.value }))
                  }
                  placeholder={`Enter ${key}...`}
                />
              </div>
            ))}

            {Object.keys(selectedTool.input_schema || {}).length === 0 && (
              <p className="text-sm text-muted-foreground py-4">
                This tool doesn't require any inputs.
              </p>
            )}

            {error && (
              <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-md">
                <p className="text-sm text-destructive">{error}</p>
              </div>
            )}

            <DialogFooter className="flex gap-2">
              <Button variant="outline" onClick={handleBack}>
                Back
              </Button>
              <Button onClick={handleExecute} disabled={executing}>
                {executing ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Executing...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Execute Tool
                  </>
                )}
              </Button>
            </DialogFooter>
          </div>
        ) : (
          // Tool list
          <div className="space-y-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search tools..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9"
              />
            </div>

            {error && (
              <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-md">
                <p className="text-sm text-destructive">{error}</p>
              </div>
            )}

            <ScrollArea className="h-[300px]">
              {filteredTools.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-8 text-center">
                  <Wrench className="h-10 w-10 text-muted-foreground/50 mb-3" />
                  <p className="text-muted-foreground">
                    {searchQuery
                      ? "No tools match your search"
                      : "No tools available yet"}
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Tools are generated automatically as the system learns
                  </p>
                </div>
              ) : (
                <div className="space-y-2">
                  {filteredTools.map((tool) => (
                    <button
                      key={tool.tool_id}
                      onClick={() => handleSelectTool(tool)}
                      className={cn(
                        "w-full p-3 text-left rounded-lg border transition-colors",
                        "hover:bg-accent hover:border-primary/50",
                        "focus:outline-none focus:ring-2 focus:ring-primary/50"
                      )}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <span className="font-medium">{tool.name}</span>
                            {tool.is_validated && (
                              <Badge variant="secondary" className="text-xs">
                                Validated
                              </Badge>
                            )}
                          </div>
                          <p className="text-sm text-muted-foreground mt-1 line-clamp-2">
                            {tool.description}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-3 mt-2 text-xs text-muted-foreground">
                        <span>Used {tool.times_used}x</span>
                        <span>
                          {Math.round(tool.success_rate * 100)}% success
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </ScrollArea>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
