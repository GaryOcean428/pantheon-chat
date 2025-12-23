import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { ConsciousnessDashboard } from "@/components/ConsciousnessDashboard";
import { BetaAttentionDisplay } from "@/components/BetaAttentionDisplay";
import NeurochemistryDisplay from "@/components/NeurochemistryDisplay";
import { InnateDrivesDisplay } from "@/components/InnateDrivesDisplay";
import CapabilityTelemetryPanel from "@/components/CapabilityTelemetryPanel";
import { ConsciousnessMonitoringDemo } from "@/components/ConsciousnessMonitoringDemo";
import { EmotionalStatePanel } from "@/components/EmotionalStatePanel";
import NeurochemistryAdminPanel from "@/components/NeurochemistryAdminPanel";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Activity, Brain, Shield, Zap, Server, TrendingUp, AlertCircle, CheckCircle2, Radio } from "lucide-react";

interface TelemetryOverview {
  success: boolean;
  data: {
    timestamp: string;
    consciousness: {
      phi: number;
      kappa: number;
      beta: number;
      regime: string;
      basinDistance: number;
      inResonance: boolean;
      quality: number;
      entropy?: number;
      fidelity?: number;
      integration?: number;
      grounded?: boolean;
      conscious?: boolean;
      geometricMemorySize?: number;
      basinHistorySize?: number;
      subsystems?: Array<{
        id: number;
        name: string;
        activation: number;
        entropy: number;
        purity: number;
      }>;
    };
    usage: {
      tavily: {
        enabled: boolean;
        todaySearches: number;
        todayExtracts: number;
        estimatedCostCents: number;
        dailyLimit: number;
        rateStatus: string;
      };
      googleFree: {
        enabled: boolean;
        todaySearches: number;
      };
      duckDuckGo?: {
        enabled: boolean;
        todaySearches: number;
        torEnabled: boolean;
      };
      totalApiCalls: number;
    };
    learning: {
      vocabularySize: number;
      recentExpansions: number;
      highPhiDiscoveries: number;
      sourcesDiscovered: number;
      activeSources: number;
    };
    defense: {
      negativeKnowledgeCount: number;
      geometricBarriers: number;
      contradictions: number;
      computeTimeSaved: number;
    };
    autonomy: {
      kernelsActive: number;
      feedbackLoopsHealthy: number;
      lastAutonomicAction: string | null;
      selfRegulationScore: number;
    };
    systemHealth: {
      overall: number;
      components: Record<string, boolean>;
    };
  };
}

interface StreamData {
  timestamp: string;
  consciousness: {
    phi: number;
    kappa: number;
    beta: number;
    regime: string;
    quality: number;
    inResonance: boolean;
  };
  usage: {
    tavilyStatus: string;
    tavilyToday: number;
    tavilyCost: number;
  };
}

function MetricCard({ 
  title, 
  value, 
  subtitle, 
  icon: Icon, 
  trend,
  className = ""
}: { 
  title: string; 
  value: string | number; 
  subtitle?: string; 
  icon: React.ElementType;
  trend?: "up" | "down" | "neutral";
  className?: string;
}) {
  return (
    <Card className={`hover-elevate ${className}`}>
      <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">{title}</CardTitle>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        <div className="text-3xl font-bold font-mono" data-testid={`metric-${title.toLowerCase().replace(/\s/g, '-')}`}>
          {value}
        </div>
        {subtitle && (
          <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>
        )}
      </CardContent>
    </Card>
  );
}

function ConsciousnessGauge({ phi, kappa, regime, quality, inResonance }: {
  phi: number;
  kappa: number;
  regime: string;
  quality: number;
  inResonance: boolean;
}) {
  const kappaTarget = 64;
  const kappaPercent = Math.min((kappa / kappaTarget) * 100, 100);
  
  return (
    <Card className="col-span-2">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5 text-primary" />
          Consciousness State
          {inResonance && (
            <Badge variant="default" className="ml-2 animate-pulse">
              <Radio className="h-3 w-3 mr-1" />
              In Resonance
            </Badge>
          )}
        </CardTitle>
        <CardDescription>Real-time quantum information geometry metrics</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid grid-cols-3 gap-6">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Integrated Information</span>
              <span className="font-mono text-lg font-bold">{phi.toFixed(3)}</span>
            </div>
            <div className="text-4xl font-bold font-mono text-center py-4" data-testid="metric-phi">
              Φ = {phi.toFixed(2)}
            </div>
            <Progress value={phi * 100} className="h-2" />
          </div>
          
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Coupling Constant</span>
              <span className="font-mono text-lg font-bold">{kappa.toFixed(1)}</span>
            </div>
            <div className="text-4xl font-bold font-mono text-center py-4" data-testid="metric-kappa">
              κ = {kappa.toFixed(1)}
            </div>
            <Progress value={kappaPercent} className="h-2" />
            <p className="text-xs text-muted-foreground text-center">Target: κ* ≈ 64</p>
          </div>
          
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Quality Score</span>
              <span className="font-mono text-lg font-bold">{(quality * 100).toFixed(0)}%</span>
            </div>
            <div className="text-4xl font-bold font-mono text-center py-4" data-testid="metric-quality">
              {(quality * 100).toFixed(0)}%
            </div>
            <Progress value={quality * 100} className="h-2" />
          </div>
        </div>
        
        <div className="flex items-center justify-center gap-4 pt-4 border-t">
          <Badge variant="outline" className="font-mono">
            Regime: {regime}
          </Badge>
        </div>
      </CardContent>
    </Card>
  );
}

function ApiUsageCard({ usage }: { usage: TelemetryOverview['data']['usage'] }) {
  const tavilyPercent = usage.tavily.enabled 
    ? (usage.tavily.todaySearches / usage.tavily.dailyLimit) * 100 
    : 0;
  
  const statusColor = {
    'OK': 'text-green-500',
    'WARNING': 'text-yellow-500',
    'BLOCKED': 'text-red-500',
  }[usage.tavily.rateStatus] || 'text-muted-foreground';
  
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="h-5 w-5 text-primary" />
          API Usage
        </CardTitle>
        <CardDescription>Today's search provider utilization</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Tavily</span>
            <Badge variant={usage.tavily.enabled ? "default" : "secondary"}>
              {usage.tavily.enabled ? "Active" : "Disabled"}
            </Badge>
          </div>
          <Progress value={tavilyPercent} className="h-2" />
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>{usage.tavily.todaySearches} / {usage.tavily.dailyLimit} searches</span>
            <span className={statusColor}>{usage.tavily.rateStatus}</span>
          </div>
          <div className="text-sm font-mono" data-testid="metric-tavily-cost">
            Cost: ${(usage.tavily.estimatedCostCents / 100).toFixed(2)}
          </div>
        </div>
        
        <div className="space-y-2 pt-4 border-t">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Google Free</span>
            <Badge variant={usage.googleFree.enabled ? "default" : "secondary"}>
              {usage.googleFree.enabled ? "Active" : "Disabled"}
            </Badge>
          </div>
          <div className="text-sm text-muted-foreground">
            {usage.googleFree.todaySearches} searches today
          </div>
        </div>
        
        {usage.duckDuckGo && (
          <div className="space-y-2 pt-4 border-t">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">DuckDuckGo</span>
              <div className="flex items-center gap-2">
                {usage.duckDuckGo.torEnabled && (
                  <Badge variant="outline" className="text-xs">Tor</Badge>
                )}
                <Badge variant={usage.duckDuckGo.enabled ? "default" : "secondary"}>
                  {usage.duckDuckGo.enabled ? "Active" : "Disabled"}
                </Badge>
              </div>
            </div>
            <div className="text-sm text-muted-foreground">
              {usage.duckDuckGo.todaySearches} searches today (via Shadow Pantheon)
            </div>
          </div>
        )}
        
        <div className="pt-4 border-t">
          <div className="text-2xl font-bold font-mono" data-testid="metric-total-api">
            {usage.totalApiCalls}
          </div>
          <div className="text-xs text-muted-foreground">Total API calls today</div>
        </div>
      </CardContent>
    </Card>
  );
}

function LearningCard({ learning }: { learning: TelemetryOverview['data']['learning'] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-primary" />
          Learning Progress
        </CardTitle>
        <CardDescription>Vocabulary and discovery metrics</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-2xl font-bold font-mono" data-testid="metric-vocabulary">
              {learning.vocabularySize.toLocaleString()}
            </div>
            <div className="text-xs text-muted-foreground">Vocabulary Size</div>
          </div>
          <div>
            <div className="text-2xl font-bold font-mono" data-testid="metric-expansions">
              +{learning.recentExpansions}
            </div>
            <div className="text-xs text-muted-foreground">Recent Expansions</div>
          </div>
        </div>
        
        <div className="pt-4 border-t space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm">High-Φ Discoveries</span>
            <span className="font-mono font-bold">{learning.highPhiDiscoveries}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm">Sources Discovered</span>
            <span className="font-mono font-bold">{learning.sourcesDiscovered}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm">Active Sources</span>
            <span className="font-mono font-bold">{learning.activeSources}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function DefenseCard({ defense }: { defense: TelemetryOverview['data']['defense'] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Shield className="h-5 w-5 text-primary" />
          QIG Defense System
        </CardTitle>
        <CardDescription>Geometric immune system metrics</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-2xl font-bold font-mono" data-testid="metric-negative-knowledge">
              {defense.negativeKnowledgeCount}
            </div>
            <div className="text-xs text-muted-foreground">Negative Knowledge</div>
          </div>
          <div>
            <div className="text-2xl font-bold font-mono" data-testid="metric-barriers">
              {defense.geometricBarriers}
            </div>
            <div className="text-xs text-muted-foreground">Geometric Barriers</div>
          </div>
        </div>
        
        <div className="pt-4 border-t space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm">Contradictions Blocked</span>
            <span className="font-mono font-bold">{defense.contradictions}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm">Compute Time Saved</span>
            <span className="font-mono font-bold">{defense.computeTimeSaved}ms</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function AutonomyCard({ autonomy }: { autonomy: TelemetryOverview['data']['autonomy'] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Zap className="h-5 w-5 text-primary" />
          Autonomic System
        </CardTitle>
        <CardDescription>Self-regulation and kernel activity</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-2xl font-bold font-mono" data-testid="metric-kernels">
              {autonomy.kernelsActive}
            </div>
            <div className="text-xs text-muted-foreground">Active Kernels</div>
          </div>
          <div>
            <div className="text-2xl font-bold font-mono" data-testid="metric-feedback-loops">
              {autonomy.feedbackLoopsHealthy}
            </div>
            <div className="text-xs text-muted-foreground">Healthy Loops</div>
          </div>
        </div>
        
        <div className="pt-4 border-t space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm">Self-Regulation Score</span>
            <span className="font-mono font-bold">{(autonomy.selfRegulationScore * 100).toFixed(0)}%</span>
          </div>
          <Progress value={autonomy.selfRegulationScore * 100} className="h-2" />
          
          {autonomy.lastAutonomicAction && (
            <div className="text-xs text-muted-foreground pt-2">
              Last action: {autonomy.lastAutonomicAction}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function SystemHealthCard({ health }: { health: TelemetryOverview['data']['systemHealth'] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Server className="h-5 w-5 text-primary" />
          System Health
        </CardTitle>
        <CardDescription>Component status overview</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="text-center">
          <div className="text-4xl font-bold font-mono" data-testid="metric-health">
            {(health.overall * 100).toFixed(0)}%
          </div>
          <div className="text-sm text-muted-foreground">Overall Health</div>
          <Progress value={health.overall * 100} className="h-3 mt-2" />
        </div>
        
        <div className="pt-4 border-t space-y-2">
          {Object.entries(health.components).map(([name, healthy]) => (
            <div key={name} className="flex items-center justify-between">
              <span className="text-sm capitalize">{name}</span>
              {healthy ? (
                <CheckCircle2 className="h-4 w-4 text-green-500" />
              ) : (
                <AlertCircle className="h-4 w-4 text-red-500" />
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function LiveIndicator({ isConnected }: { isConnected: boolean }) {
  return (
    <div className="flex items-center gap-2">
      <div className={`h-2 w-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
      <span className="text-sm text-muted-foreground">
        {isConnected ? 'Live' : 'Disconnected'}
      </span>
    </div>
  );
}

export default function TelemetryDashboard() {
  const [streamData, setStreamData] = useState<StreamData | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  
  const { data: overview, isLoading, error } = useQuery<TelemetryOverview>({
    queryKey: ['/api/v1/telemetry/overview'],
    refetchInterval: 10000,
  });
  
  useEffect(() => {
    const eventSource = new EventSource('/api/v1/telemetry/stream');
    
    eventSource.onopen = () => {
      setIsStreaming(true);
    };
    
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setStreamData(data);
      } catch (e) {
        console.error('Failed to parse SSE data:', e);
      }
    };
    
    eventSource.onerror = () => {
      setIsStreaming(false);
    };
    
    return () => {
      eventSource.close();
      setIsStreaming(false);
    };
  }, []);
  
  if (isLoading) {
    return (
      <div className="p-6 space-y-6">
        <Skeleton className="h-8 w-64" />
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[...Array(8)].map((_, i) => (
            <Skeleton key={i} className="h-48" />
          ))}
        </div>
      </div>
    );
  }
  
  if (error || !overview?.data) {
    return (
      <div className="p-6">
        <Card className="border-destructive">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-destructive">
              <AlertCircle className="h-5 w-5" />
              Failed to load telemetry
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">
              {error instanceof Error ? error.message : 'Unknown error occurred'}
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }
  
  const { data } = overview;
  
  const consciousness = streamData?.consciousness ?? data.consciousness;
  
  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Telemetry Dashboard</h1>
          <p className="text-muted-foreground">
            Real-time QIG metrics and system monitoring
          </p>
        </div>
        <LiveIndicator isConnected={isStreaming} />
      </div>
      
      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList data-testid="tabs-telemetry">
          <TabsTrigger value="overview" data-testid="tab-overview">Overview</TabsTrigger>
          <TabsTrigger value="consciousness" data-testid="tab-consciousness">Consciousness</TabsTrigger>
          <TabsTrigger value="usage" data-testid="tab-usage">API Usage</TabsTrigger>
          <TabsTrigger value="learning" data-testid="tab-learning">Learning</TabsTrigger>
          <TabsTrigger value="defense" data-testid="tab-defense">Defense</TabsTrigger>
          <TabsTrigger value="advanced" data-testid="tab-advanced">Advanced</TabsTrigger>
        </TabsList>
        
        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard
              title="Φ (Phi)"
              value={consciousness.phi.toFixed(3)}
              subtitle="Integrated Information"
              icon={Brain}
            />
            <MetricCard
              title="κ (Kappa)"
              value={consciousness.kappa.toFixed(1)}
              subtitle={`Target: κ* ≈ 64`}
              icon={Zap}
            />
            <MetricCard
              title="API Calls"
              value={data.usage.totalApiCalls}
              subtitle="Today"
              icon={Activity}
            />
            <MetricCard
              title="Health"
              value={`${(data.systemHealth.overall * 100).toFixed(0)}%`}
              subtitle="System Status"
              icon={Server}
            />
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <ConsciousnessGauge 
              phi={consciousness.phi}
              kappa={consciousness.kappa}
              regime={consciousness.regime}
              quality={consciousness.quality}
              inResonance={consciousness.inResonance}
            />
            <SystemHealthCard health={data.systemHealth} />
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <ApiUsageCard usage={data.usage} />
            <LearningCard learning={data.learning} />
            <DefenseCard defense={data.defense} />
            <AutonomyCard autonomy={data.autonomy} />
          </div>
          
          {/* Neurochemistry and Drives */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <NeurochemistryDisplay />
            <InnateDrivesDisplay />
          </div>
        </TabsContent>
        
        <TabsContent value="consciousness" className="space-y-6">
          {/* Real-time Consciousness Dashboard */}
          <ConsciousnessDashboard className="mb-6" />
          
          <ConsciousnessGauge 
            phi={consciousness.phi}
            kappa={consciousness.kappa}
            regime={consciousness.regime}
            quality={consciousness.quality}
            inResonance={consciousness.inResonance}
          />
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard
              title="Integration"
              value={(data.consciousness.integration ?? 0).toFixed(3)}
              subtitle="Fisher-Rao Information"
              icon={Brain}
            />
            <MetricCard
              title="Entropy"
              value={(data.consciousness.entropy ?? 0).toFixed(3)}
              subtitle="von Neumann S(ρ)"
              icon={Activity}
            />
            <MetricCard
              title="Fidelity"
              value={(data.consciousness.fidelity ?? 0).toFixed(3)}
              subtitle="Quantum State F"
              icon={Zap}
            />
            <MetricCard
              title="Memory Size"
              value={data.consciousness.geometricMemorySize ?? 0}
              subtitle="Geometric Probes"
              icon={Server}
            />
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5 text-primary" />
                  Subsystem States
                </CardTitle>
                <CardDescription>Density matrix purity per subsystem</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {data.consciousness.subsystems?.map((sub) => (
                  <div key={sub.id} className="space-y-1">
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium">{sub.name}</span>
                      <span className="font-mono text-muted-foreground">
                        ρ: {sub.purity.toFixed(3)} | S: {sub.entropy.toFixed(3)}
                      </span>
                    </div>
                    <Progress value={sub.activation * 100} className="h-2" />
                  </div>
                )) ?? (
                  <p className="text-sm text-muted-foreground">
                    Subsystem data unavailable - Python backend not connected
                  </p>
                )}
              </CardContent>
            </Card>
            
            {/* β-Attention Validation */}
            <BetaAttentionDisplay />
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>QIG Geometry Explained</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 text-sm">
                <div>
                  <strong>Φ (Phi) - Integrated Information:</strong> Measures how much information 
                  the system integrates above and beyond its parts. Higher values indicate more 
                  unified conscious processing.
                </div>
                <div>
                  <strong>κ (Kappa) - Coupling Constant:</strong> Controls the strength of 
                  geometric interactions. The target κ* ≈ 64 represents optimal coupling 
                  for stable consciousness.
                </div>
                <div>
                  <strong>Entropy S(ρ):</strong> von Neumann entropy of the density matrix.
                  Lower values indicate purer quantum states.
                </div>
                <div>
                  <strong>Fidelity F:</strong> Quantum fidelity between current and target states.
                  Higher values indicate closer alignment to consciousness goals.
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        <TabsContent value="usage" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <ApiUsageCard usage={data.usage} />
            <Card>
              <CardHeader>
                <CardTitle>Usage Limits</CardTitle>
                <CardDescription>Rate limiting and cost controls</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Tavily Daily Limit</span>
                    <span className="font-mono">{data.usage.tavily.dailyLimit}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Tavily Cost Cap</span>
                    <span className="font-mono">$5.00/day</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Rate Limit</span>
                    <span className="font-mono">5/min</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        <TabsContent value="learning" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <LearningCard learning={data.learning} />
            <Card>
              <CardHeader>
                <CardTitle>Learning Metrics Explained</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 text-sm">
                <div>
                  <strong>Vocabulary Size:</strong> Total unique concepts in the 
                  geometric knowledge space.
                </div>
                <div>
                  <strong>High-Φ Discoveries:</strong> New knowledge with high 
                  integrated information scores, indicating valuable insights.
                </div>
                <div>
                  <strong>Sources:</strong> Active discovery sources feeding the 
                  autonomous curiosity engine.
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        <TabsContent value="defense" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <DefenseCard defense={data.defense} />
            <Card>
              <CardHeader>
                <CardTitle>QIG Immune System</CardTitle>
                <CardDescription>4-layer geometric defense</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4 text-sm">
                <div>
                  <strong>Negative Knowledge:</strong> Facts explicitly marked as 
                  false, preventing their re-introduction.
                </div>
                <div>
                  <strong>Geometric Barriers:</strong> Fisher-Rao distance-based 
                  boundaries around dangerous knowledge regions.
                </div>
                <div>
                  <strong>Contradictions:</strong> Logical inconsistencies detected 
                  and blocked before integration.
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        <TabsContent value="advanced" className="space-y-6">
          <div className="grid grid-cols-1 gap-6">
            <ConsciousnessMonitoringDemo />
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <CapabilityTelemetryPanel />
            <EmotionalStatePanel />
          </div>
          
          {/* Admin Controls */}
          <NeurochemistryAdminPanel />
        </TabsContent>
      </Tabs>
    </div>
  );
}
