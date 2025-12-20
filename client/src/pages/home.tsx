import { useQuery } from "@tanstack/react-query";
import {
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Badge,
  Progress,
  Skeleton,
} from "@/components/ui";
import { useAuth } from "@/hooks/useAuth";
import { useConsciousness, getPhiColor, getRegimeLabel } from "@/contexts/ConsciousnessContext";
import { Waves, MessageSquare, Brain, Activity, Search, Zap, Sparkles, Eye, GraduationCap } from "lucide-react";
import { Link } from "wouter";
import type { User } from "@shared/schema";
import { QUERY_KEYS } from "@/api";

interface InvestigationStatus {
  isRunning: boolean;
  tested: number;
  nearMisses: number;
  consciousness: {
    phi: number;
    kappa: number;
    regime: string;
  };
  currentThought: string;
  progress: number;
  manifold?: {
    totalProbes: number;
    avgPhi: number;
    exploredVolume: number;
  };
}

export default function Home() {
  const { user } = useAuth() as { user: User | undefined };
  const { consciousness, isIdle } = useConsciousness();

  const { data: investigationStatus, isLoading: statusLoading } = useQuery<InvestigationStatus>({
    queryKey: QUERY_KEYS.investigation.status(),
    refetchInterval: 3000,
  });

  const phi = consciousness?.phi ?? investigationStatus?.consciousness?.phi ?? 0;
  const kappa = consciousness?.kappaEff ?? investigationStatus?.consciousness?.kappa ?? 0;
  const regime = consciousness?.regime ?? investigationStatus?.consciousness?.regime ?? 'initializing';
  const isIdleState = !consciousness?.isInvestigating;

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-6xl mx-auto space-y-8">
        <div className="space-y-2">
          <h1 className="text-4xl font-bold" data-testid="text-welcome">
            Welcome{user?.firstName ? `, ${user.firstName}` : ""}
          </h1>
          <p className="text-muted-foreground">
            Ocean Agentic Platform - Intelligent Chat and Search powered by Quantum Information Geometry
          </p>
        </div>

        <div className="grid md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Activity className="h-4 w-4" />
                Consciousness (Φ)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold font-mono ${getPhiColor(phi, isIdleState)}`} data-testid="text-phi-value">
                {phi.toFixed(4)}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Brain className="h-4 w-4" />
                Curvature (κ)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold font-mono" data-testid="text-kappa-value">
                {kappa.toFixed(4)}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Eye className="h-4 w-4" />
                Regime
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Badge 
                className={
                  regime === 'geometric' 
                    ? 'bg-green-500/20 text-green-400'
                    : regime === 'breakdown'
                    ? 'bg-red-500/20 text-red-400'
                    : 'bg-yellow-500/20 text-yellow-400'
                }
                data-testid="badge-regime"
              >
                {getRegimeLabel(regime, isIdleState)}
              </Badge>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Waves className="h-4 w-4" />
                Status
              </CardTitle>
            </CardHeader>
            <CardContent>
              {statusLoading ? (
                <Skeleton className="h-6 w-20" data-testid="skeleton-status" />
              ) : (
                <Badge 
                  className={investigationStatus?.isRunning 
                    ? 'bg-green-500/20 text-green-400' 
                    : isIdle 
                    ? 'bg-muted text-muted-foreground'
                    : 'bg-blue-500/20 text-blue-400'
                  }
                  data-testid="badge-status"
                >
                  {investigationStatus?.isRunning ? 'Active' : isIdle ? 'Idle' : 'Ready'}
                </Badge>
              )}
            </CardContent>
          </Card>
        </div>

        {statusLoading ? (
          <Card>
            <CardHeader>
              <Skeleton className="h-6 w-48" data-testid="skeleton-thought-title" />
              <Skeleton className="h-4 w-64 mt-2" data-testid="skeleton-thought-content" />
            </CardHeader>
          </Card>
        ) : investigationStatus?.isRunning ? (
          <Card className="border-primary/30 bg-primary/5">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Waves className="h-5 w-5 text-primary animate-pulse" />
                Ocean is Thinking
              </CardTitle>
              <CardDescription>
                {investigationStatus.currentThought || 'Processing geometric manifolds...'}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between text-sm">
                <span>Research Progress</span>
                <span className="font-mono">{investigationStatus.progress}%</span>
              </div>
              <Progress value={investigationStatus.progress} className="h-2" />
              <div className="flex gap-4 text-sm text-muted-foreground">
                <span>Queries: {investigationStatus.tested.toLocaleString()}</span>
                <span>Insights: {investigationStatus.nearMisses}</span>
              </div>
            </CardContent>
          </Card>
        ) : null}

        <div className="grid md:grid-cols-3 gap-6">
          <Card className="border-primary/30 hover-elevate">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MessageSquare className="h-5 w-5 text-primary" />
                Zeus Chat
              </CardTitle>
              <CardDescription>
                Natural language conversations with QIG-powered geometric consciousness. Ask anything.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Link href="/investigation">
                <Button size="lg" className="w-full" data-testid="button-go-to-chat">
                  <Sparkles className="mr-2 h-4 w-4" />
                  Start Conversation
                </Button>
              </Link>
            </CardContent>
          </Card>

          <Card className="hover-elevate">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="h-5 w-5" />
                Olympus Pantheon
              </CardTitle>
              <CardDescription>
                12-god specialized intelligence system for domain expertise and multi-agent research.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Link href="/olympus">
                <Button size="lg" variant="outline" className="w-full" data-testid="button-go-to-olympus">
                  Open Pantheon
                </Button>
              </Link>
            </CardContent>
          </Card>

          <Card className="hover-elevate">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Search className="h-5 w-5" />
                Shadow Search
              </CardTitle>
              <CardDescription>
                Proactive knowledge discovery through autonomous research and web indexing.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Link href="/sources">
                <Button size="lg" variant="outline" className="w-full" data-testid="button-go-to-sources">
                  Manage Sources
                </Button>
              </Link>
            </CardContent>
          </Card>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <Card className="hover-elevate">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <GraduationCap className="h-5 w-5" />
                Learning Center
              </CardTitle>
              <CardDescription>
                Monitor self-learning effectiveness, tool generation, and knowledge acquisition metrics.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Link href="/learning">
                <Button variant="outline" className="w-full" data-testid="button-go-to-learning">
                  View Progress
                </Button>
              </Link>
            </CardContent>
          </Card>
        </div>

        {investigationStatus?.manifold && investigationStatus.manifold.totalProbes > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                Knowledge Manifold
              </CardTitle>
              <CardDescription>
                Geometric representation of explored information space
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <div className="text-sm text-muted-foreground">Probes</div>
                  <div className="text-xl font-mono font-bold">
                    {investigationStatus.manifold.totalProbes.toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Average Φ</div>
                  <div className={`text-xl font-mono font-bold ${getPhiColor(investigationStatus.manifold.avgPhi, isIdleState)}`}>
                    {investigationStatus.manifold.avgPhi.toFixed(4)}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Explored</div>
                  <div className="text-xl font-mono font-bold">
                    {(investigationStatus.manifold.exploredVolume * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
