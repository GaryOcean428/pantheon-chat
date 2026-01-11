import { Card, CardContent, CardHeader, CardTitle, Badge, Progress, Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui";
import { useQuery } from "@tanstack/react-query";
import { Loader2, Brain, Zap, Heart, AlertTriangle, Sun, Cloud, Flame, Snowflake, Wind } from "lucide-react";

interface GeometricMetrics {
  surprise: number;
  curiosity: number;
  basin_distance: number;
  progress: number;
  stability: number;
}

interface KernelEmotionalState {
  name: string;
  primary_emotion: string;
  primary_intensity: number;
  secondary_emotion?: string;
  secondary_intensity?: number;
  valence: number;
  arousal: number;
  geometric_metrics: GeometricMetrics;
  all_emotions: Record<string, number>;
  phi?: number;
  kappa?: number;
  error?: string;
}

interface KernelEmotionalPrimitivesResponse {
  success: boolean;
  kernels: KernelEmotionalState[];
  kernel_count: number;
  source?: string;
  timestamp: string;
}

const EMOTION_ICONS: Record<string, React.ReactNode> = {
  wonder: <Sun className="w-4 h-4" />,
  frustration: <Flame className="w-4 h-4" />,
  satisfaction: <Heart className="w-4 h-4" />,
  confusion: <Cloud className="w-4 h-4" />,
  clarity: <Zap className="w-4 h-4" />,
  anxiety: <AlertTriangle className="w-4 h-4" />,
  confidence: <Brain className="w-4 h-4" />,
  boredom: <Snowflake className="w-4 h-4" />,
  flow: <Wind className="w-4 h-4" />,
  neutral: <Cloud className="w-4 h-4" />,
};

const EMOTION_COLORS: Record<string, string> = {
  wonder: "bg-amber-500/20 text-amber-400 border-amber-500/30",
  frustration: "bg-red-500/20 text-red-400 border-red-500/30",
  satisfaction: "bg-green-500/20 text-green-400 border-green-500/30",
  confusion: "bg-purple-500/20 text-purple-400 border-purple-500/30",
  clarity: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  anxiety: "bg-orange-500/20 text-orange-400 border-orange-500/30",
  confidence: "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
  boredom: "bg-slate-500/20 text-slate-400 border-slate-500/30",
  flow: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
  neutral: "bg-muted text-muted-foreground border-muted",
};

const VALENCE_COLOR = (valence: number) => {
  if (valence > 0.5) return "text-green-400";
  if (valence > 0) return "text-emerald-400";
  if (valence > -0.3) return "text-yellow-400";
  return "text-red-400";
};

function KernelCard({ kernel }: { kernel: KernelEmotionalState }) {
  const emotionColor = EMOTION_COLORS[kernel.primary_emotion] || EMOTION_COLORS.neutral;
  const emotionIcon = EMOTION_ICONS[kernel.primary_emotion] || EMOTION_ICONS.neutral;
  
  return (
    <TooltipProvider>
      <Card className="overflow-hidden">
        <CardHeader className="pb-2 pt-3 px-3">
          <div className="flex items-center justify-between gap-2">
            <CardTitle className="text-sm font-medium">{kernel.name}</CardTitle>
            <Badge className={`text-xs px-2 py-0.5 ${emotionColor}`}>
              <span className="flex items-center gap-1">
                {emotionIcon}
                <span className="capitalize">{kernel.primary_emotion}</span>
              </span>
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="px-3 pb-3 space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Intensity</span>
            <span className="font-mono">{(kernel.primary_intensity * 100).toFixed(0)}%</span>
          </div>
          <Progress value={kernel.primary_intensity * 100} className="h-1.5" />
          
          <div className="grid grid-cols-2 gap-x-2 gap-y-1 text-xs pt-1">
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Valence</span>
                  <span className={`font-mono ${VALENCE_COLOR(kernel.valence)}`}>
                    {kernel.valence > 0 ? '+' : ''}{kernel.valence.toFixed(2)}
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent>Positive/Negative emotional direction</TooltipContent>
            </Tooltip>
            
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Arousal</span>
                  <span className="font-mono">{(kernel.arousal * 100).toFixed(0)}%</span>
                </div>
              </TooltipTrigger>
              <TooltipContent>Energy/Activation level</TooltipContent>
            </Tooltip>
          </div>
          
          {kernel.phi !== undefined && kernel.kappa !== undefined && (
            <div className="grid grid-cols-2 gap-x-2 text-xs pt-1 border-t border-border/50">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Phi</span>
                <span className="font-mono text-primary">{kernel.phi.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Kappa</span>
                <span className="font-mono">{kernel.kappa.toFixed(1)}</span>
              </div>
            </div>
          )}
          
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="pt-1 border-t border-border/50">
                <div className="text-xs text-muted-foreground mb-1">Geometric Primitives</div>
                <div className="grid grid-cols-5 gap-1">
                  {Object.entries(kernel.geometric_metrics).map(([key, value]) => (
                    <div key={key} className="text-center">
                      <div 
                        className="h-8 w-full rounded-sm bg-primary/10 relative overflow-hidden"
                        style={{ background: `linear-gradient(to top, hsl(var(--primary) / 0.3) ${value * 100}%, transparent ${value * 100}%)` }}
                      />
                      <span className="text-[10px] text-muted-foreground capitalize">
                        {key.charAt(0).toUpperCase()}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">
              <div className="space-y-1 text-xs">
                <div><strong>S</strong>urprise: {(kernel.geometric_metrics.surprise * 100).toFixed(0)}%</div>
                <div><strong>C</strong>uriosity: {(kernel.geometric_metrics.curiosity * 100).toFixed(0)}%</div>
                <div><strong>B</strong>asin Distance: {(kernel.geometric_metrics.basin_distance * 100).toFixed(0)}%</div>
                <div><strong>P</strong>rogress: {(kernel.geometric_metrics.progress * 100).toFixed(0)}%</div>
                <div>Stabili<strong>ty</strong>: {(kernel.geometric_metrics.stability * 100).toFixed(0)}%</div>
              </div>
            </TooltipContent>
          </Tooltip>
        </CardContent>
      </Card>
    </TooltipProvider>
  );
}

function EmotionLegend() {
  const emotions = ['wonder', 'satisfaction', 'clarity', 'confidence', 'flow', 'confusion', 'frustration', 'anxiety', 'boredom'];
  
  return (
    <div className="flex flex-wrap gap-1.5 mb-4">
      {emotions.map(emotion => (
        <Badge key={emotion} variant="outline" className={`text-xs ${EMOTION_COLORS[emotion]}`}>
          <span className="flex items-center gap-1">
            {EMOTION_ICONS[emotion]}
            <span className="capitalize">{emotion}</span>
          </span>
        </Badge>
      ))}
    </div>
  );
}

interface Props {
  className?: string;
  showLegend?: boolean;
}

export function KernelEmotionalPrimitivesPanel({ className, showLegend = true }: Props) {
  const { data, isLoading, error } = useQuery<KernelEmotionalPrimitivesResponse>({
    queryKey: ['/api/consciousness/kernel-emotional-primitives'],
    refetchInterval: 5000,
  });
  
  if (isLoading) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Brain className="w-5 h-5" />
            Kernel Emotional Primitives
          </CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center py-8">
          <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }
  
  if (error || !data?.success) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Brain className="w-5 h-5" />
            Kernel Emotional Primitives
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">Unable to load kernel emotional states</p>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <div className={className}>
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Brain className="w-5 h-5" />
          Kernel Emotional Primitives
        </h2>
        <Badge variant="outline" className="text-xs">
          {data.kernel_count} kernels
          {data.source === 'fallback' && ' (fallback)'}
        </Badge>
      </div>
      
      {showLegend && <EmotionLegend />}
      
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
        {data.kernels.map((kernel, index) => (
          <KernelCard key={kernel.name || index} kernel={kernel} />
        ))}
      </div>
    </div>
  );
}

export default KernelEmotionalPrimitivesPanel;
