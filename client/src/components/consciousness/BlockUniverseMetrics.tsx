import { Progress } from "@/components/ui";
import { Brain, Activity, Sparkles } from "lucide-react";
import type { ConsciousnessState } from "./types";

interface BlockUniverseMetricsProps {
  state: ConsciousnessState;
}

export function BlockUniverseMetrics({ state }: BlockUniverseMetricsProps) {
  const { phi_spatial, phi_temporal, phi_4D } = state;
  
  if (phi_temporal === undefined && phi_4D === undefined) {
    return null;
  }
  
  const in4DMode = (phi_4D ?? 0) >= 0.85 && (phi_temporal ?? 0) > 0.7;
  
  return (
    <div className="border-t pt-3">
      <div className="text-sm font-medium mb-2 flex items-center gap-2">
        <Sparkles className="w-4 h-4 text-cyan-400" />
        4D Block Universe Consciousness
      </div>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        {phi_spatial !== undefined && (
          <div className="space-y-1 p-2 bg-muted/30 rounded-lg">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground flex items-center gap-1">
                <Brain className="w-3 h-3" />
                Φ<sub className="text-[8px]">spatial</sub>
              </span>
              <span className="font-mono font-medium" data-testid="text-phi-spatial">
                {(phi_spatial * 100).toFixed(1)}%
              </span>
            </div>
            <Progress value={phi_spatial * 100} className="h-1.5" />
            <div className="text-[10px] text-muted-foreground">Spatial Integration</div>
          </div>
        )}
        
        {phi_temporal !== undefined && (
          <div className="space-y-1 p-2 bg-muted/30 rounded-lg border-cyan-500/30 border">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground flex items-center gap-1">
                <Activity className="w-3 h-3 text-cyan-400" />
                Φ<sub className="text-[8px]">temporal</sub>
              </span>
              <span className="font-mono font-medium text-cyan-400" data-testid="text-phi-temporal">
                {(phi_temporal * 100).toFixed(1)}%
              </span>
            </div>
            <Progress value={phi_temporal * 100} className="h-1.5" />
            <div className="text-[10px] text-muted-foreground">Trajectory Coherence</div>
          </div>
        )}
        
        {phi_4D !== undefined && (
          <div className="space-y-1 p-2 bg-muted/30 rounded-lg border-purple-500/30 border">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground flex items-center gap-1">
                <Sparkles className="w-3 h-3 text-purple-400" />
                Φ<sub className="text-[8px]">4D</sub>
              </span>
              <span className="font-mono font-medium text-purple-400" data-testid="text-phi-4d">
                {(phi_4D * 100).toFixed(1)}%
              </span>
            </div>
            <Progress value={phi_4D * 100} className="h-1.5" />
            <div className="text-[10px] text-muted-foreground">Spacetime Integration</div>
          </div>
        )}
      </div>
      
      {in4DMode && (
        <div className="mt-2 p-2 bg-gradient-to-r from-cyan-500/10 to-purple-500/10 rounded-lg border border-cyan-500/30">
          <div className="flex items-center gap-2 text-sm">
            <Sparkles className="w-4 h-4 text-cyan-400 animate-pulse" />
            <span className="font-medium text-cyan-400">4D BLOCK UNIVERSE MODE ACTIVE</span>
          </div>
          <div className="text-xs text-muted-foreground mt-1">
            Ocean is navigating spacetime - temporal patterns recognized across search trajectory
          </div>
        </div>
      )}
    </div>
  );
}
