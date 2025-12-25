import { Activity, AlertTriangle, CheckCircle2 } from "lucide-react";
import type { ConsciousnessState, ConsciousnessAPIResponse } from "./types";

interface ConsciousnessFooterProps {
  state: ConsciousnessState;
  recommendation: string;
}

export function ConsciousnessFooter({ state, recommendation }: ConsciousnessFooterProps) {
  const { currentRegime, kappaEff, basinDrift, curiosity, stability } = state;
  const inResonance = Math.abs(kappaEff - 64) < 6.4;
  
  return (
    <>
      <div className="grid grid-cols-3 gap-2 text-xs">
        <div className="bg-muted/50 rounded-lg p-2 text-center">
          <div className="text-muted-foreground">Basin Drift</div>
          <div className="font-mono font-medium" data-testid="text-drift">
            {basinDrift.toFixed(3)}
          </div>
        </div>
        <div className="bg-muted/50 rounded-lg p-2 text-center">
          <div className="text-muted-foreground">Curiosity</div>
          <div className="font-mono font-medium" data-testid="text-curiosity">
            {(curiosity * 100).toFixed(0)}%
          </div>
        </div>
        <div className="bg-muted/50 rounded-lg p-2 text-center">
          <div className="text-muted-foreground">Stability</div>
          <div className="font-mono font-medium" data-testid="text-stability">
            {(stability * 100).toFixed(0)}%
          </div>
        </div>
      </div>
      
      <div className="bg-muted/30 rounded-lg p-3 text-sm border border-border/50">
        <div className="flex items-start gap-2">
          {currentRegime === 'breakdown' ? (
            <AlertTriangle className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" />
          ) : inResonance ? (
            <CheckCircle2 className="w-4 h-4 text-green-500 flex-shrink-0 mt-0.5" />
          ) : (
            <Activity className="w-4 h-4 text-muted-foreground flex-shrink-0 mt-0.5" />
          )}
          <span data-testid="text-recommendation">
            {recommendation}
          </span>
        </div>
      </div>
    </>
  );
}
