import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui";
import { AlertTriangle } from "lucide-react";
import type { SafetyManifest } from "./types";

interface SafetyBoundaryCardProps {
  safetyManifest: SafetyManifest;
}

export function SafetyBoundaryCard({ safetyManifest }: SafetyBoundaryCardProps) {
  return (
    <Card className="border-amber-500/20">
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-amber-400" />
          Safety Boundary Ranges
        </CardTitle>
        <CardDescription>
          Active thresholds for intervention eligibility (monitoring only)
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Min Φ (Intervention)</p>
            <p className="font-mono text-sm" data-testid="text-phi-min-intervention">
              &gt; {safetyManifest.phi_min_intervention}
            </p>
          </div>
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Min Φ (Mushroom Mod)</p>
            <p className="font-mono text-sm" data-testid="text-phi-min-mushroom">
              &gt; {safetyManifest.phi_min_mushroom_mod}
            </p>
          </div>
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Max Instability (Micro)</p>
            <p className="font-mono text-sm" data-testid="text-instability-max-mushroom">
              &lt; {(safetyManifest.instability_max_mushroom * 100).toFixed(0)}%
            </p>
          </div>
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Max Instability (Mod)</p>
            <p className="font-mono text-sm" data-testid="text-instability-max-mod">
              &lt; {(safetyManifest.instability_max_mushroom_mod * 100).toFixed(0)}%
            </p>
          </div>
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Max Coverage (Dream)</p>
            <p className="font-mono text-sm" data-testid="text-coverage-max-dream">
              &lt; {(safetyManifest.coverage_max_dream * 100).toFixed(0)}%
            </p>
          </div>
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Mushroom Cooldown</p>
            <p className="font-mono text-sm" data-testid="text-mushroom-cooldown">
              {Math.round(safetyManifest.mushroom_cooldown_seconds / 60)} min
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
