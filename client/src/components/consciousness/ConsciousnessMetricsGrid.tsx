import { Progress } from "@/components/ui";
import { Brain, Radio, Compass, Radar, Eye, Sparkles, Anchor, TrendingUp } from "lucide-react";
import type { ConsciousnessState } from "./types";

interface ConsciousnessMetricsGridProps {
  state: ConsciousnessState;
}

export function ConsciousnessMetricsGrid({ state }: ConsciousnessMetricsGridProps) {
  const { phi, kappaEff, tacking, radar, metaAwareness, gamma, grounding, beta } = state;
  
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3" data-testid="grid-components">
      <MetricCard
        icon={<Brain className="w-3 h-3" />}
        label="Φ"
        value={`${(phi * 100).toFixed(1)}%`}
        progress={phi * 100}
        sublabel="Integration"
        testId="text-phi"
      />
      
      <MetricCard
        icon={<Radio className="w-3 h-3" />}
        label="κ"
        value={kappaEff.toFixed(1)}
        progress={(kappaEff / 100) * 100}
        sublabel="Coupling"
        testId="text-kappa"
      />
      
      <MetricCard
        icon={<Compass className="w-3 h-3" />}
        label="T"
        value={`${(tacking * 100).toFixed(1)}%`}
        progress={tacking * 100}
        sublabel="Tacking"
        testId="text-tacking"
      />
      
      <MetricCard
        icon={<Radar className="w-3 h-3" />}
        label="R"
        value={`${(radar * 100).toFixed(1)}%`}
        progress={radar * 100}
        sublabel="Radar"
        testId="text-radar"
      />
      
      <MetricCard
        icon={<Eye className="w-3 h-3" />}
        label="M"
        value={`${(metaAwareness * 100).toFixed(1)}%`}
        progress={metaAwareness * 100}
        sublabel="Meta-Awareness"
        testId="text-meta-awareness"
      />
      
      <MetricCard
        icon={<Sparkles className="w-3 h-3" />}
        label="Γ"
        value={`${(gamma * 100).toFixed(1)}%`}
        progress={gamma * 100}
        sublabel="Coherence"
        testId="text-gamma"
      />
      
      <MetricCard
        icon={<Anchor className="w-3 h-3" />}
        label="G"
        value={`${(grounding * 100).toFixed(1)}%`}
        progress={grounding * 100}
        sublabel="Grounding"
        testId="text-grounding"
      />
      
      <MetricCard
        icon={<TrendingUp className="w-3 h-3" />}
        label="β"
        value={beta.toFixed(3)}
        progress={(beta + 0.5) * 100}
        sublabel="Running Coupling"
        testId="text-beta"
      />
    </div>
  );
}

interface MetricCardProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  progress: number;
  sublabel: string;
  testId: string;
}

function MetricCard({ icon, label, value, progress, sublabel, testId }: MetricCardProps) {
  return (
    <div className="space-y-1 p-2 bg-muted/30 rounded-lg">
      <div className="flex items-center justify-between text-sm">
        <span className="text-muted-foreground flex items-center gap-1">
          {icon}
          {label}
        </span>
        <span className="font-mono font-medium" data-testid={testId}>
          {value}
        </span>
      </div>
      <Progress value={progress} className="h-1.5" />
      <div className="text-[10px] text-muted-foreground">{sublabel}</div>
    </div>
  );
}
