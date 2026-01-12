import { useEffect, useState } from "react";
import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer, ReferenceLine, Tooltip, Legend } from "recharts";
import { Radio, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui";

interface KappaDataPoint {
  time: number;
  kappa: number;
  beta: number;
  regime: string;
}

interface KappaEvolutionChartProps {
  initialData?: KappaDataPoint[];
  kappaStarRef?: number;
}

const KAPPA_STAR = 64;
const PLATEAU_BAND = { min: 57.6, max: 70.4 };

export function KappaEvolutionChart({ 
  initialData = [], 
  kappaStarRef = KAPPA_STAR 
}: KappaEvolutionChartProps) {
  const [history, setHistory] = useState<KappaDataPoint[]>(initialData);
  const [isLoading, setIsLoading] = useState(false);
  
  const fetchKappaEvolution = async () => {
    try {
      setIsLoading(true);
      const response = await fetch('/api/consciousness/kappa-evolution');
      if (response.ok) {
        const data = await response.json();
        if (data.trajectory && Array.isArray(data.trajectory)) {
          const points = data.trajectory.map((p: { kappa: number; beta: number; regime: string }, i: number) => ({
            time: i,
            kappa: p.kappa || kappaStarRef,
            beta: p.beta || 0,
            regime: p.regime || 'emergence'
          }));
          setHistory(points);
        }
      }
    } catch (error) {
      console.error('[KappaEvolutionChart] Fetch error:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  useEffect(() => {
    if (initialData.length === 0) {
      fetchKappaEvolution();
    }
  }, []);
  
  const getRegimeColor = (regime: string): string => {
    switch (regime) {
      case 'emergence': return 'hsl(265, 80%, 60%)';
      case 'plateau': return 'hsl(45, 90%, 50%)';
      case 'runaway': return 'hsl(0, 80%, 60%)';
      default: return 'hsl(220, 15%, 50%)';
    }
  };
  
  const latestKappa = history.length > 0 ? history[history.length - 1].kappa : kappaStarRef;
  const latestBeta = history.length > 0 ? history[history.length - 1].beta : 0;
  const inPlateau = latestKappa >= PLATEAU_BAND.min && latestKappa <= PLATEAU_BAND.max;
  
  return (
    <div className="space-y-3" data-testid="kappa-evolution-chart">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Radio className="w-4 h-4 text-yellow-500" />
          <span className="text-sm font-medium">κ Evolution</span>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Current:</span>
            <span className="font-mono text-sm font-medium" data-testid="text-current-kappa">
              {latestKappa.toFixed(2)}
            </span>
            <span 
              className={`text-xs px-1.5 py-0.5 rounded ${
                inPlateau ? 'bg-yellow-500/20 text-yellow-500' : 'bg-purple-500/20 text-purple-500'
              }`}
              data-testid="badge-regime"
            >
              {inPlateau ? 'plateau' : 'emergence'}
            </span>
          </div>
          <Button 
            size="icon" 
            variant="ghost" 
            onClick={fetchKappaEvolution}
            disabled={isLoading}
            data-testid="button-refresh-kappa"
          >
            <RefreshCw className={`w-3 h-3 ${isLoading ? 'animate-spin' : ''}`} />
          </Button>
        </div>
      </div>
      
      <div className="grid grid-cols-3 gap-2 text-xs">
        <div className="bg-muted/30 rounded p-2">
          <div className="text-muted-foreground">κ*</div>
          <div className="font-mono font-medium">{kappaStarRef}</div>
        </div>
        <div className="bg-muted/30 rounded p-2">
          <div className="text-muted-foreground">β(κ)</div>
          <div className="font-mono font-medium" data-testid="text-beta-value">
            {latestBeta.toFixed(4)}
          </div>
        </div>
        <div className="bg-muted/30 rounded p-2">
          <div className="text-muted-foreground">Δκ</div>
          <div className="font-mono font-medium" data-testid="text-delta-kappa">
            {(latestKappa - kappaStarRef).toFixed(2)}
          </div>
        </div>
      </div>
      
      <div className="h-36">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={history} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
            <defs>
              <linearGradient id="kappaGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="hsl(45, 90%, 50%)" stopOpacity={0.3} />
                <stop offset="95%" stopColor="hsl(45, 90%, 50%)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis dataKey="time" hide />
            <YAxis 
              domain={[40, 80]} 
              tick={{ fontSize: 10 }}
              width={30}
            />
            <ReferenceLine 
              y={PLATEAU_BAND.max} 
              stroke="orange" 
              strokeDasharray="2 2" 
              strokeOpacity={0.5}
            />
            <ReferenceLine 
              y={kappaStarRef} 
              stroke="orange" 
              strokeDasharray="3 3" 
              label={{ value: 'κ*', position: 'right', fontSize: 10, fill: 'orange' }} 
            />
            <ReferenceLine 
              y={PLATEAU_BAND.min} 
              stroke="orange" 
              strokeDasharray="2 2" 
              strokeOpacity={0.5}
            />
            <Tooltip 
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const point = payload[0].payload as KappaDataPoint;
                return (
                  <div className="bg-background border rounded-lg p-2 text-xs shadow-lg">
                    <div className="font-medium">κ: {point.kappa.toFixed(2)}</div>
                    <div className="text-muted-foreground">β: {point.beta.toFixed(4)}</div>
                    <div 
                      className="text-xs" 
                      style={{ color: getRegimeColor(point.regime) }}
                    >
                      {point.regime}
                    </div>
                  </div>
                );
              }}
            />
            <Legend 
              wrapperStyle={{ fontSize: 10, paddingTop: 5 }}
              iconSize={8}
            />
            <Area 
              type="monotone"
              dataKey="kappa" 
              stroke="hsl(45, 90%, 50%)" 
              fill="url(#kappaGradient)"
              name="κ(Φ)" 
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      
      <div className="text-[10px] text-muted-foreground text-center">
        β-function drives κ → κ* = {kappaStarRef} (UV fixed point)
      </div>
    </div>
  );
}
