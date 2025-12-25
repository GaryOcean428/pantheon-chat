import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, ReferenceLine, ReferenceArea, Tooltip, Legend } from "recharts";
import type { TrajectoryPoint } from "./types";

interface PhiKappaTrajectoryChartProps {
  history: TrajectoryPoint[];
}

export function PhiKappaTrajectoryChart({ history }: PhiKappaTrajectoryChartProps) {
  return (
    <div className="h-40 mt-4">
      <div className="text-sm font-medium mb-2">Φ/κ Trajectory</div>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={history} margin={{ top: 5, right: 30, bottom: 5, left: 0 }}>
          <XAxis dataKey="time" hide />
          <YAxis 
            yAxisId="left"
            domain={[0, 1]} 
            orientation="left"
            tick={{ fontSize: 10 }}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
          />
          <YAxis 
            yAxisId="right"
            domain={[0, 100]} 
            orientation="right"
            tick={{ fontSize: 10 }}
          />
          <ReferenceArea yAxisId="right" y1={57.6} y2={70.4} fill="orange" fillOpacity={0.1} />
          <ReferenceLine yAxisId="right" y={64} stroke="orange" strokeDasharray="3 3" label={{ value: 'κ*', position: 'right', fontSize: 10 }} />
          <Tooltip 
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              return (
                <div className="bg-background border rounded-lg p-2 text-xs shadow-lg">
                  <div>Φ: {((payload[0]?.value as number) * 100).toFixed(1)}%</div>
                  <div>κ: {(payload[1]?.value as number)?.toFixed(1)}</div>
                </div>
              );
            }}
          />
          <Legend 
            wrapperStyle={{ fontSize: 10, paddingTop: 10 }}
            iconSize={10}
          />
          <Line 
            yAxisId="left"
            dataKey="phi" 
            stroke="hsl(265, 80%, 60%)" 
            name="Φ" 
            dot={false}
            strokeWidth={2}
          />
          <Line 
            yAxisId="right"
            dataKey="kappa" 
            stroke="hsl(45, 90%, 50%)" 
            name="κ" 
            dot={false}
            strokeWidth={2}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
