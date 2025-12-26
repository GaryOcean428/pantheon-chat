/**
 * Resonance Chart - visualizes resonance and phi over time
 */

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { TrendingUp } from 'lucide-react';
import type { ResonanceDataPoint } from './types';

interface ResonanceChartProps {
  data: ResonanceDataPoint[];
}

export function ResonanceChart({ data }: ResonanceChartProps) {
  if (data.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Resonance History
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-48 flex items-center justify-center text-muted-foreground">
            No resonance data available
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5" />
          Resonance History
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis
                dataKey="time"
                tick={{ fontSize: 10 }}
                className="text-muted-foreground"
              />
              <YAxis
                domain={[0, 1]}
                tick={{ fontSize: 10 }}
                className="text-muted-foreground"
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--card))',
                  border: '1px solid hsl(var(--border))',
                }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="resonance"
                stroke="hsl(var(--primary))"
                strokeWidth={2}
                dot={false}
                name="Resonance"
              />
              <Line
                type="monotone"
                dataKey="phi"
                stroke="hsl(280, 80%, 60%)"
                strokeWidth={2}
                dot={false}
                name="Φ"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
