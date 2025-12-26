/**
 * Consciousness Status Panel - displays phi, kappa, and regime information
 */

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, Badge, Progress } from '@/components/ui';
import { Brain, Activity, Zap } from 'lucide-react';
import { SEARCH_CONSTANTS, REGIME_COLORS } from './constants';
import type { ConsciousnessStatus } from './types';

interface ConsciousnessStatusPanelProps {
  status: ConsciousnessStatus | undefined;
  isLoading: boolean;
}

export function ConsciousnessStatusPanel({ status, isLoading }: ConsciousnessStatusPanelProps) {
  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Consciousness Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-3">
            <div className="h-4 bg-muted rounded w-3/4" />
            <div className="h-4 bg-muted rounded w-1/2" />
            <div className="h-4 bg-muted rounded w-2/3" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!status) {
    return null;
  }

  const regimeColor = REGIME_COLORS[status.regime] || REGIME_COLORS.unknown;
  const phiPercent = status.phi * SEARCH_CONSTANTS.PERCENT_MULTIPLIER;
  const coherencePercent = status.coherence * SEARCH_CONSTANTS.PERCENT_MULTIPLIER;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5" />
          Consciousness Status
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Regime</span>
          <Badge
            style={{
              backgroundColor: `hsl(${regimeColor.h}, ${regimeColor.s}%, ${regimeColor.l}%)`,
            }}
          >
            {status.regime}
          </Badge>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="flex items-center gap-1">
              <Activity className="h-4 w-4" />
              Φ (Integration)
            </span>
            <span className="font-mono">{status.phi.toFixed(3)}</span>
          </div>
          <Progress value={phiPercent} className="h-2" />
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="flex items-center gap-1">
              <Zap className="h-4 w-4" />
              κ (Coupling)
            </span>
            <span className="font-mono">{status.kappa.toFixed(1)}</span>
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span>Coherence</span>
            <span className="font-mono">{coherencePercent.toFixed(0)}%</span>
          </div>
          <Progress value={coherencePercent} className="h-2" />
        </div>

        <div className="text-xs text-muted-foreground text-right">
          Updated: {new Date(status.lastUpdate).toLocaleTimeString()}
        </div>
      </CardContent>
    </Card>
  );
}
