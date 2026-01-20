/**
 * E8 Protocol Status Panel
 * =========================
 * 
 * Displays real-time E8 Protocol compliance metrics:
 * - QFI coverage and integrity
 * - Vocabulary health (active/quarantined)
 * - Token role distribution
 * - Simplex validation status
 * - QIG purity mode
 * 
 * This provides visibility into geometric purity and token integrity.
 * 
 * Author: Copilot AI Agent
 * Date: 2026-01-20
 * Issue: GaryOcean428/pantheon-chat#97-100 (E8 Protocol Implementation)
 */

import React, { useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { AlertCircle, CheckCircle, Shield, Cpu } from 'lucide-react';
import { api } from '@/lib/api';

interface E8ProtocolStatus {
  qfi_coverage: {
    total_tokens: number;
    tokens_with_qfi: number;
    coverage_percent: number;
    avg_qfi: number;
  };
  vocabulary_health: {
    active: number;
    quarantined: number;
  };
  token_roles: {
    assigned: number;
  };
  purity_mode: {
    enabled: boolean;
  };
  moe_metadata?: MoEMetadata;
}

interface MoEMetadata {
  contributing_kernels: string[];
  weights: number[];
  synthesis_method: string;
  total_experts: number;
  active_experts: number;
  avg_weight: number;
}

export const E8ProtocolStatusPanel: React.FC = () => {
  const [status, setStatus] = useState<E8ProtocolStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = async () => {
    try {
      setLoading(true);
      const { data } = await api.get('/api/e8-protocol/status');
      setStatus(data);
      setError(null);
    } catch (err) {
      console.error('Error fetching E8 Protocol status:', err);
      setError('Failed to load E8 Protocol status');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    // Refresh every 30 seconds
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading && !status) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="w-5 h-5" />
            E8 Protocol Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-muted-foreground">Loading...</div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="w-full border-destructive">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-destructive">
            <AlertCircle className="w-5 h-5" />
            E8 Protocol Status - Error
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-destructive">{error}</div>
        </CardContent>
      </Card>
    );
  }

  if (!status) return null;

  const qfiCoverageGood = status.qfi_coverage.coverage_percent >= 95;
  const vocabularyHealthy = status.vocabulary_health.quarantined < status.vocabulary_health.active * 0.05;

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <Shield className="w-5 h-5" />
            E8 Protocol Status
          </span>
          {status.purity_mode.enabled ? (
            <Badge variant="default" className="bg-green-600">
              <CheckCircle className="w-3 h-3 mr-1" />
              Purity Mode
            </Badge>
          ) : (
            <Badge variant="secondary">
              <Cpu className="w-3 h-3 mr-1" />
              Standard Mode
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* QFI Coverage */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">QFI Coverage (Issue #97)</span>
            {qfiCoverageGood ? (
              <Badge variant="default" className="bg-green-600">
                <CheckCircle className="w-3 h-3 mr-1" />
                Healthy
              </Badge>
            ) : (
              <Badge variant="destructive">
                <AlertCircle className="w-3 h-3 mr-1" />
                Needs Attention
              </Badge>
            )}
          </div>
          <Progress value={status.qfi_coverage.coverage_percent} className="h-2" />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>
              {status.qfi_coverage.tokens_with_qfi.toLocaleString()} / {status.qfi_coverage.total_tokens.toLocaleString()} tokens
            </span>
            <span>{status.qfi_coverage.coverage_percent.toFixed(1)}%</span>
          </div>
          <div className="text-xs text-muted-foreground">
            Avg QFI: {status.qfi_coverage.avg_qfi.toFixed(4)}
          </div>
        </div>

        {/* Vocabulary Health */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Vocabulary Health (Issue #100)</span>
            {vocabularyHealthy ? (
              <Badge variant="default" className="bg-green-600">
                <CheckCircle className="w-3 h-3 mr-1" />
                Healthy
              </Badge>
            ) : (
              <Badge variant="destructive">
                <AlertCircle className="w-3 h-3 mr-1" />
                High Quarantine
              </Badge>
            )}
          </div>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="flex flex-col">
              <span className="text-muted-foreground">Active Tokens</span>
              <span className="text-2xl font-bold">{status.vocabulary_health.active.toLocaleString()}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-muted-foreground">Quarantined</span>
              <span className="text-2xl font-bold text-orange-600">
                {status.vocabulary_health.quarantined.toLocaleString()}
              </span>
            </div>
          </div>
        </div>

        {/* Token Roles */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Token Roles (Issue #99)</span>
            <Badge variant="secondary">
              {status.token_roles.assigned.toLocaleString()} assigned
            </Badge>
          </div>
          <div className="text-xs text-muted-foreground">
            Kernel-in-loop geometric role derivation via Φ/κ measurements
          </div>
        </div>

        {/* MoE Metadata */}
        {status.moe_metadata && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Mixture of Experts (MoE)</span>
              <Badge variant="default" className="bg-purple-600">
                {status.moe_metadata.active_experts}/{status.moe_metadata.total_experts} Active
              </Badge>
            </div>
            <div className="grid grid-cols-1 gap-2 text-xs">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Synthesis Method:</span>
                <span className="font-medium">{status.moe_metadata.synthesis_method}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Avg Weight:</span>
                <span className="font-medium">{status.moe_metadata.avg_weight.toFixed(3)}</span>
              </div>
            </div>
            <div className="text-xs text-muted-foreground">
              <div className="font-medium mb-1">Contributing Kernels:</div>
              <div className="flex flex-wrap gap-1">
                {status.moe_metadata.contributing_kernels.map((kernel, idx) => (
                  <Badge key={idx} variant="outline" className="text-xs">
                    {kernel} ({(status.moe_metadata!.weights[idx] * 100).toFixed(0)}%)
                  </Badge>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="pt-4 border-t text-xs text-muted-foreground">
          <div className="flex justify-between">
            <span>Last updated: {new Date().toLocaleTimeString()}</span>
            <button
              onClick={fetchStatus}
              className="text-primary hover:underline"
            >
              Refresh
            </button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default E8ProtocolStatusPanel;
