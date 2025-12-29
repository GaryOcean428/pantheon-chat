/**
 * Types for Capability Telemetry Panel
 */

export interface CapabilityMetrics {
  invocations: number;
  successes: number;
  failures: number;
  success_rate: number;
  avg_duration_ms: number;
  last_invoked: string | null;
}

export interface Capability {
  name: string;
  category: string;
  description: string;
  enabled: boolean;
  level: number;
  metrics: CapabilityMetrics;
}

export interface KernelSummary {
  kernel_id: string;
  kernel_name: string;
  total_capabilities: number;
  enabled: number;
  total_invocations: number;
  success_rate: number;
  strongest: string | null;
  weakest: string | null;
}

export interface FleetTelemetry {
  kernels: number;
  total_capabilities: number;
  total_invocations: number;
  fleet_success_rate: number;
  category_distribution: Record<string, number>;
  kernel_summaries: KernelSummary[];
}
