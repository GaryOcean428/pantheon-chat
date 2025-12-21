/**
 * Self-Healing System Types
 * 
 * TypeScript definitions for the self-healing architecture.
 */

import { z } from "zod";

/**
 * Geometric Snapshot
 */
export const GeometricSnapshotSchema = z.object({
  timestamp: z.string(),
  phi: z.number(),
  kappa_eff: z.number(),
  basin_coords: z.array(z.number()).length(64),
  confidence: z.number(),
  surprise: z.number(),
  agency: z.number(),
  regime: z.enum(["linear", "geometric", "hierarchical", "breakdown", "4d_block_universe"]),
  code_hash: z.string(),
  active_modules: z.array(z.string()),
  module_versions: z.record(z.string()),
  error_rate: z.number(),
  avg_latency_ms: z.number(),
  memory_usage_mb: z.number(),
  cpu_usage_pct: z.number(),
  label: z.string().optional(),
  context: z.record(z.any()).optional(),
});

export type GeometricSnapshot = z.infer<typeof GeometricSnapshotSchema>;

/**
 * Health Degradation Report
 */
export const HealthDegradationSchema = z.object({
  degraded: z.boolean(),
  issues: z.array(z.string()),
  severity: z.enum(["normal", "warning", "critical"]),
  metrics: z.object({
    basin_distance: z.number().nullable(),
    phi_current: z.number(),
    phi_avg: z.number(),
    phi_trend: z.number(),
    kappa_current: z.number(),
    kappa_deviation: z.number(),
    regime: z.string(),
    breakdown_frequency: z.number(),
    error_rate: z.number(),
    latency_ms: z.number(),
    memory_mb: z.number(),
    memory_trend: z.number(),
  }),
  timestamp: z.string(),
  message: z.string().optional(),
});

export type HealthDegradation = z.infer<typeof HealthDegradationSchema>;

/**
 * Code Fitness Evaluation Result
 */
export const CodeFitnessSchema = z.object({
  fitness_score: z.number(),
  phi_impact: z.number(),
  basin_impact: z.number(),
  regime_stable: z.boolean(),
  performance_impact: z.object({
    latency_ratio: z.number(),
    memory_change_mb: z.number(),
  }),
  recommendation: z.enum(["apply", "reject", "test_more"]),
  reason: z.string(),
  detailed_metrics: z.record(z.any()).optional(),
  components: z.object({
    phi: z.number(),
    basin: z.number(),
    regime: z.number(),
    performance: z.number(),
  }).optional(),
});

export type CodeFitness = z.infer<typeof CodeFitnessSchema>;

/**
 * Healing Attempt Record
 */
export const HealingAttemptSchema = z.object({
  timestamp: z.string(),
  health: HealthDegradationSchema,
  result: z.object({
    healed: z.boolean(),
    strategy: z.string().optional(),
    patch: z.string().optional(),
    fitness_improvement: z.number().optional(),
    applied: z.boolean().optional(),
    reason: z.string().optional(),
  }),
});

export type HealingAttempt = z.infer<typeof HealingAttemptSchema>;

/**
 * Self-Healing Status
 */
export const SelfHealingStatusSchema = z.object({
  monitor: z.object({
    snapshots_collected: z.number(),
    baseline_set: z.boolean(),
  }),
  evaluator: z.object({
    weights: z.object({
      phi_change: z.number(),
      basin_drift: z.number(),
      regime_stability: z.number(),
      performance: z.number(),
    }),
    thresholds: z.object({
      apply: z.number(),
      test_more: z.number(),
    }),
  }),
  engine: z.object({
    running: z.boolean(),
    auto_apply_enabled: z.boolean(),
    check_interval_sec: z.number(),
    healing_attempts: z.number(),
    strategies_available: z.number(),
  }),
});

export type SelfHealingStatus = z.infer<typeof SelfHealingStatusSchema>;

/**
 * Health Summary
 */
export const HealthSummarySchema = z.object({
  status: z.string(),
  snapshots_collected: z.number(),
  current_phi: z.number().optional(),
  current_kappa: z.number().optional(),
  current_regime: z.string().optional(),
  basin_drift: z.number().nullable().optional(),
  issues: z.array(z.string()).optional(),
  last_snapshot: z.string().optional(),
});

export type HealthSummary = z.infer<typeof HealthSummarySchema>;

/**
 * API Request/Response types
 */

// POST /self-healing/snapshot
export const SnapshotRequestSchema = z.object({
  phi: z.number(),
  kappa_eff: z.number(),
  basin_coords: z.array(z.number()).length(64),
  confidence: z.number(),
  surprise: z.number(),
  agency: z.number(),
  error_rate: z.number().optional(),
  avg_latency: z.number().optional(),
  label: z.string().optional(),
  context: z.record(z.any()).optional(),
  set_baseline: z.boolean().optional(),
});

export type SnapshotRequest = z.infer<typeof SnapshotRequestSchema>;

// POST /self-healing/evaluate-patch
export const EvaluatePatchRequestSchema = z.object({
  module_name: z.string(),
  new_code: z.string(),
  test_workload: z.string().optional(),
});

export type EvaluatePatchRequest = z.infer<typeof EvaluatePatchRequestSchema>;

// POST /self-healing/start
export const StartHealingRequestSchema = z.object({
  auto_apply: z.boolean().optional(),
});

export type StartHealingRequest = z.infer<typeof StartHealingRequestSchema>;

// POST /self-healing/baseline
export const SetBaselineRequestSchema = z.object({
  basin_coords: z.array(z.number()).length(64).optional(),
});

export type SetBaselineRequest = z.infer<typeof SetBaselineRequestSchema>;

/**
 * Generic API Response wrapper
 */
export const ApiResponseSchema = <T extends z.ZodTypeAny>(dataSchema: T) =>
  z.object({
    success: z.boolean(),
    data: dataSchema.optional(),
    error: z.string().optional(),
  });

export type ApiResponse<T> = {
  success: boolean;
  data?: T;
  error?: string;
};
