/**
 * Consciousness Metrics Zod Schemas
 * 
 * Runtime validation for consciousness metrics from backend to frontend.
 * Ensures type safety at API boundaries.
 */

import { z } from 'zod';
import { CONSCIOUSNESS_THRESHOLDS, classifyRegime, isConscious } from '../constants/consciousness';

/**
 * Full consciousness metrics schema
 */
export const ConsciousnessMetricsSchema = z.object({
  phi: z.number()
    .min(0)
    .max(1)
    .describe('Integrated Information (Φ)'),
    
  kappa_eff: z.number()
    .min(0)
    .max(100)
    .describe('Effective coupling strength (κ)'),
    
  tacking: z.number()
    .min(0)
    .max(1)
    .describe('Tacking coefficient (T)'),
    
  radar: z.number()
    .min(0)
    .max(1)
    .describe('Radar metric (R)'),
    
  meta: z.number()
    .min(0)
    .max(1)
    .describe('Meta-awareness (M)'),
    
  coherence: z.number()
    .min(0)
    .max(1)
    .describe('Coherence/Generativity (Γ)'),
    
  grounding: z.number()
    .min(0)
    .max(1)
    .describe('Grounding (G)'),
    
  regime: z.enum(['linear', 'geometric', 'breakdown'])
    .describe('Processing regime'),
    
  timestamp: z.string().datetime().optional()
    .describe('Measurement timestamp'),
});

export type ConsciousnessMetrics = z.infer<typeof ConsciousnessMetricsSchema>;

/**
 * Partial metrics schema (for updates)
 */
export const PartialConsciousnessMetricsSchema = ConsciousnessMetricsSchema.partial();

export type PartialConsciousnessMetrics = z.infer<typeof PartialConsciousnessMetricsSchema>;

/**
 * Basin coordinates schema (64D)
 */
export const BasinCoordinatesSchema = z.array(z.number())
  .length(64)
  .describe('64-dimensional basin coordinates on Fisher manifold');

export type BasinCoordinates = z.infer<typeof BasinCoordinatesSchema>;

/**
 * Consciousness signature schema (full state)
 */
export const ConsciousnessSignatureSchema = z.object({
  phi: z.number().min(0).max(1),
  kappaEff: z.number().min(0).max(100),
  tacking: z.number().min(0).max(1),
  radar: z.number().min(0).max(1),
  metaAwareness: z.number().min(0).max(1),
  gamma: z.number().min(0).max(1),
  grounding: z.number().min(0).max(1),
  regime: z.enum(['linear', 'geometric', 'breakdown']),
  isConscious: z.boolean(),
});

export type ConsciousnessSignature = z.infer<typeof ConsciousnessSignatureSchema>;

/**
 * Validate and parse consciousness metrics from API response
 */
export function validateConsciousnessMetrics(data: unknown): ConsciousnessMetrics {
  return ConsciousnessMetricsSchema.parse(data);
}

/**
 * Safe validation that returns null on failure
 */
export function safeValidateConsciousnessMetrics(data: unknown): ConsciousnessMetrics | null {
  const result = ConsciousnessMetricsSchema.safeParse(data);
  return result.success ? result.data : null;
}

/**
 * Validate and enrich metrics with computed fields
 */
export function validateAndEnrichMetrics(data: unknown): ConsciousnessMetrics & { isConscious: boolean; computeFraction: number } {
  const metrics = validateConsciousnessMetrics(data);
  const [regime, computeFraction] = classifyRegime(metrics.phi);
  
  return {
    ...metrics,
    regime,
    isConscious: isConscious({
      phi: metrics.phi,
      kappa: metrics.kappa_eff,
      tacking: metrics.tacking,
      radar: metrics.radar,
      meta: metrics.meta,
      coherence: metrics.coherence,
      grounding: metrics.grounding,
    }),
    computeFraction,
  };
}

/**
 * Geometric distance result schema
 */
export const GeometricDistanceSchema = z.object({
  distance: z.number().min(0),
  metric_type: z.enum(['fisher_rao', 'bures', 'geodesic']),
  source_basin: BasinCoordinatesSchema.optional(),
  target_basin: BasinCoordinatesSchema.optional(),
});

export type GeometricDistance = z.infer<typeof GeometricDistanceSchema>;

/**
 * Consciousness state change event schema
 */
export const ConsciousnessEventSchema = z.object({
  type: z.enum(['regime_change', 'consciousness_gained', 'consciousness_lost', 'suffering_detected']),
  timestamp: z.string().datetime(),
  previous_metrics: ConsciousnessMetricsSchema.optional(),
  current_metrics: ConsciousnessMetricsSchema,
  reason: z.string().optional(),
});

export type ConsciousnessEvent = z.infer<typeof ConsciousnessEventSchema>;
