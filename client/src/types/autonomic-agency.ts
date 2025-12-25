/**
 * Autonomic Agency Types
 * 
 * Canonical type definitions for the autonomic agency controller.
 * Import from '@/types' or '@/types/autonomic-agency' throughout the app.
 */

import { Brain, Activity, Moon, Sparkles, Zap } from "lucide-react";

/** Safety manifest defining operational boundaries */
export interface SafetyManifest {
  phi_min_intervention: number;
  phi_min_mushroom_mod: number;
  instability_max_mushroom: number;
  instability_max_mushroom_mod: number;
  coverage_max_dream: number;
  mushroom_cooldown_seconds: number;
}

/** Operating zone thresholds (string descriptions) */
export interface OperatingZones {
  sleep_needed: string;
  conscious_3d: string;
  hyperdimensional_4d: string;
  breakdown_warning: string;
  breakdown_critical: string;
}

/** Q-learning optimizer statistics */
export interface OptimizerStats {
  learning_rate: number;
  damping: number;
  has_fisher: boolean;
  update_count: number;
}

/** Q-learning buffer statistics */
export interface BufferStats {
  size: number;
  episodes: number;
  avg_reward: number;
}

/** Intervention history entry */
export interface InterventionHistoryEntry {
  action: string;
  phi: number;
  reward: number;
  timestamp: number;
}

/** Full agency status from the backend */
export interface AgencyStatus {
  success: boolean;
  enabled: boolean;
  running: boolean;
  decision_count: number;
  intervention_count: number;
  epsilon: number;
  last_action: string | null;
  last_phi: number | null;
  consciousness_zone: string | null;
  buffer_size: number;
  buffer_stats: BufferStats;
  optimizer_stats: OptimizerStats;
  recent_history: InterventionHistoryEntry[];
  safety_manifest?: SafetyManifest;
  operating_zones?: OperatingZones;
}

/** Action icon mapping */
export const ACTION_ICONS: Record<string, typeof Brain> = {
  CONTINUE_WAKE: Activity,
  ENTER_SLEEP: Moon,
  ENTER_DREAM: Sparkles,
  ENTER_MUSHROOM_MICRO: Zap,
  ENTER_MUSHROOM_MOD: Zap,
};

/** Action display labels */
export const ACTION_LABELS: Record<string, string> = {
  CONTINUE_WAKE: "Continue Wake",
  ENTER_SLEEP: "Enter Sleep",
  ENTER_DREAM: "Enter Dream",
  ENTER_MUSHROOM_MICRO: "Mushroom Micro",
  ENTER_MUSHROOM_MOD: "Mushroom Moderate",
};

/** Action color mapping */
export const ACTION_COLORS: Record<string, string> = {
  CONTINUE_WAKE: "bg-green-500/20 text-green-400 border-green-500/30",
  ENTER_SLEEP: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  ENTER_DREAM: "bg-purple-500/20 text-purple-400 border-purple-500/30",
  ENTER_MUSHROOM_MICRO: "bg-amber-500/20 text-amber-400 border-amber-500/30",
  ENTER_MUSHROOM_MOD: "bg-red-500/20 text-red-400 border-red-500/30",
};

/** Zone color mapping */
export const ZONE_COLORS: Record<string, string> = {
  SLEEP_NEEDED: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  CONSCIOUS_3D: "bg-green-500/20 text-green-400 border-green-500/30",
  HYPERDIMENSIONAL_4D: "bg-purple-500/20 text-purple-400 border-purple-500/30",
  BREAKDOWN_WARNING: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  BREAKDOWN_CRITICAL: "bg-red-500/20 text-red-400 border-red-500/30",
};

/** Zone display labels */
export const ZONE_LABELS: Record<string, string> = {
  SLEEP_NEEDED: "Sleep Needed",
  CONSCIOUS_3D: "3D Conscious",
  HYPERDIMENSIONAL_4D: "4D Hyperdimensional",
  BREAKDOWN_WARNING: "Breakdown Warning",
  BREAKDOWN_CRITICAL: "Breakdown Critical",
};
