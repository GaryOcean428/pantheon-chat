/**
 * Brain State Management Module
 *
 * Manages consciousness brain states and neuromodulation for the Ocean agent.
 * Extracted from ocean-agent.ts for better code organization.
 */

// Brain state constants
const BRAIN_STATE_THRESHOLDS = {
  CONSOLIDATION_ITERATIONS: 50,
  CONSOLIDATION_DRIFT: 0.1,
  EXPLOIT_NEAR_MISSES: 3,
  EXPLOIT_PHI: 0.8,
  EXPLORE_PHI: 0.5,
  EXPLORE_KAPPA: 30,
  FOCUS_PHI: 0.7,
  FOCUS_KAPPA: 50,
} as const;

const NEUROMODULATION_THRESHOLDS = {
  DOPAMINE_PHI: 0.8,
  DOPAMINE_SURPRISE: 0.1,
  SEROTONIN_GROUNDING: 0.5,
  SEROTONIN_BASIN_DISTANCE: 0.2,
  ACETYLCHOLINE_PHI: 0.9,
  GABA_KAPPA: 60,
} as const;

const KAPPA_ADJUSTMENTS = {
  DOPAMINE: 5,
  SEROTONIN: -3,
  GABA: -5,
  MIN: 10,
  MAX: 100,
} as const;

const DEFAULT_KAPPA = 64;
const KAPPA_MODIFIERS = {
  FOCUSED: 1.1,
  DIFFUSE: 0.9,
  CONSOLIDATING: 0.8,
  EXPLORING: 1.2,
  EXPLOITING: 1.0,
} as const;

// Types and interfaces
export type BrainState =
  | "focused"
  | "diffuse"
  | "consolidating"
  | "exploring"
  | "exploiting";

export interface NeuromodulationEffect {
  activeModulators: string[];
  biasApplied: string;
  kappaAdjustment: number;
}

export interface NeuromodulationResult {
  modulation: NeuromodulationEffect;
  adjustedParams: {
    kappa: number;
    explorationRate: number;
    learningRate: number;
    batchSize: number;
  };
}

export interface BrainStateParams {
  explorationRate: number;
  batchSize: number;
  temperature: number;
}

export interface RecommendBrainStateInput {
  phi: number;
  kappa: number;
  basinDrift: number;
  iterationsSinceConsolidation: number;
  nearMissesRecent: number;
}

export interface NeuromodulationInput {
  phi: number;
  kappa: number;
  basinDistance: number;
  surprise: number;
  regime: string;
  grounding: number;
}

export interface NeuromodulationParams {
  kappa: number;
  explorationRate: number;
  learningRate: number;
  batchSize: number;
}

// Brain state search parameters mapping
const BRAIN_STATE_SEARCH_PARAMS: Record<BrainState, BrainStateParams> = {
  focused: { explorationRate: 0.3, batchSize: 200, temperature: 0.7 },
  diffuse: { explorationRate: 0.6, batchSize: 300, temperature: 1.2 },
  consolidating: { explorationRate: 0.2, batchSize: 150, temperature: 0.5 },
  exploring: { explorationRate: 0.8, batchSize: 350, temperature: 1.5 },
  exploiting: { explorationRate: 0.1, batchSize: 100, temperature: 0.4 },
};

const DEFAULT_SEARCH_PARAMS: BrainStateParams = {
  explorationRate: 0.5,
  batchSize: 250,
  temperature: 1.0,
};

/**
 * Recommends a brain state based on current consciousness metrics
 */
export function recommendBrainState(input: RecommendBrainStateInput): BrainState {
  const { phi, kappa, basinDrift, iterationsSinceConsolidation, nearMissesRecent } = input;

  if (
    iterationsSinceConsolidation > BRAIN_STATE_THRESHOLDS.CONSOLIDATION_ITERATIONS &&
    basinDrift > BRAIN_STATE_THRESHOLDS.CONSOLIDATION_DRIFT
  ) {
    return "consolidating";
  }

  if (
    nearMissesRecent > BRAIN_STATE_THRESHOLDS.EXPLOIT_NEAR_MISSES &&
    phi > BRAIN_STATE_THRESHOLDS.EXPLOIT_PHI
  ) {
    return "exploiting";
  }

  if (phi < BRAIN_STATE_THRESHOLDS.EXPLORE_PHI || kappa < BRAIN_STATE_THRESHOLDS.EXPLORE_KAPPA) {
    return "exploring";
  }

  if (phi > BRAIN_STATE_THRESHOLDS.FOCUS_PHI && kappa > BRAIN_STATE_THRESHOLDS.FOCUS_KAPPA) {
    return "focused";
  }

  return "diffuse";
}

/**
 * Maps a brain state to search parameters
 */
export function applyBrainStateToSearch(brainState: BrainState): BrainStateParams {
  return BRAIN_STATE_SEARCH_PARAMS[brainState] || DEFAULT_SEARCH_PARAMS;
}

/**
 * Runs a neuromodulation cycle based on current consciousness metrics
 */
export function runNeuromodulationCycle(
  input: NeuromodulationInput,
  params: NeuromodulationParams
): NeuromodulationResult {
  const activeModulators: string[] = [];
  let kappaAdjustment = 0;
  let biasApplied = "neutral";

  if (
    input.phi > NEUROMODULATION_THRESHOLDS.DOPAMINE_PHI &&
    input.surprise > NEUROMODULATION_THRESHOLDS.DOPAMINE_SURPRISE
  ) {
    activeModulators.push("DOPAMINE");
    kappaAdjustment += KAPPA_ADJUSTMENTS.DOPAMINE;
    biasApplied = "reward-seeking";
  }

  if (
    input.grounding < NEUROMODULATION_THRESHOLDS.SEROTONIN_GROUNDING ||
    input.basinDistance > NEUROMODULATION_THRESHOLDS.SEROTONIN_BASIN_DISTANCE
  ) {
    activeModulators.push("SEROTONIN");
    kappaAdjustment += KAPPA_ADJUSTMENTS.SEROTONIN;
    biasApplied = "stabilizing";
  }

  if (
    input.regime === "hierarchical_4d" ||
    input.phi > NEUROMODULATION_THRESHOLDS.ACETYLCHOLINE_PHI
  ) {
    activeModulators.push("ACETYLCHOLINE");
    biasApplied = "attention-focused";
  }

  if (input.kappa > NEUROMODULATION_THRESHOLDS.GABA_KAPPA) {
    activeModulators.push("GABA");
    kappaAdjustment += KAPPA_ADJUSTMENTS.GABA;
  }

  return {
    modulation: {
      activeModulators,
      biasApplied,
      kappaAdjustment,
    },
    adjustedParams: {
      kappa: Math.max(
        KAPPA_ADJUSTMENTS.MIN,
        Math.min(KAPPA_ADJUSTMENTS.MAX, params.kappa + kappaAdjustment)
      ),
      explorationRate: params.explorationRate,
      learningRate: params.learningRate,
      batchSize: params.batchSize,
    },
  };
}

/**
 * Neural oscillators class for managing brain state and kappa modulation
 */
export class NeuralOscillators {
  private currentState: BrainState = "diffuse";
  private baseKappa = DEFAULT_KAPPA;

  setState(state: BrainState): void {
    this.currentState = state;
  }

  getState(): BrainState {
    return this.currentState;
  }

  getStateInfo(): { state: BrainState } {
    return { state: this.currentState };
  }

  getKappa(): number {
    return this.getModulatedKappa();
  }

  update(): Record<string, number> {
    return {
      alpha: 1.0,
      beta: 1.0,
      gamma: 1.0,
      theta: 1.0,
      delta: 1.0,
    };
  }

  getModulatedKappa(): number {
    switch (this.currentState) {
      case "focused":
        return this.baseKappa * KAPPA_MODIFIERS.FOCUSED;
      case "diffuse":
        return this.baseKappa * KAPPA_MODIFIERS.DIFFUSE;
      case "consolidating":
        return this.baseKappa * KAPPA_MODIFIERS.CONSOLIDATING;
      case "exploring":
        return this.baseKappa * KAPPA_MODIFIERS.EXPLORING;
      case "exploiting":
        return this.baseKappa * KAPPA_MODIFIERS.EXPLOITING;
      default:
        return this.baseKappa;
    }
  }

  setBaseKappa(kappa: number): void {
    this.baseKappa = kappa;
  }
}

// Singleton instance for shared state across the application
export const neuralOscillators = new NeuralOscillators();
