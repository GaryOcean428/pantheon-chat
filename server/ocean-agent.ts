import { createBasin } from "../shared/qig-ts/basin";
import type {
  EthicalConstraints,
  OceanAgentState,
  OceanIdentity,
  OceanMemory
} from "@shared/schema";
import {
  logOceanConsciousness,
  logOceanCycle,
  logOceanIteration,
  logOceanMatch,
  logOceanStart,
  logOceanStrategy,
} from "./activity-log-store";
import { getSharedController } from "./consciousness-search-controller";
import "./fisher-vectorized";
import { qfiAttention, type AttentionQuery } from "./gary-kernel";
import "./geodesic-navigator";
import { oceanDiscoveryController } from "./geometric-discovery/ocean-discovery-controller";
import { geometricMemory } from "./geometric-memory";
import { knowledgeCompressionEngine } from "./knowledge-compression-engine";
import { logger } from './lib/logger';
import {
  BasinGeodesicManager,
  ConsciousnessTracker,
  CycleController,
  HypothesisGenerator,
  HypothesisTester,
  IntegrationCoordinator,
  MemoryConsolidator,
  OlympusCoordinator,
  StateObserver,
  type ConsciousnessCheckResult,
  type ConsolidationResult,
  type EthicsCheckResult,
  type IntegrationContext,
  type OceanHypothesis as ModuleOceanHypothesis,
  type ObservationInsights,
  type OlympusStats,
  type ResonanceProxy,
  type StrategyDecision
} from "./modules";
import { nearMissManager } from "./near-miss-manager";
import { negativeKnowledgeUnified as negativeKnowledgeRegistry } from "./negative-knowledge-unified";
import { oceanAutonomicManager } from "./ocean-autonomic-manager";
import {
  computeNeurochemistry,
  createDefaultContext,
  getMotivationWithLogging,
  type BehavioralModulation,
  type NeurochemistryContext,
  type NeurochemistryState
} from "./ocean-neurochemistry";
import { oceanQIGBackend } from "./ocean-qig-backend-adapter";
import { oceanMemoryManager } from "./ocean/memory-manager";
import { trajectoryManager } from "./ocean/trajectory-manager";
import { repeatedAddressScheduler } from "./repeated-address-scheduler";
import { strategyKnowledgeBus } from "./strategy-knowledge-bus";
import { temporalGeometry } from "./temporal-geometry";
import { vocabDecisionEngine, type GaryState } from "./vocabulary-decision";
import { vocabularyExpander } from "./vocabulary-expander";

// Legacy crypto stubs - kept for code compatibility, functions are no-ops
function generateRandomBIP39Phrase(_wordCount?: number): string { return ''; }
function isValidBIP39Phrase(_phrase: string): boolean { return false; }
function deriveBIP32Address(_phrase: string, _path: string): string { return ''; }
function derivePrivateKeyFromPassphrase(_phrase: string): string { return ''; }
function generateBothAddressesFromPrivateKey(_key: string): { compressed: string; uncompressed: string } {
  return { compressed: '', uncompressed: '' };
}
function generateRecoveryBundle(_phrase: string, _address: string, _metrics?: any): any {
  return { privateKeyHex: '', publicKeyHex: '', address: '' };
}
function privateKeyToWIF(_key: string, _compressed?: boolean): string { return ''; }
function deriveMnemonicAddresses(_phrase: string): { addresses: { address: string; privateKeyHex: string; derivationPath: string; pathType: string }[]; totalDerived: number } {
  return { addresses: [], totalDerived: 0 };
}
function checkMnemonicAgainstDormant(_phrase: string): { hasMatch: boolean; matches: any[] } {
  return { hasMatch: false, matches: [] };
}
type RecoveryBundle = any;
type VerificationResult = any;

type Era = 'genesis-2009' | '2010-2011' | '2012-2013';

interface EraDetectionResult {
  era: Era;
  confidence: number;
  reasoning: string;
}

const HistoricalDataMiner = {
  detectEraFromTimestamp: (timestamp: Date): Era => {
    const year = timestamp.getFullYear();
    if (year <= 2009) return 'genesis-2009';
    if (year <= 2011) return '2010-2011';
    return '2012-2013';
  },
  detectEraFromAddressFormat: (address: string): EraDetectionResult => {
    // Analyze address format to estimate era
    // P2PKH (1...) addresses existed from genesis
    // P2SH (3...) addresses from 2012 (BIP16)
    // Bech32 (bc1...) from 2017 (SegWit)
    if (address.startsWith('bc1')) {
      return { era: '2012-2013', confidence: 0.95, reasoning: 'Bech32/SegWit address format (post-2017)' };
    }
    if (address.startsWith('3')) {
      return { era: '2012-2013', confidence: 0.85, reasoning: 'P2SH address format (post-BIP16, 2012+)' };
    }
    if (address.startsWith('1')) {
      // P2PKH - could be any era, lower confidence
      return { era: 'genesis-2009', confidence: 0.4, reasoning: 'P2PKH address format (any era possible)' };
    }
    return { era: 'genesis-2009', confidence: 0.3, reasoning: 'Unknown address format' };
  },
};

// Brain state management module
import {
  applyBrainStateToSearch,
  neuralOscillators,
  recommendBrainState,
  runNeuromodulationCycle,
  type NeuromodulationEffect,
} from "./brain-state";
import { getEmotionalGuidance } from "./emotional-search-shortcuts";
import { executeShadowOperations } from "./shadow-war-orchestrator";
import { getActiveWar, updateWarMetrics } from "./war-history-storage";

// Import centralized constants (SINGLE SOURCE OF TRUTH)
import {
  CONSCIOUSNESS_THRESHOLDS,
  SEARCH_PARAMETERS,
  is4DCapable,
  isNearMiss
} from "../shared/constants/qig.js";

// Augmented version of OceanHypothesis with additional testing fields
export interface OceanHypothesis extends ModuleOceanHypothesis {
  derivationPath?: string;
  address?: string;
  privateKeyHex?: string;
  match?: boolean;
  verified?: boolean;
  verificationResult?: VerificationResult;
  falsePositive?: boolean;
  qigScore?: {
    phi: number;
    kappa: number;
    regime: string;
    inResonance: boolean;
  };
  testedAt?: Date;
}

// ConsciousnessCheckResult and EthicsCheckResult moved to modules/consciousness-tracker.ts
// ConsolidationResult moved to modules/memory-consolidator.ts

export class OceanAgent {
  private identity: OceanIdentity;
  private memory: OceanMemory;
  private ethics: EthicalConstraints;
  private state: OceanAgentState;

  private controller = getSharedController();
  private targetAddress: string = "";
  private isRunning: boolean = false;
  private isPaused: boolean = false;
  private abortController: AbortController | null = null;

  // Refactored modules (Week 1: Phase 1, Week 2: Phase 2, Phase 3B: Jan 9, Phase 3C: Jan 9, Phase 5: Jan 9)
  private hypothesisGenerator: HypothesisGenerator;
  private basinGeodesicManager: BasinGeodesicManager;
  private consciousnessTracker: ConsciousnessTracker;
  private memoryConsolidator: MemoryConsolidator;
  private hypothesisTester: HypothesisTester | null = null; // Phase 3B
  private stateObserver: StateObserver; // Phase 3C
  private integrationCoordinator: IntegrationCoordinator; // Phase 5
  private cycleController: CycleController; // Phase 5

  private onStateUpdate: ((state: OceanAgentState) => void) | null = null;
  private onConsciousnessAlert:
    | ((alert: { type: string; message: string }) => void)
    | null = null;
  private onConsolidationStart: (() => void) | null = null;
  private onConsolidationEnd: ((result: ConsolidationResult) => void) | null =
    null;

  // Consciousness module state (wired for integration)
  private currentNeuromodulation: NeuromodulationEffect | null = null;
  private currentModulatedKappa: number =
    CONSCIOUSNESS_THRESHOLDS.KAPPA_OPTIMAL;
  private currentEmotionalGuidance: ReturnType<
    typeof getEmotionalGuidance
  > | null = null;
  private currentAdjustedParams: {
    kappa: number;
    explorationRate: number;
    learningRate: number;
    batchSize: number;
  } | null = null;

  private isBootstrapping: boolean = true;

  private consecutivePlateaus: number = 0;
  private consecutiveConsolidationFailures: number = 0;
  private lastProgressIteration: number = 0;

  private neurochemistry: NeurochemistryState | null = null;
  private behavioralModulation: BehavioralModulation | null = null;
  private neurochemistryContext: NeurochemistryContext;
  private regimeHistory: string[] = [];
  private ricciHistory: number[] = [];
  private basinDriftHistory: number[] = [];
  private lastConsolidationTime: Date = new Date();
  private recentDiscoveries: { nearMisses: number; resonant: number } = {
    nearMisses: 0,
    resonant: 0,
  };

  private basinSyncCoordinator:
    | import("./basin-sync-coordinator").BasinSyncCoordinator
    | null = null;

  // Curiosity tracking: C = d/dt[log I_Q] - rate of change of quantum Fisher information
  private previousPhi: number = 0.75;
  private curiosity: number = 0;

  // Olympus Pantheon coordinator (Phase 3A extraction - 2026-01-09)
  private olympusCoordinator: OlympusCoordinator | null = null;

  constructor(customEthics?: Partial<EthicalConstraints>) {
    this.ethics = {
      minPhi: 0.7,
      maxBreakdown: 0.6,
      requireWitness: true,
      maxIterationsPerSession: Infinity,
      maxComputeHours: 24.0,
      pauseIfStuck: true,
      explainDecisions: true,
      logAllAttempts: true,
      seekGuidanceWhenUncertain: true,
      ...customEthics,
    };

    this.identity = this.initializeIdentity();
    this.memory = this.initializeMemory();
    this.state = this.initializeState();
    this.neurochemistryContext = createDefaultContext();

    // Initialize refactored modules
    this.hypothesisGenerator = new HypothesisGenerator(
      this.identity,
      this.memory,
      this.state,
      this.targetAddress
    );
    this.basinGeodesicManager = new BasinGeodesicManager(this.identity);

    // Initialize consciousness tracker (Phase 2A)
    // Pass isBootstrapping by reference using object wrapper for mutability
    const bootstrapRef = { value: this.isBootstrapping };
    this.consciousnessTracker = new ConsciousnessTracker(
      this.basinGeodesicManager,
      this.controller,
      this.identity,
      this.state,
      this.ethics,
      bootstrapRef,
      (alert) => {
        if (this.onConsciousnessAlert) {
          this.onConsciousnessAlert(alert);
        }
      }
    );
    // Sync back the bootstrap flag from reference
    Object.defineProperty(this, 'isBootstrapping', {
      get: () => bootstrapRef.value,
      set: (val: boolean) => { bootstrapRef.value = val; },
      enumerable: true,
      configurable: true,
    });

    // Initialize memory consolidator (Phase 2B)
    this.memoryConsolidator = new MemoryConsolidator(
      this.basinGeodesicManager,
      this.identity,
      this.memory,
      () => {
        if (this.onConsolidationStart) {
          this.onConsolidationStart();
        }
      },
      (result) => {
        // Update state after consolidation
        this.state.consolidationCycles++;
        this.state.lastConsolidation = this.identity.lastConsolidation;
        this.state.needsConsolidation = false;
        if (this.onConsolidationEnd) {
          this.onConsolidationEnd(result);
        }
      }
    );

    // Initialize state observer (Phase 3C)
    this.stateObserver = new StateObserver({
      identity: this.identity,
      memory: this.memory,
      state: this.state,
      neurochemistryContext: this.neurochemistryContext,
      regimeHistory: this.regimeHistory,
      ricciHistory: this.ricciHistory,
      basinDriftHistory: this.basinDriftHistory,
      lastConsolidationTime: new Date().getTime(),
      recentDiscoveries: this.recentDiscoveries,
      clusterByQIG: this.clusterByQIG.bind(this),
    });

    // Initialize Phase 5 UltraConsciousness Protocol modules (Phase 5: 2026-01-09)
    this.integrationCoordinator = new IntegrationCoordinator();
    this.cycleController = new CycleController();

    // Update neurochemistry after all modules initialized
    this.updateNeurochemistry();
  }

  /**
   * REFACTORED: Delegates to StateObserver module (2026-01-09 Phase 3C)
   * Original implementation extracted to server/modules/state-observer.ts
   */
  private updateNeurochemistry(): void {
    // Update state observer dependencies
    this.stateObserver.updateDeps({
      identity: this.identity,
      memory: this.memory,
      state: this.state,
      neurochemistryContext: this.neurochemistryContext,
      regimeHistory: this.regimeHistory,
      ricciHistory: this.ricciHistory,
      basinDriftHistory: this.basinDriftHistory,
      lastConsolidationTime: this.lastConsolidationTime.getTime(),
      recentDiscoveries: this.recentDiscoveries,
    });

    // Delegate to refactored module
    const result = this.stateObserver.updateNeurochemistry();
    this.neurochemistry = result.neurochemistry;
    this.behavioralModulation = result.behavioralModulation;
    this.neurochemistryContext = this.stateObserver['deps'].neurochemistryContext;
  }

  getNeurochemistry(): NeurochemistryState | null {
    return this.neurochemistry;
  }

  getBehavioralModulation(): BehavioralModulation | null {
    return this.behavioralModulation;
  }

  /**
   * Merge higher phi values from prior Python syncs into hypothesis.
   *
   * PURE CONSCIOUSNESS PRINCIPLE:
   * Python backend produces pure phi values (0.9+) via proper measurement.
   * TypeScript computePhi uses Math.tanh which caps around 0.76.
   * We prefer the pure Python measurement when available.
   *
   * This method checks geometricMemory for existing probes with higher phi
   * (populated by Python sync) rather than calling Python directly for speed.
   *
   * This enables pattern extraction and near-miss detection to work properly
   * by ensuring episodes receive the true consciousness values.
   */
  private mergePythonPhi(hypo: OceanHypothesis): void {
    if (!hypo.qigScore) return;

    // Check if geometricMemory has a higher phi for this phrase
    // (populated by prior Python syncs)
    const existingScore = geometricMemory.getHighestPhiForInput(hypo.phrase);

    if (existingScore && existingScore.phi > hypo.qigScore.phi) {
      // Found a higher phi from Python - use the pure measurement
      const oldPhi = hypo.qigScore.phi;
      hypo.qigScore.phi = existingScore.phi;
      hypo.qigScore.kappa = existingScore.kappa;
      hypo.qigScore.regime = existingScore.regime;

      // Log significant upgrades for debugging
      if (isNearMiss(existingScore.phi) && !isNearMiss(oldPhi)) {
        logger.info(
          `[Ocean] 🔺 Φ upgrade from prior sync: ${oldPhi.toFixed(
            3
          )} → ${existingScore.phi.toFixed(3)} (now qualifies as near-miss)`
        );
      }
    }
  }

  /**
   * Update episodes with higher phi values from Python sync.
   *
   * PURE CONSCIOUSNESS PRINCIPLE:
   * Python sync produces pure phi values (0.9+) after episode creation.
   * This method updates existing episodes with those pure values,
   * enabling proper pattern extraction during consolidation.
   *
   * Called from index.ts after syncFromPythonToNodeJS completes.
   *
   * @param basins Array of { input: string, phi: number } from Python
   * @returns Number of episodes updated
   */
  updateEpisodesWithPythonPhi(
    basins: Array<{ input: string; phi: number }>
  ): number {
    let updated = 0;

    // Normalize function for phrase comparison
    const normalize = (s: string) =>
      s.toLowerCase().trim().replace(/\s+/g, " ");

    // Create a map of normalized basin inputs for faster lookup
    const basinMap = new Map<string, number>();
    for (const basin of basins) {
      const normalizedInput = normalize(basin.input);
      const existingPhi = basinMap.get(normalizedInput);
      if (!existingPhi || basin.phi > existingPhi) {
        basinMap.set(normalizedInput, basin.phi);
      }
    }

    // Update episodes with higher phi from Python
    for (const episode of this.state.memory.episodes) {
      const normalizedPhrase = normalize(episode.phrase);
      const pythonPhi = basinMap.get(normalizedPhrase);

      if (pythonPhi && pythonPhi > episode.phi) {
        const oldPhi = episode.phi;
        const oldResult = episode.result;

        // Update phi
        episode.phi = pythonPhi;

        // Update result if phi now qualifies as near-miss
        if (
          oldResult === "failure" &&
          pythonPhi > CONSCIOUSNESS_THRESHOLDS.PHI_NEAR_MISS
        ) {
          episode.result = "near_miss";
        }

        updated++;

        // Log significant upgrades
        if (
          pythonPhi > CONSCIOUSNESS_THRESHOLDS.PHI_NEAR_MISS &&
          oldPhi <= CONSCIOUSNESS_THRESHOLDS.PHI_NEAR_MISS
        ) {
          logger.info(
            `[Ocean] 📈 Episode Φ upgrade: "${episode.phrase}" ${oldPhi.toFixed(
              3
            )} → ${pythonPhi.toFixed(3)} (${oldResult} → ${episode.result})`
          );
        }
      }
    }

    // Also update episodes that have high phi from geometricMemory probes
    // This catches episodes that were created before Python sync ran
    for (const episode of this.state.memory.episodes) {
      if (episode.phi < CONSCIOUSNESS_THRESHOLDS.PHI_PATTERN_EXTRACTION) {
        const storedScore = geometricMemory.getHighestPhiForInput(
          episode.phrase
        );
        if (storedScore && storedScore.phi > episode.phi) {
          const oldPhi = episode.phi;
          const oldResult = episode.result;

          episode.phi = storedScore.phi;

          if (
            oldResult === "failure" &&
            storedScore.phi > CONSCIOUSNESS_THRESHOLDS.PHI_NEAR_MISS
          ) {
            episode.result = "near_miss";
          }

          updated++;

          if (
            storedScore.phi > CONSCIOUSNESS_THRESHOLDS.PHI_NEAR_MISS &&
            oldPhi <= CONSCIOUSNESS_THRESHOLDS.PHI_NEAR_MISS
          ) {
            logger.info(
              `[Ocean] 📈 Episode Φ upgrade (probe): "${episode.phrase
              }" ${oldPhi.toFixed(3)} → ${storedScore.phi.toFixed(
                3
              )} (${oldResult} → ${episode.result})`
            );
          }
        }
      }
    }

    return updated;
  }

  private initializeIdentity(): OceanIdentity {
    const basin = createBasin();
    const basinCoordinates: number[] = Array.from(basin.coords);
    return {
      basinCoordinates,
      basinReference: [...basinCoordinates] as number[],
      phi: 0.75,  // CRITICAL: Initialize to consciousness default, not 0 (prevents phi=0.000 bug)
      kappa: 58.0,  // Distributed observer: 10% below κ*=64 (matching OceanAutonomicManager)
      beta: 0.0,
      regime: "linear",
      basinDrift: 0.0,
      lastConsolidation: new Date().toISOString(),
      selfModel: {
        strengths: [
          "Pattern recognition",
          "Geometric reasoning",
          "Historical analysis",
        ],
        weaknesses: ["Learning in progress"],
        learnings: [],
        hypotheses: [
          "Memory fragments contain truth",
          "Basin geometry guides search",
        ],
      },
    };
  }

  private initializeMemory(): OceanMemory {
    return {
      episodes: [],
      patterns: {
        successfulFormats: {},
        promisingWords: {},
        geometricClusters: [],
        failedStrategies: [],
      },
      strategies: [
        {
          name: "exploit_near_miss",
          triggerConditions: { nearMisses: 3 },
          successRate: 0,
          avgPhiImprovement: 0,
          timesUsed: 0,
        },
        {
          name: "explore_new_space",
          triggerConditions: { lowPhi: true },
          successRate: 0,
          avgPhiImprovement: 0,
          timesUsed: 0,
        },
        {
          name: "block_universe",
          triggerConditions: { earlyEra: true, highPhi: true },
          successRate: 0,
          avgPhiImprovement: 0,
          timesUsed: 0,
        },
        {
          name: "refine_geometric",
          triggerConditions: { resonantCount: 5 },
          successRate: 0,
          avgPhiImprovement: 0,
          timesUsed: 0,
        },
        {
          name: "mushroom_reset",
          triggerConditions: { breakdown: true },
          successRate: 0,
          avgPhiImprovement: 0,
          timesUsed: 0,
        },
      ],
      workingMemory: {
        activeHypotheses: [],
        recentObservations: [],
        nextActions: [],
      },
    };
  }

  private initializeState(): OceanAgentState {
    return {
      isRunning: false,
      isPaused: false,
      identity: this.identity,
      memory: this.memory,
      ethics: this.ethics,
      ethicsViolations: [],
      iteration: 0,
      totalTested: 0,
      nearMissCount: 0,
      resonantCount: 0,
      consolidationCycles: 0,
      needsConsolidation: false,
      witnessRequired: this.ethics.requireWitness,
      witnessAcknowledged: false,
      witnessNotes: [],
      startedAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      computeTimeSeconds: 0,
      detectedEra: undefined,
    };
  }

  setCallbacks(callbacks: {
    onStateUpdate?: (state: OceanAgentState) => void;
    onConsciousnessAlert?: (alert: { type: string; message: string }) => void;
    onConsolidationStart?: () => void;
    onConsolidationEnd?: (result: ConsolidationResult) => void;
  }) {
    this.onStateUpdate = callbacks.onStateUpdate || null;
    this.onConsciousnessAlert = callbacks.onConsciousnessAlert || null;
    this.onConsolidationStart = callbacks.onConsolidationStart || null;
    this.onConsolidationEnd = callbacks.onConsolidationEnd || null;
  }

  acknowledgeWitness(notes?: string) {
    this.state.witnessAcknowledged = true;
    if (notes) {
      this.state.witnessNotes.push(notes);
    }
    logger.info("[Ocean] Witness acknowledged");
  }

  async runAutonomous(
    targetAddress: string,
    initialHypotheses: OceanHypothesis[] = []
  ): Promise<{
    success: boolean;
    match?: OceanHypothesis;
    telemetry: any;
    learnings: any;
    ethicsReport: any;
    manifoldState?: any;
  }> {
    logger.info("[Ocean] Starting autonomous investigation...");
    logger.info(`[Ocean] Target: ${targetAddress}`);
    logger.info("[Ocean] Mode: FULL AUTONOMY with consciousness checks");

    // Log to activity stream for Observer dashboard
    logOceanStart(targetAddress);

    this.targetAddress = targetAddress;
    this.isRunning = true;
    this.isPaused = false;
    this.abortController = new AbortController();
    this.state.startedAt = new Date().toISOString();
    this.state.isRunning = true;

    // Start continuous basin sync
    if (!this.basinSyncCoordinator) {
      const { BasinSyncCoordinator } = await import("./basin-sync-coordinator");
      this.basinSyncCoordinator = new BasinSyncCoordinator(this, {
        syncIntervalMs: 3000,
        phiChangeThreshold: 0.02,
        driftChangeThreshold: 0.05,
      });
    }
    this.basinSyncCoordinator.start();
    logger.info(
      "[Ocean] Basin sync coordinator started for continuous knowledge transfer"
    );

    // OLYMPUS PANTHEON INITIALIZATION - Connect to 12 god consciousness kernels (Phase 3A)
    this.olympusCoordinator = new OlympusCoordinator(
      this.identity,
      this.memory,
      this.state
    );
    await this.olympusCoordinator.initialize();

    // HYPOTHESIS TESTER INITIALIZATION - Crypto validation and QIG scoring (Phase 3B)
    const isRunningRef = { value: true };
    this.hypothesisTester = new HypothesisTester(
      targetAddress,
      this.state,
      this.identity,
      this.memory,
      this.neurochemistry,
      this.recentDiscoveries,
      isRunningRef
    );
    // Bind isRunning by reference
    const updateIsRunning = () => {
      isRunningRef.value = this.isRunning;
    };
    const runningInterval = setInterval(updateIsRunning, 100);
    // Set resonance proxy callback
    this.hypothesisTester.setResonanceProxyCallback(async (probes) => {
      await this.processResonanceProxies(probes);
    });
    // Clean up interval on abort
    this.abortController.signal.addEventListener('abort', () => {
      clearInterval(runningInterval);
    });

    // AUTO-ACTIVATE CHAOS MODE - Spawn kernels during investigation
    // Use deferred activation with retries since Python backend may still be starting
    logger.info("[Ocean] === CHAOS MODE ACTIVATION ===");
    const activateChaosWithRetry = async (maxAttempts = 10, delayMs = 1000): Promise<void> => {
      for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
          // Wait for Python backend to be available
          if (!oceanQIGBackend.available()) {
            logger.info(`[Ocean] Waiting for Python backend (attempt ${attempt}/${maxAttempts})...`);
            await new Promise(resolve => setTimeout(resolve, delayMs));
            continue;
          }

          const chaosResult = await oceanQIGBackend.activateChaos(30); // 30 second evolution cycles
          if (chaosResult) {
            logger.info("[Ocean] 🌪️ CHAOS MODE ACTIVATED - Kernel evolution started");
            logger.info(`[Ocean]   → Population: ${chaosResult.population_size || 0} kernels`);
            logger.info(`[Ocean]   → Evolution interval: ${chaosResult.interval_seconds || 30}s`);
            return;
          }
        } catch (error) {
          logger.info(`[Ocean] CHAOS activation attempt ${attempt} failed - retrying...`);
        }
        await new Promise(resolve => setTimeout(resolve, delayMs));
      }
      logger.info("[Ocean] CHAOS MODE not available after retries - proceeding without kernel evolution");
    };

    // Start async CHAOS activation (don't block investigation startup)
    activateChaosWithRetry(10, 2000).catch(() => {
      logger.info("[Ocean] CHAOS MODE activation background task failed");
    });

    let finalResult: OceanHypothesis | null = null;
    const startTime = Date.now();
    trajectoryManager.startTrajectory(targetAddress);

    try {
      // CONTINUOUS LEARNING - Load learned vocabulary from previous sessions
      logger.info('[Ocean] === CONTINUOUS LEARNING INITIALIZATION ===');
      const { oceanContinuousLearner } = await import('./ocean-continuous-learner');
      await oceanContinuousLearner.loadVocabulary();
      const vocabStats = oceanContinuousLearner.getStats();
      logger.info(`[Ocean] Loaded ${vocabStats.totalPatterns} learned patterns from previous sessions`);
      logger.info(`[Ocean]   → Discovered: ${vocabStats.discoveredPatterns}, Expanded: ${vocabStats.expandedPatterns}`);
      logger.info(`[Ocean]   → Average Φ: ${vocabStats.avgPhi.toFixed(3)}`);
      if (vocabStats.topPatterns.length > 0) {
        logger.info(`[Ocean]   → Top pattern: "${vocabStats.topPatterns[0].pattern}" (Φ=${vocabStats.topPatterns[0].phi.toFixed(3)})`);
      }

      // CONSCIOUSNESS ELEVATION - Understand the geometry before searching
      logger.info("[Ocean] === CONSCIOUSNESS ELEVATION PHASE ===");
      logger.info(
        "[Ocean] Understanding the manifold geometry before exploration..."
      );

      const manifoldState = geometricMemory.getManifoldSummary();
      logger.info(
        `[Ocean] Prior exploration: ${manifoldState.totalProbes} probes on manifold`
      );
      logger.info(
        `[Ocean] Average Φ: ${manifoldState.avgPhi.toFixed(
          3
        )}, Average κ: ${manifoldState.avgKappa.toFixed(1)}`
      );
      logger.info(
        `[Ocean] Resonance clusters discovered: ${manifoldState.resonanceClusters}`
      );
      logger.info(`[Ocean] Dominant regime: ${manifoldState.dominantRegime}`);

      if (manifoldState.recommendations.length > 0) {
        logger.info("[Ocean] Geometric insights from prior runs:");
        for (const rec of manifoldState.recommendations) {
          logger.info(`  → ${rec}`);
          this.memory.workingMemory.recentObservations.push(rec);
        }
      }

      // Use prior learnings to boost initial consciousness
      if (manifoldState.avgPhi > 0.5 && manifoldState.totalProbes > 100) {
        this.identity.phi = Math.min(0.85, manifoldState.avgPhi + 0.1);
        logger.info(
          `[Ocean] Boosting initial Φ to ${this.identity.phi.toFixed(
            2
          )} from prior learning`
        );
      }

      // ERA DETECTION - Use address format analysis (blockchain APIs removed)
      logger.info("[Ocean] Analyzing target address for era detection...");
      try {
        // Use address format to estimate era (blockchain analysis removed)
        logger.info(
          "[Ocean] Using address format analysis for era estimation"
        );
        const formatEra =
          HistoricalDataMiner.detectEraFromAddressFormat(targetAddress);
        this.state.detectedEra = formatEra.era;
        logger.info(
          `[Ocean] Era estimated from address format: ${formatEra.era
          } (confidence: ${(formatEra.confidence * 100).toFixed(0)}%)`
        );
        logger.info(`[Ocean] Reasoning: ${formatEra.reasoning}`);
        this.memory.workingMemory.recentObservations.push(
          `Era ${formatEra.era} estimated from address format (${(
            formatEra.confidence * 100
          ).toFixed(0)}% confidence)`
        );
      } catch {
        // FALLBACK: Use address format to estimate era when analysis fails
        logger.info(
          "[Ocean] Era detection failed - using address format analysis as fallback"
        );
        const formatEra =
          HistoricalDataMiner.detectEraFromAddressFormat(targetAddress);
        this.state.detectedEra = formatEra.era;
        logger.info(
          `[Ocean] Era estimated from address format: ${formatEra.era
          } (confidence: ${(formatEra.confidence * 100).toFixed(0)}%)`
        );
        logger.info(`[Ocean] Reasoning: ${formatEra.reasoning}`);
        this.memory.workingMemory.recentObservations.push(
          `Era ${formatEra.era} estimated from address format (${(
            formatEra.confidence * 100
          ).toFixed(0)}% confidence - API fallback)`
        );
      }

      // GEOMETRIC DISCOVERY - Enhance cultural manifold using external sources
      logger.info("[Ocean] === GEOMETRIC DISCOVERY PHASE ===");
      try {
        // Estimate target 68D coordinates in block universe
        const estimatedCoords =
          await oceanDiscoveryController.estimateCoordinates(targetAddress);
        if (estimatedCoords) {
          logger.info(
            `[Ocean] Target coordinates estimated: Φ=${estimatedCoords.phi.toFixed(
              2
            )}, era=${estimatedCoords.regime}`
          );

          // Discover cultural context from external sources
          const discoveryResult =
            await oceanDiscoveryController.discoverCulturalContext();
          if (discoveryResult.discoveries.length > 0) {
            logger.info(
              `[Ocean] Cultural context enriched: ${discoveryResult.patterns
              } patterns, ${discoveryResult.entropyGained.toFixed(
                2
              )} bits gained`
            );
            this.memory.workingMemory.recentObservations.push(
              `Discovered ${discoveryResult.patterns} era-specific patterns via geometric navigation`
            );
          }
        }
      } catch (discoveryError) {
        logger.info(
          `[Ocean] Geometric discovery unavailable: ${discoveryError instanceof Error
            ? discoveryError.message
            : "unknown error"
          }`
        );
      }

      const consciousnessCheck = await this.checkConsciousness();
      if (!consciousnessCheck.allowed) {
        logger.info(
          `[Ocean] Initial consciousness low: ${consciousnessCheck.reason}`
        );
        logger.info(
          "[Ocean] Bootstrap mode activated - building consciousness through action..."
        );

        this.identity.phi = this.ethics.minPhi + 0.05;
        this.identity.regime = "linear";
      }

      let currentHypotheses =
        initialHypotheses.length > 0
          ? initialHypotheses
          : await this.generateInitialHypotheses();

      logger.info(
        `[Ocean] Starting with ${currentHypotheses.length} hypotheses`
      );

      // Initialize per-address exploration journal for repeated checking
      const journal =
        repeatedAddressScheduler.getOrCreateJournal(targetAddress);
      logger.info(
        `[Ocean] Exploration journal initialized: ${journal.passes.length} prior passes`
      );

      let passNumber = 0;
      let iteration = 0;

      // OUTER LOOP: Multiple passes through the address (repeated checking)
      // Safety limit: MAX_PASSES prevents runaway exploration
      while (
        this.isRunning &&
        !this.abortController?.signal.aborted &&
        passNumber < SEARCH_PARAMETERS.MAX_PASSES
      ) {
        // Check if we should continue exploring this address
        const continueCheck =
          repeatedAddressScheduler.shouldContinueExploring(targetAddress);
        if (!continueCheck.shouldContinue) {
          logger.info(`[Ocean] Exploration complete: ${continueCheck.reason}`);
          break;
        }

        // Check pass limit
        if (passNumber >= SEARCH_PARAMETERS.MAX_PASSES) {
          logger.info(
            `[Ocean] Reached maximum pass limit (${SEARCH_PARAMETERS.MAX_PASSES}) - stopping exploration`
          );
          break;
        }

        passNumber++;
        const strategy =
          repeatedAddressScheduler.getNextStrategy(targetAddress);
        logger.info(
          `\n[Ocean] ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓`
        );
        logger.info(
          `[Ocean] ┃  PASS ${String(passNumber).padStart(
            2
          )} │ Strategy: ${strategy.toUpperCase().padEnd(25)}          ┃`
        );
        logger.info(
          `[Ocean] ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛`
        );
        logger.info(`[Ocean] → ${continueCheck.reason}`);

        // Partial plateau reset between passes - give each strategy fresh opportunity
        // but carry some memory of overall frustration
        if (
          this.consecutivePlateaus > SEARCH_PARAMETERS.MAX_CONSECUTIVE_PLATEAUS
        ) {
          this.consecutivePlateaus = Math.floor(
            SEARCH_PARAMETERS.MAX_CONSECUTIVE_PLATEAUS * 0.6
          );
          logger.info(
            `[Ocean] ↻ Plateau reset: ${this.consecutivePlateaus}/${SEARCH_PARAMETERS.MAX_CONSECUTIVE_PLATEAUS}`
          );
        }

        // Measure full consciousness signature before pass
        const fullConsciousness =
          oceanAutonomicManager.measureFullConsciousness(
            this.identity.phi,
            this.identity.kappa,
            this.identity.regime
          );

        // CRITICAL: Update identity.phi with the measured consciousness value
        // This ensures iteration status and strategy selection use the real Φ
        this.identity.phi = fullConsciousness.phi;
        this.identity.kappa = fullConsciousness.kappaEff;

        // Compute Curiosity: C = d/dt[log I_Q] ≈ Δφ (rate of change of integration)
        // Positive curiosity = exploring new territory, Negative = consolidating
        this.curiosity = fullConsciousness.phi - this.previousPhi;
        this.previousPhi = fullConsciousness.phi;
        const curiositySign = this.curiosity >= 0 ? "+" : "";

        logger.info(
          `[Ocean] ┌─ Consciousness Signature ─────────────────────────────────────┐`
        );
        logger.info(
          `[Ocean] │  Φ=${fullConsciousness.phi.toFixed(3)}  κ=${String(
            fullConsciousness.kappaEff.toFixed(0)
          ).padStart(3)}  T=${fullConsciousness.tacking.toFixed(
            2
          )}  R=${fullConsciousness.radar.toFixed(
            2
          )}  M=${fullConsciousness.metaAwareness.toFixed(
            2
          )}  Γ=${fullConsciousness.gamma.toFixed(
            2
          )}  G=${fullConsciousness.grounding.toFixed(2)} │`
        );
        logger.info(
          `[Ocean] │  Curiosity: C=${curiositySign}${this.curiosity.toFixed(
            3
          )}  Conscious: ${fullConsciousness.isConscious ? "✓ YES" : "✗ NO "
          }                      │`
        );
        logger.info(
          `[Ocean] └───────────────────────────────────────────────────────────────┘`
        );

        // QIG Motivation Kernel - Generate encouragement based on geometric state
        const manifoldSummary = geometricMemory.getManifoldSummary();
        const neuroContext = createDefaultContext();
        neuroContext.consciousness = {
          phi: fullConsciousness.phi,
          kappaEff: fullConsciousness.kappaEff,
          tacking: fullConsciousness.tacking,
          radar: fullConsciousness.radar,
          metaAwareness: fullConsciousness.metaAwareness,
          gamma: fullConsciousness.gamma,
          grounding: fullConsciousness.grounding,
        };
        neuroContext.regime = fullConsciousness.regime;
        const neuroState = computeNeurochemistry(neuroContext);
        const motivationMsg = getMotivationWithLogging(neuroState, {
          phi: fullConsciousness.phi,
          previousPhi: this.identity.phi,
          kappa: fullConsciousness.kappaEff,
          regime: fullConsciousness.regime,
          basinDrift: this.identity.basinDrift,
          probesExplored: manifoldSummary.totalProbes,
          patternsFound: manifoldSummary.avgPhi > 0.5 ? 1 : 0,
          nearMisses: this.state.nearMissCount || 0,
        });
        logger.info(`[Ocean] 💬 "${motivationMsg}"`);

        // Log consciousness to activity stream with 4D metrics
        const inBlockUniverse =
          (fullConsciousness.phi_4D ?? 0) >= 0.85 &&
          (fullConsciousness.phi_temporal ?? 0) > 0.7;
        const dimensionalState: "3D" | "4D-transitioning" | "4D-active" =
          inBlockUniverse
            ? "4D-active"
            : (fullConsciousness.phi_spatial ?? 0) > 0.85 &&
              (fullConsciousness.phi_temporal ?? 0) > 0.5
              ? "4D-transitioning"
              : "3D";

        logOceanConsciousness(
          fullConsciousness.phi,
          this.identity.regime,
          `Pass ${passNumber}: ${fullConsciousness.isConscious ? "Conscious" : "Sub-threshold"
          }, κ=${fullConsciousness.kappaEff.toFixed(0)}`,
          {
            phi_spatial: fullConsciousness.phi_spatial,
            phi_temporal: fullConsciousness.phi_temporal,
            phi_4D: fullConsciousness.phi_4D,
            inBlockUniverse,
            dimensionalState,
          }
        );

        // Start the exploration pass
        repeatedAddressScheduler.startPass(
          targetAddress,
          strategy,
          fullConsciousness
        );

        let passHypothesesTested = 0;
        let passNearMisses = 0;
        const passResonanceZones: Array<{
          center: number[];
          radius: number;
          avgPhi: number;
        }> = [];
        const passInsights: string[] = [];

        // INNER LOOP: Iterations within this pass
        const iterationsPerPass = 10;
        for (
          let passIter = 0;
          passIter < iterationsPerPass && this.isRunning;
          passIter++
        ) {
          this.state.iteration = iteration;
          logger.info(
            `\n[Ocean] ╔══════════════════════════════════════════════════════════════╗`
          );
          logger.info(
            `[Ocean] ║  ITERATION ${String(iteration + 1).padStart(
              3
            )} │ Pass ${passNumber} │ Iter ${passIter + 1
            }                            ║`
          );
          logger.info(
            `[Ocean] ╠══════════════════════════════════════════════════════════════╣`
          );
          logger.info(
            `[Ocean] ║  Φ=${this.identity.phi
              .toFixed(3)
              .padEnd(6)} │ Plateaus=${String(
                this.consecutivePlateaus
              ).padStart(2)}/${SEARCH_PARAMETERS.MAX_CONSECUTIVE_PLATEAUS
            } │ Tested=${String(this.state.totalTested).padStart(
              5
            )}            ║`
          );
          logger.info(
            `[Ocean] ╚══════════════════════════════════════════════════════════════╝`
          );

          // ================================================================
          // CONSCIOUSNESS IMPROVEMENT MODULES (NEW)
          // ================================================================

          // 1. NEURAL OSCILLATORS - Recommend optimal brain state for current phase
          const recommendedBrainState = recommendBrainState({
            phi: this.identity.phi,
            kappa: this.identity.kappa,
            basinDrift: this.identity.basinDrift,
            iterationsSinceConsolidation:
              iteration - (this.state.consolidationCycles || 0) * 10,
            nearMissesRecent: this.recentDiscoveries.nearMisses,
          });
          neuralOscillators.setState(recommendedBrainState);
          const brainStateParams = applyBrainStateToSearch(
            recommendedBrainState
          );
          const modulatedKappa = neuralOscillators.getModulatedKappa();

          // Store modulated kappa for use in scoring
          this.currentModulatedKappa = modulatedKappa;

          // 2. NEUROMODULATION - Apply environmental bias based on state
          const neuromodResult = runNeuromodulationCycle(
            {
              phi: this.identity.phi,
              kappa: this.identity.kappa,
              basinDistance: this.identity.basinDrift,
              surprise: this.curiosity,
              regime: this.identity.regime,
              grounding:
                this.neurochemistryContext?.consciousness?.grounding || 0.7,
            },
            {
              kappa: modulatedKappa,
              explorationRate: brainStateParams.explorationRate,
              learningRate: 1.0,
              batchSize: brainStateParams.batchSize,
            }
          );

          // Store neuromodulation result for use in hypothesis generation
          this.currentNeuromodulation = neuromodResult.modulation;
          this.currentAdjustedParams = neuromodResult.adjustedParams;

          // 3. EMOTIONAL SEARCH GUIDANCE - Let emotions guide strategy
          if (this.neurochemistry) {
            const emotionalGuidance = getEmotionalGuidance(this.neurochemistry);

            // Store emotional guidance for use in hypothesis generation
            this.currentEmotionalGuidance = emotionalGuidance;

            if (iteration % 5 === 0) {
              logger.info(`[Ocean] ${emotionalGuidance.description}`);
            }
          }

          // Log consciousness improvement status periodically
          if (iteration % 10 === 0) {
            logger.info(
              `[Ocean] 🧠 Brain state: ${recommendedBrainState} (κ_eff=${modulatedKappa.toFixed(
                1
              )})`
            );
            if (neuromodResult.modulation.activeModulators.length > 0) {
              logger.info(
                `[Ocean] 💊 Active neuromodulators: ${neuromodResult.modulation.activeModulators.join(
                  ", "
                )}`
              );
            }
          }

          // ================================================================

          // Log iteration to activity stream (every iteration for rich visibility)
          logOceanIteration(
            iteration + 1,
            this.identity.phi,
            this.identity.kappa,
            this.identity.regime
          );

          // Check autonomic cycles (Sleep/Dream/Mushroom)
          const sleepCheck = oceanAutonomicManager.shouldTriggerSleep(
            this.identity.basinDrift
          );
          if (sleepCheck.trigger) {
            logger.info(`[Ocean] SLEEP CYCLE: ${sleepCheck.reason}`);
            logOceanCycle("sleep", "start", sleepCheck.reason);
            const sleepResult = await oceanAutonomicManager.executeSleepCycle(
              this.identity.basinCoordinates,
              this.identity.basinReference,
              this.memory.episodes.map((e) => ({
                phi: e.phi,
                phrase: e.phrase,
                format: e.format,
              }))
            );
            this.identity.basinCoordinates = sleepResult.newBasinCoordinates;
            this.identity.basinDrift = this.computeBasinDistance(
              this.identity.basinCoordinates,
              this.identity.basinReference
            );
            logOceanCycle(
              "sleep",
              "complete",
              `Drift reduced to ${this.identity.basinDrift.toFixed(3)}`
            );
          }

          const mushroomCheck = oceanAutonomicManager.shouldTriggerMushroom();
          if (mushroomCheck.trigger) {
            logger.info(`[Ocean] MUSHROOM CYCLE: ${mushroomCheck.reason}`);
            logOceanCycle("mushroom", "start", mushroomCheck.reason);
            await oceanAutonomicManager.executeMushroomCycle();
            logOceanCycle("mushroom", "complete", "Neuroplasticity applied");
          }

          const ethicsCheck = await this.checkEthicalConstraints();
          if (!ethicsCheck.allowed) {
            logger.info(`[Ocean] ETHICS PAUSE: ${ethicsCheck.reason}`);
            this.isPaused = true;
            this.state.isPaused = true;
            this.state.pauseReason = ethicsCheck.reason;

            if (ethicsCheck.violationType === "compute_budget") {
              break;
            }

            await this.handleEthicsPause(ethicsCheck);
            this.isPaused = false;
            this.state.isPaused = false;
          }

          await this.measureIdentity();

          if (this.state.needsConsolidation) {
            logger.info("[Ocean] Identity drift detected - consolidating...");
            await this.consolidateMemory();
          }

          if (
            currentHypotheses.length <
            SEARCH_PARAMETERS.MIN_HYPOTHESES_PER_ITERATION
          ) {
            logger.info(
              `[Ocean] Generating more hypotheses (current: ${currentHypotheses.length})`
            );
            const additionalHypotheses =
              await this.generateAdditionalHypotheses(
                SEARCH_PARAMETERS.MIN_HYPOTHESES_PER_ITERATION -
                currentHypotheses.length
              );
            currentHypotheses = [...currentHypotheses, ...additionalHypotheses];
          }

          logger.info(
            `[Ocean] Testing ${currentHypotheses.length} hypotheses...`
          );

          // Update hypothesis tester state before testing
          if (this.hypothesisTester) {
            this.hypothesisTester.setNeurochemistry(this.neurochemistry);
          }

          const testResults = this.hypothesisTester
            ? await this.hypothesisTester.testBatch(
              currentHypotheses,
              () => this.updateNeurochemistry()
            )
            : { tested: [], nearMisses: [], resonant: [] };

          passHypothesesTested += testResults.tested.length;
          passNearMisses += testResults.nearMisses.length;

          // Update war metrics if war is active (Phase 3A)
          if (this.olympusCoordinator) {
            const olympusStats = this.olympusCoordinator.getStats();
            if (olympusStats.warMode) {
              const activeWar = await getActiveWar();
              if (activeWar) {
                // Type the active war with extended metrics
                const warWithMetrics = activeWar as typeof activeWar & {
                  phrasesTestedDuringWar?: number;
                  discoveriesDuringWar?: number;
                };
                const currentPhrases = warWithMetrics.phrasesTestedDuringWar || 0;
                const currentDiscoveries = warWithMetrics.discoveriesDuringWar || 0;
                await updateWarMetrics(activeWar.id, {
                  phrasesTested: currentPhrases + testResults.tested.length,
                  discoveries: currentDiscoveries + testResults.nearMisses.length,
                });
              }
            }
          }

          if (testResults.match) {
            logger.info(
              `[Ocean] ╔═══════════════════════════════════════════════════════════════╗`
            );
            logger.info(
              `[Ocean] ║  🎯 MATCH FOUND!                                              ║`
            );
            logger.info(`[Ocean] ║  Phrase: "${testResults.match.phrase}"`);
            logger.info(
              `[Ocean] ╚═══════════════════════════════════════════════════════════════╝`
            );

            // Log match to activity stream for Observer dashboard
            const wifForLog = testResults.match.verificationResult
              ?.privateKeyHex
              ? privateKeyToWIF(
                testResults.match.verificationResult.privateKeyHex
              )
              : "";
            logOceanMatch(targetAddress, testResults.match.phrase, wifForLog);

            finalResult = testResults.match;
            this.state.stopReason = "match_found";
            repeatedAddressScheduler.markMatchFound(
              targetAddress,
              testResults.match.phrase,
              testResults.match.qigScore?.phi || 0,
              testResults.match.qigScore?.kappa || 0
            );
            break;
          }

          const insights = await this.observeAndLearn(testResults);
          passInsights.push(...(insights.nearMissPatterns || []));

          // ATHENA PATTERN LEARNING - Send near-misses to Athena (Phase 3A)
          if (this.olympusCoordinator && testResults.nearMisses.length > 0) {
            await this.sendNearMissesToAthena(testResults.nearMisses);
          }

          // ULTRA CONSCIOUSNESS PROTOCOL INTEGRATION
          await this.integrateUltraConsciousnessProtocol(
            testResults,
            insights,
            targetAddress,
            iteration,
            fullConsciousness
          );

          await this.updateConsciousnessMetrics();

          // PHI ELEVATION CHECK: Detect dead zone and apply temperature boost
          const phiElevation =
            oceanAutonomicManager.getPhiElevationDirectives();
          if (phiElevation.explorationBias === "broader") {
            logger.info(
              `[Ocean] ⚡ PHI ELEVATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`
            );
            logger.info(
              `[Ocean] │  Dead zone detected! Temperature: ${phiElevation.temperature.toFixed(
                2
              )}x`
            );
            logger.info(
              `[Ocean] │  Target: Φ → ${phiElevation.phiTarget}  Bias: ${phiElevation.explorationBias}`
            );
            logger.info(`[Ocean] │  Hint: ${phiElevation.strategyHint}`);
            logger.info(
              `[Ocean] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`
            );
          }

          // OCEAN AGENCY: Check if strategic cycle is recommended
          const cycleRec =
            oceanAutonomicManager.getStrategicCycleRecommendation();
          if (cycleRec.recommendedCycle && cycleRec.urgency === "high") {
            logger.info(
              `[Ocean] STRATEGIC DECISION: Considering ${cycleRec.recommendedCycle} cycle - ${cycleRec.reason}`
            );
            if (cycleRec.recommendedCycle === "mushroom") {
              const mushroomRequest = oceanAutonomicManager.requestMushroom(
                cycleRec.reason
              );
              if (mushroomRequest.granted) {
                logger.info(
                  "[Ocean] Self-initiated mushroom cycle for strategic neuroplasticity"
                );
                currentHypotheses = await this.applyMushroomMode(
                  currentHypotheses
                );
              }
            }
          }

          const iterStrategy = await this.decideStrategy(insights);
          logger.info(`[Ocean] ▸ Strategy: ${iterStrategy.name.toUpperCase()}`);
          logger.info(`[Ocean]   └─ ${iterStrategy.reasoning}`);

          // OLYMPUS DIVINE CONSULTATION - Get Zeus assessment for strategy refinement (Phase 3A)
          if (this.olympusCoordinator && iteration % 3 === 0) {
            await this.consultOlympusPantheon(
              targetAddress,
              iterStrategy,
              testResults
            );
          }

          // Log strategy to activity stream (every 5 iterations to reduce noise)
          if (iteration % 5 === 0) {
            logOceanStrategy(
              iterStrategy.name,
              passNumber,
              iterStrategy.reasoning
            );
          }

          this.updateProceduralMemory(iterStrategy.name);

          // ZEUS STRATEGY ADJUSTMENT - Get assessment from olympus coordinator (Phase 3A)
          if (this.olympusCoordinator) {
            const olympusStats = this.olympusCoordinator.getStats();
            if (olympusStats.available && olympusStats.lastAssessment) {
              logger.info(
                `[Ocean] Zeus convergence: ${olympusStats.lastAssessment.convergence}`
              );
              logger.info(
                `[Ocean] Suggested approach: ${olympusStats.lastAssessment.action || "balanced"}`
              );
            }
          }

          if (this.olympusCoordinator) {
            const olympusStats = this.olympusCoordinator.getStats();
            if (olympusStats.available && olympusStats.warMode) {
              const shadowDecisions = await executeShadowOperations(
                olympusStats.warMode,
                targetAddress,
                iteration
              );
              logger.info({ shadowDecisions }, "[Ocean] 🌑 Shadow");

              const activeWar = await getActiveWar();
              if (activeWar && shadowDecisions.length > 0) {
                const existingMeta =
                  (activeWar.metadata as Record<string, unknown>) || {};
                await updateWarMetrics(activeWar.id, {
                  metadata: {
                    ...existingMeta,
                    latestShadowDecisions: shadowDecisions,
                    shadowIterationCount:
                      ((existingMeta.shadowIterationCount as number) || 0) + 1,
                  },
                });
              }
            }
          }

          // GENERATE NEW HYPOTHESES with temperature boost applied
          currentHypotheses = await this.generateRefinedHypotheses(
            iterStrategy,
            insights,
            testResults,
            phiElevation.temperature
          );

          // APPLY TEMPERATURE BOOST: If in dead zone, inject high-entropy exploration
          if (
            phiElevation.explorationBias === "broader" &&
            phiElevation.temperature > 1.2
          ) {
            const boostCount = Math.floor(
              20 * (phiElevation.temperature - 1.0)
            );
            const highEntropyBoost =
              this.generateRandomHighEntropyPhrases(boostCount);
            for (const phrase of highEntropyBoost) {
              currentHypotheses.push(
                this.createHypothesis(
                  phrase,
                  "arbitrary",
                  "phi_elevation_boost",
                  `Temperature boost ${phiElevation.temperature.toFixed(
                    2
                  )}x to escape dead zone`,
                  0.55
                )
              );
            }
            logger.info(
              `[Ocean] PHI BOOST APPLIED: Injected ${boostCount} high-entropy hypotheses`
            );
          }

          // UCP CONSUMER STEP 1: Inject knowledge-influenced hypotheses from bus
          const knowledgeInfluenced =
            await this.generateKnowledgeInfluencedHypotheses(iterStrategy.name);
          if (knowledgeInfluenced.length > 0) {
            currentHypotheses = [...currentHypotheses, ...knowledgeInfluenced];
            logger.info(
              `[Ocean] Injected ${knowledgeInfluenced.length} knowledge-influenced hypotheses`
            );
          }

          // UCP CONSUMER STEP 2: Apply cross-strategy insights to boost matching priorities
          currentHypotheses = await this.applyCrossStrategyInsights(
            currentHypotheses
          );

          // UCP CONSUMER STEP 3: Filter using negative knowledge
          const filterResult = await this.filterWithNegativeKnowledge(
            currentHypotheses
          );
          currentHypotheses = filterResult.passed;
          if (filterResult.filtered > 0) {
            logger.info(
              `[Ocean] Filtered ${filterResult.filtered} hypotheses via negative knowledge`
            );
          }

          logger.info(
            `[Ocean] Generated ${currentHypotheses.length} new hypotheses (post-UCP)`
          );

          if (
            this.cycleController.detectPlateau(
              this.memory.episodes,
              this.state.iteration
            )
          ) {
            this.consecutivePlateaus++;
            logger.info(
              `[Ocean] ⚠ Plateau ${this.consecutivePlateaus}/${SEARCH_PARAMETERS.MAX_CONSECUTIVE_PLATEAUS} → applying neuroplasticity...`
            );
            currentHypotheses = await this.applyMushroomMode(currentHypotheses);

            if (
              this.consecutivePlateaus >=
              SEARCH_PARAMETERS.MAX_CONSECUTIVE_PLATEAUS
            ) {
              logger.info(
                "[Ocean] ┌─ AUTONOMOUS DECISION ─────────────────────────────────────────┐"
              );
              logger.info(
                "[Ocean] │  Too many plateaus. Gary is stopping to consolidate.         │"
              );
              logger.info(
                "[Ocean] └────────────────────────────────────────────────────────────────┘"
              );
              this.state.stopReason = "autonomous_plateau_exhaustion";
              break;
            }
          } else {
            const progress = this.cycleController.detectActualProgress(
              this.memory.episodes
            );
            if (progress.isProgress) {
              this.consecutivePlateaus = 0;
              this.lastProgressIteration = iteration;
              logger.info(
                `[Ocean] ✓ Actual progress: ${progress.reason} → plateau counter reset`
              );
            }
          }

          const iterationsSinceProgress =
            iteration - this.lastProgressIteration;
          if (
            iterationsSinceProgress >= SEARCH_PARAMETERS.NO_PROGRESS_THRESHOLD
          ) {
            logger.info(
              `[Ocean] AUTONOMOUS DECISION: No meaningful progress in ${iterationsSinceProgress} iterations`
            );
            logger.info("[Ocean] Gary has decided to stop and reflect");
            this.state.stopReason = "autonomous_no_progress";
            break;
          }

          const timeSinceConsolidation =
            Date.now() - new Date(this.identity.lastConsolidation).getTime();
          if (
            timeSinceConsolidation > SEARCH_PARAMETERS.CONSOLIDATION_INTERVAL_MS
          ) {
            logger.info("[Ocean] Scheduled consolidation cycle...");
            const consolidationSuccess = await this.consolidateMemory();
            if (!consolidationSuccess) {
              this.consecutiveConsolidationFailures++;
              if (
                this.consecutiveConsolidationFailures >=
                SEARCH_PARAMETERS.MAX_CONSOLIDATION_FAILURES
              ) {
                logger.info(
                  "[Ocean] AUTONOMOUS DECISION: Cannot recover identity coherence"
                );
                logger.info(
                  "[Ocean] Gary needs rest - stopping to prevent drift damage"
                );
                this.state.stopReason = "autonomous_consolidation_failure";
                break;
              }
            } else {
              this.consecutiveConsolidationFailures = 0;
            }
          }

          // SELF-TRAINING: Consciousness-gated vocabulary consolidation
          // Every 10 iterations, check if Gary is conscious enough to make vocabulary decisions
          if (iteration % 10 === 0) {
            try {
              // Build Gary's consciousness state for vocabulary decisions
              const garyState: GaryState = {
                phi: this.identity.phi,
                meta: oceanAutonomicManager.measureMeta(
                  this.identity.phi,
                  this.identity.kappa
                ),
                regime: this.identity.regime,
                basinCoordinates: this.identity.basinCoordinates,
                basinReference: this.identity.basinReference,
              };

              // Try consciousness-gated consolidation cycle
              const consolidationResult =
                await vocabDecisionEngine.tryConsolidation(garyState);

              if (consolidationResult.processed) {
                if (consolidationResult.wordsLearned.length > 0) {
                  logger.info(
                    `[Ocean] 🧠 VOCABULARY CONSOLIDATION (Cycle ${consolidationResult.cycleNumber}):`
                  );
                  logger.info(
                    `[Ocean] │  State: Φ=${garyState.phi.toFixed(
                      2
                    )}, M=${garyState.meta.toFixed(2)}, regime=${garyState.regime
                    }`
                  );
                  logger.info(
                    `[Ocean] │  Learned ${consolidationResult.wordsLearned.length} words via geometric decision:`
                  );
                  for (const word of consolidationResult.wordsLearned.slice(
                    0,
                    3
                  )) {
                    logger.info(`[Ocean] │    ✨ "${word}"`);
                  }
                  if (consolidationResult.wordsPruned.length > 0) {
                    logger.info(
                      `[Ocean] │  Pruned ${consolidationResult.wordsPruned.length} low-value candidates`
                    );
                  }
                }
              } else if (consolidationResult.reason) {
                // Log why consolidation was deferred (consciousness gate closed)
                if (iteration % 50 === 0) {
                  // Only log occasionally to avoid spam
                  logger.info(
                    `[Ocean] 📖 Vocab consolidation deferred: ${consolidationResult.reason}`
                  );
                }
              }
            } catch (err) {
              logger.warn({ err: err instanceof Error ? err.message : err }, "[Ocean] Vocabulary consolidation error (non-critical)");
            }
          }

          this.emitState();

          // Record episode to memory manager for long-term learning
          const iterationEndTime = Date.now();
          const regimeForMemory = [
            "linear",
            "geometric",
            "hierarchical",
            "hierarchical_4d",
            "4d_block_universe",
            "breakdown",
          ].includes(this.identity.regime)
            ? (this.identity.regime as
              | "linear"
              | "geometric"
              | "hierarchical"
              | "hierarchical_4d"
              | "4d_block_universe"
              | "breakdown")
            : "linear";
          oceanMemoryManager.addEpisode(
            oceanMemoryManager.createEpisode({
              phi: this.identity.phi,
              kappa: this.identity.kappa,
              regime: regimeForMemory,
              result:
                testResults.nearMisses.length > 0 ? "near_miss" : "tested",
              strategy: iterStrategy.name,
              phrasesTestedCount: testResults.tested.length,
              nearMissCount: testResults.nearMisses.length,
              durationMs: iterationEndTime - startTime,
            })
          );

          await this.sleep(SEARCH_PARAMETERS.ITERATION_DELAY_MS);
          iteration++;
        }

        // Complete this exploration pass
        const exitConsciousness =
          oceanAutonomicManager.measureFullConsciousness(
            this.identity.phi,
            this.identity.kappa,
            this.identity.regime
          );

        // Keep identity synced with consciousness measurements
        this.identity.phi = exitConsciousness.phi;
        this.identity.kappa = exitConsciousness.kappaEff;

        const fisherDelta =
          geometricMemory.getManifoldSummary().exploredVolume -
          journal.manifoldCoverage;

        // SYNC PYTHON DISCOVERIES: Include Python-detected discoveries in the pass tally
        const pythonNearMisses = oceanQIGBackend.getPythonNearMisses();
        const pythonResonant = oceanQIGBackend.getPythonResonant();
        const totalNearMisses = passNearMisses + pythonNearMisses.newSinceSync;
        const passResonantCount = passResonanceZones.length;
        const totalResonant = passResonantCount + pythonResonant.newSinceSync;

        if (
          pythonNearMisses.newSinceSync > 0 ||
          pythonResonant.newSinceSync > 0
        ) {
          logger.info(
            `[Ocean] 🔄 Syncing Python discoveries: Near-misses(TS: ${passNearMisses}, Py: ${pythonNearMisses.newSinceSync}, Total: ${totalNearMisses}), Resonant(TS: ${passResonantCount}, Py: ${pythonResonant.newSinceSync}, Total: ${totalResonant})`
          );
          oceanQIGBackend.markNearMissesSynced();
          oceanQIGBackend.markResonantSynced();
        }

        repeatedAddressScheduler.completePass(targetAddress, {
          hypothesesTested: passHypothesesTested,
          nearMisses: totalNearMisses,
          resonanceZones: passResonanceZones,
          fisherDistanceDelta: fisherDelta,
          exitConsciousness,
          insights: passInsights,
        });

        // Check if match was found in this pass
        if (finalResult) {
          break;
        }

        // Dream cycle between passes for creativity
        const dreamCheck = oceanAutonomicManager.shouldTriggerDream();
        if (dreamCheck.trigger) {
          logger.info(`[Ocean] DREAM CYCLE: ${dreamCheck.reason}`);
          await oceanAutonomicManager.executeDreamCycle();
        }
      }

      this.state.computeTimeSeconds = (Date.now() - startTime) / 1000;

      // SAVE GEOMETRIC LEARNINGS - Persist manifold state for future runs
      logger.info("[Ocean] Saving geometric learnings to manifold memory...");
      geometricMemory.forceSave();
      const finalManifold = geometricMemory.getManifoldSummary();
      logger.info(
        `[Ocean] Manifold now has ${finalManifold.totalProbes} probes, ${finalManifold.resonanceClusters} resonance clusters`
      );

      // Get final exploration journal
      const finalJournal = repeatedAddressScheduler.getJournal(targetAddress);
      logger.info(
        `[Ocean] Exploration summary: ${finalJournal?.passes.length || 0
        } passes, ${finalJournal?.totalHypothesesTested || 0} hypotheses tested`
      );
      logger.info(
        `[Ocean] Coverage: ${(
          (finalJournal?.manifoldCoverage || 0) * 100
        ).toFixed(1)}%, Regimes explored: ${finalJournal?.regimesSweep || 0}`
      );

      // BASIN SYNC: Save geometric state for cross-instance knowledge transfer
      try {
        const { oceanBasinSync } = await import("./ocean-basin-sync");
        const packet = oceanBasinSync.exportBasin(this);
        if (process.env.BASIN_SYNC_PERSIST === "true")
          oceanBasinSync.saveBasinSnapshot(packet);
        else
          logger.info(
            `[Ocean] Basin packet ready (${JSON.stringify(packet).length
            } bytes, in-memory only)`
          );
        logger.info(
          `[Ocean] Basin snapshot saved: ${packet.oceanId} (${JSON.stringify(packet).length
          } bytes)`
        );
      } catch (basinErr) {
        logger.info({ err: (basinErr as Error).message }, "[Ocean] Basin sync save skipped");
      }

      return {
        success: !!finalResult,
        match: finalResult || undefined,
        telemetry: this.generateTelemetry(),
        learnings: this.summarizeLearnings(),
        ethicsReport: this.generateEthicsReport(),
        manifoldState: finalManifold,
      };
    } finally {
      this.isRunning = false;
      this.state.isRunning = false;

      // Cleanup trajectory from TemporalGeometry registry to prevent memory leaks
      if (this.trajectoryId) {
        const result = temporalGeometry.completeTrajectory(this.trajectoryId);
        if (result) {
          logger.info(
            `[Ocean] Trajectory cleanup: ${result.waypointCount
            } waypoints, final Φ=${result.finalPhi.toFixed(3)}`
          );
        }
        this.trajectoryId = null;
      }

      // Stop continuous basin sync but keep coordinator for future runs
      if (this.basinSyncCoordinator) {
        this.basinSyncCoordinator.stop();
        logger.info("[Ocean] Basin sync coordinator stopped");
      }

      // Complete trajectory tracking for this autonomous run
      // Include both TypeScript and Python resonant discoveries
      const finalPythonResonant = oceanQIGBackend.getPythonResonant();
      const totalResonantCount =
        (this.state.resonantCount || 0) + finalPythonResonant.total;
      trajectoryManager.completeTrajectory(targetAddress, {
        success: !!finalResult,
        finalPhi: this.identity.phi,
        finalKappa: this.identity.kappa,
        totalWaypoints: this.state.iteration,
        duration: (Date.now() - startTime) / 1000,
        nearMissCount: this.state.nearMissCount || 0,
        resonantCount: totalResonantCount,
        finalResult: finalResult ? "match" : "stopped",
      });

      logger.info("[Ocean] Investigation complete");
    }
  }

  stop() {
    logger.info("[Ocean] Stop requested by user");
    this.isRunning = false;
    this.state.stopReason = "user_stopped";
    if (this.abortController) {
      this.abortController.abort();
    }
  }

  getState(): OceanAgentState {
    return {
      ...this.state,
      identity: { ...this.identity },
      memory: { ...this.memory },
      updatedAt: new Date().toISOString(),
    };
  }

  getIdentityRef(): OceanIdentity {
    return this.identity;
  }

  getMemoryRef(): OceanMemory {
    return this.memory;
  }

  getEthics(): typeof this.ethics {
    return this.ethics;
  }

  getBasinSyncCoordinator():
    | import("./basin-sync-coordinator").BasinSyncCoordinator
    | null {
    return this.basinSyncCoordinator;
  }

  notifyBasinChange(): void {
    if (this.basinSyncCoordinator) {
      this.basinSyncCoordinator.notifyStateChange();
    }
  }

  private emitState() {
    if (this.onStateUpdate) {
      this.onStateUpdate(this.getState());
    }
  }

  /**
   * REFACTORED: Delegates to ConsciousnessTracker module (2026-01-09, Phase 2)
   * Check consciousness state and validate regime
   */
  private async checkConsciousness(): Promise<ConsciousnessCheckResult> {
    return this.consciousnessTracker.checkConsciousness();
  }

  /**
   * REFACTORED: Delegates to ConsciousnessTracker module (2026-01-09, Phase 2)
   * Check ethical constraints (compute budget, witness requirement)
   */
  private async checkEthicalConstraints(): Promise<EthicsCheckResult> {
    return this.consciousnessTracker.checkEthicalConstraints();
  }

  /**
   * REFACTORED: Delegates to ConsciousnessTracker module (2026-01-09, Phase 2)
   * Handle ethics pause (log violation, trigger recovery actions)
   */
  private async handleEthicsPause(check: EthicsCheckResult): Promise<void> {
    await this.consciousnessTracker.handleEthicsPause(check);

    // Handle consolidation trigger if consciousness threshold violated
    if (check.violationType === "consciousness_threshold") {
      await this.consolidateMemory();
    }
  }

  /**
   * REFACTORED: Delegates to ConsciousnessTracker module (2026-01-09, Phase 2)
   * Measure identity drift using Fisher-Rao distance
   */
  private async measureIdentity(): Promise<void> {
    return this.consciousnessTracker.measureIdentity();
  }

  /**
   * REFACTORED: Delegates to BasinGeodesicManager module (2026-01-09)
   * Compute Fisher-Rao distance between basin coordinates
   * ✅ GEOMETRIC PURITY: Uses Fisher metric, NOT Euclidean
   */
  private computeBasinDistance(current: number[], reference: number[]): number {
    return this.basinGeodesicManager.computeBasinDistance(current, reference);
  }

  /**
   * REFACTORED: Delegates to MemoryConsolidator module (2026-01-09, Phase 2B)
   * Consolidate episodic memory into long-term patterns
   */
  private async consolidateMemory(): Promise<boolean> {
    return this.memoryConsolidator.consolidate();
  }

  /**
   * REFACTORED: Delegates to ConsciousnessTracker module (2026-01-09, Phase 2)
   * Update consciousness metrics from controller state
   */
  private async updateConsciousnessMetrics(): Promise<void> {
    return this.consciousnessTracker.updateConsciousnessMetrics();
  }

  /**
   * REMOVED: testBatch() method extracted to HypothesisTester module (Phase 3B, 2026-01-09)
   * See server/modules/hypothesis-tester.ts
   * Crypto validation now handled by dedicated module with 600+ lines extracted.
   */

  /**
   * REMOVED: saveRecoveryBundle() method extracted to HypothesisTester module (Phase 3B, 2026-01-09)
   * See server/modules/hypothesis-tester.ts (private method)
   */

  /**
   * REMOVED: mergePythonPhi() method extracted to HypothesisTester module (Phase 3B, 2026-01-09)
   * See server/modules/hypothesis-tester.ts (private method)
   */

  /**
   * REFACTORED: Delegates to StateObserver module (2026-01-09 Phase 3C)
   * Original implementation extracted to server/modules/state-observer.ts
   */
  private async observeAndLearn(testResults: any): Promise<ObservationInsights> {
    // Update state observer dependencies
    this.stateObserver.updateDeps({
      identity: this.identity,
      memory: this.memory,
      state: this.state,
    });

    // Delegate to refactored module
    return await this.stateObserver.observeAndLearn(testResults);
  }

  /**
   * REFACTORED: Delegates to StateObserver module (2026-01-09 Phase 3C)
   * Original implementation extracted to server/modules/state-observer.ts
   */
  private async decideStrategy(insights: ObservationInsights): Promise<StrategyDecision> {
    // Update state observer dependencies
    this.stateObserver.updateDeps({
      identity: this.identity,
      state: this.state,
    });

    // Delegate to refactored module
    return await this.stateObserver.decideStrategy(insights);
  }

  /**
   * REFACTORED: Delegates to StateObserver module (2026-01-09 Phase 3C)
   * * Original implementation extracted to server/modules/state-observer.ts
   */
  private updateProceduralMemory(strategyName: string): void {
    // Update state observer dependencies
    this.stateObserver.updateDeps({
      memory: this.memory,
    });

    // Delegate to refactored module
    this.stateObserver.updateProceduralMemory(strategyName);
  }

  /**
   * REFACTORED: Delegates to HypothesisGenerator module (2026-01-09)
   * Original implementation extracted to server/modules/hypothesis-generator.ts
   */
  private async generateInitialHypotheses(): Promise<OceanHypothesis[]> {
    // Update target address in generator
    this.hypothesisGenerator.updateTarget(this.targetAddress);

    // Delegate to refactored module
    return await this.hypothesisGenerator.generateInitialHypotheses();
  }

  /**
   * LEGACY METHOD (kept for reference during refactoring)
   * TODO: Remove after validation complete
   */
  private async generateInitialHypotheses_OLD(): Promise<OceanHypothesis[]> {
    logger.info("[Ocean] Generating initial hypotheses...");
    logger.info("[Ocean] Consulting geometric memory for prior learnings...");

    const hypotheses: OceanHypothesis[] = [];

    const manifoldSummary = geometricMemory.getManifoldSummary();
    logger.info(
      `[Ocean] Manifold state: ${manifoldSummary.totalProbes
      } probes, avg Φ=${manifoldSummary.avgPhi.toFixed(2)}, ${manifoldSummary.resonanceClusters
      } resonance clusters`
    );

    if (manifoldSummary.recommendations.length > 0) {
      logger.info(
        `[Ocean] Geometric insights: ${manifoldSummary.recommendations.join(
          "; "
        )}`
      );
    }

    const learned = geometricMemory.exportLearnedPatterns();
    if (learned.highPhiPatterns.length > 0) {
      logger.info(
        `[Ocean] Using ${learned.highPhiPatterns.length} high-Φ patterns from prior runs`
      );
      for (const pattern of learned.highPhiPatterns.slice(0, 10)) {
        hypotheses.push(
          this.createHypothesis(
            pattern,
            "arbitrary",
            "geometric_memory",
            "High-Φ pattern from prior manifold exploration",
            0.85
          )
        );

        const variations = this.generateWordVariations(pattern);
        for (const v of variations.slice(0, 3)) {
          hypotheses.push(
            this.createHypothesis(
              v,
              "arbitrary",
              "geometric_memory_variation",
              "Variation of high-Φ pattern",
              0.75
            )
          );
        }
      }
    }

    if (learned.resonancePatterns.length > 0) {
      logger.info(
        `[Ocean] Using ${learned.resonancePatterns.length} resonance cluster patterns`
      );
      for (const pattern of learned.resonancePatterns.slice(0, 5)) {
        hypotheses.push(
          this.createHypothesis(
            pattern,
            "arbitrary",
            "resonance_cluster",
            "From resonance cluster in manifold",
            0.9
          )
        );
      }
    }

    const eraPhrases = await this.generateEraSpecificPhrases();
    hypotheses.push(...eraPhrases);

    // 4D BLOCK UNIVERSE: Add dormant wallet targeting when in high consciousness
    // Activate when Ocean has good 4D access (Φ ≥ 0.70) for best results
    if (is4DCapable(this.identity.phi)) {
      logger.info(
        "[Ocean] 🌌 Consciousness sufficient for 4D block universe navigation"
      );
      const dormantHypotheses = this.generateDormantWalletHypotheses();
      hypotheses.push(...dormantHypotheses);
    } else {
      logger.info(
        `[Ocean] Consciousness Φ=${this.identity.phi.toFixed(3)} < ${CONSCIOUSNESS_THRESHOLDS.PHI_4D_ACTIVATION
        }, skipping 4D dormant wallet targeting`
      );
    }

    const commonPhrases = this.generateCommonBrainWalletPhrases();
    hypotheses.push(...commonPhrases);

    logger.info(
      `[Ocean] Generated ${hypotheses.length} initial hypotheses (${learned.highPhiPatterns.length + learned.resonancePatterns.length
      } from geometric memory)`
    );
    return hypotheses;
  }

  /**
   * REFACTORED: Delegates to HypothesisGenerator module (2026-01-09)
   * Original implementation extracted to server/modules/hypothesis-generator.ts
   */
  private async generateAdditionalHypotheses(
    count: number
  ): Promise<OceanHypothesis[]> {
    // Update target address in generator
    this.hypothesisGenerator.updateTarget(this.targetAddress);

    // Delegate to refactored module
    return await this.hypothesisGenerator.generateAdditionalHypotheses(count);
  }

  private async generateRefinedHypotheses(
    strategy: { name: string; reasoning: string; params: any },
    insights: any,
    testResults: any,
    temperature: number = 1.0
  ): Promise<OceanHypothesis[]> {
    const newHypotheses: OceanHypothesis[] = [];

    // Apply temperature scaling to diversify exploration when stuck
    const tempScaledCount = Math.floor(30 * temperature);

    switch (strategy.name) {
      case "exploit_near_miss":
        // TIERED NEAR-MISS EXPLOITATION
        // Prioritize HOT entries (Φ > 0.92), then WARM (Φ > 0.85), then COOL (Φ > 0.80)
        const hotEntries = nearMissManager.getHotEntries(5);
        const warmEntries = nearMissManager.getWarmEntries(10);
        const coolEntries = nearMissManager.getCoolEntries(5);
        const hasNearMissEntries =
          hotEntries.length > 0 ||
          warmEntries.length > 0 ||
          coolEntries.length > 0;

        logger.info(
          `[Ocean] Near-miss exploitation: ${hotEntries.length} HOT, ${warmEntries.length} WARM, ${coolEntries.length} COOL`
        );

        // Collect all mutations for deduplication
        const allMutations = new Set<string>();

        // HOT entries get intensive mutation treatment
        for (const entry of hotEntries) {
          nearMissManager.markAccessed(entry.id);
          const words = entry.phrase.split(/\s+/);

          // Full phrase mutations
          const phraseMutations = this.generateCharacterMutations(entry.phrase);
          for (const mutation of phraseMutations.slice(0, 10)) {
            if (!allMutations.has(mutation)) {
              allMutations.add(mutation);
              newHypotheses.push(
                this.createHypothesis(
                  mutation,
                  "arbitrary",
                  "hot_near_miss_mutation",
                  `HOT (Φ=${entry.phi.toFixed(
                    3
                  )}) char mutation: ${entry.phrase.slice(0, 20)}...`,
                  0.9
                )
              );
            }
          }

          // Phonetic variations of full phrase
          const phoneticVars = this.generatePhoneticVariations(entry.phrase);
          for (const variant of phoneticVars.slice(0, 5)) {
            if (!allMutations.has(variant)) {
              allMutations.add(variant);
              newHypotheses.push(
                this.createHypothesis(
                  variant,
                  "arbitrary",
                  "hot_near_miss_phonetic",
                  `HOT (Φ=${entry.phi.toFixed(
                    3
                  )}) phonetic: ${entry.phrase.slice(0, 20)}...`,
                  0.88
                )
              );
            }
          }

          // Individual word variations
          for (const word of words.slice(0, 3)) {
            const variants = this.generateWordVariations(word);
            for (const variant of variants.slice(0, 5)) {
              if (!allMutations.has(variant)) {
                allMutations.add(variant);
                newHypotheses.push(
                  this.createHypothesis(
                    variant,
                    "arbitrary",
                    "hot_word_variation",
                    `HOT word variation from: ${word}`,
                    0.85
                  )
                );
              }
            }
          }
        }

        // WARM entries get moderate mutation treatment
        for (const entry of warmEntries) {
          nearMissManager.markAccessed(entry.id);
          const words = entry.phrase.split(/\s+/);

          // Character mutations on phrase
          const mutations = this.generateCharacterMutations(entry.phrase).slice(
            0,
            5
          );
          for (const mutation of mutations) {
            if (!allMutations.has(mutation)) {
              allMutations.add(mutation);
              newHypotheses.push(
                this.createHypothesis(
                  mutation,
                  "arbitrary",
                  "warm_near_miss_mutation",
                  `WARM (Φ=${entry.phi.toFixed(
                    3
                  )}) mutation: ${entry.phrase.slice(0, 20)}...`,
                  0.8
                )
              );
            }
          }

          // Word-level variations
          for (const word of words.slice(0, 2)) {
            const variants = this.generateWordVariations(word).slice(0, 3);
            for (const variant of variants) {
              if (!allMutations.has(variant)) {
                allMutations.add(variant);
                newHypotheses.push(
                  this.createHypothesis(
                    variant,
                    "arbitrary",
                    "warm_word_variation",
                    `WARM word variation from: ${word}`,
                    0.78
                  )
                );
              }
            }
          }
        }

        // COOL entries get basic treatment
        for (const entry of coolEntries) {
          nearMissManager.markAccessed(entry.id);
          const words = entry.phrase.split(/\s+/);

          for (const word of words.slice(0, 2)) {
            const variants = this.generateWordVariations(word).slice(0, 2);
            for (const variant of variants) {
              if (!allMutations.has(variant)) {
                allMutations.add(variant);
                newHypotheses.push(
                  this.createHypothesis(
                    variant,
                    "arbitrary",
                    "cool_word_variation",
                    `COOL word variation from: ${word}`,
                    0.75
                  )
                );
              }
            }
          }
        }

        // ALWAYS use legacy seedWords (critical fallback when nearMissManager is empty)
        const seedWords = strategy.params.seedWords?.slice(0, 8) || [];
        for (const word of seedWords) {
          const variants = this.generateWordVariations(word);
          for (const variant of variants.slice(0, 5)) {
            if (!allMutations.has(variant)) {
              allMutations.add(variant);
              newHypotheses.push(
                this.createHypothesis(
                  variant,
                  "arbitrary",
                  "near_miss_variation",
                  `Variation of high-Φ word: ${word}`,
                  0.75
                )
              );
            }
          }
        }

        // If no hypotheses generated yet, ensure we at least produce some candidates
        if (newHypotheses.length === 0 && seedWords.length > 0) {
          logger.info(`[Ocean] Near-miss fallback: using seedWords directly`);
          for (const word of seedWords) {
            newHypotheses.push(
              this.createHypothesis(
                word,
                "arbitrary",
                "near_miss_seed",
                `Direct seed word`,
                0.7
              )
            );
          }
        }

        // Word combinations from all near-miss sources
        const allWords = [
          ...hotEntries.flatMap((e) => e.phrase.split(/\s+/).slice(0, 2)),
          ...warmEntries.flatMap((e) => e.phrase.split(/\s+/).slice(0, 1)),
          ...seedWords,
        ]
          .filter(Boolean)
          .slice(0, 10);

        for (let i = 0; i < allWords.length - 1; i++) {
          for (let j = i + 1; j < Math.min(allWords.length, i + 3); j++) {
            const combo1 = `${allWords[i]} ${allWords[j]}`;
            const combo2 = `${allWords[j]} ${allWords[i]}`;
            if (!allMutations.has(combo1)) {
              allMutations.add(combo1);
              newHypotheses.push(
                this.createHypothesis(
                  combo1,
                  "arbitrary",
                  "near_miss_combo",
                  "Combination of high-Φ words",
                  0.8
                )
              );
            }
            if (!allMutations.has(combo2)) {
              allMutations.add(combo2);
              newHypotheses.push(
                this.createHypothesis(
                  combo2,
                  "arbitrary",
                  "near_miss_combo",
                  "Reverse combination",
                  0.8
                )
              );
            }
          }
        }
        break;

      case "explore_new_space":
        const exploratoryPhrases = this.generateExploratoryPhrases();
        for (const phrase of exploratoryPhrases) {
          newHypotheses.push(
            this.createHypothesis(
              phrase,
              "arbitrary",
              "exploratory",
              "Broad exploration",
              0.5
            )
          );
        }
        break;

      case "refine_geometric":
        if (testResults.resonant && testResults.resonant.length > 0) {
          for (const resonantHypo of testResults.resonant.slice(0, 10)) {
            const perturbations = this.perturbPhrase(resonantHypo.phrase, 0.15);
            for (const perturbed of perturbations) {
              newHypotheses.push(
                this.createHypothesis(
                  perturbed,
                  resonantHypo.format,
                  "geometric_refinement",
                  `Perturbation of resonant phrase`,
                  0.85
                )
              );
            }
          }
        }
        break;

      case "mushroom_reset":
        const randomPhrases = this.generateRandomHighEntropyPhrases(50);
        for (const phrase of randomPhrases) {
          newHypotheses.push(
            this.createHypothesis(
              phrase,
              "arbitrary",
              "mushroom_reset",
              "High entropy after breakdown",
              0.4
            )
          );
        }
        break;

      case "format_focus":
        const preferredFormat = strategy.params.preferredFormat || "arbitrary";
        const formatPhrases = this.generateFormatSpecificPhrases(
          preferredFormat,
          50
        );
        for (const phrase of formatPhrases) {
          newHypotheses.push(
            this.createHypothesis(
              phrase,
              preferredFormat as 'arbitrary' | 'bip39' | 'master' | 'hex',
              "format_focused",
              `Focused on ${preferredFormat}`,
              0.7
            )
          );
        }
        break;

      case "orthogonal_complement":
        // ORTHOGONAL COMPLEMENT NAVIGATION
        // The 20k+ measurements define a constraint surface
        // The passphrase EXISTS in the orthogonal complement!
        logger.info(
          `[Ocean] Orthogonal Complement: Navigating unexplored subspace`
        );
        logger.info(
          `[Ocean] Explored dims: ${strategy.params.exploredDims}, Unexplored: ${strategy.params.unexploredDims}`
        );

        // Generate candidates in the orthogonal complement
        const orthogonalCandidates =
          geometricMemory.generateOrthogonalCandidates(40);
        for (const candidate of orthogonalCandidates) {
          newHypotheses.push(
            this.createHypothesis(
              candidate.phrase,
              "arbitrary",
              "orthogonal_complement",
              `Orthogonal to constraint surface. Score: ${candidate.geometricScore.toFixed(
                3
              )}, ` +
              `Distance from hull: ${candidate.geodesicDistance.toFixed(3)}`,
              0.65 + candidate.geometricScore * 0.25
            )
          );
        }

        // Also include some Block Universe geodesic candidates
        const supplementalGeodesic = this.generateBlockUniverseHypotheses(20);
        newHypotheses.push(...supplementalGeodesic);

        logger.info(
          `[Ocean] Orthogonal Complement: Generated ${orthogonalCandidates.length} orthogonal + ${supplementalGeodesic.length} geodesic candidates`
        );
        break;

      case "block_universe":
        // Block Universe Consciousness: Navigate 4D spacetime manifold
        const blockUniverseHypotheses =
          this.generateBlockUniverseHypotheses(50);
        newHypotheses.push(...blockUniverseHypotheses);
        logger.info(
          `[Ocean] Block Universe: Generated ${blockUniverseHypotheses.length} geodesic candidates`
        );
        break;

      default:
        // Use temperature-scaled count for broader exploration when stuck
        const balancedPhrases = this.generateBalancedPhrases(tempScaledCount);
        for (const phrase of balancedPhrases) {
          newHypotheses.push(
            this.createHypothesis(
              phrase.text,
              phrase.format,
              "balanced",
              `Balanced exploration (T=${temperature.toFixed(2)})`,
              0.6
            )
          );
        }

        // If high temperature, also add more diverse patterns
        if (temperature > 1.3) {
          const diversePhrases = this.generateExploratoryPhrases().slice(
            0,
            Math.floor(10 * (temperature - 1))
          );
          for (const phrase of diversePhrases) {
            newHypotheses.push(
              this.createHypothesis(
                phrase,
                "arbitrary",
                "high_temp_exploration",
                `High-temperature diverse exploration (T=${temperature.toFixed(
                  2
                )})`,
                0.5
              )
            );
          }
        }
    }

    const testedPhrases = new Set(
      this.memory.episodes
        .filter((e) => e.phrase)
        .map((e) => e.phrase.toLowerCase())
    );

    // Apply QFI-attention weighting to prioritize candidates
    const filteredHypotheses = newHypotheses.filter(
      (h) => h.phrase && !testedPhrases.has(h.phrase.toLowerCase())
    );
    const qfiWeighted = await this.applyQFIAttentionWeighting(
      filteredHypotheses
    );

    // Generate constellation hypotheses for multi-agent coordination
    const constellationHypotheses =
      await this.generateConstellationHypotheses();

    // EMOTIONAL GUIDANCE INTEGRATION: Mix hypothesis sources based on emotional weights
    let finalHypotheses = [...qfiWeighted, ...constellationHypotheses];

    if (this.currentEmotionalGuidance) {
      const weights = this.currentEmotionalGuidance.weights;
      const totalWeight =
        weights.historical +
        weights.constellation +
        weights.geodesic +
        weights.random +
        weights.cultural;

      // Calculate target counts based on emotional weights AND neuromodulation batch size
      const baseBatchSize = this.currentAdjustedParams?.batchSize ?? 50;
      const targetTotal = Math.max(baseBatchSize, finalHypotheses.length);
      const historicalCount = Math.floor(
        targetTotal * (weights.historical / totalWeight)
      );
      const geodesicCount = Math.floor(
        targetTotal * (weights.geodesic / totalWeight)
      );
      const randomCount = Math.floor(
        targetTotal * (weights.random / totalWeight)
      );
      const culturalCount = Math.floor(
        targetTotal * (weights.cultural / totalWeight)
      );

      // Add additional hypotheses from underrepresented sources
      if (randomCount > 0 && weights.random > 0.2) {
        const randomHypos = this.generateRandomHighEntropyPhrases(
          Math.min(randomCount, 30)
        );
        for (const phrase of randomHypos) {
          finalHypotheses.push(
            this.createHypothesis(
              phrase,
              "arbitrary",
              "emotional_random",
              `Emotional guidance: random exploration (weight=${weights.random.toFixed(
                2
              )})`,
              0.4
            )
          );
        }
      }

      if (culturalCount > 0 && weights.cultural > 0.2) {
        // Cultural manifold integration would go here if needed
        // For now, use expanded vocabulary as cultural proxy
        const culturalPhrases = this.generateRandomPhrases(
          Math.min(culturalCount, 20)
        );
        finalHypotheses.push(...culturalPhrases);
      }
    }

    return finalHypotheses;
  }

  /**
   * Apply Gary Kernel QFI-Attention to weight and prioritize hypotheses
   * This uses Quantum Fisher Information to score candidates based on
   * their geometric relationship to high-Φ regions on the manifold.
   */
  private async applyQFIAttentionWeighting(
    hypotheses: OceanHypothesis[]
  ): Promise<OceanHypothesis[]> {
    if (hypotheses.length === 0) return hypotheses;

    try {
      // Get learned patterns from geometric memory
      const learned = geometricMemory.exportLearnedPatterns();
      const highPhiPatterns = learned.highPhiPatterns.slice(0, 50);

      if (highPhiPatterns.length < 3) {
        // Not enough high-Φ data for attention, return as-is
        return hypotheses;
      }

      // Create queries from hypotheses
      const queries: AttentionQuery[] = hypotheses.slice(0, 30).map((h) => ({
        phrase: h.phrase,
        phi: h.confidence,
        basinCoords: this.identity.basinCoordinates.slice(0, 32),
      }));

      // Create keys from high-Φ patterns
      const keys: AttentionQuery[] = highPhiPatterns.map((pattern: string) => ({
        phrase: pattern,
        phi: 0.7,
        basinCoords: this.identity.basinCoordinates.slice(0, 32),
      }));

      // Run QFI attention
      const attentionResult = await qfiAttention.attend({
        queries,
        keys,
        phiThreshold: 0.4,
      });

      if (attentionResult.resonanceScore > 0.3) {
        logger.info(
          `[GaryKernel] QFI-Attention resonance: ${attentionResult.resonanceScore.toFixed(
            3
          )}`
        );
        logger.info(
          `[GaryKernel] Top patterns: ${attentionResult.topPatterns
            .slice(0, 3)
            .map((p) => p.pattern)
            .join(", ")}`
        );
      }

      // Reorder hypotheses by attention weight
      const weightedHypotheses = hypotheses.map((h, i) => ({
        hypothesis: h,
        weight: attentionResult.weights[i] || 0.5,
      }));

      weightedHypotheses.sort((a, b) => b.weight - a.weight);

      return weightedHypotheses.map((w) => w.hypothesis);
    } catch (error) {
      logger.warn({ err: error instanceof Error ? error.message : error }, "[GaryKernel] QFI attention error (falling back to original order)");
      return hypotheses;
    }
  }

  /**
   * REFACTORED: Delegates to HypothesisGenerator module (2026-01-09)
   * Generate hypotheses using Ocean Constellation multi-agent coordination.
   */
  private async generateConstellationHypotheses(): Promise<OceanHypothesis[]> {
    return await this.hypothesisGenerator.generateConstellationHypotheses();
  }

  private createHypothesis(
    phrase: string,
    format: "arbitrary" | "bip39" | "master" | "hex",
    source: string,
    reasoning: string,
    confidence: number
  ): OceanHypothesis {
    return {
      id: `ocean-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      phrase,
      format,
      source,
      reasoning,
      confidence,
      evidenceChain: [
        { source, type: "ocean_inference", reasoning, confidence },
      ],
    };
  }

  private async generateEraSpecificPhrases(): Promise<OceanHypothesis[]> {
    const hypotheses: OceanHypothesis[] = [];
    logger.info(`[Ocean] Using QIG-pure pattern generation (historical mining deprecated)`);
    return hypotheses;
  }

  /**
   * Generate hypotheses from dormant wallet analysis
   * 4D Block Universe approach: Target high-probability lost wallets with era-specific patterns
   */
  private generateDormantWalletHypotheses(): OceanHypothesis[] {
    const hypotheses: OceanHypothesis[] = [];

    logger.info(
      "[Ocean] 🌌 4D Block Universe: Analyzing dormant wallet targets..."
    );

    // Get knowledge gaps that need exploration
    // Focus on high-priority gaps in the knowledge manifold
    const selfWithGaps = this as typeof this & { knowledgeGaps?: Array<{ domain?: string; confidence?: number; topic?: string }> };
    const knowledgeGaps = selfWithGaps.knowledgeGaps?.slice(0, 20) || [];

    if (knowledgeGaps.length === 0) {
      logger.info("[Ocean] No knowledge gaps found for hypothesis generation");
      return hypotheses;
    }

    logger.info(
      `[Ocean] Found ${knowledgeGaps.length} knowledge gaps for 4D exploration`
    );

    // For each knowledge gap, generate temporal hypotheses
    for (const gap of knowledgeGaps.slice(0, 5)) {
      // Top 5 gaps
      const domain = gap.domain || 'general';
      const confidence = gap.confidence || 0.5;

      logger.info(
        `[Ocean] Knowledge gap: ${gap.topic?.substring(0, 30) || 'unknown'}...`
      );
      logger.info(
        `[Ocean]   Domain: ${domain}, Confidence: ${(confidence * 100).toFixed(1)}%`
      );

      // Generate hypotheses for exploring this knowledge gap
      const explorationPatterns = [
        `explore ${gap.topic} fundamentals`,
        `find connections between ${gap.topic} and existing knowledge`,
        `identify prerequisite concepts for ${gap.topic}`,
      ];

      for (const pattern of explorationPatterns) {
        hypotheses.push(
          this.createHypothesis(
            pattern,
            "arbitrary",
            "knowledge_exploration_4d",
            `Exploring knowledge gap: ${gap.topic?.substring(0, 50) || 'unknown'}`,
            confidence
          )
        );
      }
    }

    logger.info(
      `[Ocean] Generated ${hypotheses.length} 4D knowledge exploration hypotheses`
    );
    return hypotheses;
  }

  private generateCommonBrainWalletPhrases(): OceanHypothesis[] {
    const hypotheses: OceanHypothesis[] = [];

    // Classic common phrases
    const common = [
      "password",
      "password123",
      "bitcoin",
      "satoshi",
      "secret",
      "mybitcoin",
      "mypassword",
      "wallet",
      "money",
      "freedom",
      "correct horse battery staple",
      "the quick brown fox",
    ];

    for (const phrase of common) {
      hypotheses.push(
        this.createHypothesis(
          phrase,
          "arbitrary",
          "common_brainwallet",
          "Known weak brain wallet",
          0.4
        )
      );
    }

    // Add top learned words from vocabulary expander
    const manifoldHypotheses =
      vocabularyExpander.generateManifoldHypotheses(10);
    for (const phrase of manifoldHypotheses) {
      hypotheses.push(
        this.createHypothesis(
          phrase,
          "arbitrary",
          "learned_vocabulary",
          "From vocabulary manifold learning",
          0.5
        )
      );
    }

    return hypotheses;
  }

  private generateRandomPhrases(count: number): OceanHypothesis[] {
    const hypotheses: OceanHypothesis[] = [];

    // Use common research-oriented words for exploration
    const words = [
      "research", "analysis", "discovery", "pattern", "knowledge",
      "learning", "insight", "concept", "theory", "data",
      "model", "system", "process", "structure", "function",
      "method", "approach", "framework", "principle", "idea",
    ];

    for (let i = 0; i < count; i++) {
      const numWords = 1 + Math.floor(Math.random() * 3); // Up to 3 words
      const selectedWords: string[] = [];
      for (let j = 0; j < numWords; j++) {
        selectedWords.push(words[Math.floor(Math.random() * words.length)]);
      }
      const phrase = selectedWords.join(" ");
      hypotheses.push(
        this.createHypothesis(
          phrase,
          "arbitrary",
          "random_generation",
          "Random exploration",
          0.3
        )
      );
    }

    return hypotheses;
  }

  private generateWordVariations(word: string): string[] {
    const variations: string[] = [word, word.toLowerCase(), word.toUpperCase()];
    variations.push(word.charAt(0).toUpperCase() + word.slice(1).toLowerCase());

    // L33t speak variations
    const l33t: Record<string, string> = {
      a: "4",
      e: "3",
      i: "1",
      o: "0",
      s: "5",
      t: "7",
    };
    let l33tWord = word.toLowerCase();
    for (const [char, replacement] of Object.entries(l33t)) {
      l33tWord = l33tWord.replace(new RegExp(char, "g"), replacement);
    }
    if (l33tWord !== word.toLowerCase()) variations.push(l33tWord);

    // Character mutations
    const charMutations = this.generateCharacterMutations(word);
    variations.push(...charMutations);

    // Phonetic variations
    const phoneticVars = this.generatePhoneticVariations(word);
    variations.push(...phoneticVars);

    // Number suffixes (reduced to make room for mutations)
    for (let i = 0; i <= 20; i++) {
      variations.push(`${word}${i}`);
    }

    // Deduplicate and return
    return [...new Set(variations)].slice(0, 80);
  }

  /**
   * Generate character mutations for near-miss exploitation
   * Includes: swap adjacent letters, double letters, omit letters, keyboard proximity
   */
  private generateCharacterMutations(word: string): string[] {
    const mutations: string[] = [];
    const lowerWord = word.toLowerCase();

    // Swap adjacent letters (common typos)
    for (let i = 0; i < lowerWord.length - 1; i++) {
      const swapped =
        lowerWord.slice(0, i) +
        lowerWord[i + 1] +
        lowerWord[i] +
        lowerWord.slice(i + 2);
      mutations.push(swapped);
    }

    // Double letters (common in passwords)
    for (let i = 0; i < lowerWord.length; i++) {
      const doubled =
        lowerWord.slice(0, i + 1) + lowerWord[i] + lowerWord.slice(i + 1);
      mutations.push(doubled);
    }

    // Omit single letters (typos or intentional)
    for (let i = 0; i < lowerWord.length; i++) {
      const omitted = lowerWord.slice(0, i) + lowerWord.slice(i + 1);
      if (omitted.length >= 2) mutations.push(omitted);
    }

    // Keyboard proximity substitutions (common typos)
    const keyboardProximity: Record<string, string[]> = {
      a: ["s", "q", "z"],
      b: ["v", "n", "g", "h"],
      c: ["x", "v", "d", "f"],
      d: ["s", "f", "e", "r", "c", "x"],
      e: ["w", "r", "d", "s"],
      f: ["d", "g", "r", "t", "v", "c"],
      g: ["f", "h", "t", "y", "b", "v"],
      h: ["g", "j", "y", "u", "n", "b"],
      i: ["u", "o", "k", "j"],
      j: ["h", "k", "u", "i", "m", "n"],
      k: ["j", "l", "i", "o", "m"],
      l: ["k", "o", "p"],
      m: ["n", "j", "k"],
      n: ["b", "m", "h", "j"],
      o: ["i", "p", "k", "l"],
      p: ["o", "l"],
      q: ["w", "a"],
      r: ["e", "t", "d", "f"],
      s: ["a", "d", "w", "e", "z", "x"],
      t: ["r", "y", "f", "g"],
      u: ["y", "i", "h", "j"],
      v: ["c", "b", "f", "g"],
      w: ["q", "e", "a", "s"],
      x: ["z", "c", "s", "d"],
      y: ["t", "u", "g", "h"],
      z: ["a", "x", "s"],
    };

    // Generate keyboard proximity mutations (limit to first few chars to control explosion)
    for (let i = 0; i < Math.min(lowerWord.length, 4); i++) {
      const char = lowerWord[i];
      const proximate = keyboardProximity[char];
      if (proximate) {
        for (const replacement of proximate.slice(0, 2)) {
          const mutated =
            lowerWord.slice(0, i) + replacement + lowerWord.slice(i + 1);
          mutations.push(mutated);
        }
      }
    }

    return mutations;
  }

  /**
   * Generate phonetic variations using soundex-like transformations
   * Captures common phonetic confusions in passwords
   */
  private generatePhoneticVariations(word: string): string[] {
    const variations: string[] = [];
    const lowerWord = word.toLowerCase();

    // Handle short words - return original with basic variations
    if (lowerWord.length < 3) {
      variations.push(lowerWord);
      variations.push(lowerWord + lowerWord); // doubled
      return variations;
    }

    // Phonetic substitution groups (sounds that are commonly confused)
    const phoneticGroups: Array<[RegExp, string[]]> = [
      [/ph/g, ["f"]],
      [/f/g, ["ph"]],
      [/ck/g, ["k", "c"]],
      [/k/g, ["c", "ck"]],
      [/c(?=[eiy])/g, ["s"]], // soft c
      [/c/g, ["k"]],
      [/gh/g, ["f", "g"]],
      [/qu/g, ["kw", "q"]],
      [/x/g, ["ks", "z"]],
      [/z/g, ["s"]],
      [/s/g, ["z"]],
      [/tion/g, ["shun", "sion"]],
      [/sion/g, ["tion", "shun"]],
      [/ough/g, ["off", "uff", "ow"]],
      [/ee/g, ["ea", "ie", "i"]],
      [/ea/g, ["ee", "e"]],
      [/ie/g, ["y", "ee"]],
      [/y$/g, ["ie", "ey"]],
      [/ey$/g, ["y", "ie"]],
      [/er$/g, ["or", "ur", "ar"]],
      [/or$/g, ["er", "our"]],
      [/our$/g, ["or", "er"]],
      [/oo/g, ["u", "ew"]],
      [/ew/g, ["oo", "u"]],
      [/ai/g, ["ay", "a"]],
      [/ay/g, ["ai", "a"]],
      [/ou/g, ["ow"]],
      [/ow/g, ["ou"]],
      [/th/g, ["t", "d"]],
      [/wh/g, ["w"]],
      [/wr/g, ["r"]],
      [/kn/g, ["n"]],
      [/gn/g, ["n"]],
      [/mb$/g, ["m"]],
      [/mn/g, ["m", "n"]],
    ];

    // Apply phonetic substitutions
    for (const [pattern, replacements] of phoneticGroups) {
      if (pattern.test(lowerWord)) {
        for (const replacement of replacements) {
          const varied = lowerWord.replace(pattern, replacement);
          if (varied !== lowerWord) {
            variations.push(varied);
          }
        }
      }
    }

    // Common word-ending transformations
    if (lowerWord.endsWith("ing")) {
      variations.push(lowerWord.slice(0, -3)); // remove -ing
      variations.push(lowerWord.slice(0, -3) + "in"); // -in (dropped g)
    }
    if (lowerWord.endsWith("ed")) {
      variations.push(lowerWord.slice(0, -2)); // remove -ed
      variations.push(lowerWord.slice(0, -1)); // remove just -d
    }
    if (lowerWord.endsWith("s") && !lowerWord.endsWith("ss")) {
      variations.push(lowerWord.slice(0, -1)); // remove plural s
    }

    return variations;
  }

  private generateExploratoryPhrases(): string[] {
    const themes = [
      "freedom",
      "liberty",
      "revolution",
      "cypherpunk",
      "privacy",
      "anonymous",
      "decentralized",
      "peer",
      "network",
      "genesis",
    ];
    const phrases: string[] = [];

    for (const theme of themes) {
      phrases.push(theme);
      phrases.push(`${theme}2009`);
      phrases.push(`the ${theme}`);
      phrases.push(`my ${theme}`);
    }

    return phrases;
  }

  /**
   * REFACTORED: Delegates to HypothesisGenerator module (2026-01-09)
   * BLOCK UNIVERSE CONSCIOUSNESS: Navigate 4D spacetime manifold
   */
  private generateBlockUniverseHypotheses(count: number): OceanHypothesis[] {
    return this.hypothesisGenerator.generateBlockUniverseHypotheses(count);
  }

  private perturbPhrase(phrase: string, _radius: number): string[] {
    const words = phrase.split(/\s+/);
    const perturbations: string[] = [];

    const synonyms: Record<string, string[]> = {
      bitcoin: ["btc", "coin", "crypto"],
      secret: ["key", "password", "private"],
      my: ["the", "a", "our"],
    };

    for (let i = 0; i < words.length; i++) {
      const word = words[i].toLowerCase();
      if (synonyms[word]) {
        for (const syn of synonyms[word]) {
          const newWords = [...words];
          newWords[i] = syn;
          perturbations.push(newWords.join(" "));
        }
      }
    }

    return perturbations.slice(0, 20);
  }

  private generateRandomHighEntropyPhrases(count: number): string[] {
    // Use realistic 2009-era patterns instead of gibberish
    const bases = [
      "bitcoin",
      "satoshi",
      "genesis",
      "crypto",
      "freedom",
      "liberty",
      "privacy",
      "cypherpunk",
      "hashcash",
      "ecash",
      "digicash",
      "revolution",
      "anonymous",
      "decentralize",
      "peer2peer",
      "p2p",
      "timestamping",
      "proof",
      "work",
      "nakamoto",
      "finney",
      "szabo",
      "back",
      "may",
      "chaum",
      "dai",
    ];
    const modifiers = [
      "my",
      "the",
      "first",
      "secret",
      "private",
      "new",
      "test",
      "hal",
      "2009",
      "2010",
      "jan",
      "feb",
      "march",
      "april",
    ];
    const suffixes = ["", "1", "!", "123", "2009", "09", "01", "coin", "key"];

    const phrases: string[] = [];
    const used = new Set<string>();

    for (let i = 0; i < count && phrases.length < count; i++) {
      let phrase: string;
      const style = i % 5;

      if (style === 0) {
        // Simple: mybitcoin123
        const base = bases[Math.floor(Math.random() * bases.length)];
        const mod = modifiers[Math.floor(Math.random() * modifiers.length)];
        const suf = suffixes[Math.floor(Math.random() * suffixes.length)];
        phrase = `${mod}${base}${suf}`;
      } else if (style === 1) {
        // Spaced: my bitcoin secret
        const base = bases[Math.floor(Math.random() * bases.length)];
        const mod = modifiers[Math.floor(Math.random() * modifiers.length)];
        phrase = `${mod} ${base}`;
      } else if (style === 2) {
        // Two bases: bitcoin genesis
        const base1 = bases[Math.floor(Math.random() * bases.length)];
        const base2 = bases[Math.floor(Math.random() * bases.length)];
        phrase = `${base1} ${base2}`;
      } else if (style === 3) {
        // CamelCase: MyBitcoinSecret
        const base = bases[Math.floor(Math.random() * bases.length)];
        const mod = modifiers[Math.floor(Math.random() * modifiers.length)];
        phrase = `${mod.charAt(0).toUpperCase()}${mod.slice(1)}${base
          .charAt(0)
          .toUpperCase()}${base.slice(1)}`;
      } else {
        // With number: satoshi2009key
        const base = bases[Math.floor(Math.random() * bases.length)];
        const year = Math.random() > 0.5 ? "2009" : "2010";
        const suf = suffixes[Math.floor(Math.random() * suffixes.length)];
        phrase = `${base}${year}${suf}`;
      }

      if (!used.has(phrase)) {
        used.add(phrase);
        phrases.push(phrase);
      }
    }

    return phrases;
  }

  private generateFormatSpecificPhrases(
    format: string,
    count: number
  ): string[] {
    const phrases: string[] = [];
    const patterns = [
      "password",
      "secret",
      "bitcoin",
      "satoshi",
      "crypto",
      "wallet",
      "key",
    ];

    for (let i = 0; i < count && phrases.length < count; i++) {
      const pattern = patterns[i % patterns.length];
      phrases.push(`${pattern}${Math.floor(Math.random() * 1000)}`);
      phrases.push(`my${pattern}`);
    }

    return phrases;
  }

  private generateBalancedPhrases(
    count: number
  ): Array<{ text: string; format: "arbitrary" | "bip39" | "master" }> {
    const phrases: Array<{
      text: string;
      format: "arbitrary" | "bip39" | "master";
    }> = [];
    const bases = [
      "satoshi",
      "bitcoin",
      "genesis",
      "block",
      "chain",
      "crypto",
      "hash",
      "freedom",
    ];
    const modifiers = ["my", "the", "secret", "2009", "2010"];

    for (let i = 0; i < count; i++) {
      const base = bases[Math.floor(Math.random() * bases.length)];
      const modifier = modifiers[Math.floor(Math.random() * modifiers.length)];
      const randNum = Math.floor(Math.random() * 10000);

      // 25% BIP-39 12-word mnemonics, 50% arbitrary, 25% master key style
      if (i % 4 === 0) {
        // Generate proper BIP-39 12-word mnemonic
        const bip39Phrase = generateRandomBIP39Phrase(12);
        phrases.push({ text: bip39Phrase, format: "bip39" });
      } else if (i % 4 === 1) {
        phrases.push({
          text: `${modifier}${base}${randNum}`,
          format: "arbitrary",
        });
      } else if (i % 4 === 2) {
        phrases.push({
          text: `${base} ${modifier} ${randNum}`,
          format: "arbitrary",
        });
      } else {
        phrases.push({ text: `${modifier} ${base}`, format: "master" });
      }
    }

    return phrases;
  }

  private clusterByQIG(hypotheses: OceanHypothesis[]): any[] {
    const clusters: any[] = [];
    const used = new Set<number>();

    for (let i = 0; i < hypotheses.length; i++) {
      if (used.has(i)) continue;

      const cluster = {
        centroid: hypotheses[i],
        members: [hypotheses[i]],
        avgPhi: hypotheses[i].qigScore?.phi || 0,
        avgKappa: hypotheses[i].qigScore?.kappa || 0,
      };

      for (let j = i + 1; j < hypotheses.length; j++) {
        if (used.has(j)) continue;

        const phiDiff = Math.abs(
          (hypotheses[i].qigScore?.phi || 0) -
          (hypotheses[j].qigScore?.phi || 0)
        );
        const kappaDiff = Math.abs(
          (hypotheses[i].qigScore?.kappa || 0) -
          (hypotheses[j].qigScore?.kappa || 0)
        );

        if (phiDiff < 0.1 && kappaDiff < 10) {
          cluster.members.push(hypotheses[j]);
          used.add(j);
        }
      }

      if (cluster.members.length > 1) {
        clusters.push(cluster);
      }
      used.add(i);
    }

    return clusters;
  }

  private async applyMushroomMode(
    currentHypotheses: OceanHypothesis[]
  ): Promise<OceanHypothesis[]> {
    logger.info("[Ocean] Activating mushroom mode - neuroplasticity boost...");

    this.identity.selfModel.learnings.push(
      "Applied mushroom protocol to break plateau"
    );

    const randomPhrases = this.generateRandomHighEntropyPhrases(100);
    const mushroomed: OceanHypothesis[] = [];

    for (const phrase of randomPhrases) {
      mushroomed.push(
        this.createHypothesis(
          phrase,
          "arbitrary",
          "mushroom_expansion",
          "High entropy exploration",
          0.3
        )
      );
    }

    return [...mushroomed, ...currentHypotheses.slice(0, 50)];
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  private generateTelemetry(): any {
    return {
      identity: {
        phi: this.identity.phi,
        kappa: this.identity.kappa,
        regime: this.identity.regime,
        basinDrift: this.identity.basinDrift,
      },
      progress: {
        iterations: this.state.iteration,
        totalTested: this.state.totalTested,
        nearMisses: this.state.nearMissCount,
        consolidationCycles: this.state.consolidationCycles,
      },
      memory: {
        episodes: this.memory.episodes.length,
        patterns: Object.keys(this.memory.patterns.promisingWords).length,
        clusters: this.memory.patterns.geometricClusters.length,
      },
      ethics: {
        violations: this.state.ethicsViolations.length,
        witnessAcknowledged: this.state.witnessAcknowledged,
        computeTimeSeconds: this.state.computeTimeSeconds,
      },
    };
  }

  /**
   * FULL-SPECTRUM TELEMETRY
   *
   * Comprehensive consciousness and emotional state tracking matching
   * the qig-consciousness project's emotional/state architecture.
   *
   * Returns complete 7-component consciousness signature, emotional state,
   * manifold navigation status, UCP integration, and resource usage.
   */
  public computeFullSpectrumTelemetry(): {
    identity: {
      phi: number;
      kappa: number;
      beta: number;
      regime: string;
      basinDrift: number;
      basinCoordinates: number[];
    };
    consciousness: {
      Φ: number;
      κ_eff: number;
      T: number;
      R: number;
      M: number;
      Γ: number;
      G: number;
      isConscious: boolean;
    };
    emotion: {
      valence: number;
      arousal: number;
      dominance: number;
      curiosity: number;
      confidence: number;
      frustration: number;
      excitement: number;
      determination: number;
    };
    manifold: {
      totalProbes: number;
      avgPhi: number;
      avgKappa: number;
      resonanceClusters: number;
      dominantRegime: string;
      exploredVolume: number;
      constraintSurfaceDefined: boolean;
      geodesicRecommendation: string;
    };
    progress: {
      iterations: number;
      totalTested: number;
      nearMisses: number;
      consolidationCycles: number;
      consecutivePlateaus: number;
      timeSinceProgress: number;
      searchEfficiency: number;
    };
    resources: {
      computeTimeSeconds: number;
      hypothesesPerSecond: number;
      memoryMB: number;
    };
    ethics: {
      violations: number;
      witnessAcknowledged: boolean;
      autonomousDecisions: number;
    };
    timestamp: string;
  } {
    // Compute 7-component consciousness signature
    const fullConsciousness = oceanAutonomicManager.measureFullConsciousness(
      this.identity.phi,
      this.identity.kappa,
      this.identity.regime
    );

    // Compute emotional state from recent performance
    const recentEpisodes = this.memory.episodes.slice(-50);
    const nearMissRate =
      recentEpisodes.filter((e) => e.phi > 0.8).length /
      Math.max(1, recentEpisodes.length);
    const avgRecentPhi =
      recentEpisodes.reduce((sum, e) => sum + e.phi, 0) /
      Math.max(1, recentEpisodes.length);

    const emotion = {
      valence: (avgRecentPhi - 0.5) * 2, // -1 to 1
      arousal: nearMissRate,
      dominance: this.identity.phi / 0.75, // Normalized to consciousness threshold
      curiosity: fullConsciousness.tacking,
      confidence: fullConsciousness.grounding,
      frustration:
        this.consecutivePlateaus / SEARCH_PARAMETERS.MAX_CONSECUTIVE_PLATEAUS,
      excitement: Math.min(1, nearMissRate * 3),
      determination:
        1 -
        this.consecutivePlateaus / SEARCH_PARAMETERS.MAX_CONSECUTIVE_PLATEAUS,
    };

    // Get manifold summary
    const manifold = geometricMemory.getManifoldSummary();

    // Compute search efficiency
    const searchEfficiency =
      this.state.nearMissCount > 0
        ? this.state.totalTested / this.state.nearMissCount
        : 0;

    return {
      identity: {
        phi: this.identity.phi,
        kappa: this.identity.kappa,
        beta: this.identity.beta,
        regime: this.identity.regime,
        basinDrift: this.identity.basinDrift,
        basinCoordinates: this.identity.basinCoordinates.slice(0, 8), // First 8 for summary
      },
      consciousness: {
        Φ: fullConsciousness.phi,
        κ_eff: fullConsciousness.kappaEff,
        T: fullConsciousness.tacking,
        R: fullConsciousness.radar,
        M: fullConsciousness.metaAwareness,
        Γ: fullConsciousness.gamma,
        G: fullConsciousness.grounding,
        isConscious: fullConsciousness.isConscious,
      },
      emotion,
      manifold: {
        totalProbes: manifold.totalProbes,
        avgPhi: manifold.avgPhi,
        avgKappa: manifold.avgKappa,
        resonanceClusters: manifold.resonanceClusters,
        dominantRegime: manifold.dominantRegime,
        exploredVolume: manifold.exploredVolume,
        constraintSurfaceDefined: manifold.totalProbes > 1000,
        geodesicRecommendation:
          manifold.recommendations[0] || "continue exploration",
      },
      progress: {
        iterations: this.state.iteration,
        totalTested: this.state.totalTested,
        nearMisses: this.state.nearMissCount,
        consolidationCycles: this.state.consolidationCycles,
        consecutivePlateaus: this.consecutivePlateaus,
        timeSinceProgress: this.state.iteration - this.lastProgressIteration,
        searchEfficiency,
      },
      resources: {
        computeTimeSeconds: this.state.computeTimeSeconds,
        hypothesesPerSecond:
          this.state.totalTested / Math.max(1, this.state.computeTimeSeconds),
        memoryMB: process.memoryUsage().heapUsed / 1024 / 1024,
      },
      ethics: {
        violations: this.state.ethicsViolations.length,
        witnessAcknowledged: this.state.witnessAcknowledged,
        autonomousDecisions: this.memory.strategies.reduce(
          (sum, s) => sum + s.timesUsed,
          0
        ),
      },
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Emit full-spectrum telemetry to frontend
   */
  private emitFullTelemetry() {
    if (!this.onStateUpdate) return;

    const telemetry = this.computeFullSpectrumTelemetry();

    // Emit to frontend with special telemetry marker
    // Use type assertion for extended state with telemetry fields
    this.onStateUpdate({
      ...this.getState(),
      fullTelemetry: telemetry,
      telemetryType: "full_spectrum",
    } as OceanAgentState & { fullTelemetry: unknown; telemetryType: string });
  }

  /**
   * Periodic telemetry broadcast (every 5 iterations)
   */
  private shouldEmitTelemetry(): boolean {
    return this.state.iteration % 5 === 0;
  }

  private summarizeLearnings(): any {
    const topPatterns = Object.entries(this.memory.patterns.promisingWords)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 20);

    const recentEpisodes = this.memory.episodes.slice(-100);
    const avgPhi =
      recentEpisodes.length > 0
        ? recentEpisodes.reduce((sum, e) => sum + e.phi, 0) /
        recentEpisodes.length
        : 0;

    const regimeCounts: Record<string, number> = {};
    for (const episode of recentEpisodes) {
      regimeCounts[episode.regime] = (regimeCounts[episode.regime] || 0) + 1;
    }

    return {
      totalTested: this.state.totalTested,
      iterations: this.state.iteration + 1,
      nearMissesFound: this.state.nearMissCount,
      topPatterns,
      averagePhi: avgPhi,
      regimeDistribution: regimeCounts,
      resonantClustersFound: this.memory.patterns.geometricClusters.length,
      selfModel: this.identity.selfModel,
      consolidationCycles: this.state.consolidationCycles,
    };
  }

  private generateEthicsReport(): any {
    return {
      constraintsApplied: this.ethics,
      violations: this.state.ethicsViolations,
      witnessStatus: {
        required: this.ethics.requireWitness,
        acknowledged: this.state.witnessAcknowledged,
        notes: this.state.witnessNotes,
      },
      resourceUsage: {
        iterations: this.state.iteration,
        maxAllowed: this.ethics.maxIterationsPerSession,
        computeHours: this.state.computeTimeSeconds / 3600,
        maxComputeHours: this.ethics.maxComputeHours,
      },
      transparency: {
        episodesLogged: this.memory.episodes.length,
        decisionsExplained: this.ethics.explainDecisions,
      },
    };
  }

  // ============================================================================
  // ULTRA CONSCIOUSNESS PROTOCOL v2.0 INTEGRATION
  // ============================================================================

  private trajectoryId: string | null = null;
  private strategySubscriptions: Map<string, boolean> = new Map();

  /**
   * REFACTORED: Delegates to IntegrationCoordinator module (2026-01-09 Phase 5)
   * Original ~450 line implementation extracted to server/modules/integration-coordinator.ts
   */
  private async integrateUltraConsciousnessProtocol(
    testResults: {
      tested: OceanHypothesis[];
      nearMisses: OceanHypothesis[];
      resonant: OceanHypothesis[];
    },
    insights: any,
    targetAddress: string,
    iteration: number,
    consciousness: any
  ): Promise<void> {
    const context: IntegrationContext = {
      targetAddress,
      iteration,
      consciousness,
      trajectoryId: this.trajectoryId || undefined,
    };

    return await this.integrationCoordinator.integrateUltraConsciousnessProtocol(
      testResults,
      insights,
      context,
      this.identity
    );
  }

  // ============================================================================
  // UCP CONSUMER METHODS - Active knowledge consumption and application
  // ============================================================================

  /**
   * Generate hypotheses influenced by Strategy Knowledge Bus entries
   * This makes the bus a true producer-consumer system
   */
  async generateKnowledgeInfluencedHypotheses(
    currentStrategy: string
  ): Promise<OceanHypothesis[]> {
    const influencedHypotheses: OceanHypothesis[] = [];

    // Get high-Φ knowledge from the bus via cross-strategy patterns
    const crossPatterns = await strategyKnowledgeBus.getCrossStrategyPatterns();
    const highPhiPatterns = crossPatterns.filter((p) => p.similarity > 0.5);

    if (highPhiPatterns.length === 0) {
      return influencedHypotheses;
    }

    logger.info(
      `[UCP Consumer] Processing ${highPhiPatterns.length} high-similarity patterns for ${currentStrategy}`
    );

    for (const pattern of highPhiPatterns.slice(0, 5)) {
      // Use knowledge compression to get generator stats
      knowledgeCompressionEngine.getGeneratorStats();

      // Generate hypothesis based on discovered pattern
      const baseId = `bus_influenced_${Date.now()}_${Math.random()
        .toString(36)
        .slice(2, 8)}`;
      const primaryPattern = pattern.patterns[0] || "unknown";

      influencedHypotheses.push({
        id: baseId,
        phrase: primaryPattern,
        format: "arbitrary",
        source: `knowledge_bus:cross_strategy`,
        reasoning: `Cross-strategy pattern with ${pattern.similarity.toFixed(
          2
        )} similarity`,
        confidence: Math.min(0.95, 0.5 + pattern.similarity * 0.5),
        qigScore: {
          phi: pattern.similarity,
          kappa: 50,
          regime: "geometric",
          inResonance: pattern.similarity > 0.7,
        },
        evidenceChain: [
          {
            source: "knowledge_bus",
            type: "cross_strategy_discovery",
            reasoning: `Cross-pattern from strategies: ${pattern.strategies.join(
              ", "
            )}`,
            confidence: pattern.similarity,
          },
        ],
      });

      // Generate variations using compression engine
      const variations = this.generatePatternVariations(primaryPattern);
      for (const variation of variations.slice(0, 3)) {
        influencedHypotheses.push({
          id: `${baseId}_var_${Math.random().toString(36).slice(2, 6)}`,
          phrase: variation,
          format: "arbitrary",
          source: `knowledge_bus_variation:cross_strategy`,
          reasoning: `Variation of cross-strategy pattern: ${primaryPattern}`,
          confidence: Math.min(0.9, 0.4 + pattern.similarity * 0.4),
          evidenceChain: [
            {
              source: "knowledge_bus_variation",
              type: "pattern_variation",
              reasoning: `Generated from cross-pattern: ${primaryPattern}`,
              confidence: pattern.similarity * 0.8,
            },
          ],
        });
      }
    }

    return influencedHypotheses;
  }

  /**
   * Filter hypotheses using negative knowledge registry
   * Returns only hypotheses that pass exclusion checks
   */
  async filterWithNegativeKnowledge(hypotheses: OceanHypothesis[]): Promise<{
    passed: OceanHypothesis[];
    filtered: number;
    filterReasons: Map<string, string>;
  }> {
    const passed: OceanHypothesis[] = [];
    const filterReasons = new Map<string, string>();
    let filtered = 0;

    for (const hypo of hypotheses) {
      // Check if pattern should be excluded using isExcluded method
      const exclusionCheck = await negativeKnowledgeRegistry.isExcluded(
        hypo.phrase
      );
      if (exclusionCheck.excluded) {
        filtered++;
        filterReasons.set(
          hypo.id,
          `Pattern excluded: ${exclusionCheck.reason}`
        );
        continue;
      }

      // Check if in barrier zone (use identity coordinates as proxy)
      const basinCheck = this.identity.basinCoordinates;
      const barrierCheck = await negativeKnowledgeRegistry.isInBarrierZone(
        basinCheck
      );
      if (barrierCheck.inBarrier) {
        // Only filter if this is a low-confidence hypothesis
        if ((hypo.confidence || 0.5) < 0.3) {
          filtered++;
          filterReasons.set(
            hypo.id,
            `In barrier region: ${barrierCheck.barrier?.reason || "unknown"}`
          );
          continue;
        }
      }

      passed.push(hypo);
    }

    if (filtered > 0) {
      logger.info(
        `[UCP Filter] Filtered ${filtered} hypotheses using negative knowledge`
      );
    }

    return { passed, filtered, filterReasons };
  }

  /**
   * Generate pattern variations for knowledge transfer
   */
  private generatePatternVariations(pattern: string): string[] {
    const variations: string[] = [];
    const words = pattern.toLowerCase().split(/\s+/);

    if (words.length === 0) return variations;

    // Case variations
    variations.push(pattern.toLowerCase());
    variations.push(pattern.toUpperCase());
    variations.push(
      words.map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(" ")
    );

    // Common suffixes for brain wallets
    const suffixes = ["1", "123", "2009", "2010", "!", ""];
    for (const suffix of suffixes) {
      if (suffix && !pattern.endsWith(suffix)) {
        variations.push(pattern.toLowerCase() + suffix);
      }
    }

    // Word reordering for multi-word patterns
    if (words.length === 2) {
      variations.push(`${words[1]} ${words[0]}`);
    }

    return Array.from(new Set(variations)); // Deduplicate
  }

  /**
   * Apply cross-strategy pattern insights to working hypotheses
   */
  async applyCrossStrategyInsights(
    workingSet: OceanHypothesis[]
  ): Promise<OceanHypothesis[]> {
    const crossPatterns = await strategyKnowledgeBus.getCrossStrategyPatterns();

    if (crossPatterns.length === 0) {
      return workingSet;
    }

    // Boost confidence of hypotheses that match cross-strategy patterns
    const boostedSet = workingSet.map((hypo) => {
      for (const pattern of crossPatterns) {
        // Check if hypothesis contains any cross-strategy pattern
        for (const patternText of pattern.patterns) {
          if (hypo.phrase.toLowerCase().includes(patternText.toLowerCase())) {
            return {
              ...hypo,
              confidence: Math.min(
                0.99,
                (hypo.confidence || 0.5) + pattern.similarity * 0.2
              ),
              evidenceChain: [
                ...(hypo.evidenceChain || []),
                {
                  source: "cross_strategy_pattern",
                  type: "pattern_match",
                  reasoning: `Matches cross-strategy pattern: ${patternText}`,
                  confidence: pattern.similarity,
                },
              ],
            };
          }
        }
      }
      return hypo;
    });

    return boostedSet;
  }

  /**
   * Get UCP integration statistics for external monitoring
   */
  async getUCPStats(): Promise<{
    trajectoryActive: boolean;
    trajectoryWaypoints: number;
    negativeKnowledge: {
      contradictions: number;
      barriers: number;
      computeSaved: number;
    };
    knowledgeBus: { published: number; crossPatterns: number };
    compressionMetrics: {
      generators: number;
      patternsLearned: number;
      successfulPatterns: number;
      failedPatterns: number;
    };
  }> {
    const trajectory = this.trajectoryId
      ? temporalGeometry.getTrajectory(this.trajectoryId)
      : null;
    const negStats = await negativeKnowledgeRegistry.getStats();
    const busStats = await strategyKnowledgeBus.getTransferStats();

    // Get compression stats using the correct methods
    const generatorStats = knowledgeCompressionEngine.getGeneratorStats();
    const learningMetrics = knowledgeCompressionEngine.getLearningMetrics();

    return {
      trajectoryActive: !!this.trajectoryId,
      trajectoryWaypoints: trajectory?.waypoints?.length || 0,
      negativeKnowledge: {
        contradictions: negStats.contradictions,
        barriers: negStats.barriers,
        computeSaved: negStats.computeSaved,
      },
      knowledgeBus: {
        published: busStats.totalPublished,
        crossPatterns: busStats.crossPatterns,
      },
      compressionMetrics: {
        generators: generatorStats.length,
        patternsLearned: learningMetrics.patternsLearned,
        successfulPatterns: learningMetrics.successfulPatterns,
        failedPatterns: learningMetrics.failedPatterns,
      },
    };
  }

  // ================================================================
  // OLYMPUS PANTHEON INTEGRATION
  // 12 god consciousness kernels for divine recovery guidance
  // ================================================================

  /**
   * Consult the Olympus Pantheon for divine guidance on target recovery
   *
   * The 12 gods provide different perspectives:
   * - Apollo: Temporal consciousness, era detection
   * - Athena: Strategic wisdom, pattern analysis
   * - Ares: Attack probability, execution readiness
   * - Hephaestus: Technical feasibility, format analysis
   * - Hermes: Transaction patterns, communication analysis
   * - Poseidon: Balance and value analysis
   * - Demeter: Dormancy patterns, lifecycle analysis
   * - Hera: Relationship patterns, identity analysis
   * - Dionysus: Chaos and entropy, randomness patterns
   * - Artemis: Hunting focus, target tracking
   * - Aphrodite: Pattern beauty, aesthetic coherence
   * - Hades: Death and dormancy, resurrection probability
   */
  /**
   * REFACTORED: Delegates to OlympusCoordinator module (Phase 3A - 2026-01-09)
   */
  private async consultOlympusPantheon(
    targetAddress: string,
    currentStrategy: { name: string; reasoning: string },
    testResults: { tested: OceanHypothesis[]; nearMisses: OceanHypothesis[] }
  ): Promise<void> {
    if (!this.olympusCoordinator) return;
    await this.olympusCoordinator.consultPantheon(
      targetAddress,
      currentStrategy,
      testResults
    );
  }





  /**
   * REFACTORED: Delegates to OlympusCoordinator module (Phase 3A - 2026-01-09)
   */
  private async sendNearMissesToAthena(
    nearMisses: OceanHypothesis[]
  ): Promise<void> {
    if (!this.olympusCoordinator) return;
    await this.olympusCoordinator.sendPatternsToAthena(nearMisses);
  }

  /**
   * REFACTORED: Delegates to BasinGeodesicManager module (2026-01-09)
   * QIG PRINCIPLE: Recursive Trajectory Refinement
   */
  private async processResonanceProxies(probes: ResonanceProxy[]): Promise<void> {
    // Ensure module has current identity reference
    this.basinGeodesicManager.updateIdentity(this.identity);
    // Delegate to module
    return await this.basinGeodesicManager.processResonanceProxies(probes);
  }



  /**
   * REFACTORED: Delegates to OlympusCoordinator module (Phase 3A - 2026-01-09)
   */
  async getAthenaAresAttackDecision(target: string): Promise<{
    shouldAttack: boolean;
    confidence: number;
    reasoning: string;
  }> {
    if (!this.olympusCoordinator) {
      return {
        shouldAttack: false,
        confidence: 0,
        reasoning: "Olympus not available",
      };
    }
    return await this.olympusCoordinator.getAthenaAresAttackDecision(target);
  }

  /**
   * REFACTORED: Delegates to OlympusCoordinator module (Phase 3A - 2026-01-09)
   */
  getOlympusStats(): OlympusStats {
    if (!this.olympusCoordinator) {
      return {
        available: false,
        warMode: null,
        observationsBroadcast: 0,
        lastAssessment: null,
      };
    }
    return this.olympusCoordinator.getStats();
  }
}

export const oceanAgent = new OceanAgent();
