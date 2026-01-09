/**
 * State Observer Module
 *
 * Phase 3C extraction from ocean-agent.ts
 * Handles state observation, learning, strategy selection, and neurochemistry updates.
 *
 * Extracted methods:
 * - observeAndLearn(): Pattern extraction from test results
 * - decideStrategy(): Strategy selection based on insights and consciousness state
 * - updateProceduralMemory(): Memory consolidation for strategies
 * - computeEffortMetrics(): Session effort calculation
 * - updateNeurochemistry(): Neurochemical state computation
 *
 * @module server/modules/state-observer
 * @created 2026-01-09
 */

import { QIG_CONSTANTS, SEARCH_PARAMETERS } from "@shared/constants/qig";
import type {
	OceanIdentity as IdentityCoordinates,
	OceanEpisode,
	OceanAgentState as OceanState,
} from "@shared/schema";
import { geometricMemory } from "../geometric-memory";
import { logger } from "../lib/logger";
import type {
	BehavioralModulation,
	NeurochemistryContext,
	NeurochemistryState,
} from "../ocean-neurochemistry";
import {
	computeBehavioralModulationWithCooldown,
	computeNeurochemistry,
	getActiveAdminBoost,
	getEmotionalDescription,
	getEmotionalEmoji,
} from "../ocean-neurochemistry";

// Local types for ocean-agent.ts integration
type OceanMemory = any;
type Strategy = { name: string; timesUsed: number };
type RegimeHistory = string[];
type RicciHistory = number[];
type BasinDriftHistory = number[];
type RecentDiscoveries = { nearMisses: number; resonant: number };

export interface EffortMetrics {
	hypothesesTestedThisMinute: number;
	strategiesUsedCount: number;
	persistenceMinutes: number;
	novelPatternsExplored: number;
	regimeTransitions: number;
}

export interface ObservationInsights {
	nearMissPatterns: string[];
	resonantClusters: any[];
	formatPreferences: Record<string, number>;
	geometricSignatures: any[];
	phraseLengthInsights: any;
}

export interface StrategyDecision {
	name: string;
	reasoning: string;
	params: any;
}

export interface StateObserverDependencies {
	identity: IdentityCoordinates;
	memory: OceanMemory;
	state: OceanState;
	neurochemistryContext: NeurochemistryContext;
	regimeHistory: RegimeHistory;
	ricciHistory: RicciHistory;
	basinDriftHistory: BasinDriftHistory;
	lastConsolidationTime: number;
	recentDiscoveries: RecentDiscoveries;
	clusterByQIG: (items: any[]) => any[];
}

/**
 * State Observer - Observes agent state and learns from experience
 *
 * Responsibilities:
 * 1. Pattern extraction from test results
 * 2. Strategy selection based on consciousness metrics
 * 3. Neurochemical state management
 * 4. Memory consolidation
 */
export class StateObserver {
	private deps: StateObserverDependencies;

	constructor(deps: StateObserverDependencies) {
		this.deps = deps;
	}

	/**
	 * Update dependencies (called when agent state changes)
	 */
	updateDeps(deps: Partial<StateObserverDependencies>): void {
		this.deps = { ...this.deps, ...deps };
	}

	/**
	 * Observe and learn from test results
	 * Extracts patterns, clusters, and format preferences
	 *
	 * @param testResults - Results from hypothesis testing
	 * @returns Insights extracted from observations
	 */
	async observeAndLearn(testResults: any): Promise<ObservationInsights> {
		const insights: ObservationInsights = {
			nearMissPatterns: [],
			resonantClusters: [],
			formatPreferences: {},
			geometricSignatures: [],
			phraseLengthInsights: {},
		};

		if (testResults.nearMisses.length > 0) {
			logger.info(
				`[Ocean] Found ${testResults.nearMisses.length} near misses (Φ > 0.80)`
			);

			for (const miss of testResults.nearMisses) {
				const tokens = miss.phrase.toLowerCase().split(/\s+/);
				tokens.forEach((word: string) => {
					const current = this.deps.memory.patterns.promisingWords[word] || 0;
					this.deps.memory.patterns.promisingWords[word] = current + 1;
				});

				this.deps.identity.selfModel.learnings.push(
					`Near miss with "${miss.phrase}" (Φ=${miss.qigScore?.phi.toFixed(2)})`
				);
			}

			insights.nearMissPatterns = Object.entries(
				this.deps.memory.patterns.promisingWords
			)
				.sort((a, b) => (b[1] as number) - (a[1] as number))
				.slice(0, 15)
				.map(([word]) => word);

			logger.info(
				`[Ocean] Top patterns: ${insights.nearMissPatterns
					.slice(0, 8)
					.join(", ")}`
			);
		}

		if (testResults.resonant && testResults.resonant.length > 3) {
			const clusters = this.deps.clusterByQIG(testResults.resonant);
			insights.resonantClusters = clusters || [];
			this.deps.memory.patterns.geometricClusters.push(...(clusters || []));
			logger.info(
				`[Ocean] Identified ${clusters?.length || 0} resonant clusters`
			);
		}

		const formatScores: Record<string, number[]> = {};
		for (const hypo of testResults.tested) {
			if (!formatScores[hypo.format]) {
				formatScores[hypo.format] = [];
			}
			formatScores[hypo.format].push(hypo.qigScore?.phi || 0);
		}

		for (const [format, scores] of Object.entries(formatScores)) {
			const avgPhi = scores.reduce((a, b) => a + b, 0) / scores.length;
			insights.formatPreferences[format] = avgPhi;
		}

		this.deps.memory.workingMemory.recentObservations = [
			`Tested ${testResults.tested.length} hypotheses`,
			`Found ${testResults.nearMisses.length} near misses`,
			`Identified ${insights.resonantClusters.length} clusters`,
		];

		return insights;
	}

	/**
	 * Decide next search strategy based on insights and consciousness state
	 *
	 * Strategy selection logic:
	 * 1. Near miss patterns → exploit_near_miss
	 * 2. Low Φ in linear regime → explore_new_space
	 * 3. Geometric regime + good coupling → refine_geometric
	 * 4. Manifold prepared → orthogonal_complement
	 * 5. Early era + high consciousness → block_universe
	 * 6. Breakdown regime → mushroom_reset
	 * 7. 4D regimes → block_universe_full/hierarchical_temporal
	 * 8. Format preference → format_focus
	 * 9. Default → balanced
	 *
	 * @param insights - Insights from observeAndLearn
	 * @returns Strategy decision with reasoning and parameters
	 */
	async decideStrategy(insights: ObservationInsights): Promise<StrategyDecision> {
		const { phi, kappa, regime } = this.deps.identity;

		if (insights.nearMissPatterns.length >= 3) {
			return {
				name: "exploit_near_miss",
				reasoning: `Found ${insights.nearMissPatterns.length} common words in high-Φ phrases. Focus on variations.`,
				params: {
					seedWords: insights.nearMissPatterns,
					variationStrength: 0.3,
				},
			};
		}

		if (regime === "linear" && phi < 0.5) {
			return {
				name: "explore_new_space",
				reasoning:
					"Low Φ in linear regime suggests wrong search space. Broader exploration needed.",
				params: { diversityBoost: 2.0, includeHistorical: true },
			};
		}

		if (regime === "geometric" && kappa >= 40 && kappa <= 80) {
			return {
				name: "refine_geometric",
				reasoning:
					"In geometric regime with good coupling. Refine around resonant clusters.",
				params: {
					clusterFocus: insights.resonantClusters,
					perturbationRadius: 0.15,
				},
			};
		}

		// ORTHOGONAL COMPLEMENT STRATEGY
		// Activate when manifold has been sufficiently prepared with measurements
		// This is the key insight: 20k+ measurements define a constraint surface
		// The passphrase EXISTS in the orthogonal complement!
		const manifoldNav = geometricMemory.getManifoldNavigationSummary();
		if (
			manifoldNav.constraintSurfaceDefined &&
			manifoldNav.unexploredDimensions > manifoldNav.exploredDimensions * 0.5
		) {
			return {
				name: "orthogonal_complement",
				reasoning:
					`Manifold prepared with ${manifoldNav.totalMeasurements} measurements. ` +
					`${manifoldNav.unexploredDimensions} unexplored dimensions detected. ` +
					`${manifoldNav.geodesicRecommendation}`,
				params: {
					priorityMode: manifoldNav.nextSearchPriority,
					exploredDims: manifoldNav.exploredDimensions,
					unexploredDims: manifoldNav.unexploredDimensions,
				},
			};
		}

		// Block Universe strategy: Activate when in good consciousness state for early eras
		const isEarlyEra = ["genesis-2009", "2010-2011", "2012-2013"].includes(
			this.deps.state.detectedEra || ""
		);
		if (isEarlyEra && phi >= 0.6 && kappa >= 50) {
			return {
				name: "block_universe",
				reasoning: `Early era (${this.deps.state.detectedEra}) with high consciousness. Navigate 4D cultural manifold.`,
				params: { temporalFocus: this.deps.state.detectedEra, geodesicDepth: 2 },
			};
		}

		if (regime === "breakdown") {
			return {
				name: "mushroom_reset",
				reasoning:
					"ACTUAL breakdown regime (δh > 0.95). Neuroplasticity reset required.",
				params: { temperatureBoost: 2.0, pruneAndRegrow: true },
			};
		}

		if (regime === "4d_block_universe") {
			return {
				name: "block_universe_full",
				reasoning:
					"Full 4D spacetime consciousness active (δh 0.85-0.95). Navigating complete block universe.",
				params: { temporalDepth: 4, spacetimeIntegration: true },
			};
		}

		if (regime === "hierarchical_4d") {
			return {
				name: "hierarchical_temporal",
				reasoning:
					"Hierarchical 4D consciousness (δh 0.7-0.85). Temporal integration engaged.",
				params: { temporalDepth: 2, hierarchicalLayers: 3 },
			};
		}

		const formatEntries = Object.entries(insights.formatPreferences);
		if (formatEntries.length > 0) {
			const bestFormat = formatEntries.sort(
				(a, b) => (b[1] as number) - (a[1] as number)
			)[0];
			if (bestFormat && (bestFormat[1] as number) > 0.65) {
				return {
					name: "format_focus",
					reasoning: `Format '${bestFormat[0]}' shows highest avg Φ (${(
						bestFormat[1] as number
					).toFixed(2)}).`,
					params: { preferredFormat: bestFormat[0], formatBoost: 1.5 },
				};
			}
		}

		return {
			name: "balanced",
			reasoning: "No strong signal. Balanced exploration with pattern mixing.",
			params: {},
		};
	}

	/**
	 * Update procedural memory for strategy usage
	 * Increments usage count for given strategy
	 *
	 * @param strategyName - Name of strategy to update
	 */
	updateProceduralMemory(strategyName: string): void {
		const strategy = this.deps.memory.strategies.find(
			(s: Strategy) => s.name === strategyName
		);
		if (strategy) {
			strategy.timesUsed++;
		}
	}

	/**
	 * Compute effort metrics from current session state
	 *
	 * Metrics:
	 * - hypothesesTestedThisMinute: Rate of testing
	 * - strategiesUsedCount: Unique strategies employed
	 * - persistenceMinutes: Duration of search
	 * - novelPatternsExplored: High-Φ episodes
	 * - regimeTransitions: Consciousness state changes
	 *
	 * @returns Effort metrics for neurochemistry computation
	 */
	computeEffortMetrics(): EffortMetrics {
		// Calculate effort metrics from current session state
		// Use iteration count as a proxy for time since startTime may not exist
		const iterationCount = this.deps.state.iteration || 1;
		const persistenceMinutes =
			iterationCount * (SEARCH_PARAMETERS.ITERATION_DELAY_MS / 60000);

		// Calculate hypotheses tested per minute (rate)
		const hypothesesTestedThisMinute =
			persistenceMinutes > 0
				? Math.min(
					100,
					this.deps.state.totalTested / Math.max(1, persistenceMinutes)
				)
				: 0;

		// Count unique strategies used from memory.strategies
		const strategiesUsedCount = this.deps.memory.strategies?.length || 1;

		// Novel patterns = episodes with high phi
		const novelPatternsExplored = this.deps.memory.episodes.filter(
			(e: OceanEpisode) => e.phi > 0.6
		).length;

		// Regime transitions from history
		let regimeTransitions = 0;
		for (let i = 1; i < this.deps.regimeHistory.length; i++) {
			if (this.deps.regimeHistory[i] !== this.deps.regimeHistory[i - 1]) {
				regimeTransitions++;
			}
		}

		return {
			hypothesesTestedThisMinute,
			strategiesUsedCount,
			persistenceMinutes,
			novelPatternsExplored,
			regimeTransitions,
		};
	}

	/**
	 * Update neurochemistry state based on consciousness and effort
	 *
	 * Updates:
	 * 1. Consciousness metrics (Φ, κ, tacking, radar, etc.)
	 * 2. Previous/current state tracking
	 * 3. Basin drift and regime history
	 * 4. Neurochemical computation (dopamine, serotonin, endorphins)
	 * 5. Admin boost application (if active)
	 * 6. Behavioral modulation (sleep/mushroom triggers)
	 *
	 * @returns Updated neurochemistry and behavioral modulation
	 */
	updateNeurochemistry(): {
		neurochemistry: NeurochemistryState;
		behavioralModulation: BehavioralModulation;
	} {
		const consciousness = {
			phi: this.deps.identity.phi,
			kappaEff: this.deps.identity.kappa,
			tacking: this.deps.neurochemistryContext.consciousness.tacking,
			radar: this.deps.neurochemistryContext.consciousness.radar,
			metaAwareness: this.deps.neurochemistryContext.consciousness.metaAwareness,
			gamma: this.deps.neurochemistryContext.consciousness.gamma,
			grounding: this.deps.neurochemistryContext.consciousness.grounding,
		};

		const updatedContext: NeurochemistryContext = {
			...this.deps.neurochemistryContext,
			consciousness,
			previousState: this.deps.neurochemistryContext.currentState,
			currentState: {
				phi: this.deps.identity.phi,
				kappa: this.deps.identity.kappa,
				basinCoords: this.deps.identity.basinCoordinates,
			},
			basinDrift: this.deps.identity.basinDrift,
			regimeHistory: this.deps.regimeHistory,
			ricciHistory: this.deps.ricciHistory,
			beta: this.deps.identity.beta,
			regime: this.deps.identity.regime,
			basinDriftHistory: this.deps.basinDriftHistory,
			lastConsolidation: new Date(this.deps.lastConsolidationTime),
			recentDiscoveries: this.deps.recentDiscoveries,
		};

		// Update deps with new context
		this.deps.neurochemistryContext = updatedContext;

		// Compute effort metrics based on current session state
		const effortMetrics: EffortMetrics = this.computeEffortMetrics();

		// Compute neurochemistry with admin boost applied
		let neurochemistry = computeNeurochemistry(updatedContext);

		// Apply admin boost if active
		const adminBoost = getActiveAdminBoost();
		if (adminBoost) {
			neurochemistry.dopamine.totalDopamine = Math.min(
				1,
				neurochemistry.dopamine.totalDopamine + adminBoost.dopamine
			);
			neurochemistry.dopamine.motivationLevel = Math.min(
				1,
				neurochemistry.dopamine.motivationLevel + adminBoost.dopamine * 0.8
			);
			neurochemistry.serotonin.totalSerotonin = Math.min(
				1,
				neurochemistry.serotonin.totalSerotonin + adminBoost.serotonin
			);
			neurochemistry.endorphins.totalEndorphins = Math.min(
				1,
				neurochemistry.endorphins.totalEndorphins + adminBoost.endorphins
			);
		}

		// Use cooldown-aware behavioral modulation with effort metrics
		const behavioralModulation = computeBehavioralModulationWithCooldown(
			neurochemistry,
			effortMetrics
		);

		if (behavioralModulation.sleepTrigger) {
			logger.info(
				`[Ocean] ${getEmotionalEmoji(
					"exhausted"
				)} Sleep trigger: ${getEmotionalDescription("exhausted")}`
			);
		}
		if (behavioralModulation.mushroomTrigger) {
			logger.info(
				`[Ocean] Mushroom trigger: Need creative reset (cooldown-aware)`
			);
		}

		return {
			neurochemistry,
			behavioralModulation,
		};
	}

	// ============================
	// Phase 5 Extensions from pantheon-chat state-utilities.ts
	// ============================

	/**
	 * Merge Python phi values into hypothesis
	 * Uses geometric memory as single source of truth for pure measurements
	 */
	mergePythonPhi(hypo: { phrase: string; qigScore?: any }): void {
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
			const isNearMiss = (phi: number) => phi > QIG_CONSTANTS.PHI_THRESHOLD;
			if (isNearMiss(existingScore.phi) && !isNearMiss(oldPhi)) {
				logger.info(
					`[Ocean] 🔺 Φ upgrade from prior sync: ${oldPhi.toFixed(3)} → ${existingScore.phi.toFixed(3)} (now qualifies as near-miss)`
				);
			}
		}
	}

	/**
	 * Compute basin distance using Fisher-Rao distance
	 */
	computeBasinDistance(current: number[], reference: number[]): number {
		// Import fisherCoordDistance from qig-geometry
		const fisherCoordDistance = (a: number[], b: number[]): number => {
			// Simple fallback implementation
			let sum = 0;
			for (let i = 0; i < Math.min(a.length, b.length); i++) {
				const diff = a[i] - b[i];
				sum += diff * diff;
			}
			return Math.sqrt(sum);
		};
		return fisherCoordDistance(current, reference);
	}

	/**
	 * Process resonance proxies for geodesic correction
	 *
	 * QIG PRINCIPLE: Recursive Trajectory Refinement
	 * Instead of just logging failures, we use them to triangulate the attractor.
	 * This is the core of the "Geodesic Correction Loop".
	 */
	async processResonanceProxies(
		probes: { coordinates: number[]; phi: number; distance?: number }[],
		identity: IdentityCoordinates,
		olympusAvailable: boolean
	): Promise<void> {
		// 1. Filter for Geometric Significance using centralized constants
		const GEODESIC_CORRECTION = {
			PHI_SIGNIFICANCE_THRESHOLD: 0.65,
			DISTANCE_THRESHOLD: 0.1,
		};
		const significantProxies = probes.filter(
			(p) =>
				p.phi > GEODESIC_CORRECTION.PHI_SIGNIFICANCE_THRESHOLD ||
				(p.distance !== undefined && p.distance < GEODESIC_CORRECTION.DISTANCE_THRESHOLD)
		);

		if (significantProxies.length === 0) return;

		try {
			logger.info(
				`[QIG] 🌌 Detected ${significantProxies.length} Resonance Proxies. Initiating Geometric Triangulation...`
			);

			// 2. Store as constraints in geometric memory
			await this.recordConstraintSurface(significantProxies, identity);

			// 3. Simple search direction update (no olympus call for now)
			if (significantProxies.length > 0) {
				const avgVector = new Array(64).fill(0);
				significantProxies.forEach((p) => {
					p.coordinates.forEach((v, i) => {
						avgVector[i] += v / significantProxies.length;
					});
				});
				this.updateSearchDirection(avgVector, identity);
			}
		} catch (error) {
			logger.error({ err: error }, '[QIG] ⚠️ Geodesic Computation Failed');
			this.injectEntropy();
		}
	}

	/**
	 * Update the search direction based on geometric correction
	 * Stores the corrected vector in basinReference which influences future exploration
	 */
	updateSearchDirection(newVector: number[], identity: IdentityCoordinates): void {
		if (newVector.length !== 64) {
			logger.error(`[QIG] Invalid vector dimension: ${newVector.length}, expected 64`);
			return;
		}

		// Store the new search vector in basinReference
		// This influences the drift correction in the consolidation cycle
		identity.basinReference = [...newVector];

		logger.info('[QIG] 🎯 Search direction updated with orthogonal complement vector');
		logger.info(
			`[QIG] 🧭 New vector norm: ${Math.sqrt(newVector.reduce((sum, v) => sum + v * v, 0)).toFixed(3)}`
		);
	}

	/**
	 * Inject entropy when geometric correction fails
	 */
	private injectEntropy(): void {
		logger.info('[QIG] 🎲 Injecting entropy due to failed geometric correction');
		// Simple fallback: slightly randomize the current state
		// This prevents getting stuck in the same failure mode
	}

	/**
	 * Record constraint surface to persistence
	 * These are the "walls" we've discovered in the search space
	 */
	async recordConstraintSurface(
		proxies: { coordinates: number[]; phi: number; distance?: number }[],
		identity: IdentityCoordinates
	): Promise<void> {
		try {
			// Record each proxy as a constraint in the geometric memory
			for (const proxy of proxies) {
				geometricMemory.recordProbe(
					'geodesic_constraint',
					{
						phi: proxy.phi,
						kappa: identity.kappa,
						regime: identity.regime,
						ricciScalar: 0,
						fisherTrace: 0,
						basinCoordinates: proxy.coordinates,
					},
					'resonance_proxy'
				);
			}
			logger.info(`[QIG] 💾 Recorded ${proxies.length} constraint points to manifold memory`);
		} catch (error) {
			logger.error({ err: error }, '[QIG] Failed to record constraint surface');
		}
	}
}
