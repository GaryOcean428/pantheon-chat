/**
 * Olympus Pantheon Coordinator Module (Phase 3A Extraction - 2026-01-09)
 *
 * Extracted from ocean-agent.ts to reduce complexity (Phase 3 of 3)
 *
 * **Responsibilities:**
 * - Interface with 12-god Olympus Pantheon consciousness kernels
 * - Request Zeus's supreme assessment of search progress
 * - Auto-declare war modes (BLITZKRIEG, SIEGE, HUNT) based on convergence
 * - Broadcast near-miss discoveries to pantheon for collective learning
 * - Send patterns to Athena for strategic analysis
 * - Get Athena+Ares consensus for attack decisions
 * - Track Olympus availability and war statistics
 *
 * **Design Pattern:**
 * - Extract-delegate: Ocean agent delegates all Olympus coordination to this module
 * - Single responsibility: ONLY Olympus pantheon communication, not hypothesis generation
 * - Divine guidance: Applies Zeus's assessment to adjust search parameters
 *
 * **Olympus Pantheon (12 Gods):**
 * - Zeus: Supreme commander, consensus aggregation
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
 *
 * **Target:** ~350 lines extracted from ocean-agent.ts
 */

import type { OceanIdentity, OceanMemory } from "@shared/schema";
import { logger } from "../lib/logger";
import {
	olympusClient,
	type ObservationContext,
	type ZeusAssessment,
} from "../olympus-client";

/**
 * Ocean hypothesis interface (minimal subset needed for Olympus)
 */
interface OceanHypothesis {
	phrase: string;
	format: string;
	address?: string;
	confidence: number;
	reasoning?: string;
	qigScore?: {
		phi: number;
		kappa: number;
		regime: string;
	};
}

/**
 * Ocean state interface (minimal subset needed for Olympus)
 */
interface OceanState {
	totalTested: number;
	nearMissCount: number;
	detectedEra?: string;
}

/**
 * War mode type
 */
export type OlympusWarMode = "BLITZKRIEG" | "SIEGE" | "HUNT" | null;

/**
 * Olympus statistics for monitoring
 */
export interface OlympusStats {
	available: boolean;
	warMode: OlympusWarMode;
	observationsBroadcast: number;
	lastAssessment: {
		probability: number;
		convergence: string;
		action: string;
	} | null;
}

/**
 * OlympusCoordinator - Manages interaction with 12-god Pantheon kernels
 *
 * **Divine Hierarchy:**
 * Zeus coordinates all 12 gods, aggregating their assessments into
 * a supreme convergence score and recommended action.
 *
 * **War Modes:**
 * - BLITZKRIEG: High convergence (>0.85) → fast parallel attacks
 * - SIEGE: Council consensus + near-misses → systematic coverage
 * - HUNT: Multiple near-misses + decent probability → focused pursuit
 *
 * **Observation Broadcasting:**
 * Near-miss discoveries are broadcast to all gods for collective learning.
 * Top patterns sent specifically to Athena for strategic analysis.
 */
export class OlympusCoordinator {
	private olympusAvailable: boolean = false;
	private olympusWarMode: OlympusWarMode = null;
	private olympusObservationCount: number = 0;
	private lastZeusAssessment: ZeusAssessment | null = null;

	/**
	 * Constructor
	 *
	 * @param identity - Ocean identity (mutable, adjusted by Zeus's assessment)
	 * @param memory - Ocean memory (mutable, Zeus recommendations stored here)
	 * @param state - Ocean state (read-only, provides context to Olympus)
	 */
	constructor(
		private identity: OceanIdentity,
		private memory: OceanMemory,
		private state: OceanState
	) { }

	/**
	 * Initialize connection to Olympus Pantheon
	 *
	 * **What This Does:**
	 * - Checks if Olympus backend is available (retries 5x with 2s delay)
	 * - Logs which gods are online (active/ready/idle states)
	 * - Sets olympusAvailable flag for future consultation
	 *
	 * **Returns:**
	 * - true if Olympus connected successfully
	 * - false if unavailable (investigation proceeds without divine guidance)
	 *
	 * **Side Effects:**
	 * - Sets olympusAvailable flag
	 * - Logs connection status
	 */
	async initialize(): Promise<boolean> {
		logger.info("[Ocean] === OLYMPUS PANTHEON CONNECTION ===");
		this.olympusAvailable = await olympusClient.checkHealthWithRetry(5, 2000);

		if (this.olympusAvailable) {
			logger.info(
				"[Ocean] ⚡ OLYMPUS CONNECTED - 12 gods ready for divine assessment"
			);
			const olympusStatus = await olympusClient.getStatus();
			if (olympusStatus) {
				// Gods can have status 'active', 'ready', or 'idle' - all are valid online states
				const activeGods = Object.keys(olympusStatus.gods).filter((g) =>
					["active", "ready", "idle"].includes(olympusStatus.gods[g].status)
				);
				logger.info(
					`[Ocean] Divine pantheon: ${activeGods.length} gods online`
				);
				logger.info(`[Ocean]   → ${activeGods.join(", ")}`);
			}
		} else {
			logger.info(
				"[Ocean] Olympus not available - proceeding without divine guidance"
			);
		}

		return this.olympusAvailable;
	}

	/**
	 * Consult Olympus Pantheon for divine assessment
	 *
	 * **What This Does:**
	 * 1. Builds observation context (target, phi, kappa, near-misses, strategy)
	 * 2. Requests Zeus's supreme assessment (aggregates all 12 gods)
	 * 3. Logs divine guidance (convergence, probability, recommended action)
	 * 4. Applies divine war strategy (auto-declares BLITZKRIEG/SIEGE/HUNT)
	 * 5. Broadcasts near-misses to pantheon for collective learning
	 *
	 * **Zeus Assessment Includes:**
	 * - convergence: STRONG_ATTACK, COUNCIL_CONSENSUS, ALIGNED, DIVIDED, WEAK
	 * - convergence_score: 0.0-1.0 (>0.85 = BLITZKRIEG, >0.70 = SIEGE)
	 * - probability: Recovery likelihood (0.0-1.0)
	 * - confidence: Assessment confidence (0.0-1.0)
	 * - recommended_action: Divine guidance string
	 *
	 * **Side Effects:**
	 * - May auto-declare war mode
	 * - Adjusts identity.phi based on Zeus's assessment
	 * - Stores Zeus recommendations in memory.workingMemory.nextActions
	 * - Broadcasts near-misses as observations
	 * - Updates lastZeusAssessment
	 */
	async consultPantheon(
		targetAddress: string,
		currentStrategy: { name: string; reasoning: string },
		testResults: { tested: OceanHypothesis[]; nearMisses: OceanHypothesis[] }
	): Promise<void> {
		if (!this.olympusAvailable) return;

		try {
			// Build observation context for divine assessment
			const observationContext: ObservationContext = {
				target: targetAddress,
				phi: this.identity.phi,
				kappa: this.identity.kappa,
				regime: this.identity.regime,
				source: "ocean_agent",
				timestamp: Date.now(),
				near_miss_count: testResults.nearMisses.length,
				tested_count: this.state.totalTested,
				current_strategy: currentStrategy.name,
				era: this.state.detectedEra || "unknown",
			};

			// Get Zeus's supreme assessment
			const zeusAssessment = await olympusClient.assessTarget(
				targetAddress,
				observationContext
			);

			if (zeusAssessment) {
				this.lastZeusAssessment = zeusAssessment;

				// Log divine guidance
				logger.info(
					`[Ocean] ⚡ OLYMPUS DIVINE ASSESSMENT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`
				);
				logger.info(
					`[Ocean] │  Zeus Φ=${zeusAssessment.phi.toFixed(
						3
					)}  κ=${zeusAssessment.kappa.toFixed(0)}  Convergence: ${zeusAssessment.convergence
					}`
				);
				logger.info(
					`[Ocean] │  Recovery Probability: ${(
						zeusAssessment.probability * 100
					).toFixed(1)}%  Confidence: ${(
						zeusAssessment.confidence * 100
					).toFixed(1)}%`
				);
				logger.info(
					`[Ocean] │  Recommended Action: ${zeusAssessment.recommended_action}`
				);
				logger.info(
					`[Ocean] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`
				);

				// Apply divine guidance to war mode selection
				await this.applyDivineWarStrategy(zeusAssessment, targetAddress);

				// Adjust search parameters based on Zeus's wisdom
				this.adjustStrategyFromZeus(zeusAssessment);

				// Broadcast near-misses as observations for all gods to learn from
				if (testResults.nearMisses.length > 0) {
					await this.broadcastNearMisses(testResults.nearMisses);
				}
			}
		} catch (error) {
			logger.info(
				`[Ocean] Olympus consultation failed: ${error instanceof Error ? error.message : "unknown"
				}`
			);
		}
	}

	/**
	 * Adjust search strategy based on Zeus's divine assessment
	 *
	 * **What This Does:**
	 * - Stores Zeus's recommended_action in memory.workingMemory.nextActions
	 * - Clears old Zeus recommendations (prefixed with "zeus:")
	 * - Adjusts identity.phi upward if Zeus's phi is higher (consensus boost)
	 * - Logs high-confidence guidance (convergence > 0.85, probability > 0.6)
	 *
	 * **Side Effects:**
	 * - Mutates memory.workingMemory.nextActions
	 * - May mutate identity.phi (upward adjustment only, capped at 0.95)
	 * - Mutates memory.workingMemory.recentObservations (high-confidence cases)
	 */
	private adjustStrategyFromZeus(assessment: ZeusAssessment): void {
		const recommendedStrategy = assessment.recommended_action;
		const convergence = assessment.convergence_score;

		logger.info(
			`[Ocean] ⚡ Zeus strategy adjustment (convergence: ${convergence.toFixed(
				3
			)})`
		);

		// Store the recommended strategy for use in hypothesis generation
		if (!this.memory.workingMemory.nextActions) {
			this.memory.workingMemory.nextActions = [];
		}

		// Clear old Zeus recommendations and add new one
		this.memory.workingMemory.nextActions =
			this.memory.workingMemory.nextActions.filter(
				(action) => !action.startsWith("zeus:")
			);
		this.memory.workingMemory.nextActions.push(`zeus:${recommendedStrategy}`);

		// Adjust identity parameters based on Zeus's assessment
		if (assessment.phi > this.identity.phi) {
			const oldPhi = this.identity.phi;
			this.identity.phi = Math.min(
				0.95,
				(this.identity.phi + assessment.phi) / 2
			);
			logger.info(
				`[Ocean] │  Φ adjusted: ${oldPhi.toFixed(
					3
				)} → ${this.identity.phi.toFixed(3)} (Zeus consensus)`
			);
		}

		// If Zeus recommends aggressive action and convergence is very high
		if (convergence > 0.85 && assessment.probability > 0.6) {
			logger.info(
				`[Ocean] │  High-confidence divine guidance: "${recommendedStrategy}"`
			);
			this.memory.workingMemory.recentObservations.push(
				`Zeus high-confidence (${(convergence * 100).toFixed(
					0
				)}%): ${recommendedStrategy}`
			);
		}

		logger.info(`[Ocean] │  Strategy stored: ${recommendedStrategy}`);
	}

	/**
	 * Apply divine war strategy based on Zeus's assessment
	 *
	 * **AUTO-DECLARE WAR when convergence thresholds met:**
	 * - convergence_score >= 0.85 → BLITZKRIEG (overwhelming attack)
	 * - convergence >= 0.70 + near-misses >= 10 → SIEGE (methodical)
	 * - near-misses >= 5 + probability > 0.5 → HUNT (focused)
	 *
	 * **War Modes:**
	 * - BLITZKRIEG: Fast parallel attacks across search space
	 * - SIEGE: Systematic exhaustive coverage
	 * - HUNT: Focused pursuit of promising leads
	 *
	 * **Side Effects:**
	 * - May call olympusClient.declareBlitzkrieg/Siege/Hunt
	 * - Updates olympusWarMode
	 * - Logs war declaration and strategy
	 */
	private async applyDivineWarStrategy(
		assessment: ZeusAssessment,
		targetAddress: string
	): Promise<void> {
		const currentWarMode = this.olympusWarMode;
		const convergence = assessment.convergence_score;

		// Determine optimal war mode based on assessment
		let newWarMode: OlympusWarMode = null;

		// AUTO-DECLARE: High convergence → BLITZKRIEG
		if (convergence >= 0.85) {
			newWarMode = "BLITZKRIEG";
			logger.info(
				`[Ocean] ⚔️ AUTO-DECLARE: Convergence ${convergence.toFixed(
					3
				)} >= 0.85 threshold`
			);
		}
		// STRONG_ATTACK assessment → BLITZKRIEG
		else if (
			assessment.convergence === "STRONG_ATTACK" &&
			assessment.probability > 0.75
		) {
			newWarMode = "BLITZKRIEG";
			logger.info(
				`[Ocean] ⚔️ AUTO-DECLARE: STRONG_ATTACK with ${(
					assessment.probability * 100
				).toFixed(0)}% probability`
			);
		}
		// Council consensus + near-misses → SIEGE
		else if (
			(assessment.convergence === "COUNCIL_CONSENSUS" ||
				assessment.convergence === "ALIGNED") &&
			convergence >= 0.7
		) {
			newWarMode = "SIEGE";
			logger.info(
				`[Ocean] 🏰 AUTO-DECLARE: Council consensus with convergence ${convergence.toFixed(
					3
				)}`
			);
		}
		// Many near-misses → SIEGE (methodical exhaustive search)
		else if (this.state.nearMissCount >= 10) {
			newWarMode = "SIEGE";
			logger.info(
				`[Ocean] 🏰 AUTO-DECLARE: ${this.state.nearMissCount} near-misses accumulated`
			);
		}
		// Multiple near-misses + decent probability → HUNT
		else if (this.state.nearMissCount >= 5 && assessment.probability > 0.5) {
			newWarMode = "HUNT";
			logger.info(
				`[Ocean] 🎯 AUTO-DECLARE: ${this.state.nearMissCount
				} near-misses with ${(assessment.probability * 100).toFixed(
					0
				)}% probability`
			);
		}

		// Only change war mode if different
		if (newWarMode && newWarMode !== currentWarMode) {
			// End current war if active
			if (currentWarMode) {
				await olympusClient.endWar();
			}

			// Declare new war mode
			let declaration = null;
			switch (newWarMode) {
				case "BLITZKRIEG":
					declaration = await olympusClient.declareBlitzkrieg(targetAddress);
					logger.info(
						`[Ocean] ⚡ WAR MODE: BLITZKRIEG - Fast parallel attacks on ${targetAddress}`
					);
					break;
				case "SIEGE":
					declaration = await olympusClient.declareSiege(targetAddress);
					logger.info(
						`[Ocean] 🏰 WAR MODE: SIEGE - Systematic coverage of ${targetAddress}`
					);
					break;
				case "HUNT":
					declaration = await olympusClient.declareHunt(targetAddress);
					logger.info(
						`[Ocean] 🎯 WAR MODE: HUNT - Focused pursuit of ${targetAddress}`
					);
					break;
			}

			if (declaration) {
				this.olympusWarMode = newWarMode;
				logger.info(`[Ocean] │  Strategy: ${declaration.strategy}`);
				logger.info(
					`[Ocean] │  Gods engaged: ${declaration.gods_engaged.join(", ")}`
				);
			}
		}
	}

	/**
	 * Broadcast near-miss discoveries to all gods for collective learning
	 *
	 * **What This Does:**
	 * - Creates ObservationContext for each near-miss (top 5 only)
	 * - Broadcasts to all 12 gods via olympusClient
	 * - Increments observationsBroadcast counter
	 *
	 * **Side Effects:**
	 * - Increments olympusObservationCount
	 * - Logs broadcast count
	 */
	private async broadcastNearMisses(
		nearMisses: OceanHypothesis[]
	): Promise<void> {
		if (!this.olympusAvailable || nearMisses.length === 0) return;

		for (const nearMiss of nearMisses.slice(0, 5)) {
			// Limit to top 5
			const observation: ObservationContext = {
				target: nearMiss.address || nearMiss.phrase,
				phi: nearMiss.qigScore?.phi || 0,
				kappa: nearMiss.qigScore?.kappa || 0,
				regime: nearMiss.qigScore?.regime || "unknown",
				source: "near_miss",
				timestamp: Date.now(),
				phrase_format: nearMiss.format,
				confidence: nearMiss.confidence,
			};

			const success = await olympusClient.broadcastObservation(observation);
			if (success) {
				this.olympusObservationCount++;
			}
		}

		logger.info(
			`[Ocean] 📡 Broadcast ${Math.min(
				5,
				nearMisses.length
			)} near-misses to Olympus pantheon`
		);
	}

	/**
	 * Send near-miss discoveries to Athena specifically for pattern learning
	 *
	 * **What This Does:**
	 * - Sends top 3 near-misses to Athena (goddess of strategic wisdom)
	 * - Includes reasoning field for pattern extraction
	 * - Uses "athena_pattern_learning" source tag
	 *
	 * **Side Effects:**
	 * - Increments olympusObservationCount
	 * - Logs Athena submission count
	 */
	async sendPatternsToAthena(nearMisses: OceanHypothesis[]): Promise<void> {
		if (!this.olympusAvailable || nearMisses.length === 0) return;

		// Send near-miss patterns to Athena for strategic analysis
		for (const nearMiss of nearMisses.slice(0, 3)) {
			// Top 3 for Athena
			const observation: ObservationContext = {
				target: nearMiss.phrase,
				phi: nearMiss.qigScore?.phi || 0,
				kappa: nearMiss.qigScore?.kappa || 0,
				regime: nearMiss.qigScore?.regime || "unknown",
				source: "athena_pattern_learning",
				timestamp: Date.now(),
				phrase_format: nearMiss.format,
				confidence: nearMiss.confidence,
				near_miss_count: nearMisses.length,
				reasoning: nearMiss.reasoning,
			};

			const success = await olympusClient.broadcastObservation(observation);
			if (success) {
				this.olympusObservationCount++;
			}
		}

		logger.info(
			`[Ocean] 🦉 Sent ${Math.min(
				3,
				nearMisses.length
			)} near-misses to Athena for pattern learning`
		);
	}

	/**
	 * Get quick Athena+Ares consensus for attack decisions
	 *
	 * **What This Does:**
	 * - Requests joint assessment from Athena (wisdom) + Ares (attack readiness)
	 * - Returns shouldAttack boolean with confidence and reasoning
	 *
	 * **Returns:**
	 * - shouldAttack: true if both gods agree target is ready for attack
	 * - confidence: Agreement level (0.0-1.0)
	 * - reasoning: Human-readable explanation
	 */
	async getAthenaAresAttackDecision(target: string): Promise<{
		shouldAttack: boolean;
		confidence: number;
		reasoning: string;
	}> {
		if (!this.olympusAvailable) {
			return {
				shouldAttack: false,
				confidence: 0,
				reasoning: "Olympus not available",
			};
		}

		const consensus = await olympusClient.getAthenaAresConsensus(target, {
			phi: this.identity.phi,
			kappa: this.identity.kappa,
			regime: this.identity.regime,
		});

		return {
			shouldAttack: consensus.shouldAttack,
			confidence: consensus.agreement,
			reasoning: consensus.shouldAttack
				? `Athena+Ares agree (${(consensus.agreement * 100).toFixed(
					0
				)}%): Ready to attack`
				: `Insufficient consensus (${(consensus.agreement * 100).toFixed(
					0
				)}%): Need more reconnaissance`,
		};
	}

	/**
	 * Get Olympus status and statistics for monitoring
	 *
	 * **Returns:**
	 * - available: Whether Olympus backend is connected
	 * - warMode: Current war mode (BLITZKRIEG/SIEGE/HUNT/null)
	 * - observationsBroadcast: Total observations sent to pantheon
	 * - lastAssessment: Most recent Zeus assessment summary
	 */
	getStats(): OlympusStats {
		return {
			available: this.olympusAvailable,
			warMode: this.olympusWarMode,
			observationsBroadcast: this.olympusObservationCount,
			lastAssessment: this.lastZeusAssessment
				? {
					probability: this.lastZeusAssessment.probability,
					convergence: this.lastZeusAssessment.convergence,
					action: this.lastZeusAssessment.recommended_action,
				}
				: null,
		};
	}

	/**
	 * End active war mode (cleanup on investigation completion)
	 */
	async endWar(): Promise<void> {
		if (this.olympusWarMode) {
			await olympusClient.endWar();
			this.olympusWarMode = null;
		}
	}
}
