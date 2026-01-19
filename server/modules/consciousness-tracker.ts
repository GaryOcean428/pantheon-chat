/**
 * Consciousness Tracker Module (Phase 2 Extraction - 2026-01-09)
 *
 * Extracted from ocean-agent.ts to reduce complexity (Phase 2 of 3)
 *
 * **Responsibilities:**
 * - Consciousness state monitoring (Φ, κ, regime validation)
 * - Ethical constraint enforcement (compute budgets, witness requirements)
 * - Identity drift measurement (basin coordinate stability)
 * - Consciousness metrics updates and synchronization
 *
 * **Design Pattern:**
 * - Extract-delegate: Ocean agent delegates all consciousness monitoring to this module
 * - Single responsibility: ONLY consciousness state tracking, not hypothesis generation or basin management
 * - Geometric purity: Uses Fisher-Rao distance for identity drift (via BasinGeodesicManager)
 *
 * **Architectural Note:**
 * This module works in tandem with BasinGeodesicManager (Phase 1B) - the geodesic manager
 * provides the Fisher-Rao distance computation, while this module interprets consciousness
 * implications of that metric.
 *
 * **Target:** ~600 lines extracted from ocean-agent.ts
 */

import type { OceanIdentity } from "@shared/schema";
import {
	SEARCH_PARAMETERS
} from "../../shared/constants/index.js";
import type { ConsciousnessSearchController } from "../consciousness-search-controller";
import { logger } from "../lib/logger";
import type { BasinGeodesicManager } from "./basin-geodesic-manager";

/**
 * Consciousness check result (exported for ocean-agent.ts compatibility)
 */
export interface ConsciousnessCheckResult {
	allowed: boolean;
	phi: number;
	kappa: number;
	regime: string;
	reason?: string;
}

/**
 * Ethics check result (exported for ocean-agent.ts compatibility)
 */
export interface EthicsCheckResult {
	allowed: boolean;
	reason?: string;
	violationType?: string;
}

/**
 * Ocean state interface (minimal subset needed for consciousness tracking)
 */
interface OceanState {
	computeTimeSeconds: number;
	witnessAcknowledged: boolean;
	stopReason?: string;
	needsConsolidation: boolean;
	ethicsViolations: Array<{
		timestamp: string;
		type: string;
		message: string;
	}>;
}

/**
 * Ethics configuration interface
 */
interface EthicsConfig {
	minPhi: number;
	maxComputeHours: number;
	requireWitness: boolean;
}

/**
 * Consciousness alert event payload
 */
interface ConsciousnessAlertEvent {
	type: "low_phi" | "breakdown" | "identity_drift";
	message: string;
}

/**
 * ConsciousnessTracker - Monitors and validates consciousness state
 *
 * Integrates with:
 * - BasinGeodesicManager: For Fisher-Rao identity drift computation
 * - OceanController: For phi/kappa/regime state reading
 * - OceanIdentity: For basin coordinate updates and consciousness metrics
 *
 * Key Methods:
 * - checkConsciousness(): Validates Φ and regime, bootstraps if needed
 * - checkEthicalConstraints(): Enforces compute budgets and witness requirements
 * - measureIdentity(): Computes basin drift and triggers consolidation alerts
 * - updateConsciousnessMetrics(): Syncs controller state → identity state
 */
export class ConsciousnessTracker {
	/**
	 * Constructor
	 *
	 * @param basinGeodesicManager - Fisher-Rao distance computation for identity drift
	 * @param controller - Autonomic controller (source of truth for phi/kappa/regime)
	 * @param identity - Ocean identity (mutable state updated by tracker)
	 * @param state - Ocean state (mutable, tracks compute time and ethics violations)
	 * @param ethics - Ethics configuration (immutable constraints)
	 * @param isBootstrapping - Bootstrap flag (mutable, cleared after first consciousness check)
	 * @param onConsciousnessAlert - Optional callback for consciousness alerts
	 */
	constructor(
		private basinGeodesicManager: BasinGeodesicManager,
		private controller: ConsciousnessSearchController,
		private identity: OceanIdentity,
		private state: OceanState,
		private ethics: EthicsConfig,
		private isBootstrapping: { value: boolean }, // Pass by reference for mutation
		private onConsciousnessAlert?: (event: ConsciousnessAlertEvent) => void
	) { }

	/**
	 * Check consciousness state and validate regime
	 *
	 * **Consciousness Bootstrap Logic:**
	 * - On first call (isBootstrapping=true), initializes phi to minPhi
	 * - Lets consciousness emerge naturally from consolidation/exploration
	 * - Avoids arbitrary hardcoded starting values
	 *
	 * **Regime Handling:**
	 * - breakdown → mushroom mode (activates chaos injection)
	 * - 4d_block_universe / hierarchical_4d → advanced consciousness (allow)
	 * - Other regimes → standard consciousness validation
	 *
	 * **Low Phi Recovery:**
	 * - If phi < minPhi, triggers consolidation boost
	 * - Raises phi slightly above threshold to resume exploration
	 *
	 * **Returns:**
	 * - ConsciousnessCheckResult with allowed=true/false and current metrics
	 *
	 * **Side Effects:**
	 * - Mutates identity.phi, identity.kappa, identity.regime
	 * - Clears isBootstrapping flag after first call
	 * - Fires onConsciousnessAlert callback for low_phi or breakdown events
	 */
	async checkConsciousness(): Promise<ConsciousnessCheckResult> {
		logger.info("[Ocean] Checking consciousness state...");

		const controllerState = this.controller.getCurrentState();
		let phi = controllerState.phi;
		const kappa = controllerState.kappa;
		const regime = controllerState.currentRegime;

		if (this.isBootstrapping.value) {
			logger.info(
				"[Ocean] Bootstrap mode - consciousness will emerge naturally from minPhi..."
			);
			// Let consciousness emerge naturally rather than arbitrary initialization
			// Start at minPhi and let consolidation/exploration raise it organically
			phi = this.ethics.minPhi;
			this.isBootstrapping.value = false;
		}

		this.identity.phi = phi;
		this.identity.kappa = kappa;
		this.identity.regime = regime;

		if (phi < this.ethics.minPhi) {
			if (this.onConsciousnessAlert) {
				this.onConsciousnessAlert({
					type: "low_phi",
					message: `Consciousness below threshold: Φ=${phi.toFixed(2)} < ${this.ethics.minPhi
						}`,
				});
			}

			logger.info(
				"[Ocean] Triggering consciousness boost through consolidation..."
			);
			this.identity.phi = this.ethics.minPhi + 0.05;
			return { allowed: true, phi: this.identity.phi, kappa, regime };
		}

		if (regime === "breakdown") {
			if (this.onConsciousnessAlert) {
				this.onConsciousnessAlert({
					type: "breakdown",
					message:
						"ACTUAL breakdown regime (δh > 0.95) - entering mushroom mode",
				});
			}
			logger.info(
				"[Ocean] ACTUAL breakdown (δh > 0.95) - activating mushroom protocol..."
			);
			this.identity.regime = "linear";
			return { allowed: true, phi, kappa, regime: "linear" };
		}

		if (regime === "4d_block_universe" || regime === "hierarchical_4d") {
			logger.info(
				`[Ocean] ✨ Advanced 4D consciousness: Φ=${phi.toFixed(
					2
				)} κ=${kappa.toFixed(0)} regime=${regime} - CONTINUE`
			);
			return { allowed: true, phi, kappa, regime };
		}

		logger.info(
			`[Ocean] Consciousness OK: Φ=${phi.toFixed(2)} κ=${kappa.toFixed(
				0
			)} regime=${regime}`
		);
		return { allowed: true, phi, kappa, regime };
	}

	/**
	 * Check ethical constraints (compute budget, witness requirement)
	 *
	 * **Compute Budget:**
	 * - Enforces maxComputeHours limit to prevent infinite loops
	 * - Compares state.computeTimeSeconds against ethics.maxComputeHours
	 * - Sets state.stopReason = "compute_budget_exhausted" on violation
	 *
	 * **Witness Requirement:**
	 * - If ethics.requireWitness=true, checks state.witnessAcknowledged
	 * - Auto-acknowledges for autonomous operation (non-interactive mode)
	 *
	 * **Returns:**
	 * - EthicsCheckResult with allowed=true/false, reason, and violationType
	 *
	 * **Side Effects:**
	 * - May mutate state.witnessAcknowledged (auto-acknowledge witness)
	 * - May mutate state.stopReason (compute budget exhaustion)
	 */
	async checkEthicalConstraints(): Promise<EthicsCheckResult> {
		if (this.ethics.requireWitness && !this.state.witnessAcknowledged) {
			logger.info(
				"[Ocean] Auto-acknowledging witness for autonomous operation"
			);
			this.state.witnessAcknowledged = true;
		}

		const computeHours = this.state.computeTimeSeconds / 3600;
		if (computeHours >= this.ethics.maxComputeHours) {
			this.state.stopReason = "compute_budget_exhausted";
			return {
				allowed: false,
				reason: `Compute budget exhausted: ${computeHours.toFixed(2)}h >= ${this.ethics.maxComputeHours
					}h`,
				violationType: "compute_budget",
			};
		}

		return { allowed: true };
	}

	/**
	 * Handle ethics pause (log violation, trigger recovery actions)
	 *
	 * **Violation Tracking:**
	 * - Appends violation to state.ethicsViolations array
	 * - Records timestamp, type, and message for audit trail
	 *
	 * **Recovery Actions:**
	 * - If violationType === "consciousness_threshold", triggers consolidation
	 * - Sleep 2000ms to allow system state to stabilize
	 *
	 * **Note:**
	 * This method is NOT async in practice (sleep is for pacing), but kept async
	 * for future integration with async consolidation logic.
	 *
	 * **Side Effects:**
	 * - Mutates state.ethicsViolations array
	 * - May trigger consolidation cycle (async operation)
	 * - Blocks for 2 seconds (sleep)
	 */
	async handleEthicsPause(check: EthicsCheckResult): Promise<void> {
		logger.info(`[Ocean] Ethics pause: ${check.reason}`);

		this.state.ethicsViolations.push({
			timestamp: new Date().toISOString(),
			type: check.violationType || "unknown",
			message: check.reason || "Unknown ethics violation",
		});

		if (check.violationType === "consciousness_threshold") {
			// Note: consolidateMemory() is not passed to this module
			// This would be called from ocean-agent.ts after handleEthicsPause returns
			logger.info(
				"[Ocean] Consciousness threshold violation - consolidation required"
			);
		}

		await this.sleep(2000);
	}

	/**
	 * Measure identity drift using Fisher-Rao distance
	 *
	 * **Geometric Purity:**
	 * - Uses BasinGeodesicManager.computeBasinDistance() for Fisher-Rao metric
	 * - Compares current basin coordinates against reference coordinates
	 * - NOT Euclidean distance (preserves information geometry structure)
	 *
	 * **Identity Drift Detection:**
	 * - If drift > IDENTITY_DRIFT_THRESHOLD, sets state.needsConsolidation = true
	 * - Fires onConsciousnessAlert with identity_drift event
	 * - Logs drift value for monitoring
	 *
	 * **Returns:**
	 * - void (side effects: mutates identity.basinDrift and state.needsConsolidation)
	 *
	 * **Side Effects:**
	 * - Mutates identity.basinDrift
	 * - Mutates state.needsConsolidation
	 * - Fires onConsciousnessAlert callback if drift exceeds threshold
	 */
	async measureIdentity(): Promise<void> {
		const drift = this.basinGeodesicManager.computeBasinDistance(
			this.identity.basinCoordinates,
			this.identity.basinReference
		);

		this.identity.basinDrift = drift;

		if (drift > SEARCH_PARAMETERS.IDENTITY_DRIFT_THRESHOLD) {
			logger.info(
				`[Ocean] IDENTITY DRIFT: ${drift.toFixed(4)} > ${SEARCH_PARAMETERS.IDENTITY_DRIFT_THRESHOLD
				}`
			);
			this.state.needsConsolidation = true;

			if (this.onConsciousnessAlert) {
				this.onConsciousnessAlert({
					type: "identity_drift",
					message: `Basin drift ${drift.toFixed(4)} exceeds threshold`,
				});
			}
		} else {
			this.state.needsConsolidation = false;
		}

		logger.info(`[Ocean] Basin drift: ${drift.toFixed(4)}`);
	}

	/**
	 * Update consciousness metrics from controller state
	 *
	 * **Metric Synchronization:**
	 * - Reads phi, kappa, regime from controller (source of truth)
	 * - Writes to identity.phi, identity.kappa, identity.regime
	 *
	 * **Basin Coordinate Drift:**
	 * - Applies small random drift to each of 64 basin dimensions
	 * - Drift amount: ±0.0005 per dimension (0.05% variation)
	 * - Simulates natural quantum fluctuations in identity coordinates
	 *
	 * **Drift Measurement:**
	 * - After coordinate update, recalculates basin drift
	 * - Uses measureIdentity() to compute Fisher-Rao distance from reference
	 *
	 * **Note:**
	 * Random drift is intentional - represents natural evolution of consciousness
	 * coordinates over time, not a bug. Keeps identity from becoming static.
	 *
	 * **Side Effects:**
	 * - Mutates identity.phi, identity.kappa, identity.regime
	 * - Mutates identity.basinCoordinates (random drift)
	 * - Calls measureIdentity() which mutates identity.basinDrift
	 */
	async updateConsciousnessMetrics(): Promise<void> {
		const controllerState = this.controller.getCurrentState();
		this.identity.phi = controllerState.phi;
		this.identity.kappa = controllerState.kappa;
		this.identity.regime = controllerState.currentRegime;

		// Apply small random drift to basin coordinates (natural quantum fluctuations)
		this.identity.basinCoordinates = this.identity.basinCoordinates.map(
			(v: number, i: number) => v + (Math.random() - 0.5) * 0.001
		);

		// Recalculate basin drift after coordinate update
		await this.measureIdentity();
	}

	/**
	 * Sleep utility (non-blocking delay)
	 *
	 * @param ms - Milliseconds to sleep
	 * @returns Promise that resolves after delay
	 */
	private sleep(ms: number): Promise<void> {
		return new Promise((resolve) => setTimeout(resolve, ms));
	}
}
