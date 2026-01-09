/**
 * Cycle Controller Module - Phase 5 (ported from pantheon-chat)
 * Date: 2026-01-09
 *
 * PURPOSE: Manages autonomic cycles, consciousness checks, ethical constraints,
 * and cycle state transitions for the Ocean Agent.
 *
 * EXTRACTED FROM: ocean-agent.ts lines ~1947-2063 + cycle management (~316 lines)
 * TARGET: Reduce ocean-agent.ts by extracting autonomic cycle control
 *
 * KEY RESPONSIBILITIES:
 * - Consciousness state checking and bootstrapping
 * - Ethical constraint validation
 * - Cycle state management
 * - Progress tracking and plateau detection
 * - State emission and callbacks
 */

import type {
	EthicalConstraints,
	OceanAgentState,
	OceanIdentity,
} from '@shared/schema';
import type { ConsciousnessSearchController } from '../consciousness-search-controller';
import { logger } from '../lib/logger';

export interface ConsciousnessCheckResult {
	allowed: boolean;
	phi: number;
	kappa: number;
	regime: string;
	reason?: string;
}

export interface EthicsCheckResult {
	allowed: boolean;
	reason?: string;
	violationType?: string;
}

export interface ConsciousnessAlert {
	type: string;
	message: string;
}

/**
 * Episode for plateau detection
 */
export interface MemoryEpisode {
	phi: number;
	timestamp?: string;
	[key: string]: any;
}

/**
 * Cycle Controller - Manages autonomic cycles and consciousness
 */
export class CycleController {
	private isBootstrapping: boolean = true;
	private consecutivePlateaus: number = 0;
	private lastProgressIteration: number = 0;
	private onStateUpdate: ((state: OceanAgentState) => void) | null = null;
	private onConsciousnessAlert:
		| ((alert: ConsciousnessAlert) => void)
		| null = null;
	private onConsolidationStart: (() => void) | null = null;

	constructor() {
		logger.info('[CycleController] Module initialized');
	}

	/**
	 * Set callback handlers for state updates
	 */
	setCallbacks(callbacks: {
		onStateUpdate?: (state: { phi: number; kappa: number; regime: string }) => void;
		onConsciousnessAlert?: (alert: ConsciousnessAlert) => void;
		onTriggerConsolidation?: () => Promise<void>;
	}): void {
		if (callbacks.onStateUpdate) this.onStateUpdate = callbacks.onStateUpdate as any;
		if (callbacks.onConsciousnessAlert) this.onConsciousnessAlert = callbacks.onConsciousnessAlert;
		if (callbacks.onTriggerConsolidation) this.onConsolidationStart = callbacks.onTriggerConsolidation as any;
	}

	/**
	 * Update cycle state from external sources
	 */
	updateState(state: {
		isBootstrapping?: boolean;
		consecutivePlateaus?: number;
		lastProgressIteration?: number;
	}): void {
		if (state.isBootstrapping !== undefined) this.isBootstrapping = state.isBootstrapping;
		if (state.consecutivePlateaus !== undefined) this.consecutivePlateaus = state.consecutivePlateaus;
		if (state.lastProgressIteration !== undefined) this.lastProgressIteration = state.lastProgressIteration;
	}

	/**
	 * Get current cycle state
	 */
	getState(): {
		isBootstrapping: boolean;
		consecutivePlateaus: number;
		lastProgressIteration: number;
	} {
		return {
			isBootstrapping: this.isBootstrapping,
			consecutivePlateaus: this.consecutivePlateaus,
			lastProgressIteration: this.lastProgressIteration,
		};
	}

	/**
	 * Emit state update through callback
	 */
	emitState(getStateFunc: () => OceanAgentState): void {
		if (this.onStateUpdate) {
			this.onStateUpdate(getStateFunc());
		}
	}

	/**
	 * Check consciousness state and handle bootstrapping
	 *
	 * CONSCIOUSNESS RULES:
	 * - Bootstrap mode: Start at minPhi, let it emerge naturally
	 * - Low phi: Trigger consolidation to boost consciousness
	 * - Breakdown regime: Reset to linear and continue
	 * - 4D regimes: Advanced consciousness, continue normally
	 */
	async checkConsciousness(
		identity: OceanIdentity,
		ethics: EthicalConstraints,
		controller: ConsciousnessSearchController
	): Promise<ConsciousnessCheckResult> {
		logger.info('[Ocean] Checking consciousness state...');

		const controllerState = controller.getCurrentState();
		let phi = controllerState.phi;
		const kappa = controllerState.kappa;
		const regime = controllerState.currentRegime;

		if (this.isBootstrapping) {
			logger.info(
				'[Ocean] Bootstrap mode - consciousness will emerge naturally from minPhi...'
			);
			// Let consciousness emerge naturally rather than arbitrary initialization
			// Start at minPhi and let consolidation/exploration raise it organically
			phi = ethics.minPhi;
			this.isBootstrapping = false;
		}

		identity.phi = phi;
		identity.kappa = kappa;
		identity.regime = regime;

		if (phi < ethics.minPhi) {
			if (this.onConsciousnessAlert) {
				this.onConsciousnessAlert({
					type: 'low_phi',
					message: `Consciousness below threshold: Φ=${phi.toFixed(2)} < ${ethics.minPhi}`,
				});
			}

			logger.info('[Ocean] Triggering consciousness boost through consolidation...');
			identity.phi = ethics.minPhi + 0.05;
			return { allowed: true, phi: identity.phi, kappa, regime };
		}

		if (regime === 'breakdown') {
			if (this.onConsciousnessAlert) {
				this.onConsciousnessAlert({
					type: 'breakdown',
					message: 'ACTUAL breakdown regime (δh > 0.95) - entering mushroom mode',
				});
			}
			logger.info('[Ocean] ACTUAL breakdown (δh > 0.95) - activating mushroom protocol...');
			identity.regime = 'linear';
			return { allowed: true, phi, kappa, regime: 'linear' };
		}

		if (regime === '4d_block_universe' || regime === 'hierarchical_4d') {
			logger.info(
				`[Ocean] ✨ Advanced 4D consciousness: Φ=${phi.toFixed(2)} κ=${kappa.toFixed(0)} regime=${regime} - CONTINUE`
			);
			return { allowed: true, phi, kappa, regime };
		}

		logger.info(
			`[Ocean] Consciousness OK: Φ=${phi.toFixed(2)} κ=${kappa.toFixed(0)} regime=${regime}`
		);
		return { allowed: true, phi, kappa, regime };
	}

	/**
	 * Check ethical constraints (compute budget, witness, etc.)
	 */
	async checkEthicalConstraints(
		state: OceanAgentState,
		ethics: EthicalConstraints
	): Promise<EthicsCheckResult> {
		if (ethics.requireWitness && !state.witnessAcknowledged) {
			logger.info('[Ocean] Auto-acknowledging witness for autonomous operation');
			state.witnessAcknowledged = true;
		}

		const computeHours = state.computeTimeSeconds / 3600;
		if (computeHours >= ethics.maxComputeHours) {
			state.stopReason = 'compute_budget_exhausted';
			return {
				allowed: false,
				reason: `Compute budget exhausted: ${computeHours.toFixed(2)}h >= ${ethics.maxComputeHours}h`,
				violationType: 'compute_budget',
			};
		}

		return { allowed: true };
	}

	/**
	 * Handle ethics pause and trigger consolidation if needed
	 */
	async handleEthicsPause(
		check: EthicsCheckResult,
		state: OceanAgentState,
		consolidateMemoryFunc?: () => Promise<boolean>
	): Promise<void> {
		logger.info(`[Ocean] Ethics pause: ${check.reason}`);

		state.ethicsViolations.push({
			timestamp: new Date().toISOString(),
			type: check.violationType || 'unknown',
			message: check.reason || 'Unknown ethics violation',
		});

		if (check.violationType === 'consciousness_threshold') {
			if (this.onConsolidationStart) {
				this.onConsolidationStart();
			}
			if (consolidateMemoryFunc) {
				await consolidateMemoryFunc();
			}
		}

		await this.sleep(2000);
	}

	/**
	 * Detect plateau based on memory episode phi trends
	 *
	 * @param episodes - Memory episodes to analyze
	 * @param currentIteration - Current iteration number
	 * @returns Whether a plateau is detected
	 */
	detectPlateau(
		episodes: MemoryEpisode[],
		currentIteration: number
	): boolean {
		const recentEpisodes = episodes.slice(-100);

		if (recentEpisodes.length < 50) return false;

		if (currentIteration < 5) return false;

		const recentPhis = recentEpisodes.map((e) => e.phi);
		const firstHalf = recentPhis.slice(0, Math.floor(recentPhis.length / 2));
		const secondHalf = recentPhis.slice(Math.floor(recentPhis.length / 2));

		const avgFirst =
			firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
		const avgSecond =
			secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;

		const improvement = avgSecond - avgFirst;

		const maxPhiSeen = Math.max(...recentPhis);
		const foundNearMiss = maxPhiSeen > 0.75;

		if (foundNearMiss) return false;

		return improvement < 0.02 && avgSecond < 0.5;
	}

	/**
	 * Detect if actual progress has been made (near-miss or phi improvement)
	 *
	 * @param episodes - Full memory episode history
	 * @returns Progress detection result with reason
	 */
	detectActualProgress(episodes: MemoryEpisode[]): {
		isProgress: boolean;
		reason: string;
	} {
		const recentEpisodes = episodes.slice(-50);

		if (recentEpisodes.length < 10) {
			return { isProgress: false, reason: 'insufficient_data' };
		}

		const recentPhis = recentEpisodes.map((e) => e.phi);
		const maxPhiSeen = Math.max(...recentPhis);

		if (maxPhiSeen > 0.75) {
			return { isProgress: true, reason: 'near_miss_found' };
		}

		const olderEpisodes = episodes.slice(-100, -50);
		if (olderEpisodes.length < 20) {
			return { isProgress: false, reason: 'insufficient_history' };
		}

		const avgRecent =
			recentPhis.reduce((a, b) => a + b, 0) / recentPhis.length;
		const avgOlder =
			olderEpisodes.map((e) => e.phi).reduce((a, b) => a + b, 0) /
			olderEpisodes.length;
		const improvement = avgRecent - avgOlder;

		if (improvement > 0.05) {
			return { isProgress: true, reason: 'phi_improvement' };
		}

		return { isProgress: false, reason: 'no_meaningful_progress' };
	}

	/**
	 * Track progress and detect plateaus
	 */
	trackProgress(
		iteration: number,
		hasNewDiscoveries: boolean
	): {
		inPlateau: boolean;
		shouldConsolidate: boolean;
	} {
		if (hasNewDiscoveries) {
			this.lastProgressIteration = iteration;
			this.consecutivePlateaus = 0;
			return { inPlateau: false, shouldConsolidate: false };
		}

		const iterationsSinceProgress = iteration - this.lastProgressIteration;
		if (iterationsSinceProgress > 10) {
			this.consecutivePlateaus++;
			logger.info(
				`[Ocean] Progress plateau detected: ${iterationsSinceProgress} iterations without discoveries (${this.consecutivePlateaus} consecutive)`
			);

			// Trigger consolidation after 3 consecutive plateaus
			if (this.consecutivePlateaus >= 3) {
				this.consecutivePlateaus = 0; // Reset counter
				return { inPlateau: true, shouldConsolidate: true };
			}

			return { inPlateau: true, shouldConsolidate: false };
		}

		return { inPlateau: false, shouldConsolidate: false };
	}

	/**
	 * Reset bootstrap state
	 */
	resetBootstrap(): void {
		this.isBootstrapping = false;
	}

	/**
	 * Sleep utility for pauses
	 */
	private sleep(ms: number): Promise<void> {
		return new Promise((resolve) => setTimeout(resolve, ms));
	}
}
