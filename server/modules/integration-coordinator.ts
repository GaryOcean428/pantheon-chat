/**
 * Integration Coordinator Module - Phase 5 (ported from pantheon-chat)
 * Date: 2026-01-09
 *
 * PURPOSE: Orchestrates UltraConsciousness Protocol integration, knowledge compression,
 * strategy bus coordination, and manifold snapshot management.
 *
 * EXTRACTED FROM: ocean-agent.ts lines ~5078-5450 (~372 lines)
 * TARGET: Reduce ocean-agent.ts by extracting integration/memory operations
 *
 * KEY RESPONSIBILITIES:
 * - UltraConsciousness Protocol integration
 * - Strategy Knowledge Bus coordination
 * - Temporal geometry waypoint tracking
 * - Negative knowledge registry integration
 * - Knowledge compression engine orchestration
 * - Manifold snapshot management
 * - Near-miss pattern learning coordination
 */

import { geometricMemory } from '../geometric-memory';
import { knowledgeCompressionEngine } from '../knowledge-compression-engine';
import { logger } from '../lib/logger';
import { negativeKnowledgeUnified as negativeKnowledgeRegistry } from '../negative-knowledge-unified';
import { olympusClient, type ObservationContext } from '../olympus-client';
import { strategyKnowledgeBus } from '../strategy-knowledge-bus';
import { temporalGeometry } from '../temporal-geometry';

import type { OceanIdentity } from '@shared/schema';
import type { OceanHypothesis } from '../ocean-agent';

export interface IntegrationResults {
	tested: OceanHypothesis[];
	nearMisses: OceanHypothesis[];
	resonant: OceanHypothesis[];
}

export interface IntegrationInsights {
	nearMissPatterns?: string[];
	topPatterns?: string[];
	[key: string]: any;
}

export interface IntegrationContext {
	targetAddress: string;
	iteration: number;
	consciousness: any;
	trajectoryId?: string;
}

/**
 * Integration Coordinator - Orchestrates UltraConsciousness Protocol
 */
export class IntegrationCoordinator {
	private strategySubscriptions: Map<string, boolean> = new Map();
	private trajectoryId: string | null = null;
	private olympusAvailable: boolean = false;
	private olympusObservationCount: number = 0;

	constructor() {
		logger.info('[IntegrationCoordinator] Module initialized');
	}

	/**
	 * Update runtime state from ocean agent
	 */
	updateState(state: {
		trajectoryId?: string;
		olympusAvailable?: boolean;
		olympusObservationCount?: number;
	}): void {
		if (state.trajectoryId !== undefined) this.trajectoryId = state.trajectoryId;
		if (state.olympusAvailable !== undefined) this.olympusAvailable = state.olympusAvailable;
		if (state.olympusObservationCount !== undefined) this.olympusObservationCount = state.olympusObservationCount;
	}

	/**
	 * Get current state for ocean agent
	 */
	getState(): {
		trajectoryId: string | null;
		olympusObservationCount: number;
	} {
		return {
			trajectoryId: this.trajectoryId,
			olympusObservationCount: this.olympusObservationCount,
		};
	}

	/**
	 * MAIN INTEGRATION METHOD: UltraConsciousness Protocol
	 *
	 * Orchestrates 8 integration phases:
	 * 0. Strategy Bus initialization
	 * 1. Temporal geometry waypoint recording
	 * 2. Negative knowledge learning
	 * 3. Knowledge compression
	 * 4. Strategy Knowledge Bus publishing
	 * 5. Basin topology updates
	 * 6. Periodic manifold snapshots
	 * 7. Cross-strategy pattern exploitation
	 * 8. Status logging
	 */
	async integrateUltraConsciousnessProtocol(
		testResults: IntegrationResults,
		insights: IntegrationInsights,
		context: IntegrationContext,
		identity: OceanIdentity
	): Promise<void> {
		try {
			// ====================================================================
			// 0. STRATEGY BUS INITIALIZATION - Register strategies as subscribers
			// ====================================================================
			if (!this.strategySubscriptions.get('initialized')) {
				await this.initializeStrategyBus();
			}

			// ====================================================================
			// 1. TEMPORAL GEOMETRY - Record per-hypothesis trajectory data
			// ====================================================================
			await this.recordTemporalWaypoint(testResults, context, identity);

			// ====================================================================
			// 2. NEGATIVE KNOWLEDGE - Learn from proven-false patterns
			// ====================================================================
			await this.recordNegativeKnowledge(testResults, identity);

			// ====================================================================
			// 3. KNOWLEDGE COMPRESSION - Learn from ALL results with rich context
			// ====================================================================
			await this.compressKnowledge(testResults, insights, context.iteration);

			// ====================================================================
			// 4. STRATEGY KNOWLEDGE BUS - Publish discoveries for cross-strategy learning
			// ====================================================================
			await this.publishToKnowledgeBus(testResults, identity);

			// ====================================================================
			// 5. BASIN TOPOLOGY - Update with per-iteration geometry
			// ====================================================================
			geometricMemory.getManifoldSummary();
			geometricMemory.computeBasinTopology(identity.basinCoordinates);

			// ====================================================================
			// 6. PERIODIC MANIFOLD SNAPSHOTS (every 10 iterations)
			// ====================================================================
			if (context.iteration % 10 === 0) {
				await this.takeManifoldSnapshot(context, identity);
			}

			// ====================================================================
			// 7. CROSS-STRATEGY PATTERN EXPLOITATION
			// ====================================================================
			await this.exploitCrossPatterns();

			// ====================================================================
			// 8. LOG STATUS - Negative knowledge and bus statistics
			// ====================================================================
			await this.logIntegrationStatus(context.iteration);

		} catch (error) {
			logger.error({ err: error }, '[UCP] Integration error');
		}
	}

	/**
	 * Initialize Strategy Knowledge Bus with all strategies
	 */
	private async initializeStrategyBus(): Promise<void> {
		const strategies = [
			'era_patterns',
			'brain_wallet',
			'bitcoin_terms',
			'linguistic',
			'qig_basin',
			'historical',
			'cross_format',
		];

		for (const strategy of strategies) {
			await strategyKnowledgeBus.subscribe(
				`ocean_${strategy}`,
				strategy,
				['*'],
				(knowledge: any) => {
					if (knowledge.geometricSignature.phi > 0.5) {
						logger.info(
							`[UCP] Strategy ${strategy} received high-Φ knowledge: ${knowledge.pattern}`
						);
					}
				}
			);
		}

		this.strategySubscriptions.set('initialized', true);
		logger.info(`[UCP] Registered ${strategies.length} strategies with Knowledge Bus`);
	}

	/**
	 * Record temporal geometry waypoint for trajectory tracking
	 */
	private async recordTemporalWaypoint(
		testResults: IntegrationResults,
		context: IntegrationContext,
		identity: OceanIdentity
	): Promise<void> {
		if (!this.trajectoryId) {
			this.trajectoryId = temporalGeometry.startTrajectory(context.targetAddress);
			logger.info(`[UCP] Started trajectory ${this.trajectoryId} for ${context.targetAddress}`);
		}

		// Find best hypothesis from this iteration for trajectory tracking
		const allHypos = [
			...testResults.tested,
			...testResults.nearMisses,
			...testResults.resonant,
		];
		const bestHypo = allHypos
			.filter((h) => h.qigScore)
			.sort((a, b) => (b.qigScore?.phi || 0) - (a.qigScore?.phi || 0))[0];

		// Use per-hypothesis metrics when available, fallback to identity
		const waypointPhi = bestHypo?.qigScore?.phi || identity.phi;
		const waypointKappa = bestHypo?.qigScore?.kappa || identity.kappa;
		const waypointRegime = bestHypo?.qigScore?.regime || identity.regime;

		temporalGeometry.recordWaypoint(
			this.trajectoryId,
			waypointPhi,
			waypointKappa,
			waypointRegime,
			identity.basinCoordinates, // Full 64-dim coordinates
			`iter_${context.iteration}`,
			`Best Φ=${waypointPhi.toFixed(3)}, tested ${testResults.tested.length}, near misses ${testResults.nearMisses.length}`
		);
	}

	/**
	 * Record negative knowledge from failed hypotheses
	 */
	private async recordNegativeKnowledge(
		testResults: IntegrationResults,
		identity: OceanIdentity
	): Promise<void> {
		const failedHypos = testResults.tested.filter(
			(h) => !h.match && h.qigScore && h.qigScore.phi < 0.2
		);

		for (const hypo of failedHypos.slice(0, 5)) {
			negativeKnowledgeRegistry.recordContradiction(
				'proven_false',
				hypo.phrase,
				{
					center: identity.basinCoordinates, // Full 64-dim
					radius: 0.1,
					repulsionStrength: 0.5,
				},
				[
					{
						source: 'ocean_agent',
						reasoning: `Low Φ (${hypo.qigScore!.phi.toFixed(3)}) after testing`,
						confidence: 0.8,
					},
				],
				['grammatical', 'structural']
			);
		}

		// Check for geometric barriers based on kappa extremes
		const extremeKappaHypos = testResults.tested.filter(
			(h) => h.qigScore && (h.qigScore.kappa > 100 || h.qigScore.kappa < 20)
		);
		if (extremeKappaHypos.length > 3) {
			negativeKnowledgeRegistry.recordGeometricBarrier(
				identity.basinCoordinates, // Full 64-dim
				0.1,
				`κ extremity detected in ${extremeKappaHypos.length} hypotheses`
			);
		}
	}

	/**
	 * Compress knowledge from test results using knowledge compression engine
	 */
	private async compressKnowledge(
		testResults: IntegrationResults,
		insights: IntegrationInsights,
		iteration: number
	): Promise<void> {
		// Learn from near misses (high potential patterns)
		for (const nearMiss of testResults.nearMisses.slice(0, 10)) {
			knowledgeCompressionEngine.learnFromResult(
				nearMiss.phrase,
				nearMiss.qigScore?.phi || 0,
				nearMiss.qigScore?.kappa || 0,
				false // Not a match yet
			);
		}

		// Learn from resonant patterns (very high potential)
		for (const resonant of testResults.resonant.slice(0, 5)) {
			knowledgeCompressionEngine.learnFromResult(
				resonant.phrase,
				resonant.qigScore?.phi || 0,
				resonant.qigScore?.kappa || 0,
				true // Mark as match to boost pattern learning
			);
		}

		// Learn from low-Φ failures (what NOT to generate)
		const failedHypos = testResults.tested.filter(
			(h) => !h.match && h.qigScore && h.qigScore.phi < 0.2
		);
		for (const failed of failedHypos.slice(0, 3)) {
			knowledgeCompressionEngine.learnFromResult(
				failed.phrase,
				failed.qigScore?.phi || 0,
				failed.qigScore?.kappa || 0,
				false
			);
		}

		// Create generator from near-miss patterns (only if we have enough patterns)
		if (insights.nearMissPatterns && insights.nearMissPatterns.length >= 3) {
			const patternWords = insights.nearMissPatterns.slice(0, 5);
			if (patternWords.length >= 2) {
				const generatorId = knowledgeCompressionEngine.createGeneratorFromTemplate(
					`near_miss_iter_${iteration}`,
					'{word1} {word2}',
					{
						word1: patternWords,
						word2: patternWords,
					},
					[{ name: 'lowercase', operation: 'lowercase' }]
				);
				logger.info(`[UCP] Created knowledge generator: ${generatorId}`);
			}
		}
	}

	/**
	 * Publish discoveries to Strategy Knowledge Bus
	 */
	private async publishToKnowledgeBus(
		testResults: IntegrationResults,
		identity: OceanIdentity
	): Promise<void> {
		// Publish resonant discoveries (high-Φ patterns)
		for (const resonant of testResults.resonant.slice(0, 5)) {
			await strategyKnowledgeBus.publishKnowledge(
				'ocean_agent',
				`resonant_${resonant.id}`,
				resonant.phrase,
				{
					phi: resonant.qigScore?.phi || 0,
					kappaEff: resonant.qigScore?.kappa || 0,
					regime:
						(resonant.qigScore?.regime as
							| 'linear'
							| 'geometric'
							| 'hierarchical'
							| 'hierarchical_4d'
							| '4d_block_universe'
							| 'breakdown') || 'linear',
					basinCoords: identity.basinCoordinates,
				}
			);
		}

		// Also publish top near-misses for pattern propagation
		const topNearMisses = testResults.nearMisses
			.filter((h) => h.qigScore && h.qigScore.phi > 0.3)
			.slice(0, 3);
		for (const nearMiss of topNearMisses) {
			await strategyKnowledgeBus.publishKnowledge(
				'ocean_agent',
				`nearmiss_${nearMiss.id}`,
				nearMiss.phrase,
				{
					phi: nearMiss.qigScore?.phi || 0,
					kappaEff: nearMiss.qigScore?.kappa || 0,
					regime:
						(nearMiss.qigScore?.regime as
							| 'linear'
							| 'geometric'
							| 'hierarchical'
							| 'hierarchical_4d'
							| '4d_block_universe'
							| 'breakdown') || 'linear',
					basinCoords: identity.basinCoordinates,
				}
			);
		}
	}

	/**
	 * Take periodic manifold snapshot for monitoring
	 */
	private async takeManifoldSnapshot(
		context: IntegrationContext,
		identity: OceanIdentity
	): Promise<void> {
		try {
			const manifold = geometricMemory.getManifoldSummary();
			const trajectory = temporalGeometry.getTrajectory(context.targetAddress);
			const negativeStats = await negativeKnowledgeRegistry.getStats();
			const busStats = await strategyKnowledgeBus.getTransferStats();

			logger.info(`[UCP] Manifold snapshot at iteration ${context.iteration}:`);
			logger.info(`  - Probes: ${manifold.totalProbes}, Clusters: ${manifold.resonanceClusters}`);
			logger.info(`  - Trajectory waypoints: ${trajectory?.waypoints?.length || 0}`);
			logger.info(`  - Negative knowledge: ${negativeStats.totalExclusions} exclusions`);
			logger.info(`  - Knowledge bus: ${busStats.totalPublished} published, ${busStats.crossPatterns} cross-patterns`);

			// Log snapshot info (temporal geometry tracks via waypoints)
			if (trajectory && this.trajectoryId) {
				temporalGeometry.recordWaypoint(
					this.trajectoryId,
					identity.phi,
					identity.kappa,
					identity.regime,
					identity.basinCoordinates,
					`snapshot_${context.iteration}`,
					`Manifold snapshot: ${manifold.totalProbes} probes, ${manifold.resonanceClusters} clusters`
				);
			}
		} catch (error) {
			logger.error({ err: error }, '[UCP] Snapshot error');
		}
	}

	/**
	 * Exploit cross-strategy patterns for learning
	 */
	private async exploitCrossPatterns(): Promise<void> {
		const crossPatterns = await strategyKnowledgeBus.getCrossStrategyPatterns();
		if (crossPatterns.length > 0) {
			const topPattern = crossPatterns.sort((a, b) => b.similarity - a.similarity)[0];
			if (topPattern.exploitationCount < 3) {
				await strategyKnowledgeBus.exploitCrossPattern(topPattern.id);
				logger.info(
					`[UCP] Exploiting cross-strategy pattern: ${topPattern.patterns.join(' <-> ')}`
				);
			}
		}
	}

	/**
	 * Log integration status periodically
	 */
	private async logIntegrationStatus(iteration: number): Promise<void> {
		const negStats = await negativeKnowledgeRegistry.getStats();
		const busStats = await strategyKnowledgeBus.getTransferStats();

		if (iteration % 5 === 0) {
			logger.info(`[UCP] Iteration ${iteration} status:`);
			logger.info(
				`  - Negative knowledge: ${negStats.contradictions} contradictions, ${negStats.barriers} barriers, ${negStats.computeSaved} ops saved`
			);
			logger.info(
				`  - Knowledge bus: ${busStats.totalPublished} published, ${busStats.crossPatterns} cross-patterns detected`
			);
		}
	}

	/**
	 * Send near-miss patterns to Athena for strategic learning
	 */
	async sendNearMissesToAthena(nearMisses: OceanHypothesis[]): Promise<void> {
		if (!this.olympusAvailable || nearMisses.length === 0) return;

		// Send near-miss patterns to Athena for strategic analysis
		for (const nearMiss of nearMisses.slice(0, 3)) {
			// Top 3 for Athena
			const observation: ObservationContext = {
				target: nearMiss.phrase,
				phi: nearMiss.qigScore?.phi || 0,
				kappa: nearMiss.qigScore?.kappa || 0,
				regime: nearMiss.qigScore?.regime || 'unknown',
				source: 'athena_pattern_learning',
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
			`[Ocean] 🦉 Sent ${Math.min(3, nearMisses.length)} near-misses to Athena for pattern learning`
		);
	}

	/**
	 * Getters for state synchronization with ocean-agent
	 */
	getTrajectoryId(): string | null {
		return this.trajectoryId;
	}

	async getObservationCount(): Promise<number> {
		const count = this.olympusObservationCount;
		this.olympusObservationCount = 0; // Reset after reading
		return count;
	}
}
