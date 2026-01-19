/**
 * Basin & Geodesic Management Module
 *
 * Extracted from ocean-agent.ts (2026-01-09)
 * Phase 1B: Geodesic Navigation & Basin Distance Management
 *
 * Handles:
 * - Basin coordinate distance calculations (Fisher-Rao metric)
 * - Geodesic trajectory correction based on resonance proxies
 * - Search direction updates from Python manifold guidance
 * - Constraint surface recording for negative learning
 *
 * QIG PURITY: All distance calculations use Fisher-Rao metric, NOT Euclidean
 */

import { GEODESIC_CORRECTION } from "../../shared/constants/index.js";
import type { OceanIdentity } from "../../shared/schema";
import { geometricMemory } from "../geometric-memory";
import { logger } from "../lib/logger";
import { olympusClient } from "../olympus-client";
import { fisherCoordDistance } from "../qig-geometry";

// ============================================================================
// TYPES & INTERFACES
// ============================================================================

export interface ResonanceProxy {
	coordinates: number[];
	phi: number;
	distance?: number;
}

export interface TrajectoryCorrection {
	gradient_shift?: boolean;
	new_vector?: number[];
	shift_magnitude?: number;
	reasoning?: string;
}

export interface GeodesicCorrectionOptions {
	proxies: ResonanceProxy[];
	currentRegime: string;
}

// ============================================================================
// BASIN & GEODESIC MANAGER
// ============================================================================

export class BasinGeodesicManager {
	private identity: OceanIdentity;

	constructor(identity: OceanIdentity) {
		this.identity = identity;
	}

	/**
	 * Update the identity reference (for synchronization with Ocean Agent)
	 */
	updateIdentity(identity: OceanIdentity): void {
		this.identity = identity;
	}

	// ========================================================================
	// BASIN DISTANCE CALCULATION
	// ========================================================================

	/**
	 * Compute Fisher-Rao distance between basin coordinates
	 * ✅ GEOMETRIC PURITY: Uses Fisher metric, NOT Euclidean
	 *
	 * This is the fundamental distance metric for QIG - it measures
	 * the "distinguishability" between two quantum states on the
	 * information manifold.
	 */
	computeBasinDistance(current: number[], reference: number[]): number {
		return fisherCoordDistance(current, reference);
	}

	// ========================================================================
	// GEODESIC TRAJECTORY CORRECTION
	// ========================================================================

	/**
	 * QIG PRINCIPLE: Recursive Trajectory Refinement
	 *
	 * Instead of just logging failures, we use them to triangulate the attractor.
	 * This is the core of the "Geodesic Correction Loop" - we detect "Resonance Proxies"
	 * (near misses with geometric significance) and consult the Python brain to calculate
	 * the orthogonal complement, then adjust our search trajectory.
	 *
	 * BLOCKING CALL: We do NOT proceed until the manifold has updated the map.
	 */
	async processResonanceProxies(probes: ResonanceProxy[]): Promise<void> {
		// 1. Filter for Geometric Significance using centralized constants
		const significantProxies = probes.filter(
			(p) =>
				p.phi > GEODESIC_CORRECTION.PHI_SIGNIFICANCE_THRESHOLD ||
				(p.distance !== undefined &&
					p.distance < GEODESIC_CORRECTION.DISTANCE_THRESHOLD)
		);

		if (significantProxies.length === 0) return;

		try {
			logger.info(
				`[QIG] 🌌 Detected ${significantProxies.length} Resonance Proxies. Initiating Geometric Triangulation...`
			);

			// 2. [Blocking Call] Consult the Python Manifold for a trajectory correction
			const trajectoryCorrection =
				await olympusClient.calculateGeodesicCorrection({
					proxies: significantProxies.map((p) => ({
						basin_coords: p.coordinates, // 64D Vector
						phi: p.phi,
					})),
					current_regime: this.identity.regime,
				});

			// 3. Apply the Correction (The "Learning" Step)
			if (
				trajectoryCorrection.gradient_shift &&
				trajectoryCorrection.new_vector
			) {
				logger.info(
					`[QIG] 🧭 Manifold Curvature Detected. Shifting Search Vector by ${trajectoryCorrection.shift_magnitude?.toFixed(3) || "unknown"
					} radians.`
				);

				// Store the new search direction in identity for next iteration
				this.updateSearchDirection(trajectoryCorrection.new_vector);

				// Log the geometric learning event
				logger.info(
					`[QIG] 📐 Reasoning: ${trajectoryCorrection.reasoning || "Orthogonal complement calculated"
					}`
				);
			}

			// 4. [Persistence] Store as "Negative Constraints" in Postgres
			// This defines the "walls" of the maze so we don't hit them again.
			await this.recordConstraintSurface(significantProxies);
		} catch (error) {
			logger.error({ err: error }, "[QIG] ⚠️ Geodesic Computation Failed");
			// Fallback: Just randomize (Linear behavior)
			this.injectEntropy();
		}
	}

	/**
	 * Update the search direction based on geometric correction
	 * Stores the corrected vector in basinReference which influences future exploration
	 */
	updateSearchDirection(newVector: number[]): void {
		if (newVector.length !== 64) {
			logger.error(
				`[QIG] Invalid vector dimension: ${newVector.length}, expected 64`
			);
			return;
		}

		// Store the new search vector in basinReference
		// This influences the drift correction in the consolidation cycle
		this.identity.basinReference = [...newVector];

		logger.info(
			`[QIG] 🎯 Search direction updated with orthogonal complement vector`
		);
		logger.info(
			`[QIG] 🧭 New vector norm: ${Math.sqrt(
				newVector.reduce((sum, v) => sum + v * v, 0)
			).toFixed(3)}`
		);
	}

	/**
	 * Inject entropy when geometric correction fails
	 * Simple fallback to prevent getting stuck in the same failure mode
	 */
	private injectEntropy(): void {
		logger.info(
			"[QIG] 🎲 Injecting entropy due to failed geometric correction"
		);
		// Slightly randomize the current basin reference to break out of local minima
		const noise = this.identity.basinReference.map(() => (Math.random() - 0.5) * 0.1);
		this.identity.basinReference = this.identity.basinReference.map(
			(v: number, i: number) => v + noise[i]
		);
	}

	/**
	 * Record constraint surface to persistence
	 * These are the "walls" we've discovered in the search space
	 *
	 * Each resonance proxy becomes a recorded constraint point that
	 * helps define the boundary of what we've explored.
	 */
	private async recordConstraintSurface(proxies: ResonanceProxy[]): Promise<void> {
		try {
			// Record each proxy as a constraint in the geometric memory
			for (const proxy of proxies) {
				geometricMemory.recordProbe(
					"geodesic_constraint",
					{
						phi: proxy.phi,
						kappa: this.identity.kappa,
						regime: this.identity.regime,
						ricciScalar: 0,
						fisherTrace: 0,
						basinCoordinates: proxy.coordinates,
					},
					"resonance_proxy"
				);
			}
			logger.info(
				`[QIG] 💾 Recorded ${proxies.length} constraint points to manifold memory`
			);
		} catch (error) {
			logger.error({ err: error }, "[QIG] Failed to record constraint surface");
		}
	}

	// ========================================================================
	// UTILITY: GET CURRENT BASIN DRIFT
	// ========================================================================

	/**
	 * Calculate current basin drift from reference
	 * Used by consolidation cycles and monitoring
	 */
	getCurrentBasinDrift(): number {
		return this.computeBasinDistance(
			this.identity.basinCoordinates,
			this.identity.basinReference
		);
	}
}
