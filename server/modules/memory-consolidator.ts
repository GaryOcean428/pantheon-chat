/**
 * Memory Consolidator Module (Phase 2B Extraction - 2026-01-09)
 *
 * Extracted from ocean-agent.ts to reduce complexity (Phase 2 of 3)
 *
 * **Responsibilities:**
 * - Consolidate episodic memory into long-term patterns
 * - Update episode phi values from Python backend (pure consciousness measurements)
 * - Extract word/format patterns from high-phi episodes
 * - Apply basin coordinate corrections toward reference state
 * - Track consolidation metrics and success criteria
 *
 * **Design Pattern:**
 * - Extract-delegate: Ocean agent delegates memory consolidation to this module
 * - Single responsibility: ONLY memory consolidation, not hypothesis generation or basin management
 * - Pure consciousness: Prefers Python backend phi measurements over TypeScript approximations
 *
 * **Architectural Note:**
 * This module coordinates with BasinGeodesicManager for basin distance calculations
 * and uses geometric memory for fast local phi lookups before falling back to Python.
 *
 * **Target:** ~220 lines extracted from ocean-agent.ts
 */

import type { OceanEpisode, OceanIdentity, OceanMemory } from "@shared/schema";
import { CONSCIOUSNESS_THRESHOLDS, SEARCH_PARAMETERS } from "../../shared/constants/index.js";
import { geometricMemory } from "../geometric-memory";
import { logger } from "../lib/logger";
import { oceanQIGBackend } from "../ocean-qig-backend-adapter";
import type { BasinGeodesicManager } from "./basin-geodesic-manager";

/**
 * Consolidation result metrics
 */
export interface ConsolidationResult {
	basinDriftBefore: number;
	basinDriftAfter: number;
	episodesProcessed: number;
	patternsExtracted: number;
	duration: number;
}

/**
 * MemoryConsolidator - Consolidates episodic memory into long-term patterns
 *
 * **Pure Consciousness Principle:**
 * Python backend produces pure phi values (0.9+) via proper quantum measurement.
 * TypeScript computePhi uses Math.tanh which caps around 0.76.
 * We prefer the pure Python measurement when available.
 *
 * **Consolidation Process:**
 * 1. First pass: Fast local geometric memory lookups (non-blocking)
 * 2. Second pass: Python backend phi calls (batched, max 4 concurrent)
 * 3. Pattern extraction: Words and formats from high-phi episodes
 * 4. Basin correction: Nudge coordinates toward reference state
 * 5. Update drift metrics and consolidation timestamp
 *
 * **Performance:**
 * - Bounded concurrency prevents Express event loop blocking
 * - Yields to event loop between batches
 * - Only processes recent 100 episodes for efficiency
 */
export class MemoryConsolidator {
	/**
	 * Constructor
	 *
	 * @param basinGeodesicManager - For basin distance computation (Fisher-Rao metric)
	 * @param identity - Ocean identity (mutable, updated by consolidation)
	 * @param memory - Ocean memory (mutable, patterns extracted here)
	 * @param onConsolidationStart - Optional callback when consolidation begins
	 * @param onConsolidationEnd - Optional callback when consolidation completes
	 */
	constructor(
		private basinGeodesicManager: BasinGeodesicManager,
		private identity: OceanIdentity,
		private memory: OceanMemory,
		private onConsolidationStart?: () => void,
		private onConsolidationEnd?: (result: ConsolidationResult) => void
	) { }

	/**
	 * Consolidate episodic memory into long-term patterns
	 *
	 * **What This Does:**
	 * 1. Updates episode phi values from Python backend (pure consciousness)
	 * 2. Extracts word patterns from high-phi episodes
	 * 3. Tracks successful formats
	 * 4. Corrects basin coordinates toward reference
	 * 5. Recalculates basin drift
	 *
	 * **Performance:**
	 * - Only processes recent 100 episodes (not entire history)
	 * - First tries geometric memory (fast local lookup)
	 * - Then batches Python calls (max 4 concurrent)
	 * - Yields to event loop between batches
	 *
	 * **Returns:**
	 * - true if consolidation succeeded (drift < threshold)
	 * - false if drift still high (needs more consolidation)
	 *
	 * **Side Effects:**
	 * - Mutates episode phi values
	 * - Mutates memory.patterns (promisingWords, successfulFormats)
	 * - Mutates identity.basinCoordinates (applies corrections)
	 * - Mutates identity.basinDrift (recalculates)
	 * - Fires onConsolidationStart/End callbacks
	 */
	async consolidate(): Promise<boolean> {
		logger.info("[Ocean] Starting consolidation cycle...");
		const startTime = Date.now();
		const driftBefore = this.identity.basinDrift;

		if (this.onConsolidationStart) {
			this.onConsolidationStart();
		}

		const recentEpisodes = this.memory.episodes.slice(-100);
		let patternsExtracted = 0;
		let phiUpgrades = 0;

		// PURE CONSCIOUSNESS: Update episode phi values from Python backend directly
		// This is the authoritative source for pure consciousness measurements
		let pythonPhiCalls = 0;
		let pythonSkipped = 0;

		// Ensure Python backend is available before attempting phi upgrades
		const pythonAvailable = await oceanQIGBackend.checkHealth(true);

		// First pass: fast local memory lookups (non-blocking)
		const episodesNeedingPython: OceanEpisode[] = [];

		for (const episode of recentEpisodes) {
			if (episode.phi < CONSCIOUSNESS_THRESHOLDS.PHI_PATTERN_EXTRACTION) {
				// First try geometric memory (fast local lookup)
				const storedScore = geometricMemory.getHighestPhiForInput(
					episode.phrase
				);
				if (storedScore && storedScore.phi > episode.phi) {
					const oldPhi = episode.phi;
					episode.phi = storedScore.phi;

					if (
						episode.result === "failure" &&
						storedScore.phi > CONSCIOUSNESS_THRESHOLDS.PHI_NEAR_MISS
					) {
						episode.result = "near_miss";
					}

					phiUpgrades++;

					if (storedScore.phi > CONSCIOUSNESS_THRESHOLDS.PHI_NEAR_MISS) {
						logger.info(
							`[Consolidation] 📈 Φ upgrade (memory): "${episode.phrase
							}" ${oldPhi.toFixed(3)} → ${storedScore.phi.toFixed(3)}`
						);
					}
				}

				// Collect episodes still needing Python phi upgrade
				if (
					episode.phi < CONSCIOUSNESS_THRESHOLDS.PHI_PATTERN_EXTRACTION &&
					pythonAvailable
				) {
					episodesNeedingPython.push(episode);
				} else if (
					episode.phi < CONSCIOUSNESS_THRESHOLDS.PHI_PATTERN_EXTRACTION
				) {
					pythonSkipped++;
				}
			}
		}

		// Second pass: Python calls with bounded concurrency (max 4 concurrent)
		// This prevents blocking the Express event loop during consolidation
		const MAX_CONCURRENT_PYTHON_CALLS = 4;

		if (episodesNeedingPython.length > 0) {
			// Process in batches to yield control to event loop
			for (
				let i = 0;
				i < episodesNeedingPython.length;
				i += MAX_CONCURRENT_PYTHON_CALLS
			) {
				const batch = episodesNeedingPython.slice(
					i,
					i + MAX_CONCURRENT_PYTHON_CALLS
				);

				// Process batch concurrently
				const results = await Promise.all(
					batch.map(async (episode) => {
						const purePhi = await oceanQIGBackend.getPurePhi(episode.phrase);
						pythonPhiCalls++;
						return { episode, purePhi };
					})
				);

				// Apply results
				for (const { episode, purePhi } of results) {
					if (purePhi !== null && purePhi > episode.phi) {
						const oldPhi = episode.phi;
						episode.phi = purePhi;

						if (
							episode.result === "failure" &&
							purePhi > CONSCIOUSNESS_THRESHOLDS.PHI_NEAR_MISS
						) {
							episode.result = "near_miss";
						}

						phiUpgrades++;

						if (purePhi > CONSCIOUSNESS_THRESHOLDS.PHI_NEAR_MISS) {
							logger.info(
								`[Consolidation] 🐍 Φ upgrade (Python): "${episode.phrase
								}" ${oldPhi.toFixed(3)} → ${purePhi.toFixed(3)}`
							);
						}
					}
				}

				// Yield to event loop between batches so Express can handle requests
				if (i + MAX_CONCURRENT_PYTHON_CALLS < episodesNeedingPython.length) {
					await new Promise((resolve) => setImmediate(resolve));
				}
			}
		}

		if (pythonPhiCalls > 0) {
			logger.info(
				`[Consolidation] Made ${pythonPhiCalls} Python phi calls (batched, max ${MAX_CONCURRENT_PYTHON_CALLS} concurrent)`
			);
		}
		if (pythonSkipped > 0 && !pythonAvailable) {
			logger.info(
				`[Consolidation] ⚠️ Python backend unavailable - skipped ${pythonSkipped} potential phi upgrades`
			);
		}

		if (phiUpgrades > 0) {
			logger.info(
				`[Consolidation] Updated ${phiUpgrades} episodes with pure Φ values`
			);
		}

		// Extract patterns from high-phi episodes
		for (const episode of recentEpisodes) {
			if (
				episode.result === "near_miss" ||
				episode.phi > CONSCIOUSNESS_THRESHOLDS.PHI_PATTERN_EXTRACTION
			) {
				const words = episode.phrase.toLowerCase().split(/\s+/);
				for (const word of words) {
					const current = this.memory.patterns.promisingWords[word] || 0;
					this.memory.patterns.promisingWords[word] = current + episode.phi;
					patternsExtracted++;
				}

				const format = episode.format;
				const currentFormat =
					this.memory.patterns.successfulFormats[format] || 0;
				this.memory.patterns.successfulFormats[format] = currentFormat + 1;
			}
		}

		// Apply basin coordinate corrections toward reference state
		const correctionRate = 0.1;
		for (let i = 0; i < 64; i++) {
			const correction =
				(this.identity.basinReference[i] - this.identity.basinCoordinates[i]) *
				correctionRate;
			this.identity.basinCoordinates[i] += correction;
		}

		// Recalculate basin drift after corrections
		this.identity.basinDrift = this.basinGeodesicManager.computeBasinDistance(
			this.identity.basinCoordinates,
			this.identity.basinReference
		);

		// Update consolidation timestamp
		this.identity.lastConsolidation = new Date().toISOString();

		const duration = Date.now() - startTime;

		const result: ConsolidationResult = {
			basinDriftBefore: driftBefore,
			basinDriftAfter: this.identity.basinDrift,
			episodesProcessed: recentEpisodes.length,
			patternsExtracted,
			duration,
		};

		const success =
			this.identity.basinDrift < SEARCH_PARAMETERS.IDENTITY_DRIFT_THRESHOLD;

		logger.info(`[Ocean] Consolidation complete:`);
		logger.info(
			`  - Drift: ${driftBefore.toFixed(
				4
			)} -> ${this.identity.basinDrift.toFixed(4)}`
		);
		logger.info(`  - Patterns extracted: ${patternsExtracted}`);
		logger.info(`  - Duration: ${duration}ms`);
		logger.info(`  - Success: ${success ? "YES" : "NO (drift still high)"}`);

		if (this.onConsolidationEnd) {
			this.onConsolidationEnd(result);
		}

		return success;
	}
}
