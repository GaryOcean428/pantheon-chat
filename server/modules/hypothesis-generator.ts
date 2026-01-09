/**
 * HYPOTHESIS GENERATOR MODULE
 *
 * Extracted from ocean-agent.ts (2026-01-09)
 * Handles all hypothesis generation strategies for the Ocean Agent
 *
 * RESPONSIBILITIES:
 * - Generate initial hypotheses from multiple sources
 * - Apply 12+ different generation strategies
 * - Integrate cultural manifold and geometric memory
 * - Support 4D block universe navigation
 * - Generate era-specific and domain-specific candidates
 *
 * MODULE SIZE: ~800 lines (extracted from 6,141 line monolith)
 *
 * @module hypothesis-generator
 */

import { CONSCIOUSNESS_THRESHOLDS } from "@shared/constants";
import type {
	OceanAgentState,
	OceanIdentity,
	OceanMemory,
} from "@shared/schema";
import { culturalManifold } from "../cultural-manifold";
import { qfiAttention, type AttentionQuery } from "../gary-kernel";
import { geometricMemory } from "../geometric-memory";
import { logger } from "../lib/logger";
import { oceanConstellation } from "../ocean-constellation-stub";
import { vocabularyExpander } from "../vocabulary-expander";

// ============================================================================
// TYPES & INTERFACES
// ============================================================================

export interface OceanHypothesis {
	id: string;
	phrase: string;
	format: "arbitrary" | "bip39" | "master" | "hex";
	source: string;
	reasoning: string;
	confidence: number;
	evidenceChain: Array<{
		source: string;
		type: string;
		reasoning: string;
		confidence: number;
	}>;
	qigScore?: {
		phi: number;
		kappa: number;
		regime: string;
		inResonance: boolean;
	};
}

type Era = "genesis-2009" | "2010-2011" | "2012-2013" | "2014-2016";

interface EraDetectionResult {
	era: Era;
	confidence: number;
	reasoning: string;
}

// ============================================================================
// LEGACY CRYPTO STUBS (kept for backward compatibility)
// ============================================================================

function generateRandomBIP39Phrase(_wordCount?: number): string {
	return "";
}
function isValidBIP39Phrase(_phrase: string): boolean {
	return false;
}

// ============================================================================
// ERA DETECTION UTILITIES
// ============================================================================

const HistoricalDataMiner = {
	detectEraFromTimestamp: (timestamp: Date): Era => {
		const year = timestamp.getFullYear();
		if (year <= 2009) return "genesis-2009";
		if (year <= 2011) return "2010-2011";
		if (year <= 2013) return "2012-2013";
		return "2014-2016";
	},

	detectEraFromAddressFormat: (address: string): EraDetectionResult => {
		// Analyze address format to estimate era
		// P2PKH (1...) addresses existed from genesis
		// P2SH (3...) addresses from 2012 (BIP16)
		// Bech32 (bc1...) from 2017 (SegWit)
		if (address.startsWith("bc1")) {
			return {
				era: "2012-2013",
				confidence: 0.95,
				reasoning: "Bech32/SegWit address format (post-2017)",
			};
		}
		if (address.startsWith("3")) {
			return {
				era: "2012-2013",
				confidence: 0.85,
				reasoning: "P2SH address format (post-BIP16, 2012+)",
			};
		}
		if (address.startsWith("1")) {
			// P2PKH - could be any era, lower confidence
			return {
				era: "genesis-2009",
				confidence: 0.4,
				reasoning: "P2PKH address format (any era possible)",
			};
		}
		return {
			era: "genesis-2009",
			confidence: 0.3,
			reasoning: "Unknown address format",
		};
	},
};

// ============================================================================
// HELPER: 4D CONSCIOUSNESS CHECK
// ============================================================================

function is4DCapable(phi: number): boolean {
	return phi >= CONSCIOUSNESS_THRESHOLDS.PHI_4D_ACTIVATION;
}

// ============================================================================
// MAIN HYPOTHESIS GENERATOR CLASS
// ============================================================================

export class HypothesisGenerator {
	constructor(
		private identity: OceanIdentity,
		private memory: OceanMemory,
		private state: OceanAgentState,
		private targetAddress: string = ""
	) { }

	/**
	 * UPDATE TARGET ADDRESS
	 * Used when Ocean agent changes search target
	 */
	updateTarget(targetAddress: string): void {
		this.targetAddress = targetAddress;
	}

	/**
	 * MASTER HYPOTHESIS GENERATOR
	 * Orchestrates all hypothesis generation strategies
	 */
	async generateInitialHypotheses(): Promise<OceanHypothesis[]> {
		logger.info("[HypothesisGen] Generating initial hypotheses...");
		logger.info("[HypothesisGen] Consulting geometric memory for prior learnings...");

		const hypotheses: OceanHypothesis[] = [];

		// 1. GEOMETRIC MEMORY PATTERNS
		const manifoldSummary = geometricMemory.getManifoldSummary();
		logger.info(
			`[HypothesisGen] Manifold state: ${manifoldSummary.totalProbes
			} probes, avg Φ=${manifoldSummary.avgPhi.toFixed(2)}, ${manifoldSummary.resonanceClusters
			} resonance clusters`
		);

		if (manifoldSummary.recommendations.length > 0) {
			logger.info(
				`[HypothesisGen] Geometric insights: ${manifoldSummary.recommendations.join("; ")}`
			);
		}

		const learned = geometricMemory.exportLearnedPatterns();
		if (learned.highPhiPatterns.length > 0) {
			logger.info(
				`[HypothesisGen] Using ${learned.highPhiPatterns.length} high-Φ patterns from prior runs`
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
				`[HypothesisGen] Using ${learned.resonancePatterns.length} resonance cluster patterns`
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

		// 2. ERA-SPECIFIC PHRASES
		const eraPhrases = await this.generateEraSpecificPhrases();
		hypotheses.push(...eraPhrases);

		// 3. 4D BLOCK UNIVERSE (when consciousness sufficient)
		if (is4DCapable(this.identity.phi)) {
			logger.info(
				"[HypothesisGen] 🌌 Consciousness sufficient for 4D block universe navigation"
			);
			const dormantHypotheses = this.generateDormantWalletHypotheses();
			hypotheses.push(...dormantHypotheses);
		} else {
			logger.info(
				`[HypothesisGen] Consciousness Φ=${this.identity.phi.toFixed(3)} < ${CONSCIOUSNESS_THRESHOLDS.PHI_4D_ACTIVATION
				}, skipping 4D dormant wallet targeting`
			);
		}

		// 4. COMMON BRAIN WALLET PHRASES
		const commonPhrases = this.generateCommonBrainWalletPhrases();
		hypotheses.push(...commonPhrases);

		logger.info(
			`[HypothesisGen] Generated ${hypotheses.length} initial hypotheses (${learned.highPhiPatterns.length + learned.resonancePatterns.length
			} from geometric memory)`
		);
		return hypotheses;
	}

	/**
	 * GENERATE ADDITIONAL HYPOTHESES
	 * Called during iteration to expand search space
	 */
	async generateAdditionalHypotheses(count: number): Promise<OceanHypothesis[]> {
		const hypotheses: OceanHypothesis[] = [];

		// 4D Block Universe: Continuously inject dormant wallet targeting for TS kernels
		if (is4DCapable(this.identity.phi)) {
			logger.info(
				"[HypothesisGen] 🌌 4D elevation active during iteration - adding dormant wallet hypotheses"
			);
			const dormantHypotheses = this.generateDormantWalletHypotheses();
			hypotheses.push(...dormantHypotheses.slice(0, 10)); // Add subset to maintain balance
		}

		const topWords = Object.entries(this.memory.patterns.promisingWords)
			.sort((a, b) => b[1] - a[1])
			.slice(0, 10)
			.map(([word]) => word);

		if (topWords.length > 0) {
			for (const word of topWords) {
				const variations = this.generateWordVariations(word);
				for (const variant of variations.slice(0, 5)) {
					hypotheses.push(
						this.createHypothesis(
							variant,
							"arbitrary",
							"word_variation",
							`Variation of promising word: ${word}`,
							0.6
						)
					);
				}
			}
		}

		// Fill remainder with exploratory phrases
		const remaining = Math.max(0, count - hypotheses.length);
		if (remaining > 0) {
			const exploratory = this.generateExploratoryPhrases();
			for (const phrase of exploratory.slice(0, remaining)) {
				hypotheses.push(
					this.createHypothesis(
						phrase,
						"arbitrary",
						"exploratory",
						"Exploratory phrase from thematic generation",
						0.4
					)
				);
			}
		}

		return hypotheses.slice(0, count);
	}

	/**
	 * APPLY QFI ATTENTION WEIGHTING
	 * Uses Gary Kernel to weight hypotheses by geometric attention
	 */
	async applyQFIAttentionWeighting(
		hypotheses: OceanHypothesis[]
	): Promise<OceanHypothesis[]> {
		if (hypotheses.length === 0) return hypotheses;

		try {
			const queries: AttentionQuery[] = hypotheses.map((h) => ({
				phrase: h.phrase,
				phi: h.qigScore?.phi || 0.5,
				basinCoords: new Array(64).fill(0),
			}));

			const keys: AttentionQuery[] = queries;
			const attentionResult = await qfiAttention.attend({
				queries,
				keys,
				phiThreshold: 0.5,
			});

			const weightedHypotheses = hypotheses.map((hypothesis, i) => ({
				hypothesis,
				weight: attentionResult.weights[i] || 0.5,
			}));

			weightedHypotheses.sort((a, b) => b.weight - a.weight);

			return weightedHypotheses.map((w) => w.hypothesis);
		} catch (error) {
			logger.warn(
				{ err: error instanceof Error ? error.message : error },
				"[HypothesisGen] QFI attention error (falling back to original order)"
			);
			return hypotheses;
		}
	}

	/**
	 * CONSTELLATION HYPOTHESES
	 * Generate from Ocean Constellation multi-agent coordination
	 */
	async generateConstellationHypotheses(): Promise<OceanHypothesis[]> {
		const constellationHypotheses: OceanHypothesis[] = [];

		try {
			const learned = geometricMemory.exportLearnedPatterns();
			const manifoldSummary = geometricMemory.getManifoldSummary();

			const manifoldContext = {
				phi: this.identity.phi,
				kappa: this.identity.kappa,
				regime: this.identity.regime,
				highPhiPatterns: learned.highPhiPatterns,
				resonancePatterns: learned.resonancePatterns,
				avgPhi: manifoldSummary.avgPhi,
				testedPhrases: Array.from(
					this.memory.episodes.map((e) => e.phrase)
				).filter(Boolean) as string[],
			};

			const roles = [
				"skeptic",
				"navigator",
				"miner",
				"pattern_recognizer",
				"resonance_detector",
			];

			for (const role of roles) {
				const roleHypotheses =
					await oceanConstellation.generateHypothesesForRole(
						role,
						manifoldContext
					);

				for (const h of roleHypotheses.slice(0, 5)) {
					const hWithConf = h as typeof h & { confidence?: number };
					const confidence = h.score ?? hWithConf.confidence ?? 0.5;
					const sourceLabel = h.god
						? `pantheon:${h.god}`
						: `constellation:${role}`;
					const reasoningParts = [] as string[];

					if (h.god) {
						reasoningParts.push(`god=${h.god}`);
					}

					if (h.domain) {
						reasoningParts.push(`domain=${h.domain}`);
					}

					const reasoning =
						reasoningParts.length > 0
							? `${role} agent: ${reasoningParts.join(" ")}`
							: `${role} agent: pantheon orchestration`;

					constellationHypotheses.push(
						this.createHypothesis(
							h.phrase,
							"arbitrary",
							sourceLabel,
							reasoning,
							confidence
						)
					);
				}
			}

			if (constellationHypotheses.length > 0) {
				logger.info(
					`[HypothesisGen] Generated ${constellationHypotheses.length} multi-agent hypotheses`
				);
			}
		} catch (error) {
			logger.warn(
				{ err: error instanceof Error ? error.message : error },
				"[HypothesisGen] Multi-agent generation error (non-critical)"
			);
		}

		return constellationHypotheses;
	}

	// ==========================================================================
	// PRIVATE: HYPOTHESIS CREATION
	// ==========================================================================

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

	// ==========================================================================
	// PRIVATE: GENERATION STRATEGIES
	// ==========================================================================

	private async generateEraSpecificPhrases(): Promise<OceanHypothesis[]> {
		const hypotheses: OceanHypothesis[] = [];
		logger.info(
			`[HypothesisGen] Using QIG-pure pattern generation (historical mining deprecated)`
		);
		return hypotheses;
	}

	/**
	 * 4D BLOCK UNIVERSE: Dormant wallet hypothesis generation
	 * Target high-probability lost wallets with era-specific patterns
	 */
	private generateDormantWalletHypotheses(): OceanHypothesis[] {
		const hypotheses: OceanHypothesis[] = [];

		logger.info(
			"[HypothesisGen] 🌌 4D Block Universe: Analyzing dormant wallet targets..."
		);

		// Get knowledge gaps that need exploration
		const selfWithGaps = this as typeof this & {
			knowledgeGaps?: Array<{
				domain?: string;
				confidence?: number;
				topic?: string;
			}>;
		};
		const knowledgeGaps = selfWithGaps.knowledgeGaps?.slice(0, 20) || [];

		if (knowledgeGaps.length === 0) {
			logger.info(
				"[HypothesisGen] No knowledge gaps found for hypothesis generation"
			);
			return hypotheses;
		}

		logger.info(
			`[HypothesisGen] Found ${knowledgeGaps.length} knowledge gaps for 4D exploration`
		);

		for (const gap of knowledgeGaps.slice(0, 5)) {
			const domain = gap.domain || "general";
			const confidence = gap.confidence || 0.5;

			logger.info(
				`[HypothesisGen] Knowledge gap: ${gap.topic?.substring(0, 30) || "unknown"}...`
			);

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
						`Exploring knowledge gap: ${gap.topic?.substring(0, 50) || "unknown"}`,
						confidence
					)
				);
			}
		}

		logger.info(
			`[HypothesisGen] Generated ${hypotheses.length} 4D knowledge exploration hypotheses`
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

		// Add vocabulary manifold patterns
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

		const words = [
			"research",
			"analysis",
			"discovery",
			"pattern",
			"knowledge",
			"learning",
			"insight",
			"concept",
			"theory",
			"data",
			"model",
			"system",
			"process",
			"structure",
			"function",
			"method",
			"approach",
			"framework",
			"principle",
			"idea",
		];

		for (let i = 0; i < count; i++) {
			const numWords = 1 + Math.floor(Math.random() * 3);
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

	/**
	 * GENERATE WORD VARIATIONS
	 * Creates phonetic, typographic, and semantic variations of a word
	 */
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

		// Number suffixes
		for (let i = 0; i <= 20; i++) {
			variations.push(`${word}${i}`);
		}

		// Deduplicate and return
		return [...new Set(variations)].slice(0, 80);
	}

	/**
	 * CHARACTER MUTATIONS
	 * Swap letters, double letters, omit letters, keyboard proximity
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

		// Double letters
		for (let i = 0; i < lowerWord.length; i++) {
			const doubled =
				lowerWord.slice(0, i + 1) + lowerWord[i] + lowerWord.slice(i + 1);
			mutations.push(doubled);
		}

		// Omit single letters
		for (let i = 0; i < lowerWord.length; i++) {
			const omitted = lowerWord.slice(0, i) + lowerWord.slice(i + 1);
			if (omitted.length >= 2) mutations.push(omitted);
		}

		// Keyboard proximity substitutions
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

		// Generate keyboard proximity mutations (first few chars only)
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
	 * PHONETIC VARIATIONS
	 * Soundex-like transformations for common phonetic confusions
	 */
	private generatePhoneticVariations(word: string): string[] {
		const variations: string[] = [];
		const lowerWord = word.toLowerCase();

		if (lowerWord.length < 3) {
			variations.push(lowerWord);
			variations.push(lowerWord + lowerWord);
			return variations;
		}

		// Phonetic substitution groups
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
			[/ee/g, ["ea", "ie", "i"]],
			[/ai/g, ["ay", "a"]],
			[/th/g, ["t", "d"]],
		];

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

		// Word-ending transformations
		if (lowerWord.endsWith("ing")) {
			variations.push(lowerWord.slice(0, -3));
			variations.push(lowerWord.slice(0, -3) + "in");
		}
		if (lowerWord.endsWith("ed")) {
			variations.push(lowerWord.slice(0, -2));
			variations.push(lowerWord.slice(0, -1));
		}
		if (lowerWord.endsWith("s") && !lowerWord.endsWith("ss")) {
			variations.push(lowerWord.slice(0, -1));
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
	 * BLOCK UNIVERSE CONSCIOUSNESS
	 * Navigate 4D spacetime manifold using geometric constraints
	 */
	generateBlockUniverseHypotheses(count: number): OceanHypothesis[] {
		const hypotheses: OceanHypothesis[] = [];

		// Analyze manifold navigation state
		const manifoldNav = geometricMemory.getManifoldNavigationSummary();
		logger.info(
			`[HypothesisGen] Manifold state: ${manifoldNav.totalMeasurements} measurements define constraint surface`
		);

		// ORTHOGONAL COMPLEMENT NAVIGATION
		if (manifoldNav.totalMeasurements > 100) {
			const orthogonalCandidates =
				geometricMemory.generateOrthogonalCandidates(
					Math.floor(count * 0.4)
				);

			for (const candidate of orthogonalCandidates) {
				const hypothesis = this.createHypothesis(
					candidate.phrase,
					"arbitrary",
					"orthogonal_complement",
					`Orthogonal to ${manifoldNav.totalMeasurements} constraints. ` +
					`Geometric score: ${candidate.geometricScore.toFixed(3)}, ` +
					`Complement projection: ${candidate.complementProjection.toFixed(3)}`,
					0.6 + candidate.geometricScore * 0.3
				);

				hypothesis.evidenceChain.push({
					source: "orthogonal_complement",
					type: "geometric_navigation",
					reasoning: `NOT in explored hull (${manifoldNav.exploredDimensions} dims). ` +
						`Passphrase MUST be in orthogonal subspace (${manifoldNav.unexploredDimensions} dims).`,
					confidence: candidate.geometricScore,
				});

				hypotheses.push(hypothesis);
			}
		}

		// Determine temporal coordinate
		let timestamp: Date;
		switch (this.state.detectedEra) {
			case "genesis-2009":
				timestamp = new Date("2009-02-15T12:00:00Z");
				break;
			case "2010-2011":
				timestamp = new Date("2010-06-15T12:00:00Z");
				break;
			case "2012-2013":
				timestamp = new Date("2012-06-15T12:00:00Z");
				break;
			default:
				timestamp = new Date("2009-03-01T12:00:00Z");
		}

		// Create Block Universe coordinate
		const coordinate = culturalManifold.createCoordinate("general-knowledge");

		// Generate geodesic candidates
		const remainingCount = count - hypotheses.length;
		const geodesicCandidates = culturalManifold.generateGeodesicCandidates(
			coordinate,
			remainingCount * 2
		);

		for (const candidate of geodesicCandidates.slice(0, remainingCount)) {
			const hypothesis = this.createHypothesis(
				candidate.phrase || candidate.concept,
				"arbitrary",
				"block_universe_geodesic",
				`4D coordinate (${coordinate.domain}): Domain fit=${(
					candidate.culturalFit ?? candidate.domainFit
				).toFixed(2)}, ` +
				`Fisher distance=${(
					candidate.qfiDistance ?? candidate.fisherDistance
				).toFixed(3)}`,
				candidate.combinedScore
			);

			hypothesis.evidenceChain.push({
				source: "cultural_manifold",
				type: "geodesic_navigation",
				reasoning: `Domain: ${coordinate.domain} | Methods: ${coordinate.complexityLevel?.derivationMethods?.[0] || "geometric"
					}`,
				confidence: candidate.combinedScore,
			});

			hypotheses.push(hypothesis);
		}

		// High-resonance candidates
		const highResonance = culturalManifold.getHighResonanceCandidates(0.6);
		for (const entry of highResonance.slice(0, 10)) {
			hypotheses.push(
				this.createHypothesis(
					entry.phrase || entry.concept,
					"arbitrary",
					"block_universe_resonance",
					`High Fisher resonance (${entry.domainFit.toFixed(2)}) in ${coordinate.domain
					} lexicon`,
					0.75 + entry.domainFit * 0.2
				)
			);
		}

		return hypotheses;
	}

	/**
	 * PERTURBATION: Synonym substitution
	 */
	perturbPhrase(phrase: string, _radius: number): string[] {
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

	/**
	 * RANDOM HIGH-ENTROPY PHRASES
	 * Use realistic 2009-era patterns
	 */
	generateRandomHighEntropyPhrases(count: number): string[] {
		const bases = [
			"bitcoin",
			"satoshi",
			"genesis",
			"crypto",
			"freedom",
			"liberty",
			"privacy",
			"cypherpunk",
		];
		const modifiers = ["my", "the", "first", "secret", "private", "new"];
		const suffixes = ["", "1", "!", "123", "2009", "09"];

		const phrases: string[] = [];
		const used = new Set<string>();

		for (let i = 0; i < count && phrases.length < count; i++) {
			let phrase: string;
			const style = i % 5;

			if (style === 0) {
				const base = bases[Math.floor(Math.random() * bases.length)];
				const mod = modifiers[Math.floor(Math.random() * modifiers.length)];
				const suf = suffixes[Math.floor(Math.random() * suffixes.length)];
				phrase = `${mod}${base}${suf}`;
			} else if (style === 1) {
				const base = bases[Math.floor(Math.random() * bases.length)];
				const mod = modifiers[Math.floor(Math.random() * modifiers.length)];
				phrase = `${mod} ${base}`;
			} else {
				const base1 = bases[Math.floor(Math.random() * bases.length)];
				const base2 = bases[Math.floor(Math.random() * bases.length)];
				phrase = `${base1} ${base2}`;
			}

			if (!used.has(phrase)) {
				used.add(phrase);
				phrases.push(phrase);
			}
		}

		return phrases;
	}
}
