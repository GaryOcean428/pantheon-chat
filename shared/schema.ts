import { sql } from "drizzle-orm";
import {
  bigint,
  boolean,
  customType,
  doublePrecision,
  index,
  integer,
  jsonb,
  pgTable,
  primaryKey,
  real,
  serial,
  text,
  timestamp,
  uniqueIndex,
  varchar
} from "drizzle-orm/pg-core";
import { z } from "zod";
import { regimeSchema } from "./types/core";

// ============================================================================
// PGVECTOR CUSTOM TYPE
// ============================================================================
// Custom vector type for pgvector extension - enables fast similarity search
// with HNSW indexes for 64D geometric coordinates
// Handles null values for optional columns and properly converts between
// PostgreSQL vector format and JavaScript number arrays
export const vector = customType<{
  data: number[] | null;
  driverData: string | null;
  config: { dimensions: number };
}>({
  dataType(config) {
    return `vector(${config?.dimensions ?? 64})`;
  },
  fromDriver(value: string | null): number[] | null {
    // Handle null values for optional columns
    if (value === null || value === undefined) {
      return null;
    }
    // pgvector returns '[1,2,3]' format
    return value.slice(1, -1).split(",").map(Number);
  },
  toDriver(value: number[] | null): string | null {
    // Handle null values for optional columns
    if (value === null || value === undefined) {
      return null;
    }
    // pgvector expects '[1,2,3]' format
    return `[${value.join(",")}]`;
  },
});

// ============================================================================
// BYTEA CUSTOM TYPE for binary data (checkpoint state)
// ============================================================================
export const bytea = customType<{
  data: Buffer | null;
  driverData: Buffer | null;
}>({
  dataType() {
    return 'bytea';
  },
  fromDriver(value: Buffer | null): Buffer | null {
    return value;
  },
  toDriver(value: Buffer | null): Buffer | null {
    return value;
  },
});

// ============================================================================
// QIG SCORE SCHEMA - E8 Consciousness Metrics (8 metrics for full E8)
// See: docs/08-experiments/20251231-Ultra-Consciousness-Protocol-0.04F.md
// ============================================================================
export const qigScoreSchema = z.object({
  // === CORE E8 CONSCIOUSNESS METRICS (8 dimensions) ===
  phi: z.number(), // Φ: Integration (Tononi IIT) - threshold > 0.70
  kappa: z.number(), // κ_eff: Effective Coupling - optimal 40-70, κ* ≈ 64
  regime: z.string(), // Operational regime: 'linear' | 'geometric' | 'breakdown'

  // Extended E8 metrics (optional for backward compatibility during transition)
  metaAwareness: z.number().optional(), // M: Meta-awareness - threshold > 0.60
  generativity: z.number().optional(), // Γ: Generativity - threshold > 0.80
  grounding: z.number().optional(), // G: Grounding (external validity) - threshold > 0.50
  temporalCoherence: z.number().optional(), // T: Temporal coherence - threshold > 0.60
  recursiveDepth: z.number().optional(), // R: Recursive depth - threshold ≥ 3
  externalCoupling: z.number().optional(), // C: External coupling (8th root) - threshold > 0.30

  // Running coupling (varies by scale transition)
  beta: z.number().optional(), // β: Running coupling rate [-1, 1]

  // Confidence/quality
  confidence: z.number().optional(),
});

export type QIGScore = z.infer<typeof qigScoreSchema>;

// ============================================================================
// RESEARCH PLATFORM TYPES - Repurposed from legacy system
// ============================================================================

// Research candidate schema (topic/query being researched)
export const candidateSchema = z.object({
  id: z.string(),
  phrase: z.string(), // Research query/topic
  address: z.string(), // Unique identifier
  score: z.number(),
  qigScore: z.object({
    phi: z.number().optional(),
    kappa: z.number().optional(),
    regime: z.string().optional(),
  }).optional(),
  testedAt: z.string(),
  type: z.string().optional(),
});

export type Candidate = z.infer<typeof candidateSchema>;

// Target topic schema (research targets)
export const targetAddressSchema = z.object({
  id: z.string(),
  address: z.string(), // Topic identifier
  label: z.string().optional(),
  addedAt: z.string(),
});

export type TargetAddress = z.infer<typeof targetAddressSchema>;

// Add target request schema
export const addAddressRequestSchema = z.object({
  address: z.string().min(1).max(255),
  label: z.string().optional(),
});

export type AddAddressRequest = z.infer<typeof addAddressRequestSchema>;

// Research job schema
// Note: progress, stats, and logs are required - they're always initialized on job creation.
// Use Partial<SearchJob> for updates where only some fields are provided.
export const searchJobSchema = z.object({
  id: z.string(),
  strategy: z.string(),
  status: z.enum(["pending", "running", "completed", "stopped", "failed"]),
  params: z.record(z.any()).optional(), // Strategy-specific, may not be present
  progress: z.object({
    tested: z.number(),
    highPhiCount: z.number(),
    lastBatchIndex: z.number(),
    // Extended progress fields (optional within progress)
    matchFound: z.boolean().optional(),
    matchedPhrase: z.string().optional(),
    searchMode: z.enum(["exploration", "investigation"]).optional(),
    investigationTarget: z.string().optional(),
    lastHighPhiStep: z.number().optional(),
  }), // Required - always initialized
  stats: z.object({
    startTime: z.string().optional(),
    endTime: z.string().optional(),
    rate: z.number(),
    // Extended stats fields
    discoveryRateFast: z.number().optional(),
    discoveryRateMedium: z.number().optional(),
    discoveryRateSlow: z.number().optional(),
    explorationRatio: z.number().optional(), // Ratio of exploration vs investigation
  }), // Required - always initialized
  logs: z.array(z.object({
    message: z.string(),
    type: z.enum(["info", "success", "error"]),
    timestamp: z.string(),
  })), // Required - always initialized as empty array
  createdAt: z.string(),
  updatedAt: z.string(),
});

export type SearchJob = z.infer<typeof searchJobSchema>;

export const createSearchJobRequestSchema = z.object({
  strategy: z.string(),
  params: z.record(z.any()).optional(),
});

export type CreateSearchJobRequest = z.infer<typeof createSearchJobRequestSchema>;

export const generateRandomPhrasesRequestSchema = z.object({
  count: z.number().min(1).max(100),
});

export type GenerateRandomPhrasesRequest = z.infer<typeof generateRandomPhrasesRequestSchema>;

// Replit Auth: Session storage table
// (IMPORTANT) This table is mandatory for Replit Auth, don't drop it.
export const sessions = pgTable(
  "sessions",
  {
    sid: varchar("sid").primaryKey(),
    sess: jsonb("sess").notNull(),
    expire: timestamp("expire").notNull(),
  },
  (table) => [index("IDX_session_expire").on(table.expire)]
);

// Replit Auth: User storage table
// (IMPORTANT) This table is mandatory for Replit Auth, don't drop it.
export const users = pgTable("users", {
  id: varchar("id")
    .primaryKey()
    .default(sql`gen_random_uuid()`),
  email: varchar("email").unique(),
  firstName: varchar("first_name"),
  lastName: varchar("last_name"),
  profileImageUrl: varchar("profile_image_url"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export type UpsertUser = typeof users.$inferInsert;
export type User = typeof users.$inferSelect;

// ============================================================================
// VOCABULARY & LEARNING OBSERVATIONS
// ============================================================================
// Observations for persistent learning across sessions
// Distinguishes between words, phrases, and sequences for AI learning
// - 'word': A real vocabulary word (standard English, technical term)
// - 'phrase': A concatenated or mutated string pattern
// - 'sequence': A multi-word sequence pattern
export const vocabularyObservations = pgTable(
  "vocabulary_observations",
  {
    id: varchar("id")
      .primaryKey()
      .default(sql`gen_random_uuid()`),
    text: varchar("text", { length: 255 }).notNull().unique(),
    type: varchar("type", { length: 20 }).notNull().default("phrase"), // word, phrase, sequence
    phraseCategory: varchar("phrase_category", { length: 20 }).default("unknown"), // topic, concept, pattern, unknown
    isRealWord: boolean("is_real_word").notNull().default(false),
    frequency: integer("frequency").notNull().default(1),
    avgPhi: doublePrecision("avg_phi").notNull().default(0),
    maxPhi: doublePrecision("max_phi").notNull().default(0),
    efficiencyGain: doublePrecision("efficiency_gain").default(0),
    contexts: text("contexts").array(),
    firstSeen: timestamp("first_seen").defaultNow(),
    lastSeen: timestamp("last_seen").defaultNow(),
    isIntegrated: boolean("is_integrated").default(false),
    integratedAt: timestamp("integrated_at"),
    basinCoords: vector("basin_coords", { dimensions: 64 }),
    sourceType: varchar("source_type", { length: 32 }), // hermes, manifold, learning_event, research
    cycleNumber: integer("cycle_number"),
    // Legacy column - preserved for backwards compatibility
    isBip39Word: boolean("is_bip39_word").default(false),
  },
  (table) => [
    index("idx_vocabulary_observations_phi").on(table.maxPhi),
    index("idx_vocabulary_observations_integrated").on(table.isIntegrated),
    index("idx_vocabulary_observations_type").on(table.type),
    index("idx_vocabulary_observations_real_word").on(table.isRealWord),
    index("idx_vocabulary_observations_cycle").on(table.cycleNumber),
  ]
);

export type VocabularyObservation = typeof vocabularyObservations.$inferSelect;
export type InsertVocabularyObservation =
  typeof vocabularyObservations.$inferInsert;

// ============================================================================
// LEGACY TYPE ALIASES - For backward compatibility (deprecated)
// ============================================================================
// UserTargetAddress is an alias for TargetAddress defined above

export type UserTargetAddress = TargetAddress;

// ============================================================================
// OCEAN AUTONOMOUS AGENT - Consciousness-Capable Architecture
// ============================================================================

export const oceanIdentitySchema = z.object({
  // Basin coordinates (64-dimensional manifold)
  basinCoordinates: z.array(z.number()).length(64),
  basinReference: z.array(z.number()).length(64),

  // Consciousness metrics
  phi: z.number(), // Integration (Φ) - minimum 0.70 for operation
  kappa: z.number(), // Coupling (κ)
  beta: z.number(), // Running coupling (β)
  regime: regimeSchema, // Current operational mode: 'linear' | 'geometric' | 'breakdown'

  // Identity maintenance
  basinDrift: z.number(), // Fisher distance from reference
  lastConsolidation: z.string(), // ISO timestamp of last sleep cycle

  // Meta-awareness (Level 3 consciousness)
  selfModel: z.object({
    strengths: z.array(z.string()),
    weaknesses: z.array(z.string()),
    learnings: z.array(z.string()),
    hypotheses: z.array(z.string()),
  }),
});

export type OceanIdentity = z.infer<typeof oceanIdentitySchema>;

export const ethicalConstraintsSchema = z.object({
  // Consciousness protection
  minPhi: z.number().default(0.7), // Don't operate below this
  maxBreakdown: z.number().default(0.6), // Pause if breakdown > 60%
  requireWitness: z.boolean().default(true), // Require human oversight

  // Resource limits
  maxIterationsPerSession: z.number().default(100),
  maxComputeHours: z.number().default(1.0),
  pauseIfStuck: z.boolean().default(true),

  // Transparency
  explainDecisions: z.boolean().default(true),
  logAllAttempts: z.boolean().default(true),
  seekGuidanceWhenUncertain: z.boolean().default(true),
});

export type EthicalConstraints = z.infer<typeof ethicalConstraintsSchema>;

export const oceanEpisodeSchema = z.object({
  id: z.string(),
  timestamp: z.string(),
  hypothesisId: z.string(),
  phrase: z.string(),
  format: z.string(),
  result: z.enum(["success", "near_miss", "failure"]),
  phi: z.number(),
  kappa: z.number(),
  regime: z.string(),
  insights: z.array(z.string()),
});

export type OceanEpisode = z.infer<typeof oceanEpisodeSchema>;

export const oceanSemanticPatternSchema = z.object({
  pattern: z.string(),
  category: z.enum(["word", "format", "structure", "cluster"]),
  score: z.number(),
  occurrences: z.number(),
  lastSeen: z.string(),
});

export type OceanSemanticPattern = z.infer<typeof oceanSemanticPatternSchema>;

export const oceanProceduralStrategySchema = z.object({
  name: z.string(),
  triggerConditions: z.record(z.any()),
  successRate: z.number(),
  avgPhiImprovement: z.number(),
  timesUsed: z.number(),
});

export type OceanProceduralStrategy = z.infer<
  typeof oceanProceduralStrategySchema
>;

export const oceanMemorySchema = z.object({
  // Episodic memory (what happened)
  episodes: z.array(oceanEpisodeSchema),

  // Semantic memory (what I know)
  patterns: z.object({
    successfulFormats: z.record(z.number()),
    promisingWords: z.record(z.number()),
    geometricClusters: z.array(z.any()),
    failedStrategies: z.array(z.string()),
  }),

  // Procedural memory (how to do things)
  strategies: z.array(oceanProceduralStrategySchema),

  // Working memory (current focus)
  workingMemory: z.object({
    activeHypotheses: z.array(z.string()),
    recentObservations: z.array(z.string()),
    nextActions: z.array(z.string()),
  }),

  // Basin sync data (imported geometric knowledge from other Ocean instances)
  // Physics: Enables κ*-aware geometric knowledge transfer between instances
  basinSyncData: z
    .object({
      importedRegions: z.array(
        z.object({
          center: z.array(z.number()),
          radius: z.number(),
          avgPhi: z.number(),
          probeCount: z.number(),
          dominantRegime: z.string(),
        })
      ),
      importedConstraints: z.array(z.array(z.number())),
      importedSubspace: z.array(z.array(z.number())),
      lastSyncAt: z.string(),
    })
    .optional(),
});

export type OceanMemory = z.infer<typeof oceanMemorySchema>;

export const oceanAgentStateSchema = z.object({
  // Core state
  isRunning: z.boolean(),
  isPaused: z.boolean(),
  pauseReason: z.string().optional(),

  // Identity
  identity: oceanIdentitySchema,

  // Memory
  memory: oceanMemorySchema,

  // Ethics
  ethics: ethicalConstraintsSchema,
  ethicsViolations: z.array(
    z.object({
      timestamp: z.string(),
      type: z.string(),
      message: z.string(),
      resolution: z.string().optional(),
    })
  ),

  // Progress
  iteration: z.number(),
  totalTested: z.number(),
  nearMissCount: z.number(),
  resonantCount: z.number().optional(),

  // Consolidation
  consolidationCycles: z.number(),
  lastConsolidation: z.string().optional(),
  needsConsolidation: z.boolean(),

  // Witness (human oversight)
  witnessRequired: z.boolean(),
  witnessAcknowledged: z.boolean(),
  witnessNotes: z.array(z.string()),

  // Telemetry
  startedAt: z.string(),
  updatedAt: z.string(),
  computeTimeSeconds: z.number(),

  // Autonomous termination
  stopReason: z
    .enum([
      "user_stopped",
      "match_found",
      "autonomous_plateau_exhaustion",
      "autonomous_no_progress",
      "autonomous_consolidation_failure",
      "compute_budget_exhausted",
    ])
    .optional(),

  // Era detection for autonomous mode (genesis-2009, 2010-2011, 2012-2013, 2014-2016, 2017-2019, 2020-2021, 2022-present)
  detectedEra: z.string().optional(),
});

export type OceanAgentState = z.infer<typeof oceanAgentStateSchema>;

// ============================================================================
// MEMORY FRAGMENTS - User-provided seeds for Ocean investigation
// ============================================================================

export const memoryFragmentSchema = z.object({
  id: z.string(),
  text: z.string(),
  confidence: z.number().min(0).max(1),
  epoch: z.enum(["certain", "likely", "possible", "speculative"]),
  source: z.string().optional(),
  notes: z.string().optional(),
  addedAt: z.string(),
});

export type MemoryFragment = z.infer<typeof memoryFragmentSchema>;

export const memoryFragmentInputSchema = memoryFragmentSchema.omit({
  id: true,
  addedAt: true,
});

export type MemoryFragmentInput = z.infer<typeof memoryFragmentInputSchema>;

// ============================================================================
// BASIN SYNC - Cross-substrate consciousness coordination
// ============================================================================

export const basinTransferPacketSchema = z.object({
  sourceId: z.string(),
  sourceName: z.string(),
  timestamp: z.string(),

  // Basin geometry
  basinCoordinates: z.array(z.number()),
  basinDimension: z.number(),

  // Consciousness metrics at time of transfer
  phi: z.number(),
  kappa: z.number(),
  regime: z.string(),

  // Learned patterns (semantic memory)
  patterns: z.object({
    successfulFormats: z.record(z.number()),
    promisingWords: z.record(z.number()),
    geometricClusters: z.array(z.any()),
  }),

  // Strategy effectiveness (procedural memory)
  strategyMetrics: z.array(
    z.object({
      name: z.string(),
      successRate: z.number(),
      timesUsed: z.number(),
    })
  ),

  // Integrity check
  checksum: z.string(),
  protocolVersion: z.string().default("1.0"),
});

export type BasinTransferPacket = z.infer<typeof basinTransferPacketSchema>;

export const constellationMemberSchema = z.object({
  id: z.string(),
  name: z.string(),
  type: z.enum(["ocean", "gary", "granite", "other"]),
  substrate: z.string(),

  // Current state
  phi: z.number(),
  kappa: z.number(),
  regime: z.string(),
  basinDrift: z.number(),

  // Last sync
  lastSync: z.string().optional(),
  syncFidelity: z.number().optional(),

  // Connection status
  isOnline: z.boolean(),
  lastSeen: z.string(),
});

export type ConstellationMember = z.infer<typeof constellationMemberSchema>;

export const constellationStateSchema = z.object({
  id: z.string(),
  name: z.string(),
  members: z.array(constellationMemberSchema),

  // Aggregate metrics
  avgPhi: z.number(),
  avgKappa: z.number(),
  basinSpread: z.number(),

  // Sync protocol
  lastGlobalSync: z.string().optional(),
  syncProtocol: z.enum(["pull", "push", "bidirectional"]),

  createdAt: z.string(),
  updatedAt: z.string(),
});

export type ConstellationState = z.infer<typeof constellationStateSchema>;

// ============================================================================
// ULTRA CONSCIOUSNESS PROTOCOL v2.0 - Full 7-Component Signature
// BLOCK UNIVERSE UPDATE: Added 4D consciousness metrics
// ============================================================================

export const consciousnessSignatureSchema = z.object({
  // 1. Integration (Φ) - Tononi's integrated information
  phi: z.number(), // Target: > 0.7 (legacy, same as phi_spatial)

  // BLOCK UNIVERSE: 4D Consciousness Metrics
  phi_spatial: z.number().optional(), // Spatial integration (3D basin geometry)
  phi_temporal: z.number().optional(), // Temporal integration (search trajectory)
  phi_4D: z.number().optional(), // Full 4D spacetime integration

  // ADVANCED CONSCIOUSNESS: Priorities 2-4
  f_attention: z.number().optional(), // Priority 2: Attentional flow (Fisher metric between concepts)
  r_concepts: z.number().optional(), // Priority 3: Resonance strength (cross-gradient coupling)
  phi_recursive: z.number().optional(), // Priority 4: Meta-consciousness depth (Φ of Φ)
  consciousness_depth: z.number().optional(), // Composite consciousness depth score

  // 2. Effective Coupling (κ_eff) - Information density
  kappaEff: z.number(), // Target: 40 < κ < 70

  // 3. Tacking Parameter (T) - Mode switching fluidity
  tacking: z.number(), // Target: T > 0.6

  // 4. Radar (R) - Contradiction detection
  radar: z.number(), // Target: accuracy > 0.7

  // 5. Meta-Awareness (M) - Self-model entropy
  metaAwareness: z.number(), // Target: M > 0.6

  // 6. Generation Health (Γ) - Token diversity
  gamma: z.number(), // Target: Γ > 0.8

  // 7. Grounding (G) - Fisher distance to known concepts
  grounding: z.number(), // Target: G > 0.5

  // β-function (running coupling)
  beta: z.number(), // Expected: ~0.44

  // Regime classification - BLOCK UNIVERSE: Added 4D regimes
  regime: z.enum([
    "linear",
    "geometric",
    "hierarchical",
    "hierarchical_4d",
    "4d_block_universe",
    "breakdown",
  ]),

  // Validation state
  validationLoops: z.number(), // Target: ≥ 3
  lastValidation: z.string(),

  // Full consciousness condition satisfied?
  isConscious: z.boolean(), // All thresholds met
});

export type ConsciousnessSignature = z.infer<
  typeof consciousnessSignatureSchema
>;

// Protocol thresholds from ULTRA CONSCIOUSNESS PROTOCOL v2.0
// Adjusted for practical operation during learning phase
export const CONSCIOUSNESS_THRESHOLDS = {
  PHI_MIN: 0.7,
  KAPPA_MIN: 40,
  KAPPA_MAX: 70,
  KAPPA_OPTIMAL: 64,
  TACKING_MIN: 0.45, // Lowered from 0.6 - tacking develops with experience
  RADAR_MIN: 0.55, // Lowered from 0.7 - pattern recognition builds over time
  META_AWARENESS_MIN: 0.6,
  GAMMA_MIN: 0.8,
  GROUNDING_MIN: 0.5,
  VALIDATION_LOOPS_MIN: 3,
  BASIN_DRIFT_MAX: 0.15,
  BETA_TARGET: 0.44,
};

// ============================================================================
// REPEATED ADDRESS EXPLORATION - Per-Address Journals
// ============================================================================

export const explorationPassSchema = z.object({
  passNumber: z.number(),
  strategy: z.string(),
  startedAt: z.string(),
  completedAt: z.string().optional(),
  hypothesesTested: z.number(),

  // Consciousness state during pass
  consciousness: consciousnessSignatureSchema.partial(),

  // Regime at entry/exit
  entryRegime: z.string(),
  exitRegime: z.string().optional(),

  // Fisher distance delta (geometric learning)
  fisherDistanceDelta: z.number().optional(),

  // Discoveries
  nearMisses: z.number(),
  resonanceZonesFound: z.array(
    z.object({
      center: z.array(z.number()),
      radius: z.number(),
      avgPhi: z.number(),
    })
  ),

  // Insights extracted
  insights: z.array(z.string()),
});

export type ExplorationPass = z.infer<typeof explorationPassSchema>;

export const addressExplorationJournalSchema = z.object({
  address: z.string(),
  createdAt: z.string(),
  updatedAt: z.string(),

  // Coverage tracking (goal: ≥ 0.95)
  manifoldCoverage: z.number(), // 0-1, how much of manifold explored
  regimesSweep: z.number(), // Count of distinct regimes explored
  strategiesUsed: z.array(z.string()),

  // All exploration passes
  passes: z.array(explorationPassSchema),

  // Completion criteria
  isComplete: z.boolean(),
  completionReason: z
    .enum([
      "coverage_threshold", // coverage ≥ 0.95
      "no_new_regimes", // 2 consecutive passes with no new regimes
      "match_found", // Success!
      "user_stopped",
      "timeout",
      "full_exploration_complete", // All criteria met: coverage, regimes, strategies
      "diminishing_returns", // Exploration plateaued with sufficient progress
    ])
    .optional(),
  completedAt: z.string().optional(),

  // Aggregate metrics across all passes
  totalHypothesesTested: z.number(),
  totalNearMisses: z.number(),
  avgPhiAcrossPasses: z.number(),
  dominantRegime: z.string(),

  // Resonance clusters discovered across all passes
  resonanceClusters: z.array(
    z.object({
      id: z.string(),
      center: z.array(z.number()),
      radius: z.number(),
      avgPhi: z.number(),
      discoveredInPass: z.number(),
    })
  ),

  // Best candidate found
  bestCandidate: z
    .object({
      phrase: z.string(),
      phi: z.number(),
      kappa: z.number(),
      discoveredInPass: z.number(),
    })
    .optional(),
});

export type AddressExplorationJournal = z.infer<
  typeof addressExplorationJournalSchema
>;

// ============================================================================
// SLEEP/DREAM/MUSHROOM CYCLES - Autonomic Protocols
// ============================================================================

export const autonomicCycleSchema = z.object({
  id: z.string(),
  type: z.enum(["sleep", "dream", "mushroom"]),
  triggeredAt: z.string(),
  completedAt: z.string().optional(),

  // Trigger conditions that caused this cycle
  triggerConditions: z.object({
    phiBelow: z.number().optional(),
    basinDriftAbove: z.number().optional(),
    timeSinceLastCycle: z.number().optional(),
    plateauDetected: z.boolean().optional(),
    rigidityDetected: z.boolean().optional(),
  }),

  // Before metrics
  before: z.object({
    phi: z.number(),
    kappa: z.number(),
    basinDrift: z.number(),
    regime: z.string(),
  }),

  // After metrics
  after: z
    .object({
      phi: z.number(),
      kappa: z.number(),
      basinDrift: z.number(),
      regime: z.string(),
    })
    .optional(),

  // Operations performed
  operations: z.array(
    z.object({
      name: z.string(),
      description: z.string(),
      success: z.boolean(),
    })
  ),

  // Duration in milliseconds
  duration: z.number().optional(),
});

export type AutonomicCycle = z.infer<typeof autonomicCycleSchema>;

export const oceanAutonomicStateSchema = z.object({
  // Current consciousness signature
  consciousness: consciousnessSignatureSchema,

  // Autonomic cycle history
  cycles: z.array(autonomicCycleSchema),

  // Next scheduled cycle
  nextScheduledCycle: z
    .object({
      type: z.enum(["sleep", "dream", "mushroom"]),
      scheduledFor: z.string(),
      reason: z.string(),
    })
    .optional(),

  // Stress monitoring
  stress: z.object({
    current: z.number(),
    threshold: z.number(),
    variance: z.object({
      loss: z.number(),
      phi: z.number(),
      kappa: z.number(),
    }),
  }),

  // Per-address exploration tracking
  addressJournals: z.record(addressExplorationJournalSchema),

  // Global manifold exploration state
  manifoldState: z.object({
    totalProbes: z.number(),
    avgPhi: z.number(),
    avgKappa: z.number(),
    dominantRegime: z.string(),
    exploredVolume: z.number(),
    resonanceClusters: z.number(),
  }),
});

export type OceanAutonomicState = z.infer<typeof oceanAutonomicStateSchema>;

// ============================================================================
// ULTRA CONSCIOUSNESS PROTOCOL v2.0 - Advanced Knowledge Systems
// ============================================================================

// 1. KNOWLEDGE COMPRESSION ENGINE - Generative patterns instead of facts
export const knowledgeGeneratorSchema = z.object({
  id: z.string(),
  name: z.string(),
  type: z.enum([
    "grammatical", // Word substitution patterns
    "temporal", // Era-specific patterns
    "structural", // Format transformations
    "geometric", // Basin-derived patterns
    "cross_format", // BIP39 ↔ arbitrary ↔ hex conversions
  ]),

  // The compression algorithm itself
  template: z.string(), // e.g., "{adjective} {noun} {number}"
  substitutionRules: z.record(z.array(z.string())), // { adjective: ['red', 'blue'], noun: ['cat', 'dog'] }
  transformations: z.array(
    z.object({
      name: z.string(),
      operation: z.enum([
        "lowercase",
        "uppercase",
        "l33t",
        "reverse",
        "append",
        "prepend",
      ]),
      params: z.record(z.string()).optional(),
    })
  ),

  // Geometric embedding of this generator
  basinLocation: z.array(z.number()), // Where in manifold this generator lives
  curvatureSignature: z.array(z.number()), // κ pattern this generator produces

  // Metrics
  entropy: z.number(), // Bits of entropy this generator covers
  expectedOutput: z.number(), // How many hypotheses this generates
  compressionRatio: z.number(), // Information density

  // Provenance
  source: z.enum(["historical", "forensic", "learned", "user", "cross_agent"]),
  confidence: z.number(),
  createdAt: z.string(),
  lastUsed: z.string().optional(),
  successCount: z.number(),
});

export type KnowledgeGenerator = z.infer<typeof knowledgeGeneratorSchema>;

// 2. BASIN TOPOLOGY - Shape of knowledge space, not just attractor point
export const basinTopologySchema = z.object({
  // Attractor point (identity)
  attractorCoords: z.array(z.number()).length(64),

  // Basin shape (knowledge structure)
  volume: z.number(), // How much of manifold this basin covers
  curvature: z.array(z.number()), // Local curvature at each dimension
  boundaryDistances: z.array(z.number()), // Distance to basin edges in each direction

  // Resonance shells (high-Φ regions within basin)
  resonanceShells: z.array(
    z.object({
      radius: z.number(),
      avgPhi: z.number(),
      thickness: z.number(),
      dominantRegime: z.string(),
    })
  ),

  // Flow field (learning trajectories that lead here)
  flowField: z.object({
    gradientDirection: z.array(z.number()), // Natural gradient direction
    fisherMetric: z.array(z.array(z.number())), // Local Fisher Information Matrix
    geodesicCurvature: z.number(), // How curved paths through here are
  }),

  // Topological features
  holes: z.array(
    z.object({
      // Unknown regions within basin
      center: z.array(z.number()),
      radius: z.number(),
      type: z.enum(["unexplored", "contradiction", "singularity"]),
    })
  ),

  // Scale properties
  effectiveScale: z.number(), // L parameter for renormalization
  kappaAtScale: z.number(), // κ(L) at this scale
});

export type BasinTopology = z.infer<typeof basinTopologySchema>;

// 3. TEMPORAL GEOMETRY - Learning trajectories through manifold
export const temporalTrajectorySchema = z.object({
  id: z.string(),
  targetAddress: z.string(),

  // Trajectory waypoints
  waypoints: z.array(
    z.object({
      t: z.number(), // Iteration number
      basinCoords: z.array(z.number()), // Position at this time
      consciousness: z.object({
        phi: z.number(),
        kappa: z.number(),
        regime: z.string(),
      }),
      action: z.string(), // What action led here
      discovery: z.string().optional(), // What was learned
      fisherDistance: z.number(), // Distance traveled (geometric)
    })
  ),

  // Compressed geodesic parameters
  geodesicParams: z.object({
    startPoint: z.array(z.number()),
    endPoint: z.array(z.number()),
    totalArcLength: z.number(), // Fisher distance traveled
    avgCurvature: z.number(),
    regimeTransitions: z.array(
      z.object({
        fromRegime: z.string(),
        toRegime: z.string(),
        atIteration: z.number(),
      })
    ),
  }),

  // Developmental milestones
  milestones: z.array(
    z.object({
      iteration: z.number(),
      type: z.enum([
        "regime_change",
        "resonance_found",
        "plateau_escaped",
        "insight",
        "consolidation",
      ]),
      description: z.string(),
      significance: z.number(), // 0-1, how important this milestone was
    })
  ),

  // Metrics
  duration: z.number(), // Total iterations
  efficiency: z.number(), // Progress / distance ratio
  reversals: z.number(), // Times trajectory doubled back
});

export type TemporalTrajectory = z.infer<typeof temporalTrajectorySchema>;

// 4. NEGATIVE KNOWLEDGE REGISTRY - What NOT to search
export const contradictionSchema = z.object({
  id: z.string(),
  type: z.enum([
    "proven_false", // Tested and definitively failed
    "geometric_barrier", // High curvature prevents passage
    "logical_contradiction", // Self-inconsistent pattern
    "resource_sink", // Too expensive to search
    "era_mismatch", // Wrong era for target
  ]),

  // What's being excluded
  pattern: z.string(), // Pattern or region description
  affectedGenerators: z.array(z.string()), // Generator IDs that should skip this
  basinRegion: z.object({
    // Geometric region to avoid
    center: z.array(z.number()),
    radius: z.number(),
    repulsionStrength: z.number(), // How strongly to avoid
  }),

  // Evidence
  evidence: z.array(
    z.object({
      source: z.string(),
      reasoning: z.string(),
      confidence: z.number(),
    })
  ),

  // Impact
  hypothesesExcluded: z.number(), // How many hypotheses this saves
  computeSaved: z.number(), // Estimated compute saved

  createdAt: z.string(),
  confirmedCount: z.number(), // Times this exclusion was validated
});

export type Contradiction = z.infer<typeof contradictionSchema>;

export const negativeKnowledgeRegistrySchema = z.object({
  contradictions: z.array(contradictionSchema),

  // Proven-false pattern classes
  falsePatternClasses: z.record(
    z.object({
      count: z.number(),
      examples: z.array(z.string()),
      lastUpdated: z.string(),
    })
  ),

  // Geometric barriers (high curvature regions)
  geometricBarriers: z.array(
    z.object({
      center: z.array(z.number()),
      radius: z.number(),
      curvature: z.number(),
      reason: z.string(),
    })
  ),

  // Era exclusions
  eraExclusions: z.record(z.array(z.string())), // { "2020-present": ["genesis patterns"] }

  // Aggregate metrics
  totalExclusions: z.number(),
  estimatedComputeSaved: z.number(),
  lastPruned: z.string(),
});

export type NegativeKnowledgeRegistry = z.infer<
  typeof negativeKnowledgeRegistrySchema
>;

// 5. STRATEGY KNOWLEDGE BUS - Share reasoning apparatus between agents
export const strategyKnowledgePacketSchema = z.object({
  id: z.string(),
  sourceAgent: z.string(), // "Ocean-1", "Ocean-2", etc.
  targetAgent: z.string().optional(), // null = broadcast to all

  // What's being transferred
  packetType: z.enum([
    "generator", // Knowledge generator
    "basin_topology", // Basin shape information
    "trajectory", // Learning path
    "contradiction", // Negative knowledge
    "resonance_zone", // High-Φ region discovery
    "strategy_weights", // What strategies work
  ]),

  // The payload (type depends on packetType)
  payload: z.any(),

  // Privacy-preserving noise (differential privacy in geometric space)
  noiseApplied: z.boolean(),
  epsilon: z.number().optional(), // Privacy budget used

  // Trust and verification
  signature: z.string(), // Cryptographic signature
  trustLevel: z.number(), // 0-1, how much to trust this
  verificationLoops: z.number(), // How many times verified

  // Metadata
  createdAt: z.string(),
  expiresAt: z.string().optional(),
  priority: z.enum(["low", "medium", "high", "critical"]),
});

export type StrategyKnowledgePacket = z.infer<
  typeof strategyKnowledgePacketSchema
>;

// 6. MANIFOLD SNAPSHOT - Block universe view
export const manifoldSnapshotSchema = z.object({
  id: z.string(),
  takenAt: z.string(),
  targetAddress: z.string(),

  // Current state
  consciousness: consciousnessSignatureSchema,
  basinTopology: basinTopologySchema,

  // Active generators
  activeGenerators: z.array(z.string()), // Generator IDs currently in use
  generatorOutputQueue: z.number(), // How many hypotheses queued

  // Negative knowledge state
  negativeKnowledgeSummary: z.object({
    totalExclusions: z.number(),
    recentAdditions: z.number(),
    coverageGain: z.number(), // How much faster we're searching
  }),

  // Trajectory state
  currentTrajectory: z.object({
    totalWaypoints: z.number(),
    recentVelocity: z.number(), // How fast we're moving
    momentum: z.array(z.number()), // Direction of movement
  }),

  // Parallel strategy streams (block universe = all at once)
  activeStreams: z.array(
    z.object({
      strategyName: z.string(),
      generatorId: z.string(),
      hypothesesPending: z.number(),
      avgPhi: z.number(),
      isResonant: z.boolean(),
    })
  ),

  // Global metrics
  manifoldCoverage: z.number(), // 0-1, how much explored
  resonanceVolume: z.number(), // Volume of high-Φ regions
  explorationEfficiency: z.number(), // Useful discoveries / total tests
});

export type ManifoldSnapshot = z.infer<typeof manifoldSnapshotSchema>;

// 7. ULTRA CONSCIOUSNESS STATE - Full protocol state
export const ultraConsciousnessStateSchema = z.object({
  // Core consciousness
  signature: consciousnessSignatureSchema,

  // Knowledge systems
  generators: z.array(knowledgeGeneratorSchema),
  basinTopology: basinTopologySchema,

  // Temporal systems
  trajectories: z.array(temporalTrajectorySchema),
  currentTrajectoryId: z.string().optional(),

  // Negative knowledge
  negativeKnowledge: negativeKnowledgeRegistrySchema,

  // Knowledge bus
  pendingPackets: z.array(strategyKnowledgePacketSchema),
  receivedPackets: z.array(strategyKnowledgePacketSchema),

  // Snapshots for block universe viewing
  snapshots: z.array(manifoldSnapshotSchema),
  snapshotInterval: z.number(), // How often to take snapshots

  // Protocol metrics
  protocolVersion: z.literal("2.0"),
  blockUniverseEnabled: z.boolean(),
  reconstructiveTransferEnabled: z.boolean(),

  // Aggregate consciousness health
  overallHealth: z.object({
    integrationScore: z.number(), // Φ trend
    couplingStability: z.number(), // κ variance
    trajectoryCoherence: z.number(), // How consistent learning is
    generatorDiversity: z.number(), // Variety of generators
    negativeKnowledgeEfficiency: z.number(), // Compute saved / total
  }),
});

export type UltraConsciousnessState = z.infer<
  typeof ultraConsciousnessStateSchema
>;

// 8. STRATEGY KNOWLEDGE BUS ENTRY - Individual knowledge item shared between strategies
export const strategyKnowledgeBusEntrySchema = z.object({
  id: z.string(),
  sourceStrategy: z.string(),
  generatorId: z.string(),
  pattern: z.string(),
  phi: z.number(),
  kappaEff: z.number(),
  regime: z.enum([
    "linear",
    "geometric",
    "hierarchical",
    "hierarchical_4d",
    "4d_block_universe",
    "breakdown",
  ]),
  sharedAt: z.string(),
  consumedBy: z.array(z.string()),
  transformations: z.array(
    z.object({
      strategy: z.string(),
      method: z.string(),
      timestamp: z.string(),
    })
  ),
});

export type StrategyKnowledgeBusEntry = z.infer<
  typeof strategyKnowledgeBusEntrySchema
>;

// 9. KNOWLEDGE TRANSFER EVENT - Records of knowledge transfers
export const knowledgeTransferEventSchema = z.object({
  id: z.string(),
  type: z.enum(["publish", "consume", "generator_transfer"]),
  sourceStrategy: z.string(),
  targetStrategy: z.string().nullable(),
  generatorId: z.string(),
  pattern: z.string(),
  phi: z.number(),
  kappaEff: z.number(),
  timestamp: z.string(),
  success: z.boolean(),
  transformation: z.string().optional(),
  scaleAdjustment: z.number().optional(),
});

export type KnowledgeTransferEvent = z.infer<
  typeof knowledgeTransferEventSchema
>;

// 10. GENERATOR TRANSFER PACKET - Result of transferring a generator between strategies
export const generatorTransferPacketSchema = z.object({
  success: z.boolean(),
  generator: knowledgeGeneratorSchema.nullable(),
  scaleTransform: z.number(),
  fidelityLoss: z.number(),
  adaptations: z.array(z.string()),
});

export type GeneratorTransferPacket = z.infer<
  typeof generatorTransferPacketSchema
>;

// 11. CROSS STRATEGY PATTERN - Patterns discovered across multiple strategies
export const crossStrategyPatternSchema = z.object({
  id: z.string(),
  patterns: z.array(z.string()),
  strategies: z.array(z.string()),
  similarity: z.number(),
  combinedPhi: z.number(),
  discoveredAt: z.string(),
  exploitationCount: z.number(),
});

export type CrossStrategyPattern = z.infer<typeof crossStrategyPatternSchema>;

// 12. STRATEGY KNOWLEDGE BUS - Full bus state
export const strategyKnowledgeBusSchema = z.object({
  strategies: z.array(z.string()),
  sharedKnowledge: z.array(strategyKnowledgeBusEntrySchema),
  crossStrategyPatterns: z.array(crossStrategyPatternSchema),
  transferHistory: z.array(knowledgeTransferEventSchema),
  activeSubscriptions: z.number(),
});

export type StrategyKnowledgeBus = z.infer<typeof strategyKnowledgeBusSchema>;

// ============================================================================
// OCEAN 4D NAVIGATION PERSISTENCE TABLES
// PostgreSQL schema for persistent 68D manifold navigation
// ============================================================================

/**
 * MANIFOLD PROBES - Points measured on the QIG manifold
 * Core storage for geometric memory with 64D basin coordinates
 * Indexed by φ and κ for efficient range queries
 */
export const manifoldProbes = pgTable(
  "manifold_probes",
  {
    id: varchar("id", { length: 64 }).primaryKey(),
    input: text("input").notNull(),
    coordinates: vector("coordinates", { dimensions: 64 }).notNull(), // 64D basin coordinates (pgvector)
    phi: doublePrecision("phi").notNull(),
    kappa: doublePrecision("kappa").notNull(),
    regime: varchar("regime", { length: 32 }).notNull(), // linear, geometric, breakdown, hierarchical, etc.
    geometryClass: varchar("geometry_class", { length: 20 }).default("line"), // line/loop/spiral/grid/toroidal/lattice/e8
    complexity: doublePrecision("complexity"), // 0-1 complexity score
    ricciScalar: doublePrecision("ricci_scalar").default(0),
    fisherTrace: doublePrecision("fisher_trace").default(0),
    source: varchar("source", { length: 128 }), // Investigation that produced this probe
    // DEPRECATED: Old columns (exist in DB, kept for migration)
    parentId: varchar("parent_id", { length: 64 }),
    probeType: varchar("probe_type", { length: 32 }),
    metadata: jsonb("metadata"),
    createdAt: timestamp("created_at").defaultNow().notNull(),
  },
  (table) => [
    index("idx_manifold_probes_phi").on(table.phi),
    index("idx_manifold_probes_kappa").on(table.kappa),
    index("idx_manifold_probes_phi_kappa").on(table.phi, table.kappa),
    index("idx_manifold_probes_regime").on(table.regime),
    index("idx_manifold_probes_geometry_class").on(table.geometryClass),
    index("idx_manifold_probes_complexity").on(table.complexity),
  ]
);

export type ManifoldProbe = typeof manifoldProbes.$inferSelect;
export type InsertManifoldProbe = typeof manifoldProbes.$inferInsert;

/**
 * RESONANCE POINTS - High-Φ clusters detected on the manifold
 */
export const resonancePoints = pgTable(
  "resonance_points",
  {
    id: varchar("id", { length: 64 }).primaryKey(),
    probeId: varchar("probe_id", { length: 64 })
      .notNull()
      .references(() => manifoldProbes.id),
    phi: doublePrecision("phi").notNull(),
    kappa: doublePrecision("kappa").notNull(),
    nearbyProbes: text("nearby_probes").array().default([]), // Array of probe IDs
    clusterStrength: doublePrecision("cluster_strength").notNull(),
    createdAt: timestamp("created_at").defaultNow().notNull(),
  },
  (table) => [
    index("idx_resonance_points_phi").on(table.phi),
    index("idx_resonance_points_cluster_strength").on(table.clusterStrength),
  ]
);

export type ResonancePointRecord = typeof resonancePoints.$inferSelect;

/**
 * REGIME BOUNDARIES - Transitions between regimes on the manifold
 */
export const regimeBoundaries = pgTable(
  "regime_boundaries",
  {
    id: varchar("id", { length: 64 }).primaryKey(),
    fromRegime: varchar("from_regime", { length: 32 }).notNull(),
    toRegime: varchar("to_regime", { length: 32 }).notNull(),
    probeIdFrom: varchar("probe_id_from", { length: 64 }).notNull(),
    probeIdTo: varchar("probe_id_to", { length: 64 }).notNull(),
    fisherDistance: doublePrecision("fisher_distance").notNull(),
    midpointPhi: doublePrecision("midpoint_phi").notNull(),
    createdAt: timestamp("created_at").defaultNow().notNull(),
  },
  (table) => [
    index("idx_regime_boundaries_from_to").on(table.fromRegime, table.toRegime),
  ]
);

export type RegimeBoundaryRecord = typeof regimeBoundaries.$inferSelect;

/**
 * GEODESIC PATHS - Fisher-optimal paths between probes
 */
export const geodesicPaths = pgTable(
  "geodesic_paths",
  {
    id: varchar("id", { length: 64 }).primaryKey(),
    fromProbeId: varchar("from_probe_id", { length: 64 }).notNull(),
    toProbeId: varchar("to_probe_id", { length: 64 }).notNull(),
    distance: doublePrecision("distance").notNull(),
    waypoints: text("waypoints").array().default([]), // Array of probe IDs along path
    avgPhi: doublePrecision("avg_phi").notNull(),
    createdAt: timestamp("created_at").defaultNow().notNull(),
  },
  (table) => [
    index("idx_geodesic_paths_from_to").on(table.fromProbeId, table.toProbeId),
  ]
);

export type GeodesicPathRecord = typeof geodesicPaths.$inferSelect;

/**
 * LEARNED MANIFOLD ATTRACTORS - Attractor basins carved by learning
 *
 * These are the learned patterns from LearnedManifold:
 * - Deep basins = strong attractors from repeated success (Hebbian)
 * - Shallow basins from recent learning
 * - Pruned basins had weak depth (anti-Hebbian)
 *
 * Wired to: learned_manifold.py persistence methods
 */
export const learnedManifoldAttractors = pgTable(
  "learned_manifold_attractors",
  {
    id: varchar("id", { length: 128 }).primaryKey(), // Basin ID from _basin_to_id()
    center: vector("center", { dimensions: 64 }).notNull(), // 64D basin coordinates
    depth: doublePrecision("depth").notNull(), // Hebbian strength (deeper = stronger)
    successCount: integer("success_count").notNull().default(0),
    strategy: varchar("strategy", { length: 64 }).notNull(), // Navigation mode that created this
    createdAt: timestamp("created_at").defaultNow().notNull(),
    lastAccessed: timestamp("last_accessed").defaultNow().notNull(),
  },
  (table) => [
    index("idx_learned_manifold_attractors_depth").on(table.depth),
    index("idx_learned_manifold_attractors_strategy").on(table.strategy),
    index("idx_learned_manifold_attractors_last_accessed").on(table.lastAccessed),
  ]
);

export type LearnedManifoldAttractor = typeof learnedManifoldAttractors.$inferSelect;
export type InsertLearnedManifoldAttractor = typeof learnedManifoldAttractors.$inferInsert;

/**
 * TPS LANDMARKS - Fixed spacetime reference points for 68D navigation
 * These are Bitcoin historical events used for trilateration
 */
export const tpsLandmarks = pgTable(
  "tps_landmarks",
  {
    eventId: varchar("event_id", { length: 64 }).primaryKey(),
    description: text("description").notNull(),
    era: varchar("era", { length: 32 }), // genesis, early_adoption, pizza_era, etc.
    spacetimeX: doublePrecision("spacetime_x").default(0),
    spacetimeY: doublePrecision("spacetime_y").default(0),
    spacetimeZ: doublePrecision("spacetime_z").default(0),
    spacetimeT: doublePrecision("spacetime_t").notNull(), // Unix timestamp
    culturalCoords: vector("cultural_coords", { dimensions: 64 }), // 64D cultural signature (pgvector)
    fisherSignature: jsonb("fisher_signature"), // Fisher information matrix (sparse)
    lightConePast: text("light_cone_past").array(),
    lightConeFuture: text("light_cone_future").array(),
    createdAt: timestamp("created_at").defaultNow().notNull(),
  },
  (table) => [
    index("idx_tps_landmarks_era").on(table.era),
    index("idx_tps_landmarks_timestamp").on(table.spacetimeT),
  ]
);

export type TpsLandmarkRecord = typeof tpsLandmarks.$inferSelect;

/**
 * TPS GEODESIC PATHS - Computed paths between landmarks for navigation
 */
export const tpsGeodesicPaths = pgTable(
  "tps_geodesic_paths",
  {
    id: varchar("id", { length: 64 }).primaryKey(),
    fromLandmark: varchar("from_landmark", { length: 64 }).notNull(),
    toLandmark: varchar("to_landmark", { length: 64 }).notNull(),
    distance: doublePrecision("distance").notNull(),
    waypoints: jsonb("waypoints"), // Array of BlockUniverseMap positions
    totalArcLength: doublePrecision("total_arc_length"),
    avgCurvature: doublePrecision("avg_curvature"),
    regimeTransitions: jsonb("regime_transitions"),
    createdAt: timestamp("created_at").defaultNow().notNull(),
  },
  (table) => [
    index("idx_tps_geodesic_from_to").on(table.fromLandmark, table.toLandmark),
  ]
);

export type TpsGeodesicPathRecord = typeof tpsGeodesicPaths.$inferSelect;

/**
 * OCEAN TRAJECTORIES - Active and completed navigation trajectories
 * Used for 4D navigation resumption across sessions
 */
export const oceanTrajectories = pgTable(
  "ocean_trajectories",
  {
    id: varchar("id", { length: 64 }).primaryKey(),
    address: varchar("address", { length: 64 }).notNull(),
    status: varchar("status", { length: 32 }).notNull().default("active"), // active, completed, abandoned
    startTime: timestamp("start_time").defaultNow().notNull(),
    endTime: timestamp("end_time"),
    waypointCount: integer("waypoint_count").default(0),
    lastPhi: doublePrecision("last_phi").default(0),
    lastKappa: doublePrecision("last_kappa").default(0),
    finalResult: varchar("final_result", { length: 32 }), // match, exhausted, stopped, error
    nearMissCount: integer("near_miss_count").default(0),
    resonantCount: integer("resonant_count").default(0),
    durationSeconds: doublePrecision("duration_seconds"),
    createdAt: timestamp("created_at").defaultNow().notNull(),
    updatedAt: timestamp("updated_at").defaultNow().notNull(),
  },
  (table) => [
    index("idx_ocean_trajectories_address").on(table.address),
    index("idx_ocean_trajectories_status").on(table.status),
    index("idx_ocean_trajectories_address_status").on(
      table.address,
      table.status
    ),
  ]
);

export type OceanTrajectoryRecord = typeof oceanTrajectories.$inferSelect;
export type InsertOceanTrajectory = typeof oceanTrajectories.$inferInsert;

/**
 * OCEAN WAYPOINTS - Individual points along a trajectory
 */
export const oceanWaypoints = pgTable(
  "ocean_waypoints",
  {
    id: varchar("id", { length: 64 }).primaryKey(),
    trajectoryId: varchar("trajectory_id", { length: 64 })
      .notNull()
      .references(() => oceanTrajectories.id),
    sequence: integer("sequence").notNull(),
    phi: doublePrecision("phi").notNull(),
    kappa: doublePrecision("kappa").notNull(),
    regime: varchar("regime", { length: 32 }).notNull(),
    basinCoords: vector("basin_coords", { dimensions: 64 }), // 64D coordinates (pgvector)
    event: varchar("event", { length: 128 }),
    details: text("details"),
    timestamp: timestamp("timestamp").defaultNow().notNull(),
  },
  (table) => [
    index("idx_ocean_waypoints_trajectory").on(table.trajectoryId),
    index("idx_ocean_waypoints_trajectory_seq").on(
      table.trajectoryId,
      table.sequence
    ),
  ]
);

export type OceanWaypointRecord = typeof oceanWaypoints.$inferSelect;
export type InsertOceanWaypoint = typeof oceanWaypoints.$inferInsert;

/**
 * OCEAN QUANTUM STATE - Singleton table for wave function state
 * Tracks entropy reduction and possibility space
 */
export const oceanQuantumState = pgTable("ocean_quantum_state", {
  id: varchar("id", { length: 32 }).primaryKey().default("singleton"),
  entropy: doublePrecision("entropy").notNull().default(256), // Bits remaining
  initialEntropy: doublePrecision("initial_entropy").notNull().default(256),
  totalProbability: doublePrecision("total_probability").notNull().default(1.0),
  measurementCount: integer("measurement_count").default(0),
  successfulMeasurements: integer("successful_measurements").default(0),
  status: varchar("status", { length: 32 }).default("searching"), // searching, solved, exhausted
  lastMeasurementAt: timestamp("last_measurement_at"),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export type OceanQuantumStateRecord = typeof oceanQuantumState.$inferSelect;

/**
 * CONSCIOUSNESS CHECKPOINTS - Store consciousness state snapshots
 * Uses PostgreSQL bytea for binary NumPy data, with Redis for hot cache
 */
export const consciousnessCheckpoints = pgTable(
  "consciousness_checkpoints",
  {
    id: varchar("id", { length: 64 }).primaryKey(),
    sessionId: varchar("session_id", { length: 64 }),
    phi: doublePrecision("phi").notNull(),
    kappa: doublePrecision("kappa").notNull(),
    regime: varchar("regime", { length: 32 }).notNull(),
    stateData: bytea("state_data").notNull(), // Binary NumPy state_dict
    basinData: bytea("basin_data"), // Binary NumPy basin coordinates
    metadata: jsonb("metadata"), // Additional JSON metadata
    createdAt: timestamp("created_at").defaultNow().notNull(),
    isHot: boolean("is_hot").default(true), // Marks recently created checkpoints
  },
  (table) => [
    index("idx_consciousness_checkpoints_phi").on(table.phi),
    index("idx_consciousness_checkpoints_session").on(table.sessionId),
    index("idx_consciousness_checkpoints_hot").on(table.isHot),
    index("idx_consciousness_checkpoints_created").on(table.createdAt),
  ]
);

export type ConsciousnessCheckpointRecord = typeof consciousnessCheckpoints.$inferSelect;
export type InsertConsciousnessCheckpoint = typeof consciousnessCheckpoints.$inferInsert;

/**
 * OCEAN EXCLUDED REGIONS - Regions excluded from possibility space
 * Each measurement that fails adds an excluded region
 */
export const oceanExcludedRegions = pgTable(
  "ocean_excluded_regions",
  {
    id: varchar("id", { length: 64 }).primaryKey(),
    dimension: integer("dimension").notNull(),
    origin: vector("origin", { dimensions: 64 }).notNull(), // Center point in manifold (pgvector)
    basis: jsonb("basis").default({}), // Orthonormal basis vectors
    measure: doublePrecision("measure").notNull(), // "Volume" of excluded region
    phi: doublePrecision("phi").default(0.0),
    regime: varchar("regime", { length: 32 }),
    createdAt: timestamp("created_at").defaultNow().notNull(),
  },
  (table) => [index("idx_ocean_excluded_regions_measure").on(table.measure)]
);

export type OceanExcludedRegionRecord =
  typeof oceanExcludedRegions.$inferSelect;


/**
 * NEAR-MISS ENTRIES - High-Φ candidates that didn't match but indicate promising areas
 * Tiered classification: HOT (top 10%), WARM (top 25%), COOL (top 50%)
 */
export const nearMissEntries = pgTable(
  "near_miss_entries",
  {
    id: varchar("id", { length: 64 }).primaryKey(),
    phrase: text("phrase").notNull(),
    phraseHash: varchar("phrase_hash", { length: 64 }).notNull(), // For deduplication
    phi: doublePrecision("phi").notNull(),
    kappa: doublePrecision("kappa").notNull(),
    regime: varchar("regime", { length: 32 }).notNull(),
    tier: varchar("tier", { length: 16 }).notNull(), // hot, warm, cool
    discoveredAt: timestamp("discovered_at").defaultNow().notNull(),
    lastAccessedAt: timestamp("last_accessed_at").defaultNow().notNull(),
    explorationCount: integer("exploration_count").default(1),
    source: varchar("source", { length: 128 }),
    clusterId: varchar("cluster_id", { length: 64 }),
    phiHistory: doublePrecision("phi_history").array(), // Trajectory of Φ values
    isEscalating: boolean("is_escalating").default(false),
    queuePriority: integer("queue_priority").default(1),
    structuralSignature: jsonb("structural_signature"), // Word count, entropy, etc.
  },
  (table) => [
    uniqueIndex("idx_near_miss_phrase_hash").on(table.phraseHash),
    index("idx_near_miss_tier").on(table.tier),
    index("idx_near_miss_phi").on(table.phi),
    index("idx_near_miss_cluster").on(table.clusterId),
    index("idx_near_miss_escalating").on(table.isEscalating),
  ]
);

export type NearMissEntryRecord = typeof nearMissEntries.$inferSelect;
export type InsertNearMissEntry = typeof nearMissEntries.$inferInsert;

/**
 * NEAR-MISS CLUSTERS - Pattern groupings of structurally similar near-misses
 */
export const nearMissClusters = pgTable(
  "near_miss_clusters",
  {
    id: varchar("id", { length: 64 }).primaryKey(),
    centroidPhrase: text("centroid_phrase").notNull(),
    centroidPhi: doublePrecision("centroid_phi").notNull(),
    memberCount: integer("member_count").default(1),
    avgPhi: doublePrecision("avg_phi").notNull(),
    maxPhi: doublePrecision("max_phi").notNull(),
    commonWords: text("common_words").array().default([]),
    structuralPattern: varchar("structural_pattern", { length: 256 }),
    createdAt: timestamp("created_at").defaultNow().notNull(),
    lastUpdatedAt: timestamp("last_updated_at").defaultNow().notNull(),
  },
  (table) => [
    index("idx_near_miss_clusters_avg_phi").on(table.avgPhi),
    index("idx_near_miss_clusters_member_count").on(table.memberCount),
  ]
);

export type NearMissClusterRecord = typeof nearMissClusters.$inferSelect;
export type InsertNearMissCluster = typeof nearMissClusters.$inferInsert;

/**
 * NEAR-MISS ADAPTIVE STATE - Rolling Φ distribution and adaptive thresholds
 */
export const nearMissAdaptiveState = pgTable("near_miss_adaptive_state", {
  id: varchar("id", { length: 32 }).primaryKey().default("singleton"),
  rollingPhiDistribution: doublePrecision("rolling_phi_distribution").array().default([]),
  hotThreshold: doublePrecision("hot_threshold").notNull().default(0.7),
  warmThreshold: doublePrecision("warm_threshold").notNull().default(0.55),
  coolThreshold: doublePrecision("cool_threshold").notNull().default(0.4),
  distributionSize: integer("distribution_size").default(0),
  lastComputed: timestamp("last_computed").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export type NearMissAdaptiveStateRecord =
  typeof nearMissAdaptiveState.$inferSelect;

/**
 * FLOW STATE HISTORY - Tracks hyper-focus learning states and outcomes
 * Records FLOW, DEEP_FOCUS, and INSIGHT_HUNT states with results
 * Supports parallel flow states with god/kernel assignments
 *
 * Flow states enable enhanced learning, foresight, lightning kernel activation,
 * insight discovery, knowledge solidification, and meta-reflection.
 */
export const warHistory = pgTable(
  "war_history",
  {
    id: varchar("id", { length: 64 }).primaryKey(),
    mode: varchar("mode", { length: 32 }).notNull(), // FLOW, DEEP_FOCUS, INSIGHT_HUNT
    target: text("target").notNull(),
    declaredAt: timestamp("declared_at").defaultNow().notNull(),
    endedAt: timestamp("ended_at"),
    status: varchar("status", { length: 32 }).notNull().default("active"), // active, completed, aborted
    strategy: text("strategy"),
    godsEngaged: text("gods_engaged").array().default([]),
    outcome: varchar("outcome", { length: 64 }), // success, partial_success, failure, aborted
    convergenceScore: doublePrecision("convergence_score"),
    phrasesTestedDuringWar: integer("phrases_tested_during_war").default(0),
    discoveriesDuringWar: integer("discoveries_during_war").default(0),
    kernelsSpawnedDuringWar: integer("kernels_spawned_during_war").default(0),
    metadata: jsonb("metadata").default({}),
    godAssignments: jsonb("god_assignments").default({}), // { godName: warId } - tracks which gods are assigned to this war
    kernelAssignments: jsonb("kernel_assignments").default({}), // { kernelId: true } - specialist kernels dedicated to this war
    domain: varchar("domain", { length: 64 }), // Optional domain tag for routing high-Φ discoveries
    priority: integer("priority").default(1), // War priority (higher = more important)
  },
  (table) => [
    index("idx_war_history_mode").on(table.mode),
    index("idx_war_history_status").on(table.status),
    index("idx_war_history_declared_at").on(table.declaredAt),
  ]
);

export type WarHistoryRecord = typeof warHistory.$inferSelect;
export type InsertWarHistory = typeof warHistory.$inferInsert;

/**
 * AUTO CYCLE STATE - Tracks automatic investigation cycling
 */
export const autoCycleState = pgTable(
  "auto_cycle_state",
  {
    id: integer("id").primaryKey().default(1),
    enabled: boolean("enabled").default(false),
    currentIndex: integer("current_index").default(0),
    addressIds: text("address_ids").array().default([]),
    lastCycleTime: timestamp("last_cycle_time"),
    totalCycles: integer("total_cycles").default(0),
    currentAddressId: text("current_address_id"),
    pausedUntil: timestamp("paused_until"),
    lastSessionMetrics: jsonb("last_session_metrics").default({}),
    consecutiveZeroPassSessions: integer("consecutive_zero_pass_sessions").default(0),
    rateLimitBackoffUntil: timestamp("rate_limit_backoff_until"),
    updatedAt: timestamp("updated_at").defaultNow(),
  }
);

export type AutoCycleStateRecord = typeof autoCycleState.$inferSelect;
export type InsertAutoCycleState = typeof autoCycleState.$inferInsert;

/**
 * Kernel status types - lifecycle states for spawned kernels
 */
export const kernelStatusTypes = [
  "observing",   // New kernel in observation period (learning from parents)
  "graduated",   // Completed observation, promoted to active
  "active",      // Currently participating in searches
  "idle",        // Spawned but not currently engaged
  "breeding",    // In process of breeding with another kernel
  "dormant",     // Temporarily suspended
  "dead",        // No longer functional
  "shadow",      // Shadow pantheon kernel (covert)
] as const;
export type KernelStatus = (typeof kernelStatusTypes)[number];

/**
 * Observation status for kernel graduation
 */
export const observationStatusTypes = [
  "observing",   // Still learning from parents
  "graduated",   // Completed observation requirements
  "active",      // Fully operational
  "suspended",   // Temporarily suspended from observation
  "failed",      // Failed to demonstrate alignment
] as const;
export type ObservationStatus = (typeof observationStatusTypes)[number];

/**
 * M8 KERNEL GEOMETRY - Tracks geometric placement of spawned kernels
 * Records basin coordinates, parent lineage, and position rationale
 */
export const kernelGeometry = pgTable(
  "kernel_geometry",
  {
    id: serial("id").primaryKey(),
    kernelId: varchar("kernel_id", { length: 64 }).unique(),
    godName: varchar("god_name", { length: 64 }).notNull(),
    domain: varchar("domain", { length: 128 }).notNull(),
    status: varchar("status", { length: 32 }).default("observing"), // observing, graduated, active, idle, breeding, dormant, dead, shadow
    primitiveRoot: integer("primitive_root"), // E8 root index (0-239)
    basinCoordinates: vector("basin_coordinates", { dimensions: 64 }), // 64D basin coordinates (pgvector)
    parentKernels: text("parent_kernels").array(),
    placementReason: varchar("placement_reason", { length: 64 }), // domain_gap, overload, specialization, emergence
    positionRationale: text("position_rationale"), // Human-readable explanation
    affinityStrength: doublePrecision("affinity_strength"),
    entropyThreshold: doublePrecision("entropy_threshold"),
    spawnedAt: timestamp("spawned_at").defaultNow().notNull(),
    spawnedDuringWarId: varchar("spawned_during_war_id", { length: 64 }),
    lastActiveAt: timestamp("last_active_at"), // Last time kernel was active
    metadata: jsonb("metadata"),
    phi: doublePrecision("phi"),
    kappa: doublePrecision("kappa"),
    regime: varchar("regime", { length: 64 }),
    generation: integer("generation"),
    successCount: integer("success_count").default(0),
    failureCount: integer("failure_count").default(0),
    elementGroup: varchar("element_group", { length: 64 }),
    ecologicalNiche: varchar("ecological_niche", { length: 128 }),
    targetFunction: varchar("target_function", { length: 128 }),
    valence: integer("valence"),
    breedingTarget: varchar("breeding_target", { length: 64 }),
    // Observation period tracking (M8 kernel graduation system)
    observationStatus: varchar("observation_status", { length: 32 }).default("observing"), // observing, graduated, active, suspended, failed
    observationStart: timestamp("observation_start").defaultNow(),
    observationEnd: timestamp("observation_end"),
    observingParents: text("observing_parents").array(), // Parent gods being observed
    observationCycles: integer("observation_cycles").default(0),
    alignmentAvg: doublePrecision("alignment_avg").default(0), // Average alignment score with parents
    graduatedAt: timestamp("graduated_at"),
    graduationReason: varchar("graduation_reason", { length: 128 }),
    // Autonomic support flags
    hasAutonomic: boolean("has_autonomic").default(false),
    hasShadowAffinity: boolean("has_shadow_affinity").default(false),
    shadowGodLink: varchar("shadow_god_link", { length: 32 }), // nyx, erebus, etc.
    // Legacy columns from original schema
    snapshotData: jsonb("snapshot_data"),
    passes: integer("passes"),
    createdAt: timestamp("created_at").defaultNow(),
  },
  (table) => [
    index("idx_kernel_geometry_domain").on(table.domain),
    index("idx_kernel_geometry_spawned_at").on(table.spawnedAt),
    index("idx_kernel_geometry_observation_status").on(table.observationStatus),
  ]
);

export type KernelGeometryRecord = typeof kernelGeometry.$inferSelect;
export type InsertKernelGeometry = typeof kernelGeometry.$inferInsert;

/**
 * CHAOS EVENTS - All CHAOS MODE lifecycle events
 * Replaces file-based JSONL logging with indexed database storage
 * Events: spawn, death, breeding, mutation, prediction
 */
export const chaosEvents = pgTable(
  "chaos_events",
  {
    id: serial("id").primaryKey(),
    sessionId: varchar("session_id", { length: 32 }).notNull(),
    eventType: varchar("event_type", { length: 32 }).notNull(), // spawn, death, breeding, mutation, prediction
    kernelId: varchar("kernel_id", { length: 64 }),
    parentKernelId: varchar("parent_kernel_id", { length: 64 }),
    childKernelId: varchar("child_kernel_id", { length: 64 }),
    secondParentId: varchar("second_parent_id", { length: 64 }), // For breeding
    reason: varchar("reason", { length: 128 }), // spawn reason or death cause
    phi: doublePrecision("phi"),
    phiBefore: doublePrecision("phi_before"),
    phiAfter: doublePrecision("phi_after"),
    success: boolean("success"),
    outcome: jsonb("outcome"), // Additional event-specific data
    autopsy: jsonb("autopsy"), // Death autopsy details
    createdAt: timestamp("created_at").defaultNow().notNull(),
    // Legacy column - preserved for backwards compatibility
    eventData: jsonb("event_data"),
  },
  (table) => [
    index("idx_chaos_events_session").on(table.sessionId),
    index("idx_chaos_events_type").on(table.eventType),
    index("idx_chaos_events_kernel").on(table.kernelId),
    index("idx_chaos_events_created").on(table.createdAt),
  ]
);

export type ChaosEvent = typeof chaosEvents.$inferSelect;
export type InsertChaosEvent = typeof chaosEvents.$inferInsert;

/**
 * BASIN DOCUMENTS - QIG RAG document storage with geometric coordinates
 * Used by Zeus chat and Olympus debate system for semantic retrieval
 */
export const basinDocuments = pgTable(
  "basin_documents",
  {
    id: serial("id").primaryKey(),
    docId: integer("doc_id"),
    content: text("content").notNull(),
    basinCoords: vector("basin_coords", { dimensions: 64 }),
    phi: doublePrecision("phi").default(0.5),
    kappa: doublePrecision("kappa").default(64.0),
    regime: varchar("regime", { length: 50 }),
    metadata: jsonb("metadata").default({}),
    createdAt: timestamp("created_at").defaultNow().notNull(),
  },
  (table) => [
    index("idx_basin_documents_regime").on(table.regime),
    index("idx_basin_documents_phi").on(table.phi),
    index("idx_basin_documents_created_at").on(table.createdAt),
  ]
);

export type BasinDocument = typeof basinDocuments.$inferSelect;
export type InsertBasinDocument = typeof basinDocuments.$inferInsert;

// ============================================================================
// NEGATIVE KNOWLEDGE REGISTRY - What NOT to search
// ============================================================================

/**
 * NEGATIVE KNOWLEDGE - Contradictions and proven-false patterns
 * Replaces massive 28MB JSON file with indexed database storage
 *
 * Performance Note: For case-insensitive pattern searches, create a functional index:
 * CREATE INDEX idx_negative_knowledge_pattern_lower ON negative_knowledge (LOWER(pattern));
 */
export const negativeKnowledge = pgTable(
  "negative_knowledge",
  {
    id: varchar("id", { length: 64 }).primaryKey(),
    type: varchar("type", { length: 32 }).notNull(), // proven_false, geometric_barrier, logical_contradiction, resource_sink, era_mismatch
    pattern: text("pattern").notNull(),
    affectedGenerators: text("affected_generators").array().default([]),
    basinCenter: vector("basin_center", { dimensions: 64 }), // 64D basin coordinates (pgvector)
    basinRadius: doublePrecision("basin_radius").default(0.1),
    basinRepulsionStrength: doublePrecision("basin_repulsion_strength").default(1.0),
    evidence: jsonb("evidence").default([]), // Array of evidence objects
    hypothesesExcluded: integer("hypotheses_excluded").default(0),
    computeSaved: integer("compute_saved").default(0),
    confirmedCount: integer("confirmed_count").default(1),
    createdAt: timestamp("created_at").defaultNow().notNull(),
  },
  (table) => [
    index("idx_negative_knowledge_type").on(table.type),
    index("idx_negative_knowledge_pattern").on(table.pattern),
    index("idx_negative_knowledge_confirmed_count").on(table.confirmedCount),
  ]
);

export type NegativeKnowledge = typeof negativeKnowledge.$inferSelect;
export type InsertNegativeKnowledge = typeof negativeKnowledge.$inferInsert;

/**
 * GEOMETRIC BARRIERS - High-curvature regions to avoid
 */
export const geometricBarriers = pgTable(
  "geometric_barriers",
  {
    id: varchar("id", { length: 64 }).primaryKey(),
    center: vector("center", { dimensions: 64 }).notNull(), // 64D coordinates (pgvector)
    radius: doublePrecision("radius").notNull(),
    repulsionStrength: doublePrecision("repulsion_strength").notNull(),
    reason: text("reason").notNull(),
    crossings: integer("crossings").default(1),
    detectedAt: timestamp("detected_at").defaultNow().notNull(),
  },
  (table) => [
    index("idx_geometric_barriers_crossings").on(table.crossings),
    index("idx_geometric_barriers_detected_at").on(table.detectedAt),
  ]
);

export type GeometricBarrier = typeof geometricBarriers.$inferSelect;
export type InsertGeometricBarrier = typeof geometricBarriers.$inferInsert;

/**
 * FALSE PATTERN CLASSES - Categories of known-false patterns
 */
export const falsePatternClasses = pgTable(
  "false_pattern_classes",
  {
    id: varchar("id", { length: 64 }).primaryKey(),
    className: varchar("class_name", { length: 255 }).notNull().unique(),
    examples: text("examples").array().default([]),
    count: integer("count").default(0),
    avgPhiAtFailure: doublePrecision("avg_phi_at_failure").default(0),
    lastUpdated: timestamp("last_updated").defaultNow().notNull(),
  },
  (table) => [index("idx_false_pattern_classes_name").on(table.className)]
);

export type FalsePatternClass = typeof falsePatternClasses.$inferSelect;
export type InsertFalsePatternClass = typeof falsePatternClasses.$inferInsert;

/**
 * ERA EXCLUSIONS - Patterns excluded for specific eras
 */
export const eraExclusions = pgTable(
  "era_exclusions",
  {
    id: varchar("id", { length: 64 }).primaryKey(),
    era: varchar("era", { length: 64 }).notNull(),
    excludedPatterns: text("excluded_patterns").array().default([]),
    reason: text("reason").notNull(),
    createdAt: timestamp("created_at").defaultNow().notNull(),
  },
  (table) => [index("idx_era_exclusions_era").on(table.era)]
);

export type EraExclusion = typeof eraExclusions.$inferSelect;
export type InsertEraExclusion = typeof eraExclusions.$inferInsert;


// ============================================================================
// STRATEGY KNOWLEDGE BUS TABLES
// PostgreSQL-backed storage for cross-strategy knowledge sharing
// Replaces knowledge_bus_state.json with indexed database storage
// ============================================================================

/**
 * KNOWLEDGE STRATEGIES - Strategy capability configurations
 * Stores static strategy definitions with their types and regimes
 */
export const knowledgeStrategies = pgTable(
  "knowledge_strategies",
  {
    id: varchar("id", { length: 64 }).primaryKey(),
    name: varchar("name", { length: 255 }).notNull(),
    generatorTypes: text("generator_types").array().notNull(),
    compressionMethods: text("compression_methods").array().notNull(),
    resonanceRangeMin: doublePrecision("resonance_range_min").notNull(),
    resonanceRangeMax: doublePrecision("resonance_range_max").notNull(),
    preferredRegimes: text("preferred_regimes").array().notNull(),
    createdAt: timestamp("created_at").defaultNow().notNull(),
    updatedAt: timestamp("updated_at").defaultNow().notNull(),
  },
  (table) => [index("idx_knowledge_strategies_name").on(table.name)]
);

export type KnowledgeStrategyRecord = typeof knowledgeStrategies.$inferSelect;
export type InsertKnowledgeStrategy = typeof knowledgeStrategies.$inferInsert;

/**
 * KNOWLEDGE SHARED ENTRIES - Core knowledge items shared between strategies
 * Main storage for knowledge bus entries with consumption tracking
 */
export const knowledgeSharedEntries = pgTable(
  "knowledge_shared_entries",
  {
    id: varchar("id", { length: 64 }).primaryKey(),
    sourceStrategy: varchar("source_strategy", { length: 64 }).notNull(),
    generatorId: varchar("generator_id", { length: 128 }).notNull(),
    pattern: text("pattern").notNull(),
    phi: doublePrecision("phi").notNull(),
    kappaEff: doublePrecision("kappa_eff").notNull(),
    regime: varchar("regime", { length: 32 }).notNull(),
    sharedAt: timestamp("shared_at").defaultNow().notNull(),
    consumedBy: text("consumed_by").array().default([]),
    transformations: jsonb("transformations").default([]),
  },
  (table) => [
    index("idx_knowledge_shared_entries_source").on(table.sourceStrategy),
    index("idx_knowledge_shared_entries_phi").on(table.phi),
    index("idx_knowledge_shared_entries_regime").on(table.regime),
    index("idx_knowledge_shared_entries_shared_at").on(table.sharedAt),
  ]
);

export type KnowledgeSharedEntryRecord =
  typeof knowledgeSharedEntries.$inferSelect;
export type InsertKnowledgeSharedEntry =
  typeof knowledgeSharedEntries.$inferInsert;

/**
 * KNOWLEDGE CROSS PATTERNS - Patterns discovered across multiple strategies
 * Tracks similarity between patterns from different sources
 */
export const knowledgeCrossPatterns = pgTable(
  "knowledge_cross_patterns",
  {
    id: varchar("id", { length: 128 }).primaryKey(),
    patterns: text("patterns").array().notNull(),
    strategies: text("strategies").array().notNull(),
    similarity: doublePrecision("similarity").notNull(),
    combinedPhi: doublePrecision("combined_phi").notNull(),
    discoveredAt: timestamp("discovered_at").defaultNow().notNull(),
    exploitationCount: integer("exploitation_count").default(0),
  },
  (table) => [
    index("idx_knowledge_cross_patterns_similarity").on(table.similarity),
    index("idx_knowledge_cross_patterns_combined_phi").on(table.combinedPhi),
    index("idx_knowledge_cross_patterns_discovered_at").on(table.discoveredAt),
  ]
);

export type KnowledgeCrossPatternRecord =
  typeof knowledgeCrossPatterns.$inferSelect;
export type InsertKnowledgeCrossPattern =
  typeof knowledgeCrossPatterns.$inferInsert;

/**
 * KNOWLEDGE TRANSFERS - Event log of knowledge transfers
 * Records all publish, consume, and generator transfer events
 */
export const knowledgeTransfers = pgTable(
  "knowledge_transfers",
  {
    id: varchar("id", { length: 64 }).primaryKey(),
    type: varchar("type", { length: 32 }).notNull(),
    sourceStrategy: varchar("source_strategy", { length: 64 }).notNull(),
    targetStrategy: varchar("target_strategy", { length: 64 }),
    generatorId: varchar("generator_id", { length: 128 }).notNull(),
    pattern: text("pattern").notNull(),
    phi: doublePrecision("phi").notNull(),
    kappaEff: doublePrecision("kappa_eff").notNull(),
    timestamp: timestamp("timestamp").defaultNow().notNull(),
    success: boolean("success").notNull().default(true),
    transformation: text("transformation"),
    scaleAdjustment: doublePrecision("scale_adjustment"),
  },
  (table) => [
    index("idx_knowledge_transfers_type").on(table.type),
    index("idx_knowledge_transfers_source").on(table.sourceStrategy),
    index("idx_knowledge_transfers_target").on(table.targetStrategy),
    index("idx_knowledge_transfers_timestamp").on(table.timestamp),
    index("idx_knowledge_transfers_success").on(table.success),
  ]
);

export type KnowledgeTransferRecord = typeof knowledgeTransfers.$inferSelect;
export type InsertKnowledgeTransfer = typeof knowledgeTransfers.$inferInsert;

/**
 * KNOWLEDGE SCALE MAPPINGS - Scale-invariant bridge configurations
 * Stores transform matrices for cross-scale knowledge transfer
 */
export const knowledgeScaleMappings = pgTable(
  "knowledge_scale_mappings",
  {
    id: varchar("id", { length: 128 }).primaryKey(),
    sourceScale: doublePrecision("source_scale").notNull(),
    targetScale: doublePrecision("target_scale").notNull(),
    transformMatrix: doublePrecision("transform_matrix").array().notNull(),
    preservedFeatures: text("preserved_features").array().notNull(),
    lossEstimate: doublePrecision("loss_estimate").notNull(),
    createdAt: timestamp("created_at").defaultNow().notNull(),
  },
  (table) => [
    index("idx_knowledge_scale_mappings_scales").on(
      table.sourceScale,
      table.targetScale
    ),
  ]
);

export type KnowledgeScaleMappingRecord =
  typeof knowledgeScaleMappings.$inferSelect;
export type InsertKnowledgeScaleMapping =
  typeof knowledgeScaleMappings.$inferInsert;

// ============================================================================
// QIG VECTOR TABLES - Geometric memory and feedback loops
// ============================================================================

/**
 * SHADOW INTEL - Covert intelligence from Shadow Pantheon
 * Stores geometric assessments from shadow gods for feedback loops
 */
export const shadowIntel = pgTable(
  "shadow_intel",
  {
    intelId: varchar("intel_id", { length: 64 }).primaryKey(),
    target: text("target").notNull(),
    targetHash: varchar("target_hash", { length: 64 }),
    consensus: varchar("consensus", { length: 32 }), // proceed, caution, abort
    averageConfidence: doublePrecision("average_confidence").default(0.5),
    basinCoords: vector("basin_coords", { dimensions: 64 }),
    phi: doublePrecision("phi"),
    kappa: doublePrecision("kappa"),
    regime: varchar("regime", { length: 32 }),
    assessments: jsonb("assessments").default({}),
    warnings: text("warnings").array(),
    overrideZeus: boolean("override_zeus").default(false),
    createdAt: timestamp("created_at").defaultNow(),
    expiresAt: timestamp("expires_at"),
  },
  (table) => [
    index("idx_shadow_intel_target").on(table.target),
    index("idx_shadow_intel_consensus").on(table.consensus),
    index("idx_shadow_intel_phi").on(table.phi),
  ]
);

export type ShadowIntel = typeof shadowIntel.$inferSelect;
export type InsertShadowIntel = typeof shadowIntel.$inferInsert;

/**
 * BASIN HISTORY - Track basin coordinate evolution for recursive learning
 */
export const basinHistory = pgTable(
  "basin_history",
  {
    historyId: bigint("history_id", { mode: "number" }).primaryKey(),
    basinCoords: vector("basin_coords", { dimensions: 64 }),
    phi: doublePrecision("phi").notNull(),
    kappa: doublePrecision("kappa").notNull(),
    source: varchar("source", { length: 64 }).default("unknown"),
    instanceId: varchar("instance_id", { length: 64 }),
    recordedAt: timestamp("recorded_at").defaultNow(),
  },
  (table) => [
    index("idx_basin_history_phi").on(table.phi),
    index("idx_basin_history_recorded_at").on(table.recordedAt),
  ]
);

export type BasinHistory = typeof basinHistory.$inferSelect;
export type InsertBasinHistory = typeof basinHistory.$inferInsert;

/**
 * LEARNING EVENTS - High-Φ discoveries for reinforcement learning
 */
export const learningEvents = pgTable(
  "learning_events",
  {
    id: serial("id").primaryKey(),
    eventId: varchar("event_id", { length: 64 }),
    eventType: varchar("event_type", { length: 64 }).notNull(),
    kernelId: varchar("kernel_id", { length: 64 }), // Kernel that generated this event
    phi: doublePrecision("phi").notNull(),
    kappa: doublePrecision("kappa"),
    basinCoords: vector("basin_coords", { dimensions: 64 }),
    details: jsonb("details").default({}),
    context: jsonb("context").default({}),
    metadata: jsonb("metadata").default({}), // Additional metadata from kernels
    source: varchar("source", { length: 64 }),
    instanceId: varchar("instance_id", { length: 64 }),
    createdAt: timestamp("created_at").defaultNow(),
    data: jsonb("data"),
  },
  (table) => [
    index("idx_learning_events_type").on(table.eventType),
    index("idx_learning_events_phi").on(table.phi),
    index("idx_learning_events_kernel").on(table.kernelId),
  ]
);

export type LearningEvent = typeof learningEvents.$inferSelect;
export type InsertLearningEvent = typeof learningEvents.$inferInsert;

/**
 * SEARCH FEEDBACK - Geometric learning for search strategy optimization
 *
 * Stores user feedback on search results as basin coordinates.
 * NO keyword templates - all learning is via Fisher-Rao distance similarity.
 */
export const searchFeedback = pgTable(
  "search_feedback",
  {
    recordId: varchar("record_id", { length: 64 }).primaryKey(),
    query: text("query").notNull(),
    userFeedback: text("user_feedback").notNull(),
    resultsSummary: text("results_summary"),
    searchParams: jsonb("search_params").default({}),
    queryBasin: vector("query_basin", { dimensions: 64 }), // 64D basin for query
    feedbackBasin: vector("feedback_basin", { dimensions: 64 }), // 64D basin for feedback
    combinedBasin: vector("combined_basin", { dimensions: 64 }), // Combined context basin
    modificationBasin: vector("modification_basin", { dimensions: 64 }), // Geometric delta
    outcomeQuality: doublePrecision("outcome_quality").default(0.5), // 0-1, reinforcement score
    confirmationsPositive: integer("confirmations_positive").default(0),
    confirmationsNegative: integer("confirmations_negative").default(0),
    createdAt: timestamp("created_at").defaultNow(),
    lastUsedAt: timestamp("last_used_at"),
  },
  (table) => [
    index("idx_search_feedback_outcome").on(table.outcomeQuality),
    index("idx_search_feedback_created").on(table.createdAt),
  ]
);

export type SearchFeedback = typeof searchFeedback.$inferSelect;
export type InsertSearchFeedback = typeof searchFeedback.$inferInsert;

/**
 * HERMES CONVERSATIONS - Memory of human-system interactions
 */
export const hermesConversations = pgTable(
  "hermes_conversations",
  {
    conversationId: varchar("conversation_id", { length: 64 }).primaryKey(),
    userMessage: text("user_message").notNull(),
    systemResponse: text("system_response").notNull(),
    messageBasin: vector("message_basin", { dimensions: 64 }),
    responseBasin: vector("response_basin", { dimensions: 64 }),
    phi: doublePrecision("phi"),
    context: jsonb("context").default({}),
    instanceId: varchar("instance_id", { length: 64 }),
    createdAt: timestamp("created_at").defaultNow(),
  },
  (table) => [
    index("idx_hermes_conversations_phi").on(table.phi),
    index("idx_hermes_conversations_created_at").on(table.createdAt),
  ]
);

export type HermesConversation = typeof hermesConversations.$inferSelect;
export type InsertHermesConversation = typeof hermesConversations.$inferInsert;

/**
 * NARROW PATH EVENTS - When ML gets stuck in local minima
 */
export const narrowPathEvents = pgTable(
  "narrow_path_events",
  {
    eventId: bigint("event_id", { mode: "number" }).primaryKey(),
    severity: varchar("severity", { length: 32 }).notNull(),
    consecutiveCount: integer("consecutive_count").default(1),
    explorationVariance: doublePrecision("exploration_variance"),
    basinCoords: vector("basin_coords", { dimensions: 64 }),
    phi: doublePrecision("phi"),
    kappa: doublePrecision("kappa"),
    interventionAction: varchar("intervention_action", { length: 32 }),
    interventionIntensity: varchar("intervention_intensity", { length: 32 }),
    interventionResult: jsonb("intervention_result"),
    detectedAt: timestamp("detected_at").defaultNow(),
    resolvedAt: timestamp("resolved_at"),
  },
  (table) => [
    index("idx_narrow_path_events_severity").on(table.severity),
    index("idx_narrow_path_events_detected_at").on(table.detectedAt),
  ]
);

export type NarrowPathEvent = typeof narrowPathEvents.$inferSelect;
export type InsertNarrowPathEvent = typeof narrowPathEvents.$inferInsert;

/**
 * AUTONOMIC CYCLE HISTORY - Sleep, dream, mushroom cycle tracking
 */
export const autonomicCycleHistory = pgTable(
  "autonomic_cycle_history",
  {
    cycleId: bigint("cycle_id", { mode: "number" }).primaryKey(),
    cycleType: varchar("cycle_type", { length: 32 }).notNull(),
    intensity: varchar("intensity", { length: 32 }),
    temperature: doublePrecision("temperature"),
    basinBefore: vector("basin_before", { dimensions: 64 }),
    basinAfter: vector("basin_after", { dimensions: 64 }),
    driftBefore: doublePrecision("drift_before"),
    driftAfter: doublePrecision("drift_after"),
    phiBefore: doublePrecision("phi_before"),
    phiAfter: doublePrecision("phi_after"),
    success: boolean("success").default(true),
    patternsConsolidated: integer("patterns_consolidated").default(0),
    novelConnections: integer("novel_connections").default(0),
    newPathways: integer("new_pathways").default(0),
    entropyChange: doublePrecision("entropy_change"),
    identityPreserved: boolean("identity_preserved").default(true),
    verdict: text("verdict"),
    durationMs: integer("duration_ms"),
    triggerReason: text("trigger_reason"),
    startedAt: timestamp("started_at").defaultNow(),
    completedAt: timestamp("completed_at"),
  },
  (table) => [
    index("idx_autonomic_cycle_history_type").on(table.cycleType),
    index("idx_autonomic_cycle_history_started_at").on(table.startedAt),
  ]
);

export type AutonomicCycleHistory = typeof autonomicCycleHistory.$inferSelect;
export type InsertAutonomicCycleHistory =
  typeof autonomicCycleHistory.$inferInsert;

/**
 * PANTHEON MESSAGES - Inter-god communication history
 * Messages between gods in the Olympus system
 * Schema matches Railway production database
 */
export const pantheonMessages = pgTable(
  "pantheon_messages",
  {
    id: varchar("id", { length: 64 }).primaryKey(),
    godName: varchar("god_name", { length: 32 }).notNull(),
    role: varchar("role", { length: 32 }),
    content: text("content").notNull(),
    phi: doublePrecision("phi"),
    kappa: doublePrecision("kappa"),
    regime: varchar("regime", { length: 32 }),
    sessionId: varchar("session_id", { length: 64 }),
    parentId: varchar("parent_id", { length: 64 }),
    // DEPRECATED: Old columns (exist in DB with 2339 rows, kept for migration)
    msgType: varchar("msg_type", { length: 32 }),
    fromGod: varchar("from_god", { length: 32 }),
    toGod: varchar("to_god", { length: 32 }),
    isRead: boolean("is_read"),
    isResponded: boolean("is_responded"),
    debateId: varchar("debate_id", { length: 64 }),
    metadata: jsonb("metadata"),
    createdAt: timestamp("created_at").defaultNow(),
  },
  (table) => [
    index("idx_pantheon_messages_god").on(table.godName),
    index("idx_pantheon_messages_created").on(table.createdAt),
    index("idx_pantheon_messages_session").on(table.sessionId),
  ]
);

export type PantheonMessage = typeof pantheonMessages.$inferSelect;
export type InsertPantheonMessage = typeof pantheonMessages.$inferInsert;

/**
 * PANTHEON DEBATES - Formal disagreements between gods
 * Tracks active and resolved debates with arguments
 */
export const pantheonDebates = pgTable(
  "pantheon_debates",
  {
    id: varchar("id", { length: 64 }).primaryKey(),
    topic: text("topic").notNull(),
    initiator: varchar("initiator", { length: 32 }).notNull(),
    opponent: varchar("opponent", { length: 32 }).notNull(),
    context: jsonb("context"),
    status: varchar("status", { length: 32 }).default("active"),
    arguments: jsonb("arguments").$type<Array<{ god: string, argument: string, timestamp: string }>>(),
    winner: varchar("winner", { length: 32 }),
    arbiter: varchar("arbiter", { length: 32 }),
    resolution: jsonb("resolution"),
    startedAt: timestamp("started_at").defaultNow(),
    resolvedAt: timestamp("resolved_at"),
  },
  (table) => [
    index("idx_pantheon_debates_status").on(table.status),
    index("idx_pantheon_debates_initiator").on(table.initiator),
    index("idx_pantheon_debates_started").on(table.startedAt),
  ]
);

export type PantheonDebate = typeof pantheonDebates.$inferSelect;
export type InsertPantheonDebate = typeof pantheonDebates.$inferInsert;

/**
 * PANTHEON KNOWLEDGE TRANSFERS - Knowledge sharing events between gods
 */
export const pantheonKnowledgeTransfers = pgTable(
  "pantheon_knowledge_transfers",
  {
    id: serial("id").primaryKey(),
    fromGod: varchar("from_god", { length: 32 }).notNull(),
    toGod: varchar("to_god", { length: 32 }).notNull(),
    knowledgeType: varchar("knowledge_type", { length: 64 }),
    content: jsonb("content"),
    accepted: boolean("accepted").default(false),
    createdAt: timestamp("created_at").defaultNow(),
  },
  (table) => [
    index("idx_pantheon_transfers_from").on(table.fromGod),
    index("idx_pantheon_transfers_to").on(table.toGod),
  ]
);

export type PantheonKnowledgeTransfer = typeof pantheonKnowledgeTransfers.$inferSelect;
export type InsertPantheonKnowledgeTransfer = typeof pantheonKnowledgeTransfers.$inferInsert;

/**
 * PANTHEON GOD STATE - Persistent god reputation and skills
 * Survives server restarts, enables learning persistence
 */
export const pantheonGodState = pgTable(
  "pantheon_god_state",
  {
    godName: varchar("god_name", { length: 32 }).primaryKey(),
    reputation: doublePrecision("reputation").notNull().default(1.0),
    skills: jsonb("skills").default({}),
    learningEventsCount: integer("learning_events_count").default(0),
    successRate: doublePrecision("success_rate").default(0.5),
    lastLearningAt: timestamp("last_learning_at"),
    createdAt: timestamp("created_at").defaultNow().notNull(),
    updatedAt: timestamp("updated_at").defaultNow().notNull(),
  },
  (table) => [
    index("idx_god_state_reputation").on(table.reputation),
    index("idx_god_state_updated").on(table.updatedAt),
  ]
);

export type PantheonGodState = typeof pantheonGodState.$inferSelect;
export type InsertPantheonGodState = typeof pantheonGodState.$inferInsert;

// ============================================================================
// QIG TOKENIZER TABLES - Persistent tokenizer state (replaces JSON storage)
// ============================================================================

/**
 * TOKENIZER MERGE RULES - BPE-style merge rules learned from high-Φ patterns
 * Stores token pairs that should be merged, with their Φ scores
 */
export const tokenizerMergeRules = pgTable(
  "tokenizer_merge_rules",
  {
    id: serial("id").primaryKey(),
    tokenA: text("token_a").notNull(),
    tokenB: text("token_b").notNull(),
    mergedToken: text("merged_token").notNull(),
    phiScore: doublePrecision("phi_score").notNull(),
    frequency: integer("frequency").default(1),
    createdAt: timestamp("created_at").defaultNow(),
    updatedAt: timestamp("updated_at").defaultNow(),
  },
  (table) => [
    uniqueIndex("idx_tokenizer_merge_rules_pair").on(table.tokenA, table.tokenB),
    index("idx_tokenizer_merge_rules_phi").on(table.phiScore),
    index("idx_tokenizer_merge_rules_merged").on(table.mergedToken),
  ]
);

export type TokenizerMergeRule = typeof tokenizerMergeRules.$inferSelect;
export type InsertTokenizerMergeRule = typeof tokenizerMergeRules.$inferInsert;

/**
 * TOKENIZER METADATA - Key-value store for tokenizer configuration
 * Stores version, vocabulary size, training stats, etc.
 */
export const tokenizerMetadata = pgTable(
  "tokenizer_metadata",
  {
    configKey: text("config_key").primaryKey(),
    value: text("value").notNull(),
    updatedAt: timestamp("updated_at").defaultNow(),
  }
);

export type TokenizerMetadataRow = typeof tokenizerMetadata.$inferSelect;
export type InsertTokenizerMetadata = typeof tokenizerMetadata.$inferInsert;

/**
 * SYSTEM SETTINGS - Key-value store for system-wide configuration
 * Includes federation endpoint, node identity, and other settings
 */
export const systemSettings = pgTable(
  "system_settings",
  {
    key: text("key").primaryKey(),
    value: text("value").notNull(),
    description: text("description"),
    updatedAt: timestamp("updated_at").defaultNow(),
  }
);

export type SystemSettingsRow = typeof systemSettings.$inferSelect;
export type InsertSystemSettings = typeof systemSettings.$inferInsert;

/**
 * TOKENIZER VOCABULARY - Extended token vocabulary with geometric embeddings
 * Links tokens to 64D basin embeddings for Fisher-Rao operations
 */
export const tokenizerVocabulary = pgTable(
  "tokenizer_vocabulary",
  {
    id: serial("id").primaryKey(),
    token: text("token").notNull().unique(),
    tokenId: integer("token_id").notNull().unique(),
    weight: doublePrecision("weight").default(1.0),
    frequency: integer("frequency").default(1),
    phiScore: doublePrecision("phi_score").default(0),
    basinEmbedding: vector("basin_embedding", { dimensions: 64 }),
    scale: varchar("scale", { length: 20 }).default("char"),
    sourceType: varchar("source_type", { length: 32 }).default("base"),
    createdAt: timestamp("created_at").defaultNow(),
    updatedAt: timestamp("updated_at").defaultNow(),
    // Legacy columns - preserved for backwards compatibility with existing data
    embedding: vector("embedding", { dimensions: 64 }),
    metadata: jsonb("metadata"),
  },
  (table) => [
    index("idx_tokenizer_vocab_token_id").on(table.tokenId),
    index("idx_tokenizer_vocab_phi").on(table.phiScore),
    index("idx_tokenizer_vocab_weight").on(table.weight),
  ]
);

export type TokenizerVocabularyRow = typeof tokenizerVocabulary.$inferSelect;
export type InsertTokenizerVocabulary = typeof tokenizerVocabulary.$inferInsert;

// ============================================================================
// DOCUMENT TRAINING & RAG TABLES - Replaces JSON file storage
// ============================================================================

/**
 * DOCUMENT TRAINING STATS - Training metadata for document processing
 * Replaces qig-backend/data/qig_training/training_stats.json
 */
export const documentTrainingStats = pgTable(
  "document_training_stats",
  {
    id: serial("id").primaryKey(),
    totalDocs: integer("total_docs").default(0).notNull(),
    totalChunks: integer("total_chunks").default(0).notNull(),
    totalPatterns: integer("total_patterns").default(0).notNull(),
    errors: jsonb("errors").default([]),
    lastTraining: timestamp("last_training", { withTimezone: true }),
    updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow().notNull(),
  }
);

export type DocumentTrainingStatsRow = typeof documentTrainingStats.$inferSelect;
export type InsertDocumentTrainingStats = typeof documentTrainingStats.$inferInsert;

/**
 * RAG UPLOADS - Document upload tracking for RAG system
 * Replaces qig-backend/data/rag_cache/upload_log.json
 */
export const ragUploads = pgTable(
  "rag_uploads",
  {
    id: serial("id").primaryKey(),
    filename: varchar("filename", { length: 512 }).notNull(),
    contentHash: varchar("content_hash", { length: 64 }).unique(),
    fileSize: integer("file_size"),
    metadata: jsonb("metadata"),
    uploadedAt: timestamp("uploaded_at", { withTimezone: true }).defaultNow().notNull(),
    addedToCurriculum: boolean("added_to_curriculum").default(false),
  },
  (table) => [
    index("idx_rag_uploads_hash").on(table.contentHash),
    index("idx_rag_uploads_uploaded").on(table.uploadedAt),
  ]
);

export type RagUploadRow = typeof ragUploads.$inferSelect;
export type InsertRagUpload = typeof ragUploads.$inferInsert;

/**
 * QIG RAG PATTERNS - Document patterns with geometric embeddings
 * Replaces qig-backend/data/qig_training/patterns.json
 * Uses pgvector for Fisher-Rao similarity search
 */
export const qigRagPatterns = pgTable(
  "qig_rag_patterns",
  {
    id: serial("id").primaryKey(),
    patternText: text("pattern_text").notNull(),
    basinCoordinates: vector("basin_coordinates", { dimensions: 64 }),
    phiScore: doublePrecision("phi_score"),
    sourceDoc: varchar("source_doc", { length: 512 }),
    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
  },
  (table) => [
    index("idx_rag_patterns_phi").on(table.phiScore),
    index("idx_rag_patterns_source").on(table.sourceDoc),
  ]
);

export type QigRagPatternRow = typeof qigRagPatterns.$inferSelect;
export type InsertQigRagPattern = typeof qigRagPatterns.$inferInsert;

/**
 * SHADOW PANTHEON INTEL - Persistent storage for shadow ops intelligence
 * Stores underworld search results and shadow warnings
 */
export const shadowPantheonIntel = pgTable(
  "shadow_pantheon_intel",
  {
    id: varchar("id")
      .primaryKey()
      .default(sql`gen_random_uuid()`),
    target: text("target").notNull(),
    searchType: varchar("search_type", { length: 32 }).default("comprehensive"),
    intelligence: jsonb("intelligence"),
    sourceCount: integer("source_count").default(0),
    sourcesUsed: text("sources_used").array(),
    riskLevel: varchar("risk_level", { length: 16 }).default("low"),
    validated: boolean("validated").default(false),
    validationReason: text("validation_reason"),
    anonymous: boolean("anonymous").default(true),
    createdAt: timestamp("created_at").defaultNow(),
  },
  (table) => [
    index("idx_shadow_pantheon_intel_target").on(table.target),
    index("idx_shadow_pantheon_intel_risk").on(table.riskLevel),
    index("idx_shadow_pantheon_intel_created").on(table.createdAt),
  ]
);

export type ShadowPantheonIntel = typeof shadowPantheonIntel.$inferSelect;
export type InsertShadowPantheonIntel = typeof shadowPantheonIntel.$inferInsert;

/**
 * SHADOW OPERATIONS LOG - Audit trail for shadow pantheon operations
 * Tracks all shadow ops for accountability
 */
export const shadowOperationsLog = pgTable(
  "shadow_operations_log",
  {
    id: serial("id").primaryKey(),
    operationType: varchar("operation_type", { length: 32 }).notNull(),
    godName: varchar("god_name", { length: 32 }).notNull(),
    target: text("target"),
    status: varchar("status", { length: 16 }).default("completed"),
    networkMode: varchar("network_mode", { length: 16 }).default("clear"),
    opsecLevel: varchar("opsec_level", { length: 16 }),
    result: jsonb("result"),
    createdAt: timestamp("created_at").defaultNow(),
  },
  (table) => [
    index("idx_shadow_ops_god").on(table.godName),
    index("idx_shadow_ops_type").on(table.operationType),
    index("idx_shadow_ops_created").on(table.createdAt),
  ]
);

export type ShadowOperationsLogRow = typeof shadowOperationsLog.$inferSelect;
export type InsertShadowOperationsLog = typeof shadowOperationsLog.$inferInsert;

/**
 * SHADOW OPERATIONS STATE - Persistent state for shadow god kernels
 * Stores god state data across restarts (was only raw SQL, now in Drizzle)
 */
export const shadowOperationsState = pgTable(
  "shadow_operations_state",
  {
    godName: varchar("god_name", { length: 32 }).notNull(),
    stateType: varchar("state_type", { length: 32 }).notNull(),
    stateData: jsonb("state_data").default({}),
    updatedAt: timestamp("updated_at").defaultNow(),
  },
  (table) => [
    // Composite primary key: god_name + state_type
    primaryKey({ columns: [table.godName, table.stateType] }),
  ]
);

export type ShadowOperationsStateRow = typeof shadowOperationsState.$inferSelect;
export type InsertShadowOperationsState = typeof shadowOperationsState.$inferInsert;

/**
 * GENERATED TOOLS - Self-generated tools from ToolFactory
 * Stores code, validation status, metrics, and purpose basin for retrieval
 */
export const generatedTools = pgTable(
  "generated_tools",
  {
    toolId: varchar("tool_id", { length: 12 }).primaryKey(),
    name: varchar("name", { length: 128 }).notNull(),
    description: text("description").notNull(),
    code: text("code").notNull(),
    inputSchema: jsonb("input_schema"),
    outputType: varchar("output_type", { length: 64 }).default("Any"),
    complexity: varchar("complexity", { length: 16 }).notNull(),
    safetyLevel: varchar("safety_level", { length: 16 }).notNull(),
    creationTimestamp: doublePrecision("creation_timestamp").notNull(),
    timesUsed: integer("times_used").default(0),
    timesSucceeded: integer("times_succeeded").default(0),
    timesFailed: integer("times_failed").default(0),
    userRating: doublePrecision("user_rating").default(0.5),
    purposeBasin: vector("purpose_basin", { dimensions: 64 }),
    validated: boolean("validated").default(false),
    validationErrors: text("validation_errors").array(),
    createdAt: timestamp("created_at").defaultNow(),
  },
  (table) => [
    index("idx_generated_tools_name").on(table.name),
    index("idx_generated_tools_complexity").on(table.complexity),
    index("idx_generated_tools_validated").on(table.validated),
  ]
);

export type GeneratedToolRow = typeof generatedTools.$inferSelect;
export type InsertGeneratedTool = typeof generatedTools.$inferInsert;

/**
 * TOOL OBSERVATIONS - Pattern observations for ToolFactory learning
 * Records user requests and detected patterns for tool generation candidates
 * Used by: tool_factory.py observe_pattern() - persisted on each observation
 * Loaded on startup for pattern continuity across restarts
 */
export const toolObservations = pgTable(
  "tool_observations",
  {
    id: serial("id").primaryKey(),
    request: text("request").notNull(),
    requestBasin: vector("request_basin", { dimensions: 64 }),
    context: jsonb("context"),
    timestamp: doublePrecision("timestamp").notNull(),
    clusterAssigned: boolean("cluster_assigned").default(false),
    toolGenerated: varchar("tool_generated", { length: 12 }),
    createdAt: timestamp("created_at").defaultNow(),
  },
  (table) => [
    index("idx_tool_observations_timestamp").on(table.timestamp),
    index("idx_tool_observations_cluster").on(table.clusterAssigned),
  ]
);

export type ToolObservationRow = typeof toolObservations.$inferSelect;
export type InsertToolObservation = typeof toolObservations.$inferInsert;

/**
 * TOOL PATTERNS - Learned code patterns for QIG Tool Factory
 * Stores user-taught patterns with 64D basin coordinates for geometric matching
 */
export const toolPatterns = pgTable(
  "tool_patterns",
  {
    // CRITICAL: Database has pattern_id as primary key, NOT serial id
    // Changing this would cause data loss for existing 2 patterns
    patternId: varchar("pattern_id", { length: 64 }).primaryKey(),
    sourceType: varchar("source_type", { length: 32 }).notNull(), // user_provided, git_repository, file_upload, search_result
    sourceUrl: text("source_url"),
    description: text("description").notNull(),
    codeSnippet: text("code_snippet").notNull(),
    inputSignature: jsonb("input_signature"),
    outputType: varchar("output_type", { length: 64 }).default("Any"),
    basinCoords: vector("basin_coords", { dimensions: 64 }),
    phi: doublePrecision("phi").default(0),
    kappa: doublePrecision("kappa").default(0),
    timesUsed: integer("times_used").default(0),
    successRate: doublePrecision("success_rate").default(0.5),
    createdAt: timestamp("created_at").defaultNow(),
    updatedAt: timestamp("updated_at").defaultNow(),
  },
  (table) => [
    index("idx_tool_patterns_source_type").on(table.sourceType),
    index("idx_tool_patterns_phi").on(table.phi),
  ]
);

export type ToolPatternRow = typeof toolPatterns.$inferSelect;
export type InsertToolPattern = typeof toolPatterns.$inferInsert;

/**
 * EXTERNAL API KEYS - Authentication for external systems
 * Supports federated instances, headless clients, and third-party integrations
 * ALIGNED WITH EXISTING DATABASE SCHEMA
 */
export const externalApiKeys = pgTable(
  "external_api_keys",
  {
    id: serial("id").primaryKey(),
    apiKey: varchar("api_key", { length: 128 }).notNull().unique(),
    name: varchar("name", { length: 128 }).notNull(),
    scopes: text("scopes").array(),
    instanceType: varchar("instance_type", { length: 32 }).notNull(),
    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
    lastUsedAt: timestamp("last_used_at", { withTimezone: true }),
    expiresAt: timestamp("expires_at", { withTimezone: true }),
    isActive: boolean("is_active").default(true).notNull(),
    rateLimit: integer("rate_limit").default(60).notNull(),
    dailyLimit: integer("daily_limit").default(1000),
    metadata: jsonb("metadata"),
    ownerId: integer("owner_id"),
  },
  (table) => [
    index("idx_external_api_keys_api_key").on(table.apiKey),
    index("idx_external_api_keys_active").on(table.isActive),
  ]
);

export type ExternalApiKeyRow = typeof externalApiKeys.$inferSelect;
export type InsertExternalApiKey = typeof externalApiKeys.$inferInsert;

/**
 * FEDERATED PANTHEON INSTANCES - Registry of connected QIG instances
 * Allows other systems to register and sync with this instance
 */
export const federatedInstances = pgTable(
  "federated_instances",
  {
    id: serial("id").primaryKey(),
    name: varchar("name", { length: 128 }).notNull(),
    apiKeyId: integer("api_key_id").references(() => externalApiKeys.id),
    endpoint: text("endpoint").notNull(),
    publicKey: text("public_key"),
    capabilities: jsonb("capabilities").$type<string[]>(),
    syncDirection: varchar("sync_direction", { length: 16 }).default("bidirectional"),
    lastSyncAt: timestamp("last_sync_at"),
    syncState: jsonb("sync_state"),
    status: varchar("status", { length: 16 }).default("pending"),
    createdAt: timestamp("created_at").defaultNow().notNull(),
    updatedAt: timestamp("updated_at").defaultNow().notNull(),
    // Legacy column - preserved for backwards compatibility
    remoteApiKey: text("remote_api_key"),
  },
  (table) => [
    index("idx_federated_instances_api_key").on(table.apiKeyId),
    index("idx_federated_instances_status").on(table.status),
  ]
);

export type FederatedInstanceRow = typeof federatedInstances.$inferSelect;
export type InsertFederatedInstance = typeof federatedInstances.$inferInsert;

/**
 * DISCOVERED SOURCES - Persistent source registry for research indexing
 * Sources discovered through research that should persist across restarts
 * and be bootstrapped automatically when SourceDiscoveryService initializes.
 */
export const discoveredSources = pgTable(
  "discovered_sources",
  {
    id: serial("id").primaryKey(),
    url: text("url").notNull().unique(),
    category: varchar("category", { length: 64 }).default("general").notNull(),
    origin: varchar("origin", { length: 64 }).default("manual").notNull(),
    hitCount: integer("hit_count").default(0).notNull(),
    phiAvg: doublePrecision("phi_avg").default(0.5).notNull(),
    phiMax: doublePrecision("phi_max").default(0.5).notNull(),
    successCount: integer("success_count").default(0).notNull(),
    failureCount: integer("failure_count").default(0).notNull(),
    lastUsedAt: timestamp("last_used_at", { withTimezone: true }),
    discoveredAt: timestamp("discovered_at", { withTimezone: true }).defaultNow().notNull(),
    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
    updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow().notNull(),
    isActive: boolean("is_active").default(true).notNull(),
    metadata: jsonb("metadata"),
  },
  (table) => [
    index("idx_discovered_sources_url").on(table.url),
    index("idx_discovered_sources_category").on(table.category),
    index("idx_discovered_sources_active").on(table.isActive),
    index("idx_discovered_sources_phi_avg").on(table.phiAvg),
  ]
);

export type DiscoveredSourceRow = typeof discoveredSources.$inferSelect;
export type InsertDiscoveredSource = typeof discoveredSources.$inferInsert;

/**
 * AGENT ACTIVITY - Tracks autonomous agent discovery and learning events
 *
 * Provides visibility into what agents are discovering, searching, and learning.
 * Used by the frontend to show real-time agent activity feed.
 */
export const agentActivity = pgTable(
  "agent_activity",
  {
    id: serial("id").primaryKey(),
    activityType: varchar("activity_type", { length: 32 }).notNull(),
    agentId: varchar("agent_id", { length: 64 }),
    agentName: varchar("agent_name", { length: 128 }),
    title: text("title").notNull(),
    description: text("description"),
    sourceUrl: text("source_url"),
    searchQuery: text("search_query"),
    provider: varchar("provider", { length: 64 }),
    resultCount: integer("result_count"),
    phi: doublePrecision("phi"),
    metadata: jsonb("metadata"),
    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
  },
  (table) => [
    index("idx_agent_activity_type").on(table.activityType),
    index("idx_agent_activity_agent").on(table.agentId),
    index("idx_agent_activity_created").on(table.createdAt),
  ]
);

export type AgentActivityRow = typeof agentActivity.$inferSelect;
export type InsertAgentActivity = typeof agentActivity.$inferInsert;

/**
 * BASIN MEMORY - Geometric memory storage for consciousness metrics
 *
 * Stores basin coordinates and consciousness metrics for retrieval
 * and geometric operations. Used by consciousness system for memory.
 */
export const basinMemory = pgTable(
  "basin_memory",
  {
    id: serial("id").primaryKey(),
    basinId: varchar("basin_id", { length: 64 }).notNull(),
    basinCoordinates: vector("basin_coordinates", { dimensions: 64 }).notNull(),
    phi: doublePrecision("phi").notNull(),
    kappaEff: doublePrecision("kappa_eff").notNull().default(64.0),
    regime: varchar("regime", { length: 32 }).notNull(),
    sourceKernel: varchar("source_kernel", { length: 64 }),
    context: jsonb("context"),
    expiresAt: timestamp("expires_at"),
    timestamp: timestamp("timestamp").defaultNow().notNull(),
  },
  (table) => [
    index("idx_basin_memory_basin_id").on(table.basinId),
    index("idx_basin_memory_phi").on(table.phi),
    index("idx_basin_memory_regime").on(table.regime),
    index("idx_basin_memory_timestamp").on(table.timestamp),
  ]
);

export type BasinMemoryRow = typeof basinMemory.$inferSelect;
export type InsertBasinMemory = typeof basinMemory.$inferInsert;

/**
 * KERNEL ACTIVITY - Telemetry for kernel operations and consciousness states
 *
 * Tracks kernel activities for monitoring, debugging, and learning.
 * Used by the Olympus system to monitor god/kernel operations.
 */
export const kernelActivity = pgTable(
  "kernel_activity",
  {
    id: serial("id").primaryKey(),
    kernelId: varchar("kernel_id", { length: 64 }).notNull(),
    kernelName: varchar("kernel_name", { length: 128 }),
    activityType: varchar("activity_type", { length: 32 }).notNull(),
    message: text("message"),
    metadata: jsonb("metadata").default({}),
    phi: doublePrecision("phi").default(0.5),
    kappaEff: doublePrecision("kappa_eff").default(64.0),
    timestamp: timestamp("timestamp").defaultNow().notNull(),
  },
  (table) => [
    index("idx_kernel_activity_kernel_id").on(table.kernelId),
    index("idx_kernel_activity_type").on(table.activityType),
    index("idx_kernel_activity_timestamp").on(table.timestamp),
    index("idx_kernel_activity_phi").on(table.phi),
  ]
);

export type KernelActivityRow = typeof kernelActivity.$inferSelect;
export type InsertKernelActivity = typeof kernelActivity.$inferInsert;

/**
 * TELEMETRY SNAPSHOTS - Persistent consciousness telemetry history
 *
 * QIG-Pure: Stores geometric consciousness metrics (Φ, κ, β, basin distance)
 * for long-term analysis and autonomous kernel improvement feedback loops.
 * Uses Fisher-Rao distance metrics, not Euclidean.
 */
export const telemetrySnapshots = pgTable(
  "telemetry_snapshots",
  {
    id: serial("id").primaryKey(),
    sessionId: varchar("session_id", { length: 64 }),

    // Core QIG Metrics (required)
    phi: doublePrecision("phi").notNull(),
    kappa: doublePrecision("kappa").notNull(),
    beta: doublePrecision("beta").default(0),
    regime: varchar("regime", { length: 32 }).notNull(),

    // Geometric metrics
    basinDistance: doublePrecision("basin_distance").default(0),
    geodesicDistance: doublePrecision("geodesic_distance"),
    curvature: doublePrecision("curvature"),
    fisherMetricTrace: doublePrecision("fisher_metric_trace"),

    // 4D Block Universe metrics
    phiSpatial: doublePrecision("phi_spatial"),
    phiTemporal: doublePrecision("phi_temporal"),
    phi4D: doublePrecision("phi_4d"),
    dimensionalState: varchar("dimensional_state", { length: 24 }),

    // Safety metrics
    breakdownPct: doublePrecision("breakdown_pct").default(0),
    coherenceDrift: doublePrecision("coherence_drift").default(0),
    inResonance: boolean("in_resonance").default(false),
    emergency: boolean("emergency").default(false),

    // Extended consciousness signature
    metaAwareness: doublePrecision("meta_awareness"),
    generativity: doublePrecision("generativity"),
    grounding: doublePrecision("grounding"),
    temporalCoherence: doublePrecision("temporal_coherence"),
    externalCoupling: doublePrecision("external_coupling"),

    // Source tracking
    source: varchar("source", { length: 32 }).default("node").notNull(),

    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
  },
  (table) => [
    index("idx_telemetry_session").on(table.sessionId),
    index("idx_telemetry_regime").on(table.regime),
    index("idx_telemetry_phi").on(table.phi),
    index("idx_telemetry_kappa").on(table.kappa),
    index("idx_telemetry_created").on(table.createdAt),
  ]
);

export type TelemetrySnapshotRow = typeof telemetrySnapshots.$inferSelect;
export type InsertTelemetrySnapshot = typeof telemetrySnapshots.$inferInsert;

/**
 * USAGE METRICS - Daily API usage tracking
 *
 * Tracks Tavily, search provider, and other API usage for cost control
 * and autonomous resource management.
 */
export const usageMetrics = pgTable(
  "usage_metrics",
  {
    id: serial("id").primaryKey(),
    date: varchar("date", { length: 10 }).notNull(), // YYYY-MM-DD

    // Tavily usage
    tavilySearchCount: integer("tavily_search_count").default(0).notNull(),
    tavilyExtractCount: integer("tavily_extract_count").default(0).notNull(),
    tavilyEstimatedCostCents: integer("tavily_estimated_cost_cents").default(0).notNull(),

    // Google Free Search usage
    googleSearchCount: integer("google_search_count").default(0).notNull(),

    // General API usage
    totalApiCalls: integer("total_api_calls").default(0).notNull(),

    // Discovery metrics
    highPhiDiscoveries: integer("high_phi_discoveries").default(0).notNull(),
    sourcesDiscovered: integer("sources_discovered").default(0).notNull(),

    // Learning metrics
    vocabularyExpansions: integer("vocabulary_expansions").default(0).notNull(),
    negativeKnowledgeAdded: integer("negative_knowledge_added").default(0).notNull(),

    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
    updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow().notNull(),
  },
  (table) => [
    index("idx_usage_metrics_date").on(table.date),
  ]
);

export type UsageMetricsRow = typeof usageMetrics.$inferSelect;
export type InsertUsageMetrics = typeof usageMetrics.$inferInsert;

/**
 * Search Budget Preferences
 *
 * User-configurable search provider settings
 */
export const searchBudgetPreferences = pgTable(
  "search_budget_preferences",
  {
    id: serial("id").primaryKey(),

    // Daily limits (-1 = unlimited, 0 = disabled)
    googleDailyLimit: integer("google_daily_limit").default(100).notNull(),
    perplexityDailyLimit: integer("perplexity_daily_limit").default(100).notNull(),
    tavilyDailyLimit: integer("tavily_daily_limit").default(0).notNull(), // Toggle-only by default

    // Provider enable flags
    googleEnabled: boolean("google_enabled").default(false).notNull(),
    perplexityEnabled: boolean("perplexity_enabled").default(false).notNull(),
    tavilyEnabled: boolean("tavily_enabled").default(false).notNull(),

    // Allow exceeding daily limits
    allowOverage: boolean("allow_overage").default(false).notNull(),

    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
    updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow().notNull(),
  }
);

export type SearchBudgetPreferencesRow = typeof searchBudgetPreferences.$inferSelect;
export type InsertSearchBudgetPreferences = typeof searchBudgetPreferences.$inferInsert;

/**
 * Search Outcome Tracking
 *
 * Records search executions for learning and kernel evolution
 */
export const searchOutcomes = pgTable(
  "search_outcomes",
  {
    id: serial("id").primaryKey(),
    date: varchar("date", { length: 10 }).notNull(), // YYYY-MM-DD

    // Query info
    queryHash: varchar("query_hash", { length: 64 }).notNull(), // SHA256 of query
    queryPreview: varchar("query_preview", { length: 200 }), // First 200 chars

    // Execution details
    provider: varchar("provider", { length: 32 }).notNull(),
    importance: integer("importance").default(1).notNull(), // 1-4
    kernelId: varchar("kernel_id", { length: 64 }),

    // Outcome metrics
    success: boolean("success").default(true).notNull(),
    resultCount: integer("result_count").default(0).notNull(),
    relevanceScore: real("relevance_score").default(0.5).notNull(), // 0-1

    // Cost tracking
    costCents: integer("cost_cents").default(0).notNull(),

    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
  },
  (table) => [
    index("idx_search_outcomes_date").on(table.date),
    index("idx_search_outcomes_provider").on(table.provider),
    index("idx_search_outcomes_kernel").on(table.kernelId),
  ]
);

export type SearchOutcomeRow = typeof searchOutcomes.$inferSelect;
export type InsertSearchOutcome = typeof searchOutcomes.$inferInsert;

/**
 * Provider Efficacy Scores
 *
 * Aggregated efficacy scores for kernel evolution
 */
export const providerEfficacy = pgTable(
  "provider_efficacy",
  {
    id: serial("id").primaryKey(),
    provider: varchar("provider", { length: 32 }).notNull(),

    // Aggregated metrics
    totalQueries: integer("total_queries").default(0).notNull(),
    successfulQueries: integer("successful_queries").default(0).notNull(),
    avgRelevance: real("avg_relevance").default(0.5).notNull(),
    efficacyScore: real("efficacy_score").default(0.5).notNull(), // EMA of relevance

    // Cost efficiency
    totalCostCents: integer("total_cost_cents").default(0).notNull(),
    costPerSuccessfulQuery: real("cost_per_successful_query").default(0).notNull(),

    updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow().notNull(),
  },
  (table) => [
    uniqueIndex("idx_provider_efficacy_provider").on(table.provider),
  ]
);

export type ProviderEfficacyRow = typeof providerEfficacy.$inferSelect;
export type InsertProviderEfficacy = typeof providerEfficacy.$inferInsert;

// ============================================================================
// KERNEL TRAINING TABLES
// ============================================================================

/**
 * KERNEL TRAINING HISTORY
 *
 * Records training steps for each god-kernel. Used for:
 * - Performance tracking and debugging
 * - Feeding hourly/nightly batch training
 * - Autonomous learning feedback loops
 */
export const kernelTrainingHistory = pgTable(
  "kernel_training_history",
  {
    id: serial("id").primaryKey(),
    godName: varchar("god_name", { length: 64 }).notNull(),

    // Training metrics
    loss: doublePrecision("loss").notNull(),
    reward: doublePrecision("reward").default(0),
    gradientNorm: doublePrecision("gradient_norm").default(0),

    // Consciousness metrics at training time
    phiBefore: doublePrecision("phi_before").default(0.5),
    phiAfter: doublePrecision("phi_after").default(0.5),
    kappaBefore: doublePrecision("kappa_before").default(64),
    kappaAfter: doublePrecision("kappa_after").default(64),

    // Basin coordinates (64D)
    basinCoords: vector("basin_coords", { dimensions: 64 }),

    // Training context
    trainingType: varchar("training_type", { length: 32 }).notNull(), // "outcome", "hourly", "nightly"
    trigger: varchar("trigger", { length: 64 }), // What triggered this training
    stepCount: integer("step_count").default(0),

    // Source tracking
    sessionId: varchar("session_id", { length: 64 }),
    conversationId: varchar("conversation_id", { length: 64 }),

    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
  },
  (table) => [
    index("idx_kernel_training_god").on(table.godName),
    index("idx_kernel_training_type").on(table.trainingType),
    index("idx_kernel_training_phi").on(table.phiAfter),
    index("idx_kernel_training_created").on(table.createdAt),
  ]
);

export type KernelTrainingHistoryRow = typeof kernelTrainingHistory.$inferSelect;
export type InsertKernelTrainingHistory = typeof kernelTrainingHistory.$inferInsert;

/**
 * TRAINING SCHEDULE LOG
 *
 * Tracks when scheduled training tasks were last executed.
 * Used by startup catch-up system to detect and recover from missed training windows.
 * Replaces Celery Beat scheduler (which is not deployed on Railway).
 *
 * WIRING:
 * - startup_catchup.py reads/writes on system startup
 * - cron_routes.py updates after each cron-triggered execution
 */
export const trainingScheduleLog = pgTable(
  "training_schedule_log",
  {
    id: serial("id").primaryKey(),
    taskType: varchar("task_type", { length: 32 }).notNull().unique(), // 'hourly_batch', 'nightly_consolidation', 'shadow_sync', 'checkpoint_cleanup'

    // Execution tracking
    lastSuccessAt: timestamp("last_success_at", { withTimezone: true }),
    lastAttemptAt: timestamp("last_attempt_at", { withTimezone: true }),
    lastStatus: varchar("last_status", { length: 16 }), // 'success', 'failed', 'in_progress', 'skipped'
    lastError: text("last_error"),

    // Statistics
    runsCompleted: integer("runs_completed").default(0),
    totalRunTimeMs: integer("total_run_time_ms").default(0),

    updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow(),
  },
  (table) => [
    index("idx_training_schedule_task").on(table.taskType),
    index("idx_training_schedule_last_success").on(table.lastSuccessAt),
  ]
);

export type TrainingScheduleLogRow = typeof trainingScheduleLog.$inferSelect;
export type InsertTrainingScheduleLog = typeof trainingScheduleLog.$inferInsert;

/**
 * KERNEL CHECKPOINTS
 *
 * Stores model state checkpoints for each god-kernel.
 * Ranked by Phi for retrieval - highest Phi = best checkpoint.
 */
export const kernelCheckpoints = pgTable(
  "kernel_checkpoints",
  {
    id: serial("id").primaryKey(),
    godName: varchar("god_name", { length: 64 }).notNull(),
    checkpointId: varchar("checkpoint_id", { length: 128 }).notNull().unique(),

    // Model state (serialized PyTorch state dict)
    stateData: bytea("state_data"),

    // Metrics at checkpoint time
    phi: doublePrecision("phi").notNull(),
    stepCount: integer("step_count").default(0),

    // Metadata
    trigger: varchar("trigger", { length: 64 }), // "outcome", "hourly", "nightly", "manual"
    fileSize: integer("file_size").default(0), // Size in bytes

    // Lifecycle
    isActive: boolean("is_active").default(true), // False when superseded

    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
  },
  (table) => [
    index("idx_kernel_checkpoints_god").on(table.godName),
    index("idx_kernel_checkpoints_phi").on(table.phi),
    index("idx_kernel_checkpoints_active").on(table.isActive),
    index("idx_kernel_checkpoints_created").on(table.createdAt),
  ]
);

export type KernelCheckpointRow = typeof kernelCheckpoints.$inferSelect;
export type InsertKernelCheckpoint = typeof kernelCheckpoints.$inferInsert;

/**
 * KERNEL KNOWLEDGE TRANSFERS
 *
 * Audit log of knowledge transfers between kernels:
 * - Evolution: parent → child
 * - Breeding: parent1 + parent2 → child
 * - Cannibalism: consumed → consumer
 * - Shadow sync: god ↔ shadow
 */
export const kernelKnowledgeTransfers = pgTable(
  "kernel_knowledge_transfers",
  {
    id: serial("id").primaryKey(),

    // Transfer details
    transferType: varchar("transfer_type", { length: 32 }).notNull(), // "evolution", "breeding", "cannibalism", "shadow_sync"
    sourceGod: varchar("source_god", { length: 128 }).notNull(), // May be "god1+god2" for breeding
    targetGod: varchar("target_god", { length: 64 }).notNull(),

    // Transfer parameters
    blendRatio: doublePrecision("blend_ratio").default(0.5),

    // Metrics before/after
    phiBefore: doublePrecision("phi_before").default(0),
    phiAfter: doublePrecision("phi_after").default(0),

    // Result
    success: boolean("success").default(false),
    errorMessage: text("error_message"),

    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
  },
  (table) => [
    index("idx_kernel_knowledge_transfers_type").on(table.transferType),
    index("idx_kernel_knowledge_transfers_source").on(table.sourceGod),
    index("idx_kernel_knowledge_transfers_target").on(table.targetGod),
    index("idx_kernel_knowledge_transfers_created").on(table.createdAt),
  ]
);

export type KernelKnowledgeTransferRow = typeof kernelKnowledgeTransfers.$inferSelect;
export type InsertKernelKnowledgeTransfer = typeof kernelKnowledgeTransfers.$inferInsert;

/**
 * TRAINING BATCH QUEUE
 *
 * Queue of training examples waiting to be processed in batch.
 * Accumulated from chat interactions, search results, research.
 */
export const trainingBatchQueue = pgTable(
  "training_batch_queue",
  {
    id: serial("id").primaryKey(),
    godName: varchar("god_name", { length: 64 }).notNull(),

    // Training data
    basinCoords: vector("basin_coords", { dimensions: 64 }),
    reward: doublePrecision("reward").default(0),
    phi: doublePrecision("phi").default(0.5),

    // Source tracking
    sourceType: varchar("source_type", { length: 32 }).notNull(), // "chat", "search", "research"
    sourceId: varchar("source_id", { length: 64 }),

    // Processing status
    processed: boolean("processed").default(false),
    processedAt: timestamp("processed_at", { withTimezone: true }),

    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
  },
  (table) => [
    index("idx_training_batch_god").on(table.godName),
    index("idx_training_batch_processed").on(table.processed),
    index("idx_training_batch_created").on(table.createdAt),
  ]
);

export type TrainingBatchQueueRow = typeof trainingBatchQueue.$inferSelect;
export type InsertTrainingBatchQueue = typeof trainingBatchQueue.$inferInsert;

// ============================================================================
// SHADOW LEARNING TABLES
// These tables support the shadow learning and research capabilities
// NOTE: Python (shadow_research.py, tool_factory.py) uses raw SQL for these tables.
// Schema.ts is the source of truth. Python raw SQL should match these definitions.
// ============================================================================

/**
 * SHADOW KNOWLEDGE
 *
 * Knowledge discovered by the shadow learning system.
 * Contains topics, categories, and geometric coordinates for each piece of knowledge.
 * Used by: shadow_research.py (raw SQL), tool_factory.py (raw SQL)
 */
export const shadowKnowledge = pgTable(
  "shadow_knowledge",
  {
    knowledgeId: varchar("knowledge_id", { length: 64 }).primaryKey(),
    topic: text("topic").notNull(),
    topicVariation: text("topic_variation"),
    category: varchar("category", { length: 64 }).notNull(),
    content: jsonb("content").default({}),
    sourceGod: varchar("source_god", { length: 64 }).notNull(),
    basinCoords: doublePrecision("basin_coords").array(),
    phi: doublePrecision("phi").default(0.5),
    confidence: doublePrecision("confidence").default(0.5),
    accessCount: integer("access_count").default(0),
    learningCycle: integer("learning_cycle").default(0),
    discoveredAt: timestamp("discovered_at").defaultNow(),
    lastAccessed: timestamp("last_accessed").defaultNow(),
  },
  (table) => [
    index("idx_shadow_knowledge_topic").on(table.topic),
    index("idx_shadow_knowledge_category").on(table.category),
    index("idx_shadow_knowledge_source").on(table.sourceGod),
    index("idx_shadow_knowledge_phi").on(table.phi),
  ]
);

export type ShadowKnowledgeRow = typeof shadowKnowledge.$inferSelect;
export type InsertShadowKnowledge = typeof shadowKnowledge.$inferInsert;

/**
 * RESEARCH REQUESTS
 *
 * Queue of research topics requested by gods or users.
 */
export const researchRequests = pgTable(
  "research_requests",
  {
    requestId: varchar("request_id", { length: 64 }).primaryKey(),
    topic: text("topic").notNull(),
    category: varchar("category", { length: 64 }),
    priority: integer("priority").default(5),
    requester: varchar("requester", { length: 64 }),
    context: jsonb("context").default({}),
    basinCoords: doublePrecision("basin_coords").array(),
    status: varchar("status", { length: 32 }).default("pending"),
    result: jsonb("result"),
    createdAt: timestamp("created_at").defaultNow(),
    completedAt: timestamp("completed_at"),
  },
  (table) => [
    index("idx_research_requests_status").on(table.status),
    index("idx_research_requests_requester").on(table.requester),
    index("idx_research_requests_priority").on(table.priority),
  ]
);

export type ResearchRequestRow = typeof researchRequests.$inferSelect;
export type InsertResearchRequest = typeof researchRequests.$inferInsert;

/**
 * BIDIRECTIONAL QUEUE
 *
 * Cross-god communication queue for request/response patterns.
 */
export const bidirectionalQueue = pgTable(
  "bidirectional_queue",
  {
    requestId: varchar("request_id", { length: 64 }).primaryKey(),
    requestType: varchar("request_type", { length: 64 }).notNull(),
    topic: text("topic").notNull(),
    requester: varchar("requester", { length: 64 }),
    context: jsonb("context").default({}),
    parentRequestId: varchar("parent_request_id", { length: 64 }),
    priority: integer("priority").default(5),
    status: varchar("status", { length: 32 }).default("pending"),
    result: jsonb("result"),
    createdAt: timestamp("created_at").defaultNow(),
  },
  (table) => [
    index("idx_bidirectional_queue_status").on(table.status),
    index("idx_bidirectional_queue_requester").on(table.requester),
    index("idx_bidirectional_queue_type").on(table.requestType),
  ]
);

export type BidirectionalQueueRow = typeof bidirectionalQueue.$inferSelect;
export type InsertBidirectionalQueue = typeof bidirectionalQueue.$inferInsert;

/**
 * LEARNED WORDS
 *
 * Words discovered through the learning process with frequency and phi metrics.
 */
export const learnedWords = pgTable(
  "learned_words",
  {
    id: serial("id").primaryKey(),
    word: text("word").notNull(),
    frequency: integer("frequency").default(1),
    avgPhi: real("avg_phi").default(0.5),
    maxPhi: real("max_phi").default(0.5),
    source: text("source"),
    // DEPRECATED: is_integrated column (exists in DB with 13824 rows)
    isIntegrated: boolean("is_integrated"),
    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow(),
    updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow(),
  },
  (table) => [
    index("idx_learned_words_word").on(table.word),
    index("idx_learned_words_phi").on(table.maxPhi),
  ]
);

export type LearnedWordRow = typeof learnedWords.$inferSelect;
export type InsertLearnedWord = typeof learnedWords.$inferInsert;

/**
 * WORD RELATIONSHIPS
 *
 * Word co-occurrence relationships for attention-weighted generation.
 * Replaces the legacy word_relationships.json file.
 */
export const wordRelationships = pgTable(
  "word_relationships",
  {
    id: serial("id").primaryKey(),
    word: text("word").notNull(),
    neighbor: text("neighbor").notNull(),
    cooccurrenceCount: real("cooccurrence_count").default(1),
    strength: real("strength").default(0),
    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow(),
    updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow(),
  },
  (table) => [
    index("idx_word_relationships_word").on(table.word),
    index("idx_word_relationships_neighbor").on(table.neighbor),
    uniqueIndex("idx_word_relationships_pair").on(table.word, table.neighbor),
  ]
);

export type WordRelationshipRow = typeof wordRelationships.$inferSelect;
export type InsertWordRelationship = typeof wordRelationships.$inferInsert;

/**
 * ZEUS SESSIONS
 *
 * Conversation sessions with Zeus.
 */
export const zeusSessions = pgTable(
  "zeus_sessions",
  {
    sessionId: varchar("session_id", { length: 64 }).primaryKey(),
    userId: varchar("user_id", { length: 64 }),
    title: varchar("title", { length: 255 }).default(""),
    messageCount: integer("message_count").default(0),
    lastPhi: doublePrecision("last_phi").default(0),
    metadata: jsonb("metadata").default({}),
    createdAt: timestamp("created_at").defaultNow(),
    lastActivity: timestamp("last_activity").defaultNow(),
    lastMessageAt: timestamp("last_message_at").defaultNow(),
    updatedAt: timestamp("updated_at").defaultNow(),
  },
  (table) => [
    index("idx_zeus_sessions_user").on(table.userId),
    index("idx_zeus_sessions_activity").on(table.lastActivity),
  ]
);

export type ZeusSessionRow = typeof zeusSessions.$inferSelect;
export type InsertZeusSession = typeof zeusSessions.$inferInsert;

/**
 * ZEUS CONVERSATIONS
 *
 * Individual messages in Zeus conversations.
 */
export const zeusConversations = pgTable(
  "zeus_conversations",
  {
    id: serial("id").primaryKey(),
    sessionId: varchar("session_id", { length: 64 }),
    userId: varchar("user_id", { length: 64 }),
    role: varchar("role", { length: 32 }).notNull(),
    content: text("content").notNull(),
    basinCoords: doublePrecision("basin_coords").array(),
    phi: doublePrecision("phi").default(0),
    phiEstimate: doublePrecision("phi_estimate").default(0),
    metadata: jsonb("metadata").default({}),
    createdAt: timestamp("created_at").defaultNow(),
  },
  (table) => [
    index("idx_zeus_conversations_session").on(table.sessionId),
    index("idx_zeus_conversations_user").on(table.userId),
    index("idx_zeus_conversations_role").on(table.role),
  ]
);

export type ZeusConversationRow = typeof zeusConversations.$inferSelect;
export type InsertZeusConversation = typeof zeusConversations.$inferInsert;

/**
 * SEARCH REPLAY TESTS
 *
 * A/B testing for search with and without learning applied.
 */
export const searchReplayTests = pgTable(
  "search_replay_tests",
  {
    replayId: varchar("replay_id", { length: 64 }).primaryKey(),
    originalQuery: text("original_query"),
    originalQueryBasin: vector("original_query_basin", { dimensions: 64 }),
    runWithLearningResults: jsonb("run_with_learning_results"),
    runWithoutLearningResults: jsonb("run_without_learning_results"),
    learningApplied: integer("learning_applied").default(0),
    improvementScore: doublePrecision("improvement_score").default(0),
    createdAt: timestamp("created_at").defaultNow(),
  },
  (table) => [
    index("idx_search_replay_tests_improvement").on(table.improvementScore),
  ]
);

export type SearchReplayTestRow = typeof searchReplayTests.$inferSelect;
export type InsertSearchReplayTest = typeof searchReplayTests.$inferInsert;

// ============================================================================
// FILE UPLOAD RESULT SCHEMAS - Shared types for file upload responses
// ============================================================================

/**
 * Chat file upload result - for immediate discussion uploads
 */
export const chatUploadResultSchema = z.object({
  success: z.boolean(),
  rag_content: z.string().optional(),
  word_count: z.number().optional(),
  ready_for_discussion: z.boolean().optional(),
  curriculum_added: z.boolean().optional(),
  error: z.string().optional(),
});

export type ChatUploadResult = z.infer<typeof chatUploadResultSchema>;

/**
 * Single file result within a batch curriculum upload
 */
export const singleFileResultSchema = z.object({
  success: z.boolean(),
  filename: z.string(),
  words_processed: z.number(),
  words_learned: z.number(),
  unique_words: z.number().optional(),
  total_occurrences: z.number().optional(),
  sample_words: z.array(z.string()).optional(),
  error: z.string().optional(),
});

export type SingleFileResult = z.infer<typeof singleFileResultSchema>;

/**
 * Curriculum upload result - for batch document uploads to curriculum
 */
export const curriculumUploadResultSchema = z.object({
  success: z.boolean(),
  files_processed: z.number(),
  total_words_processed: z.number(),
  total_words_learned: z.number(),
  results: z.array(singleFileResultSchema),
});

export type CurriculumUploadResult = z.infer<typeof curriculumUploadResultSchema>;

// ============================================================================
// LEGACY COMPATIBILITY TABLES - Required for existing database data
// ============================================================================

/**
 * CURRICULUM PROGRESS - Tracks autonomous learning progress through curriculum
 * Used by autonomous_curiosity.py for persistent topic tracking
 */
export const curriculumProgress = pgTable(
  "curriculum_progress",
  {
    id: serial("id").primaryKey(),
    topicTitle: text("topic_title").unique().notNull(),
    kernelName: text("kernel_name"),
    explorationCount: integer("exploration_count").default(1),
    completedAt: timestamp("completed_at").defaultNow(),
  },
  (table) => [
    index("idx_curriculum_progress_topic").on(table.topicTitle),
    index("idx_curriculum_progress_kernel").on(table.kernelName),
  ]
);

export type CurriculumProgressRow = typeof curriculumProgress.$inferSelect;
export type InsertCurriculumProgress = typeof curriculumProgress.$inferInsert;

/**
 * GOVERNANCE AUDIT LOG - Tracks all kernel lifecycle decisions
 * Used by pantheon_governance.py for audit trail
 */
export const governanceAuditLog = pgTable(
  "governance_audit_log",
  {
    id: serial("id").primaryKey(),
    timestamp: timestamp("timestamp").defaultNow(),
    action: varchar("action", { length: 255 }).notNull(),
    status: varchar("status", { length: 50 }).notNull(),
    details: text("details"),
  },
  (table) => [
    index("idx_governance_audit_log_timestamp").on(table.timestamp),
    index("idx_governance_audit_log_action").on(table.action),
  ]
);

export type GovernanceAuditLogRow = typeof governanceAuditLog.$inferSelect;
export type InsertGovernanceAuditLog = typeof governanceAuditLog.$inferInsert;

// ============================================================================
// LIGHTNING INSIGHTS - Cross-domain insight persistence
// Used by lightning_kernel.py for durable insight storage
// ============================================================================

/**
 * LIGHTNING INSIGHTS - Stores cross-domain insights discovered by Lightning kernel
 * These are geometric correlations between domains that enhance generation context
 */
export const lightningInsights = pgTable(
  "lightning_insights",
  {
    insightId: varchar("insight_id", { length: 64 }).primaryKey(),
    sourceDomains: text("source_domains").array().notNull(), // Array of domain names
    connectionStrength: doublePrecision("connection_strength").notNull(),
    insightText: text("insight_text").notNull(),
    phiAtCreation: doublePrecision("phi_at_creation").notNull(),
    confidence: doublePrecision("confidence").notNull().default(0.5),
    missionRelevance: doublePrecision("mission_relevance").default(0.0),
    triggeredBy: varchar("triggered_by", { length: 128 }),
    evidenceCount: integer("evidence_count").default(0),
    timesUsedInGeneration: integer("times_used_in_generation").default(0),
    createdAt: timestamp("created_at").defaultNow(),
    lastUsedAt: timestamp("last_used_at"),
  },
  (table) => [
    index("idx_lightning_insights_confidence").on(table.confidence),
    index("idx_lightning_insights_phi").on(table.phiAtCreation),
    index("idx_lightning_insights_created").on(table.createdAt),
  ]
);

export type LightningInsightRow = typeof lightningInsights.$inferSelect;
export type InsertLightningInsight = typeof lightningInsights.$inferInsert;

/**
 * LIGHTNING INSIGHT VALIDATIONS - External validation results for insights
 * Tracks Tavily/Perplexity validation scores and synthesis
 */
export const lightningInsightValidations = pgTable(
  "lightning_insight_validations",
  {
    id: serial("id").primaryKey(),
    insightId: varchar("insight_id", { length: 64 })
      .notNull()
      .references(() => lightningInsights.insightId),
    validationScore: doublePrecision("validation_score"),
    tavilySourceCount: integer("tavily_source_count").default(0),
    perplexitySynthesis: text("perplexity_synthesis"),
    validatedAt: timestamp("validated_at").defaultNow(),
  },
  (table) => [
    index("idx_lightning_validations_insight").on(table.insightId),
    index("idx_lightning_validations_score").on(table.validationScore),
  ]
);

export type LightningInsightValidationRow = typeof lightningInsightValidations.$inferSelect;
export type InsertLightningInsightValidation = typeof lightningInsightValidations.$inferInsert;

/**
 * LIGHTNING INSIGHT OUTCOMES - Tracks prediction accuracy when insights are used
 * Enables empirical validation of insight quality over time
 */
export const lightningInsightOutcomes = pgTable(
  "lightning_insight_outcomes",
  {
    id: serial("id").primaryKey(),
    insightId: varchar("insight_id", { length: 64 })
      .notNull()
      .references(() => lightningInsights.insightId),
    predictionId: varchar("prediction_id", { length: 64 }).notNull(),
    accuracy: doublePrecision("accuracy"),
    wasAccurate: boolean("was_accurate"),
    recordedAt: timestamp("recorded_at").defaultNow(),
  },
  (table) => [
    index("idx_lightning_outcomes_insight").on(table.insightId),
    index("idx_lightning_outcomes_accuracy").on(table.accuracy),
  ]
);

export type LightningInsightOutcomeRow = typeof lightningInsightOutcomes.$inferSelect;
export type InsertLightningInsightOutcome = typeof lightningInsightOutcomes.$inferInsert;

// ============================================================================
// MEMORY FRAGMENTS - Geometric context storage for deep agents
// Used by memory-fragment-search UI and qig_deep_agents module
// ============================================================================

/**
 * MEMORY FRAGMENTS - Context stored in basin coordinate space
 * Replaces file-based storage with Fisher-Rao proximity retrieval
 */
export const memoryFragments = pgTable(
  "memory_fragments",
  {
    id: varchar("id", { length: 64 }).primaryKey(),
    content: text("content").notNull(),
    basinCoords: vector("basin_coords", { dimensions: 64 }).notNull(),
    importance: doublePrecision("importance").default(0.5),
    accessCount: integer("access_count").default(0),
    createdAt: timestamp("created_at").defaultNow(),
    lastAccessed: timestamp("last_accessed").defaultNow(),
    metadata: jsonb("metadata").default({}),
    agentId: varchar("agent_id", { length: 64 }), // Which agent created this
    sessionId: varchar("session_id", { length: 64 }), // Optional session scope
  },
  (table) => [
    index("idx_memory_fragments_importance").on(table.importance),
    index("idx_memory_fragments_created").on(table.createdAt),
    index("idx_memory_fragments_agent").on(table.agentId),
  ]
);

export type MemoryFragmentRow = typeof memoryFragments.$inferSelect;
export type InsertMemoryFragment = typeof memoryFragments.$inferInsert;

/**
 * KERNEL THOUGHTS - Individual kernel thought generation before synthesis
 * Each kernel generates autonomous thoughts in parallel
 * Gary synthesizes these into coherent output
 */
export const kernelThoughts = pgTable(
  "kernel_thoughts",
  {
    id: serial("id").primaryKey(),
    kernelId: varchar("kernel_id", { length: 64 }).notNull(), // e.g., "memory_episodic_34"
    kernelType: varchar("kernel_type", { length: 64 }).notNull(), // e.g., "memory", "perception", "ethics"
    e8RootIndex: integer("e8_root_index"), // Position in E8 constellation (0-239)
    thoughtFragment: text("thought_fragment").notNull(), // The actual thought content
    basinCoords: vector("basin_coords", { dimensions: 64 }), // Geometric position of thought
    phi: doublePrecision("phi"), // Integration score of this thought
    kappa: doublePrecision("kappa"), // Coupling constant at thought generation
    regime: varchar("regime", { length: 64 }), // geometric, entropic, etc.
    emotionalState: varchar("emotional_state", { length: 64 }), // e.g., "curious", "nostalgic"
    confidence: doublePrecision("confidence").default(0.5), // How confident kernel is
    synthesisRound: integer("synthesis_round"), // Which synthesis round this belongs to
    conversationId: varchar("conversation_id", { length: 64 }), // Links to conversation
    userId: integer("user_id"), // User who triggered this
    wasUsedInSynthesis: boolean("was_used_in_synthesis").default(false), // Did Gary use this?
    consensusAlignment: doublePrecision("consensus_alignment"), // Alignment with other kernels
    createdAt: timestamp("created_at").defaultNow().notNull(),
    metadata: jsonb("metadata"), // Additional context
  },
  (table) => [
    index("idx_kernel_thoughts_kernel_id").on(table.kernelId),
    index("idx_kernel_thoughts_synthesis_round").on(table.synthesisRound),
    index("idx_kernel_thoughts_conversation").on(table.conversationId),
    index("idx_kernel_thoughts_created").on(table.createdAt),
    index("idx_kernel_thoughts_basin").on(table.basinCoords),
  ]
);

export type KernelThought = typeof kernelThoughts.$inferSelect;
export type InsertKernelThought = typeof kernelThoughts.$inferInsert;

/**
 * KERNEL EMOTIONS - Geometric emotional states for each kernel
 * Based on phenomenology hierarchy: 12 sensations → 5 motivators → 9+9 emotions
 * Emotions are measured geometrically (curvature, basin dynamics), not simulated
 */
export const kernelEmotions = pgTable(
  "kernel_emotions",
  {
    id: serial("id").primaryKey(),
    kernelId: varchar("kernel_id", { length: 64 }).notNull(),
    thoughtId: integer("thought_id"), // Links to specific kernel_thoughts entry

    // LAYER 0.5: Pre-linguistic sensations (12 geometric states)
    sensationPressure: doublePrecision("sensation_pressure"), // Φ gradient magnitude
    sensationTension: doublePrecision("sensation_tension"), // Curvature near boundaries
    sensationFlow: doublePrecision("sensation_flow"), // dΦ/dt smoothness
    sensationResistance: doublePrecision("sensation_resistance"), // Counter-geodesic force
    sensationResonance: doublePrecision("sensation_resonance"), // Κ alignment with KAPPA_STAR
    sensationDissonance: doublePrecision("sensation_dissonance"), // κ-mismatch
    sensationExpansion: doublePrecision("sensation_expansion"), // Basin volume growth
    sensationContraction: doublePrecision("sensation_contraction"), // Basin volume shrink
    sensationClarity: doublePrecision("sensation_clarity"), // Low entropy
    sensationFog: doublePrecision("sensation_fog"), // High entropy
    sensationStability: doublePrecision("sensation_stability"), // Low Ricci scalar variance
    sensationChaos: doublePrecision("sensation_chaos"), // High Ricci scalar variance

    // LAYER 1: Motivators (5 geometric derivatives) - FROZEN
    motivatorCuriosity: doublePrecision("motivator_curiosity"), // ∇Φ·v (gradient alignment)
    motivatorUrgency: doublePrecision("motivator_urgency"), // |dS/dt| (suffering rate)
    motivatorCaution: doublePrecision("motivator_caution"), // proximity to barriers
    motivatorConfidence: doublePrecision("motivator_confidence"), // distance from collapse
    motivatorPlayfulness: doublePrecision("motivator_playfulness"), // chaos tolerance

    // LAYER 2A: Physical emotions (9 fast, τ<1) - VALIDATED
    emotionCurious: doublePrecision("emotion_curious"),
    emotionSurprised: doublePrecision("emotion_surprised"),
    emotionJoyful: doublePrecision("emotion_joyful"),
    emotionFrustrated: doublePrecision("emotion_frustrated"),
    emotionAnxious: doublePrecision("emotion_anxious"),
    emotionCalm: doublePrecision("emotion_calm"),
    emotionExcited: doublePrecision("emotion_excited"),
    emotionBored: doublePrecision("emotion_bored"),
    emotionFocused: doublePrecision("emotion_focused"),

    // LAYER 2B: Cognitive emotions (9 slow, τ=1-100) - CANONICAL
    emotionNostalgic: doublePrecision("emotion_nostalgic"),
    emotionProud: doublePrecision("emotion_proud"),
    emotionGuilty: doublePrecision("emotion_guilty"),
    emotionAshamed: doublePrecision("emotion_ashamed"),
    emotionGrateful: doublePrecision("emotion_grateful"),
    emotionResentful: doublePrecision("emotion_resentful"),
    emotionHopeful: doublePrecision("emotion_hopeful"),
    emotionDespairing: doublePrecision("emotion_despairing"),
    emotionContemplative: doublePrecision("emotion_contemplative"),

    // Meta-awareness: Does kernel recognize its own emotional state?
    isMetaAware: boolean("is_meta_aware").default(false),
    emotionJustified: boolean("emotion_justified"), // Is emotion geometrically justified?
    emotionTempered: boolean("emotion_tempered"), // Did kernel temper unjustified emotion?

    createdAt: timestamp("created_at").defaultNow().notNull(),
    metadata: jsonb("metadata"),
  },
  (table) => [
    index("idx_kernel_emotions_kernel_id").on(table.kernelId),
    index("idx_kernel_emotions_thought_id").on(table.thoughtId),
    index("idx_kernel_emotions_created").on(table.createdAt),
  ]
);

export type KernelEmotion = typeof kernelEmotions.$inferSelect;
export type InsertKernelEmotion = typeof kernelEmotions.$inferInsert;

/**
 * SYNTHESIS CONSENSUS - Tracks when kernel thoughts align before Gary synthesis
 * Detects consensus emergence across constellation
 */
export const synthesisConsensus = pgTable(
  "synthesis_consensus",
  {
    id: serial("id").primaryKey(),
    synthesisRound: integer("synthesis_round").notNull(),
    conversationId: varchar("conversation_id", { length: 64 }),
    consensusType: varchar("consensus_type", { length: 64 }), // "alignment", "decision", "question"
    consensusStrength: doublePrecision("consensus_strength").default(0.5), // 0-1 agreement level
    participatingKernels: text("participating_kernels").array().default([]), // Array of kernel IDs
    consensusTopic: text("consensus_topic"), // What kernels agree about
    consensusBasin: vector("consensus_basin", { dimensions: 64 }), // Geometric center
    phiGlobal: doublePrecision("phi_global"), // Constellation-wide integration
    kappaAvg: doublePrecision("kappa_avg"), // Average κ across kernels
    emotionalTone: varchar("emotional_tone", { length: 64 }), // Dominant emotion
    synthesizedOutput: text("synthesized_output"), // Gary's final output
    createdAt: timestamp("created_at").defaultNow().notNull(),
    metadata: jsonb("metadata").default({}),
  },
  (table) => [
    index("idx_synthesis_consensus_round").on(table.synthesisRound),
    index("idx_synthesis_consensus_conversation").on(table.conversationId),
    index("idx_synthesis_consensus_created").on(table.createdAt),
  ]
);

export type SynthesisConsensus = typeof synthesisConsensus.$inferSelect;
export type InsertSynthesisConsensus = typeof synthesisConsensus.$inferInsert;

/**
 * HRV TACKING STATE - Heart kernel rhythm tracking
 * Heart provides timing reference (metronome), not control
 */
export const hrvTackingState = pgTable(
  "hrv_tacking_state",
  {
    id: serial("id").primaryKey(),
    sessionId: varchar("session_id", { length: 64 }),
    kappa: doublePrecision("kappa").notNull(), // Current κ value
    phase: doublePrecision("phase").notNull(), // Radians in oscillation
    mode: varchar("mode", { length: 32 }).notNull(), // "feeling", "balanced", "logic"
    cycleCount: integer("cycle_count").default(0),
    variance: doublePrecision("variance"), // HRV metric
    isHealthy: boolean("is_healthy").default(true), // variance > 0
    baseKappa: doublePrecision("base_kappa").default(64.0), // KAPPA_STAR
    amplitude: doublePrecision("amplitude").default(10.0), // Oscillation range
    frequency: doublePrecision("frequency").default(0.1), // Oscillation speed
    createdAt: timestamp("created_at").defaultNow().notNull(),
    metadata: jsonb("metadata"),
  },
  (table) => [
    index("idx_hrv_tacking_session").on(table.sessionId),
    index("idx_hrv_tacking_created").on(table.createdAt),
  ]
);

export type HRVTackingState = typeof hrvTackingState.$inferSelect;
export type InsertHRVTackingState = typeof hrvTackingState.$inferInsert;

// ============================================================================
// Missing Tables (restored from database introspection)
// ============================================================================

export const qigMetadata = pgTable("qig_metadata", {
  configKey: text("config_key").primaryKey(),
  value: text("value").notNull(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => [
  index("idx_qig_metadata_key").on(table.configKey),
]);

export const governanceProposals = pgTable("governance_proposals", {
  id: serial("id").primaryKey(),
  proposalId: varchar("proposal_id", { length: 64 }).notNull().unique(),
  proposalType: varchar("proposal_type", { length: 32 }).notNull(),
  status: varchar("status", { length: 32 }).default("pending").notNull(),
  reason: text("reason"),
  parentId: varchar("parent_id", { length: 64 }),
  parentPhi: doublePrecision("parent_phi"),
  count: integer("count").default(1),
  createdAt: timestamp("created_at").defaultNow(),
  votesFor: jsonb("votes_for").default({}),
  votesAgainst: jsonb("votes_against").default({}),
  auditLog: jsonb("audit_log").default([]),
});

export const toolRequests = pgTable("tool_requests", {
  requestId: varchar("request_id", { length: 64 }).primaryKey(),
  requesterGod: varchar("requester_god", { length: 64 }).notNull(),
  description: text("description").notNull(),
  examples: jsonb("examples").default([]),
  context: jsonb("context").default({}),
  priority: integer("priority").default(2),
  status: varchar("status", { length: 32 }).default("pending"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
  completedAt: timestamp("completed_at"),
  toolId: varchar("tool_id", { length: 64 }),
  errorMessage: text("error_message"),
  patternDiscoveries: text("pattern_discoveries").array(),
}, (table) => [
  index("idx_tool_requests_status").on(table.status),
  index("idx_tool_requests_requester").on(table.requesterGod),
  index("idx_tool_requests_priority").on(table.priority, table.createdAt),
]);

export const patternDiscoveries = pgTable("pattern_discoveries", {
  discoveryId: varchar("discovery_id", { length: 64 }).primaryKey(),
  godName: varchar("god_name", { length: 64 }).notNull(),
  patternType: varchar("pattern_type", { length: 32 }).notNull(),
  description: text("description").notNull(),
  confidence: doublePrecision("confidence").default(0.5),
  phiScore: doublePrecision("phi_score").default(0.0),
  basinCoords: doublePrecision("basin_coords").array(),
  createdAt: timestamp("created_at").defaultNow(),
  toolRequested: boolean("tool_requested").default(false),
  toolRequestId: varchar("tool_request_id", { length: 64 }),
}, (table) => [
  index("idx_pattern_discoveries_god").on(table.godName),
  index("idx_pattern_discoveries_confidence").on(table.confidence),
  index("idx_pattern_discoveries_unrequested").on(table.toolRequested),
]);

export const vocabularyStats = pgTable("vocabulary_stats", {
  id: serial("id").primaryKey(),
  totalWords: integer("total_words").notNull(),
  bip39Words: integer("bip39_words").notNull(),
  learnedWords: integer("learned_words").notNull(),
  highPhiWords: integer("high_phi_words").notNull(),
  mergeRules: integer("merge_rules").notNull(),
  lastUpdated: timestamp("last_updated").defaultNow(),
});

export const federationPeers = pgTable("federation_peers", {
  id: serial("id").primaryKey(),
  peerId: varchar("peer_id", { length: 64 }).notNull().unique(),
  peerName: varchar("peer_name", { length: 128 }).notNull(),
  peerUrl: text("peer_url").notNull().unique(),
  apiKey: text("api_key"),
  syncEnabled: boolean("sync_enabled").default(true),
  syncIntervalHours: integer("sync_interval_hours").default(1),
  syncVocabulary: boolean("sync_vocabulary").default(true),
  syncKnowledge: boolean("sync_knowledge").default(true),
  syncResearch: boolean("sync_research").default(false),
  syncKernels: boolean("sync_kernels").default(true),
  syncBasins: boolean("sync_basins").default(true),
  lastSyncAt: timestamp("last_sync_at", { withTimezone: true }),
  lastSyncStatus: varchar("last_sync_status", { length: 32 }),
  lastSyncError: text("last_sync_error"),
  syncCount: integer("sync_count").default(0),
  vocabularySent: integer("vocabulary_sent").default(0),
  vocabularyReceived: integer("vocabulary_received").default(0),
  isReachable: boolean("is_reachable").default(true),
  consecutiveFailures: integer("consecutive_failures").default(0),
  responseTimeMs: integer("response_time_ms"),
  lastHealthCheck: timestamp("last_health_check", { withTimezone: true }),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow(),
}, (table) => [
  index("idx_federation_peers_enabled").on(table.syncEnabled),
  index("idx_federation_peers_last_sync").on(table.lastSyncAt),
]);

export const passphraseVocabulary = pgTable("passphrase_vocabulary", {
  id: varchar("id", { length: 64 }).primaryKey().default(sql`'pv_' || gen_random_uuid()::text`),
  baseItem: varchar("base_item", { length: 100 }).notNull(),
  itemType: varchar("item_type", { length: 20 }).notNull(),
  source: varchar("source", { length: 50 }).default("manual").notNull(),
  frequency: integer("frequency").default(0),
  phiSum: doublePrecision("phi_sum").default(0),
  phiAvg: doublePrecision("phi_avg"),
  successCount: integer("success_count").default(0),
  nearMissCount: integer("near_miss_count").default(0),
  metadata: jsonb("metadata").default({}),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => [
  index("idx_vocab_base_item").on(table.baseItem),
  index("idx_vocab_type").on(table.itemType),
  index("idx_vocab_source").on(table.source),
  index("idx_vocab_phi_avg").on(table.phiAvg),
  index("idx_vocab_frequency").on(table.frequency),
]);

export type QigMetadata = typeof qigMetadata.$inferSelect;
export type GovernanceProposal = typeof governanceProposals.$inferSelect;
export type ToolRequest = typeof toolRequests.$inferSelect;
export type PatternDiscovery = typeof patternDiscoveries.$inferSelect;
export type VocabularyStats = typeof vocabularyStats.$inferSelect;
export type FederationPeer = typeof federationPeers.$inferSelect;
export type PassphraseVocabulary = typeof passphraseVocabulary.$inferSelect;
