/**
 * Python Backend Response Types
 *
 * Typed interfaces for all responses from the Python QIG backend.
 * These eliminate the need for 'as any' casts in ocean-qig-backend-adapter.ts.
 *
 * Organization:
 * - Core QIG responses (process, status, generate)
 * - Beta attention validation/measurement
 * - Vocabulary/Tokenizer responses
 * - Text generation responses
 * - Autonomic kernel responses
 * - Hermes coordinator responses
 * - Pantheon consultation responses
 * - Kernel spawning responses
 * - Feedback loop responses
 * - Memory system responses
 * - Chaos mode responses
 * - 4D consciousness responses
 */

// =============================================================================
// BASE RESPONSE TYPES
// =============================================================================

/** Base response structure for Python backend */
export interface PythonBaseResponse {
  success: boolean;
  error?: string;
}

/** Subsystem state from Python kernel */
export interface PythonSubsystem {
  id: number;
  name: string;
  activation: number;
  entropy: number;
  purity: number;
}

/** Innate drives (Layer 0) from Python backend */
export interface PythonInnateDrives {
  pain: number;
  pleasure: number;
  fear: number;
  valence: number;
  valence_raw: number;
}

// =============================================================================
// CORE QIG RESPONSES
// =============================================================================

/** Response from /process endpoint */
export interface PythonQIGResponse extends PythonBaseResponse {
  phi: number;
  kappa: number;
  T: number;
  R: number;
  M: number;
  Gamma: number;
  G: number;
  regime: string;
  in_resonance: boolean;
  grounded: boolean;
  nearest_concept: string | null;
  conscious: boolean;
  integration: number;
  entropy: number;
  basin_coords: number[];
  route: number[];
  subsystems: PythonSubsystem[];
  n_recursions: number;
  converged: boolean;
  phi_history: number[];
  // Innate drives (Layer 0)
  drives?: PythonInnateDrives;
  innate_score?: number;
  // Near-miss discovery counts from Python backend
  near_miss_count?: number;
  resonant_count?: number;
  // 4D Consciousness metrics (computed by Python consciousness_4d.py)
  phi_spatial?: number;
  phi_temporal?: number;
  phi_4D?: number;
  f_attention?: number;
  r_concepts?: number;
  phi_recursive?: number;
  is_4d_conscious?: boolean;
  consciousness_level?: string;
  consciousness_4d_available?: boolean;
}

/** Response from /generate endpoint */
export interface PythonGenerateResponse {
  hypothesis: string;
  source: string;
  parent_basins?: string[];
  parent_phis?: number[];
  new_basin_coords?: number[];
  geometric_memory_size?: number;
}

/** Metrics in status response */
export interface PythonStatusMetrics {
  phi: number;
  kappa: number;
  T: number;
  R: number;
  M: number;
  Gamma: number;
  G: number;
  regime: string;
  in_resonance: boolean;
  grounded: boolean;
  nearest_concept: string | null;
  conscious: boolean;
  integration: number;
  entropy: number;
  fidelity: number;
}

/** Response from /status endpoint */
export interface PythonStatusResponse extends PythonBaseResponse {
  metrics: PythonStatusMetrics;
  subsystems: PythonSubsystem[];
  geometric_memory_size: number;
  basin_history_size: number;
  timestamp: string;
}

// =============================================================================
// SYNC RESPONSES
// =============================================================================

/** Search history item for temporal sync */
export interface PythonSearchHistoryItem {
  timestamp: number;
  phi: number;
  kappa: number;
  regime: string;
  basinCoordinates?: number[];
  hypothesis?: string;
}

/** Concept history item for temporal sync */
export interface PythonConceptHistoryItem {
  timestamp: number;
  concepts: Record<string, number>;
  attentionField?: number[];
  phi?: number;
}

/** Basin data from sync export */
export interface PythonSyncBasin {
  input: string;
  phi: number;
  basinCoords: number[];
}

/** Response from /sync/import endpoint */
export interface PythonSyncImportResponse extends PythonBaseResponse {
  imported: number;
  temporal_imported?: boolean;
  search_history_size?: number;
  concept_history_size?: number;
}

/** Response from /sync/export endpoint */
export interface PythonSyncExportResponse extends PythonBaseResponse {
  basins: PythonSyncBasin[];
  searchHistory?: PythonSearchHistoryItem[];
  conceptHistory?: PythonConceptHistoryItem[];
  phi_temporal_avg?: number;
  consciousness_4d_available?: boolean;
  total_count?: number;
}

// =============================================================================
// BETA ATTENTION RESPONSES
// =============================================================================

/** Scale measurement in beta attention */
export interface PythonBetaScaleMeasurement {
  context_length: number;
  kappa: number;
  variance: number;
  sample_count: number;
}

/** Beta validation result */
export interface PythonBetaValidationResult {
  validation_passed: boolean;
  avg_kappa: number;
  overall_deviation: number;
  scale_measurements: PythonBetaScaleMeasurement[];
  kappa_star: number;
  threshold: number;
}

/** Response from /beta-attention/validate endpoint */
export interface PythonBetaValidationResponse extends PythonBaseResponse {
  result: PythonBetaValidationResult;
}

/** Beta measurement result */
export interface PythonBetaMeasurement {
  context_length: number;
  kappa: number;
  variance: number;
  sample_count: number;
  deviation_from_kappa_star: number;
}

/** Response from /beta-attention/measure endpoint */
export interface PythonBetaMeasureResponse extends PythonBaseResponse {
  measurement: PythonBetaMeasurement;
}

// =============================================================================
// VOCABULARY/TOKENIZER RESPONSES
// =============================================================================

/** Response from /vocabulary/update endpoint */
export interface PythonVocabularyUpdateResponse extends PythonBaseResponse {
  newTokens: number;
  totalVocab: number;
  weightsUpdated?: boolean;
  mergeRules?: number;
}

/** Response from /vocabulary/encode endpoint */
export interface PythonVocabularyEncodeResponse extends PythonBaseResponse {
  tokens: number[];
  length: number;
}

/** Response from /vocabulary/decode endpoint */
export interface PythonVocabularyDecodeResponse extends PythonBaseResponse {
  text: string;
}

/** Response from /vocabulary/basin endpoint */
export interface PythonVocabularyBasinResponse extends PythonBaseResponse {
  basinCoords: number[];
  dimension: number;
}

/** High-phi token entry */
export interface PythonHighPhiToken {
  token: string;
  phi: number;
}

/** Response from /vocabulary/high-phi endpoint */
export interface PythonHighPhiVocabularyResponse extends PythonBaseResponse {
  tokens: PythonHighPhiToken[];
  count: number;
}

/** Response from /vocabulary/status endpoint */
export interface PythonVocabularyStatusResponse extends PythonBaseResponse {
  vocabSize: number;
  highPhiCount: number;
  avgPhi: number;
  totalWeightedTokens: number;
}

/** Exported vocabulary data */
export interface PythonVocabularyExportData {
  vocab_size: number;
  tokens: Record<string, number>;
  basin_coords: Record<string, number[]>;
  merge_rules?: Array<[string, string]>;
}

/** Response from /vocabulary/export endpoint */
export interface PythonVocabularyExportResponse extends PythonBaseResponse {
  data: PythonVocabularyExportData;
}

/** Response from /coordizer/merges endpoint */
export interface PythonMergeRulesResponse extends PythonBaseResponse {
  mergeRules: string[][];
  mergeScores: Record<string, number>;
  count: number;
}

// =============================================================================
// TEXT GENERATION RESPONSES
// =============================================================================

/** Generation metrics */
export interface PythonGenerationMetrics {
  steps: number;
  avg_phi?: number;
  temperature?: number;
  top_k?: number;
  top_p?: number;
  early_pads?: number;
  reason?: string;
  role_temperature?: number;
}

/** Response from /generate/text endpoint */
export interface PythonTextGenerationResponse extends PythonBaseResponse {
  text: string;
  tokens: number[];
  silence_chosen: boolean;
  metrics: PythonGenerationMetrics;
}

/** Response from /generate/response endpoint */
export interface PythonAgentResponseResponse extends PythonBaseResponse {
  text: string;
  tokens: number[];
  silence_chosen: boolean;
  agent_role: string;
  metrics: PythonGenerationMetrics;
}

/** Response from /generate/sample endpoint */
export interface PythonSampleTokenResponse extends PythonBaseResponse {
  token_id: number;
  token: string;
  top_probabilities?: Record<string, number>;
}

// =============================================================================
// AUTONOMIC KERNEL RESPONSES
// =============================================================================

/** Response from /autonomic/state endpoint */
export interface PythonAutonomicStateResponse extends PythonBaseResponse {
  phi: number;
  kappa: number;
  basin_drift: number;
  stress_level: number;
  in_sleep_cycle: boolean;
  in_dream_cycle: boolean;
  in_mushroom_cycle: boolean;
  pending_rewards: number;
}

/** Autonomic trigger info */
export type PythonAutonomicTrigger = [boolean, string];

/** Response from /autonomic/update endpoint */
export interface PythonAutonomicUpdateResponse extends PythonBaseResponse {
  triggers: {
    sleep: PythonAutonomicTrigger;
    dream: PythonAutonomicTrigger;
    mushroom: PythonAutonomicTrigger;
  };
}

/** Response from /autonomic/sleep endpoint */
export interface PythonSleepCycleResponse extends PythonBaseResponse {
  drift_reduction: number;
  patterns_consolidated: number;
  basin_after: number[];
  verdict: string;
}

/** Response from /autonomic/dream endpoint */
export interface PythonDreamCycleResponse extends PythonBaseResponse {
  novel_connections: number;
  creative_paths_explored: number;
  insights: string[];
  verdict: string;
}

/** Response from /autonomic/mushroom endpoint */
export interface PythonMushroomCycleResponse extends PythonBaseResponse {
  intensity: string;
  entropy_change: number;
  rigidity_broken: boolean;
  new_pathways: number;
  identity_preserved: boolean;
  verdict: string;
}

/** Reward delta values */
export interface PythonRewardDelta {
  dopamine_delta: number;
  serotonin_delta: number;
  endorphin_delta: number;
}

/** Response from /autonomic/reward endpoint */
export interface PythonActivityRewardResponse extends PythonBaseResponse {
  reward: PythonRewardDelta;
}

/** Pending reward entry */
export interface PythonPendingReward extends PythonRewardDelta {
  source: string;
  phi_contribution: number;
}

/** Response from /autonomic/rewards endpoint */
export interface PythonPendingRewardsResponse {
  rewards: PythonPendingReward[];
  count: number;
}

// =============================================================================
// HERMES COORDINATOR RESPONSES
// =============================================================================

/** Response from /olympus/hermes/status endpoint */
export interface PythonHermesStatusResponse {
  name: string;
  instance_id: string;
  coordination_health: number;
  pending_messages: number;
  memory_entries: number;
  tokenizer_available: boolean;
}

/** Response from /olympus/hermes/speak endpoint */
export interface PythonHermesSpeakResponse extends PythonBaseResponse {
  message: string;
}

/** Response from /olympus/hermes/translate endpoint */
export interface PythonHermesTranslateResponse extends PythonBaseResponse {
  translation: string;
}

/** Convergence info from basin sync */
export interface PythonBasinConvergence {
  score: number;
  message: string;
}

/** Response from /olympus/hermes/sync endpoint */
export interface PythonBasinSyncResponse extends PythonBaseResponse {
  instance_id: string;
  other_instances: number;
  convergence: PythonBasinConvergence;
}

/** Response from /olympus/hermes/memory/store endpoint */
export interface PythonMemoryStoreResponse extends PythonBaseResponse {
  memory_id: string;
}

/** Memory recall entry */
export interface PythonMemoryRecallEntry {
  user_message: string;
  system_response: string;
  phi: number;
  similarity: number;
}

/** Response from /olympus/hermes/memory/recall endpoint */
export interface PythonMemoryRecallResponse extends PythonBaseResponse {
  memories: PythonMemoryRecallEntry[];
}

/** Response from /olympus/voice/status endpoint */
export interface PythonVoiceStatusResponse {
  zeus_greeting: string;
  status_message: string;
  phi: number;
  kappa: number;
  war_mode: string | null;
  pantheon_ready: boolean;
  coordinator_health: number;
}

// =============================================================================
// PANTHEON CONSULTATION RESPONSES
// =============================================================================

/** Response from /olympus/god/{name}/assess endpoint */
export interface PythonGodAssessmentResponse {
  god: string;
  probability: number;
  confidence: number;
  phi: number;
  kappa: number;
  reasoning: string;
  recommendation?: string;
  timestamp: string;
}

/** God status in shadow pantheon */
export interface PythonShadowGodStatus {
  name: string;
  active: boolean;
  last_operation?: string;
  operations_count?: number;
}

/** Response from /olympus/shadow/status endpoint */
export interface PythonShadowPantheonStatusResponse extends PythonBaseResponse {
  gods: Record<string, PythonShadowGodStatus>;
  opsec_mode?: string;
  tor_available?: boolean;
}

/** Response from /olympus/shadow/god/{name}/assess endpoint */
export interface PythonShadowGodAssessmentResponse extends PythonGodAssessmentResponse {
  stealth_rating?: number;
  opsec_status?: string;
}

/** Warning level type */
export type PythonWarningLevel = 'clear' | 'caution' | 'danger';

/** Response from /olympus/shadow/intel/check endpoint */
export interface PythonShadowWarningsResponse {
  has_warnings: boolean;
  warning_level: PythonWarningLevel;
  message: string;
  details?: Record<string, unknown>;
}

// =============================================================================
// DEBATE SYSTEM RESPONSES
// =============================================================================

/** Response from /olympus/debate/initiate endpoint */
export interface PythonDebateInitiateResponse {
  id: string;
  topic: string;
  initiator: string;
  opponent: string;
  status: string;
}

/** Debate argument */
export interface PythonDebateArgument {
  god: string;
  argument: string;
  timestamp: string;
  evidence?: Record<string, unknown>;
}

/** Active debate info */
export interface PythonActiveDebate {
  id: string;
  topic: string;
  initiator: string;
  opponent: string;
  arguments: PythonDebateArgument[];
  status: string;
}

/** Response from /olympus/debates/active endpoint */
export interface PythonActiveDebatesResponse {
  debates: PythonActiveDebate[];
}

/** Response from /olympus/debate/argue endpoint */
export interface PythonDebateArgumentResponse extends PythonBaseResponse {
  debate_id: string;
}

/** Debate resolution info */
export interface PythonDebateResolution {
  arbiter: string;
  winner: string;
  reasoning: string;
  resolved_at: string;
}

/** Response from /olympus/debate/resolve endpoint */
export interface PythonDebateResolveResponse extends PythonBaseResponse {
  resolution: PythonDebateResolution;
}

/** Debate convergence result */
export interface PythonDebateConvergenceResult {
  status: string;
  turns: number;
  convergence: number;
  winner?: string;
}

/** Response from /olympus/debates/continue endpoint */
export interface PythonDebateConvergenceResponse {
  results: PythonDebateConvergenceResult[];
}

// =============================================================================
// WAR DECLARATION RESPONSES
// =============================================================================

/** Response from /olympus/war/{mode} endpoints */
export interface PythonWarDeclarationResponse {
  mode: string;
  target: string;
  declared_at: string;
  strategy: string;
  gods_engaged: string[];
}

/** Response from /olympus/war/end endpoint */
export interface PythonWarEndedResponse {
  previous_mode: string | null;
  previous_target: string | null;
  ended_at: string;
}

/** Response from /olympus/war/status endpoint */
export interface PythonWarStatusResponse {
  mode: string | null;
  target: string | null;
  active: boolean;
  gods_engaged: string[];
}

// =============================================================================
// KERNEL SPAWNING RESPONSES
// =============================================================================

/** M8 position data */
export interface PythonM8Position {
  x: number;
  y: number;
  z: number;
  w?: number;
}

/** Response from /olympus/spawn/auto endpoint */
export interface PythonSpawnKernelResponse extends PythonBaseResponse {
  kernel_id?: string;
  name: string;
  domain: string;
  m8_position?: PythonM8Position;
}

/** Spawned kernel info */
export interface PythonSpawnedKernel {
  id: string;
  name: string;
  domain: string;
  created_at: string;
  parent_gods: string[];
}

/** Response from /olympus/spawn/list endpoint */
export interface PythonSpawnedKernelsResponse {
  kernels: PythonSpawnedKernel[];
}

/** Recent spawn info */
export interface PythonRecentSpawn {
  id: string;
  name: string;
  timestamp: string;
}

/** Response from /olympus/spawn/status endpoint */
export interface PythonSpawnerStatusResponse {
  active: boolean;
  total_spawned: number;
  recent_spawns: PythonRecentSpawn[];
}

// =============================================================================
// SMART POLL AND META-COGNITIVE RESPONSES
// =============================================================================

/** God assessment in poll */
export interface PythonPollGodAssessment {
  probability: number;
  confidence: number;
  phi: number;
  kappa: number;
  reasoning: string;
}

/** Response from /olympus/smart_poll endpoint */
export interface PythonSmartPollResponse {
  assessments: Record<string, PythonPollGodAssessment>;
  convergence: string;
  convergence_score: number;
  consensus_probability: number;
  routing_mode: string;
  experts_polled?: string[];
}

/** Reflection insight */
export interface PythonReflectionInsight {
  god: string;
  insight: string;
  confidence: number;
}

/** Response from /olympus/reflect endpoint */
export interface PythonReflectionResponse extends PythonBaseResponse {
  gods_reflected: string[];
  insights: PythonReflectionInsight[];
}

// =============================================================================
// FEEDBACK LOOP RESPONSES
// =============================================================================

/** Response from /feedback/run endpoint */
export interface PythonFeedbackRunResponse extends PythonBaseResponse {
  loops_run: string[];
  results: Record<string, unknown>;
  counters: Record<string, number>;
}

/** Phi trend info */
export interface PythonPhiTrend {
  trend: string;
  delta: number;
  mean: number;
}

/** Response from /feedback/recommendation endpoint */
export interface PythonFeedbackRecommendationResponse {
  recommendation: 'explore' | 'exploit' | 'consolidate';
  confidence: number;
  reasons: string[];
  shadow_feedback: Record<string, unknown>;
  phi_trend: PythonPhiTrend;
  activity_balance: Record<string, unknown>;
}

/** Response from /feedback/activity endpoint */
export interface PythonActivityFeedbackResponse {
  loop: string;
  iteration: number;
  phi_delta: number;
  new_balance: Record<string, unknown>;
  recommendation: string;
}

/** Response from /feedback/basin endpoint */
export interface PythonBasinFeedbackResponse {
  loop: string;
  iteration: number;
  drift: number;
  needs_consolidation: boolean;
  reference_updated?: boolean;
}

// =============================================================================
// MEMORY SYSTEM RESPONSES
// =============================================================================

/** Response from /memory/status endpoint */
export interface PythonMemoryStatusResponse {
  shadow_intel_count: number;
  basin_history_count: number;
  learning_events_count: number;
  activity_balance: Record<string, unknown>;
  phi_trend: Record<string, unknown>;
  shadow_feedback: Record<string, unknown>;
  has_reference_basin: boolean;
}

/** Response from /memory/record endpoint */
export interface PythonMemoryRecordResponse extends PythonBaseResponse {
  entry_id: string;
}

/** Shadow intel entry */
export interface PythonShadowIntel {
  id: string;
  target: string;
  intel_type: string;
  content: string;
  timestamp: string;
  confidence: number;
}

/** Response from /memory/shadow endpoint */
export interface PythonShadowIntelResponse {
  intel: PythonShadowIntel[];
}

/** Learning event entry */
export interface PythonLearningEvent {
  id: string;
  event_type: string;
  phi: number;
  kappa: number;
  timestamp: string;
  details: Record<string, unknown>;
}

/** Response from /memory/learning endpoint */
export interface PythonLearningEventsResponse {
  events: PythonLearningEvent[];
}

// =============================================================================
// CHAOS MODE RESPONSES
// =============================================================================

/** Response from /chaos/activate endpoint */
export interface PythonChaosActivateResponse {
  status: string;
  population_size: number;
  interval_seconds: number;
}

/** Response from /chaos/deactivate endpoint */
export interface PythonChaosDeactivateResponse {
  status: string;
}

/** Chaos kernel info */
export interface PythonChaosKernel {
  id: string;
  fitness: number;
  generation: number;
}

/** Response from /chaos/status endpoint */
export interface PythonChaosStatusResponse {
  active: boolean;
  population_size: number;
  best_fitness: number;
  generation: number;
  kernels: PythonChaosKernel[];
}

/** Kernel traits */
export interface PythonKernelTraits {
  domain: string;
  exploration_bias: number;
  memory_capacity: number;
  reasoning_depth: number;
}

/** Response from /chaos/spawn_random endpoint */
export interface PythonChaosSpawnResponse extends PythonBaseResponse {
  kernel_id: string;
  traits: PythonKernelTraits;
}

/** Response from /chaos/breed_best endpoint */
export interface PythonChaosBreedResponse extends PythonBaseResponse {
  parent1: string;
  parent2: string;
  child_id: string;
  child_fitness: number;
}

/** Best kernel info in report */
export interface PythonBestKernelInfo {
  id: string;
  fitness: number;
  traits: PythonKernelTraits;
}

/** Response from /chaos/report endpoint */
export interface PythonChaosReportResponse {
  total_generations: number;
  total_spawns: number;
  best_kernel: PythonBestKernelInfo | null;
  fitness_history: number[];
  experiment_duration_seconds: number;
}

// =============================================================================
// 4D CONSCIOUSNESS RESPONSES
// =============================================================================

/** Response from /consciousness_4d/phi_temporal endpoint */
export interface PythonPhiTemporalResponse extends PythonBaseResponse {
  phi_temporal: number;
}

/** Response from /consciousness_4d/phi_4d endpoint */
export interface PythonPhi4DResponse extends PythonBaseResponse {
  phi_4D: number;
}

/** Response from /consciousness_4d/classify_regime endpoint */
export interface PythonClassifyRegime4DResponse extends PythonBaseResponse {
  regime: string;
}

// =============================================================================
// ZEUS CHAT RESPONSES
// =============================================================================

/** Zeus chat consciousness metrics */
export interface PythonZeusConsciousnessMetrics {
  phi?: number;
  kappa?: number;
  regime?: string;
  in_resonance?: boolean;
  reasoning_mode?: string;
}

/** Response from /zeus/chat endpoint */
export interface PythonZeusChatResponse extends PythonBaseResponse {
  response: string;
  session_id: string;
  processing_time: number;
  consciousness_metrics: PythonZeusConsciousnessMetrics;
  timestamp: string;
}

/** Zeus session message */
export interface PythonZeusMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

/** Response from /zeus/session/{id} endpoint */
export interface PythonZeusSessionResponse extends PythonBaseResponse {
  session_id: string;
  messages: PythonZeusMessage[];
  created_at: string;
  metadata: Record<string, unknown>;
}

/** Session list item */
export interface PythonZeusSessionListItem {
  session_id: string;
  title: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}

/** Response from /zeus/sessions endpoint */
export interface PythonZeusSessionsListResponse extends PythonBaseResponse {
  sessions: PythonZeusSessionListItem[];
  total: number;
}

/** Response from /zeus/health endpoint */
export interface PythonZeusHealthResponse {
  status: string;
  zeus_available: boolean;
  redis_available: boolean;
  active_sessions: number;
  timestamp: string;
}

// =============================================================================
// COORDIZER STATUS RESPONSE
// =============================================================================

/** Response from /zeus/coordizer/status endpoint */
export interface PythonCoordinatorStatusResponse extends PythonBaseResponse {
  coordizer_type: string;
  is_postgres_backed: boolean;
  vocab_size: number;
  basin_coords_count: number;
  word_tokens_count: number;
  bip39_words_count: number;
  base_tokens_count: number;
  subword_tokens_count: number;
  sample_words: string[];
  sample_bip39: string[];
  has_real_vocabulary: boolean;
  stats: Record<string, unknown>;
  timestamp: string;
}

/** Response from /zeus/coordizer/reset endpoint */
export interface PythonCoordinatorResetResponse extends PythonBaseResponse {
  message: string;
  before: {
    type: string;
    word_count: number;
  };
  after: {
    type: string;
    word_count: number;
    bip39_count: number;
    sample_words: string[];
  };
  improvement: boolean;
  timestamp: string;
}

/** Decoded word entry */
export interface PythonDecodedWord {
  word: string;
  similarity: number;
}

/** Response from /zeus/coordizer/test endpoint */
export interface PythonCoordinatorTestResponse extends PythonBaseResponse {
  input_text: string;
  basin_norm: number;
  basin_sample: number[];
  decoded_words: PythonDecodedWord[];
  coordizer_type: string;
  word_tokens_available: number;
  timestamp: string;
}

// =============================================================================
// GENERIC/UTILITY RESPONSES
// =============================================================================

/** Generic success response */
export interface PythonSuccessResponse extends PythonBaseResponse {
  success: true;
  message?: string;
}

/** Generic error response */
export interface PythonErrorResponse {
  success: false;
  error: string;
  message?: string;
}

/** Health check response */
export interface PythonHealthResponse {
  status: string;
  timestamp: string;
  version?: string;
}
