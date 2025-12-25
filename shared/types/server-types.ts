/**
 * Server-Side Types
 *
 * Typed interfaces for server-side constructs including:
 * - User authentication (Replit Auth)
 * - Workflow progress tracking
 * - Recovery constraints
 * - Session state extensions
 * - War tracking state
 */

// =============================================================================
// USER AUTHENTICATION TYPES (Replit Auth)
// =============================================================================

/** User claims from Replit Auth JWT */
export interface ReplitUserClaims {
  sub?: string;  // User ID
  name?: string;
  email?: string;
  picture?: string;
  iat?: number;
  exp?: number;
}

/** Authenticated user object attached to Express request (OIDC session) */
export interface AuthenticatedUser {
  claims?: ReplitUserClaims;
  access_token?: string;
  refresh_token?: string;
  expires_at?: number;
  cachedProfile?: { cachedAt: number; [key: string]: unknown };
}

/** Express Request with authenticated user */
export interface AuthenticatedRequest extends Express.Request {
  user?: AuthenticatedUser;
}

// =============================================================================
// WORKFLOW PROGRESS TYPES
// =============================================================================

/** Base progress tracking */
export interface BaseProgress {
  current: number;
  total: number;
  rate?: number;
}

/** Constrained search progress within workflow */
export interface ConstrainedSearchProgress {
  searchJobId?: string;
  constraintsIdentified: string[];
  patternsFound: number;
  estimatedSearchSpace: number;
  currentPhase: 'identifying' | 'searching' | 'validating' | 'complete';
}

/** Estate progress within workflow */
export interface EstateProgress {
  documentsAnalyzed: number;
  contactsIdentified: number;
  leadsGenerated: number;
}

/** Temporal progress within workflow */
export interface TemporalProgress {
  epochsScanned: number;
  patternsDetected: number;
  anomaliesFound: number;
}

/** Social progress within workflow */
export interface SocialProgress {
  profilesAnalyzed: number;
  connectionsFound: number;
  hintsExtracted: number;
}

/** Combined workflow progress */
export interface WorkflowProgress {
  phase: 'init' | 'analyzing' | 'searching' | 'validating' | 'complete';
  notes?: string[];
  constrainedSearchProgress?: ConstrainedSearchProgress;
  estateProgress?: EstateProgress;
  temporalProgress?: TemporalProgress;
  socialProgress?: SocialProgress;
  [key: string]: unknown;  // Allow additional properties
}

// =============================================================================
// RECOVERY CONSTRAINT TYPES
// =============================================================================

/** Normalized constraint breakdown (Task 7 schema) */
export interface RecoveryConstraints {
  // Entity linkage (normalized from linkedEntities)
  entityLinkage?: number;
  entityConfidence?: number;
  
  // Artifact density (normalized from artifactCount)
  artifactDensity?: number;
  
  // Temporal precision
  temporalPrecisionHours?: number;
  
  // Graph connectivity (normalized from graphDegree)
  graphSignature?: number;
  clusterSize?: number;
  
  // Value patterns
  hasRoundNumbers?: boolean;
  isCoinbase?: boolean;
  valuePatternStrength?: number;
  
  // Script/software fingerprint
  hasSoftwareFingerprint?: boolean;
  scriptComplexity?: number;
  
  // Allow additional properties for flexibility
  [key: string]: unknown;
}

/** Legacy constraint format (pre-Task 7) - includes old field names */
export interface LegacyRecoveryConstraints extends RecoveryConstraints {
  // Legacy field names (will be normalized to new names)
  linkedEntities?: number;
  artifactCount?: number;
  graphDegree?: number;
}

// =============================================================================
// SESSION STATE EXTENSIONS
// =============================================================================

/** Ocean agent state extension for sessions */
export interface OceanSessionState {
  identity: {
    phi: number;
    kappa: number;
    regime: string;
    basinDrift: number;
    selfModel: string;
  };
  memory: {
    episodeCount: number;
    patternCount: number;
    clusterCount: number;
  };
  ethics: {
    violations: number;
    witnessAcknowledged: boolean;
  };
  consolidation: {
    cycles: number;
    lastConsolidation: string | null;
    needsConsolidation: boolean;
  };
  computeTimeSeconds: number;
  detectedEra?: string;
}

/** Agent state summary */
export interface AgentStateSummary {
  iteration?: number;
  totalTested?: number;
  nearMissCount?: number;
  currentStrategy?: string;
  topPatterns?: string[];
  consciousness?: {
    phi: number;
    kappa: number;
    regime: string;
  };
  detectedEra?: string;
}

// =============================================================================
// WAR TRACKING TYPES
// =============================================================================

/** War declaration result with history ID */
export interface WarDeclarationResult {
  mode: string;
  target: string;
  declared_at: string;
  strategy: string;
  gods_engaged: string[];
  warHistoryId?: string;
}

/** War end result with history ID */
export interface WarEndResult {
  previous_mode: string | null;
  previous_target: string | null;
  ended_at: string;
  warHistoryId?: string;
}

/** Active war with metrics */
export interface ActiveWarWithMetrics {
  id: string;
  mode: string;
  target: string;
  status: 'active' | 'completed';
  startedAt: string;
  endedAt?: string;
  godsEngaged: string[];
  phrasesTestedDuringWar?: number;
  discoveriesDuringWar?: number;
  kernelsSpawnedDuringWar?: number;
}

// =============================================================================
// HYPOTHESIS EXTENSION TYPES
// =============================================================================

/** Extended hypothesis with address derivation info */
export interface ExtendedHypothesis {
  id: string;
  phrase: string;
  format: string;
  source: string;
  confidence: number;
  // Address derivation extensions
  addressCompressed?: string;
  addressUncompressed?: string;
  matchedFormat?: 'compressed' | 'uncompressed';
  derivationPath?: string;
  pathType?: string;
  isMnemonicDerived?: boolean;
  dormantMatch?: unknown;
  hdAddressCount?: number;
  recoveryBundle?: unknown;
  // QIG scoring
  qigScore?: {
    phi: number;
    kappa: number;
    regime: string;
    inResonance?: boolean;
  };
  evidenceChain?: Array<{
    source: string;
    type: string;
    reasoning: string;
    confidence: number;
  }>;
}

// =============================================================================
// KERNEL TYPES
// =============================================================================

/** Kernel status values */
export type KernelStatus = 'active' | 'idle' | 'breeding' | 'dormant' | 'dead' | 'shadow' | 'observing';

/** Kernel record from storage */
export interface StoredKernel {
  kernelId: string;
  godName: string;
  domain: string;
  status?: KernelStatus;
  primitiveRoot?: number;
  basinCoordinates?: number[];
  parentKernels?: string[];
  placementReason?: string;
  positionRationale?: string;
  affinityStrength?: number;
  entropyThreshold?: number;
  spawnedAt?: Date;
  lastActiveAt?: Date;
  spawnedDuringWarId?: string;
  phi?: number;
  kappa?: number;
  regime?: string;
  generation?: number;
  successCount?: number;
  failureCount?: number;
  elementGroup?: string;
  ecologicalNiche?: string;
  targetFunction?: string;
  valence?: number;
  breedingTarget?: string;
  metadata?: Record<string, unknown>;
}

// =============================================================================
// OCEAN AGENT TYPES
// =============================================================================

/** Pattern analysis results */
export interface PatternAnalysisResults {
  resonantClusters: Array<{
    centroid: number[];
    members: string[];
    avgPhi: number;
  }>;
  balanceHotspots: Array<{
    coords: number[];
    balance: number;
    addresses: string[];
  }>;
  geometricSignatures: Array<{
    pattern: string;
    frequency: number;
    avgPhi: number;
  }>;
  phraseLengthInsights: Record<number, {
    count: number;
    avgPhi: number;
    bestPhrase?: string;
  }>;
}

/** Consolidated pattern analysis */
export interface ConsolidatedAnalysis {
  consolidatedAt: string;
  totalTested: number;
  topPatterns: PatternAnalysisResults;
  learnings?: Record<string, unknown>;
}
