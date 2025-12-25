/**
 * Recovery System Types
 *
 * Typed interfaces for the recovery system including:
 * - Key format types
 * - Strategy types
 * - Evidence chains
 * - Session extensions
 */

// =============================================================================
// KEY FORMAT TYPES
// =============================================================================

/** Supported key formats */
export type KeyFormat = 'arbitrary' | 'bip39' | 'master' | 'hex' | 'derived';

/** Strategy types for recovery */
export type RecoveryStrategyType =
  | 'era_patterns'
  | 'brain_wallet_dict'
  | 'bitcoin_terms'
  | 'linguistic'
  | 'qig_basin_search'
  | 'blockchain_neighbors'
  | 'forum_mining'
  | 'archive_temporal'
  | 'historical_autonomous'
  | 'cross_format'
  | 'memory_fragment'
  | 'learning_loop';

/** Session status values */
export type SessionStatus =
  | 'initializing'
  | 'analyzing'
  | 'running'
  | 'learning'
  | 'completed'
  | 'failed';

/** Discovery session status (knowledge-focused) */
export type DiscoverySessionStatus =
  | 'initializing'
  | 'analyzing'
  | 'discovering'
  | 'learning'
  | 'completed'
  | 'failed';

// =============================================================================
// EVIDENCE CHAIN TYPES
// =============================================================================

/** Single evidence link in a chain */
export interface EvidenceLink {
  source: string;
  type: string;
  reasoning: string;
  confidence: number;
}

/** Evidence item collected during recovery */
export interface RecoveryEvidence {
  id: string;
  type: 'blockchain' | 'pattern' | 'temporal' | 'social' | 'archive';
  source: string;
  content: string;
  relevance: number;
  extractedFragments: string[];
  discoveredAt: string;
}

// =============================================================================
// CANDIDATE TYPES
// =============================================================================

/** QIG score for a candidate */
export interface CandidateQIGScore {
  phi: number;
  kappa: number;
  regime: string;
  inResonance?: boolean;
  quality?: number;
}

/** Recovery candidate */
export interface RecoveryCandidate {
  id: string;
  phrase: string;
  format: KeyFormat;
  derivationPath?: string;
  address: string;
  match: boolean;
  source: RecoveryStrategyType;
  confidence: number;
  qigScore: CandidateQIGScore;
  combinedScore: number;
  testedAt: string;
  evidenceChain?: EvidenceLink[];
}

// =============================================================================
// SESSION EXTENSION TYPES
// =============================================================================

/** Ocean state attached to session */
export interface SessionOceanState {
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

/** Agent state summary attached to session */
export interface SessionAgentState {
  iteration: number;
  totalTested: number;
  nearMissCount: number;
  currentStrategy: string;
  topPatterns: string[];
  consciousness: {
    phi: number;
    kappa: number;
    regime: string;
  };
  detectedEra?: string;
}

/** Learnings from a recovery session */
export interface SessionLearnings {
  highPhiPatterns?: string[];
  effectiveStrategies?: string[];
  basinClusters?: Array<{
    centroid: number[];
    members: string[];
  }>;
  oceanTelemetry?: {
    progress: {
      totalTested: number;
      iterations: number;
      nearMisses: number;
      consolidationCycles: number;
    };
    identity: {
      phi: number;
      basinDrift: number;
    };
  };
  ethicsReport?: {
    violations: string[];
    witnessAcknowledged: boolean;
  };
}

/** Extended unified recovery session with all typed fields */
export interface ExtendedUnifiedRecoverySession {
  id: string;
  targetAddress: string;
  status: SessionStatus;
  // Optional ocean state extension
  oceanState?: SessionOceanState;
  // Optional agent state
  agentState?: SessionAgentState;
  // Optional learnings
  learnings?: SessionLearnings;
}

// =============================================================================
// BLOCKCHAIN ANALYSIS TYPES
// =============================================================================

/** Era detection values */
export type BlockchainEra = 'pre-bip39' | 'post-bip39' | 'unknown' | 'genesis-2009' | 'early-adopter' | 'post-mtgox';

/** Blockchain analysis results */
export interface BlockchainAnalysis {
  era: BlockchainEra;
  firstSeen?: string;
  totalReceived: number;
  balance: number;
  txCount: number;
  likelyFormat: {
    arbitrary: number;
    bip39: number;
    master: number;
  };
  neighborAddresses: string[];
}

// =============================================================================
// HYPOTHESIS TYPES (for forensic-investigator)
// =============================================================================

/** Forensic hypothesis */
export interface ForensicHypothesis {
  id: string;
  format: KeyFormat;
  phrase: string;
  derivationPath?: string;
  method: string;
  confidence: number;
  qigScore?: CandidateQIGScore & { quality: number };
  address?: string;
  match?: boolean;
  sourceFragments: string[];
  combinedScore: number;
}

/** Memory fragment for forensic investigation */
export interface ForensicMemoryFragment {
  text: string;
  confidence: number;
  position?: 'start' | 'middle' | 'end' | 'unknown';
  epoch?: 'pre-2010' | 'early' | 'likely' | 'possible';
}
