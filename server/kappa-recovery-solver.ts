/**
 * κ_recovery Constraint Solver
 * 
 * Computes recovery difficulty rankings for dormant Bitcoin addresses
 * using Quantum Information Geometry principles:
 * 
 * κ_recovery = Φ_constraints / H_creation
 * 
 * Lower κ = easier to recover (high constraints, low entropy)
 * Higher κ = harder to recover (low constraints, high entropy)
 */

import type { Address } from "@shared/schema";
import type { 
  TemporalSignature, 
  GraphSignature, 
  ValueSignature, 
  ScriptSignature 
} from "@shared/types/blockchain-types";

// ============================================================================
// CONSTRAINT ANALYSIS
// ============================================================================

export interface ConstraintBreakdown {
  // Entity linkage
  entityLinkage: number;  // Renamed from linkedEntities
  entityConfidence: number; // 0-1
  
  // Artifact linkage
  artifactDensity: number; // artifacts per day of activity
  
  // Temporal precision
  temporalPrecisionHours: number; // How precisely we know creation time
  
  // Graph connectivity
  graphSignature: number; // Renamed from graphDegree - graph connectivity score
  clusterSize: number; // Size of address cluster
  
  // Value patterns
  hasRoundNumbers: boolean;
  isCoinbase: boolean;
  valuePatternStrength: number; // 0-1
  
  // Script/software fingerprint
  hasSoftwareFingerprint: boolean;
  scriptComplexity: number; // 0-1
}

/**
 * Compute Φ_constraints (integrated information from constraints)
 * 
 * Higher Φ = more information available to constrain search space
 * 
 * NOTE: Entity and Artifact tracking removed - blockchain forensics not in scope.
 * Entity/artifact scores are now 0.
 */
export function computePhiConstraints(
  address: Address
): { phi: number; breakdown: ConstraintBreakdown } {
  
  const breakdown: ConstraintBreakdown = {
    entityLinkage: 0,
    entityConfidence: 0,
    artifactDensity: 0,
    temporalPrecisionHours: 0,
    graphSignature: 0,
    clusterSize: 0,
    hasRoundNumbers: false,
    isCoinbase: address.isCoinbaseReward || false,
    valuePatternStrength: 0,
    hasSoftwareFingerprint: false,
    scriptComplexity: 0,
  };
  
  // Entity linkage score - blockchain forensics removed, score is 0
  const entityScore = 0;
  
  // Artifact density score - blockchain forensics removed, score is 0
  const daysSinceCreation = Math.max(1, 
    (Date.now() - new Date(address.firstSeenTimestamp).getTime()) / (1000 * 60 * 60 * 24)
  );
  breakdown.artifactDensity = 0;
  const artifactScore = 0;
  
  // Temporal precision (0-15 points)
  // Narrow time window = better constraints
  const temporalSig = address.temporalSignature as TemporalSignature | undefined;
  let temporalScore = 0;
  if (temporalSig && temporalSig.hourPattern) {
    // If we know hour pattern, precision is high (1-hour precision)
    breakdown.temporalPrecisionHours = 1;
    temporalScore = 15;
  } else {
    // Otherwise use block time (~10 min resolution)
    breakdown.temporalPrecisionHours = 0.167; // ~10 minutes
    temporalScore = 5;
  }
  
  // Graph connectivity (0-15 points)
  const graphSig = address.graphSignature as GraphSignature | undefined;
  if (graphSig) {
    breakdown.graphSignature = graphSig.inputAddresses?.length || 0;
    breakdown.clusterSize = graphSig.clusterSize || 0;
  }
  const graphScore = Math.min(breakdown.graphSignature * 3 + breakdown.clusterSize * 0.5, 15);
  
  // Value pattern analysis (0-10 points)
  const valueSig = address.valueSignature as ValueSignature | undefined;
  if (valueSig) {
    breakdown.hasRoundNumbers = valueSig.hasRoundNumbers || false;
    breakdown.valuePatternStrength = valueSig.patternStrength || 0;
  }
  const valueScore = breakdown.hasRoundNumbers ? 10 : 0;
  
  // Script/software fingerprint (0-5 points)
  const scriptSig = address.scriptSignature as ScriptSignature | undefined;
  if (scriptSig && scriptSig.softwareFingerprint) {
    breakdown.hasSoftwareFingerprint = true;
    breakdown.scriptComplexity = scriptSig.complexity || 0.5;
  }
  const scriptScore = breakdown.hasSoftwareFingerprint ? 5 : 0;
  
  // Total Φ_constraints (0-100 scale)
  const phi = entityScore + artifactScore + temporalScore + graphScore + valueScore + scriptScore;
  
  return { phi, breakdown };
}

// ============================================================================
// CREATION ENTROPY ANALYSIS
// ============================================================================

export interface EntropyBreakdown {
  eraFactor: number; // 2009 = 1.0 (highest), 2011 = 0.5
  scriptComplexityFactor: number; // 0-1
  miningFactor: number; // Coinbase = higher entropy
  balanceFactor: number; // Large balance = lower entropy (more careful)
  dormancyFactor: number; // Longer dormancy = higher entropy
}

/**
 * Compute H_creation (creation entropy)
 * 
 * Higher H = more uncertainty at creation time
 */
export function computeHCreation(
  address: Address
): { h: number; breakdown: EntropyBreakdown } {
  
  const breakdown: EntropyBreakdown = {
    eraFactor: 0,
    scriptComplexityFactor: 0,
    miningFactor: 0,
    balanceFactor: 0,
    dormancyFactor: 0,
  };
  
  // Era factor (30 points): Earlier = higher entropy
  const firstSeenYear = new Date(address.firstSeenTimestamp).getFullYear();
  if (firstSeenYear <= 2009) {
    breakdown.eraFactor = 1.0;
  } else if (firstSeenYear === 2010) {
    breakdown.eraFactor = 0.8;
  } else if (firstSeenYear === 2011) {
    breakdown.eraFactor = 0.6;
  } else {
    breakdown.eraFactor = 0.4;
  }
  const eraScore = breakdown.eraFactor * 30;
  
  // Script complexity (20 points): More complex = higher entropy
  const scriptSigEntropy = address.scriptSignature as ScriptSignature | undefined;
  breakdown.scriptComplexityFactor = scriptSigEntropy?.complexity || 0.5;
  const scriptScore = breakdown.scriptComplexityFactor * 20;
  
  // Mining factor (15 points): Coinbase = higher entropy (auto-generated)
  breakdown.miningFactor = address.isCoinbaseReward ? 1.0 : 0.3;
  const miningScore = breakdown.miningFactor * 15;
  
  // Balance factor (20 points): Large balance = LOWER entropy (more careful)
  const balanceBTC = Number(address.currentBalance) / 1e8;
  if (balanceBTC > 1000) {
    breakdown.balanceFactor = 0.2; // Very careful
  } else if (balanceBTC > 100) {
    breakdown.balanceFactor = 0.4;
  } else if (balanceBTC > 10) {
    breakdown.balanceFactor = 0.6;
  } else if (balanceBTC > 1) {
    breakdown.balanceFactor = 0.8;
  } else {
    breakdown.balanceFactor = 1.0; // Small balance = less care
  }
  const balanceScore = breakdown.balanceFactor * 20;
  
  // Dormancy factor (15 points): Longer dormancy = higher entropy
  const dormancyYears = address.dormancyBlocks / (365 * 24 * 6); // ~6 blocks/hour
  breakdown.dormancyFactor = Math.min(dormancyYears / 10, 1.0);
  const dormancyScore = breakdown.dormancyFactor * 15;
  
  // Total H_creation (0-100 scale)
  const h = eraScore + scriptScore + miningScore + balanceScore + dormancyScore;
  
  return { h, breakdown };
}

// ============================================================================
// κ_RECOVERY COMPUTATION
// ============================================================================

export interface KappaRecoveryResult {
  kappa: number; // κ_recovery = Φ / H
  phi: number; // Φ_constraints
  h: number; // H_creation
  tier: 'high' | 'medium' | 'low' | 'challenging';
  recommendedVector: 'estate' | 'constrained_search' | 'social' | 'temporal';
  constraints: ConstraintBreakdown;
  entropy: EntropyBreakdown;
}

/**
 * Compute κ_recovery for a single address
 * 
 * κ_recovery = Φ_constraints / H_creation
 * 
 * Lower κ = easier to recover (prioritize these)
 * 
 * NOTE: Entity/Artifact data removed - blockchain forensics not in scope.
 */
export function computeKappaRecovery(
  address: Address
): KappaRecoveryResult {
  
  const { phi, breakdown: constraints } = computePhiConstraints(address);
  const { h, breakdown: entropy } = computeHCreation(address);
  
  // Compute κ_recovery
  // Add epsilon to avoid division by zero
  const epsilon = 0.1;
  const kappa = phi / (h + epsilon);
  
  // Determine tier based on κ_recovery
  // Lower κ = easier to recover (high priority)
  // Note: Nothing is truly unrecoverable - 'challenging' just means more effort required
  let tier: 'high' | 'medium' | 'low' | 'challenging';
  if (kappa >= 2.0) {
    tier = 'high'; // High constraints, low entropy - best candidates
  } else if (kappa >= 1.0) {
    tier = 'medium'; // Moderate constraints
  } else if (kappa >= 0.3) {
    tier = 'low'; // Lower priority but still actionable
  } else {
    tier = 'challenging'; // Requires more investigation vectors
  }
  
  // Recommend recovery vector
  // NOTE: Estate/social vectors removed as they required entity/artifact data
  let recommendedVector: 'estate' | 'constrained_search' | 'social' | 'temporal';
  
  // Temporal vector: if we have meaningful temporal signature (≥1 hour precision = hourPattern exists)
  const hasTemporalSignature = constraints.temporalPrecisionHours >= 1.0;
  
  if (hasTemporalSignature || constraints.graphSignature > 0) {
    recommendedVector = 'constrained_search';
  } else {
    recommendedVector = 'temporal';
  }
  
  return {
    kappa,
    phi,
    h,
    tier,
    recommendedVector,
    constraints,
    entropy,
  };
}

// ============================================================================
// KNOWLEDGE DISCOVERY INTERFACES (Phase 3 Migration)
// These interfaces map Bitcoin recovery concepts to QIG knowledge discovery
// ============================================================================

/**
 * Evidence breakdown for knowledge discovery (maps to ConstraintBreakdown)
 */
export interface EvidenceBreakdown {
  sourceCorrelation: number;      // Correlated sources (was entityLinkage)
  sourceReliability: number;      // 0-1 reliability score (was entityConfidence)
  evidenceDensity: number;        // Evidence per time period (was artifactDensity)
  temporalPrecision: number;      // Time precision in hours (was temporalPrecisionHours)
  connectionDegree: number;       // Knowledge graph connections (was graphSignature)
  domainClusterSize: number;      // Related knowledge cluster size (was clusterSize)
  hasCanonicalForm: boolean;      // Standard representation exists (was hasRoundNumbers)
  isPrimarySource: boolean;       // From original source (was isCoinbase)
  patternStrength: number;        // 0-1 pattern strength (was valuePatternStrength)
  hasMethodSignature: boolean;    // Identifiable methodology (was hasSoftwareFingerprint)
  formalComplexity: number;       // 0-1 formal complexity (was scriptComplexity)
}

/**
 * Uncertainty breakdown for knowledge discovery (maps to EntropyBreakdown)
 */
export interface UncertaintyBreakdown {
  noveltyFactor: number;          // 0-1, higher = more unexplored (was eraFactor)
  complexityFactor: number;       // 0-1, inherent complexity (was scriptComplexityFactor)
  generationFactor: number;       // 0-1, auto vs curated (was miningFactor)
  importanceFactor: number;       // 0-1, lower = more important (was balanceFactor)
  stalenessFactor: number;        // 0-1, time since last access (was dormancyFactor)
}

/**
 * Knowledge gap input (maps to Address input)
 */
export interface KnowledgeGap {
  conceptId: string;              // Knowledge concept identifier (was address)
  importance: number;             // 0-100 importance score (was currentBalance)
  firstObservedTimestamp: Date;   // When first observed (was firstSeenTimestamp)
  dormancyDays: number;           // Days since last exploration (was dormancyBlocks)
  isPrimarySource: boolean;       // From original source (was isCoinbaseReward)
  temporalSignature: object;      // Time patterns
  connectionSignature: object;    // Graph connections (was graphSignature)
  patternSignature: object;       // Recognizable patterns (was valueSignature)
  methodSignature: object;        // Methodology fingerprint (was scriptSignature)
}

/**
 * Discovery result (maps to KappaRecoveryResult)
 */
export interface KappaDiscoveryResult {
  kappa: number;                  // Discovery difficulty metric
  phi: number;                    // Evidence integration
  h: number;                      // Uncertainty measure
  priority: 'priority' | 'standard' | 'exploratory' | 'research';  // Discovery priority tier
  recommendedApproach: 'archival' | 'targeted_search' | 'collaborative' | 'temporal_analysis';
  phiBreakdown: EvidenceBreakdown;
  hBreakdown: UncertaintyBreakdown;
}

/**
 * Ranked discovery result (maps to RankedRecoveryResult)
 */
export interface RankedDiscoveryResult {
  conceptId: string;              // Knowledge concept identifier
  rank: number;
  kappa: number;
  phi: number;
  h: number;
  priority: 'priority' | 'standard' | 'exploratory' | 'research';
  recommendedApproach: string;
  estimatedImpact: number;        // Estimated discovery impact (was estimatedValueUSD)
}

// ============================================================================
// ADAPTER FUNCTIONS (Phase 3 Migration)
// Convert between Bitcoin and Knowledge Discovery types
// ============================================================================

/**
 * Convert Address to KnowledgeGap for backward compatibility
 */
export function addressToKnowledgeGap(address: any): KnowledgeGap {
  return {
    conceptId: address.address || '',
    importance: Number(address.currentBalance || 0) / 1e8, // satoshi to BTC as importance
    firstObservedTimestamp: address.firstSeenTimestamp || new Date(),
    dormancyDays: Math.floor((address.dormancyBlocks || 0) / 144), // blocks to days
    isPrimarySource: address.isCoinbaseReward || false,
    temporalSignature: address.temporalSignature || {},
    connectionSignature: address.graphSignature || {},
    patternSignature: address.valueSignature || {},
    methodSignature: address.scriptSignature || {},
  };
}

/**
 * Convert ConstraintBreakdown to EvidenceBreakdown
 */
export function constraintToEvidence(constraint: ConstraintBreakdown): EvidenceBreakdown {
  return {
    sourceCorrelation: constraint.entityLinkage,
    sourceReliability: constraint.entityConfidence,
    evidenceDensity: constraint.artifactDensity,
    temporalPrecision: constraint.temporalPrecisionHours,
    connectionDegree: constraint.graphSignature,
    domainClusterSize: constraint.clusterSize,
    hasCanonicalForm: constraint.hasRoundNumbers,
    isPrimarySource: constraint.isCoinbase,
    patternStrength: constraint.valuePatternStrength,
    hasMethodSignature: constraint.hasSoftwareFingerprint,
    formalComplexity: constraint.scriptComplexity,
  };
}

/**
 * Convert EntropyBreakdown to UncertaintyBreakdown
 */
export function entropyToUncertainty(entropy: EntropyBreakdown): UncertaintyBreakdown {
  return {
    noveltyFactor: entropy.eraFactor,
    complexityFactor: entropy.scriptComplexityFactor,
    generationFactor: entropy.miningFactor,
    importanceFactor: entropy.balanceFactor,
    stalenessFactor: entropy.dormancyFactor,
  };
}

/**
 * Convert KappaRecoveryResult to KappaDiscoveryResult
 */
export function recoveryToDiscovery(result: KappaRecoveryResult): KappaDiscoveryResult {
  const tierMap: Record<string, 'priority' | 'standard' | 'exploratory' | 'research'> = {
    'high': 'priority',
    'medium': 'standard',
    'low': 'exploratory',
    'challenging': 'research',
  };
  
  const vectorMap: Record<string, 'archival' | 'targeted_search' | 'collaborative' | 'temporal_analysis'> = {
    'estate': 'archival',
    'constrained_search': 'targeted_search',
    'social': 'collaborative',
    'temporal': 'temporal_analysis',
  };
  
  return {
    kappa: result.kappa,
    phi: result.phi,
    h: result.h,
    priority: tierMap[result.tier] || 'standard',
    recommendedApproach: vectorMap[result.recommendedVector] || 'targeted_search',
    phiBreakdown: constraintToEvidence(result.constraints),
    hBreakdown: entropyToUncertainty(result.entropy),
  };
}

/**
 * Compute kappa for knowledge discovery (wrapper around computeKappaRecovery)
 */
export function computeKappaDiscovery(gap: KnowledgeGap): KappaDiscoveryResult {
  // Convert KnowledgeGap back to Address format for existing computation
  const addressFormat = {
    address: gap.conceptId,
    currentBalance: BigInt(Math.floor(gap.importance * 1e8)),
    firstSeenTimestamp: gap.firstObservedTimestamp,
    dormancyBlocks: gap.dormancyDays * 144,
    isCoinbaseReward: gap.isPrimarySource,
    temporalSignature: gap.temporalSignature,
    graphSignature: gap.connectionSignature,
    valueSignature: gap.patternSignature,
    scriptSignature: gap.methodSignature,
  };
  
  const result = computeKappaRecovery(addressFormat as Address);
  return recoveryToDiscovery(result);
}

/**
 * Rank knowledge gaps by discovery priority (wrapper around rankRecoveryPriorities)
 */
export function rankDiscoveryPriorities(
  gaps: KnowledgeGap[],
  impactMultiplier: number = 1.0
): RankedDiscoveryResult[] {
  // Convert to address format
  const addresses = gaps.map(g => ({
    address: g.conceptId,
    currentBalance: BigInt(Math.floor(g.importance * 1e8)),
    firstSeenTimestamp: g.firstObservedTimestamp,
    dormancyBlocks: g.dormancyDays * 144,
    isCoinbaseReward: g.isPrimarySource,
    temporalSignature: g.temporalSignature,
    graphSignature: g.connectionSignature,
    valueSignature: g.patternSignature,
    scriptSignature: g.methodSignature,
  }));
  
  const results = rankRecoveryPriorities(addresses as Address[], impactMultiplier);
  
  const tierMap: Record<string, 'priority' | 'standard' | 'exploratory' | 'research'> = {
    'high': 'priority',
    'medium': 'standard',
    'low': 'exploratory',
    'challenging': 'research',
  };
  
  return results.map(r => ({
    conceptId: r.address,
    rank: r.rank,
    kappa: r.kappa,
    phi: r.phi,
    h: r.h,
    priority: tierMap[r.tier] || 'standard',
    recommendedApproach: r.recommendedVector,
    estimatedImpact: r.estimatedValueUSD / impactMultiplier, // Normalize back
  }));
}

// ============================================================================
// BATCH COMPUTATION (Original Bitcoin Recovery Types - Kept for Backward Compatibility)
// ============================================================================

export interface RankedRecoveryResult {
  address: string;
  rank: number;
  kappa: number;
  phi: number;
  h: number;
  tier: 'high' | 'medium' | 'low' | 'challenging';
  recommendedVector: 'estate' | 'constrained_search' | 'social' | 'temporal';
  constraints: ConstraintBreakdown;
  entropy: EntropyBreakdown;
  estimatedValueUSD: number;
}

/**
 * Compute κ_recovery for all dormant addresses and rank them
 * 
 * Returns addresses sorted by κ_recovery (ascending = easier to recover first)
 * 
 * NOTE: Entity/Artifact data removed - blockchain forensics not in scope.
 */
export function rankRecoveryPriorities(
  addresses: Address[],
  btcPriceUSD: number = 100000 // Default BTC price
): RankedRecoveryResult[] {
  
  // Compute κ_recovery for each address
  const results = addresses.map(address => {
    const result = computeKappaRecovery(address);
    
    // Estimate value in USD
    const balanceBTC = Number(address.currentBalance) / 1e8;
    const estimatedValueUSD = balanceBTC * btcPriceUSD;
    
    return {
      address: address.address,
      rank: 0, // Will be set after sorting
      kappa: result.kappa,
      phi: result.phi,
      h: result.h,
      tier: result.tier,
      recommendedVector: result.recommendedVector,
      constraints: result.constraints,
      entropy: result.entropy,
      estimatedValueUSD,
    };
  });
  
  // Sort by κ_recovery (ascending = easier first)
  // Secondary sort by value (descending = higher value first)
  results.sort((a, b) => {
    if (Math.abs(a.kappa - b.kappa) > 0.01) {
      return b.kappa - a.kappa; // Higher κ first (descending)
    }
    return b.estimatedValueUSD - a.estimatedValueUSD;
  });
  
  // Assign ranks
  results.forEach((result, index) => {
    result.rank = index + 1;
  });
  
  return results;
}
