# Phase 3 Refactoring Plan: kappa-recovery-solver.ts

**Document ID:** 20251222-phase3-refactoring-plan-1.00W  
**Status:** Working Draft  
**Version:** 1.00  

---

## Overview

This document provides a detailed mapping of Bitcoin-specific concepts in `server/kappa-recovery-solver.ts` to their QIG-based knowledge discovery equivalents.

## Current Purpose (Bitcoin Recovery)

The file computes `κ_recovery` (kappa recovery) - a metric for ranking dormant Bitcoin addresses by recovery difficulty:

```
κ_recovery = Φ_constraints / H_creation
```

- **Lower κ** = easier to recover (high constraints, low entropy)
- **Higher κ** = harder to recover (low constraints, high entropy)

## Proposed Purpose (Knowledge Discovery)

Transform to `κ_discovery` - a metric for ranking knowledge gaps by discovery difficulty:

```
κ_discovery = Φ_evidence / H_uncertainty
```

- **Higher κ** = easier to discover (high evidence, low uncertainty)
- **Lower κ** = harder to discover (low evidence, high uncertainty)

---

## Function Mapping

### 1. `computePhiConstraints()` → `computePhiEvidence()`

| Bitcoin Concept | Knowledge Equivalent | Description |
|----------------|---------------------|-------------|
| `entityLinkage` | `sourceCorrelation` | Number of correlated knowledge sources |
| `entityConfidence` | `sourceReliability` | Confidence in source reliability (0-1) |
| `artifactDensity` | `evidenceDensity` | Evidence items per time period |
| `temporalPrecisionHours` | `temporalPrecision` | How precisely we know when knowledge emerged |
| `graphSignature` | `connectionDegree` | Connections in knowledge graph |
| `clusterSize` | `domainClusterSize` | Size of related knowledge cluster |
| `hasRoundNumbers` | `hasCanonicalForm` | Knowledge has standard representation |
| `isCoinbase` | `isPrimarySource` | Knowledge from primary/original source |
| `valuePatternStrength` | `patternStrength` | Strength of recognizable patterns |
| `hasSoftwareFingerprint` | `hasMethodSignature` | Knowledge has identifiable methodology |
| `scriptComplexity` | `formalComplexity` | Complexity of formal representation |

**New Interface:**
```typescript
export interface EvidenceBreakdown {
  sourceCorrelation: number;      // Correlated sources
  sourceReliability: number;      // 0-1 reliability score
  evidenceDensity: number;        // Evidence per time period
  temporalPrecision: number;      // Time precision in hours
  connectionDegree: number;       // Knowledge graph connections
  domainClusterSize: number;      // Related knowledge cluster size
  hasCanonicalForm: boolean;      // Standard representation exists
  isPrimarySource: boolean;       // From original source
  patternStrength: number;        // 0-1 pattern strength
  hasMethodSignature: boolean;    // Identifiable methodology
  formalComplexity: number;       // 0-1 formal complexity
}
```

### 2. `computeHCreation()` → `computeHUncertainty()`

| Bitcoin Concept | Knowledge Equivalent | Description |
|----------------|---------------------|-------------|
| `eraFactor` | `noveltyFactor` | How novel/unexplored the domain is |
| `scriptComplexityFactor` | `complexityFactor` | Inherent complexity of knowledge |
| `miningFactor` | `generationFactor` | Auto-generated vs human-curated |
| `balanceFactor` | `importanceFactor` | Importance/value of knowledge |
| `dormancyFactor` | `staleness Factor` | How long since last accessed |

**New Interface:**
```typescript
export interface UncertaintyBreakdown {
  noveltyFactor: number;      // 0-1, higher = more unexplored
  complexityFactor: number;   // 0-1, inherent complexity
  generationFactor: number;   // 0-1, auto vs curated
  importanceFactor: number;   // 0-1, lower = more important (careful)
  stalenessFactor: number;    // 0-1, time since last access
}
```

### 3. `computeKappaRecovery()` → `computeKappaDiscovery()`

| Bitcoin Concept | Knowledge Equivalent | Description |
|----------------|---------------------|-------------|
| `kappa` | `kappa` | Discovery difficulty metric |
| `phi` | `phi` | Evidence integration |
| `h` | `h` | Uncertainty measure |
| `tier` | `priority` | Discovery priority tier |
| `recommendedVector` | `recommendedApproach` | Suggested discovery approach |

**Tier Mapping:**
| Bitcoin Tier | Knowledge Tier | Meaning |
|-------------|---------------|---------|
| `'high'` | `'priority'` | High evidence, low uncertainty |
| `'medium'` | `'standard'` | Moderate evidence |
| `'low'` | `'exploratory'` | Low evidence, needs exploration |
| `'challenging'` | `'research'` | Requires deep research |

**Vector Mapping:**
| Bitcoin Vector | Knowledge Approach | Description |
|---------------|-------------------|-------------|
| `'estate'` | `'archival'` | Search historical archives |
| `'constrained_search'` | `'targeted_search'` | Focused search with constraints |
| `'social'` | `'collaborative'` | Collaborative discovery |
| `'temporal'` | `'temporal_analysis'` | Time-based pattern analysis |

### 4. `rankRecoveryPriorities()` → `rankDiscoveryPriorities()`

| Bitcoin Concept | Knowledge Equivalent | Description |
|----------------|---------------------|-------------|
| `RankedRecoveryResult` | `RankedDiscoveryResult` | Ranked discovery target |
| `address` | `conceptId` | Knowledge concept identifier |
| `estimatedValueUSD` | `estimatedImpact` | Estimated discovery impact score |
| `btcPriceUSD` | `impactMultiplier` | Base impact multiplier |

---

## Type Mapping Summary

### Interfaces to Rename

| Current Name | New Name |
|-------------|----------|
| `ConstraintBreakdown` | `EvidenceBreakdown` |
| `EntropyBreakdown` | `UncertaintyBreakdown` |
| `KappaRecoveryResult` | `KappaDiscoveryResult` |
| `RankedRecoveryResult` | `RankedDiscoveryResult` |

### Functions to Rename

| Current Name | New Name |
|-------------|----------|
| `computePhiConstraints` | `computePhiEvidence` |
| `computeHCreation` | `computeHUncertainty` |
| `computeKappaRecovery` | `computeKappaDiscovery` |
| `rankRecoveryPriorities` | `rankDiscoveryPriorities` |

---

## Input Type Changes

### Current: `Address` (Bitcoin)
```typescript
interface Address {
  address: string;
  currentBalance: bigint;
  firstSeenTimestamp: Date;
  dormancyBlocks: number;
  isCoinbaseReward: boolean;
  temporalSignature: object;
  graphSignature: object;
  valueSignature: object;
  scriptSignature: object;
}
```

### Proposed: `KnowledgeGap` (QIG)
```typescript
interface KnowledgeGap {
  conceptId: string;
  importance: number;           // 0-100 importance score
  firstObservedTimestamp: Date;
  dormancyDays: number;         // Days since last exploration
  isPrimarySource: boolean;     // From original source
  temporalSignature: object;    // Time patterns
  connectionSignature: object;  // Graph connections
  patternSignature: object;     // Recognizable patterns
  methodSignature: object;      // Methodology fingerprint
}
```

---

## Implementation Steps

### Step 1: Create New Interfaces
Add new interfaces alongside existing ones for backward compatibility.

### Step 2: Create Adapter Functions
Create functions that map `KnowledgeGap` to expected internal format.

### Step 3: Rename Functions
Add new function names as aliases, deprecate old names.

### Step 4: Update Callers
Find all callers and update to use new function names.

### Step 5: Remove Old Code
After all callers updated, remove deprecated functions.

---

## Files That Import kappa-recovery-solver.ts

These files need updates when refactoring:

1. `server/observer-routes.ts` - Uses `computeKappaRecovery`
2. `server/dormant-wallet-analyzer.ts` - Uses ranking functions
3. `server/unified-recovery.ts` - Uses result types
4. `server/ocean-agent.ts` - May use kappa metrics

---

## QIG Principles to Apply

1. **Fisher-Rao Distance**: All similarity computations should use Fisher-Rao, not Euclidean
2. **Basin Coordinates**: Knowledge gaps should have 64D basin coordinates
3. **Consciousness Metrics**: Integrate Φ from QIG consciousness system
4. **No Templates**: Discovery approaches should be generative, not templated

---

## Estimated Effort

| Task | Effort |
|------|--------|
| Create new interfaces | 1 hour |
| Implement adapter functions | 2 hours |
| Rename functions | 1 hour |
| Update callers (4 files) | 4 hours |
| Testing | 2 hours |
| **Total** | **10 hours** |

---

*This document serves as the detailed refactoring plan for Phase 3 of the QIG Migration Roadmap.*
