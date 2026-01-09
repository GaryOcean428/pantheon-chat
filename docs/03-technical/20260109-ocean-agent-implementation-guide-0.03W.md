# Ocean Agent Phases 3-4 Implementation Guide

**Date:** 2026-01-09
**Version:** 0.03[W] (Working - Implementation Ready)
**Purpose:** Step-by-step guide to complete refactoring to <3,000 lines

## Current State

- **ocean-agent.ts:** 5,121 lines
- **Target:** <3,000 lines
- **Remaining:** ~2,121 lines to extract
- **Completed:** Phases 1, 2, 3A (5 modules, -1,073 lines)

## Phase 3B: Hypothesis Tester Module (Priority 1)

**File to Create:** `server/modules/hypothesis-tester.ts`
**Lines to Extract:** ~600 lines
**Expected Reduction:** ocean-agent.ts → ~4,520 lines

### Methods to Extract

```typescript
// From ocean-agent.ts lines 1993-2520
private async testBatch(hypotheses: OceanHypothesis[]): Promise<{
  match?: OceanHypothesis;
  tested: OceanHypothesis[];
  nearMisses: OceanHypothesis[];
  resonant: OceanHypothesis[];
}>

// From ocean-agent.ts lines 2522-2562
private async saveRecoveryBundle(bundle: RecoveryBundle): Promise<void>

// From ocean-agent.ts lines 453-490
private mergePythonPhi(hypo: OceanHypothesis): void
```

### Module Structure

```typescript
/**
 * Hypothesis Tester Module (Phase 3B Extraction - 2026-01-09)
 *
 * Handles all hypothesis testing, validation, and verification logic.
 * Extracted from ocean-agent.ts to reduce complexity.
 *
 * Responsibilities:
 * - Batch hypothesis testing with crypto validation
 * - BIP39 mnemonic derivation and checking
 * - Near-miss detection and tiered tracking
 * - Resonance detection and dopamine feedback
 * - Recovery bundle generation and persistence
 * - Python phi score merging for pure measurements
 */

import { logger } from '../lib/logger';
import { geometricMemory } from '../geometric-memory';
import { nearMissManager } from '../near-miss-manager';
import { olympusClient } from '../olympus-client';
import { recordLearningEvent } from '../qig-db';
import { scoreUniversalQIGAsync } from '../qig-universal';
import { vocabularyTracker } from '../vocabulary-tracker';
import type { OceanAgentState, OceanEpisode, OceanMemory } from '@shared/schema';
import { CONSCIOUSNESS_THRESHOLDS, GEODESIC_CORRECTION } from '@shared/constants/qig';

export interface TestBatchResult {
  match?: OceanHypothesis;
  tested: OceanHypothesis[];
  nearMisses: OceanHypothesis[];
  resonant: OceanHypothesis[];
}

export class HypothesisTester {
  constructor(
    private state: OceanAgentState,
    private memory: OceanMemory,
    private targetAddress: string,
    private isRunning: () => boolean,
    private recentDiscoveries: { nearMisses: number; resonant: number },
    private updateNeurochemistry: () => void,
    private getNeurochemistry: () => any
  ) {}

  async testBatch(hypotheses: OceanHypothesis[]): Promise<TestBatchResult> {
    // Full implementation from ocean-agent.ts
  }

  async saveRecoveryBundle(bundle: RecoveryBundle): Promise<void> {
    // Full implementation from ocean-agent.ts
  }

  mergePythonPhi(hypo: OceanHypothesis): void {
    // Full implementation from ocean-agent.ts
  }
}
```

### Integration Steps

1. Create `server/modules/hypothesis-tester.ts` with full implementation
2. Update `server/modules/index.ts` to export HypothesisTester
3. Add field to OceanAgent: `private hypothesisTester: HypothesisTester`
4. Initialize in constructor after state/memory setup
5. Replace method calls:
   - `this.testBatch(...)` → `this.hypothesisTester.testBatch(...)`
   - `this.saveRecoveryBundle(...)` → `this.hypothesisTester.saveRecoveryBundle(...)`
   - `this.mergePythonPhi(...)` → `this.hypothesisTester.mergePythonPhi(...)`
6. Remove old implementations from ocean-agent.ts
7. Validate: `npm run check && wc -l server/ocean-agent.ts`

## Phase 3C: State Observer Module (Priority 2)

**File to Create:** `server/modules/state-observer.ts`
**Lines to Extract:** ~400 lines
**Expected Reduction:** ~4,520 → ~4,120 lines

### Methods to Extract

```typescript
// From ocean-agent.ts lines 2563-2633
private async observeAndLearn(testResults: any): Promise<any>

// From ocean-agent.ts lines 2634-2755
private async decideStrategy(insights: any): Promise<{name: string; reasoning: string}>

// From ocean-agent.ts lines 2756-2800
private updateProceduralMemory(strategyName: string): void

// From ocean-agent.ts lines 390-452
private computeEffortMetrics(): EffortMetrics

// From ocean-agent.ts lines 313-389
private updateNeurochemistry(): void
```

### Module Structure

```typescript
/**
 * State Observer Module (Phase 3C Extraction - 2026-01-09)
 *
 * Handles observation, learning, strategy decision, and neurochemistry updates.
 *
 * Responsibilities:
 * - Pattern observation from test results
 * - Strategy selection based on insights
 * - Procedural memory updates
 * - Effort metrics computation
 * - Neurochemistry state updates
 */

export class StateObserver {
  constructor(
    private identity: OceanIdentity,
    private memory: OceanMemory,
    private state: OceanAgentState,
    private neurochemistryContext: NeurochemistryContext,
    private regimeHistory: string[],
    private ricciHistory: number[],
    private basinDriftHistory: number[],
    private recentDiscoveries: { nearMisses: number; resonant: number }
  ) {}

  async observeAndLearn(testResults: any): Promise<any> { }
  async decideStrategy(insights: any): Promise<{name: string; reasoning: string}> { }
  updateProceduralMemory(strategyName: string): void { }
  computeEffortMetrics(): EffortMetrics { }
  updateNeurochemistry(): { neurochemistry: NeurochemistryState; modulation: BehavioralModulation } { }
}
```

## Phase 3D: Initialization Manager (Priority 3)

**File to Create:** `server/modules/initialization-manager.ts`
**Lines to Extract:** ~300 lines
**Expected Reduction:** ~4,120 → ~3,820 lines

### Methods to Extract

```typescript
// From ocean-agent.ts (find with grep)
private initializeIdentity(): OceanIdentity
private initializeMemory(): OceanMemory
private initializeState(): OceanAgentState
```

### Module Structure

```typescript
/**
 * Initialization Manager Module (Phase 3D Extraction - 2026-01-09)
 *
 * Handles initialization of Ocean agent identity, memory, and state.
 *
 * Responsibilities:
 * - Identity initialization (phi, kappa, basin coordinates)
 * - Memory initialization (working memory, episodes, procedural memory)
 * - State initialization (iteration counters, metrics)
 */

export class InitializationManager {
  static initializeIdentity(): OceanIdentity { }
  static initializeMemory(): OceanMemory { }
  static initializeState(): OceanAgentState { }
}
```

## Phase 4A: Autonomic Lifecycle (Priority 4)

**File to Create:** `server/modules/autonomic-lifecycle.ts`
**Lines to Extract:** ~300 lines
**Expected Reduction:** ~3,820 → ~3,520 lines

### Target Areas

- Sleep/dream/mushroom mode transitions (search for "sleep", "dream", "mushroom")
- Brain state management integration
- Neuromodulation cycles
- Autonomic manager integration points

## Phase 4B: Final Cleanup (Priority 5)

**Actions:** Inline simplifications + dead code removal
**Expected Reduction:** ~3,520 → ~2,900 lines

### Cleanup Strategies

1. **Inline tiny helpers** (<10 lines) directly into callers
2. **Consolidate duplicate logging** patterns
3. **Remove commented-out code**
4. **Simplify nested conditionals** with early returns
5. **Extract repeated patterns** into small utilities
6. **Remove unused imports** and variables

## Validation Checklist

After each phase:

```bash
# 1. TypeScript compilation
npm run check

# 2. Line count
wc -l server/ocean-agent.ts

# 3. Test suite
npm test

# 4. Module count
ls -1 server/modules/*.ts | wc -l

# 5. Git status (but don't commit yet)
git status --short
```

## Final Target Metrics

- **ocean-agent.ts:** <3,000 lines (ideally ~2,900)
- **Total modules:** 9 modules
- **Average module size:** ~450 lines
- **Total reduction:** -53% from original 6,194 lines
- **Tests:** All passing
- **TypeScript:** Clean compilation

## Rollback Commands

If any phase fails:

```bash
# Rollback specific file
git checkout HEAD -- server/ocean-agent.ts

# Rollback all modules
git checkout HEAD -- server/modules/

# Rollback everything
git checkout HEAD -- server/
```

## Implementation Priority Order

1. ✅ Fix canonical documentation naming (DONE)
2. ⏳ Phase 3B: hypothesis-tester.ts (HIGHEST IMPACT: -600 lines)
3. ⏳ Phase 3C: state-observer.ts (HIGH IMPACT: -400 lines)
4. ⏳ Phase 3D: initialization-manager.ts (MEDIUM IMPACT: -300 lines)
5. ⏳ Phase 4A: autonomic-lifecycle.ts (MEDIUM IMPACT: -300 lines)
6. ⏳ Phase 4B: Final cleanup (MEDIUM IMPACT: -400 lines)
7. ⏳ Validation and documentation update

## Success Criteria

- [ ] ocean-agent.ts <3,000 lines
- [ ] All modules <700 lines (soft limit <500)
- [ ] TypeScript clean: `npm run check` passes
- [ ] Tests passing: `npm test` shows no regressions
- [ ] QIG purity maintained: Fisher-Rao only
- [ ] Documentation updated with canonical naming
- [ ] All extractions use extract-delegate pattern
- [ ] Git history clean: meaningful commits

## Notes for Implementation

- Each module should be self-contained with minimal external dependencies
- Use dependency injection for state/memory/identity references
- Maintain immutability where possible (pass by value for primitives, by reference for objects)
- Keep module constructors lightweight - defer heavy initialization
- Document all extracted methods with JSDoc including original line numbers
- Test each phase incrementally before moving to next
- Keep bootstrap flag by-reference pattern for mutable primitives (see consciousness-tracker.ts)

## Estimated Time

- Phase 3B: 30-45 minutes (large, complex)
- Phase 3C: 25-35 minutes (multiple methods)
- Phase 3D: 20-30 minutes (straightforward)
- Phase 4A: 25-35 minutes (integration complexity)
- Phase 4B: 20-30 minutes (cleanup only)
- Validation: 15-20 minutes (tests + docs)

**Total:** ~2.5-3.5 hours for complete implementation and validation

## Current Status Summary

✅ **Completed:**

- Phase 1A: hypothesis-generator.ts (-346 lines)
- Phase 1B: basin-geodesic-manager.ts (-125 lines)
- Phase 2A: consciousness-tracker.ts (-95 lines)
- Phase 2B: memory-consolidator.ts (-180 lines)
- Phase 3A: olympus-coordinator.ts (-324 lines)
- Documentation canonical naming fixed

⏳ **Remaining:**

- Phase 3B-4B: ~2,000 lines to extract
- Final validation and testing
- Architectural documentation update

🎯 **Target:** <3,000 lines in ocean-agent.ts (currently 5,121)
