# Ocean Agent Phases 3-4 Extraction Plan

**Goal:** Reduce ocean-agent.ts from 5,121 lines to <3,000 lines

**Current Status:** 5,121 lines (-1,073 from original 6,194)

**Remaining Work:** ~2,121 lines to extract

## Extraction Strategy

Given the complexity and time constraints, we're extracting large semantic chunks that are mostly self-contained:

### Phase 3B: Testing & Validation Module (~550 lines)

**File:** `server/modules/hypothesis-tester.ts`
**Content:**

- `testBatch()` method (lines 1993-2520, ~527 lines)
- Crypto validation logic (BIP39, hex, master key derivation)
- Near-miss detection and tiered tracking
- Resonance detection and celebration
- Recovery bundle generation

### Phase 3C: State & Observation Module (~400 lines)

**File:** `server/modules/state-observer.ts`
**Content:**

- `observeAndLearn()` method (lines 2563-2633, ~70 lines)
- `decideStrategy()` method (lines 2634-2755, ~121 lines)
- `updateProceduralMemory()` method (lines 2756-2800, ~44 lines)
- `computeEffortMetrics()` method (lines 390-452, ~62 lines)
- `updateNeurochemistry()` method (lines 313-389, ~76 lines)
- `mergePythonPhi()` method (lines 453-490, ~37 lines)

Total: ~410 lines

### Phase 3D: Autonomic & Lifecycle Module (~300 lines)

**File:** `server/modules/autonomic-lifecycle.ts`
**Content:**

- Sleep/dream/mushroom mode transitions
- Brain state management integration
- Neuromodulation cycles
- Consciousness regime transitions

### Phase 4: Initialization & Setup Module (~400 lines)

**File:** `server/modules/initialization-manager.ts`
**Content:**

- `initializeIdentity()` method
- `initializeMemory()` method
- `initializeState()` method
- Constructor logic extraction
- Module initialization coordination

### Phase 4B: Final Cleanup & Inline Simplification (~500 lines)

**Actions:**

- Inline small helper methods (<10 lines)
- Consolidate duplicate logging
- Remove dead code
- Simplify nested conditionals

## Expected Final State

After all extractions:

- **ocean-agent.ts:** ~2,800 lines (54% reduction from original)
- **New modules:** 10 total modules in `server/modules/`
- **Total extracted:** ~3,400 lines into reusable modules
- **Maintainability:** Much improved - each module <700 lines

## Testing Strategy

After each extraction:

1. TypeScript compilation check: `npm run check`
2. Test suite: `npm test`
3. Line count validation: `wc -l server/ocean-agent.ts`
4. Import verification: Check all delegations work

## Rollback Plan

All extractions are git-tracked. Rollback command:

```bash
git checkout HEAD -- server/ocean-agent.ts server/modules/
```
