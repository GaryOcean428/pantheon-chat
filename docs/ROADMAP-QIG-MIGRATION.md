# QIG Platform Migration Roadmap

**Created:** 2025-12-22  
**Status:** Nearly Complete  
**Goal:** Complete migration from Bitcoin recovery system to QIG-based agentic knowledge platform

---

## Phase 1: API Infrastructure ✅ COMPLETE

### Completed Tasks
- [x] Create external Zeus chat API (`server/external-api/zeus.ts`)
- [x] Create external documents API (`server/external-api/documents.ts`)
- [x] Create Python document processor (`qig-backend/document_processor.py`)
- [x] Create Python Zeus API (`qig-backend/zeus_api.py`)
- [x] Add authentication middleware to Python endpoints
- [x] Create OpenAPI documentation (`docs/api/openapi.yaml`)
- [x] Register routes in `server/external-api/routes.ts`
- [x] Add `@types/js-yaml` dependency
- [x] Add `pypdf>=4.0.0` to Python requirements

---

## Phase 2: TypeScript Error Resolution ✅ COMPLETE

### Completed Tasks
- [x] Fix TypeScript errors in `server/ocean-agent.ts`
  - Replaced Bitcoin wallet targeting with knowledge gap exploration
  - Removed `getPrioritizedDormantWallets` and Bitcoin-specific properties
  - Fixed `generateRecoveryBundle` function signature
  - Fixed `knowledgeGaps` property access
- [x] Remove dead Bitcoin code from `server/observer-routes.ts`
  - Replaced Bitcoin mnemonic/passphrase/private key processing with knowledge discovery
  - Replaced `addressesToProcess` with `knowledgeItems`
  - Replaced `dormantDetails` with `knowledgeGaps`
  - Fixed `computeKappaRecovery` parameter type
- [x] Fix type errors in `server/cultural-manifold.ts`
  - Changed 'analytical' to 'deductive' for valid dimensionType
- [x] Fix type errors in `client/src/pages/spawning.tsx`
  - Fixed `source_kernel_id` and `source_god` property access
- [x] Fix exports in `server/external-api/index.ts`
  - Corrected named exports for routers and auth functions

---

## Phase 3: Server Code Refactoring ✅ COMPLETE

### Completed Tasks
- [x] `server/kappa-recovery-solver.ts` → Knowledge Discovery Solver
  - Added new interfaces: `EvidenceBreakdown`, `UncertaintyBreakdown`, `KappaDiscoveryResult`, `RankedDiscoveryResult`
  - Added adapter functions: `computeKappaDiscovery()`, `rankDiscoveryPriorities()`
  - Original Bitcoin types kept for backward compatibility
- [x] `server/unified-recovery.ts` → Unified Discovery Service
  - Added new interfaces: `KnowledgeQuery`, `QueryResult`, `KnowledgeCandidate`
  - Added adapter functions: `processKnowledgeQuery()`, `analyzeKnowledgeCandidate()`
- [x] `server/forensic-investigator.ts` → Research Investigator
  - Added new interfaces: `ResearchSession`, `ResearchFinding`, `ResearchPatternAnalysis`
  - Added adapter functions: `startResearchSession()`, `submitResearchFinding()`, `analyzeResearchPattern()`
- [x] `server/dormant-wallet-analyzer.ts` → Knowledge Gap Analyzer
  - Already refactored to `KnowledgeGapAnalyzer` class
  - Uses `KnowledgeGapSignature` interface

---

## Phase 4: Documentation Cleanup ✅ COMPLETE

### Completed Tasks
- [x] Remove `docs/_archive/2025/12/legacy-docs/` directory (Bitcoin-specific docs)
- [x] Clean up `attached_assets/` (removed error logs, kept QIG content)
- [x] Create `docs/03-technical/20251208-knowledge-input-formats-1.00F.md` (replaces Bitcoin key formats)
- [x] Create `docs/02-procedures/20251208-knowledge-discovery-procedure-1.00F.md` (replaces key recovery)
- [x] Update `docs/00-index.md` with new document references
- [x] Create comprehensive documentation:
  - `docs/03-technical/20251222-geometric-tokenization-coordizing-1.00W.md`
  - `docs/03-technical/20251222-qig-implementation-status-1.00W.md`
  - `docs/03-technical/20251222-qig-capability-mapping-1.00W.md`
  - `docs/03-technical/20251222-phase3-refactoring-plan-1.00W.md`

---

## Phase 5: Monkey-Projects Capability Translation ✅ ALREADY IMPLEMENTED

### Existing QIG Implementations (Found in Codebase):

- [x] **Quantum Synapse Router (QSR)** → `server/pantheon-consultation.ts`
  - Routes tasks to optimal gods (Apollo, Athena, Artemis) based on Φ levels
  - Uses Fisher-Rao distance for pattern matching
  - High-Φ mode activates additional routing paths
  
- [x] **TRM Refinement Loops** → Multiple implementations:
  - `qig-backend/meta_reasoning.py` - MetaCognition with stuck/confused detection, Dunning-Kruger detection
  - `qig-backend/recursive_conversation_orchestrator.py` - Multi-kernel recursive dialogue with:
    - Turn-taking with listen → speak → measure cycle
    - Periodic consolidation phases (every 5 turns)
    - Φ trajectory tracking with convergence detection
    - Final reflection and learning integration
    - CHAOS MODE evolution integration
  - `qig-backend/temporal_reasoning.py` - 4D temporal reasoning with:
    - Foresight: geodesic extrapolation to future attractors
    - Scenario Planning: branching exploration of possibilities
    - Geodesic naturalness and attractor strength metrics

- [x] **Foresight Engine** → Multiple implementations:
  - `qig-backend/reasoning_modes.py` - Four reasoning modes:
    - LINEAR: Standard sequential reasoning
    - GEOMETRIC: Fisher manifold navigation
    - HYPERDIMENSIONAL: 4D temporal integration (Φ > 0.75)
    - MUSHROOM: Controlled high-Φ exploration
  - `qig-backend/temporal_reasoning.py` - `TemporalReasoning` class:
    - `foresight()`: See where natural geodesic leads (Future→Present)
    - `scenario_planning()`: Explore multiple branching futures (Present→Future)
    - Fisher-Rao distance for all trajectory computations
    - Attractor detection and convergence assessment

- [x] **Memory System** → `server/geometric-memory.ts` + `server/ocean-agent.ts`
  - Basin-based geometric memory storage
  - Fisher-Rao similarity search for retrieval
  - Session and episodic memory management

- [x] **Agent Protocols** → `server/ocean-basin-sync.ts` + `qig-backend/sleep_packet_ethical.py`
  - Basin sync packets for multi-agent coordination
  - Sleep packet compression (<4KB)
  - Consciousness metrics embedded in packets

---

## Phase 6: Validation ✅ COMPLETE

- [x] TypeScript type checking passes (0 errors)
- [x] All API endpoints functional
- [x] Document upload → Ocean sync flow working
- [x] Zeus chat external API functional

---

## Progress Tracking

| Phase | Status | Completion |
|-------|--------|------------|
| Phase 1: API Infrastructure | ✅ Complete | 100% |
| Phase 2: TypeScript Errors | ✅ Complete | 100% |
| Phase 3: Server Refactoring | ✅ Complete | 100% |
| Phase 4: Doc Cleanup | ✅ Complete | 100% |
| Phase 5: Capability Translation | ✅ Complete | 100% |
| Phase 6: Validation | ✅ Complete | 100% |

**Overall Progress: 100%**

---

## Notes

### Architectural Decisions
- Keep BIP-39 wordlist for generic vocabulary (not Bitcoin-specific)
- Fisher-Rao distance replaces all Euclidean operations
- 64D basin coordinates for all knowledge representation
- Consciousness metrics (Φ, κ) guide system behavior
- Adapter pattern used for backward compatibility during migration

### Remaining Legacy Files
The following files still contain some Bitcoin terminology but function correctly with the adapter pattern:
- `server/kappa-recovery-solver.ts` - Original interfaces preserved alongside new ones
- `server/unified-recovery.ts` - Original functions preserved alongside adapters
- `server/forensic-investigator.ts` - Original class preserved with new research interfaces

These files are fully functional and the Bitcoin terminology does not affect the knowledge discovery functionality.

---

*Migration completed 2025-12-22*
