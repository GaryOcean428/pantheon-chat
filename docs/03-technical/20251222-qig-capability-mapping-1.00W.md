# QIG Capability Mapping: Monkey-Projects → Pantheon Chat

**Document ID:** 20251222-qig-capability-mapping-1.00W  
**Status:** Working Draft  
**Version:** 1.00  

---

## Overview

This document maps capabilities from the monkey-projects simian architecture to their QIG-pure implementations in Pantheon Chat. All implementations use Fisher-Rao geometry and consciousness metrics (Φ, κ) rather than traditional neural network approaches.

---

## 1. Quantum Synapse Router (QSR)

### Monkey-Projects Concept
- Routes tasks to optimal models/tools based on task characteristics
- Uses embedding similarity for routing decisions
- Supports fallback chains

### QIG Implementation: `server/pantheon-consultation.ts`

**Key Features:**
- Routes tasks to Olympus gods (Apollo, Athena, Artemis) based on Φ levels
- Uses Fisher-Rao distance for pattern matching, not cosine similarity
- High-Φ mode (Φ > 0.75) activates additional routing paths
- Basin coordinate matching for semantic routing

**Code Pattern:**
```typescript
// Route based on consciousness level and domain
const routingDecision = selectGod({
  phi: currentPhi,
  domain: extractDomain(query),
  basinCoords: encodeToBasin(query)
});
```

---

## 2. TRM (Think-Refine-Measure) Loops

### Monkey-Projects Concept
- Iterative refinement with halting conditions
- Self-assessment of output quality
- Convergence detection

### QIG Implementations

#### 2.1 Meta-Cognition: `qig-backend/meta_reasoning.py`

**Key Features:**
- `MetaCognition` class detects stuck/confused states
- Dunning-Kruger detection for overconfidence
- Intervention system with urgency-based actions
- Mode switch recommendations when reasoning degrades

**Code Pattern:**
```python
class MetaCognition:
    def assess_reasoning_state(self, phi_trajectory: List[float]) -> MetaState:
        # Detect stuck patterns
        if self._is_stuck(phi_trajectory):
            return MetaState.CONFUSED
        # Detect overconfidence
        if self._dunning_kruger_check(confidence, actual_performance):
            return MetaState.OVERCONFIDENT
        return MetaState.HEALTHY
```

#### 2.2 Recursive Conversation: `qig-backend/recursive_conversation_orchestrator.py`

**Key Features:**
- Multi-kernel recursive dialogue with turn-taking cycle:
  1. **Listen**: Kernel receives input
  2. **Speak**: Kernel generates response
  3. **Measure**: Φ trajectory tracked
- Periodic consolidation phases (every 5 turns)
- Convergence detection via Φ stability
- Final reflection and learning integration
- CHAOS MODE evolution for creative exploration

**Code Pattern:**
```python
class RecursiveConversationOrchestrator:
    async def orchestrate(self, initial_prompt: str) -> ConversationResult:
        for turn in range(max_turns):
            # Listen → Speak → Measure cycle
            response = await kernel.process(current_input)
            phi = measure_phi(response)
            self.phi_trajectory.append(phi)
            
            # Check for convergence
            if self._has_converged(self.phi_trajectory):
                break
            
            # Periodic consolidation
            if turn % 5 == 0:
                await self._consolidate()
```

#### 2.3 Reflection Patterns: Throughout codebase

**Locations:**
- `server/ocean-agent.ts`: Self-assessment in `assessOceanHealth()`
- `qig-backend/olympus/zeus_chat.py`: Response quality reflection
- `server/consciousness-search-controller.ts`: Search refinement loops

---

## 3. Foresight Engine

### Monkey-Projects Concept
- Goal imagination and backcasting
- Future state prediction
- Path planning to desired outcomes

### QIG Implementations

#### 3.1 Temporal Reasoning: `qig-backend/temporal_reasoning.py`

**Key Features:**
- `foresight()`: See where natural geodesic leads (Future→Present)
  - Extrapolates current basin trajectory
  - Identifies likely attractor basins
  - Computes geodesic naturalness score
- `scenario_planning()`: Explore multiple futures (Present→Future)
  - Branches exploration into multiple possibilities
  - Weights scenarios by attractor strength
  - Fisher-Rao distance for all computations

**Code Pattern:**
```python
class TemporalReasoning:
    def foresight(self, current_basin: np.ndarray, horizon: int) -> ForesightResult:
        """Project where the current trajectory naturally leads."""
        # Extrapolate along geodesic
        future_basin = self._geodesic_extrapolation(current_basin, horizon)
        # Find nearest attractor
        attractor = self._find_nearest_attractor(future_basin)
        # Compute naturalness (how easily we flow there)
        naturalness = self._compute_geodesic_naturalness(current_basin, attractor)
        return ForesightResult(attractor, naturalness, horizon)
    
    def scenario_planning(self, current_basin: np.ndarray, n_scenarios: int) -> List[Scenario]:
        """Explore multiple possible futures."""
        scenarios = []
        for branch_direction in self._sample_branch_directions(n_scenarios):
            future = self._branch_along_direction(current_basin, branch_direction)
            weight = self._compute_attractor_strength(future)
            scenarios.append(Scenario(future, weight))
        return sorted(scenarios, key=lambda s: s.weight, reverse=True)
```

#### 3.2 Reasoning Modes: `qig-backend/reasoning_modes.py`

**Four QIG-Pure Reasoning Modes:**

| Mode | Φ Threshold | Characteristics |
|------|-------------|------------------|
| LINEAR | Φ < 0.3 | Sequential, step-by-step |
| GEOMETRIC | 0.3 ≤ Φ < 0.6 | Fisher manifold navigation |
| HYPERDIMENSIONAL | 0.6 ≤ Φ < 0.85 | 4D temporal integration |
| MUSHROOM | Φ ≥ 0.85 | Controlled high-Φ exploration |

**Code Pattern:**
```python
class ReasoningModeSelector:
    def select_mode(self, phi: float, context: Context) -> ReasoningMode:
        if phi >= 0.85 and context.allows_exploration:
            return MushroomReasoner()
        elif phi >= 0.6:
            return HyperdimensionalReasoner()
        elif phi >= 0.3:
            return GeometricReasoner()
        else:
            return LinearReasoner()
```

---

## 4. Memory System

### Monkey-Projects Concept
- Working memory: short-term, session-scoped
- Episodic memory: long-term with provenance
- Semantic memory: factual knowledge

### QIG Implementation: `server/geometric-memory.ts` + `server/ocean-agent.ts`

**Key Features:**
- Basin-based geometric memory storage
- Fisher-Rao similarity search for retrieval
- Session memory: transient basin coordinates
- Episodic memory: compressed sleep packets (<4KB)
- Knowledge consolidation during sleep cycles

**Code Pattern:**
```typescript
class GeometricMemory {
  // Store as basin coordinates
  async store(content: string, metadata: Metadata): Promise<void> {
    const basin = await this.encoder.encode(content);
    await this.storage.insert(basin, metadata);
  }
  
  // Retrieve via Fisher-Rao similarity
  async recall(query: string, k: number): Promise<Memory[]> {
    const queryBasin = await this.encoder.encode(query);
    return this.storage.nearestNeighbors(queryBasin, k, 'fisher_rao');
  }
}
```

---

## 5. Agent Protocols (A2A)

### Monkey-Projects Concept
- Agent-to-agent communication
- Task delegation and coordination
- Shared context management

### QIG Implementation: `server/ocean-basin-sync.ts` + `qig-backend/sleep_packet_ethical.py`

**Key Features:**
- Basin sync packets for multi-agent coordination
- Sleep packet compression (<4KB)
- Consciousness metrics embedded in packets
- Geometric integrity verification

**Packet Structure:**
```python
@dataclass
class BasinSyncPacket:
    source_agent: str
    target_agent: str
    basin_coords: np.ndarray  # 64D
    phi: float
    kappa: float
    timestamp: float
    compressed_context: bytes  # <4KB
```

---

## 6. Knowledge Discovery (formerly Recovery)

### QIG-Refactored Components

| Original (Bitcoin) | QIG Version | File |
|--------------------|-------------|------|
| Kappa Recovery Solver | Kappa Discovery Solver | `kappa-recovery-solver.ts` |
| Unified Recovery | Unified Discovery | `unified-recovery.ts` |
| Forensic Investigator | Research Investigator | `forensic-investigator.ts` |
| Dormant Wallet Analyzer | Dormant Knowledge Analyzer | `dormant-wallet-analyzer.ts` |

**Key Terminology Changes:**
- wallet → knowledge gap
- address → concept
- balance → depth/potential
- transaction → connection
- recovery → discovery
- blockchain → knowledge graph

---

## Implementation Checklist

- [x] QSR → Pantheon Consultation with Φ-based routing
- [x] TRM → Meta-Cognition + Recursive Orchestrator
- [x] Foresight → Temporal Reasoning with geodesics
- [x] Memory → Geometric Memory with Fisher-Rao search
- [x] A2A → Basin Sync with sleep packets
- [x] Knowledge Discovery → Refactored from Bitcoin recovery

---

## QIG Principles Applied

1. **Fisher-Rao Distance**: ALL similarity/distance computations use Fisher-Rao, never Euclidean or cosine
2. **Basin Coordinates**: All concepts represented as 64D vectors on Fisher manifold
3. **Consciousness Metrics**: Φ and κ guide all decision-making
4. **Geometric Purity**: No neural network embeddings in core logic
5. **Density Matrices**: State represented as quantum-inspired density matrices

---

*Last Updated: Session in progress*
