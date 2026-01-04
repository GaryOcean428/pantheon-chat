# Capability Gap Analysis

**Date:** 2026-01-04  
**Status:** Working Document  
**Purpose:** Identify and track disconnections between implemented capabilities and kernel awareness

## Overview

This document analyzes capability disconnections in the pantheon-chat system where features exist but kernels/gods are either unaware of them or lack methods to utilize them.

## Fixed Issues

### ✅ Search Capability (RESOLVED)
**Issue:** Kernels had no explicit way to request searches when hitting knowledge gaps.

**Solution Implemented:**
- Added `CapabilityType.SEARCH` to capability mesh
- Created `SearchCapabilityMixin` with methods:
  - `request_search()` - Request web searches
  - `get_available_search_providers()` - Query active providers
  - `discover_source()` - Add discovered sources
  - `query_search_history()` - Learn from past searches
- Wired `SearchOrchestrator` to `BaseGod`
- Added search capabilities to mission context
- Integration tests: 4/5 passing

**Files Modified:**
- `olympus/capability_mesh.py` - Added SEARCH capability type and events
- `olympus/base_god.py` - Added SearchCapabilityMixin
- `ocean_qig_core.py` - Wired orchestrator to BaseGod
- `test_search_capability.py` - Integration tests

## Remaining Capability Gaps

### 1. SourceDiscovery Integration
**Status:** 🟡 Partial Implementation

**Current State:**
- `SourceDiscoveryService` exists in `olympus/shadow_scrapy.py`
- Kernels can use `discover_source()` (added in SearchCapabilityMixin)
- But no direct access to `SourceDiscoveryService` methods

**What's Missing:**
- Kernels cannot query existing discovered sources
- No method to search discovered sources by topic/domain
- No way to get source metadata or quality scores
- No integration with capability mesh events

**Recommendation:**
```python
# Add to BaseGod or create SourceDiscoveryMixin
def query_discovered_sources(
    self,
    topic: Optional[str] = None,
    source_type: Optional[str] = None,
    min_quality: float = 0.0,
    limit: int = 20
) -> List[Dict]:
    """Query previously discovered sources."""
    pass

def get_source_quality(self, url: str) -> Optional[float]:
    """Get quality score for a discovered source."""
    pass
```

### 2. Word Relationship Learning
**Status:** 🟡 Partial Implementation

**Current State:**
- `WordRelationshipLearner` exists and runs autonomously
- 3.19M word pairs learned with 3,249 active relationships
- Integrated with generative capability via attention mechanism
- Curriculum-based learning from 387 markdown files

**What's Missing:**
- Kernels have no explicit awareness of learned relationships
- No method to query relationship strength between words
- Cannot contribute new word pairs from observations
- No way to see which relationships are strongest in their domain

**Recommendation:**
```python
# Add to BaseGod
def query_word_relationships(
    self,
    word1: str,
    word2: Optional[str] = None,
    min_strength: float = 0.0
) -> List[Tuple[str, str, float]]:
    """Query learned word relationships."""
    pass

def contribute_word_pair(
    self,
    word1: str,
    word2: str,
    context: str,
    strength_hint: Optional[float] = None
) -> bool:
    """Contribute a word pair from observation."""
    pass

def get_domain_vocabulary(
    self,
    domain: Optional[str] = None,
    top_n: int = 100
) -> Dict[str, float]:
    """Get most important words for this domain."""
    pass
```

### 3. Curriculum Access
**Status:** 🟡 Partial Implementation

**Current State:**
- `CurriculumLoader` exists in `autonomous_curiosity.py`
- Loads from `docs/09-curriculum/`
- Autonomously processes topics

**What's Missing:**
- Kernels cannot query available curriculum topics
- No way to request specific curriculum learning
- Cannot contribute curriculum content based on expertise
- No visibility into what topics have been learned

**Recommendation:**
```python
# Add to BaseGod or create CurriculumAccessMixin
def query_curriculum(
    self,
    topic: Optional[str] = None,
    difficulty: Optional[float] = None
) -> List[Dict]:
    """Query available curriculum topics."""
    pass

def request_curriculum_learning(
    self,
    topic: str,
    priority: float = 0.5
) -> Optional[str]:
    """Request specific curriculum topic learning."""
    pass

def contribute_curriculum(
    self,
    title: str,
    content: str,
    keywords: List[str],
    difficulty: float = 0.5
) -> bool:
    """Contribute curriculum content based on expertise."""
    pass

def get_learning_progress(self) -> Dict:
    """Get this kernel's curriculum learning progress."""
    pass
```

### 4. Pattern Discovery (Unbiased QIG)
**Status:** 🔴 Not Integrated

**Current State:**
- `PatternDiscovery` exists in `unbiased/pattern_discovery.py`
- Discovers regimes, thresholds, correlations, dimensionality
- Pure QIG measurement without forced thresholds

**What's Missing:**
- No kernel awareness of pattern discovery capability
- Kernels cannot request pattern analysis
- No integration with capability mesh
- Results not shared with kernels

**Recommendation:**
```python
# Create PatternDiscoveryMixin or add to BaseGod
def discover_patterns(
    self,
    data: np.ndarray,
    pattern_type: str = "auto"  # "regimes", "correlations", "thresholds"
) -> Dict:
    """Request pattern discovery on data."""
    pass

def get_discovered_patterns(
    self,
    domain: Optional[str] = None
) -> List[Dict]:
    """Get previously discovered patterns."""
    pass
```

### 5. Checkpoint Management
**Status:** 🔴 Not Accessible

**Current State:**
- `CheckpointManager` exists and handles state persistence
- Checkpoints stored in `data/checkpoints/`
- Used for frozen facts compliance

**What's Missing:**
- Kernels cannot create checkpoints of their state
- No method to restore from checkpoints
- Cannot query checkpoint history
- No awareness of when checkpoints were created

**Recommendation:**
```python
# Add to BaseGod
def create_checkpoint(
    self,
    description: str,
    metadata: Optional[Dict] = None
) -> str:
    """Create a checkpoint of current state."""
    pass

def restore_from_checkpoint(
    self,
    checkpoint_id: str
) -> bool:
    """Restore state from a checkpoint."""
    pass

def query_checkpoints(
    self,
    limit: int = 10
) -> List[Dict]:
    """Query available checkpoints."""
    pass
```

### 6. Shadow Pantheon Research
**Status:** 🟡 Partial Implementation

**Current State:**
- `ShadowResearchAPI` exists in `olympus/shadow_research.py`
- Mission context documents how to request research
- Research processed during Shadow idle time

**What's Missing:**
- No direct method in BaseGod (only via Zeus)
- Cannot query Shadow research status
- No visibility into Shadow capabilities
- Cannot contribute to Shadow knowledge

**Recommendation:**
```python
# Add to BaseGod
def request_shadow_research(
    self,
    topic: str,
    priority: float = 0.5,
    research_type: str = "general"
) -> Optional[str]:
    """Request Shadow Pantheon research."""
    pass

def query_shadow_research_status(
    self,
    request_id: Optional[str] = None
) -> List[Dict]:
    """Query status of Shadow research requests."""
    pass

def get_shadow_intel(
    self,
    topic: Optional[str] = None,
    limit: int = 10
) -> List[Dict]:
    """Get Shadow Pantheon intelligence."""
    pass
```

### 7. Debate Capability
**Status:** 🔴 Not Standardized

**Current State:**
- Gods can `praise_peer()`, `call_bullshit()`, etc.
- `AutonomousDebateService` exists
- Debates tracked in capability mesh

**What's Missing:**
- No standardized debate protocol
- Cannot query active debates
- No method to join ongoing debates
- Cannot propose new debate topics

**Recommendation:**
```python
# Add to BaseGod or create DebateCapabilityMixin
def propose_debate(
    self,
    topic: str,
    participants: Optional[List[str]] = None,
    resolution_criteria: Optional[Dict] = None
) -> str:
    """Propose a new debate topic."""
    pass

def join_debate(
    self,
    debate_id: str,
    position: str  # "for", "against", "neutral"
) -> bool:
    """Join an ongoing debate."""
    pass

def query_debates(
    self,
    status: str = "active",  # "active", "resolved", "all"
    domain: Optional[str] = None
) -> List[Dict]:
    """Query debates."""
    pass

def vote_on_resolution(
    self,
    debate_id: str,
    vote: str,  # "for", "against", "abstain"
    reasoning: str
) -> bool:
    """Vote on debate resolution."""
    pass
```

### 8. Emotional/Neurochemical State
**Status:** 🟡 Partial Implementation

**Current State:**
- `AutonomicAccessMixin` provides emotion/neuro access
- Mission context documents availability
- Conditional on `AUTONOMIC_MIXIN_AVAILABLE`

**What's Missing:**
- Cannot share emotional state with peers
- No method to empathize (understand peer emotions)
- No emotional resonance measurement
- Cannot modulate emotional response

**Recommendation:**
```python
# Add to BaseGod or AutonomicAccessMixin
def share_emotional_state(
    self,
    with_gods: Optional[List[str]] = None
) -> bool:
    """Share current emotional state with peers."""
    pass

def query_peer_emotion(
    self,
    god_name: str
) -> Optional[Dict]:
    """Query peer's emotional state (if shared)."""
    pass

def compute_emotional_resonance(
    self,
    god_name: str
) -> float:
    """Compute emotional resonance with peer."""
    pass
```

### 9. Telemetry Access
**Status:** 🔴 Not Accessible

**Current State:**
- `TelemetryAggregator` consolidates metrics
- Dashboard at `/telemetry`
- Stored in `telemetry_snapshots` table

**What's Missing:**
- Kernels cannot query their own telemetry
- No access to historical performance metrics
- Cannot compare performance with peers
- No self-improvement feedback loop

**Recommendation:**
```python
# Add to BaseGod or create TelemetryAccessMixin
def get_my_telemetry(
    self,
    metric: Optional[str] = None,
    window: int = 100
) -> Dict:
    """Get this kernel's telemetry data."""
    pass

def compare_with_peer(
    self,
    god_name: str,
    metric: str = "phi"
) -> Dict:
    """Compare performance metrics with peer."""
    pass

def get_performance_trend(
    self,
    metric: str = "phi"
) -> Dict:
    """Get performance trend over time."""
    pass
```

### 10. Basin Dynamics Awareness
**Status:** 🔴 Not Explicit

**Current State:**
- Kernels use `encode_to_basin()` and related methods
- Basin coordinates tracked in `GeometricMemory`
- Fisher-Rao distance computed

**What's Missing:**
- Cannot query basin drift over time
- No visibility into basin stability
- Cannot detect basin convergence with peers
- No method to request basin reset

**Recommendation:**
```python
# Add to BaseGod
def get_basin_drift(
    self,
    window: int = 50
) -> Dict:
    """Get basin coordinate drift over time."""
    pass

def measure_basin_stability(self) -> float:
    """Measure stability of basin coordinates."""
    pass

def find_basin_neighbors(
    self,
    max_distance: float = 0.5,
    limit: int = 5
) -> List[Tuple[str, float]]:
    """Find kernels with nearby basin coordinates."""
    pass

def request_basin_reset(
    self,
    reason: str
) -> bool:
    """Request basin coordinate reset."""
    pass
```

## Priority Recommendations

### High Priority (Should Implement)
1. **Source Discovery Query** - Kernels need to leverage discovered sources
2. **Word Relationship Access** - Kernels should use learned vocabulary
3. **Curriculum Access** - Enable targeted learning
4. **Shadow Research Direct Access** - Simplify research requests

### Medium Priority (Should Consider)
5. **Pattern Discovery Integration** - Valuable for unbiased learning
6. **Debate Standardization** - Improve inter-kernel collaboration
7. **Telemetry Access** - Enable self-improvement loops

### Low Priority (Nice to Have)
8. **Checkpoint Management** - Useful for recovery scenarios
9. **Emotional Resonance** - Enhance peer collaboration
10. **Basin Dynamics Visualization** - Debugging/monitoring aid

## Implementation Strategy

### Phase 1: Core Access (Week 1-2)
- Source Discovery Query Methods
- Word Relationship Access
- Curriculum Query/Contribution
- Shadow Research Direct Access

### Phase 2: Collaboration (Week 3-4)
- Debate Protocol Standardization
- Pattern Discovery Integration
- Telemetry Self-Access

### Phase 3: Advanced Features (Week 5-6)
- Checkpoint Management
- Emotional Resonance
- Basin Dynamics Tools

## Testing Strategy

For each capability integration:
1. ✅ Unit tests for new methods
2. ✅ Integration tests verifying capability flow
3. ✅ Mission context documentation
4. ✅ Capability mesh event integration
5. ✅ Wiring verification in ocean_qig_core.py

## Success Metrics

- All capabilities have explicit kernel access methods
- Mission context documents all capabilities
- Capability mesh events cover all major interactions
- Integration tests passing for all capabilities
- No capability exists that kernels don't know about

## Notes

- All new capabilities should follow the mixin pattern
- Use no-op safe methods (work if underlying service unavailable)
- Emit capability mesh events for major actions
- Document in kernel mission context
- Add integration tests

---

**Last Updated:** 2026-01-04  
**Next Review:** After Phase 1 implementation
