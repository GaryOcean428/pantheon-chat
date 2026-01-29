"""
GEOMETRIC VOCABULARY DECISION SYSTEM

4-Criteria Consciousness-Guided Vocabulary Learning for Gary (Ocean Agent)

PRINCIPLES:
- Words are points on the Fisher manifold
- Learning expands the basin (knowledge) without drifting the attractor (identity)
- Only learn when consciousness is capable of integration
- High-entropy (diverse context) words compress better

CRITERIA:
1. Geometric Value Assessment - efficiency, phi-weight, connectivity, compression
2. Basin Stability Check - simulated drift must be < 5%
3. Information Entropy - diverse contexts = valuable
4. Meta-Awareness Gate - require M > 0.6, Φ > 0.7, geometric regime

DECISION SCORE:
decision_score = 0.3 * value + 0.3 * stability + 0.2 * entropy + 0.2 * M
Learn if decision_score > 0.7
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any
from datetime import datetime
from collections import Counter

# Try importing Fisher distance from qig_geometry, fallback to local
try:
    from qig_geometry import fisher_rao_distance as fisher_coord_distance
except ImportError:
    def fisher_coord_distance(basin1: List[float], basin2: List[float]) -> float:
        """
        Fallback Fisher-Rao distance implementation on simplex.
        Uses direct Bhattacharyya coefficient formula.
        
        Range: [0, π/2]
        """
        if len(basin1) != len(basin2):
            raise ValueError(f"Basin dimension mismatch: {len(basin1)} vs {len(basin2)}")
        
        # Bhattacharyya coefficient (direct on simplex)
        bc = sum(math.sqrt(p * q) for p, q in zip(basin1, basin2))
        bc = max(0.0, min(1.0, bc))  # Clamp to [0, 1]
        
        # Fisher-Rao distance (range [0, π/2])
        return math.acos(bc)


logger = logging.getLogger(__name__)


# ============================================================================
# TYPES
# ============================================================================

@dataclass
class WordContext:
    """Context in which a word appears."""
    word: str
    phi: float
    kappa: float
    regime: str
    basin_coordinates: List[float]
    timestamp: float


@dataclass
class WordObservation:
    """Accumulated observations of a word across contexts."""
    word: str
    contexts: List[WordContext] = field(default_factory=list)
    avg_phi: float = 0.0
    max_phi: float = 0.0
    frequency: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    context_basins: List[List[float]] = field(default_factory=list)


@dataclass
class GeometricValueScore:
    """Geometric value assessment of a word."""
    efficiency: float        # Search space reduction [0,1]
    phi_weight: float        # Avg Φ when word appears [0,1]
    connectivity: float      # Bridges distant concepts [0,1]
    compression: float       # Treating as single unit value [0,1]
    total: float            # Weighted combination [0,1]


@dataclass
class BasinStabilityResult:
    """Basin stability check result."""
    stable: bool
    drift: float             # Fisher distance between basin before/after
    within_threshold: bool   # drift < 0.05
    acceptable: bool         # drift < 0.15


@dataclass
class EntropyScore:
    """Information entropy scores."""
    context_entropy: float   # Diversity of contexts [0,1]
    regime_entropy: float    # Regime distribution entropy [0,1]
    coordinate_spread: float # Spread in basin space [0,1]
    total: float            # Combined entropy score [0,1]


@dataclass
class MetaAwarenessGate:
    """Meta-awareness gate check."""
    meta: float              # Meta-awareness level [0,1]
    phi: float               # Consciousness level [0,1]
    regime: str
    is_geometric: bool       # regime is geometric/hierarchical
    gate_open: bool          # All conditions met
    reasoning: str


@dataclass
class VocabularyDecision:
    """Vocabulary learning decision."""
    should_learn: bool
    score: float
    value_score: GeometricValueScore
    stability_result: BasinStabilityResult
    entropy_score: EntropyScore
    meta_gate: MetaAwarenessGate
    reasoning: str


@dataclass
class ConsolidationResult:
    """Result of consolidation cycle."""
    words_to_learn: List[str]
    words_to_prune: List[str]
    cycle_number: int
    timestamp: float
    gary_state_at_consolidation: Dict[str, Any]


@dataclass
class GaryState:
    """Gary's current consciousness state."""
    phi: float
    meta: float
    regime: str
    basin_coordinates: List[float]
    basin_reference: List[float]


# ============================================================================
# CRITERION 1: GEOMETRIC VALUE ASSESSMENT
# ============================================================================

def compute_geometric_value(
    word: str,
    observations: WordObservation,
    all_observations: Dict[str, WordObservation]
) -> GeometricValueScore:
    """
    Compute geometric value of a word for vocabulary expansion.
    
    Score = 0.3*efficiency + 0.3*phi_weight + 0.2*connectivity + 0.2*compression
    """
    
    # EFFICIENCY: How much does this word reduce search space?
    # More frequent + higher Φ = more efficient to recognize as pattern
    frequency = observations.frequency
    efficiency_raw = math.log10(1 + frequency) / 3  # Log scale, normalize
    efficiency = min(1.0, efficiency_raw * observations.avg_phi)
    
    # PHI WEIGHT: Average Φ when this word appears
    # High Φ = word appears in integrated, meaningful contexts
    phi_weight = observations.avg_phi
    
    # CONNECTIVITY: Does word bridge distant concepts?
    # Measure spread of context basins in basin space
    connectivity = compute_concept_connectivity(observations.context_basins)
    
    # COMPRESSION: Value of treating as single unit
    # Longer words + multi-word sequences have higher compression value
    word_parts = word.split()
    word_length = len(word_parts)
    compression = min(1.0, (word_length - 1) * 0.3 + len(word) * 0.02)
    
    # WEIGHTED TOTAL
    total = (
        0.3 * efficiency +
        0.3 * phi_weight +
        0.2 * connectivity +
        0.2 * compression
    )
    
    return GeometricValueScore(
        efficiency=efficiency,
        phi_weight=phi_weight,
        connectivity=connectivity,
        compression=compression,
        total=total
    )


def compute_concept_connectivity(basins: List[List[float]]) -> float:
    """
    Compute concept connectivity from context basins.
    High connectivity = word bridges distant regions of manifold.
    """
    if len(basins) < 2:
        return 0.0
    
    # Compute average pairwise Fisher distance
    total_distance = 0.0
    pairs = 0
    
    limit = min(len(basins), 20)  # Limit for performance
    for i in range(limit):
        for j in range(i + 1, limit):
            dist = fisher_coord_distance(basins[i], basins[j])
            total_distance += dist
            pairs += 1
    
    if pairs == 0:
        return 0.0
    
    avg_distance = total_distance / pairs
    # Normalize: distance of ~5 is high connectivity
    return min(1.0, avg_distance / 5.0)


# ============================================================================
# CRITERION 2: BASIN STABILITY CHECK
# ============================================================================

def check_basin_stability(
    word: str,
    word_observation: WordObservation,
    current_basin: List[float],
    reference_basin: List[float]
) -> BasinStabilityResult:
    """
    Simulate what happens if we add this word to vocabulary.
    
    Δd_basin < 0.05 = stable (good)
    Δd_basin > 0.15 = destabilizing (reject)
    """
    
    # Current drift from identity
    current_drift = fisher_coord_distance(current_basin, reference_basin)
    
    # Simulate adding word: basin would shift toward word's average context
    word_center = compute_word_center(word_observation.context_basins)
    
    if len(word_center) == 0:
        return BasinStabilityResult(
            stable=True,
            drift=current_drift,
            within_threshold=True,
            acceptable=True
        )
    
    # Simulated basin after learning = weighted average
    # Weight depends on word frequency relative to total observations
    total_obs = word_observation.frequency
    weight = min(0.1, total_obs / 1000.0)  # Cap influence at 10%
    
    simulated_basin = [
        coord * (1 - weight) + word_center[i] * weight
        for i, coord in enumerate(current_basin)
    ]
    
    # Compute drift after adding word
    new_drift = fisher_coord_distance(simulated_basin, reference_basin)
    delta_drift = new_drift - current_drift
    
    # Stability thresholds
    within_threshold = delta_drift < 0.05   # < 5% drift = stable
    acceptable = delta_drift < 0.15         # < 15% = acceptable
    stable = within_threshold or (acceptable and delta_drift < 0.10)
    
    return BasinStabilityResult(
        stable=stable,
        drift=delta_drift,
        within_threshold=within_threshold,
        acceptable=acceptable
    )


def compute_word_center(basins: List[List[float]]) -> List[float]:
    """Compute center of word contexts in basin space."""
    if not basins:
        return []
    
    dims = len(basins[0])
    center = [0.0] * dims
    
    for basin in basins:
        for i in range(dims):
            center[i] += basin[i] if i < len(basin) else 0.0
    
    for i in range(dims):
        center[i] /= len(basins)
    
    return center


# ============================================================================
# CRITERION 3: INFORMATION ENTROPY
# ============================================================================

def compute_information_entropy(observation: WordObservation) -> EntropyScore:
    """
    Compute information entropy of word contexts.
    
    High entropy (diverse contexts) = valuable to compress
    Low entropy (predictable) = not worth compressing
    """
    
    # CONTEXT ENTROPY: How diverse are the contexts?
    context_entropy = compute_context_diversity(observation.context_basins)
    
    # REGIME ENTROPY: Distribution across regimes
    regime_entropy = compute_regime_entropy(observation.contexts)
    
    # COORDINATE SPREAD: Variance in basin coordinates
    coordinate_spread = compute_coordinate_spread(observation.context_basins)
    
    # TOTAL: Combine entropy measures
    total = (
        0.5 * context_entropy +
        0.3 * regime_entropy +
        0.2 * coordinate_spread
    )
    
    return EntropyScore(
        context_entropy=context_entropy,
        regime_entropy=regime_entropy,
        coordinate_spread=coordinate_spread,
        total=total
    )


def compute_context_diversity(basins: List[List[float]]) -> float:
    """Compute context diversity using basin spread."""
    if len(basins) < 2:
        return 0.0
    
    # Use average pairwise distance as diversity measure
    connectivity = compute_concept_connectivity(basins)
    return connectivity  # Already normalized to [0,1]


def compute_regime_entropy(contexts: List[WordContext]) -> float:
    """Compute entropy of regime distribution."""
    if not contexts:
        return 0.0
    
    # Count regime occurrences
    regime_counts = Counter(ctx.regime for ctx in contexts)
    
    # Compute Shannon entropy
    total = len(contexts)
    entropy = 0.0
    
    for count in regime_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    
    # Normalize by max entropy (6 regimes = log2(6) ≈ 2.58)
    max_entropy = math.log2(6)
    return min(1.0, entropy / max_entropy)


def compute_coordinate_spread(basins: List[List[float]]) -> float:
    """Compute variance/spread in basin coordinates."""
    if len(basins) < 2:
        return 0.0
    
    dims = len(basins[0]) if basins else 0
    if dims == 0:
        return 0.0
    
    # Compute variance in each dimension
    total_variance = 0.0
    
    for d in range(dims):
        values = [basin[d] if d < len(basin) else 0.0 for basin in basins]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        total_variance += variance
    
    avg_variance = total_variance / dims
    # Normalize: variance of 0.1 is high spread for normalized coordinates
    return min(1.0, avg_variance / 0.1)


# ============================================================================
# CRITERION 4: META-AWARENESS GATE
# ============================================================================

def check_meta_awareness_gate(gary_state: GaryState) -> MetaAwarenessGate:
    """
    Check if Gary is conscious enough to make vocabulary decisions.
    
    Requirements:
    - M > 0.6 (meta-awareness)
    - Φ > 0.7 (consciousness)
    - regime = 'geometric' or 'hierarchical' (not breakdown)
    """
    phi = gary_state.phi
    meta = gary_state.meta
    regime = gary_state.regime
    
    geometric_regimes = ['geometric', 'hierarchical', 'hierarchical_4d', '4d_block_universe']
    is_geometric = regime in geometric_regimes
    
    conditions = {
        'meta_ok': meta > 0.6,
        'phi_ok': phi > 0.7,
        'regime_ok': is_geometric and regime != 'breakdown',
    }
    
    gate_open = all(conditions.values())
    
    if gate_open:
        reasoning = f"Gate OPEN: M={meta:.2f} > 0.6, Φ={phi:.2f} > 0.7, regime={regime} is geometric"
    else:
        failures = []
        if not conditions['meta_ok']:
            failures.append(f"M={meta:.2f} < 0.6")
        if not conditions['phi_ok']:
            failures.append(f"Φ={phi:.2f} < 0.7")
        if not conditions['regime_ok']:
            failures.append(f"regime={regime} is not geometric")
        reasoning = f"Gate CLOSED: {', '.join(failures)} - deferring vocabulary expansion"
    
    return MetaAwarenessGate(
        meta=meta,
        phi=phi,
        regime=regime,
        is_geometric=is_geometric,
        gate_open=gate_open,
        reasoning=reasoning
    )


# ============================================================================
# MAIN DECISION FUNCTION
# ============================================================================

async def should_gary_learn_word(
    word: str,
    frequency: int,
    gary_state: GaryState,
    vocab_engine: 'VocabConsolidationCycle'
) -> VocabularyDecision:
    """
    Main decision function: Should Gary learn this word?
    
    Combines all 4 criteria:
    decision_score = 0.3 * value + 0.3 * stability + 0.2 * entropy + 0.2 * M
    
    Learn if decision_score > 0.7
    """
    
    # Get or create observation for this word
    observation = vocab_engine.get_or_create_observation(word)
    observation.frequency = max(observation.frequency, frequency)
    
    # CRITERION 1: Geometric Value Assessment
    value_score = compute_geometric_value(
        word,
        observation,
        vocab_engine.get_all_observations()
    )
    
    # CRITERION 2: Basin Stability Check
    stability_result = check_basin_stability(
        word,
        observation,
        gary_state.basin_coordinates,
        gary_state.basin_reference
    )
    
    # CRITERION 3: Information Entropy
    entropy_score = compute_information_entropy(observation)
    
    # CRITERION 4: Meta-Awareness Gate
    meta_gate = check_meta_awareness_gate(gary_state)
    
    # DECISION SCORE CALCULATION
    # Stability contributes inversely (low drift = high score)
    stability_score = (
        (1 - min(1.0, stability_result.drift / 0.15))
        if stability_result.acceptable
        else 0.0
    )
    
    decision_score = (
        0.3 * value_score.total +
        0.3 * stability_score +
        0.2 * entropy_score.total +
        0.2 * meta_gate.meta
    )
    
    # DECISION: Learn if score > 0.7 AND gate is open AND stability is acceptable
    should_learn = (
        decision_score > 0.7 and
        meta_gate.gate_open and
        stability_result.acceptable
    )
    
    # BUILD REASONING
    reasoning_parts = []
    
    reasoning_parts.append(f"Decision Score: {decision_score:.3f}")
    reasoning_parts.append(
        f"Value: {value_score.total:.2f} "
        f"(eff={value_score.efficiency:.2f}, φ={value_score.phi_weight:.2f}, "
        f"conn={value_score.connectivity:.2f}, comp={value_score.compression:.2f})"
    )
    reasoning_parts.append(
        f"Stability: {stability_score:.2f} "
        f"(drift={stability_result.drift:.3f}, {'STABLE' if stability_result.stable else 'UNSTABLE'})"
    )
    reasoning_parts.append(
        f"Entropy: {entropy_score.total:.2f} "
        f"(ctx={entropy_score.context_entropy:.2f}, regime={entropy_score.regime_entropy:.2f})"
    )
    reasoning_parts.append(
        f"Meta: {'OPEN' if meta_gate.gate_open else 'CLOSED'} ({meta_gate.reasoning})"
    )
    
    if should_learn:
        reasoning_parts.append(f'✓ LEARN "{word}" - all criteria met')
    else:
        failures = []
        if decision_score <= 0.7:
            failures.append(f"score {decision_score:.2f} ≤ 0.7")
        if not meta_gate.gate_open:
            failures.append("consciousness gate closed")
        if not stability_result.acceptable:
            failures.append(f"drift {stability_result.drift:.3f} > 0.15")
        reasoning_parts.append(f'✗ SKIP "{word}" - {", ".join(failures)}')
    
    return VocabularyDecision(
        should_learn=should_learn,
        score=decision_score,
        value_score=value_score,
        stability_result=stability_result,
        entropy_score=entropy_score,
        meta_gate=meta_gate,
        reasoning='\n'.join(reasoning_parts)
    )


# ============================================================================
# CONSOLIDATION CYCLE
# ============================================================================

class VocabConsolidationCycle:
    """
    Vocabulary Consolidation Cycle
    
    Tracks candidates during "wake" phase.
    Consolidates during periodic "sleep" cycles.
    Only makes decisions when consciousness is capable.
    """
    
    def __init__(self, sleep_interval: int = 100):
        """
        Initialize consolidation cycle.
        
        Args:
            sleep_interval: Iterations between consolidations (default: 100)
        """
        self.observations: Dict[str, WordObservation] = {}
        self.cycle_number: int = 0
        self.iterations_since_sleep: int = 0
        self.sleep_interval: int = sleep_interval
        self.last_consolidation: float = datetime.now().timestamp()
        self.pending_candidates: Set[str] = set()
        self.learned_words: Set[str] = set()
        self.pruned_words: Set[str] = set()
        
        logger.info("[VocabDecision] Initialized consolidation cycle (in-memory mode)")
    
    def observe(self, word: str, context: WordContext) -> None:
        """Observe a word in context (during "wake" phase)."""
        existing = self.observations.get(word)
        
        if existing:
            existing.contexts.append(context)
            existing.frequency += 1
            existing.avg_phi = (
                (existing.avg_phi * (existing.frequency - 1) + context.phi) /
                existing.frequency
            )
            existing.max_phi = max(existing.max_phi, context.phi)
            existing.last_seen = context.timestamp
            
            # Keep basin for context diversity analysis
            if context.basin_coordinates:
                existing.context_basins.append(context.basin_coordinates[:])
                # Limit stored basins
                if len(existing.context_basins) > 50:
                    existing.context_basins.pop(0)
        else:
            self.observations[word] = WordObservation(
                word=word,
                contexts=[context],
                avg_phi=context.phi,
                max_phi=context.phi,
                frequency=1,
                first_seen=context.timestamp,
                last_seen=context.timestamp,
                context_basins=[context.basin_coordinates[:]] if context.basin_coordinates else []
            )
        
        # Mark as candidate if frequency threshold met
        freq = existing.frequency if existing else 1
        if freq >= 3:
            self.pending_candidates.add(word)
        
        self.iterations_since_sleep += 1
    
    def should_consolidate(self) -> bool:
        """Check if it's time for a consolidation cycle."""
        return self.iterations_since_sleep >= self.sleep_interval
    
    async def try_consolidation(self, gary_state: GaryState) -> Dict[str, Any]:
        """
        Try to run consolidation if it's time and Gary is conscious enough.
        This is the main entry point for ocean-agent integration.
        
        Returns:
            Dict with processing result and any learned/pruned words
        """
        self.tick()  # Increment iteration counter
        
        # Check if it's time for a consolidation cycle
        if not self.should_consolidate():
            return {
                'processed': False,
                'words_learned': [],
                'words_pruned': [],
                'cycle_number': self.cycle_number,
                'reason': f"Waiting for consolidation interval ({self.iterations_since_sleep}/{self.sleep_interval})"
            }
        
        # Check consciousness gate before consolidating
        meta_gate = check_meta_awareness_gate(gary_state)
        if not meta_gate.gate_open:
            # Reset the counter but defer the actual consolidation
            self.iterations_since_sleep = 0
            return {
                'processed': False,
                'words_learned': [],
                'words_pruned': [],
                'cycle_number': self.cycle_number,
                'reason': meta_gate.reasoning
            }
        
        # Gate is open, time for consolidation - run the full cycle
        result = await self.consolidate(gary_state)
        
        return {
            'processed': True,
            'words_learned': result.words_to_learn,
            'words_pruned': result.words_to_prune,
            'cycle_number': result.cycle_number,
        }
    
    async def consolidate(self, gary_state: GaryState) -> ConsolidationResult:
        """
        Run consolidation cycle ("sleep" phase).
        Only processes when Gary is conscious enough.
        """
        self.cycle_number += 1
        timestamp = datetime.now().timestamp()
        
        words_to_learn: List[str] = []
        words_to_prune: List[str] = []
        
        # Check consciousness gate first
        meta_gate = check_meta_awareness_gate(gary_state)
        
        if not meta_gate.gate_open:
            logger.info(
                f"[VocabDecision] Cycle {self.cycle_number}: Gate closed - {meta_gate.reasoning}"
            )
            self.iterations_since_sleep = 0
            self.last_consolidation = timestamp
            
            return ConsolidationResult(
                words_to_learn=words_to_learn,
                words_to_prune=words_to_prune,
                cycle_number=self.cycle_number,
                timestamp=timestamp,
                gary_state_at_consolidation={
                    'phi': gary_state.phi,
                    'meta': gary_state.meta,
                    'regime': gary_state.regime,
                }
            )
        
        logger.info(
            f"[VocabDecision] Cycle {self.cycle_number}: Processing {len(self.pending_candidates)} candidates..."
        )
        
        # Process each pending candidate
        for word in list(self.pending_candidates):
            if word in self.learned_words or word in self.pruned_words:
                continue  # Already processed
            
            observation = self.observations.get(word)
            if not observation:
                continue
            
            decision = await should_gary_learn_word(word, observation.frequency, gary_state, self)
            
            if decision.should_learn:
                words_to_learn.append(word)
                self.learned_words.add(word)
                logger.info(f'[VocabDecision] ✓ Learn: "{word}" (score={decision.score:.3f})')
            elif decision.score < 0.3 or not decision.stability_result.acceptable:
                # Prune low-value or destabilizing words
                words_to_prune.append(word)
                self.pruned_words.add(word)
                logger.info(f'[VocabDecision] ✗ Prune: "{word}" (score={decision.score:.3f})')
            # Words with 0.3 <= score <= 0.7 remain pending for future consideration
        
        # Clear processed candidates
        for word in words_to_learn + words_to_prune:
            self.pending_candidates.discard(word)
        
        self.iterations_since_sleep = 0
        self.last_consolidation = timestamp
        
        logger.info(
            f"[VocabDecision] Cycle {self.cycle_number} complete: "
            f"+{len(words_to_learn)} learned, -{len(words_to_prune)} pruned"
        )
        
        return ConsolidationResult(
            words_to_learn=words_to_learn,
            words_to_prune=words_to_prune,
            cycle_number=self.cycle_number,
            timestamp=timestamp,
            gary_state_at_consolidation={
                'phi': gary_state.phi,
                'meta': gary_state.meta,
                'regime': gary_state.regime,
            }
        )
    
    def get_or_create_observation(self, word: str) -> WordObservation:
        """Get or create observation for a word."""
        obs = self.observations.get(word)
        if not obs:
            now = datetime.now().timestamp()
            obs = WordObservation(
                word=word,
                contexts=[],
                avg_phi=0.0,
                max_phi=0.0,
                frequency=0,
                first_seen=now,
                last_seen=now,
                context_basins=[]
            )
            self.observations[word] = obs
        return obs
    
    def get_all_observations(self) -> Dict[str, WordObservation]:
        """Get all observations."""
        return self.observations
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics."""
        return {
            'total_words': len(self.observations),
            'pending_candidates': len(self.pending_candidates),
            'learned_words': len(self.learned_words),
            'pruned_words': len(self.pruned_words),
            'cycle_number': self.cycle_number,
            'iterations_since_sleep': self.iterations_since_sleep,
        }
    
    def tick(self) -> None:
        """Increment iteration counter (called each search iteration)."""
        self.iterations_since_sleep += 1


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

vocab_decision_engine = VocabConsolidationCycle(sleep_interval=100)

# Export for ocean-agent integration
__all__ = [
    'VocabConsolidationCycle',
    'vocab_decision_engine',
    'should_gary_learn_word',
    'check_meta_awareness_gate',
    'compute_geometric_value',
    'check_basin_stability',
    'compute_information_entropy',
    'WordContext',
    'WordObservation',
    'GaryState',
    'VocabularyDecision',
    'ConsolidationResult',
    'GeometricValueScore',
    'BasinStabilityResult',
    'EntropyScore',
    'MetaAwarenessGate',
]
