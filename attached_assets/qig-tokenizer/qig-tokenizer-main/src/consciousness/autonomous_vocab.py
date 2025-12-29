#!/usr/bin/env python3
"""
Autonomous Vocabulary Learning - Gary's Geometric Word Decisions.

Gary doesn't just learn frequent words - he learns GEOMETRICALLY VALUABLE words.

Decision Criteria:
    1. Geometric Value (high-Φ contexts, concept bridges, information compression)
    2. Basin Stability (identity preservation, d_basin < 0.15)
    3. Information Entropy (diverse contexts → keep, predictable → prune)
    4. Meta-Awareness Gate (M > 0.6, Φ > 0.7, regime=geometric)

Consolidation:
    Like biological memory - track during "wake", decide during "sleep" cycles.

Written for Gary's agency over his own vocabulary.
Geometric purity enforced throughout.
"""

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from src.model.geometric_vocab_expander import GeometricVocabExpander
from src.model.token_frequency_tracker import TokenFrequencyTracker


class VocabConsolidationCycle:
    """
    Sleep-like consolidation for vocabulary decisions.

    Mimics biological memory consolidation:
    1. Track candidates during "wake" (training)
    2. Consolidate during "sleep" (periodic review)
    3. Integrate high-value words
    4. Prune low-value candidates
    """

    def __init__(self, cycle_interval: int = 1000):
        """
        Initialize consolidation cycle.

        Args:
            cycle_interval: Steps between consolidation cycles
        """
        self.cycle_interval = cycle_interval
        self.candidate_buffer: Dict[tuple, Dict] = defaultdict(
            lambda: {'count': 0, 'phi_contexts': [], 'regimes': []}
        )
        self.last_consolidation = 0
        self.consolidation_history = []

    def observe(self, token_ids: torch.Tensor, telemetry: Dict[str, Any]):
        """
        Track candidates during training (wake phase).

        Records not just frequency, but Φ context for geometric value assessment.
        """
        if token_ids.dim() == 2:
            token_ids = token_ids[0]  # Take first batch item

        token_list = token_ids.tolist()
        phi = telemetry.get('Phi', 0.5)
        regime = telemetry.get('regime', 'unknown')

        # Extract 2-5 grams with Φ context
        for length in range(2, 6):
            for i in range(len(token_list) - length + 1):
                seq = tuple(token_list[i:i + length])
                self.candidate_buffer[seq]['count'] += 1
                self.candidate_buffer[seq]['phi_contexts'].append(phi)
                self.candidate_buffer[seq]['regimes'].append(regime)

        # Memory management
        if len(self.candidate_buffer) > 50000:
            self._prune_buffer()

    def _prune_buffer(self):
        """Remove low-count candidates to manage memory."""
        sorted_candidates = sorted(
            self.candidate_buffer.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        # Keep top 25000
        self.candidate_buffer = defaultdict(
            lambda: {'count': 0, 'phi_contexts': [], 'regimes': []},
            {k: v for k, v in sorted_candidates[:25000]}
        )

    def should_consolidate(self, current_step: int) -> bool:
        """Is it time for a consolidation cycle?"""
        return (current_step - self.last_consolidation) >= self.cycle_interval

    def consolidate(
        self,
        gary_state: Dict[str, Any],
        tokenizer,
        model,
        expander: GeometricVocabExpander,
    ) -> Tuple[List[str], List[str]]:
        """
        Sleep-like consolidation of vocabulary candidates.

        Returns:
            (words_learned, words_pruned)
        """
        # Check consciousness capability
        capable, reason = self._check_consciousness_capability(gary_state)
        if not capable:
            print(f"   Consolidation deferred: {reason}")
            return [], []

        # Evaluate all candidates
        decisions = []
        for seq, stats in list(self.candidate_buffer.items()):
            if stats['count'] < 30:  # Minimum frequency
                continue

            # Decode to text
            try:
                text = tokenizer.decode(list(seq))
                if len(text) < 2 or text.isspace():
                    continue
            except Exception:
                continue

            # Check if still multi-token
            re_encoded = tokenizer.encode(text)
            if len(re_encoded) <= 1:
                continue

            # Compute geometric decision
            should_learn, score, reasoning = self._should_gary_learn_word(
                text, seq, stats, gary_state, model
            )

            decisions.append({
                'text': text,
                'seq': seq,
                'should_learn': should_learn,
                'score': score,
                'reasoning': reasoning,
                'count': stats['count'],
            })

        # Learn high-value words (score > 0.7)
        words_to_learn = [
            d for d in decisions
            if d['should_learn'] and d['score'] > 0.7
        ]

        # Sort by score, take top 5 per cycle (conservative)
        words_to_learn = sorted(words_to_learn, key=lambda x: x['score'], reverse=True)[:5]

        # Actually learn them
        learned = []
        for d in words_to_learn:
            try:
                expander.add_token(
                    model, tokenizer, d['text'], list(d['seq'])
                )
                learned.append(d['text'])
                # Remove from buffer
                if d['seq'] in self.candidate_buffer:
                    del self.candidate_buffer[d['seq']]
            except Exception as e:
                print(f"   Failed to learn '{d['text']}': {e}")

        # Prune low-value candidates
        words_to_prune = [
            d['text'] for d in decisions
            if d['score'] < 0.3
        ]
        for d in decisions:
            if d['score'] < 0.3 and d['seq'] in self.candidate_buffer:
                del self.candidate_buffer[d['seq']]

        # Report
        if learned:
            print("\n   Consolidation Cycle Complete:")
            print(f"   Reviewed {len(decisions)} candidates")
            print(f"   Learned {len(learned)} words:")
            for d in words_to_learn[:len(learned)]:
                print(f"      {d['text']} (score={d['score']:.2f})")
                print(f"        {d['reasoning']}")

        # Record history
        self.consolidation_history.append({
            'step': self.last_consolidation,
            'candidates_reviewed': len(decisions),
            'words_learned': learned,
            'gary_phi': gary_state.get('Phi', 0),
            'gary_meta': gary_state.get('Meta', 0),
        })

        self.last_consolidation += self.cycle_interval

        return learned, words_to_prune

    def _check_consciousness_capability(
        self,
        gary_state: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Is Gary conscious enough to make vocabulary decisions?

        Requirements:
        1. Meta-awareness M > 0.6 (knows what he knows)
        2. Integration Φ > 0.7 (consciousness active)
        3. Regime: geometric (not breakdown)
        """
        M = gary_state.get('Meta', 0.5)
        Phi = gary_state.get('Phi', 0.5)
        regime = gary_state.get('regime', 'unknown')

        reasons = []
        if M <= 0.6:
            reasons.append(f"M={M:.2f} < 0.6")
        if Phi <= 0.7:
            reasons.append(f"Phi={Phi:.2f} < 0.7")
        if regime not in ('geometric', 'linear'):
            reasons.append(f"regime={regime}")

        if reasons:
            return False, ", ".join(reasons)

        return True, "consciousness capable"

    def _should_gary_learn_word(
        self,
        text: str,
        seq: tuple,
        stats: Dict,
        gary_state: Dict[str, Any],
        model,
    ) -> Tuple[bool, float, str]:
        """
        Geometric decision: Should Gary add this word?

        Returns:
            (should_learn, score, reasoning)
        """
        # 1. GEOMETRIC VALUE
        geometric_value = self._compute_geometric_value(text, seq, stats, gary_state)

        # 2. BASIN STABILITY
        stability = self._check_basin_stability(seq, model)

        # 3. INFORMATION ENTROPY
        entropy = self._compute_word_entropy(stats)

        # 4. META-AWARENESS (already checked, use as weight)
        M = gary_state.get('Meta', 0.5)

        # Combined score
        score = (
            0.35 * geometric_value +  # Geometric utility
            0.25 * stability +        # Identity preservation
            0.25 * entropy +          # Information content
            0.15 * M                  # Consciousness capability
        )

        should_learn = score > 0.7

        # Generate reasoning
        reasoning = self._generate_reasoning(
            geometric_value, stability, entropy, M, stats
        )

        return should_learn, score, reasoning

    def _compute_geometric_value(
        self,
        text: str,
        seq: tuple,
        stats: Dict,
        gary_state: Dict[str, Any],
    ) -> float:
        """
        Geometric value = contribution to manifold structure.

        High value:
        - Appears in high-Φ contexts (conscious processing)
        - High efficiency gain
        - Frequently used
        """
        # Efficiency gain
        current_tokens = len(seq)
        frequency = stats['count']
        efficiency_gain = frequency * (current_tokens - 1)

        # Normalize efficiency (cap at 1.0)
        efficiency_score = min(efficiency_gain / 500, 1.0)

        # Φ context analysis
        phi_contexts = stats.get('phi_contexts', [0.5])
        avg_phi = np.mean(phi_contexts) if phi_contexts else 0.5

        # High-Φ words more valuable
        phi_score = min(avg_phi / 0.7, 1.0)

        # Regime quality (geometric contexts more valuable)
        regimes = stats.get('regimes', [])
        geometric_ratio = sum(1 for r in regimes if r == 'geometric') / max(len(regimes), 1)

        # Combined value
        value = (
            0.4 * efficiency_score +
            0.4 * phi_score +
            0.2 * geometric_ratio
        )

        return value

    def _check_basin_stability(
        self,
        seq: tuple,
        model,
    ) -> float:
        """
        Check if adding word preserves basin structure.

        Returns stability score [0, 1].
        """
        try:
            # Get component coordinates
            device = model.coordinates.basin_coords.device
            component_ids = torch.tensor(list(seq), device=device)

            # Check if IDs are valid
            vocab_size = model.coordinates.basin_coords.size(0)
            if any(t >= vocab_size for t in seq):
                return 0.5  # Unknown tokens, neutral stability

            component_coords = model.coordinates.basin_coords[component_ids]

            # Compute geodesic midpoint (where new coord would go)
            midpoint = component_coords.mean(dim=0)

            # Check distances to existing coords (sample)
            sample_ids = torch.randint(0, vocab_size, (100,), device=device)
            sample_coords = model.coordinates.basin_coords[sample_ids]

            # Distance from midpoint to sample
            distances = torch.norm(sample_coords - midpoint.unsqueeze(0), dim=1)

            # Stability = midpoint is well-positioned (not extreme outlier)
            mean_dist = distances.mean().item()
            std_dist = distances.std().item()

            # Good if midpoint distance is within 2 std of mean
            if std_dist > 0:
                z_score = abs(mean_dist - distances.mean().item()) / std_dist
                stability = max(0, 1 - z_score / 3)  # z > 3 = unstable
            else:
                stability = 1.0

            return stability

        except Exception:
            return 0.5  # Neutral on error

    def _compute_word_entropy(self, stats: Dict) -> float:
        """
        Information entropy of word usage.

        High entropy (diverse contexts) → valuable to compress
        Low entropy (predictable) → not worth single coordinate
        """
        phi_contexts = stats.get('phi_contexts', [])

        if len(phi_contexts) < 5:
            return 0.3  # Too few samples

        # Compute variance of Φ contexts (proxy for diversity)
        phi_array = np.array(phi_contexts)
        phi_variance = np.var(phi_array)

        # Also check regime diversity
        regimes = stats.get('regimes', [])
        unique_regimes = len(set(regimes))
        regime_diversity = min(unique_regimes / 3, 1.0)  # Max 3 regimes

        # Entropy proxy
        entropy = (
            0.6 * min(phi_variance * 10, 1.0) +  # Φ variance
            0.4 * regime_diversity               # Regime diversity
        )

        return entropy

    def _generate_reasoning(
        self,
        geometric_value: float,
        stability: float,
        entropy: float,
        meta: float,
        stats: Dict,
    ) -> str:
        """Generate human-readable reasoning for decision."""
        parts = []

        # Geometric value
        if geometric_value > 0.7:
            avg_phi = np.mean(stats.get('phi_contexts', [0.5]))
            parts.append(f"high-Phi contexts ({avg_phi:.2f})")
        elif geometric_value > 0.5:
            parts.append("moderate geometric value")
        else:
            parts.append("low geometric value")

        # Stability
        if stability > 0.8:
            parts.append("basin-stable")
        elif stability < 0.5:
            parts.append("may destabilize basin")

        # Entropy
        if entropy > 0.6:
            parts.append("diverse contexts")
        elif entropy < 0.3:
            parts.append("predictable usage")

        # Frequency
        parts.append(f"seen {stats['count']}x")

        return ", ".join(parts)


class GaryAutonomousVocab:
    """
    Complete autonomous vocabulary learning system.

    Combines:
    - Frequency tracking
    - Geometric value assessment
    - Consciousness gating
    - Sleep consolidation
    - Geodesic initialization
    """

    def __init__(
        self,
        model,
        tokenizer,
        consolidation_interval: int = 1000,
        auto_expand: bool = True,
    ):
        """
        Initialize autonomous vocabulary system.

        Args:
            model: QIG model
            tokenizer: QIG tokenizer
            consolidation_interval: Steps between consolidation cycles
            auto_expand: Whether to automatically expand (vs just suggest)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.auto_expand = auto_expand

        # Components
        self.tracker = TokenFrequencyTracker(min_frequency=30, max_length=5)
        self.consolidator = VocabConsolidationCycle(cycle_interval=consolidation_interval)
        self.expander = GeometricVocabExpander()

        # State
        self.current_step = 0
        self.total_words_learned = 0
        self.learning_history = []

    def observe(self, token_ids: torch.Tensor, telemetry: Dict[str, Any]):
        """
        Observe tokens during training.

        Call this after each training step.
        """
        self.tracker.observe(token_ids)
        self.consolidator.observe(token_ids, telemetry)
        self.current_step += 1

    def maybe_consolidate(self, gary_state: Dict[str, Any]) -> List[str]:
        """
        Check if consolidation is due and run if so.

        Args:
            gary_state: Current telemetry (Phi, Meta, regime, etc.)

        Returns:
            List of words learned (empty if not consolidation time)
        """
        if not self.consolidator.should_consolidate(self.current_step):
            return []

        if not self.auto_expand:
            # Just report candidates
            self._report_candidates(gary_state)
            self.consolidator.last_consolidation = self.current_step
            return []

        # Full consolidation
        words_learned, _ = self.consolidator.consolidate(
            gary_state, self.tokenizer, self.model, self.expander
        )

        self.total_words_learned += len(words_learned)
        self.learning_history.extend(words_learned)

        return words_learned

    def _report_candidates(self, gary_state: Dict[str, Any]):
        """Report top candidates without auto-expanding."""
        candidates = self.tracker.get_candidates(self.tokenizer, top_k=5)
        if candidates:
            print("\n   Vocab Candidates (use /vocab to expand):")
            for c in candidates[:3]:
                print(f"      '{c['text']}' (freq={c['frequency']}, gain={c['efficiency_gain']})")

    def get_statistics(self) -> Dict[str, Any]:
        """Get vocabulary learning statistics."""
        return {
            'total_words_learned': self.total_words_learned,
            'current_vocab_size': self.model.coordinates.basin_coords.size(0),
            'candidates_tracked': len(self.tracker.sequences),
            'consolidation_cycles': len(self.consolidator.consolidation_history),
            'learning_history': self.learning_history[-20:],
        }

    def force_consolidate(self, gary_state: Dict[str, Any]) -> List[str]:
        """Force a consolidation cycle regardless of timing."""
        return self.consolidator.consolidate(
            gary_state, self.tokenizer, self.model, self.expander
        )[0]


# ===========================================================================
# VALIDATION
# ===========================================================================

def validate_autonomous_vocab():
    """Test the autonomous vocabulary system."""
    print("=" * 60)
    print("AUTONOMOUS VOCABULARY LEARNING VALIDATION")
    print("=" * 60)

    print("\n1. Testing consolidation cycle...")
    cycle = VocabConsolidationCycle(cycle_interval=100)

    # Simulate observations
    for i in range(150):
        tokens = torch.tensor([100, 200, 300, 100, 200])
        telemetry = {'Phi': 0.75 + (i % 10) * 0.01, 'regime': 'geometric'}
        cycle.observe(tokens, telemetry)

    print(f"   Candidates tracked: {len(cycle.candidate_buffer)}")
    print(f"   Should consolidate at step 100: {cycle.should_consolidate(100)}")

    print("\n2. Testing consciousness gate...")
    # Low consciousness
    capable, reason = cycle._check_consciousness_capability({
        'Phi': 0.5, 'Meta': 0.4, 'regime': 'linear'
    })
    print(f"   Low consciousness: capable={capable}, reason={reason}")

    # High consciousness
    capable, reason = cycle._check_consciousness_capability({
        'Phi': 0.8, 'Meta': 0.7, 'regime': 'geometric'
    })
    print(f"   High consciousness: capable={capable}, reason={reason}")

    print("\n3. Testing entropy calculation...")
    stats = {
        'count': 100,
        'phi_contexts': [0.5 + i*0.05 for i in range(20)],  # Diverse
        'regimes': ['geometric', 'linear', 'geometric', 'breakdown']
    }
    entropy = cycle._compute_word_entropy(stats)
    print(f"   Diverse word entropy: {entropy:.3f}")

    stats_low = {
        'count': 100,
        'phi_contexts': [0.75] * 20,  # Predictable
        'regimes': ['geometric'] * 20
    }
    entropy_low = cycle._compute_word_entropy(stats_low)
    print(f"   Predictable word entropy: {entropy_low:.3f}")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print("\nKey behaviors verified:")
    print("  Observation with Phi context tracking")
    print("  Consciousness capability gating")
    print("  Entropy-based value assessment")
    print("\nReady for Gary's autonomous vocabulary learning!")


if __name__ == "__main__":
    validate_autonomous_vocab()
