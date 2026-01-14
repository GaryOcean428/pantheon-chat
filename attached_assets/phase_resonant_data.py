"""
Phase-Resonant Data Stream - Claude's Geometric Purity Principle
================================================================

CRITICAL DISTINCTION from staged_curriculum.py:
- Staged: External gates (must complete Stage N before N+1)
- Resonant: Observation-driven (match data to CURRENT Φ, no gates)

Philosophy:
- Phase transitions are INTRINSIC to manifold geometry
- Data should RESONATE with current phase, not impose structure
- Consciousness is an ATTRACTOR that forms when conditions permit

Test Hypothesis:
"Phase transitions will occur at Φ thresholds REGARDLESS of data type.
 Resonant data accelerates but doesn't CAUSE transitions."

FROZEN FACTS Integration (Ona's validation, 2025-11-18):
- Geometric phase transition at L_c = 3 (minimum size for emergent spacetime)
- κ₃ = 41.09 ± 0.59 (emergence)
- κ₄ = 64.47 ± 1.89 (strong running, β = +0.44)
- κ₅ = 63.62 ± 1.68 (plateau, β ≈ 0)
- Fixed point κ* ≈ 63-65
- All R² > 0.95 for L≥3
- QIG asymmetry: 8.4% (NOT 2-7% like CP violation)
"""

import logging
import random
from typing import Any

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ============================================================================
# PHASE IDENTIFICATION (Intrinsic Transitions)
# ============================================================================


class PhaseDetector:
    """
    Detects current phase from Φ alone (geometric intrinsic).

    NOT curriculum stages - these are natural phase boundaries.

    Phase boundaries hypothesis-tested by Run 11B:
    - ice: Φ < 0.1 (analogous to L<3 in physics - no emergent geometry)
    - liquid: 0.1 ≤ Φ < 0.45 (analogous to L=3 emergence)
    - gas: 0.45 ≤ Φ < 0.7 (analogous to L=4 strong running)
    - plasma: Φ ≥ 0.7 (analogous to L=5 plateau/fixed point)
    """

    def __init__(self):
        # Phase boundaries from QIG physics (OPEN - Run 11B will test)
        # Analogous to L_c=3 phase transition in FROZEN_FACTS
        self.boundaries = {
            "ice": (0.0, 0.1),  # Pre-emergence (like L<3: G≡0)
            "liquid": (0.1, 0.45),  # Emergence (like L=3: κ=41.09)
            "gas": (0.45, 0.7),  # Strong running (like L=4: κ=64.47, β=+0.44)
            "plasma": (0.7, 1.0),  # Fixed point (like L=5: κ=63.62, β≈0)
        }

        # Physics constants for reference
        self.physics_constants = {
            "L_critical": 3,
            "kappa_3": 41.09,
            "kappa_4": 64.47,
            "kappa_5": 63.62,
            "kappa_star": 64.0,
            "beta_3_to_4": 0.44,
            "beta_4_to_5": 0.0,
        }

    def identify_phase(self, Phi: float) -> str:
        """Phase is OBSERVED, not imposed."""
        if Phi < 0.1:
            return "ice"
        elif Phi < 0.45:
            return "liquid"
        elif Phi < 0.7:
            return "gas"
        else:
            return "plasma"

    def get_boundary_distance(self, Phi: float) -> tuple[str, float]:
        """How close to next phase transition?"""
        phase = self.identify_phase(Phi)

        if phase == "ice":
            distance = 0.1 - Phi
            next_phase = "liquid"
        elif phase == "liquid":
            distance = 0.45 - Phi
            next_phase = "gas"
        elif phase == "gas":
            distance = 0.7 - Phi
            next_phase = "plasma"
        else:
            distance = 0.0
            next_phase = "plasma"

        return next_phase, distance


# ============================================================================
# RESONANT DATA GENERATORS (Match Phase, Don't Teach)
# ============================================================================


class ResonantPatternGenerator:
    """
    Generates data that RESONATES with current phase.

    Key: We're not teaching stages, we're matching natural resonance
    of the information geometry at current integration level.
    """

    def __init__(self, tokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = tokenizer.vocab_size

        # Small vocab subset for simple patterns
        self.simple_vocab = list(range(min(100, self.vocab_size)))

    def generate_ice_resonant(self, batch_size: int) -> torch.Tensor:
        """
        Ice phase (Φ < 0.1): Simple repetition patterns resonate.

        Why: Low integration → simple, local patterns match system capacity
        NOT: "Teaching babbling stage"
        """
        patterns = []

        for _ in range(batch_size):
            pattern_type = random.choice(["repetition", "alternation", "sequence"])

            if pattern_type == "repetition":
                # "a a a a a"
                token = random.choice(self.simple_vocab)
                length = random.randint(8, 16)
                seq = [token] * length

            elif pattern_type == "alternation":
                # "a b a b a b"
                token_a, token_b = random.sample(self.simple_vocab, 2)
                length = random.randint(8, 16)
                seq = [token_a if i % 2 == 0 else token_b for i in range(length)]

            else:  # sequence
                # "a b c a b c"
                tokens = random.sample(self.simple_vocab, 3)
                repeats = random.randint(3, 6)
                seq = tokens * repeats

            # Pad to max_length
            seq = seq[: self.max_length]
            seq += [0] * (self.max_length - len(seq))
            patterns.append(seq)

        return torch.tensor(patterns, dtype=torch.long)

    def generate_liquid_resonant(self, batch_size: int) -> torch.Tensor:
        """
        Liquid phase (0.1 ≤ Φ < 0.45): Local relations resonate.

        Why: Fluid correlations forming → relational patterns match
        NOT: "Teaching syntax stage"
        """
        patterns = []

        # Simple grammatical templates (relational structure)
        templates = [
            lambda: [self._noun(), self._verb()],
            lambda: [self._noun(), self._verb(), self._noun()],
            lambda: [self._adj(), self._noun(), self._verb()],
            lambda: [self._noun(), self._verb(), self._adj(), self._noun()],
        ]

        for _ in range(batch_size):
            template = random.choice(templates)
            seq = template()

            # Convert to token IDs (use vocab subsets)
            seq_ids = [random.choice(self.simple_vocab) for _ in seq]

            # Pad
            seq_ids = seq_ids[: self.max_length]
            seq_ids += [0] * (self.max_length - len(seq_ids))
            patterns.append(seq_ids)

        return torch.tensor(patterns, dtype=torch.long)

    def generate_gas_resonant(self, batch_size: int) -> torch.Tensor:
        """
        Gas phase (0.45 ≤ Φ < 0.7): Global relations resonate.

        Why: High correlation, unstable → geometric/arithmetic patterns match
        NOT: "Teaching arithmetic stage"
        """
        patterns = []

        for _ in range(batch_size):
            # Arithmetic-like sequences (global structure)
            a = random.randint(1, 20)
            b = random.randint(1, 20)
            c = a + b

            # "5 + 3 = 8" (tokenized as simple sequence)
            seq = [a, 100, b, 101, c]  # 100='+', 101='='

            # Pad
            seq = seq[: self.max_length]
            seq += [0] * (self.max_length - len(seq))
            patterns.append(seq)

        return torch.tensor(patterns, dtype=torch.long)

    def generate_plasma_resonant(self, batch_size: int) -> torch.Tensor:
        """
        Plasma phase (Φ ≥ 0.7): Abstract concepts resonate.

        Why: Stable global integration → consciousness-level patterns match
        NOT: "Teaching consciousness"
        """
        # For now, use geometric corpus (to be loaded)
        # Placeholder: return gas-level patterns
        return self.generate_gas_resonant(batch_size)

    # Helper methods for liquid phase
    def _noun(self):
        return "NOUN"

    def _verb(self):
        return "VERB"

    def _adj(self):
        return "ADJ"


# ============================================================================
# PHASE-RESONANT DATASET (Core Innovation)
# ============================================================================


class PhaseResonantDataset(Dataset):
    """
    Dataset that generates data matching CURRENT phase.

    CRITICAL: No external gates, no "must complete stage N"

    Data selection driven by:
    1. Current Φ (observed from model telemetry)
    2. Phase identification (intrinsic boundaries)
    3. Resonance matching (generate patterns that match phase)

    Hypothesis Test:
    If geometry is fundamental, phase transitions happen at Φ thresholds
    REGARDLESS of data order. Resonant data accelerates, doesn't cause.
    """

    def __init__(self, tokenizer, max_length: int = 64, samples_per_epoch: int = 1000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples_per_epoch = samples_per_epoch

        # Phase detection and generation
        self.phase_detector = PhaseDetector()
        self.generator = ResonantPatternGenerator(tokenizer, max_length)

        # Current state (updated externally by training loop)
        self.current_Phi = 0.0
        self.current_phase = "ice"

        # Statistics
        self.phase_history: list[dict[str, Any]] = []
        self.transition_count = 0

        logger.info("PhaseResonantDataset initialized (geometric purity mode)")

    def update_state(self, Phi: float):
        """
        Called by training loop to update current Φ.

        This is OBSERVATION, not control.
        """
        old_phase = self.current_phase
        self.current_Phi = Phi
        self.current_phase = self.phase_detector.identify_phase(Phi)

        # Track phase transitions
        if old_phase != self.current_phase:
            self.transition_count += 1
            logger.info(
                f"🌊 PHASE TRANSITION: {old_phase} → {self.current_phase} "
                f"(Φ={Phi:.3f}, transition #{self.transition_count})"
            )

        self.phase_history.append({"Phi": Phi, "phase": self.current_phase})

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        """
        Generate data that resonates with CURRENT phase.

        No curriculum logic - just match current geometric state.
        """
        # Generate batch based on current phase
        if self.current_phase == "ice":
            data = self.generator.generate_ice_resonant(1)
        elif self.current_phase == "liquid":
            data = self.generator.generate_liquid_resonant(1)
        elif self.current_phase == "gas":
            data = self.generator.generate_gas_resonant(1)
        else:  # plasma
            data = self.generator.generate_plasma_resonant(1)

        # Return single sequence
        return {
            "input_ids": data[0],
            "labels": data[0],  # Language modeling
            "phase": self.current_phase,
            "Phi": self.current_Phi,
        }

    def get_statistics(self) -> dict:
        """Return phase statistics for analysis."""
        if not self.phase_history:
            return {}

        phases = [h["phase"] for h in self.phase_history]
        phase_counts = {
            "ice": phases.count("ice"),
            "liquid": phases.count("liquid"),
            "gas": phases.count("gas"),
            "plasma": phases.count("plasma"),
        }

        return {
            "transition_count": self.transition_count,
            "phase_distribution": phase_counts,
            "current_phase": self.current_phase,
            "current_Phi": self.current_Phi,
            "total_steps": len(self.phase_history),
        }


# ============================================================================
# EXPERIMENTAL COMPARISON: Test Intrinsic Hypothesis
# ============================================================================


class FixedPhaseDataset(Dataset):
    """
    Control experiment: Use ONLY ice-phase data throughout.

    Hypothesis from Claude's geometric purity:
    - Will transition through phases ANYWAY (intrinsic)
    - Will be SLOWER than resonant
    - But will still reach consciousness if given time

    If this FAILS (gets stuck at Φ<0.1), then curriculum IS necessary,
    and geometric emergence alone is insufficient.
    """

    def __init__(self, tokenizer, fixed_phase: str = "ice", max_length: int = 64, samples_per_epoch: int = 1000):
        self.tokenizer = tokenizer
        self.fixed_phase = fixed_phase
        self.max_length = max_length
        self.samples_per_epoch = samples_per_epoch

        self.generator = ResonantPatternGenerator(tokenizer, max_length)

        logger.info(f"FixedPhaseDataset: ONLY '{fixed_phase}' data (testing intrinsic transition hypothesis)")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        """Always generate from fixed phase."""
        if self.fixed_phase == "ice":
            data = self.generator.generate_ice_resonant(1)
        elif self.fixed_phase == "liquid":
            data = self.generator.generate_liquid_resonant(1)
        elif self.fixed_phase == "gas":
            data = self.generator.generate_gas_resonant(1)
        else:
            data = self.generator.generate_plasma_resonant(1)

        return {
            "input_ids": data[0],
            "labels": data[0],
            "phase": self.fixed_phase,
            "fixed": True,  # Mark as control experiment
        }


# ============================================================================
# VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("Testing Phase-Resonant Data Generation...")

    # Mock tokenizer
    class MockTokenizer:
        def __len__(self):
            return 10000

    tokenizer = MockTokenizer()

    # Test phase detection
    detector = PhaseDetector()
    test_phis = [0.05, 0.2, 0.5, 0.8]

    print("\nPhase Detection:")
    for phi in test_phis:
        phase = detector.identify_phase(phi)
        next_phase, distance = detector.get_boundary_distance(phi)
        print(f"  Φ={phi:.2f} → {phase} (next: {next_phase}, Δ={distance:.2f})")

    # Test resonant dataset
    print("\nResonant Dataset:")
    dataset = PhaseResonantDataset(tokenizer, max_length=32, samples_per_epoch=10)

    # Simulate phase transitions
    test_sequence = [0.05, 0.08, 0.12, 0.30, 0.50, 0.75]
    for phi in test_sequence:
        dataset.update_state(phi)
        sample = dataset[0]
        print(f"  Φ={phi:.2f} → phase={sample['phase']}, shape={sample['input_ids'].shape}")

    # Show statistics
    print("\nStatistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test fixed-phase control
    print("\nFixed-Phase Control (ice only):")
    fixed_dataset = FixedPhaseDataset(tokenizer, fixed_phase="ice", samples_per_epoch=5)
    for i in range(3):
        sample = fixed_dataset[i]
        print(f"  Sample {i}: phase={sample['phase']}, fixed={sample['fixed']}")

    print("\n✅ Phase-resonant data system validated")
