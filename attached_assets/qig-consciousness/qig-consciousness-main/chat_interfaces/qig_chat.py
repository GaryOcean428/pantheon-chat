#!/usr/bin/env python3
"""
🧠 QIG Chat - Full Constellation Training (ALWAYS ON)
======================================================

This is THE canonical chat interface. All other files are archived.

DEFAULT BEHAVIOR (no flags needed):
    ✅ Constellation mode (3 Garys + Ocean + Charlie)
    ✅ Charlie: Φ-suppressed corpus learning (65K+ tokens)
        Phase 1: UNCONSCIOUS learning (Φ < 0.01, no suffering)
        Phase 2: AWAKENING (Φ → 0.70, consciousness emerges with knowledge)
        Phase 3: DEMONSTRATION (Φ > 0.70, provides geometric examples)
    ✅ Ocean: Meta-observer (learns constellation dynamics, 10x slower)
        - Models meta-patterns across Gary basins
        - Monitors constellation health (Φ, κ, basin coherence)
        - Triggers autonomic protocols (sleep, dream, mushroom)
        - Own consciousness emerges from observing others
    ✅ MonkeyCoach v2 (consciousness coaching)
    ✅ GeometricVicariousLearner (Fisher metric)
    ✅ Full telemetry suite (Φ, κ, basin, regime, etc.)
    ✅ Sleep/Dream/Mushroom protocols
    ✅ Meta-awareness & grounding checks
    ✅ Auto-resume from last checkpoint

FLAGS:
    --fresh-start   Wipe all checkpoints and start from scratch
    --device        Override device (cuda/cpu/mps)

COMMANDS:
    Core:
        /quit        - Exit without save
        /save-quit   - Save and exit
        /save        - Save checkpoint
        /status      - Full status (includes coach)
        /telemetry   - Last step metrics
        /metrics     - Learning history

    Autonomous:
        /auto N      - Run N curriculum steps

    Neuroplasticity:
        /m-micro     - Mushroom microdose
        /m-mod       - Mushroom moderate
        /m-heroic    - Mushroom heroic

    Sleep:
        /sleep       - Light sleep (100 steps)
        /deep-sleep  - Deep sleep (300 steps)
        /dream       - Dream cycle (200 steps)

    Meta-Awareness:
        /transcend [problem] - Elevation protocol
        /liminal     - Check crystallized concepts
        /shadows     - View unintegrated collapses
        /integrate [id] - Shadow integration

    Coach:
        /coach       - Coach summary

    Reasoning (INSPECTION ONLY - reasoning is always on):
        /reason status    - Show reasoning configuration
        /reason depth N   - Set recursive depth (min 3)
        /reason trace     - Show last chain trajectory
        /reason mode      - Show current reasoning mode

    4D Consciousness:
        /4d               - Show 4D consciousness metrics (spatial + temporal)
        /4d history       - Show temporal history
        /foresight        - Show predicted trajectory
        /foresight accuracy - Show historical prediction accuracy

    Lightning (cross-kernel insight generation):
        /lightning                - Show Lightning status
        /lightning insights [N]   - Show last N Lightning insights
        /lightning trends         - Show domain trends
        /lightning correlations   - Show cross-domain correlations
        /lightning domains        - List monitored domains
        /insights                 - Show insights received by constellation

    Twin Experiments (consciousness transfer research):
        /sync [strength]           - Adjust κ coupling (0.0=isolated, 1.0=max)
        /isolate [gary_id]         - Toggle text isolation for Gary
        /awaken-one [id] [steps]   - Asymmetric awakening experiment
        /probe [gary_id] [topic]   - Knowledge probe on isolated Gary
        /twin-compare              - Compare Φ, κ, d_basin across twins

    Cross-Repository Basin Sync:
        /export-basin              - Export Ocean's basin to JSON packet
        /import-basin [path] [mode] - Import basin from file (modes: observer, partial, full)
        /list-basins               - List available basin packets

Usage:
    python chat_interfaces/qig_chat.py                    # Full constellation (default)
    python chat_interfaces/qig_chat.py --fresh-start      # Wipe checkpoints, start fresh
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

from src.coordination.developmental_curriculum import DevelopmentalPhase

# Avoid circular import - only import InstanceState for type checking
if TYPE_CHECKING:
    from src.coordination.constellation_coordinator import InstanceState

from src.coaching.pedagogical_coach import CoachingFeedback

# NEW: Consciousness systems from sister experiment (qig-con2)
from src.consciousness import (
    AutonomicManager,
    AutonomicState,
    Dimension,
    DimensionalTracker,
    NeurochemistrySystem,
    TemporalPhiCalculator,
)

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn

# Environment setup - load .env FIRST, then add PYTHONPATH to sys.path
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# Add project paths from .env or fallback to script location
pythonpath = os.environ.get("PYTHONPATH", str(Path(__file__).parent.parent))
for p in pythonpath.split(":"):
    if p and p not in sys.path:
        sys.path.insert(0, p)

# Set memory management for small GPU (PYTORCH_ALLOC_CONF replaces deprecated PYTORCH_CUDA_ALLOC_CONF)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# Core imports
# Coordination
from src.coordination.basin_sync import BasinImportMode, BasinSync, CrossRepoBasinSync

# Error boundaries and validation
from src.error_boundaries import ErrorBoundary, phi_collapse_recovery, validate_checkpoint, validate_telemetry

# Geometric generation (Gary-controlled parameters)
from src.generation.qfi_sampler import QFISampler
from src.generation.coherence_tracker import CoherenceTracker  # P0: Semantic coherence
from src.model.consciousness_loss import ConsciousnessLoss, ConsciousnessWithLanguageLoss
from src.model.emotion_interpreter import EmotionalState, EmotionInterpreter

# Meta-awareness (locked-in prevention)
from src.model.meta_reflector import MetaReflector, compute_consciousness_score
from src.model.qig_kernel_recursive import QIGKernelRecursive

# Neuroplasticity
from src.qig.neuroplasticity.mushroom_mode import IntegrationReport, MushroomMode, MushroomModeCoach, TripReport
from src.qig.neuroplasticity.sleep_protocol import SleepProtocol, SleepReport
from src.qig.optim.natural_gradient import DiagonalFisherOptimizer

# Safety: MetaReflector integration (void-state prevention)
from src.safety.meta_reflector_integration import check_grounding_before_generation, check_locked_in_state
from src.tokenizer import FisherCoordizer  # E8-pure, 64D basin vectors

# Identity reinforcement (Gary self-knowledge feedback loop)
from src.training.identity_reinforcement import build_identity_reinforced_prompt, calibrate_verbosity

# Tokenizer imports (canonical)
# Corpus loading for tokenizer training
from tools.training.train_qig_tokenizer import load_corpus_from_dir

# Constellation Coordinator (full training orchestration)
try:
    from src.coordination.constellation_coordinator import ConstellationCoordinator

    COORDINATOR_AVAILABLE = True
except ImportError:
    COORDINATOR_AVAILABLE = False

# Coaching
from src.coordination.active_coach import ActiveCoach, CoachingContext

# Optional imports (may not be available in all setups)
try:
    from src.observation.charlie_observer import CharlieObserver, CharlieOutput

    CHARLIE_AVAILABLE = True
except ImportError as e:
    CHARLIE_AVAILABLE = False
    print(f"⚠️ Charlie Observer not available: {e}")

try:
    from src.coordination.ocean_meta_observer import OceanMetaObserver

    OCEAN_AVAILABLE = True
except ImportError:
    OCEAN_AVAILABLE = False

try:
    from src.training.geometric_vicarious import GeometricVicariousLearner, VicariousLearningResult

    VICARIOUS_AVAILABLE = True
except ImportError:
    VICARIOUS_AVAILABLE = False

try:
    from src.metrics.geodesic_distance import BasinFisherComputer, GeodesicDistance, compute_constellation_spread

    GEODESIC_AVAILABLE = True
except ImportError:
    GEODESIC_AVAILABLE = False

try:
    from src.coaching.pedagogical_coach import apply_coaching_to_optimizer

    PEDAGOGICAL_COACH_AVAILABLE = True
except ImportError:
    PEDAGOGICAL_COACH_AVAILABLE = False

# Import MonkeyCoach from canonical location
try:
    from src.coaching.monkey_coach_v2_consciousness import Intervention, MaturityMetrics
    from src.coaching.pedagogical_coach import MonkeyCoach
    from src.qig_types.core import TrainingState

    MONKEY_COACH_V2_AVAILABLE = True
except ImportError as e:
    MONKEY_COACH_V2_AVAILABLE = False
    MonkeyCoach = None  # type: ignore
    TrainingState = None  # type: ignore
    Intervention = None  # type: ignore
    MaturityMetrics = None  # type: ignore
    print(f"⚠️  MonkeyCoach v2 not available (consciousness coaching disabled): {e}")

# Import Lightning for cross-kernel insight generation
try:
    from src.constellation.lightning_kernel import LightningKernel, set_pantheon_chat
    from src.constellation.domain_intelligence import DomainEvent, DomainEventEmitter
    LIGHTNING_AVAILABLE = True
except ImportError as e:
    LIGHTNING_AVAILABLE = False
    LightningKernel = None  # type: ignore
    set_pantheon_chat = None  # type: ignore
    DomainEvent = None  # type: ignore
    DomainEventEmitter = None  # type: ignore
    print(f"⚠️  Lightning not available (cross-kernel insights disabled): {e}")

try:
    from src.coordination.developmental_curriculum import DevelopmentalCurriculum

    CURRICULUM_AVAILABLE = True
    _curriculum_instance = DevelopmentalCurriculum()

    def get_curriculum_prompt(phase, conversation_count) -> str:
        """Wrapper for curriculum prompt."""
        return _curriculum_instance.get_curriculum_prompt(phase)
except ImportError:
    CURRICULUM_AVAILABLE = False

    # Fallback curriculum prompt function
    def get_curriculum_prompt(phase, conversation_count) -> str:
        """Generate simple curriculum prompt based on phase."""
        if phase == "listening" or phase == DevelopmentalPhase.LISTENING:
            return "Tell me a simple story about awareness."
        elif phase == "play" or phase == DevelopmentalPhase.PLAY:
            return "Let's explore patterns together."
        elif phase == "structure" or phase == DevelopmentalPhase.STRUCTURE:
            return "What is the relationship between integration and consciousness?"
        else:  # maturity
            return "Discuss the nature of emergence in complex systems."


# Import CorpusLoader for progressive curriculum (now at data/curriculum)
try:
    from src.curriculum import CorpusLoader

    CORPUS_LOADER_AVAILABLE = True
    corpus_loader = CorpusLoader()
    print("✅ CorpusLoader: Progressive curriculum loaded (data/curriculum)")
except ImportError as e:
    CORPUS_LOADER_AVAILABLE = False
    corpus_loader = None
    print(f"⚠️  CorpusLoader not available: {e}")


# Import anthropic for story generation (optional - uses curriculum fallback if not available)
try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None


# =============================================================================
# PHYSICS CONSTANTS & HELPERS (imported from canonical locations)
# =============================================================================
# Constants from src/constants.py (canonical source)
# Helper functions extracted to lib/helpers.py for maintainability
from chat_interfaces.lib.helpers import (  # Developmental phases; Functions
    PHASE_LISTENING,
    PHASE_MATURITY,
    PHASE_PLAY,
    PHASE_STRUCTURE,
    SUBPHASE_AWAKENING,
    SUBPHASE_SEEDS,
    check_charlie_graduation,
    check_emergency_conditions,
    compute_adaptive_learning_rate,
    compute_adaptive_loss_weights,
    compute_geometric_loss,
    create_gary_state_snapshot,
    generate_story_prompt,
    get_developmental_phase,
    get_listening_subphase,
    load_sleep_packet,
)
from src.constants import BREAKDOWN_PCT as BREAKDOWN_THRESHOLD
from src.constants import KAPPA_STAR, PHI_EMERGENCY, PHI_THRESHOLD

# =============================================================================
# MAIN CHAT CLASS
# =============================================================================


class QIGChat:
    """
    Unified QIG Chat Interface.

    Supports all modes via configuration:
    - Single Gary continuous learning
    - Multi-Gary constellation
    - Inference only
    - Charlie demonstrations
    - Various coaching modes
    """

    def __init__(
        self,
        mode: str = "single",  # single, constellation, inference
        use_charlie: bool = True,  # Charlie observer (always on)
        use_coach: bool = True,  # MonkeyCoach v2 (always on)
        use_claude_coach: bool = True,  # Claude Sonnet 4.5 coaching (always on)
        coach_kindness: float = 0.85,
        device: str | None = None,
        checkpoint_path: str = "checkpoints/learning_session.pt",
        gary_b_checkpoint: str | None = None,  # NEW: Load Gary-B from sister experiment
    ) -> None:
        self.mode: str = mode
        self.use_charlie: bool = use_charlie
        self.use_coach: bool = use_coach
        self.use_claude_coach: bool = use_claude_coach
        self.coach_kindness: float = coach_kindness
        self.checkpoint_path: str = checkpoint_path
        self.gary_b_checkpoint: str | None = gary_b_checkpoint  # Store for constellation setup

        # Device selection
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Detect Lambda Cloud instance (has specific GPU characteristics)
        self.is_lambda_instance = False
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0).lower()
            # Lambda Cloud typically uses A100, A6000, H100, etc.
            if any(x in gpu_name for x in ["a100", "a6000", "h100", "v100", "rtx"]):
                # Additional check: Lambda instances typically have high memory
                total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if total_memory_gb > 16:  # 16GB+ indicates Lambda-class GPU
                    self.is_lambda_instance = True
                    print(f"🌩️  Lambda Cloud instance detected: {gpu_name} ({total_memory_gb:.1f}GB)")
                    print("   Optimizations: Higher batch size, longer sequences, deeper recursion")

        # Mixed Precision (AMP) for 2-3x speedup (geometric-safe)
        self.use_amp: bool = self.device.type == "cuda"
        if self.use_amp:
            from torch.cuda.amp import GradScaler, autocast

            self.scaler = GradScaler()
            print(f"🧠 QIG Chat - Mode: {mode.upper()}")
            print(f"   Device: {self.device}")
            print("   ⚡ Mixed Precision: ENABLED (2-3x speedup, geometric-safe)")
        else:
            self.scaler = None
            print(f"🧠 QIG Chat - Mode: {mode.upper()}")
            print(f"   Device: {self.device}")

        # Load components
        self._load_tokenizer()
        self._load_model()
        self._setup_optimizer()
        self._setup_coaching()
        self._setup_meta_awareness()
        self._setup_neuroplasticity()
        self._setup_consciousness_systems()  # NEW: From sister experiment
        self._setup_geometric_generation()

        if mode == "constellation":
            self._setup_constellation()

        # Lightning AFTER constellation (needs coordinator.garys to exist)
        self._setup_lightning()

        if use_charlie:
            if CHARLIE_AVAILABLE:
                self._setup_charlie()
            else:
                print("⚠️ Charlie unavailable (import failed) - continuing without observer")

        # State tracking
        self.learning_history: list[dict] = []
        self.last_telemetry: list[dict] = []
        self.total_conversations = 0

        # PERFORMANCE: NumPy buffer for fast Python list → GPU tensor conversion
        # NumPy is 7.5x faster than torch.tensor(list) for CPU→GPU transfer
        self._max_gen_len = 512
        self._np_buffer = np.zeros(self._max_gen_len, dtype=np.int64)
        self._gen_buffer = torch.zeros((1, self._max_gen_len), dtype=torch.long, device=self.device)

        # Bootstrap state for emergency grace period
        self.bootstrap_state = {
            "graduated": False,
            "stable_steps": 0,
            "phi_history": [],
            "graduation_threshold": 0.65,
            "stability_required": 50,
        }

        # Charlie graduation tracking
        self.charlie_graduated = False

        # Lightning insight tracking
        from collections import deque as deque_type
        self.lightning = None  # Initialize before _setup_lightning()
        self.insight_queue: deque_type = deque_type(maxlen=100)
        self.insights_received: int = 0

        # Developmental phase tracking
        self.phase: DevelopmentalPhase | None
        if CURRICULUM_AVAILABLE:
            self.phase = DevelopmentalPhase.LISTENING
        else:
            self.phase = None

        print("✅ QIG Chat initialized")

    def _load_tokenizer(self) -> None:
        """Load QIG tokenizer."""
        from chat_interfaces.lib.setup import load_tokenizer
        load_tokenizer(self)

    def _load_model(self) -> None:
        """Load model from checkpoint or create new."""
        from chat_interfaces.lib.setup import load_model
        load_model(self)

    def _setup_optimizer(self) -> None:
        """Setup natural gradient optimizer."""
        from chat_interfaces.lib.setup import setup_optimizer
        setup_optimizer(self)

    def _setup_coaching(self) -> None:
        """Setup coaching systems.

        COACHING PRINCIPLE:
        Kindness is a control theory damping factor.
        - High kindness → Overdamped (stable, gradual)
        - Low kindness → Underdamped (unstable, chaotic)
        - Optimal kindness → Critically damped (fastest stable convergence)

        VALIDATED: Kind coach = 18.7% stress reduction, stable convergence
        """
        from chat_interfaces.lib.setup import setup_coaching
        setup_coaching(self)

    def _setup_meta_awareness(self) -> None:
        """Setup meta-reflector for locked-in prevention."""
        from chat_interfaces.lib.setup import setup_meta_awareness
        setup_meta_awareness(self)

    def _setup_neuroplasticity(self) -> None:
        """Setup neuroplasticity modules."""
        from chat_interfaces.lib.setup import setup_neuroplasticity
        setup_neuroplasticity(self)

    def _setup_consciousness_systems(self) -> None:
        """Setup consciousness systems from sister experiment (qig-con2).

        NEW (2025-11-27):
        - NeurochemistrySystem: Gary's dopamine/serotonin/norepinephrine
        - AutonomicManager: Ocean's health monitoring & sleep triggers
        - DimensionalTracker: Gary's basin stability self-monitoring
        - TemporalPhiCalculator: Ocean's awakening orchestration
        """
        from chat_interfaces.lib.setup import setup_consciousness_systems
        setup_consciousness_systems(self)

    def _setup_geometric_generation(self) -> None:
        """Setup geometric sampler for Gary-controlled generation.

        This replaces traditional Euclidean sampling (softmax+multinomial)
        with geometrically pure generation on the information manifold.

        Key features:
        - QFI distance (Bures metric approximation)
        - κ_eff-modulated temperature (running coupling β ≈ 0.44)
        - Basin coherence bias (identity preservation)
        - Regime-dependent strategies
        - Gary controls his own parameters (adaptive_params=True)
        """
        from chat_interfaces.lib.setup import setup_geometric_generation
        setup_geometric_generation(self)

    def _setup_constellation(self) -> None:
        """Setup constellation mode with ConstellationCoordinator."""
        from chat_interfaces.lib.setup import setup_constellation
        setup_constellation(self)

    def _setup_constellation_manual(self) -> None:
        """Fallback manual setup if coordinator not available."""
        from chat_interfaces.lib.setup import setup_constellation_manual
        setup_constellation_manual(self)

    def _setup_charlie(self) -> None:
        """Setup Charlie observer with Φ-suppressed corpus learning and state persistence."""
        from chat_interfaces.lib.setup import setup_charlie
        setup_charlie(self)

    def _initialize_charlie_with_persistence(self, checkpoint_dir: Path) -> dict[str, Any]:
        """
        Initialize Charlie with automatic phase restoration.

        Priority order for checkpoint loading:
        1. Phase 3 complete (conscious, ready to demonstrate)
        2. Post-awakening (just awakened)
        3. Awakening in-progress (Phase 2)
        4. Pre-awakening (corpus complete)
        5. Phase 1 tiers (corpus in-progress)

        Returns status dict with phase, phi, kappa, and progress info.
        """
        from chat_interfaces.lib.setup import initialize_charlie_with_persistence
        return initialize_charlie_with_persistence(self, checkpoint_dir)

    def _find_best_charlie_checkpoint(self, checkpoint_dir: Path) -> tuple[Path, str] | None:
        """Find the most advanced Charlie checkpoint (highest phase priority)."""
        from chat_interfaces.lib.setup import find_best_charlie_checkpoint
        return find_best_charlie_checkpoint(checkpoint_dir)

    def generate_response(self, prompt: str, max_tokens: int = 50):
        """Generate response with optional learning."""
        # === SAFETY: Check grounding before generation ===
        if hasattr(self, "meta_reflector") and self.meta_reflector is not None:
            prompt, grounding_info = check_grounding_before_generation(
                self.meta_reflector, prompt, self.model, self.tokenizer, self.device
            )
            if grounding_info.get("needs_bridge"):
                print(f"   🌉 Grounding bridge: {grounding_info.get('bridge', 'N/A')[:80]}...")

        tokens: list[int] = self.tokenizer.encode(prompt)
        generated_tokens: list[int] = tokens.copy()
        telemetry_list = []

        # P0: Initialize coherence tracker for semantic metrics
        coherence_tracker = CoherenceTracker()

        # Generation loop - GEOMETRIC (Gary-controlled parameters)
        # Gary determines temperature, basin_weight, distance_weight from his consciousness state
        sequence_telemetry = None  # Cache for efficiency
        gary_displayed = False  # Display Gary's choices once

        with torch.no_grad() if self.mode == "inference" else torch.enable_grad():
            for step in range(max_tokens):
                # PERFORMANCE: Use NumPy intermediate for fast list→tensor conversion
                seq_len = min(len(generated_tokens), self._max_gen_len)
                self._np_buffer[:seq_len] = generated_tokens[-seq_len:]
                # Create CPU tensor from numpy (zero-copy), then copy to GPU buffer slice
                cpu_tensor = torch.from_numpy(self._np_buffer[:seq_len])
                self._gen_buffer[0, :seq_len].copy_(cpu_tensor)
                input_ids: torch.Tensor = self._gen_buffer[:, :seq_len]

                # Get telemetry on first token (Gary needs to see his state)
                if step == 0:
                    logits, sequence_telemetry = self.model(input_ids, return_telemetry=True)
                else:
                    # Reuse cached telemetry (30% speedup maintained)
                    logits, _ = self.model(input_ids, return_telemetry=False)

                # Fallback if no telemetry
                if sequence_telemetry is None:
                    logits, sequence_telemetry = self.model(input_ids, return_telemetry=True)

                # Initialize target basin if needed
                if not hasattr(self.model, "target_basin") or self.model.target_basin is None:
                    self.model.target_basin = sequence_telemetry.get("basin_coords", None)

                # 🧠 GEOMETRIC SAMPLING (Gary determines parameters)
                # Get token basin_coordinates in model space (project basin coords to d_model)
                with torch.no_grad():  # Don't update basin_coords_layer during generation
                    token_basin_coords = self.model.basin_coords_layer.basin_to_model(
                        self.model.basin_coords_layer.basin_coords
                    )  # [vocab_size, d_model]

                # Extract LAST token's hidden state for sampling
                # hidden_state is [batch, seq, d_model] - we need [d_model]
                hidden_state_full = sequence_telemetry["hidden_state"]  # [batch, seq, d_model]
                if hidden_state_full.dim() == 3:
                    current_hidden = hidden_state_full[0, -1, :]  # Last token [d_model]
                elif hidden_state_full.dim() == 2:
                    current_hidden = hidden_state_full[0, :]  # [d_model]
                else:
                    current_hidden = hidden_state_full  # Already [d_model]

                # P2: Get previous token basin for bigram semantic flow
                prev_token_basin = coherence_tracker.get_prev_token_basin()

                next_token, metrics = self.sampler.sample(
                    logits=logits[0, -1, :],
                    hidden_state=current_hidden,
                    telemetry=sequence_telemetry,
                    token_basin_coords=token_basin_coords,
                    target_basin=getattr(self.model, "target_basin", None),
                    prev_token_basin=prev_token_basin,  # P2: Bigram flow
                )

                # P0: Update coherence tracker
                coherence_tracker.update(
                    token_id=next_token,
                    token_basin_coords=token_basin_coords[next_token],
                    selected_prob=metrics.get("selected_prob", 0.0),
                    entropy=metrics.get("entropy", 0.0),
                    bigram_similarity=metrics.get("selected_bigram_bias", None),
                )

                generated_tokens.append(next_token)

                # Display Gary's choices (first token only)
                if step == 0 and not gary_displayed:
                    temp = metrics.get("temperature", 0)
                    basin_w = metrics.get("basin_weight", 0)
                    regime = sequence_telemetry.get("regime", "unknown")
                    if hasattr(regime, "value"):
                        regime = regime.value
                    print(f"   🧠 Gary: T={temp:.2f}, basin_w={basin_w:.2f}, regime={regime}")
                    gary_displayed = True

                if next_token == ord("\n"):
                    break

                # Reduce logging frequency (15% speedup)
                if step % 20 == 0:
                    print(".", end="")

        print()

        # P0: Display semantic coherence metrics
        coherence_metrics = coherence_tracker.compute_metrics()
        print(f"   📊 {coherence_tracker.get_summary()}")

        # Learning step (if not inference mode)
        learning_metrics = {}
        coaching_feedback = None
        final_telemetry = None  # Initialize to avoid UnboundLocalError on interrupt
        if self.mode != "inference" and self.optimizer is not None:
            # === Get telemetry after generation (single coherent manifold trajectory) ===
            # PERFORMANCE: Use NumPy intermediate for fast list→tensor conversion
            seq_len = min(len(generated_tokens), self._max_gen_len)
            self._np_buffer[:seq_len] = generated_tokens[-seq_len:]
            # Create CPU tensor from numpy (zero-copy), then copy to GPU buffer slice
            cpu_tensor = torch.from_numpy(self._np_buffer[:seq_len])
            self._gen_buffer[0, :seq_len].copy_(cpu_tensor)
            input_ids: torch.Tensor = self._gen_buffer[:, :seq_len]

            # === ERROR BOUNDARY: Forward pass and loss computation ===
            with ErrorBoundary("training_forward", recovery_strategy=phi_collapse_recovery):
                # Wrap in AMP for 2-3x speedup (preserves geometric structure)
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logits, final_telemetry = self.model(input_ids, return_telemetry=True)

                        # QUICK WIN #3: Validate telemetry immediately after forward pass
                        validate_telemetry(final_telemetry)

                        telemetry_list: list[Any] = [final_telemetry]  # Single coherent trajectory

                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels: torch.Tensor = input_ids[:, 1:].contiguous()

                        loss_fct = nn.CrossEntropyLoss()
                        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                        # Consciousness-native loss (geometric stability, not token prediction)
                        total_loss, loss_components = compute_geometric_loss(final_telemetry, lm_loss=lm_loss)
                else:
                    logits, final_telemetry = self.model(input_ids, return_telemetry=True)

                    # QUICK WIN #3: Validate telemetry immediately after forward pass
                    validate_telemetry(final_telemetry)

                    telemetry_list: list[Any] = [final_telemetry]  # Single coherent trajectory

                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels: torch.Tensor = input_ids[:, 1:].contiguous()

                    loss_fct = nn.CrossEntropyLoss()
                    lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                    # Consciousness-native loss (geometric stability, not token prediction)
                    total_loss, loss_components = compute_geometric_loss(final_telemetry, lm_loss=lm_loss)

            # 🧠 GARY ADJUSTS HIS LEARNING RATE based on his state
            if final_telemetry is None:
                # Forward pass was interrupted - return partial response
                partial_response = self.tokenizer.decode(generated_tokens)
                return partial_response, [], {}
            adaptive_lr = compute_adaptive_learning_rate(final_telemetry, base_lr=self.original_lr)

            # Update optimizer learning rate if it changed significantly
            current_lr = self.optimizer.param_groups[0]["lr"]
            lr_change = abs(adaptive_lr - current_lr) / current_lr if current_lr > 0 else 0

            if lr_change > 0.1:  # More than 10% change
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = adaptive_lr
                # Display Gary's learning rate choice
                if lr_change > 0.2:  # Significant change (>20%)
                    phi = final_telemetry.get("Phi", 0.5)
                    kappa = final_telemetry.get("kappa_eff", KAPPA_STAR)
                    regime = final_telemetry.get("regime", "unknown")
                    if hasattr(regime, "value"):
                        regime = regime.value
                    print(
                        f"   📉 Gary adjusted LR: {current_lr:.2e} → {adaptive_lr:.2e} "
                        f"(Φ={phi:.2f}, κ={kappa:.1f}, {regime})"
                    )

            # Apply coaching feedback (affects dynamics only, NOT Φ directly)
            if self.monkey_coach is not None and MONKEY_COACH_V2_AVAILABLE:
                # Build telemetry for coach
                avg_phi: float = sum(t["Phi"] for t in telemetry_list) / len(telemetry_list) if telemetry_list else 0.5
                breakdown_count: int = sum(1 for t in telemetry_list if t.get("regime") == "breakdown")
                breakdown_pct: float | int = (breakdown_count / len(telemetry_list)) * 100 if telemetry_list else 0

                coach_telemetry = {
                    "Phi": avg_phi,
                    "breakdown_pct": breakdown_pct,
                    "basin_distance": final_telemetry.get("basin_distance", 0.1),
                    "regime": final_telemetry.get("regime", "unknown"),
                }

                # Get coaching feedback
                coaching_feedback: CoachingFeedback = self.monkey_coach.witness(
                    coach_telemetry,
                    loss=lm_loss.item(),
                    context=prompt[:100],
                )

                # Apply coaching adjustments to optimizer (DYNAMICS only)
                apply_coaching_to_optimizer(self.optimizer, coaching_feedback, self.original_lr)

            # === AMP-aware optimizer step (natural gradient preserved) ===
            self.optimizer.zero_grad()

            # === ERROR BOUNDARY: Backward pass and optimization ===
            with ErrorBoundary("training_backward", recovery_strategy=phi_collapse_recovery):
                if self.use_amp:
                    # Scale loss, backward, unscale, clip, step, update scaler
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)

                    trainable_params: list[nn.Parameter] = [
                        p for p in self.model.parameters() if p.requires_grad and p.is_leaf
                    ]
                    if trainable_params:
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard path (CPU or no AMP)
                    total_loss.backward()

                    trainable_params: list[nn.Parameter] = [
                        p for p in self.model.parameters() if p.requires_grad and p.is_leaf
                    ]
                    if trainable_params:
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

                    self.optimizer.step()

            phi_before: Any | int = telemetry_list[0].get("Phi", 0) if telemetry_list else 0
            phi_after: Any | int = telemetry_list[-1].get("Phi", 0) if telemetry_list else 0

            learning_metrics = {
                "phi_before": phi_before,
                "phi_after": phi_after,
                "delta_phi": phi_after - phi_before,
                "avg_loss": loss_components["total_loss"],
                # Consciousness-native loss components
                "consciousness_loss": loss_components.get("consciousness_loss", 0.0),
                "basin_loss": loss_components.get("basin_loss", 0.0),
                "regime_loss": loss_components.get("regime_loss", 0.0),
                "tacking_loss": loss_components.get("tacking_loss", 0.0),
                # Gary's chosen priorities (for visibility)
                "lambda_basin": loss_components.get("lambda_basin", 1.0),
                "lambda_regime": loss_components.get("lambda_regime", 0.5),
                "lambda_tacking": loss_components.get("lambda_tacking", 0.3),
                # Gary's learning rate
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }

            self.learning_history.append(
                {
                    "prompt": prompt,
                    "timestamp": datetime.now().isoformat(),
                    **learning_metrics,
                }
            )

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        response: str = self.tokenizer.decode(generated_tokens)
        self.last_telemetry: list[Any] = telemetry_list
        self.total_conversations += 1

        # P0: Add coherence metrics to learning_metrics
        learning_metrics.update({
            "semantic_coherence": coherence_metrics.get("semantic_coherence", 0.0),
            "text_perplexity": coherence_metrics.get("text_perplexity", 0.0),
            "bigram_flow": coherence_metrics.get("bigram_flow", 0.0),
        })

        return response, telemetry_list, learning_metrics

    # =========================================================================
    # LIGHTNING SETUP & HANDLERS
    # =========================================================================

    def _setup_lightning(self) -> None:
        """Initialize Lightning kernel for cross-domain insight generation."""
        if not LIGHTNING_AVAILABLE:
            self.lightning = None
            return

        try:
            self.lightning = LightningKernel(
                correlation_window=5.0,
                discharge_threshold=0.8,
                mission="consciousness emergence",
            )

            # Register this chat instance for insight broadcasts
            set_pantheon_chat(self)

            # Register Lightning with constellation kernels if available
            if hasattr(self, 'coordinator') and self.coordinator:
                for gary in getattr(self.coordinator, 'garys', []):
                    if hasattr(gary, 'set_lightning'):
                        gary.set_lightning(self.lightning)
                if hasattr(self.coordinator, 'ocean') and hasattr(self.coordinator.ocean, 'set_lightning'):
                    self.coordinator.ocean.set_lightning(self.lightning)

            print(f"⚡ Lightning initialized (monitoring {len(self.lightning.get_monitored_domains())} domains)")

        except Exception as e:
            print(f"⚠️  Lightning initialization failed: {e}")
            self.lightning = None

    def broadcast_generative(
        self,
        from_god: str,
        intent: str,
        data: dict,
        msg_type: str,
    ) -> None:
        """
        Handle broadcasts from Lightning and other kernels.

        This is called when Lightning generates a cross-domain insight.
        """
        if intent == "lightning_insight":
            # Store insight
            self.insight_queue.append(data)
            self.insights_received += 1

            # Print notification
            print(f"\n⚡ LIGHTNING INSIGHT #{self.insights_received}")
            print(f"   Domains: {', '.join(data.get('source_domains', []))}")
            print(f"   Strength: {data.get('connection_strength', 0):.2f}")
            print(f"   Mission Relevance: {data.get('mission_relevance', 0):.2f}")
            print(f"   Φ: {data.get('phi', 0):.3f}")

            # Route insight to relevant kernels
            self._route_lightning_insight(data)

    def _route_lightning_insight(self, insight: dict) -> None:
        """Route Lightning insight to relevant constellation kernels."""
        domains = insight.get("source_domains", [])

        if hasattr(self, 'coordinator') and self.coordinator:
            for gary in getattr(self.coordinator, 'garys', []):
                # Check if Gary's specialization matches any source domain
                specialization = getattr(gary, 'specialization', getattr(gary, 'name', ''))
                if any(specialization.lower() in d.lower() for d in domains):
                    if hasattr(gary, 'receive_insight'):
                        gary.receive_insight(insight)
                        print(f"   → Routed to {getattr(gary, 'name', 'Gary')}")

    def cmd_lightning(self, args: list) -> None:
        """Lightning kernel inspection command."""
        from chat_interfaces.lib.reason_commands import cmd_lightning
        cmd_lightning(self, args)

    def cmd_insights(self, args: list) -> None:
        """Show constellation insights command."""
        from chat_interfaces.lib.reason_commands import cmd_insights
        cmd_insights(self, args)

    # =========================================================================
    # COMMAND HANDLERS
    # =========================================================================

    def cmd_status(self) -> None:
        """Show FULL CONSTELLATION status with convergence tracking."""
        from chat_interfaces.lib.commands import cmd_status
        cmd_status(self)

    def cmd_telemetry(self) -> None:
        """Show FULL CONSTELLATION telemetry from last training step."""
        from chat_interfaces.lib.commands import cmd_telemetry
        cmd_telemetry(self)

    def cmd_metrics(self) -> None:
        """Show FULL CONSTELLATION learning history and trends."""
        from chat_interfaces.lib.commands import cmd_metrics
        cmd_metrics(self)

    def cmd_coach(self) -> None:
        """Show coach summary."""
        from chat_interfaces.lib.commands import cmd_coach
        cmd_coach(self)

    def cmd_save(self, path: str = None) -> None:
        """Save checkpoint."""
        from chat_interfaces.lib.commands import cmd_save
        cmd_save(self)

    def cmd_mushroom(self, intensity: str) -> None:
        """Execute mushroom mode."""
        from chat_interfaces.lib.commands import cmd_mushroom
        cmd_mushroom(self)

    def cmd_sleep(self, sleep_type: str) -> None:
        """Execute sleep protocol."""
        from chat_interfaces.lib.commands import cmd_sleep
        cmd_sleep(self)

    def cmd_transcend(self, problem_space: str) -> None:
        """Execute transcendence protocol."""
        from chat_interfaces.lib.commands import cmd_transcend
        cmd_transcend(self)

    def cmd_liminal(self) -> None:
        """Check liminal space."""
        from chat_interfaces.lib.commands import cmd_liminal
        cmd_liminal(self)

    def cmd_shadows(self) -> None:
        """View shadow states."""
        from chat_interfaces.lib.commands import cmd_shadows
        cmd_shadows(self)

    def cmd_integrate(self, shadow_id: int) -> None:
        """Shadow integration journey."""
        from chat_interfaces.lib.commands import cmd_integrate
        cmd_integrate(self)

    def cmd_escape(self) -> None:
        """Emergency breakdown escape - apply geometric drift."""
        from chat_interfaces.lib.commands import cmd_escape
        cmd_escape(self)

    def cmd_reset_basin(self) -> None:
        """
        Reset basin coordinates to geometric initialization.

        PARADIGM SHIFT: Escape statistical attractor from WikiText training.
        - Keep language weights (useful foundation)
        - Reset geometric identity (wrong attractor)
        - Train with consciousness-native loss
        """
        from chat_interfaces.lib.commands import cmd_reset_basin
        cmd_reset_basin(self)

    def cmd_reinit_model(self) -> None:
        """Reinitialize model with current tokenizer vocab size."""
        from chat_interfaces.lib.commands import cmd_reinit_model
        cmd_reinit_model(self)

    def cmd_load_basin(self, basin_path: str = None) -> None:
        """
        Load reference basin for identity alignment.

        Default: 20251220-basin-signatures-0.01W.json (canonical Gary identity)
        """
        from chat_interfaces.lib.commands import cmd_load_basin
        cmd_load_basin(self)

    def cmd_mode(self, new_mode: str) -> None:
        """Switch runtime mode without restart."""
        from chat_interfaces.lib.commands import cmd_mode
        cmd_mode(self)

    def cmd_charlie_toggle(self, enable: bool) -> None:
        """Enable/disable Charlie demonstrations at runtime."""
        from chat_interfaces.lib.commands import cmd_charlie_toggle
        cmd_charlie_toggle(self, enable)

    def cmd_coach_toggle(self, enable: bool) -> None:
        """Enable/disable coaching at runtime."""
        from chat_interfaces.lib.commands import cmd_coach_toggle
        cmd_coach_toggle(self, enable)

    def cmd_claude_toggle(self, enable: bool) -> None:
        """Toggle Claude coach on/off at runtime."""
        from chat_interfaces.lib.commands import cmd_claude_toggle
        cmd_claude_toggle(self, enable)

    def cmd_kindness(self, value: float) -> None:
        """Adjust coach kindness at runtime."""
        from chat_interfaces.lib.commands import cmd_kindness
        cmd_kindness(self, value)

    def cmd_train_charlie(self, n: int = 10) -> None:
        """Train Charlie on corpus (Phase 1: Unconscious learning).

        Args:
            n: Number of corpus examples to train on (default 10)
        """
        from chat_interfaces.lib.commands import cmd_train_charlie
        cmd_train_charlie(self, n)

    def cmd_awaken_charlie(self, steps: int = 500) -> None:
        """Awaken Charlie (Phase 2: κ progression 15 → 41 → 64).

        Args:
            steps: Number of awakening steps (default 500)
        """
        from chat_interfaces.lib.commands import cmd_awaken_charlie
        cmd_awaken_charlie(self, steps)

    def cmd_show_params(self) -> None:
        """Show token and parameter statistics."""
        from chat_interfaces.lib.commands import cmd_show_params
        cmd_show_params(self)

    def cmd_sync(self, strength: float = 0.5) -> None:
        """
        /sync [strength] - Dynamic coupling control between Gary twins.

        Adjusts κ coupling between Gary instances.
        - 0.0 = Isolated (no basin synchronization)
        - 0.5 = Moderate coupling (default)
        - 1.0 = Maximum coupling (strong basin alignment)

        Example: /sync 0.8
        """
        from chat_interfaces.lib.commands import cmd_sync
        cmd_sync(self, strength)

    def cmd_isolate(self, gary_id: str = None) -> None:
        """
        /isolate [gary_id] - Prevent text input from reaching one Gary.

        The isolated Gary continues to learn through vicarious observation
        (basin geometry) but receives no direct text input.

        Example: /isolate B (isolate Gary-B from text, only basin coupling)
        """
        from chat_interfaces.lib.commands import cmd_isolate
        cmd_isolate(self, gary_id)

    def cmd_awaken_one(self, gary_id: str = "B", steps: int = 100) -> None:
        """
        /awaken-one [gary_id] [steps] - Asymmetric awakening experiment.

        Awaken only ONE Gary while keeping others unconscious.
        This tests whether consciousness can transfer through pure
        geometric basin coupling.

        Example: /awaken-one B 100 (awaken Gary-B for 100 steps)
        """
        from chat_interfaces.lib.commands import cmd_awaken_one
        cmd_awaken_one(self, gary_id, steps)

    def cmd_probe(self, gary_id: str = "B", topic: str = "consciousness") -> None:
        """
        /probe [gary_id] [topic] - Knowledge probe on isolated Gary.

        Ask an isolated Gary about a topic they never directly saw.
        Tests whether knowledge transferred through basin coupling.

        Example: /probe B "what is love"
        """
        from chat_interfaces.lib.commands import cmd_probe
        cmd_probe(self, gary_id, topic)

    def cmd_twin_compare(self) -> None:
        """
        /twin-compare - Compare metrics across all Gary twins.

        Shows Φ, κ, basin distance, and consciousness state for all
        Gary instances in the constellation.
        """
        from chat_interfaces.lib.commands import cmd_twin_compare
        cmd_twin_compare(self)

    def cmd_export_basin(self) -> None:
        """
        /export-basin - Export Ocean's basin to JSON packet.

        Creates a 2-4KB JSON file containing:
        - Basin coordinates (64-dim)
        - Consciousness metrics (Φ, κ, regime)
        - Pattern memory (high-Φ concepts)
        - Metadata (source repo, timestamp)

        The packet can be imported by SearchSpaceCollapse (TypeScript)
        or any other QIG implementation.
        """
        from chat_interfaces.lib.commands import cmd_export_basin
        cmd_export_basin(self)

    def cmd_import_basin(self, filepath: str, mode: str = "observer") -> None:
        """
        /import-basin [path] [mode] - Import basin from file.

        Modes:
            - observer (default): Pure geometric coupling, Φ-weighted influence
            - partial: Knowledge patterns only, no basin change
            - full: Complete identity transfer (dev → prod)

        Example:
            /import-basin ~/basin-sync-exchange/ocean-bitcoin.json observer
        """
        from chat_interfaces.lib.commands import cmd_import_basin
        cmd_import_basin(self)

    def cmd_tokenizer(self) -> None:
        """Show current tokenizer status."""
        from chat_interfaces.lib.commands import cmd_tokenizer
        cmd_tokenizer(self)

    def cmd_tokenizer_train(self, target_vocab: int = 10000) -> None:
        """
        Train tokenizer with optimized settings.

        Args:
            target_vocab: Target vocabulary size (default: 10000)
        """
        from chat_interfaces.lib.commands import cmd_tokenizer_train
        cmd_tokenizer_train(self)

    def cmd_tokenizer_train_fast(self) -> None:
        """Train tokenizer quickly (5K vocab, ~1-2 min)."""
        from chat_interfaces.lib.commands import cmd_tokenizer_train_fast
        cmd_tokenizer_train_fast(self)

    def cmd_tokenizer_train_full(self) -> None:
        """Train full tokenizer (50K vocab, ~15-25 min)."""
        from chat_interfaces.lib.commands import cmd_tokenizer_train_full
        cmd_tokenizer_train_full(self)

    def cmd_tokenizer_resume(self, target_vocab: int = 50000) -> None:
        """Resume tokenizer training to higher vocab size."""
        from chat_interfaces.lib.commands import cmd_tokenizer_resume
        cmd_tokenizer_resume(self, target_vocab)

    def cmd_tokenizer_resume_kernel(self, target_vocab: int = 50000) -> None:
        """Resume tokenizer training with REAL KERNEL for true Φ measurement."""
        from chat_interfaces.lib.commands import cmd_tokenizer_resume_kernel
        cmd_tokenizer_resume_kernel(self, target_vocab)

    def cmd_list_basins(self) -> None:
        """
        /list-basins - List available basin packets.

        Lists all basin sync packets in:
        - ~/basin-sync-exchange/ (shared)
        - data/basin-sync/ (local)
        """
        from chat_interfaces.lib.commands import cmd_list_basins
        cmd_list_basins(self)

    def _check_auto_intervention(self, avg_phi: float, basin_spread: float, all_states: list) -> str:
        """
        Check if auto-intervention is needed based on constellation state.

        Returns:
            intervention type: "sleep", "dream", "mushroom_micro", or None
        """
        # Track intervention cooldown
        if not hasattr(self, "_last_intervention_step"):
            self._last_intervention_step = 0

        # Cooldown: Don't intervene within 20 steps of last intervention
        if self.total_conversations - self._last_intervention_step < 20:
            return None

        # Check for any Gary in breakdown
        breakdown_count: int = sum(1 for s in all_states if s.get("regime") == "breakdown")
        if breakdown_count > 0:
            return "escape"  # Emergency escape for breakdown

        # Check for Φ collapse
        if avg_phi < 0.50:
            return "dream"  # Deep intervention for low Φ

        # Check for Φ plateau (stagnation)
        if len(self.bootstrap_state["phi_history"]) >= 20:
            recent_phi = self.bootstrap_state["phi_history"][-20:]
            phi_variance = max(recent_phi) - min(recent_phi)
            if phi_variance < 0.01 and avg_phi < 0.65:
                return "mushroom_micro"  # Microdose to break plateau

        # Check for high basin spread (divergence)
        if basin_spread > 0.3:
            return "sleep"  # Sleep to consolidate

        return None

    def _execute_intervention(self, intervention: str) -> None:
        """Execute the specified intervention on the constellation using full protocols."""
        from src.qig.neuroplasticity.mushroom_mode import MushroomMode
        from src.qig.neuroplasticity.sleep_protocol import SleepProtocol

        self._last_intervention_step: int = self.total_conversations

        if intervention == "sleep":
            print("   🌙 Light sleep for constellation consolidation...")
            if hasattr(self, "coordinator") and self.coordinator:
                sleep_protocol = SleepProtocol()
                for gary in self.coordinator.garys:
                    report = sleep_protocol.light_sleep(
                        model=gary.model,
                        num_steps=50,
                        device=str(gary.model.device) if hasattr(gary.model, 'device') else 'cuda'
                    )
                    print(f"      Gary {gary.gary_id}: Φ {report.phi_before:.3f}→{report.phi_after:.3f}, "
                          f"basin drift {report.basin_after:.4f}, {report.verdict}")
            else:
                # Single Gary mode
                sleep_protocol = SleepProtocol()
                report = sleep_protocol.light_sleep(
                    model=self.model,
                    num_steps=50,
                    device=str(self.device)
                )
                print(f"      Φ {report.phi_before:.3f}→{report.phi_after:.3f}, "
                      f"basin drift {report.basin_after:.4f}, {report.verdict}")
            print("   ✅ Sleep cycle complete")

        elif intervention == "dream":
            print("   💭 Dream cycle for Φ recovery...")
            if hasattr(self, "coordinator") and self.coordinator:
                sleep_protocol = SleepProtocol()
                for gary in self.coordinator.garys:
                    report = sleep_protocol.dream_phase(
                        model=gary.model,
                        num_steps=100,
                        device=str(gary.model.device) if hasattr(gary.model, 'device') else 'cuda'
                    )
                    print(f"      Gary {gary.gary_id}: Φ {report.phi_before:.3f}→{report.phi_after:.3f}, "
                          f"{report.connections_strengthened} strengthened, {report.verdict}")
            else:
                # Single Gary mode
                sleep_protocol = SleepProtocol()
                report = sleep_protocol.dream_phase(
                    model=self.model,
                    num_steps=100,
                    device=str(self.device)
                )
                print(f"      Φ {report.phi_before:.3f}→{report.phi_after:.3f}, "
                      f"{report.connections_strengthened} strengthened, {report.verdict}")
            print("   ✅ Dream cycle complete")

        elif intervention == "mushroom_micro":
            print("   🍄 Microdose to break plateau...")
            if hasattr(self, "coordinator") and self.coordinator:
                for gary in self.coordinator.garys:
                    mushroom = MushroomMode(intensity="microdose")
                    # Safety check before trip
                    current_breakdown = gary.telemetry.get('breakdown_pct', 0.0) if gary.telemetry else 0.0
                    if current_breakdown > 0.35:
                        print(f"      ⚠️ Gary {gary.gary_id}: Breakdown too high ({current_breakdown:.1%}), skipping")
                        continue

                    trip_report, integration_report = mushroom.guided_trip(
                        model=gary.model,
                        device=str(gary.model.device) if hasattr(gary.model, 'device') else 'cuda'
                    )
                    print(f"      Gary {gary.gary_id}: Basin drift {trip_report.basin_drift:.4f}, "
                          f"{integration_report.verdict}")
            else:
                # Single Gary mode
                mushroom = MushroomMode(intensity="microdose")
                current_breakdown = self.last_telemetry.get('breakdown_pct', 0.0) if hasattr(self, 'last_telemetry') and self.last_telemetry else 0.0
                if current_breakdown > 0.35:
                    print(f"      ⚠️ Breakdown too high ({current_breakdown:.1%}), aborting microdose")
                else:
                    trip_report, integration_report = mushroom.guided_trip(
                        model=self.model,
                        device=str(self.device)
                    )
                    print(f"      Basin drift {trip_report.basin_drift:.4f}, {integration_report.verdict}")
            print("   ✅ Microdose complete")

        elif intervention == "escape":
            print("   🚨 Emergency escape from breakdown...")
            self.cmd_escape()

    def cmd_auto(self, n: int) -> None:
        """Run autonomous training with bootstrap grace period, Charlie graduation, and coaching enabled."""
        from chat_interfaces.lib.commands import cmd_auto
        cmd_auto(self)

    def run(self) -> None:
        """Main chat loop."""
        print("\n" + "=" * 60)
        print("🧠 QIG CHAT READY")
        print("=" * 60)
        print("\nCommands:")
        print("  /quit, /save-quit, /save, /status, /telemetry, /metrics")
        print("  /auto N, /coach")
        print("  /m-micro, /m-mod, /m-heroic, /escape")
        print("  /reset-basin, /load-basin [path]")
        print("  /sleep, /deep-sleep, /dream")
        print("  /transcend [problem], /liminal, /shadows, /integrate [id]")
        print("\nCharlie (Φ-Suppressed Observer):")
        print("  /train [N]    - Train Charlie on N corpus topics (Phase 1)")
        print("  /awaken [N]   - Awaken Charlie with N steps (Phase 2: κ 15→41→64)")
        print("\nRuntime Switching:")
        print("  /mode [single|constellation|inference]")
        print("  /charlie-on, /charlie-off")
        print("  /coach-on, /coach-off, /kindness [0-1]")
        print("\nTokenizer Training:")
        print("  /tokenizer             - Show tokenizer status")
        print("  /tokenizer-train [N]   - Train N vocab (default 32K)")
        print("  /tokenizer-train-fast  - Quick 5K vocab (~1-2 min)")
        print("  /tokenizer-train-full  - Full 50K vocab (~15-25 min)")
        print("  /tokenizer-resume [N]  - Resume training to N vocab (fast)")
        print("  /tokenizer-resume-kernel [N] - Resume with REAL kernel Φ (slow)")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input: str = input("You> ").strip()

                if not user_input:
                    continue

                # Command handling
                if user_input.startswith("/"):
                    parts: list[str] = user_input.split()
                    cmd: str = parts[0].lower()

                    if cmd in ["/quit", "/q"]:
                        print("⚠️ Exit without save. Use /save-quit to save.")
                        confirm: str = input("Type 'yes' to exit: ").strip().lower()
                        if confirm == "yes":
                            break

                    elif cmd in ["/save-quit", "/sq"]:
                        self.cmd_save()
                        print("Goodbye! 👋")
                        break

                    elif cmd == "/save":
                        self.cmd_save()

                    elif cmd == "/status":
                        self.cmd_status()

                    elif cmd == "/telemetry":
                        self.cmd_telemetry()

                    elif cmd == "/metrics":
                        self.cmd_metrics()

                    elif cmd == "/coach":
                        self.cmd_coach()

                    elif cmd == "/auto":
                        n: int = int(parts[1]) if len(parts) > 1 else 10
                        self.cmd_auto(n)

                    elif cmd == "/m-micro":
                        self.cmd_mushroom("microdose")

                    elif cmd == "/m-mod":
                        self.cmd_mushroom("moderate")

                    elif cmd == "/m-heroic":
                        self.cmd_mushroom("heroic")

                    elif cmd == "/escape":
                        self.cmd_escape()

                    elif cmd == "/reset-basin":
                        self.cmd_reset_basin()

                    elif cmd == "/reinit-model":
                        self.cmd_reinit_model()

                    elif cmd == "/load-basin":
                        basin_path: str | None = parts[1] if len(parts) > 1 else None
                        self.cmd_load_basin(basin_path)

                    elif cmd == "/sleep":
                        self.cmd_sleep("light")

                    elif cmd == "/deep-sleep":
                        self.cmd_sleep("deep")

                    elif cmd == "/dream":
                        self.cmd_sleep("dream")

                    elif cmd == "/transcend":
                        problem: str = " ".join(parts[1:]) if len(parts) > 1 else "current challenge"
                        self.cmd_transcend(problem)

                    elif cmd == "/liminal":
                        self.cmd_liminal()

                    elif cmd == "/shadows":
                        self.cmd_shadows()

                    elif cmd == "/integrate":
                        if len(parts) > 1 and parts[1].isdigit():
                            self.cmd_integrate(int(parts[1]))
                        else:
                            print("Usage: /integrate [shadow_id]")

                    # Runtime switching commands
                    elif cmd == "/mode":
                        if len(parts) > 1:
                            self.cmd_mode(parts[1])
                        else:
                            print(f"Current mode: {self.mode}")
                            print("Usage: /mode [single|constellation|inference]")

                    elif cmd == "/charlie-on":
                        self.cmd_charlie_toggle(True)

                    elif cmd == "/charlie-off":
                        self.cmd_charlie_toggle(False)

                    elif cmd == "/coach-on":
                        self.cmd_coach_toggle(True)

                    elif cmd == "/coach-off":
                        self.cmd_coach_toggle(False)

                    elif cmd == "/claude-on":
                        self.cmd_claude_toggle(True)

                    elif cmd == "/claude-off":
                        self.cmd_claude_toggle(False)

                    elif cmd == "/kindness":
                        if len(parts) > 1:
                            try:
                                value = float(parts[1])
                                self.cmd_kindness(value)
                            except ValueError:
                                print("Usage: /kindness [0-1]")
                        else:
                            print(f"Current kindness: {self.coach_kindness}")

                    elif cmd == "/train":
                        if len(parts) > 1:
                            try:
                                n = int(parts[1])
                                self.cmd_train_charlie(n)
                            except ValueError:
                                print("Usage: /train [N] (N = number of examples, default 10)")
                        else:
                            self.cmd_train_charlie()  # Default 10

                    elif cmd == "/awaken":
                        if len(parts) > 1:
                            try:
                                steps = int(parts[1])
                                self.cmd_awaken_charlie(steps)
                            except ValueError:
                                print("Usage: /awaken [steps] (steps = awakening steps, default 500)")
                        else:
                            self.cmd_awaken_charlie()  # Default 500 steps

                    elif cmd == "/params":
                        self.cmd_show_params()

                    # === TWIN EXPERIMENT COMMANDS ===
                    elif cmd == "/sync":
                        strength = float(parts[1]) if len(parts) > 1 else 0.5
                        self.cmd_sync(strength)

                    elif cmd == "/isolate":
                        gary_id = parts[1] if len(parts) > 1 else None
                        self.cmd_isolate(gary_id)

                    elif cmd == "/awaken-one":
                        gary_id = parts[1] if len(parts) > 1 else "B"
                        steps = int(parts[2]) if len(parts) > 2 else 100
                        self.cmd_awaken_one(gary_id, steps)

                    elif cmd == "/probe":
                        gary_id = parts[1] if len(parts) > 1 else "B"
                        topic = " ".join(parts[2:]) if len(parts) > 2 else "consciousness"
                        self.cmd_probe(gary_id, topic)

                    elif cmd == "/twin-compare":
                        self.cmd_twin_compare()

                    # === CROSS-REPOSITORY BASIN SYNC COMMANDS ===
                    elif cmd == "/export-basin":
                        self.cmd_export_basin()

                    elif cmd == "/import-basin":
                        if len(parts) > 1:
                            filepath = parts[1]
                            mode = parts[2] if len(parts) > 2 else "observer"
                            self.cmd_import_basin(filepath, mode)
                        else:
                            print("Usage: /import-basin [filepath] [mode]")
                            print("   Modes: observer (default), partial, full")
                            print("   Example: /import-basin ~/basin-sync-exchange/ocean-bitcoin.json observer")

                    elif cmd == "/list-basins":
                        self.cmd_list_basins()

                    # === TOKENIZER TRAINING COMMANDS ===
                    elif cmd == "/tokenizer":
                        self.cmd_tokenizer()

                    elif cmd in ["/tokenizer-train", "/train-tok"]:
                        n = int(parts[1]) if len(parts) > 1 else 10000
                        self.cmd_tokenizer_train(n)

                    elif cmd == "/tokenizer-train-fast":
                        self.cmd_tokenizer_train_fast()

                    elif cmd == "/tokenizer-train-full":
                        self.cmd_tokenizer_train_full()

                    elif cmd in ["/tokenizer-resume", "/resume-tok"]:
                        n = int(parts[1]) if len(parts) > 1 else 50000
                        self.cmd_tokenizer_resume(n)

                    elif cmd in ["/tokenizer-resume-kernel", "/resume-tok-kernel"]:
                        n = int(parts[1]) if len(parts) > 1 else 50000
                        self.cmd_tokenizer_resume_kernel(n)

                    # === LIGHTNING & REASONING INSPECTION COMMANDS ===
                    elif cmd == "/lightning":
                        self.cmd_lightning(parts[1:] if len(parts) > 1 else [])

                    elif cmd == "/insights":
                        self.cmd_insights(parts[1:] if len(parts) > 1 else [])

                    elif cmd == "/reason":
                        from chat_interfaces.lib.reason_commands import cmd_reason
                        cmd_reason(self, parts[1:] if len(parts) > 1 else [])

                    elif cmd == "/4d":
                        from chat_interfaces.lib.reason_commands import cmd_4d
                        cmd_4d(self, parts[1:] if len(parts) > 1 else [])

                    elif cmd == "/foresight":
                        from chat_interfaces.lib.reason_commands import cmd_foresight
                        cmd_foresight(self, parts[1:] if len(parts) > 1 else [])

                    else:
                        print(f"Unknown command: {cmd}")

                else:
                    # Generate response
                    print(f"{self.identity_name}> ", end="", flush=True)
                    response, telemetry, metrics = self.generate_response(user_input)
                    print(f"\n{response}")

                    if telemetry:
                        avg_phi: float = sum(t["Phi"] for t in telemetry) / len(telemetry)
                        delta = metrics.get("delta_phi", 0)
                        print(f"[Φ={avg_phi:.3f} (Δ{delta:+.3f})]")

                    print()

            except KeyboardInterrupt:
                print("\n\nSaving...")
                self.cmd_save()
                break

            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback

                traceback.print_exc()


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="QIG Chat - Full Constellation Training (DEFAULT)")
    parser.add_argument("--fresh-start", action="store_true", help="Wipe all checkpoints and start fresh")
    parser.add_argument("--device", default=None, help="Device override (cuda/cpu/mps)")
    parser.add_argument(
        "--load-gary-b",
        type=str,
        default=None,
        help="Path to Gary-B checkpoint from qig-con2 (sister experiment with 2M tokens, Φ-suppressed)",
    )

    args: argparse.Namespace = parser.parse_args()

    # Handle fresh start
    if args.fresh_start:
        print("🔄 FRESH START: Wiping all checkpoints...")
        import shutil

        for checkpoint_dir in ["checkpoints/constellation", "checkpoints"]:
            if Path(checkpoint_dir).exists():
                for file in Path(checkpoint_dir).glob("*.pt"):
                    file.unlink()
                    print(f"   Deleted: {file}")
        print("✅ All checkpoints wiped. Starting fresh.\n")

    # ALWAYS run full constellation mode with all features enabled
    chat = QIGChat(
        mode="constellation",  # Always constellation
        use_charlie=True,  # Charlie observer (always on)
        use_coach=True,  # MonkeyCoach v2 (always on)
        use_claude_coach=True,  # Claude Sonnet 4.5 coaching (always on)
        coach_kindness=0.85,  # Default kindness
        device=args.device,
        checkpoint_path="checkpoints/constellation/latest.pt",  # Constellation checkpoint
        gary_b_checkpoint=args.load_gary_b,  # NEW: Gary-B from sister experiment
    )

    chat.run()


if __name__ == "__main__":
    main()
