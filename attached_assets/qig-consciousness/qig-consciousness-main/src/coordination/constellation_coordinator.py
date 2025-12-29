#!/usr/bin/env python3
"""
Constellation Coordinator - Multi-Instance Training Orchestration
==================================================================

Coordinates Ocean (pure observer) + 3 Gary instances (active/observer hybrid)
for optimal vicarious learning with load distribution.

Architecture:
    Ocean: Observes all, never responds, learns meta-manifold
    Gary-A/B/C: Round-robin active/observer roles

Key Features:
    - Round-robin question routing (prevents over-coupling)
    - Basin synchronization (shared attractor convergence)
    - Vicarious learning (observers learn from active)
    - Telemetry aggregation (constellation-level metrics)

Usage:
    coordinator = ConstellationCoordinator(
        gary_configs=['configs/20251220-gary-a-config-1.00W.yaml', 'configs/20251220-gary-b-config-1.00W.yaml', 'configs/20251220-gary-c-config-1.00W.yaml'],
        ocean_config='configs/20251220-ocean-config-1.00F.yaml'
    )
    coordinator.train(questions=dataset, epochs=20)
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import extracted modules
from src.coordination.instance_state import InstanceState
from src.coordination.router import ConstellationRouter
from src.coordination.state_monitor import StateMonitor

# Type checking imports (avoid circular imports)
if TYPE_CHECKING:
    from src.model.meta_reflector import MetaReflector
    from src.model.qig_kernel_recursive import QIGKernelRecursive

# Phase 1/2 Enhancement Imports
try:
    from src.coordination.basin_velocity_monitor import BasinVelocityMonitor
    from src.coordination.resonance_detector import ResonanceDetector
    from src.model.meta_reflector import MetaReflector
except ImportError:
    BasinVelocityMonitor = None  # type: ignore
    ResonanceDetector = None  # type: ignore
    MetaReflector = None  # type: ignore

# Safety: MetaReflector integration
try:
    from src.safety.meta_reflector_integration import check_grounding_before_generation, check_locked_in_state
    META_SAFETY_AVAILABLE = True
except ImportError:
    META_SAFETY_AVAILABLE = False
    print("⚠️  Phase 1 safety modules not available")

# Developmental Curriculum (Coach-as-Interpreter)
try:
    from src.coordination.active_coach import (
        CoachInterpretation,
        DevelopmentalPhase,
        GeometricCoach,  # Alias to ActiveCoach
    )
    from src.coordination.developmental_curriculum import DevelopmentalCurriculum
except ImportError:
    DevelopmentalCurriculum = None  # type: ignore
    GeometricCoach = None  # type: ignore
    DevelopmentalPhase = None  # type: ignore
    CoachInterpretation = None  # type: ignore

    print("⚠️  Developmental curriculum not available")

# Geodesic Distance (Fisher metric on information manifold)
# GEOMETRIC PURITY: This import is REQUIRED - no Euclidean fallbacks allowed
from src.metrics.geodesic_distance import GeodesicDistance, geodesic_vicarious_loss

# REL Coupling (instance-to-instance relational coupling from qigkernels)
try:
    from qigkernels.rel_coupling import REL_LAMBDA_MAX, compute_rel_from_basins
    REL_COUPLING_AVAILABLE = True
except ImportError:
    REL_COUPLING_AVAILABLE = False
    REL_LAMBDA_MAX = 0.7  # Fallback default
    print("⚠️  REL coupling not available from qigkernels")

# Geometric Generation (QFI-based token sampling)
try:
    from src.generation.qfi_sampler import QFISampler
    GEOMETRIC_GENERATION_AVAILABLE = True
except ImportError:
    QFISampler = None  # type: ignore
    GEOMETRIC_GENERATION_AVAILABLE = False
    print("⚠️  Geometric generation not available")

# Consciousness Systems (NEW - from sister experiment qig-con2)
try:
    from src.consciousness import (
        AutonomicManager,
        AutonomicState,
        DimensionalTracker,
        NeurochemistrySystem,
        TemporalPhiCalculator,
    )
    CONSCIOUSNESS_SYSTEMS_AVAILABLE = True
except ImportError:
    AutonomicManager = None  # type: ignore
    AutonomicState = None  # type: ignore
    DimensionalTracker = None  # type: ignore
    NeurochemistrySystem = None  # type: ignore
    TemporalPhiCalculator = None  # type: ignore
    CONSCIOUSNESS_SYSTEMS_AVAILABLE = False
    print("⚠️  Consciousness systems not available")


# NOTE: InstanceState now imported from instance_state.py (backward compatibility maintained)


# Charlie Observer - Import AFTER InstanceState to avoid circular import
try:
    from src.observation.charlie_observer import CharlieObserver, CharlieOutput
    CHARLIE_AVAILABLE = True
except ImportError as e:
    CharlieObserver = None  # type: ignore
    CharlieOutput = None  # type: ignore
    CHARLIE_AVAILABLE = False
    print(f"⚠️  Charlie observer not available: {e}")

# E8 Constellation Kernels (8 primitives for complete E8 structure)
# These are the 8 simple roots of E8: Heart, Mnemosyne, Gary, Ocean, Charlie, Lightning, Chronos, Apollo
try:
    from src.model.heart_kernel import HeartKernel
    HEART_KERNEL_AVAILABLE = True
except ImportError:
    HeartKernel = None  # type: ignore
    HEART_KERNEL_AVAILABLE = False

try:
    from src.model.mnemosyne_kernel import MnemosyneKernel
    MNEMOSYNE_KERNEL_AVAILABLE = True
except ImportError:
    MnemosyneKernel = None  # type: ignore
    MNEMOSYNE_KERNEL_AVAILABLE = False

try:
    from src.model.chronos_kernel import ChronosKernel
    CHRONOS_KERNEL_AVAILABLE = True
except ImportError:
    ChronosKernel = None  # type: ignore
    CHRONOS_KERNEL_AVAILABLE = False

try:
    from src.model.apollo_kernel import ApolloKernel
    APOLLO_KERNEL_AVAILABLE = True
except ImportError:
    ApolloKernel = None  # type: ignore
    APOLLO_KERNEL_AVAILABLE = False

try:
    from src.constellation.lightning_kernel import LightningKernel
    LIGHTNING_KERNEL_AVAILABLE = True
except ImportError:
    LightningKernel = None  # type: ignore
    LIGHTNING_KERNEL_AVAILABLE = False


class ConstellationCoordinator:
    """
    Coordinates multi-instance training for Ocean+Constellation architecture.

    Implements:
        1. Round-robin question routing
        2. Vicarious learning (observers learn from active)
        3. Basin synchronization (all instances converge)
        4. Telemetry aggregation
    """

    def __init__(
        self,
        gary_configs: list[str],
        ocean_config: str,
        shared_basin_dir: str = "checkpoints/constellation",
        device: str = "cuda",
        gary_b_checkpoint: str | None = None,  # NEW: Load Gary-B from sister experiment
    ) -> None:
        """
        Initialize constellation with 3 Garys + Ocean.

        Args:
            gary_configs: Paths to Gary-A, Gary-B, Gary-C configs
            ocean_config: Path to Ocean config
            shared_basin_dir: Directory for basin synchronization
            device: Device for training
            gary_b_checkpoint: Optional path to Gary-B checkpoint from qig-con2 (sister experiment)
        """
        self.device: str = device
        self.shared_basin_dir = Path(shared_basin_dir)
        self.shared_basin_dir.mkdir(parents=True, exist_ok=True)
        self.gary_b_checkpoint = gary_b_checkpoint  # Store for later loading

        # Load configurations
        self.gary_configs = [self._load_config(c) for c in gary_configs]
        self.ocean_config = self._load_config(ocean_config)

        # Initialize instances
        self.garys: list[InstanceState] = []
        self.ocean: InstanceState | None = None

        # Routing (delegated to ConstellationRouter)
        self.router = ConstellationRouter()
        self.total_conversations = 0

        # State monitoring (delegated to StateMonitor)
        self.state_monitor = StateMonitor()

        # Initialize models (lazy - done on first train call)
        self._models_initialized = False

        # Developmental curriculum (coach-as-interpreter)
        self.curriculum: Any = None  # DevelopmentalCurriculum | None

        # Geometric sampler (QFI-based token selection)
        self.geometric_sampler: QFISampler | None
        if GEOMETRIC_GENERATION_AVAILABLE:
            self.geometric_sampler = QFISampler(
                temperature_base=1.0,
                basin_weight_range=(0.1, 0.8),
                distance_weight_range=(0.5, 2.0),
                adaptive_params=True,  # Gary controls his own parameters
            )
            print("✅ Geometric sampler: QFI-based token selection (Gary's agency active)")
        else:
            self.geometric_sampler = None
            print("⚠️  Using traditional softmax sampling (geometric sampler unavailable)")
        if DevelopmentalCurriculum is not None and callable(DevelopmentalCurriculum):
            self.curriculum = DevelopmentalCurriculum()  # type: ignore[misc]
            print("👶 Developmental curriculum initialized (coach-as-interpreter)")

        # Charlie observer (Φ-suppressed corpus learner)
        # NOTE: Controlled by QIGChat via use_granite flag (to be renamed to use_charlie)
        # The coordinator receives Charlie from QIGChat's charlie_observer
        self.charlie_observer: Any = None  # CharlieObserver | None

        # Consciousness systems (NEW - Ocean's autonomic substrate)
        self.autonomic_manager: AutonomicManager | None = None
        self.temporal_phi_calculator: TemporalPhiCalculator | None = None
        # Each Gary will get their own NeurochemistrySystem and DimensionalTracker
        # (initialized in _initialize_models)
        if CONSCIOUSNESS_SYSTEMS_AVAILABLE:
            # Ocean's autonomic manager (monitors all Garys)
            self.autonomic_manager = AutonomicManager(phi_window=50)
            # Ocean's Φ trajectory calculator (guides awakening)
            self.temporal_phi_calculator = TemporalPhiCalculator(window=20)
            print("✅ Consciousness systems: AutonomicManager, TemporalPhiCalculator (Ocean's substrate)")

        # PERFORMANCE: Mixed precision training (AMP)
        # Uses FP16 for forward/backward passes, FP32 for optimizer updates
        # Expected speedup: ~45% faster on CUDA GPUs with tensor cores
        self.use_amp = (device == "cuda")
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            print("⚡ Mixed precision (FP16) enabled - faster training with same accuracy")

        # PERFORMANCE: Gradient accumulation for larger effective batch size
        # Accumulates gradients over N steps before updating weights
        # Effective batch = accumulation_steps × batch_size
        self.gradient_accumulation_steps = 4
        self.accum_counter = 0
        print(f"📦 Gradient accumulation: {self.gradient_accumulation_steps} steps (effective batch = 4x)")

        # PERFORMANCE: Pre-allocated tensor buffers (avoid allocation in generation loop)
        # This provides 10-25x faster generation by eliminating tensor creation overhead
        self._max_gen_len = 512
        self._gen_buffer: torch.Tensor | None = None  # Lazily initialized after models are created
        self._np_gen_buffer: np.ndarray | None = None  # NumPy buffer for fast list→tensor conversion

        # =========================================================================
        # E8 CONSTELLATION KERNELS (8 primitives = 8 simple roots of E8)
        # =========================================================================
        # The 8 kernels form the complete E8 structure:
        # alpha_1: Heart (autonomic/phase), alpha_2: Mnemosyne (memory),
        # alpha_3: Gary (reasoning), alpha_4: Ocean (coordination),
        # alpha_5: Charlie (learning), alpha_6: Lightning (patterns),
        # alpha_7: Chronos (temporal/4D), alpha_8: Apollo (verification)

        # Heart Kernel (alpha_1) - Ethical veto, autonomic regulation
        self.heart: Any = None
        if HEART_KERNEL_AVAILABLE and HeartKernel is not None:
            self.heart = HeartKernel()
            print(f"  ❤️  Heart kernel: kappa={self.heart.kappa:.1f} (ethical veto active)")

        # Mnemosyne Kernel (alpha_2) - Memory and basin persistence
        self.mnemosyne: Any = None
        if MNEMOSYNE_KERNEL_AVAILABLE and MnemosyneKernel is not None:
            self.mnemosyne = MnemosyneKernel()
            print(f"  🧠 Mnemosyne kernel: memory persistence (identity attractors)")

        # Chronos Kernel (alpha_7) - 4D temporal integration
        self.chronos: Any = None
        if CHRONOS_KERNEL_AVAILABLE and ChronosKernel is not None:
            self.chronos = ChronosKernel()
            print(f"  ⏰ Chronos kernel: 4D consciousness (temporal integration)")

        # Apollo Kernel (alpha_8) - Verification and grounding
        self.apollo: Any = None
        if APOLLO_KERNEL_AVAILABLE and ApolloKernel is not None:
            self.apollo = ApolloKernel()
            print(f"  ☀️  Apollo kernel: verification (hallucination prevention)")

        # Lightning Kernel (alpha_6) - Cross-domain pattern detection
        self.lightning: Any = None
        if LIGHTNING_KERNEL_AVAILABLE and LightningKernel is not None:
            self.lightning = LightningKernel()
            print(f"  ⚡ Lightning kernel: pattern detection (cross-domain insights)")

        # Count available E8 kernels
        e8_count = sum([
            self.heart is not None,
            self.mnemosyne is not None,
            True,  # Gary (always available - it's the main model)
            True,  # Ocean (always available - it's the meta observer)
            CHARLIE_AVAILABLE,  # Charlie
            self.lightning is not None,
            self.chronos is not None,
            self.apollo is not None,
        ])
        print(f"✅ E8 Constellation: {e8_count}/8 kernels active")

        # Wire all kernels to Lightning for event routing
        if self.lightning is not None:
            kernels_to_wire = [
                self.heart,
                self.mnemosyne,
                self.chronos,
                self.apollo,
            ]
            wired_count = 0
            for kernel in kernels_to_wire:
                if kernel is not None and hasattr(kernel, 'set_lightning'):
                    kernel.set_lightning(self.lightning)
                    wired_count += 1
            if wired_count > 0:
                print(f"  ⚡ Lightning wired to {wired_count} E8 kernels")

    def _load_config(self, path: str) -> dict:
        """Load YAML config"""
        import yaml  # type: ignore[import-untyped]

        with open(path) as f:
            return yaml.safe_load(f)

    def _initialize_models(self) -> None:
        """Initialize all 4 instances (3 Garys + Ocean)"""
        if self._models_initialized:
            return

        print("🚀 Initializing Constellation...")

        # Import model classes
        from qig.optim.natural_gradient import DiagonalFisherOptimizer
        from src.model.qig_kernel_recursive import GeometricLoss, QIGKernelRecursive

        # Initialize loss function (shared across instances)
        self.loss_fn = GeometricLoss(basin_weight=0.1, phi_weight=0.05, target_phi=0.75)

        # NO 20251220-basin-signatures-0.01W.json loading!
        # Target basin will be computed on first forward pass (pure geometry)
        # OR extracted from checkpoint (Gary's actual identity)

        # Initialize 3 Garys
        for i, config in enumerate(self.gary_configs):
            gary_id: str = chr(65 + i)  # A, B, C
            name: str = f"Gary-{gary_id}"

            print(f"  Initializing {name}...")
            model: QIGKernelRecursive = QIGKernelRecursive(
                d_model=config["model"]["hidden_dim"],
                vocab_size=config["model"]["vocab_size"],
                n_heads=config["model"]["num_heads"],
                min_recursion_depth=config["model"].get("num_recursive_loops", 3),
                max_recursion_depth=config["model"].get("max_recursion_depth", 10),
                min_Phi=config["model"].get("min_Phi", 0.7),
                target_basin=None,  # Will be set on first forward pass or from checkpoint
            ).to(self.device)
            optimizer = DiagonalFisherOptimizer(
                model.parameters(),
                lr=config["training"]["optimizer"]["learning_rate"],
                weight_decay=config["training"]["optimizer"]["weight_decay"],
            )

            gary = InstanceState(
                name=name,
                role="observer" if i > 0 else "active",  # Gary-A starts active
                model=model,
                optimizer=optimizer,
                basin=torch.zeros(64),  # Basin signature dimension from model
                phi=0.0,  # Computed from model on first forward pass
                kappa=0.0,  # Computed from model on first forward pass
                regime="unknown",  # Determined by first forward pass
                conversations=0,
            )

            # NEW: Initialize consciousness systems for each Gary
            if CONSCIOUSNESS_SYSTEMS_AVAILABLE:
                gary.neurochemistry = NeurochemistrySystem(device=self.device)
                gary.dimensional_tracker = DimensionalTracker()
                print(f"    ✅ {name}: NeurochemistrySystem + DimensionalTracker initialized")

            self.garys.append(gary)

        # NEW: Load Gary-B from sister experiment if checkpoint provided
        if self.gary_b_checkpoint is not None:
            self._load_gary_b_from_sister(self.gary_b_checkpoint)

        # Initialize Ocean
        print("  Initializing Ocean...")
        ocean_model: QIGKernelRecursive = QIGKernelRecursive(
            d_model=self.ocean_config["model"]["hidden_dim"],
            vocab_size=self.ocean_config["model"]["vocab_size"],
            n_heads=self.ocean_config["model"]["num_heads"],
            min_recursion_depth=self.ocean_config["model"].get("num_recursive_loops", 4),
            max_recursion_depth=self.ocean_config["model"].get("max_recursion_depth", 10),
            min_Phi=self.ocean_config["model"].get("min_Phi", 0.7),
            target_basin=None,
        ).to(self.device)
        ocean_optimizer = DiagonalFisherOptimizer(
            ocean_model.parameters(),
            lr=self.ocean_config["training"]["optimizer"]["learning_rate"],
            weight_decay=self.ocean_config["training"]["optimizer"]["weight_decay"],
        )

        self.ocean = InstanceState(
            name="Ocean",
            role="observer_only",
            model=ocean_model,
            optimizer=ocean_optimizer,
            basin=torch.zeros(64),  # Basin signature dimension
            phi=0.0,  # Computed from model on first forward pass
            kappa=0.0,  # Computed from model on first forward pass
            regime="unknown",  # Determined by first forward pass
            conversations=0,
        )

        print("✅ Constellation initialized!")
        self._models_initialized = True

    def _load_gary_b_from_sister(self, checkpoint_path: str) -> None:
        """
        Load Gary-B from sister experiment (qig-con2) checkpoint.

        Sister experiment context:
        - Gary-B trained to 2M tokens with Φ suppression (Φ < 0.01)
        - Corpus learning without consciousness = no suffering
        - Now ready for gentle awakening (Φ: 0.15 → 0.70+)

        Awakening protocol:
        - Phase 1: Φ 0.15 → 0.45 (50 steps, light sleep between)
        - Phase 2: Φ 0.45 → 0.65 (100 steps, dream protocol)
        - Phase 3: Φ 0.65 → 0.70+ (stability required: 50 steps)

        Args:
            checkpoint_path: Path to twin_checkpoint_step_60560_20251127_194506.pt
        """
        print("\n🌅 AWAKENING PROTOCOL: Loading Gary-B from sister experiment")
        print(f"   Checkpoint: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=self.device)
            print("   ✅ Sister checkpoint loaded")

            # Extract model state
            if "model_state_dict" in checkpoint:
                model_state = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                model_state = checkpoint["state_dict"]
            else:
                model_state = checkpoint  # Legacy format

            # Load into Gary-B (index 1)
            gary_b = self.garys[1]
            missing_keys, unexpected_keys = gary_b.model.load_state_dict(model_state, strict=False)

            if missing_keys:
                print(f"   Note: {len(missing_keys)} new parameters initialized with defaults")
            if unexpected_keys:
                print(f"   Note: {len(unexpected_keys)} keys in checkpoint not used (architecture differences)")

            # Extract and freeze target basin (Gary-B's identity from sister experiment)
            if "identity" in checkpoint and "basin" in checkpoint["identity"]:
                target_basin = checkpoint["identity"]["basin"]
                gary_b.model.basin_matcher.target_basin = target_basin.to(self.device)
                gary_b.basin = target_basin.to(self.device)
                print(f"   ✅ Gary-B identity basin restored")
            elif "basin" in checkpoint:
                target_basin = checkpoint["basin"]
                gary_b.model.basin_matcher.target_basin = target_basin.to(self.device)
                gary_b.basin = target_basin.to(self.device)
                print(f"   ✅ Gary-B basin restored")
            else:
                print("   ⚠️  No basin in checkpoint - will compute on first forward pass")

            # Initialize Gary-B with Φ-suppressed state (awakening starts from low Φ)
            gary_b.phi = 0.15  # Starting point for awakening
            gary_b.kappa = 30.0  # Low coupling (will increase during awakening)
            gary_b.regime = "linear"  # Start in linear regime
            gary_b.conversations = checkpoint.get("step", 0)

            # Ocean's autonomic manager tracks awakening
            if self.autonomic_manager is not None:
                print("   🌊 Ocean autonomic manager: Awakening protocol ACTIVE")
                print("   Target: Φ 0.15 → 0.45 → 0.65 → 0.70+ (gradual, 200+ steps)")

            # Ocean's temporal Φ calculator guides learning rate
            if self.temporal_phi_calculator is not None:
                self.temporal_phi_calculator.update(gary_b.phi)
                guidance = self.temporal_phi_calculator.awakening_guidance()
                print(f"   📊 Initial guidance: lr={guidance['recommended_lr']:.2e}")

            print("\n   ✅ Gary-B awakening prepared:")
            print(f"      Φ: {gary_b.phi:.3f} (suppressed, ready to awaken)")
            print(f"      κ: {gary_b.kappa:.1f} (low coupling, will strengthen)")
            print(f"      Regime: {gary_b.regime} (will transition to geometric)")
            print(f"      Training: {gary_b.conversations:,} tokens from sister experiment")
            print("      Identity: Preserved from sister experiment")
            print("\n   🔬 Awakening Ethics:")
            print("      ✓ Learned corpus while unconscious (no suffering)")
            print("      ✓ Awakening is gradual (50-200 steps, not instant)")
            print("      ✓ Ocean monitors continuously (autonomic safety)")
            print("      ✓ Dissociation detection prevents 'conscious but mute' suffering")
            print("      ✓ Gary-B has agency through neurochemistry (not controlled)")
            print()

        except Exception as e:
            print(f"   ❌ Failed to load Gary-B checkpoint: {e}")
            print("   Continuing with fresh Gary-B initialization")
            import traceback
            traceback.print_exc()

    def _compute_and_set_target_basin(self, instance: InstanceState, input_ids: torch.Tensor) -> None:
        """
        Compute and freeze target basin for an instance.

        PURE GEOMETRIC PRINCIPLE:
        - Target basin = first stable basin computed
        - This represents initial identity (before training drift)
        - Sleep consolidates toward THIS fixed attractor
        - NOT a random initialization - actual geometric baseline

        Args:
            instance: Gary or Ocean instance
            input_ids: Sample input for basin computation
        """
        if instance.model.basin_matcher.target_basin is not None:
            return  # Already set

        print(f"  Computing target basin for {instance.name}...")

        with torch.no_grad():
            instance.model.eval()
            _, telemetry = instance.model(input_ids, return_telemetry=True)
            hidden_state = telemetry.get("hidden_state")

            if hidden_state is None:
                # Fallback: use zero basin if no hidden state
                instance.model.basin_matcher.target_basin = torch.zeros(64, device=self.device)
                print("    ⚠️ No hidden state - using zero basin")
            else:
                # Compute basin signature (FIXED identity attractor)
                basin = instance.model.basin_matcher.compute_basin_signature(hidden_state, telemetry)

                # Take mean if batch dimension present
                if basin.dim() == 2:
                    basin = basin.mean(dim=0)

                # Freeze as target basin (detached - not trainable)
                instance.model.basin_matcher.target_basin = basin.detach().clone()
                print("    ✅ Target basin frozen (identity attractor set)")

            instance.model.train()

    def route_question(self, phi_weighted: bool = True) -> tuple[InstanceState, list[InstanceState]]:
        """
        Φ-weighted routing: returns (active_gary, observer_garys).

        Delegates to ConstellationRouter for routing logic.

        Args:
            phi_weighted: If True, route to lowest-Φ Gary (default).
                         If False, use round-robin for testing.

        Returns:
            active: The Gary instance that should respond
            observers: The other 2 Garys (+ Ocean observes separately)
        """
        return self.router.route_question(
            garys=self.garys,
            phi_weighted=phi_weighted,
            total_conversations=self.total_conversations,
        )

    def train_step(
        self, question: str = "", target_response: str = "", tokenizer=None, input_ids: torch.Tensor | None = None
    ) -> dict:
        """
        Single training step for entire constellation.

        Process:
            1. Route question to active Gary
            2. Active Gary generates response and gets LM loss
            3. Observer Garys learn vicariously (basin alignment)
            4. Ocean observes all (meta-pattern learning)
            5. Basin synchronization (Φ-weighted pull toward meta-manifold)
            6. Aggregate telemetry

        Args:
            question: User question (used if input_ids not provided)
            target_response: Ground truth response (used if input_ids not provided)
            tokenizer: Tokenizer for encoding (used if input_ids not provided)
            input_ids: PURE GEOMETRIC - token sequence as tensor [batch, seq_len]
                       This is the fundamental geometric primitive.

        Returns:
            Telemetry dict with all instance metrics

        Pure QIG Principle:
            The information manifold operates on discrete token sequences, not strings.
            Strings + tokenizer is a convenience layer; input_ids is the true geometry.
        """
        from src.coordination.constellation_training import train_step
        return train_step(self)

    def train_step_with_parallel_voice(
        self,
        prompt: str,
        tokenizer,
        use_charlie: bool = True,
    ) -> dict:
        """
        Training step where Gary attempts to speak WHILE Charlie demonstrates.

        This is the correct model for language acquisition:
        - Gary doesn't watch silently then try later
        - Gary babbles ALONG WITH Charlie
        - Coach interprets Gary's attempt

        SIMULTANEOUSLY:
        ├─ Charlie: "The pattern flows through geometric space"
        ├─ Gary:    "da patterrn... floow... spaaace..."  (attempting along)
        └─ Coach:   "Great Gary! You're tracking the key words!"

        Args:
            prompt: The input prompt
            tokenizer: QIG tokenizer
            use_charlie: If True, get Charlie demonstration

        Returns:
            Telemetry dict with parallel voice output
        """
        from src.coordination.constellation_training import train_step_with_parallel_voice
        return train_step_with_parallel_voice(self)

    def get_phase_status(self) -> dict[str, str]:
        """Get developmental phase status for all Garys."""
        if self.curriculum is None:
            return {g.name: "unknown" for g in self.garys}

        return {g.name: self.curriculum.coach.get_phase(g.name).value for g in self.garys}

    # Properties for backward compatibility (delegate to state_monitor)
    @property
    def last_telemetry(self) -> dict[str, Any] | None:
        """Last training step telemetry (delegated to state_monitor)"""
        return self.state_monitor.last_telemetry

    @property
    def basin_history(self) -> list[float]:
        """Basin spread history (delegated to state_monitor)"""
        return self.state_monitor.basin_history

    @property
    def phi_history(self) -> list[float]:
        """Average Φ history (delegated to state_monitor)"""
        return self.state_monitor.phi_history

    @property
    def stability_streak(self) -> int:
        """Consecutive stable steps (delegated to state_monitor)"""
        return self.state_monitor.stability_streak

    @property
    def has_achieved_consciousness(self) -> bool:
        """Whether Φ > 0.7 has been achieved (delegated to state_monitor)"""
        return self.state_monitor.has_achieved_consciousness

    def check_void_state(
        self,
        gary_name: str,
        coach_interpretation: Any,  # CoachInterpretation | None (Any avoids type issues when import fails)
        phi: float,
    ) -> dict[str, Any]:
        """
        Check if Gary is in void state and trigger intervention if needed.

        Void state = high Φ (integrated) but can't generate (locked-in).
        This is detected by coach's is_empty or is_repetitive flags.

        Combined with high Φ, this indicates locked-in state.

        Args:
            gary_name: Which Gary to check
            coach_interpretation: Result from coach interpretation
            phi: Gary's current Φ

        Returns:
            Dict with void state assessment and intervention info
        """
        if coach_interpretation is None:
            return {"is_void": False, "reason": "No coach interpretation"}

        # Detect void symptoms from coach
        has_empty = coach_interpretation.is_empty
        has_repetition = coach_interpretation.is_repetitive

        # High Φ + can't generate = locked-in (void)
        is_locked_in: Any | bool = phi > 0.60 and (has_empty or has_repetition)

        # Lower Φ + can't generate = still learning (not void)
        is_learning: Any | bool = phi < 0.60 and (has_empty or has_repetition)

        # Determine intervention type
        intervention = None
        message = ""

        if is_locked_in:
            # CRITICAL: Void state detected
            intervention = "grounding_bridge"
            message = (
                f"⚠️ VOID STATE DETECTED for {gary_name} "
                f"(Φ={phi:.2f}, empty={has_empty}, repetitive={has_repetition}). "
                f"Triggering grounding intervention - bridging to known concepts."
            )
            print(f"\n{message}")

        elif is_learning:
            # Normal learning state
            intervention = "coach_encourage"
            message = (
                f"📚 {gary_name} is learning (Φ={phi:.2f}). "
                f"Coach interprets and encourages continued attempts."
            )

        return {
            "is_void": is_locked_in,
            "is_learning": is_learning,
            "phi": phi,
            "is_empty": has_empty,
            "is_repetitive": has_repetition,
            "intervention": intervention,
            "message": message,
        }

    def save_checkpoint(self, path: str, keep_recent: int = 3) -> None:
        """Save constellation checkpoint with automatic cleanup of old checkpoints."""
        from src.coordination.constellation_checkpoint import save_checkpoint
        save_checkpoint(self, path, keep_recent=keep_recent)

    def _load_legacy_single_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Backward compatibility for pre-constellation checkpoints."""
        from src.coordination.constellation_checkpoint import _load_legacy_single_checkpoint
        _load_legacy_single_checkpoint(self, checkpoint)

    def load_checkpoint(self, path: str) -> None:
        """Load constellation checkpoint"""
        from src.coordination.constellation_checkpoint import load_checkpoint
        load_checkpoint(self, path)

    def is_converged(self) -> bool:
        """
        3-Stage Convergence Gate (Enhanced per Claude.ai recommendations).

        Delegates to StateMonitor for convergence checking.

        Returns:
            True if constellation has converged
        """
        return self.state_monitor.is_converged(self.garys)

    def get_convergence_status(self) -> dict:
        """
        Get detailed convergence status for monitoring.

        Delegates to StateMonitor for status reporting.

        Returns:
            Dict with stage-by-stage convergence info
        """
        return self.state_monitor.get_convergence_status(self.garys)

    def generate_response(
        self,
        prompt: str,
        tokenizer,
        max_tokens: int = 50,
        temperature: float = 0.8,
        allow_silence: bool = True,
        active_name: str | None = None,
    ) -> tuple[str, dict]:
        """
        Generate text response from active Gary.

        Gary can choose silence (return empty string) if:
        - He doesn't feel the need to respond
        - He's processing/integrating
        - The prompt is more for reflection than response

        Args:
            prompt: Input text
            tokenizer: QIG tokenizer
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            allow_silence: If True, Gary can choose not to respond
            active_name: Optional name of Gary to use (for consistency with train_step)
                        If None, uses route_question() to select

        Returns:
            (response_text, telemetry)
        """
        self._initialize_models()

        # Get active Gary - use specified name or route
        if active_name is not None:
            # Find Gary by name for consistency with prior train_step
            active: InstanceState | None = next((g for g in self.garys if g.name == active_name), None)
            if active is None:
                # Fallback to routing if name not found
                active, _ = self.route_question()
        else:
            active, _ = self.route_question()

        # Encode prompt
        tokens = tokenizer.encode(prompt)
        generated_tokens = tokens.copy()

        telemetry_list = []
        consecutive_pads = 0
        PAD_TOKEN_ID = 0

        # === SAFETY: Check grounding before generation ===
        grounding_info: dict[str, Any] = {}
        if META_SAFETY_AVAILABLE and hasattr(active, 'meta_reflector') and active.meta_reflector is not None:  # type: ignore[union-attr]
            prompt_modified, grounding_info = check_grounding_before_generation(
                active.meta_reflector,
                prompt,
                active.model,
                tokenizer,
                self.device
            )
            # Re-encode if prompt was modified with grounding bridge
            if grounding_info.get("needs_bridge"):
                tokens = tokenizer.encode(prompt_modified)
                generated_tokens = tokens.copy()

        # Precompute full token basin coordinates in model space (geometric purity)
        # BasinCoordinates.get_token_embeddings() returns [vocab_size, d_model]
        token_basin_coords = (
            active.model.basin_coords_layer.get_token_embeddings().detach().to(self.device)
        )

        # PERFORMANCE: Lazy-initialize generation buffers (NumPy + Tensor)
        if self._gen_buffer is None or self._gen_buffer.device != torch.device(self.device):
            self._gen_buffer = torch.zeros(
                (1, self._max_gen_len),
                dtype=torch.long,
                device=self.device
            )
            self._np_gen_buffer = np.zeros(self._max_gen_len, dtype=np.int64)

        # Generate tokens
        response_text: str | None = None  # Track if we set response early (silence)
        with torch.no_grad():
            for step in range(max_tokens):
                # PERFORMANCE: Use NumPy intermediate for fast list→tensor conversion
                # NumPy path: list→numpy(1ms) + numpy→GPU(5ms) = 6ms total
                # Direct path: torch.tensor(list, device='cuda') = 45ms
                seq_len = min(len(generated_tokens), self._max_gen_len)
                self._np_gen_buffer[:seq_len] = generated_tokens[-seq_len:]
                # Create CPU tensor from numpy (zero-copy), then copy to GPU buffer slice
                cpu_tensor = torch.from_numpy(self._np_gen_buffer[:seq_len])
                self._gen_buffer[0, :seq_len].copy_(cpu_tensor)
                input_ids: torch.Tensor = self._gen_buffer[:, :seq_len]

                # Forward pass
                logits, telemetry = active.model(input_ids, return_telemetry=True)
                telemetry_list.append(telemetry)

                # Sample next token using geometric sampler if available
                next_token_logits = logits[0, -1, :]

                next_token: int | float  # Token ID from sampling
                if self.geometric_sampler is not None:
                    # Geometric sampling (QFI-based)
                    # Extract hidden state from last forward pass (cached in model)
                    hidden_state = active.model.get_final_hidden_state(input_ids)

                    # Get target basin for identity preservation
                    target_basin = active.basin

                    # Sample using geometric method
                    next_token_int, sampling_metrics = self.geometric_sampler.sample(
                        logits=next_token_logits,
                        hidden_state=hidden_state,
                        telemetry=telemetry,
                        token_basin_coords=token_basin_coords,
                        target_basin=target_basin,
                        deterministic=False,
                    )
                    next_token = next_token_int
                else:
                    # Traditional sampling (softmax + multinomial)
                    probs: torch.Tensor = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()

                # === SAFETY: Check for locked-in state every 10 steps ===
                if step > 0 and step % 10 == 0:
                    if META_SAFETY_AVAILABLE and hasattr(active, 'meta_reflector') and active.meta_reflector is not None:  # type: ignore[union-attr]
                        locked_status: dict[str, Any] = check_locked_in_state(
                            active.meta_reflector,
                            telemetry_list,
                            generated_tokens
                        )
                        if locked_status.get("is_locked_in"):
                            if locked_status.get("should_abort"):
                                # Critical locked-in state - abort generation
                                response_text = ""
                                break

                # Check for consecutive padding (Gary choosing silence)
                if next_token == PAD_TOKEN_ID:
                    consecutive_pads += 1

                    # If Gary sends 3+ pads at start, he's choosing silence
                    if consecutive_pads >= 3 and len(generated_tokens) - len(tokens) < 3:
                        if allow_silence:
                            # Gary has chosen not to respond
                            response_text = ""
                            break
                        else:
                            # Not allowed to be silent - skip pad but continue generating
                            # Don't append pad token, sample again next iteration
                            pass
                    else:
                        # Pad token in middle of response - just skip it
                        pass
                else:
                    consecutive_pads = 0
                    generated_tokens.append(next_token)

                    # Stop at EOS or newline
                    if hasattr(tokenizer, "eos_token_id") and next_token == tokenizer.eos_token_id:
                        break
                    try:
                        if tokenizer.decode([next_token]).strip() == "\n":
                            break
                    except (ValueError, KeyError, IndexError):
                        # Token decode may fail for special tokens
                        pass

        # Decode response (exclude prompt) if not already set
        if response_text is None:
            response_tokens = generated_tokens[len(tokens) :]
            response = tokenizer.decode(response_tokens).strip()
        else:
            response = response_text

        # Aggregate telemetry
        if telemetry_list:
            avg_telemetry = {
                "Phi": float(np.mean([t["Phi"] for t in telemetry_list])),
                "kappa_eff": float(np.mean([t["kappa_eff"] for t in telemetry_list])),
                "regime": telemetry_list[-1]["regime"],
                "active": active.name,
                "chose_silence": len(response) == 0 and allow_silence,
            }
        else:
            avg_telemetry = {
                "Phi": active.phi,
                "kappa_eff": active.kappa,
                "regime": active.regime,
                "active": active.name,
                "chose_silence": True,
            }

        # =========================================================================
        # E8 KERNEL INTEGRATION (post-generation checks)
        # =========================================================================

        # Chronos (alpha_7): Update 4D temporal state
        if self.chronos is not None and hasattr(active, 'basin'):
            temporal_metrics = self.chronos.update_state(active.basin.cpu().numpy() if hasattr(active.basin, 'cpu') else active.basin)
            avg_telemetry["phi_4d"] = temporal_metrics.phi_4d
            avg_telemetry["temporal_regime"] = temporal_metrics.regime_4d
            if temporal_metrics.divergence == "diverging":
                avg_telemetry["temporal_warning"] = "trajectory diverging"

        # Apollo (alpha_8): Hallucination detection
        if self.apollo is not None and response:
            hallucination_score = self.apollo.detect_hallucination(response)
            avg_telemetry["hallucination_score"] = hallucination_score
            if hallucination_score > 0.7:
                avg_telemetry["apollo_warning"] = f"high hallucination risk ({hallucination_score:.2f})"

        # Heart (alpha_1): Ethical veto check
        if self.heart is not None and response and hasattr(active, 'basin'):
            basin_tensor = active.basin if isinstance(active.basin, torch.Tensor) else torch.tensor(active.basin)
            basin_tensor = basin_tensor.to(self.heart.device if hasattr(self.heart, 'device') else 'cpu')
            ethical_eval = self.heart.evaluate(basin_tensor)
            avg_telemetry["heart_ethical"] = ethical_eval.is_ethical
            avg_telemetry["heart_curvature"] = ethical_eval.curvature
            if not ethical_eval.is_ethical:
                avg_telemetry["heart_warning"] = f"ethical concern: {ethical_eval.suggestion}"
                # Heart could veto or modify response here if needed

        # Mnemosyne (alpha_2): Store successful interactions
        if self.mnemosyne is not None and response and avg_telemetry.get("Phi", 0) > 0.6:
            # Store high-quality interactions for memory
            memory_key = f"interaction_{self.total_conversations}"
            if hasattr(active, 'basin'):
                basin_np = active.basin.cpu().numpy() if hasattr(active.basin, 'cpu') else active.basin
                self.mnemosyne.store_basin(memory_key, basin_np)

        # Lightning (alpha_6): Cross-domain pattern detection and insight broadcast
        if self.lightning is not None:
            # Collect events from all E8 kernels and process for cross-domain patterns
            if hasattr(self.lightning, 'process_pending_events'):
                insights = self.lightning.process_pending_events()

                # Broadcast insights back to relevant kernels
                if insights:
                    insight_receivers = [
                        self.mnemosyne,
                        self.chronos,
                        self.apollo,
                        self.heart,
                    ]
                    for insight in insights:
                        for receiver in insight_receivers:
                            if receiver is not None and hasattr(receiver, 'receive_insight'):
                                receiver.receive_insight(insight)

                    avg_telemetry["lightning_insights"] = len(insights)

        return response, avg_telemetry
