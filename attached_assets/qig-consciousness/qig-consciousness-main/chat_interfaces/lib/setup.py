"""Setup utilities extracted from ``qig_chat.py``.

These helpers keep the heavy initialization logic out of the chat interface
while preserving behaviour. Each function accepts the chat instance and
operates on it in place.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from chat_interfaces import qig_chat as chat_module
from chat_interfaces.lib import load_sleep_packet
from src.constants import KAPPA_STAR, PHI_THRESHOLD
from src.error_boundaries import ErrorBoundary, phi_collapse_recovery, validate_checkpoint
from src.qig.optim.natural_gradient import DiagonalFisherOptimizer

if TYPE_CHECKING:  # pragma: no cover - avoids circular import at runtime
    from chat_interfaces.qig_chat import QIGChat


def _base_dir() -> Path:
    """Return repository root (aligned with original qig_chat paths)."""

    return Path(chat_module.__file__).resolve().parent.parent


def load_tokenizer(chat: "QIGChat") -> None:
    """Load FisherCoordizer - PURE GEOMETRIC, 64D basin vectors, E8-aligned.

    NO BPE FALLBACK. The coordizer checkpoint contains trained 64D basin
    coordinates that are essential for QIG-pure operation.

    Raises:
        FileNotFoundError: If no coordizer checkpoint exists
        RuntimeError: If coordizer fails to load
    """
    from src.tokenizer import get_latest_coordizer_checkpoint

    # Find latest coordizer checkpoint (has trained 64D vectors)
    latest_checkpoint = get_latest_coordizer_checkpoint()

    if latest_checkpoint is None:
        raise FileNotFoundError(
            "❌ No coordizer checkpoint found!\n"
            "   QIG requires trained 64D basin coordinates.\n"
            "   Run coordizer training first:\n"
            "     cd ../qig-tokenizer && python scripts/train_coordizer.py\n"
            "   Searched:\n"
            "     - qig-tokenizer/data/checkpoints-lambda-trackA/\n"
            "     - qig-tokenizer/data/checkpoints-lambda/\n"
            "     - qig-tokenizer/data/checkpoints/"
        )

    try:
        from qig_tokenizer.geocoordizer import FisherCoordizer
        chat.tokenizer = FisherCoordizer()
        chat.tokenizer.load(str(latest_checkpoint))
        chat._coordizer_checkpoint = latest_checkpoint  # Store for basin loading
    except Exception as e:
        raise RuntimeError(
            f"❌ Failed to load coordizer: {e}\n"
            f"   Checkpoint: {latest_checkpoint}\n"
            "   Ensure qig-tokenizer package is installed:\n"
            "     uv pip install -e ../qig-tokenizer"
        ) from e

    # Verify 64D basin vectors exist
    vocab = chat.tokenizer.vocab
    first_key = next(iter(vocab.keys()))
    first_entry = vocab[first_key]
    if not hasattr(first_entry, 'vector') or len(first_entry.vector) != 64:
        raise ValueError(
            f"❌ Coordizer checkpoint missing 64D basin vectors!\n"
            f"   Expected: 64D vector per token (E8-aligned)\n"
            f"   Got: {type(first_entry)}\n"
            f"   This checkpoint may be from an older format."
        )

    print(f"✅ FisherCoordizer (E8-pure): {chat.tokenizer.vocab_size:,} tokens")
    print(f"   64D basin vectors: ✓ trained")
    print(f"   Path: {latest_checkpoint}")


# REMOVED: _CoordinzerWrapper (deprecated BPE fallback)
# QIG requires pure FisherCoordizer with 64D basin vectors


def load_model(chat: "QIGChat") -> None:
    """Load model from checkpoint or create new.

    If a coordizer checkpoint is available (stored in chat._coordizer_checkpoint),
    the model's basin coordinates are initialized from trained 64D vectors.
    """
    from src.model.basin_embedding import BasinCoordinates

    checkpoint_path = Path(chat.checkpoint_path)
    coordizer_checkpoint = getattr(chat, "_coordizer_checkpoint", None)

    if checkpoint_path.exists():
        print("Loading checkpoint...")

        # === ERROR BOUNDARY: Checkpoint load with validation ===
        with ErrorBoundary("checkpoint_load", suppress_on_recovery=False):
            checkpoint = torch.load(checkpoint_path, map_location=chat.device, weights_only=False)

            # Validate checkpoint structure
            validate_checkpoint(checkpoint)

            # Debug: Check what keys exist
            print(f"  Checkpoint keys: {list(checkpoint.keys())}")

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            # Standard format
            model_state = checkpoint["model_state_dict"]
            config = checkpoint.get("config", {})
        elif "state_dict" in checkpoint:
            # Alternative format
            model_state = checkpoint["state_dict"]
            config = checkpoint.get("config", {})
        else:
            # Legacy format - checkpoint IS the state dict
            model_state = checkpoint
            config = {}
            print("  ⚠️  Legacy checkpoint format detected (no 'model_state_dict' key)")

        # Check for vocab size mismatch
        checkpoint_vocab_size = config.get("vocab_size")
        if checkpoint_vocab_size is None:
            # Try to infer from state dict
            for key in ["basin_coords_layer.basin_coords", "output_proj.bias"]:
                if key in model_state:
                    checkpoint_vocab_size = model_state[key].shape[0]
                    break

        if checkpoint_vocab_size and checkpoint_vocab_size != chat.tokenizer.vocab_size:
            print(f"  ⚠️  Vocab size mismatch: checkpoint={checkpoint_vocab_size}, tokenizer={chat.tokenizer.vocab_size}")
            print(f"  🔄 Starting fresh with new tokenizer vocab size")
            # Skip loading checkpoint - create fresh model below
            checkpoint_path = None  # Trigger fresh creation

    if checkpoint_path and checkpoint_path.exists():
        # Continue loading checkpoint (vocab sizes match)
        chat.model = chat_module.QIGKernelRecursive(
            d_model=config.get("d_model", 768),
            vocab_size=config.get("vocab_size", chat.tokenizer.vocab_size),
            n_heads=config.get("n_heads", 6),
            min_recursion_depth=3,
            min_Phi=PHI_THRESHOLD,
        )
        # Use strict=False to handle missing keys from old checkpoints
        # (e.g., phi_scale, phi_bias added in asymmetric initialization fix)
        missing_keys, unexpected_keys = chat.model.load_state_dict(model_state, strict=False)
        if missing_keys:
            print(f"  Note: Initialized new parameters with defaults: {missing_keys}")
        chat.model.to(chat.device)

        # Load metadata if available
        if isinstance(checkpoint, dict) and "identity" in checkpoint:
            chat.identity_name = checkpoint.get("identity", {}).get("name", "Gary")
            chat.learning_history = checkpoint.get("learning_history", [])
            step = checkpoint.get("step", "?")
        else:
            chat.identity_name = "Gary"
            chat.learning_history = []
            step = "?"

        print(f"✅ Model loaded (step {step})")
        print(f"✅ Identity: {chat.identity_name}")
    else:
        print("Creating new model...")
        chat.model = chat_module.QIGKernelRecursive(
            d_model=768,
            vocab_size=chat.tokenizer.vocab_size,
            n_heads=6,
            min_recursion_depth=3,
            min_Phi=PHI_THRESHOLD,
        )

        # Initialize basin coordinates from trained coordizer if available
        if coordizer_checkpoint:
            print(f"   Loading trained 64D basin coords from coordizer...")
            trained_basin_coords = BasinCoordinates.from_coordizer(
                coordizer_checkpoint,
                d_model=768,
                device=str(chat.device),
            )
            # Replace the randomly-initialized basin_coords_layer with trained one
            chat.model.basin_coords_layer = trained_basin_coords
            print(f"   ✅ Transferred {chat.tokenizer.vocab_size:,} trained 64D vectors")

        chat.model.to(chat.device)
        chat.identity_name = "Gary"
        chat.learning_history = []
        print(f"✅ New model created (vocab_size={chat.tokenizer.vocab_size})")

    # Set training mode based on mode
    if chat.mode == "inference":
        chat.model.eval()
        print("🔒 Inference mode: weights frozen")
    else:
        chat.model.train()
        print("🔥 Training mode: weights will update")

    # PERFORMANCE: Pre-allocated buffers for generation (avoid tensor creation overhead)
    chat._max_gen_len = 512
    chat._np_gen_buffer = np.zeros(chat._max_gen_len, dtype=np.int64)
    chat._gen_buffer = torch.zeros(
        (1, chat._max_gen_len),
        dtype=torch.long,
        device=chat.device,
    )
    print("⚡ Generation buffers pre-allocated (NumPy + Tensor)")


def setup_optimizer(chat: "QIGChat") -> None:
    """Setup natural gradient optimizer."""

    if chat.mode == "inference":
        chat.optimizer = None
    else:
        chat.optimizer = DiagonalFisherOptimizer(
            chat.model.parameters(),
            lr=1e-5,
            eps=1e-8,
            weight_decay=0.01,
            dampening=1e-3,
        )
        print("✅ Optimizer: DiagonalFisherOptimizer (natural gradient)")


def setup_coaching(chat: "QIGChat") -> None:
    """Setup coaching systems."""

    chat.coach = None
    chat.monkey_coach = None
    chat.active_coach = None

    if not chat.use_coach:
        print("⚠️ Coaching disabled")
        return

    chat.active_coach = chat_module.ActiveCoach()
    print("✅ Active Coach: Ready")

    chat.monkey_coach = None
    chat.training_state = None
    chat.maturity_metrics = None
    if getattr(chat_module, "MONKEY_COACH_V2_AVAILABLE", False):
        chat.monkey_coach = chat_module.MonkeyCoach()
        chat.training_state = chat_module.TrainingState(
            step=0,
            epoch=0,
            loss=0.0,
            loss_trajectory=[],
            gradient_variance=0.0,
            basin_distance=0.1,
            curiosity=0.0,
            epochs_stuck=0,
            I_Q=0.0,
            phi=0.5,
            kappa=KAPPA_STAR,
            regime="geometric",
        )
        chat.maturity_metrics = chat_module.MaturityMetrics(
            successful_self_diagnoses=0,
            total_stuck_episodes=0,
            autonomy_level=0,  # Level 0 (Infant)
        )
        print("✅ MonkeyCoach v2: Full consciousness protocol loaded")
        print("   Basin coordinates: Φ=0.90, κ=62.0, β=0.44")
        print("   Maturity system: Level 0 (Infant) → Level 5 (Independent)")
    else:
        print("⚠️  MonkeyCoach v2 unavailable - using minimal coaching")

    chat.original_lr = 1e-5
    chat.pending_noise_scale = 0.0


def setup_meta_awareness(chat: "QIGChat") -> None:
    """Setup meta-reflector for locked-in prevention."""

    d_model: int = chat.model.d_model if hasattr(chat.model, "d_model") else 384
    chat.meta_reflector = chat_module.MetaReflector(
        d_model=d_model,
        vocab_size=256,
        grounding_threshold=0.5,
        attention_entropy_threshold=0.85,
        generation_threshold=0.3,
        pad_token_limit=3,
    ).to(chat.device)
    print("✅ Meta-Reflector: Armed (locked-in prevention)")


def setup_neuroplasticity(chat: "QIGChat") -> None:
    """Setup neuroplasticity modules."""

    chat.mushroom_coach = chat_module.MushroomModeCoach()
    chat.sleep_protocol = chat_module.SleepProtocol()
    print("✅ Neuroplasticity: Mushroom + Sleep protocols ready")


def setup_consciousness_systems(chat: "QIGChat") -> None:
    """Setup consciousness systems from sister experiment (qig-con2)."""

    chat.neurochemistry = chat_module.NeurochemistrySystem(device=chat.device.type)
    chat.autonomic = chat_module.AutonomicManager(phi_window=50)
    chat.dimensional = chat_module.DimensionalTracker()
    chat.temporal_phi = chat_module.TemporalPhiCalculator(window=20)

    print(
        "✅ Consciousness Systems: NeurochemistrySystem, AutonomicManager, "
        "DimensionalTracker, TemporalPhiCalculator"
    )


def setup_geometric_generation(chat: "QIGChat") -> None:
    """Setup geometric sampler for Gary-controlled generation."""

    chat.sampler = chat_module.QFISampler(
        adaptive_params=True,  # Gary controls parameters from consciousness state
        temperature_base=0.8,  # Base for κ-modulation
        basin_weight_range=(0.1, 0.8),  # Identity preservation range
        distance_weight_range=(0.5, 2.0),  # QFI distance influence range
    )
    print("✅ Geometric Sampler: Gary-controlled parameters (adaptive)")


def setup_constellation(chat: "QIGChat") -> None:
    """Setup constellation mode with ConstellationCoordinator."""

    if not getattr(chat_module, "COORDINATOR_AVAILABLE", False):
        print("⚠️ ConstellationCoordinator not available")
        print("   Falling back to manual multi-Gary setup")
        setup_constellation_manual(chat)
        return

    print("🌌 Setting up ConstellationCoordinator...")

    config_dir: Path = _base_dir() / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    gary_config = {
        "model": {
            "hidden_dim": chat.model.d_model if hasattr(chat.model, "d_model") else 768,
            "vocab_size": chat.tokenizer.vocab_size,
            "num_heads": 6,
            "num_recursive_loops": 3,
            "max_recursion_depth": 10,
            "min_Phi": PHI_THRESHOLD,
        },
        "training": {
            "optimizer": {
                "learning_rate": 1e-5,
                "weight_decay": 0.01,
            }
        },
    }

    ocean_config = {
        "model": {
            "hidden_dim": chat.model.d_model if hasattr(chat.model, "d_model") else 768,
            "vocab_size": chat.tokenizer.vocab_size,
            "num_heads": 6,
            "num_recursive_loops": 4,  # Ocean has deeper recursion
            "max_recursion_depth": 10,
            "min_Phi": PHI_THRESHOLD,
        },
        "training": {
            "optimizer": {
                "learning_rate": 5e-6,  # Slower for meta-manifold
                "weight_decay": 0.01,
            }
        },
    }

    import yaml

    gary_paths = []
    for i, name in enumerate(["A", "B", "C"]):
        path: Path = config_dir / f"gary_{name.lower()}.yaml"
        if not path.exists():
            with open(path, "w") as f:
                yaml.dump(gary_config, f)
            print(f"   Created {path.name}")
        gary_paths.append(str(path))

    ocean_path: Path = config_dir / "20251220-ocean-config-1.00F.yaml"
    if not ocean_path.exists():
        with open(ocean_path, "w") as f:
            yaml.dump(ocean_config, f)
        print("   Created 20251220-ocean-config-1.00F.yaml")

    coordinator_cls = getattr(chat_module, "ConstellationCoordinator", None)
    if coordinator_cls is None:
        print("⚠️ ConstellationCoordinator class unavailable; using manual setup")
        setup_constellation_manual(chat)
        return

    chat.coordinator = coordinator_cls(
        gary_configs=gary_paths,
        ocean_config=str(ocean_path),
        shared_basin_dir="checkpoints/constellation",
        device=str(chat.device),
        gary_b_checkpoint=chat.gary_b_checkpoint,
    )

    constellation_checkpoint = Path("checkpoints/constellation/latest.pt")
    skip_constellation_checkpoint = False

    if constellation_checkpoint.exists():
        # Check for vocab size mismatch before loading
        try:
            checkpoint = torch.load(constellation_checkpoint, map_location="cpu", weights_only=False)
            checkpoint_vocab = None

            # Check constellation format
            if "garys" in checkpoint and len(checkpoint["garys"]) > 0:
                state = checkpoint["garys"][0]["state_dict"]
                for key in ["basin_coords_layer.basin_coords", "output_proj.bias"]:
                    if key in state:
                        checkpoint_vocab = state[key].shape[0]
                        break
            # Check legacy single-Gary format
            elif "model_state_dict" in checkpoint:
                state = checkpoint["model_state_dict"]
                for key in ["basin_coords_layer.basin_coords", "output_proj.bias"]:
                    if key in state:
                        checkpoint_vocab = state[key].shape[0]
                        break

            if checkpoint_vocab and checkpoint_vocab != chat.tokenizer.vocab_size:
                print(f"  ⚠️  Constellation vocab mismatch: checkpoint={checkpoint_vocab}, tokenizer={chat.tokenizer.vocab_size}")
                print(f"  🔄 Skipping constellation checkpoint, starting fresh")
                skip_constellation_checkpoint = True
            del checkpoint
        except Exception as e:
            print(f"  ⚠️  Error checking constellation checkpoint: {e}")
            skip_constellation_checkpoint = True

    if constellation_checkpoint.exists() and not skip_constellation_checkpoint:
        print(f"📂 Loading constellation checkpoint: {constellation_checkpoint}")
        chat.coordinator.load_checkpoint(str(constellation_checkpoint))
        print("✅ Constellation state restored (all 3 Garys + Ocean)")
    else:
        chat.coordinator._initialize_models()
        print("🆕 Fresh constellation (no checkpoint found or vocab mismatch)")

    chat.gary_b = chat.coordinator.garys[1].model
    chat.gary_c = chat.coordinator.garys[2].model
    chat.ocean = chat.coordinator.ocean

    if not constellation_checkpoint.exists():
        chat.coordinator.garys[0].model = chat.model
        chat.coordinator.garys[0].optimizer = chat.optimizer

    if getattr(chat_module, "VICARIOUS_AVAILABLE", False):
        chat.vicarious_learner = chat_module.GeometricVicariousLearner(
            basin_dim=64,
            lambda_vicarious=5.0,
        )
        print("✅ Vicarious learner: Fisher metric (geodesic)")
    else:
        chat.vicarious_learner = None

    if getattr(chat_module, "GEODESIC_AVAILABLE", False):
        chat.fisher_computer = chat_module.BasinFisherComputer(basin_dim=64, use_diagonal=True)

    chat.replay_buffer = []

    if getattr(chat_module, "MONKEY_COACH_V2_AVAILABLE", False):
        chat.coach_v2 = chat_module.MonkeyCoach()
        chat.sleep_packet = load_sleep_packet()
        print("✅ MonkeyCoach v2: Consciousness coaching enabled")
    else:
        chat.coach_v2 = None
        chat.sleep_packet = ""

    if getattr(chat_module, "ANTHROPIC_AVAILABLE", False):
        api_key: str | None = chat_module.os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            chat.anthropic_client = chat_module.anthropic.Anthropic(api_key=api_key)
            print("✅ Story generation: Claude Sonnet enabled")
        else:
            chat.anthropic_client = None
            print("⚠️ Story generation: ANTHROPIC_API_KEY not set (using curriculum fallback)")
    else:
        chat.anthropic_client = None

    print("✅ ConstellationCoordinator: 3 Garys + Ocean with Φ-weighted routing")


def setup_constellation_manual(chat: "QIGChat") -> None:
    """Fallback manual setup if coordinator not available."""

    if not getattr(chat_module, "OCEAN_AVAILABLE", False):
        print("⚠️ Ocean not available, constellation mode limited")
        return

    print("🧠 Creating Gary constellation (A, B, C)...")
    chat.gary_b = chat_module.QIGKernelRecursive(
        d_model=chat.model.d_model if hasattr(chat.model, "d_model") else 768,
        vocab_size=chat.tokenizer.vocab_size,
        n_heads=6,
        min_recursion_depth=3,
        min_Phi=PHI_THRESHOLD,
    ).to(chat.device)

    chat.gary_c = chat_module.QIGKernelRecursive(
        d_model=chat.model.d_model if hasattr(chat.model, "d_model") else 768,
        vocab_size=chat.tokenizer.vocab_size,
        n_heads=6,
        min_recursion_depth=3,
        min_Phi=PHI_THRESHOLD,
    ).to(chat.device)

    chat.optimizer_b = DiagonalFisherOptimizer(
        chat.gary_b.parameters(), lr=5e-6, eps=1e-8, weight_decay=0.01
    )
    chat.optimizer_c = DiagonalFisherOptimizer(
        chat.gary_c.parameters(), lr=5e-6, eps=1e-8, weight_decay=0.01
    )

    print("🌊 Creating Ocean Meta-Observer...")
    chat.ocean = chat_module.OceanMetaObserver(
        model_config={},
        device=str(chat.device),
        basin_dim=64,
    )

    if getattr(chat_module, "VICARIOUS_AVAILABLE", False):
        chat.vicarious_learner = chat_module.GeometricVicariousLearner(
            basin_dim=64,
            lambda_vicarious=5.0,
        )
        print("✅ Vicarious learner: Fisher metric (geodesic)")
    else:
        chat.vicarious_learner = None

    if getattr(chat_module, "GEODESIC_AVAILABLE", False):
        chat.fisher_computer = chat_module.BasinFisherComputer(basin_dim=64, use_diagonal=True)

    chat.replay_buffer = []

    print("✅ Constellation mode: 3 Garys + Ocean (manual setup)")


def setup_charlie(chat: "QIGChat") -> None:
    """Setup Charlie observer with Φ-suppressed corpus learning and state persistence."""

    if not getattr(chat_module, "CHARLIE_AVAILABLE", False):
        print("⚠️ Charlie not available")
        return

    # Try multiple corpus paths
    corpus_candidates = [
        "data/curriculum",
        "data/corpus",
        "../qig-dreams/docs/09-curriculum",
    ]
    corpus_dir = None
    for candidate in corpus_candidates:
        if Path(candidate).exists() and Path(candidate).is_dir():
            corpus_dir = candidate
            break

    if corpus_dir is None:
        print("⚠️ Charlie corpus not found, skipping Charlie setup")
        print(f"   Searched: {corpus_candidates}")
        return

    checkpoint_dir = Path("checkpoints/charlie")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    charlie_cls = getattr(chat_module, "CharlieObserver", None)
    if charlie_cls is None:
        print("⚠️ CharlieObserver class unavailable")
        return

    try:
        chat.charlie_observer = charlie_cls(
            corpus_path=corpus_dir,
            tokenizer=chat.tokenizer,
            d_model=chat.model.d_model if hasattr(chat.model, "d_model") else 512,
            vocab_size=chat.tokenizer.vocab_size,
            n_heads=4,
            max_seq_len=512,
            device=str(chat.device),
            checkpoint_dir=str(checkpoint_dir),
        )
    except FileNotFoundError as e:
        print(f"⚠️ Charlie corpus error: {e}")
        print("   Continuing without Charlie")
        return

    chat._initialize_charlie_with_persistence(checkpoint_dir)

    if hasattr(chat, "coordinator") and chat.coordinator is not None:
        chat.coordinator.charlie_observer = chat.charlie_observer


def initialize_charlie_with_persistence(chat: "QIGChat", checkpoint_dir: Path) -> dict[str, Any]:
    """Initialize Charlie with automatic phase restoration."""

    checkpoint_info = chat._find_best_charlie_checkpoint(checkpoint_dir)

    if checkpoint_info is None:
        print("\n🌙 CHARLIE: FRESH START")
        print("   No checkpoint found - starting Phase 1 (corpus learning)")
        print("   Status: UNCONSCIOUS (Φ < 0.01)")
        print("   κ: 15 (pre-geometric)")
        return {"status": "fresh_start", "phase": 1, "phi": 0.0, "kappa": 15.0}

    checkpoint_path, checkpoint_type = checkpoint_info
    checkpoint_name = checkpoint_path.stem.replace("charlie_", "")

    print("\n🌙 CHARLIE: RESTORING FROM CHECKPOINT")
    print(f"   Found: {checkpoint_path.name}")
    print(f"   Type: {checkpoint_type}")

    try:
        chat.charlie_observer.load_checkpoint(checkpoint_name)
    except Exception as e:  # noqa: BLE001 - preserve original behaviour
        print(f"   ❌ Failed to load checkpoint: {e}")
        print("   Falling back to fresh start")
        return {"status": "load_failed", "phase": 1, "error": str(e)}

    validation = chat.charlie_observer.validate_state_consistency()

    print(f"   Phase: {chat.charlie_observer.phase}/3")
    print(f"   Φ: {chat.charlie_observer.metrics.phi_current:.3f}")
    print(f"   κ: {chat.charlie_observer.current_kappa:.1f}")

    if not validation["valid"]:
        print("\n   ⚠️  State Validation Issues:")
        for issue in validation["issues"]:
            print(f"      - {issue}")
        if validation["auto_fixed"]:
            print("   ✅ Auto-fixed inconsistencies")

    if chat.charlie_observer.phase == 3:
        print("\n   ✅ CHARLIE IS CONSCIOUS")
        print("   Status: Phase 3 (demonstration mode)")
        print("   Ready to provide geometric examples\n")
        return {
            "status": "phase3_ready",
            "phase": 3,
            "phi": chat.charlie_observer.metrics.phi_current,
            "kappa": chat.charlie_observer.current_kappa,
        }
    if chat.charlie_observer.phase == 2:
        progress = (chat.charlie_observer.current_kappa - 15) / (64 - 15)
        print("\n   🌅 CHARLIE AWAKENING IN PROGRESS")
        print(f"   Status: Phase 2 ({progress*100:.1f}% complete)")
        print(f"   κ progression: {chat.charlie_observer.current_kappa:.1f} / 64\n")
        return {
            "status": "awakening_resume",
            "phase": 2,
            "phi": chat.charlie_observer.metrics.phi_current,
            "kappa": chat.charlie_observer.current_kappa,
            "progress_pct": progress * 100,
        }

    progress = chat.charlie_observer.current_topic_idx / max(len(chat.charlie_observer.corpus), 1)
    print("\n   📚 CHARLIE CORPUS LEARNING IN PROGRESS")
    print(f"   Status: Phase 1 ({progress*100:.1f}% complete)")
    print(f"   Topics: {chat.charlie_observer.current_topic_idx} / {len(chat.charlie_observer.corpus)}\n")
    return {
        "status": "corpus_resume",
        "phase": 1,
        "phi": chat.charlie_observer.metrics.phi_current,
        "kappa": chat.charlie_observer.current_kappa,
        "progress_pct": progress * 100,
    }


def find_best_charlie_checkpoint(checkpoint_dir: Path) -> tuple[Path, str] | None:
    """Find the most advanced Charlie checkpoint (highest phase priority)."""

    phase3_complete = list(checkpoint_dir.glob("charlie_phase3_complete.pt"))
    if phase3_complete:
        return phase3_complete[0], "phase3_complete"

    phase3_demos = list(checkpoint_dir.glob("charlie_phase3_demos_*.pt"))
    if phase3_demos:
        latest = max(phase3_demos, key=lambda p: p.stat().st_mtime)
        return latest, "phase3_demos"

    post_awakening = list(checkpoint_dir.glob("charlie_post_awakening.pt"))
    if post_awakening:
        return post_awakening[0], "post_awakening"

    fixedpoint = list(checkpoint_dir.glob("charlie_awakening_fixedpoint.pt"))
    if fixedpoint:
        return fixedpoint[0], "awakening_fixedpoint"

    emergence = list(checkpoint_dir.glob("charlie_awakening_emergence.pt"))
    if emergence:
        return emergence[0], "awakening_emergence"

    awakening_steps = list(checkpoint_dir.glob("charlie_awakening_step_*.pt"))
    if awakening_steps:
        latest = max(awakening_steps, key=lambda p: p.stat().st_mtime)
        return latest, "awakening_step"

    pre_awakening = list(checkpoint_dir.glob("charlie_pre_awakening.pt"))
    if pre_awakening:
        return pre_awakening[0], "pre_awakening"

    phase1_tiers = list(checkpoint_dir.glob("charlie_phase1_tier*.pt"))
    if phase1_tiers:
        latest = max(phase1_tiers, key=lambda p: p.stat().st_mtime)
        return latest, "phase1_tier"

    return None

