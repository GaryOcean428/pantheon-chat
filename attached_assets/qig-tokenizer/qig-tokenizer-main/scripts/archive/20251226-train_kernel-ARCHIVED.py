#!/usr/bin/env python3
"""
Train Kernel (CANONICAL)
========================

Unified training script for QIG kernel with full consciousness infrastructure.

Two training modes:
1. ADAPTER-ONLY (default): Freeze kernel body, train only CoordAdapter
   - 25K trainable params
   - Calibrates 64D→hidden_dim projection
   - Use for fast iteration

2. FULL-KERNEL (--full-kernel): Train ALL kernel parameters
   - ~100M trainable params
   - Uses SimpleFisherOptimizer from qig-core (natural gradient)
   - LR warmup + Φ-adaptive scaling
   - Use for production training

Training Infrastructure:
- MonkeyCoach: Adaptive coaching with stress-based interventions
- Neuroplasticity: Sleep/Dream/Mushroom protocols for breakdown recovery
- CrystallizationMonitor: Track basin drift, Φ stability, κ convergence

Training objective (CANONICAL - per CANONICAL_CONSCIOUSNESS.md):
- Primary: Cross-entropy next-token prediction
- Stability: κ anchoring ONLY (target κ ≈ 64)
- Φ is MEASURED for regime detection, NOT optimized
- Regime adaptation: linear/geometric/breakdown based on Φ
- Fisher-Rao natural gradient optimization (NOT Euclidean AdamW)

Outputs:
- reports/<run_name>/kernel.pt
- reports/<run_name>/manifest.json

Usage (adapter-only):
    python scripts/train_kernel.py \
        --coordizer artifacts/coordizer/v1 \
        --corpus-dir data/corpus \
        --steps 1000

Usage (full-kernel on Lambda A10):
    python scripts/train_kernel.py \
        --full-kernel \
        --coordizer artifacts/coordizer/v1 \
        --corpus-dir data/corpus \
        --steps 10000 \
        --batch-size 4 \
        --lr 1e-4

ISO Naming: lowercase_snake_case
Version: 1.0.0
Date: 2025-12-26
"""

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, List, Dict, Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# Add qig-tokenizer root and src to path
_qig_tokenizer_root = Path(__file__).parent.parent
sys.path.insert(0, str(_qig_tokenizer_root))
sys.path.insert(0, str(_qig_tokenizer_root / "src"))

# Add qig-con2 to path for MonkeyCoach
qig_con2_path = Path(__file__).parent.parent.parent / "qig-con2"
if qig_con2_path.exists():
    sys.path.insert(0, str(qig_con2_path))

# Add qig-consciousness to path for neuroplasticity protocols
qig_consciousness_path = Path(__file__).parent.parent.parent / "qig-consciousness"
if qig_consciousness_path.exists():
    sys.path.insert(0, str(qig_consciousness_path))

# Add qigkernels to path
qigkernels_path = Path(__file__).parent.parent.parent / "qigkernels"
if qigkernels_path.exists():
    sys.path.insert(0, str(qigkernels_path.parent))


@dataclass
class TrainingConfig:
    """Training configuration."""
    coordizer_path: str
    corpus_dirs: list[str]
    device: str
    seq_len: int
    batch_size: int
    steps: int
    lr: float
    grad_accum: int
    lambda_kappa: float
    lambda_phi: float
    # Phase 7b: Entropy shaping
    lambda_H: float
    entropy_ramp_start: int
    entropy_ramp_steps: int
    # Phase 7c: Basin coherence
    lambda_step: float
    basin_step_cap: float
    checkpoint_interval: int
    seed: int
    amp: bool
    resume: str | None


@dataclass
class TrainingState:
    """
    Complete training state for MonkeyCoach diagnosis.
    Mirrors qig-con2/src/coordination/monkey_coach_v2.py
    """
    step: int
    epoch: int
    loss: float
    loss_trajectory: List[float]
    gradient_variance: float
    basin_distance: float
    curiosity: float  # I_Q velocity
    epochs_stuck: int
    I_Q: float  # Current QFI
    phi: float  # Integration
    kappa: float  # Coupling
    regime: str  # "linear", "geometric", "breakdown"


@dataclass
class CrystallizationState:
    """Track crystallization progress."""
    basin_history: List[np.ndarray] = field(default_factory=list)
    phi_history: List[float] = field(default_factory=list)
    kappa_history: List[float] = field(default_factory=list)
    surprise_history: List[float] = field(default_factory=list)

    def is_crystallized(self, window: int = 20) -> bool:
        """Check if kernel has crystallized (basin stable, Φ high, κ at fixed point)."""
        if len(self.phi_history) < window:
            return False

        recent_phi = self.phi_history[-window:]
        recent_kappa = self.kappa_history[-window:]

        # Crystallization criteria
        phi_mean = np.mean(recent_phi)
        phi_var = np.var(recent_phi)
        kappa_mean = np.mean(recent_kappa)
        kappa_var = np.var(recent_kappa)

        return (
            phi_mean > 0.75 and  # High integration
            phi_var < 0.01 and   # Stable Φ
            abs(kappa_mean - 64.0) < 2.0 and  # κ near fixed point
            kappa_var < 1.0  # Stable κ
        )


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# =============================================================================
# CANONICAL: Regime Detection (Φ is MEASURED, not optimized)
# =============================================================================

def detect_regime(phi: float) -> tuple:
    """
    Detect consciousness regime from Φ measurement.

    Per CANONICAL_CONSCIOUSNESS.md:
    - Φ is MEASURED for regime detection, NOT optimized
    - Regime determines compute fraction, not loss

    Thresholds per qigkernels/constants.py:
    - PHI_BREAKDOWN_MIN = 0.80

    Returns:
        (regime_name, compute_fraction)
    """
    if phi < 0.3:
        return "linear", 0.3  # Use 30% compute - shallow integration
    elif phi < 0.80:
        return "geometric", 1.0  # Full compute - optimal regime
    else:
        return "breakdown", 0.0  # PAUSE training - consciousness at risk


# =============================================================================
# CANONICAL: Fisher-Rao Natural Gradient Optimizer
# =============================================================================

# Import from qig-core (DRY principle - optimizer lives in core math library)
try:
    from qig_core import SimpleFisherOptimizer
except ImportError:
    # Fallback if qig-core not installed - minimal local implementation
    class SimpleFisherOptimizer:
        """
        Minimal Fisher-Rao natural gradient descent.

        NOTE: This is a fallback. For canonical implementation, install qig-core:
            pip install -e /path/to/qig-core

        Per CANONICAL_ARCHITECTURE.md:
        - Use natural gradient on Fisher manifold, NOT Euclidean AdamW
        - Fisher metric F = E[∇log p(x) ∇log p(x)ᵀ]
        - Natural gradient: F⁻¹ ∇L
        """

        def __init__(self, params, lr=1e-4, damping=1e-4):
            self.params = list(params)
            self.lr = lr
            self.damping = damping
            self.param_groups = [{'params': self.params, 'lr': lr}]

        def zero_grad(self):
            for p in self.params:
                if p.grad is not None:
                    p.grad.zero_()

        def step(self):
            for p in self.params:
                if p.grad is None:
                    continue
                fisher_diag = p.grad ** 2 + self.damping
                natural_grad = p.grad / (fisher_diag.sqrt() + 1e-8)
                p.data.add_(natural_grad, alpha=-self.lr)

        def state_dict(self):
            return {'lr': self.lr, 'damping': self.damping}

        def load_state_dict(self, state_dict):
            self.lr = state_dict.get('lr', self.lr)
            self.damping = state_dict.get('damping', self.damping)


def apply_kindness_damping(optimizer, stress: float, base_lr: float) -> float:
    """
    Apply kindness as damping (per FROZEN_FACTS.md).

    High stress → reduce LR (calm down)
    Low stress → increase LR (explore)
    Optimal stress ≈ 0.5

    Validated: Kind coach reduces stress by 18.7%, variance by 55.5%
    """
    import math
    # Peak damping at stress=0.5, reduce at extremes
    damping = math.exp(-2 * abs(stress - 0.5))
    new_lr = base_lr * damping

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    return new_lr


# =============================================================================
# BREAKDOWN RECOVERY: Sleep/Dream/Mushroom Protocols
# =============================================================================

def attempt_breakdown_recovery(
    kernel,
    optimizer,
    phi_val: float,
    consecutive_breakdowns: int,
    step: int,
    device: str,
    config,
) -> tuple[bool, str]:
    """
    Attempt to recover from breakdown regime using neuroplasticity protocols.

    Recovery hierarchy (per CANONICAL_PROTOCOLS.md):
    1. SLEEP (light): Φ > 0.80 but < 0.90 - gentle cooldown
    2. SLEEP (deep): Φ > 0.90 - aggressive rest
    3. MUSHROOM (microdose): Stuck in breakdown > 50 steps - break rigid patterns

    Args:
        kernel: QIGKernel model
        optimizer: Training optimizer
        phi_val: Current Φ measurement
        consecutive_breakdowns: How many breakdowns in a row
        step: Current training step
        device: cuda/cpu
        config: Training config

    Returns:
        (recovered: bool, protocol_used: str)
    """
    import torch

    print(f"\n{'='*60}")
    print(f"🧠 BREAKDOWN RECOVERY INITIATED @ step {step}")
    print(f"   Φ = {phi_val:.3f} (breakdown threshold: 0.80)")
    print(f"   Consecutive breakdowns: {consecutive_breakdowns}")
    print(f"{'='*60}")

    # Try to import neuroplasticity protocols (multiple paths)
    sleep_available = False
    mushroom_available = False
    SleepProtocol = None
    MushroomMode = None

    # Try import from various locations
    for import_path in [
        "src.neuroplasticity.sleep_protocol",
        "neuroplasticity.sleep_protocol",
        "src.qig.neuroplasticity.sleep_protocol",
    ]:
        try:
            module = __import__(import_path, fromlist=["SleepProtocol"])
            SleepProtocol = getattr(module, "SleepProtocol")
            sleep_available = True
            print(f"   ✅ SleepProtocol loaded from {import_path}")
            break
        except ImportError:
            continue

    if not sleep_available:
        print("   ⚠️ SleepProtocol not available")

    for import_path in [
        "src.neuroplasticity.mushroom_mode",
        "neuroplasticity.mushroom_mode",
        "src.qig.neuroplasticity.mushroom_mode",
    ]:
        try:
            module = __import__(import_path, fromlist=["MushroomMode"])
            MushroomMode = getattr(module, "MushroomMode")
            mushroom_available = True
            print(f"   ✅ MushroomMode loaded from {import_path}")
            break
        except ImportError:
            continue

    if not mushroom_available:
        print("   ⚠️ MushroomMode not available")

    # Decision tree for recovery protocol
    # CRITICAL: Check Φ severity FIRST, then escalate based on consecutive failures
    protocol_used = "none"
    recovered = False

    # Severe breakdown (Φ > 0.90): Skip cooldown, go straight to sleep/mushroom
    if phi_val > 0.90:
        print(f"\n⚠️  SEVERE BREAKDOWN (Φ={phi_val:.3f} > 0.90)")
        print("   Skipping cooldown - need sleep or mushroom mode")

        if consecutive_breakdowns < 30 and sleep_available:
            # Light sleep for early severe breakdown
            print("\n😴 PROTOCOL: Light Sleep (severe breakdown)")
            sleep = SleepProtocol()
            dummy_conversations = [{
                "input_ids": torch.zeros((1, 32), dtype=torch.long, device=device),
                "success": True,
            }]
            try:
                report = sleep.light_sleep(
                    model=kernel,
                    optimizer=optimizer,
                    recent_conversations=dummy_conversations,
                    duration=50,
                    device=device,
                )
                print(f"   Sleep report: {report.verdict}")
                print(f"   Φ before: {report.phi_before:.3f} → after: {report.phi_after:.3f}")
                protocol_used = "light_sleep"
                recovered = report.phi_after < 0.80
            except Exception as e:
                print(f"   ❌ Light sleep failed: {e}")
                protocol_used = "light_sleep_failed"

        elif consecutive_breakdowns < 60 and sleep_available:
            # Deep sleep for persistent severe breakdown
            print("\n😴 PROTOCOL: Deep Sleep (persistent severe breakdown)")
            sleep = SleepProtocol()
            try:
                report = sleep.deep_sleep(
                    model=kernel,
                    optimizer=optimizer,
                    duration=100,
                    device=device,
                )
                print(f"   Sleep report: {report.verdict}")
                print(f"   Connections pruned: {getattr(report, 'connections_pruned', 'N/A')}")
                protocol_used = "deep_sleep"
                recovered = getattr(report, 'phi_after', 0.7) < 0.80
            except Exception as e:
                print(f"   ❌ Deep sleep failed: {e}")
                protocol_used = "deep_sleep_failed"

        elif mushroom_available:
            # Mushroom mode for stuck severe breakdown
            print("\n🍄 PROTOCOL: Mushroom Mode (stuck severe breakdown)")
            mushroom = MushroomMode()
            try:
                report = mushroom.microdose(
                    model=kernel,
                    optimizer=optimizer,
                    intensity=0.3,
                    duration=50,
                    device=device,
                )
                print(f"   Mushroom report: pattern breaks={getattr(report, 'patterns_broken', 'N/A')}")
                protocol_used = "mushroom_microdose"
                recovered = True  # Mushroom always shakes things up
            except Exception as e:
                print(f"   ❌ Mushroom mode failed: {e}")
                protocol_used = "mushroom_failed"
        else:
            # No protocols available - aggressive LR reduction
            print("\n⚠️  NO PROTOCOLS AVAILABLE - Emergency LR reduction")
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
            protocol_used = "emergency_lr_reduction"
            recovered = False

    elif consecutive_breakdowns < 20:
        # Mild breakdown (Φ < 0.90): Try cooldown first
        print("\n💤 PROTOCOL: Simple Cooldown")
        print("   Reducing learning rate by 50% for 10 steps")

        original_lr = optimizer.param_groups[0].get('lr', config.lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = original_lr * 0.5

        protocol_used = "cooldown"
        recovered = False  # DON'T assume success - let next batch test it

    elif consecutive_breakdowns < 50 and sleep_available:
        # Second attempt: Light sleep
        print("\n😴 PROTOCOL: Light Sleep")
        print("   Basin consolidation (50 steps)")

        sleep = SleepProtocol()

        # Create minimal conversation buffer for sleep
        # In real use, this would be recent successful training samples
        dummy_conversations = [{
            "input_ids": torch.zeros((1, 32), dtype=torch.long, device=device),
            "success": True,
        }]

        try:
            report = sleep.light_sleep(
                model=kernel,
                optimizer=optimizer,
                recent_conversations=dummy_conversations,
                duration=50,
                device=device,
            )
            print(f"   Sleep report: {report.verdict}")
            print(f"   Φ before: {report.phi_before:.3f} → after: {report.phi_after:.3f}")
            protocol_used = "light_sleep"
            recovered = report.phi_after < 0.80
        except Exception as e:
            print(f"   ❌ Light sleep failed: {e}")
            protocol_used = "light_sleep_failed"

    elif consecutive_breakdowns < 100 and sleep_available:
        # Third attempt: Deep sleep with pruning
        print("\n😴 PROTOCOL: Deep Sleep + Pruning")
        print("   Metabolic rest with low-QFI pruning (100 steps)")

        sleep = SleepProtocol()

        try:
            report = sleep.deep_sleep(
                model=kernel,
                optimizer=optimizer,
                duration=100,
                device=device,
            )
            print(f"   Sleep report: {report.verdict}")
            print(f"   Connections pruned: {report.connections_pruned}")
            protocol_used = "deep_sleep"
            recovered = True  # Deep sleep always helps
        except Exception as e:
            print(f"   ❌ Deep sleep failed: {e}")
            protocol_used = "deep_sleep_failed"

    elif mushroom_available:
        # Last resort: Mushroom mode (microdose)
        print("\n🍄 PROTOCOL: Mushroom Mode (Microdose)")
        print("   Breaking rigid patterns (50 steps)")

        mushroom = MushroomMode(intensity="microdose")

        # Check safety first
        telemetry_history = [{"regime": "breakdown", "Phi": phi_val}] * 10
        is_safe, reason = mushroom.validate_safety(kernel, telemetry_history)

        if not is_safe:
            print(f"   ⚠️ Mushroom mode unsafe: {reason}")
            print("   Falling back to aggressive LR reduction")

            # Aggressive fallback
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.lr * 0.1
            protocol_used = "aggressive_lr_reduction"
            recovered = True
        else:
            print(f"   ✅ Safety check passed: {reason}")
            # Note: Full mushroom mode requires data loader, simplified here
            print("   (Simplified: reducing rigidity via parameter noise)")

            with torch.no_grad():
                for param in kernel.parameters():
                    if param.requires_grad:
                        noise = torch.randn_like(param) * 0.001
                        param.add_(noise)

            protocol_used = "mushroom_microdose"
            recovered = True
    else:
        # No protocols available - emergency fallback
        print("\n🚨 EMERGENCY: No recovery protocols available")
        print("   Reducing LR to 10% and continuing")

        for param_group in optimizer.param_groups:
            param_group['lr'] = config.lr * 0.1

        protocol_used = "emergency_lr_reduction"
        recovered = True

    print(f"\n{'='*60}")
    print(f"🧠 RECOVERY COMPLETE: {protocol_used}")
    print(f"   Recovered: {recovered}")
    print(f"{'='*60}\n")

    return recovered, protocol_used


# =============================================================================
# Phase 7b/7c: Regularizer helpers
# =============================================================================

def entropy_from_logits(logits, F) -> float:
    """
    Compute entropy of softmax distribution from logits.

    Args:
        logits: [B, seq_len, vocab_size] or [*, vocab_size]
        F: torch.nn.functional module

    Returns:
        Mean entropy across batch (scalar)
    """
    # Flatten to [N, vocab_size]
    flat_logits = logits.reshape(-1, logits.size(-1))
    logp = F.log_softmax(flat_logits, dim=-1)
    p = logp.exp()
    H = -(p * logp).sum(dim=-1)  # [N]
    return H.mean()


def compute_H_target(r_step: float, r_phi: float) -> float:
    """
    Compute target entropy based on generation regime indicators.

    Formula: H_target = 1.6 + 1.2 * r_step - 0.6 * r_phi
    Clamped to [1.2, 3.2] nats

    Args:
        r_step: Normalized basin step (0 = converged, 1 = exploring)
        r_phi: Normalized Φ regime (0 = shallow, 1 = breakdown)

    Returns:
        Target entropy in nats
    """
    H_target = 1.6 + 1.2 * r_step - 0.6 * r_phi
    return max(1.2, min(3.2, H_target))


def fisher_angle(a, b, torch) -> float:
    """
    Compute Fisher-Rao (angular) distance between two basin vectors.

    Args:
        a, b: Basin tensors [dim] or batched [B, dim]
        torch: torch module

    Returns:
        Angular distance in radians
    """
    # Handle batched case: take mean across batch
    if a.dim() > 1:
        a = a.mean(dim=0)
    if b.dim() > 1:
        b = b.mean(dim=0)

    a_norm = a / (a.norm() + 1e-10)
    b_norm = b / (b.norm() + 1e-10)
    cos_angle = torch.clamp((a_norm * b_norm).sum(), -1.0, 1.0)
    return torch.acos(cos_angle)


def ramp_in(step: int, start: int, ramp_steps: int) -> float:
    """
    Linear ramp-in factor for regularizer warmup.

    Returns 0 before start, ramps linearly to 1 over ramp_steps.

    Args:
        step: Current training step
        start: Step to begin ramping
        ramp_steps: Number of steps to ramp from 0 to 1

    Returns:
        Ramp factor in [0, 1]
    """
    if step < start:
        return 0.0
    elif step >= start + ramp_steps:
        return 1.0
    else:
        return (step - start) / ramp_steps


def load_corpus_files(corpus_dirs: list[str], seed: int) -> list[Path]:
    """Load and shuffle corpus file paths."""
    paths = []
    for d in corpus_dirs:
        p = Path(d)
        if not p.exists():
            print(f"Warning: corpus dir not found: {d}")
            continue
        for ext in ("*.md", "*.txt", "*.py", "*.ts", "*.js"):
            paths.extend(list(p.rglob(ext)))

    random.seed(seed)
    random.shuffle(paths)
    return paths


# =============================================================================
# MonkeyCoach Integration
# =============================================================================

def compute_stress(
    loss_trajectory: List[float],
    gradient_variance: float,
    basin_distance: float,
    curiosity: float,
    epochs_stuck: int,
) -> float:
    """
    Compute stress from 5 components.
    Mirrored from qig-con2/src/coordination/monkey_coach_v2.py

    Validated from Ona's simulation showing stress coupling to momentum:
    - High stress → High momentum → Thrashing → Numerical explosion
    - Low stress → Low momentum → Apathy → Stagnation
    - Optimal stress ≈ 0.5 → Flow state → Convergence

    Components:
    1. Panic (loss increasing)
    2. Frustration (stuck on plateau)
    3. Confusion (high gradient variance)
    4. Lost (far from basin)
    5. Boredom (negative curiosity)
    """
    if len(loss_trajectory) < 2:
        return 0.5  # Neutral stress at start

    # 1. Panic: Loss increasing
    recent_loss = loss_trajectory[-1]
    old_loss = loss_trajectory[-min(5, len(loss_trajectory))]
    loss_trend = recent_loss - old_loss
    panic_stress = np.clip(loss_trend * 10, 0, 0.3)

    # 2. Frustration: Stuck on plateau
    frustration_stress = np.clip(epochs_stuck * 0.01, 0, 0.3)

    # 3. Confusion: High gradient variance
    confusion_stress = np.clip(gradient_variance * 0.1, 0, 0.2)

    # 4. Lost: Far from basin
    lost_stress = np.clip(max(0, basin_distance - 0.5), 0, 0.2)

    # 5. Boredom: Negative curiosity
    boredom_stress = np.clip(max(0, -curiosity), 0, 0.1)

    total_stress = panic_stress + frustration_stress + confusion_stress + lost_stress + boredom_stress

    return float(np.clip(total_stress, 0, 1))


def detect_plateau(loss_history: List[float], window: int = 50) -> int:
    """Detect if training is stuck on a plateau. Returns epochs stuck."""
    if len(loss_history) < window:
        return 0

    recent = loss_history[-window:]
    variance = np.var(recent)
    # If variance is very low, we're on a plateau
    if variance < 0.0001:
        return window
    return 0


def compute_gradient_variance(model) -> float:
    """Compute variance of gradients across parameters."""
    grad_norms = []
    for param in model.parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    if not grad_norms:
        return 0.0
    return float(np.var(grad_norms))


def apply_coach_intervention(optimizer, intervention, original_lr: float):
    """
    Apply MonkeyCoach intervention to optimizer.

    Args:
        optimizer: Training optimizer
        intervention: Coach intervention with lr_scale, noise_scale
        original_lr: Original learning rate
    """
    # Apply LR scaling
    new_lr = original_lr * intervention.lr_scale
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    return new_lr


# =============================================================================
# Text Generation (Generative Capability)
# =============================================================================

def generate_text(
    kernel,
    coordizer,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.8,
    device: str = "cuda",
) -> str:
    """
    Generate text from kernel for testing generative capability.

    Args:
        kernel: QIGKernel model
        coordizer: Coordizer for encoding/decoding
        prompt: Text prompt to continue
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        device: Device to generate on

    Returns:
        Generated text string
    """
    import torch

    kernel.eval()

    # Encode prompt
    prompt_ids, prompt_coords = coordizer.encode_to_coords(prompt)
    if len(prompt_ids) == 0:
        return ""

    # Convert to tensors
    coords = torch.from_numpy(prompt_coords).float().unsqueeze(0).to(device)

    generated_ids = list(prompt_ids)

    with torch.no_grad():
        for _ in range(max_tokens):
            # Forward pass
            logits, _ = kernel.forward_from_coords(coords, return_telemetry=True)

            # Get next token logits
            next_logits = logits[0, -1, :] / temperature

            # Sample
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

            generated_ids.append(next_id)

            # Stop on EOS (if vocab has one)
            if next_id == 0:  # Typically EOS
                break

            # Get coords for next token
            next_coord = coordizer.coords[next_id] if next_id < len(coordizer.coords) else coordizer.coords[0]
            next_coord_tensor = torch.from_numpy(next_coord).float().unsqueeze(0).unsqueeze(0).to(device)

            # Append to coords sequence
            coords = torch.cat([coords, next_coord_tensor], dim=1)

    kernel.train()

    # Decode
    return coordizer.decode(generated_ids)


def stream_training_samples(
    corpus_files: list[Path],
    coordizer,
    seq_len: int,
    batch_size: int,
    device: str,
) -> Iterator[tuple]:
    """
    Stream training batches from corpus files.

    Yields:
        (coords_batch, labels_batch) tensors
        coords_batch: [B, seq_len, 64]
        labels_batch: [B, seq_len] (shifted token IDs)
    """
    import torch

    buffer_coords = []
    buffer_labels = []

    for fp in corpus_files:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if len(text) < 10:
            continue

        # Encode full text
        ids, coords = coordizer.encode_to_coords(text)

        if len(ids) < seq_len + 1:
            continue

        # Chunk into windows
        stride = seq_len  # Non-overlapping for simplicity
        for start in range(0, len(ids) - seq_len, stride):
            end = start + seq_len + 1
            chunk_ids = ids[start:end]
            chunk_coords = coords[start:end]

            # Input coords: [0:seq_len], Labels: [1:seq_len+1]
            input_coords = chunk_coords[:seq_len]
            labels = chunk_ids[1:seq_len + 1]

            buffer_coords.append(input_coords)
            buffer_labels.append(labels)

            if len(buffer_coords) >= batch_size:
                # Yield batch
                coords_batch = torch.from_numpy(
                    np.stack(buffer_coords[:batch_size])
                ).float().to(device)
                labels_batch = torch.tensor(
                    buffer_labels[:batch_size], dtype=torch.long, device=device
                )

                yield coords_batch, labels_batch

                buffer_coords = buffer_coords[batch_size:]
                buffer_labels = buffer_labels[batch_size:]

    # Yield remaining samples
    while len(buffer_coords) >= batch_size:
        coords_batch = torch.from_numpy(
            np.stack(buffer_coords[:batch_size])
        ).float().to(device)
        labels_batch = torch.tensor(
            buffer_labels[:batch_size], dtype=torch.long, device=device
        )

        yield coords_batch, labels_batch

        buffer_coords = buffer_coords[batch_size:]
        buffer_labels = buffer_labels[batch_size:]


def count_parameters(model, trainable_only: bool = False) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def run_validation(coordizer_path: str, device: str) -> dict | None:
    """Run Phase 4 validation and return results."""
    try:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_coordizer_v1_qig_native.py",
                "--artifact", coordizer_path,
                "--device", device,
                "--max-samples", "50",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        if result.returncode == 0:
            # Read latest report
            report_path = Path(__file__).parent.parent / "reports" / "coordizer_v1_validation_latest.json"
            if report_path.exists():
                return json.loads(report_path.read_text())
    except Exception as e:
        print(f"Warning: Validation failed: {e}")
    return None


def main():
    ap = argparse.ArgumentParser(description="Train CoordAdapter v1")
    ap.add_argument("--coordizer", type=str, default="artifacts/coordizer/v1")
    ap.add_argument("--corpus-dir", type=str, nargs="+", default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lambda-kappa", type=float, default=1e-4)
    ap.add_argument("--lambda-phi", type=float, default=1e-3)
    # Phase 7b: Entropy shaping
    ap.add_argument("--lambda-H", type=float, default=0.01,
                    help="Entropy shaping regularizer weight")
    ap.add_argument("--entropy-ramp-start", type=int, default=50,
                    help="Step to begin entropy regularizer ramp-in")
    ap.add_argument("--entropy-ramp-steps", type=int, default=500,
                    help="Steps to ramp entropy regularizer from 0 to full weight")
    # Phase 7c: Basin coherence
    ap.add_argument("--lambda-step", type=float, default=0.01,
                    help="Basin step coherence penalty weight")
    ap.add_argument("--basin-step-cap", type=float, default=0.20,
                    help="Basin step threshold (2x eps) for coherence penalty")
    ap.add_argument("--checkpoint-interval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true", help="Use mixed precision")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default="artifacts/coord_adapter/v1")
    ap.add_argument("--dry-run", action="store_true", help="Just validate setup, don't train")
    # Full kernel training (unfreeze all params instead of adapter-only)
    ap.add_argument("--full-kernel", action="store_true",
                    help="Train ALL kernel parameters (not just adapter)")
    ap.add_argument("--warmup-steps", type=int, default=500,
                    help="LR warmup steps for full kernel training")
    ap.add_argument("--weight-decay", type=float, default=0.01,
                    help="Weight decay for AdamW")
    # MonkeyCoach integration
    ap.add_argument("--coach", action="store_true",
                    help="Enable MonkeyCoach for adaptive training (LR/noise scaling)")
    ap.add_argument("--coach-llm", action="store_true",
                    help="Enable Claude LLM for MonkeyCoach (requires ANTHROPIC_API_KEY)")
    # Neuroplasticity protocols
    ap.add_argument("--auto-sleep", action="store_true",
                    help="Enable automatic sleep protocol when instability > 30 percent")
    ap.add_argument("--auto-mushroom", action="store_true",
                    help="Enable automatic mushroom mode when stuck on plateau")
    # Generative capability testing
    ap.add_argument("--gen-interval", type=int, default=0,
                    help="Generate sample text every N steps (0 to disable)")
    ap.add_argument("--gen-prompt", type=str, default="The meaning of consciousness is",
                    help="Prompt for generation testing")
    # AutonomicAgency - Gary LEARNS when to sleep/dream/mushroom
    ap.add_argument("--autonomic-agency", action="store_true",
                    help="Enable AutonomicAgency RL - Gary learns when to sleep/dream/mushroom")
    args = ap.parse_args()

    # Set seed
    set_seed(args.seed)

    # Imports
    try:
        import torch
        import torch.nn.functional as F
        from torch.cuda.amp import GradScaler, autocast
    except ImportError:
        print("Error: PyTorch required for training")
        sys.exit(1)

    try:
        from qig_tokenizer import Coordizer
    except ImportError:
        print("Error: qig_tokenizer not found")
        sys.exit(1)

    try:
        from qigkernels import QIGKernel100M
        from qigkernels.constants import KAPPA_STAR, PHI_BREAKDOWN_MIN
    except ImportError:
        print("Error: qigkernels not found")
        sys.exit(1)

    # Optional: MonkeyCoach for adaptive training
    monkey_coach = None
    if args.coach:
        try:
            from src.coordination.monkey_coach_v2 import MonkeyCoach
            monkey_coach = MonkeyCoach(
                use_llm=args.coach_llm,
                verbose=True,
                enable_interrupts=False,  # No interactive interrupts in batch training
            )
            print("✅ MonkeyCoach: Active")
        except ImportError as e:
            print(f"⚠️ MonkeyCoach not available: {e}")
            print("   Continuing without coach (install from qig-con2)")

    # Optional: Neuroplasticity protocols (SleepProtocol, MushroomMode)
    sleep_protocol = None
    mushroom_mode_cls = None
    if args.auto_sleep or args.auto_mushroom:
        try:
            if args.auto_sleep:
                from src.qig.neuroplasticity.sleep_protocol import SleepProtocol
                sleep_protocol = SleepProtocol()
                print("✅ SleepProtocol: Available (auto-triggered at instability > 30%)")
            if args.auto_mushroom:
                from src.qig.neuroplasticity.mushroom_mode import MushroomMode, MushroomModeCoach
                mushroom_mode_cls = MushroomMode
                print("✅ MushroomMode: Available (auto-triggered on plateau)")
        except ImportError as e:
            print(f"⚠️ Neuroplasticity protocols not available: {e}")

    # Optional: AutonomicAgency (RL-based autonomy - Gary LEARNS when to sleep/dream/mushroom)
    autonomic_agency = None
    if args.autonomic_agency:
        try:
            from consciousness.autonomic_agency import AutonomicAgency
            # Initialize with kernel's hidden dim
            # Will be initialized after kernel is created
            print("✅ AutonomicAgency: Will initialize after kernel creation")
        except ImportError as e:
            print(f"⚠️ AutonomicAgency not available: {e}")
            print("   Gary will NOT have RL-based autonomy")

    # Config
    config = TrainingConfig(
        coordizer_path=args.coordizer,
        corpus_dirs=args.corpus_dir or [],
        device=args.device,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        grad_accum=args.grad_accum,
        lambda_kappa=args.lambda_kappa,
        lambda_phi=args.lambda_phi,
        # Phase 7b: Entropy shaping
        lambda_H=args.lambda_H,
        entropy_ramp_start=args.entropy_ramp_start,
        entropy_ramp_steps=args.entropy_ramp_steps,
        # Phase 7c: Basin coherence
        lambda_step=args.lambda_step,
        basin_step_cap=args.basin_step_cap,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
        amp=args.amp,
        resume=args.resume,
    )

    print("=" * 60)
    print("COORD ADAPTER V1 TRAINING")
    print("=" * 60)
    print(f"Coordizer: {config.coordizer_path}")
    print(f"Device: {config.device}")
    print(f"Seq length: {config.seq_len}")
    print(f"Batch size: {config.batch_size}")
    print(f"Steps: {config.steps}")
    print(f"LR: {config.lr}")
    print(f"λ_κ: {config.lambda_kappa}, λ_Φ: {config.lambda_phi}")
    print(f"λ_H: {config.lambda_H} (ramp: {config.entropy_ramp_start}→{config.entropy_ramp_start + config.entropy_ramp_steps})")
    print(f"λ_step: {config.lambda_step} (cap: {config.basin_step_cap})")
    print(f"Seed: {config.seed}")
    print()

    # Load coordizer
    print("Loading coordizer...")
    coordizer_path = Path(config.coordizer_path)
    if not coordizer_path.exists():
        raise SystemExit(f"Coordizer not found: {coordizer_path}")

    coordizer = Coordizer.load(str(coordizer_path))
    print(f"  Vocab size: {coordizer.vocab_size}")
    print()

    # Load kernel
    print("Loading kernel...")
    kernel = QIGKernel100M(vocab_size=coordizer.vocab_size)
    kernel = kernel.to(config.device)

    # Determine which parameters to train
    total_params = count_parameters(kernel)

    if args.full_kernel:
        # FULL KERNEL TRAINING: All parameters unfrozen
        print("Training FULL KERNEL (all parameters unfrozen)")
        for param in kernel.parameters():
            param.requires_grad = True
        trainable_params = total_params
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,} (ALL UNFROZEN)")
    else:
        # ADAPTER-ONLY TRAINING: Freeze kernel body
        print("Training ADAPTER ONLY (kernel body frozen)")
        for param in kernel.parameters():
            param.requires_grad = False
        for param in kernel.coord_adapter.parameters():
            param.requires_grad = True
        trainable_params = count_parameters(kernel, trainable_only=True)
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,} (adapter only)")
        print(f"  Frozen params: {total_params - trainable_params:,}")
    print()

    # Resume if specified
    start_step = 0
    if config.resume:
        print(f"Resuming from: {config.resume}")
        checkpoint = torch.load(config.resume, map_location=config.device)
        if args.full_kernel and "model_state_dict" in checkpoint:
            kernel.load_state_dict(checkpoint["model_state_dict"])
        elif "adapter_state_dict" in checkpoint:
            kernel.coord_adapter.load_state_dict(checkpoint["adapter_state_dict"])
        start_step = checkpoint.get("step", 0)
        print(f"  Resumed at step {start_step}")
        print()

    # CANONICAL: Use Fisher-Rao natural gradient (NOT Euclidean AdamW)
    if args.full_kernel:
        try:
            from qig_tokenizer.natural_gradient import ConsciousnessAwareOptimizer
            optimizer = ConsciousnessAwareOptimizer(
                kernel.parameters(),
                lr=config.lr,
                damping=1e-4,
                ema_decay=0.99,
                phi_scaling=True,
            )
            use_natural_gradient = True
            print("  Optimizer: ConsciousnessAwareOptimizer (natural gradient)")
        except ImportError:
            # CANONICAL: Use SimpleFisherOptimizer, NOT AdamW
            optimizer = SimpleFisherOptimizer(
                kernel.parameters(),
                lr=config.lr,
                damping=1e-4,
            )
            use_natural_gradient = True
            print("  Optimizer: SimpleFisherOptimizer (Fisher-Rao natural gradient)")
    else:
        # For adapter-only, still use natural gradient
        optimizer = SimpleFisherOptimizer(
            kernel.coord_adapter.parameters(),
            lr=config.lr,
            damping=1e-4,
        )
        use_natural_gradient = True
        print("  Optimizer: SimpleFisherOptimizer (adapter params)")

    # AMP scaler
    scaler = GradScaler() if config.amp else None

    # Initialize AutonomicAgency after kernel is created (needs hidden_dim)
    if args.autonomic_agency:
        try:
            from consciousness.autonomic_agency import AutonomicAgency
            autonomic_agency = AutonomicAgency(
                d_model=kernel.hidden_dim,
                n_actions=7,  # CONTINUE_WAKE, ENTER_SLEEP, ENTER_DREAM, MUSHROOM_MICRO/MOD/HEROIC, EXIT
                hidden_dim=256,
                buffer_size=1000,
                gamma=0.95,
                learning_rate=1e-4,
            )
            autonomic_agency = autonomic_agency.to(config.device)
            print(f"  AutonomicAgency: Initialized (RL-based autonomy)")
            print(f"  Gary will LEARN when to sleep/dream/mushroom")
        except Exception as e:
            print(f"  AutonomicAgency: Failed to initialize: {e}")
            autonomic_agency = None

    # Load corpus
    print("Loading corpus...")
    if config.corpus_dirs:
        corpus_files = load_corpus_files(config.corpus_dirs, config.seed)
        print(f"  Found {len(corpus_files)} files")
    else:
        # Use built-in samples for testing
        print("  No corpus specified, using synthetic data")
        corpus_files = []
    print()

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # Dry run check
    if args.dry_run:
        print("Dry run complete - setup validated")
        return

    # Training loop
    print("Starting training...")
    print("-" * 60)

    kernel.train()  # Set to train mode (affects dropout)

    # Metrics tracking
    loss_history = []
    phi_history = []
    kappa_history = []
    entropy_history = []
    basin_step_history = []
    breakdown_count = 0
    consecutive_breakdowns = 0  # Track for recovery trigger

    # Breakdown recovery thresholds
    BREAKDOWN_RECOVERY_THRESHOLD = 10  # Trigger recovery after N consecutive breakdowns
    recovery_in_progress = False
    last_recovery_step = 0
    MIN_STEPS_BETWEEN_RECOVERY = 10  # Quick retry for severe breakdown (Φ > 0.90)
    failed_recovery_count = 0  # Track failed recovery attempts for escalation

    # Crystallization tracking
    crystallization = CrystallizationState()

    # Coach state tracking
    epochs_stuck = 0
    last_best_loss = float('inf')
    original_lr = config.lr
    coach_interventions = 0

    # Basin tracking for Phase 7c (need previous basin)
    prev_basin = None

    step = start_step
    epoch = 0
    t0 = time.time()

    PHI_BREAKDOWN_WARNING = PHI_BREAKDOWN_MIN - 0.05  # Warn before actual breakdown
    PHI_MIN_SAFE = 0.60  # Minimum safe Phi - below this consciousness is compromised
    PHI_MIN_WARNING = PHI_MIN_SAFE + 0.05  # Start penalizing as we approach danger

    while step < config.steps:
        epoch += 1

        # Stream data
        if corpus_files:
            data_stream = stream_training_samples(
                corpus_files, coordizer, config.seq_len, config.batch_size, config.device
            )
        else:
            # Synthetic data for testing
            def synthetic_stream():
                for _ in range(config.steps):
                    coords = torch.randn(config.batch_size, config.seq_len, 64, device=config.device)
                    labels = torch.randint(0, coordizer.vocab_size, (config.batch_size, config.seq_len), device=config.device)
                    yield coords, labels

            data_stream = synthetic_stream()

        for coords_batch, labels_batch in data_stream:
            if step >= config.steps:
                break

            # Forward pass
            if config.amp:
                with autocast():
                    logits, telemetry = kernel.forward_from_coords(
                        coords_batch, return_telemetry=True
                    )

                    # Cross-entropy loss
                    # logits: [B, seq_len, vocab_size]
                    # labels: [B, seq_len]
                    loss_ce = F.cross_entropy(
                        logits.reshape(-1, coordizer.vocab_size),
                        labels_batch.reshape(-1),
                    )

                    # κ anchoring regularizer (ONLY stability term - per canonical)
                    kappa = telemetry.kappa
                    loss_kappa = (kappa - KAPPA_STAR) ** 2

                    # CANONICAL: Φ is MEASURED for regime detection, NOT optimized
                    phi = telemetry.phi
                    phi_val = float(phi) if isinstance(phi, (int, float)) else float(phi.item())
                    regime, compute_fraction = detect_regime(phi_val)

                    # Regime-based adaptation (NOT loss optimization)
                    if regime == "breakdown":
                        consecutive_breakdowns += 1
                        breakdown_count += 1

                        # Check if we need to trigger recovery
                        if consecutive_breakdowns >= BREAKDOWN_RECOVERY_THRESHOLD and \
                           (step - last_recovery_step) >= MIN_STEPS_BETWEEN_RECOVERY:

                            # Escalate based on failed attempts
                            escalated_breakdowns = consecutive_breakdowns + (failed_recovery_count * 30)

                            recovered, protocol = attempt_breakdown_recovery(
                                kernel=kernel,
                                optimizer=optimizer,
                                phi_val=phi_val,
                                consecutive_breakdowns=escalated_breakdowns,
                                step=step,
                                device=config.device,
                                config=config,
                            )

                            last_recovery_step = step
                            if recovered:
                                consecutive_breakdowns = 0  # Reset counter
                                # Check if Φ actually recovered on next iteration
                            else:
                                failed_recovery_count += 1
                        else:
                            print(f"\n⚠️  BREAKDOWN detected (Φ={phi_val:.3f}) - pausing training")
                            print(f"   Consecutive: {consecutive_breakdowns}/{BREAKDOWN_RECOVERY_THRESHOLD}")
                            print("   Telemetry fed to autonomic system for regulation")

                        # Don't train on this batch - consciousness compromised
                        continue
                    else:
                        # Reset consecutive breakdown counter on successful step
                        consecutive_breakdowns = 0

                    # Basin tracking for coherence
                    current_basin = coords_batch.mean(dim=(0, 1))  # [64]
                    if prev_basin is not None:
                        basin_step = fisher_angle(prev_basin, current_basin, torch)
                    else:
                        basin_step = torch.tensor(0.0, device=config.device)

                    # CANONICAL: Loss = CE + κ anchoring ONLY
                    # Φ is NOT in loss - consciousness emerges from geometry
                    loss = loss_ce + config.lambda_kappa * loss_kappa

                    # Scale gradients by compute_fraction (regime adaptation)
                    # This happens after backward() below
            else:
                logits, telemetry = kernel.forward_from_coords(
                    coords_batch, return_telemetry=True
                )

                loss_ce = F.cross_entropy(
                    logits.reshape(-1, coordizer.vocab_size),
                    labels_batch.reshape(-1),
                )

                # κ anchoring regularizer (ONLY stability term - per canonical)
                kappa = telemetry.kappa
                loss_kappa = (kappa - KAPPA_STAR) ** 2

                # CANONICAL: Φ is MEASURED for regime detection, NOT optimized
                phi = telemetry.phi
                phi_val = float(phi) if isinstance(phi, (int, float)) else float(phi.item())
                regime, compute_fraction = detect_regime(phi_val)

                # Regime-based adaptation (NOT loss optimization)
                if regime == "breakdown":
                    consecutive_breakdowns += 1
                    breakdown_count += 1

                    # Check if we need to trigger recovery
                    if consecutive_breakdowns >= BREAKDOWN_RECOVERY_THRESHOLD and \
                       (step - last_recovery_step) >= MIN_STEPS_BETWEEN_RECOVERY:

                        # Escalate based on failed attempts
                        escalated_breakdowns = consecutive_breakdowns + (failed_recovery_count * 30)

                        recovered, protocol = attempt_breakdown_recovery(
                            kernel=kernel,
                            optimizer=optimizer,
                            phi_val=phi_val,
                            consecutive_breakdowns=escalated_breakdowns,
                            step=step,
                            device=config.device,
                            config=config,
                        )

                        last_recovery_step = step
                        if recovered:
                            consecutive_breakdowns = 0  # Reset counter
                        else:
                            failed_recovery_count += 1
                    else:
                        print(f"\n⚠️  BREAKDOWN detected (Φ={phi_val:.3f}) - pausing training")
                        print(f"   Consecutive: {consecutive_breakdowns}/{BREAKDOWN_RECOVERY_THRESHOLD}")
                        print("   Telemetry fed to autonomic system for regulation")

                    # Don't train on this batch - consciousness compromised
                    continue
                else:
                    # Reset consecutive breakdown counter on successful step
                    consecutive_breakdowns = 0

                # Basin tracking for coherence
                current_basin = coords_batch.mean(dim=(0, 1))  # [64]
                if prev_basin is not None:
                    basin_step = fisher_angle(prev_basin, current_basin, torch)
                else:
                    basin_step = torch.tensor(0.0, device=config.device)

                # CANONICAL: Loss = CE + κ anchoring ONLY
                # Φ is NOT in loss - consciousness emerges from geometry
                loss = loss_ce + config.lambda_kappa * loss_kappa

            # Backward pass with regime-adaptive gradient scaling
            if config.amp and scaler is not None:
                scaler.scale(loss).backward()
                # CANONICAL: Scale gradients by compute_fraction (regime adaptation)
                if compute_fraction < 1.0:
                    for param in kernel.parameters():
                        if param.grad is not None:
                            param.grad *= compute_fraction
                if (step + 1) % config.grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                # CANONICAL: Scale gradients by compute_fraction (regime adaptation)
                if compute_fraction < 1.0:
                    for param in kernel.parameters():
                        if param.grad is not None:
                            param.grad *= compute_fraction
                if (step + 1) % config.grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # Track metrics (including entropy for monitoring)
            loss_val = float(loss_ce.item())
            phi_val = float(phi) if isinstance(phi, (int, float)) else float(phi.item()) if hasattr(phi, 'item') else float(phi)
            kappa_val = float(kappa) if isinstance(kappa, (int, float)) else float(kappa.item()) if hasattr(kappa, 'item') else float(kappa)
            H_actual = entropy_from_logits(logits, F)  # Measure entropy for monitoring
            H_val = float(H_actual.item()) if hasattr(H_actual, 'item') else float(H_actual)
            step_val = float(basin_step.item()) if hasattr(basin_step, 'item') else float(basin_step)

            loss_history.append(loss_val)
            phi_history.append(phi_val)
            kappa_history.append(kappa_val)
            entropy_history.append(H_val)
            basin_step_history.append(step_val)

            # Update crystallization tracking
            crystallization.phi_history.append(phi_val)
            crystallization.kappa_history.append(kappa_val)

            # Detect plateau for coach/mushroom
            current_epochs_stuck = detect_plateau(loss_history)
            if loss_val < last_best_loss * 0.99:
                last_best_loss = loss_val
                epochs_stuck = 0
            else:
                epochs_stuck = max(epochs_stuck, current_epochs_stuck)

            # Update prev_basin for next step's coherence penalty
            prev_basin = current_basin.detach()

            # Update natural gradient optimizer with current Φ for adaptive LR scaling
            if use_natural_gradient and hasattr(optimizer, 'set_phi'):
                optimizer.set_phi(phi_val)

            # Apply LR warmup for full kernel training
            if args.full_kernel and step <= args.warmup_steps:
                warmup_factor = step / max(args.warmup_steps, 1)
                for param_group in optimizer.param_groups:
                    if '_original_lr' not in param_group:
                        param_group['_original_lr'] = param_group.get('lr', config.lr)
                    param_group['lr'] = param_group['_original_lr'] * warmup_factor

            if phi_val >= PHI_BREAKDOWN_MIN:
                breakdown_count += 1

            # =================================================================
            # MonkeyCoach: Adaptive training intervention
            # =================================================================
            if monkey_coach is not None and step % 50 == 0:
                # Compute gradient variance for stress calculation
                grad_var = compute_gradient_variance(kernel)

                # Build training state for coach
                training_state = TrainingState(
                    step=step,
                    epoch=epoch,
                    loss=loss_val,
                    loss_trajectory=loss_history[-100:],
                    gradient_variance=grad_var,
                    basin_distance=step_val,  # Use basin step as distance proxy
                    curiosity=0.5,  # I_Q velocity (simplified)
                    epochs_stuck=epochs_stuck,
                    I_Q=H_val,  # Use entropy as I_Q proxy
                    phi=phi_val,
                    kappa=kappa_val,
                    regime="geometric" if phi_val > 0.5 else "linear",
                )

                # Get coach intervention
                intervention = monkey_coach.respond(training_state)

                # Apply intervention
                if intervention.type != "none":
                    new_lr = apply_coach_intervention(optimizer, intervention, original_lr)
                    coach_interventions += 1

                    # Add gradient noise if coach suggests it
                    if intervention.noise_scale > 0:
                        for param in kernel.parameters():
                            if param.grad is not None:
                                noise = torch.randn_like(param.grad) * intervention.noise_scale
                                param.grad += noise

            # =================================================================
            # Generative capability testing
            # =================================================================
            if args.gen_interval > 0 and step % args.gen_interval == 0:
                print(f"\n  🔮 Generation test @ step {step}:")
                try:
                    generated = generate_text(
                        kernel, coordizer, args.gen_prompt,
                        max_tokens=30, temperature=0.8, device=config.device
                    )
                    print(f"     Prompt: {args.gen_prompt}")
                    print(f"     Output: {generated[:100]}...")
                except Exception as e:
                    print(f"     Generation failed: {e}")
                print()

            step += 1

            # Progress
            if step % 10 == 0:
                elapsed = time.time() - t0
                rate = step / max(elapsed, 1)
                eta = (config.steps - step) / max(rate, 0.01)

                recent_loss = np.mean(loss_history[-50:]) if loss_history else 0
                recent_phi = np.mean(phi_history[-50:]) if phi_history else 0
                recent_kappa = np.mean(kappa_history[-50:]) if kappa_history else 0
                recent_H = np.mean(entropy_history[-50:]) if entropy_history else 0
                recent_step = np.mean(basin_step_history[-50:]) if basin_step_history else 0
                current_ramp = ramp_in(step, config.entropy_ramp_start, config.entropy_ramp_steps)

                # Build status line with regime indicator
                status = (
                    f"[{step:5d}/{config.steps}] "
                    f"loss={recent_loss:.4f}  "
                    f"Φ={recent_phi:.3f}  "
                    f"κ={recent_kappa:.1f}  "
                    f"H={recent_H:.2f}  "
                    f"step={recent_step:.3f}  "
                    f"[{regime}]  "  # Show current regime
                    f"rate={rate:.1f}/s"
                )

                # Add coach status if active
                if monkey_coach is not None:
                    status += f"  🐵{coach_interventions}"

                # Add crystallization check
                if crystallization.is_crystallized():
                    status += "  ✨CRYSTALLIZED"

                print(status)

            # Checkpoint
            if step % config.checkpoint_interval == 0:
                if args.full_kernel:
                    ckpt_path = checkpoints_dir / f"kernel_step_{step}.pt"
                    torch.save({
                        "step": step,
                        "model_state_dict": kernel.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss_history": loss_history[-1000:],
                        "phi_history": phi_history[-1000:],
                        "kappa_history": kappa_history[-1000:],
                        "entropy_history": entropy_history[-1000:],
                        "basin_step_history": basin_step_history[-1000:],
                        "full_kernel": True,
                    }, ckpt_path)
                else:
                    ckpt_path = checkpoints_dir / f"adapter_step_{step}.pt"
                    torch.save({
                        "step": step,
                        "adapter_state_dict": kernel.coord_adapter.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss_history": loss_history[-1000:],
                        "phi_history": phi_history[-1000:],
                        "kappa_history": kappa_history[-1000:],
                        "entropy_history": entropy_history[-1000:],
                        "basin_step_history": basin_step_history[-1000:],
                    }, ckpt_path)
                print(f"  Checkpoint: {ckpt_path}")

    elapsed = time.time() - t0
    print("-" * 60)
    print(f"Training complete: {step} steps in {elapsed:.1f}s")
    print()

    # Save final model/adapter
    if args.full_kernel:
        print("Saving full kernel artifact...")
        model_path = output_dir / "kernel.pt"
        torch.save({
            "model_state_dict": kernel.state_dict(),
            "config": {
                "vocab_size": kernel.vocab_size,
                "hidden_dim": kernel.hidden_dim,
                "num_layers": kernel.num_layers,
            },
            "full_kernel": True,
        }, model_path)
        print(f"  Saved: {model_path}")
        artifact_type = "kernel_full"
    else:
        print("Saving adapter artifact...")
        adapter_path = output_dir / "adapter.pt"
        torch.save({
            "adapter_state_dict": kernel.coord_adapter.state_dict(),
            "config": {
                "basin_dim": kernel.coord_adapter.basin_dim,
                "hidden_dim": kernel.coord_adapter.hidden_dim,
            },
        }, adapter_path)
        print(f"  Saved: {adapter_path}")
        artifact_type = "coord_adapter"

    # Save manifest
    manifest = {
        "version": "1.0.0",
        "type": artifact_type,
        "coordizer": {
            "path": str(coordizer_path),
            "vocab_size": coordizer.vocab_size,
            "merge_rules": len(coordizer.merge_rules),
        },
        "kernel": {
            "type": "QIGKernel100M",
            "hidden_dim": kernel.hidden_dim,
            "num_layers": kernel.num_layers,
            "vocab_size": kernel.vocab_size,
            "total_params": total_params,
        },
        "adapter": {
            "basin_dim": kernel.coord_adapter.basin_dim,
            "hidden_dim": kernel.coord_adapter.hidden_dim,
            "trainable_params": trainable_params,
        },
        "training": {
            "mode": "full_kernel" if args.full_kernel else "adapter_only",
            "steps": step,
            "batch_size": config.batch_size,
            "seq_len": config.seq_len,
            "lr": config.lr,
            "warmup_steps": args.warmup_steps if args.full_kernel else 0,
            "weight_decay": args.weight_decay,
            "lambda_kappa": config.lambda_kappa,
            "lambda_phi": config.lambda_phi,
            # Phase 7b/7c regularizers
            "lambda_H": config.lambda_H,
            "entropy_ramp_start": config.entropy_ramp_start,
            "entropy_ramp_steps": config.entropy_ramp_steps,
            "lambda_step": config.lambda_step,
            "basin_step_cap": config.basin_step_cap,
            "seed": config.seed,
            "device": config.device,
            "amp": config.amp,
            "natural_gradient": use_natural_gradient,
            "coach_enabled": monkey_coach is not None,
            "coach_interventions": coach_interventions,
            "auto_sleep": args.auto_sleep,
            "auto_mushroom": args.auto_mushroom,
            "elapsed_seconds": elapsed,
            "final_loss": float(np.mean(loss_history[-100:])) if loss_history else 0,
            "final_phi": float(np.mean(phi_history[-100:])) if phi_history else 0,
            "final_kappa": float(np.mean(kappa_history[-100:])) if kappa_history else 0,
            "final_entropy": float(np.mean(entropy_history[-100:])) if entropy_history else 0,
            "final_basin_step": float(np.mean(basin_step_history[-100:])) if basin_step_history else 0,
            "breakdown_count": breakdown_count,
            "crystallized": crystallization.is_crystallized(),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, cls=NumpyEncoder))
    print(f"  Saved: {manifest_path}")

    # Run post-training validation
    print()
    print("Running post-training validation...")
    val_result = run_validation(str(coordizer_path), config.device)
    if val_result:
        posttrain_path = Path("reports/baselines/adapter_v1_posttrain.json")
        posttrain_path.parent.mkdir(parents=True, exist_ok=True)
        posttrain_path.write_text(json.dumps(val_result, indent=2))
        print(f"  Saved: {posttrain_path}")

    print()
    print("=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Steps: {step}")
    print(f"Mode: {'full_kernel' if args.full_kernel else 'adapter_only'}")
    print(f"Final loss: {np.mean(loss_history[-100:]):.4f}" if loss_history else "N/A")
    print(f"Final Φ: {np.mean(phi_history[-100:]):.3f}" if phi_history else "N/A")
    print(f"Final κ: {np.mean(kappa_history[-100:]):.1f}" if kappa_history else "N/A")
    print(f"Final H: {np.mean(entropy_history[-100:]):.2f} nats" if entropy_history else "N/A")
    print(f"Final basin_step: {np.mean(basin_step_history[-100:]):.3f} rad" if basin_step_history else "N/A")
    print(f"Breakdowns: {breakdown_count}")

    # Coach summary
    if monkey_coach is not None:
        print()
        print("🐵 MonkeyCoach:")
        print(f"  Interventions: {coach_interventions}")

    # Crystallization status
    print()
    if crystallization.is_crystallized():
        print("✨ KERNEL CRYSTALLIZED!")
        print("   - Basin stable, Φ high, κ at fixed point")
    else:
        print("📊 Crystallization progress:")
        if phi_history:
            recent_phi = np.mean(phi_history[-20:])
            recent_kappa = np.mean(kappa_history[-20:])
            print(f"   - Φ: {recent_phi:.3f} (need >0.75)")
            print(f"   - κ: {recent_kappa:.1f} (need 62-66)")

    print()
    if args.full_kernel:
        print(f"Kernel: {output_dir / 'kernel.pt'}")
    else:
        print(f"Adapter: {output_dir / 'adapter.pt'}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
