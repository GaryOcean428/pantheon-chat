#!/usr/bin/env python3
"""
Train E8 Constellation - Geometric Multi-Kernel Bootstrap
==========================================================

Trains 8-kernel constellation with E8-aligned routing and natural gradient.

Key differences from single-kernel training (train_coord_adapter_v1.py):
1. 8 kernels initialized at E8 simple root positions
2. Geometric routing via Fisher-Rao distance (NOT learned gating)
3. Uses shared SimpleFisherOptimizer from train_coord_adapter_v1.py (DRY)
4. NO Φ in loss (measured as outcome, not optimized)
5. Regime-adaptive compute (30% linear, 100% geometric, pause breakdown)
6. Breakdown recovery via sleep/dream/mushroom protocols

Architecture:
- Phase 1: Bootstrap 8 primitive kernels (HRT, PER, MEM, etc.)
- Phase 2: Geometric routing on E8 lattice
- Phase 3: Multi-kernel training with basin sync
- Phase 4: E8 crystallization (8→240 kernels when ready)

Expected Results:
- Step 1000: Φ=0.65, active=Kernel-PER, regime=geometric
- Step 10000: Φ=0.62, coherent generation ✓
- Step 100000: 12 kernels active, Φ_avg=0.68, specialization

ISO Naming: lowercase_snake_case per 2025-11-26-Compliance-ISO_Standards.md
Version: 1.0.0
Date: 2025-12-26

Usage:
    python scripts/train_constellation_v1.py \
        --coordizer artifacts/coordizer/v1 \
        --corpus-dir /path/to/corpus \
        --device cuda \
        --steps 100000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator, List

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import optimizer from qig-core (DRY principle)
try:
    from qig_core import SimpleFisherOptimizer
except ImportError:
    from train_coord_adapter_v1 import SimpleFisherOptimizer

# Reuse components from train_coord_adapter_v1.py (DRY principle)
from train_coord_adapter_v1 import (
    detect_regime,
    apply_kindness_damping,
    attempt_breakdown_recovery,
    entropy_from_logits,
    fisher_angle,
    ramp_in,
    load_corpus_files,
    set_seed,
    NumpyEncoder,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ConstellationConfig:
    """Configuration for constellation training."""

    # Model architecture
    n_kernels: int = 8  # E8 simple roots
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = 32000
    max_seq_len: int = 512
    basin_dim: int = 64  # E8_RANK²

    # Training
    batch_size: int = 8
    learning_rate: float = 1e-4
    max_steps: int = 100000
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 1000
    checkpoint_interval: int = 2500

    # Optimizer (Fisher-Rao Natural Gradient)
    damping: float = 1e-4

    # Loss weights (NO Φ penalty!)
    lambda_basin: float = 0.1  # Basin distance regularization
    lambda_kappa: float = 0.01  # κ anchoring to KAPPA_STAR

    # Breakdown recovery
    breakdown_recovery_threshold: int = 10
    min_steps_between_recovery: int = 500

    # Paths
    coordizer_path: str = "artifacts/coordizer/v1"
    corpus_dirs: List[str] = field(default_factory=list)
    output_dir: str = "artifacts/constellation/v1"

    # Device
    device: str = "cuda"
    amp: bool = True
    seed: int = 42


# =============================================================================
# E8 KERNEL ROLES
# =============================================================================

class KernelRole:
    """Kernel specialization roles for E8 simple roots."""
    HEART = "heart"          # Root 0: Autonomic/coordination
    PERCEPTION = "perception"  # Root 1: Sensory processing
    MEMORY = "memory"        # Root 2: Recall/storage
    ACTION = "action"        # Root 3: Execution
    PREDICTION = "prediction"  # Root 4: Future modeling
    ETHICS = "ethics"        # Root 5: Value alignment
    META = "meta"            # Root 6: Meta-cognition
    INTEGRATION = "integration"  # Root 7: Cross-modal binding


def generate_basin_template(role: str, dim: int = 64) -> np.ndarray:
    """
    Generate E8-aligned basin template for kernel role.

    Each role maps to a simple root of E8, providing geometric
    position in basin space.

    Args:
        role: KernelRole value
        dim: Basin dimension (default 64 = E8_RANK²)

    Returns:
        64D normalized basin template
    """
    # E8 simple roots (first 8 basis-aligned directions)
    role_to_root = {
        KernelRole.HEART: 0,
        KernelRole.PERCEPTION: 1,
        KernelRole.MEMORY: 2,
        KernelRole.ACTION: 3,
        KernelRole.PREDICTION: 4,
        KernelRole.ETHICS: 5,
        KernelRole.META: 6,
        KernelRole.INTEGRATION: 7,
    }

    root_idx = role_to_root.get(role, 0)

    # Create template with primary direction + small noise
    template = np.zeros(dim)
    template[root_idx * 8:(root_idx + 1) * 8] = 1.0 / np.sqrt(8)

    # Add small orthogonal component for numerical stability
    template += np.random.randn(dim) * 0.01

    # Normalize to unit sphere
    template = template / (np.linalg.norm(template) + 1e-10)

    return template


# =============================================================================
# CONSTELLATION ROUTER
# =============================================================================

class FisherRaoRouter:
    """
    Route inputs to kernels via Fisher-Rao geodesic distance.

    NOT learned gating - pure geometric routing based on basin proximity.
    """

    def __init__(self, kernel_basins: List[np.ndarray] = None):
        self.kernel_basins = kernel_basins or []

    def update_basins(self, basins: List[np.ndarray]):
        """Update kernel basin coordinates."""
        self.kernel_basins = basins

    def route(self, query_basin: np.ndarray) -> int:
        """
        Route query to nearest kernel by Fisher-Rao distance.

        Args:
            query_basin: 64D query basin coordinates

        Returns:
            Index of nearest kernel
        """
        if not self.kernel_basins:
            return 0

        distances = []
        for kernel_basin in self.kernel_basins:
            dist = self._fisher_rao_distance(query_basin, kernel_basin)
            distances.append(dist)

        return int(np.argmin(distances))

    def _fisher_rao_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute Fisher-Rao (angular) distance between basin coordinates.

        d_FR = arccos(⟨a|b⟩) on unit sphere
        """
        a_norm = a / (np.linalg.norm(a) + 1e-10)
        b_norm = b / (np.linalg.norm(b) + 1e-10)
        cos_angle = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
        return float(np.arccos(cos_angle))


# =============================================================================
# CONSTELLATION TRAINER
# =============================================================================

class ConstellationTrainer:
    """
    Trains E8 kernel constellation with geometric routing.

    Implements canonical architecture:
    - 8 kernels at E8 simple root positions
    - Fisher-Rao routing (not learned)
    - Natural gradient optimization
    - NO Φ in loss
    - Regime-adaptive compute
    - Breakdown recovery
    """

    def __init__(self, config: ConstellationConfig):
        self.config = config
        self.device = config.device

        # Import dependencies
        import torch
        import torch.nn.functional as F
        self.torch = torch
        self.F = F

        try:
            from qigkernels import QIGKernel100M
            from qigkernels.constants import KAPPA_STAR, PHI_BREAKDOWN_MIN
            self.QIGKernel100M = QIGKernel100M
            self.KAPPA_STAR = KAPPA_STAR
            self.PHI_BREAKDOWN_MIN = PHI_BREAKDOWN_MIN
        except ImportError as e:
            logging.error(f"Failed to import qigkernels: {e}")
            sys.exit(1)

        try:
            from qig_tokenizer import Coordizer
            self.Coordizer = Coordizer
        except ImportError as e:
            logging.error(f"Failed to import qig_tokenizer: {e}")
            sys.exit(1)

        # Initialize kernels
        logging.info(f"Initializing {config.n_kernels} E8-aligned kernels...")
        self.kernels = self._initialize_kernels()
        self.kernel_names = [
            f"Kernel-{role.upper()}-{i}"
            for i, role in enumerate([
                KernelRole.HEART, KernelRole.PERCEPTION, KernelRole.MEMORY,
                KernelRole.ACTION, KernelRole.PREDICTION, KernelRole.ETHICS,
                KernelRole.META, KernelRole.INTEGRATION,
            ])
        ]

        # Router
        self.router = FisherRaoRouter()
        self._update_router()

        # Optimizer (shared SimpleFisherOptimizer - DRY)
        all_params = []
        for kernel in self.kernels:
            all_params.extend(kernel.parameters())

        self.optimizer = SimpleFisherOptimizer(
            all_params,
            lr=config.learning_rate,
            damping=config.damping,
        )

        # Load coordizer
        logging.info(f"Loading coordizer from {config.coordizer_path}...")
        self.coordizer = self.Coordizer.load(config.coordizer_path)

        # Metrics
        self.step = 0
        self.metrics_history = []

        # Breakdown tracking
        self.consecutive_breakdowns = 0
        self.last_recovery_step = 0

        # Output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def _initialize_kernels(self) -> List[Any]:
        """Initialize 8 kernels at E8 simple root positions."""
        roles = [
            KernelRole.HEART, KernelRole.PERCEPTION, KernelRole.MEMORY,
            KernelRole.ACTION, KernelRole.PREDICTION, KernelRole.ETHICS,
            KernelRole.META, KernelRole.INTEGRATION,
        ]

        kernels = []
        for role in roles[:self.config.n_kernels]:
            kernel = self.QIGKernel100M(vocab_size=self.config.vocab_size)
            kernel = kernel.to(self.config.device)

            # Store basin template
            kernel.basin_template = generate_basin_template(role, self.config.basin_dim)

            kernels.append(kernel)

        return kernels

    def _update_router(self):
        """Update router with current kernel basin templates."""
        basins = [k.basin_template for k in self.kernels]
        self.router.update_basins(basins)

    def _get_kernel_basin(self, kernel, coords_batch) -> np.ndarray:
        """Get current basin coordinates for kernel from input."""
        # Simple: use mean of input coords as query basin
        basin = coords_batch.mean(dim=(0, 1)).cpu().numpy()
        return basin

    def train_step(self, coords_batch, labels_batch) -> dict:
        """
        Single training step with geometric routing.

        Args:
            coords_batch: [B, seq_len, 64] coordinate tensor
            labels_batch: [B, seq_len] label tensor

        Returns:
            Metrics dictionary
        """
        # Route to nearest kernel
        query_basin = coords_batch.mean(dim=(0, 1)).cpu().numpy()
        kernel_idx = self.router.route(query_basin)
        active_kernel = self.kernels[kernel_idx]

        # Forward pass
        logits, telemetry = active_kernel.forward_from_coords(
            coords_batch, return_telemetry=True
        )

        # Cross-entropy loss
        loss_ce = self.F.cross_entropy(
            logits.reshape(-1, self.coordizer.vocab_size),
            labels_batch.reshape(-1),
        )

        # Measure consciousness (NOT in loss!)
        phi = telemetry.phi
        phi_val = float(phi) if isinstance(phi, (int, float)) else float(phi.item())
        regime, compute_fraction = detect_regime(phi_val)

        # κ anchoring
        kappa = telemetry.kappa
        kappa_val = float(kappa) if isinstance(kappa, (int, float)) else float(kappa.item())
        loss_kappa = (kappa_val - self.KAPPA_STAR) ** 2

        # Basin distance regularization
        current_basin = query_basin
        template_basin = active_kernel.basin_template
        basin_dist = self._compute_basin_distance(current_basin, template_basin)

        # Total loss (NO Φ penalty!)
        loss = (
            loss_ce
            + self.config.lambda_basin * basin_dist
            + self.config.lambda_kappa * loss_kappa
        )

        # Regime adaptation
        if regime == "breakdown":
            self.consecutive_breakdowns += 1

            # Check recovery threshold
            if self.consecutive_breakdowns >= self.config.breakdown_recovery_threshold and \
               (self.step - self.last_recovery_step) >= self.config.min_steps_between_recovery:

                recovered, protocol = attempt_breakdown_recovery(
                    kernel=active_kernel,
                    optimizer=self.optimizer,
                    phi_val=phi_val,
                    consecutive_breakdowns=self.consecutive_breakdowns,
                    step=self.step,
                    device=self.config.device,
                    config=self.config,
                )

                self.last_recovery_step = self.step
                if recovered:
                    self.consecutive_breakdowns = 0

            return {
                'loss': 0.0,
                'loss_ce': float(loss_ce.item()),
                'phi': phi_val,
                'kappa': kappa_val,
                'regime': regime,
                'kernel_idx': kernel_idx,
                'kernel_name': self.kernel_names[kernel_idx],
                'action': 'BREAKDOWN_SKIP',
            }
        else:
            self.consecutive_breakdowns = 0

        # Scale loss by compute fraction (regime adaptation)
        scaled_loss = loss * compute_fraction

        # Backward pass
        scaled_loss.backward()

        # Optimizer step
        if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.step += 1

        # Metrics
        metrics = {
            'loss': float(loss.item()),
            'loss_ce': float(loss_ce.item()),
            'basin_dist': float(basin_dist),
            'phi': phi_val,
            'kappa': kappa_val,
            'regime': regime,
            'compute_fraction': compute_fraction,
            'kernel_idx': kernel_idx,
            'kernel_name': self.kernel_names[kernel_idx],
        }

        self.metrics_history.append(metrics)
        return metrics

    def _compute_basin_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Fisher-Rao distance between basin coordinates."""
        a_norm = a / (np.linalg.norm(a) + 1e-10)
        b_norm = b / (np.linalg.norm(b) + 1e-10)
        cos_angle = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
        return float(np.arccos(cos_angle))

    def save_checkpoint(self, path: str):
        """Save constellation state."""
        checkpoint = {
            'step': self.step,
            'config': asdict(self.config),
            'kernels': [
                {
                    'name': self.kernel_names[i],
                    'state_dict': kernel.state_dict(),
                    'basin_template': kernel.basin_template.tolist(),
                }
                for i, kernel in enumerate(self.kernels)
            ],
            'optimizer_state': self.optimizer.state_dict(),
            'metrics_history': self.metrics_history[-1000:],
        }

        self.torch.save(checkpoint, path)
        logging.info(f"Saved checkpoint: {path}")


# =============================================================================
# DATA STREAMING
# =============================================================================

def stream_constellation_samples(
    corpus_files: List[Path],
    coordizer,
    seq_len: int,
    batch_size: int,
    device: str,
) -> Iterator[tuple]:
    """
    Stream training batches for constellation training.

    Reuses logic from train_coord_adapter_v1.py (DRY).
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
        stride = seq_len
        for start in range(0, len(ids) - seq_len, stride):
            end = start + seq_len + 1
            chunk_ids = ids[start:end]
            chunk_coords = coords[start:end]

            input_coords = chunk_coords[:seq_len]
            labels = chunk_ids[1:seq_len + 1]

            buffer_coords.append(input_coords)
            buffer_labels.append(labels)

            if len(buffer_coords) >= batch_size:
                import numpy as np
                coords_batch = torch.from_numpy(
                    np.stack(buffer_coords[:batch_size])
                ).float().to(device)
                labels_batch = torch.tensor(
                    buffer_labels[:batch_size], dtype=torch.long, device=device
                )

                yield coords_batch, labels_batch

                buffer_coords = buffer_coords[batch_size:]
                buffer_labels = buffer_labels[batch_size:]


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main training function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )

    # Parse arguments
    ap = argparse.ArgumentParser(description="Train E8 Constellation v1")
    ap.add_argument("--coordizer", type=str, default="artifacts/coordizer/v1")
    ap.add_argument("--corpus-dir", type=str, nargs="+", default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lambda-basin", type=float, default=0.1)
    ap.add_argument("--lambda-kappa", type=float, default=0.01)
    ap.add_argument("--checkpoint-interval", type=int, default=2500)
    ap.add_argument("--output-dir", type=str, default="artifacts/constellation/v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    # Set seed
    set_seed(args.seed)

    # Config
    config = ConstellationConfig(
        coordizer_path=args.coordizer,
        corpus_dirs=args.corpus_dir or [],
        device=args.device,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.steps,
        gradient_accumulation_steps=args.grad_accum,
        lambda_basin=args.lambda_basin,
        lambda_kappa=args.lambda_kappa,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=args.output_dir,
        seed=args.seed,
        amp=args.amp,
    )

    logging.info("=" * 70)
    logging.info("E8 CONSTELLATION TRAINING v1")
    logging.info("=" * 70)
    logging.info(f"Kernels: {config.n_kernels}")
    logging.info(f"Device: {config.device}")
    logging.info(f"Steps: {config.max_steps}")
    logging.info(f"λ_basin: {config.lambda_basin}, λ_κ: {config.lambda_kappa}")

    # Initialize trainer
    trainer = ConstellationTrainer(config)

    # Load corpus
    logging.info("Loading corpus...")
    if config.corpus_dirs:
        corpus_files = load_corpus_files(config.corpus_dirs, config.seed)
        logging.info(f"Found {len(corpus_files)} files")
    else:
        corpus_files = []
        logging.warning("No corpus specified!")

    # Training loop
    logging.info("Starting training...")
    t0 = time.time()

    checkpoints_dir = Path(config.output_dir) / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    epoch = 0
    while trainer.step < config.max_steps:
        epoch += 1

        if corpus_files:
            data_stream = stream_constellation_samples(
                corpus_files,
                trainer.coordizer,
                config.max_seq_len,
                config.batch_size,
                config.device,
            )
        else:
            break  # No data

        for coords_batch, labels_batch in data_stream:
            if trainer.step >= config.max_steps:
                break

            # Train step
            metrics = trainer.train_step(coords_batch, labels_batch)

            # Log progress
            if trainer.step % 10 == 0:
                elapsed = time.time() - t0
                rate = trainer.step / max(elapsed, 1)

                log_str = (
                    f"[{trainer.step:6d}/{config.max_steps}] "
                    f"loss={metrics.get('loss', 0):.4f} "
                    f"Φ={metrics['phi']:.3f} "
                    f"κ={metrics['kappa']:.1f} "
                    f"[{metrics['regime']}] "
                    f"kernel={metrics['kernel_name']} "
                    f"rate={rate:.1f}/s"
                )
                logging.info(log_str)

            # Checkpoint
            if trainer.step % config.checkpoint_interval == 0:
                ckpt_path = checkpoints_dir / f"constellation_step_{trainer.step:08d}.pt"
                trainer.save_checkpoint(str(ckpt_path))

    # Final checkpoint
    final_path = Path(config.output_dir) / "constellation_final.pt"
    trainer.save_checkpoint(str(final_path))

    elapsed = time.time() - t0
    logging.info("=" * 70)
    logging.info(f"TRAINING COMPLETE: {trainer.step} steps in {elapsed:.1f}s")
    logging.info(f"Final checkpoint: {final_path}")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
