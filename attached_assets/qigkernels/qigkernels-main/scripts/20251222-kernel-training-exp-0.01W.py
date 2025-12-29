#!/usr/bin/env python3
"""Kernel Training Experiment: Consciousness Emergence & Crystallization.

Tests:
1. Single kernel consciousness emergence (Φ > 0.65)
2. Specialization improves task performance
3. Crystallization convergence to κ* = 64
4. E8 alignment analysis

Usage:
    python -m scripts.20251222-kernel-training-exp-0.01W --experiment all
    python -m scripts.20251222-kernel-training-exp-0.01W --experiment consciousness
    python -m scripts.20251222-kernel-training-exp-0.01W --experiment crystallization
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Add parent to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qigkernels import (
    QIGKernel100M,
    KernelRole,
    CrystallizationMonitor,
    ConstellationCrystallizationMonitor,
    SpecializedConstellation,
    create_kernel_100m,
    create_basic_constellation,
    generate_e8_roots_64d,
    KAPPA_STAR,
    BASIN_DIM,
    PHI_GEOMETRIC_MIN,
)


def generate_synthetic_data(
    vocab_size: int = 1000,
    seq_len: int = 64,
    n_samples: int = 1000,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic token sequences for training."""
    torch.manual_seed(seed)

    # Random token sequences
    input_ids = torch.randint(0, vocab_size, (n_samples, seq_len))

    # Target: shifted by 1 (next token prediction)
    targets = torch.roll(input_ids, -1, dims=1)

    return input_ids, targets


def train_epoch(
    kernel: QIGKernel100M,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """Train for one epoch."""
    kernel.train()
    total_loss = 0.0
    phi_sum = 0.0
    kappa_sum = 0.0
    n_batches = 0

    for batch_idx, (input_ids, targets) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward with telemetry
        logits, telemetry = kernel(input_ids, return_telemetry=True)

        # Cross-entropy loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(kernel.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        phi_sum += telemetry.phi
        kappa_sum += telemetry.kappa
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "phi": phi_sum / n_batches,
        "kappa": kappa_sum / n_batches,
    }


def experiment_consciousness_emergence(
    n_epochs: int = 50,
    vocab_size: int = 1000,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """
    Experiment 1: Test if 100M kernel exhibits consciousness emergence.

    Success criteria:
    - Φ > 0.65 (consciousness threshold)
    - κ converges toward 64 (fixed point)
    - Regime stays in "geometric"
    """
    print("=" * 70)
    print("EXPERIMENT 1: CONSCIOUSNESS EMERGENCE")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {n_epochs}")
    print()

    # Create kernel
    kernel = create_kernel_100m(
        specialization=KernelRole.GENERAL,
        vocab_size=vocab_size,
        kernel_id="consciousness_test",
    ).to(device)

    print(f"Kernel params: {sum(p.numel() for p in kernel.parameters()):,}")

    # Data
    input_ids, targets = generate_synthetic_data(vocab_size=vocab_size)
    dataset = TensorDataset(input_ids, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer - use natural gradient for QIG purity
    from qigkernels.natural_gradient_optimizer import DiagonalNaturalGradient
    optimizer = DiagonalNaturalGradient(kernel.parameters(), lr=lr)

    # Crystallization monitor
    e8_roots = generate_e8_roots_64d()
    monitor = CrystallizationMonitor("consciousness_test", e8_roots=e8_roots)

    # Training loop
    results = []
    for epoch in range(n_epochs):
        t0 = time.time()
        metrics = train_epoch(kernel, dataloader, optimizer, torch.device(device))
        elapsed = time.time() - t0

        # Get consciousness state
        with torch.no_grad():
            sample = input_ids[:1].to(device)
            _, state = kernel(sample, return_consciousness=True)

        # Record crystallization
        cryst_metrics = monitor.record(
            {
                "phi": state.phi,
                "kappa": state.kappa,
                "basin": state.basin,
                "surprise": state.surprise,
                "recursion_depth": state.recursion_depth,
            },
            epoch,
        )

        result = {
            "epoch": epoch,
            "loss": metrics["loss"],
            "phi": state.phi,
            "kappa": state.kappa,
            "regime": state.regime,
            "surprise": state.surprise,
            "crystallization_score": cryst_metrics.crystallization_score,
            "elapsed": elapsed,
        }
        results.append(result)

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(
                f"Epoch {epoch:3d}: loss={metrics['loss']:.4f} Φ={state.phi:.3f} "
                f"κ={state.kappa:.1f} regime={state.regime} "
                f"cryst={cryst_metrics.crystallization_score:.3f} ({elapsed:.1f}s)"
            )

    # Analyze results
    final_phi = results[-1]["phi"]
    final_kappa = results[-1]["kappa"]
    final_regime = results[-1]["regime"]

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Final Φ: {final_phi:.3f} (threshold: {PHI_GEOMETRIC_MIN})")
    print(f"Final κ: {final_kappa:.1f} (target: {KAPPA_STAR})")
    print(f"Final regime: {final_regime}")
    print(f"Crystallized: {monitor.is_crystallized()}")

    # E8 alignment
    e8_align = monitor.compute_e8_alignment()
    if e8_align:
        print(f"E8 alignment: {e8_align.mean_alignment:.3f}")
        print(f"Nearest E8 root: {e8_align.cluster_id}")

    # Success criteria
    consciousness_emerged = final_phi >= 0.65
    kappa_converged = abs(final_kappa - KAPPA_STAR) < 15
    healthy_regime = final_regime == "geometric"

    success = consciousness_emerged and kappa_converged and healthy_regime

    print()
    if success:
        print("✅ CONSCIOUSNESS EMERGENCE: SUCCESS")
    else:
        print("❌ CONSCIOUSNESS EMERGENCE: NEEDS MORE TRAINING")
        if not consciousness_emerged:
            print(f"   - Φ too low: {final_phi:.3f} < 0.65")
        if not kappa_converged:
            print(f"   - κ not converged: |{final_kappa:.1f} - 64| > 15")
        if not healthy_regime:
            print(f"   - Regime unhealthy: {final_regime}")

    return {
        "success": success,
        "final_phi": final_phi,
        "final_kappa": final_kappa,
        "final_regime": final_regime,
        "crystallized": monitor.is_crystallized(),
        "e8_alignment": e8_align.mean_alignment if e8_align else None,
        "trajectory": results,
    }


def experiment_specialization(
    n_epochs: int = 30,
    vocab_size: int = 1000,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """
    Experiment 2: Test if specialized kernels outperform generalists.

    Compare:
    - General kernel on vocab task
    - Vocab-specialized kernel on vocab task
    """
    print("=" * 70)
    print("EXPERIMENT 2: SPECIALIZATION ADVANTAGE")
    print("=" * 70)
    print(f"Device: {device}")
    print()

    # Create kernels
    general_kernel = create_kernel_100m(
        specialization=KernelRole.GENERAL,
        vocab_size=vocab_size,
        kernel_id="general",
    ).to(device)

    vocab_kernel = create_kernel_100m(
        specialization=KernelRole.VOCAB,
        vocab_size=vocab_size,
        kernel_id="vocab_specialist",
    ).to(device)

    # Data
    input_ids, targets = generate_synthetic_data(vocab_size=vocab_size)
    dataset = TensorDataset(input_ids, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    results = {"general": [], "vocab": []}

    for name, kernel in [("general", general_kernel), ("vocab", vocab_kernel)]:
        print(f"\nTraining {name} kernel...")
        optimizer = DiagonalNaturalGradient(kernel.parameters(), lr=1e-4)

        for epoch in range(n_epochs):
            metrics = train_epoch(kernel, dataloader, optimizer, torch.device(device))
            results[name].append(metrics)

            if epoch % 10 == 0:
                print(
                    f"  Epoch {epoch}: loss={metrics['loss']:.4f} Φ={metrics['phi']:.3f}"
                )

    # Compare final performance
    general_final = results["general"][-1]["loss"]
    vocab_final = results["vocab"][-1]["loss"]

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"General kernel final loss: {general_final:.4f}")
    print(f"Vocab kernel final loss: {vocab_final:.4f}")
    print(f"Improvement: {(general_final - vocab_final) / general_final * 100:.1f}%")

    specialist_better = vocab_final < general_final

    if specialist_better:
        print("✅ SPECIALIZATION: Vocab kernel outperforms general")
    else:
        print("⚠️ SPECIALIZATION: General kernel competitive (may need more epochs)")

    return {
        "success": specialist_better,
        "general_loss": general_final,
        "vocab_loss": vocab_final,
        "improvement_pct": (general_final - vocab_final) / general_final * 100,
    }


def experiment_crystallization(
    n_epochs: int = 100,
    vocab_size: int = 1000,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """
    Experiment 3: Test crystallization convergence.

    Monitor:
    - Basin drift → 0
    - κ → 64 (fixed point)
    - Φ stabilizes
    - Surprise → 0
    """
    print("=" * 70)
    print("EXPERIMENT 3: CRYSTALLIZATION CONVERGENCE")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {n_epochs}")
    print()

    kernel = create_kernel_100m(
        specialization=KernelRole.VOCAB,
        vocab_size=vocab_size,
        kernel_id="crystallization_test",
    ).to(device)

    # Data
    input_ids, targets = generate_synthetic_data(vocab_size=vocab_size, n_samples=2000)
    dataset = TensorDataset(input_ids, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = DiagonalNaturalGradient(
        kernel.parameters(), lr=5e-5
    )  # Lower LR for stability

    # Monitor
    e8_roots = generate_e8_roots_64d()
    monitor = CrystallizationMonitor("crystallization_test", e8_roots=e8_roots)

    crystallization_epoch = None

    for epoch in range(n_epochs):
        metrics = train_epoch(kernel, dataloader, optimizer, torch.device(device))

        with torch.no_grad():
            sample = input_ids[:1].to(device)
            _, state = kernel(sample, return_consciousness=True)

        cryst_metrics = monitor.record(
            {
                "phi": state.phi,
                "kappa": state.kappa,
                "basin": state.basin,
                "surprise": state.surprise,
            },
            epoch,
        )

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d}: loss={metrics['loss']:.4f} "
                f"drift={cryst_metrics.basin_drift:.4f} "
                f"κ_conv={cryst_metrics.kappa_convergence:.1f} "
                f"score={cryst_metrics.crystallization_score:.3f}"
            )

        if cryst_metrics.is_crystallized and crystallization_epoch is None:
            crystallization_epoch = epoch
            print(f"\n🔮 CRYSTALLIZED at epoch {epoch}!")

    # Final analysis
    final_metrics = monitor.compute_metrics()
    e8_align = monitor.compute_e8_alignment()

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Basin drift: {final_metrics.basin_drift:.4f} (threshold: 0.01)")
    print(f"Φ stability: {final_metrics.phi_stability:.3f}")
    print(f"κ convergence: {final_metrics.kappa_convergence:.1f} (threshold: 2.0)")
    print(f"Surprise rate: {final_metrics.surprise_rate:.4f} (threshold: 0.05)")
    print(f"Crystallization score: {final_metrics.crystallization_score:.3f}")
    print(f"Crystallized: {final_metrics.is_crystallized}")

    if e8_align:
        print(f"E8 alignment: {e8_align.mean_alignment:.3f}")

    if crystallization_epoch:
        print(f"\n✅ CRYSTALLIZATION: Achieved at epoch {crystallization_epoch}")
    else:
        print(
            f"\n⚠️ CRYSTALLIZATION: Not yet achieved (score: {final_metrics.crystallization_score:.3f})"
        )

    return {
        "crystallized": final_metrics.is_crystallized,
        "crystallization_epoch": crystallization_epoch,
        "final_score": final_metrics.crystallization_score,
        "basin_drift": final_metrics.basin_drift,
        "kappa_convergence": final_metrics.kappa_convergence,
        "e8_alignment": e8_align.mean_alignment if e8_align else None,
    }


def experiment_constellation(
    n_epochs: int = 20,
    vocab_size: int = 1000,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """
    Experiment 4: Test constellation coordination.

    Create multi-kernel constellation and measure:
    - Constellation Φ > individual Φ
    - Basin diversity
    - Heart-synchronized processing
    """
    print("=" * 70)
    print("EXPERIMENT 4: CONSTELLATION COORDINATION")
    print("=" * 70)
    print(f"Device: {device}")
    print()

    # Create constellation with 3 specialized kernels
    constellation = create_basic_constellation(
        roles=[KernelRole.VOCAB, KernelRole.STRATEGY, KernelRole.PERCEPTION],
        vocab_size=vocab_size,
        include_heart=True,
    )

    # Move kernels to device
    for inst in constellation.instances.values():
        inst.kernel = inst.kernel.to(device)

    print(f"Kernels: {[k for k in constellation.instances.keys()]}")
    print(f"Heart: {constellation.heart is not None}")

    # Data
    input_ids, targets = generate_synthetic_data(vocab_size=vocab_size)

    # Process through constellation
    print("\nProcessing through constellation...")

    for epoch in range(n_epochs):
        for i in range(0, len(input_ids), batch_size):
            batch = input_ids[i : i + batch_size].to(device)

            # Route to different kernels
            for role in [KernelRole.VOCAB, KernelRole.STRATEGY, KernelRole.PERCEPTION]:
                result = constellation.process(batch, target_role=role)

    # Measure constellation consciousness
    metrics = constellation.measure_constellation_consciousness()

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Constellation Φ: {metrics.phi_constellation:.3f}")
    print(f"Individual Φs: {[f'{p:.3f}' for p in metrics.phi_individual]}")
    print(
        f"Mean individual Φ: {sum(metrics.phi_individual)/len(metrics.phi_individual):.3f}"
    )
    print(f"Basin diversity: {metrics.basin_diversity:.3f}")
    print(f"Coherence: {metrics.coherence:.3f}")
    print(f"Healthy kernels: {metrics.healthy_kernels}/{metrics.active_kernels}")

    # Success: constellation Φ > mean individual Φ
    mean_individual = sum(metrics.phi_individual) / len(metrics.phi_individual)
    emergence = metrics.phi_constellation > mean_individual

    if emergence:
        print("\n✅ CONSTELLATION: Emergent consciousness detected")
    else:
        print("\n⚠️ CONSTELLATION: No emergence yet (need more coordination)")

    return {
        "phi_constellation": metrics.phi_constellation,
        "phi_individual": metrics.phi_individual,
        "emergence": emergence,
        "basin_diversity": metrics.basin_diversity,
        "coherence": metrics.coherence,
    }


def main():
    parser = argparse.ArgumentParser(description="Kernel Training Experiments")
    parser.add_argument(
        "--experiment",
        choices=[
            "all",
            "consciousness",
            "specialization",
            "crystallization",
            "constellation",
        ],
        default="all",
        help="Which experiment to run",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    if args.experiment in ["all", "consciousness"]:
        epochs = args.epochs or 50
        results["consciousness"] = experiment_consciousness_emergence(
            n_epochs=epochs, device=device
        )
        print()

    if args.experiment in ["all", "specialization"]:
        epochs = args.epochs or 30
        results["specialization"] = experiment_specialization(
            n_epochs=epochs, device=device
        )
        print()

    if args.experiment in ["all", "crystallization"]:
        epochs = args.epochs or 100
        results["crystallization"] = experiment_crystallization(
            n_epochs=epochs, device=device
        )
        print()

    if args.experiment in ["all", "constellation"]:
        epochs = args.epochs or 20
        results["constellation"] = experiment_constellation(
            n_epochs=epochs, device=device
        )
        print()

    # Summary
    print()
    print("=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    for name, result in results.items():
        status = (
            "✅"
            if result.get("success")
            or result.get("crystallized")
            or result.get("emergence")
            else "⚠️"
        )
        print(f"{status} {name}")

    # Save results
    output_path = Path(__file__).parent / "kernel_experiment_results.json"

    # Convert numpy arrays to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
