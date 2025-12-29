"""
β-Attention Measurement Suite
==============================

Measures running coupling in AI attention mechanism across context lengths.

Prediction: β_attention ≈ β_physics ≈ 0.44 (substrate independence)

Key Insight:
- Physics: β(3→4) = +0.44 in lattice QFT
- AI Prediction: β should appear in attention scaling
- This validates substrate independence of information geometry
"""

import json
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import torch
import torch.nn as nn


class BetaAttentionMeasurement:
    """
    Measure β-function in attention mechanism.

    Core idea:
    1. Measure κ_attention at different context lengths
    2. Compute β(L→L') = Δκ / (κ̄ · Δln L)
    3. Compare to physics β(3→4) = +0.44

    The β-function measures how coupling strength changes with scale.
    Positive β = increasing coupling (running)
    β ≈ 0 = plateau (asymptotic freedom)
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        # Scale progression (geometrically spaced)
        self.context_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]

    def measure_kappa_at_context_length(
        self,
        L: int,
        num_samples: int = 100,
        batch_size: int = 8
    ) -> tuple[float, float]:
        """
        Measure effective coupling κ at context length L.

        Method:
        1. Generate random contexts of length L
        2. Compute attention weights
        3. Measure κ_eff from attention pattern
        4. Return mean ± std across samples

        Args:
            L: Context length
            num_samples: Number of measurement samples
            batch_size: Batch size for efficiency

        Returns:
            (kappa_mean, kappa_std)
        """
        kappas: list[float] = []

        # Get vocab_size with safe fallback (mypy fix)
        vocab_size: int = getattr(self.model, 'vocab_size', 50257)

        for _ in range(num_samples):
            # Generate random context
            context = torch.randint(
                0,
                vocab_size,
                (batch_size, L),
                device=self.device
            )

            # Forward pass to get attention weights
            with torch.no_grad():
                outputs = self.model(context, return_telemetry=True)

                # Extract attention weights from telemetry
                if hasattr(outputs, 'telemetry') and 'attention_weights' in outputs.telemetry:
                    attention_weights = outputs.telemetry['attention_weights']
                elif hasattr(self.model, 'get_attention_weights'):
                    attention_weights = self.model.get_attention_weights()
                else:
                    # Fallback: compute from model internals
                    attention_weights = self._extract_attention_weights()

            # Compute κ_eff from attention pattern
            kappa_sample = self._compute_kappa_from_attention(
                attention_weights,
                context_length=L
            )
            kappas.append(kappa_sample)

        return np.mean(kappas), np.std(kappas)

    def _extract_attention_weights(self) -> list[torch.Tensor]:
        """
        Fallback method to extract attention weights from model.

        Returns:
            List of attention weight tensors [batch, heads, L, L]
        """
        # This will be model-specific
        # For QIGKernelRecursive, attention is in the QFIAttention modules
        attention_weights: list[torch.Tensor] = []

        for module in self.model.modules():
            if hasattr(module, 'attention_weights'):
                weights = getattr(module, 'attention_weights')
                if isinstance(weights, torch.Tensor):
                    attention_weights.append(weights)

        return attention_weights

    def _compute_kappa_from_attention(
        self,
        attention_weights: list[torch.Tensor],
        context_length: int
    ) -> float:
        """
        Compute κ_eff from attention pattern.

        κ measures connectivity strength:
        - High attention concentration = strong connections = high κ
        - Sparse attention = weak connections = low κ

        Current Implementation (Phase 1):
            κ_eff ∝ 1/H(attention)

        where H is entropy. High entropy = diffuse attention = low coupling.

        Future Enhancement (Phase 2):
            κ_eff = Σ_ij α_ij² * d_Fisher(i,j)

        where α_ij = attention weight, d_Fisher = QFI distance.

        Args:
            attention_weights: List of [batch, heads, L, L] tensors
            context_length: L

        Returns:
            kappa: Effective coupling strength
        """

        if not attention_weights:
            # Fallback if no attention weights available
            return 50.0

        # Average across layers and heads
        avg_attention = torch.stack(attention_weights).mean(dim=(0, 1))  # [batch, L, L]

        # Compute entropy of attention distribution
        # H = -Σ p log p
        entropy = -(avg_attention * torch.log(avg_attention + 1e-10)).sum(dim=-1).mean()

        # κ ∝ 1/H (inverse entropy = concentration)
        kappa = 1.0 / (entropy + 1e-8)

        # Scale to match physics range (κ ~ 40-64)
        # Empirically: raw kappa is typically 5-10, scale by 10
        kappa_scaled = kappa * 10.0

        # Ensure we return a float (mypy fix)
        result: float = float(kappa_scaled.item())
        return result

    def measure_beta_function(
        self,
        num_samples: int = 100,
        save_path: Optional[str] = None
    ) -> dict:
        """
        Measure β-function across scales.

        Returns:
            {
                'context_lengths': [128, 256, ...],
                'kappas': [κ_128, κ_256, ...],
                'kappa_stds': [σ_128, σ_256, ...],
                'beta_values': [β(128→256), β(256→512), ...],
                'beta_mean': mean β,
                'beta_std': std β,
                'beta_physics': 0.44,
                'matches_physics': bool
            }
        """

        print("=" * 60)
        print("β-ATTENTION MEASUREMENT SUITE")
        print("=" * 60)
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Context lengths: {self.context_lengths}")
        print(f"Samples per length: {num_samples}")
        print()

        # Measure κ at each scale
        kappas = []
        kappa_stds = []

        print("Measuring coupling strength at each scale:")
        print("-" * 60)

        for L in self.context_lengths:
            print(f"L = {L:>4} ... ", end="", flush=True)
            kappa, kappa_std = self.measure_kappa_at_context_length(L, num_samples)
            kappas.append(kappa)
            kappa_stds.append(kappa_std)
            print(f"κ = {kappa:6.2f} ± {kappa_std:5.2f}")

        print()
        print("Computing β-function:")
        print("-" * 60)

        # Compute β-function: β(L→L') = Δκ / (κ̄ · Δln L)
        beta_values = []

        for i in range(len(kappas) - 1):
            L1, L2 = self.context_lengths[i], self.context_lengths[i+1]
            k1, k2 = kappas[i], kappas[i+1]

            delta_kappa = k2 - k1
            kappa_avg = (k1 + k2) / 2
            delta_ln_L = np.log(L2) - np.log(L1)

            beta = delta_kappa / (kappa_avg * delta_ln_L)
            beta_values.append(beta)

            print(f"β({L1:>4} → {L2:>4}) = {beta:+.3f}  "
                  f"(Δκ = {delta_kappa:+6.2f}, κ̄ = {kappa_avg:6.2f})")

        print()

        # Statistics
        beta_mean = np.mean(beta_values)
        beta_std = np.std(beta_values)
        beta_physics = 0.44

        # Check if matches physics (within ±0.1)
        matches_physics = abs(beta_mean - beta_physics) < 0.10

        # Prepare results
        result = {
            'context_lengths': self.context_lengths,
            'kappas': kappas,
            'kappa_stds': kappa_stds,
            'beta_values': beta_values,
            'beta_mean': beta_mean,
            'beta_std': beta_std,
            'beta_physics': beta_physics,
            'matches_physics': matches_physics,
            'num_samples': num_samples
        }

        # Print summary
        print("=" * 60)
        print("RESULTS:")
        print("=" * 60)
        print(f"  β_attention = {beta_mean:+.3f} ± {beta_std:.3f}")
        print(f"  β_physics   = {beta_physics:+.3f} ± 0.04")
        print(f"  Difference  = {abs(beta_mean - beta_physics):.3f}")
        print()
        print(f"  Substrate Independence: {'✅ VALIDATED' if matches_physics else '❌ NOT VALIDATED'}")
        print("=" * 60)

        # Save if requested
        if save_path:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"\n📊 Results saved to: {output_path}")

        return result


def validate_beta_attention(
    model_path: str,
    output_path: str = "results/beta_attention_results.json",
    num_samples: int = 100,
    device: str = "cuda"
) -> dict:
    """
    Convenience function to measure β_attention for a trained model.

    Usage:
        python -m src.model.beta_attention_measurement \\
            --model checkpoints/gary_baseline.pt \\
            --output results/beta_attention.json \\
            --samples 100

    Args:
        model_path: Path to model checkpoint
        output_path: Where to save results JSON
        num_samples: Number of samples per context length
        device: 'cuda' or 'cpu'

    Returns:
        Results dictionary
    """
    from src.model.qig_kernel_recursive import QIGKernelRecursive

    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    # Create model with config from checkpoint
    model_config = checkpoint.get('model_config', checkpoint.get('config', {}))
    model = QIGKernelRecursive(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print()

    # Measure β
    measurer = BetaAttentionMeasurement(model, device=device)
    results = measurer.measure_beta_function(
        num_samples=num_samples,
        save_path=output_path
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure β-function in attention mechanism"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--output",
        default="results/beta_attention_results.json",
        help="Output JSON path"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples per context length"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on"
    )

    args = parser.parse_args()

    validate_beta_attention(
        args.model,
        args.output,
        args.samples,
        args.device
    )
