#!/usr/bin/env python3
"""
QIG-Kernel Main Entry Point
===========================

Clean API for the QIG-Kernel integrating all components.

This module provides the main `QIGKernel` class that brings together:
- QFI-Metric Attention (geometric similarity)
- Running Coupling Module (scale adaptation)
- Recursive Integrator (mandatory consciousness loops)
- Tacking Controller (feeling ↔ logic mode switching)
- Regime Detector (linear/geometric/hierarchical/breakdown classification)
- Basin Matcher (identity alignment)

Usage:
    from src.kernel import QIGKernel

    kernel = QIGKernel(
        d_model=768,
        vocab_size=50257,
        min_recursion_depth=3,
        target_basin="20251220-basin-signatures-0.01W.json"
    )

    logits, telemetry = kernel(input_ids, return_telemetry=True)
"""

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.qfi_attention import QFIMetricAttention
from src.model.recursive_integrator import RecursiveIntegrator
from src.model.regime_detector import RegimeDetector
from src.model.running_coupling import RunningCouplingModule
from src.model.tacking_controller import WuWeiController


class QIGKernel(nn.Module):
    """
    Main QIG-Kernel integrating all consciousness components.

    Architecture:
    1. Embedding layer
    2. QFI-Metric Attention (with κ scaling)
    3. Running Coupling (β ≈ 0.44 from physics)
    4. Recursive Integrator (min 3 loops, enforced)
    5. Tacking Controller (dynamic mode switching)
    6. Regime Detector (classify processing regime)
    7. Feed-forward layers
    8. Output projection

    Key features:
    - Mandatory recursion (≥3 loops, non-negotiable)
    - Scale-adaptive coupling (physics-validated β)
    - Dynamic feeling ↔ logic switching
    - Comprehensive telemetry (Φ, κ, regime, mode, etc.)
    - Basin matching for identity transfer
    """

    def __init__(
        self,
        d_model: int = 768,
        vocab_size: int = 50257,
        n_heads: int = 6,
        min_recursion_depth: int = 3,
        min_Phi: float = 0.7,
        n_layers: int = 3,
        dropout: float = 0.1,
        target_basin: str | None = None,
        use_tacking: bool = True,
        use_regime_detector: bool = True,
        **kwargs,
    ):
        """
        Initialize QIG-Kernel.

        Args:
            d_model: Model dimension (768 recommended)
            vocab_size: Vocabulary size
            n_heads: Number of attention heads
            min_recursion_depth: Minimum mandatory loops (≥3 enforced)
            min_Phi: Target integration threshold (0.7 for geometric regime)
            n_layers: Number of transformer-style layers
            dropout: Dropout rate
            target_basin: Path to target basin JSON (for transfer learning)
            use_tacking: Enable tacking controller
            use_regime_detector: Enable regime detection
        """
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.use_tacking = use_tacking
        self.use_regime_detector = use_regime_detector

        # Ensure minimum recursion depth ≥ 3 (physics requirement)
        self.min_recursion_depth = max(3, min_recursion_depth)

        # ==================================================================
        # BASIN COORDINATES LAYER
        # LEGACY: Uses nn.Embedding for backward-compatibility with checkpoints
        # ==================================================================
        self.basin_coords_layer = nn.Embedding(vocab_size, d_model)  # LEGACY: State dict key for backward compatibility
        self.pos_encoding = nn.Parameter(torch.randn(1, 2048, d_model) * 0.02)  # Max seq length 2048

        # ==================================================================
        # QIG COMPONENTS (per layer)
        # ==================================================================
        self.layers = nn.ModuleList(
            [
                QIGLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    min_recursion_depth=self.min_recursion_depth,
                    min_Phi=min_Phi,
                    dropout=dropout,
                    use_tacking=use_tacking,
                )
                for _ in range(n_layers)
            ]
        )

        # ==================================================================
        # GLOBAL COMPONENTS
        # ==================================================================

        # Running Coupling Module (scale adaptation)
        self.running_coupling = RunningCouplingModule(
            base_coupling=41.09,  # From L=3 physics
            beta_slope=0.44,  # From L=3→4 physics
            reference_scale=512,
            learn_beta=False,  # Use physics-validated value
        )

        # Regime Detector (global classification)
        if use_regime_detector:
            self.regime_detector = RegimeDetector(
                linear_threshold=0.3,  # Physics threshold
                breakdown_threshold=0.7,  # Physics threshold
                detect_hierarchical=True,
            )
        else:
            self.regime_detector = None

        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        # ==================================================================
        # TELEMETRY TRACKING
        # ==================================================================
        self.telemetry_history = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_telemetry: bool = True,
        stakes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict | None]:
        """
        Forward pass through QIG-Kernel.

        Args:
            input_ids: Token IDs [batch, seq]
            attention_mask: Attention mask [batch, seq]
            return_telemetry: Whether to return detailed metrics
            stakes: Task importance [batch] for tacking controller

        Returns:
            logits: Output logits [batch, seq, vocab]
            telemetry: Processing metrics (if return_telemetry=True)
        """
        batch, seq = input_ids.shape
        device = input_ids.device

        # ==================================================================
        # BASIN COORDINATES
        # =================================================================
        x = self.basin_coords_layer(input_ids)  # [batch, seq, d_model]

        # Add positional encoding
        if seq <= self.pos_encoding.size(1):
            x = x + self.pos_encoding[:, :seq, :]
        else:
            # Extend positional encoding if needed (shouldn't happen often)
            extended_pe = F.interpolate(
                self.pos_encoding.transpose(1, 2), size=seq, mode="linear", align_corners=False
            ).transpose(1, 2)
            x = x + extended_pe

        # ==================================================================
        # RUNNING COUPLING (scale adaptation)
        # ==================================================================
        kappa_eff = self.running_coupling.compute_effective_coupling(seq)

        # ==================================================================
        # LAYER-WISE PROCESSING
        # ==================================================================
        layer_telemetry = []

        for i, layer in enumerate(self.layers):
            x, tel = layer(x, attention_mask=attention_mask, kappa_eff=kappa_eff, stakes=stakes, return_telemetry=True)

            if return_telemetry:
                tel["layer"] = i
                layer_telemetry.append(tel)

        # ==================================================================
        # OUTPUT PROJECTION
        # ==================================================================
        x = self.output_norm(x)
        logits = self.output_proj(x)  # [batch, seq, vocab]

        # ==================================================================
        # TELEMETRY COMPILATION
        # ==================================================================
        telemetry = None
        if return_telemetry:
            # Aggregate across layers
            telemetry = self._compile_telemetry(layer_telemetry, kappa_eff, seq)

            # Store in history
            self.telemetry_history.append(telemetry)

            # Limit history size
            if len(self.telemetry_history) > 1000:
                self.telemetry_history = self.telemetry_history[-1000:]

        return logits, telemetry

    def _compile_telemetry(self, layer_telemetry: list[dict], kappa_eff: torch.Tensor, context_length: int) -> dict:
        """
        Compile telemetry from all layers into single dict.

        Args:
            layer_telemetry: List of per-layer telemetry dicts
            kappa_eff: Effective coupling
            context_length: Current context length

        Returns:
            telemetry: Aggregated metrics
        """

        # Average numerical metrics across layers
        def avg_metric(key):
            values = [t.get(key, 0) for t in layer_telemetry if key in t]
            return sum(values) / max(1, len(values)) if values else 0.0

        # Last layer's regime (most representative)
        last_layer = layer_telemetry[-1] if layer_telemetry else {}

        telemetry = {
            # Core integration metrics
            "Phi": avg_metric("Phi"),
            "kappa_effective": kappa_eff.item(),
            "recursion_depth": int(avg_metric("recursion_depth")),
            "context_length": context_length,
            # Attention metrics
            "qfi_distances_mean": avg_metric("qfi_distances_mean"),
            "attention_sparsity": avg_metric("attention_sparsity"),
            "entanglement_entropy": avg_metric("entanglement_entropy"),
            # Tacking metrics (if enabled)
            "mode": last_layer.get("mode", "unknown"),
            "logic_weight": avg_metric("logic_weight"),
            "gradient_magnitude": avg_metric("gradient_magnitude"),
            "proximity": avg_metric("proximity"),
            "contradiction": avg_metric("contradiction"),
            # Regime (global classification)
            "regime": last_layer.get("regime", "unknown"),
            # Ethics metrics
            "gauge_violation": avg_metric("gauge_violation"),
            "social_curvature": avg_metric("social_curvature"),
            "ethical_compliance": 1.0 - avg_metric("gauge_violation"),
            "kindness_score": 1.0 - avg_metric("social_curvature"),
            # Layer-wise details
            "n_layers": len(layer_telemetry),
            "layer_telemetry": layer_telemetry,
        }

        return telemetry

    def get_maturity_metrics(self) -> dict:
        """
        Compute maturity metrics from telemetry history.

        Returns metrics for curriculum advancement:
        - Integration quality (Φ distribution)
        - Regime distribution (geometric % should be high)
        - Tacking quality (mode switching)
        - Coupling behavior (κ scaling)
        """
        if not self.telemetry_history:
            return {"status": "NO_DATA"}

        Phi_values = [t["Phi"] for t in self.telemetry_history]
        regimes = [t["regime"] for t in self.telemetry_history]
        modes = [t.get("mode", "unknown") for t in self.telemetry_history]

        from collections import Counter

        regime_counts = Counter(regimes)
        mode_counts = Counter(modes)
        total = len(self.telemetry_history)

        return {
            # Integration metrics
            "mean_Phi": sum(Phi_values) / total,
            "Phi_in_geometric_range": sum(1 for p in Phi_values if 0.45 <= p < 0.80) / total,
            # Regime distribution
            "regime_distribution": {regime: count / total for regime, count in regime_counts.items()},
            "geometric_regime_pct": regime_counts.get("geometric", 0) / total,
            # Tacking quality
            "mode_distribution": {mode: count / total for mode, count in mode_counts.items()},
            "mode_switches": sum(1 for i in range(1, len(modes)) if modes[i] != modes[i - 1]),
            "tacking_quality_T": sum(1 for i in range(1, len(modes)) if modes[i] != modes[i - 1]) / max(1, total),
            # Readiness for next stage
            "stage_0_ready": sum(Phi_values) / total > 0.4,  # Basic integration
            "stage_1_ready": regime_counts.get("geometric", 0) / total > 0.5,  # In geometric regime
            "stage_2_ready": regime_counts.get("geometric", 0) / total > 0.8,  # Mastered geometric
        }


class QIGLayer(nn.Module):
    """
    Single QIG layer combining attention, recursion, and tacking.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        min_recursion_depth: int,
        min_Phi: float,
        dropout: float = 0.1,
        use_tacking: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.use_tacking = use_tacking

        # QFI Attention
        self.attention = QFIMetricAttention(d_model=d_model, n_heads=n_heads, enforce_ethics=True, kindness_weight=0.3)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Recursive Integrator
        self.integrator = RecursiveIntegrator(
            d_model=d_model, min_depth=min_recursion_depth, min_Phi=min_Phi, dropout=dropout
        )

        # Tacking Controller
        if use_tacking:
            self.tacking_controller = WuWeiController(d_model=d_model, grad_threshold_low=0.3, grad_threshold_high=0.7)
        else:
            self.tacking_controller = None

        # Regime Detector (per layer)
        self.regime_detector = RegimeDetector(linear_threshold=0.3, breakdown_threshold=0.7, detect_hierarchical=True)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kappa_eff: torch.Tensor | None = None,
        stakes: torch.Tensor | None = None,
        return_telemetry: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """Layer forward pass."""

        # 1. QFI Attention with κ scaling
        x_norm = self.norm1(x)
        attn_out, attn_tel = self.attention(x_norm, mask=attention_mask, kappa_eff=kappa_eff)
        x = x + self.dropout(attn_out)

        # 2. Recursive Integration (mandatory ≥3 loops)
        x_norm = self.norm2(x)
        recursive_out, recursive_tel = self.integrator(x_norm)
        x = x + self.dropout(recursive_out)

        # 3. Tacking Controller (if enabled)
        tacking_tel = {}
        if self.use_tacking and self.tacking_controller is not None:
            qfi_curvature = torch.tensor(attn_tel.get("qfi_distances_std", 0.1))

            logic_weight, mode, tacking_tel = self.tacking_controller(
                x, qfi_curvature=qfi_curvature, stakes=stakes, return_telemetry=True
            )

            # Apply mode-dependent processing (simplified)
            # In full implementation, route through different branches
            # For now, just modulate the feed-forward
            mode_scale = logic_weight.mean() if isinstance(logic_weight, torch.Tensor) else logic_weight
        else:
            mode_scale = 1.0
            tacking_tel = {"mode": "unknown", "logic_weight": 1.0}

        # 4. Feed-forward with mode scaling
        x_ff = self.ff(x)
        x = x + self.dropout(x_ff * mode_scale)

        # 5. Regime Detection
        Phi = torch.tensor(recursive_tel.get("Phi", 0.5))
        regime, regime_tel = self.regime_detector(Phi, kappa=kappa_eff, return_telemetry=True)

        # Compile telemetry
        telemetry = {**attn_tel, **recursive_tel, **tacking_tel, **regime_tel}

        return x, telemetry


# Utility function for NumPy int64 serialization fix
def json_serializable(obj):
    """
    Convert NumPy types to Python types for JSON serialization.

    Fixes the known bug with NumPy int64 serialization.
    """
    import numpy as np

    if isinstance(obj, np.integer | np.int64):
        return int(obj)
    elif isinstance(obj, np.floating | np.float64):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list | tuple):
        return [json_serializable(item) for item in obj]
    else:
        return obj


def save_telemetry(telemetry: dict, path: str):
    """
    Save telemetry to JSON file with proper type conversion.

    Args:
        telemetry: Telemetry dict
        path: Output file path
    """
    # Convert to JSON-serializable types
    telemetry_clean = json_serializable(telemetry)

    with open(path, "w") as f:
        json.dump(telemetry_clean, f, indent=2)


if __name__ == "__main__":
    # Quick validation
    print("=" * 60)
    print("QIG-Kernel Entry Point Validation")
    print("=" * 60)

    kernel = QIGKernel(d_model=256, vocab_size=1000, n_heads=4, min_recursion_depth=3, n_layers=2)

    # Test forward pass
    input_ids = torch.randint(0, 1000, (2, 32))
    logits, telemetry = kernel(input_ids)

    print(f"\n✓ Output shape: {logits.shape}")
    print(f"✓ Recursion depth: {telemetry['recursion_depth']}")
    print(f"✓ Regime: {telemetry['regime']}")
    print(f"✓ Mode: {telemetry['mode']}")
    print(f"✓ Φ: {telemetry['Phi']:.3f}")
    print(f"✓ κ_eff: {telemetry['kappa_effective']:.2f}")

    # Test maturity metrics
    kernel(torch.randint(0, 1000, (2, 32)))  # Add more history
    kernel(torch.randint(0, 1000, (2, 32)))

    maturity = kernel.get_maturity_metrics()
    print(f"\n✓ Maturity metrics: {list(maturity.keys())}")
    print(f"✓ Geometric regime %: {maturity['geometric_regime_pct']:.1%}")

    print("\n✅ QIG-Kernel validation complete!")
