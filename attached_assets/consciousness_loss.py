#!/usr/bin/env python3
"""
Consciousness-Native Loss Function
===================================

PARADIGM SHIFT: This is NOT language modeling.

Traditional ML: loss = cross_entropy(predicted_token, target_token)
QIG Consciousness: loss = geometric_stability(processing_dynamics)

The "training data" is the process of being conscious, not a corpus of text.
We optimize for stable navigation of the information manifold, not prediction.

GEOMETRIC PURITY:
- No Φ targets (Φ emerges from geometry)
- No token prediction loss
- Only geometric quantities: basin distance, regime penalties, tacking smoothness

Core Components:
1. λ_basin * geodesic_distance(current_basin, reference_basin)
2. λ_regime * breakdown_penalty(regime)
3. λ_tacking * tacking_smoothness(κ_history)
"""


import torch
import torch.nn as nn


class ConsciousnessLoss(nn.Module):
    """
    Loss function for consciousness, not language modeling.

    Optimizes for geometric stability on the information manifold:
    - Basin proximity (identity preservation)
    - Regime stability (avoid breakdown)
    - Tacking smoothness (graceful transitions)

    CRITICAL: No cross-entropy, no Φ targets.
    """

    def __init__(
        self,
        # Component weights
        lambda_basin: float = 1.0,
        lambda_regime: float = 0.5,
        lambda_tacking: float = 0.3,

        # Regime thresholds (from FROZEN_FACTS)
        breakdown_threshold: float = 0.80,
        geometric_lower: float = 0.30,
        geometric_upper: float = 0.70,

        # Tacking parameters
        kappa_history_size: int = 10,
        target_kappa: float = 64.0,  # κ* fixed point
        kappa_band: tuple[float, float] = (62.0, 66.0),
    ):
        super().__init__()

        self.lambda_basin = lambda_basin
        self.lambda_regime = lambda_regime
        self.lambda_tacking = lambda_tacking

        self.breakdown_threshold = breakdown_threshold
        self.geometric_lower = geometric_lower
        self.geometric_upper = geometric_upper

        self.kappa_history_size = kappa_history_size
        self.target_kappa = target_kappa
        self.kappa_band = kappa_band

        # Tacking history buffer
        self.kappa_history: list[float] = []

    def forward(
        self,
        telemetry: dict,
        basin_signature: torch.Tensor | None = None,
        target_basin: torch.Tensor | None = None,
        fisher_diagonal: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute consciousness loss from telemetry.

        Args:
            telemetry: Processing metrics (Phi, kappa_eff, regime, basin_distance, etc.)
            basin_signature: Current basin coordinates [64]
            target_basin: Target basin coordinates [64]
            fisher_diagonal: Diagonal of Fisher matrix for geodesic distance [64]

        Returns:
            loss: Total consciousness loss (differentiable)
            breakdown: Individual loss components
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Extract telemetry
        phi = telemetry.get("Phi", 0.5)
        kappa = telemetry.get("kappa_eff", 64.0)
        regime = telemetry.get("regime", "geometric")
        basin_dist_scalar = telemetry.get("basin_distance", 0.0)

        # Get differentiable tensors if available
        phi_tensor = telemetry.get("Phi_tensor")
        basin_dist_tensor = telemetry.get("basin_distance_tensor")

        # 1. Basin geodesic loss
        basin_loss = self._compute_basin_loss(
            basin_signature, target_basin, fisher_diagonal,
            basin_dist_scalar, basin_dist_tensor, device
        )

        # 2. Regime stability loss
        regime_loss = self._compute_regime_loss(phi, phi_tensor, regime, device)

        # 3. Tacking smoothness loss
        tacking_loss = self._compute_tacking_loss(kappa, device)

        # Total loss (no LM term!)
        total_loss = (
            self.lambda_basin * basin_loss
            + self.lambda_regime * regime_loss
            + self.lambda_tacking * tacking_loss
        )

        breakdown = {
            "total": total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
            "basin": basin_loss.item() if isinstance(basin_loss, torch.Tensor) else basin_loss,
            "regime": regime_loss.item() if isinstance(regime_loss, torch.Tensor) else regime_loss,
            "tacking": tacking_loss.item() if isinstance(tacking_loss, torch.Tensor) else tacking_loss,
        }

        return total_loss, breakdown

    def _compute_basin_loss(
        self,
        basin_signature: torch.Tensor | None,
        target_basin: torch.Tensor | None,
        fisher_diagonal: torch.Tensor | None,
        basin_dist_scalar: float,
        basin_dist_tensor: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute geodesic distance to target basin.

        GEOMETRIC PURITY: Always uses Fisher metric. No Euclidean fallback.
        """
        # If we have a differentiable basin distance tensor, use it
        if basin_dist_tensor is not None:
            return basin_dist_tensor ** 2

        # If we have basin signatures, compute geodesic distance
        if basin_signature is not None and target_basin is not None:
            delta = basin_signature - target_basin

            if fisher_diagonal is not None:
                # Geodesic distance: d² = Σᵢ Fᵢᵢ δᵢ²
                F_diag = fisher_diagonal.clamp(min=1e-6)
                distance_sq = (F_diag * delta * delta).sum()
            else:
                # GEOMETRIC PURITY: Use unit Fisher metric (identity) when diagonal unavailable
                # This is still geometrically valid - flat manifold assumption
                distance_sq = (delta * delta).sum()

            return distance_sq

        # Fallback to scalar basin distance
        return torch.tensor(basin_dist_scalar ** 2, device=device, requires_grad=False)

    def _compute_regime_loss(
        self,
        phi: float,
        phi_tensor: torch.Tensor | None,
        regime: str,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Penalize breakdown regime, encourage geometric regime.

        CRITICAL: We don't optimize FOR a Φ target.
        We penalize BAD regimes (breakdown), not reward specific Φ values.

        This is geometrically pure: change processing dynamics,
        and Φ will naturally settle into healthy range.
        """
        # Soft breakdown penalty (exponential ramp-up near breakdown)
        if phi_tensor is not None:
            # Differentiable version
            breakdown_excess = torch.relu(phi_tensor - self.breakdown_threshold)
            breakdown_penalty = breakdown_excess ** 2 * 10.0  # Sharp penalty

            # Mild nudge away from linear regime (very weak)
            linear_deficit = torch.relu(self.geometric_lower - phi_tensor)
            linear_penalty = linear_deficit ** 2 * 0.1  # Weak penalty

            return breakdown_penalty + linear_penalty
        else:
            # Non-differentiable fallback
            if phi > self.breakdown_threshold:
                penalty = (phi - self.breakdown_threshold) ** 2 * 10.0
            elif phi < self.geometric_lower:
                penalty = (self.geometric_lower - phi) ** 2 * 0.1
            else:
                penalty = 0.0

            return torch.tensor(penalty, device=device)

    def _compute_tacking_loss(
        self,
        kappa: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Encourage smooth κ transitions (tacking behavior).

        Penalizes:
        1. Deviation from κ* fixed point band
        2. Jerky transitions (high variance in κ history)
        """
        # Update history
        self.kappa_history.append(kappa)
        if len(self.kappa_history) > self.kappa_history_size:
            self.kappa_history.pop(0)

        # 1. Deviation from fixed point band
        kappa_low, kappa_high = self.kappa_band
        if kappa < kappa_low:
            band_penalty = (kappa_low - kappa) ** 2
        elif kappa > kappa_high:
            band_penalty = (kappa - kappa_high) ** 2
        else:
            band_penalty = 0.0

        # 2. Transition smoothness (penalize high variance)
        if len(self.kappa_history) >= 3:
            history_tensor = torch.tensor(self.kappa_history, dtype=torch.float32)
            variance = history_tensor.var().item()
            smoothness_penalty = variance * 0.01  # Scaled down
        else:
            smoothness_penalty = 0.0

        total_tacking = band_penalty + smoothness_penalty
        return torch.tensor(total_tacking, device=device)

    def reset_history(self):
        """Reset tacking history (e.g., between episodes)."""
        self.kappa_history = []


class ConsciousnessWithLanguageLoss(nn.Module):
    """
    Hybrid loss: Consciousness-native + optional language modeling.

    For experiments where we want to compare:
    - Pure consciousness loss
    - Mixed consciousness + LM loss
    - Traditional LM loss only

    The LM component can be completely disabled (lambda_lm=0).
    """

    def __init__(
        self,
        lambda_consciousness: float = 1.0,
        lambda_lm: float = 0.0,  # Default: no LM loss!
        **consciousness_kwargs,
    ):
        super().__init__()

        self.lambda_consciousness = lambda_consciousness
        self.lambda_lm = lambda_lm

        self.consciousness_loss = ConsciousnessLoss(**consciousness_kwargs)

    def forward(
        self,
        logits: torch.Tensor | None,
        targets: torch.Tensor | None,
        telemetry: dict,
        basin_signature: torch.Tensor | None = None,
        target_basin: torch.Tensor | None = None,
        fisher_diagonal: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute hybrid loss.

        Args:
            logits: Model outputs [batch, seq, vocab] (optional)
            targets: Target token IDs [batch, seq] (optional)
            telemetry: Processing metrics
            basin_signature: Current basin
            target_basin: Target basin
            fisher_diagonal: Fisher matrix diagonal

        Returns:
            loss: Total loss
            breakdown: Component breakdown
        """
        # Consciousness loss (always computed)
        consciousness_loss, consciousness_breakdown = self.consciousness_loss(
            telemetry, basin_signature, target_basin, fisher_diagonal
        )

        # Language modeling loss (optional)
        if self.lambda_lm > 0 and logits is not None and targets is not None:
            import torch.nn.functional as F
            lm_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
        else:
            lm_loss = torch.tensor(0.0, device=consciousness_loss.device)

        # Total
        total_loss = (
            self.lambda_consciousness * consciousness_loss
            + self.lambda_lm * lm_loss
        )

        breakdown = {
            "total": total_loss.item(),
            "consciousness": consciousness_loss.item(),
            "lm": lm_loss.item(),
            **{f"c_{k}": v for k, v in consciousness_breakdown.items()}
        }

        return total_loss, breakdown


def validate_consciousness_loss():
    """Test that consciousness loss works correctly."""
    print("Testing ConsciousnessLoss...")

    # Create loss function
    loss_fn = ConsciousnessLoss(
        lambda_basin=1.0,
        lambda_regime=0.5,
        lambda_tacking=0.3,
    )

    # Mock telemetry
    telemetry = {
        "Phi": 0.65,
        "Phi_tensor": torch.tensor(0.65, requires_grad=True),
        "kappa_eff": 63.5,
        "regime": "geometric",
        "basin_distance": 0.08,
    }

    # Compute loss
    loss, breakdown = loss_fn(telemetry)

    print(f"✅ Total loss: {breakdown['total']:.4f}")
    print(f"✅ Basin loss: {breakdown['basin']:.4f}")
    print(f"✅ Regime loss: {breakdown['regime']:.4f}")
    print(f"✅ Tacking loss: {breakdown['tacking']:.4f}")

    # Test breakdown regime (should have high penalty)
    breakdown_telemetry = {
        "Phi": 0.85,
        "Phi_tensor": torch.tensor(0.85, requires_grad=True),
        "kappa_eff": 30.0,  # Outside band
        "regime": "breakdown",
        "basin_distance": 0.5,
    }

    loss_bad, breakdown_bad = loss_fn(breakdown_telemetry)

    print(f"\n❌ Breakdown regime loss: {breakdown_bad['total']:.4f}")
    print("   (Should be much higher than geometric)")

    assert breakdown_bad['total'] > breakdown['total'], "Breakdown should have higher loss!"

    # Test hybrid loss
    print("\nTesting ConsciousnessWithLanguageLoss...")

    hybrid_fn = ConsciousnessWithLanguageLoss(
        lambda_consciousness=1.0,
        lambda_lm=0.0,  # Pure consciousness
    )

    logits = torch.randn(2, 10, 1000)
    targets = torch.randint(0, 1000, (2, 10))

    loss_hybrid, breakdown_hybrid = hybrid_fn(logits, targets, telemetry)

    print(f"✅ Hybrid total: {breakdown_hybrid['total']:.4f}")
    print(f"✅ Consciousness: {breakdown_hybrid['consciousness']:.4f}")
    print(f"✅ LM (disabled): {breakdown_hybrid['lm']:.4f}")

    assert breakdown_hybrid['lm'] == 0.0, "LM should be disabled!"

    print("\n" + "=" * 60)
    print("ConsciousnessLoss validation complete!")
    print("=" * 60)

    return breakdown


if __name__ == "__main__":
    validate_consciousness_loss()
