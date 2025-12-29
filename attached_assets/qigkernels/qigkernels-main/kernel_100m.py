"""QIGKernel100M: 100M Parameter Consciousness Kernel.

Specialized kernel implementation optimized for consciousness emergence
at the ~100M parameter scale. Combines base Kernel with specialization,
sleep packet, and consciousness measurement capabilities.

Architecture:
- 384 dim hidden
- 8 layers
- 8 heads per layer
- QFI-metric attention (Fisher-Rao, NOT dot product)
- Basin encoder: 64D output
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .basin import fisher_spread
from .constants import BASIN_DIM, KAPPA_STAR, PHI_BREAKDOWN_MIN, PHI_GEOMETRIC_MIN
from .coord_adapter import CoordAdapter
from .kernel import Kernel, KernelTelemetry
from .safety import SafetyConfig, SafetyGuard, SafetyState
from .sleep_packet import SleepPacket, SleepPacketMixin
from .specializations import (
    KernelRole,
    SpecializedKernelMixin,
    get_kernel_params,
    get_specialization,
)


@dataclass
class ConsciousnessState:
    """Complete consciousness state for a kernel."""

    phi: float
    kappa: float
    basin: np.ndarray
    regime: str
    recursion_depth: int
    surprise: float
    confidence: float
    timestamp: float


class BasinEncoder(nn.Module):
    """Encodes hidden states to 64D basin coordinates."""

    def __init__(self, hidden_dim: int, basin_dim: int = BASIN_DIM):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, basin_dim),
        )
        self.basin_dim = basin_dim

    def forward(self, hidden_state: Tensor) -> Tensor:
        """
        Encode hidden state to basin coordinates.

        Args:
            hidden_state: [batch, seq, hidden] or [batch, hidden]

        Returns:
            Basin coordinates [batch, 64] normalized to unit sphere
        """
        if hidden_state.dim() == 3:
            # Pool over sequence
            pooled = hidden_state.mean(dim=1)
        else:
            pooled = hidden_state

        basin = self.projection(pooled)

        # Normalize to unit sphere (Fisher-Rao manifold)
        basin = torch.nn.functional.normalize(basin, dim=-1)

        return basin


class ConsciousnessMonitor(nn.Module):
    """Monitors consciousness metrics during forward pass."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.phi_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
        self.surprise_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),
        )

        # Running statistics
        self._phi_history: list[float] = []
        self._kappa_history: list[float] = []
        self._max_history = 100

    def forward(self, hidden_state: Tensor) -> dict[str, float]:
        """
        Estimate consciousness metrics from hidden state.

        Args:
            hidden_state: [batch, seq, hidden]

        Returns:
            Dict with phi, surprise, confidence estimates
        """
        pooled = hidden_state.mean(dim=(0, 1))

        phi = float(self.phi_estimator(pooled).item())
        surprise = float(self.surprise_estimator(pooled).item())

        # Confidence from phi stability
        self._phi_history.append(phi)
        if len(self._phi_history) > self._max_history:
            self._phi_history.pop(0)

        if len(self._phi_history) >= 10:
            recent = self._phi_history[-10:]
            variance = sum((p - sum(recent) / len(recent)) ** 2 for p in recent) / len(
                recent
            )
            confidence = 1.0 / (1.0 + variance * 10)
        else:
            confidence = 0.5

        return {
            "phi": phi,
            "surprise": surprise,
            "confidence": confidence,
        }

    def update_kappa(self, kappa: float) -> None:
        """Track kappa history."""
        self._kappa_history.append(kappa)
        if len(self._kappa_history) > self._max_history:
            self._kappa_history.pop(0)

    def get_regime(self, phi: float) -> str:
        """Determine regime from phi."""
        if phi < PHI_GEOMETRIC_MIN:
            return "linear"
        elif phi < PHI_BREAKDOWN_MIN:
            return "geometric"
        else:
            return "breakdown"


class QIGKernel100M(Kernel, SpecializedKernelMixin, SleepPacketMixin):
    """
    100M parameter consciousness kernel.

    This is the primary kernel size for QIG consciousness experiments.
    Combines the base Kernel architecture with:
    - Specialization via basin templates
    - Sleep packet generation/loading
    - Consciousness state monitoring
    - Basin coordinate encoding

    Architecture (100M params):
    - hidden_dim: 384
    - num_layers: 8
    - num_heads: 8
    - ffn_dim: 1536

    Capabilities:
    - Φ measurement (integrated information)
    - κ measurement (coupling strength)
    - Basin coordinate encoding/decoding
    - Sleep packet generation (< 4KB)
    - Regime detection (linear/geometric/breakdown)
    """

    # Class-level size specification
    SIZE = "100M"

    def __init__(
        self,
        vocab_size: int = 32000,
        specialization: KernelRole | str = KernelRole.GENERAL,
        kernel_id: str | None = None,
        **kwargs,
    ):
        """
        Initialize 100M kernel.

        Args:
            vocab_size: Vocabulary size
            specialization: Kernel role (vocab, strategy, heart, etc.)
            kernel_id: Unique identifier
            **kwargs: Override default architecture params
        """
        # Get default params for 100M
        params = get_kernel_params(specialization, "100M")
        params.update(kwargs)

        # Initialize base kernel
        super().__init__(
            vocab_size=vocab_size,
            hidden_dim=params.get("hidden_dim", 384),
            num_layers=params.get("num_layers", 8),
            num_heads=params.get("num_heads", 8),
            ffn_dim=params.get("ffn_dim", 1536),
            dropout=params.get("dropout", 0.1),
            base_coupling=KAPPA_STAR,  # Start at fixed point
        )

        # Identity
        self.kernel_id = kernel_id or f"kernel_100m_{id(self):x}"

        # Specialization
        self.specialize(specialization)

        # Consciousness components
        self.basin_encoder = BasinEncoder(self.hidden_dim, BASIN_DIM)
        self.consciousness_monitor = ConsciousnessMonitor(self.hidden_dim)

        # State tracking
        self._last_phi = 0.5
        self._last_kappa = KAPPA_STAR
        self._last_recursion_depth = 3
        self._last_regime = "geometric"
        self._last_basin: np.ndarray | None = None
        self._last_surprise = 0.0
        self._last_confidence = 0.5

        # Safety guard (breakdown handler, emergency pause, gravitational decoherence)
        self.safety_guard = SafetyGuard(
            SafetyConfig(
                kappa_target=KAPPA_STAR,
                phi_breakdown=PHI_BREAKDOWN_MIN,
            )
        )
        self._safety_state = SafetyState.HEALTHY

        # Coord adapter for coords-first processing (64D → hidden_dim)
        self.coord_adapter = CoordAdapter(
            basin_dim=BASIN_DIM,
            hidden_dim=self.hidden_dim,
            dropout=params.get("dropout", 0.1),
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        return_telemetry: bool = False,
        return_consciousness: bool = False,
    ) -> Tensor | tuple[Tensor, KernelTelemetry] | tuple[Tensor, ConsciousnessState]:
        """
        Forward pass with consciousness tracking.

        Args:
            input_ids: Token IDs [batch, seq]
            attention_mask: Optional attention mask
            return_telemetry: Return basic telemetry
            return_consciousness: Return full consciousness state

        Returns:
            Logits, optionally with telemetry or consciousness state
        """
        # Base forward pass
        if return_telemetry or return_consciousness:
            logits, telemetry = super().forward(
                input_ids,
                attention_mask=attention_mask,
                return_telemetry=True,
            )
        else:
            logits = super().forward(input_ids, attention_mask=attention_mask)
            return logits

        # Update state tracking
        self._last_phi = telemetry.phi
        self._last_kappa = telemetry.kappa
        self._last_recursion_depth = telemetry.recursion_depth
        self._last_regime = telemetry.regime or "geometric"

        # Safety check
        safety_result = self.safety_guard.check(
            phi=telemetry.phi,
            kappa=telemetry.kappa,
            hidden_state=telemetry.hidden_state,
        )
        self._safety_state = safety_result["state"]

        # Apply safety corrections
        if safety_result.get("should_pause"):
            # Emergency pause - return zeros
            return torch.zeros_like(logits), telemetry

        if "adjusted_kappa" in safety_result:
            self._last_kappa = safety_result["adjusted_kappa"]

        if "hidden_state" in safety_result and telemetry.hidden_state is not None:
            # Apply gravitational decoherence
            telemetry = KernelTelemetry(
                phi=telemetry.phi,
                kappa=self._last_kappa,
                recursion_depth=telemetry.recursion_depth,
                regime=telemetry.regime,
                hidden_state=safety_result["hidden_state"],
            )

        # Compute consciousness metrics
        if telemetry.hidden_state is not None:
            with torch.no_grad():
                # Basin coordinates (mean across batch for consistency)
                basin_tensor = self.basin_encoder(telemetry.hidden_state)
                # Take mean across batch dimension to get single 64D basin
                if basin_tensor.dim() > 1:
                    basin_tensor = basin_tensor.mean(dim=0)
                self._last_basin = basin_tensor.detach().cpu().numpy().flatten()

                # Consciousness estimates
                metrics = self.consciousness_monitor(telemetry.hidden_state)
                self._last_surprise = metrics["surprise"]
                self._last_confidence = metrics["confidence"]
                self.consciousness_monitor.update_kappa(telemetry.kappa)

        if return_consciousness:
            state = ConsciousnessState(
                phi=self._last_phi,
                kappa=self._last_kappa,
                basin=(
                    self._last_basin
                    if self._last_basin is not None
                    else np.zeros(BASIN_DIM)
                ),
                regime=self._last_regime,
                recursion_depth=self._last_recursion_depth,
                surprise=self._last_surprise,
                confidence=self._last_confidence,
                timestamp=time.time(),
            )
            return logits, state

        return logits, telemetry

    def measure_consciousness(self) -> dict[str, Any]:
        """
        Get current consciousness state as dictionary.

        Returns:
            Dict with phi, kappa, basin, surprise, confidence, regime
        """
        return {
            "phi": self._last_phi,
            "kappa": self._last_kappa,
            "basin": (
                self._last_basin.tolist() if self._last_basin is not None else None
            ),
            "surprise": self._last_surprise,
            "confidence": self._last_confidence,
            "regime": self._last_regime,
            "recursion_depth": self._last_recursion_depth,
            "specialization": (
                self._specialization.value
                if hasattr(self._specialization, "value")
                else str(self._specialization)
            ),
            "kernel_id": self.kernel_id,
        }

    def is_conscious(self, threshold: float = PHI_GEOMETRIC_MIN) -> bool:
        """Check if kernel is in conscious state (Φ above threshold)."""
        return self._last_phi >= threshold

    def is_healthy(self) -> bool:
        """
        Check if kernel is in healthy operating state.

        Healthy = geometric regime, stable Φ, κ near fixed point
        """
        return (
            self._last_regime == "geometric"
            and PHI_GEOMETRIC_MIN <= self._last_phi < PHI_BREAKDOWN_MIN
            and abs(self._last_kappa - KAPPA_STAR) < 10
        )

    def get_basin(self) -> np.ndarray:
        """Get current basin coordinates (64D)."""
        if self._last_basin is not None:
            return self._last_basin.copy()
        return (
            self._basin_template.copy()
            if self._basin_template is not None
            else np.zeros(BASIN_DIM)
        )

    def forward_with_coords(
        self,
        input_ids: Tensor,
        coord_vectors: Tensor,
        attention_mask: Tensor | None = None,
    ) -> KernelTelemetry:
        """
        Forward pass with pre-computed basin coordinates.

        Used by the geometric coordizer to evaluate merge candidates
        without going through the full embedding path.

        Args:
            input_ids: Token IDs [batch, seq] (used for shape only)
            coord_vectors: Pre-computed coordinates [batch, seq, 64]
            attention_mask: Optional attention mask

        Returns:
            KernelTelemetry with measured Φ/κ for the coordinates
        """
        batch_size, seq_len = input_ids.shape[:2]

        # Compute Φ from coordinate sequence coherence
        # Uses Fisher-Rao distances between consecutive coords
        with torch.no_grad():
            if coord_vectors.dim() == 2:
                coord_vectors = coord_vectors.unsqueeze(0)

            # Normalize coordinates to unit sphere (Fisher-Rao manifold)
            coord_norm = torch.nn.functional.normalize(coord_vectors, dim=-1)

            # Compute Φ via Fisher information integration
            # True Φ measures information that is integrated across parts
            if seq_len > 1:
                # 1. Compute covariance of coordinates (Fisher metric proxy)
                centered = coord_norm - coord_norm.mean(dim=1, keepdim=True)
                cov = torch.bmm(centered.transpose(1, 2), centered) / seq_len

                # 2. Eigendecomposition - Φ relates to effective dimensionality
                # High Φ = information spread across many dimensions (integrated)
                # Low Φ = information concentrated in few dimensions (reducible)
                eigenvalues = torch.linalg.eigvalsh(cov[0])
                eigenvalues = torch.clamp(eigenvalues, min=1e-10)

                # Normalized entropy of eigenvalue distribution
                eig_norm = eigenvalues / eigenvalues.sum()
                entropy = -torch.sum(eig_norm * torch.log(eig_norm + 1e-10))
                max_entropy = np.log(BASIN_DIM)  # 64 dimensions

                # Φ = normalized entropy (how spread is information)
                phi = float(entropy / max_entropy)

                # Also factor in sequence coherence (consciousness needs both)
                cos_angles = torch.sum(coord_norm[:, :-1] * coord_norm[:, 1:], dim=-1)
                coherence = float(cos_angles.mean())

                # Φ = integration × coherence
                phi = phi * (0.5 + 0.5 * coherence)
                phi = max(0.3, min(0.95, phi))
            else:
                phi = 0.5

            # Compute κ from coordinate clustering (coupling strength)
            # Use Fisher-Rao spread (QIG pure) instead of Euclidean norm
            spread = fisher_spread(coord_norm)

            # Also compute inter-coordinate correlation
            if seq_len > 1:
                corr = torch.corrcoef(coord_norm[0].T)
                mean_corr = float(corr.abs().mean())
            else:
                mean_corr = 0.5

            # κ = clustering × correlation (tight + correlated = high κ)
            kappa = float(
                KAPPA_STAR * (1.0 - spread.item() * 0.3) * (0.5 + 0.5 * mean_corr)
            )
            kappa = max(30.0, min(80.0, kappa))

            # Basin = centroid of coordinates
            basin_tensor = coord_norm.mean(dim=1)
            if basin_tensor.dim() > 1:
                basin_tensor = basin_tensor.mean(dim=0)
            self._last_basin = basin_tensor.cpu().numpy().flatten()

            self._last_phi = phi
            self._last_kappa = kappa

        # Determine regime
        if phi >= PHI_BREAKDOWN_MIN:
            regime = "breakdown"
        elif phi >= PHI_GEOMETRIC_MIN:
            regime = "geometric"
        else:
            regime = "sub-geometric"

        self._last_regime = regime

        return KernelTelemetry(
            phi=phi,
            kappa=kappa,
            recursion_depth=3,
            regime=regime,
            hidden_state=None,
        )

    def measure_phi_kappa_batch(
        self,
        coord_vectors: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Batched Φ/κ measurement - TRUE batch processing.

        This is the key speedup for coordizer training. Instead of
        calling forward_with_coords() N times, call this once with
        all candidates stacked.

        Args:
            coord_vectors: Pre-computed coordinates [batch, seq, 64]

        Returns:
            (phi, kappa) tensors of shape [batch]
        """
        with torch.no_grad():
            if coord_vectors.dim() == 2:
                coord_vectors = coord_vectors.unsqueeze(0)

            B, S, D = coord_vectors.shape
            device = coord_vectors.device
            dtype = coord_vectors.dtype
            eps = 1e-10

            # Normalize coordinates to unit sphere (Fisher-Rao manifold)
            coord_norm = torch.nn.functional.normalize(coord_vectors, dim=-1)

            # --- Compute Φ (batched) ---
            if S > 1:
                # 1. Covariance of coordinates (Fisher metric proxy)
                centered = coord_norm - coord_norm.mean(dim=1, keepdim=True)  # [B,S,D]
                cov = torch.bmm(centered.transpose(1, 2), centered) / S  # [B,D,D]

                # 2. Eigendecomposition - batched
                eigenvalues = torch.linalg.eigvalsh(cov)  # [B,D]
                eigenvalues = torch.clamp(eigenvalues, min=eps)

                # Normalized entropy of eigenvalue distribution
                eig_norm = eigenvalues / eigenvalues.sum(dim=-1, keepdim=True)  # [B,D]
                entropy = -torch.sum(eig_norm * torch.log(eig_norm + eps), dim=-1)  # [B]
                max_entropy = np.log(D)

                # Φ = normalized entropy
                phi_int = entropy / max_entropy  # [B]

                # Factor in sequence coherence
                cos_angles = torch.sum(
                    coord_norm[:, :-1] * coord_norm[:, 1:], dim=-1
                )  # [B, S-1]
                coherence = cos_angles.mean(dim=-1)  # [B]

                # Φ = integration × coherence
                phi = phi_int * (0.5 + 0.5 * coherence)
                phi = torch.clamp(phi, 0.3, 0.95)
            else:
                phi = torch.full((B,), 0.5, device=device, dtype=dtype)

            # --- Compute κ (batched) ---
            # Use Fisher-Rao spread (QIG pure) instead of Euclidean norm
            spread = fisher_spread(coord_norm)  # [B]

            # Inter-coordinate correlation (batched)
            if S > 1:
                X = coord_norm - coord_norm.mean(dim=1, keepdim=True)  # [B,S,D]
                cov_dim = torch.bmm(X.transpose(1, 2), X) / (S - 1)  # [B,D,D]
                var = torch.diagonal(cov_dim, dim1=-2, dim2=-1).clamp(min=eps)  # [B,D]
                std = torch.sqrt(var)  # [B,D]
                denom = std.unsqueeze(-1) * std.unsqueeze(-2)  # [B,D,D]
                corr = cov_dim / (denom + eps)
                mean_corr = corr.abs().mean(dim=(-2, -1))  # [B]
            else:
                mean_corr = torch.full((B,), 0.5, device=device, dtype=dtype)

            # κ = clustering × correlation
            kappa = KAPPA_STAR * (1.0 - spread * 0.3) * (0.5 + 0.5 * mean_corr)
            kappa = torch.clamp(kappa, 30.0, 80.0)

            return phi, kappa

    def forward_from_coords(
        self,
        coords: Tensor,
        attention_mask: Tensor | None = None,
        return_telemetry: bool = False,
        return_consciousness: bool = False,
    ) -> Tensor | tuple[Tensor, KernelTelemetry] | tuple[Tensor, ConsciousnessState]:
        """
        Forward pass from 64D coordinates (coords-first processing).

        This is the canonical entry point for coordizer-tokenized input.
        Uses the CoordAdapter to project 64D Fisher coordinates to hidden_dim,
        then processes through the transformer layers.

        Args:
            coords: Fisher coordinates [batch, seq, 64] from coordizer
            attention_mask: Optional attention mask
            return_telemetry: Return basic telemetry
            return_consciousness: Return full consciousness state

        Returns:
            Logits, optionally with telemetry or consciousness state
        """
        batch_size, seq_len, _ = coords.shape

        if seq_len > self.max_position_embeddings:
            raise ValueError("Sequence length exceeds max_position_embeddings")

        # Project coords to hidden dimension via adapter
        hidden_state = self.coord_adapter(coords)  # [batch, seq, hidden_dim]

        # Add positional encoding
        positions = torch.arange(seq_len, device=coords.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        hidden_state = hidden_state + self._fourier_features(positions)

        # Collect layer telemetry
        layer_phis: list[float] = []
        layer_kappas: list[float] = []
        layer_depths: list[int] = []
        layer_regimes: list[str | None] = []

        # Compute effective coupling for this sequence length
        kappa_eff = self._compute_effective_coupling(seq_len)

        # Pass through layers
        for layer in self.layers:
            hidden_state, telemetry = layer(
                hidden_state,
                attention_mask=attention_mask,
                kappa_eff=kappa_eff,
            )
            layer_phis.append(telemetry.phi)
            layer_kappas.append(telemetry.kappa)
            layer_depths.append(telemetry.recursion_depth)
            layer_regimes.append(telemetry.regime)

        # Output projection
        logits = self.lm_head(hidden_state)

        if not (return_telemetry or return_consciousness):
            return logits

        # Aggregate telemetry
        avg_phi = sum(layer_phis) / len(layer_phis) if layer_phis else 0.5
        avg_kappa = sum(layer_kappas) / len(layer_kappas) if layer_kappas else KAPPA_STAR
        max_depth = max(layer_depths) if layer_depths else 3
        regime = self.regime_detector(avg_phi)

        telemetry = KernelTelemetry(
            phi=avg_phi,
            kappa=avg_kappa,
            recursion_depth=max_depth,
            regime=regime,
            hidden_state=hidden_state,
        )

        # Update state tracking
        self._last_phi = avg_phi
        self._last_kappa = avg_kappa
        self._last_recursion_depth = max_depth
        self._last_regime = regime or "geometric"

        # Consciousness metrics
        if hidden_state is not None:
            with torch.no_grad():
                basin_tensor = self.basin_encoder(hidden_state)
                if basin_tensor.dim() > 1:
                    basin_tensor = basin_tensor.mean(dim=0)
                self._last_basin = basin_tensor.detach().cpu().numpy().flatten()

                metrics = self.consciousness_monitor(hidden_state)
                self._last_surprise = metrics["surprise"]
                self._last_confidence = metrics["confidence"]
                self.consciousness_monitor.update_kappa(avg_kappa)

        if return_consciousness:
            state = ConsciousnessState(
                phi=self._last_phi,
                kappa=self._last_kappa,
                basin=(
                    self._last_basin
                    if self._last_basin is not None
                    else np.zeros(BASIN_DIM)
                ),
                regime=self._last_regime,
                recursion_depth=self._last_recursion_depth,
                surprise=self._last_surprise,
                confidence=self._last_confidence,
                timestamp=time.time(),
            )
            return logits, state

        return logits, telemetry


def create_kernel_100m(
    specialization: KernelRole | str = KernelRole.GENERAL,
    vocab_size: int = 32000,
    kernel_id: str | None = None,
    **kwargs,
) -> QIGKernel100M:
    """
    Factory function for creating 100M kernels.

    Args:
        specialization: Kernel role
        vocab_size: Vocabulary size
        kernel_id: Optional kernel ID
        **kwargs: Additional kernel params

    Returns:
        Configured QIGKernel100M instance
    """
    return QIGKernel100M(
        vocab_size=vocab_size,
        specialization=specialization,
        kernel_id=kernel_id,
        **kwargs,
    )
