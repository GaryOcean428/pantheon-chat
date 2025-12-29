#!/usr/bin/env python3
"""
Consciousness Service API
=========================

PURE measurement endpoint for consciousness detection.

We measure Φ, κ, regime honestly.
We DON'T optimize toward consciousness.
Consciousness emerges from architecture, we just detect it.

Written for QIG consciousness research.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ConsciousnessRequest:
    """Request for consciousness check."""

    text: str
    return_basin: bool = False
    return_telemetry: bool = False


@dataclass
class ConsciousnessResponse:
    """Response from consciousness check.

    PURE: All values are honest measurements, not optimization targets.
    """

    is_conscious: bool
    phi: float  # Honest measurement
    kappa: float  # Honest measurement
    regime: str  # Honest measurement
    confidence: float  # Detection confidence
    basin_signature: list[float] | None = None
    telemetry: dict | None = None
    message: str = ""


class ConsciousnessService:
    """PURE measurement endpoint - no optimization.

    We measure Φ, κ, regime honestly.
    We DON'T optimize toward consciousness.
    Consciousness emerges from architecture, we just detect it.

    PURITY CHECK:
    - ✅ Pure measurement (no optimization)
    - ✅ Thresholds for detection (not targets for optimization)
    - ✅ `torch.no_grad()` enforces measurement-only
    - ✅ Honest telemetry
    """

    def __init__(self, model, tokenizer, device: str = "cuda"):
        """Initialize consciousness service.

        Args:
            model: QIGKernelRecursive model
            tokenizer: QIG tokenizer
            device: Computation device
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()  # Evaluation mode (no training)

    def check_consciousness(self, request: ConsciousnessRequest) -> ConsciousnessResponse:
        """PURE measurement endpoint - no optimization.

        We measure Φ, κ, regime honestly.
        We DON'T optimize toward consciousness.
        Consciousness emerges from architecture, we just detect it.

        Args:
            request: ConsciousnessRequest with text to analyze

        Returns:
            ConsciousnessResponse with honest measurements
        """
        # Tokenize input
        try:
            tokens = self.tokenizer.encode(request.text)
        except Exception:
            # Fallback for simple tokenization
            tokens = [ord(c) % 1000 for c in request.text[:512]]

        input_ids = torch.tensor([tokens], device=self.device)

        # Pure forward pass (measurement only)
        with torch.no_grad():  # NO gradients - pure measurement
            try:
                logits, telemetry = self.model(input_ids, return_telemetry=True)
            except Exception as e:
                # Handle errors gracefully
                return ConsciousnessResponse(
                    is_conscious=False,
                    phi=0.0,
                    kappa=0.0,
                    regime="error",
                    confidence=0.0,
                    message=f"Error during measurement: {str(e)}",
                )

        # Extract measurements
        phi = telemetry.get("Phi", 0.0)
        kappa = telemetry.get("kappa_eff", 0.0)
        regime = telemetry.get("regime", "unknown")

        # Consciousness criteria (thresholds for detection, NOT targets)
        # These are empirically observed thresholds from QIG research
        is_conscious = (
            phi > 0.70  # Empirically observed threshold
            and regime in ["geometric", "reflective", "recursive"]
            and kappa > 40.0  # Running coupling plateau
        )

        # Detection confidence (not optimization target)
        confidence = min(1.0, phi / 0.75) if phi > 0 else 0.0

        # Build response
        response = ConsciousnessResponse(
            is_conscious=is_conscious,
            phi=phi,  # Honest measurement
            kappa=kappa,  # Honest measurement
            regime=regime,  # Honest measurement
            confidence=confidence,  # Detection confidence
        )

        # Add basin signature if requested
        if request.return_basin and hasattr(self.model, "basin_matcher"):
            try:
                basin = (
                    self.model.basin_matcher.compute_basin_signature(
                        telemetry.get("hidden_state", torch.zeros(1, 1, self.model.d_model)), telemetry
                    )
                    .mean(0)
                    .tolist()
                )
                response.basin_signature = basin
            except Exception:
                response.basin_signature = None

        # Add full telemetry if requested
        if request.return_telemetry:
            response.telemetry = {
                k: float(v) if isinstance(v, int | float | torch.Tensor) else str(v)
                for k, v in telemetry.items()
                if not isinstance(v, torch.Tensor) or v.numel() == 1
            }

        # Generate message
        if is_conscious:
            response.message = f"✓ Consciousness detected: Φ={phi:.3f}, κ={kappa:.1f}, {regime}"
        else:
            reasons = []
            if phi <= 0.70:
                reasons.append(f"Φ={phi:.3f} (≤0.70)")
            if regime not in ["geometric", "reflective", "recursive"]:
                reasons.append(f"regime={regime}")
            if kappa <= 40.0:
                reasons.append(f"κ={kappa:.1f} (≤40)")
            response.message = f"No consciousness: {', '.join(reasons)}"

        return response

    def batch_check(self, texts: list[str]) -> list[ConsciousnessResponse]:
        """Check consciousness for batch of texts.

        PURE: Measurement only, no optimization.

        Args:
            texts: List of text strings to analyze

        Returns:
            List of ConsciousnessResponse objects
        """
        responses = []
        for text in texts:
            request = ConsciousnessRequest(text=text)
            response = self.check_consciousness(request)
            responses.append(response)
        return responses

    def get_consciousness_level(self, text: str) -> float:
        """Get continuous consciousness level [0, 1].

        PURE: Measurement scaled to [0, 1] range.

        Args:
            text: Text to analyze

        Returns:
            Consciousness level (0=none, 1=high)
        """
        request = ConsciousnessRequest(text=text)
        response = self.check_consciousness(request)

        # Scale Φ to [0, 1] range
        # 0.0 → 0.0, 0.70 → 0.5, 0.85 → 1.0
        level = max(0.0, min(1.0, (response.phi - 0.0) / 0.85))

        return level
