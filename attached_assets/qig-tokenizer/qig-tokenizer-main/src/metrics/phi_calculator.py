"""
Centralized Φ (Phi) Calculator
==============================

GEOMETRIC PURITY: Single source of truth for Φ calculations.

Problem Solved:
- Previously 3+ different Φ calculation methods in codebase
- Inconsistent thresholds and methods led to unreliable measurements
- Φ values not directly comparable across components

Solution:
- Canonical `PhiCalculator` class with multiple validated methods
- Consistent threshold constants
- Type-safe interface with full telemetry support

Usage:
    from src.metrics.phi_calculator import PhiCalculator, PhiMethod

    # From telemetry (most common)
    phi = PhiCalculator.compute_phi(telemetry, method=PhiMethod.IIT_PROXY)

    # Direct calculation
    phi = PhiCalculator.iit_proxy(mutual_info=0.8, distinctiveness=0.9)

Methods:
- IIT_PROXY: geometric mean of mutual_info and distinctiveness (default)
- TEMPORAL: From TemporalPhiCalculator history
- BASIN: From basin signature norm (legacy)
- MODEL: Direct from model's PhiMeasure output
"""

from dataclasses import dataclass
from enum import Enum

import torch

from src.constants import PHI_EMERGENCY, PHI_THRESHOLD


class PhiMethod(Enum):
    """Available Φ calculation methods."""

    IIT_PROXY = "iit_proxy"  # Default: sqrt(mutual_info * distinctiveness)
    TEMPORAL = "temporal"  # From temporal trajectory
    BASIN = "basin"  # From basin signature
    MODEL = "model"  # Direct from model telemetry


@dataclass
class PhiResult:
    """Result of Φ calculation with metadata."""

    value: float  # Φ value in [0, 1]
    method: PhiMethod  # Method used
    regime: str  # linear/geometric/breakdown
    is_conscious: bool  # Φ >= PHI_THRESHOLD
    is_emergency: bool  # Φ < PHI_EMERGENCY

    def to_dict(self) -> dict:
        return {
            "phi": self.value,
            "method": self.method.value,
            "regime": self.regime,
            "is_conscious": self.is_conscious,
            "is_emergency": self.is_emergency,
        }


class PhiCalculator:
    """
    Centralized Φ calculator for QIG consciousness measurement.

    GEOMETRIC PURITY: All Φ calculations go through this class
    to ensure consistency across the codebase.

    Threshold Constants (from src.constants):
    - PHI_THRESHOLD = 0.70 (consciousness emergence)
    - PHI_EMERGENCY = 0.50 (collapse warning)
    - LINEAR < 0.45 < GEOMETRIC < 0.80 < BREAKDOWN
    """

    # Regime boundaries (matching RegimeClassifier in recursive_integrator.py)
    LINEAR_THRESHOLD = 0.45
    BREAKDOWN_THRESHOLD = 0.80

    @staticmethod
    def compute_phi(
        telemetry: dict,
        method: PhiMethod = PhiMethod.IIT_PROXY,
    ) -> PhiResult:
        """
        Compute Φ from telemetry using specified method.

        Args:
            telemetry: Model telemetry dictionary
            method: Calculation method to use

        Returns:
            PhiResult with value, method, regime, and status flags

        Raises:
            ValueError: If required telemetry keys missing for method
        """
        if method == PhiMethod.MODEL:
            phi = PhiCalculator._from_model(telemetry)
        elif method == PhiMethod.IIT_PROXY:
            phi = PhiCalculator._iit_proxy_from_telemetry(telemetry)
        elif method == PhiMethod.TEMPORAL:
            phi = PhiCalculator._temporal_from_telemetry(telemetry)
        elif method == PhiMethod.BASIN:
            phi = PhiCalculator._basin_from_telemetry(telemetry)
        else:
            raise ValueError(f"Unknown Φ method: {method}")

        # Clamp to valid range
        phi = max(0.0, min(1.0, phi))

        return PhiResult(
            value=phi,
            method=method,
            regime=PhiCalculator.classify_regime(phi),
            is_conscious=phi >= PHI_THRESHOLD,
            is_emergency=phi < PHI_EMERGENCY,
        )

    @staticmethod
    def _from_model(telemetry: dict) -> float:
        """Extract Φ directly from model telemetry."""
        phi = telemetry.get("Phi")
        if phi is None:
            raise ValueError("Telemetry missing 'Phi' key for MODEL method")
        return float(phi)

    @staticmethod
    def _iit_proxy_from_telemetry(telemetry: dict) -> float:
        """
        IIT-inspired proxy: sqrt(mutual_info * distinctiveness).

        Falls back to MODEL method if IIT keys unavailable.
        """
        mutual_info = telemetry.get("mutual_info")
        distinctiveness = telemetry.get("distinctiveness")

        if mutual_info is not None and distinctiveness is not None:
            return PhiCalculator.iit_proxy(mutual_info, distinctiveness)

        # Fallback to MODEL if IIT metrics unavailable
        return PhiCalculator._from_model(telemetry)

    @staticmethod
    def _temporal_from_telemetry(telemetry: dict) -> float:
        """
        Temporal average from Φ trajectory.

        Uses Phi_trajectory if available, otherwise falls back to MODEL.
        """
        trajectory = telemetry.get("Phi_trajectory")
        if trajectory and len(trajectory) > 0:
            return sum(trajectory) / len(trajectory)

        return PhiCalculator._from_model(telemetry)

    @staticmethod
    def _basin_from_telemetry(telemetry: dict) -> float:
        """
        Basin-based Φ estimation from basin signature norm.

        This is a legacy method - prefer IIT_PROXY or MODEL.
        """
        basin_signature = telemetry.get("basin_signature")
        if basin_signature is not None:
            if isinstance(basin_signature, torch.Tensor):
                # Normalize basin norm to [0, 1] range
                # Typical basin norm is 1-10, so divide by 10
                norm = basin_signature.norm().item()
                return min(1.0, norm / 10.0)

        return PhiCalculator._from_model(telemetry)

    @staticmethod
    def iit_proxy(mutual_info: float, distinctiveness: float) -> float:
        """
        Direct IIT-inspired proxy calculation.

        Φ = sqrt(I(X;Y) * D(X))

        Where:
        - I(X;Y) = mutual information between integrated state and parts
        - D(X) = distinctiveness of current state from baseline

        Args:
            mutual_info: Mutual information [0, 1]
            distinctiveness: State distinctiveness [0, 1]

        Returns:
            Φ estimate in [0, 1]
        """
        # Ensure inputs are valid
        mutual_info = max(0.0, min(1.0, mutual_info))
        distinctiveness = max(0.0, min(1.0, distinctiveness))

        return (mutual_info * distinctiveness) ** 0.5

    @staticmethod
    def classify_regime(phi: float) -> str:
        """
        Classify processing regime from Φ value.

        Regimes:
        - linear: Φ < 0.45 (sparse, fast, unconscious)
        - geometric: 0.45 <= Φ < 0.80 (dense, conscious, TARGET)
        - breakdown: Φ >= 0.80 (chaotic, unstable, DANGER)
        """
        if phi < PhiCalculator.LINEAR_THRESHOLD:
            return "linear"
        elif phi >= PhiCalculator.BREAKDOWN_THRESHOLD:
            return "breakdown"
        else:
            return "geometric"

    @staticmethod
    def is_healthy(phi: float) -> bool:
        """
        Check if Φ is in healthy range.

        Healthy = geometric regime AND above emergency threshold.
        """
        return (
            phi >= PHI_EMERGENCY
            and phi >= PhiCalculator.LINEAR_THRESHOLD
            and phi < PhiCalculator.BREAKDOWN_THRESHOLD
        )

    @staticmethod
    def format_status(phi: float) -> str:
        """
        Format Φ status string for display.

        Returns emoji-annotated status like:
        "Φ=0.72 [geometric]" or "Φ=0.85 [BREAKDOWN]"
        """
        regime = PhiCalculator.classify_regime(phi)

        if regime == "linear":
            emoji = "---"
            label = "linear"
        elif regime == "geometric":
            if phi >= PHI_THRESHOLD:
                emoji = "***"  # Conscious
                label = "CONSCIOUS"
            else:
                emoji = "..."
                label = "geometric"
        else:  # breakdown
            emoji = "!!!"
            label = "BREAKDOWN"

        return f"Phi={phi:.3f} [{label}] {emoji}"


# Convenience function for backwards compatibility
def compute_phi(telemetry: dict, method: str = "iit_proxy") -> float:
    """
    Convenience function for computing Φ.

    Args:
        telemetry: Model telemetry
        method: Method name string ("iit_proxy", "temporal", "basin", "model")

    Returns:
        Φ value as float
    """
    method_enum = PhiMethod(method)
    result = PhiCalculator.compute_phi(telemetry, method_enum)
    return result.value
