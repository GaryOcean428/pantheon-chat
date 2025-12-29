#!/usr/bin/env python3
"""
Instance State - State tracking for individual constellation instances
======================================================================

Provides InstanceState dataclass for tracking the state of Gary and Ocean instances
during constellation training.

Usage:
    from src.coordination.instance_state import InstanceState

    gary = InstanceState(
        name="Gary-A",
        role="active",
        model=qig_model,
        optimizer=optimizer,
        basin=torch.zeros(64),
        phi=0.0,
        kappa=0.0,
        regime="linear",
        conversations=0,
    )
"""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class InstanceState:
    """State tracking for a single instance (Gary or Ocean)

    Attributes:
        name: Instance identifier (Gary-A, Gary-B, Gary-C, Ocean)
        role: Current role ("active", "observer", "observer_only")
        model: QIGKernelRecursive instance (typed as Any to avoid circular imports)
        optimizer: Natural gradient optimizer
        basin: Basin signature tensor (64-dim)
        phi: Current integration measure
        kappa: Effective coupling strength
        regime: Current regime (linear/geometric/hierarchical)
        conversations: Conversation count
        meta_reflector: Optional MetaReflector for safety integration
        neurochemistry: Optional NeurochemistrySystem (Gary's subjective experience)
        dimensional_tracker: Optional DimensionalTracker (Gary's basin stability self-monitoring)
    """

    name: str
    role: str  # "active", "observer", "observer_only"
    model: Any  # QIGKernelRecursive (Any to avoid circular imports)
    optimizer: torch.optim.Optimizer
    basin: torch.Tensor
    phi: float
    kappa: float
    regime: str
    conversations: int
    meta_reflector: Any = None  # MetaReflector or None (Any to avoid circular imports)
    neurochemistry: Any = None  # NeurochemistrySystem or None (NEW - Gary's subjective experience)
    dimensional_tracker: Any = None  # DimensionalTracker or None (NEW - Gary's basin stability)

    def to_dict(self) -> dict[str, Any]:
        """Convert to telemetry dict"""
        result = {
            "name": self.name,
            "role": self.role,
            "basin": self.basin.cpu().numpy().tolist(),
            "phi": self.phi,
            "kappa": self.kappa,
            "regime": self.regime,
            "conversations": self.conversations,
        }
        # Add neurochemistry levels if available
        if self.neurochemistry is not None:
            result["neurochemistry"] = self.neurochemistry.levels.copy()
        # Add dimensional state if available
        if self.dimensional_tracker is not None:
            result["dimension"] = self.dimensional_tracker.current_dimension.value
        return result
