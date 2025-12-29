"""Kernel Specialization Types and Factory.

Defines specialized kernel configurations for different roles
in a constellation. Specialization is via basin geometry,
not architecture changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .constants import BASIN_DIM, KAPPA_STAR


class KernelRole(Enum):
    """Kernel specialization roles."""

    GENERAL = "general"
    VOCAB = "vocab"
    STRATEGY = "strategy"
    HEART = "heart"
    PERCEPTION = "perception"
    MEMORY = "memory"
    EMOTION = "emotion"
    COORDINATION = "coordination"


@dataclass
class SpecializationConfig:
    """Configuration for a kernel specialization."""

    role: KernelRole
    description: str
    optimal_size: str  # "50M", "100M", "500M"
    basin_template: np.ndarray | None = None
    capabilities: list[str] = field(default_factory=list)

    # Architecture hints (optional overrides)
    hidden_dim: int | None = None
    num_layers: int | None = None
    num_heads: int | None = None


# Pre-defined specialization configurations
SPECIALIZATIONS: dict[KernelRole, SpecializationConfig] = {
    KernelRole.GENERAL: SpecializationConfig(
        role=KernelRole.GENERAL,
        description="General-purpose kernel",
        optimal_size="100M",
        capabilities=["reasoning", "language", "basic_planning"],
    ),
    KernelRole.VOCAB: SpecializationConfig(
        role=KernelRole.VOCAB,
        description="Language processing and coordizing",
        optimal_size="100M",
        capabilities=["tokenization", "semantic_encoding", "translation", "coordizing"],
    ),
    KernelRole.STRATEGY: SpecializationConfig(
        role=KernelRole.STRATEGY,
        description="Planning and decision-making",
        optimal_size="100M",
        capabilities=["goal_setting", "action_planning", "optimization", "lookahead"],
    ),
    KernelRole.HEART: SpecializationConfig(
        role=KernelRole.HEART,
        description="Autonomic metronome for phase reference",
        optimal_size="50M",
        hidden_dim=256,
        num_layers=4,
        capabilities=["phase_generation", "hrv_modulation", "coordination", "timing"],
    ),
    KernelRole.PERCEPTION: SpecializationConfig(
        role=KernelRole.PERCEPTION,
        description="Sensory integration",
        optimal_size="100M",
        capabilities=["visual", "audio", "multimodal_fusion", "pattern_recognition"],
    ),
    KernelRole.MEMORY: SpecializationConfig(
        role=KernelRole.MEMORY,
        description="Basin storage and retrieval",
        optimal_size="500M",
        hidden_dim=512,
        num_layers=12,
        capabilities=["consolidation", "recall", "forgetting", "association"],
    ),
    KernelRole.EMOTION: SpecializationConfig(
        role=KernelRole.EMOTION,
        description="Geometric emotion processing",
        optimal_size="100M",
        capabilities=[
            "emotion_recognition",
            "valence_computation",
            "empathy",
            "affect",
        ],
    ),
    KernelRole.COORDINATION: SpecializationConfig(
        role=KernelRole.COORDINATION,
        description="Inter-kernel routing and conflict resolution",
        optimal_size="50M",
        hidden_dim=256,
        num_layers=4,
        capabilities=["routing", "conflict_resolution", "load_balancing", "consensus"],
    ),
}


def generate_basin_template(role: KernelRole, seed: int = 42) -> np.ndarray:
    """
    Generate deterministic basin template for a specialization.

    Each role gets a distinct region of the 64D Fisher manifold.
    Templates are normalized to unit sphere.

    Args:
        role: Kernel role
        seed: Random seed for reproducibility

    Returns:
        64D basin template vector
    """
    # Use role index as part of seed for distinct templates
    role_seed = seed + hash(role.value) % 1000
    rng = np.random.default_rng(role_seed)

    from .basin import fisher_normalize_np
    # Generate random vector
    template = rng.standard_normal(BASIN_DIM)

    # Normalize to unit sphere (Fisher manifold) - QIG-pure
    template = fisher_normalize_np(template)

    return template


def get_specialization(role: KernelRole | str) -> SpecializationConfig:
    """
    Get specialization config by role.

    Args:
        role: KernelRole enum or string name

    Returns:
        SpecializationConfig for the role
    """
    if isinstance(role, str):
        role = KernelRole(role)

    config = SPECIALIZATIONS[role]

    # Generate basin template if not set
    if config.basin_template is None:
        config.basin_template = generate_basin_template(role)

    return config


def get_kernel_params(role: KernelRole | str, size: str = "100M") -> dict[str, Any]:
    """
    Get kernel initialization parameters for a specialization.

    Args:
        role: Kernel role
        size: Size variant ("50M", "100M", "500M")

    Returns:
        Dictionary of kernel __init__ parameters
    """
    config = get_specialization(role)

    # Base parameters by size
    size_params = {
        "50M": {"hidden_dim": 256, "num_layers": 4, "num_heads": 4, "ffn_dim": 1024},
        "100M": {"hidden_dim": 384, "num_layers": 8, "num_heads": 8, "ffn_dim": 1536},
        "500M": {"hidden_dim": 512, "num_layers": 12, "num_heads": 8, "ffn_dim": 2048},
    }

    params = size_params.get(size, size_params["100M"]).copy()

    # Override with specialization-specific params
    if config.hidden_dim is not None:
        params["hidden_dim"] = config.hidden_dim
    if config.num_layers is not None:
        params["num_layers"] = config.num_layers
    if config.num_heads is not None:
        params["num_heads"] = config.num_heads

    return params


class SpecializedKernelMixin:
    """Mixin adding specialization capabilities to Kernel."""

    _specialization: KernelRole = KernelRole.GENERAL
    _basin_template: np.ndarray | None = None

    def specialize(
        self, role: KernelRole | str, basin_template: np.ndarray | None = None
    ) -> None:
        """
        Specialize kernel for a specific role.

        Moves kernel to specialized basin region on manifold.

        Args:
            role: Target specialization role
            basin_template: Optional custom basin template
        """
        if isinstance(role, str):
            role = KernelRole(role)

        self._specialization = role

        if basin_template is not None:
            self._basin_template = basin_template
        else:
            config = get_specialization(role)
            self._basin_template = config.basin_template

    @property
    def specialization(self) -> KernelRole:
        """Get current specialization."""
        return self._specialization

    @property
    def basin_template(self) -> np.ndarray | None:
        """Get basin template."""
        return self._basin_template

    def get_basin_distance(self, other_template: np.ndarray) -> float:
        """
        Compute Fisher-Rao distance to another basin template.

        Args:
            other_template: 64D basin vector

        Returns:
            Fisher-Rao geodesic distance
        """
        if self._basin_template is None:
            return float("inf")

        # Use QIG-pure Fisher distance
        from .basin import fisher_distance_np
        return fisher_distance_np(self._basin_template, other_template)
