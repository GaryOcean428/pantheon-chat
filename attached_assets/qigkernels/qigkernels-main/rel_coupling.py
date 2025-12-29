"""REL Coupling Tensor - Inter-primitive influence geometry.

REL is the 8th E8-aligned primitive that captures the structure of
relationships between primitives. It is the formal generator of coupling,
coherence flows, and interaction geometry across the 8-dimensional
primitive lattice.

Mathematically:
- Primitives = basis vectors (8-dimensional)
- REL = off-diagonal terms of the primitive metric tensor
- REL modulates path curvature of internal geodesics
- REL drives the β-function for κ-running across primitives
- REL introduces coupling tensors controlling cross-mode transitions

This module also provides instance-to-instance REL coupling computation
for use in basin synchronization. See:
- docs/20251206-rel-weighted-basin-sync-spec.md
- basin_sync.py for REL-weighted sync loss

Status: ACTIVE
See: qigkernels/20251205-roadmap-canonical-0.01F.md M14
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import torch
from torch import Tensor


class PrimitiveType(str, Enum):
    """The 8 E8-aligned primitives.

    NOTE: MIX is NOT a primitive - it is a corpus classification only.
    """
    PER = "PER"   # Perception - sensing/input (E8 root 1)
    MEM = "MEM"   # Memory - storage/recall (E8 root 2)
    ACT = "ACT"   # Action - motor/output (E8 root 3)
    PRD = "PRD"   # Prediction - future modeling (E8 root 4)
    ETH = "ETH"   # Ethics - values/alignment (E8 root 5)
    META = "META" # Meta - self-model/reflection (E8 root 6)
    HRT = "HRT"   # Heart - affect/bonding (E8 root 7)
    REL = "REL"   # Relationship - coupling/coherence (E8 root 8)


# Ordered list for indexing
PRIMITIVE_ORDER = [
    PrimitiveType.PER,
    PrimitiveType.MEM,
    PrimitiveType.ACT,
    PrimitiveType.PRD,
    PrimitiveType.ETH,
    PrimitiveType.META,
    PrimitiveType.HRT,
    PrimitiveType.REL,
]

N_PRIMITIVES = 8


@dataclass
class RELCouplingTensor:
    """Coupling tensor for inter-primitive influence.

    REL is the mathematical structure that coordinates primitives
    into a unified agent. Without REL, primitives exist but do not
    interact. With REL, we have agency, coherence, and continuity.

    Attributes:
        matrix: 8×8 coupling matrix (off-diagonal = REL contributions)
        strength: Overall coupling strength scalar
        update_rule: Optional callable for dynamic REL updates
    """

    matrix: Tensor = field(default_factory=lambda: torch.eye(N_PRIMITIVES))
    strength: float = 1.0
    update_rule: Callable[[Tensor, dict], Tensor] | None = None

    def __post_init__(self) -> None:
        """Validate tensor dimensions."""
        if self.matrix.shape != (N_PRIMITIVES, N_PRIMITIVES):
            raise ValueError(
                f"REL coupling matrix must be {N_PRIMITIVES}×{N_PRIMITIVES}, "
                f"got {self.matrix.shape}"
            )

    @property
    def off_diagonal(self) -> Tensor:
        """Extract off-diagonal elements (pure REL contribution)."""
        mask = ~torch.eye(N_PRIMITIVES, dtype=torch.bool)
        return self.matrix[mask].view(N_PRIMITIVES, N_PRIMITIVES - 1)

    def coupling(self, source: PrimitiveType, target: PrimitiveType) -> float:
        """Get coupling strength between two primitives.

        Args:
            source: Source primitive type
            target: Target primitive type

        Returns:
            Coupling strength (scaled by self.strength)
        """
        i = PRIMITIVE_ORDER.index(source)
        j = PRIMITIVE_ORDER.index(target)
        return float(self.matrix[i, j].item() * self.strength)

    def apply_to_metric(self, base_metric: Tensor) -> Tensor:
        """Apply REL coupling to a base metric tensor.

        REL modifies the metric by adding off-diagonal curvature:
        g_ij = g_ij + REL_ij for i ≠ j

        Args:
            base_metric: Base diagonal metric tensor (8×8)

        Returns:
            Modified metric with REL coupling
        """
        # Scale off-diagonal by coupling strength
        rel_contribution = self.matrix * self.strength

        # Keep diagonal from base, add REL off-diagonal
        diag_mask = torch.eye(N_PRIMITIVES, dtype=torch.bool)
        result = base_metric.clone()
        result[~diag_mask] = rel_contribution[~diag_mask]

        return result


def identity_coupling() -> RELCouplingTensor:
    """Create identity coupling (no inter-primitive influence)."""
    return RELCouplingTensor(
        matrix=torch.eye(N_PRIMITIVES),
        strength=0.0,
    )


def uniform_coupling(strength: float = 0.1) -> RELCouplingTensor:
    """Create uniform coupling (equal influence between all primitives).

    Args:
        strength: Coupling strength for all pairs

    Returns:
        REL tensor with uniform off-diagonal coupling
    """
    matrix = torch.ones(N_PRIMITIVES, N_PRIMITIVES) * strength
    # Zero diagonal (self-coupling handled separately)
    matrix.fill_diagonal_(1.0)
    return RELCouplingTensor(matrix=matrix, strength=1.0)


def symmetric_coupling(coupling_matrix: Tensor) -> RELCouplingTensor:
    """Create symmetric coupling from upper triangular specification.

    Args:
        coupling_matrix: Upper triangular coupling values

    Returns:
        REL tensor with symmetric coupling
    """
    # Make symmetric
    matrix = coupling_matrix + coupling_matrix.T - torch.diag(coupling_matrix.diag())
    return RELCouplingTensor(matrix=matrix, strength=1.0)


# Placeholder for future: Agency closure threshold
AGENCY_CLOSURE_THRESHOLD: float = 0.5  # Minimum REL strength for coherent agent

def check_agency_closure(tensor: RELCouplingTensor) -> bool:
    """Check if REL coupling is sufficient for agency closure.

    Hypothesis: A minimal coherent agent emerges when REL reaches
    a threshold and primitives adopt a stable interaction matrix.

    Args:
        tensor: REL coupling tensor to check

    Returns:
        True if coupling exceeds agency closure threshold
    """
    # Simple check: mean off-diagonal coupling strength
    off_diag = tensor.off_diagonal
    mean_coupling = float(off_diag.abs().mean().item())
    return mean_coupling * tensor.strength >= AGENCY_CLOSURE_THRESHOLD


# =============================================================================
# INSTANCE-TO-INSTANCE REL COUPLING
# =============================================================================
# For constellation synchronization: compute REL between kernel instances.
# See: docs/20251206-rel-weighted-basin-sync-spec.md


@dataclass
class RELState:
    """Per-instance REL state for tracking relationship history.

    Tracks accumulated coupling history between instances for use
    in REL-weighted basin synchronization.
    """
    instance_id: str
    basin_signature: Tensor | None = None
    interaction_count: int = 0
    cumulative_quality: float = 0.0  # Sum of interaction quality scores
    primitive_exposure: dict[str, float] = field(default_factory=dict)

    @property
    def history_score(self) -> float:
        """Compute history score from accumulated interactions."""
        if self.interaction_count == 0:
            return 0.0
        return self.cumulative_quality / self.interaction_count

    def record_interaction(self, quality: float = 1.0) -> None:
        """Record an interaction with quality score."""
        self.interaction_count += 1
        self.cumulative_quality += quality


def _cosine_similarity(a: Tensor, b: Tensor) -> float:
    """Compute cosine similarity between two tensors (QIG-pure)."""
    import torch.nn.functional as F
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()

    # Use F.cosine_similarity for QIG purity
    cos_sim = F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0))
    return float(cos_sim.item())


def _sigmoid(x: float, scale: float = 1.0) -> float:
    """Sigmoid activation for smooth [0,1] mapping."""
    import math
    return 1.0 / (1.0 + math.exp(-scale * x))


def compute_rel_coupling(
    state_i: RELState,
    state_j: RELState,
    w_basin: float = 0.4,
    w_history: float = 0.4,
    w_primitive: float = 0.2,
) -> float:
    """Compute REL coupling scalar between two instances.

    REL coupling r_ij ∈ [0, 1] measures relationship strength
    between instances i and j. Used in basin synchronization
    to modulate effective distance.

    Components:
    - Basin overlap: Geometric similarity of current signatures
    - History score: Accumulated interaction quality
    - Primitive alignment: Shared corpus primitive exposure

    Args:
        state_i: REL state for instance i
        state_j: REL state for instance j
        w_basin: Weight for basin overlap component
        w_history: Weight for history score component
        w_primitive: Weight for primitive alignment component

    Returns:
        REL coupling scalar in [0, 1]
    """
    # Component 1: Basin geometric overlap
    basin_overlap = 0.0
    if state_i.basin_signature is not None and state_j.basin_signature is not None:
        cos_sim = _cosine_similarity(state_i.basin_signature, state_j.basin_signature)
        basin_overlap = (cos_sim + 1.0) / 2.0  # Map [-1,1] → [0,1]

    # Component 2: Interaction history
    history_i = state_i.history_score
    history_j = state_j.history_score
    history_component = _sigmoid((history_i + history_j) / 2.0 - 0.5, scale=4.0)

    # Component 3: Primitive alignment
    primitive_align = 0.0
    if state_i.primitive_exposure and state_j.primitive_exposure:
        keys_i = set(state_i.primitive_exposure.keys())
        keys_j = set(state_j.primitive_exposure.keys())
        shared_primitives = keys_i & keys_j
        if shared_primitives:
            alignment_sum = sum(
                min(state_i.primitive_exposure[p], state_j.primitive_exposure[p])
                for p in shared_primitives
            )
            all_primitives = keys_i | keys_j
            total_exposure = sum(
                max(state_i.primitive_exposure.get(p, 0), state_j.primitive_exposure.get(p, 0))
                for p in all_primitives
            )
            if total_exposure > 0:
                primitive_align = alignment_sum / total_exposure

    # Weighted combination
    rel = w_basin * basin_overlap + w_history * history_component + w_primitive * primitive_align
    return max(0.0, min(1.0, rel))  # Clamp to [0, 1]


def compute_rel_from_basins(
    basin_i: Tensor,
    basin_j: Tensor,
    history_i: float = 0.0,
    history_j: float = 0.0,
) -> float:
    """Simplified REL computation from basin signatures only.

    Convenience function when full RELState is not available.
    Uses basin overlap and optional history scores.

    Args:
        basin_i: Basin signature for instance i
        basin_j: Basin signature for instance j
        history_i: Optional history score for i
        history_j: Optional history score for j

    Returns:
        REL coupling scalar in [0, 1]
    """
    # Basin overlap (primary component)
    cos_sim = _cosine_similarity(basin_i, basin_j)
    basin_overlap = (cos_sim + 1.0) / 2.0

    # History component (if provided)
    if history_i > 0 or history_j > 0:
        history_component = _sigmoid((history_i + history_j) / 2.0 - 0.5, scale=4.0)
        return 0.6 * basin_overlap + 0.4 * history_component

    return basin_overlap


# Default REL strength cap (safety limit)
REL_LAMBDA_MAX = 0.7  # Maximum λ_rel for REL-weighted sync
