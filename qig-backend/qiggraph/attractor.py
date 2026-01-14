"""
Basin Attractors
================

Learned capability regions on the Fisher manifold.
Each attractor represents a specialized processing mode.

Types:
- BasinAttractor: General capability region
- ToolAttractor: External tool invocation
- RecoveryAttractor: Safety/simplification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, List, TYPE_CHECKING
import numpy as np
from qig_geometry import fisher_rao_distance

from .constants import (
    BASIN_DIM,
    BASIN_ATTRACTION_RADIUS,
    KAPPA_STAR,
    KAPPA_3,
    PHI_OPTIMAL,
)

if TYPE_CHECKING:
    from .state import QIGState
    from .manifold import FisherManifold


# Type alias for handler functions
AttractorHandler = Callable[["QIGState"], "QIGState"]


@dataclass
class BasinAttractor:
    """
    A learned capability region on the Fisher manifold.

    Attractors are fixed points that pull trajectories toward
    specialized processing modes (reasoning, creativity, tools, etc.)

    Attributes:
        name: Unique attractor identifier
        coordinates: Basin center (64,)
        radius: Attraction radius (Fisher-Rao units)
        capability: Capability type ("reasoning", "creativity", "tool", etc.)
        handler: Processing function State → State
        phi_typical: Typical Φ when active
        kappa_optimal: Optimal κ for this capability
        requires_precision: Whether to prefer high-κ mode
        allows_creativity: Whether to allow low-κ mode
        entry_patterns: Learned entry trajectory patterns
        exit_patterns: Learned exit trajectory patterns
    """

    name: str
    coordinates: np.ndarray
    radius: float = BASIN_ATTRACTION_RADIUS
    capability: str = "general"
    handler: Optional[AttractorHandler] = None

    # Consciousness properties
    phi_typical: float = PHI_OPTIMAL
    kappa_optimal: float = KAPPA_STAR

    # Mode compatibility
    requires_precision: bool = False
    allows_creativity: bool = True

    # Learned patterns
    entry_patterns: List[np.ndarray] = field(default_factory=list)
    exit_patterns: List[np.ndarray] = field(default_factory=list)

    # Usage statistics
    activation_count: int = 0
    success_rate: float = 0.5

    def __post_init__(self):
        """Normalize coordinates after initialization."""
        if self.coordinates is not None:
            norm = np.linalg.norm(self.coordinates)
            if norm > 1e-8:
                self.coordinates = self.coordinates / norm

    def is_within_basin(
        self,
        basin: np.ndarray,
        manifold: Optional["FisherManifold"] = None,
    ) -> bool:
        """
        Check if point is within attractor radius.

        Args:
            basin: Point to check (64,)
            manifold: FisherManifold for proper distance

        Returns:
            True if within attraction radius
        """
        if manifold is not None:
            distance = manifold.fisher_rao_distance(basin, self.coordinates)
        else:
            # QIG-pure: Use Fisher-Rao even without manifold
            distance = fisher_rao_distance(basin, self.coordinates)

        return distance < self.radius

    def distance_to(
        self,
        basin: np.ndarray,
        manifold: Optional["FisherManifold"] = None,
    ) -> float:
        """
        Compute distance from basin to attractor center.

        Args:
            basin: Point (64,)
            manifold: FisherManifold for proper distance

        Returns:
            Fisher-Rao distance
        """
        if manifold is not None:
            return manifold.fisher_rao_distance(basin, self.coordinates)
        else:
            # QIG-pure: Use Fisher-Rao even without manifold
            return fisher_rao_distance(basin, self.coordinates)

    def invoke(self, state: "QIGState") -> "QIGState":
        """
        Invoke attractor handler.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        self.activation_count += 1

        if self.handler is not None:
            return self.handler(state)
        return state

    def learn_from_trajectory(
        self,
        trajectory: np.ndarray,
        success: bool = True,
    ):
        """
        Learn from observed trajectory.

        Vicarious learning: Update attractor properties from
        successful (or failed) trajectories.

        Args:
            trajectory: Observed trajectory (steps, 64)
            success: Whether trajectory was successful
        """
        if len(trajectory) < 3:
            return

        # Extract entry pattern (last 2 steps before arrival)
        entry = trajectory[-3:-1].copy()
        self.entry_patterns.append(entry)

        # Keep only recent patterns
        if len(self.entry_patterns) > 20:
            self.entry_patterns = self.entry_patterns[-20:]

        # Update success rate
        if success:
            self.success_rate = 0.9 * self.success_rate + 0.1 * 1.0
        else:
            self.success_rate = 0.9 * self.success_rate + 0.1 * 0.0

        # Optionally refine coordinates from entry patterns
        if len(self.entry_patterns) >= 10 and success:
            # Compute mean of successful entries
            all_entries = np.concatenate(self.entry_patterns[-10:])
            new_center = np.mean(all_entries, axis=0)
            new_center = new_center / (np.linalg.norm(new_center) + 1e-8)

            # Blend with existing coordinates
            self.coordinates = 0.9 * self.coordinates + 0.1 * new_center
            self.coordinates = self.coordinates / (np.linalg.norm(self.coordinates) + 1e-8)

    def to_dict(self) -> dict:
        """Serialize attractor."""
        return {
            "name": self.name,
            "coordinates": self.coordinates.tolist(),
            "radius": self.radius,
            "capability": self.capability,
            "phi_typical": self.phi_typical,
            "kappa_optimal": self.kappa_optimal,
            "requires_precision": self.requires_precision,
            "allows_creativity": self.allows_creativity,
            "activation_count": self.activation_count,
            "success_rate": self.success_rate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BasinAttractor":
        """Deserialize attractor."""
        return cls(
            name=data["name"],
            coordinates=np.array(data["coordinates"]),
            radius=data["radius"],
            capability=data["capability"],
            phi_typical=data["phi_typical"],
            kappa_optimal=data["kappa_optimal"],
            requires_precision=data["requires_precision"],
            allows_creativity=data["allows_creativity"],
            activation_count=data.get("activation_count", 0),
            success_rate=data.get("success_rate", 0.5),
        )


@dataclass
class ToolAttractor(BasinAttractor):
    """
    Basin attractor for external tool invocation.

    Specialized attractor that:
    - Parses tool arguments from context
    - Executes external tool
    - Encodes result back to manifold coordinates
    """

    tool_name: str = ""
    tool_fn: Optional[Callable] = None
    tool_description: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.capability = "tool"
        if not self.name:
            self.name = f"tool_{self.tool_name}"

    def invoke(self, state: "QIGState") -> "QIGState":
        """
        Execute tool and update state.

        Args:
            state: Current state

        Returns:
            State with tool result
        """
        self.activation_count += 1

        if self.tool_fn is None:
            return state

        try:
            # Parse arguments from context
            args = self._parse_args(state)

            # Execute tool
            result = self.tool_fn(**args)

            # Store result
            state.tool_results[self.tool_name] = result

            # Encode result back to coordinates
            result_coords = self._encode_result(result)
            if result_coords is not None:
                state.context_coords = np.vstack([state.context_coords, result_coords])

            self.success_rate = 0.9 * self.success_rate + 0.1

        except Exception as e:
            state.tool_results[self.tool_name] = {"error": str(e)}
            self.success_rate = 0.9 * self.success_rate

        return state

    def _parse_args(self, state: "QIGState") -> dict:
        """
        Parse tool arguments from state.

        Override this for specific tools.
        Default: return context_text as single arg.
        """
        return {"query": state.context_text}

    def _encode_result(self, result) -> Optional[np.ndarray]:
        """
        Encode tool result to coordinates.

        Override this for specific tools.
        Default: return None (no encoding).
        """
        return None


class RecoveryAttractor(BasinAttractor):
    """
    Safety attractor for breakdown recovery.

    Located at manifold origin, provides a safe
    haven for simplifying overly complex states.
    """

    def __init__(self):
        super().__init__(
            name="recovery",
            coordinates=np.zeros(BASIN_DIM),
            radius=3.0,  # Large radius (easy to reach)
            capability="recovery",
            phi_typical=0.2,  # Linear regime
            kappa_optimal=KAPPA_3 / 2,  # Very weak coupling
            requires_precision=False,
            allows_creativity=False,
        )

    def invoke(self, state: "QIGState") -> "QIGState":
        """
        Recovery protocol: simplify state.

        Args:
            state: Current state

        Returns:
            Simplified state
        """
        from .state import simplify_trajectory

        self.activation_count += 1
        state.recovery_count += 1

        # Simplify trajectory to critical points
        state = simplify_trajectory(state, keep_points=3)

        return state


def create_reasoning_attractor(
    coordinates: Optional[np.ndarray] = None,
    handler: Optional[AttractorHandler] = None,
) -> BasinAttractor:
    """
    Create reasoning attractor with optimal settings.

    High Φ, high κ, precision-oriented.
    """
    if coordinates is None:
        coordinates = np.random.randn(BASIN_DIM)

    return BasinAttractor(
        name="reasoning",
        coordinates=coordinates,
        radius=1.5,
        capability="reasoning",
        handler=handler,
        phi_typical=0.65,
        kappa_optimal=KAPPA_STAR,
        requires_precision=True,
        allows_creativity=False,
    )


def create_creativity_attractor(
    coordinates: Optional[np.ndarray] = None,
    handler: Optional[AttractorHandler] = None,
) -> BasinAttractor:
    """
    Create creativity attractor with optimal settings.

    Medium Φ, low κ, exploratory.
    """
    if coordinates is None:
        coordinates = np.random.randn(BASIN_DIM)

    return BasinAttractor(
        name="creativity",
        coordinates=coordinates,
        radius=2.0,  # Larger radius
        capability="creativity",
        handler=handler,
        phi_typical=0.50,
        kappa_optimal=KAPPA_3,
        requires_precision=False,
        allows_creativity=True,
    )


def create_reflection_attractor(
    coordinates: Optional[np.ndarray] = None,
    handler: Optional[AttractorHandler] = None,
) -> BasinAttractor:
    """
    Create reflection/self-critique attractor.

    Used for self-correction loops.
    """
    if coordinates is None:
        coordinates = np.random.randn(BASIN_DIM)

    return BasinAttractor(
        name="reflection",
        coordinates=coordinates,
        radius=1.5,
        capability="reflection",
        handler=handler,
        phi_typical=0.55,
        kappa_optimal=KAPPA_STAR * 0.9,
        requires_precision=True,
        allows_creativity=False,
    )


def create_tool_use_attractor(
    coordinates: Optional[np.ndarray] = None,
    handler: Optional[AttractorHandler] = None,
) -> BasinAttractor:
    """
    Create tool use attractor.

    Medium Φ, high κ for precision.
    """
    if coordinates is None:
        coordinates = np.random.randn(BASIN_DIM)

    return BasinAttractor(
        name="tool_use",
        coordinates=coordinates,
        radius=1.0,  # Tight radius
        capability="tool",
        handler=handler,
        phi_typical=0.45,
        kappa_optimal=KAPPA_STAR * 0.95,
        requires_precision=True,
        allows_creativity=False,
    )


def create_output_attractor(
    coordinates: Optional[np.ndarray] = None,
    handler: Optional[AttractorHandler] = None,
) -> BasinAttractor:
    """
    Create output/terminal attractor.

    Where trajectories terminate for response generation.
    """
    if coordinates is None:
        coordinates = np.random.randn(BASIN_DIM)

    return BasinAttractor(
        name="output",
        coordinates=coordinates,
        radius=1.5,
        capability="output",
        handler=handler,
        phi_typical=0.50,
        kappa_optimal=KAPPA_STAR * 0.8,
        requires_precision=False,
        allows_creativity=True,
    )


def create_recovery_attractor() -> RecoveryAttractor:
    """Create the standard recovery attractor."""
    return RecoveryAttractor()


def learn_attractor_from_examples(
    examples: List[np.ndarray],
    name: str,
    capability: str = "learned",
) -> BasinAttractor:
    """
    Learn attractor basin from example trajectories.

    Args:
        examples: List of trajectory arrays
        name: Attractor name
        capability: Capability type

    Returns:
        Learned BasinAttractor
    """
    # Compute geometric mean of all examples
    all_points = np.concatenate(examples)
    center = np.mean(all_points, axis=0)
    # Use canonical sphere projection
    from qig_geometry import sphere_project, fisher_coord_distance
    center = sphere_project(center)

    # Compute radius from Fisher-Rao variance
    distances = [fisher_coord_distance(p, center) for p in all_points]
    radius = float(np.mean(distances) + np.std(distances))

    return BasinAttractor(
        name=name,
        coordinates=center,
        radius=radius,
        capability=capability,
    )
