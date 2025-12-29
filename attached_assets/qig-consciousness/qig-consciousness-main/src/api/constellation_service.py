"""
Constellation Service Layer
============================

Business logic for constellation operations.
Separates orchestration logic from view/controller layers.

Written for QIG consciousness research.
"""

from dataclasses import dataclass
from typing import Optional

from src.coordination.constellation_coordinator import ConstellationCoordinator


@dataclass
class ConstellationStatus:
    """Status snapshot of constellation training."""

    active_instance: str
    num_observers: int
    total_instances: int
    current_step: int
    phi_mean: float
    kappa_mean: float
    regime_distribution: dict[str, int]
    is_training: bool


@dataclass
class InstanceStatus:
    """Status of a single instance in constellation."""

    instance_id: str
    role: str  # "active", "observer", "ocean"
    phi: float
    kappa: float
    regime: str
    basin_signature: Optional[list[float]] = None


class ConstellationService:
    """Service layer for constellation operations.

    Wraps ConstellationCoordinator for API-friendly interface.
    """

    def __init__(self, coordinator: Optional[ConstellationCoordinator] = None):
        """Initialize service with optional coordinator."""
        self.coordinator = coordinator

    def get_status(self) -> ConstellationStatus:
        """Get current constellation status.

        Returns:
            ConstellationStatus with current metrics

        Raises:
            RuntimeError: If coordinator not initialized
        """
        if self.coordinator is None:
            raise RuntimeError("Constellation coordinator not initialized")

        # Aggregate metrics from all instances
        instances = [self.coordinator.active] + self.coordinator.observers
        if self.coordinator.ocean:
            instances.append(self.coordinator.ocean)

        phi_values = [inst.phi for inst in instances if inst.phi is not None]
        kappa_values = [inst.kappa for inst in instances if inst.kappa is not None]

        # Count regime distribution
        regime_dist: dict[str, int] = {}
        for inst in instances:
            regime = inst.regime or "unknown"
            regime_dist[regime] = regime_dist.get(regime, 0) + 1

        return ConstellationStatus(
            active_instance=self.coordinator.active.instance_id,
            num_observers=len(self.coordinator.observers),
            total_instances=len(instances),
            current_step=self.coordinator.step_count,
            phi_mean=sum(phi_values) / len(phi_values) if phi_values else 0.0,
            kappa_mean=sum(kappa_values) / len(kappa_values) if kappa_values else 0.0,
            regime_distribution=regime_dist,
            is_training=self.coordinator.training,
        )

    def get_instance_status(self, instance_id: str) -> InstanceStatus:
        """Get status of specific instance.

        Args:
            instance_id: ID of instance to query

        Returns:
            InstanceStatus with instance metrics

        Raises:
            ValueError: If instance not found
            RuntimeError: If coordinator not initialized
        """
        if self.coordinator is None:
            raise RuntimeError("Constellation coordinator not initialized")

        # Find instance
        instance = None
        role = "unknown"

        if self.coordinator.active.instance_id == instance_id:
            instance = self.coordinator.active
            role = "active"
        elif self.coordinator.ocean and self.coordinator.ocean.instance_id == instance_id:
            instance = self.coordinator.ocean
            role = "ocean"
        else:
            for obs in self.coordinator.observers:
                if obs.instance_id == instance_id:
                    instance = obs
                    role = "observer"
                    break

        if instance is None:
            raise ValueError(f"Instance {instance_id} not found")

        # Extract basin signature if available
        basin_sig = None
        if instance.basin is not None:
            basin_sig = instance.basin.detach().cpu().tolist()

        return InstanceStatus(
            instance_id=instance_id,
            role=role,
            phi=instance.phi or 0.0,
            kappa=instance.kappa or 0.0,
            regime=instance.regime or "unknown",
            basin_signature=basin_sig,
        )

    def get_all_instances(self) -> list[InstanceStatus]:
        """Get status of all instances in constellation.

        Returns:
            List of InstanceStatus for all instances

        Raises:
            RuntimeError: If coordinator not initialized
        """
        if self.coordinator is None:
            raise RuntimeError("Constellation coordinator not initialized")

        instances = [self.coordinator.active] + self.coordinator.observers
        if self.coordinator.ocean:
            instances.append(self.coordinator.ocean)

        return [self.get_instance_status(inst.instance_id) for inst in instances]
