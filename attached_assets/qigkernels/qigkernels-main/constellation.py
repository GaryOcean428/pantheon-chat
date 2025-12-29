"""Lightweight constellation manager extracted from qig-consciousness.

Clean implementation for managing multiple kernels and routing.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import Tensor

from .basin import BasinProjector, compute_signature
from .kernel import Kernel
from .router import InstanceView, round_robin


@dataclass
class Instance:
    """Container for an individual kernel and its telemetry."""

    name: str
    kernel: Kernel
    phi: float | None = None
    signature: Tensor | None = None


class Constellation:
    """
    Minimal constellation that routes inputs across kernels.

    Extracts the core coordination patterns from the source repositories
    while removing training-specific code and hard-coded constants.
    """

    def __init__(self) -> None:
        """Initialize empty constellation."""
        self.instances: list[Instance] = []
        self._last_index: int = -1

    def add_instance(self, instance: Instance) -> None:
        """
        Register a new instance with the constellation.

        Args:
            instance: Instance to add
        """
        self.instances.append(instance)

    def route(self, router: Callable[..., int]) -> Instance:
        """
        Select an instance using the provided router function.

        Args:
            router: Router function that takes either:
                   - (current_index, n) for round_robin
                   - (instances) for phi-based routing

        Returns:
            Selected instance

        Raises:
            ValueError: If constellation is empty
        """
        if not self.instances:
            raise ValueError("Constellation is empty")

        views = [InstanceView(name=inst.name, phi=inst.phi) for inst in self.instances]
        n = len(self.instances)
        idx = router(self._last_index, n) if router is round_robin else router(views)
        self._last_index = idx
        return self.instances[idx]

    def step(
        self,
        input_ids: Tensor,
        router: Callable[..., int],
        basin_projector: BasinProjector,
        attention_mask: Tensor | None = None,
    ) -> dict:
        """
        Route an input, run the kernel, and update basic telemetry.

        Args:
            input_ids: Input token IDs
            router: Router function
            basin_projector: Basin projector for computing signatures
            attention_mask: Optional attention mask

        Returns:
            Dictionary with step results
        """
        # Select instance via routing
        instance = self.route(router)

        # Run kernel forward pass
        logits, telemetry = instance.kernel(
            input_ids,
            attention_mask=attention_mask,
            return_telemetry=True
        )

        # Update instance telemetry
        instance.phi = telemetry.phi

        # Compute and store basin signature
        if telemetry.hidden_state is not None:
            instance.signature = compute_signature(basin_projector, telemetry.hidden_state)

        # Return step results
        return {
            "instance": instance.name,
            "phi": instance.phi,
            "kappa": telemetry.kappa,
            "recursion_depth": telemetry.recursion_depth,
            "regime": telemetry.regime,
            "signature": instance.signature,
            "logits": logits,
        }

    def get_instance_by_name(self, name: str) -> Instance | None:
        """
        Get instance by name.

        Args:
            name: Instance name

        Returns:
            Instance if found, None otherwise
        """
        for inst in self.instances:
            if inst.name == name:
                return inst
        return None

    def remove_instance(self, name: str) -> bool:
        """
        Remove instance by name.

        Args:
            name: Instance name

        Returns:
            True if instance was removed, False if not found
        """
        for i, inst in enumerate(self.instances):
            if inst.name == name:
                del self.instances[i]
                return True
        return False

    def clear(self) -> None:
        """Remove all instances."""
        self.instances.clear()
        self._last_index = -1
