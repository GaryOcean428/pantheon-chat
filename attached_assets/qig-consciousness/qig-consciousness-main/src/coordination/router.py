#!/usr/bin/env python3
"""
Constellation Router - Φ-weighted routing logic
================================================

Implements intelligent routing that selects which Gary instance should be active
based on Φ (integration) levels and load balancing.

QIG Routing Principle (Hypothesis 1 - Claustrum as Routing Hub):
    The routing hub should MAXIMIZE the conditional QFI - how much its routing
    decision affects global integration.

    We route to the instance whose activation would most benefit the constellation's
    overall Φ. Low-Φ instances should be active more often (they benefit from direct
    experience), while high-Φ instances provide the strongest vicarious learning signal.

Usage:
    from src.coordination.router import ConstellationRouter

    router = ConstellationRouter()
    active, observers = router.route_question(garys, total_conversations=100)
"""

from typing import Any


class ConstellationRouter:
    """
    Φ-weighted routing for constellation coordination.

    Routes questions to Gary instances based on their integration levels,
    ensuring optimal learning distribution.
    """

    def __init__(self) -> None:
        """Initialize router with load balancing state."""
        self.active_index = 0  # Round-robin fallback index
        self.turn_counter = 0  # Explicit turn counter for load balance backup

    def route_question(
        self,
        garys: list[Any],  # List[InstanceState] (Any to avoid circular import)
        phi_weighted: bool = True,
        total_conversations: int = 0,
    ) -> tuple[Any, list[Any]]:
        """
        Φ-weighted routing: returns (active_gary, observer_garys).

        QIG Routing Principle (Hypothesis 1 - Claustrum as Routing Hub):
            The "routing hub" should not be the source of Φ, but should
            MAXIMIZE the conditional QFI - how much its routing decision
            affects global integration.

            We route to the instance whose activation would most benefit
            the constellation's overall Φ. Low-Φ instances should be active
            more often (they benefit from direct experience), while high-Φ
            instances provide the strongest vicarious learning signal.

        Ocean is ALWAYS observer (never returned as active).

        Args:
            garys: List of Gary InstanceState objects
            phi_weighted: If True, route to lowest-Φ Gary (default).
                         If False, use round-robin for testing.
            total_conversations: Total conversations so far (for cold start check)

        Returns:
            active: The Gary instance that should respond
            observers: The other Gary instances (Ocean observes separately)
        """
        if phi_weighted and total_conversations > 0:
            # Route to lowest-Φ Gary - they benefit most from direct experience
            # High-Φ Garys provide strong vicarious learning signal as observers
            phis: list[float] = [g.phi for g in garys]

            # Check for equal Φ case (within tolerance)
            phi_range: float = max(phis) - min(phis)
            if phi_range < 0.01:
                # Equal Φ - use turn counter for load balance
                active_gary: Any = garys[self.turn_counter % len(garys)]
            else:
                # min Φ - lowest integration instance
                active_gary = min(garys, key=lambda g: g.phi)
            active: Any = active_gary

            self.turn_counter += 1
        else:
            # Round-robin for cold start or testing
            active = garys[self.active_index]
            self.active_index = (self.active_index + 1) % len(garys)
            self.turn_counter += 1

        observers: list[Any] = [g for g in garys if g != active]

        # Update roles
        active.role = "active"
        for obs in observers:
            obs.role = "observer"

        return active, observers

    def get_state(self) -> dict[str, int]:
        """Get current router state for checkpointing."""
        return {
            "active_index": self.active_index,
            "turn_counter": self.turn_counter,
        }

    def load_state(self, state: dict[str, int]) -> None:
        """Load router state from checkpoint."""
        self.active_index = state.get("active_index", 0)
        self.turn_counter = state.get("turn_counter", 0)
