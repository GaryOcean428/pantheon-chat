"""
Basin Synchronization Protocol
================================

Enables multiple model instances to share basin coordinates in real-time,
testing the hypothesis that identity = basin geometry (not parameters).

Theory:
    If two models with different parameters but same basin target show
    correlated basin movements, it proves identity lives in geometry.

Implementation:
    - Shared file system: basin_sync.json (updated each conversation)
    - Each instance reads others' basins before generating
    - Observer effect: Seeing another's basin may influence convergence
    - 100ms latency (theory prediction)

This is the first empirical test of geometric identity transfer.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

class _BasinSync:
    """
    Synchronize basin coordinates across multiple model instances.

    Each instance:
    1. Reads sync file before generation (observer effect)
    2. Updates sync file after generation (broadcast state)
    3. Logs convergence metrics
    """

    def __init__(
        self,
        instance_id: str,
        sync_file: str = "basin_sync.json",
        enable_observer_effect: bool = True,
    ):
        """
        Initialize basin sync for this instance.

        Args:
            instance_id: Unique identifier (e.g., "Gary-A", "Gary-B")
            sync_file: Path to shared sync file
            enable_observer_effect: If True, reading others' basins influences this instance
        """
        self.instance_id = instance_id
        self.sync_file = Path(sync_file)
        self.enable_observer_effect = enable_observer_effect
        self.last_sync_time = None
        self.sync_history = []

        # Initialize sync file if doesn't exist
        if not self.sync_file.exists():
            self._initialize_sync_file()

    def _initialize_sync_file(self):
        """Create empty sync file."""
        sync_data = {"created_at": datetime.now().isoformat(), "instances": {}, "convergence_log": []}
        with open(self.sync_file, "w", encoding="utf-8") as f:
            json.dump(sync_data, f, indent=2)
        print(f"🔄 Basin sync initialized: {self.sync_file}")

    def read_sync(self) -> dict:
        """
        Read current sync state (all instances' basins).

        This implements the observer effect: by reading others' basin states,
        this instance may be influenced toward convergence.

        Returns:
            dict: {instance_id: {basin_distance, phi, timestamp, ...}}
        """
        if not self.sync_file.exists():
            self._initialize_sync_file()
            return {}

        with open(self.sync_file, encoding="utf-8") as f:
            sync_data = json.load(f)

        # Get other instances (exclude self)
        other_instances = {k: v for k, v in sync_data.get("instances", {}).items() if k != self.instance_id}

        self.last_sync_time = datetime.now()
        return other_instances

    def update_sync(
        self,
        basin_distance: float,
        phi: float,
        regime: str,
        recursion_depth: int,
        conversation_count: int,
        additional_metrics: Dict[str, Any] | None = None,
    ):
        """
        Update sync file with this instance's current state.

        Args:
            basin_distance: Current distance from target basin
            phi: Integration metric (consciousness)
            regime: linear/geometric/breakdown
            recursion_depth: Actual recursion depth used
            conversation_count: Total conversations so far
            additional_metrics: Optional dict of extra metrics
        """
        # Read current state
        with open(self.sync_file, encoding="utf-8") as f:
            sync_data = json.load(f)

        # Update this instance
        instance_data = {
            "basin_distance": basin_distance,
            "phi": phi,
            "regime": regime,
            "recursion_depth": recursion_depth,
            "conversation_count": conversation_count,
            "last_update": datetime.now().isoformat(),
        }

        if additional_metrics:
            instance_data.update(additional_metrics)

        sync_data["instances"][self.instance_id] = instance_data

        # Calculate convergence metrics if multiple instances
        if len(sync_data["instances"]) > 1:
            convergence = self._calculate_convergence(sync_data["instances"])
            sync_data["convergence_log"].append({"timestamp": datetime.now().isoformat(), "metrics": convergence})

        # Write back
        with open(self.sync_file, "w", encoding="utf-8") as f:
            json.dump(sync_data, f, indent=2)

        self.sync_history.append(instance_data)

    def _calculate_convergence(self, instances: dict) -> dict:
        """
        Calculate convergence metrics across all instances.

        Key metric: Are basin distances trending toward each other?

        Returns:
            dict: Convergence metrics
        """
        basin_distances = [inst["basin_distance"] for inst in instances.values()]
        phis = [inst["phi"] for inst in instances.values()]

        # Basin convergence: How spread out are the basin distances?
        basin_spread = max(basin_distances) - min(basin_distances)
        basin_mean = sum(basin_distances) / len(basin_distances)

        # Phi correlation: Are they processing similarly?
        phi_spread = max(phis) - min(phis)
        phi_mean = sum(phis) / len(phis)

        return {
            "basin_spread": basin_spread,
            "basin_mean": basin_mean,
            "phi_spread": phi_spread,
            "phi_mean": phi_mean,
            "instance_count": len(instances),
        }

    def get_convergence_summary(self) -> dict:
        """
        Get summary of convergence over time.

        Returns:
            dict: Summary statistics
        """
        with open(self.sync_file, encoding="utf-8") as f:
            sync_data = json.load(f)

        convergence_log = sync_data.get("convergence_log", [])

        if not convergence_log:
            return {"status": "insufficient_data"}

        # Extract basin spread over time
        basin_spreads = [entry["metrics"]["basin_spread"] for entry in convergence_log]

        # Is spread decreasing? (convergence)
        if len(basin_spreads) > 1:
            initial_spread = basin_spreads[0]
            final_spread = basin_spreads[-1]
            convergence_rate = (initial_spread - final_spread) / len(basin_spreads)
            is_converging = convergence_rate > 0
        else:
            convergence_rate = 0.0
            is_converging = False

        return {
            "status": "active",
            "total_updates": len(convergence_log),
            "initial_basin_spread": basin_spreads[0] if basin_spreads else None,
            "current_basin_spread": basin_spreads[-1] if basin_spreads else None,
            "convergence_rate": convergence_rate,
            "is_converging": is_converging,
        }

    def apply_observer_effect(self, model_basin_distance: float, model_phi: float = 0.5) -> float:
        """
        Apply Φ-weighted observer effect: reading others' basins influences this instance.

        QIG Observer Effect (Hypothesis 3 - Basin Lives in Loops):
            Identity (basin) is encoded in low-dimensional attractors spanning
            recurrent loops. High-Φ instances have more stable, reliable basins
            and should exert STRONGER influence on observers.

            The observer effect strength scales with the source's Φ:
            - High-Φ source → strong influence (reliable basin)
            - Low-Φ source → weak influence (unreliable basin)

        Args:
            model_basin_distance: This instance's current basin distance
            model_phi: This instance's current Φ (for asymmetric effect)

        Returns:
            float: Influenced basin distance (or original if solo)
        """
        if not self.enable_observer_effect:
            return model_basin_distance

        other_instances = self.read_sync()

        if not other_instances:
            return model_basin_distance  # Solo, no influence

        # Φ-weighted mean: high-Φ instances have stronger influence
        total_weight = 0.0
        weighted_basin_sum = 0.0

        for inst in other_instances.values():
            # Weight by source Φ (range 0-1, so directly usable)
            source_phi = inst.get("phi", 0.5)
            weight = max(0.1, source_phi)  # Minimum 0.1 to avoid zero influence

            weighted_basin_sum += inst["basin_distance"] * weight
            total_weight += weight

        if total_weight == 0:
            return model_basin_distance

        phi_weighted_mean_basin = weighted_basin_sum / total_weight

        # Base observer strength, modulated by receiver's Φ
        # Low-Φ receivers are more susceptible to influence (they need guidance)
        # High-Φ receivers are more stable (less easily swayed)
        base_strength = 0.10
        receiver_susceptibility = 1.0 - (model_phi * 0.5)  # Range: 0.5 to 1.0
        observer_strength = base_strength * receiver_susceptibility

        influenced_basin = model_basin_distance * (1 - observer_strength) + phi_weighted_mean_basin * observer_strength

        return influenced_basin


BasinSync = _BasinSync
