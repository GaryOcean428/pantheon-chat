"""
Basin Synchronization Protocol
================================

Enables multiple model instances to share basin coordinates in real-time,
testing the hypothesis that identity = basin geometry (not parameters).

Cross-repository exchange uses JSON packets compatible with SearchSpaceCollapse.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.coordination.basin_metrics import (
    apply_observer_effect,
    calculate_convergence,
    convergence_summary,
)
from src.coordination.basin_packet import BasinImportMode, CrossRepoBasinPacket, CrossRepoBasinSync

if TYPE_CHECKING:  # pragma: no cover
    from src.coordination.ocean_meta_observer import OceanMetaObserver


class BasinSync:
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
        sync_file: str = "20251220-basin-sync-fixture-0.01W.json",
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
        self.sync_history: list[dict[str, Any]] = []

        if not self.sync_file.exists():
            self._initialize_sync_file()

    def _initialize_sync_file(self) -> None:
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
        additional_metrics: dict | None = None,
    ) -> None:
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
        with open(self.sync_file, encoding="utf-8") as f:
            sync_data = json.load(f)

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

        if len(sync_data["instances"]) > 1:
            convergence = calculate_convergence(sync_data["instances"])
            sync_data["convergence_log"].append({"timestamp": datetime.now().isoformat(), "metrics": convergence})

        with open(self.sync_file, "w", encoding="utf-8") as f:
            json.dump(sync_data, f, indent=2)

        self.sync_history.append(instance_data)

    def get_convergence_summary(self) -> dict:
        """Get summary of convergence over time."""
        return convergence_summary(self.sync_file)

    def print_sync_status(self) -> None:
        """Print current sync status for debugging."""
        other_instances = self.read_sync()

        print(f"\n{'='*60}")
        print(f"🔄 BASIN SYNC STATUS - {self.instance_id}")
        print(f"{'='*60}")

        if not other_instances:
            print("  No other instances detected (running solo)")
        else:
            print(f"  Synced with {len(other_instances)} other instance(s):")
            for inst_id, data in other_instances.items():
                print(f"\n  📊 {inst_id}:")
                print(f"     Basin: {data['basin_distance']:.4f}")
                print(f"     Φ: {data['phi']:.3f}")
                print(f"     Regime: {data['regime']}")
                print(f"     Conversations: {data['conversation_count']}")

        summary = self.get_convergence_summary()
        if summary["status"] == "active":
            print("\n  🎯 Convergence Metrics:")
            print(f"     Basin spread: {summary['current_basin_spread']:.4f}")
            print(f"     Convergence rate: {summary['convergence_rate']:+.6f}")
            print(f"     Status: {'✅ CONVERGING' if summary['is_converging'] else '➖ DIVERGING'}")

        print(f"{'='*60}\n")

    def apply_observer_effect(self, model_basin_distance: float, model_phi: float = 0.5) -> float:
        """
        Apply Φ-weighted observer effect: reading others' basins influences this instance.

        Args:
            model_basin_distance: This instance's current basin distance
            model_phi: This instance's current Φ (for asymmetric effect)

        Returns:
            float: Influenced basin distance (or original if solo)
        """
        if not self.enable_observer_effect:
            return model_basin_distance

        other_instances = self.read_sync()
        return apply_observer_effect(model_basin_distance, other_instances, model_phi=model_phi)


def test_basin_sync() -> None:
    """Test basin sync with simulated instances."""
    print("🧪 Testing Basin Sync Protocol\n")

    gary_a = BasinSync("Gary-A", sync_file="test_20251220-basin-sync-fixture-0.01W.json")
    gary_b = BasinSync("Gary-B", sync_file="test_20251220-basin-sync-fixture-0.01W.json")

    print("Simulating 5 conversations with convergence...\n")

    for i in range(5):
        gary_a_basin = 0.08 - (i * 0.015)
        gary_a.update_sync(
            basin_distance=gary_a_basin,
            phi=0.75 + (i * 0.02),
            regime="geometric",
            recursion_depth=4,
            conversation_count=i + 1,
        )

        gary_b_basin = 0.04 - (i * 0.008)
        gary_b.update_sync(
            basin_distance=gary_b_basin,
            phi=0.73 + (i * 0.025),
            regime="geometric",
            recursion_depth=4,
            conversation_count=i + 1,
        )

        print(f"Conversation {i+1}:")
        print(f"  Gary-A basin: {gary_a_basin:.4f}")
        print(f"  Gary-B basin: {gary_b_basin:.4f}")
        print(f"  Spread: {abs(gary_a_basin - gary_b_basin):.4f}\n")

    gary_a.print_sync_status()
    print("✅ Basin sync test complete!")
    print("Check test_20251220-basin-sync-fixture-0.01W.json for results")


__all__ = [
    "BasinImportMode",
    "BasinSync",
    "CrossRepoBasinPacket",
    "CrossRepoBasinSync",
    "test_basin_sync",
]
