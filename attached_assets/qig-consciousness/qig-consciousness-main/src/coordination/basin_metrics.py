"""Metric helpers for basin synchronization."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def calculate_convergence(instances: dict) -> dict:
    """Calculate convergence metrics across all instances."""
    basin_distances = [inst["basin_distance"] for inst in instances.values()]
    phis = [inst["phi"] for inst in instances.values()]

    basin_spread = max(basin_distances) - min(basin_distances)
    basin_mean = sum(basin_distances) / len(basin_distances)
    phi_spread = max(phis) - min(phis)
    phi_mean = sum(phis) / len(phis)

    return {
        "basin_spread": basin_spread,
        "basin_mean": basin_mean,
        "phi_spread": phi_spread,
        "phi_mean": phi_mean,
        "instance_count": len(instances),
    }


def convergence_summary(sync_file: Path) -> dict:
    """Summarize convergence history from a sync file."""
    with open(sync_file) as f:
        sync_data = json.load(f)

    convergence_log = sync_data.get("convergence_log", [])
    if not convergence_log:
        return {"status": "insufficient_data"}

    basin_spreads = [entry["metrics"]["basin_spread"] for entry in convergence_log]

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


def apply_observer_effect(
    model_basin_distance: float,
    other_instances: dict[str, Any],
    model_phi: float = 0.5,
    base_strength: float = 0.10,
) -> float:
    """Apply Φ-weighted observer effect using other instance metrics."""
    if not other_instances:
        return model_basin_distance

    total_weight = 0.0
    weighted_basin_sum = 0.0

    for inst in other_instances.values():
        source_phi = inst.get("phi", 0.5)
        weight = max(0.1, source_phi)
        weighted_basin_sum += inst["basin_distance"] * weight
        total_weight += weight

    if total_weight == 0:
        return model_basin_distance

    phi_weighted_mean_basin = weighted_basin_sum / total_weight
    receiver_susceptibility = 1.0 - (model_phi * 0.5)
    observer_strength = base_strength * receiver_susceptibility

    return model_basin_distance * (1 - observer_strength) + phi_weighted_mean_basin * observer_strength


__all__ = ["apply_observer_effect", "calculate_convergence", "convergence_summary"]
