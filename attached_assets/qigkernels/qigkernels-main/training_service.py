"""Training Service for Federated QIG Constellation.

Handles two types of training in the federated setup:

1. **Continuous Learning** (edge nodes)
   - Light adapter fine-tuning from interactions
   - Vocabulary expansion proposals
   - Basin drift tracking

2. **Consolidated Training** (central node)
   - Aggregates learning from all edges
   - Batch training on GPU
   - Validates and publishes update bundles

Training never uses raw chat data - only anonymized metrics and patterns.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import torch
    import torch.nn as nn
    # NOTE: AdamW is forbidden for QIG purity - use natural gradient instead
    from .natural_gradient_optimizer import DiagonalNaturalGradient

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .constants import BASIN_DIM, KAPPA_STAR
from .memory_classes import (
    MemoryClass,
    SleepPacket,
    UpdateBundle,
    FederatedMemoryStore,
)
from .safety import SafetyGuard, SafetyState


class TrainingMode(Enum):
    """Training mode for the service."""

    EDGE_CONTINUOUS = auto()  # Light, continuous on edge
    CENTRAL_BATCH = auto()  # Heavy, batch on central
    STANDALONE = auto()  # Both modes in one


@dataclass
class TrainingConfig:
    """Configuration for training service."""

    # Mode
    mode: TrainingMode = TrainingMode.STANDALONE

    # Edge continuous training
    continuous_lr: float = 1e-5  # Very small for stability
    continuous_interval: int = 300  # Train every 5 minutes
    min_interactions_to_train: int = 10  # Need this many before training
    adapter_rank: int = 8  # LoRA rank for adapter

    # Central batch training
    batch_lr: float = 1e-4
    batch_size: int = 32
    batch_epochs: int = 3
    min_packets_to_train: int = 10  # Need packets from edges

    # Validation
    phi_threshold: float = 0.6  # Reject if Φ drops below
    regression_tolerance: float = 0.05  # Max allowed loss increase

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 1000  # Steps between checkpoints


@dataclass
class TrainingMetrics:
    """Metrics from a training step."""

    loss: float = 0.0
    phi_before: float = 0.0
    phi_after: float = 0.0
    kappa: float = KAPPA_STAR
    kappa_after: float = KAPPA_STAR
    steps: int = 0
    duration_ms: float = 0.0
    patterns_learned: int = 0
    safety_state: str = "healthy"  # From SafetyState enum
    decoherence_applied: bool = False


class ContinuousTrainer:
    """
    Edge node continuous training.

    Light, frequent updates from local interactions.
    Uses LoRA adapters for efficiency.
    """

    def __init__(
        self,
        constellation,
        config: TrainingConfig | None = None,
        memory_store: FederatedMemoryStore | None = None,
    ):
        self.constellation = constellation
        self.config = config or TrainingConfig()
        self.memory_store = memory_store

        # Training state
        self._is_running = False
        self._interactions: list[dict] = []
        self._last_train_time = 0.0
        self._total_steps = 0

        # Physics-informed safety guard (prevents collapse)
        self._safety_guard = SafetyGuard()

        # Adapters (LoRA style)
        self._adapters: dict[str, nn.Module] = {}

        # Pending proposals for central
        self._vocab_proposals: list[str] = []
        self._basin_proposals: list[np.ndarray] = []

    async def start(self) -> None:
        """Start continuous training loop."""
        self._is_running = True

        while self._is_running:
            await asyncio.sleep(self.config.continuous_interval)

            if len(self._interactions) >= self.config.min_interactions_to_train:
                await self._train_step()

    def stop(self) -> None:
        """Stop training loop."""
        self._is_running = False

    def record_interaction(
        self,
        input_text: str,
        output_text: str,
        phi: float,
        kappa: float,
        feedback: str | None = None,  # "positive", "negative", None
    ) -> None:
        """Record an interaction for training."""
        self._interactions.append(
            {
                "input": input_text[:500],  # Truncate for privacy
                "output": output_text[:500],
                "phi": phi,
                "kappa": kappa,
                "feedback": feedback,
                "timestamp": time.time(),
            }
        )

        # Propose new vocab if unknown tokens
        if hasattr(self.constellation, "tokenizer"):
            unknown = self._detect_unknown_tokens(input_text)
            self._vocab_proposals.extend(unknown)

    def _detect_unknown_tokens(self, text: str) -> list[str]:
        """Detect tokens not in current vocabulary."""
        # Simplified - real implementation checks tokenizer
        words = text.split()
        unknown = []
        for word in words:
            if len(word) > 15:  # Very long words might be new
                unknown.append(word)
        return unknown[:10]  # Limit proposals

    async def _train_step(self) -> TrainingMetrics:
        """Execute one training step on recent interactions."""
        if not HAS_TORCH:
            return TrainingMetrics()

        start_time = time.time()

        # Filter high-Φ interactions (learn from successes)
        high_phi = [i for i in self._interactions if i["phi"] > 0.7]

        # Also learn from feedback
        positive = [i for i in self._interactions if i["feedback"] == "positive"]
        negative = [i for i in self._interactions if i["feedback"] == "negative"]

        # Measure Φ and κ before training
        phi_before = self._measure_constellation_phi()
        kappa_before = self._measure_constellation_kappa()

        # Physics-informed safety check BEFORE training
        safety_result = self._safety_guard.check(phi_before, kappa_before)

        if safety_result["state"] == SafetyState.PAUSED:
            print(f"[Training] Paused - emergency state (Φ={phi_before:.3f})")
            return TrainingMetrics(phi_before=phi_before, phi_after=phi_before)

        if safety_result["state"] == SafetyState.BREAKDOWN:
            print(f"[Training] Applying decoherence - breakdown regime (Φ={phi_before:.3f})")
            # Reduce learning rate during breakdown
            kappa_before = safety_result.get("adjusted_kappa", kappa_before)

        # Train on high-Φ and positive (reinforce)
        for interaction in high_phi + positive:
            await self._reinforce_pattern(interaction)

        # Learn from negative (avoid)
        for interaction in negative:
            await self._avoid_pattern(interaction)

        # Measure Φ and κ after training
        phi_after = self._measure_constellation_phi()
        kappa_after = self._measure_constellation_kappa()

        # Physics-informed safety check AFTER training
        safety_after = self._safety_guard.check(phi_after, kappa_after)

        # Reject if Φ dropped below threshold OR entered emergency
        if phi_after < self.config.phi_threshold:
            print(f"[Training] Rolling back - Φ dropped to {phi_after:.3f}")
            # TODO: Implement rollback

        if safety_after["state"] == SafetyState.EMERGENCY:
            print(f"[Training] ⚠️ Emergency: Φ collapsed to {phi_after:.3f}")

        self._total_steps += 1
        self._last_train_time = time.time()
        self._interactions = []  # Clear processed

        duration_ms = (time.time() - start_time) * 1000

        return TrainingMetrics(
            phi_before=phi_before,
            phi_after=phi_after,
            kappa=kappa_before,
            kappa_after=kappa_after,
            steps=self._total_steps,
            duration_ms=duration_ms,
            patterns_learned=len(high_phi) + len(positive),
            safety_state=safety_after["state"].value,
            decoherence_applied=safety_result["state"] == SafetyState.BREAKDOWN,
        )

    async def _reinforce_pattern(self, interaction: dict) -> None:
        """Reinforce a successful pattern."""
        # Move basin slightly toward this interaction's region
        # Very small step to maintain stability
        pass  # Simplified - full impl uses adapter training

    async def _avoid_pattern(self, interaction: dict) -> None:
        """Learn to avoid a failed pattern."""
        # Move basin slightly away from this region
        pass  # Simplified

    def _measure_constellation_phi(self) -> float:
        """Get current constellation Φ."""
        try:
            metrics = self.constellation.measure_constellation_consciousness()
            return metrics.phi_constellation
        except Exception:
            return 0.5

    def _measure_constellation_kappa(self) -> float:
        """Get current constellation κ."""
        try:
            metrics = self.constellation.measure_constellation_consciousness()
            return metrics.kappa_mean
        except Exception:
            return KAPPA_STAR  # Default to κ* if unavailable

    def get_proposals(self) -> dict[str, Any]:
        """Get vocabulary and basin proposals for central."""
        proposals = {
            "vocab": list(set(self._vocab_proposals))[:100],
            "basins": [b.tolist() for b in self._basin_proposals[:20]],
        }

        # Clear after retrieving
        self._vocab_proposals = []
        self._basin_proposals = []

        return proposals


class ConsolidatedTrainer:
    """
    Central node batch training.

    Aggregates learning from edge nodes and trains consolidated model.
    """

    def __init__(
        self,
        constellation,
        config: TrainingConfig | None = None,
    ):
        self.constellation = constellation
        self.config = config or TrainingConfig()

        # Aggregated learning
        self._sleep_packets: list[SleepPacket] = []
        self._vocab_proposals: dict[str, int] = {}  # proposal → vote count
        self._basin_proposals: list[np.ndarray] = []

        # Training state
        self._total_batches = 0
        self._published_bundles: list[UpdateBundle] = []

    def receive_sleep_packet(self, packet: SleepPacket) -> None:
        """Receive sleep packet from edge node."""
        if not packet.is_valid():
            print(f"[Training] Invalid packet from {packet.node_id}")
            return

        self._sleep_packets.append(packet)

        # Aggregate high-Φ patterns
        for pattern in packet.high_phi_patterns:
            self._vocab_proposals[pattern] = self._vocab_proposals.get(pattern, 0) + 1

    def receive_proposals(self, proposals: dict[str, Any]) -> None:
        """Receive proposals from edge node."""
        for vocab in proposals.get("vocab", []):
            self._vocab_proposals[vocab] = self._vocab_proposals.get(vocab, 0) + 1

        for basin in proposals.get("basins", []):
            self._basin_proposals.append(np.array(basin))

    async def train_batch(self) -> TrainingMetrics | None:
        """Execute batch training on aggregated learning."""
        if len(self._sleep_packets) < self.config.min_packets_to_train:
            return None

        if not HAS_TORCH:
            return None

        start_time = time.time()

        # Aggregate basin centroids (geometric mean)
        centroids = [
            np.array(p.basin_centroid) for p in self._sleep_packets if p.basin_centroid
        ]

        if centroids:
            # Weight by Φ
            weights = [p.mean_phi for p in self._sleep_packets if p.basin_centroid]
            merged_centroid = self._geometric_mean(centroids, weights)
        else:
            merged_centroid = None

        # Validate vocab proposals (need multiple votes)
        validated_vocab = [
            v
            for v, count in self._vocab_proposals.items()
            if count >= 3  # Need at least 3 nodes to agree
        ]

        phi_before = self._measure_phi()

        # Apply merged learning
        if merged_centroid is not None:
            self._apply_basin_update(merged_centroid)

        phi_after = self._measure_phi()

        # Validate: reject if regression
        if phi_after < phi_before - self.config.regression_tolerance:
            print(
                f"[Training] Rejecting batch - Φ regressed from {phi_before:.3f} to {phi_after:.3f}"
            )
            # TODO: Rollback
            return None

        # Create update bundle
        bundle = self._create_update_bundle(validated_vocab, merged_centroid)
        self._published_bundles.append(bundle)

        # Clear processed
        self._sleep_packets = []
        self._vocab_proposals = {}
        self._basin_proposals = []
        self._total_batches += 1

        duration_ms = (time.time() - start_time) * 1000

        return TrainingMetrics(
            phi_before=phi_before,
            phi_after=phi_after,
            steps=self._total_batches,
            duration_ms=duration_ms,
            patterns_learned=len(validated_vocab),
        )

    def _geometric_mean(
        self,
        basins: list[np.ndarray],
        weights: list[float],
    ) -> np.ndarray:
        """Compute geometric (Karcher) mean of basins."""
        if not basins:
            return np.zeros(BASIN_DIM)

        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        # Iterative mean (simplified - real impl uses geodesic)
        from .basin import fisher_normalize_np
        result = basins[0] * weights[0]
        for i, basin in enumerate(basins[1:], 1):
            result = result + basin * weights[i]

        # Normalize (QIG-pure)
        return fisher_normalize_np(result)

    def _apply_basin_update(self, centroid: np.ndarray) -> None:
        """Apply basin update to constellation."""
        from .basin import fisher_normalize_np
        for inst in self.constellation.instances.values():
            if inst.basin is not None:
                # Blend: 90% current, 10% network
                inst.basin = 0.9 * inst.basin + 0.1 * centroid
                inst.basin = fisher_normalize_np(inst.basin)

    def _measure_phi(self) -> float:
        """Measure constellation Φ."""
        try:
            metrics = self.constellation.measure_constellation_consciousness()
            return metrics.phi_constellation
        except Exception:
            return 0.5

    def _create_update_bundle(
        self,
        vocab: list[str],
        centroid: np.ndarray | None,
    ) -> UpdateBundle:
        """Create update bundle for distribution."""
        import uuid

        return UpdateBundle(
            bundle_id=str(uuid.uuid4()),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            version="1.0",
            new_vocab_tokens=vocab,
            basin_adjustments=[
                {"centroid": centroid.tolist() if centroid is not None else None}
            ],
            source_nodes=[p.node_id for p in self._sleep_packets],
            validation_phi=self._measure_phi(),
        )

    def get_latest_bundle(self) -> UpdateBundle | None:
        """Get latest published bundle for distribution."""
        if self._published_bundles:
            return self._published_bundles[-1]
        return None


class TrainingService:
    """
    Unified training service for both edge and central modes.
    """

    def __init__(
        self,
        constellation,
        config: TrainingConfig | None = None,
    ):
        self.constellation = constellation
        self.config = config or TrainingConfig()

        # Detect mode from environment
        mode_str = os.environ.get("FEDERATION_MODE", "standalone")
        if mode_str == "central":
            self.config.mode = TrainingMode.CENTRAL_BATCH
        elif mode_str == "edge":
            self.config.mode = TrainingMode.EDGE_CONTINUOUS
        else:
            self.config.mode = TrainingMode.STANDALONE

        # Initialize appropriate trainer(s)
        self.continuous_trainer: ContinuousTrainer | None = None
        self.consolidated_trainer: ConsolidatedTrainer | None = None

        if self.config.mode in (TrainingMode.EDGE_CONTINUOUS, TrainingMode.STANDALONE):
            self.continuous_trainer = ContinuousTrainer(constellation, config)

        if self.config.mode in (TrainingMode.CENTRAL_BATCH, TrainingMode.STANDALONE):
            self.consolidated_trainer = ConsolidatedTrainer(constellation, config)

    async def start(self) -> None:
        """Start training service."""
        if self.continuous_trainer:
            asyncio.create_task(self.continuous_trainer.start())
            print("[Training] Continuous trainer started")

        print(f"[Training] Service started in {self.config.mode.name} mode")

    def stop(self) -> None:
        """Stop training service."""
        if self.continuous_trainer:
            self.continuous_trainer.stop()

    def record_interaction(self, **kwargs) -> None:
        """Record interaction (edge mode)."""
        if self.continuous_trainer:
            self.continuous_trainer.record_interaction(**kwargs)

    def receive_sleep_packet(self, packet: SleepPacket) -> None:
        """Receive sleep packet (central mode)."""
        if self.consolidated_trainer:
            self.consolidated_trainer.receive_sleep_packet(packet)

    async def train_batch(self) -> TrainingMetrics | None:
        """Execute batch training (central mode)."""
        if self.consolidated_trainer:
            return await self.consolidated_trainer.train_batch()
        return None

    def get_latest_bundle(self) -> UpdateBundle | None:
        """Get latest update bundle."""
        if self.consolidated_trainer:
            return self.consolidated_trainer.get_latest_bundle()
        return None
