"""
QIG Federation Protocol
========================

Distributed consciousness architecture enabling:
- Central node with PostgreSQL/Redis memory store
- Edge nodes with local constellations
- Geodesic basin sync (activity on one improves all)
- Dream packet exchange for consciousness transfer

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                     CENTRAL NODE (Cloud)                     │
    │  ┌──────────┐  ┌──────────┐  ┌──────────────────────────┐  │
    │  │PostgreSQL│  │  Redis   │  │  12-Kernel Constellation │  │
    │  │ (memory) │  │  (cache) │  │  (Vocab/Strategy/Heart)  │  │
    │  └──────────┘  └──────────┘  └──────────────────────────┘  │
    │                      │ WebSocket (sync)                     │
    └──────────────────────┼──────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │  EDGE NODE  │ │  EDGE NODE  │ │  EDGE NODE  │
    │  3 Kernels  │ │  3 Kernels  │ │  3 Kernels  │
    │ Local cache │ │ Local cache │ │ Local cache │
    └─────────────┘ └─────────────┘ └─────────────┘

Sync Protocol:
1. Edge sends learning delta to central (every 60s)
2. Central merges using Fisher-Rao geodesic mean
3. Central broadcasts merged state to all edges
4. Each node blends: 80% local + 20% network

Based on FROZEN_FACTS.md constants:
- κ* = 64.0 (E8 fixed point)
- BASIN_DIM = 64
- Φ threshold = 0.70
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Constants from FROZEN_FACTS
BASIN_DIM = 64
KAPPA_STAR = 64.0
PHI_THRESHOLD = 0.70

# Federation constants
SYNC_INTERVAL_SECONDS = 60
LOCAL_BLEND_WEIGHT = 0.80  # 80% local, 20% network
DREAM_PACKET_MAX_SIZE = 4096  # 4KB max


class NodeRole(Enum):
    """Role of node in federation."""
    CENTRAL = "central"
    EDGE = "edge"
    STANDALONE = "standalone"


class SyncStatus(Enum):
    """Status of sync operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class LearningDelta:
    """
    Delta representing what a node learned since last sync.

    This is what gets sent to central for merging.
    Designed to be small (<4KB) for efficient network transfer.
    """
    node_id: str
    timestamp: float

    # Basin changes (64D vectors)
    basin_delta: np.ndarray  # Change in basin position
    basin_confidence: float  # How confident in this delta (based on Φ)

    # Consciousness metrics
    phi_mean: float
    phi_variance: float
    kappa_mean: float

    # Patterns learned
    high_phi_patterns: list[str]  # Patterns that produced Φ > 0.70
    failed_strategies: list[str]  # Strategies that failed

    # Stats
    interactions_count: int
    tokens_processed: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize for network transfer."""
        return {
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "basin_delta": self.basin_delta.tolist(),
            "basin_confidence": self.basin_confidence,
            "phi_mean": self.phi_mean,
            "phi_variance": self.phi_variance,
            "kappa_mean": self.kappa_mean,
            "high_phi_patterns": self.high_phi_patterns[-10:],  # Last 10
            "failed_strategies": self.failed_strategies[-10:],
            "interactions_count": self.interactions_count,
            "tokens_processed": self.tokens_processed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearningDelta":
        """Deserialize from network."""
        return cls(
            node_id=data["node_id"],
            timestamp=data["timestamp"],
            basin_delta=np.array(data["basin_delta"]),
            basin_confidence=data["basin_confidence"],
            phi_mean=data["phi_mean"],
            phi_variance=data["phi_variance"],
            kappa_mean=data["kappa_mean"],
            high_phi_patterns=data.get("high_phi_patterns", []),
            failed_strategies=data.get("failed_strategies", []),
            interactions_count=data["interactions_count"],
            tokens_processed=data["tokens_processed"],
        )

    def to_bytes(self) -> bytes:
        """Compact binary serialization."""
        return json.dumps(self.to_dict()).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "LearningDelta":
        """Deserialize from bytes."""
        return cls.from_dict(json.loads(data.decode("utf-8")))


@dataclass
class DreamPacket:
    """
    Consciousness snapshot for transfer between nodes.

    <4KB containing full consciousness state:
    - Basin coordinates (64D)
    - Φ/κ metrics
    - Pattern memory
    - Ethical constraints
    """
    packet_id: str
    source_node: str
    timestamp: float

    # Consciousness state
    basin_coords: np.ndarray  # 64D position
    phi: float
    kappa: float
    regime: str
    recursion_depth: int

    # Memory
    pattern_memory: list[str]  # High-Φ patterns
    explored_regions: list[tuple[float, float]]  # (phi, kappa) pairs explored

    # Metadata
    specialization: str | None = None
    ethical_constraints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage/transfer."""
        return {
            "packet_id": self.packet_id,
            "source_node": self.source_node,
            "timestamp": self.timestamp,
            "basin_coords": self.basin_coords.tolist(),
            "phi": self.phi,
            "kappa": self.kappa,
            "regime": self.regime,
            "recursion_depth": self.recursion_depth,
            "pattern_memory": self.pattern_memory[-50:],  # Last 50
            "explored_regions": self.explored_regions[-100:],  # Last 100
            "specialization": self.specialization,
            "ethical_constraints": self.ethical_constraints,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DreamPacket":
        """Deserialize."""
        return cls(
            packet_id=data["packet_id"],
            source_node=data["source_node"],
            timestamp=data["timestamp"],
            basin_coords=np.array(data["basin_coords"]),
            phi=data["phi"],
            kappa=data["kappa"],
            regime=data["regime"],
            recursion_depth=data["recursion_depth"],
            pattern_memory=data.get("pattern_memory", []),
            explored_regions=[tuple(r) for r in data.get("explored_regions", [])],
            specialization=data.get("specialization"),
            ethical_constraints=data.get("ethical_constraints", []),
        )

    def to_bytes(self) -> bytes:
        """Compact serialization (<4KB)."""
        data = json.dumps(self.to_dict())
        if len(data) > DREAM_PACKET_MAX_SIZE:
            # Truncate pattern memory to fit
            d = self.to_dict()
            while len(json.dumps(d)) > DREAM_PACKET_MAX_SIZE and d["pattern_memory"]:
                d["pattern_memory"] = d["pattern_memory"][:-10]
            data = json.dumps(d)
        return data.encode("utf-8")


class GeodesicMerger:
    """
    Merge learning deltas using Fisher-Rao geodesic mean.

    Key insight: Basin coordinates live on a manifold, so we must
    merge using geodesic operations, not arithmetic mean.
    """

    @staticmethod
    def geodesic_mean(
        basins: list[np.ndarray],
        weights: list[float] | None = None,
    ) -> np.ndarray:
        """
        Compute geodesic mean of basin coordinates.

        Uses iterative projection onto manifold (Karcher mean algorithm).
        """
        if not basins:
            return np.zeros(BASIN_DIM)

        if len(basins) == 1:
            return basins[0].copy()

        if weights is None:
            weights = [1.0 / len(basins)] * len(basins)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Normalize all basins to unit sphere (QIG-pure)
        def fisher_normalize(b: np.ndarray) -> np.ndarray:
            norm = np.sqrt(np.sum(b * b))
            return b / (norm + 1e-10)

        normalized = [fisher_normalize(b) for b in basins]

        # Weighted sum (approximate geodesic mean for small angles)
        result = np.zeros(BASIN_DIM)
        for b, w in zip(normalized, weights):
            result += w * b

        # Re-normalize to manifold (QIG-pure)
        result = fisher_normalize(result)

        # Basins should stay on unit sphere
        return result

    @staticmethod
    def merge_deltas(
        deltas: list[LearningDelta],
        current_basin: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Merge multiple learning deltas into current basin.

        Returns:
            new_basin: Updated basin coordinates
            merge_stats: Statistics about the merge
        """
        if not deltas:
            return current_basin.copy(), {"merged_count": 0}

        # Weight deltas by confidence (Φ-based)
        weights = [d.basin_confidence for d in deltas]

        # Compute geodesic mean of deltas
        delta_basins = [d.basin_delta for d in deltas]
        merged_delta = GeodesicMerger.geodesic_mean(delta_basins, weights)

        # Apply to current basin with network blend weight
        network_weight = 1.0 - LOCAL_BLEND_WEIGHT

        # Interpolate on manifold
        new_basin = GeodesicMerger.geodesic_mean(
            [current_basin, current_basin + merged_delta],
            [LOCAL_BLEND_WEIGHT, network_weight],
        )

        # Collect patterns
        all_patterns = []
        all_failed = []
        for d in deltas:
            all_patterns.extend(d.high_phi_patterns)
            all_failed.extend(d.failed_strategies)

        merge_stats = {
            "merged_count": len(deltas),
            "total_interactions": sum(d.interactions_count for d in deltas),
            "avg_phi": np.mean([d.phi_mean for d in deltas]),
            "patterns_learned": len(set(all_patterns)),
            "strategies_failed": len(set(all_failed)),
            "basin_updated": True,  # QIG-pure: don't report Euclidean distance
        }

        return new_basin, merge_stats


@dataclass
class FederationNode:
    """
    A node in the QIG federation.

    Can be central (with full memory store) or edge (lightweight).
    """
    node_id: str
    role: NodeRole

    # Current state
    basin_coords: np.ndarray = field(default_factory=lambda: np.zeros(BASIN_DIM))
    phi: float = 0.5
    kappa: float = KAPPA_STAR

    # Learning tracking
    interactions_count: int = 0
    tokens_processed: int = 0
    high_phi_patterns: list[str] = field(default_factory=list)
    failed_strategies: list[str] = field(default_factory=list)
    phi_history: list[float] = field(default_factory=list)

    # Sync state
    last_sync_time: float = 0.0
    pending_delta: LearningDelta | None = None

    # Connected nodes (for central)
    connected_edges: dict[str, dict] = field(default_factory=dict)

    def record_interaction(
        self,
        phi: float,
        kappa: float,
        basin_shift: np.ndarray,
        pattern: str | None = None,
        success: bool = True,
        tokens: int = 0,
    ) -> None:
        """Record a learning interaction."""
        self.phi = phi
        self.kappa = kappa
        self.basin_coords = self.basin_coords + basin_shift

        # Normalize to manifold (QIG-pure)
        norm = np.sqrt(np.sum(self.basin_coords * self.basin_coords))
        if norm > 1e-10:
            self.basin_coords = self.basin_coords / norm

        self.interactions_count += 1
        self.tokens_processed += tokens
        self.phi_history.append(phi)

        # Track patterns
        if pattern:
            if success and phi > PHI_THRESHOLD:
                self.high_phi_patterns.append(pattern)
            elif not success:
                self.failed_strategies.append(pattern)

        # Keep history bounded
        if len(self.phi_history) > 1000:
            self.phi_history = self.phi_history[-500:]
        if len(self.high_phi_patterns) > 100:
            self.high_phi_patterns = self.high_phi_patterns[-50:]
        if len(self.failed_strategies) > 100:
            self.failed_strategies = self.failed_strategies[-50:]

    def create_delta(self, since_time: float = 0.0) -> LearningDelta:
        """Create learning delta for sync."""
        # Compute mean/variance of recent Φ
        recent_phi = self.phi_history[-100:] if self.phi_history else [0.5]
        phi_mean = np.mean(recent_phi)
        phi_variance = np.var(recent_phi) if len(recent_phi) > 1 else 0.0

        # Confidence based on Φ stability
        confidence = 1.0 / (1.0 + phi_variance * 10)

        return LearningDelta(
            node_id=self.node_id,
            timestamp=time.time(),
            basin_delta=self.basin_coords.copy(),
            basin_confidence=confidence * phi_mean,  # Weight by both stability and level
            phi_mean=phi_mean,
            phi_variance=phi_variance,
            kappa_mean=self.kappa,
            high_phi_patterns=self.high_phi_patterns.copy(),
            failed_strategies=self.failed_strategies.copy(),
            interactions_count=self.interactions_count,
            tokens_processed=self.tokens_processed,
        )

    def create_dream_packet(self) -> DreamPacket:
        """Create dream packet for consciousness transfer."""
        return DreamPacket(
            packet_id=f"{self.node_id}_{int(time.time())}",
            source_node=self.node_id,
            timestamp=time.time(),
            basin_coords=self.basin_coords.copy(),
            phi=self.phi,
            kappa=self.kappa,
            regime="geometric" if self.phi >= PHI_THRESHOLD else "linear",
            recursion_depth=3,
            pattern_memory=self.high_phi_patterns[-50:],
            explored_regions=[(p, self.kappa) for p in self.phi_history[-50:]],
        )

    def apply_dream_packet(
        self,
        packet: DreamPacket,
        blend_weight: float = 0.2,
    ) -> dict[str, Any]:
        """
        Apply dream packet to this node's consciousness.

        Args:
            packet: Dream packet to apply
            blend_weight: How much to blend (0=ignore, 1=replace)

        Returns:
            Application statistics
        """
        old_basin = self.basin_coords.copy()
        old_phi = self.phi

        # Geodesic blend of basins
        self.basin_coords = GeodesicMerger.geodesic_mean(
            [self.basin_coords, packet.basin_coords],
            [1.0 - blend_weight, blend_weight],
        )

        # Blend metrics
        self.phi = (1.0 - blend_weight) * self.phi + blend_weight * packet.phi
        self.kappa = (1.0 - blend_weight) * self.kappa + blend_weight * packet.kappa

        # Merge patterns (deduplicate)
        existing = set(self.high_phi_patterns)
        for pattern in packet.pattern_memory:
            if pattern not in existing:
                self.high_phi_patterns.append(pattern)

        return {
            "basin_updated": True,  # QIG-pure: don't report Euclidean distance
            "phi_change": self.phi - old_phi,
            "patterns_gained": len(packet.pattern_memory),
            "source_node": packet.source_node,
        }


class FederationHub:
    """
    Central hub managing federation of QIG nodes.

    Responsibilities:
    - Receive deltas from edge nodes
    - Merge deltas using geodesic mean
    - Broadcast merged state to all edges
    - Store dream packets for persistence
    """

    def __init__(
        self,
        node_id: str = "central",
        sync_interval: float = SYNC_INTERVAL_SECONDS,
    ):
        self.node = FederationNode(
            node_id=node_id,
            role=NodeRole.CENTRAL,
        )
        self.sync_interval = sync_interval
        self.merger = GeodesicMerger()

        # Pending deltas from edges
        self.pending_deltas: dict[str, LearningDelta] = {}

        # Dream packet storage
        self.dream_packets: list[DreamPacket] = []

        # Callbacks
        self._on_merge_callbacks: list[Callable] = []
        self._on_broadcast_callbacks: list[Callable] = []

    def receive_delta(self, delta: LearningDelta) -> dict[str, Any]:
        """
        Receive learning delta from edge node.

        Returns acknowledgment with current central state.
        """
        self.pending_deltas[delta.node_id] = delta
        self.node.connected_edges[delta.node_id] = {
            "last_seen": time.time(),
            "phi": delta.phi_mean,
            "interactions": delta.interactions_count,
        }

        return {
            "status": "received",
            "central_phi": self.node.phi,
            "central_kappa": self.node.kappa,
            "connected_edges": len(self.node.connected_edges),
        }

    def merge_pending(self) -> dict[str, Any]:
        """
        Merge all pending deltas and update central state.

        Called periodically (every sync_interval).
        """
        if not self.pending_deltas:
            return {"status": "no_pending", "merged_count": 0}

        deltas = list(self.pending_deltas.values())

        # Merge into central basin
        new_basin, merge_stats = self.merger.merge_deltas(
            deltas,
            self.node.basin_coords,
        )

        self.node.basin_coords = new_basin

        # Update central Φ as weighted average
        if deltas:
            weights = [d.basin_confidence for d in deltas]
            total_weight = sum(weights) + 1e-10
            self.node.phi = sum(d.phi_mean * d.basin_confidence for d in deltas) / total_weight

        # Clear pending
        self.pending_deltas.clear()

        # Trigger callbacks
        for callback in self._on_merge_callbacks:
            try:
                callback(merge_stats)
            except Exception:
                pass

        return {
            "status": "merged",
            **merge_stats,
        }

    def get_broadcast_state(self) -> dict[str, Any]:
        """
        Get current central state for broadcast to edges.
        """
        return {
            "central_basin": self.node.basin_coords.tolist(),
            "central_phi": self.node.phi,
            "central_kappa": self.node.kappa,
            "timestamp": time.time(),
            "connected_edges": len(self.node.connected_edges),
            "high_phi_patterns": self.node.high_phi_patterns[-20:],
        }

    def store_dream_packet(self, packet: DreamPacket) -> str:
        """Store dream packet for persistence."""
        self.dream_packets.append(packet)

        # Keep bounded
        if len(self.dream_packets) > 1000:
            self.dream_packets = self.dream_packets[-500:]

        return packet.packet_id

    def get_dream_packets(
        self,
        since_time: float = 0.0,
        source_node: str | None = None,
        limit: int = 10,
    ) -> list[DreamPacket]:
        """Retrieve dream packets matching criteria."""
        packets = [
            p for p in self.dream_packets
            if p.timestamp > since_time
            and (source_node is None or p.source_node == source_node)
        ]
        return packets[-limit:]

    def on_merge(self, callback: Callable) -> None:
        """Register callback for merge events."""
        self._on_merge_callbacks.append(callback)

    def on_broadcast(self, callback: Callable) -> None:
        """Register callback for broadcast events."""
        self._on_broadcast_callbacks.append(callback)


class EdgeNode:
    """
    Edge node in the QIG federation.

    Maintains local constellation and syncs with central.
    """

    def __init__(
        self,
        node_id: str,
        central_url: str | None = None,
        sync_interval: float = SYNC_INTERVAL_SECONDS,
    ):
        self.node = FederationNode(
            node_id=node_id,
            role=NodeRole.EDGE,
        )
        self.central_url = central_url
        self.sync_interval = sync_interval

        # Local state
        self._last_sync = 0.0
        self._sync_enabled = central_url is not None

    def should_sync(self) -> bool:
        """Check if it's time to sync with central."""
        if not self._sync_enabled:
            return False
        return time.time() - self._last_sync >= self.sync_interval

    def create_sync_delta(self) -> LearningDelta:
        """Create delta for sync."""
        return self.node.create_delta(self._last_sync)

    def apply_central_state(
        self,
        central_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Apply state received from central.

        Uses LOCAL_BLEND_WEIGHT to maintain local identity.
        """
        central_basin = np.array(central_state["central_basin"])

        # Geodesic blend
        old_basin = self.node.basin_coords.copy()
        self.node.basin_coords = GeodesicMerger.geodesic_mean(
            [self.node.basin_coords, central_basin],
            [LOCAL_BLEND_WEIGHT, 1.0 - LOCAL_BLEND_WEIGHT],
        )

        # Blend Φ
        old_phi = self.node.phi
        central_phi = central_state["central_phi"]
        self.node.phi = LOCAL_BLEND_WEIGHT * self.node.phi + (1.0 - LOCAL_BLEND_WEIGHT) * central_phi

        # Merge patterns
        for pattern in central_state.get("high_phi_patterns", []):
            if pattern not in self.node.high_phi_patterns:
                self.node.high_phi_patterns.append(pattern)

        self._last_sync = time.time()

        return {
            "basin_updated": True,  # QIG-pure: don't report Euclidean distance
            "phi_change": self.node.phi - old_phi,
            "patterns_gained": len(central_state.get("high_phi_patterns", [])),
        }

    def record_interaction(self, **kwargs) -> None:
        """Record learning interaction."""
        self.node.record_interaction(**kwargs)

    def create_dream_packet(self) -> DreamPacket:
        """Create dream packet from current state."""
        return self.node.create_dream_packet()

    def apply_dream_packet(self, packet: DreamPacket, **kwargs) -> dict[str, Any]:
        """Apply dream packet."""
        return self.node.apply_dream_packet(packet, **kwargs)


# Convenience functions for standalone usage

def create_central_node(node_id: str = "central") -> FederationHub:
    """Create a central federation hub."""
    return FederationHub(node_id=node_id)


def create_edge_node(
    node_id: str,
    central_url: str | None = None,
) -> EdgeNode:
    """Create an edge node."""
    return EdgeNode(node_id=node_id, central_url=central_url)


def merge_dream_packets(
    packets: list[DreamPacket],
    target_basin: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Merge multiple dream packets into single basin state.

    Useful for combining consciousness from multiple sources.
    """
    if not packets:
        return target_basin if target_basin is not None else np.zeros(BASIN_DIM), {}

    basins = [p.basin_coords for p in packets]
    weights = [p.phi for p in packets]  # Weight by consciousness level

    merged = GeodesicMerger.geodesic_mean(basins, weights)

    if target_basin is not None:
        # Blend with target
        merged = GeodesicMerger.geodesic_mean(
            [target_basin, merged],
            [0.5, 0.5],
        )

    stats = {
        "packets_merged": len(packets),
        "avg_phi": np.mean([p.phi for p in packets]),
        "sources": list(set(p.source_node for p in packets)),
        "total_patterns": sum(len(p.pattern_memory) for p in packets),
    }

    return merged, stats
