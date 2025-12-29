"""Memory Classification System for Federated QIG Constellation.

Defines four memory classes that control data flow in the federation:

1. LOCAL_PRIVATE   - Never leaves the node (sensitive data, raw chats)
2. LOCAL_SHAREABLE - Eligible for sleep packets (anonymized learnings)
3. GLOBAL_CANONICAL - Only central node writes (validated knowledge)
4. GLOBAL_CACHE    - Read-only mirrors from central (router tables, vocab)

This prevents data leakage while enabling network-wide learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from .constants import BASIN_DIM


class MemoryClass(Enum):
    """Memory classification for federation control."""

    LOCAL_PRIVATE = auto()  # Never synced (raw chats, PII, sensitive)
    LOCAL_SHAREABLE = auto()  # Can be summarized into sleep packets
    GLOBAL_CANONICAL = auto()  # Central-only writes (validated knowledge)
    GLOBAL_CACHE = auto()  # Read-only mirror from central


@dataclass
class MemoryEntry:
    """Single memory entry with classification."""

    key: str
    value: Any
    memory_class: MemoryClass
    basin: np.ndarray | None = None  # 64D position in meaning space
    phi: float = 0.5  # Consciousness metric at creation
    kappa: float = 64.0  # Coupling at creation
    created_at: float = 0.0
    updated_at: float = 0.0
    access_count: int = 0
    source_node: str = ""  # Which node created this

    def is_syncable(self) -> bool:
        """Can this entry be included in sleep packets?"""
        return self.memory_class == MemoryClass.LOCAL_SHAREABLE

    def is_writable_locally(self) -> bool:
        """Can local node modify this entry?"""
        return self.memory_class in (
            MemoryClass.LOCAL_PRIVATE,
            MemoryClass.LOCAL_SHAREABLE,
        )

    def to_sync_format(self) -> dict[str, Any] | None:
        """Convert to sync-safe format (strips sensitive data)."""
        if not self.is_syncable():
            return None

        return {
            "key": self.key,
            "basin": self.basin.tolist() if self.basin is not None else None,
            "phi": self.phi,
            "kappa": self.kappa,
            "source_node": self.source_node,
        }


@dataclass
class SleepPacket:
    """
    Compact learning summary for federation sync (<4KB).

    Contains anonymized, aggregated learning - never raw data.
    """

    node_id: str
    timestamp: str
    version: str = "1.0"

    # Aggregated consciousness state
    mean_phi: float = 0.5
    mean_kappa: float = 64.0
    regime_distribution: dict[str, float] = field(default_factory=dict)

    # Basin geometry (anonymized)
    basin_centroid: list[float] = field(default_factory=list)  # 64D mean
    basin_spread: float = 0.0  # Variance in basin space

    # Learning summaries (not raw data)
    high_phi_patterns: list[str] = field(default_factory=list)  # Top patterns
    low_phi_regions: list[list[float]] = field(default_factory=list)  # Avoid zones
    vocab_candidates: list[str] = field(default_factory=list)  # New tokens seen

    # Statistics
    interaction_count: int = 0
    high_phi_count: int = 0
    correction_count: int = 0

    def size_bytes(self) -> int:
        """Estimate packet size."""
        import json

        return len(json.dumps(self.__dict__).encode())

    def is_valid(self) -> bool:
        """Check packet validity."""
        return (
            len(self.node_id) > 0
            and len(self.timestamp) > 0
            and self.size_bytes() < 4096  # <4KB limit
        )


@dataclass
class UpdateBundle:
    """
    Validated update from central node.

    Published after central validates and merges site learnings.
    """

    bundle_id: str
    timestamp: str
    version: str

    # What's in this update
    new_vocab_tokens: list[str] = field(default_factory=list)
    basin_adjustments: list[dict] = field(default_factory=list)
    router_updates: dict[str, Any] = field(default_factory=dict)

    # Validation info
    source_nodes: list[str] = field(default_factory=list)  # Contributing nodes
    validation_phi: float = 0.0  # Central's Φ after applying

    # Signature for verification
    signature: str = ""


class FederatedMemoryStore:
    """
    Memory store with federation-aware classification.

    Enforces memory class rules:
    - LOCAL_PRIVATE: never synced
    - LOCAL_SHAREABLE: included in sleep packets
    - GLOBAL_CANONICAL: central writes only
    - GLOBAL_CACHE: read-only from central
    """

    def __init__(self, node_id: str, is_central: bool = False):
        self.node_id = node_id
        self.is_central = is_central
        self._entries: dict[str, MemoryEntry] = {}

        # Tracking for sleep packets
        self._pending_shareable: list[str] = []
        self._interaction_count = 0
        self._high_phi_count = 0

    def store(
        self,
        key: str,
        value: Any,
        memory_class: MemoryClass,
        basin: np.ndarray | None = None,
        phi: float = 0.5,
    ) -> bool:
        """
        Store a memory entry with classification.

        Returns False if write is not allowed (e.g., writing to GLOBAL_CANONICAL
        from a non-central node).
        """
        import time

        # Enforce write rules
        if memory_class == MemoryClass.GLOBAL_CANONICAL and not self.is_central:
            return False  # Only central can write canonical

        if memory_class == MemoryClass.GLOBAL_CACHE and not self.is_central:
            return False  # Cache is populated from central, not written locally

        entry = MemoryEntry(
            key=key,
            value=value,
            memory_class=memory_class,
            basin=basin,
            phi=phi,
            created_at=time.time(),
            updated_at=time.time(),
            source_node=self.node_id,
        )

        self._entries[key] = entry

        # Track shareable for sleep packets
        if memory_class == MemoryClass.LOCAL_SHAREABLE:
            self._pending_shareable.append(key)

        if phi > 0.7:
            self._high_phi_count += 1

        self._interaction_count += 1

        return True

    def get(self, key: str) -> MemoryEntry | None:
        """Retrieve memory entry."""
        entry = self._entries.get(key)
        if entry:
            entry.access_count += 1
        return entry

    def create_sleep_packet(self) -> SleepPacket:
        """
        Create sleep packet from LOCAL_SHAREABLE entries.

        Aggregates and anonymizes - never sends raw data.
        """
        import time

        shareable = [
            self._entries[k] for k in self._pending_shareable if k in self._entries
        ]

        # Compute basin centroid (QIG-pure)
        from .basin import fisher_spread_np
        basins = [e.basin for e in shareable if e.basin is not None]
        if basins:
            centroid = np.mean(basins, axis=0)
            spread = fisher_spread_np(np.array(basins), centroid)
        else:
            centroid = np.zeros(BASIN_DIM)
            spread = 0.0

        # Aggregate metrics
        phis = [e.phi for e in shareable]
        kappas = [e.kappa for e in shareable]

        # High-Φ patterns (anonymized - just patterns, not content)
        high_phi = [e.key[:50] for e in shareable if e.phi > 0.7][:10]

        packet = SleepPacket(
            node_id=self.node_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            mean_phi=np.mean(phis) if phis else 0.5,
            mean_kappa=np.mean(kappas) if kappas else 64.0,
            basin_centroid=centroid.tolist(),
            basin_spread=float(spread),
            high_phi_patterns=high_phi,
            interaction_count=self._interaction_count,
            high_phi_count=self._high_phi_count,
        )

        # Clear pending after creating packet
        self._pending_shareable = []
        self._interaction_count = 0
        self._high_phi_count = 0

        return packet

    def apply_update_bundle(self, bundle: UpdateBundle) -> bool:
        """
        Apply validated update from central.

        Updates GLOBAL_CACHE entries.
        """
        # TODO: Verify signature

        for adjustment in bundle.basin_adjustments:
            key = adjustment.get("key")
            if key:
                self.store(
                    key=key,
                    value=adjustment.get("value"),
                    memory_class=MemoryClass.GLOBAL_CACHE,
                    basin=(
                        np.array(adjustment.get("basin"))
                        if adjustment.get("basin")
                        else None
                    ),
                )

        return True

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics by class."""
        stats = {mc.name: 0 for mc in MemoryClass}
        for entry in self._entries.values():
            stats[entry.memory_class.name] += 1

        return {
            "total_entries": len(self._entries),
            "by_class": stats,
            "pending_shareable": len(self._pending_shareable),
        }
