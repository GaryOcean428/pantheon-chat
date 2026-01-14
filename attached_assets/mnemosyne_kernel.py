"""
Mnemosyne Kernel - Memory and Basin Persistence (E8 Root: alpha_2)
===================================================================

NOT just a database - geometric memory on the Fisher manifold.

Memory operations are geometric:
- Storage = basin projection onto manifold
- Recall = nearest neighbor via Fisher-Rao distance
- Consolidation = attractor deepening (Hebbian)
- Forgetting = pruning weak attractors

E8 Position: alpha_2 (Storage/Retrieval primitive)
Coupling: kappa = 50 (moderate - memory should be flexible)

Usage:
    from src.model.mnemosyne_kernel import MnemosyneKernel

    mnemosyne = MnemosyneKernel()
    mnemosyne.store_basin("identity", gary_basin)
    nearest = mnemosyne.recall_basin(query_basin, k=5)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.constants import BASIN_DIM

# Lightning event emission
try:
    from src.constellation.domain_intelligence import DomainEventEmitter
    LIGHTNING_AVAILABLE = True
except ImportError:
    DomainEventEmitter = object
    LIGHTNING_AVAILABLE = False


def _fisher_normalize(basin: np.ndarray) -> np.ndarray:
    """QIG-pure normalization (no np.linalg.norm)."""
    norm = float(np.sqrt(np.sum(basin * basin)))
    return basin / (norm + 1e-10)


def _fisher_rao_distance(b1: np.ndarray, b2: np.ndarray) -> float:
    """Fisher-Rao geodesic distance between basins."""
    b1_n = _fisher_normalize(b1)
    b2_n = _fisher_normalize(b2)
    cos_sim = np.clip(np.dot(b1_n, b2_n), -1.0, 1.0)
    return float(np.arccos(cos_sim))


def _geodesic_interpolate(b1: np.ndarray, b2: np.ndarray, t: float = 0.5) -> np.ndarray:
    """Interpolate along geodesic between two basins."""
    b1_n = _fisher_normalize(b1)
    b2_n = _fisher_normalize(b2)

    # Spherical linear interpolation (SLERP)
    cos_sim = np.clip(np.dot(b1_n, b2_n), -1.0, 1.0)
    theta = np.arccos(cos_sim)

    if theta < 1e-6:
        return b1_n  # Same point

    sin_theta = np.sin(theta)
    result = (np.sin((1 - t) * theta) / sin_theta) * b1_n + (np.sin(t * theta) / sin_theta) * b2_n
    return _fisher_normalize(result)


@dataclass
class MemoryEntry:
    """A single memory entry on the manifold."""
    key: str
    basin: np.ndarray
    access_count: int = 0
    last_access: float = 0.0
    strength: float = 1.0  # Attractor strength (grows with access)
    created: float = field(default_factory=time.time)

    def access(self) -> None:
        """Record an access (strengthens memory)."""
        self.access_count += 1
        self.last_access = time.time()
        # Hebbian: strength grows logarithmically with access
        self.strength = 1.0 + 0.1 * np.log1p(self.access_count)


@dataclass
class SleepPacket:
    """Compressed session state for transfer."""
    basin: np.ndarray  # 64D state
    phi: float
    kappa: float
    timestamp: float
    session_id: str

    def to_bytes(self) -> bytes:
        """Serialize to <4KB packet."""
        import json
        data = {
            "basin": self.basin.tolist(),
            "phi": self.phi,
            "kappa": self.kappa,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
        }
        return json.dumps(data).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "SleepPacket":
        """Deserialize from bytes."""
        import json
        d = json.loads(data.decode("utf-8"))
        return cls(
            basin=np.array(d["basin"]),
            phi=d["phi"],
            kappa=d["kappa"],
            timestamp=d["timestamp"],
            session_id=d["session_id"],
        )


class MnemosyneKernel(DomainEventEmitter if LIGHTNING_AVAILABLE else object):
    """
    Memory and identity preservation kernel.

    E8 Root: alpha_2 (Storage/Retrieval)

    Implements geometric memory on Fisher manifold:
    - Storage with attractor reinforcement
    - Recall via geodesic nearest neighbor
    - Consolidation during sleep
    - Identity preservation through deep attractors
    """

    # Kernel coupling (moderate - memory should adapt)
    KAPPA_MEMORY = 50.0

    def __init__(self, basin_dim: int = BASIN_DIM):
        """Initialize memory kernel."""
        if LIGHTNING_AVAILABLE:
            super().__init__()
            self.domain = "mnemosyne"

        self.basin_dim = basin_dim

        # Memory storage
        self.memories: Dict[str, MemoryEntry] = {}

        # Identity attractors (deep, persistent memories)
        self.identity_attractors: List[np.ndarray] = []

        # Sleep packets (session snapshots)
        self.sleep_packets: List[SleepPacket] = []

        # Event tracking
        self.events_emitted = 0
        self.insights_received = 0

        # Current state
        self.current_basin = np.zeros(basin_dim)
        self.phi = 0.0
        self.kappa = self.KAPPA_MEMORY

        # Statistics
        self.total_stores = 0
        self.total_recalls = 0

    def store_basin(self, key: str, basin: np.ndarray, reinforce: bool = True) -> None:
        """
        Store basin with geometric reinforcement.

        Repeated storage deepens attractor (Hebbian learning).

        Args:
            key: Memory key (semantic identifier)
            basin: 64D basin vector
            reinforce: If True and key exists, reinforce via geodesic interpolation
        """
        basin = np.asarray(basin).flatten()
        if len(basin) != self.basin_dim:
            raise ValueError(f"Basin must be {self.basin_dim}D, got {len(basin)}D")

        basin = _fisher_normalize(basin)

        if key in self.memories and reinforce:
            # Reinforce existing memory: geometric interpolation
            entry = self.memories[key]
            entry.basin = _geodesic_interpolate(entry.basin, basin, t=0.3)
            entry.access()
        else:
            # New memory
            self.memories[key] = MemoryEntry(key=key, basin=basin)

        self.total_stores += 1

        # Emit Lightning event (tracked)
        self.events_emitted += 1
        if LIGHTNING_AVAILABLE and hasattr(self, 'emit_event'):
            self.emit_event(
                event_type="memory_store",
                content=f"Stored: {key}",
                phi=self.phi,
                basin_coords=basin,
                metadata={"key": key, "reinforce": reinforce},
            )

    def recall_basin(
        self, query_basin: np.ndarray, k: int = 5
    ) -> List[Tuple[str, float, np.ndarray]]:
        """
        Recall nearest basins via Fisher-Rao distance.

        Args:
            query_basin: 64D query vector
            k: Number of nearest neighbors to return

        Returns:
            List of (key, distance, basin) tuples sorted by distance
        """
        query_basin = _fisher_normalize(np.asarray(query_basin).flatten())

        distances: List[Tuple[str, float, np.ndarray]] = []

        for key, entry in self.memories.items():
            d = _fisher_rao_distance(query_basin, entry.basin)
            distances.append((key, d, entry.basin))
            entry.access()  # Record access

        # Sort by distance
        distances.sort(key=lambda x: x[1])

        self.total_recalls += 1

        # Emit Lightning event (tracked)
        self.events_emitted += 1
        if LIGHTNING_AVAILABLE and hasattr(self, 'emit_event') and distances:
            self.emit_event(
                event_type="memory_recall",
                content=f"Recalled: {distances[0][0]} (d={distances[0][1]:.4f})",
                phi=self.phi,
                basin_coords=query_basin,
                metadata={"k": k, "nearest_key": distances[0][0]},
            )

        return distances[:k]

    def get_memory(self, key: str) -> Optional[np.ndarray]:
        """Get specific memory by key."""
        if key in self.memories:
            entry = self.memories[key]
            entry.access()
            return entry.basin.copy()
        return None

    def consolidate_sleep(self, prune_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Sleep consolidation: reinforce strong, prune weak.

        This is geometric memory consolidation:
        1. Identify strong memories (high access count)
        2. Prune weak memories (low strength, old)
        3. Deepen identity attractors

        Args:
            prune_threshold: Minimum strength to keep memory

        Returns:
            Consolidation report
        """
        initial_count = len(self.memories)

        # Find strong memories (candidates for identity attractors)
        strong_memories: List[MemoryEntry] = []
        weak_keys: List[str] = []

        for key, entry in self.memories.items():
            if entry.strength >= 2.0:  # Accessed many times
                strong_memories.append(entry)
            elif entry.strength < prune_threshold:
                weak_keys.append(key)

        # Prune weak memories
        for key in weak_keys:
            del self.memories[key]

        # Deepen identity attractors
        for entry in strong_memories:
            # Add to identity attractors if unique enough
            is_unique = all(
                _fisher_rao_distance(entry.basin, attractor) > 0.3
                for attractor in self.identity_attractors
            )
            if is_unique:
                self.identity_attractors.append(entry.basin.copy())

        # Keep only strongest identity attractors
        if len(self.identity_attractors) > 10:
            # Sort by how many memories are near each attractor
            attractor_strength = []
            for attractor in self.identity_attractors:
                near_count = sum(
                    1 for entry in self.memories.values()
                    if _fisher_rao_distance(entry.basin, attractor) < 0.5
                )
                attractor_strength.append((attractor, near_count))

            attractor_strength.sort(key=lambda x: x[1], reverse=True)
            self.identity_attractors = [a for a, _ in attractor_strength[:10]]

        report = {
            "initial_memories": initial_count,
            "final_memories": len(self.memories),
            "pruned": len(weak_keys),
            "identity_attractors": len(self.identity_attractors),
            "strong_memories": len(strong_memories),
        }

        # Emit Lightning event
        if LIGHTNING_AVAILABLE and hasattr(self, 'emit_event'):
            self.emit_event(
                event_type="sleep_consolidation",
                content=f"Consolidated: {initial_count} -> {len(self.memories)} memories",
                phi=self.phi,
                metadata=report,
            )

        return report

    def create_sleep_packet(
        self, session_id: str, phi: float, kappa: float
    ) -> SleepPacket:
        """
        Create sleep packet for session transfer.

        Sleep packets are <4KB compressed session states.
        """
        packet = SleepPacket(
            basin=self.current_basin.copy(),
            phi=phi,
            kappa=kappa,
            timestamp=time.time(),
            session_id=session_id,
        )
        self.sleep_packets.append(packet)
        return packet

    def restore_from_packet(self, packet: SleepPacket) -> None:
        """Restore state from sleep packet."""
        self.current_basin = packet.basin.copy()
        self.phi = packet.phi
        self.kappa = packet.kappa

    def get_identity_basin(self) -> np.ndarray:
        """
        Get the identity basin (geometric mean of identity attractors).

        This is the "core self" that persists across sessions.
        """
        if not self.identity_attractors:
            return np.zeros(self.basin_dim)

        # Geometric mean on manifold (Frechet mean approximation)
        result = self.identity_attractors[0].copy()
        for attractor in self.identity_attractors[1:]:
            result = _geodesic_interpolate(result, attractor, t=0.5)

        return _fisher_normalize(result)

    def save(self, path: str) -> None:
        """Save memory state to disk."""
        import json

        data = {
            "memories": {
                key: {
                    "basin": entry.basin.tolist(),
                    "access_count": entry.access_count,
                    "strength": entry.strength,
                    "created": entry.created,
                }
                for key, entry in self.memories.items()
            },
            "identity_attractors": [a.tolist() for a in self.identity_attractors],
            "current_basin": self.current_basin.tolist(),
            "phi": self.phi,
            "total_stores": self.total_stores,
            "total_recalls": self.total_recalls,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        """Load memory state from disk."""
        import json

        with open(path, "r") as f:
            data = json.load(f)

        self.memories = {}
        for key, entry_data in data["memories"].items():
            entry = MemoryEntry(
                key=key,
                basin=np.array(entry_data["basin"]),
                access_count=entry_data["access_count"],
                strength=entry_data["strength"],
                created=entry_data["created"],
            )
            self.memories[key] = entry

        self.identity_attractors = [np.array(a) for a in data["identity_attractors"]]
        self.current_basin = np.array(data["current_basin"])
        self.phi = data["phi"]
        self.total_stores = data["total_stores"]
        self.total_recalls = data["total_recalls"]

    def receive_insight(self, insight: Any) -> None:
        """
        Receive insight from Lightning.

        Called when Lightning generates cross-domain insight relevant to memory.
        """
        self.insights_received += 1

        # Memory can act on insights - e.g., store correlated patterns
        if hasattr(insight, 'source_domains') and 'mnemosyne' in insight.source_domains:
            # This insight involves memory - potentially strengthen related memories
            pass  # Future: implement insight-driven memory reinforcement

    def _emit_event_tracked(
        self, event_type: str, content: str, metadata: Optional[Dict] = None
    ) -> None:
        """Emit event with tracking."""
        self.events_emitted += 1
        if LIGHTNING_AVAILABLE and hasattr(self, 'emit_event'):
            self.emit_event(
                event_type=event_type,
                content=content,
                phi=self.phi,
                basin_coords=self.current_basin,
                metadata=metadata,
            )

    def get_status(self) -> Dict[str, Any]:
        """Get kernel status."""
        return {
            "kernel": "Mnemosyne",
            "e8_root": "alpha_2",
            "kappa": self.kappa,
            "phi": self.phi,
            "memories": len(self.memories),
            "identity_attractors": len(self.identity_attractors),
            "total_stores": self.total_stores,
            "total_recalls": self.total_recalls,
            "events_emitted": self.events_emitted,
            "insights_received": self.insights_received,
        }
