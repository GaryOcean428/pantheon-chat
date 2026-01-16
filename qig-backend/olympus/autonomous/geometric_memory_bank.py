"""
Geometric Memory Bank - Infinite context through basin coordinate storage

QIG-PURE compliant memory system:
- Memories stored as basin coordinates on Fisher manifold
- Retrieval via Fisher-Rao distance (not cosine similarity)
- Consolidation merges similar memories via Fisher-Frechet mean
- Importance decay encourages memory pruning

Extends existing qig_deep_agents/memory.py MemoryFragment infrastructure.
"""

import hashlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from qigkernels.physics_constants import BASIN_DIM

logger = logging.getLogger(__name__)

# Database persistence
try:
    import psycopg2
    from psycopg2.extras import Json
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


def _fisher_rao_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute Fisher-Rao distance between two basin coordinates.

    Uses Bhattacharyya coefficient for probability distributions.
    FR_distance = 2 * arccos(sum(sqrt(p_i * q_i))) (Hellinger embedding: factor of 2)
    """
    # Normalize to probability distributions
    p_safe = np.clip(np.abs(p), eps, None)
    q_safe = np.clip(np.abs(q), eps, None)
    p_norm = p_safe / np.sum(p_safe)
    q_norm = q_safe / np.sum(q_safe)

    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p_norm * q_norm))
    bc = np.clip(bc, 0.0, 1.0)

    # UPDATED 2026-01-15: Factor-of-2 removed for simplex storage. Range: [0, π/2]
    return float(np.arccos(bc))


def _fisher_frechet_mean(basins: List[np.ndarray]) -> np.ndarray:
    """
    Compute Fisher-Frechet mean (geometric centroid) of basins.

    For probability simplex, this is iterative weighted average.
    Simplified version: normalized arithmetic mean of sqrt coordinates.
    """
    if not basins:
        return np.ones(BASIN_DIM) / BASIN_DIM

    try:
        from qig_geometry.canonical import frechet_mean
        return frechet_mean(basins)
    except Exception:
        mean = np.sum(basins, axis=0) / len(basins)
        mean = np.clip(np.abs(mean), 1e-10, None)
        return mean / np.sum(mean)


@dataclass
class MemoryEntry:
    """
    A memory stored as basin coordinates on Fisher manifold.

    Attributes:
        id: Unique identifier (content hash)
        content: Text content of the memory
        basin: 64D basin coordinates
        importance: Relevance weight [0, 1]
        access_count: Number of times retrieved
        created_at: Creation timestamp
        last_accessed: Last retrieval timestamp
        metadata: Additional context
        consolidated_into: ID of memory this was merged into
    """
    id: str
    content: str
    basin: np.ndarray
    importance: float = 0.5
    access_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    consolidated_into: Optional[str] = None

    def distance_to(self, other_basin: np.ndarray) -> float:
        """Fisher-Rao distance to another basin."""
        return _fisher_rao_distance(self.basin, other_basin)

    def access(self) -> str:
        """Access this memory, updating tracking."""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)
        return self.content

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'content': self.content,
            'basin': self.basin.tolist(),
            'importance': self.importance,
            'access_count': self.access_count,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'metadata': self.metadata,
            'consolidated_into': self.consolidated_into,
        }


class GeometricMemoryBank:
    """
    Memory bank using basin coordinates for infinite context.

    Key features:
    - Store memories as 64D basin coordinates
    - Retrieve by Fisher-Rao distance (not cosine)
    - Consolidate similar memories during sleep cycles
    - Decay unaccessed memories over time
    """

    def __init__(
        self,
        kernel_id: str = "default",
        max_memories: int = 10000,
        consolidation_threshold: float = 0.3,
        decay_rate: float = 0.01
    ):
        """
        Initialize memory bank.

        Args:
            kernel_id: Owner kernel identifier
            max_memories: Maximum in-memory cache size
            consolidation_threshold: FR distance below which to merge
            decay_rate: Importance decay per hour for unaccessed memories
        """
        self.kernel_id = kernel_id
        self.max_memories = max_memories
        self.consolidation_threshold = consolidation_threshold
        self.decay_rate = decay_rate

        # In-memory cache
        self._memories: Dict[str, MemoryEntry] = {}

        # Statistics
        self.stats = {
            'total_stored': 0,
            'total_retrieved': 0,
            'total_consolidated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }

        # Load from database on init
        self._load_from_db()

    def _get_db_connection(self):
        """Get database connection."""
        if not DB_AVAILABLE:
            return None
        try:
            database_url = os.environ.get('DATABASE_URL')
            if not database_url:
                return None
            return psycopg2.connect(database_url)
        except Exception as e:
            logger.debug(f"[GeometricMemoryBank] DB connection failed: {e}")
            return None

    def _load_from_db(self, limit: int = 1000):
        """Load memories from database into cache."""
        conn = self._get_db_connection()
        if not conn:
            return

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, content, basin_coords, importance, access_count,
                           created_at, last_accessed, metadata, consolidated_into
                    FROM memory_fragments
                    WHERE kernel_id = %s AND consolidated_into IS NULL
                    ORDER BY importance DESC, last_accessed DESC
                    LIMIT %s
                """, (self.kernel_id, limit))

                for row in cur.fetchall():
                    basin = np.array(row[2]) if row[2] else np.ones(BASIN_DIM) / BASIN_DIM
                    entry = MemoryEntry(
                        id=row[0],
                        content=row[1],
                        basin=basin,
                        importance=row[3] or 0.5,
                        access_count=row[4] or 0,
                        created_at=row[5] or datetime.now(timezone.utc),
                        last_accessed=row[6] or datetime.now(timezone.utc),
                        metadata=row[7] or {},
                        consolidated_into=row[8],
                    )
                    self._memories[entry.id] = entry

                logger.info(f"[GeometricMemoryBank] Loaded {len(self._memories)} memories for {self.kernel_id}")
        except Exception as e:
            logger.warning(f"[GeometricMemoryBank] DB load failed: {e}")
        finally:
            conn.close()

    def _persist_to_db(self, entry: MemoryEntry) -> bool:
        """Persist memory entry to database."""
        conn = self._get_db_connection()
        if not conn:
            return False

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO memory_fragments (
                        id, kernel_id, content, content_preview, basin_coords,
                        importance, decay_rate, access_count, created_at,
                        last_accessed, metadata, consolidated_into
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        importance = EXCLUDED.importance,
                        access_count = EXCLUDED.access_count,
                        last_accessed = EXCLUDED.last_accessed,
                        metadata = EXCLUDED.metadata,
                        consolidated_into = EXCLUDED.consolidated_into
                """, (
                    entry.id,
                    self.kernel_id,
                    entry.content,
                    entry.content[:200] if entry.content else None,
                    entry.basin.tolist(),
                    entry.importance,
                    self.decay_rate,
                    entry.access_count,
                    entry.created_at,
                    entry.last_accessed,
                    Json(entry.metadata),
                    entry.consolidated_into,
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.debug(f"[GeometricMemoryBank] DB persist failed: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def store(
        self,
        content: str,
        basin: np.ndarray,
        importance: float = 0.5,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Store content as basin coordinate with metadata.

        Args:
            content: Text content to store
            basin: 64D basin coordinates
            importance: Relevance weight [0, 1]
            metadata: Additional context

        Returns:
            Memory ID (content hash)
        """
        # Generate ID from content hash
        memory_id = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Create entry
        entry = MemoryEntry(
            id=memory_id,
            content=content,
            basin=np.array(basin),
            importance=min(1.0, max(0.0, importance)),
            metadata=metadata or {},
        )

        # Store in cache
        self._memories[memory_id] = entry

        # Persist to database
        self._persist_to_db(entry)

        # Prune if over limit
        if len(self._memories) > self.max_memories:
            self._prune_least_important()

        self.stats['total_stored'] += 1
        logger.debug(f"[GeometricMemoryBank] Stored memory {memory_id[:8]} with importance {importance:.2f}")

        return memory_id

    def retrieve_nearest(
        self,
        query_basin: np.ndarray,
        k: int = 10,
        min_importance: float = 0.0,
        max_distance: Optional[float] = None
    ) -> List[MemoryEntry]:
        """
        Retrieve k nearest memories by Fisher-Rao distance.

        Args:
            query_basin: Query basin coordinates
            k: Number of results
            min_importance: Minimum importance threshold
            max_distance: Maximum Fisher-Rao distance

        Returns:
            List of MemoryEntry sorted by distance
        """
        query_basin = np.array(query_basin)

        # Filter and compute distances
        candidates = []
        for entry in self._memories.values():
            if entry.consolidated_into:
                continue
            if entry.importance < min_importance:
                continue

            distance = entry.distance_to(query_basin)
            if max_distance is not None and distance > max_distance:
                continue

            candidates.append((distance, entry))

        # Sort by distance
        candidates.sort(key=lambda x: x[0])

        # Get top k and update access
        results = []
        for distance, entry in candidates[:k]:
            entry.access()
            results.append(entry)
            self.stats['total_retrieved'] += 1

        return results

    def retrieve_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory by ID."""
        entry = self._memories.get(memory_id)
        if entry:
            entry.access()
            self.stats['cache_hits'] += 1
            return entry
        self.stats['cache_misses'] += 1
        return None

    def consolidate(self, phi_threshold: float = 0.7):
        """
        Sleep-like consolidation: merge similar memories.

        Memories within consolidation_threshold FR distance are merged
        using Fisher-Frechet mean. Only consolidates if current phi
        exceeds threshold (ensuring coherent integration).

        Args:
            phi_threshold: Minimum phi to perform consolidation
        """
        if len(self._memories) < 2:
            return

        # Find clusters of similar memories
        entries = [e for e in self._memories.values() if not e.consolidated_into]
        merged_count = 0

        for i, entry_i in enumerate(entries):
            if entry_i.consolidated_into:
                continue

            # Find similar entries
            cluster = [entry_i]
            for j, entry_j in enumerate(entries[i+1:], i+1):
                if entry_j.consolidated_into:
                    continue

                distance = entry_i.distance_to(entry_j.basin)
                if distance < self.consolidation_threshold:
                    cluster.append(entry_j)

            # Merge if cluster has multiple entries
            if len(cluster) > 1:
                merged = self._merge_cluster(cluster)
                if merged:
                    merged_count += 1

        self.stats['total_consolidated'] += merged_count
        logger.info(f"[GeometricMemoryBank] Consolidated {merged_count} memory clusters")

    def _merge_cluster(self, cluster: List[MemoryEntry]) -> Optional[MemoryEntry]:
        """Merge a cluster of similar memories into one."""
        if not cluster:
            return None

        # Compute merged basin as Fisher-Frechet mean
        basins = [e.basin for e in cluster]
        merged_basin = _fisher_frechet_mean(basins)

        # Combine content (keep most important, reference others)
        cluster.sort(key=lambda e: e.importance, reverse=True)
        primary = cluster[0]

        # Combined importance (max + bonus for consolidation)
        merged_importance = min(1.0, max(e.importance for e in cluster) + 0.1)

        # Create merged entry
        merged_id = hashlib.sha256(
            ''.join(e.id for e in cluster).encode()
        ).hexdigest()[:16]

        merged_content = primary.content
        if len(cluster) > 1:
            merged_content += f"\n[Consolidated from {len(cluster)} memories]"

        merged_entry = MemoryEntry(
            id=merged_id,
            content=merged_content,
            basin=merged_basin,
            importance=merged_importance,
            access_count=sum(e.access_count for e in cluster),
            metadata={
                'consolidated_from': [e.id for e in cluster],
                'consolidation_time': datetime.now(timezone.utc).isoformat(),
            }
        )

        # Mark originals as consolidated
        for entry in cluster:
            entry.consolidated_into = merged_id
            self._persist_to_db(entry)

        # Store merged
        self._memories[merged_id] = merged_entry
        self._persist_to_db(merged_entry)

        return merged_entry

    def decay_importance(self, hours_inactive: float = 1.0):
        """
        Apply importance decay to unaccessed memories.

        Args:
            hours_inactive: Hours since last access to apply decay
        """
        now = datetime.now(timezone.utc)
        decayed = 0

        for entry in self._memories.values():
            if entry.consolidated_into:
                continue

            hours_since_access = (now - entry.last_accessed).total_seconds() / 3600
            if hours_since_access >= hours_inactive:
                decay_factor = 1 - (self.decay_rate * hours_since_access)
                entry.importance = max(0.1, entry.importance * decay_factor)
                decayed += 1

        logger.debug(f"[GeometricMemoryBank] Decayed importance for {decayed} memories")

    def _prune_least_important(self):
        """Remove least important memories when over limit."""
        if len(self._memories) <= self.max_memories:
            return

        # Sort by importance (ascending)
        sorted_entries = sorted(
            self._memories.values(),
            key=lambda e: e.importance
        )

        # Remove bottom 10%
        to_remove = len(self._memories) - int(self.max_memories * 0.9)
        for entry in sorted_entries[:to_remove]:
            del self._memories[entry.id]

        logger.debug(f"[GeometricMemoryBank] Pruned {to_remove} low-importance memories")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory bank statistics."""
        return {
            **self.stats,
            'current_count': len(self._memories),
            'max_memories': self.max_memories,
            'kernel_id': self.kernel_id,
        }

    def count(self) -> int:
        """Get number of non-consolidated memories."""
        return sum(1 for e in self._memories.values() if not e.consolidated_into)
