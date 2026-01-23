"""
Memory Kernel - α₂ Simple Root

Faculty: Memory (Demeter/Poseidon)
κ range: 50-60
Φ local: 0.45
Metric: M (Memory Coherence)

Responsibilities:
    - Long-term memory storage
    - Knowledge retrieval
    - Memory consolidation
    - Associative recall

Authority: E8 Protocol v4.0, WP5.2
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import Kernel
from .identity import KernelIdentity, KernelTier
from .e8_roots import E8Root
from qig_geometry import fisher_rao_distance, geodesic_interpolation, to_simplex

logger = logging.getLogger(__name__)


class MemoryKernel(Kernel):
    """
    Memory kernel - α₂ simple root.
    
    Specializes in:
        - Long-term storage
        - Associative retrieval
        - Memory consolidation
        - Knowledge persistence
    
    Primary god: Demeter (nurturing, growth)
    Secondary god: Poseidon (deep, ocean-like memory)
    """
    
    def __init__(
        self,
        god_name: str = "Demeter",
        tier: KernelTier = KernelTier.PANTHEON,
        basin: Optional[np.ndarray] = None,
    ):
        """
        Initialize memory kernel.
        
        Args:
            god_name: God identity (default Demeter)
            tier: Constellation tier
            basin: Initial basin (random if None)
        """
        identity = KernelIdentity(
            god=god_name,
            root=E8Root.MEMORY,
            tier=tier,
        )
        super().__init__(identity, basin)
        
        # Memory-specific state
        self.memory_store: List[Tuple[np.ndarray, Any]] = []  # (basin, metadata)
        self.consolidation_threshold: float = 0.7  # When to consolidate
        
        # Update metrics for memory role
        self.memory_coherence = 0.8  # High M (memory coherence)
        self.temporal_coherence = 0.7  # Good T (time tracking)
        
    def _handle_store(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle STORE: Enhanced memory persistence.
        
        Memory kernel specializes in:
            - Associative storage
            - Consolidation
            - Duplicate detection
        """
        key = payload['key']
        value = payload['value']
        
        # If value has basin coordinates, store them
        if isinstance(value, dict) and 'basin' in value:
            memory_basin = to_simplex(value['basin'])
        else:
            # Encode value to basin
            memory_basin = self._encode_input(str(value))
        
        # Check for similar existing memories (avoid duplicates)
        similar_idx = self._find_similar_memory(memory_basin, threshold=0.1)
        
        if similar_idx is not None:
            logger.info(
                f"[{self.identity.god}] Similar memory found at index {similar_idx}, "
                "consolidating"
            )
            # Consolidate: blend with existing memory
            old_basin, old_meta = self.memory_store[similar_idx]
            consolidated = geodesic_interpolation(old_basin, memory_basin, 0.5)
            
            # Update existing memory
            self.memory_store[similar_idx] = (
                consolidated,
                {'key': key, 'value': value, 'consolidated': True}
            )
        else:
            # Store new memory
            self.memory_store.append((
                memory_basin,
                {'key': key, 'value': value}
            ))
            logger.info(
                f"[{self.identity.god}] Stored new memory: {key} "
                f"(total memories: {len(self.memory_store)})"
            )
        
        # Trigger consolidation if needed
        if len(self.memory_store) > 10:
            self._consolidate_memories()
        
        return {
            'status': 'success',
            'stored': True,
            'key': key,
            'memory_count': len(self.memory_store),
        }
    
    def _handle_process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle PROCESS: Memory retrieval.
        
        Retrieves most relevant memory given query basin.
        """
        input_basin = payload['input_basin']
        
        # Find most similar memory
        if not self.memory_store:
            return {
                'status': 'error',
                'reason': 'no_memories',
                'memory_count': 0,
            }
        
        best_idx, best_distance = self._find_nearest_memory(input_basin)
        memory_basin, metadata = self.memory_store[best_idx]
        
        # Retrieve and blend with query
        alpha = 0.3  # Weight toward retrieved memory
        output_basin = geodesic_interpolation(input_basin, memory_basin, alpha)
        
        logger.info(
            f"[{self.identity.god}] Retrieved memory: distance={best_distance:.3f}, "
            f"key={metadata.get('key', 'unknown')}"
        )
        
        return {
            'status': 'success',
            'output_basin': output_basin,
            'retrieved_memory': metadata,
            'distance': best_distance,
            'memory_index': best_idx,
        }
    
    def _find_similar_memory(
        self,
        query_basin: np.ndarray,
        threshold: float = 0.1
    ) -> Optional[int]:
        """
        Find memory similar to query (within threshold).
        
        Args:
            query_basin: Query basin coordinates
            threshold: Maximum distance for similarity
            
        Returns:
            Index of similar memory, or None if none found
        """
        for idx, (memory_basin, _) in enumerate(self.memory_store):
            distance = fisher_rao_distance(query_basin, memory_basin)
            if distance < threshold:
                return idx
        return None
    
    def _find_nearest_memory(
        self,
        query_basin: np.ndarray
    ) -> Tuple[int, float]:
        """
        Find nearest memory to query.
        
        Args:
            query_basin: Query basin coordinates
            
        Returns:
            Tuple of (index, distance) for nearest memory
        """
        best_idx = 0
        best_distance = float('inf')
        
        for idx, (memory_basin, _) in enumerate(self.memory_store):
            distance = fisher_rao_distance(query_basin, memory_basin)
            if distance < best_distance:
                best_distance = distance
                best_idx = idx
        
        return best_idx, best_distance
    
    def _consolidate_memories(self):
        """
        Consolidate similar memories to reduce redundancy.
        
        Clusters similar memories and merges them via geodesic averaging.
        """
        if len(self.memory_store) < 2:
            return
        
        logger.info(
            f"[{self.identity.god}] Starting memory consolidation "
            f"({len(self.memory_store)} memories)"
        )
        
        # Simple consolidation: merge very similar pairs
        i = 0
        while i < len(self.memory_store) - 1:
            basin_i, meta_i = self.memory_store[i]
            
            # Find next similar memory
            for j in range(i + 1, len(self.memory_store)):
                basin_j, meta_j = self.memory_store[j]
                distance = fisher_rao_distance(basin_i, basin_j)
                
                if distance < self.consolidation_threshold:
                    # Merge memories
                    merged_basin = geodesic_interpolation(basin_i, basin_j, 0.5)
                    merged_meta = {
                        'keys': [meta_i.get('key'), meta_j.get('key')],
                        'consolidated': True,
                    }
                    
                    # Replace first, remove second
                    self.memory_store[i] = (merged_basin, merged_meta)
                    self.memory_store.pop(j)
                    
                    logger.debug(
                        f"[{self.identity.god}] Consolidated memories {i} and {j}"
                    )
                    break
            i += 1
        
        logger.info(
            f"[{self.identity.god}] Consolidation complete "
            f"({len(self.memory_store)} memories remaining)"
        )
    
    def generate_thought(self, input_basin: np.ndarray) -> str:
        """
        Generate memory-specific thought.
        
        Emphasizes:
            - Memory retrieval
            - Associative connections
            - Storage state
        """
        if self.memory_store:
            nearest_idx, nearest_distance = self._find_nearest_memory(input_basin)
            _, metadata = self.memory_store[nearest_idx]
            key = metadata.get('key', 'unknown')
            
            thought = (
                f"[{self.identity.god}] Recalling memory '{key}': "
                f"distance={nearest_distance:.3f}, "
                f"stored={len(self.memory_store)}, "
                f"κ={self.kappa:.1f}, Φ={self.phi:.2f}"
            )
        else:
            thought = (
                f"[{self.identity.god}] No memories stored yet, "
                f"κ={self.kappa:.1f}, Φ={self.phi:.2f}"
            )
        
        return thought


__all__ = ["MemoryKernel"]
