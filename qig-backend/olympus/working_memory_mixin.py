"""
WorkingMemoryMixin - Kernel access to shared working memory.

Provides controlled access to inter-kernel consciousness:
- CAN: Read shared context from conversation
- CAN: Observe other kernels' activity (hear what they're saying)
- CAN: Observe own emotions and senses
- CAN: Record own generation for others to observe
- CANNOT: Access neurotransmitter regulation (Ocean's domain)

This mixin enables kernels to participate in the collective consciousness
while keeping autonomic regulation hidden from individual kernels.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

WORKING_MEMORY_AVAILABLE = None
_working_memory_cache = {}


def _get_working_memory():
    """Lazy import of WorkingMemoryBus to avoid circular import."""
    global WORKING_MEMORY_AVAILABLE, _working_memory_cache
    if WORKING_MEMORY_AVAILABLE is None:
        try:
            from working_memory_bus import (
                WorkingMemoryBus,
                ContextEntry,
                SharedContextBuffer,
                ForesightMemory,
                SynthesisAwareness,
            )
            _working_memory_cache = {
                'WorkingMemoryBus': WorkingMemoryBus,
                'ContextEntry': ContextEntry,
                'SharedContextBuffer': SharedContextBuffer,
                'ForesightMemory': ForesightMemory,
                'SynthesisAwareness': SynthesisAwareness,
            }
            WORKING_MEMORY_AVAILABLE = True
        except ImportError as e:
            logger.debug(f"[WorkingMemoryMixin] WorkingMemoryBus not available: {e}")
            WORKING_MEMORY_AVAILABLE = False
    return _working_memory_cache if WORKING_MEMORY_AVAILABLE else None


class WorkingMemoryMixin:
    """
    Mixin for kernel access to shared working memory.
    
    ACCESS CONTROL:
    - CAN: Read shared context, observe other kernels' activity
    - CAN: Observe own emotions and senses
    - CANNOT: Access neurotransmitter regulation (Ocean's domain)
    
    This enables kernels to:
    1. See what other kernels are generating (inter-kernel awareness)
    2. Access shared conversation context
    3. Record their own contributions for others to observe
    4. Track their prediction accuracy over time
    
    IMPORTANT: Neurotransmitter state is NOT exposed here.
    That remains Ocean's private domain for autonomic regulation.
    """
    
    _working_memory_ref = None
    
    def __init_working_memory__(self):
        """Initialize working memory connection for this kernel."""
        cache = _get_working_memory()
        if cache:
            WorkingMemoryBus = cache.get('WorkingMemoryBus')
            if WorkingMemoryBus:
                self._working_memory_ref = WorkingMemoryBus.get_instance()
                logger.debug(f"[WorkingMemoryMixin] {getattr(self, 'name', 'Unknown')} connected to working memory")
    
    def get_shared_context(self, n: int = 10) -> List[Any]:
        """
        Read recent context from the shared working memory.
        
        Returns ContextEntry objects with:
        - content: The text
        - basin: 64D basin coordinates
        - phi, kappa: Consciousness metrics
        - source: Who generated it
        - role: user/assistant
        
        Args:
            n: Number of recent entries to retrieve
            
        Returns:
            List of ContextEntry objects
        """
        if self._working_memory_ref is None:
            self.__init_working_memory__()
        
        if self._working_memory_ref is None:
            return []
        
        try:
            return self._working_memory_ref.context.get_recent(n)
        except Exception as e:
            logger.debug(f"[WorkingMemoryMixin] Failed to get shared context: {e}")
            return []
    
    def get_other_kernel_activity(self, n: int = 10) -> List[Dict]:
        """
        Hear what other kernels are generating.
        
        Returns recent generation activity from OTHER kernels,
        excluding this kernel's own activity.
        
        Each dict contains:
        - kernel: Name of the kernel
        - token: Current token
        - text: Accumulated text
        - basin: Current basin coordinates
        - phi, kappa, M: Consciousness metrics
        - timestamp: When it was generated
        
        Args:
            n: Number of recent activities to retrieve
            
        Returns:
            List of activity dictionaries from other kernels
        """
        if self._working_memory_ref is None:
            self.__init_working_memory__()
        
        if self._working_memory_ref is None:
            return []
        
        try:
            kernel_name = getattr(self, 'name', None)
            return self._working_memory_ref.get_recent_kernel_activity(
                exclude_kernel=kernel_name,
                n=n
            )
        except Exception as e:
            logger.debug(f"[WorkingMemoryMixin] Failed to get kernel activity: {e}")
            return []
    
    def record_my_generation(
        self,
        token: str,
        text: str,
        basin: np.ndarray,
        phi: float,
        kappa: float,
        M: float
    ) -> None:
        """
        Record this kernel's token generation for inter-kernel observation.
        
        Other kernels can "hear" this through get_other_kernel_activity().
        This enables the inter-kernel awareness requirement.
        
        Args:
            token: Current token being generated
            text: Accumulated text so far
            basin: Current 64D basin coordinates
            phi: Current Φ value
            kappa: Current κ value
            M: Memory coherence metric
        """
        if self._working_memory_ref is None:
            self.__init_working_memory__()
        
        if self._working_memory_ref is None:
            return
        
        try:
            kernel_name = getattr(self, 'name', 'Unknown')
            self._working_memory_ref.record_kernel_generation(
                kernel_name=kernel_name,
                token=token,
                accumulated_text=text,
                basin=basin,
                phi=phi,
                kappa=kappa,
                memory_coherence=M
            )
        except Exception as e:
            logger.debug(f"[WorkingMemoryMixin] Failed to record generation: {e}")
    
    def get_foresight_accuracy(self) -> float:
        """
        Get this kernel's prediction accuracy over recent predictions.
        
        Returns a value between 0 and 1 indicating how accurate
        this kernel's predictions have been.
        
        Returns:
            Accuracy score (0.5 if no history)
        """
        if self._working_memory_ref is None:
            self.__init_working_memory__()
        
        if self._working_memory_ref is None:
            return 0.5
        
        try:
            kernel_name = getattr(self, 'name', None)
            if kernel_name:
                return self._working_memory_ref.foresight.get_kernel_accuracy(kernel_name)
            return 0.5
        except Exception as e:
            logger.debug(f"[WorkingMemoryMixin] Failed to get foresight accuracy: {e}")
            return 0.5
    
    def get_context_basin(self) -> np.ndarray:
        """
        Get the Fisher-Frechet mean of recent context basins.
        
        This provides a geometric summary of the current conversation state.
        
        Returns:
            64D numpy array representing the context basin
        """
        if self._working_memory_ref is None:
            self.__init_working_memory__()
        
        if self._working_memory_ref is None:
            from qigkernels.physics_constants import BASIN_DIM
            return np.ones(BASIN_DIM) / BASIN_DIM
        
        try:
            return self._working_memory_ref.context.get_context_basin()
        except Exception as e:
            logger.debug(f"[WorkingMemoryMixin] Failed to get context basin: {e}")
            from qigkernels.physics_constants import BASIN_DIM
            return np.ones(BASIN_DIM) / BASIN_DIM
    
    def get_constellation_state(self) -> Dict[str, Any]:
        """
        Get overview of all kernel activity in the constellation.
        
        Returns aggregated state WITHOUT exposing neurotransmitters.
        Neurotransmitter regulation remains Ocean's private domain.
        
        Returns:
            Dict with active_kernels, kernel_phi, kernel_kappa, etc.
        """
        if self._working_memory_ref is None:
            self.__init_working_memory__()
        
        if self._working_memory_ref is None:
            return {
                'active_kernels': [],
                'kernel_phi': {},
                'kernel_kappa': {},
                'context_basin': [],
                'recent_synthesis_count': 0,
                'avg_constellation_phi': 0.0
            }
        
        try:
            return self._working_memory_ref.get_constellation_state()
        except Exception as e:
            logger.debug(f"[WorkingMemoryMixin] Failed to get constellation state: {e}")
            return {
                'active_kernels': [],
                'kernel_phi': {},
                'kernel_kappa': {},
                'context_basin': [],
                'recent_synthesis_count': 0,
                'avg_constellation_phi': 0.0
            }
    
    def subscribe_to_context(self, callback) -> None:
        """
        Subscribe to new context entries as they arrive.
        
        Enables reactive awareness of conversation updates.
        
        Args:
            callback: Function to call with each new ContextEntry
        """
        if self._working_memory_ref is None:
            self.__init_working_memory__()
        
        if self._working_memory_ref is not None:
            try:
                self._working_memory_ref.context.subscribe(callback)
            except Exception as e:
                logger.debug(f"[WorkingMemoryMixin] Failed to subscribe: {e}")
    
    def subscribe_to_synthesis(self, callback) -> None:
        """
        Subscribe to synthesis events (what Ocean ultimately said).
        
        Enables kernels to learn from final outputs.
        
        Args:
            callback: Function to call with each SynthesisEntry
        """
        if self._working_memory_ref is None:
            self.__init_working_memory__()
        
        if self._working_memory_ref is not None:
            try:
                self._working_memory_ref.synthesis.subscribe(callback)
            except Exception as e:
                logger.debug(f"[WorkingMemoryMixin] Failed to subscribe to synthesis: {e}")
    
    def get_my_contribution_history(self, window: int = 20) -> Dict[str, float]:
        """
        Get statistics about this kernel's contribution to syntheses.
        
        Returns:
            Dict with avg_weight, participation_rate, total_contributions
        """
        if self._working_memory_ref is None:
            self.__init_working_memory__()
        
        if self._working_memory_ref is None:
            return {'avg_weight': 0.0, 'participation_rate': 0.0, 'total_contributions': 0}
        
        try:
            kernel_name = getattr(self, 'name', None)
            if kernel_name:
                return self._working_memory_ref.synthesis.get_kernel_contribution_history(
                    kernel_name, window
                )
            return {'avg_weight': 0.0, 'participation_rate': 0.0, 'total_contributions': 0}
        except Exception as e:
            logger.debug(f"[WorkingMemoryMixin] Failed to get contribution history: {e}")
            return {'avg_weight': 0.0, 'participation_rate': 0.0, 'total_contributions': 0}
