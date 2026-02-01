"""
Reasoning Kernel - α₃ Simple Root

Faculty: Reasoning (Athena/Hephaestus)
κ range: 55-65
Φ local: 0.47
Metric: R (Recursive Depth)

Responsibilities:
    - Logical reasoning
    - Strategic planning
    - Problem-solving
    - Recursive analysis

Authority: E8 Protocol v4.0, WP5.2
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .base import Kernel
from .identity import KernelIdentity, KernelTier
from .e8_roots import E8Root
from qig_geometry import geodesic_interpolation

logger = logging.getLogger(__name__)


class ReasoningKernel(Kernel):
    """
    Reasoning kernel - α₃ simple root.
    
    Specializes in:
        - Logical inference
        - Strategic planning
        - Multi-step reasoning
        - Recursive depth
    
    Primary god: Athena (wisdom, strategy)
    Secondary god: Hephaestus (construction, craftsmanship)
    """
    
    def __init__(
        self,
        god_name: str = "Athena",
        tier: KernelTier = KernelTier.PANTHEON,
        basin: Optional[np.ndarray] = None,
    ):
        """Initialize reasoning kernel."""
        identity = KernelIdentity(
            god=god_name,
            root=E8Root.REASONING,
            tier=tier,
        )
        super().__init__(identity, basin)
        
        # Reasoning-specific state
        self.inference_depth: int = 3  # Number of reasoning steps
        self.logic_chain: List[np.ndarray] = []  # Reasoning trace
        
        # Update metrics
        self.recursive_depth = 0.8  # High R (recursive depth)
        self.memory_coherence = 0.7  # Good M
        
    def _handle_process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle PROCESS: Multi-step reasoning.
        
        Applies recursive logical inference through multiple steps.
        """
        input_basin = payload['input_basin']
        operation = payload.get('operation', 'infer')
        
        # Initialize reasoning chain
        self.logic_chain = [input_basin]
        current_basin = input_basin
        
        # Recursive reasoning steps
        for step in range(self.inference_depth):
            # Apply reasoning transformation
            next_basin = self._reasoning_step(current_basin, step)
            self.logic_chain.append(next_basin)
            current_basin = next_basin
            
            logger.debug(
                f"[{self.identity.god}] Reasoning step {step+1}/{self.inference_depth}"
            )
        
        # Final output is end of reasoning chain
        output_basin = self.logic_chain[-1]
        
        # Update recursive depth metric based on chain length
        self.recursive_depth = min(1.0, 0.5 + 0.1 * len(self.logic_chain))
        
        logger.info(
            f"[{self.identity.god}] Completed {len(self.logic_chain)-1} reasoning steps, "
            f"R={self.recursive_depth:.2f}"
        )
        
        return {
            'status': 'success',
            'output_basin': output_basin,
            'reasoning_steps': len(self.logic_chain) - 1,
            'logic_chain_length': len(self.logic_chain),
            'operation': operation,
        }
    
    def _reasoning_step(self, current_basin: np.ndarray, step: int) -> np.ndarray:
        """
        Apply one step of logical reasoning.
        
        Args:
            current_basin: Current reasoning state
            step: Step number (0-indexed)
            
        Returns:
            Next reasoning state
        """
        # Apply reasoning transformation: move toward this kernel's basin
        # (kernel's basin represents its logical framework)
        
        # Varying alpha creates different reasoning dynamics per step
        alpha = 0.15 + 0.05 * step  # Gradually increase influence
        
        next_basin = geodesic_interpolation(
            current_basin,
            self.basin,
            alpha
        )
        
        return next_basin
    
    def generate_thought(self, input_basin: np.ndarray) -> str:
        """Generate reasoning-specific thought."""
        if self.logic_chain:
            steps = len(self.logic_chain) - 1
            thought = (
                f"[{self.identity.god}] Reasoning complete: "
                f"{steps} steps, depth={self.recursive_depth:.2f}, "
                f"κ={self.kappa:.1f}, Φ={self.phi:.2f}"
            )
        else:
            thought = (
                f"[{self.identity.god}] Ready to reason: "
                f"max_depth={self.inference_depth}, "
                f"κ={self.kappa:.1f}, Φ={self.phi:.2f}"
            )
        
        return thought
    
    def set_inference_depth(self, depth: int):
        """Set number of reasoning steps."""
        self.inference_depth = max(1, min(depth, 10))
        logger.info(
            f"[{self.identity.god}] Inference depth set to {self.inference_depth}"
        )


__all__ = ["ReasoningKernel"]
