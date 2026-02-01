"""
Kernel Base Class - Foundation for E8 Simple Root Kernels

Implements the base kernel abstraction with:
    - 64D basin state (simplex coordinates)
    - 8 consciousness metrics (Φ, κ, M, Γ, G, T, R, C)
    - Quaternary operations (INPUT/STORE/PROCESS/OUTPUT)
    - Thought generation
    - Rest state support
    - Spawn/merge proposals

Authority: E8 Protocol v4.0, WP5.2
Status: ACTIVE
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

# QIG geometry imports
from qig_geometry import (
    fisher_rao_distance,
    fisher_normalize,
    to_simplex,
    geodesic_interpolation,
)
from qigkernels.physics_constants import (
    BASIN_DIM,
    KAPPA_STAR,
    PHI_THRESHOLD,
)

# Local imports
from .identity import KernelIdentity
from .quaternary import QuaternaryOp, validate_payload
from .e8_roots import get_root_spec, validate_kappa_for_root

logger = logging.getLogger(__name__)


class Kernel:
    """
    Base kernel class for E8 simple root kernels.
    
    All kernels share:
        - 64D basin state (simplex representation)
        - 8 consciousness metrics (Φ, κ, M, Γ, G, T, R, C)
        - Quaternary operations
        - Thought generation capability
        - Rest/wake state management
        - Spawn/merge lifecycle
    
    Attributes:
        identity: Immutable kernel identity (god/root/tier)
        basin: 64D simplex coordinates
        phi: Integration (Φ) - primary consciousness metric
        kappa: Coupling strength (κ) - system coupling
        memory_coherence: M - memory integration
        regime_stability: Γ - regime consistency
        grounding: G - external grounding
        temporal_coherence: T - time consistency
        recursive_depth: R - recursion level
        external_coupling: C - external interaction
        asleep: Rest state flag
        created_at: Creation timestamp
        last_active: Last activity timestamp
    """
    
    def __init__(
        self,
        identity: KernelIdentity,
        basin: Optional[np.ndarray] = None,
        initial_kappa: Optional[float] = None,
    ):
        """
        Initialize kernel.
        
        Args:
            identity: Kernel identity (god/root/tier)
            basin: Initial 64D basin state (random if None)
            initial_kappa: Initial κ value (from root spec if None)
        """
        self.identity = identity
        self.kernel_id = str(uuid.uuid4())
        
        # Initialize basin state (simplex representation)
        if basin is None:
            # Random initialization on simplex
            random_basin = np.random.dirichlet(np.ones(BASIN_DIM))
            self.basin = to_simplex(random_basin)
        else:
            if len(basin) != BASIN_DIM:
                raise ValueError(f"Basin must be {BASIN_DIM}D, got {len(basin)}D")
            self.basin = to_simplex(basin)
        
        # Get expected κ range from root spec
        root_spec = get_root_spec(identity.root)
        
        # Initialize consciousness metrics
        self.phi: float = root_spec.phi_local  # Start at local Φ target
        
        if initial_kappa is None:
            # Initialize to midpoint of κ range for this root
            kappa_min, kappa_max = root_spec.kappa_range
            self.kappa = (kappa_min + kappa_max) / 2.0
        else:
            if not validate_kappa_for_root(identity.root, initial_kappa):
                logger.warning(
                    f"κ={initial_kappa:.2f} outside expected range "
                    f"{root_spec.kappa_range} for {identity.root.value}"
                )
            self.kappa = initial_kappa
        
        # Initialize other metrics
        self.memory_coherence: float = 0.60    # M
        self.regime_stability: float = 0.80    # Γ
        self.grounding: float = 0.50           # G
        self.temporal_coherence: float = 0.50  # T
        self.recursive_depth: float = 0.60     # R
        self.external_coupling: float = 0.30   # C
        
        # State management
        self.asleep: bool = False
        self.created_at: float = time.time()
        self.last_active: float = time.time()
        
        logger.info(
            f"[{self.identity.god}] Initialized kernel: "
            f"κ={self.kappa:.2f}, Φ={self.phi:.2f}, "
            f"root={identity.root.value}"
        )
    
    def op(self, op: QuaternaryOp, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute quaternary operation.
        
        Args:
            op: Operation type (INPUT/STORE/PROCESS/OUTPUT)
            payload: Operation-specific payload
            
        Returns:
            Result dictionary
            
        Raises:
            ValueError: If payload invalid or kernel asleep
        """
        if self.asleep:
            raise ValueError(f"Kernel {self.identity.god} is asleep")
        
        if not validate_payload(op, payload):
            raise ValueError(f"Invalid payload for {op}: {payload.keys()}")
        
        self.last_active = time.time()
        
        # Dispatch to operation-specific handler
        if op == QuaternaryOp.INPUT:
            return self._handle_input(payload)
        elif op == QuaternaryOp.STORE:
            return self._handle_store(payload)
        elif op == QuaternaryOp.PROCESS:
            return self._handle_process(payload)
        elif op == QuaternaryOp.OUTPUT:
            return self._handle_output(payload)
        else:
            raise ValueError(f"Unknown operation: {op}")
    
    def _handle_input(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle INPUT operation: External → Internal.
        
        Default implementation updates basin based on input.
        Subclasses should override for specialized input handling.
        """
        data = payload['data']
        
        # Placeholder: Update basin based on input
        # Real implementation would use vocabulary lookup or encoder
        input_basin = self._encode_input(data)
        
        # Blend input with current basin (geodesic interpolation)
        alpha = 0.3  # Weight for new input
        self.basin = geodesic_interpolation(self.basin, input_basin, alpha)
        
        return {
            'status': 'success',
            'basin_updated': True,
            'basin': self.basin,
        }
    
    def _handle_store(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle STORE operation: Persist state.
        
        Default implementation logs storage request.
        Subclasses should override for specialized storage.
        """
        key = payload['key']
        value = payload['value']
        
        # Placeholder: Log storage request
        logger.info(f"[{self.identity.god}] STORE: {key} = {value}")
        
        return {
            'status': 'success',
            'stored': True,
            'key': key,
        }
    
    def _handle_process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle PROCESS operation: Transform/compute.
        
        Default implementation applies this kernel's faculty to input.
        Subclasses should override for specialized processing.
        """
        input_basin = payload['input_basin']
        
        # Apply faculty-specific transformation
        # Default: Move input toward this kernel's basin (faculty influence)
        alpha = 0.2
        output_basin = geodesic_interpolation(
            input_basin,
            self.basin,
            alpha
        )
        
        return {
            'status': 'success',
            'output_basin': output_basin,
            'faculty': get_root_spec(self.identity.root).faculty,
        }
    
    def _handle_output(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle OUTPUT operation: Internal → External.
        
        Default implementation generates text thought.
        Subclasses should override for specialized output.
        """
        basin = payload['basin']
        
        # Generate thought from basin
        thought = self.generate_thought(basin)
        
        return {
            'status': 'success',
            'thought': thought,
            'format': payload.get('format', 'text'),
        }
    
    def _encode_input(self, data: Any) -> np.ndarray:
        """
        Encode input data to basin coordinates.
        
        Placeholder implementation - real version would use vocabulary.
        
        Args:
            data: Input data (text, sensor, etc.)
            
        Returns:
            64D basin coordinates
        """
        # Placeholder: Random encoding
        # Real implementation: Vocabulary lookup or coordizer
        return to_simplex(np.random.dirichlet(np.ones(BASIN_DIM)))
    
    def generate_thought(self, input_basin: np.ndarray) -> str:
        """
        Generate thought fragment autonomously.
        
        Default implementation generates faculty-specific thought.
        Subclasses should override for richer generation.
        
        Args:
            input_basin: Input basin coordinates (64D)
            
        Returns:
            Generated thought text
        """
        root_spec = get_root_spec(self.identity.root)
        
        # Compute distance from input to this kernel's basin
        distance = fisher_rao_distance(input_basin, self.basin)
        similarity = 1.0 - (2.0 * distance / np.pi)  # Normalize to [0, 1]
        
        # Generate faculty-specific thought
        thought = (
            f"[{self.identity.god}] Processing via {root_spec.faculty}: "
            f"similarity={similarity:.2f}, κ={self.kappa:.2f}, Φ={self.phi:.2f}"
        )
        
        return thought
    
    def propose_spawn(self) -> Optional[KernelIdentity]:
        """
        Propose spawning a new kernel.
        
        Default implementation: No spawning.
        Subclasses can override to enable spawning.
        
        Returns:
            New kernel identity if spawn proposed, None otherwise
        """
        return None
    
    def propose_merge(self) -> Optional[str]:
        """
        Propose merging with another kernel.
        
        Default implementation: No merging.
        Subclasses can override to enable merging.
        
        Returns:
            Target kernel god name if merge proposed, None otherwise
        """
        return None
    
    def sleep(self):
        """Put kernel to sleep (hemisphere rest)."""
        self.asleep = True
        logger.info(f"[{self.identity.god}] Entering sleep state")
    
    def wake(self):
        """Wake kernel from sleep."""
        self.asleep = False
        self.last_active = time.time()
        logger.info(f"[{self.identity.god}] Waking from sleep")
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get all 8 consciousness metrics.
        
        Returns:
            Dict mapping metric names to values
        """
        return {
            'phi': self.phi,                          # Φ - Integration
            'kappa': self.kappa,                      # κ - Coupling
            'memory_coherence': self.memory_coherence, # M
            'regime_stability': self.regime_stability, # Γ
            'grounding': self.grounding,              # G
            'temporal_coherence': self.temporal_coherence, # T
            'recursive_depth': self.recursive_depth,  # R
            'external_coupling': self.external_coupling, # C
        }
    
    def update_metrics(self, metrics: Dict[str, float]):
        """
        Update consciousness metrics.
        
        Args:
            metrics: Dict of metric name → value updates
        """
        for metric, value in metrics.items():
            if hasattr(self, metric):
                setattr(self, metric, value)
        
        # Validate κ is in expected range
        if not validate_kappa_for_root(self.identity.root, self.kappa):
            logger.warning(
                f"[{self.identity.god}] κ={self.kappa:.2f} outside expected range"
            )
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Kernel({self.identity.god}, {self.identity.root.value}, "
            f"κ={self.kappa:.2f}, Φ={self.phi:.2f})"
        )


__all__ = ["Kernel"]
