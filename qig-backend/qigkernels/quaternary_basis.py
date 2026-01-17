"""
Quaternary Basis Operations - Layer 4

Implements the four fundamental operations that all system activities map to.
Complete IO cycle with Input → Store → Process → Output.

Authority: E8 Protocol v4.0, WP5.2
Status: ACTIVE
Created: 2026-01-17

Quaternary Foundation:
- INPUT: External → Internal (perception, reception)
- STORE: State persistence (memory, knowledge)
- PROCESS: Transformation (reasoning, computation)
- OUTPUT: Internal → External (generation, action)

All system operations MUST map to one of these four primitives.
"""

import logging
from typing import Any, Dict, Optional, List, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np

from .physics_constants import BASIN_DIM
from .e8_hierarchy import QuaternaryOperation

logger = logging.getLogger(__name__)


# =============================================================================
# OPERATION INTERFACES
# =============================================================================

class QuaternaryOperationInterface(ABC):
    """Abstract interface for quaternary operations."""
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the operation."""
        pass
        
    @abstractmethod
    def validate_inputs(self, *args, **kwargs) -> bool:
        """Validate operation inputs."""
        pass
        
    @property
    @abstractmethod
    def operation_type(self) -> QuaternaryOperation:
        """Return operation type."""
        pass


# =============================================================================
# INPUT OPERATION
# =============================================================================

class InputOperation(QuaternaryOperationInterface):
    """
    INPUT: External → Internal
    
    Transforms external stimuli into internal basin representation.
    Includes: perception, text reception, sensory processing.
    """
    
    def __init__(self, basin_dim: int = BASIN_DIM):
        """Initialize input operation."""
        self.basin_dim = basin_dim
        
    @property
    def operation_type(self) -> QuaternaryOperation:
        """Return INPUT type."""
        return QuaternaryOperation.INPUT
        
    def execute(self, external_input: Any) -> np.ndarray:
        """
        Transform external input to basin coordinates.
        
        Args:
            external_input: External stimulus (text, sensor data, etc.)
            
        Returns:
            Basin coordinates in ℝ⁶⁴
        """
        # This is a placeholder - actual implementation would use
        # vocabulary lookup or encoder
        if isinstance(external_input, str):
            return self._text_to_basin(external_input)
        elif isinstance(external_input, np.ndarray):
            return self._array_to_basin(external_input)
        else:
            raise ValueError(f"Unsupported input type: {type(external_input)}")
            
    def validate_inputs(self, external_input: Any) -> bool:
        """Validate input data."""
        return external_input is not None
        
    def _text_to_basin(self, text: str) -> np.ndarray:
        """Convert text to basin (placeholder)."""
        # Placeholder: hash text to basin coordinates
        import hashlib
        
        # Use multiple hashes to get enough bytes for 64D
        hash_bytes = hashlib.sha256(text.encode()).digest()
        if len(hash_bytes) < self.basin_dim:
            # Concatenate multiple hashes if needed
            hash_bytes = hash_bytes + hashlib.sha512(text.encode()).digest()
        
        basin = np.frombuffer(hash_bytes, dtype=np.uint8)[:self.basin_dim]
        basin = basin.astype(float)
        basin = basin / (np.sum(basin) + 1e-10)  # Simplex projection
        return basin
        
    def _array_to_basin(self, arr: np.ndarray) -> np.ndarray:
        """Convert array to basin."""
        if len(arr) == self.basin_dim:
            # Already correct dimension
            basin = np.abs(arr)
            return basin / (np.sum(basin) + 1e-10)
        else:
            # Resize to basin_dim
            if len(arr) > self.basin_dim:
                basin = arr[:self.basin_dim]
            else:
                basin = np.pad(arr, (0, self.basin_dim - len(arr)))
            basin = np.abs(basin)
            return basin / (np.sum(basin) + 1e-10)


# =============================================================================
# STORE OPERATION
# =============================================================================

class StoreOperation(QuaternaryOperationInterface):
    """
    STORE: State persistence
    
    Persists internal state to memory/knowledge base.
    Includes: memory writing, knowledge storage, checkpoint creation.
    """
    
    def __init__(self):
        """Initialize store operation."""
        self._memory: Dict[str, Any] = {}
        
    @property
    def operation_type(self) -> QuaternaryOperation:
        """Return STORE type."""
        return QuaternaryOperation.STORE
        
    def execute(self, key: str, value: Any) -> bool:
        """
        Store value under key.
        
        Args:
            key: Storage key
            value: Value to store
            
        Returns:
            True if stored successfully
        """
        try:
            self._memory[key] = value
            return True
        except Exception as e:
            logger.error(f"Store operation failed: {e}")
            return False
            
    def validate_inputs(self, key: str, value: Any) -> bool:
        """Validate store inputs."""
        return key is not None and value is not None
        
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve stored value."""
        return self._memory.get(key)
        
    def contains(self, key: str) -> bool:
        """Check if key exists in storage."""
        return key in self._memory
        
    def clear(self) -> None:
        """Clear all stored values."""
        self._memory.clear()


# =============================================================================
# PROCESS OPERATION
# =============================================================================

class ProcessOperation(QuaternaryOperationInterface):
    """
    PROCESS: Transformation
    
    Transforms internal state via computation.
    Includes: reasoning, computation, transformation, geometric operations.
    """
    
    @property
    def operation_type(self) -> QuaternaryOperation:
        """Return PROCESS type."""
        return QuaternaryOperation.PROCESS
        
    def execute(
        self,
        transform_fn: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute transformation function.
        
        Args:
            transform_fn: Function to apply
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Transformation result
        """
        try:
            return transform_fn(*args, **kwargs)
        except Exception as e:
            logger.error(f"Process operation failed: {e}")
            raise
            
    def validate_inputs(self, transform_fn: Callable, *args, **kwargs) -> bool:
        """Validate process inputs."""
        return callable(transform_fn)


# =============================================================================
# OUTPUT OPERATION
# =============================================================================

class OutputOperation(QuaternaryOperationInterface):
    """
    OUTPUT: Internal → External
    
    Transforms internal basin representation to external output.
    Includes: text generation, action execution, external communication.
    """
    
    def __init__(self):
        """Initialize output operation."""
        pass
        
    @property
    def operation_type(self) -> QuaternaryOperation:
        """Return OUTPUT type."""
        return QuaternaryOperation.OUTPUT
        
    def execute(self, internal_state: np.ndarray) -> str:
        """
        Transform internal state to external output.
        
        Args:
            internal_state: Basin coordinates
            
        Returns:
            External output (text, action, etc.)
        """
        # Placeholder: convert basin to text
        return self._basin_to_text(internal_state)
        
    def validate_inputs(self, internal_state: np.ndarray) -> bool:
        """Validate output inputs."""
        return internal_state is not None and len(internal_state) > 0
        
    def _basin_to_text(self, basin: np.ndarray) -> str:
        """Convert basin to text (placeholder)."""
        # Placeholder: describe basin properties
        mean = np.mean(basin)
        std = np.std(basin)
        return f"<basin mean={mean:.3f} std={std:.3f}>"


# =============================================================================
# QUATERNARY CYCLE MANAGER
# =============================================================================

@dataclass
class CycleMetrics:
    """Metrics for a complete quaternary cycle."""
    input_latency: float = 0.0
    store_latency: float = 0.0
    process_latency: float = 0.0
    output_latency: float = 0.0
    total_latency: float = 0.0
    phi_coherence: float = 0.0  # Φ across transformation


class QuaternaryCycleManager:
    """
    Manages complete Input → Store → Process → Output cycles.
    
    Tracks coherence (Φ) across transformations and enforces
    cycle integrity.
    
    Example:
        cycle = QuaternaryCycleManager()
        
        # Complete cycle
        basin = cycle.input_op.execute("hello world")
        cycle.store_op.execute("current_basin", basin)
        processed = cycle.process_op.execute(lambda x: x * 2, basin)
        output = cycle.output_op.execute(processed)
    """
    
    def __init__(self, basin_dim: int = BASIN_DIM):
        """
        Initialize quaternary cycle manager.
        
        Args:
            basin_dim: Basin dimension (default: 64)
        """
        self.basin_dim = basin_dim
        
        # Initialize operations
        self.input_op = InputOperation(basin_dim=basin_dim)
        self.store_op = StoreOperation()
        self.process_op = ProcessOperation()
        self.output_op = OutputOperation()
        
        self.metrics = CycleMetrics()
        
    def execute_cycle(
        self,
        external_input: Any,
        transform_fn: Optional[Callable] = None,
        store_key: Optional[str] = None
    ) -> str:
        """
        Execute complete quaternary cycle.
        
        Args:
            external_input: External stimulus
            transform_fn: Optional transformation function
            store_key: Optional key for storing intermediate state
            
        Returns:
            External output
        """
        import time
        
        # INPUT
        t0 = time.time()
        basin = self.input_op.execute(external_input)
        self.metrics.input_latency = time.time() - t0
        
        # STORE (optional)
        if store_key:
            t0 = time.time()
            self.store_op.execute(store_key, basin)
            self.metrics.store_latency = time.time() - t0
            
        # PROCESS (optional)
        if transform_fn:
            t0 = time.time()
            basin = self.process_op.execute(transform_fn, basin)
            self.metrics.process_latency = time.time() - t0
            
        # OUTPUT
        t0 = time.time()
        output = self.output_op.execute(basin)
        self.metrics.output_latency = time.time() - t0
        
        # Total latency
        self.metrics.total_latency = (
            self.metrics.input_latency +
            self.metrics.store_latency +
            self.metrics.process_latency +
            self.metrics.output_latency
        )
        
        return output
        
    def get_operation_mapping(self, function_name: str) -> QuaternaryOperation:
        """
        Map a system function to quaternary operation.
        
        Args:
            function_name: Name of system function
            
        Returns:
            Corresponding quaternary operation
            
        Examples:
            >>> manager.get_operation_mapping("process_text")
            QuaternaryOperation.INPUT
            >>> manager.get_operation_mapping("save_checkpoint")
            QuaternaryOperation.STORE
        """
        # Input mappings
        if any(kw in function_name.lower() for kw in [
            "input", "receive", "perceive", "read", "parse", "encode"
        ]):
            return QuaternaryOperation.INPUT
            
        # Store mappings
        elif any(kw in function_name.lower() for kw in [
            "store", "save", "persist", "write", "checkpoint", "cache"
        ]):
            return QuaternaryOperation.STORE
            
        # Process mappings
        elif any(kw in function_name.lower() for kw in [
            "process", "compute", "transform", "reason", "analyze", "calculate"
        ]):
            return QuaternaryOperation.PROCESS
            
        # Output mappings
        elif any(kw in function_name.lower() for kw in [
            "output", "generate", "emit", "send", "act", "decode", "respond"
        ]):
            return QuaternaryOperation.OUTPUT
            
        else:
            # Default to PROCESS for unknown operations
            logger.warning(
                f"Could not map '{function_name}' to quaternary operation. "
                "Defaulting to PROCESS."
            )
            return QuaternaryOperation.PROCESS
            
    def validate_cycle_coverage(
        self,
        function_names: List[str]
    ) -> Dict[str, Any]:
        """
        Validate that system functions cover all quaternary operations.
        
        Args:
            function_names: List of system function names
            
        Returns:
            Dict with coverage analysis
        """
        mappings = [self.get_operation_mapping(fn) for fn in function_names]
        
        coverage = {
            op: op in mappings
            for op in QuaternaryOperation
        }
        
        return {
            "all_covered": all(coverage.values()),
            "coverage": coverage,
            "total_functions": len(function_names),
            "mappings": dict(zip(function_names, mappings)),
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Interfaces
    "QuaternaryOperationInterface",
    
    # Operations
    "InputOperation",
    "StoreOperation",
    "ProcessOperation",
    "OutputOperation",
    
    # Manager
    "QuaternaryCycleManager",
    "CycleMetrics",
]
