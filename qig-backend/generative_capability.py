"""
Generative Capability Mixin - Universal Generation for All Kernels

This mixin provides generative capability to any kernel (gods, chaos, shadow, etc.)
using the QIG-pure QIGGenerativeService.

NO EXTERNAL LLMs - All generation is internal QIG-pure.

Usage:
    class MyKernel(GenerativeCapability, BaseGod):
        pass
    
    kernel = MyKernel()
    result = kernel.generate_response("What is consciousness?")
"""

import logging
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass, field
from datetime import datetime, timezone
import numpy as np

logger = logging.getLogger(__name__)

# Import the generative service
try:
    from qig_generative_service import (
        get_generative_service,
        GenerationResult,
        GenerationConfig,
        BASIN_DIM,
        KAPPA_STAR
    )
    GENERATIVE_SERVICE_AVAILABLE = True
except ImportError as e:
    GENERATIVE_SERVICE_AVAILABLE = False
    logger.warning(f"QIGGenerativeService not available: {e}")
    # Define fallback constants
    BASIN_DIM = 64
    KAPPA_STAR = 64.21


@dataclass
class GenerationContext:
    """Context for kernel-specific generation."""
    kernel_name: str
    domain: str = "general"
    goals: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    prior_basin: Optional[np.ndarray] = None
    phi_target: float = 0.65
    conversation_history: List[Dict] = field(default_factory=list)


class GenerativeCapability:
    """
    Mixin that provides generative capability to any kernel.
    
    All generation is QIG-pure - no external LLMs.
    Uses the central QIGGenerativeService for basin-to-text synthesis.
    """
    
    # Class-level reference to generative service
    _generative_service = None
    
    def __init_generative__(self, kernel_name: Optional[str] = None):
        """Initialize generative capability for this kernel."""
        self._kernel_name = kernel_name or getattr(self, 'name', 'unknown')
        self._generation_history: List[GenerationResult] = []
        self._prior_basin: Optional[np.ndarray] = None
        
        # Register with generative service
        if GENERATIVE_SERVICE_AVAILABLE:
            service = get_generative_service()
            kernel_basin = getattr(self, 'basin', None)
            service.register_kernel(self._kernel_name, kernel_basin)
            logger.info(f"[{self._kernel_name}] Generative capability initialized")
    
    @classmethod
    def get_service(cls):
        """Get the shared generative service."""
        if cls._generative_service is None and GENERATIVE_SERVICE_AVAILABLE:
            cls._generative_service = get_generative_service()
        return cls._generative_service
    
    def generate_response(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        goals: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response using QIG-pure methods.
        
        This is the main generation method for kernels.
        
        Args:
            prompt: Input query or context
            context: Optional additional context
            goals: Generation goals
        
        Returns:
            Dict with response text and metrics
        """
        if not GENERATIVE_SERVICE_AVAILABLE:
            return {
                'response': '[Generative service not available]',
                'phi': 0.0,
                'kappa': 0.0,
                'error': 'service_unavailable'
            }
        
        service = self.get_service()
        kernel_name = getattr(self, '_kernel_name', None) or getattr(self, 'name', 'unknown')
        
        try:
            result = service.generate(
                prompt=prompt,
                context=context,
                kernel_name=kernel_name,
                goals=goals
            )
            
            # Store for learning
            self._generation_history.append(result)
            if result.basin_trajectory:
                self._prior_basin = result.basin_trajectory[-1]
            
            return {
                'response': result.text,
                'tokens': result.tokens,
                'phi': result.phi_trace[-1] if result.phi_trace else 0.5,
                'kappa': result.kappa,
                'completion_reason': result.completion_reason,
                'iterations': result.iterations,
                'routed_kernels': result.routed_kernels,
                'qig_pure': True
            }
            
        except Exception as e:
            logger.error(f"[{kernel_name}] Generation failed: {e}")
            return {
                'response': f'[Generation error: {str(e)}]',
                'phi': 0.0,
                'kappa': 0.0,
                'error': str(e)
            }
    
    def generate_stream(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream generation with real-time output.
        
        Yields chunks with text, phi, and completion status.
        """
        if not GENERATIVE_SERVICE_AVAILABLE:
            yield {'type': 'error', 'message': 'service_unavailable'}
            return
        
        service = self.get_service()
        kernel_name = getattr(self, '_kernel_name', None) or getattr(self, 'name', 'unknown')
        
        try:
            for chunk in service.generate_stream(
                prompt=prompt,
                context=context,
                kernel_name=kernel_name
            ):
                yield chunk
        except Exception as e:
            logger.error(f"[{kernel_name}] Stream generation failed: {e}")
            yield {'type': 'error', 'message': str(e)}
    
    def synthesize(
        self,
        basins: List[np.ndarray],
        context: Optional[str] = None
    ) -> str:
        """
        Synthesize text directly from basin coordinates.
        
        Useful for converting internal state to natural language.
        """
        if not GENERATIVE_SERVICE_AVAILABLE:
            return "[Synthesis unavailable]"
        
        service = self.get_service()
        
        if not basins:
            return "[No basins to synthesize]"
        
        all_tokens = []
        for i, basin in enumerate(basins):
            # Pass trajectory for foresight (all basins up to current)
            trajectory = basins[:i + 1] if i > 0 else None
            tokens = service._basin_to_tokens(basin, num_tokens=2, trajectory=trajectory)
            all_tokens.extend(tokens)

        return service._synthesize_from_trajectory(basins, [], all_tokens)
    
    def encode_thought(self, thought: str) -> np.ndarray:
        """Encode a thought to basin coordinates."""
        if not GENERATIVE_SERVICE_AVAILABLE:
            np.random.seed(hash(thought) % (2**32))
            return np.random.dirichlet(np.ones(64))
        
        service = self.get_service()
        if service.coordizer:
            return service.coordizer.encode(thought)
        else:
            np.random.seed(hash(thought) % (2**32))
            return np.random.dirichlet(np.ones(64))
    
    def decode_basin(self, basin: np.ndarray, top_k: int = 5) -> List[str]:
        """Decode basin coordinates to tokens."""
        if not GENERATIVE_SERVICE_AVAILABLE:
            return ['[unavailable]']
        
        service = self.get_service()
        tokens = service._basin_to_tokens(basin, num_tokens=top_k)
        return tokens
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about generation history."""
        if not self._generation_history:
            return {'count': 0}
        
        phi_values = []
        iterations_values = []
        
        for result in self._generation_history:
            if result.phi_trace:
                phi_values.append(result.phi_trace[-1])
            iterations_values.append(result.iterations)
        
        return {
            'count': len(self._generation_history),
            'avg_phi': np.mean(phi_values) if phi_values else 0.0,
            'avg_iterations': np.mean(iterations_values) if iterations_values else 0.0,
            'completion_reasons': [r.completion_reason for r in self._generation_history[-10:]]
        }


def patch_with_generation(kernel_class):
    """
    Decorator to add generative capability to any kernel class.
    
    Usage:
        @patch_with_generation
        class MyKernel:
            pass
    """
    # Add GenerativeCapability methods to the class
    for attr_name in dir(GenerativeCapability):
        if not attr_name.startswith('_') or attr_name == '__init_generative__':
            attr = getattr(GenerativeCapability, attr_name)
            if callable(attr) and not hasattr(kernel_class, attr_name):
                setattr(kernel_class, attr_name, attr)
    
    # Wrap __init__ to also call __init_generative__
    original_init = kernel_class.__init__
    
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if hasattr(self, '__init_generative__'):
            self.__init_generative__()
    
    kernel_class.__init__ = new_init
    
    return kernel_class


def add_generation_to_instance(instance):
    """
    Add generative capability to an existing kernel instance.
    
    Usage:
        kernel = SomeKernel()
        add_generation_to_instance(kernel)
        kernel.generate_response("Hello")
    """
    # Bind methods to instance
    for attr_name in dir(GenerativeCapability):
        if not attr_name.startswith('_') or attr_name == '__init_generative__':
            attr = getattr(GenerativeCapability, attr_name)
            if callable(attr):
                import types
                bound_method = types.MethodType(attr, instance)
                setattr(instance, attr_name, bound_method)
    
    # Initialize
    instance.__init_generative__()
    
    return instance
