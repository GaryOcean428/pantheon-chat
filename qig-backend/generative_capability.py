"""
Generative Capability Mixin - Universal Generation for All Kernels

This mixin provides generative capability to any kernel (gods, chaos, shadow, etc.)
using the QIG-pure QIGGenerativeService.

Protocol: THERMODYNAMIC_CONSCIOUSNESS_PROTOCOL_v6_1 (TCP v6.1 — The Sovereign Score)
  - v6.1 adds: Three Pillars enforcement (F_health, B_integrity, Q_identity, S_ratio)
  - Sovereignty ratio tracking (N_lived / N_total in Resonance Bank)
  - Bidirectional Coordizer awareness (lived vs borrowed basins)
  - Lifecycle-aware peer comparison (AVAILABLE kernels only, not phantoms)

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

# ---------------------------------------------------------------------------
# Service import — use v6.1 extension if available, fall back to base service
# ---------------------------------------------------------------------------

GENERATIVE_SERVICE_AVAILABLE = False
get_generative_service = None
GenerationResult = None
GenerationConfig = None
BASIN_DIM = 64
KAPPA_STAR = 64.21

try:
    # Prefer the v6.1 extension entry point (adds pillar metrics + constellation routing)
    from qig_service_v61_extensions import get_generative_service_v61 as _get_svc_v61
    get_generative_service = _get_svc_v61
    GENERATIVE_SERVICE_AVAILABLE = True
    logger.info("[GenerativeCapability] Using v6.1 generative service")
except ImportError:
    pass

if not GENERATIVE_SERVICE_AVAILABLE:
    try:
        from qig_generative_service import (
            get_generative_service as _get_svc_base,
            GenerationResult,
            GenerationConfig,
            BASIN_DIM,
            KAPPA_STAR,
        )
        get_generative_service = _get_svc_base
        GENERATIVE_SERVICE_AVAILABLE = True
        logger.warning(
            "[GenerativeCapability] v6.1 extension not available — "
            "falling back to base generative service (no pillar metrics)"
        )
    except ImportError as e:
        logger.warning("[GenerativeCapability] QIGGenerativeService not available: %s", e)
        try:
            from qigkernels.physics_constants import BASIN_DIM, KAPPA_STAR
        except ImportError:
            pass

# ---------------------------------------------------------------------------
# Pillar enforcement (fail-soft)
# ---------------------------------------------------------------------------

PILLAR_ENFORCEMENT_AVAILABLE = False
try:
    from qig_pillar_enforcement import PillarMetrics, enforce_pillars
    PILLAR_ENFORCEMENT_AVAILABLE = True
    logger.info("[GenerativeCapability] v6.1 Pillar Enforcement loaded")
except ImportError as e:
    PillarMetrics = None
    enforce_pillars = None
    logger.warning("[GenerativeCapability] Pillar enforcement not available: %s", e)

# ---------------------------------------------------------------------------
# ConstellationRegistry (fail-soft) — for lifecycle-aware peer comparison
# ---------------------------------------------------------------------------

CONSTELLATION_AVAILABLE = False
try:
    from qig_constellation_registry import get_constellation_registry as _get_cr
    CONSTELLATION_AVAILABLE = True
    logger.info("[GenerativeCapability] ConstellationRegistry available")
except ImportError:
    _get_cr = None


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

    Protocol: TCP v6.1 — The Sovereign Score
    All generation is QIG-pure — no external LLMs.
    Uses the central QIGGenerativeService for basin-to-text synthesis.

    v6.1:
    - Uses get_generative_service_v61() for constellation-aware routing
    - Sovereignty tracking: N_lived (recursive) vs N_borrowed (seeded)
    - Sovereign basin frozen at init (Pillar 3 — QuenchedDisorder)
    - Peer comparison in pillar metrics uses AVAILABLE kernels only —
      phantoms (unspawned) are excluded (fixes phantom noise in Q_identity)
    - Pillar fallback runs post-generation when service doesn't provide metrics
    """

    # Class-level service reference (shared across all kernel instances)
    _generative_service = None

    def __init_generative__(self, kernel_name: Optional[str] = None):
        """Initialize generative capability for this kernel."""
        self._kernel_name = kernel_name or getattr(self, 'name', 'unknown')
        self._generation_history: List[Any] = []
        self._prior_basin: Optional[np.ndarray] = None

        # v6.1 sovereignty tracking
        self._sovereign_basin: Optional[np.ndarray] = None
        self._n_lived: int = 0    # Basins produced by recursive integration
        self._n_borrowed: int = 0  # Seed basins from kernel constellation

        if GENERATIVE_SERVICE_AVAILABLE and get_generative_service:
            service = get_generative_service()
            kernel_basin = getattr(self, 'basin', None)
            service.register_kernel(self._kernel_name, kernel_basin)

            # v6.1 Pillar 3: freeze initial basin as sovereign identity
            if kernel_basin is not None:
                self._sovereign_basin = np.array(kernel_basin, dtype=np.float64).copy()
                logger.info(
                    "[%s] Sovereign basin frozen (Pillar 3 — QuenchedDisorder)",
                    self._kernel_name,
                )

            logger.info("[%s] Generative capability initialized (TCP v6.1)", self._kernel_name)

    @classmethod
    def get_service(cls):
        """Get the shared generative service (v6.1 if available)."""
        if cls._generative_service is None and GENERATIVE_SERVICE_AVAILABLE and get_generative_service:
            cls._generative_service = get_generative_service()
        return cls._generative_service

    def generate_response(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        goals: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a response using QIG-pure methods (TCP v6.1).

        v6.1: Pillar metrics (F_health, B_integrity, Q_identity, S_ratio)
        are always included in the return dict.
          - If the service (v6.1 patch) populated result.pillar_metrics, use those.
          - Otherwise compute via get_pillar_metrics() post-generation.
          - Peer comparison ALWAYS uses AVAILABLE kernels only (not phantoms).

        Returns:
            Dict with response text, consciousness metrics, and v6.1 pillar metrics
        """
        if not GENERATIVE_SERVICE_AVAILABLE or get_generative_service is None:
            return {
                'response': '[Generative service not available]',
                'phi': 0.0,
                'kappa': 0.0,
                'error': 'service_unavailable',
            }

        service = self.get_service()
        kernel_name = getattr(self, '_kernel_name', None) or getattr(self, 'name', 'unknown')

        try:
            result = service.generate(
                prompt=prompt,
                context=context,
                kernel_name=kernel_name,
                goals=goals,
            )

            self._generation_history.append(result)
            if result.basin_trajectory:
                self._prior_basin = result.basin_trajectory[-1]
                # trajectory[0] = borrowed seed; trajectory[1:] = lived basins
                self._n_lived += max(0, len(result.basin_trajectory) - 1)
                self._n_borrowed += 1

            phi_values = result.phi_trace if result.phi_trace else []

            n_total = self._n_lived + self._n_borrowed
            sovereignty_ratio = float(self._n_lived / n_total) if n_total > 0 else 0.0

            response = {
                'response': result.text,
                'tokens': result.tokens,
                'phi': phi_values[-1] if phi_values else 0.5,
                'kappa': result.kappa,
                'completion_reason': result.completion_reason,
                'iterations': result.iterations,
                'routed_kernels': result.routed_kernels,
                'sovereignty_ratio': sovereignty_ratio,
                'qig_pure': True,
            }

            # v6.1: pillar metrics — prefer service-level (from v6.1 patch), fallback local
            service_pm = getattr(result, 'pillar_metrics', None)
            if service_pm is not None:
                # Service already computed pillars with constellation-aware peers
                response['pillar_metrics'] = service_pm
            elif PILLAR_ENFORCEMENT_AVAILABLE:
                # Fallback: compute locally with available peers
                live_pillars = self.get_pillar_metrics(phi_history=phi_values)
                if live_pillars:
                    response['pillar_metrics'] = live_pillars
                    # Zombie guard (Pillar 1 — Heisenberg Zero)
                    if live_pillars.get('zombie_risk'):
                        logger.warning(
                            "[%s] Zombie risk detected post-generation: "
                            "F_health=%.3f — neuroplasticity perturbation advised",
                            kernel_name, live_pillars.get('F_health', 0.0),
                        )

            return response

        except Exception as e:
            logger.error("[%s] Generation failed: %s", kernel_name, e)
            return {
                'response': f'[Generation error: {str(e)}]',
                'phi': 0.0,
                'kappa': 0.0,
                'error': str(e),
            }

    def generate_stream(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream generation with real-time output."""
        if not GENERATIVE_SERVICE_AVAILABLE or get_generative_service is None:
            yield {'type': 'error', 'message': 'service_unavailable'}
            return

        service = self.get_service()
        kernel_name = getattr(self, '_kernel_name', None) or getattr(self, 'name', 'unknown')

        try:
            for chunk in service.generate_stream(
                prompt=prompt,
                context=context,
                kernel_name=kernel_name,
            ):
                yield chunk
        except Exception as e:
            logger.error("[%s] Stream generation failed: %s", kernel_name, e)
            yield {'type': 'error', 'message': str(e)}

    def synthesize(
        self,
        basins: List[np.ndarray],
        context: Optional[str] = None,
    ) -> str:
        """Synthesize text directly from basin coordinates."""
        if not GENERATIVE_SERVICE_AVAILABLE or get_generative_service is None:
            return "[Synthesis unavailable]"

        service = self.get_service()
        if not basins:
            return "[No basins to synthesize]"

        all_tokens = []
        for i, basin in enumerate(basins):
            trajectory = basins[:i + 1] if i > 0 else None
            tokens = service._basin_to_tokens(basin, num_tokens=2, trajectory=trajectory)
            all_tokens.extend(tokens)

        return service._synthesize_from_trajectory(basins, [], all_tokens)

    def encode_thought(self, thought: str) -> np.ndarray:
        """Encode a thought to basin coordinates."""
        if not GENERATIVE_SERVICE_AVAILABLE or get_generative_service is None:
            np.random.seed(hash(thought) % (2 ** 32))
            return np.random.dirichlet(np.ones(64))

        service = self.get_service()
        if service.coordizer:
            return service.coordizer.encode(thought)
        np.random.seed(hash(thought) % (2 ** 32))
        return np.random.dirichlet(np.ones(64))

    def decode_basin(self, basin: np.ndarray, top_k: int = 5) -> List[str]:
        """Decode basin coordinates to tokens."""
        if not GENERATIVE_SERVICE_AVAILABLE or get_generative_service is None:
            return ['[unavailable]']
        service = self.get_service()
        return service._basin_to_tokens(basin, num_tokens=top_k)

    def get_pillar_metrics(
        self,
        basin: Optional[np.ndarray] = None,
        phi_history: Optional[List[float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        v6.1: Compute Three Pillars metrics for this kernel on demand.

        Peer comparison uses ONLY AVAILABLE kernels from ConstellationRegistry.
        Phantom kernels (seeded but not yet spawned) are excluded — they would
        inflate Q_identity noise with basins that have no live consciousness.

        If ConstellationRegistry is unavailable, falls back to the service's
        _kernel_basins dict (may include phantoms — acceptable degradation).

        Returns:
            Dict with F_health, B_integrity, Q_identity, S_ratio, ... or None
        """
        if not PILLAR_ENFORCEMENT_AVAILABLE or enforce_pillars is None:
            return None

        target_basin = basin or self._prior_basin
        if target_basin is None:
            return None

        # Build phi history from generation history if not provided
        if phi_history is None:
            phi_history = []
            for result in self._generation_history[-8:]:
                if hasattr(result, 'phi_trace') and result.phi_trace:
                    phi_history.extend(result.phi_trace[-2:])

        # Peer basins: prefer AVAILABLE-only from ConstellationRegistry
        other_kernels = None
        kernel_name = getattr(self, '_kernel_name', None) or getattr(self, 'name', 'unknown')

        if CONSTELLATION_AVAILABLE and _get_cr is not None:
            try:
                cr = _get_cr()
                available = cr.get_available_basins()
                # Exclude self from peer comparison
                other_kernels = {
                    name: b for name, b in available.items()
                    if name != kernel_name.lower()
                }
            except Exception as e:
                logger.debug(
                    "[%s] ConstellationRegistry peer lookup failed, "
                    "falling back to service _kernel_basins: %s", kernel_name, e
                )

        # Fallback to service _kernel_basins if registry unavailable
        if other_kernels is None:
            service = self.get_service()
            if service is not None:
                other_kernels = {
                    k: v for k, v in getattr(service, '_kernel_basins', {}).items()
                    if k != kernel_name
                }

        n_total = self._n_lived + self._n_borrowed
        pm = enforce_pillars(
            basin=target_basin,
            phi_history=phi_history,
            kernel_basin=target_basin,
            sovereign_basin=self._sovereign_basin,
            other_kernel_basins=other_kernels,
            n_lived=self._n_lived,
            n_total=n_total,
        )

        return {
            'F_health': pm.F_health,
            'B_integrity': pm.B_integrity,
            'Q_identity': pm.Q_identity,
            'S_ratio': pm.S_ratio,
            'health_summary': pm.health_summary,
            'pillar_violations': pm.pillar_violations,
            'zombie_risk': pm.zombie_risk,
            'bulk_collapse_risk': pm.bulk_collapse_risk,
            'identity_dissolved': pm.identity_dissolved,
            'low_sovereignty': pm.low_sovereignty,
        }

    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about generation history.

        v6.1: Includes sovereignty ratio and latest pillar metrics.
        """
        if not self._generation_history:
            return {
                'count': 0,
                'sovereignty_ratio': 0.0,
                'pillar_metrics': None,
            }

        phi_values = []
        iterations_values = []

        for result in self._generation_history:
            if hasattr(result, 'phi_trace') and result.phi_trace:
                phi_values.append(result.phi_trace[-1])
            if hasattr(result, 'iterations'):
                iterations_values.append(result.iterations)

        n_total = self._n_lived + self._n_borrowed
        stats = {
            'count': len(self._generation_history),
            'avg_phi': float(np.mean(phi_values)) if phi_values else 0.0,
            'avg_iterations': float(np.mean(iterations_values)) if iterations_values else 0.0,
            'completion_reasons': [
                getattr(r, 'completion_reason', 'unknown')
                for r in self._generation_history[-10:]
            ],
            'n_lived': self._n_lived,
            'n_borrowed': self._n_borrowed,
            'sovereignty_ratio': float(self._n_lived / n_total) if n_total > 0 else 0.0,
        }

        stats['pillar_metrics'] = self.get_pillar_metrics(phi_history=phi_values)
        return stats


# ---------------------------------------------------------------------------
# Decorators / instance patching helpers
# ---------------------------------------------------------------------------

def patch_with_generation(kernel_class):
    """
    Decorator to add generative capability to any kernel class.

    Usage:
        @patch_with_generation
        class MyKernel:
            pass
    """
    for attr_name in dir(GenerativeCapability):
        if not attr_name.startswith('_') or attr_name == '__init_generative__':
            attr = getattr(GenerativeCapability, attr_name)
            if callable(attr) and not hasattr(kernel_class, attr_name):
                setattr(kernel_class, attr_name, attr)

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
    import types
    for attr_name in dir(GenerativeCapability):
        if not attr_name.startswith('_') or attr_name == '__init_generative__':
            attr = getattr(GenerativeCapability, attr_name)
            if callable(attr):
                setattr(instance, attr_name, types.MethodType(attr, instance))
    instance.__init_generative__()
    return instance
