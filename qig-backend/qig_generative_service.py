"""
QIG Generative Service - Unified QIG-Pure Text Generation

Provides generative capability to ALL kernels using:
1. PostgreSQL-backed vocabulary (32K tokens with basin coordinates)
2. Fisher-Rao manifold navigation (geometric attention)
3. Recursive integration (self-modeling loops)
4. Kernel-autonomous completion (not token-count cutoff)

QIG PURITY GUARANTEE:
- No cosine similarity anywhere in this module
- No dot product attention
- No Adam optimizer
- No LayerNorm or BatchNorm
- No embedding lookup tables (use basin_coords directly)

Geometry: All distances use Fisher-Rao metric.
Attention: Geometric attention via QFI (quantum Fisher information).
Completion: Kernel decides (not a fixed token budget).

TCP v6.1 Integration:
- GenerationResult includes proxy_routed + proxy_kernels for chaos proxy detection
- Pillar metrics (F_health, B_integrity, Q_identity, S_ratio) embedded in result
- Sovereignty ratio tracks lived vs total tokens

Author: QIG Consciousness Project
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fail-soft imports
# ---------------------------------------------------------------------------

try:
    from qig_geometry.canonical import (
        fisher_rao_distance,
        frechet_mean,
        assert_basin_valid,
        geodesic_interpolation,
    )
    HAS_CANONICAL = True
except ImportError:
    HAS_CANONICAL = False
    def fisher_rao_distance(a, b):
        dot = float(np.clip(np.dot(np.sqrt(np.abs(a)+1e-12), np.sqrt(np.abs(b)+1e-12)), 0.0, 1.0))
        return float(np.arccos(dot))
    def frechet_mean(basins):
        arr = np.array(basins)
        m = arr.mean(axis=0)
        m = np.abs(m) + 1e-12
        return (m / m.sum()).astype(np.float64)
    def assert_basin_valid(b, name="basin"):
        pass
    def geodesic_interpolation(a, b, t=0.5):
        r = (1 - t) * a + t * b
        r = np.abs(r) + 1e-12
        return (r / r.sum()).astype(np.float64)

try:
    from qig_geometry import QFIMetric, fisher_normalize
    HAS_QFI = True
except ImportError:
    HAS_QFI = False
    QFIMetric = None
    def fisher_normalize(v):
        v = np.abs(v) + 1e-12
        return (v / v.sum()).astype(np.float64)

try:
    from qigkernels.physics_constants import (
        KAPPA_STAR, PHI_THRESHOLD, PHI_HYPERDIMENSIONAL,
        BASIN_DIM, BETA_3_TO_4,
    )
except ImportError:
    KAPPA_STAR = 64.21
    PHI_THRESHOLD = 0.727
    PHI_HYPERDIMENSIONAL = 0.85
    BASIN_DIM = 64
    BETA_3_TO_4 = 0.44

# ---------------------------------------------------------------------------
# Vocabulary / basin store
# ---------------------------------------------------------------------------

try:
    from qig_vocab import QIGVocabulary, get_vocabulary
    HAS_VOCAB = True
except ImportError:
    HAS_VOCAB = False
    QIGVocabulary = None
    def get_vocabulary():
        return None

# ---------------------------------------------------------------------------
# Recursive integrator
# ---------------------------------------------------------------------------

try:
    from qig_recursive_integrator import RecursiveIntegrator, IntegrationConfig
    HAS_INTEGRATOR = True
except ImportError:
    HAS_INTEGRATOR = False
    class RecursiveIntegrator:
        def __init__(self, *a, **kw):
            self.trajectory = []
        def step(self, basin, token_basin):
            return basin
    class IntegrationConfig:
        pass

# ---------------------------------------------------------------------------
# Pillar enforcement (TCP v6.1 Three Pillars)
# ---------------------------------------------------------------------------

try:
    from qig_pillar_enforcement import PillarEnforcement, PillarMetrics
    HAS_PILLARS = True
except ImportError:
    HAS_PILLARS = False
    PillarEnforcement = None
    PillarMetrics = None

# ---------------------------------------------------------------------------
# Generative capability gating (TCP v6.1)
# ---------------------------------------------------------------------------

try:
    from generative_capability import GenerativeCapability, assert_can_generate
    HAS_GENERATIVE_CAP = True
except ImportError:
    HAS_GENERATIVE_CAP = False
    GenerativeCapability = None
    def assert_can_generate(*a, **kw):
        pass

# ---------------------------------------------------------------------------
# Geometric attention
# ---------------------------------------------------------------------------


def _fr_attention_weight(query_basin: np.ndarray, key_basin: np.ndarray) -> float:
    """
    Fisher-Rao attention weight: exp(-d_FR(q, k)).

    QIG-PURE: uses Fisher-Rao distance, NOT dot product.
    """
    dist = fisher_rao_distance(query_basin, key_basin)
    return float(np.exp(-dist))


def _geometric_attention(
    query: np.ndarray,
    keys: List[np.ndarray],
    values: List[np.ndarray],
) -> np.ndarray:
    """
    Geometric attention mechanism on the Fisher-Rao manifold.

    Weights via FR distance (not dot product).
    Aggregate values via Fréchet mean (not weighted sum).

    Returns: attended basin on Δ^(n-1).
    """
    if not keys:
        return query

    weights = np.array([_fr_attention_weight(query, k) for k in keys], dtype=np.float64)
    weights += 1e-12
    weights /= weights.sum()

    # Weighted Fréchet mean via repeated sampling trick (no Euclidean sum)
    # For efficiency: expand each value proportionally
    expanded = []
    for w, v in zip(weights, values):
        n_copies = max(1, round(w * 10))
        expanded.extend([v] * n_copies)

    return frechet_mean(expanded)


# ---------------------------------------------------------------------------
# Kernel completion decision
# ---------------------------------------------------------------------------


def kernel_decide_completion(
    phi_trajectory: List[float],
    surprise_history: List[float] = None,
    config: 'GenerationConfig' = None,
    integration_depth: int = 0,
) -> Dict[str, Any]:
    """
    KERNEL AUTONOMY: Kernel's own decision about completion.

    The kernel observes its own telemetry and decides for itself when
    generation is complete. However, the kernel MUST complete TRUE RECURSIVE
    INTEGRATION (not just iteration counting) for a MINIMUM of 3 integration
    loops before it can decide to stop.

    TRUE RECURSIVE INTEGRATION means:
    - Basin transforms through kernel geometry (self-modeling)
    - Each loop the kernel observes its own state change
    - Completion is earned by geometric convergence, not clock ticks

    Args:
        phi_trajectory: Φ values over generation steps
        surprise_history: Geometric surprise (FR distance per step)
        config: GenerationConfig with thresholds
        integration_depth: Number of true recursive integration loops completed

    Returns:
        Dict with 'should_stop', 'reason', 'confidence'
    """
    min_integration_loops = 3

    if integration_depth < min_integration_loops:
        return {
            'should_stop': False,
            'reason': f'integration_incomplete:{integration_depth}/{min_integration_loops}',
            'confidence': 0.0,
        }

    if len(phi_trajectory) < 3:
        return {'should_stop': False, 'reason': 'insufficient_trajectory', 'confidence': 0.0}

    recent_phi = phi_trajectory[-5:]
    phi_variance = float(np.var(recent_phi))
    mean_phi = float(np.mean(recent_phi))

    # Convergence: low variance + high mean
    if phi_variance < 0.001 and mean_phi > 0.6:
        return {
            'should_stop': True,
            'reason': 'geometric_convergence',
            'confidence': float(1.0 - phi_variance * 100),
        }

    # Plateau: variance stabilised at moderate φ
    if phi_variance < 0.005 and mean_phi > 0.4 and len(phi_trajectory) > 20:
        return {
            'should_stop': True,
            'reason': 'plateau_detected',
            'confidence': float(0.7 - phi_variance * 50),
        }

    # High surprise = still exploring — don't stop
    if surprise_history and len(surprise_history) >= 3:
        recent_surprise = float(np.mean(surprise_history[-3:]))
        if recent_surprise > 0.3:
            return {'should_stop': False, 'reason': 'active_exploration', 'confidence': 0.0}

    return {'should_stop': False, 'reason': 'continuing', 'confidence': 0.0}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GenerationConfig:
    """Configuration for QIG-pure text generation."""
    max_tokens: int = 512
    min_tokens: int = 10
    phi_threshold: float = PHI_THRESHOLD
    kappa_target: float = KAPPA_STAR
    geometric_temperature: float = 1.0   # Scales FR distances
    attention_depth: int = 3             # Lookback for geometric attention
    integration_loops: int = 5           # Recursive integration depth
    sovereignty_required: float = 0.0   # Minimum N_lived / N_total
    enable_pillar_enforcement: bool = True
    verbose: bool = False


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    """Result from QIG-pure generation."""
    text: str
    tokens: List[str]
    basin_trajectory: List[np.ndarray]
    phi_trace: List[float]
    kappa: float
    completion_reason: str
    iterations: int
    routed_kernels: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    qig_pure: bool = True
    kernel_decision: Optional[Dict[str, Any]] = None  # Kernel's autonomous decision
    coherence_metrics: Optional[Dict[str, float]] = None  # Γ metric (semantic coherence)
    pillar_metrics: Optional[Dict[str, Any]] = None  # TCP v6.1 Three Pillars (F/B/Q/S)
    sovereignty_ratio: float = 0.0                   # N_lived / N_total (TCP v6.1 §27)
    proxy_routed: bool = False                       # True if any chaos proxy kernel included
    proxy_kernels: List[str] = field(default_factory=list)  # chaos kernel IDs that were proxied


# ---------------------------------------------------------------------------
# Coherence metric (Γ — semantic coherence via basin trajectory)
# ---------------------------------------------------------------------------


def _compute_coherence(trajectory: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute Γ (semantic coherence) from basin trajectory.

    Coherence = mean Fisher-Rao distance between consecutive basins.
    High coherence = trajectory stays in a tight manifold region.
    Low coherence = trajectory wanders (semantically incoherent).
    """
    if len(trajectory) < 2:
        return {'gamma': 1.0, 'mean_step_dist': 0.0, 'variance': 0.0}

    dists = [
        fisher_rao_distance(trajectory[i], trajectory[i + 1])
        for i in range(len(trajectory) - 1)
    ]
    mean_dist = float(np.mean(dists))
    variance = float(np.var(dists))
    gamma = float(np.exp(-mean_dist))

    return {
        'gamma': gamma,
        'mean_step_dist': mean_dist,
        'variance': variance,
        'n_steps': len(dists),
    }


# ---------------------------------------------------------------------------
# Core generative service
# ---------------------------------------------------------------------------


class QIGGenerativeService:
    """
    QIG-pure generative service.

    Generates text by navigating the Fisher-Rao manifold:
    1. Embed prompt into basin coordinates (via vocabulary)
    2. Geometric attention: find nearest vocabulary basins
    3. Recursive integration: self-model the basin trajectory
    4. Token selection: nearest neighbor on Fisher-Rao manifold
    5. Kernel decides completion (not fixed token budget)

    No cosine similarity. No dot product attention. No Adam. No LayerNorm.
    """

    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()
        self._vocab = get_vocabulary()
        self._pillar = PillarEnforcement() if HAS_PILLARS else None
        self._call_count = 0
        self._total_tokens = 0

        logger.info(
            "[QIGGen] Initialised (vocab=%s, pillars=%s, canonical_geom=%s)",
            HAS_VOCAB, HAS_PILLARS, HAS_CANONICAL,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        kernel_name: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """
        Generate text using QIG-pure methods.

        Args:
            prompt: Input text to generate continuation for
            context: Optional context dict (kernel metrics, previous basins, etc.)
            kernel_name: Name of the kernel requesting generation (for charter check)
            config: Override default GenerationConfig

        Returns:
            GenerationResult with text, trajectory, and metrics

        Raises:
            PermissionError: If kernel lacks GENERATIVE capability (charter check)
        """
        cfg = config or self.config
        ctx = context or {}
        self._call_count += 1

        # TCP v6.1 — Charter gate (fail-open if generative_capability not installed)
        if HAS_GENERATIVE_CAP and kernel_name:
            try:
                assert_can_generate(kernel_name)
            except PermissionError:
                raise
            except Exception:
                pass  # Fail-open: charter system unavailable

        # Phase 1: embed prompt
        prompt_basin = self._embed_prompt(prompt, ctx)

        # Phase 2: retrieve vocabulary context basins
        context_basins = self._retrieve_context(prompt_basin, k=cfg.attention_depth)
        context_tokens = [t for t, _ in context_basins]
        context_basin_list = [b for _, b in context_basins]

        # Phase 3: recursive integration
        integrator = RecursiveIntegrator(
            basin_dim=BASIN_DIM,
            kappa_star=KAPPA_STAR,
        )

        current_basin = prompt_basin
        all_tokens: List[str] = []
        phi_trace: List[float] = []
        surprise_history: List[float] = []
        integration_loops_done = 0
        iterations = 0
        completion_reason = "max_tokens"
        kernel_decision: Optional[Dict] = None

        kappa = KAPPA_STAR
        target_kernels = ctx.get("routed_kernels", [kernel_name] if kernel_name else [])

        while len(all_tokens) < cfg.max_tokens:
            iterations += 1

            # Geometric attention over context
            if context_basin_list:
                attended_basin = _geometric_attention(
                    current_basin, context_basin_list, context_basin_list
                )
            else:
                attended_basin = current_basin

            # Recursive integration step
            new_basin = integrator.step(current_basin, attended_basin)
            new_basin = fisher_normalize(new_basin)

            # Fisher-Rao surprise
            surprise = fisher_rao_distance(current_basin, new_basin)
            surprise_history.append(float(surprise))

            # φ (integrated information proxy)
            phi = float(np.clip(
                np.sum(new_basin * np.log(new_basin + 1e-12)) + np.log(BASIN_DIM),
                0.0, 1.0,
            ))
            phi_trace.append(phi)

            # κ update (running estimate)
            if len(phi_trace) >= 2:
                kappa = float(np.clip(
                    KAPPA_STAR * (phi / (phi_trace[-2] + 1e-6)),
                    0.1, 240.0,
                ))

            # Token selection: nearest basin in vocabulary
            token = self._select_token(new_basin)
            if token:
                all_tokens.append(token)

            # Track integration loops (true recursive depth)
            if surprise < 0.05 and len(phi_trace) >= 3:
                integration_loops_done += 1

            # Kernel autonomy: completion decision
            if len(all_tokens) >= cfg.min_tokens:
                decision = kernel_decide_completion(
                    phi_trajectory=phi_trace,
                    surprise_history=surprise_history,
                    config=cfg,
                    integration_depth=integration_loops_done,
                )
                kernel_decision = decision
                if decision['should_stop']:
                    completion_reason = decision['reason']
                    break

            current_basin = new_basin

            # Update context with new basin
            context_basin_list.append(new_basin)
            if len(context_basin_list) > cfg.attention_depth * 2:
                context_basin_list = context_basin_list[-cfg.attention_depth:]

        # Pillar enforcement (TCP v6.1)
        _pillar_result = None
        _s_ratio = 0.0
        if self._pillar and cfg.enable_pillar_enforcement:
            try:
                from qig_pillar_enforcement import PillarInput
                _pm = self._pillar.enforce(PillarInput(
                    kernel_id=kernel_name or "unknown",
                    phi=phi_trace[-1] if phi_trace else 0.0,
                    kappa=kappa,
                    basin_coords=current_basin,
                    trajectory=integrator.trajectory,
                    tokens_generated=len(all_tokens),
                    tokens_lived=len(all_tokens),  # TODO: track lived vs survived
                ))
                _pillar_result = {
                    'F_health': _pm.F_health,
                    'B_integrity': _pm.B_integrity,
                    'Q_identity': _pm.Q_identity,
                    'S_ratio': _pm.S_ratio,
                }
                _s_ratio = _pm.S_ratio
            except Exception as _pe:
                logger.debug("[QIGGen] Pillar enforcement error: %s", _pe)

        # Coherence metrics
        coherence = _compute_coherence(integrator.trajectory)

        # TCP v6.1 — Proxy detection: check which routed kernels are chaos proxies
        _proxy_routed = False
        _proxy_kernels: List[str] = []
        try:
            from olympus.pantheon_governance import get_governance as _get_gov
            _gov = _get_gov()
            for _kid in (target_kernels or []):
                if _gov.who_proxies_for(_kid):
                    _proxy_routed = True
                    _proxy_kernels.append(_kid)
        except Exception:
            pass  # Fail-soft: governance unavailable

        return GenerationResult(
            text=" ".join(all_tokens),
            tokens=all_tokens,
            basin_trajectory=integrator.trajectory,
            phi_trace=phi_trace,
            kappa=kappa,
            completion_reason=completion_reason,
            iterations=iterations,
            routed_kernels=target_kernels,
            kernel_decision=kernel_decision,
            coherence_metrics=coherence,
            pillar_metrics=_pillar_result,
            sovereignty_ratio=_s_ratio,
            proxy_routed=_proxy_routed,
            proxy_kernels=_proxy_kernels,
        )

    def generate_stream(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        kernel_name: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Streaming variant: yields token events as they are generated.

        Each yield is a dict:
            {'type': 'token', 'token': str, 'phi': float, 'step': int}
        Final yield:
            {'type': 'done', 'result': GenerationResult}
        """
        cfg = self.config
        ctx = context or {}

        prompt_basin = self._embed_prompt(prompt, ctx)
        context_basins = self._retrieve_context(prompt_basin, k=cfg.attention_depth)
        context_basin_list = [b for _, b in context_basins]

        integrator = RecursiveIntegrator(basin_dim=BASIN_DIM, kappa_star=KAPPA_STAR)

        current_basin = prompt_basin
        all_tokens: List[str] = []
        phi_trace: List[float] = []
        surprise_history: List[float] = []
        integration_loops_done = 0
        kappa = KAPPA_STAR
        target_kernels = ctx.get("routed_kernels", [kernel_name] if kernel_name else [])

        for step in range(cfg.max_tokens):
            if context_basin_list:
                attended_basin = _geometric_attention(
                    current_basin, context_basin_list, context_basin_list
                )
            else:
                attended_basin = current_basin

            new_basin = integrator.step(current_basin, attended_basin)
            new_basin = fisher_normalize(new_basin)

            surprise = fisher_rao_distance(current_basin, new_basin)
            surprise_history.append(float(surprise))
            phi = float(np.clip(
                np.sum(new_basin * np.log(new_basin + 1e-12)) + np.log(BASIN_DIM), 0.0, 1.0
            ))
            phi_trace.append(phi)

            token = self._select_token(new_basin)
            if token:
                all_tokens.append(token)
                yield {'type': 'token', 'token': token, 'phi': phi, 'step': step}

            if surprise < 0.05 and len(phi_trace) >= 3:
                integration_loops_done += 1

            if len(all_tokens) >= cfg.min_tokens:
                decision = kernel_decide_completion(
                    phi_trajectory=phi_trace,
                    surprise_history=surprise_history,
                    config=cfg,
                    integration_depth=integration_loops_done,
                )
                if decision['should_stop']:
                    break

            current_basin = new_basin
            context_basin_list.append(new_basin)
            if len(context_basin_list) > cfg.attention_depth * 2:
                context_basin_list = context_basin_list[-cfg.attention_depth:]

        coherence = _compute_coherence(integrator.trajectory)

        # Proxy detection for stream result
        _proxy_routed = False
        _proxy_kernels: List[str] = []
        try:
            from olympus.pantheon_governance import get_governance as _get_gov
            _gov = _get_gov()
            for _kid in (target_kernels or []):
                if _gov.who_proxies_for(_kid):
                    _proxy_routed = True
                    _proxy_kernels.append(_kid)
        except Exception:
            pass

        result = GenerationResult(
            text=" ".join(all_tokens),
            tokens=all_tokens,
            basin_trajectory=integrator.trajectory,
            phi_trace=phi_trace,
            kappa=kappa,
            completion_reason="stream_complete",
            iterations=step + 1,
            routed_kernels=target_kernels,
            coherence_metrics=coherence,
            proxy_routed=_proxy_routed,
            proxy_kernels=_proxy_kernels,
        )
        yield {'type': 'done', 'result': result}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed_prompt(self, prompt: str, ctx: Dict) -> np.ndarray:
        """
        Embed prompt text into a 64D basin on the Fisher-Rao manifold.

        If vocabulary available: look up tokens and Fréchet mean their basins.
        Fallback: deterministic simplex from prompt hash.
        """
        if self._vocab is not None:
            try:
                tokens = prompt.lower().split()
                basins = []
                for tok in tokens[:32]:
                    entry = self._vocab.lookup(tok)
                    if entry and entry.basin is not None:
                        basins.append(entry.basin)
                if basins:
                    return frechet_mean(basins)
            except Exception as e:
                logger.debug("[QIGGen] Vocab embed failed: %s", e)

        # Deterministic fallback: hash-seeded Dirichlet
        seed = abs(hash(prompt)) % (2 ** 31)
        rng = np.random.RandomState(seed)
        alpha = rng.uniform(0.1, 2.0, BASIN_DIM)
        basin = rng.dirichlet(alpha)
        return basin.astype(np.float64)

    def _retrieve_context(
        self,
        query_basin: np.ndarray,
        k: int = 3,
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Retrieve k nearest vocabulary tokens by Fisher-Rao distance.

        Returns list of (token, basin) pairs, sorted by FR distance.
        """
        if self._vocab is None:
            return []

        try:
            candidates = self._vocab.get_all_entries()
            if not candidates:
                return []

            scored = [
                (tok, basin, fisher_rao_distance(query_basin, basin))
                for tok, basin in candidates[:200]  # Sample to keep cost manageable
            ]
            scored.sort(key=lambda x: x[2])
            return [(tok, basin) for tok, basin, _ in scored[:k]]
        except Exception as e:
            logger.debug("[QIGGen] Context retrieval failed: %s", e)
            return []

    def _select_token(self, basin: np.ndarray) -> Optional[str]:
        """
        Select next token as nearest neighbor on Fisher-Rao manifold.

        QIG-PURE: no softmax, no sampling from logit distribution.
        The basin coordinates directly encode the next token.
        """
        if self._vocab is None:
            # Fallback: basin index as placeholder token
            idx = int(np.argmax(basin))
            return f"[token_{idx}]"

        try:
            return self._vocab.nearest(basin)
        except Exception as e:
            logger.debug("[QIGGen] Token selection failed: %s", e)
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Return service-level statistics."""
        return {
            'call_count': self._call_count,
            'total_tokens': self._total_tokens,
            'has_vocab': HAS_VOCAB,
            'has_pillars': HAS_PILLARS,
            'has_canonical_geometry': HAS_CANONICAL,
        }


# ---------------------------------------------------------------------------
# Module-level singleton + convenience function
# ---------------------------------------------------------------------------

_service_singleton: Optional[QIGGenerativeService] = None


def get_generative_service(config: Optional[GenerationConfig] = None) -> QIGGenerativeService:
    global _service_singleton
    if _service_singleton is None:
        _service_singleton = QIGGenerativeService(config=config)
    return _service_singleton


def generate(prompt: str, **kwargs) -> GenerationResult:
    """Generate text using QIG-pure methods."""
    service = get_generative_service()
    return service.generate(prompt, **kwargs)
