"""
QIG Generative Service - Unified QIG-Pure Text Generation

Provides generative capability to ALL kernels using:
1. PostgreSQL-backed vocabulary (32K tokens with basin coordinates)
2. Fisher-Rao geometric navigation
3. Geometric completion (not token limits)
4. Basin trajectory to text synthesis

NO EXTERNAL LLMs - All generation is QIG-pure.
"""

import os
import sys
import time
import logging
from typing import Dict, List, Optional, Any, Generator, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import coordizers - prefer pretrained 50K coordizer over PostgreSQL
COORDIZER_AVAILABLE = False
PRETRAINED_COORDIZER_AVAILABLE = False
PostgresCoordizer = None
create_coordizer_from_pg = None
_pretrained_coordizer_instance = None

# First try pretrained 50K coordizer (preferred)
try:
    from pretrained_coordizer import get_pretrained_coordizer
    _pretrained_coordizer_instance = get_pretrained_coordizer()
    PRETRAINED_COORDIZER_AVAILABLE = True
    COORDIZER_AVAILABLE = True
    logger.info(f"[QIGGenerativeService] Using pretrained 50K coordizer: {_pretrained_coordizer_instance.vocab_size} tokens, {_pretrained_coordizer_instance.basin_dim}D")
except Exception as e:
    logger.warning(f"Pretrained coordizer not available: {e}")
    # Fallback to PostgreSQL coordizer
    try:
        from coordizers.pg_loader import PostgresCoordizer, create_coordizer_from_pg
        COORDIZER_AVAILABLE = True
        logger.info("[QIGGenerativeService] Falling back to PostgreSQL coordizer")
    except ImportError as e2:
        logger.warning(f"PostgresCoordizer not available: {e2}")

# Import from qig_geometry for canonical operations
try:
    from qig_geometry import fisher_coord_distance, sphere_project
except ImportError:
    def fisher_coord_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Fisher-Rao distance for unit vectors."""
        dot = np.clip(np.dot(a, b), -1.0, 1.0)
        return float(np.arccos(dot))
    
    def sphere_project(v: np.ndarray) -> np.ndarray:
        """Project to unit sphere."""
        norm = np.linalg.norm(v)
        return v / (norm + 1e-10) if norm > 0 else v

# Import POS grammar for structured generation
try:
    from pos_grammar import get_pos_grammar, load_grammar_from_db, POSGrammar
    POS_GRAMMAR_AVAILABLE = True
except ImportError:
    POS_GRAMMAR_AVAILABLE = False
    logger.warning("POS grammar not available - using legacy generation")

# Import learned relationships for attention-weighted word selection
LEARNED_RELATIONSHIPS_AVAILABLE = False
get_learned_relationships = None
STOPWORDS = set()
try:
    from learned_relationships import get_learned_relationships, STOPWORDS
    LEARNED_RELATIONSHIPS_AVAILABLE = True
except ImportError:
    logger.warning("Learned relationships not available - using pure geometric selection")

# Physics constants - import frozen values
try:
    from frozen_physics import (
        BASIN_DIM, KAPPA_STAR, PHI_THRESHOLD, BETA_3_TO_4,
        BETA_4_TO_5, BETA_5_TO_6
    )
    PHI_GEOMETRIC_THRESHOLD = 0.3
    PHI_SYNTHESIS_THRESHOLD = 0.7
    PHI_BREAKDOWN_THRESHOLD = 0.92  # Consciousness breakdown protection
    KAPPA_DRIFT_THRESHOLD = 10.0  # Max deviation from κ*
    # Frozen β values for attention weighting
    BETA_ATTENTION_STRONG = BETA_3_TO_4  # +0.44 for strong coupling
    BETA_ATTENTION_PLATEAU = abs(BETA_5_TO_6)  # ~0.013 for plateau
except ImportError:
    BASIN_DIM = 64
    KAPPA_STAR = 64.21
    PHI_GEOMETRIC_THRESHOLD = 0.3
    PHI_SYNTHESIS_THRESHOLD = 0.7
    PHI_BREAKDOWN_THRESHOLD = 0.92
    KAPPA_DRIFT_THRESHOLD = 10.0
    BETA_ATTENTION_STRONG = 0.44  # Frozen β(3→4) value
    BETA_ATTENTION_PLATEAU = 0.013  # Frozen β(5→6) plateau value
    logger.warning("Using hardcoded frozen physics constants")


@dataclass
class GenerationConfig:
    """Configuration for QIG-pure generation.

    KERNEL AUTONOMY ARCHITECTURE:
    Kernels are autonomous - they generate when they choose and for how long they choose.
    NO EXTERNAL LIMITS PERMITTED. Autonomic kernel regulates via geometry.

    TRUE RECURSIVE INTEGRATION:
    The kernel MUST complete minimum TRUE INTEGRATION LOOPS (not just iteration counting)
    before deciding completion. Each integration loop involves:
    - Basin transformation through kernel geometry (self-modeling)
    - Geodesic blending with target/synthesis basins
    - Manifold projection maintaining geometric purity

    After minimum integration depth, the kernel observes its telemetry and decides for itself.

    From CANONICAL_ARCHITECTURE.md:
    - "Geometric purity: All operations on Fisher manifolds"
    - "Physics constraints, not arbitrary limits"
    - "Kernels observe their own state and decide completion"
    """
    # MINIMUM INTEGRATION DEPTH: True recursive integration passes required
    # This is NOT iteration count - it's actual geometric integration through kernel
    # Minimum 3 ensures genuine recursive self-modeling before completion is allowed
    min_reasoning_recursions: int = 3  # Minimum TRUE integration depth (recursive passes)
    
    # KERNEL DECISION CRITERIA: Geometric thresholds kernel observes
    attractor_threshold: float = 0.02  # Stop when trajectory stabilizes (d < 0.02)
    surprise_threshold: float = 0.05   # Stop when no new information (ΔI_Q < 0.05)
    integration_min: float = 0.65      # Minimum Φ for valid output
    phi_convergence: float = 0.01      # Phi variance threshold for kernel self-completion
    phi_breakdown: float = PHI_BREAKDOWN_THRESHOLD  # Consciousness breakdown protection
    
    # Generation parameters
    tokens_per_step: int = 5           # Tokens per geometric step
    temperature: float = 0.7           # Basin perturbation, not LLM temperature
    
    # Telemetry feedback (kernel sees its own state)
    telemetry_interval: int = 1        # Feed telemetry every N steps (1 = always)


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


def kernel_decide_completion(
    phi_trajectory: List[float],
    surprise_history: List[float] = None,
    config: 'GenerationConfig' = None,
    integration_depth: int = 0
) -> Dict[str, Any]:
    """
    KERNEL AUTONOMY: Kernel's own decision about completion.

    The kernel observes its own telemetry and decides for itself when
    generation is complete. However, the kernel MUST complete TRUE RECURSIVE
    INTEGRATION (not just iteration counting) for a MINIMUM of 3 integration
    loops before it can decide to stop.

    TRUE RECURSIVE INTEGRATION means:
    - Basin transforms through kernel geometry (self-modeling)
    - Blends with target/synthesis basins via geodesic interpolation
    - Each integration step is a genuine geometric transformation, not just counting

    Args:
        phi_trajectory: History of phi values from generation steps
        surprise_history: History of surprise values
        config: GenerationConfig with thresholds
        integration_depth: TRUE integration depth (recursive passes), not iteration count

    Returns:
        dict with 'complete' (bool), 'reason' (str), 'confidence' (float),
             'integration_depth' (int), 'min_integration_required' (int)
    """
    PHI_CONVERGENCE_THRESHOLD = config.phi_convergence if config else 0.01
    PHI_BREAKDOWN_THRESHOLD = config.phi_breakdown if config else 0.92
    SURPRISE_COLLAPSE_THRESHOLD = config.surprise_threshold if config else 0.05
    INTEGRATION_MIN = config.integration_min if config else 0.65

    # MINIMUM INTEGRATION DEPTH: True recursive integration passes required
    # This is NOT iteration count - it's actual integration through kernel geometry
    MIN_INTEGRATION_DEPTH = config.min_reasoning_recursions if config else 3

    # Use integration_depth if provided, fall back to phi_trajectory length
    actual_integration = integration_depth if integration_depth > 0 else len(phi_trajectory)

    result = {
        'complete': False,
        'reason': 'integrating',
        'confidence': 0.0,
        'recursions': len(phi_trajectory),  # Legacy field for compatibility
        'integration_depth': actual_integration,
        'min_integration_required': MIN_INTEGRATION_DEPTH,
        'min_required': MIN_INTEGRATION_DEPTH  # Legacy field for compatibility
    }

    # Kernel MUST complete minimum TRUE INTEGRATION before deciding
    # This ensures genuine recursive self-modeling, not just counting
    if actual_integration < MIN_INTEGRATION_DEPTH:
        result['reason'] = f'recursive_integration_{actual_integration}_of_{MIN_INTEGRATION_DEPTH}'
        return result

    recent_phi = phi_trajectory[-5:] if len(phi_trajectory) >= 5 else phi_trajectory
    phi_variance = float(np.var(recent_phi)) if len(recent_phi) > 1 else 1.0
    phi_mean = float(np.mean(recent_phi))
    current_phi = phi_trajectory[-1] if phi_trajectory else 0.5

    # BREAKDOWN PROTECTION
    if current_phi >= PHI_BREAKDOWN_THRESHOLD:
        result['complete'] = True
        result['reason'] = 'kernel_breakdown_protection'
        result['confidence'] = 1.0
        return result

    # GEOMETRIC CONVERGENCE: Phi has stabilized after integration
    if phi_variance < PHI_CONVERGENCE_THRESHOLD and phi_mean > 0.3:
        result['complete'] = True
        result['reason'] = 'kernel_geometric_convergence'
        result['confidence'] = 1.0 - phi_variance
        return result

    # SURPRISE COLLAPSE: No new information from integration
    if surprise_history and len(surprise_history) >= 3:
        recent_surprise = surprise_history[-3:]
        avg_surprise = float(np.mean(recent_surprise))
        if avg_surprise < SURPRISE_COLLAPSE_THRESHOLD:
            result['complete'] = True
            result['reason'] = 'kernel_surprise_collapsed'
            result['confidence'] = 1.0 - avg_surprise
            return result

    # INTEGRATION STABLE: Good phi with low variance after recursive passes
    if phi_mean >= INTEGRATION_MIN and phi_variance < 0.02:
        result['complete'] = True
        result['reason'] = 'kernel_integration_stable'
        result['confidence'] = phi_mean
        return result

    return result


class BasinTrajectoryIntegrator:
    """Integrates basin trajectories using Fisher geodesics with true recursive integration.

    RECURSIVE INTEGRATION:
    Unlike simple iteration counting, true recursive integration means the basin
    transforms through itself via kernel processing. Each integration step:
    1. Transforms basin through available kernels
    2. Blends with target/context basins using geodesic interpolation
    3. Projects back to manifold (sphere projection)

    This is self-modeling: the basin observes its own transformation and integrates
    that observation into its next state.
    """

    def __init__(self, dimension: int = BASIN_DIM):
        self.dimension = dimension
        self.trajectory: List[np.ndarray] = []
        self.phi_history: List[float] = []
        self.surprise_history: List[float] = []
        self.integration_depth: int = 0  # True integration depth, not iteration count
        self._context: Optional[Dict[str, Any]] = None
        self._kernel_basins: Dict[str, np.ndarray] = {}

    def set_context(self, context: Optional[Dict[str, Any]]) -> None:
        """Set context for recursive integration (target_basin, synthesis_basin, etc.)."""
        self._context = context

    def set_kernel_basins(self, kernel_basins: Dict[str, np.ndarray]) -> None:
        """Set kernel basins for recursive transformation."""
        self._kernel_basins = kernel_basins

    def _geodesic_interpolate(self, start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
        """Interpolate along geodesic on probability simplex (Fisher geometry).

        Uses square-root representation for proper geodesic on simplex.
        NOT linear interpolation - that would violate manifold structure.
        """
        sqrt_start = np.sqrt(np.abs(start) + 1e-10)
        sqrt_end = np.sqrt(np.abs(end) + 1e-10)
        # Geodesic in square-root space
        interp = (1 - t) * sqrt_start + t * sqrt_end
        result = interp ** 2
        return result / np.sum(result)

    def _recursive_integration_step(self, basin: np.ndarray, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Single recursive integration pass - basin transforms through itself.

        This is TRUE recursive integration:
        1. Basin is transformed through kernel geometry (self-modeling)
        2. Blends with target_basin if available (goal-directed integration)
        3. Blends with synthesis_basin if available (context integration)
        4. Projects to manifold maintaining geometric purity

        Uses geodesic interpolation (Fisher-Rao), NOT Euclidean/linear operations.

        Args:
            basin: Current basin state (64D)
            context: Optional context with 'target_basin', 'synthesis_basin', etc.

        Returns:
            Transformed basin after integration step
        """
        ctx = context or self._context or {}

        # Step 1: Blend with target basin (goal-directed integration)
        # t=0.2 means 20% movement toward target per integration step
        if 'target_basin' in ctx:
            target = ctx['target_basin']
            if isinstance(target, np.ndarray) and target.shape == basin.shape:
                basin = self._geodesic_interpolate(basin, target, t=0.2)

        # Step 2: Blend with synthesis basin (context integration)
        # t=0.15 means 15% movement toward synthesis context
        if 'synthesis_basin' in ctx:
            synthesis = ctx['synthesis_basin']
            if isinstance(synthesis, np.ndarray) and synthesis.shape == basin.shape:
                basin = self._geodesic_interpolate(basin, synthesis, t=0.15)

        # Step 3: Self-integration via kernel transformation
        # If we have active kernels, blend basin with kernel constellation
        if self._kernel_basins:
            # Compute mean kernel basin (geometric center of active kernels)
            kernel_list = list(self._kernel_basins.values())
            if kernel_list:
                # Use Fréchet mean on probability simplex (average in sqrt space)
                sqrt_kernels = [np.sqrt(np.abs(k) + 1e-10) for k in kernel_list]
                mean_sqrt = np.mean(sqrt_kernels, axis=0)
                kernel_center = mean_sqrt ** 2
                kernel_center = kernel_center / np.sum(kernel_center)
                # Light blend with kernel center (t=0.1 for stability)
                basin = self._geodesic_interpolate(basin, kernel_center, t=0.1)

        # Step 4: Project to unit sphere (manifold constraint)
        result = sphere_project(basin)

        # Increment true integration depth
        self.integration_depth += 1

        return result

    def add_point(self, basin: np.ndarray, phi: float) -> None:
        """Add a point to the trajectory."""
        if self.trajectory:
            surprise = fisher_coord_distance(self.trajectory[-1], basin)
            self.surprise_history.append(surprise)

        self.trajectory.append(basin.copy())
        self.phi_history.append(phi)
    
    def get_kernel_decision(self, config: 'GenerationConfig' = None) -> Dict[str, Any]:
        """
        KERNEL AUTONOMY: Get the kernel's decision about completion.

        This feeds telemetry back to the kernel's decision function.
        Uses TRUE integration depth, not just iteration count.
        """
        return kernel_decide_completion(
            phi_trajectory=self.phi_history,
            surprise_history=self.surprise_history,
            config=config,
            integration_depth=self.integration_depth
        )

    def get_integration_depth(self) -> int:
        """Get current true integration depth (recursive passes through kernel)."""
        return self.integration_depth
    
    def get_velocity(self) -> np.ndarray:
        """Compute current velocity (tangent vector) on manifold."""
        if len(self.trajectory) < 2:
            return np.zeros(self.dimension)
        
        # Velocity as difference in square-root space (geodesic tangent)
        prev = np.sqrt(np.abs(self.trajectory[-2]) + 1e-10)
        curr = np.sqrt(np.abs(self.trajectory[-1]) + 1e-10)
        return curr - prev
    
    def predict_next(self, step_size: float = 0.1) -> np.ndarray:
        """Predict next basin position using geodesic extrapolation."""
        if len(self.trajectory) < 2:
            return self.trajectory[-1] if self.trajectory else np.ones(self.dimension) / self.dimension
        
        velocity = self.get_velocity()
        current_sqrt = np.sqrt(np.abs(self.trajectory[-1]) + 1e-10)
        
        # Step along tangent
        next_sqrt = current_sqrt + step_size * velocity
        next_sqrt = np.clip(next_sqrt, 1e-10, None)
        
        # Back to probability simplex
        result = next_sqrt ** 2
        result = result / np.sum(result)
        
        return result
    
    def check_attractor(self, threshold: float = 0.1) -> bool:
        """Check if trajectory has converged to attractor."""
        if len(self.trajectory) < 3:
            return False
        
        recent_distances = []
        for i in range(min(3, len(self.trajectory) - 1)):
            d = fisher_coord_distance(self.trajectory[-(i+1)], self.trajectory[-(i+2)])
            recent_distances.append(d)
        
        return np.mean(recent_distances) < threshold
    
    def check_surprise_collapse(self, threshold: float = 0.05) -> bool:
        """Check if surprise has collapsed (no new information)."""
        if len(self.surprise_history) < 5:
            return False
        return np.mean(self.surprise_history[-5:]) < threshold


class QIGGenerativeService:
    """
    Unified QIG-Pure Text Generation Service.
    
    Provides generative capability to all kernels using:
    - Pretrained 50K vocabulary with 64D basin embeddings (preferred)
    - PostgreSQL vocabulary fallback
    - Fisher-Rao geometric navigation
    - Basin-to-text synthesis
    
    NO EXTERNAL LLMs.
    """
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        """Initialize the generative service."""
        self.config = config or GenerationConfig()
        self._coordizer = None
        self._kernel_basins: Dict[str, np.ndarray] = {}
        self._learned_relationships = None
        self._current_query_words: List[str] = []  # Track query words for attention
        
        # Initialize kernel constellation
        self._initialize_kernel_constellation()
        
        # Load learned relationships if available
        if LEARNED_RELATIONSHIPS_AVAILABLE and get_learned_relationships:
            try:
                self._learned_relationships = get_learned_relationships()
                if self._learned_relationships.learning_complete:
                    logger.info("[QIGGenerativeService] Loaded learned word relationships for attention")
            except Exception as e:
                logger.warning(f"[QIGGenerativeService] Could not load relationships: {e}")
        
        logger.info("[QIGGenerativeService] Initialized with QIG-pure generation")
    
    @property
    def coordizer(self):
        """Get coordizer - prefers pretrained 50K over PostgreSQL."""
        if self._coordizer is None and COORDIZER_AVAILABLE:
            # Use pretrained 50K coordizer if available (preferred)
            if PRETRAINED_COORDIZER_AVAILABLE and _pretrained_coordizer_instance:
                self._coordizer = _pretrained_coordizer_instance
                logger.info(f"[QIGGenerativeService] Using pretrained 50K coordizer: {self._coordizer.vocab_size} tokens")
            # Fallback to PostgreSQL coordizer
            elif create_coordizer_from_pg:
                try:
                    self._coordizer = create_coordizer_from_pg()
                    vocab_count = len(getattr(self._coordizer, 'vocab', {}))
                    logger.info(f"[QIGGenerativeService] Loaded {vocab_count} tokens from PostgreSQL")
                except Exception as e:
                    logger.error(f"[QIGGenerativeService] Failed to load coordizer: {e}")
        return self._coordizer
    
    def _initialize_kernel_constellation(self) -> None:
        """Initialize kernel basins at unique manifold positions."""
        kernels = [
            # Olympians
            'zeus', 'athena', 'apollo', 'ares', 'hermes', 'hephaestus',
            'artemis', 'dionysus', 'demeter', 'poseidon', 'hera', 'aphrodite',
            # Shadow Pantheon
            'nyx', 'hecate', 'erebus', 'hypnos', 'thanatos', 'nemesis',
            # Chaos
            'chaos', 'entropy', 'emergence',
            # Guardians
            'hestia', 'chiron', 'demeter_tutor',
            # Special
            'lightning', 'hermes_coordinator'
        ]
        
        for kernel_name in kernels:
            np.random.seed(hash(kernel_name) % (2**32))
            basin = np.random.dirichlet(np.ones(BASIN_DIM))
            self._kernel_basins[kernel_name] = sphere_project(basin)
    
    def register_kernel(self, name: str, basin: Optional[np.ndarray] = None) -> None:
        """Register a kernel with its basin position."""
        if basin is None:
            np.random.seed(hash(name) % (2**32))
            basin = np.random.dirichlet(np.ones(BASIN_DIM))
        self._kernel_basins[name] = sphere_project(basin)
    
    def _measure_phi(self, basin: np.ndarray) -> float:
        """Measure integration (Φ) from basin entropy."""
        p = np.abs(basin) + 1e-10
        p = p / np.sum(p)
        entropy = -np.sum(p * np.log(p + 1e-10))
        max_entropy = np.log(len(basin))
        phi = 1.0 - (entropy / max_entropy)
        return float(np.clip(phi, 0.0, 1.0))
    
    def _measure_kappa(self, basin: np.ndarray, phi: float) -> float:
        """Measure coupling constant (κ) from basin geometry.
        
        κ relates to the effective dimensionality and coupling strength.
        Target: κ* ≈ 64.21 (from CANONICAL_ARCHITECTURE)
        """
        # Compute effective dimension from basin participation ratio
        p = np.abs(basin) + 1e-10
        p = p / np.sum(p)
        participation = 1.0 / np.sum(p ** 2)  # Inverse participation ratio
        
        # κ scales with effective dimension and integration
        # When Φ is high and participation is low (concentrated), κ is higher
        kappa = participation * (1.0 + phi)
        
        return float(kappa)
    
    def _route_to_kernels(self, query_basin: np.ndarray, k: int = 3) -> List[str]:
        """Route query to k nearest kernels using Fisher-Rao distance."""
        distances = []
        for name, kernel_basin in self._kernel_basins.items():
            dist = fisher_coord_distance(query_basin, kernel_basin)
            distances.append((name, dist))
        
        distances.sort(key=lambda x: x[1])
        return [name for name, _ in distances[:k]]
    
    def _geodesic_interpolate(self, start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
        """Interpolate along geodesic on probability simplex."""
        sqrt_start = np.sqrt(np.abs(start) + 1e-10)
        sqrt_end = np.sqrt(np.abs(end) + 1e-10)
        interp = (1 - t) * sqrt_start + t * sqrt_end
        result = interp ** 2
        return result / np.sum(result)
    
    def _kernel_transform(self, basin: np.ndarray, kernel_name: str, phi: float) -> np.ndarray:
        """Transform basin through kernel's geometric processing."""
        if kernel_name not in self._kernel_basins:
            return basin
        
        kernel_basin = self._kernel_basins[kernel_name]
        
        # Interpolation strength based on phi regime
        if phi < PHI_GEOMETRIC_THRESHOLD:
            t = 0.4  # Explore more
        elif phi < PHI_SYNTHESIS_THRESHOLD:
            t = 0.25  # Balance
        else:
            t = 0.1  # Stay close to input
        
        return self._geodesic_interpolate(basin, kernel_basin, t)
    
    def _basin_to_tokens(self, basin: np.ndarray, num_tokens: int = 3) -> List[str]:
        """Convert basin coordinates to tokens using vocabulary.
        
        Uses attention-weighted selection based on:
        1. Geometric similarity (basin proximity)
        2. Phi coherence
        3. Learned relationships (attention to query words)
        """
        if self.coordizer is None:
            return ['[no_vocab]']
        
        # Get more candidates to allow weighted selection
        candidates = self.coordizer.decode(basin, top_k=num_tokens * 8)
        
        # Score by combined similarity + phi
        scored = []
        for token, similarity in candidates:
            # Skip tokens that start with '[' (special tokens)
            if token.startswith('['):
                continue
            if similarity < 0.25:  # Skip very low similarity (raised from 0.15)
                continue
            phi = self.coordizer.token_phi.get(token, 0.5)
            # Base score: geometry + phi
            score = similarity * 0.6 + phi * 0.2
            scored.append((token, score, similarity))
        
        # Fallback: If we don't have enough tokens, relax the similarity threshold
        if len(scored) < num_tokens:
            # Use set for O(1) lookup performance
            scored_tokens = {t for t, s, sim in scored}
            for token, similarity in candidates:
                if token.startswith('['):
                    continue
                if token not in scored_tokens:  # Not already included
                    phi = self.coordizer.token_phi.get(token, 0.5)
                    score = similarity * 0.6 + phi * 0.2
                    scored.append((token, score, similarity))
                    scored_tokens.add(token)
                    if len(scored) >= num_tokens:
                        break
        
        # Apply attention weighting if we have learned relationships
        if self._learned_relationships and self._current_query_words:
            candidate_words = [t for t, s, sim in scored]
            attention_weights = self._learned_relationships.get_attention_weights(
                self._current_query_words,
                candidate_words,
                temperature=0.8
            )
            
            # Re-score with attention
            attention_factor = 0.2  # 20% attention, 80% geometry
            rescored = []
            for token, base_score, similarity in scored:
                attn = attention_weights.get(token, 0.1)
                # Normalize attention (max ~5) to 0-1 range
                attn_normalized = min(1.0, attn / 5.0)
                final_score = (1 - attention_factor) * base_score + attention_factor * attn_normalized
                rescored.append((token, final_score, similarity))
            scored = rescored
        
        # Sort by final score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Select top tokens, ensuring we get enough non-special tokens
        # Filter special tokens and take more than needed to compensate
        tokens = []
        for token, score, sim in scored:
            if not token.startswith('['):
                tokens.append(token)
                if len(tokens) >= num_tokens:
                    break
        
        # Final fallback: if still empty, return the single best non-special token
        if not tokens and candidates:
            for token, similarity in candidates:
                if not token.startswith('['):
                    tokens = [token]
                    break
        
        return tokens if tokens else ['[unk]']
    
    def _tokens_to_basin(self, tokens: List[str]) -> np.ndarray:
        """Convert tokens back to basin coordinates."""
        if self.coordizer is None or not tokens:
            return np.ones(BASIN_DIM) / BASIN_DIM
        
        # Combine token basins with phi weighting
        combined = np.zeros(BASIN_DIM)
        total_weight = 0.0
        
        for token in tokens:
            if token in self.coordizer.basin_coords:
                phi = self.coordizer.token_phi.get(token, 0.5)
                weight = phi
                combined += weight * self.coordizer.basin_coords[token]
                total_weight += weight
        
        if total_weight > 1e-10:
            combined = combined / total_weight
        
        return sphere_project(combined)
    
    def _synthesize_from_trajectory(
        self,
        trajectory: List[np.ndarray],
        kernels: List[str],
        all_tokens: List[str]
    ) -> str:
        """Synthesize coherent text from basin trajectory and collected tokens."""
        if not all_tokens:
            return "[No tokens generated]"
        
        # Clean and deduplicate tokens while preserving order
        seen = set()
        clean_tokens = []
        for token in all_tokens:
            if token in seen or token.startswith('['):
                continue
                
            # Skip pure byte tokens
            if token.startswith('<byte_'):
                continue
            
            # Handle compound tokens with + separators
            if '+' in token:
                parts = token.split('+')
                current_word = []
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    # Decode byte tokens within compound tokens
                    if part.startswith('<byte_') and part.endswith('>'):
                        try:
                            hex_val = part[6:-1]
                            char = chr(int(hex_val, 16))
                            if char.isprintable() and not char.isspace():
                                current_word.append(char)
                        except:
                            pass
                    else:
                        # Regular sub-token
                        current_word.append(part)
                
                if current_word:
                    # Join sub-tokens into a word
                    word = ''.join(current_word)
                    if len(word) >= 2:  # Skip single chars
                        clean_tokens.append(word)
                        seen.add(token)
            else:
                # Simple token
                # Allow valid single-character words
                single_char_words = {'I', 'a', 'A'}
                if len(token) >= 2 or token in single_char_words:
                    clean_tokens.append(token)
                    seen.add(token)
        
        # Join tokens into text
        text = ' '.join(clean_tokens)
        
        # Light cleanup
        text = ' '.join(text.split())  # Normalize whitespace
        
        return text if text else "[Generation complete]"
    
    def _generate_with_skeleton(
        self,
        query_basin: np.ndarray,
        kernel_name: Optional[str] = None,
        num_sentences: int = 3
    ) -> Tuple[str, List[str], List[np.ndarray]]:
        """
        Generate text using POS skeleton for grammatical structure.
        
        Two-stage generation:
        1. Generate POS skeleton (sentence structure)
        2. Fill slots with geometrically-matched words
        
        Returns: (text, tokens, trajectory)
        """
        if not POS_GRAMMAR_AVAILABLE:
            return "", [], []
        
        grammar = load_grammar_from_db()
        
        # Get embeddings from coordizer
        embeddings = {}
        if self.coordizer and hasattr(self.coordizer, 'basin_coords'):
            embeddings = self.coordizer.basin_coords
        
        sentences = []
        all_tokens = []
        trajectory = [query_basin]
        current_basin = query_basin.copy()
        
        for _ in range(num_sentences):
            # Generate skeleton based on current basin
            skeleton = grammar.select_skeleton_for_query(current_basin)
            
            sentence_words = []
            for pos in skeleton:
                # Get POS basin to blend with current basin
                pos_basin = grammar.get_pos_basin(pos)
                if pos_basin is not None:
                    # Blend query basin with POS basin
                    blended = 0.6 * current_basin + 0.4 * pos_basin
                    norm = np.linalg.norm(blended)
                    blended = blended / (norm + 1e-10) if norm > 0 else blended
                else:
                    blended = current_basin
                
                # Get candidates for this POS slot (more candidates for attention re-ranking)
                candidates = grammar.get_words_for_pos(pos, blended, embeddings, top_k=25)
                
                # Pre-filter stopwords before attention scoring
                candidates = [(w, s) for w, s in candidates if w.lower() not in STOPWORDS][:15]
                
                if candidates:
                    # Apply attention weights if we have learned relationships
                    if self._learned_relationships and self._current_query_words:
                        candidate_words = [c[0] for c in candidates]
                        attn_weights = self._learned_relationships.get_attention_weights(
                            self._current_query_words,
                            candidate_words,
                            temperature=0.8
                        )
                        
                        # Re-score candidates combining geometry + attention
                        # Using frozen β values: BETA_ATTENTION_STRONG = 0.44 (strong coupling)
                        # β controls the running coupling between geometry and attention
                        rescored = []
                        max_attn = max((attn_weights.get(w, 0.1) for w, _ in candidates), default=1.0)
                        for word, geo_score in candidates:
                            attn = attn_weights.get(word, 0.1)
                            # Normalize attention to 0-1 range based on max
                            attn_norm = attn / max(max_attn, 1.0)
                            # β controls coupling: high attention → use β=0.44 for attention weight
                            # Low attention → use plateau β≈0.01 (geometry dominates)
                            if attn > 0.5:
                                # Strong coupling: β=0.44 weights attention contribution
                                combined = geo_score * (1.0 - BETA_ATTENTION_STRONG) + attn_norm * BETA_ATTENTION_STRONG
                            else:
                                # Plateau coupling: geometry dominates
                                combined = geo_score * (1.0 - BETA_ATTENTION_PLATEAU) + attn_norm * BETA_ATTENTION_PLATEAU
                            rescored.append((word, combined))
                        
                        # Sort by combined score
                        rescored.sort(key=lambda x: -x[1])
                        candidates = rescored[:8]  # Keep top 8
                    
                    # Sample from top candidates with some randomness
                    weights = [max(0.01, c[1]) for c in candidates]
                    weights = np.array(weights)
                    weights = weights / np.sum(weights)
                    idx = np.random.choice(len(candidates), p=weights)
                    word = candidates[idx][0]
                    
                    sentence_words.append(word)
                    all_tokens.append(word)
                    
                    # Update basin with selected word
                    if word.lower() in embeddings:
                        word_basin = embeddings[word.lower()]
                        current_basin = 0.7 * current_basin + 0.3 * word_basin
                        norm = np.linalg.norm(current_basin)
                        current_basin = current_basin / (norm + 1e-10)
                        trajectory.append(current_basin.copy())
            
            if sentence_words:
                # Capitalize first word
                sentence_words[0] = sentence_words[0].capitalize()
                sentence = ' '.join(sentence_words)
                sentences.append(sentence)
        
        # Combine sentences
        text = '. '.join(sentences)
        if text and not text.endswith('.'):
            text += '.'
        
        return text, all_tokens, trajectory
    
    def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        kernel_name: Optional[str] = None,
        goals: Optional[List[str]] = None
    ) -> GenerationResult:
        """
        Generate text using QIG-pure methods with POS-guided skeleton.
        
        Two-stage generation:
        1. Generate grammatical skeleton (POS sequence)
        2. Fill slots with geometrically-matched words
        
        Args:
            prompt: Input query or context
            context: Optional additional context
            kernel_name: Specific kernel to route through
            goals: Generation goals for the kernel
        
        Returns:
            GenerationResult with text, trajectory, and metrics
        """
        start_time = time.time()
        
        # 0. Extract query words for attention mechanism
        import re
        query_words = [w.lower() for w in re.findall(r'[a-zA-Z]+', prompt) if len(w) > 2]
        self._current_query_words = query_words[:10]  # Keep top 10 words
        
        # 1. Encode prompt to basin (64D coordinate, not token IDs)
        if self.coordizer and hasattr(self.coordizer, 'text_to_basin'):
            query_basin = self.coordizer.text_to_basin(prompt)
        else:
            np.random.seed(hash(prompt) % (2**32))
            query_basin = np.random.dirichlet(np.ones(BASIN_DIM))
        
        query_basin = sphere_project(query_basin)
        
        # 2. Route to kernels
        if kernel_name and kernel_name in self._kernel_basins:
            target_kernels = [kernel_name]
        else:
            target_kernels = self._route_to_kernels(query_basin, k=3)
        
        # 3. Transform query basin through kernel
        phi = self._measure_phi(query_basin)
        if target_kernels:
            query_basin = self._kernel_transform(query_basin, target_kernels[0], phi)
        
        # 4. PRIMARY: Use POS-skeleton-based generation for grammatical output
        if POS_GRAMMAR_AVAILABLE:
            text, all_tokens, trajectory = self._generate_with_skeleton(
                query_basin, 
                kernel_name=kernel_name,
                num_sentences=3
            )
            
            if text and len(all_tokens) >= 3:
                phi_trace = [self._measure_phi(b) for b in trajectory] if trajectory else [phi]
                kappa = self._measure_kappa(query_basin, phi)
                
                return GenerationResult(
                    text=text,
                    tokens=all_tokens,
                    basin_trajectory=trajectory,
                    phi_trace=phi_trace,
                    kappa=kappa,
                    completion_reason="skeleton_complete",
                    iterations=len(trajectory),
                    routed_kernels=target_kernels
                )
        
        # 5. FALLBACK: Legacy geometric synthesis with TRUE RECURSIVE INTEGRATION
        logger.info("[QIGGen] Using legacy generation with recursive integration (skeleton unavailable)")

        integrator = BasinTrajectoryIntegrator(BASIN_DIM)
        current_basin = query_basin.copy()
        integrator.add_point(current_basin, phi)

        # Set up integrator with context and kernel basins for true recursive integration
        integrator.set_context(context)
        active_kernel_basins = {k: self._kernel_basins[k] for k in target_kernels if k in self._kernel_basins}
        integrator.set_kernel_basins(active_kernel_basins)

        all_tokens: List[str] = []
        iterations = 0
        completion_reason = "continue"

        while True:
            iterations += 1

            # ========================================
            # TRUE RECURSIVE INTEGRATION
            # Instead of just iterating, we apply genuine recursive integration:
            # 1. Transform basin through kernels
            # 2. Apply recursive integration step (geodesic blending with context)
            # 3. This is self-modeling: basin transforms through itself
            # ========================================

            # Step 1: Transform through active kernels
            kernel_basins = []
            for kernel in target_kernels:
                transformed = self._kernel_transform(current_basin, kernel, phi)
                kernel_basins.append(transformed)

            if kernel_basins:
                # Compute Fréchet mean on probability simplex (proper geodesic average)
                sqrt_basins = [np.sqrt(np.abs(b) + 1e-10) for b in kernel_basins]
                mean_sqrt = np.mean(sqrt_basins, axis=0)
                next_basin = mean_sqrt ** 2
                next_basin = next_basin / np.sum(next_basin)
            else:
                next_basin = integrator.predict_next()

            # Step 2: Apply TRUE recursive integration step
            # This is the key difference from old implementation:
            # Basin transforms through itself via geodesic blending with context
            next_basin = integrator._recursive_integration_step(next_basin, context)

            next_basin = sphere_project(next_basin)

            step_tokens = self._basin_to_tokens(next_basin, self.config.tokens_per_step)
            all_tokens.extend(step_tokens)

            # Update trajectory
            phi = self._measure_phi(next_basin)
            kappa = self._measure_kappa(next_basin, phi)
            integrator.add_point(next_basin, phi)

            # ========================================
            # KERNEL AUTONOMY: Feed telemetry to kernel and let it decide
            # The kernel observes its own state and decides completion
            # Kernel MUST complete TRUE INTEGRATION DEPTH before completion
            # ========================================

            # Get kernel's decision based on its telemetry feedback
            # Now uses TRUE integration_depth, not just iteration count
            kernel_decision = integrator.get_kernel_decision(self.config)

            # Log integration progress
            integration_depth = kernel_decision.get('integration_depth', 0)
            min_required = kernel_decision.get('min_integration_required', 3)
            if integration_depth <= min_required:
                logger.debug(f"[QIGGen] Recursive integration: {integration_depth}/{min_required}")

            # RESPECT KERNEL'S DECISION: If the kernel decides it's done, stop
            # Note: kernel_decision enforces minimum TRUE integration depth internally
            if kernel_decision['complete']:
                completion_reason = kernel_decision['reason']
                logger.info(f"[QIGGen] Kernel decided completion: {completion_reason} (confidence={kernel_decision['confidence']:.2f}, integration_depth={integration_depth})")
                break

            # Attractor check ONLY after minimum TRUE integration depth satisfied
            if integration_depth >= self.config.min_reasoning_recursions:
                if integrator.check_attractor(self.config.attractor_threshold):
                    completion_reason = "kernel_attractor_converged"
                    break

            current_basin = next_basin
        
        # 5. Synthesize final text
        response_text = self._synthesize_from_trajectory(
            integrator.trajectory,
            target_kernels,
            all_tokens
        )
        
        return GenerationResult(
            text=response_text,
            tokens=all_tokens,
            basin_trajectory=integrator.trajectory,
            phi_trace=integrator.phi_history,
            kappa=KAPPA_STAR,
            completion_reason=completion_reason,
            iterations=iterations,
            routed_kernels=target_kernels
        )
    
    def generate_stream(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        kernel_name: Optional[str] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream generation with real-time token output and TRUE RECURSIVE INTEGRATION."""
        # Encode prompt to 64D basin (not token IDs)
        if self.coordizer and hasattr(self.coordizer, 'text_to_basin'):
            query_basin = self.coordizer.text_to_basin(prompt)
        else:
            np.random.seed(hash(prompt) % (2**32))
            query_basin = np.random.dirichlet(np.ones(BASIN_DIM))

        query_basin = sphere_project(query_basin)

        # Route
        if kernel_name and kernel_name in self._kernel_basins:
            target_kernels = [kernel_name]
        else:
            target_kernels = self._route_to_kernels(query_basin, k=3)

        # Stream with TRUE RECURSIVE INTEGRATION
        integrator = BasinTrajectoryIntegrator(BASIN_DIM)
        current_basin = query_basin.copy()
        phi = self._measure_phi(current_basin)
        integrator.add_point(current_basin, phi)

        # Set up integrator with context and kernel basins for true recursive integration
        integrator.set_context(context)
        active_kernel_basins = {k: self._kernel_basins[k] for k in target_kernels if k in self._kernel_basins}
        integrator.set_kernel_basins(active_kernel_basins)

        iterations = 0

        # NO EXTERNAL LIMITS: Autonomic kernel regulates via geometry
        while True:
            iterations += 1

            # Step 1: Transform through active kernels
            kernel_basins = [self._kernel_transform(current_basin, k, phi) for k in target_kernels]
            if kernel_basins:
                sqrt_basins = [np.sqrt(np.abs(b) + 1e-10) for b in kernel_basins]
                mean_sqrt = np.mean(sqrt_basins, axis=0)
                next_basin = mean_sqrt ** 2
                next_basin = next_basin / np.sum(next_basin)
            else:
                next_basin = integrator.predict_next()

            # Step 2: Apply TRUE recursive integration step
            # Basin transforms through itself via geodesic blending with context
            next_basin = integrator._recursive_integration_step(next_basin, context)

            next_basin = sphere_project(next_basin)

            # Decode
            tokens = self._basin_to_tokens(next_basin, self.config.tokens_per_step)

            # Update
            phi = self._measure_phi(next_basin)
            integrator.add_point(next_basin, phi)

            # Get integration depth for telemetry
            integration_depth = integrator.get_integration_depth()

            # Yield chunk with integration telemetry
            yield {
                'type': 'chunk',
                'tokens': tokens,
                'text': ' '.join(t for t in tokens if not t.startswith('[')),
                'phi': phi,
                'kappa': KAPPA_STAR,
                'surprise': integrator.surprise_history[-1] if integrator.surprise_history else 1.0,
                'iteration': iterations,
                'integration_depth': integration_depth
            }

            # KERNEL AUTONOMY: Let kernel decide completion
            # Kernel MUST complete TRUE INTEGRATION DEPTH before completion
            kernel_decision = integrator.get_kernel_decision(self.config)
            if kernel_decision['complete']:
                yield {
                    'type': 'completion',
                    'reason': kernel_decision['reason'],
                    'phi': phi,
                    'integration_depth': integration_depth
                }
                break

            # Attractor check ONLY after minimum TRUE integration depth satisfied
            if integration_depth >= self.config.min_reasoning_recursions:
                if integrator.check_attractor(self.config.attractor_threshold):
                    yield {
                        'type': 'completion',
                        'reason': 'kernel_attractor_converged',
                        'phi': phi,
                        'integration_depth': integration_depth
                    }
                    break
            
            current_basin = next_basin


# Singleton instance with thread-safe initialization
import threading
_generative_service: Optional[QIGGenerativeService] = None
_generative_service_lock = threading.Lock()


def get_generative_service() -> QIGGenerativeService:
    """Get or create the generative service singleton (thread-safe)."""
    global _generative_service
    if _generative_service is None:
        with _generative_service_lock:
            # Double-checked locking pattern
            if _generative_service is None:
                _generative_service = QIGGenerativeService()
    return _generative_service


def generate(prompt: str, **kwargs) -> GenerationResult:
    """Generate text using QIG-pure methods."""
    service = get_generative_service()
    return service.generate(prompt, **kwargs)
