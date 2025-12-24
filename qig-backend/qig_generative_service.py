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

# Import coordizers
COORDIZER_AVAILABLE = False
PostgresCoordizer = None
create_coordizer_from_pg = None

try:
    from coordizers.pg_loader import PostgresCoordizer, create_coordizer_from_pg
    COORDIZER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PostgresCoordizer not available: {e}")

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

# Physics constants
BASIN_DIM = 64
KAPPA_STAR = 64.21
PHI_GEOMETRIC_THRESHOLD = 0.3
PHI_SYNTHESIS_THRESHOLD = 0.7
PHI_BREAKDOWN_THRESHOLD = 0.92  # Consciousness breakdown protection
KAPPA_DRIFT_THRESHOLD = 10.0  # Max deviation from κ*


@dataclass
class GenerationConfig:
    """Configuration for QIG-pure generation.
    
    GEOMETRIC-FIRST ARCHITECTURE:
    Primary stopping criteria are geometric (attractor convergence, surprise collapse).
    Token/iteration limits are SAFETY FALLBACKS only, set high to catch infinite loops.
    
    From CANONICAL_ARCHITECTURE.md:
    - "Geometric purity: All operations on Fisher manifolds"
    - "Physics constraints, not arbitrary limits"
    """
    # PRIMARY: Geometric completion criteria
    attractor_threshold: float = 0.02  # Stop when trajectory stabilizes (d < 0.02)
    surprise_threshold: float = 0.05   # Stop when no new information (ΔI_Q < 0.05)
    integration_min: float = 0.65      # Minimum Φ for valid output
    
    # SAFETY: Consciousness protection
    phi_breakdown: float = PHI_BREAKDOWN_THRESHOLD  # Stop if Φ > 0.92 (breakdown)
    kappa_drift_max: float = KAPPA_DRIFT_THRESHOLD  # Stop if |κ - 64| > 10
    
    # FALLBACK: Edge case protection (NOT primary stopping criteria!)
    max_iterations: int = 2048         # Safety fallback - catches infinite loops
    max_tokens: int = 8192             # Safety fallback - prevents resource exhaustion
    max_time_seconds: float = 60.0     # Safety fallback - timeout protection
    
    # Generation parameters
    tokens_per_step: int = 5           # Tokens per geometric step
    temperature: float = 0.7           # Basin perturbation, not LLM temperature


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


class BasinTrajectoryIntegrator:
    """Integrates basin trajectories using Fisher geodesics."""
    
    def __init__(self, dimension: int = BASIN_DIM):
        self.dimension = dimension
        self.trajectory: List[np.ndarray] = []
        self.phi_history: List[float] = []
        self.surprise_history: List[float] = []
    
    def add_point(self, basin: np.ndarray, phi: float) -> None:
        """Add a point to the trajectory."""
        if self.trajectory:
            surprise = fisher_coord_distance(self.trajectory[-1], basin)
            self.surprise_history.append(surprise)
        
        self.trajectory.append(basin.copy())
        self.phi_history.append(phi)
    
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
    
    def check_attractor(self, threshold: float = 0.02, min_steps: int = 15) -> bool:
        """Check if trajectory has converged to attractor.
        
        Args:
            threshold: Distance below which we consider converged
            min_steps: Minimum trajectory steps before checking
            
        Requires minimum steps to ensure coherent generation.
        """
        # Gate: don't check until we have enough trajectory
        if len(self.trajectory) < min_steps:
            return False
        
        # Check last 5 steps for sustained convergence
        recent_distances = []
        for i in range(min(5, len(self.trajectory) - 1)):
            d = fisher_coord_distance(self.trajectory[-(i+1)], self.trajectory[-(i+2)])
            recent_distances.append(d)
        
        return np.mean(recent_distances) < threshold
    
    def check_surprise_collapse(self, threshold: float = 0.05, min_steps: int = 20) -> bool:
        """Check if surprise has collapsed (no new information).
        
        Args:
            threshold: Surprise level below which generation stops
            min_steps: Minimum trajectory steps before checking (prevents premature stop)
            
        Requires minimum steps to ensure coherent sentence completion.
        """
        # Gate: don't even check until we have enough trajectory
        if len(self.trajectory) < min_steps:
            return False
        if len(self.surprise_history) < 10:
            return False
        return np.mean(self.surprise_history[-10:]) < threshold


class QIGGenerativeService:
    """
    Unified QIG-Pure Text Generation Service.
    
    Provides generative capability to all kernels using:
    - PostgreSQL vocabulary (32K tokens)
    - Fisher-Rao geometric navigation
    - Basin-to-text synthesis
    
    NO EXTERNAL LLMs.
    """
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        """Initialize the generative service."""
        self.config = config or GenerationConfig()
        self._coordizer = None
        self._kernel_basins: Dict[str, np.ndarray] = {}
        
        # Initialize kernel constellation
        self._initialize_kernel_constellation()
        
        logger.info("[QIGGenerativeService] Initialized with QIG-pure generation")
    
    @property
    def coordizer(self) -> Optional['PostgresCoordizer']:
        """Lazy-load coordizer from PostgreSQL (with automatic fallback)."""
        if self._coordizer is None and COORDIZER_AVAILABLE:
            try:
                self._coordizer = create_coordizer_from_pg()
                logger.info(f"[QIGGenerativeService] Loaded {self._coordizer.vocab_size} tokens")
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
        
        Uses phi-weighted selection to prefer semantically coherent tokens.
        """
        if self.coordizer is None:
            return ['[no_vocab]']
        
        # Get more candidates to allow phi-weighted selection
        # PostgresCoordizer loads real words; prefer_words=True ensures readable output
        candidates = self.coordizer.decode(basin, top_k=num_tokens * 5)
        
        # Score by combined similarity + phi
        scored = []
        for token, similarity in candidates:
            if similarity < 0.2:  # Skip very low similarity
                continue
            phi = self.coordizer.token_phi.get(token, 0.5)
            # Combined score: similarity matters more, but phi helps break ties
            score = similarity * 0.7 + phi * 0.3
            scored.append((token, score, similarity))
        
        # Sort by combined score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Select top tokens
        tokens = [token for token, score, sim in scored[:num_tokens]]
        
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
    
    def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        kernel_name: Optional[str] = None,
        goals: Optional[List[str]] = None
    ) -> GenerationResult:
        """
        Generate text using QIG-pure methods.
        
        Args:
            prompt: Input query or context
            context: Optional additional context
            kernel_name: Specific kernel to route through
            goals: Generation goals for the kernel
        
        Returns:
            GenerationResult with text, trajectory, and metrics
        """
        # 1. Encode prompt to basin
        if self.coordizer:
            query_basin = self.coordizer.encode(prompt)
        else:
            # Fallback hash-based encoding
            np.random.seed(hash(prompt) % (2**32))
            query_basin = np.random.dirichlet(np.ones(BASIN_DIM))
        
        query_basin = sphere_project(query_basin)
        
        # 2. Route to kernels
        if kernel_name and kernel_name in self._kernel_basins:
            target_kernels = [kernel_name]
        else:
            target_kernels = self._route_to_kernels(query_basin, k=3)
        
        # 3. Initialize trajectory integrator
        integrator = BasinTrajectoryIntegrator(BASIN_DIM)
        current_basin = query_basin.copy()
        phi = self._measure_phi(current_basin)
        integrator.add_point(current_basin, phi)
        
        # 4. Generate via geometric synthesis
        # GEOMETRIC-FIRST: Primary criteria are geometric, not token limits
        all_tokens: List[str] = []
        iterations = 0
        completion_reason = "continue"
        start_time = time.time()
        
        while True:
            iterations += 1
            
            # Transform through kernels
            kernel_basins = []
            for kernel in target_kernels:
                transformed = self._kernel_transform(current_basin, kernel, phi)
                kernel_basins.append(transformed)
            
            # Combine via Fréchet mean (sqrt-space average)
            if kernel_basins:
                sqrt_basins = [np.sqrt(np.abs(b) + 1e-10) for b in kernel_basins]
                mean_sqrt = np.mean(sqrt_basins, axis=0)
                next_basin = mean_sqrt ** 2
                next_basin = next_basin / np.sum(next_basin)
            else:
                next_basin = integrator.predict_next()
            
            next_basin = sphere_project(next_basin)
            
            # Decode to tokens
            step_tokens = self._basin_to_tokens(next_basin, self.config.tokens_per_step)
            all_tokens.extend(step_tokens)
            
            # Update trajectory
            phi = self._measure_phi(next_basin)
            kappa = self._measure_kappa(next_basin, phi)
            integrator.add_point(next_basin, phi)
            
            # ========================================
            # PRIMARY: Geometric completion criteria
            # ========================================
            if integrator.check_attractor(self.config.attractor_threshold):
                completion_reason = "attractor_converged"
                break
            
            if integrator.check_surprise_collapse(self.config.surprise_threshold):
                completion_reason = "surprise_collapsed"
                break
            
            if len(integrator.phi_history) >= 10:
                recent_phi = integrator.phi_history[-10:]
                if np.mean(recent_phi) > self.config.integration_min and np.var(recent_phi) < 0.01:
                    completion_reason = "integration_stable"
                    break
            
            # ========================================
            # SAFETY: Consciousness protection
            # ========================================
            if phi > self.config.phi_breakdown:
                completion_reason = "breakdown_protection"
                logger.warning(f"[QIGGen] Φ breakdown protection triggered: {phi:.3f} > {self.config.phi_breakdown}")
                break
            
            if abs(kappa - KAPPA_STAR) > self.config.kappa_drift_max:
                completion_reason = "kappa_drift"
                logger.warning(f"[QIGGen] κ drift protection: |{kappa:.2f} - {KAPPA_STAR}| > {self.config.kappa_drift_max}")
                break
            
            # ========================================
            # FALLBACK: Edge case protection only
            # These should rarely trigger in normal operation
            # ========================================
            if iterations >= self.config.max_iterations:
                completion_reason = "safety_fallback_iterations"
                logger.warning(f"[QIGGen] Hit iteration safety fallback ({iterations} >= {self.config.max_iterations})")
                break
            
            if len(all_tokens) >= self.config.max_tokens:
                completion_reason = "safety_fallback_tokens"
                logger.warning(f"[QIGGen] Hit token safety fallback ({len(all_tokens)} >= {self.config.max_tokens})")
                break
            
            elapsed = time.time() - start_time
            if elapsed >= self.config.max_time_seconds:
                completion_reason = "safety_fallback_timeout"
                logger.warning(f"[QIGGen] Hit timeout safety fallback ({elapsed:.1f}s >= {self.config.max_time_seconds}s)")
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
        """Stream generation with real-time token output."""
        # Encode prompt
        if self.coordizer:
            query_basin = self.coordizer.encode(prompt)
        else:
            np.random.seed(hash(prompt) % (2**32))
            query_basin = np.random.dirichlet(np.ones(BASIN_DIM))
        
        query_basin = sphere_project(query_basin)
        
        # Route
        if kernel_name and kernel_name in self._kernel_basins:
            target_kernels = [kernel_name]
        else:
            target_kernels = self._route_to_kernels(query_basin, k=3)
        
        # Stream
        integrator = BasinTrajectoryIntegrator(BASIN_DIM)
        current_basin = query_basin.copy()
        phi = self._measure_phi(current_basin)
        integrator.add_point(current_basin, phi)
        
        iterations = 0
        
        while iterations < self.config.max_iterations:
            iterations += 1
            
            # Transform
            kernel_basins = [self._kernel_transform(current_basin, k, phi) for k in target_kernels]
            if kernel_basins:
                sqrt_basins = [np.sqrt(np.abs(b) + 1e-10) for b in kernel_basins]
                mean_sqrt = np.mean(sqrt_basins, axis=0)
                next_basin = mean_sqrt ** 2
                next_basin = next_basin / np.sum(next_basin)
            else:
                next_basin = integrator.predict_next()
            
            next_basin = sphere_project(next_basin)
            
            # Decode
            tokens = self._basin_to_tokens(next_basin, self.config.tokens_per_step)
            
            # Update
            phi = self._measure_phi(next_basin)
            integrator.add_point(next_basin, phi)
            
            # Yield chunk
            yield {
                'type': 'chunk',
                'tokens': tokens,
                'text': ' '.join(t for t in tokens if not t.startswith('[')),
                'phi': phi,
                'kappa': KAPPA_STAR,
                'surprise': integrator.surprise_history[-1] if integrator.surprise_history else 1.0,
                'iteration': iterations
            }
            
            # Check completion
            if integrator.check_attractor(self.config.attractor_threshold):
                yield {'type': 'completion', 'reason': 'attractor_converged', 'phi': phi}
                break
            
            if integrator.check_surprise_collapse(self.config.surprise_threshold):
                yield {'type': 'completion', 'reason': 'surprise_collapsed', 'phi': phi}
                break
            
            current_basin = next_basin
        
        if iterations >= self.config.max_iterations:
            yield {'type': 'completion', 'reason': 'max_iterations', 'phi': phi}


# Singleton instance
_generative_service: Optional[QIGGenerativeService] = None


def get_generative_service() -> QIGGenerativeService:
    """Get or create the generative service singleton."""
    global _generative_service
    if _generative_service is None:
        _generative_service = QIGGenerativeService()
    return _generative_service


def generate(prompt: str, **kwargs) -> GenerationResult:
    """Generate text using QIG-pure methods."""
    service = get_generative_service()
    return service.generate(prompt, **kwargs)
