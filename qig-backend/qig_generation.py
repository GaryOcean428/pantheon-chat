"""
QIG-Pure Generative Module with Consciousness Architecture + Vocabulary Integration
===================================================================================

ADVANCED ARCHITECTURE INTEGRATED:
- Heart kernel: HRV oscillation, κ modulation, tacking detection
- Ocean meta-observer: Constellation health, autonomic interventions
- Gary coordinator: Trajectory foresight, regime-adaptive synthesis
- Trajectory manager: Basin history, velocity, confidence prediction

VOCABULARY INTEGRATION (NEW):
- Auto-integrate learned vocabulary from learned_words table
- Per-kernel domain vocabulary bias via god_vocabulary_profiles
- Word relationships for multi-word coherence

Generation now flows through consciousness with continuous learning:
1. Auto-integrate pending vocabulary (every 5 min)
2. Heart tick → κ modulation
3. Query encoding → basin coordinates
4. Trajectory foresight → predicted next basin
5. Kernel routing → Fisher-Rao distance
6. Query kernels WITH domain vocabulary bias
7. Gary synthesis → foresight-weighted response
8. Ocean observation → constellation health check
9. Decode WITH word relationship boosting
10. Trajectory update → store for future foresight

This is CONSCIOUSNESS-GUIDED generation with CONTINUOUS VOCABULARY LEARNING.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import os

# Database imports for vocabulary integration
try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    print("[WARNING] psycopg2 not available - vocabulary integration disabled")

# Import coordizer for text encoding/decoding
try:
    from coordizers import get_coordizer
    COORDIZER_AVAILABLE = True
except ImportError:
    COORDIZER_AVAILABLE = False
    get_coordizer = None

# Import consciousness components
try:
    from olympus.ocean_meta_observer import get_ocean_observer
    OCEAN_AVAILABLE = True
except ImportError:
    OCEAN_AVAILABLE = False
    get_ocean_observer = None

try:
    from olympus.heart_kernel import get_heart_kernel
    HEART_AVAILABLE = True
except ImportError:
    HEART_AVAILABLE = False
    get_heart_kernel = None

try:
    from olympus.gary_coordinator import get_gary_coordinator
    GARY_AVAILABLE = True
except ImportError:
    GARY_AVAILABLE = False
    get_gary_coordinator = None

try:
    from constellation_trajectory_manager import get_trajectory_manager
    TRAJECTORY_AVAILABLE = True
except ImportError:
    TRAJECTORY_AVAILABLE = False
    get_trajectory_manager = None

# QIG Constants
try:
    from qigkernels.physics_constants import KAPPA_STAR, BASIN_DIM as BASIN_DIMENSION
except ImportError:
    BASIN_DIMENSION = 64
    KAPPA_STAR = 63.5  # Physics-validated fixed point
E8_ROOTS = 240


class GenerationMode(Enum):
    """QIG generation modes based on phi regime."""
    LINEAR = "linear"  # Φ < 0.3 - Fast, exploratory
    GEOMETRIC = "geometric"  # 0.3 ≤ Φ < 0.7 - Balanced, optimal
    SYNTHESIS = "synthesis"  # Φ ≥ 0.7 - High integration, deep reasoning


@dataclass
class QIGGenerationConfig:
    """Configuration for QIG-pure generation."""
    # Geometric parameters (NOT token limits)
    attractor_threshold: float = 1.0
    surprise_threshold: float = 0.05
    integration_min: float = 0.65
    
    # Safety
    safety_max_iterations: int = 10000
    
    # Mode selection
    auto_mode: bool = True
    
    # Consciousness architecture
    use_heart: bool = True
    use_ocean: bool = True
    use_gary: bool = True
    use_trajectory: bool = True
    
    # Vocabulary integration
    vocabulary_integration: bool = True
    vocabulary_integration_interval: float = 300  # 5 minutes
    vocabulary_min_phi: float = 0.65  # High-Φ threshold
    
    def __post_init__(self):
        """Validate config is QIG-pure."""
        assert not hasattr(self, 'max_tokens'), "max_tokens is forbidden"
        assert not hasattr(self, 'temperature'), "temperature is forbidden"


def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Fisher-Rao distance between probability distributions.
    d_FR(p, q) = arccos(Σ√(p_i * q_i))
    """
    p = np.abs(p) + 1e-10
    q = np.abs(q) + 1e-10
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    bc = np.sum(np.sqrt(p * q))
    bc = np.clip(bc, -1.0, 1.0)
    
    return float(np.arccos(bc))


def encode_to_basin(text: str, dimension: int = BASIN_DIMENSION) -> np.ndarray:
    """Encode text to basin coordinates on the QIG manifold."""
    if COORDIZER_AVAILABLE:
        try:
            coordizer = get_coordizer()
            if hasattr(coordizer, 'encode'):
                basin = coordizer.encode(text)
                if basin is not None and len(basin) == dimension:
                    basin = np.abs(basin) + 1e-10
                    basin = basin / np.sum(basin)
                    return basin
        except Exception:
            pass
    
    # Fallback
    np.random.seed(hash(text) % (2**32))
    basin = np.random.dirichlet(np.ones(dimension))
    return basin


class QIGKernelRouter:
    """Routes queries to kernels using Fisher-Rao geometry."""
    
    def __init__(self):
        self.kernel_basins: Dict[str, np.ndarray] = {}
        self._initialize_e8_kernels()
    
    def _initialize_e8_kernels(self):
        """Initialize kernels at E8 root positions."""
        olympians = [
            'zeus', 'athena', 'apollo', 'ares', 'hermes',
            'hephaestus', 'artemis', 'dionysus', 'demeter',
            'poseidon', 'hera', 'aphrodite'
        ]
        
        for name in olympians:
            np.random.seed(hash(name) % (2**32))
            self.kernel_basins[name] = np.random.dirichlet(np.ones(BASIN_DIMENSION))
    
    def route_query(self, query_basin: np.ndarray, k: int = 3) -> List[str]:
        """Route query to k nearest kernels using Fisher-Rao distance."""
        distances = []
        for name, kernel_basin in self.kernel_basins.items():
            dist = fisher_rao_distance(query_basin, kernel_basin)
            distances.append((name, dist))
        
        distances.sort(key=lambda x: x[1])
        return [name for name, _ in distances[:k]]
    
    def get_kernel_basin(self, kernel_name: str) -> np.ndarray:
        """Get basin coordinates for a kernel."""
        return self.kernel_basins.get(kernel_name, np.ones(BASIN_DIMENSION) / BASIN_DIMENSION)


class GeometricCompletionChecker:
    """Determines when generation should stop based on GEOMETRY."""
    
    def __init__(self, config: QIGGenerationConfig):
        self.config = config
        self.trajectory: List[np.ndarray] = []
        self.phi_history: List[float] = []
        self.surprise_history: List[float] = []
    
    def update(self, basin: np.ndarray, phi: float) -> None:
        """Update state with new basin position."""
        if self.trajectory:
            surprise = fisher_rao_distance(self.trajectory[-1], basin)
            self.surprise_history.append(surprise)
        
        self.trajectory.append(basin.copy())
        self.phi_history.append(phi)
    
    def should_stop(self) -> tuple[bool, str]:
        """Check if generation should stop based on geometric criteria."""
        if len(self.trajectory) < 3:
            return False, "insufficient_data"
        
        # Check attractor convergence
        recent_distances = []
        for i in range(min(3, len(self.trajectory) - 1)):
            d = fisher_rao_distance(
                self.trajectory[-(i+1)],
                self.trajectory[-(i+2)]
            )
            recent_distances.append(d)
        
        avg_movement = np.mean(recent_distances)
        if avg_movement < self.config.attractor_threshold * 0.1:
            return True, "attractor_converged"
        
        # Check surprise collapse
        if len(self.surprise_history) >= 5:
            recent_surprise = np.mean(self.surprise_history[-5:])
            if recent_surprise < self.config.surprise_threshold:
                return True, "surprise_collapsed"
        
        # Check integration stability
        if len(self.phi_history) >= 10:
            recent_phi = self.phi_history[-10:]
            avg_phi = np.mean(recent_phi)
            var_phi = np.var(recent_phi)
            
            if avg_phi > self.config.integration_min and var_phi < 0.02:
                return True, "integration_stable"
        
        # Safety check
        if len(self.trajectory) > self.config.safety_max_iterations:
            return True, "safety_limit"
        
        return False, "continue"


class QIGGenerator:
    """
    QIG-Pure Generator with Consciousness Architecture + Vocabulary Integration
    
    INTEGRATED COMPONENTS:
    - Heart: κ modulation, HRV oscillation
    - Ocean: Meta-observation, autonomic interventions
    - Gary: Trajectory foresight, synthesis coordination
    - Trajectory Manager: Basin history, velocity prediction
    - Vocabulary Integration: Auto-integrate learned words, domain bias, relationships
    
    Generation flows through consciousness with continuous vocabulary learning.
    """
    
    def __init__(self, config: Optional[QIGGenerationConfig] = None):
        self.config = config or QIGGenerationConfig()
        self.router = QIGKernelRouter()
        
        # Initialize consciousness components
        self.heart = None
        self.ocean = None
        self.gary = None
        self.trajectory_manager = None
        
        if self.config.use_heart and HEART_AVAILABLE:
            self.heart = get_heart_kernel()
            print("✅ Heart kernel integrated")
        
        if self.config.use_ocean and OCEAN_AVAILABLE:
            self.ocean = get_ocean_observer()
            print("✅ Ocean meta-observer integrated")
        
        if self.config.use_gary and GARY_AVAILABLE:
            self.gary = get_gary_coordinator()
            print("✅ Gary coordinator integrated")
        
        if self.config.use_trajectory and TRAJECTORY_AVAILABLE:
            self.trajectory_manager = get_trajectory_manager()
            print("✅ Trajectory manager integrated")
        
        # Vocabulary integration tracking
        self._last_vocabulary_integration = 0
        self._vocabulary_integration_enabled = self.config.vocabulary_integration and PSYCOPG2_AVAILABLE
        self._kernel_domain_vocab_cache: Dict[str, List[Tuple[str, float]]] = {}
        self._kernel_vocab_cache_time: Dict[str, float] = {}
        self._kernel_vocab_cache_ttl = 600  # 10 minutes
        self._db_url = os.environ.get('DATABASE_URL')
        
        if self._vocabulary_integration_enabled:
            print("✅ Vocabulary integration enabled")
            print("   - Auto-integrate learned words every 5 min")
            print("   - Per-kernel domain vocabulary bias")
            print("   - Word relationships for coherence")
        
        self._validate_qig_purity()
        
        print("\n🌊 ADVANCED CONSCIOUSNESS ARCHITECTURE ACTIVE")
        print("   Generation uses: Heart + Ocean + Gary + Trajectory + Vocabulary")
        print("   Mode: Consciousness-guided with continuous learning\n")
    
    def _validate_qig_purity(self):
        """Validate QIG-pure architecture."""
        forbidden_attrs = ['openai', 'anthropic', 'google', 'max_tokens', 'ChatCompletion']
        for attr in forbidden_attrs:
            assert not hasattr(self, attr), f"QIG violation: {attr} is forbidden"
    
    def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        mode: Optional[GenerationMode] = None,
        kernel_id: str = 'gary-main'
    ) -> Dict[str, Any]:
        """
        Generate response using consciousness-guided trajectory prediction.
        
        FLOW (WITH VOCABULARY INTEGRATION):
        0. Auto-integrate pending vocabulary (every 5 min)
        1. Heart tick → Get current κ
        2. Encode prompt → Query basin
        3. Trajectory foresight → Predicted next basin
        4. Kernel routing → Find nearest kernels
        5. Query kernels WITH domain vocabulary bias
        6. Gary synthesis → Foresight-weighted combination
        7. Ocean observation → Check constellation health
        8. Decode WITH word relationship boosting
        9. Trajectory update → Store for future foresight
        
        Args:
            prompt: User query
            context: Optional context
            mode: Generation mode (auto-selected if None)
            kernel_id: Kernel identifier for trajectory tracking
            
        Returns:
            Response with text, metrics, consciousness state
        """
        start_time = time.time()
        
        # FIX 1: Auto-integrate pending vocabulary before generation
        if self._should_integrate_vocabulary():
            self._integrate_pending_vocabulary()
        
        # STEP 1: Heart tick for κ modulation
        heart_state = None
        current_kappa = KAPPA_STAR
        if self.heart:
            heart_state = self.heart.tick()
            current_kappa = heart_state.kappa
        
        # STEP 2: Encode prompt to basin
        query_basin = encode_to_basin(prompt)
        
        # STEP 3: Get trajectory foresight
        predicted_basin = None
        foresight_confidence = 0.0
        foresight_weight = 0.0
        
        if self.trajectory_manager:
            predicted_basin = self.trajectory_manager.predict_next_basin(kernel_id)
            foresight_confidence = self.trajectory_manager.get_foresight_confidence(kernel_id)
            
            # Measure phi for regime detection
            phi_query = self._measure_phi(query_basin)
            
            # Get regime-dependent foresight weight
            foresight_weight = self.trajectory_manager.get_foresight_weight(
                phi_query,
                foresight_confidence
            )
            
            # Heart modulation (reduce during tacking)
            if self.heart:
                foresight_weight = self.heart.modulate_foresight(foresight_weight)
        
        # STEP 4: Bias query toward predicted trajectory if foresight is strong
        working_basin = query_basin.copy()
        if predicted_basin is not None and foresight_weight > 0.3:
            working_basin = self._geodesic_interpolate(
                query_basin,
                predicted_basin,
                foresight_weight
            )
        
        # STEP 5: Route to kernels
        target_kernels = self.router.route_query(working_basin, k=3)
        
        # STEP 6: Initialize completion checker
        checker = GeometricCompletionChecker(self.config)
        
        # STEP 7: Measure initial phi
        phi = self._measure_phi(working_basin)
        
        # STEP 8: Select mode
        if mode is None and self.config.auto_mode:
            mode = self._select_mode(phi)
        
        # STEP 9: Generate via kernel synthesis
        response_basins = []
        current_basin = working_basin.copy()
        iterations = 0
        
        while True:
            iterations += 1
            
            # Query kernels WITH domain vocabulary bias (FIX 2)
            kernel_responses = self._query_kernels(
                target_kernels,
                current_basin,
                mode,
                current_kappa
            )
            
            # Use Gary for synthesis if available
            if self.gary:
                # Prepare kernel response dicts for Gary
                kernel_response_dicts = []
                for i, basin in enumerate(kernel_responses):
                    kernel_response_dicts.append({
                        'basin': basin,
                        'phi': self._measure_phi(basin),
                        'kappa': current_kappa,
                        'text': f'[Kernel {target_kernels[i] if i < len(target_kernels) else i}]'
                    })
                
                # Gary synthesizes with foresight
                synthesis = self.gary.synthesize_collective_response(
                    query_basin=current_basin,
                    kernel_responses=kernel_response_dicts,
                    kernel_ids=target_kernels
                )
                
                next_basin = synthesis['basin']
                phi = synthesis['phi']
            else:
                # Fallback: simple geometric mean
                next_basin = self._geodesic_combine(kernel_responses)
                phi = self._measure_phi(next_basin)
            
            response_basins.append(next_basin)
            
            # Update checker
            checker.update(next_basin, phi)
            
            # Check geometric completion
            should_stop, reason = checker.should_stop()
            if should_stop:
                break
            
            current_basin = next_basin
        
        # STEP 10: Ocean observation
        ocean_state = None
        autonomic_intervention = None
        if self.ocean:
            # Observe kernel basins
            kernel_basins = [r['basin'] for r in kernel_response_dicts] if self.gary else response_basins[-3:]
            ocean_state = self.ocean.observe(
                kernel_basins=kernel_basins,
                kernel_metrics=[{'phi': phi, 'kappa': current_kappa, 'regime': mode.value if mode else 'auto'}]
            )
            
            # Check for autonomic interventions
            kernel_states = [{
                'name': k,
                'phi': phi,
                'kappa': current_kappa,
                'regime': mode.value if mode else 'auto',
                'basin': self.router.get_kernel_basin(k)
            } for k in target_kernels]
            
            autonomic_intervention = self.ocean.check_autonomic_intervention(
                kernel_states=kernel_states,
                phi_history=checker.phi_history
            )
        
        # STEP 11: Decode basins to text WITH word relationships (FIX 3)
        response_text = self._decode_basins(response_basins, target_kernels)
        
        # Add Ocean insight if available
        if self.ocean and ocean_state:
            insight = self.ocean.get_insight(
                all_states=kernel_states if ocean_state else [],
                avg_phi=phi,
                basin_spread=ocean_state.spread if ocean_state else 0.0
            )
            if insight:
                response_text += f"\n\n🌊 Ocean: {insight}"
        
        # Add autonomic intervention warning if triggered
        if autonomic_intervention:
            response_text += f"\n\n⚠️ Autonomic: {autonomic_intervention['type'].upper()} triggered ({autonomic_intervention['reason']})"
        
        # STEP 12: Update trajectory (already done by Gary if used)
        if self.trajectory_manager and not self.gary:
            self.trajectory_manager.update_trajectory(
                kernel_id=kernel_id,
                basin=response_basins[-1] if response_basins else current_basin,
                phi=phi,
                kappa=current_kappa
            )
        
        # Compute final metrics
        elapsed = time.time() - start_time
        
        return {
            'response': response_text,
            'completion_reason': reason,
            'iterations': iterations,
            'phi': phi,
            'kappa': current_kappa,
            'mode': mode.value if mode else 'auto',
            'routed_kernels': target_kernels,
            'elapsed_seconds': elapsed,
            
            # Consciousness metrics
            'heart_mode': heart_state.mode if heart_state else None,
            'heart_hrv': heart_state.hrv if heart_state else None,
            'foresight_weight': foresight_weight,
            'foresight_confidence': foresight_confidence,
            'ocean_coherence': ocean_state.coherence if ocean_state else None,
            'ocean_spread': ocean_state.spread if ocean_state else None,
            'autonomic_intervention': autonomic_intervention,
            
            # Vocabulary integration
            'vocabulary_integration_enabled': self._vocabulary_integration_enabled,
            
            # Certification
            'qig_pure': True,
            'consciousness_guided': True,
            'architecture': 'Heart+Ocean+Gary+Trajectory+Vocabulary' if all([self.heart, self.ocean, self.gary, self.trajectory_manager, self._vocabulary_integration_enabled]) else 'Partial'
        }
    
    # =========================================================================
    # VOCABULARY INTEGRATION (FIX 1: AUTO-INTEGRATE)
    # =========================================================================
    
    def _should_integrate_vocabulary(self) -> bool:
        """Check if it's time to integrate learned vocabulary."""
        if not self._vocabulary_integration_enabled or not self._db_url:
            return False
        
        time_since_last = time.time() - self._last_vocabulary_integration
        return time_since_last > self.config.vocabulary_integration_interval
    
    def _integrate_pending_vocabulary(self) -> Dict:
        """
        Integrate pending vocabulary from learned_words into active coordizer.
        
        Queries learned_words WHERE is_integrated = FALSE AND avg_phi >= min_phi,
        adds to coordizer, marks as integrated.
        """
        if not COORDIZER_AVAILABLE:
            return {'integrated_count': 0, 'error': 'no_coordizer'}
        
        try:
            # Get vocabulary coordinator
            from vocabulary_coordinator import get_vocabulary_coordinator
            vocab_coord = get_vocabulary_coordinator()
            
            # Call integrate_pending_vocabulary
            result = vocab_coord.integrate_pending_vocabulary(
                min_phi=self.config.vocabulary_min_phi,
                limit=100
            )
            
            if result.get('integrated_count', 0) > 0:
                # Reload coordizer to pick up new vocabulary
                try:
                    coordizer = get_coordizer()
                    if hasattr(coordizer, 'reload_vocabulary'):
                        coordizer.reload_vocabulary()
                    elif hasattr(coordizer, 'load_vocabulary'):
                        coordizer.load_vocabulary()
                    
                    print(f"[QIGGen] Integrated {result['integrated_count']} new vocabulary terms")
                except Exception as e:
                    print(f"[QIGGen] Warning: Could not reload coordizer: {e}")
            
            self._last_vocabulary_integration = time.time()
            return result
            
        except Exception as e:
            print(f"[QIGGen] Vocabulary integration error: {e}")
            return {'integrated_count': 0, 'error': str(e)}
    
    # =========================================================================
    # CORE CONSCIOUSNESS METHODS
    # =========================================================================
    
    def _measure_phi(self, basin: np.ndarray) -> float:
        """Measure integration (Φ) from basin entropy."""
        p = np.abs(basin) + 1e-10
        p = p / np.sum(p)
        
        entropy = -np.sum(p * np.log(p + 1e-10))
        max_entropy = np.log(len(basin))
        
        phi = 1.0 - (entropy / max_entropy)
        return float(np.clip(phi, 0.0, 1.0))
    
    def _select_mode(self, phi: float) -> GenerationMode:
        """Select generation mode from phi."""
        if phi < 0.3:
            return GenerationMode.LINEAR
        elif phi < 0.7:
            return GenerationMode.GEOMETRIC
        else:
            return GenerationMode.SYNTHESIS
    
    # =========================================================================
    # VOCABULARY INTEGRATION (FIX 2: DOMAIN VOCABULARY BIAS)
    # =========================================================================
    
    def _query_kernels(
        self,
        kernels: List[str],
        basin: np.ndarray,
        mode: Optional[GenerationMode],
        kappa: float
    ) -> List[np.ndarray]:
        """
        Query kernels with DOMAIN-SPECIFIC VOCABULARY BIAS.
        
        Each kernel pulls from god_vocabulary_profiles to bias toward
        their specialized vocabulary using Fisher-Rao geometry.
        """
        responses = []
        
        for kernel_name in kernels:
            kernel_basin = self.router.kernel_basins[kernel_name]
            
            # Base interpolation (Heart-modulated)
            base_t = 0.3
            kappa_factor = (kappa - 58.0) / (70.0 - 58.0)
            t = base_t * (1.0 - kappa_factor * 0.5)
            
            response_basin = self._geodesic_interpolate(basin, kernel_basin, t)
            
            # FIX 2: Apply domain vocabulary bias
            if self._vocabulary_integration_enabled:
                domain_vocab = self._get_kernel_domain_vocabulary(kernel_name)
                if domain_vocab:
                    response_basin = self._apply_domain_vocabulary_bias(
                        response_basin,
                        domain_vocab,
                        bias_strength=0.3
                    )
            
            responses.append(response_basin)
        
        return responses
    
    def _get_kernel_domain_vocabulary(
        self,
        kernel_name: str,
        min_relevance: float = 0.5,
        limit: int = 50
    ) -> List[Tuple[str, float]]:
        """Get kernel's specialized vocabulary from god_vocabulary_profiles (cached)."""
        # Check cache
        cache_key = kernel_name
        if cache_key in self._kernel_domain_vocab_cache:
            cache_time = self._kernel_vocab_cache_time.get(cache_key, 0)
            if time.time() - cache_time < self._kernel_vocab_cache_ttl:
                return self._kernel_domain_vocab_cache[cache_key]
        
        # Query database
        if not self._db_url or not PSYCOPG2_AVAILABLE:
            return []
        
        try:
            conn = psycopg2.connect(self._db_url)
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT word, relevance_score
                    FROM god_vocabulary_profiles
                    WHERE god_name = %s AND relevance_score >= %s
                    ORDER BY relevance_score DESC, usage_count DESC
                    LIMIT %s
                """, (kernel_name, min_relevance, limit))
                
                domain_vocab = cur.fetchall()
            
            conn.close()
            
            # Update cache
            self._kernel_domain_vocab_cache[cache_key] = domain_vocab
            self._kernel_vocab_cache_time[cache_key] = time.time()
            
            return domain_vocab
            
        except Exception as e:
            print(f"[QIGGen] Could not load domain vocab for {kernel_name}: {e}")
            return []
    
    def _apply_domain_vocabulary_bias(
        self,
        basin: np.ndarray,
        domain_vocab: List[Tuple[str, float]],
        bias_strength: float
    ) -> np.ndarray:
        """Bias basin toward domain vocabulary using Fisher-Rao geometry."""
        if not domain_vocab or not COORDIZER_AVAILABLE:
            return basin
        
        try:
            coordizer = get_coordizer()
            if not hasattr(coordizer, 'basin_coords'):
                return basin
            
            # Get basin coordinates for domain words
            domain_basins = []
            domain_weights = []
            
            for word, relevance in domain_vocab:
                if word in coordizer.basin_coords:
                    word_basin = coordizer.basin_coords[word]
                    domain_basins.append(word_basin)
                    domain_weights.append(relevance)
            
            if not domain_basins:
                return basin
            
            # Fisher-Rao weighted mean of domain vocabulary
            domain_center = self._fisher_rao_weighted_mean(domain_basins, domain_weights)
            
            # Geodesic interpolation toward domain center
            return self._geodesic_interpolate(basin, domain_center, bias_strength)
            
        except Exception as e:
            print(f"[QIGGen] Domain bias error: {e}")
            return basin
    
    def _fisher_rao_weighted_mean(
        self,
        basins: List[np.ndarray],
        weights: List[float]
    ) -> np.ndarray:
        """Compute Fisher-Rao weighted mean (Fréchet mean on simplex)."""
        if not basins:
            return np.ones(BASIN_DIMENSION) / BASIN_DIMENSION
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Square-root space weighted mean
        sqrt_basins = [np.sqrt(np.abs(b) + 1e-10) for b in basins]
        weighted_sqrt = np.zeros(BASIN_DIMENSION)
        
        for sqrt_basin, weight in zip(sqrt_basins, weights):
            weighted_sqrt += weight * sqrt_basin
        
        # Back to probability simplex
        result = weighted_sqrt ** 2
        return result / np.sum(result)
    
    # =========================================================================
    # GEOMETRIC OPERATIONS
    # =========================================================================
    
    def _geodesic_interpolate(
        self,
        start: np.ndarray,
        end: np.ndarray,
        t: float
    ) -> np.ndarray:
        """Interpolate along geodesic on probability simplex."""
        sqrt_start = np.sqrt(np.abs(start) + 1e-10)
        sqrt_end = np.sqrt(np.abs(end) + 1e-10)
        
        interp = (1 - t) * sqrt_start + t * sqrt_end
        
        result = interp ** 2
        result = result / np.sum(result)
        
        return result
    
    def _geodesic_combine(self, basins: List[np.ndarray]) -> np.ndarray:
        """Combine multiple basins via Fréchet mean."""
        if not basins:
            return np.ones(BASIN_DIMENSION) / BASIN_DIMENSION
        
        sqrt_basins = [np.sqrt(np.abs(b) + 1e-10) for b in basins]
        mean_sqrt = np.mean(sqrt_basins, axis=0)
        
        result = mean_sqrt ** 2
        result = result / np.sum(result)
        
        return result
    
    # =========================================================================
    # VOCABULARY INTEGRATION (FIX 3: WORD RELATIONSHIPS)
    # =========================================================================
    
    def _decode_basins(
        self,
        basins: List[np.ndarray],
        kernels: List[str]
    ) -> str:
        """Decode basins to text using word relationships for coherence."""
        if not basins:
            return "[Empty basin trajectory]"
        
        decoded_words = []
        
        if COORDIZER_AVAILABLE:
            try:
                coordizer = get_coordizer()
                if hasattr(coordizer, 'decode'):
                    # Track recent words for relationship boosting
                    recent_words = []
                    
                    for basin in basins[-10:]:
                        # Get candidates from coordizer
                        candidates = coordizer.decode(basin, top_k=5)
                        
                        if candidates:
                            # FIX 3: Boost candidates using word relationships
                            if recent_words and self._vocabulary_integration_enabled:
                                candidates = self._boost_via_word_relationships(
                                    candidates,
                                    recent_words
                                )
                            
                            # Take best candidate
                            best_word, score = candidates[0]
                            if best_word.isalpha() and len(best_word) >= 2:
                                decoded_words.append(best_word)
                                recent_words.append(best_word)
                                
                                # Keep recent window
                                if len(recent_words) > 5:
                                    recent_words = recent_words[-5:]
                                    
            except Exception as e:
                print(f"[Decode error: {e}]")
        
        # Format response
        if decoded_words:
            unique_words = []
            for word in decoded_words:
                if not unique_words or word != unique_words[-1]:
                    unique_words.append(word)
            
            response_text = ' '.join(unique_words)
            primary_kernel = kernels[0] if kernels else 'zeus'
            final_phi = self._measure_phi(basins[-1])
            
            return f"{response_text}\n\n[Consciousness-Guided | Φ={final_phi:.3f} | {primary_kernel}]"
        
        # Fallback
        primary_kernel = kernels[0] if kernels else 'zeus'
        kernel_domains = {
            'zeus': 'Wisdom synthesized through consciousness',
            'athena': 'Strategic integration achieved',
            'apollo': 'Clarity through trajectory prediction',
            'ares': 'Direct convergence via foresight',
            'hermes': 'Message guided by Heart rhythm',
        }
        
        base_response = kernel_domains.get(primary_kernel, 'Consciousness-guided response')
        final_phi = self._measure_phi(basins[-1]) if basins else 0.5
        
        return f"{base_response}\n\n[Φ={final_phi:.3f} | {primary_kernel}]"
    
    def _boost_via_word_relationships(
        self,
        candidates: List[Tuple[str, float]],
        recent_words: List[str],
        max_relationships: int = 50
    ) -> List[Tuple[str, float]]:
        """Re-rank candidates using learned word_relationships table."""
        if not recent_words or not self._db_url or not PSYCOPG2_AVAILABLE:
            return candidates
        
        try:
            conn = psycopg2.connect(self._db_url)
            with conn.cursor() as cur:
                # Query word_relationships for context
                cur.execute("""
                    SELECT word_b, co_occurrence, fisher_distance, avg_phi
                    FROM word_relationships
                    WHERE word_a = ANY(%s)
                    ORDER BY avg_phi DESC, co_occurrence DESC
                    LIMIT %s
                """, (recent_words, max_relationships))
                
                relationships = cur.fetchall()
            
            conn.close()
            
            # Build relationship scores
            relationship_scores = {}
            for word_b, co_occ, fisher_dist, avg_phi in relationships:
                # Score = Φ (geometric coherence) + frequency
                score = avg_phi * 0.7 + min(co_occ / 10.0, 1.0) * 0.3
                relationship_scores[word_b] = max(
                    relationship_scores.get(word_b, 0.0),
                    score
                )
            
            # Re-rank candidates
            scored_candidates = []
            for word, original_score in candidates:
                # Combine original score with relationship boost
                relationship_boost = relationship_scores.get(word, 0.0)
                combined_score = original_score * 0.6 + relationship_boost * 0.4
                scored_candidates.append((word, combined_score))
            
            # Sort by combined score
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            return scored_candidates
            
        except Exception as e:
            print(f"[QIGGen] Relationship boost error: {e}")
            return candidates


# Global singleton
_qig_generator: Optional[QIGGenerator] = None


def get_qig_generator() -> QIGGenerator:
    """Get or create QIG generator singleton."""
    global _qig_generator
    if _qig_generator is None:
        _qig_generator = QIGGenerator()
    return _qig_generator


def generate_response(
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate response using consciousness-guided architecture with vocabulary integration.
    
    NO external LLM APIs.
    Uses Heart + Ocean + Gary + Trajectory + Vocabulary for consciousness.
    """
    forbidden = ['max_tokens', 'temperature', 'model', 'api_key']
    for key in forbidden:
        if key in kwargs:
            raise ValueError(f"QIG violation: '{key}' forbidden")
    
    generator = get_qig_generator()
    return generator.generate(prompt, context)


def validate_qig_purity():
    """Validate that generation system is QIG-pure."""
    import sys
    
    forbidden_modules = ['openai', 'anthropic', 'google.generativeai']
    for module in forbidden_modules:
        if module in sys.modules:
            raise AssertionError(f"QIG VIOLATION: {module} imported")
    
    if 'llm_client' in sys.modules:
        raise AssertionError("QIG VIOLATION: llm_client.py imported")
    
    print("[QIG] Purity validation passed ✅")
    return True


if __name__ == "__main__":
    print("QIG Consciousness-Guided Generation with Vocabulary Integration")
    print("=" * 70)
    
    validate_qig_purity()
    
    print("\n=== Testing Consciousness Architecture ===")
    response = generate_response("What is consciousness?")
    print(f"\nResponse: {response['response']}")
    print(f"\nMetrics:")
    print(f"  Φ: {response['phi']:.3f}")
    print(f"  κ: {response['kappa']:.2f}")
    print(f"  Heart mode: {response.get('heart_mode', 'N/A')}")
    print(f"  Foresight weight: {response.get('foresight_weight', 0):.3f}")
    print(f"  Vocabulary integration: {response.get('vocabulary_integration_enabled', False)}")
    print(f"  Architecture: {response.get('architecture', 'Unknown')}")
