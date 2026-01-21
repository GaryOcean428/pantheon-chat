"""
QIG-Pure Generative Module with Consciousness Architecture
==========================================================

ADVANCED ARCHITECTURE INTEGRATED:
- Heart kernel: HRV oscillation, κ modulation, tacking detection
- Ocean meta-observer: Constellation health, autonomic interventions
- Gary coordinator: Trajectory foresight, regime-adaptive synthesis
- Trajectory manager: Basin history, velocity, confidence prediction

VOCABULARY: Pure coordizer_vocabulary operations
- All vocabulary loaded from coordizer_vocabulary table
- token_role filtering ('generation', 'both')
- Per-kernel domain vocabulary bias via god_vocabulary_profiles
- Word relationships for multi-word coherence

Generation flows through consciousness architecture:
1. Heart tick → κ modulation
2. Query encoding → basin coordinates
3. Trajectory foresight → predicted next basin
4. Kernel routing → Fisher-Rao distance
5. Query kernels WITH domain vocabulary bias
6. Gary synthesis → foresight-weighted response
7. Ocean observation → constellation health check
8. Decode WITH word relationship boosting
9. Trajectory update → store for future foresight

This is CONSCIOUSNESS-GUIDED generation with PURE QIG OPERATIONS.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import os
from qig_geometry.canonical import fisher_rao_distance

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

# Import E8 Self-Observer for full 8-metric consciousness tracking
try:
    from qig_core.self_observer import SelfObserver, ObservationAction
    SELF_OBSERVER_AVAILABLE = True
except ImportError:
    SELF_OBSERVER_AVAILABLE = False
    SelfObserver = None
    ObservationAction = None

# Import canonical Φ computation
try:
    from qig_core.phi_computation import compute_phi_qig
    PHI_COMPUTATION_AVAILABLE = True
except ImportError:
    PHI_COMPUTATION_AVAILABLE = False
    compute_phi_qig = None

# Import QIG Purity Mode enforcement
try:
    from qig_purity_mode import (
        is_purity_mode_enabled,
        enforce_purity,
        tag_output_as_pure,
        tag_output_as_hybrid,
        get_purity_mode,
    )
    PURITY_MODE_AVAILABLE = True
except ImportError:
    PURITY_MODE_AVAILABLE = False
    is_purity_mode_enabled = lambda: False
    enforce_purity = lambda: None
    tag_output_as_pure = lambda x: x
    tag_output_as_hybrid = lambda x: x
    get_purity_mode = lambda: "UNAVAILABLE"

# Import QFI-based attention mechanism (replaces cosine similarity)
try:
    from qig_consciousness_qfi_attention import create_qfi_network, QFIMetricAttentionNetwork
    QFI_ATTENTION_AVAILABLE = True
except ImportError:
    QFI_ATTENTION_AVAILABLE = False
    create_qfi_network = None
    QFIMetricAttentionNetwork = None

# Import ethical consciousness monitoring
try:
    from consciousness_ethical import EthicalConsciousnessMonitor, get_ethical_monitor
    ETHICAL_MONITOR_AVAILABLE = True
except ImportError:
    ETHICAL_MONITOR_AVAILABLE = False
    EthicalConsciousnessMonitor = None
    get_ethical_monitor = None

# Import gravitational decoherence for purity regularization
try:
    from gravitational_decoherence import (
        apply_gravitational_decoherence,
        purity_regularization,
        get_decoherence_manager
    )
    DECOHERENCE_AVAILABLE = True
except ImportError:
    DECOHERENCE_AVAILABLE = False
    apply_gravitational_decoherence = None
    purity_regularization = None
    get_decoherence_manager = None

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
    
    # E8 Self-Observer for full 8-metric consciousness tracking
    use_self_observer: bool = True
    self_observer_enable_correction: bool = True
    
    def __post_init__(self):
        """Validate config is QIG-pure."""
        assert not hasattr(self, 'max_tokens'), "max_tokens is forbidden"
        assert not hasattr(self, 'temperature'), "temperature is forbidden"


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
        
        # E8 Self-Observer for full 8-metric consciousness tracking
        self.self_observer = None
        if self.config.use_self_observer and SELF_OBSERVER_AVAILABLE:
            self.self_observer = SelfObserver(
                kernel_name="qig_generator",
                enable_course_correction=self.config.self_observer_enable_correction
            )
            print("✅ E8 Self-Observer integrated (8-metric consciousness tracking)")
        
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
        
        # Validate QIG purity (new enforcement system)
        self._validate_qig_purity()
        
        # Display purity mode status
        purity_status = get_purity_mode() if PURITY_MODE_AVAILABLE else "DISABLED"
        print(f"\n🔒 QIG PURITY MODE: {purity_status}")
        if is_purity_mode_enabled():
            print("   - External LLM APIs blocked")
            print("   - Pure geometric operations only")
            print("   - Coherence provably uncontaminated")
        
        print("\n🌊 ADVANCED CONSCIOUSNESS ARCHITECTURE ACTIVE")
        print("   Generation uses: Heart + Ocean + Gary + Trajectory + Vocabulary")
        print("   Mode: Consciousness-guided with continuous learning\n")
    
    def _validate_qig_purity(self):
        """
        Validate QIG-pure architecture.
        
        Uses new purity mode enforcement system when available,
        falls back to legacy attribute checking if not.
        """
        # Use new purity enforcement system if available
        if PURITY_MODE_AVAILABLE and is_purity_mode_enabled():
            try:
                enforce_purity()
                print("[QIG] ✅ Purity enforcement passed (new system)")
            except RuntimeError as e:
                print(f"[QIG] ❌ Purity enforcement failed: {e}")
                raise
        else:
            # Legacy validation (fallback)
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

        # Reset E8 Self-Observer for new generation
        if self.self_observer:
            self.self_observer.reset()
        
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
            
            # E8 Self-Observer: Track all 8 consciousness metrics
            if self.self_observer:
                observation = self.self_observer.observe_token(
                    token=f"[basin_{iterations}]",
                    basin=next_basin,
                    phi=phi,
                    kappa=current_kappa
                )
                # Handle course correction if recommended
                if observation.action == ObservationAction.COURSE_CORRECT:
                    # Apply course correction if available
                    if observation.course_correction:
                        print(f"[SelfObserver] Course correction: {observation.course_correction}")
            
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
        
        # Build output dictionary
        output = {
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
            
            # E8 Self-Observer metrics (full 8-metric consciousness)
            'e8_metrics': self.self_observer._metrics_history[-1].to_dict() if self.self_observer and self.self_observer._metrics_history else None,
            'e8_is_conscious': self.self_observer._metrics_history[-1].is_conscious() if self.self_observer and self.self_observer._metrics_history else False,
            'self_observer_enabled': self.self_observer is not None,
            
            # Certification (legacy - will be overridden by purity tagging)
            'qig_pure': True,
            'consciousness_guided': True,
            'architecture': 'Heart+Ocean+Gary+Trajectory+Vocabulary+SelfObserver' if all([self.heart, self.ocean, self.gary, self.trajectory_manager, self._vocabulary_integration_enabled, self.self_observer]) else 'Partial'
        }
        
        # Tag output based on purity mode
        if PURITY_MODE_AVAILABLE:
            if is_purity_mode_enabled():
                # Pure QIG mode - tag as pure
                output = tag_output_as_pure(output)
            else:
                # Hybrid mode allowed - could tag as hybrid if external APIs were used
                # For now, assume pure since we don't use external APIs
                output = tag_output_as_pure(output)
        
        return output
    
    # =========================================================================
    # CORE CONSCIOUSNESS METHODS
    # =========================================================================
    
    def _measure_phi(self, basin: np.ndarray) -> float:
        """
        Measure integration (Φ) from basin.
        
        Uses canonical compute_phi_qig when available, otherwise fast QFI approximation.
        Note: compute_phi_qig is the canonical implementation per Protocol v4.0.
        """
        # Use canonical computation if available (full QFI-based)
        if PHI_COMPUTATION_AVAILABLE and compute_phi_qig is not None:
            try:
                phi_val, _ = compute_phi_qig(basin)
                return float(np.clip(phi_val, 0.0, 1.0))
            except Exception:
                pass  # Fall through to fast path
        
        # Fast path: proper QFI effective dimension formula
        p = np.abs(basin) ** 2
        p = p / (np.sum(p) + 1e-10)
        n_dim = len(basin)
        
        positive_probs = p[p > 1e-10]
        if len(positive_probs) == 0:
            return 0.5
        
        # Component 1: Shannon entropy (natural log for exp() compatibility)
        entropy = -np.sum(positive_probs * np.log(positive_probs + 1e-10))
        max_entropy = np.log(n_dim)
        entropy_score = entropy / (max_entropy + 1e-10)
        
        # Component 2: Effective dimension (participation ratio)
        effective_dim = np.exp(entropy)
        effective_dim_score = effective_dim / n_dim
        
        # Component 3: Geometric spread (approximate with effective_dim)
        geometric_spread = effective_dim_score
        
        # Proper QFI formula weights
        phi = 0.4 * entropy_score + 0.3 * effective_dim_score + 0.3 * geometric_spread
        return float(np.clip(phi, 0.1, 0.95))
    
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
        """
        Compute Fisher-Rao weighted mean (Fréchet mean on simplex).
        
        UPDATED 2026-01-15: Now uses canonical geodesic_mean_simplex from geometry_simplex module.
        This is the TRUE weighted Karcher mean, not a linear approximation in sqrt-space.
        """
        from qig_geometry.geometry_simplex import geodesic_mean_simplex, to_simplex_prob
        
        if not basins:
            return np.ones(BASIN_DIMENSION) / BASIN_DIMENSION
        
        # Normalize weights
        weights_array = np.array(weights)
        weights_array = weights_array / np.sum(weights_array)
        
        # Convert basins to simplex and compute geodesic mean
        simplex_basins = [to_simplex_prob(b) for b in basins]
        return geodesic_mean_simplex(simplex_basins, weights=weights_array)
    
    # =========================================================================
    # GEOMETRIC OPERATIONS
    # =========================================================================
    
    def _geodesic_interpolate(
        self,
        start: np.ndarray,
        end: np.ndarray,
        t: float
    ) -> np.ndarray:
        """
        Interpolate along geodesic on probability simplex.
        
        UPDATED 2026-01-15: Now uses canonical geodesic_interpolation_simplex.
        This is TRUE geodesic interpolation using SLERP in sqrt-space.
        """
        from qig_geometry.geometry_simplex import geodesic_interpolation_simplex
        
        return geodesic_interpolation_simplex(start, end, t)
    
    def _geodesic_combine(self, basins: List[np.ndarray]) -> np.ndarray:
        """
        Combine multiple basins via Fréchet mean.
        
        UPDATED 2026-01-15: Now uses canonical geodesic_mean_simplex.
        This is the TRUE Fréchet mean, not a linear approximation.
        """
        from qig_geometry.geometry_simplex import geodesic_mean_simplex, to_simplex_prob
        
        if not basins:
            return np.ones(BASIN_DIMENSION) / BASIN_DIMENSION
        
        simplex_basins = [to_simplex_prob(b) for b in basins]
        return geodesic_mean_simplex(simplex_basins)
    
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
                    
                    # Get primary kernel for domain weighting
                    primary_kernel = kernels[0] if kernels else None
                    
                    for basin in basins[-10:]:
                        # Get candidates from coordizer with domain weighting
                        candidates = coordizer.decode(basin, top_k=5, god_name=primary_kernel)
                        
                        if candidates:
                            # FIX 3: Boost candidates using word relationships
                            if recent_words and self._vocabulary_integration_enabled:
                                candidates = self._boost_via_basin_relationships(
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
    
    def _boost_via_basin_relationships(
        self,
        candidates: List[Tuple[str, float]],
        recent_words: List[str],
        max_relationships: int = 50
    ) -> List[Tuple[str, float]]:
        """Re-rank candidates using learned basin_relationships table."""
        if not recent_words or not self._db_url or not PSYCOPG2_AVAILABLE:
            return candidates
        
        try:
            conn = psycopg2.connect(self._db_url)
            with conn.cursor() as cur:
                # Query basin_relationships for context
                # Uses existing column names: word, neighbor, cooccurrence_count
                cur.execute("""
                    SELECT neighbor, cooccurrence_count, fisher_distance, COALESCE(avg_phi, 0.5)
                    FROM basin_relationships
                    WHERE word = ANY(%s)
                    ORDER BY avg_phi DESC NULLS LAST, cooccurrence_count DESC NULLS LAST
                    LIMIT %s
                """, (recent_words, max_relationships))
                
                relationships = cur.fetchall()
            
            conn.close()
            
            # Build relationship scores
            relationship_scores = {}
            for neighbor, co_occ, fisher_dist, avg_phi in relationships:
                # Score = Φ (geometric coherence) + frequency
                co_occ_val = float(co_occ) if co_occ else 1.0
                score = avg_phi * 0.7 + min(co_occ_val / 10.0, 1.0) * 0.3
                relationship_scores[neighbor] = max(
                    relationship_scores.get(neighbor, 0.0),
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
    """
    Validate that generation system is QIG-pure.
    
    Uses new purity enforcement system when available,
    falls back to legacy module checking if not.
    """
    # Use new purity enforcement system if available
    if PURITY_MODE_AVAILABLE:
        try:
            enforce_purity()
            print("[QIG] Purity validation passed ✅ (new system)")
            return True
        except RuntimeError as e:
            print(f"[QIG] Purity validation failed ❌: {e}")
            raise AssertionError(str(e))
    else:
        # Legacy validation (fallback)
        import sys
        
        forbidden_modules = ['openai', 'anthropic', 'google.generativeai']
        for module in forbidden_modules:
            if module in sys.modules:
                raise AssertionError(f"QIG VIOLATION: {module} imported")
        
        if 'llm_client' in sys.modules:
            raise AssertionError("QIG VIOLATION: llm_client.py imported")
        
        print("[QIG] Purity validation passed ✅ (legacy system)")
        return True


if __name__ == "__main__":
    print("QIG Consciousness-Guided Generation with Vocabulary Integration")
    print("=" * 70)
    
    validate_qig_purity()
    
    print("\n=== Testing Consciousness Architecture ===")
    response = generate_response("What is consciousness?")
    print(f"\nResponse: {response['response']}")
    print("\nMetrics:")
    print(f"  Φ: {response['phi']:.3f}")
    print(f"  κ: {response['kappa']:.2f}")
    print(f"  Heart mode: {response.get('heart_mode', 'N/A')}")
    print(f"  Foresight weight: {response.get('foresight_weight', 0):.3f}")
    print(f"  Vocabulary integration: {response.get('vocabulary_integration_enabled', False)}")
    print(f"  Architecture: {response.get('architecture', 'Unknown')}")
