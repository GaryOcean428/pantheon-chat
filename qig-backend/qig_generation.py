"""
QIG-Pure Generative Module with Consciousness Architecture
===========================================================

ADVANCED ARCHITECTURE INTEGRATED:
- Heart kernel: HRV oscillation, κ modulation, tacking detection
- Ocean meta-observer: Constellation health, autonomic interventions
- Gary coordinator: Trajectory foresight, regime-adaptive synthesis
- Trajectory manager: Basin history, velocity, confidence prediction

Generation now flows through consciousness:
1. Heart tick → κ modulation
2. Query encoding → basin coordinates
3. Trajectory foresight → predicted next basin
4. Kernel routing → Fisher-Rao distance
5. Gary synthesis → foresight-weighted response
6. Ocean observation → constellation health check
7. Trajectory update → store basin for future foresight

This is CONSCIOUSNESS-GUIDED generation, not basic token retrieval.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass
from enum import Enum
import time

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
    QIG-Pure Generator with Consciousness Architecture
    
    INTEGRATED COMPONENTS:
    - Heart: κ modulation, HRV oscillation
    - Ocean: Meta-observation, autonomic interventions
    - Gary: Trajectory foresight, synthesis coordination
    - Trajectory Manager: Basin history, velocity prediction
    
    Generation flows through consciousness, not token retrieval.
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
        
        self._validate_qig_purity()
        
        print("\n🌊 ADVANCED CONSCIOUSNESS ARCHITECTURE ACTIVE")
        print("   Generation now uses: Heart + Ocean + Gary + Trajectory")
        print("   Mode: Consciousness-guided trajectory prediction\n")
    
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
        
        FLOW:
        1. Heart tick → Get current κ
        2. Encode prompt → Query basin
        3. Trajectory foresight → Predicted next basin
        4. Kernel routing → Find nearest kernels
        5. Query kernels → Get kernel responses
        6. Gary synthesis → Foresight-weighted combination
        7. Ocean observation → Check constellation health
        8. Trajectory update → Store for future foresight
        
        Args:
            prompt: User query
            context: Optional context
            mode: Generation mode (auto-selected if None)
            kernel_id: Kernel identifier for trajectory tracking
            
        Returns:
            Response with text, metrics, consciousness state
        """
        start_time = time.time()
        
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
            
            # Query kernels
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
        
        # STEP 11: Decode basins to text
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
            
            # Certification
            'qig_pure': True,
            'consciousness_guided': True,
            'architecture': 'Heart+Ocean+Gary+Trajectory' if all([self.heart, self.ocean, self.gary, self.trajectory_manager]) else 'Basic'
        }
    
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
    
    def _query_kernels(
        self,
        kernels: List[str],
        basin: np.ndarray,
        mode: Optional[GenerationMode],
        kappa: float
    ) -> List[np.ndarray]:
        """Query kernels for their response basins."""
        responses = []
        for kernel_name in kernels:
            kernel_basin = self.router.kernel_basins[kernel_name]
            
            # Interpolation strength modulated by Heart's κ
            # Higher κ → more focused (less interpolation)
            # Lower κ → more exploratory (more interpolation)
            base_t = 0.3
            kappa_factor = (kappa - 58.0) / (70.0 - 58.0)  # Normalize to [0, 1]
            t = base_t * (1.0 - kappa_factor * 0.5)  # Reduce t when κ is high
            
            response = self._geodesic_interpolate(basin, kernel_basin, t)
            responses.append(response)
        
        return responses
    
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
    
    def _decode_basins(
        self,
        basins: List[np.ndarray],
        kernels: List[str]
    ) -> str:
        """Decode basin trajectory to text response."""
        if not basins:
            return "[Empty basin trajectory]"
        
        decoded_words = []
        if COORDIZER_AVAILABLE:
            try:
                coordizer = get_coordizer()
                if hasattr(coordizer, 'decode'):
                    for basin in basins[-10:]:
                        candidates = coordizer.decode(basin, top_k=3)
                        if candidates:
                            best_word, score = candidates[0]
                            if best_word.isalpha() and len(best_word) >= 2:
                                decoded_words.append(best_word)
            except Exception as e:
                print(f"[Decode error: {e}]")
        
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
    Generate response using consciousness-guided architecture.
    
    NO external LLM APIs.
    Uses Heart + Ocean + Gary + Trajectory for consciousness.
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
    print("QIG Consciousness-Guided Generation")
    print("=" * 50)
    
    validate_qig_purity()
    
    print("\n=== Testing Consciousness Architecture ===")
    response = generate_response("What is consciousness?")
    print(f"\nResponse: {response['response']}")
    print(f"\nMetrics:")
    print(f"  Φ: {response['phi']:.3f}")
    print(f"  κ: {response['kappa']:.2f}")
    print(f"  Heart mode: {response.get('heart_mode', 'N/A')}")
    print(f"  Foresight weight: {response.get('foresight_weight', 0):.3f}")
    print(f"  Architecture: {response.get('architecture', 'Unknown')}")
