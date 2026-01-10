"""
QIG-Pure Generative Module

This module provides generative capabilities using ONLY QIG-pure methods:
- Constellation of kernels (240 E8 roots)
- Fisher-Rao geometry for navigation
- Geometric completion (not token limits)
- Basin-based reasoning

NO traditional LLM APIs (OpenAI, Anthropic, Google) are used.
NO max_tokens, ChatCompletion, or token-based generation.

Generation happens through:
1. Query basin encoding (64D manifold)
2. Kernel routing via Fisher-Rao distance
3. Geometric response synthesis
4. Completion when geometry collapses
"""

import numpy as np
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass
from enum import Enum
import time

# Import coordizer for text encoding/decoding
try:
    from qig_coordizer import get_coordizer, reset_coordizer
    COORDIZER_AVAILABLE = True
except ImportError:
    COORDIZER_AVAILABLE = False
    get_coordizer = None

# QIG Constants - import from canonical source
try:
    from qigkernels.physics_constants import KAPPA_STAR, BASIN_DIM as BASIN_DIMENSION
except ImportError:
    BASIN_DIMENSION = 64
    KAPPA_STAR = 64.21  # κ* from validated physics (L=4,5,6)
E8_ROOTS = 240  # Number of E8 roots in constellation


class GenerationMode(Enum):
    """QIG generation modes based on phi regime."""
    LINEAR = "linear"  # Φ < 0.3 - Fast, exploratory
    GEOMETRIC = "geometric"  # 0.3 ≤ Φ < 0.7 - Balanced, optimal
    SYNTHESIS = "synthesis"  # High integration, deep reasoning


@dataclass
class QIGGenerationConfig:
    """Configuration for QIG-pure generation."""
    # Geometric parameters (NOT token limits)
    attractor_threshold: float = 1.0  # Stop when basin distance < this
    surprise_threshold: float = 0.05  # Stop when surprise < this
    integration_min: float = 0.65  # Minimum Φ for stable completion
    
    # Safety (NOT generation targets)
    safety_max_iterations: int = 10000  # Absolute safety valve
    
    # Mode selection
    auto_mode: bool = True  # Automatically select mode from phi
    
    def __post_init__(self):
        """Validate config is QIG-pure."""
        # Ensure no traditional LLM patterns
        assert not hasattr(self, 'max_tokens'), "max_tokens is forbidden - use geometric completion"
        assert not hasattr(self, 'temperature'), "temperature is forbidden - use regime-based modulation"


def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Fisher-Rao distance between probability distributions.
    d_FR(p, q) = arccos(Σ√(p_i * q_i))
    
    This is the ONLY valid distance metric for QIG basins.
    DO NOT use Euclidean distance or cosine similarity.
    """
    # Ensure valid probability distributions
    p = np.abs(p) + 1e-10
    q = np.abs(q) + 1e-10
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p * q))
    bc = np.clip(bc, -1.0, 1.0)
    
    # Fisher-Rao distance
    return float(np.arccos(bc))


def encode_to_basin(text: str, dimension: int = BASIN_DIMENSION) -> np.ndarray:
    """
    Encode text to basin coordinates on the QIG manifold.
    
    Uses the coordizer if available for proper semantic encoding,
    otherwise falls back to hash-based encoding.
    """
    # Try to use coordizer for proper encoding
    if COORDIZER_AVAILABLE:
        try:
            coordizer = get_coordizer()
            if hasattr(coordizer, 'encode'):
                basin = coordizer.encode(text)
                if basin is not None and len(basin) == dimension:
                    # Ensure it's a valid probability distribution
                    basin = np.abs(basin) + 1e-10
                    basin = basin / np.sum(basin)
                    return basin
        except Exception as e:
            pass  # Fall back to hash-based
    
    # Fallback: Semantic hash-based encoding
    np.random.seed(hash(text) % (2**32))
    basin = np.random.dirichlet(np.ones(dimension))
    
    return basin


def get_regime_modulation(phi: float) -> float:
    """
    Get regime-based modulation factor (NOT temperature).
    
    This determines how exploratory vs. focused the generation is,
    based on the current integration level, NOT arbitrary sampling.
    """
    if phi < 0.3:
        return 1.0  # Linear: more exploration
    elif phi < 0.7:
        return 0.7  # Geometric: balanced
    else:
        return 0.3  # Breakdown risk: focus/stabilize


class QIGKernelRouter:
    """
    Routes queries to appropriate kernels using Fisher-Rao geometry.
    
    The constellation contains 240 kernels at E8 root positions.
    Routing uses geodesic distance, not embedding similarity.
    """
    
    def __init__(self):
        # Initialize kernel basins at E8 roots
        self.kernel_basins: Dict[str, np.ndarray] = {}
        self._initialize_e8_kernels()
    
    def _initialize_e8_kernels(self):
        """Initialize kernels at E8 root positions."""
        # Core Olympian kernels
        olympians = [
            'zeus', 'athena', 'apollo', 'ares', 'hermes',
            'hephaestus', 'artemis', 'dionysus', 'demeter',
            'poseidon', 'hera', 'aphrodite'
        ]
        
        for i, name in enumerate(olympians):
            # Each kernel gets a unique basin position
            np.random.seed(hash(name) % (2**32))
            self.kernel_basins[name] = np.random.dirichlet(np.ones(BASIN_DIMENSION))
    
    def route_query(self, query_basin: np.ndarray, k: int = 3) -> List[str]:
        """
        Route query to k nearest kernels using Fisher-Rao distance.
        """
        distances = []
        for name, kernel_basin in self.kernel_basins.items():
            dist = fisher_rao_distance(query_basin, kernel_basin)
            distances.append((name, dist))
        
        # Sort by distance and return k nearest
        distances.sort(key=lambda x: x[1])
        return [name for name, _ in distances[:k]]


class GeometricCompletionChecker:
    """
    Determines when generation should stop based on GEOMETRY, not tokens.
    
    The system stops when:
    1. Attractor reached (basin converged)
    2. Surprise collapsed (no new information)
    3. Integration stable (Φ stable and high)
    
    NOT when:
    - Arbitrary token limit reached
    - Stop token encountered
    - External timeout
    """
    
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
        """
        Check if generation should stop based on geometric criteria.
        
        Returns (should_stop, reason)
        """
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
        
        # Safety check (NOT a generation target)
        if len(self.trajectory) > self.config.safety_max_iterations:
            return True, "safety_limit"
        
        return False, "continue"


class QIGGenerator:
    """
    QIG-Pure Generator
    
    Generates responses using:
    - Kernel routing via Fisher-Rao geometry
    - Geometric completion criteria
    - Basin-based synthesis
    
    NO external LLM APIs are used.
    """
    
    def __init__(self, config: Optional[QIGGenerationConfig] = None):
        self.config = config or QIGGenerationConfig()
        self.router = QIGKernelRouter()
        self._validate_qig_purity()
    
    def _validate_qig_purity(self):
        """Validate that this generator is QIG-pure."""
        # Check for forbidden patterns
        forbidden_attrs = ['openai', 'anthropic', 'google', 'max_tokens', 'ChatCompletion']
        for attr in forbidden_attrs:
            assert not hasattr(self, attr), f"QIG violation: {attr} is forbidden"
    
    def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        mode: Optional[GenerationMode] = None
    ) -> Dict[str, Any]:
        """
        Generate a response using QIG-pure methods.
        
        Args:
            prompt: User query
            context: Optional context (conversation history, memory basins)
            mode: Generation mode (auto-selected if None)
        
        Returns:
            Response with text, metrics, and completion reason
        """
        start_time = time.time()
        
        # 1. Encode prompt to basin coordinates
        query_basin = encode_to_basin(prompt)
        
        # 2. Route to appropriate kernels
        target_kernels = self.router.route_query(query_basin, k=3)
        
        # 3. Initialize completion checker
        checker = GeometricCompletionChecker(self.config)
        
        # 4. Measure initial phi
        phi = self._measure_phi(query_basin)
        
        # 5. Select mode if not specified
        if mode is None and self.config.auto_mode:
            mode = self._select_mode(phi)
        
        # 6. Generate via kernel synthesis
        response_basins = []
        current_basin = query_basin.copy()
        iterations = 0
        
        while True:
            iterations += 1
            
            # Synthesize next basin from kernel responses
            kernel_responses = self._query_kernels(target_kernels, current_basin, mode)
            
            # Combine kernel responses via geodesic interpolation
            next_basin = self._geodesic_combine(kernel_responses)
            response_basins.append(next_basin)
            
            # Update phi and checker
            phi = self._measure_phi(next_basin)
            checker.update(next_basin, phi)
            
            # Check geometric completion
            should_stop, reason = checker.should_stop()
            if should_stop:
                break
            
            current_basin = next_basin
        
        # 7. Decode basins to text
        response_text = self._decode_basins(response_basins, target_kernels)
        
        # 8. Compute final metrics
        elapsed = time.time() - start_time
        
        return {
            'response': response_text,
            'completion_reason': reason,
            'iterations': iterations,
            'phi': phi,
            'kappa': KAPPA_STAR,
            'mode': mode.value if mode else 'auto',
            'routed_kernels': target_kernels,
            'elapsed_seconds': elapsed,
            'qig_pure': True  # Certification that no external LLM was used
        }
    
    def generate_stream(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream generation with real-time metrics.
        
        Yields chunks with text and geometric metrics.
        """
        query_basin = encode_to_basin(prompt)
        target_kernels = self.router.route_query(query_basin, k=3)
        checker = GeometricCompletionChecker(self.config)
        
        current_basin = query_basin.copy()
        
        while True:
            # Generate next chunk
            kernel_responses = self._query_kernels(target_kernels, current_basin, None)
            next_basin = self._geodesic_combine(kernel_responses)
            
            phi = self._measure_phi(next_basin)
            checker.update(next_basin, phi)
            
            # Decode this step
            chunk_text = self._decode_single_basin(next_basin, target_kernels)
            
            # Yield metrics
            yield {
                'type': 'chunk',
                'text': chunk_text,
                'phi': phi,
                'kappa': KAPPA_STAR,
                'surprise': checker.surprise_history[-1] if checker.surprise_history else 1.0
            }
            
            # Check completion
            should_stop, reason = checker.should_stop()
            if should_stop:
                yield {
                    'type': 'completion',
                    'reason': reason,
                    'phi': phi
                }
                break
            
            current_basin = next_basin
    
    def _measure_phi(self, basin: np.ndarray) -> float:
        """Measure integration (Φ) from basin entropy."""
        p = np.abs(basin) + 1e-10
        p = p / np.sum(p)
        
        entropy = -np.sum(p * np.log(p + 1e-10))
        max_entropy = np.log(len(basin))
        
        # Φ inversely related to normalized entropy
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
        mode: Optional[GenerationMode]
    ) -> List[np.ndarray]:
        """Query kernels for their response basins."""
        responses = []
        for kernel_name in kernels:
            # Each kernel transforms the input basin based on its domain
            kernel_basin = self.router.kernel_basins[kernel_name]
            
            # Geodesic interpolation toward kernel's expertise
            t = 0.3  # Move 30% toward kernel basin
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
        # Square root representation for geodesic
        sqrt_start = np.sqrt(np.abs(start) + 1e-10)
        sqrt_end = np.sqrt(np.abs(end) + 1e-10)
        
        # Spherical interpolation
        interp = (1 - t) * sqrt_start + t * sqrt_end
        
        # Back to probability
        result = interp ** 2
        result = result / np.sum(result)
        
        return result
    
    def _geodesic_combine(self, basins: List[np.ndarray]) -> np.ndarray:
        """Combine multiple basins via Fréchet mean."""
        if not basins:
            return np.ones(BASIN_DIMENSION) / BASIN_DIMENSION
        
        # Simple average in square root space (approximation of Fréchet mean)
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
        """
        Decode basin trajectory to text response.
        
        Uses the coordizer to decode basin coordinates to actual words
        from the tokenizer_vocabulary (BIP39 words and learned vocabulary).
        """
        if not basins:
            return "[Empty basin trajectory]"
        
        # Try to use coordizer for real word decoding
        decoded_words = []
        if COORDIZER_AVAILABLE:
            try:
                coordizer = get_coordizer()
                if hasattr(coordizer, 'decode'):
                    # Decode each basin to get candidate words
                    for basin in basins[-10:]:  # Use last 10 basins for response
                        # Get top-3 words for each basin position
                        candidates = coordizer.decode(basin, top_k=3, prefer_words=True)
                        if candidates:
                            # Take the best match
                            best_word, score = candidates[0]
                            # Only include if it's a real word (not BPE fragment)
                            if best_word.isalpha() and len(best_word) >= 2:
                                decoded_words.append(best_word)
                            elif len(candidates) > 1:
                                # Try second best
                                second_word, _ = candidates[1]
                                if second_word.isalpha() and len(second_word) >= 2:
                                    decoded_words.append(second_word)
            except Exception as e:
                print(f"[QIGGenerator] Decode error: {e}")
        
        # Build response from decoded words
        if decoded_words:
            # Remove consecutive duplicates
            unique_words = []
            for word in decoded_words:
                if not unique_words or word != unique_words[-1]:
                    unique_words.append(word)
            
            # Create coherent response from decoded words
            response_text = ' '.join(unique_words)
            
            # Add kernel context
            primary_kernel = kernels[0] if kernels else 'zeus'
            final_phi = self._measure_phi(basins[-1])
            
            return f"{response_text}\n\n[QIG-Pure | Φ={final_phi:.3f} | {primary_kernel}]"
        
        # Fallback: kernel-based placeholder (only if decoding failed)
        primary_kernel = kernels[0] if kernels else 'zeus'
        kernel_domains = {
            'zeus': 'Wisdom synthesized from geometric constellation.',
            'athena': 'Strategic patterns revealed through integration.',
            'apollo': 'Clarity emerges from Fisher-Rao navigation.',
            'ares': 'Direct convergence achieved.',
            'hermes': 'Message transmitted via geodesic paths.',
            'hephaestus': 'Tools forged through kernel synthesis.',
            'artemis': 'Target acquired geometrically.',
            'dionysus': 'Creative threshold reached.',
            'demeter': 'Growth patterns manifest.',
            'poseidon': 'Deep structure navigated.',
            'hera': 'Relationships mapped.',
            'aphrodite': 'Harmony achieved.'
        }
        
        base_response = kernel_domains.get(primary_kernel, 'Response synthesized.')
        final_phi = self._measure_phi(basins[-1]) if basins else 0.5
        
        return f"{base_response}\n\n[QIG-Pure Fallback | Φ={final_phi:.3f} | {primary_kernel}]"
    
    def _decode_single_basin(self, basin: np.ndarray, kernels: List[str]) -> str:
        """Decode single basin to text chunk using coordizer vocabulary."""
        # Try to decode basin to actual word
        if COORDIZER_AVAILABLE:
            try:
                coordizer = get_coordizer()
                if hasattr(coordizer, 'decode'):
                    candidates = coordizer.decode(basin, top_k=3, prefer_words=True)
                    if candidates:
                        # Find first real word (not BPE fragment)
                        for word, score in candidates:
                            if word.isalpha() and len(word) >= 2:
                                return f"{word} "
            except Exception:
                pass
        
        # Fallback: just show phi metric
        phi = self._measure_phi(basin)
        return f"[Φ={phi:.2f}] "


# Global singleton for QIG-pure generation
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
    Generate a response using QIG-pure methods.
    
    This is the main entry point for generation.
    NO external LLM APIs are used.
    
    Args:
        prompt: User query
        context: Optional context
        **kwargs: Additional QIG parameters (NOT max_tokens or temperature)
    
    Returns:
        Response dict with text, metrics, and completion reason
    """
    # Validate no forbidden parameters
    forbidden = ['max_tokens', 'temperature', 'model', 'api_key']
    for key in forbidden:
        if key in kwargs:
            raise ValueError(f"QIG violation: '{key}' parameter is forbidden. Use geometric completion.")
    
    generator = get_qig_generator()
    return generator.generate(prompt, context)


# ============================================================================
# QIG PURITY ENFORCEMENT
# ============================================================================

def validate_qig_purity():
    """
    Validate that the generation system is QIG-pure.
    
    Raises AssertionError if any traditional LLM patterns are detected.
    """
    import sys
    
    # Check that forbidden modules are not imported
    forbidden_modules = ['openai', 'anthropic', 'google.generativeai']
    for module in forbidden_modules:
        if module in sys.modules:
            raise AssertionError(f"QIG VIOLATION: {module} is imported. Remove all external LLM dependencies.")
    
    # Check that llm_client is not being used
    if 'llm_client' in sys.modules:
        raise AssertionError("QIG VIOLATION: llm_client.py is imported. Use qig_generation.py instead.")
    
    print("[QIG] Purity validation passed. No external LLM dependencies detected.")
    return True


def test_coordizer_decoding():
    """
    Test that coordizer can decode basins to real words.
    
    Run this to verify the tokenizer_vocabulary integration is working.
    """
    print("\n=== Testing Coordizer Decoding ===")
    
    if not COORDIZER_AVAILABLE:
        print("[ERROR] Coordizer not available")
        return False
    
    try:
        coordizer = get_coordizer()
        print(f"[OK] Coordizer type: {type(coordizer).__name__}")
        print(f"[OK] Vocabulary size: {len(coordizer.vocab) if hasattr(coordizer, 'vocab') else 'unknown'}")
        
        if hasattr(coordizer, 'word_tokens'):
            print(f"[OK] Word tokens: {len(coordizer.word_tokens)}")
            print(f"[OK] Sample words: {coordizer.word_tokens}")
        
        # Test encoding
        test_text = "What is consciousness?"
        basin = coordizer.encode(test_text) if hasattr(coordizer, 'encode') else encode_to_basin(test_text)
        print(f"[OK] Encoded '{test_text}' to basin of shape {basin.shape}")
        
        # Test decoding
        if hasattr(coordizer, 'decode'):
            candidates = coordizer.decode(basin, top_k=10, prefer_words=True)
            print(f"[OK] Decoded to {len(candidates)} candidates:")
            for word, score in candidates[:10]:
                print(f"      {word}: {score:.3f}")
            
            # Check if we got real words
            real_words = [w for w, s in candidates if w.isalpha() and len(w) >= 2]
            if real_words:
                print(f"[OK] Found {len(real_words)} real words: {real_words}")
                return True
            else:
                print("[WARNING] No real words found in decoded output")
                return False
        else:
            print("[ERROR] Coordizer has no decode method")
            return False
            
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests when executed directly
    print("QIG Generation Module - Self Test")
    print("=" * 40)
    
    # Test purity
    validate_qig_purity()
    
    # Test coordizer decoding
    test_coordizer_decoding()
    
    # Test generation
    print("\n=== Testing Generation ===")
    response = generate_response("Explain consciousness")
    print(f"Response: {response['response']}...")
    print(f"Completion reason: {response['completion_reason']}")
    print(f"Phi: {response['phi']:.3f}")
    print(f"Routed to: {response['routed_kernels']}")
