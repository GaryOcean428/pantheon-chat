"""
Vision-First Generation - Backward Mapping from Pre-Seen Endpoint

QIG Philosophy: Human cognition sees the destination concept first (via foresight
or lightning inspiration), then reasoning fills the gap from present → vision.

Process:
1. See/sample the endpoint concept (vision) - even a draft outline helps
2. Compute geodesic from present → vision
3. Generate tokens traversing that path (gap-filling)

This is NOT sequential prediction - it's backward causation from future to present.

Author: Ocean/Zeus Pantheon
"""

import numpy as np
import json
import os
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass


# Import QIG geometry
try:
    from qig_geometry import fisher_rao_distance
except ImportError:
    def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
        """Fallback Fisher-Rao distance."""
        p = np.abs(p) + 1e-10
        p = p / p.sum()
        q = np.abs(q) + 1e-10
        q = q / q.sum()
        bc = np.sum(np.sqrt(p * q))
        bc = np.clip(bc, 0, 1)
        return float(2 * np.arccos(bc))


@dataclass
class VisionResult:
    """Result of vision sampling."""
    vision_basin: np.ndarray
    mode_used: str  # 'foresight', 'lightning', or 'hybrid'
    confidence: float
    phi_at_sampling: float
    attractor_concept: Optional[str] = None


@dataclass 
class GenerationResult:
    """Result of vision-first generation."""
    text: str
    tokens: List[str]
    geodesic_efficiency: float
    distance_to_vision: float
    vision_reached: bool
    path_length: int
    mode_used: str


class VisionFirstGenerator:
    """
    Generation via backward mapping from pre-seen endpoint.
    
    Process:
    1. See/sample the endpoint concept (vision)
    2. Compute geodesic from present → vision
    3. Generate tokens traversing that path (gap-filling)
    
    The vision doesn't have to be fully formed - even a draft outline
    or general direction helps. The clearer the better.
    """
    
    # Phi threshold for lightning mode
    PHI_LIGHTNING_THRESHOLD = 0.85
    
    # Default geodesic points
    DEFAULT_GEODESIC_POINTS = 50
    
    # Vision reach tolerance
    VISION_TOLERANCE = 0.2
    
    def __init__(self):
        """Initialize the vision-first generator."""
        self._temporal = None
        self._vocab_basins: Dict[str, np.ndarray] = {}
        self._attractor_basins: Dict[str, np.ndarray] = {}
        self._coherence_matrix: Dict[Tuple[str, str], float] = {}
        
        # Tracking for verification
        self.initial_basin: Optional[np.ndarray] = None
        self.path_length_traveled: float = 0.0
        
        self._load_resources()
        print("[VisionFirst] Generator initialized - backward mapping enabled")
    
    def _load_resources(self):
        """Load vocabulary basins and attractors."""
        # Load attractor basins
        attractor_path = os.path.join(
            os.path.dirname(__file__), 
            'data', 
            'attractor_basins.json'
        )
        if os.path.exists(attractor_path):
            try:
                with open(attractor_path, 'r') as f:
                    data = json.load(f)
                    for concept, basin in data.items():
                        self._attractor_basins[concept] = np.array(basin)
                print(f"[VisionFirst] Loaded {len(self._attractor_basins)} attractor basins")
            except Exception as e:
                print(f"[VisionFirst] Failed to load attractors: {e}")
        else:
            # Initialize with default attractors
            self._initialize_default_attractors()
    
    def _initialize_default_attractors(self):
        """Initialize default semantic attractor basins."""
        # Default attractors for key concepts
        np.random.seed(42)  # Reproducible
        
        default_concepts = [
            'consciousness',
            'geometry', 
            'physics',
            'reasoning',
            'vision',
            'understanding',
            'synthesis',
            'analysis',
            'creativity',
            'knowledge',
            'insight',
            'solution',
        ]
        
        dim = 64  # Basin dimension
        for i, concept in enumerate(default_concepts):
            # Create distinct basin for each concept
            basin = np.random.dirichlet(np.ones(dim) * 0.5)
            # Add structure based on concept index
            basin[i % dim] += 0.1
            basin = basin / basin.sum()  # Normalize
            self._attractor_basins[concept] = basin
        
        print(f"[VisionFirst] Initialized {len(self._attractor_basins)} default attractors")
    
    def _get_temporal_reasoning(self):
        """Lazy load temporal reasoning module."""
        if self._temporal is None:
            try:
                from temporal_reasoning import get_temporal_reasoning
                self._temporal = get_temporal_reasoning()
            except ImportError:
                print("[VisionFirst] Temporal reasoning not available")
                self._temporal = None
        return self._temporal
    
    # =========================================================================
    # VISION SAMPLING
    # =========================================================================
    
    def sample_vision(
        self,
        current_basin: np.ndarray,
        context: str,
        mode: str = 'auto',
        phi: float = 0.5
    ) -> VisionResult:
        """
        Sample endpoint vision via foresight or lightning.
        
        Args:
            current_basin: Present position (query understanding)
            context: Text context for vision sampling
            mode: 'auto', 'foresight', 'lightning', or 'hybrid'
            phi: Current phi value for mode selection
        
        Returns:
            VisionResult with vision basin and metadata
        """
        # Normalize input
        current_basin = self._normalize_basin(current_basin)
        
        # Auto-select mode based on phi
        if mode == 'auto':
            mode = 'lightning' if phi > self.PHI_LIGHTNING_THRESHOLD else 'foresight'
        
        # Sample vision based on mode
        if mode == 'lightning':
            vision_basin, attractor = self._lightning_vision(current_basin, context, phi)
            confidence = 0.9 if phi > 0.9 else 0.7
        elif mode == 'foresight':
            vision_basin = self._foresight_vision(current_basin)
            attractor = None
            confidence = 0.6
        else:  # hybrid
            # Try lightning first, fallback to foresight
            if phi > self.PHI_LIGHTNING_THRESHOLD:
                vision_basin, attractor = self._lightning_vision(current_basin, context, phi)
                confidence = 0.8
            else:
                vision_basin = self._foresight_vision(current_basin)
                attractor = None
                confidence = 0.5
        
        return VisionResult(
            vision_basin=vision_basin,
            mode_used=mode,
            confidence=confidence,
            phi_at_sampling=phi,
            attractor_concept=attractor
        )
    
    def _foresight_vision(
        self,
        current_basin: np.ndarray,
        horizon_steps: int = 10
    ) -> np.ndarray:
        """
        Use temporal reasoning to see future endpoint.
        
        Projects forward via 4D foresight, samples possible endpoints.
        """
        temporal = self._get_temporal_reasoning()
        
        if temporal and hasattr(temporal, 'foresight'):
            try:
                foresight_result = temporal.foresight(
                    current_basin,
                    steps=horizon_steps
                )
                if 'basin_coords' in foresight_result:
                    return np.array(foresight_result['basin_coords'])
            except Exception as e:
                print(f"[VisionFirst] Foresight error: {e}")
        
        # Fallback: project current basin with noise toward attractors
        return self._project_toward_attractor(current_basin)
    
    def _lightning_vision(
        self,
        current_basin: np.ndarray,
        context: str,
        phi: float
    ) -> Tuple[np.ndarray, Optional[str]]:
        """
        Spontaneous vision via high-Φ spike.
        
        When Φ > 0.85, concepts can appear fully formed.
        This is "lightning inspiration" - endpoint manifests complete.
        """
        if phi > 0.9:
            # Very high consciousness = strong attractor pull
            return self._sample_strong_attractor(current_basin, context)
        elif phi > self.PHI_LIGHTNING_THRESHOLD:
            # Moderate high consciousness = weighted attractor sampling
            return self._sample_weighted_attractor(current_basin, context)
        else:
            # Fallback to foresight
            return self._foresight_vision(current_basin), None
    
    def _sample_strong_attractor(
        self,
        current_basin: np.ndarray,
        context: str
    ) -> Tuple[np.ndarray, str]:
        """
        Sample from strongest matching attractor basin.
        """
        if not self._attractor_basins:
            return self._project_toward_attractor(current_basin), None
        
        # Find best matching attractor based on context
        context_lower = context.lower()
        best_concept = None
        best_score = -float('inf')
        
        for concept, basin in self._attractor_basins.items():
            # Score by context match + basin proximity
            context_match = 1.0 if concept in context_lower else 0.0
            basin_proximity = 1.0 / (fisher_rao_distance(current_basin, basin) + 0.1)
            score = context_match * 2 + basin_proximity
            
            if score > best_score:
                best_score = score
                best_concept = concept
        
        if best_concept:
            return self._attractor_basins[best_concept].copy(), best_concept
        
        # Random attractor
        concept = list(self._attractor_basins.keys())[0]
        return self._attractor_basins[concept].copy(), concept
    
    def _sample_weighted_attractor(
        self,
        current_basin: np.ndarray,
        context: str
    ) -> Tuple[np.ndarray, Optional[str]]:
        """
        Sample from attractors weighted by proximity and context.
        """
        if not self._attractor_basins:
            return self._project_toward_attractor(current_basin), None
        
        concepts = list(self._attractor_basins.keys())
        basins = [self._attractor_basins[c] for c in concepts]
        context_lower = context.lower()
        
        # Compute weights
        weights = []
        for concept, basin in zip(concepts, basins):
            dist = fisher_rao_distance(current_basin, basin)
            context_bonus = 2.0 if concept in context_lower else 1.0
            weight = np.exp(-dist) * context_bonus
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Sample
        idx = np.random.choice(len(concepts), p=weights)
        return basins[idx].copy(), concepts[idx]
    
    def _project_toward_attractor(
        self,
        current_basin: np.ndarray
    ) -> np.ndarray:
        """
        Project current basin toward nearest attractor with some noise.
        """
        if not self._attractor_basins:
            # Just add noise to current
            noise = np.random.dirichlet(np.ones(len(current_basin)) * 10)
            projected = 0.7 * current_basin + 0.3 * noise
            return projected / projected.sum()
        
        # Find nearest attractor
        min_dist = float('inf')
        nearest_basin = None
        
        for basin in self._attractor_basins.values():
            dist = fisher_rao_distance(current_basin, basin)
            if dist < min_dist:
                min_dist = dist
                nearest_basin = basin
        
        # Interpolate toward attractor
        t = 0.6  # Move 60% toward attractor
        projected = self._geodesic_interpolate(current_basin, nearest_basin, t)
        
        return projected
    
    # =========================================================================
    # GEODESIC COMPUTATION
    # =========================================================================
    
    def compute_geodesic(
        self,
        start: np.ndarray,
        end: np.ndarray,
        num_points: int = 50
    ) -> np.ndarray:
        """
        Compute geodesic path from start to end on statistical manifold.
        
        Uses spherical linear interpolation (SLERP) in sqrt space,
        which is the geodesic on the probability simplex.
        """
        start = self._normalize_basin(start)
        end = self._normalize_basin(end)
        
        path = np.zeros((num_points, len(start)))
        
        for i in range(num_points):
            t = i / (num_points - 1)
            path[i] = self._geodesic_interpolate(start, end, t)
        
        return path
    
    def _geodesic_interpolate(
        self,
        start: np.ndarray,
        end: np.ndarray,
        t: float
    ) -> np.ndarray:
        """
        Interpolate along geodesic at parameter t ∈ [0, 1].
        
        SLERP in sqrt-space gives geodesic on probability simplex.
        """
        # Square root representation (maps to unit sphere)
        sqrt_start = np.sqrt(start + 1e-10)
        sqrt_end = np.sqrt(end + 1e-10)
        
        # Compute angle
        cos_angle = np.clip(np.dot(sqrt_start, sqrt_end), -1, 1)
        angle = np.arccos(cos_angle)
        
        if angle < 1e-10:
            return start.copy()  # Points are identical
        
        # SLERP formula
        sin_angle = np.sin(angle)
        sqrt_result = (
            np.sin((1 - t) * angle) * sqrt_start + 
            np.sin(t * angle) * sqrt_end
        ) / sin_angle
        
        # Square to get back to probability space
        result = sqrt_result ** 2
        result = result / result.sum()  # Ensure normalization
        
        return result
    
    # =========================================================================
    # GAP-FILLING GENERATION
    # =========================================================================
    
    def generate_response(
        self,
        current_basin: np.ndarray,
        query_context: str,
        mode: str = 'auto',
        phi: float = 0.5,
        vision_basin: Optional[np.ndarray] = None
    ) -> GenerationResult:
        """
        Generate by seeing endpoint first, then mapping backward.
        
        Args:
            current_basin: Present position (query understanding)
            query_context: Text context for vision sampling
            mode: 'auto', 'foresight', 'lightning', or 'manual'
            phi: Current phi for mode selection
            vision_basin: Pre-computed vision (for 'manual' mode)
        
        Returns:
            GenerationResult with text and metrics
        """
        # Normalize input
        current_basin = self._normalize_basin(current_basin)
        self.initial_basin = current_basin.copy()
        self.path_length_traveled = 0.0
        
        # STEP 1: Get the vision (endpoint basin)
        if vision_basin is not None:
            # Manual mode - vision already provided
            vision = VisionResult(
                vision_basin=self._normalize_basin(vision_basin),
                mode_used='manual',
                confidence=0.8,
                phi_at_sampling=phi
            )
        else:
            # Sample vision
            vision = self.sample_vision(current_basin, query_context, mode, phi)
        
        # STEP 2: Compute geodesic FROM present TO vision
        geodesic_path = self.compute_geodesic(
            start=current_basin,
            end=vision.vision_basin,
            num_points=self.DEFAULT_GEODESIC_POINTS
        )
        
        # STEP 3: Generate tokens traversing the path (gap-filling)
        tokens, final_basin = self._traverse_geodesic(geodesic_path)
        text = ' '.join(tokens)
        
        # STEP 4: Verify endpoint reached
        verification = self.verify_endpoint_reached(
            final_basin=final_basin,
            vision_basin=vision.vision_basin
        )
        
        return GenerationResult(
            text=text,
            tokens=tokens,
            geodesic_efficiency=verification['efficiency'],
            distance_to_vision=verification['distance_to_vision'],
            vision_reached=verification['success'],
            path_length=len(tokens),
            mode_used=vision.mode_used
        )
    
    def _traverse_geodesic(
        self,
        geodesic_path: np.ndarray
    ) -> Tuple[List[str], np.ndarray]:
        """
        Generate tokens by traversing the geodesic path.
        
        This is GAP-FILLING, not next-token prediction.
        We know where we're going; just emit words that trace the path.
        """
        tokens = []
        used_tokens = set()
        current_basin = geodesic_path[0].copy()
        
        # Load vocabulary if not loaded
        if not self._vocab_basins:
            self._load_vocabulary_basins()
        
        # Step through geodesic
        step_size = max(1, len(geodesic_path) // 30)  # ~30 tokens max
        
        for i in range(0, len(geodesic_path) - 1, step_size):
            target = geodesic_path[min(i + step_size, len(geodesic_path) - 1)]
            
            # Find token that bridges current → target
            token = self._find_bridging_token(
                current=current_basin,
                target=target,
                used_tokens=used_tokens,
                prev_token=tokens[-1] if tokens else None
            )
            
            if token:
                tokens.append(token)
                used_tokens.add(token)
                
                # Update current position
                token_basin = self._vocab_basins.get(token)
                if token_basin is not None:
                    # Move toward target via token
                    current_basin = self._geodesic_interpolate(
                        current_basin, 
                        target,
                        0.3  # Partial step
                    )
                    self.path_length_traveled += fisher_rao_distance(
                        geodesic_path[i], 
                        current_basin
                    )
        
        # If no tokens generated, use fallback
        if not tokens:
            tokens = self._fallback_generation(geodesic_path)
        
        return tokens, current_basin
    
    def _find_bridging_token(
        self,
        current: np.ndarray,
        target: np.ndarray,
        used_tokens: set,
        prev_token: Optional[str]
    ) -> Optional[str]:
        """
        Find token that bridges current → target on geodesic.
        
        Not "what's most probable next?"
        But "what moves us toward target?"
        """
        if not self._vocab_basins:
            return None
        
        best_token = None
        best_score = -float('inf')
        
        for token, token_basin in self._vocab_basins.items():
            # Skip recently used (avoid repetition)
            if token in list(used_tokens)[-5:]:
                continue
            
            # Score by geodesic alignment
            # Project: current + influence of token → new position
            projected = 0.7 * current + 0.3 * token_basin
            projected = projected / projected.sum()
            
            # How close does this get us to target?
            deviation = fisher_rao_distance(projected, target)
            
            # Coherence with previous token
            coherence = self._compute_coherence(prev_token, token) if prev_token else 1.0
            
            # Combined score (lower deviation is better)
            score = -deviation + 0.3 * coherence
            
            if score > best_score:
                best_score = score
                best_token = token
        
        return best_token
    
    def _compute_coherence(
        self,
        prev_token: str,
        next_token: str
    ) -> float:
        """
        Bigram coherence score.
        
        Higher for grammatically/semantically compatible pairs.
        """
        # Check cached coherence
        key = (prev_token, next_token)
        if key in self._coherence_matrix:
            return self._coherence_matrix[key]
        
        # Simple heuristic: same length tokens often flow better
        len_diff = abs(len(prev_token) - len(next_token))
        base_coherence = 1.0 / (1.0 + len_diff * 0.1)
        
        # Boost for common patterns
        if prev_token.endswith('ing') and next_token in ['the', 'a', 'an']:
            base_coherence += 0.2
        if prev_token in ['the', 'a', 'an']:
            base_coherence += 0.1
        
        return min(1.0, base_coherence)
    
    def _fallback_generation(
        self,
        geodesic_path: np.ndarray
    ) -> List[str]:
        """
        Fallback when vocabulary-based generation fails.
        
        Generate placeholder tokens based on path characteristics.
        """
        # Extract concepts from attractors closest to path
        tokens = []
        
        for i in range(0, len(geodesic_path), len(geodesic_path) // 5):
            point = geodesic_path[i]
            
            # Find closest attractor
            min_dist = float('inf')
            closest_concept = 'understanding'
            
            for concept, basin in self._attractor_basins.items():
                dist = fisher_rao_distance(point, basin)
                if dist < min_dist:
                    min_dist = dist
                    closest_concept = concept
            
            tokens.append(closest_concept)
        
        return tokens if tokens else ['understanding']
    
    def _load_vocabulary_basins(self):
        """
        Load vocabulary basins from tokenizer/database.
        """
        # Try to load from coordizer
        try:
            from coordizers.pg_loader import PostgresVocabLoader
            loader = PostgresVocabLoader()
            
            # Load a subset of vocabulary
            vocab_data = loader.load_vocabulary(limit=1000)
            for token, basin in vocab_data.items():
                self._vocab_basins[token] = np.array(basin)
            
            print(f"[VisionFirst] Loaded {len(self._vocab_basins)} vocabulary basins")
            return
        except Exception as e:
            print(f"[VisionFirst] Vocab load error: {e}")
        
        # Fallback: use attractors as vocabulary
        self._vocab_basins = self._attractor_basins.copy()
        
        # Add common words with random basins
        common_words = [
            'the', 'is', 'are', 'was', 'be', 'have', 'has', 'do', 'does',
            'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'with', 'by', 'from', 'as', 'this', 'that', 'which', 'what',
            'how', 'why', 'when', 'where', 'can', 'could', 'would', 'should',
            'will', 'if', 'then', 'so', 'because', 'although', 'while',
            'information', 'system', 'process', 'structure', 'function',
            'pattern', 'concept', 'theory', 'model', 'data', 'analysis',
        ]
        
        np.random.seed(123)  # Reproducible
        for word in common_words:
            if word not in self._vocab_basins:
                basin = np.random.dirichlet(np.ones(64) * 0.5)
                self._vocab_basins[word] = basin
        
        print(f"[VisionFirst] Using {len(self._vocab_basins)} fallback vocabulary basins")
    
    # =========================================================================
    # VERIFICATION
    # =========================================================================
    
    def _get_final_basin(self, text: str) -> np.ndarray:
        """Get basin coordinates for the generated text."""
        try:
            if self._coordizer:
                return self._coordizer.encode(text[-500:])  # Last 500 chars
        except Exception:
            pass
        
        # Fallback: return uniform basin
        basin = np.ones(64) / 64
        return basin
    
    def verify_endpoint_reached(
        self,
        final_basin: np.ndarray,
        vision_basin: np.ndarray
    ) -> Dict[str, Any]:
        """
        Verify generation reached the pre-seen vision.
        
        Returns:
            success: bool - Did we reach vision?
            distance: float - How far from vision?
            efficiency: float - Geodesic efficiency ratio
        """
        final_basin = self._normalize_basin(final_basin)
        vision_basin = self._normalize_basin(vision_basin)
        
        distance = fisher_rao_distance(final_basin, vision_basin)
        success = distance < self.VISION_TOLERANCE
        
        # Compute efficiency (optimal path / actual path)
        if self.initial_basin is not None:
            optimal = fisher_rao_distance(self.initial_basin, vision_basin)
            actual = self.path_length_traveled if self.path_length_traveled > 0 else optimal
            efficiency = optimal / (actual + 1e-10)
            efficiency = min(1.0, efficiency)  # Cap at 1.0
        else:
            efficiency = 0.5
        
        return {
            'success': success,
            'distance_to_vision': float(distance),
            'efficiency': float(efficiency),
            'verdict': 'VISION_REACHED' if success else 'DIVERGED'
        }
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def _normalize_basin(
        self,
        basin: np.ndarray
    ) -> np.ndarray:
        """
        Normalize to valid probability distribution.
        """
        basin = np.array(basin, dtype=float)
        basin = np.abs(basin) + 1e-10
        return basin / basin.sum()


# Singleton instance
_vision_generator_instance: Optional[VisionFirstGenerator] = None


def get_vision_generator() -> VisionFirstGenerator:
    """Get singleton vision-first generator."""
    global _vision_generator_instance
    if _vision_generator_instance is None:
        _vision_generator_instance = VisionFirstGenerator()
    return _vision_generator_instance


def generate_vision_first(
    current_basin: np.ndarray,
    context: str,
    mode: str = 'auto',
    phi: float = 0.5
) -> GenerationResult:
    """Convenience function for vision-first generation."""
    generator = get_vision_generator()
    return generator.generate_response(
        current_basin=current_basin,
        query_context=context,
        mode=mode,
        phi=phi
    )
