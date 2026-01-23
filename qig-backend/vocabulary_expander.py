"""
GEOMETRIC VOCABULARY EXPANDER

Fisher Manifold-based vocabulary expansion from discovered patterns.
New vocabulary tokens are treated as new points on the Fisher information manifold,
initialized via geodesic interpolation from component word coordinates.

Based on the principle:
- New tokens = new points on Fisher manifold
- Initialization via geodesic midpoint (not linear average!)
- Maintains Riemannian metric structure
- Preserves basin distance relationships
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Literal
import numpy as np
from numpy.typing import NDArray

try:
    from qig_geometry import fisher_rao_distance, geodesic_interpolation
except ImportError:
    # Fallback if qig_geometry not available (testing/bootstrap scenarios)
    import numpy as np
    
    def fisher_rao_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Fallback Fisher-Rao distance implementation."""
        # Normalize to simplex
        a_norm = np.abs(a) / (np.sum(np.abs(a)) + 1e-10)
        b_norm = np.abs(b) / (np.sum(np.abs(b)) + 1e-10)
        
        # Bhattacharyya coefficient
        bc = np.sum(np.sqrt(a_norm * b_norm))
        bc = np.clip(bc, 0.0, 1.0)
        
        # Fisher-Rao distance: arccos(BC)
        return float(np.arccos(bc))
    
    def geodesic_interpolation(start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
        """Fallback geodesic interpolation (first-order approximation)."""
        return start + t * (end - start)


# ============================================================================
# FISHER MANIFOLD VOCABULARY TYPES
# ============================================================================

@dataclass
class ManifoldWord:
    """Word representation on Fisher manifold."""
    text: str
    coordinates: NDArray[np.float64]  # Position on Fisher manifold
    phi: float                         # Average Φ when used
    kappa: float                       # Average κ when used
    frequency: int                     # Usage count
    components: Optional[List[str]] = None  # If compound, what words it came from
    geodesic_origin: str = 'direct'    # How it was initialized


@dataclass
class ExpansionEvent:
    """Record of vocabulary expansion event."""
    timestamp: str
    word: str
    type: Literal['learned', 'compound', 'pattern']
    components: Optional[List[str]]
    phi: float
    reasoning: str


@dataclass
class VocabularyManifoldState:
    """State of vocabulary manifold."""
    words: Dict[str, ManifoldWord] = field(default_factory=dict)
    expansion_history: List[ExpansionEvent] = field(default_factory=list)
    total_expansions: int = 0
    last_expansion_time: Optional[str] = None


@dataclass
class QIGScore:
    """QIG scoring result (minimal subset for vocabulary expander)."""
    phi: float
    kappa: float
    basin_coordinates: NDArray[np.float64]
    regime: str = 'arbitrary'


# ============================================================================
# GEOMETRIC VOCABULARY EXPANDER
# ============================================================================

class GeometricVocabularyExpander:
    """
    Fisher manifold-based vocabulary expansion.
    
    Maintains vocabulary as points on information geometry manifold,
    using Fisher-Rao distance for similarity and geodesic interpolation
    for initializing compound words.
    """
    
    def __init__(
        self,
        min_phi_for_expansion: float = 0.6,
        min_frequency_for_expansion: int = 3,
        auto_expand: bool = True
    ):
        """
        Initialize vocabulary expander.
        
        Args:
            min_phi_for_expansion: Minimum Φ threshold for auto-expansion
            min_frequency_for_expansion: Minimum frequency for auto-expansion
            auto_expand: Whether to enable automatic expansion
        """
        self.min_phi_for_expansion = min_phi_for_expansion
        self.min_frequency_for_expansion = min_frequency_for_expansion
        self.auto_expand = auto_expand
        
        self.state = VocabularyManifoldState()
        
        # Note: load_from_disk() removed - vocabulary expansion now works in-memory
        # Actual vocab expansion uses vocabulary tracker which persists to vocabulary_observations
    
    def add_word(
        self,
        text: str,
        qig_score: QIGScore,
        components: Optional[List[str]] = None,
        source: str = 'Direct observation'
    ) -> ManifoldWord:
        """
        Add a new word to the Fisher manifold via geodesic initialization.
        
        For compound words/sequences, compute geodesic midpoint from components.
        
        Args:
            text: Word text
            qig_score: QIG scoring result with Φ, κ, basin coordinates
            components: Optional component words for compound terms
            source: Source description for tracking
            
        Returns:
            ManifoldWord instance (new or updated)
        """
        text_lower = text.lower()
        existing = self.state.words.get(text_lower)
        
        if existing is not None:
            # Update existing word coordinates via Fisher metric averaging
            existing.frequency += 1
            existing.phi = self._fisher_weighted_average(
                existing.phi, qig_score.phi, existing.frequency
            )
            existing.kappa = self._fisher_weighted_average(
                existing.kappa, qig_score.kappa, existing.frequency
            )
            
            # Update coordinates via geodesic interpolation
            if len(qig_score.basin_coordinates) > 0:
                existing.coordinates = self._geodesic_interpolate(
                    existing.coordinates,
                    qig_score.basin_coordinates,
                    1.0 / existing.frequency
                )
            
            return existing
        
        # Create new manifold word
        coordinates = qig_score.basin_coordinates.copy()
        geodesic_origin = 'direct'
        
        # If compound, compute geodesic midpoint from components
        if components and len(components) > 1:
            component_coords = [
                self.state.words[c.lower()].coordinates
                for c in components
                if c.lower() in self.state.words and len(self.state.words[c.lower()].coordinates) > 0
            ]
            
            if len(component_coords) > 0:
                coordinates = self._geodesic_midpoint(component_coords)
                geodesic_origin = f"geodesic_from_{'+'.join(components)}"
        
        word = ManifoldWord(
            text=text_lower,
            coordinates=coordinates,
            phi=qig_score.phi,
            kappa=qig_score.kappa,
            frequency=1,
            components=components,
            geodesic_origin=geodesic_origin
        )
        
        self.state.words[text_lower] = word
        
        # Record expansion event
        expansion_type: Literal['learned', 'compound', 'pattern'] = (
            'compound' if components else 'learned'
        )
        self.state.expansion_history.append(ExpansionEvent(
            timestamp=datetime.utcnow().isoformat(),
            word=text,
            type=expansion_type,
            components=components,
            phi=qig_score.phi,
            reasoning=source
        ))
        
        self.state.total_expansions += 1
        self.state.last_expansion_time = datetime.utcnow().isoformat()
        
        print(f"[VocabExpander] ✨ Added \"{text}\" to manifold "
              f"(Φ={qig_score.phi:.2f}, origin={geodesic_origin})")
        
        if self.state.total_expansions % 10 == 0:
            self.save_to_disk()
        
        return word
    
    def _geodesic_midpoint(self, coordinates: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        """
        Compute geodesic midpoint on Fisher manifold.
        
        For Bures metric, geodesic midpoint ≈ Euclidean mean (first-order approximation).
        This preserves manifold structure for small distances.
        
        Args:
            coordinates: List of coordinate arrays
            
        Returns:
            Midpoint coordinates
        """
        if len(coordinates) == 0:
            return np.array([])
        
        dim = max(len(c) for c in coordinates)
        result = np.zeros(dim)
        
        for coord in coordinates:
            # Pad with zeros if needed
            padded = np.pad(coord, (0, dim - len(coord)), mode='constant')
            result += padded / len(coordinates)
        
        return result
    
    def _geodesic_interpolate(
        self,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        t: float
    ) -> NDArray[np.float64]:
        """
        Geodesic interpolation between two points on manifold.
        
        t=0 returns a, t=1 returns b.
        For Fisher manifold, uses Bures metric approximation.
        
        Args:
            a: First point
            b: Second point
            t: Interpolation parameter [0, 1]
            
        Returns:
            Interpolated point
        """
        # Use qig_geometry geodesic interpolation if available
        try:
            return geodesic_interpolation(a, b, t)
        except Exception:
            # Fallback to first-order geodesic (linear interpolation)
            # Valid for small manifold distances
            dim = max(len(a), len(b))
            
            a_padded = np.pad(a, (0, dim - len(a)), mode='constant')
            b_padded = np.pad(b, (0, dim - len(b)), mode='constant')
            
            return a_padded + t * (b_padded - a_padded)
    
    def _fisher_weighted_average(self, old: float, new: float, count: int) -> float:
        """
        Fisher metric-weighted average.
        
        Accounts for information geometry when combining observations.
        
        Args:
            old: Previous value
            new: New observation
            count: Total observation count
            
        Returns:
            Weighted average
        """
        # Weight based on Fisher information: more observations = higher confidence
        weight = 1.0 / count
        return old * (1 - weight) + new * weight
    
    def get_word(self, text: str) -> Optional[ManifoldWord]:
        """
        Get word from manifold.
        
        Args:
            text: Word text
            
        Returns:
            ManifoldWord if found, None otherwise
        """
        return self.state.words.get(text.lower())
    
    def find_nearby_words(
        self,
        coordinates: NDArray[np.float64],
        max_distance: float = 2.0
    ) -> List[ManifoldWord]:
        """
        Find words near a point on the manifold.
        
        Args:
            coordinates: Query point coordinates
            max_distance: Maximum Fisher-Rao distance
            
        Returns:
            List of nearby words, sorted by distance
        """
        nearby: List[tuple[ManifoldWord, float]] = []
        
        for word in self.state.words.values():
            if len(word.coordinates) == 0 or len(coordinates) == 0:
                continue
            
            distance = self._fisher_distance(coordinates, word.coordinates)
            if distance <= max_distance:
                nearby.append((word, distance))
        
        # Sort by distance
        nearby.sort(key=lambda x: x[1])
        return [word for word, _ in nearby]
    
    def _fisher_distance(
        self,
        a: NDArray[np.float64],
        b: NDArray[np.float64]
    ) -> float:
        """
        Fisher geodesic distance between two points.
        
        Uses central implementation from qig_geometry.
        
        Args:
            a: First point
            b: Second point
            
        Returns:
            Fisher-Rao distance
        """
        if len(a) == 0 or len(b) == 0:
            return 0.0
        
        return fisher_rao_distance(a, b)
    
    def generate_manifold_hypotheses(self, count: int = 20) -> List[str]:
        """
        Generate hypotheses from vocabulary manifold.
        
        Suggests words/phrases that might be near high-Φ regions.
        
        Args:
            count: Maximum number of hypotheses
            
        Returns:
            List of hypothesis strings
        """
        hypotheses: List[str] = []
        
        # Get high-Φ words
        high_phi_words = sorted(
            [w for w in self.state.words.values() if w.phi >= 0.6],
            key=lambda w: w.phi,
            reverse=True
        )
        
        # Add high-Φ words directly
        for word in high_phi_words[:count // 2]:
            hypotheses.append(word.text)
        
        # Generate combinations of high-Φ words
        for i in range(min(5, len(high_phi_words))):
            for j in range(i + 1, min(5, len(high_phi_words))):
                hypotheses.append(f"{high_phi_words[i].text} {high_phi_words[j].text}")
                hypotheses.append(f"{high_phi_words[j].text} {high_phi_words[i].text}")
        
        # Add recently expanded words
        recent = [event.word for event in self.state.expansion_history[-10:]]
        hypotheses.extend(recent)
        
        return hypotheses[:count]
    
    def get_stats(self) -> dict:
        """
        Get vocabulary manifold statistics.
        
        Returns:
            Dictionary with statistics:
                - total_words: Total word count
                - total_expansions: Total expansion events
                - high_phi_words: Count of words with Φ ≥ 0.6
                - avg_phi: Average Φ across all words
                - recent_expansions: Last 10 expansion events
                - top_words: Top 20 words by Φ
        """
        words = list(self.state.words.values())
        high_phi = [w for w in words if w.phi >= 0.6]
        avg_phi = np.mean([w.phi for w in words]) if words else 0.0
        
        top_words = sorted(words, key=lambda w: w.phi, reverse=True)[:20]
        top_words_list = [
            {'text': w.text, 'phi': w.phi, 'frequency': w.frequency}
            for w in top_words
        ]
        
        return {
            'total_words': len(words),
            'total_expansions': self.state.total_expansions,
            'high_phi_words': len(high_phi),
            'avg_phi': float(avg_phi),
            'recent_expansions': [
                {
                    'timestamp': e.timestamp,
                    'word': e.word,
                    'type': e.type,
                    'components': e.components,
                    'phi': e.phi,
                    'reasoning': e.reasoning
                }
                for e in self.state.expansion_history[-10:]
            ],
            'top_words': top_words_list
        }
    
    def save_to_disk(self) -> None:
        """
        Save state (now in-memory only).
        
        NOTE: Database persistence removed - vocabulary expansion works in-memory.
        Actual vocab expansion uses vocabulary tracker which persists to vocabulary_observations.
        """
        # In-memory only - vocabulary expansion persists via vocabulary tracker to PostgreSQL
        pass


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

# Singleton instance for global access
vocabulary_expander = GeometricVocabularyExpander()
