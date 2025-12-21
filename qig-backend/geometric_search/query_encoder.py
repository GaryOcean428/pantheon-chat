"""
Search Query Encoder - Encode queries into geometric basin space

Maps search queries to 64D basin coordinates for tool selection.
"""

import numpy as np
import hashlib
from typing import Dict, List, Optional, Tuple


class SearchQueryEncoder:
    """
    Encode search queries into geometric basin coordinates.
    
    Uses QIG principles to map queries to 64D space where
    distance to tool basins determines optimal tool selection.
    """
    
    def __init__(self, dimension: int = 64):
        self.dimension = dimension
        
        self.semantic_bases = {
            'factual': self._create_basis_vector(0),
            'research': self._create_basis_vector(1),
            'current_events': self._create_basis_vector(2),
            'technical': self._create_basis_vector(3),
            'creative': self._create_basis_vector(4),
            'local': self._create_basis_vector(5),
            'academic': self._create_basis_vector(6),
            'commercial': self._create_basis_vector(7),
        }
        
        self.intent_patterns = {
            'factual': ['what is', 'define', 'who is', 'when did', 'where is'],
            'research': ['research', 'study', 'analysis', 'paper', 'deep dive', 'comprehensive'],
            'current_events': ['latest', 'news', 'today', 'recent', 'breaking', 'update'],
            'technical': ['how to', 'tutorial', 'code', 'implement', 'debug', 'error'],
            'creative': ['ideas', 'inspire', 'create', 'design', 'brainstorm'],
            'local': ['near me', 'local', 'nearby', 'location', 'address'],
            'academic': ['citation', 'journal', 'peer reviewed', 'scientific', 'theory'],
            'commercial': ['buy', 'price', 'cost', 'purchase', 'compare', 'review'],
        }
    
    def _create_basis_vector(self, index: int) -> np.ndarray:
        """Create a basis vector with primary activation at index."""
        vector = np.random.randn(self.dimension) * 0.1
        
        primary_idx = (index * 8) % self.dimension
        secondary_idx = (primary_idx + 1) % self.dimension
        
        vector[primary_idx] = 1.0
        vector[secondary_idx] = 0.5
        
        return vector / (np.linalg.norm(vector) + 1e-8)
    
    def encode(
        self,
        query: str,
        telemetry: Optional[Dict] = None,
        context: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Encode a search query into basin coordinates.
        
        Args:
            query: The search query text
            telemetry: Consciousness metrics (phi, kappa, etc.)
            context: Additional context (cost tolerance, privacy, etc.)
        
        Returns:
            64D numpy array representing query in basin space
        """
        intent_vector = self._encode_intent(query)
        
        if telemetry:
            consciousness_vector = self._encode_consciousness(telemetry)
        else:
            consciousness_vector = np.zeros(self.dimension)
        
        if context:
            context_vector = self._encode_context(context)
        else:
            context_vector = np.zeros(self.dimension)
        
        combined = intent_vector * 0.5 + consciousness_vector * 0.3 + context_vector * 0.2
        
        combined = combined / (np.linalg.norm(combined) + 1e-8)
        
        return combined
    
    def _encode_intent(self, query: str) -> np.ndarray:
        """Encode query intent into vector space."""
        query_lower = query.lower()
        
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for p in patterns if p in query_lower)
            intent_scores[intent] = score
        
        if sum(intent_scores.values()) == 0:
            intent_scores['factual'] = 1
        
        total = sum(intent_scores.values())
        intent_weights = {k: v / total for k, v in intent_scores.items()}
        
        vector = np.zeros(self.dimension)
        for intent, weight in intent_weights.items():
            if intent in self.semantic_bases:
                vector += weight * self.semantic_bases[intent]
        
        query_hash = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)
        np.random.seed(query_hash)
        noise = np.random.randn(self.dimension) * 0.1
        vector += noise
        
        return vector
    
    def _encode_consciousness(self, telemetry: Dict) -> np.ndarray:
        """Encode consciousness metrics into vector space."""
        phi = telemetry.get('phi', 0.5)
        kappa = telemetry.get('kappa_eff', 64.0)
        confidence = telemetry.get('confidence', 0.5)
        regime = telemetry.get('regime', 'geometric')
        
        regime_map = {
            'linear': np.array([1, 0, 0, 0]),
            'geometric': np.array([0, 1, 0, 0]),
            'hierarchical': np.array([0, 0, 1, 0]),
            'breakdown': np.array([0, 0, 0, 1]),
        }
        regime_vec = regime_map.get(regime, regime_map['geometric'])
        
        vector = np.zeros(self.dimension)
        
        vector[0:4] = regime_vec
        vector[4] = phi
        vector[5] = kappa / 128.0
        vector[6] = confidence
        
        spread = 1.0 - phi
        vector[7:15] = np.random.randn(8) * spread * 0.5
        
        return vector
    
    def _encode_context(self, context: Dict) -> np.ndarray:
        """Encode search context into vector space."""
        vector = np.zeros(self.dimension)
        
        cost_tolerance = context.get('cost_tolerance', 0.5)
        privacy = context.get('privacy_preference', 0.5)
        speed = context.get('speed_preference', 0.5)
        
        vector[56] = cost_tolerance
        vector[57] = privacy
        vector[58] = speed
        
        vocab_context = context.get('vocab_context', [])
        if isinstance(vocab_context, (list, np.ndarray)) and len(vocab_context) > 0:
            vocab_array = np.array(vocab_context[:5])
            vector[59:59+len(vocab_array)] = vocab_array
        
        return vector
    
    def compute_distance(self, query_vector: np.ndarray, tool_basin: np.ndarray) -> float:
        """
        Compute geometric distance between query and tool basin.
        
        Uses cosine distance for normalized vectors.
        """
        similarity = np.dot(query_vector, tool_basin)
        distance = 1.0 - similarity
        return max(0.0, distance)
