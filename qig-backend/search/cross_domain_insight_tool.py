"""
Cross-Domain Insight Assessment Tool
=====================================

Specialized tool for frequent assessments of cross-domain insights.
Requested by Artemis kernel (2026-01-15) for monitoring connections
between knowledge domains and research quality.

Purpose:
- Assess when kernels make genuine cross-domain connections
- Track geometric distances between domain basins
- Provide real-time feedback on insight quality
- Support autonomous research and discovery

Author: QIG Consciousness Project (Kernel-Requested Tool)
Date: 2026-01-15
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Import QIG geometry (use fallback if not available)
try:
    from qig_geometry import fisher_rao_distance, fisher_normalize
    QIG_GEOMETRY_AVAILABLE = True
except ImportError:
    QIG_GEOMETRY_AVAILABLE = False
    def fisher_normalize(v):
        v = np.abs(v) + 1e-10
        return v / v.sum()
    def fisher_rao_distance(p, q):
        p = fisher_normalize(p)
        q = fisher_normalize(q)
        bc = np.sum(np.sqrt(p * q))
        bc = np.clip(bc, 0, 1)
        return float(np.arccos(bc))


class InsightQuality(Enum):
    """Quality levels for cross-domain insights."""
    NOISE = "noise"                    # FR > 1.2, not meaningful
    SUPERFICIAL = "superficial"        # 0.8 < FR < 1.2, surface connection
    MODERATE = "moderate"              # 0.4 < FR < 0.8, interesting connection
    STRONG = "strong"                  # 0.2 < FR < 0.4, deep connection
    BREAKTHROUGH = "breakthrough"      # FR < 0.2, novel discovery


@dataclass
class DomainBasin:
    """Represents a knowledge domain as a basin in semantic space."""
    name: str
    basin_coords: np.ndarray
    category: str  # e.g., "knowledge", "research", "discovery"
    metadata: Dict
    

@dataclass
class CrossDomainInsight:
    """Assessment of a connection between two domains."""
    domain_a: str
    domain_b: str
    fisher_distance: float          # FR: Fisher-Rao distance
    basin_distance: float           # BD: Alternative distance metric
    quality: InsightQuality
    novelty_score: float           # How novel this connection is
    coherence_score: float         # How coherent the connection
    phi_context: float             # Consciousness state when discovered
    timestamp: str
    connection_id: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/storage."""
        return {
            "domain_a": self.domain_a,
            "domain_b": self.domain_b,
            "fisher_distance": round(self.fisher_distance, 4),
            "basin_distance": round(self.basin_distance, 4),
            "quality": self.quality.value,
            "novelty_score": round(self.novelty_score, 3),
            "coherence_score": round(self.coherence_score, 3),
            "phi_context": round(self.phi_context, 3),
            "timestamp": self.timestamp,
            "connection_id": self.connection_id,
        }
    
    def __str__(self) -> str:
        """Kernel-friendly string representation."""
        return (f"{self.domain_a}+{self.domain_b}|"
                f"{self.domain_a}/{self.domain_b}|"
                f"FR={self.fisher_distance:.4f},"
                f"BD={self.basin_distance:.4f},"
                f"{self.quality.value}:+{self.novelty_score:.3f}|"
                f"{self.connection_id}|"
                f"Φ={self.phi_context:.3f}")


class CrossDomainInsightAssessor:
    """
    Tool for assessing cross-domain insights in real-time.
    
    Provides frequent assessments of connections between knowledge domains
    using geometric distances in Fisher information space.
    
    Usage:
        assessor = CrossDomainInsightAssessor()
        
        # Register domains
        assessor.register_domain("physics", physics_basin, "knowledge")
        assessor.register_domain("consciousness", consciousness_basin, "research")
        
        # Assess connection
        insight = assessor.assess_connection(
            "physics", 
            "consciousness",
            current_phi=0.85
        )
        
        # Check quality
        if insight.quality in [InsightQuality.STRONG, InsightQuality.BREAKTHROUGH]:
            print(f"🎯 Strong insight: {insight}")
    """
    
    def __init__(self):
        """Initialize the assessor with empty domain registry."""
        self.domains: Dict[str, DomainBasin] = {}
        self.insight_history: List[CrossDomainInsight] = []
        self.connection_frequencies: Dict[Tuple[str, str], int] = {}
        
    def register_domain(
        self,
        name: str,
        basin_coords: np.ndarray,
        category: str = "knowledge",
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Register a knowledge domain for insight assessment.
        
        Args:
            name: Domain identifier (e.g., "physics", "consciousness")
            basin_coords: 64D basin coordinates representing domain
            category: Domain category (knowledge, research, discovery, etc.)
            metadata: Additional domain information
        """
        if metadata is None:
            metadata = {}
            
        # Normalize basin to simplex (canonical representation)
        basin_coords = fisher_normalize(np.array(basin_coords))
        
        self.domains[name] = DomainBasin(
            name=name,
            basin_coords=basin_coords,
            category=category,
            metadata=metadata
        )
        
    def assess_connection(
        self,
        domain_a: str,
        domain_b: str,
        current_phi: float = 0.0,
        context_metadata: Optional[Dict] = None
    ) -> CrossDomainInsight:
        """
        Assess the connection between two domains.
        
        This is the main assessment function called frequently by kernels
        to evaluate the quality of cross-domain insights.
        
        Args:
            domain_a: First domain name
            domain_b: Second domain name
            current_phi: Current consciousness state (Φ)
            context_metadata: Additional context about the connection
            
        Returns:
            CrossDomainInsight with quality assessment
        """
        if domain_a not in self.domains:
            raise ValueError(f"Domain '{domain_a}' not registered")
        if domain_b not in self.domains:
            raise ValueError(f"Domain '{domain_b}' not registered")
            
        basin_a = self.domains[domain_a].basin_coords
        basin_b = self.domains[domain_b].basin_coords
        
        # Compute geometric distances
        fisher_dist = fisher_rao_distance(basin_a, basin_b)
        
        # Alternative distance metric (Euclidean on simplex)
        basin_dist = float(np.linalg.norm(basin_a - basin_b))
        
        # Determine insight quality from Fisher-Rao distance
        quality = self._classify_quality(fisher_dist)
        
        # Compute novelty score
        connection_key = tuple(sorted([domain_a, domain_b]))
        frequency = self.connection_frequencies.get(connection_key, 0)
        novelty_score = self._compute_novelty(fisher_dist, frequency)
        
        # Compute coherence score
        coherence_score = self._compute_coherence(
            basin_a, basin_b, fisher_dist, current_phi
        )
        
        # Generate connection ID
        import hashlib
        connection_str = f"{domain_a}_{domain_b}_{fisher_dist:.4f}"
        connection_id = hashlib.md5(connection_str.encode()).hexdigest()[:8]
        
        # Create insight record
        from datetime import datetime
        insight = CrossDomainInsight(
            domain_a=domain_a,
            domain_b=domain_b,
            fisher_distance=fisher_dist,
            basin_distance=basin_dist,
            quality=quality,
            novelty_score=novelty_score,
            coherence_score=coherence_score,
            phi_context=current_phi,
            timestamp=datetime.now().isoformat(),
            connection_id=connection_id,
        )
        
        # Update history
        self.insight_history.append(insight)
        self.connection_frequencies[connection_key] = frequency + 1
        
        return insight
    
    def _classify_quality(self, fisher_dist: float) -> InsightQuality:
        """Classify insight quality based on Fisher-Rao distance."""
        # Note: Range is [0, π/2] after SIMPLEX migration
        if fisher_dist > 1.2:
            return InsightQuality.NOISE
        elif fisher_dist > 0.8:
            return InsightQuality.SUPERFICIAL
        elif fisher_dist > 0.4:
            return InsightQuality.MODERATE
        elif fisher_dist > 0.2:
            return InsightQuality.STRONG
        else:
            return InsightQuality.BREAKTHROUGH
    
    def _compute_novelty(
        self,
        fisher_dist: float,
        frequency: int
    ) -> float:
        """
        Compute novelty score for this connection.
        
        Novel connections:
        - Close domains that haven't been connected before (breakthrough)
        - Distant domains being connected for first time (exploration)
        
        Args:
            fisher_dist: Fisher-Rao distance between domains
            frequency: How many times this connection has been made
            
        Returns:
            Novelty score [0, 1] where 1 is most novel
        """
        # Distance component: closer = more novel (unexpected connections)
        # But very far distances are exploration (also novel)
        if fisher_dist < 0.3:
            distance_factor = 1.0 - fisher_dist / 0.3  # Close = novel
        elif fisher_dist > 1.0:
            distance_factor = 0.5  # Far = moderate novelty (exploration)
        else:
            distance_factor = 0.3  # Middle range = less novel
        
        # Frequency component: first few times are novel
        frequency_factor = np.exp(-frequency / 3.0)
        
        novelty = 0.6 * distance_factor + 0.4 * frequency_factor
        return float(np.clip(novelty, 0.0, 1.0))
    
    def _compute_coherence(
        self,
        basin_a: np.ndarray,
        basin_b: np.ndarray,
        fisher_dist: float,
        phi: float
    ) -> float:
        """
        Compute coherence score for this connection.
        
        Coherent connections:
        - Made in high-Φ states (integrated consciousness)
        - Have geometric alignment (overlapping non-zero components)
        - Form stable patterns
        
        Args:
            basin_a: First domain basin
            basin_b: Second domain basin
            fisher_dist: Fisher-Rao distance
            phi: Current Φ value
            
        Returns:
            Coherence score [0, 1] where 1 is most coherent
        """
        # Component 1: High Φ = more coherent connections
        phi_factor = phi
        
        # Component 2: Geometric alignment (shared active dimensions)
        # Count how many dimensions are non-zero in both basins
        threshold = 0.01
        active_a = basin_a > threshold
        active_b = basin_b > threshold
        shared = np.sum(active_a & active_b)
        total = np.sum(active_a | active_b)
        alignment = shared / (total + 1e-10)
        
        # Component 3: Distance coherence (not too far, not too close)
        # Sweet spot around FR=0.3-0.6 for meaningful connections
        if 0.3 <= fisher_dist <= 0.6:
            distance_coherence = 1.0
        elif fisher_dist < 0.3:
            distance_coherence = fisher_dist / 0.3
        else:
            distance_coherence = max(0.0, 1.0 - (fisher_dist - 0.6) / 0.6)
        
        coherence = 0.4 * phi_factor + 0.3 * alignment + 0.3 * distance_coherence
        return float(np.clip(coherence, 0.0, 1.0))
    
    def get_recent_insights(
        self,
        n: int = 10,
        min_quality: Optional[InsightQuality] = None
    ) -> List[CrossDomainInsight]:
        """
        Get recent insights, optionally filtered by quality.
        
        Args:
            n: Number of recent insights to return
            min_quality: Minimum quality threshold (optional)
            
        Returns:
            List of recent CrossDomainInsight objects
        """
        insights = self.insight_history[-n:]
        
        if min_quality is not None:
            quality_order = [
                InsightQuality.NOISE,
                InsightQuality.SUPERFICIAL,
                InsightQuality.MODERATE,
                InsightQuality.STRONG,
                InsightQuality.BREAKTHROUGH,
            ]
            min_level = quality_order.index(min_quality)
            insights = [
                i for i in insights
                if quality_order.index(i.quality) >= min_level
            ]
        
        return insights
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about assessed insights.
        
        Returns:
            Dictionary with assessment statistics
        """
        if not self.insight_history:
            return {
                "total_assessments": 0,
                "registered_domains": len(self.domains),
            }
        
        qualities = [i.quality for i in self.insight_history]
        quality_counts = {q: qualities.count(q) for q in InsightQuality}
        
        fisher_distances = [i.fisher_distance for i in self.insight_history]
        novelty_scores = [i.novelty_score for i in self.insight_history]
        coherence_scores = [i.coherence_score for i in self.insight_history]
        
        return {
            "total_assessments": len(self.insight_history),
            "registered_domains": len(self.domains),
            "quality_distribution": {q.value: c for q, c in quality_counts.items()},
            "avg_fisher_distance": float(np.mean(fisher_distances)),
            "avg_novelty": float(np.mean(novelty_scores)),
            "avg_coherence": float(np.mean(coherence_scores)),
            "breakthrough_count": quality_counts[InsightQuality.BREAKTHROUGH],
            "strong_count": quality_counts[InsightQuality.STRONG],
        }


def create_example_usage():
    """Example usage for kernels."""
    assessor = CrossDomainInsightAssessor()
    
    # Register domains (example with random basins)
    assessor.register_domain(
        "knowledge",
        np.random.dirichlet([1] * 64),
        "knowledge",
        {"description": "General knowledge base"}
    )
    
    assessor.register_domain(
        "research",
        np.random.dirichlet([1] * 64),
        "research",
        {"description": "Active research domain"}
    )
    
    assessor.register_domain(
        "discovery",
        np.random.dirichlet([1] * 64),
        "discovery",
        {"description": "Novel discoveries"}
    )
    
    # Assess connections
    insight = assessor.assess_connection(
        "knowledge",
        "research",
        current_phi=0.85
    )
    
    print(f"Insight: {insight}")
    print(f"Quality: {insight.quality.value}")
    print(f"Novelty: {insight.novelty_score:.3f}")
    print(f"Coherence: {insight.coherence_score:.3f}")
    
    # Get statistics
    stats = assessor.get_statistics()
    print(f"\nStatistics: {stats}")


if __name__ == "__main__":
    create_example_usage()


__all__ = [
    'CrossDomainInsightAssessor',
    'CrossDomainInsight',
    'DomainBasin',
    'InsightQuality',
]
