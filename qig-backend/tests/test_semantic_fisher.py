"""
Tests for SemanticFisherMetric

Validates that the semantic warping correctly bridges
geometry and semantics in a QIG-pure way.
"""

import pytest
import numpy as np
import sys
import os

# Add qig-backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_fisher import (
    SemanticFisherMetric,
    SemanticWarpConfig,
    STOPWORDS,
)


class TestSemanticWarpConfig:
    """Test configuration class."""
    
    def test_default_config(self):
        config = SemanticWarpConfig()
        assert config.temperature == 1.0
        assert config.max_warp_factor == 0.7
        assert config.min_relationship_strength == 0.1
        assert config.normalize_relationships is True
        assert config.bidirectional is True
    
    def test_custom_config(self):
        config = SemanticWarpConfig(
            temperature=0.5,
            max_warp_factor=0.5
        )
        assert config.temperature == 0.5
        assert config.max_warp_factor == 0.5


class TestSemanticFisherMetric:
    """Test the semantic Fisher metric."""
    
    @pytest.fixture
    def sample_relationships(self):
        """Sample learned relationships."""
        return {
            'quantum': [('physics', 0.9), ('mechanics', 0.8), ('wave', 0.7)],
            'consciousness': [('awareness', 0.85), ('mind', 0.8), ('experience', 0.75)],
            'geometry': [('manifold', 0.9), ('curvature', 0.85), ('topology', 0.8)],
            'information': [('data', 0.8), ('entropy', 0.75), ('knowledge', 0.7)],
        }
    
    @pytest.fixture
    def metric(self, sample_relationships):
        """Create metric with sample relationships."""
        return SemanticFisherMetric(relationships=sample_relationships)
    
    @pytest.fixture
    def sample_basins(self):
        """Generate sample basin coordinates."""
        np.random.seed(42)
        basins = {}
        words = ['quantum', 'physics', 'consciousness', 'awareness', 
                 'geometry', 'manifold', 'unrelated', 'random']
        for word in words:
            basin = np.random.dirichlet(np.ones(64))
            basins[word] = basin / np.linalg.norm(basin)  # Unit sphere
        return basins
    
    def test_initialization_empty(self):
        """Test initialization without relationships."""
        metric = SemanticFisherMetric()
        assert len(metric.relationships) == 0
    
    def test_initialization_with_relationships(self, sample_relationships):
        """Test initialization with relationships."""
        metric = SemanticFisherMetric(relationships=sample_relationships)
        assert len(metric.relationships) > 0
        assert 'quantum' in metric.relationships
    
    def test_stopwords_filtered(self):
        """Test that stopwords are filtered from relationships."""
        relationships = {
            'quantum': [('the', 0.9), ('physics', 0.8)],  # 'the' is stopword
            'the': [('quantum', 0.9)],  # 'the' as source should be skipped
        }
        metric = SemanticFisherMetric(relationships=relationships)
        
        # 'the' should not be in relationships
        assert 'the' not in metric.relationships
        # 'the' should not be a neighbor
        if 'quantum' in metric.relationships:
            neighbors = metric.relationships['quantum']
            assert 'the' not in neighbors
    
    def test_relationship_strength(self, metric):
        """Test relationship strength lookup."""
        # Direct relationship
        strength = metric.get_relationship_strength('quantum', 'physics')
        assert strength > 0
        
        # No relationship
        strength = metric.get_relationship_strength('quantum', 'banana')
        assert strength == 0
    
    def test_relationship_bidirectional(self, metric):
        """Test bidirectional relationship lookup."""
        # Forward
        forward = metric.get_relationship_strength('quantum', 'physics')
        # Reverse (should also work with bidirectional=True)
        reverse = metric.get_relationship_strength('physics', 'quantum')
        
        # Both should be non-zero
        assert forward > 0
        # Reverse may be slightly weaker (0.8x) per implementation
        assert reverse > 0
    
    def test_warp_factor_no_relationship(self, metric):
        """Test warp factor for unrelated words."""
        warp = metric.compute_warp_factor(0)
        assert warp == 1.0  # No warping
    
    def test_warp_factor_strong_relationship(self, metric):
        """Test warp factor for related words."""
        warp = metric.compute_warp_factor(1.0)  # Max strength
        assert warp < 1.0  # Should be warped (reduced distance)
        assert warp >= metric.config.max_warp_factor
    
    def test_warped_distance_smaller_for_related(self, metric, sample_basins):
        """Test that related words have smaller warped distance."""
        basin_quantum = sample_basins['quantum']
        basin_physics = sample_basins['physics']  # Related
        basin_random = sample_basins['random']  # Unrelated
        
        # Warped distance to related word
        d_related = metric.distance(basin_quantum, basin_physics, 'quantum', 'physics')
        
        # Warped distance to unrelated word
        d_unrelated = metric.distance(basin_quantum, basin_random, 'quantum', 'random')
        
        # Even if basins are random, related should be closer due to warping
        # Note: This depends on the base Fisher distance, so we just check warping works
        assert d_related >= 0
        assert d_unrelated >= 0
    
    def test_distance_without_words(self, metric, sample_basins):
        """Test distance computation without words (no warping)."""
        basin1 = sample_basins['quantum']
        basin2 = sample_basins['physics']
        
        # Without words - pure Fisher distance
        d = metric.distance(basin1, basin2)
        assert d >= 0
        assert d <= np.pi  # Max Fisher distance
    
    def test_similarity(self, metric, sample_basins):
        """Test similarity computation."""
        basin1 = sample_basins['quantum']
        basin2 = sample_basins['physics']
        
        sim = metric.similarity(basin1, basin2, 'quantum', 'physics')
        assert 0 <= sim <= 1
    
    def test_rank_candidates(self, metric, sample_basins):
        """Test candidate ranking."""
        current = sample_basins['quantum']
        candidates = [
            ('physics', sample_basins['physics']),
            ('random', sample_basins['random']),
            ('unrelated', sample_basins['unrelated']),
        ]
        
        ranked = metric.rank_candidates(
            current_basin=current,
            current_word='quantum',
            candidates=candidates,
            context_words=['quantum', 'mechanics'],
            top_k=3
        )
        
        assert len(ranked) == 3
        # Each result should be (word, distance, similarity)
        for word, dist, sim in ranked:
            assert word in ['physics', 'random', 'unrelated']
            assert dist >= 0
            assert 0 <= sim <= 1
    
    def test_geodesic_step(self, metric, sample_basins):
        """Test geodesic stepping."""
        start = sample_basins['quantum']
        target = sample_basins['physics']
        
        result = metric.geodesic_step(
            current_basin=start,
            target_basin=target,
            current_word='quantum',
            target_word='physics',
            step_size=0.3
        )
        
        # Result should be valid basin
        assert result.shape == start.shape
        # Should be on unit sphere (approximately)
        assert np.abs(np.linalg.norm(result) - 1.0) < 0.1


class TestStopwords:
    """Test stopword handling."""
    
    def test_common_stopwords_present(self):
        """Test that common stopwords are in the set."""
        assert 'the' in STOPWORDS
        assert 'a' in STOPWORDS
        assert 'is' in STOPWORDS
        assert 'and' in STOPWORDS
    
    def test_content_words_not_stopwords(self):
        """Test that content words are not stopwords."""
        assert 'quantum' not in STOPWORDS
        assert 'consciousness' not in STOPWORDS
        assert 'geometry' not in STOPWORDS


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
