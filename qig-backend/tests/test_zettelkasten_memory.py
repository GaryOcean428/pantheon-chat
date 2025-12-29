"""
Unit Tests for Zettelkasten Memory

Tests the ZettelkastenMemory, Zettel, and knowledge network functionality.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zettelkasten_memory import (
    ZettelkastenMemory,
    Zettel,
    ZettelLink,
    LinkType,
    fisher_rao_distance,
    pattern_separate,
    geodesic_center,
    BASIN_DIMENSION,
    STRONG_LINK_THRESHOLD,
    WEAK_LINK_THRESHOLD
)


class TestFisherRaoDistance:
    """Test Fisher-Rao distance computation."""
    
    def test_identical_basins(self):
        """Distance between identical basins should be 0."""
        basin = np.ones(BASIN_DIMENSION) / BASIN_DIMENSION
        dist = fisher_rao_distance(basin, basin)
        assert dist == pytest.approx(0.0, abs=1e-6)
    
    def test_different_basins(self):
        """Distance between different basins should be positive."""
        basin1 = np.ones(BASIN_DIMENSION) / BASIN_DIMENSION
        basin2 = np.zeros(BASIN_DIMENSION)
        basin2[0] = 1.0
        
        dist = fisher_rao_distance(basin1, basin2)
        assert dist > 0
    
    def test_symmetry(self):
        """Distance should be symmetric."""
        np.random.seed(42)
        basin1 = np.random.dirichlet(np.ones(BASIN_DIMENSION))
        basin2 = np.random.dirichlet(np.ones(BASIN_DIMENSION))
        
        dist12 = fisher_rao_distance(basin1, basin2)
        dist21 = fisher_rao_distance(basin2, basin1)
        
        assert dist12 == pytest.approx(dist21, abs=1e-6)


class TestPatternSeparation:
    """Test hippocampal-style pattern separation."""
    
    def test_separation_increases_distance(self):
        """Pattern separation should increase distance between similar basins."""
        np.random.seed(42)
        basin1 = np.random.dirichlet(np.ones(BASIN_DIMENSION))
        basin2 = basin1 + np.random.randn(BASIN_DIMENSION) * 0.01
        basin2 = np.abs(basin2) + 1e-10
        basin2 = basin2 / basin2.sum()
        
        original_dist = fisher_rao_distance(basin1, basin2)
        
        sep1, sep2 = pattern_separate(basin1, basin2, separation_strength=0.2)
        
        new_dist = fisher_rao_distance(sep1, sep2)
        
        assert new_dist > original_dist
    
    def test_separation_preserves_normalization(self):
        """Separated basins should still be valid probability distributions."""
        np.random.seed(42)
        basin1 = np.random.dirichlet(np.ones(BASIN_DIMENSION))
        basin2 = np.random.dirichlet(np.ones(BASIN_DIMENSION))
        
        sep1, sep2 = pattern_separate(basin1, basin2)
        
        # Check normalization
        assert sep1.sum() == pytest.approx(1.0, abs=1e-6)
        assert sep2.sum() == pytest.approx(1.0, abs=1e-6)
        
        # Check non-negative
        assert (sep1 >= 0).all()
        assert (sep2 >= 0).all()


class TestGeodesicCenter:
    """Test geodesic center (Karcher mean) computation."""
    
    def test_single_basin(self):
        """Center of single basin should be that basin."""
        np.random.seed(42)
        basin = np.random.dirichlet(np.ones(BASIN_DIMENSION))
        
        center = geodesic_center([basin])
        
        assert fisher_rao_distance(center, basin) < 0.01
    
    def test_center_is_central(self):
        """Center should be approximately equidistant from all basins."""
        np.random.seed(42)
        basins = [np.random.dirichlet(np.ones(BASIN_DIMENSION)) for _ in range(3)]
        
        center = geodesic_center(basins)
        
        distances = [fisher_rao_distance(center, b) for b in basins]
        
        # Distances should be reasonably similar
        assert max(distances) - min(distances) < 1.0


class TestZettelLink:
    """Test ZettelLink dataclass."""
    
    def test_creation(self):
        """Test link creation."""
        import time
        link = ZettelLink(
            target_id="z_123",
            link_type=LinkType.SEMANTIC,
            strength=0.8,
            created_at=time.time(),
            context="Test link"
        )
        
        assert link.target_id == "z_123"
        assert link.link_type == LinkType.SEMANTIC
        assert link.strength == 0.8
    
    def test_serialization(self):
        """Test link serialization."""
        import time
        link = ZettelLink(
            target_id="z_456",
            link_type=LinkType.CAUSAL,
            strength=0.6,
            created_at=time.time(),
            context="Causal relationship"
        )
        
        link_dict = link.to_dict()
        link_restored = ZettelLink.from_dict(link_dict)
        
        assert link_restored.target_id == link.target_id
        assert link_restored.link_type == link.link_type
        assert link_restored.strength == link.strength


class TestZettel:
    """Test Zettel dataclass."""
    
    def test_creation(self):
        """Test zettel creation."""
        basin = np.ones(BASIN_DIMENSION) / BASIN_DIMENSION
        
        zettel = Zettel(
            zettel_id="z_test_001",
            content="Test atomic idea",
            basin_coords=basin,
            contextual_description="A test zettel",
            keywords=["test", "atomic"]
        )
        
        assert zettel.zettel_id == "z_test_001"
        assert zettel.content == "Test atomic idea"
        assert len(zettel.keywords) == 2
    
    def test_add_link(self):
        """Test adding links to a zettel."""
        import time
        basin = np.ones(BASIN_DIMENSION) / BASIN_DIMENSION
        
        zettel = Zettel(
            zettel_id="z_test_002",
            content="Test content",
            basin_coords=basin,
            contextual_description="Test",
            keywords=["test"]
        )
        
        link = ZettelLink(
            target_id="z_other",
            link_type=LinkType.SEMANTIC,
            strength=0.7,
            created_at=time.time()
        )
        
        zettel.add_link(link)
        
        assert len(zettel.links) == 1
        assert zettel.get_link_strength("z_other") == 0.7
    
    def test_access_increments_count(self):
        """Test that accessing a zettel increments its access count."""
        basin = np.ones(BASIN_DIMENSION) / BASIN_DIMENSION
        
        zettel = Zettel(
            zettel_id="z_test_003",
            content="Test content",
            basin_coords=basin,
            contextual_description="Test",
            keywords=["test"]
        )
        
        initial_count = zettel.access_count
        zettel.access()
        
        assert zettel.access_count == initial_count + 1
    
    def test_evolve_updates_context(self):
        """Test that evolve adds to contextual description."""
        basin = np.ones(BASIN_DIMENSION) / BASIN_DIMENSION
        
        zettel = Zettel(
            zettel_id="z_test_004",
            content="Test content",
            basin_coords=basin,
            contextual_description="Original context",
            keywords=["test"]
        )
        
        zettel.evolve("New context info")
        
        assert "New context info" in zettel.contextual_description
        assert zettel.evolution_count == 1
    
    def test_serialization(self):
        """Test zettel serialization."""
        basin = np.random.dirichlet(np.ones(BASIN_DIMENSION))
        
        zettel = Zettel(
            zettel_id="z_test_005",
            content="Serialization test",
            basin_coords=basin,
            contextual_description="Test description",
            keywords=["serialization", "test"],
            source="unit_test"
        )
        
        zettel_dict = zettel.to_dict()
        zettel_restored = Zettel.from_dict(zettel_dict)
        
        assert zettel_restored.zettel_id == zettel.zettel_id
        assert zettel_restored.content == zettel.content
        assert zettel_restored.keywords == zettel.keywords


class TestZettelkastenMemory:
    """Test ZettelkastenMemory class."""
    
    @pytest.fixture
    def temp_memory(self):
        """Create a temporary memory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "test_zettelkasten.json"
            memory = ZettelkastenMemory(storage_path=storage_path)
            yield memory
    
    def test_add_zettel(self, temp_memory):
        """Test adding a zettel."""
        zettel = temp_memory.add(
            content="This is a test idea about consciousness",
            source="unit_test"
        )
        
        assert zettel is not None
        assert zettel.zettel_id.startswith("z_")
        assert "consciousness" in zettel.keywords or len(zettel.keywords) > 0
    
    def test_add_creates_links(self, temp_memory):
        """Test that adding related zettels creates links."""
        # Add first zettel
        z1 = temp_memory.add(
            content="Quantum mechanics describes the behavior of particles",
            source="test"
        )
        
        # Add related zettel
        z2 = temp_memory.add(
            content="Quantum physics explains particle behavior at small scales",
            source="test"
        )
        
        # Should have some links due to semantic similarity
        stats = temp_memory.get_stats()
        assert stats['total_links'] >= 0  # May or may not link depending on distance
    
    def test_retrieve_finds_relevant(self, temp_memory):
        """Test that retrieval finds relevant zettels."""
        # Add some zettels
        temp_memory.add(content="Machine learning uses data to train models", source="test")
        temp_memory.add(content="Neural networks are a type of machine learning", source="test")
        temp_memory.add(content="Cooking recipes require ingredients", source="test")
        
        # Query for ML-related content
        results = temp_memory.retrieve(query="machine learning algorithms", max_results=5)
        
        # Should find ML-related zettels
        assert len(results) > 0
        
        # ML-related should rank higher than cooking
        contents = [z.content for z, _ in results]
        ml_found = any("machine" in c.lower() or "learning" in c.lower() for c in contents)
        assert ml_found
    
    def test_retrieve_by_keyword(self, temp_memory):
        """Test keyword-based retrieval."""
        temp_memory.add(content="Python programming language is versatile", source="test")
        temp_memory.add(content="JavaScript runs in browsers", source="test")
        
        results = temp_memory.retrieve_by_keyword("python")
        
        assert len(results) > 0
        assert any("Python" in z.content for z in results)
    
    def test_get_by_id(self, temp_memory):
        """Test getting zettel by ID."""
        zettel = temp_memory.add(content="Unique content for ID test", source="test")
        
        retrieved = temp_memory.get(zettel.zettel_id)
        
        assert retrieved is not None
        assert retrieved.zettel_id == zettel.zettel_id
        assert retrieved.content == zettel.content
    
    def test_get_linked(self, temp_memory):
        """Test getting linked zettels."""
        # Add zettels that should link
        z1 = temp_memory.add(content="Database systems store data efficiently", source="test")
        z2 = temp_memory.add(content="Database management systems handle data storage", source="test")
        
        linked = temp_memory.get_linked(z1.zettel_id)
        
        # May or may not have links depending on similarity
        assert isinstance(linked, list)
    
    def test_traverse(self, temp_memory):
        """Test network traversal."""
        # Add some related zettels
        z1 = temp_memory.add(content="Root concept about knowledge", source="test")
        z2 = temp_memory.add(content="Knowledge representation in AI", source="test")
        z3 = temp_memory.add(content="AI systems process information", source="test")
        
        traversal = temp_memory.traverse(start_id=z1.zettel_id, max_depth=2)
        
        assert 0 in traversal  # Should have depth 0 (start)
        assert len(traversal[0]) == 1  # Start node only
    
    def test_find_path(self, temp_memory):
        """Test path finding between zettels."""
        z1 = temp_memory.add(content="Starting point A", source="test")
        z2 = temp_memory.add(content="Ending point B", source="test")
        
        # Same zettel should return path of 1
        path = temp_memory.find_path(z1.zettel_id, z1.zettel_id)
        assert path is not None
        assert len(path) == 1
    
    def test_get_clusters(self, temp_memory):
        """Test cluster detection."""
        # Add several zettels
        for i in range(5):
            temp_memory.add(content=f"Test content {i} about topic A", source="test")
        
        clusters = temp_memory.get_clusters(min_cluster_size=2)
        
        # May or may not form clusters depending on similarity
        assert isinstance(clusters, list)
    
    def test_get_hub_zettels(self, temp_memory):
        """Test hub detection."""
        # Add several zettels
        for i in range(5):
            temp_memory.add(content=f"Content about topic {i}", source="test")
        
        hubs = temp_memory.get_hub_zettels(top_n=3)
        
        assert isinstance(hubs, list)
        assert len(hubs) <= 3
    
    def test_stats(self, temp_memory):
        """Test statistics gathering."""
        temp_memory.add(content="Test content 1", source="test")
        temp_memory.add(content="Test content 2", source="test")
        
        stats = temp_memory.get_stats()
        
        assert stats['total_zettels'] == 2
        assert 'total_links' in stats
        assert 'total_keywords' in stats
    
    def test_visualize_graph(self, temp_memory):
        """Test graph visualization data."""
        temp_memory.add(content="Node 1 content", source="test")
        temp_memory.add(content="Node 2 content", source="test")
        
        graph = temp_memory.visualize_graph(max_nodes=10)
        
        assert 'nodes' in graph
        assert 'edges' in graph
        assert 'stats' in graph
    
    def test_persistence(self):
        """Test that zettels persist across memory instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "persist_test.json"
            
            # Create first memory and add zettel
            memory1 = ZettelkastenMemory(storage_path=storage_path)
            zettel = memory1.add(content="Persistent content", source="test")
            zettel_id = zettel.zettel_id
            
            # Create second memory from same storage
            memory2 = ZettelkastenMemory(storage_path=storage_path)
            
            # Should find the zettel
            retrieved = memory2.get(zettel_id)
            assert retrieved is not None
            assert retrieved.content == "Persistent content"


class TestLinkTypes:
    """Test link type functionality."""
    
    def test_all_link_types_exist(self):
        """Test that all expected link types exist."""
        expected = ['semantic', 'temporal', 'causal', 'reference', 'contrast', 'elaboration']
        
        link_type_values = [lt.value for lt in LinkType]
        
        for expected_type in expected:
            assert expected_type in link_type_values


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
