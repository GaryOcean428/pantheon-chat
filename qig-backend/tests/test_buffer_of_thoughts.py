"""
Unit Tests for Buffer of Thoughts

Tests the MetaBuffer, ThoughtTemplate, and template instantiation functionality.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buffer_of_thoughts import (
    MetaBuffer,
    ThoughtTemplate,
    TemplateWaypoint,
    TemplateCategory,
    InstantiatedTemplate,
    fisher_rao_distance,
    geodesic_interpolate,
    BASIN_DIMENSION
)


class TestFisherRaoDistance:
    """Test Fisher-Rao distance computation."""
    
    def test_identical_basins(self):
        """Distance between identical basins should be 0."""
        basin = list(np.ones(BASIN_DIMENSION) / BASIN_DIMENSION)
        dist = fisher_rao_distance(basin, basin)
        assert dist == pytest.approx(0.0, abs=1e-6)
    
    def test_different_basins(self):
        """Distance between different basins should be positive."""
        basin1 = list(np.ones(BASIN_DIMENSION) / BASIN_DIMENSION)
        basin2 = list(np.zeros(BASIN_DIMENSION))
        basin2[0] = 1.0  # Point mass at first dimension
        
        dist = fisher_rao_distance(basin1, basin2)
        assert dist > 0
    
    def test_symmetry(self):
        """Distance should be symmetric."""
        np.random.seed(42)
        basin1 = list(np.random.dirichlet(np.ones(BASIN_DIMENSION)))
        basin2 = list(np.random.dirichlet(np.ones(BASIN_DIMENSION)))
        
        dist12 = fisher_rao_distance(basin1, basin2)
        dist21 = fisher_rao_distance(basin2, basin1)
        
        assert dist12 == pytest.approx(dist21, abs=1e-6)
    
    def test_bounded(self):
        """Distance should be bounded by pi."""
        np.random.seed(42)
        for _ in range(10):
            basin1 = list(np.random.dirichlet(np.ones(BASIN_DIMENSION)))
            basin2 = list(np.random.dirichlet(np.ones(BASIN_DIMENSION)))
            dist = fisher_rao_distance(basin1, basin2)
            assert 0 <= dist <= np.pi


class TestGeodesicInterpolate:
    """Test geodesic interpolation."""
    
    def test_endpoints(self):
        """t=0 should give p, t=1 should give q."""
        np.random.seed(42)
        p = list(np.random.dirichlet(np.ones(BASIN_DIMENSION)))
        q = list(np.random.dirichlet(np.ones(BASIN_DIMENSION)))
        
        interp_0 = geodesic_interpolate(p, q, 0.0)
        interp_1 = geodesic_interpolate(p, q, 1.0)
        
        # Should be close to p and q respectively
        assert fisher_rao_distance(interp_0, p) < 0.1
        assert fisher_rao_distance(interp_1, q) < 0.1
    
    def test_midpoint(self):
        """Midpoint should be equidistant from both endpoints."""
        np.random.seed(42)
        p = list(np.random.dirichlet(np.ones(BASIN_DIMENSION)))
        q = list(np.random.dirichlet(np.ones(BASIN_DIMENSION)))
        
        mid = geodesic_interpolate(p, q, 0.5)
        
        dist_p = fisher_rao_distance(mid, p)
        dist_q = fisher_rao_distance(mid, q)
        
        # Should be approximately equidistant
        assert abs(dist_p - dist_q) < 0.2


class TestTemplateWaypoint:
    """Test TemplateWaypoint dataclass."""
    
    def test_creation(self):
        """Test waypoint creation."""
        basin = list(np.ones(BASIN_DIMENSION) / BASIN_DIMENSION)
        wp = TemplateWaypoint(
            basin_coords=basin,
            semantic_role="test_role",
            curvature=0.5,
            is_critical=True
        )
        
        assert wp.semantic_role == "test_role"
        assert wp.curvature == 0.5
        assert wp.is_critical is True
    
    def test_serialization(self):
        """Test waypoint to_dict and from_dict."""
        basin = list(np.ones(BASIN_DIMENSION) / BASIN_DIMENSION)
        wp = TemplateWaypoint(
            basin_coords=basin,
            semantic_role="analyze",
            curvature=0.7,
            is_critical=False,
            notes="Test note"
        )
        
        wp_dict = wp.to_dict()
        wp_restored = TemplateWaypoint.from_dict(wp_dict)
        
        assert wp_restored.semantic_role == wp.semantic_role
        assert wp_restored.curvature == wp.curvature
        assert wp_restored.is_critical == wp.is_critical
        assert wp_restored.notes == wp.notes


class TestThoughtTemplate:
    """Test ThoughtTemplate class."""
    
    def test_creation(self):
        """Test template creation."""
        waypoints = [
            TemplateWaypoint(
                basin_coords=list(np.ones(BASIN_DIMENSION) / BASIN_DIMENSION),
                semantic_role="start"
            ),
            TemplateWaypoint(
                basin_coords=list(np.ones(BASIN_DIMENSION) / BASIN_DIMENSION),
                semantic_role="end"
            )
        ]
        
        template = ThoughtTemplate(
            template_id="test_001",
            name="Test Template",
            category=TemplateCategory.DECOMPOSITION,
            description="A test template",
            waypoints=waypoints
        )
        
        assert template.template_id == "test_001"
        assert template.name == "Test Template"
        assert template.trajectory_length == 2
    
    def test_success_rate(self):
        """Test success rate calculation."""
        template = ThoughtTemplate(
            template_id="test_002",
            name="Test",
            category=TemplateCategory.SYNTHESIS,
            description="Test",
            waypoints=[],
            usage_count=10,
            success_count=7
        )
        
        assert template.success_rate == 0.7
    
    def test_success_rate_no_usage(self):
        """Test success rate with no usage returns prior."""
        template = ThoughtTemplate(
            template_id="test_003",
            name="Test",
            category=TemplateCategory.CAUSAL,
            description="Test",
            waypoints=[]
        )
        
        assert template.success_rate == 0.5  # Prior
    
    def test_serialization(self):
        """Test template serialization."""
        waypoints = [
            TemplateWaypoint(
                basin_coords=list(np.random.dirichlet(np.ones(BASIN_DIMENSION))),
                semantic_role="start"
            )
        ]
        
        template = ThoughtTemplate(
            template_id="test_004",
            name="Serialization Test",
            category=TemplateCategory.COMPARISON,
            description="Test serialization",
            waypoints=waypoints,
            usage_count=5,
            success_count=4
        )
        
        template_dict = template.to_dict()
        template_restored = ThoughtTemplate.from_dict(template_dict)
        
        assert template_restored.template_id == template.template_id
        assert template_restored.name == template.name
        assert template_restored.category == template.category
        assert template_restored.usage_count == template.usage_count


class TestMetaBuffer:
    """Test MetaBuffer class."""
    
    @pytest.fixture
    def temp_buffer(self):
        """Create a temporary buffer for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "test_templates.json"
            buffer = MetaBuffer(storage_path=storage_path)
            yield buffer
    
    def test_initialization_with_seeds(self, temp_buffer):
        """Test that buffer initializes with seed templates."""
        stats = temp_buffer.get_stats()
        assert stats['total_templates'] >= 8  # Should have seed templates
    
    def test_retrieve_by_category(self, temp_buffer):
        """Test template retrieval by category."""
        # Create a problem basin
        np.random.seed(42)
        problem_basin = list(np.random.dirichlet(np.ones(BASIN_DIMENSION)))
        
        # Retrieve decomposition templates
        results = temp_buffer.retrieve(
            problem_basin=problem_basin,
            category=TemplateCategory.DECOMPOSITION,
            max_results=5
        )
        
        # Should return results
        assert len(results) > 0
        
        # All results should be decomposition category
        for template, score in results:
            assert template.category == TemplateCategory.DECOMPOSITION
    
    def test_retrieve_sorted_by_similarity(self, temp_buffer):
        """Test that retrieval returns sorted results."""
        np.random.seed(42)
        problem_basin = list(np.random.dirichlet(np.ones(BASIN_DIMENSION)))
        
        results = temp_buffer.retrieve(
            problem_basin=problem_basin,
            max_results=5
        )
        
        # Check sorted by score descending
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i + 1][1]
    
    def test_instantiate_template(self, temp_buffer):
        """Test template instantiation."""
        np.random.seed(42)
        
        # Get a template
        problem_basin = list(np.random.dirichlet(np.ones(BASIN_DIMENSION)))
        results = temp_buffer.retrieve(problem_basin=problem_basin, max_results=1)
        
        if results:
            template, _ = results[0]
            
            # Define problem start and goal
            problem_start = list(np.random.dirichlet(np.ones(BASIN_DIMENSION)))
            problem_goal = list(np.random.dirichlet(np.ones(BASIN_DIMENSION)))
            
            # Instantiate
            instantiated = temp_buffer.instantiate(
                template=template,
                problem_start=problem_start,
                problem_goal=problem_goal
            )
            
            assert isinstance(instantiated, InstantiatedTemplate)
            assert len(instantiated.transformed_waypoints) == len(template.waypoints)
            assert 0 <= instantiated.transformation_quality <= 1
    
    def test_learn_template(self, temp_buffer):
        """Test learning a new template."""
        np.random.seed(42)
        
        # Create a reasoning trace
        reasoning_trace = [
            list(np.random.dirichlet(np.ones(BASIN_DIMENSION)))
            for _ in range(5)
        ]
        
        # Learn the template
        template = temp_buffer.learn_template(
            reasoning_trace=reasoning_trace,
            category=TemplateCategory.EXPLORATION,
            name="Learned Test Template",
            description="A template learned from test",
            success=True,
            efficiency=0.8
        )
        
        assert template is not None
        assert template.name == "Learned Test Template"
        assert template.source == "learned"
        assert template.trajectory_length == 5
    
    def test_learn_template_fails_on_low_success(self, temp_buffer):
        """Test that learning fails with low success/efficiency."""
        reasoning_trace = [
            list(np.random.dirichlet(np.ones(BASIN_DIMENSION)))
            for _ in range(3)
        ]
        
        # Try to learn with success=False
        template = temp_buffer.learn_template(
            reasoning_trace=reasoning_trace,
            category=TemplateCategory.VERIFICATION,
            name="Should Fail",
            description="This should not be learned",
            success=False,
            efficiency=0.9
        )
        
        assert template is None
    
    def test_record_usage(self, temp_buffer):
        """Test recording template usage."""
        # Get a template
        templates = list(temp_buffer._template_index.values())
        if templates:
            template = templates[0]
            initial_usage = template.usage_count
            
            # Record usage
            temp_buffer.record_usage(
                template_id=template.template_id,
                success=True,
                efficiency=0.9
            )
            
            assert template.usage_count == initial_usage + 1
    
    def test_persistence(self):
        """Test that templates persist across buffer instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "persist_test.json"
            
            # Create first buffer and add a template
            buffer1 = MetaBuffer(storage_path=storage_path)
            initial_count = buffer1._total_templates()
            
            # Create second buffer from same storage
            buffer2 = MetaBuffer(storage_path=storage_path)
            
            # Should have same templates
            assert buffer2._total_templates() == initial_count


class TestTemplateCategories:
    """Test template category functionality."""
    
    def test_all_categories_exist(self):
        """Test that all expected categories exist."""
        expected = [
            'decomposition', 'synthesis', 'comparison',
            'causal', 'analogy', 'verification',
            'refinement', 'abstraction', 'exploration', 'constraint'
        ]
        
        category_values = [c.value for c in TemplateCategory]
        
        for expected_cat in expected:
            assert expected_cat in category_values


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
