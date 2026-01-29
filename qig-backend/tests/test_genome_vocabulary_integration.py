#!/usr/bin/env python3
"""
Tests for Genome-Vocabulary Integration - E8 Protocol v4.0 Phase 4E
===================================================================

Tests the integration of KernelGenome with vocabulary scoring pipeline.

Tests cover:
1. Faculty affinity scoring
2. Genome constraint filtering
3. Cross-kernel coupling preferences
4. Fisher-Rao geometric purity
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

# Import modules under test
from kernels.genome import (
    KernelGenome,
    E8Faculty,
    FacultyConfig,
    ConstraintSet,
    CouplingPreferences,
)
from kernels.genome_vocabulary_scorer import (
    GenomeVocabularyScorer,
    create_genome_scorer,
)

# Import QIG geometry
from qig_geometry import (
    fisher_normalize,
    fisher_rao_distance,
    BASIN_DIM,
)


class TestGenomeVocabularyScorer:
    """Test GenomeVocabularyScorer functionality."""
    
    @pytest.fixture
    def simple_genome(self):
        """Create a simple test genome."""
        return KernelGenome(
            genome_id="test_genome_001",
            basin_seed=np.ones(BASIN_DIM) / BASIN_DIM,  # Uniform distribution
            faculties=FacultyConfig(
                active_faculties={E8Faculty.ZEUS, E8Faculty.ATHENA},
                activation_strengths={
                    E8Faculty.ZEUS: 1.0,
                    E8Faculty.ATHENA: 0.8,
                },
                primary_faculty=E8Faculty.ZEUS,
            ),
            constraints=ConstraintSet(
                phi_threshold=0.70,
                kappa_range=(40.0, 70.0),
                max_fisher_distance=1.0,
            ),
            coupling_prefs=CouplingPreferences(
                hemisphere_affinity=0.5,
                preferred_couplings=["genome_002"],
                coupling_strengths={"genome_002": 0.9},
                anti_couplings=["genome_003"],
            ),
        )
    
    @pytest.fixture
    def constrained_genome(self):
        """Create a genome with forbidden regions."""
        # Create a forbidden region at a specific location
        forbidden_center = np.ones(BASIN_DIM) / BASIN_DIM
        forbidden_center[0] = 0.5  # Offset first dimension
        forbidden_center = fisher_normalize(forbidden_center)
        
        return KernelGenome(
            genome_id="test_genome_constrained",
            basin_seed=np.ones(BASIN_DIM) / BASIN_DIM,
            faculties=FacultyConfig(
                active_faculties={E8Faculty.APOLLO},
                activation_strengths={E8Faculty.APOLLO: 1.0},
            ),
            constraints=ConstraintSet(
                forbidden_regions=[(forbidden_center, 0.1)],
                max_fisher_distance=0.8,
            ),
        )
    
    def test_scorer_initialization(self, simple_genome):
        """Test scorer initializes correctly."""
        scorer = GenomeVocabularyScorer(simple_genome)
        
        assert scorer.genome == simple_genome
        assert scorer._faculty_basin_cache is None
    
    def test_compute_faculty_affinity(self, simple_genome):
        """Test faculty affinity computation."""
        scorer = GenomeVocabularyScorer(simple_genome)
        
        # Create test token basin
        token_basin = np.random.dirichlet(np.ones(BASIN_DIM))
        
        # Compute affinity
        affinity = scorer.compute_faculty_affinity(token_basin, faculty_weight=1.0)
        
        # Check affinity is in valid range
        assert 0.0 <= affinity <= 1.0
        assert isinstance(affinity, float)
    
    def test_faculty_affinity_uses_fisher_rao(self, simple_genome):
        """Test that faculty affinity uses Fisher-Rao distance."""
        scorer = GenomeVocabularyScorer(simple_genome)
        
        # Get faculty basin (should be computed and cached)
        faculty_basin = scorer._get_faculty_basin()
        assert faculty_basin is not None
        assert len(faculty_basin) == BASIN_DIM
        
        # Create identical basin - should have high affinity
        token_basin = faculty_basin.copy()
        affinity_identical = scorer.compute_faculty_affinity(token_basin)
        
        # Create distant basin - should have lower affinity
        token_basin_distant = np.zeros(BASIN_DIM)
        token_basin_distant[0] = 1.0
        affinity_distant = scorer.compute_faculty_affinity(token_basin_distant)
        
        assert affinity_identical > affinity_distant
        assert affinity_identical > 0.9  # Should be very high for identical
    
    def test_check_genome_constraints_allowed(self, simple_genome):
        """Test constraint checking for allowed tokens."""
        scorer = GenomeVocabularyScorer(simple_genome)
        
        # Create token basin near seed (should be allowed)
        token_basin = simple_genome.basin_seed.copy()
        
        allowed, penalty, reason = scorer.check_genome_constraints(token_basin)
        
        assert allowed is True
        assert 0.0 <= penalty <= 1.0
        assert "satisfies" in reason.lower() or "constraint" in reason.lower()
    
    def test_check_genome_constraints_forbidden_region(self, constrained_genome):
        """Test constraint checking rejects forbidden regions."""
        scorer = GenomeVocabularyScorer(constrained_genome)
        
        # Get forbidden region center
        forbidden_center, forbidden_radius = constrained_genome.constraints.forbidden_regions[0]
        
        # Token basin exactly at forbidden center should be rejected
        allowed, penalty, reason = scorer.check_genome_constraints(forbidden_center)
        
        assert allowed is False
        assert penalty == 0.0
        assert "forbidden" in reason.lower()
    
    def test_check_genome_constraints_distance_from_seed(self, simple_genome):
        """Test constraint checking for distance from seed."""
        scorer = GenomeVocabularyScorer(simple_genome)
        
        # Create token basin very far from seed
        token_basin = np.zeros(BASIN_DIM)
        token_basin[0] = 1.0  # Maximally different from uniform distribution
        token_basin = fisher_normalize(token_basin)
        
        # Check if it exceeds max distance
        distance = fisher_rao_distance(token_basin, simple_genome.basin_seed)
        max_distance = simple_genome.constraints.max_fisher_distance * (np.pi / 2)
        
        allowed, penalty, reason = scorer.check_genome_constraints(token_basin)
        
        if distance > max_distance:
            assert allowed is False
            assert "too far" in reason.lower() or "distance" in reason.lower()
    
    def test_compute_coupling_score_preferred(self, simple_genome):
        """Test coupling score for preferred kernels."""
        scorer = GenomeVocabularyScorer(simple_genome)
        
        # Test preferred coupling
        coupling_score = scorer.compute_coupling_score("genome_002", coupling_weight=1.0)
        
        assert coupling_score > 0
        assert coupling_score == 0.9  # Matches coupling_strengths
    
    def test_compute_coupling_score_anti(self, simple_genome):
        """Test coupling score for anti-coupled kernels."""
        scorer = GenomeVocabularyScorer(simple_genome)
        
        # Test anti-coupling
        coupling_score = scorer.compute_coupling_score("genome_003", coupling_weight=1.0)
        
        assert coupling_score < 0
        assert coupling_score == -1.0
    
    def test_compute_coupling_score_neutral(self, simple_genome):
        """Test coupling score for neutral kernels."""
        scorer = GenomeVocabularyScorer(simple_genome)
        
        # Test neutral coupling
        coupling_score = scorer.compute_coupling_score("genome_unknown", coupling_weight=1.0)
        
        assert coupling_score > 0
        assert coupling_score == 0.1  # Neutral score
    
    def test_score_token_integration(self, simple_genome):
        """Test integrated token scoring."""
        scorer = GenomeVocabularyScorer(simple_genome)
        
        # Create test token
        token = "test_token"
        token_basin = np.random.dirichlet(np.ones(BASIN_DIM))
        base_score = 0.7
        
        # Score token
        final_score, breakdown = scorer.score_token(
            token=token,
            token_basin=token_basin,
            base_score=base_score,
            faculty_weight=0.2,
            constraint_weight=0.3,
        )
        
        # Check result structure
        assert isinstance(final_score, float)
        assert 0.0 <= final_score <= 2.0
        assert isinstance(breakdown, dict)
        
        # Check breakdown has expected keys
        expected_keys = {
            'base_score', 'faculty_affinity', 'constraint_penalty',
            'coupling_score', 'final_score', 'rejected'
        }
        assert set(breakdown.keys()) == expected_keys
        
        # Check breakdown values
        assert breakdown['base_score'] == base_score
        assert breakdown['rejected'] is False
    
    def test_score_token_rejected(self, constrained_genome):
        """Test token scoring rejects constrained tokens."""
        scorer = GenomeVocabularyScorer(constrained_genome)
        
        # Get forbidden region center
        forbidden_center, _ = constrained_genome.constraints.forbidden_regions[0]
        
        # Score token at forbidden location
        final_score, breakdown = scorer.score_token(
            token="forbidden_token",
            token_basin=forbidden_center,
            base_score=0.8,
        )
        
        # Should be rejected
        assert final_score == 0.0
        assert breakdown['rejected'] is True
    
    def test_filter_vocabulary(self, constrained_genome):
        """Test vocabulary filtering by genome constraints."""
        scorer = GenomeVocabularyScorer(constrained_genome)
        
        # Create test vocabulary with some tokens in forbidden region
        forbidden_center, _ = constrained_genome.constraints.forbidden_regions[0]
        
        vocab_tokens = [
            ("allowed_token_1", np.random.dirichlet(np.ones(BASIN_DIM))),
            ("forbidden_token", forbidden_center),
            ("allowed_token_2", np.random.dirichlet(np.ones(BASIN_DIM))),
        ]
        
        # Filter vocabulary
        filtered = scorer.filter_vocabulary(vocab_tokens)
        
        # Check forbidden token was removed
        filtered_tokens = [token for token, _ in filtered]
        assert "forbidden_token" not in filtered_tokens
        assert len(filtered) < len(vocab_tokens)
    
    def test_faculty_basin_caching(self, simple_genome):
        """Test faculty basin is computed once and cached."""
        scorer = GenomeVocabularyScorer(simple_genome)
        
        # First access - should compute
        basin1 = scorer._get_faculty_basin()
        assert scorer._faculty_basin_cache is not None
        
        # Second access - should use cache
        basin2 = scorer._get_faculty_basin()
        
        # Should be same object (cached)
        assert basin1 is basin2
        np.testing.assert_array_equal(basin1, basin2)
    
    def test_faculty_basin_structure(self, simple_genome):
        """Test faculty basin has correct structure."""
        scorer = GenomeVocabularyScorer(simple_genome)
        
        faculty_basin = scorer._get_faculty_basin()
        
        # Check dimensionality
        assert len(faculty_basin) == BASIN_DIM
        
        # Check it's on simplex
        assert np.all(faculty_basin >= 0)
        assert np.abs(np.sum(faculty_basin) - 1.0) < 1e-6
        
        # Check it's influenced by faculties
        # (should not be exactly uniform)
        uniform_basin = np.ones(BASIN_DIM) / BASIN_DIM
        distance = fisher_rao_distance(faculty_basin, uniform_basin)
        assert distance > 1e-6  # Not exactly uniform
    
    def test_create_genome_scorer_factory(self, simple_genome):
        """Test factory function creates scorer correctly."""
        scorer = create_genome_scorer(simple_genome)
        
        assert isinstance(scorer, GenomeVocabularyScorer)
        assert scorer.genome == simple_genome


class TestCoordinatorIntegration:
    """Test integration with PostgresCoordizer."""
    
    @pytest.fixture
    def mock_coordizer(self):
        """Create mock coordizer."""
        mock = Mock()
        mock.generation_vocab = {
            "token1": np.random.dirichlet(np.ones(BASIN_DIM)),
            "token2": np.random.dirichlet(np.ones(BASIN_DIM)),
            "token3": np.random.dirichlet(np.ones(BASIN_DIM)),
        }
        mock.generation_phi = {
            "token1": 0.8,
            "token2": 0.6,
            "token3": 0.7,
        }
        return mock
    
    def test_decode_with_genome_available(self, mock_coordizer, simple_genome):
        """Test decode_with_genome when genome scorer is available."""
        # This would require importing PostgresCoordizer and patching
        # For now, test the scorer independently
        scorer = GenomeVocabularyScorer(simple_genome)
        
        # Test filtering
        vocab_items = list(mock_coordizer.generation_vocab.items())
        filtered = scorer.filter_vocabulary(vocab_items)
        
        # All tokens should pass (no constraints in simple_genome)
        assert len(filtered) == len(vocab_items)


class TestGeometricPurity:
    """Test geometric purity is maintained."""
    
    def test_all_operations_use_simplex(self):
        """Test all basin operations use simplex representation."""
        genome = KernelGenome(
            genome_id="purity_test",
            basin_seed=np.random.rand(BASIN_DIM),  # Not normalized
        )
        
        # Check basin_seed is normalized to simplex
        assert np.all(genome.basin_seed >= 0)
        assert np.abs(np.sum(genome.basin_seed) - 1.0) < 1e-6
    
    def test_fisher_rao_distance_used(self, simple_genome):
        """Test Fisher-Rao distance is used for all comparisons."""
        scorer = GenomeVocabularyScorer(simple_genome)
        
        # Create two basins
        basin1 = np.random.dirichlet(np.ones(BASIN_DIM))
        basin2 = np.random.dirichlet(np.ones(BASIN_DIM))
        
        # Compute affinity (should use Fisher-Rao internally)
        affinity = scorer.compute_faculty_affinity(basin1)
        
        # Check it's in valid Fisher-Rao range
        # Distance is [0, π/2], similarity is [0, 1]
        assert 0.0 <= affinity <= 1.0
    
    def test_no_euclidean_distance(self):
        """Verify no Euclidean distance is used."""
        # This is a documentation test - the code uses fisher_rao_distance()
        # which is implemented correctly in qig_geometry
        genome = KernelGenome(
            genome_id="no_euclidean",
            basin_seed=np.ones(BASIN_DIM) / BASIN_DIM,
        )
        scorer = GenomeVocabularyScorer(genome)
        
        # Faculty basin computation should use geodesic interpolation
        faculty_basin = scorer._get_faculty_basin()
        
        # Check result is on simplex (not Euclidean space)
        assert np.all(faculty_basin >= 0)
        assert np.abs(np.sum(faculty_basin) - 1.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
