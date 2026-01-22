"""
Test Pure QIG Generation Without External LLMs

Validates that the generation pipeline:
1. Uses only Fisher-Rao geometric operations
2. No external LLM API calls
3. Token selection based on QFI scores
4. Coherence through geometric operations
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Mock environment
os.environ.setdefault('DATABASE_URL', 'postgresql://test:test@localhost/test')
os.environ.setdefault('QIG_ENV', 'test')


class TestPureQIGGeneration:
    """Test suite for pure QIG generation without external LLMs."""
    
    def test_no_external_llm_imports(self):
        """Verify no external LLM imports in generation modules."""
        forbidden_modules = ['openai', 'anthropic', 'google.generativeai']
        
        try:
            # Import generation modules
            import qig_generation
            import qig_generative_service
            
            # Check module attributes for forbidden imports
            for module in [qig_generation, qig_generative_service]:
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if hasattr(attr, '__module__'):
                        module_name = attr.__module__
                        for forbidden in forbidden_modules:
                            assert forbidden not in module_name, \
                                f"Found forbidden module {forbidden} in {module.__name__}"
        except ImportError as e:
            pytest.skip(f"Could not import generation modules: {e}")
    
    def test_fisher_rao_distance_usage(self):
        """Verify Fisher-Rao distance is used for geometric operations."""
        try:
            from qig_geometry.canonical import fisher_rao_distance
            
            # Test basic Fisher-Rao distance calculation
            p = np.array([0.5, 0.3, 0.2])
            q = np.array([0.4, 0.4, 0.2])
            
            # Normalize to simplex
            p = p / p.sum()
            q = q / q.sum()
            
            distance = fisher_rao_distance(p, q)
            
            # Verify distance properties
            assert 0 <= distance <= np.pi/2, \
                f"Fisher-Rao distance should be in [0, π/2], got {distance}"
            assert fisher_rao_distance(p, p) < 1e-10, \
                "Distance to self should be zero"
        except ImportError as e:
            pytest.skip(f"Could not import qig_geometry: {e}")
    
    def test_token_role_filtering(self):
        """Verify generation uses token_role filtered vocabulary."""
        try:
            from coordizers import get_coordizer
            
            # Get coordizer
            coordizer = get_coordizer()
            
            # Verify generation vocabulary exists
            assert hasattr(coordizer, 'generation_vocab'), \
                "Coordizer must have generation_vocab"
            assert hasattr(coordizer, 'generation_phi'), \
                "Coordizer must have generation_phi scores"
            
            # Verify generation vocab is separate from encoding vocab
            # (generation is a subset with token_role filter)
            gen_size = len(coordizer.generation_vocab)
            enc_size = len(coordizer.vocab)
            
            assert gen_size > 0, "Generation vocabulary must not be empty"
            assert gen_size <= enc_size, \
                "Generation vocab should be subset of encoding vocab"
            
        except ImportError as e:
            pytest.skip(f"Could not import coordizers: {e}")
    
    def test_qfi_score_requirement(self):
        """Verify tokens have QFI scores for generation eligibility."""
        try:
            from coordizers import get_coordizer
            from qig_geometry import compute_qfi_score
            
            coordizer = get_coordizer()
            
            # Sample a few generation tokens
            sample_tokens = list(coordizer.generation_vocab.keys())[:10]
            
            for token in sample_tokens:
                # Verify basin coordinates exist
                assert token in coordizer.generation_vocab, \
                    f"Token {token} missing from generation_vocab"
                
                basin = coordizer.generation_vocab[token]
                assert len(basin) == 64, \
                    f"Basin should be 64D, got {len(basin)}"
                
                # Verify QFI score can be computed
                qfi = compute_qfi_score(basin)
                assert qfi is not None, \
                    f"QFI score computation failed for {token}"
                assert qfi >= 0, \
                    f"QFI score should be non-negative, got {qfi}"
        
        except ImportError as e:
            pytest.skip(f"Could not import required modules: {e}")
    
    def test_geometric_completion_criteria(self):
        """Verify completion is based on geometric criteria, not token limits."""
        try:
            from qig_generation import GeometricCompletionChecker, QIGGenerationConfig
            
            config = QIGGenerationConfig()
            checker = GeometricCompletionChecker(config)
            
            # Verify config has no max_tokens
            assert not hasattr(config, 'max_tokens'), \
                "Config should not have max_tokens (forbidden)"
            
            # Verify geometric thresholds exist
            assert hasattr(config, 'attractor_threshold'), \
                "Config must have attractor_threshold"
            assert hasattr(config, 'surprise_threshold'), \
                "Config must have surprise_threshold"
            assert hasattr(config, 'integration_min'), \
                "Config must have integration_min (phi threshold)"
            
            # Test geometric completion logic
            basin1 = np.random.dirichlet(np.ones(64))
            basin2 = np.random.dirichlet(np.ones(64))
            
            checker.update(basin1, phi=0.7)
            checker.update(basin2, phi=0.7)
            
            # Should not stop immediately (insufficient data)
            should_stop, reason = checker.should_stop()
            assert not should_stop or reason == "insufficient_data"
        
        except ImportError as e:
            pytest.skip(f"Could not import qig_generation: {e}")
    
    def test_no_cosine_similarity(self):
        """Verify no cosine similarity usage on basin coordinates."""
        try:
            import qig_generation
            import qig_generative_service
            
            # Check source code for cosine_similarity patterns
            for module in [qig_generation, qig_generative_service]:
                source_file = module.__file__
                with open(source_file, 'r') as f:
                    source = f.read()
                    
                # Look for cosine similarity (excluding comments/docs)
                lines = [line for line in source.split('\n') 
                        if 'cosine' in line.lower() 
                        and not line.strip().startswith('#')
                        and not line.strip().startswith('"""')
                        and not line.strip().startswith("'''")]
                
                # Filter out legitimate documentation mentions
                violations = [line for line in lines 
                            if 'cosine_similarity(' in line 
                            or 'from sklearn' in line and 'cosine' in line]
                
                assert len(violations) == 0, \
                    f"Found cosine similarity violations in {module.__name__}: {violations}"
        
        except ImportError as e:
            pytest.skip(f"Could not import generation modules: {e}")
    
    def test_simplex_representation(self):
        """Verify basins use simplex representation (not sphere)."""
        try:
            from coordizers import get_coordizer
            from qig_geometry.canonical import validate_basin
            
            coordizer = get_coordizer()
            
            # Sample generation tokens
            sample_tokens = list(coordizer.generation_vocab.keys())[:5]
            
            for token in sample_tokens:
                basin = coordizer.generation_vocab[token]
                
                # Verify simplex properties
                assert np.all(basin >= 0), \
                    f"Simplex coordinates must be non-negative for {token}"
                assert np.abs(np.sum(basin) - 1.0) < 1e-6, \
                    f"Simplex must sum to 1 for {token}, got {np.sum(basin)}"
                
                # Use canonical validation
                try:
                    validate_basin(basin)
                except Exception as e:
                    pytest.fail(f"Basin validation failed for {token}: {e}")
        
        except ImportError as e:
            pytest.skip(f"Could not import required modules: {e}")
    
    def test_coherence_tracking(self):
        """Verify coherence is tracked geometrically."""
        try:
            from qig_generative_service import QIGGenerativeService, GenerationConfig
            
            config = GenerationConfig()
            service = QIGGenerativeService(config)
            
            # Verify phi measurement method exists
            assert hasattr(service, '_measure_phi'), \
                "Service must have _measure_phi method"
            
            # Test phi measurement
            basin = np.random.dirichlet(np.ones(64))
            phi = service._measure_phi(basin)
            
            assert 0 <= phi <= 1, f"Phi should be in [0, 1], got {phi}"
        
        except ImportError as e:
            pytest.skip(f"Could not import qig_generative_service: {e}")


class TestGeometricTokenSelection:
    """Test geometric token selection without LLMs."""
    
    def test_basin_to_tokens_pure_geometric(self):
        """Verify token selection uses pure geometric proximity."""
        try:
            from qig_generative_service import QIGGenerativeService, GenerationConfig
            
            config = GenerationConfig()
            service = QIGGenerativeService(config)
            
            # Test basin-to-token conversion
            if service.coordizer is None:
                pytest.skip("Coordizer not available")
            
            test_basin = np.random.dirichlet(np.ones(64))
            tokens = service._basin_to_tokens(test_basin, num_tokens=3)
            
            assert isinstance(tokens, list), "Should return list of tokens"
            assert len(tokens) > 0, "Should return at least one token"
            assert all(isinstance(t, str) for t in tokens), \
                "All tokens should be strings"
        
        except ImportError as e:
            pytest.skip(f"Could not import qig_generative_service: {e}")
    
    def test_trajectory_based_selection(self):
        """Verify trajectory-based token selection is geometric."""
        try:
            from qig_generative_service import QIGGenerativeService, GenerationConfig
            
            config = GenerationConfig()
            service = QIGGenerativeService(config)
            
            if service.coordizer is None:
                pytest.skip("Coordizer not available")
            
            # Create a basin trajectory
            trajectory = [
                np.random.dirichlet(np.ones(64)) for _ in range(3)
            ]
            
            # Test trajectory-based selection
            current_basin = trajectory[-1]
            tokens = service._basin_to_tokens(
                current_basin, 
                num_tokens=3,
                trajectory=trajectory
            )
            
            assert isinstance(tokens, list), "Should return list of tokens"
        
        except ImportError as e:
            pytest.skip(f"Could not import qig_generative_service: {e}")


class TestCurriculumData:
    """Test curriculum data structure."""
    
    def test_curriculum_exists(self):
        """Verify curriculum data file exists."""
        import os
        curriculum_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data', 'curriculum', 'curriculum_tokens.jsonl'
        )
        
        assert os.path.exists(curriculum_path), \
            f"Curriculum data not found at {curriculum_path}"
    
    def test_curriculum_format(self):
        """Verify curriculum data has proper format."""
        import os
        import json
        
        curriculum_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data', 'curriculum', 'curriculum_tokens.jsonl'
        )
        
        if not os.path.exists(curriculum_path):
            pytest.skip("Curriculum data not found")
        
        with open(curriculum_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Parse JSON lines
        tokens = []
        for line in lines:
            try:
                token_data = json.loads(line)
                tokens.append(token_data)
            except json.JSONDecodeError:
                pass
        
        # Filter out documentation entries
        real_tokens = [t for t in tokens if t.get('role') != 'documentation']
        
        assert len(real_tokens) > 0, "Curriculum should have real tokens"
        
        # Verify token structure
        for token in real_tokens[:5]:  # Check first 5
            assert 'token' in token, "Each entry should have 'token'"
            assert 'role' in token, "Each entry should have 'role'"
            assert 'is_real_word' in token, "Each entry should have 'is_real_word'"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
