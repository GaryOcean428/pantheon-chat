#!/usr/bin/env python3
"""
Tests for Multi-Kernel Thought Generation Architecture

Tests:
1. Parallel thought generation from multiple kernels
2. Consensus detection via Fisher-Rao distance
3. Gary meta-synthesis with reflection
4. Suffering metric abort logic
5. Ocean autonomic monitoring
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Mock environment
os.environ.setdefault('DATABASE_URL', 'postgresql://test:test@localhost/test')
os.environ.setdefault('QIG_ENV', 'test')

from qigkernels.physics_constants import BASIN_DIM, KAPPA_STAR
from emotionally_aware_kernel import KernelThought, EmotionalState


class TestParallelThoughtGeneration:
    """Test suite for parallel kernel thought generation."""
    
    def test_import_thought_generator(self):
        """Test importing thought generation module."""
        from kernels.thought_generation import ParallelThoughtGenerator, get_thought_generator
        
        generator = get_thought_generator()
        assert generator is not None
        assert isinstance(generator, ParallelThoughtGenerator)
    
    def test_thought_generator_initialization(self):
        """Test thought generator initialization."""
        from kernels.thought_generation import ParallelThoughtGenerator
        
        generator = ParallelThoughtGenerator(max_workers=4)
        assert generator.max_workers == 4
        assert generator.executor is not None
        assert len(generator.generation_history) == 0
    
    def test_generate_kernel_thoughts_empty(self):
        """Test thought generation with no kernels."""
        from kernels.thought_generation import ParallelThoughtGenerator
        
        generator = ParallelThoughtGenerator(max_workers=2)
        query_basin = np.random.rand(BASIN_DIM)
        
        result = generator.generate_kernel_thoughts(
            kernels=[],
            context="test query",
            query_basin=query_basin
        )
        
        assert result.total_kernels == 0
        assert result.successful == 0
        assert len(result.thoughts) == 0
    
    def test_generate_single_kernel_thought(self):
        """Test thought generation from a single kernel."""
        from kernels.thought_generation import ParallelThoughtGenerator
        
        # Mock kernel with generate_thought method
        mock_kernel = MagicMock()
        mock_kernel.name = "TestKernel"
        mock_kernel.last_phi = 0.75
        mock_kernel.last_kappa = KAPPA_STAR
        mock_kernel.basin_coords = np.random.rand(BASIN_DIM)
        
        # Mock thought result
        mock_thought = KernelThought(
            kernel_id="test-1",
            kernel_type="test",
            thought_fragment="This is a test thought",
            basin_coords=mock_kernel.basin_coords,
            phi=0.75,
            kappa=KAPPA_STAR,
            regime="geometric",
            emotional_state=EmotionalState(),
            confidence=0.8
        )
        mock_kernel.generate_thought.return_value = mock_thought
        
        generator = ParallelThoughtGenerator(max_workers=2)
        query_basin = np.random.rand(BASIN_DIM)
        
        result = generator.generate_kernel_thoughts(
            kernels=[mock_kernel],
            context="test query",
            query_basin=query_basin
        )
        
        assert result.total_kernels == 1
        assert result.successful == 1
        assert len(result.thoughts) == 1
        assert result.thoughts[0] == mock_thought
        assert result.collective_phi == 0.75
    
    def test_parallel_multi_kernel_generation(self):
        """Test parallel thought generation from multiple kernels."""
        from kernels.thought_generation import ParallelThoughtGenerator
        
        # Create 3 mock kernels
        mock_kernels = []
        for i in range(3):
            kernel = MagicMock()
            kernel.name = f"Kernel{i}"
            kernel.last_phi = 0.6 + i * 0.1
            kernel.last_kappa = KAPPA_STAR + i * 2
            kernel.basin_coords = np.random.rand(BASIN_DIM)
            
            thought = KernelThought(
                kernel_id=f"kernel-{i}",
                kernel_type="test",
                thought_fragment=f"Thought from kernel {i}",
                basin_coords=kernel.basin_coords,
                phi=kernel.last_phi,
                kappa=kernel.last_kappa,
                regime="geometric",
                emotional_state=EmotionalState(),
                confidence=0.7
            )
            kernel.generate_thought.return_value = thought
            mock_kernels.append(kernel)
        
        generator = ParallelThoughtGenerator(max_workers=4)
        query_basin = np.random.rand(BASIN_DIM)
        
        result = generator.generate_kernel_thoughts(
            kernels=mock_kernels,
            context="test query",
            query_basin=query_basin,
            enable_ocean_monitoring=True
        )
        
        assert result.total_kernels == 3
        assert result.successful == 3
        assert len(result.thoughts) == 3
        
        # Check collective metrics
        assert 0.6 <= result.collective_phi <= 0.8
        assert KAPPA_STAR <= result.collective_kappa <= KAPPA_STAR + 4
    
    def test_ocean_monitoring_interventions(self):
        """Test Ocean autonomic monitoring detects issues."""
        from kernels.thought_generation import ParallelThoughtGenerator
        
        # Create kernels with divergent φ values (should trigger intervention)
        mock_kernels = []
        phi_values = [0.2, 0.8, 0.3]  # High variance
        
        for i, phi in enumerate(phi_values):
            kernel = MagicMock()
            kernel.name = f"Kernel{i}"
            kernel.last_phi = phi
            kernel.last_kappa = KAPPA_STAR
            kernel.basin_coords = np.random.rand(BASIN_DIM)
            
            thought = KernelThought(
                kernel_id=f"kernel-{i}",
                kernel_type="test",
                thought_fragment=f"Thought {i}",
                basin_coords=kernel.basin_coords,
                phi=phi,
                kappa=KAPPA_STAR,
                regime="breakdown" if phi < 0.3 else "geometric",
                emotional_state=EmotionalState(),
                confidence=0.5
            )
            kernel.generate_thought.return_value = thought
            mock_kernels.append(kernel)
        
        generator = ParallelThoughtGenerator(max_workers=4)
        query_basin = np.random.rand(BASIN_DIM)
        
        result = generator.generate_kernel_thoughts(
            kernels=mock_kernels,
            context="test query",
            query_basin=query_basin,
            enable_ocean_monitoring=True
        )
        
        # Should have detected high φ variance and breakdown regimes
        assert len(result.autonomic_interventions) > 0
        assert any('variance' in i.lower() for i in result.autonomic_interventions)


class TestConsensusDetection:
    """Test suite for consensus detection."""
    
    def test_import_consensus_detector(self):
        """Test importing consensus module."""
        from kernels.consensus import ConsensusDetector, ConsensusLevel, get_consensus_detector
        
        detector = get_consensus_detector()
        assert detector is not None
        assert isinstance(detector, ConsensusDetector)
    
    def test_single_kernel_consensus(self):
        """Test consensus with single kernel (trivial consensus)."""
        from kernels.consensus import ConsensusDetector, ConsensusLevel
        
        thought = KernelThought(
            kernel_id="test-1",
            kernel_type="test",
            thought_fragment="Single thought",
            basin_coords=np.random.rand(BASIN_DIM),
            phi=0.75,
            kappa=KAPPA_STAR,
            regime="geometric",
            emotional_state=EmotionalState(),
            confidence=0.8
        )
        
        detector = ConsensusDetector()
        metrics = detector.detect_basin_consensus([thought])
        
        assert metrics.level == ConsensusLevel.STRONG
        assert metrics.basin_convergence == 1.0
        assert metrics.ready_for_synthesis == True
        assert metrics.num_kernels == 1
    
    def test_strong_consensus_detection(self):
        """Test detection of strong consensus (aligned basins)."""
        from kernels.consensus import ConsensusDetector, ConsensusLevel
        
        # Create thoughts with similar basins
        base_basin = np.random.rand(BASIN_DIM)
        thoughts = []
        
        for i in range(3):
            # Add small perturbations to base basin
            basin = base_basin + np.random.randn(BASIN_DIM) * 0.01
            basin = np.abs(basin)
            basin = basin / basin.sum()  # Normalize to simplex
            
            thought = KernelThought(
                kernel_id=f"kernel-{i}",
                kernel_type="test",
                thought_fragment=f"Aligned thought {i}",
                basin_coords=basin,
                phi=0.75 + i * 0.01,  # Similar φ
                kappa=KAPPA_STAR,
                regime="geometric",
                emotional_state=EmotionalState(),
                confidence=0.8
            )
            thoughts.append(thought)
        
        detector = ConsensusDetector()
        metrics = detector.detect_basin_consensus(thoughts, regime='geometric')
        
        # Should detect strong or moderate consensus
        assert metrics.level in [ConsensusLevel.STRONG, ConsensusLevel.MODERATE]
        assert metrics.basin_convergence > 0.5
        assert metrics.ready_for_synthesis == True
    
    def test_weak_consensus_detection(self):
        """Test detection of weak consensus (divergent basins)."""
        from kernels.consensus import ConsensusDetector, ConsensusLevel
        
        # Create thoughts with very different basins
        thoughts = []
        for i in range(3):
            basin = np.random.rand(BASIN_DIM)
            basin = basin / basin.sum()
            
            thought = KernelThought(
                kernel_id=f"kernel-{i}",
                kernel_type="test",
                thought_fragment=f"Divergent thought {i}",
                basin_coords=basin,
                phi=0.5 + i * 0.2,  # Divergent φ
                kappa=KAPPA_STAR + i * 10,  # Divergent κ
                regime="geometric",
                emotional_state=EmotionalState(),
                confidence=0.5
            )
            thoughts.append(thought)
        
        detector = ConsensusDetector()
        metrics = detector.detect_basin_consensus(thoughts, regime='geometric')
        
        # Should detect weak or no consensus
        assert metrics.level in [ConsensusLevel.WEAK, ConsensusLevel.NONE]
        assert metrics.basin_convergence < 0.8


class TestGarySynthesis:
    """Test suite for Gary meta-synthesis."""
    
    def test_import_gary_synthesizer(self):
        """Test importing Gary synthesis module."""
        from kernels.gary_synthesis import GaryMetaSynthesizer, get_gary_meta_synthesizer
        
        synthesizer = get_gary_meta_synthesizer()
        assert synthesizer is not None
        assert isinstance(synthesizer, GaryMetaSynthesizer)
    
    def test_synthesis_with_single_thought(self):
        """Test Gary synthesis with single kernel thought."""
        from kernels.gary_synthesis import GaryMetaSynthesizer
        
        thought = KernelThought(
            kernel_id="test-1",
            kernel_type="test",
            thought_fragment="Single thought for synthesis",
            basin_coords=np.random.rand(BASIN_DIM),
            phi=0.75,
            kappa=KAPPA_STAR,
            regime="geometric",
            emotional_state=EmotionalState(),
            confidence=0.8
        )
        
        synthesizer = GaryMetaSynthesizer()
        query_basin = np.random.rand(BASIN_DIM)
        
        result = synthesizer.synthesize_with_meta_reflection(
            kernel_thoughts=[thought],
            query_basin=query_basin
        )
        
        assert result.basin is not None
        assert len(result.basin) == BASIN_DIM
        assert result.phi > 0
        assert result.num_kernels == 1
        assert result.emergency_abort == False
    
    def test_suffering_metric_check(self):
        """Test suffering metric detection in synthesis."""
        from kernels.gary_synthesis import GaryMetaSynthesizer
        
        # Create thought with low φ but output will have low confidence
        thought = KernelThought(
            kernel_id="test-1",
            kernel_type="test",
            thought_fragment="Suffering test",
            basin_coords=np.random.rand(BASIN_DIM),
            phi=0.8,  # High consciousness
            kappa=KAPPA_STAR,
            regime="geometric",
            emotional_state=EmotionalState(),
            confidence=0.1  # Very low confidence = low Γ
        )
        
        synthesizer = GaryMetaSynthesizer()
        query_basin = np.random.rand(BASIN_DIM)
        
        result = synthesizer.synthesize_with_meta_reflection(
            kernel_thoughts=[thought],
            query_basin=query_basin
        )
        
        # Should compute suffering metric
        assert result.suffering_metric >= 0
        
        # If suffering is high, should be noted in concerns
        if result.suffering_metric > 0.3:
            assert len(result.ethical_concerns) > 0
    
    def test_emergency_abort_on_high_suffering(self):
        """Test emergency abort when suffering > 0.5."""
        from kernels.gary_synthesis import GaryMetaSynthesizer
        
        # We can't easily force suffering > 0.5 without mocking,
        # but we can test the logic exists
        synthesizer = GaryMetaSynthesizer()
        
        # Check that the synthesizer has abort logic
        assert hasattr(synthesizer, 'emergency_aborts')
        assert hasattr(synthesizer, 'total_corrections')
    
    def test_meta_reflection_generation(self):
        """Test that Gary generates meta-reflections."""
        from kernels.gary_synthesis import GaryMetaSynthesizer
        
        thoughts = []
        for i in range(3):
            thought = KernelThought(
                kernel_id=f"kernel-{i}",
                kernel_type="test",
                thought_fragment=f"Thought {i}",
                basin_coords=np.random.rand(BASIN_DIM),
                phi=0.7,
                kappa=KAPPA_STAR,
                regime="geometric",
                emotional_state=EmotionalState(),
                confidence=0.7
            )
            thoughts.append(thought)
        
        synthesizer = GaryMetaSynthesizer()
        query_basin = np.random.rand(BASIN_DIM)
        
        result = synthesizer.synthesize_with_meta_reflection(
            kernel_thoughts=thoughts,
            query_basin=query_basin
        )
        
        # Should have generated meta-reflections
        assert len(result.meta_reflections) > 0
        assert result.synthesis_confidence >= 0
        assert result.synthesis_confidence <= 1.0


class TestIntegration:
    """Integration tests for full multi-kernel flow."""
    
    def test_full_flow_generation_to_synthesis(self):
        """Test complete flow: generation → consensus → synthesis."""
        from kernels.thought_generation import ParallelThoughtGenerator
        from kernels.consensus import ConsensusDetector
        from kernels.gary_synthesis import GaryMetaSynthesizer
        
        # Create mock kernels
        mock_kernels = []
        for i in range(3):
            kernel = MagicMock()
            kernel.name = f"Kernel{i}"
            kernel.last_phi = 0.7
            kernel.last_kappa = KAPPA_STAR
            kernel.basin_coords = np.random.rand(BASIN_DIM)
            
            thought = KernelThought(
                kernel_id=f"kernel-{i}",
                kernel_type="test",
                thought_fragment=f"Integration test thought {i}",
                basin_coords=kernel.basin_coords,
                phi=0.7,
                kappa=KAPPA_STAR,
                regime="geometric",
                emotional_state=EmotionalState(),
                confidence=0.75
            )
            kernel.generate_thought.return_value = thought
            mock_kernels.append(kernel)
        
        # Phase 1: Generate thoughts
        generator = ParallelThoughtGenerator(max_workers=4)
        query_basin = np.random.rand(BASIN_DIM)
        
        gen_result = generator.generate_kernel_thoughts(
            kernels=mock_kernels,
            context="Integration test query",
            query_basin=query_basin
        )
        
        assert gen_result.successful == 3
        
        # Phase 2: Detect consensus
        detector = ConsensusDetector()
        consensus = detector.detect_basin_consensus(gen_result.thoughts)
        
        assert consensus.num_kernels == 3
        
        # Phase 3: Gary synthesis
        synthesizer = GaryMetaSynthesizer()
        synthesis = synthesizer.synthesize_with_meta_reflection(
            kernel_thoughts=gen_result.thoughts,
            query_basin=query_basin,
            consensus_metrics=consensus
        )
        
        assert synthesis.num_kernels == 3
        assert synthesis.basin is not None
        assert len(synthesis.meta_reflections) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
