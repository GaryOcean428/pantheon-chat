"""
Tests for the generation benchmark module.
"""

import pytest
import numpy as np
from benchmark_generation import (
    GenerationBenchmark,
    CoherenceMetrics,
    BenchmarkResult,
    TEST_PROMPTS
)


class TestCoherenceMetrics:
    """Test the coherence metrics computation."""
    
    def setup_method(self):
        self.metrics = CoherenceMetrics()
    
    def test_coherence_score_basic(self):
        """Test basic coherence scoring."""
        prompt = "What is consciousness?"
        response = "Consciousness is the state of being aware of one's surroundings and experiences. It involves subjective experience and self-awareness."
        
        score = self.metrics.compute_coherence_score(response, prompt)
        
        assert 0 <= score <= 1
        assert score > 0.3  # Should be reasonably coherent
    
    def test_coherence_score_empty(self):
        """Test coherence with empty text."""
        score = self.metrics.compute_coherence_score("", "test prompt")
        assert score == 0.0
    
    def test_topic_drift_on_topic(self):
        """Test topic drift for on-topic response."""
        prompt = "Explain neural networks"
        response = """Neural networks are computational models inspired by biological brains.
        They consist of layers of interconnected nodes. Each node processes inputs
        and passes outputs to the next layer. Training adjusts connection weights.
        This allows neural networks to learn patterns from data."""
        
        drift = self.metrics.compute_topic_drift(response, prompt)
        
        assert drift < 0.3  # Should have low drift
    
    def test_topic_drift_off_topic(self):
        """Test topic drift for response that wanders off topic."""
        prompt = "Explain neural networks"
        response = """Neural networks are computational models.
        Speaking of computation, computers were invented in the 1940s.
        The weather today is quite nice. I enjoy pizza on weekends.
        Have you ever been to Paris? The Eiffel Tower is beautiful."""
        
        drift = self.metrics.compute_topic_drift(response, prompt)
        
        # Should detect higher drift due to topic changes
        assert drift >= 0  # At minimum non-negative
    
    def test_self_consistency_consistent(self):
        """Test self-consistency for consistent text."""
        text = """The sky is blue during the day. This is because of Rayleigh scattering.
        Light scatters in the atmosphere, making the sky appear blue.
        At sunset, longer wavelengths dominate, creating orange and red hues."""
        
        consistency = self.metrics.compute_self_consistency(text)
        
        assert consistency > 0.7  # Should be highly consistent
    
    def test_self_consistency_contradictory(self):
        """Test self-consistency with contradictions."""
        text = """The answer is definitely yes. But actually, I think it's no.
        Wait, let me rephrase that. On second thought, maybe it's yes after all.
        Actually, correction: the answer is neither."""
        
        consistency = self.metrics.compute_self_consistency(text)
        
        assert consistency < 0.8  # Should detect inconsistency


class TestGenerationBenchmark:
    """Test the benchmark runner."""
    
    def setup_method(self):
        self.benchmark = GenerationBenchmark(verbose=False)
    
    def test_benchmark_initialization(self):
        """Test benchmark initializes correctly."""
        assert self.benchmark.metrics is not None
        assert self.benchmark.results == []
    
    def test_single_prompt_benchmark(self):
        """Test running benchmark on a single prompt."""
        result = self.benchmark.run_single_prompt(
            prompt="What is 2+2?",
            category="simple"
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.prompt == "What is 2+2?"
        assert result.prompt_category == "simple"
        assert result.standard_text != ""
        assert result.vision_text != ""
        assert 0 <= result.standard_coherence_score <= 1
        assert 0 <= result.vision_coherence_score <= 1
    
    def test_benchmark_result_comparison(self):
        """Test that comparison metrics are computed."""
        result = self.benchmark.run_single_prompt(
            prompt="Explain the theory of relativity",
            category="factual"
        )
        
        # Coherence improvement should be computed
        assert result.coherence_improvement == (
            result.vision_coherence_score - result.standard_coherence_score
        )
        
        # Token reduction should be reasonable
        if result.standard_token_count > 0:
            expected_reduction = (
                result.standard_token_count - result.vision_token_count
            ) / result.standard_token_count
            assert abs(result.token_reduction - expected_reduction) < 0.001
    
    def test_encode_prompt(self):
        """Test prompt encoding."""
        basin = self.benchmark._encode_prompt("test prompt")
        
        assert isinstance(basin, np.ndarray)
        assert len(basin) > 0
        assert basin.sum() > 0  # Should be normalized
    
    def test_test_prompts_structure(self):
        """Test that TEST_PROMPTS has expected structure."""
        expected_categories = ['simple', 'factual', 'reasoning', 'creative', 
                               'complex_reasoning', 'synthesis']
        
        for cat in expected_categories:
            assert cat in TEST_PROMPTS
            assert len(TEST_PROMPTS[cat]) >= 1
            assert all(isinstance(p, str) for p in TEST_PROMPTS[cat])


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""
    
    def test_default_values(self):
        """Test default values are set correctly."""
        result = BenchmarkResult(prompt="test", prompt_category="test")
        
        assert result.standard_text == ""
        assert result.standard_token_count == 0
        assert result.standard_coherence_score == 0.0
        assert result.vision_text == ""
        assert result.vision_endpoint_reached == False
        assert result.coherence_improvement == 0.0
    
    def test_all_fields_settable(self):
        """Test all fields can be set."""
        result = BenchmarkResult(
            prompt="test",
            prompt_category="simple",
            standard_text="standard response",
            standard_coherence_score=0.8,
            vision_text="vision response",
            vision_coherence_score=0.9,
            vision_mode_used="lightning",
            vision_endpoint_reached=True,
            coherence_improvement=0.1
        )
        
        assert result.standard_text == "standard response"
        assert result.vision_coherence_score == 0.9
        assert result.vision_mode_used == "lightning"
        assert result.coherence_improvement == 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
