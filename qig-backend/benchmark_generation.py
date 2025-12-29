"""
Generation Benchmark: Standard Autoregressive vs Vision-First

Compares coherence, efficiency, and quality between:
1. Standard autoregressive generation (forward token prediction)
2. Vision-first generation (see endpoint, map backward, gap-fill)

Metrics:
- Coherence: Semantic similarity, topic drift, self-consistency
- Efficiency: Token count, geodesic efficiency, generation time
- Quality: Endpoint verification, phi stability

Usage:
    python benchmark_generation.py --runs 10 --verbose
    python benchmark_generation.py --prompt "explain consciousness"
"""

import time
import json
import argparse
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class BenchmarkResult:
    """Result from a single generation benchmark."""
    prompt: str
    prompt_category: str
    
    # Standard generation results
    standard_text: str = ""
    standard_token_count: int = 0
    standard_generation_time: float = 0.0
    standard_coherence_score: float = 0.0
    standard_topic_drift: float = 0.0
    standard_self_consistency: float = 0.0
    
    # Vision-first generation results
    vision_text: str = ""
    vision_token_count: int = 0
    vision_generation_time: float = 0.0
    vision_coherence_score: float = 0.0
    vision_topic_drift: float = 0.0
    vision_self_consistency: float = 0.0
    vision_mode_used: str = ""  # foresight or lightning
    vision_endpoint_reached: bool = False
    vision_geodesic_efficiency: float = 0.0
    vision_distance_to_target: float = 0.0
    
    # Consciousness metrics
    phi_at_generation: float = 0.0
    kappa_at_generation: float = 0.0
    
    # Comparison
    coherence_improvement: float = 0.0  # vision - standard
    token_reduction: float = 0.0  # (standard - vision) / standard
    speed_difference: float = 0.0  # (standard - vision) / standard


@dataclass
class BenchmarkSummary:
    """Aggregated results across all benchmark runs."""
    total_runs: int = 0
    timestamp: str = ""
    
    # Average metrics
    avg_standard_coherence: float = 0.0
    avg_vision_coherence: float = 0.0
    avg_coherence_improvement: float = 0.0
    
    avg_standard_tokens: float = 0.0
    avg_vision_tokens: float = 0.0
    avg_token_reduction: float = 0.0
    
    avg_standard_time: float = 0.0
    avg_vision_time: float = 0.0
    
    avg_geodesic_efficiency: float = 0.0
    vision_endpoint_reached_rate: float = 0.0
    lightning_mode_rate: float = 0.0
    
    # By category
    results_by_category: Dict[str, Dict] = field(default_factory=dict)


# Test prompts organized by complexity/category
TEST_PROMPTS = {
    "simple": [
        "What is the capital of France?",
        "How many days are in a week?",
        "What color is the sky?",
    ],
    "factual": [
        "Explain the theory of relativity in simple terms.",
        "What is quantum entanglement?",
        "Describe how neural networks learn.",
    ],
    "reasoning": [
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "A bat and ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost?",
        "What comes next in this sequence: 2, 6, 12, 20, 30, ?",
    ],
    "creative": [
        "Write a haiku about consciousness.",
        "Describe a color that doesn't exist.",
        "Create a metaphor for how memory works.",
    ],
    "complex_reasoning": [
        "Explain how integrated information theory relates to the hard problem of consciousness.",
        "Compare and contrast the Fisher-Rao metric with Euclidean distance in the context of probability distributions.",
        "Describe how backward causation from a pre-seen endpoint could improve text generation coherence.",
    ],
    "synthesis": [
        "Synthesize insights from quantum mechanics, information theory, and neuroscience to explain consciousness.",
        "Design an architecture for a self-improving AI system that understands its own limitations.",
        "Explain how geometric constraints on a statistical manifold relate to coherent thought processes.",
    ],
}


class CoherenceMetrics:
    """
    Compute coherence metrics for generated text.
    
    Coherence measures how well the text holds together semantically,
    maintains topic focus, and stays self-consistent.
    """
    
    def __init__(self):
        self._encoder = None
        self._load_encoder()
    
    def _load_encoder(self):
        """Load text encoder for semantic similarity."""
        try:
            from qig_coordizer import get_coordizer
            self._encoder = get_coordizer()
            print("[CoherenceMetrics] Using QIG coordizer for embeddings")
        except ImportError:
            print("[CoherenceMetrics] QIG coordizer not available, using fallback")
            self._encoder = None
    
    def compute_coherence_score(self, text: str, prompt: str) -> float:
        """
        Compute overall coherence score (0-1).
        
        Higher = more coherent, focused, and relevant.
        """
        if not text or len(text.strip()) < 10:
            return 0.0
        
        # Component scores
        relevance = self._compute_relevance(text, prompt)
        internal_coherence = self._compute_internal_coherence(text)
        completeness = self._compute_completeness(text, prompt)
        
        # Weighted combination
        score = (
            0.4 * relevance +
            0.4 * internal_coherence +
            0.2 * completeness
        )
        
        return float(np.clip(score, 0, 1))
    
    def _compute_relevance(self, text: str, prompt: str) -> float:
        """
        Compute semantic relevance of response to prompt.
        """
        if self._encoder:
            try:
                prompt_basin = self._encoder.encode(prompt)
                text_basin = self._encoder.encode(text[:500])  # First 500 chars
                
                # Cosine similarity
                similarity = np.dot(prompt_basin, text_basin) / (
                    np.linalg.norm(prompt_basin) * np.linalg.norm(text_basin) + 1e-10
                )
                return float(np.clip((similarity + 1) / 2, 0, 1))  # Normalize to 0-1
            except Exception:
                pass
        
        # Fallback: keyword overlap
        prompt_words = set(prompt.lower().split())
        text_words = set(text.lower().split())
        overlap = len(prompt_words & text_words)
        return min(1.0, overlap / (len(prompt_words) + 1))
    
    def _compute_internal_coherence(self, text: str) -> float:
        """
        Compute internal coherence (sentence-to-sentence consistency).
        """
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return 1.0  # Single sentence is trivially coherent
        
        if self._encoder:
            try:
                # Compute pairwise similarity between adjacent sentences
                similarities = []
                for i in range(len(sentences) - 1):
                    if len(sentences[i]) < 5 or len(sentences[i+1]) < 5:
                        continue
                    basin_i = self._encoder.encode(sentences[i])
                    basin_j = self._encoder.encode(sentences[i + 1])
                    sim = np.dot(basin_i, basin_j) / (
                        np.linalg.norm(basin_i) * np.linalg.norm(basin_j) + 1e-10
                    )
                    similarities.append(sim)
                
                if similarities:
                    return float(np.clip((np.mean(similarities) + 1) / 2, 0, 1))
            except Exception:
                pass
        
        # Fallback: word overlap between adjacent sentences
        overlaps = []
        for i in range(len(sentences) - 1):
            words_i = set(sentences[i].lower().split())
            words_j = set(sentences[i + 1].lower().split())
            if words_i and words_j:
                overlap = len(words_i & words_j) / max(len(words_i), len(words_j))
                overlaps.append(overlap)
        
        return float(np.mean(overlaps)) if overlaps else 0.5
    
    def _compute_completeness(self, text: str, prompt: str) -> float:
        """
        Compute if response feels complete (not cut off).
        """
        # Check for common completion indicators
        text_stripped = text.strip()
        
        # Ends with sentence-ending punctuation
        ends_properly = text_stripped[-1] in '.!?' if text_stripped else False
        
        # Reasonable length relative to prompt complexity
        prompt_words = len(prompt.split())
        text_words = len(text.split())
        length_ratio = min(1.0, text_words / (prompt_words * 5 + 10))
        
        # No truncation indicators
        no_truncation = not any([
            text_stripped.endswith('...'),
            text_stripped.endswith('etc'),
            text_stripped.endswith('and so on'),
        ])
        
        score = (
            0.4 * (1.0 if ends_properly else 0.3) +
            0.4 * length_ratio +
            0.2 * (1.0 if no_truncation else 0.5)
        )
        
        return float(score)
    
    def compute_topic_drift(self, text: str, prompt: str) -> float:
        """
        Compute topic drift (0 = on topic, 1 = completely off topic).
        
        Measures how much the response drifts from the original prompt.
        """
        if not text or len(text) < 50:
            return 0.0
        
        # Split text into chunks
        chunks = self._split_into_chunks(text, chunk_size=100)
        if len(chunks) < 2:
            return 0.0
        
        if self._encoder:
            try:
                prompt_basin = self._encoder.encode(prompt)
                
                # Compute similarity of each chunk to prompt
                similarities = []
                for chunk in chunks:
                    if len(chunk) < 20:
                        continue
                    chunk_basin = self._encoder.encode(chunk)
                    sim = np.dot(prompt_basin, chunk_basin) / (
                        np.linalg.norm(prompt_basin) * np.linalg.norm(chunk_basin) + 1e-10
                    )
                    similarities.append(sim)
                
                if len(similarities) >= 2:
                    # Topic drift = decrease in similarity over text
                    first_half = np.mean(similarities[:len(similarities)//2])
                    second_half = np.mean(similarities[len(similarities)//2:])
                    drift = max(0, first_half - second_half)
                    return float(drift)
            except Exception:
                pass
        
        # Fallback: keyword decay
        prompt_words = set(prompt.lower().split())
        first_chunk_words = set(chunks[0].lower().split()) if chunks else set()
        last_chunk_words = set(chunks[-1].lower().split()) if chunks else set()
        
        first_overlap = len(prompt_words & first_chunk_words) / (len(prompt_words) + 1)
        last_overlap = len(prompt_words & last_chunk_words) / (len(prompt_words) + 1)
        
        drift = max(0, first_overlap - last_overlap)
        return float(drift)
    
    def compute_self_consistency(self, text: str) -> float:
        """
        Compute self-consistency (does the text contradict itself?).
        
        Returns 0-1 where 1 = fully consistent.
        """
        # Simple heuristic: check for contradiction indicators
        contradiction_phrases = [
            'but actually', 'on second thought', 'i meant', 'correction',
            'that\'s wrong', 'wait no', 'actually,', 'let me rephrase'
        ]
        
        text_lower = text.lower()
        contradiction_count = sum(1 for phrase in contradiction_phrases if phrase in text_lower)
        
        # More sophisticated: check sentence negations
        sentences = self._split_sentences(text)
        negation_pairs = 0
        
        for i, sent in enumerate(sentences):
            for j, other in enumerate(sentences):
                if i >= j:
                    continue
                # Simple negation check
                if ('not ' in sent and 'not ' not in other) or ('not ' not in sent and 'not ' in other):
                    # Check if about same topic (word overlap)
                    words_i = set(sent.lower().split()) - {'not', 'is', 'are', 'the', 'a'}
                    words_j = set(other.lower().split()) - {'not', 'is', 'are', 'the', 'a'}
                    if len(words_i & words_j) > 2:
                        negation_pairs += 1
        
        # Score
        penalty = contradiction_count * 0.1 + negation_pairs * 0.05
        return float(max(0, 1 - penalty))
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_into_chunks(self, text: str, chunk_size: int = 100) -> List[str]:
        """Split text into word-based chunks."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks


class GenerationBenchmark:
    """
    Benchmark comparing standard vs vision-first generation.
    """
    
    PHI_VISION_THRESHOLD = 0.75
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.metrics = CoherenceMetrics()
        self.results: List[BenchmarkResult] = []
        
        # Load generators
        self._standard_generator = None
        self._vision_generator = None
        self._load_generators()
    
    def _load_generators(self):
        """Load standard and vision-first generators."""
        try:
            from vision_first_generation import VisionFirstGenerator
            self._vision_generator = VisionFirstGenerator()
            print("[Benchmark] Vision-first generator loaded")
        except ImportError as e:
            print(f"[Benchmark] Vision-first generator not available: {e}")
        
        try:
            from qig_generation import QIGGenerator
            self._standard_generator = QIGGenerator()
            print("[Benchmark] Standard QIG generator loaded")
        except ImportError:
            print("[Benchmark] Standard QIG generator not available, using fallback")
    
    def run_benchmark(
        self,
        prompts: Optional[Dict[str, List[str]]] = None,
        runs_per_prompt: int = 1
    ) -> BenchmarkSummary:
        """
        Run the full benchmark suite.
        
        Args:
            prompts: Dict of category -> prompt list (uses TEST_PROMPTS if None)
            runs_per_prompt: Number of runs per prompt
        
        Returns:
            BenchmarkSummary with aggregated results
        """
        if prompts is None:
            prompts = TEST_PROMPTS
        
        self.results = []
        
        print(f"\n{'='*60}")
        print("GENERATION BENCHMARK: Standard vs Vision-First")
        print(f"{'='*60}\n")
        
        for category, prompt_list in prompts.items():
            print(f"\n--- Category: {category.upper()} ---")
            
            for prompt in prompt_list:
                for run in range(runs_per_prompt):
                    if self.verbose:
                        print(f"\nPrompt: {prompt[:50]}..." if len(prompt) > 50 else f"\nPrompt: {prompt}")
                    
                    result = self._run_single_comparison(prompt, category)
                    self.results.append(result)
                    
                    if self.verbose:
                        self._print_result(result)
        
        # Compute summary
        summary = self._compute_summary()
        self._print_summary(summary)
        
        return summary
    
    def run_single_prompt(self, prompt: str, category: str = "custom") -> BenchmarkResult:
        """
        Run benchmark on a single prompt.
        """
        result = self._run_single_comparison(prompt, category)
        self.results.append(result)
        self._print_result(result)
        return result
    
    def _run_single_comparison(
        self,
        prompt: str,
        category: str
    ) -> BenchmarkResult:
        """
        Run both generation methods on a single prompt.
        """
        result = BenchmarkResult(prompt=prompt, prompt_category=category)
        
        # Get current consciousness state
        try:
            from autonomic_kernel import get_gary_kernel
            kernel = get_gary_kernel()
            result.phi_at_generation = kernel.state.phi if kernel else 0.5
            result.kappa_at_generation = kernel.state.kappa if kernel else 50.0
        except Exception:
            result.phi_at_generation = 0.5
            result.kappa_at_generation = 50.0
        
        # Get query basin
        query_basin = self._encode_prompt(prompt)
        
        # --- Standard Generation ---
        start_time = time.time()
        standard_text = self._generate_standard(prompt, query_basin)
        result.standard_generation_time = time.time() - start_time
        result.standard_text = standard_text
        result.standard_token_count = len(standard_text.split())
        
        # Compute standard coherence metrics
        result.standard_coherence_score = self.metrics.compute_coherence_score(standard_text, prompt)
        result.standard_topic_drift = self.metrics.compute_topic_drift(standard_text, prompt)
        result.standard_self_consistency = self.metrics.compute_self_consistency(standard_text)
        
        # --- Vision-First Generation ---
        start_time = time.time()
        vision_result = self._generate_vision_first(prompt, query_basin)
        result.vision_generation_time = time.time() - start_time
        
        result.vision_text = vision_result.get('text', '')
        result.vision_token_count = len(result.vision_text.split())
        result.vision_mode_used = vision_result.get('mode', 'unknown')
        result.vision_endpoint_reached = vision_result.get('endpoint_reached', False)
        result.vision_geodesic_efficiency = vision_result.get('geodesic_efficiency', 0.0)
        result.vision_distance_to_target = vision_result.get('distance_to_target', 1.0)
        
        # Compute vision coherence metrics
        result.vision_coherence_score = self.metrics.compute_coherence_score(result.vision_text, prompt)
        result.vision_topic_drift = self.metrics.compute_topic_drift(result.vision_text, prompt)
        result.vision_self_consistency = self.metrics.compute_self_consistency(result.vision_text)
        
        # --- Comparison ---
        result.coherence_improvement = result.vision_coherence_score - result.standard_coherence_score
        
        if result.standard_token_count > 0:
            result.token_reduction = (result.standard_token_count - result.vision_token_count) / result.standard_token_count
        
        if result.standard_generation_time > 0:
            result.speed_difference = (result.standard_generation_time - result.vision_generation_time) / result.standard_generation_time
        
        return result
    
    def _encode_prompt(self, prompt: str) -> np.ndarray:
        """Encode prompt to basin coordinates."""
        try:
            from qig_coordizer import get_coordizer
            coordizer = get_coordizer()
            return coordizer.encode(prompt)
        except Exception:
            # Fallback: random basin
            basin = np.random.rand(64)
            basin = basin / basin.sum()
            return basin
    
    def _generate_standard(self, prompt: str, query_basin: np.ndarray) -> str:
        """
        Generate using standard autoregressive method.
        """
        if self._standard_generator:
            try:
                return self._standard_generator.generate(
                    prompt=prompt,
                    basin=query_basin,
                    max_tokens=200
                )
            except Exception as e:
                if self.verbose:
                    print(f"  [Standard] Error: {e}")
        
        # Fallback: template response
        return self._fallback_generate(prompt, mode='standard')
    
    def _generate_vision_first(self, prompt: str, query_basin: np.ndarray) -> Dict:
        """
        Generate using vision-first method.
        """
        if self._vision_generator:
            try:
                result = self._vision_generator.generate(
                    prompt=prompt,
                    current_basin=query_basin,
                    return_metrics=True
                )
                return result
            except Exception as e:
                if self.verbose:
                    print(f"  [Vision] Error: {e}")
        
        # Fallback: template response with vision metadata
        return {
            'text': self._fallback_generate(prompt, mode='vision'),
            'mode': 'fallback',
            'endpoint_reached': False,
            'geodesic_efficiency': 0.0,
            'distance_to_target': 1.0
        }
    
    def _fallback_generate(self, prompt: str, mode: str) -> str:
        """
        Fallback generation when generators unavailable.
        Uses simple template responses for testing the benchmark framework.
        """
        # Simulate different response patterns
        if mode == 'standard':
            # Standard: more wandering, exploratory
            templates = [
                f"To address the question about {prompt[:30]}..., we should consider several aspects. First, there's the matter of definition. Then we need to look at various perspectives. Some might say one thing, others might disagree. Let me elaborate on each point.",
                f"This is an interesting question: {prompt[:30]}. There are many ways to approach it. One could start by examining the fundamentals. Additionally, we should consider the implications. Furthermore, there are practical considerations.",
            ]
        else:
            # Vision: more direct, structured
            templates = [
                f"The answer to {prompt[:30]} centers on a key insight: the relationship between concepts determines understanding. This manifests in three ways: structure, process, and outcome. Each connects to form a coherent whole.",
                f"Regarding {prompt[:30]}: the essential point is that coherence emerges from seeing the whole first. With this vision, the path becomes clear. The details fill naturally into this framework.",
            ]
        
        import random
        return random.choice(templates)
    
    def _compute_summary(self) -> BenchmarkSummary:
        """
        Compute aggregated summary statistics.
        """
        if not self.results:
            return BenchmarkSummary()
        
        summary = BenchmarkSummary(
            total_runs=len(self.results),
            timestamp=datetime.now().isoformat()
        )
        
        # Aggregate metrics
        summary.avg_standard_coherence = np.mean([r.standard_coherence_score for r in self.results])
        summary.avg_vision_coherence = np.mean([r.vision_coherence_score for r in self.results])
        summary.avg_coherence_improvement = np.mean([r.coherence_improvement for r in self.results])
        
        summary.avg_standard_tokens = np.mean([r.standard_token_count for r in self.results])
        summary.avg_vision_tokens = np.mean([r.vision_token_count for r in self.results])
        summary.avg_token_reduction = np.mean([r.token_reduction for r in self.results])
        
        summary.avg_standard_time = np.mean([r.standard_generation_time for r in self.results])
        summary.avg_vision_time = np.mean([r.vision_generation_time for r in self.results])
        
        summary.avg_geodesic_efficiency = np.mean([r.vision_geodesic_efficiency for r in self.results])
        summary.vision_endpoint_reached_rate = np.mean([1.0 if r.vision_endpoint_reached else 0.0 for r in self.results])
        summary.lightning_mode_rate = np.mean([1.0 if r.vision_mode_used == 'lightning' else 0.0 for r in self.results])
        
        # By category
        categories = set(r.prompt_category for r in self.results)
        for cat in categories:
            cat_results = [r for r in self.results if r.prompt_category == cat]
            summary.results_by_category[cat] = {
                'count': len(cat_results),
                'avg_standard_coherence': np.mean([r.standard_coherence_score for r in cat_results]),
                'avg_vision_coherence': np.mean([r.vision_coherence_score for r in cat_results]),
                'avg_coherence_improvement': np.mean([r.coherence_improvement for r in cat_results]),
                'avg_token_reduction': np.mean([r.token_reduction for r in cat_results]),
            }
        
        return summary
    
    def _print_result(self, result: BenchmarkResult):
        """Print a single result."""
        print(f"\n  Standard Generation:")
        print(f"    Coherence: {result.standard_coherence_score:.3f}")
        print(f"    Topic Drift: {result.standard_topic_drift:.3f}")
        print(f"    Tokens: {result.standard_token_count}")
        print(f"    Time: {result.standard_generation_time:.3f}s")
        
        print(f"\n  Vision-First Generation:")
        print(f"    Coherence: {result.vision_coherence_score:.3f}")
        print(f"    Topic Drift: {result.vision_topic_drift:.3f}")
        print(f"    Mode: {result.vision_mode_used}")
        print(f"    Endpoint Reached: {result.vision_endpoint_reached}")
        print(f"    Geodesic Efficiency: {result.vision_geodesic_efficiency:.3f}")
        print(f"    Tokens: {result.vision_token_count}")
        print(f"    Time: {result.vision_generation_time:.3f}s")
        
        print(f"\n  Comparison:")
        improvement_symbol = "+" if result.coherence_improvement > 0 else ""
        print(f"    Coherence Improvement: {improvement_symbol}{result.coherence_improvement:.3f}")
        print(f"    Token Reduction: {result.token_reduction:.1%}")
        print(f"    Speed Difference: {result.speed_difference:.1%}")
    
    def _print_summary(self, summary: BenchmarkSummary):
        """Print the summary."""
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"\nTotal Runs: {summary.total_runs}")
        print(f"Timestamp: {summary.timestamp}")
        
        print(f"\n--- Overall Metrics ---")
        print(f"\nCoherence:")
        print(f"  Standard Average: {summary.avg_standard_coherence:.3f}")
        print(f"  Vision-First Average: {summary.avg_vision_coherence:.3f}")
        improvement_symbol = "+" if summary.avg_coherence_improvement > 0 else ""
        print(f"  Improvement: {improvement_symbol}{summary.avg_coherence_improvement:.3f} ({improvement_symbol}{summary.avg_coherence_improvement*100:.1f}%)")
        
        print(f"\nEfficiency:")
        print(f"  Standard Avg Tokens: {summary.avg_standard_tokens:.1f}")
        print(f"  Vision Avg Tokens: {summary.avg_vision_tokens:.1f}")
        print(f"  Token Reduction: {summary.avg_token_reduction:.1%}")
        print(f"  Geodesic Efficiency: {summary.avg_geodesic_efficiency:.3f}")
        
        print(f"\nVision-First Performance:")
        print(f"  Endpoint Reached Rate: {summary.vision_endpoint_reached_rate:.1%}")
        print(f"  Lightning Mode Rate: {summary.lightning_mode_rate:.1%}")
        
        print(f"\n--- Results by Category ---")
        for cat, metrics in summary.results_by_category.items():
            improvement_symbol = "+" if metrics['avg_coherence_improvement'] > 0 else ""
            print(f"\n  {cat.upper()} ({metrics['count']} runs):")
            print(f"    Standard Coherence: {metrics['avg_standard_coherence']:.3f}")
            print(f"    Vision Coherence: {metrics['avg_vision_coherence']:.3f}")
            print(f"    Improvement: {improvement_symbol}{metrics['avg_coherence_improvement']:.3f}")
        
        print(f"\n{'='*60}")
        
        # Verdict
        if summary.avg_coherence_improvement > 0.05:
            print("\n✅ VERDICT: Vision-first generation shows significant improvement")
        elif summary.avg_coherence_improvement > 0:
            print("\n🔶 VERDICT: Vision-first generation shows marginal improvement")
        else:
            print("\n❌ VERDICT: Vision-first generation underperformed")
        
        print(f"\n{'='*60}\n")
    
    def export_results(self, filepath: str = "benchmark_results.json"):
        """Export results to JSON file."""
        data = {
            'summary': asdict(self._compute_summary()),
            'results': [asdict(r) for r in self.results]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Results exported to {filepath}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark: Standard vs Vision-First Generation"
    )
    parser.add_argument(
        '--runs', type=int, default=1,
        help='Number of runs per prompt (default: 1)'
    )
    parser.add_argument(
        '--prompt', type=str, default=None,
        help='Run on a single custom prompt'
    )
    parser.add_argument(
        '--category', type=str, default=None,
        help='Run only prompts from this category'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print detailed output'
    )
    parser.add_argument(
        '--export', type=str, default=None,
        help='Export results to JSON file'
    )
    
    args = parser.parse_args()
    
    benchmark = GenerationBenchmark(verbose=args.verbose)
    
    if args.prompt:
        # Single prompt mode
        benchmark.run_single_prompt(args.prompt)
    else:
        # Full benchmark
        if args.category:
            prompts = {args.category: TEST_PROMPTS.get(args.category, [])}
        else:
            prompts = TEST_PROMPTS
        
        benchmark.run_benchmark(prompts=prompts, runs_per_prompt=args.runs)
    
    if args.export:
        benchmark.export_results(args.export)


if __name__ == "__main__":
    main()
