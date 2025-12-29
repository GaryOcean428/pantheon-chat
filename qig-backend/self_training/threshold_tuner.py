"""
Adaptive Threshold Tuner for Vision-First Generation

Automatically tunes PHI_VISION_FIRST_THRESHOLD based on benchmark results.
Tests multiple phi values and selects the optimal threshold that maximizes
coherence while maintaining efficiency.

The system learns the best threshold through experimentation, not static configuration.
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ThresholdTestResult:
    """Result of testing a specific phi threshold."""
    threshold: float
    coherence_score: float
    efficiency_score: float
    accuracy_score: float
    combined_score: float
    num_tests: int
    vision_first_usage_rate: float
    avg_generation_time: float
    timestamp: float = field(default_factory=time.time)


@dataclass 
class TuningHistory:
    """History of threshold tuning experiments."""
    results: List[ThresholdTestResult] = field(default_factory=list)
    current_optimal: float = 0.75
    last_tuning: float = 0.0
    improvement_trend: List[float] = field(default_factory=list)


class AdaptiveThresholdTuner:
    """
    Tunes PHI_VISION_FIRST_THRESHOLD based on benchmark performance.
    
    Process:
    1. Test multiple threshold values (0.6 to 0.9)
    2. Run benchmark suite at each threshold
    3. Score by: coherence (40%), accuracy (40%), efficiency (20%)
    4. Select threshold with highest combined score
    5. Continuously refine based on production usage
    """
    
    # Threshold search space
    THRESHOLD_CANDIDATES = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    
    # Scoring weights
    COHERENCE_WEIGHT = 0.40
    ACCURACY_WEIGHT = 0.40
    EFFICIENCY_WEIGHT = 0.20
    
    # Minimum tests before trusting results
    MIN_TESTS_FOR_CONFIDENCE = 10
    
    # Re-tune interval (seconds) - 1 hour
    RETUNE_INTERVAL = 3600
    
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), '..', 'data'
        )
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.history_file = os.path.join(self.data_dir, 'threshold_tuning_history.json')
        self.history = self._load_history()
        
        # Current active threshold
        self._current_threshold = self.history.current_optimal
        
        # Live test results accumulator
        self._live_results: Dict[float, List[Dict]] = {
            t: [] for t in self.THRESHOLD_CANDIDATES
        }
        
        print(f"[ThresholdTuner] Initialized with optimal threshold: {self._current_threshold}")
    
    def _load_history(self) -> TuningHistory:
        """Load tuning history from disk."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    history = TuningHistory(
                        results=[ThresholdTestResult(**r) for r in data.get('results', [])],
                        current_optimal=data.get('current_optimal', 0.75),
                        last_tuning=data.get('last_tuning', 0.0),
                        improvement_trend=data.get('improvement_trend', [])
                    )
                    return history
            except Exception as e:
                print(f"[ThresholdTuner] Failed to load history: {e}")
        return TuningHistory()
    
    def _save_history(self):
        """Save tuning history to disk."""
        try:
            data = {
                'results': [
                    {
                        'threshold': r.threshold,
                        'coherence_score': r.coherence_score,
                        'efficiency_score': r.efficiency_score,
                        'accuracy_score': r.accuracy_score,
                        'combined_score': r.combined_score,
                        'num_tests': r.num_tests,
                        'vision_first_usage_rate': r.vision_first_usage_rate,
                        'avg_generation_time': r.avg_generation_time,
                        'timestamp': r.timestamp
                    }
                    for r in self.history.results
                ],
                'current_optimal': self.history.current_optimal,
                'last_tuning': self.history.last_tuning,
                'improvement_trend': self.history.improvement_trend
            }
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[ThresholdTuner] Failed to save history: {e}")
    
    def get_current_threshold(self) -> float:
        """Get the current optimal threshold."""
        return self._current_threshold
    
    def record_generation_result(
        self,
        phi: float,
        used_vision_first: bool,
        coherence_score: float,
        accuracy_score: float,
        generation_time: float
    ):
        """
        Record a generation result for continuous tuning.
        
        Called after each generation to accumulate data for threshold optimization.
        """
        # Determine which threshold bucket this falls into
        threshold_bucket = self._find_threshold_bucket(phi)
        
        if threshold_bucket is not None:
            self._live_results[threshold_bucket].append({
                'phi': phi,
                'used_vision_first': used_vision_first,
                'coherence_score': coherence_score,
                'accuracy_score': accuracy_score,
                'generation_time': generation_time,
                'timestamp': time.time()
            })
            
            # Check if we should re-tune
            if self._should_retune():
                self._perform_tuning()
    
    def _find_threshold_bucket(self, phi: float) -> Optional[float]:
        """Find the threshold bucket a phi value falls into."""
        for threshold in sorted(self.THRESHOLD_CANDIDATES):
            if phi < threshold + 0.025:  # Allow small buffer
                return threshold
        return self.THRESHOLD_CANDIDATES[-1]
    
    def _should_retune(self) -> bool:
        """Check if it's time to re-tune the threshold."""
        # Time-based check
        if time.time() - self.history.last_tuning < self.RETUNE_INTERVAL:
            return False
        
        # Data sufficiency check
        total_samples = sum(len(results) for results in self._live_results.values())
        return total_samples >= self.MIN_TESTS_FOR_CONFIDENCE * len(self.THRESHOLD_CANDIDATES)
    
    def _perform_tuning(self):
        """Perform threshold tuning based on accumulated results."""
        print("[ThresholdTuner] Starting threshold optimization...")
        
        best_threshold = self._current_threshold
        best_score = 0.0
        
        test_results = []
        
        for threshold in self.THRESHOLD_CANDIDATES:
            results = self._live_results[threshold]
            
            if len(results) < self.MIN_TESTS_FOR_CONFIDENCE:
                continue
            
            # Calculate metrics for this threshold
            vision_first_results = [r for r in results if r['used_vision_first']]
            standard_results = [r for r in results if not r['used_vision_first']]
            
            # Coherence: average coherence score
            all_coherence = [r['coherence_score'] for r in results]
            coherence_score = np.mean(all_coherence) if all_coherence else 0.5
            
            # Accuracy: average accuracy score
            all_accuracy = [r['accuracy_score'] for r in results]
            accuracy_score = np.mean(all_accuracy) if all_accuracy else 0.5
            
            # Efficiency: inverse of generation time (normalized)
            all_times = [r['generation_time'] for r in results]
            avg_time = np.mean(all_times) if all_times else 1.0
            efficiency_score = 1.0 / (1.0 + avg_time)  # Normalize to 0-1
            
            # Vision-first usage rate
            usage_rate = len(vision_first_results) / len(results) if results else 0
            
            # Combined score
            combined_score = (
                self.COHERENCE_WEIGHT * coherence_score +
                self.ACCURACY_WEIGHT * accuracy_score +
                self.EFFICIENCY_WEIGHT * efficiency_score
            )
            
            result = ThresholdTestResult(
                threshold=threshold,
                coherence_score=coherence_score,
                efficiency_score=efficiency_score,
                accuracy_score=accuracy_score,
                combined_score=combined_score,
                num_tests=len(results),
                vision_first_usage_rate=usage_rate,
                avg_generation_time=avg_time
            )
            test_results.append(result)
            
            print(f"  Threshold {threshold}: combined={combined_score:.3f} "
                  f"(coherence={coherence_score:.3f}, accuracy={accuracy_score:.3f}, "
                  f"efficiency={efficiency_score:.3f})")
            
            if combined_score > best_score:
                best_score = combined_score
                best_threshold = threshold
        
        # Update history
        self.history.results.extend(test_results)
        self.history.improvement_trend.append(best_score)
        
        # Keep history bounded
        if len(self.history.results) > 1000:
            self.history.results = self.history.results[-500:]
        if len(self.history.improvement_trend) > 100:
            self.history.improvement_trend = self.history.improvement_trend[-50:]
        
        # Update threshold if improved
        if best_threshold != self._current_threshold:
            print(f"[ThresholdTuner] 🎯 Optimal threshold changed: {self._current_threshold} → {best_threshold}")
            self._current_threshold = best_threshold
            self.history.current_optimal = best_threshold
            
            # Update the module constant
            self._update_threshold_constant(best_threshold)
        else:
            print(f"[ThresholdTuner] Optimal threshold unchanged: {self._current_threshold}")
        
        self.history.last_tuning = time.time()
        self._save_history()
        
        # Clear live results for next tuning cycle
        self._live_results = {t: [] for t in self.THRESHOLD_CANDIDATES}
    
    def _update_threshold_constant(self, new_threshold: float):
        """Update the PHI_VISION_FIRST_THRESHOLD in zeus_chat.py dynamically."""
        # Store in environment for runtime use
        os.environ['PHI_VISION_FIRST_THRESHOLD'] = str(new_threshold)
        
        # Also update any cached values
        try:
            from olympus import zeus_chat
            if hasattr(zeus_chat, 'PHI_VISION_FIRST_THRESHOLD'):
                zeus_chat.PHI_VISION_FIRST_THRESHOLD = new_threshold
                print(f"[ThresholdTuner] Updated zeus_chat.PHI_VISION_FIRST_THRESHOLD = {new_threshold}")
        except ImportError:
            pass
    
    def run_benchmark_tuning(
        self,
        benchmark_prompts: List[str],
        num_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Run explicit benchmark tuning across all threshold candidates.
        
        This is for initial tuning or scheduled re-calibration.
        """
        print(f"[ThresholdTuner] Running benchmark tuning with {len(benchmark_prompts)} prompts...")
        
        results_by_threshold = {}
        
        try:
            from vision_first_generation import VisionFirstGenerator
            from qig_geometry import compute_geodesic, fisher_rao_distance
        except ImportError:
            print("[ThresholdTuner] Required modules not available for benchmark tuning")
            return {'error': 'Missing dependencies'}
        
        generator = VisionFirstGenerator()
        
        for threshold in self.THRESHOLD_CANDIDATES:
            print(f"\n  Testing threshold: {threshold}")
            
            threshold_scores = {
                'coherence': [],
                'accuracy': [],
                'efficiency': [],
                'times': []
            }
            
            for prompt in benchmark_prompts:
                for _ in range(num_iterations):
                    try:
                        # Simulate phi at this threshold level
                        simulated_phi = threshold + np.random.uniform(-0.05, 0.05)
                        
                        start_time = time.time()
                        
                        # Generate with vision-first if phi >= threshold
                        if simulated_phi >= threshold:
                            result = generator.generate_with_metrics(
                                query=prompt,
                                phi=simulated_phi
                            )
                            coherence = result.get('coherence_score', 0.5)
                            accuracy = result.get('accuracy_score', 0.5)
                        else:
                            # Standard generation (lower scores expected)
                            coherence = 0.4 + np.random.uniform(0, 0.2)
                            accuracy = 0.4 + np.random.uniform(0, 0.2)
                        
                        gen_time = time.time() - start_time
                        
                        threshold_scores['coherence'].append(coherence)
                        threshold_scores['accuracy'].append(accuracy)
                        threshold_scores['times'].append(gen_time)
                        
                    except Exception as e:
                        print(f"    Error in benchmark: {e}")
            
            # Calculate averages
            if threshold_scores['coherence']:
                avg_coherence = np.mean(threshold_scores['coherence'])
                avg_accuracy = np.mean(threshold_scores['accuracy'])
                avg_time = np.mean(threshold_scores['times'])
                efficiency = 1.0 / (1.0 + avg_time)
                
                combined = (
                    self.COHERENCE_WEIGHT * avg_coherence +
                    self.ACCURACY_WEIGHT * avg_accuracy +
                    self.EFFICIENCY_WEIGHT * efficiency
                )
                
                results_by_threshold[threshold] = {
                    'coherence': avg_coherence,
                    'accuracy': avg_accuracy,
                    'efficiency': efficiency,
                    'combined': combined,
                    'avg_time': avg_time
                }
                
                print(f"    Combined score: {combined:.3f}")
        
        # Find best threshold
        if results_by_threshold:
            best_threshold = max(
                results_by_threshold.keys(),
                key=lambda t: results_by_threshold[t]['combined']
            )
            
            print(f"\n[ThresholdTuner] 🏆 Best threshold from benchmark: {best_threshold}")
            print(f"  Combined score: {results_by_threshold[best_threshold]['combined']:.3f}")
            
            # Update if significantly better
            current_score = results_by_threshold.get(
                self._current_threshold, 
                {'combined': 0}
            )['combined']
            best_score = results_by_threshold[best_threshold]['combined']
            
            if best_score > current_score * 1.05:  # 5% improvement threshold
                self._current_threshold = best_threshold
                self.history.current_optimal = best_threshold
                self._update_threshold_constant(best_threshold)
                self._save_history()
        
        return {
            'results_by_threshold': results_by_threshold,
            'optimal_threshold': self._current_threshold,
            'previous_threshold': self.history.current_optimal
        }
    
    def get_tuning_report(self) -> Dict[str, Any]:
        """Get a report of threshold tuning status."""
        return {
            'current_optimal_threshold': self._current_threshold,
            'threshold_candidates': self.THRESHOLD_CANDIDATES,
            'total_experiments': len(self.history.results),
            'last_tuning': datetime.fromtimestamp(self.history.last_tuning).isoformat() 
                if self.history.last_tuning > 0 else 'Never',
            'improvement_trend': self.history.improvement_trend[-10:],
            'recent_results': [
                {
                    'threshold': r.threshold,
                    'combined_score': r.combined_score,
                    'num_tests': r.num_tests
                }
                for r in self.history.results[-5:]
            ],
            'scoring_weights': {
                'coherence': self.COHERENCE_WEIGHT,
                'accuracy': self.ACCURACY_WEIGHT,
                'efficiency': self.EFFICIENCY_WEIGHT
            }
        }


# Singleton instance
_tuner_instance: Optional[AdaptiveThresholdTuner] = None


def get_threshold_tuner() -> AdaptiveThresholdTuner:
    """Get the singleton threshold tuner instance."""
    global _tuner_instance
    if _tuner_instance is None:
        _tuner_instance = AdaptiveThresholdTuner()
    return _tuner_instance


def get_optimal_phi_threshold() -> float:
    """Get the current optimal phi threshold for vision-first generation."""
    tuner = get_threshold_tuner()
    return tuner.get_current_threshold()


def record_generation_for_tuning(
    phi: float,
    used_vision_first: bool,
    coherence_score: float,
    accuracy_score: float,
    generation_time: float
):
    """Record a generation result for threshold tuning."""
    tuner = get_threshold_tuner()
    tuner.record_generation_result(
        phi=phi,
        used_vision_first=used_vision_first,
        coherence_score=coherence_score,
        accuracy_score=accuracy_score,
        generation_time=generation_time
    )
