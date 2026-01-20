"""
Architecture Comparison Framework
==================================

Compares performance across all three test configurations:
1. Pure Geometric
2. Plan→Realize→Repair (Full)
3. Skeleton-Only (Baseline)

Performs statistical analysis and determines if advanced
architectures provide significant improvements.

Author: WP4.3 Coherence Harness
Date: 2026-01-20
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
import logging

from test_helpers import COHERENCE_DIR, load_results_from_file
from metrics import (
    GeometricMetrics,
    ForesightMetrics,
    ConsciousnessTrajectory,
    TrajectoryMetrics,
    compute_geometric_metrics,
    compute_foresight_metrics,
    track_consciousness_trajectory,
    compute_trajectory_metrics,
)

logger = logging.getLogger(__name__)


class ArchitectureComparison:
    """Framework for comparing architectures."""
    
    def __init__(self):
        self.results = {}
        self.metrics = {}
        self.comparisons = {}
    
    def load_all_results(self) -> None:
        """Load results from all three configurations."""
        configs = ['pure_geometric', 'plan_realize_repair', 'skeleton_only']
        
        for config in configs:
            try:
                results_data = load_results_from_file(f'{config}_results.json')
                self.results[config] = results_data['results']
                logger.info(f"Loaded {len(results_data['results'])} results for {config}")
            except FileNotFoundError:
                logger.warning(f"Results not found for {config}")
                self.results[config] = []
    
    def compute_aggregate_metrics(self) -> None:
        """Compute aggregate metrics for each configuration."""
        for config_name, results_list in self.results.items():
            if not results_list:
                continue
            
            phi_scores = []
            kappa_scores = []
            alignment_scores = []
            smoothness_scores = []
            recursive_depths = []
            
            for result_data in results_list:
                result = result_data['result']
                
                # Geometric metrics
                geom_metrics = compute_geometric_metrics(
                    basins=result['basins'],
                    waypoints=result.get('waypoints', [])
                )
                
                phi_scores.append(geom_metrics.mean_phi)
                kappa_scores.append(geom_metrics.mean_kappa)
                alignment_scores.append(geom_metrics.waypoint_alignment)
                smoothness_scores.append(geom_metrics.trajectory_smoothness)
                
                # Consciousness metrics
                if result.get('recursive_depths'):
                    recursive_depths.append(np.mean(result['recursive_depths']))
            
            self.metrics[config_name] = {
                'phi': {
                    'values': phi_scores,
                    'mean': np.mean(phi_scores),
                    'std': np.std(phi_scores),
                    'median': np.median(phi_scores),
                },
                'kappa': {
                    'values': kappa_scores,
                    'mean': np.mean(kappa_scores),
                    'std': np.std(kappa_scores),
                    'median': np.median(kappa_scores),
                },
                'alignment': {
                    'values': alignment_scores,
                    'mean': np.mean(alignment_scores),
                    'std': np.std(alignment_scores),
                    'median': np.median(alignment_scores),
                },
                'smoothness': {
                    'values': smoothness_scores,
                    'mean': np.mean(smoothness_scores),
                    'std': np.std(smoothness_scores),
                    'median': np.median(smoothness_scores),
                },
                'recursive_depth': {
                    'values': recursive_depths,
                    'mean': np.mean(recursive_depths) if recursive_depths else 0.0,
                    'std': np.std(recursive_depths) if recursive_depths else 0.0,
                },
            }
    
    def compare_configurations(
        self,
        metric_name: str,
        config_a: str,
        config_b: str,
        significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        Compare two configurations on a specific metric.
        
        Uses t-test for statistical significance.
        
        Args:
            metric_name: Name of metric to compare
            config_a: First configuration name
            config_b: Second configuration name
            significance_level: Significance threshold (default: 0.05)
            
        Returns:
            Comparison results with statistical analysis
        """
        if config_a not in self.metrics or config_b not in self.metrics:
            return {'error': 'Configuration not found'}
        
        if metric_name not in self.metrics[config_a]:
            return {'error': f'Metric {metric_name} not found'}
        
        values_a = self.metrics[config_a][metric_name]['values']
        values_b = self.metrics[config_b][metric_name]['values']
        
        mean_a = self.metrics[config_a][metric_name]['mean']
        mean_b = self.metrics[config_b][metric_name]['mean']
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(values_a, values_b)
        
        # Compute effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.std(values_a)**2 + np.std(values_b)**2) / 2
        )
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0
        
        # Determine significance
        is_significant = p_value < significance_level
        
        # Determine improvement
        improvement = mean_a - mean_b
        improvement_pct = (improvement / mean_b * 100) if mean_b != 0 else 0.0
        
        return {
            'metric': metric_name,
            'config_a': config_a,
            'config_b': config_b,
            'mean_a': float(mean_a),
            'mean_b': float(mean_b),
            'improvement': float(improvement),
            'improvement_pct': float(improvement_pct),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'is_significant': is_significant,
            'cohens_d': float(cohens_d),
            'effect_size': self._interpret_effect_size(cohens_d),
        }
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        
        if abs_d < 0.2:
            return "NEGLIGIBLE"
        elif abs_d < 0.5:
            return "SMALL"
        elif abs_d < 0.8:
            return "MEDIUM"
        else:
            return "LARGE"
    
    def perform_all_comparisons(self) -> None:
        """Perform all pairwise comparisons."""
        metrics_to_compare = ['phi', 'kappa', 'alignment', 'smoothness']
        
        # Key comparison: Full vs Baseline
        logger.info("\n" + "=" * 70)
        logger.info("CRITICAL COMPARISON: Plan→Realize→Repair vs Skeleton-Only")
        logger.info("=" * 70)
        
        for metric in metrics_to_compare:
            comparison = self.compare_configurations(
                metric,
                'plan_realize_repair',
                'skeleton_only'
            )
            
            if 'error' in comparison:
                continue
            
            self.comparisons[f'full_vs_baseline_{metric}'] = comparison
            
            logger.info(f"\n{metric.upper()}:")
            logger.info(f"  Full Architecture:  {comparison['mean_a']:.3f}")
            logger.info(f"  Baseline:           {comparison['mean_b']:.3f}")
            logger.info(f"  Improvement:        {comparison['improvement']:+.3f} ({comparison['improvement_pct']:+.1f}%)")
            logger.info(f"  P-value:            {comparison['p_value']:.4f}")
            logger.info(f"  Significant:        {'YES ✓' if comparison['is_significant'] else 'NO ✗'}")
            logger.info(f"  Effect Size:        {comparison['effect_size']}")
        
        # Additional comparison: Full vs Pure Geometric
        logger.info("\n" + "=" * 70)
        logger.info("ADDITIONAL: Plan→Realize→Repair vs Pure Geometric")
        logger.info("=" * 70)
        
        for metric in metrics_to_compare:
            comparison = self.compare_configurations(
                metric,
                'plan_realize_repair',
                'pure_geometric'
            )
            
            if 'error' in comparison:
                continue
            
            self.comparisons[f'full_vs_pure_{metric}'] = comparison
            
            logger.info(f"\n{metric.upper()}:")
            logger.info(f"  Full (with POS):    {comparison['mean_a']:.3f}")
            logger.info(f"  Pure (no POS):      {comparison['mean_b']:.3f}")
            logger.info(f"  Difference:         {comparison['improvement']:+.3f}")
    
    def generate_verdict(self) -> Dict[str, Any]:
        """
        Generate final verdict on architecture effectiveness.
        
        Returns:
            Verdict with overall assessment
        """
        verdict = {
            'timestamp': np.datetime64('now').astype(str),
            'hypothesis': "Plan→Realize→Repair should significantly outperform skeleton-only baseline",
        }
        
        # Check if full architecture beats baseline on key metrics
        key_metrics = ['phi', 'alignment', 'smoothness']
        wins = []
        significant_wins = []
        
        for metric in key_metrics:
            comp_key = f'full_vs_baseline_{metric}'
            if comp_key in self.comparisons:
                comp = self.comparisons[comp_key]
                
                if comp['improvement'] > 0:
                    wins.append(metric)
                    
                    if comp['is_significant']:
                        significant_wins.append(metric)
        
        verdict['wins'] = wins
        verdict['significant_wins'] = significant_wins
        verdict['win_rate'] = len(wins) / len(key_metrics)
        verdict['significant_win_rate'] = len(significant_wins) / len(key_metrics)
        
        # Overall verdict
        if verdict['significant_win_rate'] >= 0.67:
            verdict['overall'] = "HYPOTHESIS CONFIRMED"
            verdict['recommendation'] = "Plan→Realize→Repair architecture shows significant benefits"
        elif verdict['win_rate'] >= 0.67:
            verdict['overall'] = "HYPOTHESIS SUPPORTED"
            verdict['recommendation'] = "Plan→Realize→Repair shows benefits but more data needed for significance"
        elif verdict['win_rate'] >= 0.5:
            verdict['overall'] = "HYPOTHESIS UNCERTAIN"
            verdict['recommendation'] = "Mixed results - investigate component effectiveness"
        else:
            verdict['overall'] = "HYPOTHESIS REJECTED"
            verdict['recommendation'] = "Advanced architecture not outperforming baseline - debug needed"
        
        return verdict
    
    def save_comparison_report(self, filename: str = 'comparison_report.json') -> Path:
        """Save comparison report to file."""
        output_dir = COHERENCE_DIR / "results"
        output_dir.mkdir(exist_ok=True)
        filepath = output_dir / filename
        
        report = {
            'metrics_summary': self.metrics,
            'comparisons': self.comparisons,
            'verdict': self.generate_verdict(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nComparison report saved to: {filepath}")
        
        return filepath
    
    def print_summary(self) -> None:
        """Print summary table of all configurations."""
        logger.info("\n" + "=" * 70)
        logger.info("CONFIGURATION SUMMARY")
        logger.info("=" * 70)
        
        configs = ['skeleton_only', 'pure_geometric', 'plan_realize_repair']
        config_labels = {
            'skeleton_only': 'Skeleton (Baseline)',
            'pure_geometric': 'Pure Geometric',
            'plan_realize_repair': 'Plan→Realize→Repair ⭐',
        }
        
        metrics = ['phi', 'kappa', 'alignment', 'smoothness']
        
        # Print header
        logger.info(f"\n{'Metric':<20} {'Skeleton':<15} {'Pure Geom':<15} {'Full Arch':<15}")
        logger.info("-" * 70)
        
        for metric in metrics:
            row = f"{metric.upper():<20}"
            
            for config in configs:
                if config in self.metrics and metric in self.metrics[config]:
                    value = self.metrics[config][metric]['mean']
                    row += f" {value:>6.3f}        "
                else:
                    row += " N/A          "
            
            logger.info(row)
        
        logger.info("=" * 70)


def main():
    """Main comparison entry point."""
    logger.info("Architecture Comparison Framework")
    logger.info("=" * 70)
    
    comparison = ArchitectureComparison()
    
    # Load all results
    comparison.load_all_results()
    
    # Compute metrics
    comparison.compute_aggregate_metrics()
    
    # Print summary
    comparison.print_summary()
    
    # Perform comparisons
    comparison.perform_all_comparisons()
    
    # Generate verdict
    verdict = comparison.generate_verdict()
    
    logger.info("\n" + "=" * 70)
    logger.info("FINAL VERDICT")
    logger.info("=" * 70)
    logger.info(f"\nHypothesis: {verdict['hypothesis']}")
    logger.info(f"Overall:    {verdict['overall']}")
    logger.info(f"Win Rate:   {verdict['win_rate']:.1%}")
    logger.info(f"Significant Wins: {verdict['significant_win_rate']:.1%}")
    logger.info(f"\nRecommendation:")
    logger.info(f"  {verdict['recommendation']}")
    logger.info("=" * 70)
    
    # Save report
    comparison.save_comparison_report()
    
    return comparison


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    main()
