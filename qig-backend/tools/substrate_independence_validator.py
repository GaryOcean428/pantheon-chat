#!/usr/bin/env python3
"""
Substrate Independence Validator
=================================

GFP:
  role: validation
  status: WORKING
  phase: TESTING
  dim: 3
  scope: cross-repo
  version: 2025-12-29
  owner: pantheon-chat

Validates substrate independence by comparing β-functions and κ values
across different substrates (physics, semantic, biological).

Background:
-----------
QIG hypothesis: Universal fixed point κ* ≈ 64 should appear in all
substrates, with similar β-function patterns during convergence.

This tool:
1. Loads frozen physics results from qig-verification
2. Measures semantic β from pantheon-chat/Ocean
3. Compares substrates for publication-ready metrics

Usage:
------
    # Compare physics and semantic
    python substrate_independence_validator.py \
        --physics qig-verification/FROZEN_FACTS.json \
        --semantic pantheon-chat/beta_results.json \
        --output substrate_comparison.json
    
    # Generate publication plots
    python substrate_independence_validator.py \
        --physics data/physics.json \
        --semantic data/semantic.json \
        --plot publication_figure.png

References:
-----------
- Consciousness Protocol v4.0 §1 Task 4
- qig-verification/FROZEN_FACTS.md
- Paper 1: "Universal Fixed Point in Information Geometry"
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import sys

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not available, some features disabled")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class SubstrateData:
    """Data for a single substrate."""
    name: str
    kappa_star: float
    kappa_star_error: float
    beta_emergence: float
    beta_plateau: float
    beta_fixed_point: float
    n_measurements: int
    method: str  # e.g., 'DMRG', 'semantic', 'biological'
    source: str


@dataclass
class ComparisonResult:
    """Result of substrate comparison."""
    substrate_a: str
    substrate_b: str
    kappa_match_pct: float
    beta_emergence_match_pct: float
    beta_plateau_match_pct: float
    beta_fixed_point_match_pct: float
    overall_match_pct: float
    verdict: str


class SubstrateIndependenceValidator:
    """
    Validates substrate independence hypothesis.
    
    Compares physics, semantic, and other substrates to verify
    that QIG principles are universal.
    """
    
    def __init__(self):
        self.substrates: Dict[str, SubstrateData] = {}
        
        # Load frozen physics constants
        self._load_frozen_physics()
    
    def _load_frozen_physics(self):
        """Load frozen physics constants as reference."""
        # Hard-coded from qig-verification FROZEN_FACTS
        physics_data = SubstrateData(
            name='physics',
            kappa_star=64.21,
            kappa_star_error=0.92,
            beta_emergence=0.443,
            beta_plateau=-0.013,
            beta_fixed_point=0.013,
            n_measurements=108,  # L=6: 3 seeds × 36 perts
            method='DMRG',
            source='qig-verification/FROZEN_FACTS.md (2025-12-08)'
        )
        self.substrates['physics'] = physics_data
    
    def load_substrate_from_file(self, filepath: Path, substrate_name: str):
        """
        Load substrate data from JSON file.
        
        Expected format:
        {
            "kappa_star": 64.21,
            "kappa_star_error": 0.92,
            "beta_emergence": 0.443,
            "beta_plateau": -0.013,
            "beta_fixed_point": 0.013,
            "n_measurements": 108,
            "method": "DMRG",
            "source": "qig-verification"
        }
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        substrate = SubstrateData(
            name=substrate_name,
            kappa_star=data['kappa_star'],
            kappa_star_error=data.get('kappa_star_error', 0.0),
            beta_emergence=data['beta_emergence'],
            beta_plateau=data['beta_plateau'],
            beta_fixed_point=data['beta_fixed_point'],
            n_measurements=data.get('n_measurements', 0),
            method=data.get('method', 'unknown'),
            source=data.get('source', str(filepath))
        )
        
        self.substrates[substrate_name] = substrate
    
    def add_substrate(self, substrate: SubstrateData):
        """Add substrate data directly."""
        self.substrates[substrate.name] = substrate
    
    def compare_substrates(
        self,
        substrate_a: str,
        substrate_b: str
    ) -> ComparisonResult:
        """
        Compare two substrates for substrate independence.
        
        Args:
            substrate_a: Name of first substrate
            substrate_b: Name of second substrate
            
        Returns:
            ComparisonResult with detailed comparison
        """
        if substrate_a not in self.substrates:
            raise ValueError(f"Substrate '{substrate_a}' not loaded")
        if substrate_b not in self.substrates:
            raise ValueError(f"Substrate '{substrate_b}' not loaded")
        
        sub_a = self.substrates[substrate_a]
        sub_b = self.substrates[substrate_b]
        
        # Compare κ*
        kappa_diff = abs(sub_a.kappa_star - sub_b.kappa_star)
        kappa_relative = kappa_diff / sub_a.kappa_star
        kappa_match_pct = max(0.0, 100.0 * (1.0 - kappa_relative))
        
        # Compare β-functions
        beta_matches = {}
        for beta_type in ['emergence', 'plateau', 'fixed_point']:
            val_a = getattr(sub_a, f'beta_{beta_type}')
            val_b = getattr(sub_b, f'beta_{beta_type}')
            
            # Handle near-zero betas specially
            if abs(val_a) < 1e-6 and abs(val_b) < 1e-6:
                match_pct = 100.0
            elif abs(val_a) < 1e-6 or abs(val_b) < 1e-6:
                # One is zero, other is not
                match_pct = 50.0 if abs(val_b - val_a) < 0.03 else 0.0
            else:
                diff = abs(val_a - val_b)
                relative = diff / abs(val_a)
                match_pct = max(0.0, 100.0 * (1.0 - relative))
            
            beta_matches[beta_type] = match_pct
        
        # Overall match (weighted average)
        overall = (
            0.3 * kappa_match_pct +
            0.3 * beta_matches['emergence'] +
            0.2 * beta_matches['plateau'] +
            0.2 * beta_matches['fixed_point']
        )
        
        # Verdict
        if overall > 95.0:
            verdict = 'SUBSTRATE INDEPENDENCE VALIDATED'
        elif overall > 85.0:
            verdict = 'SUBSTRATE INDEPENDENCE CONFIRMED'
        elif overall > 70.0:
            verdict = 'PARTIAL MATCH'
        else:
            verdict = 'SUBSTRATE MISMATCH'
        
        return ComparisonResult(
            substrate_a=substrate_a,
            substrate_b=substrate_b,
            kappa_match_pct=kappa_match_pct,
            beta_emergence_match_pct=beta_matches['emergence'],
            beta_plateau_match_pct=beta_matches['plateau'],
            beta_fixed_point_match_pct=beta_matches['fixed_point'],
            overall_match_pct=overall,
            verdict=verdict
        )
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate full comparison report for all substrates.
        
        Returns:
            Dict with comparisons and summary statistics
        """
        comparisons = []
        substrate_names = list(self.substrates.keys())
        
        # Compare all pairs
        for i, sub_a in enumerate(substrate_names):
            for sub_b in substrate_names[i+1:]:
                comparison = self.compare_substrates(sub_a, sub_b)
                comparisons.append(asdict(comparison))
        
        # Summary statistics
        if comparisons:
            overall_matches = [c['overall_match_pct'] for c in comparisons]
            mean_match = sum(overall_matches) / len(overall_matches) if HAS_NUMPY else 0.0
        else:
            mean_match = 0.0
        
        return {
            'substrates': {
                name: asdict(data) 
                for name, data in self.substrates.items()
            },
            'comparisons': comparisons,
            'summary': {
                'n_substrates': len(self.substrates),
                'n_comparisons': len(comparisons),
                'mean_match_pct': mean_match,
                'hypothesis_confirmed': mean_match > 90.0
            }
        }
    
    def plot_comparison(self, output_path: Optional[Path] = None):
        """
        Generate publication-ready comparison plot.
        
        Args:
            output_path: Path to save figure (shows if None)
        """
        if not HAS_MATPLOTLIB:
            print("Error: matplotlib required for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: κ* comparison
        ax = axes[0, 0]
        substrate_names = list(self.substrates.keys())
        kappa_values = [self.substrates[s].kappa_star for s in substrate_names]
        kappa_errors = [self.substrates[s].kappa_star_error for s in substrate_names]
        
        ax.errorbar(range(len(substrate_names)), kappa_values, 
                    yerr=kappa_errors, fmt='o', capsize=5)
        ax.axhline(y=64.21, color='r', linestyle='--', label='κ* = 64.21')
        ax.set_xticks(range(len(substrate_names)))
        ax.set_xticklabels(substrate_names, rotation=45)
        ax.set_ylabel('κ*')
        ax.set_title('Fixed Point Coupling Across Substrates')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: β-function comparison
        ax = axes[0, 1]
        beta_types = ['emergence', 'plateau', 'fixed_point']
        x = np.arange(len(beta_types)) if HAS_NUMPY else range(len(beta_types))
        width = 0.35
        
        for i, substrate_name in enumerate(substrate_names):
            substrate = self.substrates[substrate_name]
            betas = [
                substrate.beta_emergence,
                substrate.beta_plateau,
                substrate.beta_fixed_point
            ]
            offset = width * (i - len(substrate_names)/2 + 0.5)
            if HAS_NUMPY:
                ax.bar(x + offset, betas, width, label=substrate_name)
            else:
                ax.bar([xi + offset for xi in x], betas, width, label=substrate_name)
        
        ax.set_xlabel('Scale')
        ax.set_ylabel('β')
        ax.set_title('β-Function Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(beta_types)
        ax.legend()
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Match percentages
        ax = axes[1, 0]
        comparisons = []
        substrate_names_list = list(self.substrates.keys())
        for i, sub_a in enumerate(substrate_names_list):
            for sub_b in substrate_names_list[i+1:]:
                comp = self.compare_substrates(sub_a, sub_b)
                comparisons.append(comp)
        
        if comparisons:
            labels = [f"{c.substrate_a}-{c.substrate_b}" for c in comparisons]
            matches = [c.overall_match_pct for c in comparisons]
            
            colors = ['green' if m > 95 else 'yellow' if m > 85 else 'red' 
                     for m in matches]
            ax.barh(range(len(labels)), matches, color=colors)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.set_xlabel('Match %')
            ax.set_title('Substrate Match Quality')
            ax.axvline(x=95, color='g', linestyle='--', alpha=0.5)
            ax.axvline(x=85, color='y', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3, axis='x')
        
        # Plot 4: Summary text
        ax = axes[1, 1]
        ax.axis('off')
        
        report = self.generate_report()
        summary_text = f"""
Substrate Independence Validation
{'=' * 40}

Substrates: {report['summary']['n_substrates']}
Comparisons: {report['summary']['n_comparisons']}

Mean Match: {report['summary']['mean_match_pct']:.1f}%

Hypothesis: {'✅ CONFIRMED' if report['summary']['hypothesis_confirmed'] else '❌ REJECTED'}

Details:
"""
        for name, data in report['substrates'].items():
            summary_text += f"\n{name}:"
            summary_text += f"\n  κ* = {data['kappa_star']:.2f} ± {data['kappa_star_error']:.2f}"
            summary_text += f"\n  β(em) = {data['beta_emergence']:.3f}"
            summary_text += f"\n  N = {data['n_measurements']}"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                fontfamily='monospace', verticalalignment='top',
                fontsize=9)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {output_path}")
        else:
            plt.show()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Validate substrate independence hypothesis'
    )
    parser.add_argument(
        '--physics',
        type=Path,
        help='Path to physics results JSON (optional, uses frozen if not provided)'
    )
    parser.add_argument(
        '--semantic',
        type=Path,
        help='Path to semantic results JSON'
    )
    parser.add_argument(
        '--biological',
        type=Path,
        help='Path to biological results JSON (optional)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output JSON path for results'
    )
    parser.add_argument(
        '--plot',
        type=Path,
        help='Generate comparison plot (PNG)'
    )
    
    args = parser.parse_args()
    
    # Create validator
    validator = SubstrateIndependenceValidator()
    
    # Load additional substrates
    if args.semantic:
        validator.load_substrate_from_file(args.semantic, 'semantic')
    
    if args.biological:
        validator.load_substrate_from_file(args.biological, 'biological')
    
    # Generate report
    report = validator.generate_report()
    
    # Print to console
    print("\n" + "=" * 60)
    print("SUBSTRATE INDEPENDENCE VALIDATION REPORT")
    print("=" * 60 + "\n")
    
    print(f"Substrates analyzed: {report['summary']['n_substrates']}")
    print(f"Comparisons made: {report['summary']['n_comparisons']}")
    print(f"Mean match: {report['summary']['mean_match_pct']:.1f}%\n")
    
    for comparison in report['comparisons']:
        print(f"\n{comparison['substrate_a']} vs {comparison['substrate_b']}:")
        print(f"  κ* match: {comparison['kappa_match_pct']:.1f}%")
        print(f"  β(emergence): {comparison['beta_emergence_match_pct']:.1f}%")
        print(f"  β(plateau): {comparison['beta_plateau_match_pct']:.1f}%")
        print(f"  Overall: {comparison['overall_match_pct']:.1f}%")
        print(f"  Verdict: {comparison['verdict']}")
    
    print("\n" + "=" * 60)
    if report['summary']['hypothesis_confirmed']:
        print("✅ SUBSTRATE INDEPENDENCE HYPOTHESIS CONFIRMED")
    else:
        print("⚠️ SUBSTRATE INDEPENDENCE REQUIRES FURTHER INVESTIGATION")
    print("=" * 60 + "\n")
    
    # Save JSON
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✅ Report saved to {args.output}")
    
    # Generate plot
    if args.plot:
        validator.plot_comparison(args.plot)
    
    return 0 if report['summary']['hypothesis_confirmed'] else 1


if __name__ == '__main__':
    sys.exit(main())
