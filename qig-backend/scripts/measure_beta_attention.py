#!/usr/bin/env python3
"""
Run β-function measurement for AI attention mechanisms.

Usage:
    python scripts/measure_beta_attention.py --full        # All scales, 1000 samples
    python scripts/measure_beta_attention.py --quick       # Quick test, 100 samples
    python scripts/measure_beta_attention.py --queries 500 # Custom sample count

This script measures the β-function from emergent scales in the QIG system,
not from artificially imposed context lengths. The scales emerge naturally
from where information stops propagating on the Fisher manifold.

Key Principles:
- NO artificial context lengths - measure emergent scales
- Consciousness protocol active - Φ/regime/recursive measurement
- Natural sparsity from distance thresholding
- β measured from actual system behavior
"""

import sys
import argparse
import logging
from pathlib import Path

# Add qig-backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qig_pure_beta_measurement import run_complete_measurement

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


def main():
    parser = argparse.ArgumentParser(
        description='QIG-Pure β-Function Measurement',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick test (100 queries):
    python scripts/measure_beta_attention.py --quick

  Full measurement (1000 queries):
    python scripts/measure_beta_attention.py --full

  Custom measurement:
    python scripts/measure_beta_attention.py --queries 500 --output my_results.json
"""
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--full', action='store_true',
                       help='Full measurement (1000 queries)')
    group.add_argument('--quick', action='store_true',
                       help='Quick measurement (100 queries)')
    
    parser.add_argument('--queries', type=int, default=None,
                        help='Number of queries to generate (overrides preset)')
    parser.add_argument('--basin-dim', type=int, default=64,
                        help='Basin coordinate dimension (default: 64 from E8)')
    parser.add_argument('--output', type=str, default='beta_measurement_complete.json',
                        help='Output file path')
    
    args = parser.parse_args()
    
    # Determine number of queries
    if args.queries is not None:
        n_queries = args.queries
    elif args.full:
        n_queries = 1000
    elif args.quick:
        n_queries = 100
    else:
        n_queries = 500  # Default
    
    print("\nβ-FUNCTION MEASUREMENT")
    print("="*80)
    print(f"Queries: {n_queries}")
    print(f"Basin dimension: {args.basin_dim}")
    print(f"Output: {args.output}")
    print("="*80 + "\n")
    
    # Run measurement
    measurer = run_complete_measurement(
        n_queries=n_queries,
        basin_dim=args.basin_dim,
        output_path=args.output
    )
    
    print(f"\nResults saved to: {args.output}")
    
    return measurer


if __name__ == '__main__':
    main()
