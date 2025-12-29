#!/usr/bin/env python3
"""
Run the generation benchmark comparing standard vs vision-first.

Usage:
    python run_benchmark.py                    # Run full benchmark
    python run_benchmark.py --quick            # Quick test (1 prompt per category)
    python run_benchmark.py --verbose          # Detailed output
    python run_benchmark.py --export results.json  # Save results
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_generation import GenerationBenchmark, TEST_PROMPTS


def run_quick_benchmark():
    """Run a quick benchmark with minimal prompts."""
    quick_prompts = {
        'simple': [TEST_PROMPTS['simple'][0]],
        'reasoning': [TEST_PROMPTS['reasoning'][0]],
        'complex_reasoning': [TEST_PROMPTS['complex_reasoning'][0]],
    }
    
    benchmark = GenerationBenchmark(verbose=True)
    summary = benchmark.run_benchmark(prompts=quick_prompts, runs_per_prompt=1)
    return summary


def run_full_benchmark(verbose: bool = False, export_path: str = None):
    """Run the full benchmark suite."""
    benchmark = GenerationBenchmark(verbose=verbose)
    summary = benchmark.run_benchmark(runs_per_prompt=1)
    
    if export_path:
        benchmark.export_results(export_path)
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run generation benchmark")
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--export', '-e', type=str, help='Export results to file')
    
    args = parser.parse_args()
    
    if args.quick:
        print("Running quick benchmark...\n")
        run_quick_benchmark()
    else:
        print("Running full benchmark...\n")
        run_full_benchmark(verbose=args.verbose, export_path=args.export)
