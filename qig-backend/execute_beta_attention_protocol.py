#!/usr/bin/env python3
"""
Execute β-attention measurement protocol and save results.

This script runs the designed β-attention validation protocol to test
substrate independence: β_attention ≈ β_physics ≈ 0.44

Results are saved to docs/04-records/ for permanent record.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add qig-backend to path
sys.path.insert(0, str(Path(__file__).parent))

from beta_attention_measurement import run_beta_attention_validation


def main():
    """Execute β-attention protocol and save results."""
    
    print("=" * 80)
    print("β-ATTENTION MEASUREMENT PROTOCOL - EXECUTION")
    print("Testing substrate independence: β_attention ≈ β_physics ≈ 0.44")
    print("=" * 80)
    print()
    
    # Run validation with 100 samples per scale
    result = run_beta_attention_validation(samples_per_scale=100)
    
    # Save results to JSON
    output_dir = Path(__file__).parent.parent / 'docs' / '04-records'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f'{timestamp}-beta-attention-protocol-results.json'
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print()
    print("=" * 80)
    print(f"RESULTS SAVED: {output_file}")
    print("=" * 80)
    print()
    
    # Create markdown report
    md_file = output_dir / f'20260113-beta-attention-protocol-execution-1.00W.md'
    
    with open(md_file, 'w') as f:
        f.write("# β-Attention Measurement Protocol - Execution Results\n\n")
        f.write("**Date**: 2026-01-13\n")
        f.write("**Status**: ✅ EXECUTED\n")
        f.write("**Version**: 1.00W\n\n")
        f.write("---\n\n")
        
        f.write("## Executive Summary\n\n")
        
        status = "PASSED ✓" if result['validation_passed'] else "FAILED ✗"
        f.write(f"**Validation Status**: {status}\n\n")
        
        f.write(f"**Average κ**: {result['avg_kappa']:.2f}\n")
        f.write(f"**κ Range**: [{result['kappa_range'][0]:.2f}, {result['kappa_range'][1]:.2f}]\n")
        f.write(f"**Overall Deviation**: {result['overall_deviation']:.3f}\n")
        f.write(f"**Substrate Independence**: {'✓' if result['substrate_independence'] else '✗'}\n")
        f.write(f"**Plateau Detected**: {'✓' if result['plateau_detected'] else '✗'}")
        
        if result['plateau_scale']:
            f.write(f" at L={result['plateau_scale']}\n")
        else:
            f.write("\n")
        
        f.write("\n---\n\n")
        f.write("## Hypothesis\n\n")
        f.write("**Substrate Independence Prediction**: β_attention ≈ β_physics ≈ 0.44\n\n")
        f.write("If validated, this confirms that information geometry is substrate-independent,\n")
        f.write("with the same running coupling emerging in both physics and AI systems.\n\n")
        
        f.write("---\n\n")
        f.write("## Measurements\n\n")
        f.write("| Context Length | κ | Φ | Variance |\n")
        f.write("|----------------|---|---|----------|\n")
        
        for m in result['measurements']:
            f.write(f"| {m['context_length']} | {m['kappa']:.3f} | {m['phi']:.3f} | {m['variance']:.4f} |\n")
        
        f.write("\n---\n\n")
        f.write("## β-Function Trajectory\n\n")
        f.write("| Scale Transition | β_attention | β_physics | Deviation | Status |\n")
        f.write("|------------------|-------------|-----------|-----------|--------|\n")
        
        for b in result['beta_trajectory']:
            status_symbol = "✓" if b['within_acceptance'] else "✗"
            f.write(f"| {b['from_scale']}→{b['to_scale']} | {b['beta']:+.3f} | ")
            f.write(f"{b['reference_beta']:+.3f} | {b['deviation']:.3f} | {status_symbol} |\n")
        
        f.write("\n---\n\n")
        f.write("## Conclusion\n\n")
        
        if result['validation_passed']:
            f.write("**VALIDATION PASSED**: Substrate independence confirmed!\n\n")
            f.write("The β-function in AI attention mechanisms matches physics predictions,\n")
            f.write("supporting the hypothesis that information geometry is universal across substrates.\n")
        else:
            f.write("**VALIDATION FAILED**: Substrate independence not confirmed.\n\n")
            f.write("The β-function in AI attention mechanisms deviates from physics predictions.\n")
            f.write("Further investigation needed to understand substrate-specific effects.\n")
        
        f.write("\n---\n\n")
        f.write("## References\n\n")
        f.write("- β-attention measurement suite: `qig-backend/beta_attention_measurement.py`\n")
        f.write("- Physics validation: β(3→4) = +0.44 (frozen_physics.py)\n")
        f.write("- Raw results: `{}`\n".format(output_file.name))
        f.write("- Master roadmap: `docs/00-roadmap/20260112-master-roadmap-1.00W.md`\n")
    
    print(f"MARKDOWN REPORT: {md_file}")
    print()
    
    return 0 if result['validation_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
