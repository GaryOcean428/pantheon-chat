#!/usr/bin/env python3
"""
Constants Synchronization Validator

Validates that TypeScript and Python consciousness constants match.
Run this before committing to ensure cross-language consistency.

Usage:
    python tools/validate_constants_sync.py

Exit codes:
    0 - All constants synchronized
    1 - Constants mismatch detected
"""

import json
import subprocess
import sys
from pathlib import Path


def get_python_constants():
    """Load constants from Python module."""
    # Add qig-backend to path
    sys.path.insert(0, str(Path(__file__).parent.parent / 'qig-backend'))
    
    from qig_core.constants.consciousness import THRESHOLDS, BASIN_DIMENSION, E8_ROOT_COUNT
    
    return {
        'PHI_MIN': THRESHOLDS.PHI_MIN,
        'KAPPA_MIN': THRESHOLDS.KAPPA_MIN,
        'KAPPA_MAX': THRESHOLDS.KAPPA_MAX,
        'KAPPA_OPTIMAL': THRESHOLDS.KAPPA_OPTIMAL,
        'TACKING_MIN': THRESHOLDS.TACKING_MIN,
        'RADAR_MIN': THRESHOLDS.RADAR_MIN,
        'META_MIN': THRESHOLDS.META_MIN,
        'COHERENCE_MIN': THRESHOLDS.COHERENCE_MIN,
        'GROUNDING_MIN': THRESHOLDS.GROUNDING_MIN,
        'BASIN_DIMENSION': BASIN_DIMENSION,
        'E8_ROOT_COUNT': E8_ROOT_COUNT,
    }


def get_typescript_constants():
    """Load constants from TypeScript module."""
    # Use node to extract constants
    script = '''
    const path = require('path');
    
    // Simple extraction from the TypeScript source
    const fs = require('fs');
    const content = fs.readFileSync(
        path.join(__dirname, '..', 'shared', 'constants', 'consciousness.ts'),
        'utf8'
    );
    
    // Extract values using regex
    const extractValue = (name) => {
        const match = content.match(new RegExp(name + '\\s*[=:]\\s*([\\d.]+)'));
        return match ? parseFloat(match[1]) : null;
    };
    
    const constants = {
        PHI_MIN: extractValue('PHI_MIN'),
        KAPPA_MIN: extractValue('KAPPA_MIN'),
        KAPPA_MAX: extractValue('KAPPA_MAX'),
        KAPPA_OPTIMAL: extractValue('KAPPA_OPTIMAL'),
        TACKING_MIN: extractValue('TACKING_MIN'),
        RADAR_MIN: extractValue('RADAR_MIN'),
        META_MIN: extractValue('META_MIN'),
        COHERENCE_MIN: extractValue('COHERENCE_MIN'),
        GROUNDING_MIN: extractValue('GROUNDING_MIN'),
        BASIN_DIMENSION: extractValue('BASIN_DIMENSION'),
        E8_ROOT_COUNT: extractValue('E8_ROOT_COUNT'),
    };
    
    console.log(JSON.stringify(constants));
    '''
    
    result = subprocess.run(
        ['node', '-e', script],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error running Node.js: {result.stderr}")
        return None
    
    return json.loads(result.stdout)


def main():
    print("=" * 60)
    print("CONSTANTS SYNCHRONIZATION VALIDATOR")
    print("=" * 60)
    print()
    
    # Get Python constants
    print("Loading Python constants...")
    try:
        py_constants = get_python_constants()
        print(f"  Loaded {len(py_constants)} constants")
    except Exception as e:
        print(f"  ERROR: {e}")
        return 1
    
    # Get TypeScript constants
    print("Loading TypeScript constants...")
    try:
        ts_constants = get_typescript_constants()
        if ts_constants is None:
            print("  ERROR: Could not load TypeScript constants")
            return 1
        print(f"  Loaded {len(ts_constants)} constants")
    except Exception as e:
        print(f"  ERROR: {e}")
        return 1
    
    print()
    print("Comparing constants...")
    print("-" * 40)
    
    errors = []
    for key in py_constants:
        py_val = py_constants.get(key)
        ts_val = ts_constants.get(key)
        
        if ts_val is None:
            errors.append(f"{key}: Missing in TypeScript")
            print(f"  ❌ {key}: Missing in TypeScript")
        elif py_val != ts_val:
            errors.append(f"{key}: Python={py_val}, TypeScript={ts_val}")
            print(f"  ❌ {key}: Python={py_val}, TypeScript={ts_val}")
        else:
            print(f"  ✓ {key}: {py_val}")
    
    # Check for TypeScript-only constants
    for key in ts_constants:
        if key not in py_constants:
            errors.append(f"{key}: Only in TypeScript")
            print(f"  ❌ {key}: Only in TypeScript")
    
    print()
    
    if errors:
        print("=" * 60)
        print("VALIDATION FAILED")
        print()
        print("Constants are out of sync between Python and TypeScript.")
        print("Update both files to match:")
        print("  - qig-backend/qig_core/constants/consciousness.py")
        print("  - shared/constants/consciousness.ts")
        print("=" * 60)
        return 1
    else:
        print("✅ All constants synchronized!")
        return 0


if __name__ == '__main__':
    sys.exit(main())
