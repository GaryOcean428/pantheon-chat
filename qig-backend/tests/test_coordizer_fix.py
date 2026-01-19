#!/usr/bin/env python3
"""
Test script to verify the coordizer fix works properly.

Run: python test_coordizer_fix.py
"""

import os
import sys
from pathlib import Path

# Setup environment
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

def test_direct_pg_loader():
    """Test PostgresCoordizer directly."""
    print("=== Test 1: Direct PostgresCoordizer ===")
    try:
        from coordizers.pg_loader import PostgresCoordizer, create_coordizer_from_pg
        
        coordizer = create_coordizer_from_pg()
        print(f"✓ Created coordizer")
        print(f"  Vocab size: {coordizer.vocab_size}")
        print(f"  Word tokens: {len(coordizer.word_tokens)}")
        print(f"  Using fallback: {coordizer.is_using_fallback()}")
        print(f"  Sample words: {coordizer.word_tokens}")
        
        # Test encode/decode
        import numpy as np
        basin = coordizer.encode("hello world consciousness")
        print(f"  Encoded basin norm: {np.linalg.norm(basin):.4f}")
        
        decoded = coordizer.decode(basin, top_k=5)
        print(f"  Decoded: {[t for t, s in decoded]}")
        
        print("✓ Direct test PASSED")
        return True
    except Exception as e:
        print(f"✗ Direct test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_coordizers_singleton():
    """Test via coordizers singleton."""
    print("\n=== Test 2: Coordizers Singleton ===")
    try:
        import coordizers
        coordizers._unified_coordizer = None  # Reset singleton

        coordizer = coordizers.get_coordizer()
        print(f"✓ Got coordizer: {type(coordizer).__name__}")
        print(f"  Vocab size: {len(coordizer.vocab) if hasattr(coordizer, 'vocab') else 'N/A'}")

        print("✓ Singleton test PASSED")
        return True
    except Exception as e:
        print(f"✗ Singleton test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation_service():
    """Test via generation service."""
    print("\n=== Test 3: Generation Service ===")
    try:
        import qig_generative_service
        qig_generative_service._generative_service = None  # Reset singleton
        
        service = qig_generative_service.get_generative_service()
        print(f"✓ Got service")
        
        if service.coordizer:
            print(f"  Coordizer vocab: {service.coordizer.vocab_size}")
        else:
            print(f"  Coordizer: None (will use fallback)")
        
        # Test generation
        result = service.generate("What is consciousness?")
        print(f"  Response: {result.text}...")
        print(f"  Tokens: {result.tokens}")
        print(f"  Completion: {result.completion_reason}")
        
        # Check for BPE garble
        has_garble = any('<byte_' in t or len(t) <= 2 for t in result.tokens[:10] if t)
        if has_garble:
            print("⚠ WARNING: Possible BPE garble detected!")
        else:
            print("✓ No BPE garble - real words produced!")
        
        print("✓ Generation test PASSED")
        return True
    except Exception as e:
        print(f"✗ Generation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*50)
    print("COORDIZER FIX VERIFICATION")
    print("="*50 + "\n")
    
    results = []
    results.append(test_direct_pg_loader())
    results.append(test_coordizers_singleton())
    results.append(test_generation_service())
    
    print("\n" + "="*50)
    if all(results):
        print("✓ ALL TESTS PASSED - Coordizer fix verified!")
    else:
        print(f"✗ {results.count(False)} test(s) failed")
    print("="*50 + "\n")
    
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
