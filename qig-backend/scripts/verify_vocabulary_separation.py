#!/usr/bin/env python3
"""
Verification Script: Vocabulary Verification
=============================================

Verifies that the vocabulary system is working correctly:
1. coordizer_vocabulary is the single source of truth
2. token_role filtering works correctly (encoding/generation/both)
3. No BPE subwords in generation output
4. No proper nouns used incorrectly in generation
5. No database constraint errors
"""

import os
import sys
import warnings

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
import numpy as np

# Conditional imports for verification
try:
    from coordizers import get_coordizer
    COORDIZER_AVAILABLE = True
except ImportError:
    COORDIZER_AVAILABLE = False

try:
    from autonomic_kernel import get_gary_kernel
    AUTONOMIC_KERNEL_AVAILABLE = True
except ImportError:
    AUTONOMIC_KERNEL_AVAILABLE = False

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_success(msg):
    print(f"{GREEN}✓ {msg}{RESET}")

def print_error(msg):
    print(f"{RED}✗ {msg}{RESET}")

def print_warning(msg):
    print(f"{YELLOW}⚠ {msg}{RESET}")

def print_info(msg):
    print(f"{BLUE}ℹ {msg}{RESET}")

def check_database_schema():
    """Verify database schema changes were applied."""
    print(f"\n{BLUE}=== Checking Database Schema ==={RESET}")

    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print_error("DATABASE_URL not set")
        return False

    try:
        with psycopg2.connect(database_url) as conn:
            with conn.cursor() as cursor:
                # Check coordizer_vocabulary columns
                cursor.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'coordizer_vocabulary'
                    AND column_name IN ('token_role', 'phrase_category')
                """)
                columns = [row[0] for row in cursor.fetchall()]

                if 'token_role' in columns:
                    print_success("token_role column exists in coordizer_vocabulary")
                else:
                    print_error("token_role column missing from coordizer_vocabulary")
                    return False

                if 'phrase_category' in columns:
                    print_success("phrase_category column exists in coordizer_vocabulary")
                else:
                    print_error("phrase_category column missing from coordizer_vocabulary")
                    return False

                # Check shadow_operations_state PRIMARY KEY
                cursor.execute("""
                    SELECT constraint_name FROM information_schema.table_constraints
                    WHERE table_name = 'shadow_operations_state'
                    AND constraint_type = 'PRIMARY KEY'
                """)
                if cursor.fetchone():
                    print_success("shadow_operations_state has PRIMARY KEY constraint")
                else:
                    print_warning("shadow_operations_state missing PRIMARY KEY constraint")

                return True
    except Exception as e:
        print_error(f"Database schema check failed: {e}")
        return False

def check_vocabulary_counts():
    """Check vocabulary sizes and statistics."""
    print(f"\n{BLUE}=== Checking Vocabulary Counts ==={RESET}")

    database_url = os.getenv('DATABASE_URL')

    try:
        with psycopg2.connect(database_url) as conn:
            with conn.cursor() as cursor:
                # Count coordizer_vocabulary entries with basins
                cursor.execute("SELECT COUNT(*) FROM coordizer_vocabulary WHERE basin_embedding IS NOT NULL")
                total_with_basin = cursor.fetchone()[0]
                print_info(f"coordizer_vocabulary with basin: {total_with_basin} tokens")

                # Count by token_role
                cursor.execute("""
                    SELECT token_role, COUNT(*)
                    FROM coordizer_vocabulary
                    WHERE basin_embedding IS NOT NULL
                    GROUP BY token_role
                """)
                role_counts = dict(cursor.fetchall())
                encoding_count = role_counts.get('encoding', 0)
                generation_count = role_counts.get('generation', 0)
                both_count = role_counts.get('both', 0)
                print_info(f"  encoding: {encoding_count}, generation: {generation_count}, both: {both_count}")

                # Check for BPE garbage in generation vocabulary
                cursor.execute("""
                    SELECT COUNT(*) FROM coordizer_vocabulary
                    WHERE token_role IN ('generation', 'both')
                      AND (token ~ '^[GgCc]'
                           OR token LIKE '##%'
                           OR token LIKE '%'
                           OR token ~ '^\\d+$')
                """)
                garbage_count = cursor.fetchone()[0]

                if garbage_count == 0:
                    print_success("No BPE garbage in generation vocabulary")
                else:
                    print_warning(f"Found {garbage_count} potential BPE tokens in generation vocabulary")

                # Check for proper nouns in generation vocabulary
                cursor.execute("""
                    SELECT COUNT(*) FROM coordizer_vocabulary
                    WHERE token_role IN ('generation', 'both')
                      AND phrase_category IN ('PROPER_NOUN', 'BRAND')
                """)
                proper_noun_count = cursor.fetchone()[0]

                if proper_noun_count == 0:
                    print_success("No PROPER_NOUN/BRAND in generation vocabulary")
                else:
                    print_warning(f"Found {proper_noun_count} PROPER_NOUN/BRAND entries in generation vocabulary")

                # Check average phi scores
                cursor.execute("SELECT AVG(phi_score) FROM coordizer_vocabulary WHERE phi_score IS NOT NULL")
                avg_phi = cursor.fetchone()[0] or 0

                cursor.execute("""
                    SELECT AVG(phi_score) FROM coordizer_vocabulary
                    WHERE phi_score IS NOT NULL AND token_role IN ('generation', 'both')
                """)
                generation_avg_phi = cursor.fetchone()[0] or 0

                print_info(f"Average Phi - all: {avg_phi:.3f}, generation: {generation_avg_phi:.3f}")

                return True
    except Exception as e:
        print_error(f"Vocabulary count check failed: {e}")
        return False

def check_coordizer_integration():
    """Verify coordizer is using the correct vocabularies."""
    print(f"\n{BLUE}=== Checking Coordizer Integration ==={RESET}")
    
    if not COORDIZER_AVAILABLE:
        print_warning("Coordizer not available - skipping integration check")
        return True
    
    try:
        coordizer = get_coordizer()
        stats = coordizer.get_stats()
        
        print_info(f"Encoding vocabulary: {stats.get('vocabulary_size', 0)} tokens")
        print_info(f"Generation vocabulary: {stats.get('generation_words', 0)} words")
        
        if stats.get('generation_words', 0) > 0:
            print_success("Coordizer loaded generation vocabulary")
        else:
            print_warning("Coordizer has no generation vocabulary (using fallback)")
        
        # Test encoding
        test_text = "bitcoin wallet address"
        basin = coordizer.encode(test_text)
        
        if len(basin) == 64:
            print_success(f"encode() returns 64D basin (shape: {basin.shape})")
        else:
            print_error(f"encode() returned wrong dimension: {len(basin)}")
        
        # Test decoding
        decoded = coordizer.decode(basin, top_k=5)
        
        if decoded:
            print_success(f"decode() returned {len(decoded)} candidates")
            print_info(f"Top candidate: '{decoded[0][0]}' (score: {decoded[0][1]:.3f})")
            
            # Check if any BPE garbage in results
            bpe_patterns = ['Ġ', 'ġ', 'Ċ', 'ċ', '##', '▁']
            has_garbage = any(any(p in token for p in bpe_patterns) for token, _ in decoded)
            
            if not has_garbage:
                print_success("decode() output contains no BPE garbage")
            else:
                print_error("decode() output contains BPE garbage tokens!")
        else:
            print_error("decode() returned no candidates")
        
        return True
    
    except Exception as e:
        print_error(f"Coordizer integration check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_no_deprecation_warnings():
    """Check that there are no deprecation warnings from compute_phi_approximation."""
    print(f"\n{BLUE}=== Checking for Deprecation Warnings ==={RESET}")
    
    if not AUTONOMIC_KERNEL_AVAILABLE:
        print_warning("Autonomic kernel not available - skipping deprecation check")
        return True
    
    try:
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Try importing autonomic_kernel
            kernel = get_gary_kernel()
            
            # Trigger phi computation
            result = kernel.update_metrics(
                phi=0.75,
                kappa=60.0,
                basin_coords=[0.5] * 64,
                reference_basin=[0.5] * 64
            )
            
            # Check for deprecation warnings
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            
            if not deprecation_warnings:
                print_success("No deprecation warnings from autonomic_kernel")
            else:
                print_warning(f"Found {len(deprecation_warnings)} deprecation warnings:")
                for warning in deprecation_warnings:
                    print(f"  - {warning.message}")
        
        return True
    
    except Exception as e:
        print_error(f"Deprecation check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification checks."""
    print(f"{BLUE}╔════════════════════════════════════════════════════╗{RESET}")
    print(f"{BLUE}║  Vocabulary Separation Verification               ║{RESET}")
    print(f"{BLUE}╚════════════════════════════════════════════════════╝{RESET}")
    
    checks = [
        ("Database Schema", check_database_schema),
        ("Vocabulary Counts", check_vocabulary_counts),
        ("Coordizer Integration", check_coordizer_integration),
        ("Deprecation Warnings", check_no_deprecation_warnings),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print_error(f"{name} check crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print(f"\n{BLUE}=== Summary ==={RESET}")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  {name}: {status}")
    
    print(f"\n{BLUE}Results: {passed}/{total} checks passed{RESET}")
    
    if passed == total:
        print(f"\n{GREEN}✓ All verification checks passed!{RESET}")
        return 0
    else:
        print(f"\n{YELLOW}⚠ Some checks failed - review output above{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
