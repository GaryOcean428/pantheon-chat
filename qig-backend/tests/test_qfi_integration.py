#!/usr/bin/env python3
"""
Integration tests for QFI Integrity Gate implementation.

Tests file structure, SQL syntax, and script executability.
Does not require database connection.

Related: Issue #97 - QFI Integrity Gate
"""

import os
import sys
import re

# Paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QIG_BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(QIG_BACKEND_DIR)


def test_quarantine_script_exists():
    """Test quarantine script exists and is executable."""
    script_path = os.path.join(QIG_BACKEND_DIR, 'scripts', 'quarantine_low_qfi_tokens.py')
    
    assert os.path.exists(script_path), f"Quarantine script not found: {script_path}"
    assert os.path.isfile(script_path), "Quarantine script is not a file"
    
    # Check shebang
    with open(script_path, 'r') as f:
        first_line = f.readline()
        assert first_line.startswith('#!/usr/bin/env python'), "Missing or invalid shebang"
    
    print("✓ Quarantine script exists and has shebang")


def test_quarantine_script_structure():
    """Test quarantine script has required components."""
    script_path = os.path.join(QIG_BACKEND_DIR, 'scripts', 'quarantine_low_qfi_tokens.py')
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for required imports
    assert 'import argparse' in content, "Missing argparse import"
    assert 'import psycopg2' in content, "Missing psycopg2 import"
    
    # Check for special symbols list
    assert 'SPECIAL_SYMBOLS' in content, "Missing SPECIAL_SYMBOLS constant"
    assert '<UNK>' in content, "Missing <UNK> in special symbols"
    assert '<PAD>' in content, "Missing <PAD> in special symbols"
    
    # Check for main function
    assert 'def main()' in content, "Missing main() function"
    assert 'def identify_low_qfi_tokens' in content, "Missing identify_low_qfi_tokens function"
    
    # Check for threshold handling
    assert '0.01' in content, "Missing QFI threshold value"
    assert '--threshold' in content, "Missing threshold argument"
    assert '--dry-run' in content, "Missing dry-run argument"
    
    print("✓ Quarantine script has all required components")


def test_migration_0015_exists():
    """Test migration 0015 exists."""
    migration_path = os.path.join(QIG_BACKEND_DIR, 'migrations', '0015_special_symbols_qfi.sql')
    
    assert os.path.exists(migration_path), f"Migration 0015 not found: {migration_path}"
    assert os.path.isfile(migration_path), "Migration 0015 is not a file"
    
    print("✓ Migration 0015 exists")


def test_migration_0015_structure():
    """Test migration 0015 has required SQL statements."""
    migration_path = os.path.join(QIG_BACKEND_DIR, 'migrations', '0015_special_symbols_qfi.sql')
    
    with open(migration_path, 'r') as f:
        content = f.read()
    
    # Check for transaction
    assert 'BEGIN;' in content, "Missing BEGIN transaction"
    assert 'COMMIT;' in content, "Missing COMMIT transaction"
    
    # Check for special symbols table
    assert 'CREATE TABLE IF NOT EXISTS special_symbols' in content, "Missing special_symbols table creation"
    
    # Check for special symbol backfill
    assert '<PAD>' in content, "Missing <PAD> symbol"
    assert '<UNK>' in content, "Missing <UNK> symbol"
    assert '<BOS>' in content, "Missing <BOS> symbol"
    assert '<EOS>' in content, "Missing <EOS> symbol"
    
    # Check for QFI values
    assert '1.0' in content, "Missing UNK QFI value (1.0)"
    assert '0.016' in content, "Missing PAD QFI value (0.016)"
    assert '0.015' in content, "Missing BOS/EOS QFI value (0.015)"
    
    # Check for constraints
    assert 'CHECK' in content, "Missing CHECK constraint"
    assert 'special_symbols_require_qfi' in content, "Missing special_symbols_require_qfi constraint"
    assert 'special_symbols_never_quarantined' in content, "Missing never_quarantined constraint"
    
    print("✓ Migration 0015 has all required SQL statements")


def test_migration_0016_exists():
    """Test migration 0016 exists."""
    migration_path = os.path.join(QIG_BACKEND_DIR, 'migrations', '0016_qfi_generation_view.sql')
    
    assert os.path.exists(migration_path), f"Migration 0016 not found: {migration_path}"
    assert os.path.isfile(migration_path), "Migration 0016 is not a file"
    
    print("✓ Migration 0016 exists")


def test_migration_0016_structure():
    """Test migration 0016 has required SQL statements."""
    migration_path = os.path.join(QIG_BACKEND_DIR, 'migrations', '0016_qfi_generation_view.sql')
    
    with open(migration_path, 'r') as f:
        content = f.read()
    
    # Check for transaction
    assert 'BEGIN;' in content, "Missing BEGIN transaction"
    assert 'COMMIT;' in content, "Missing COMMIT transaction"
    
    # Check for generation view
    assert 'CREATE OR REPLACE VIEW coordizer_vocabulary_generation_safe' in content, \
        "Missing coordizer_vocabulary_generation_safe view"
    
    # Check for QFI filtering
    assert 'qfi_score >= 0.01' in content, "Missing QFI threshold filter"
    assert "token_status = 'active'" in content, "Missing active status filter"
    assert 'basin_embedding IS NOT NULL' in content, "Missing basin filter"
    
    # Check for retrieval view
    assert 'CREATE OR REPLACE VIEW coordizer_vocabulary_retrieval' in content, \
        "Missing coordizer_vocabulary_retrieval view"
    
    # Check for indexes
    assert 'CREATE INDEX' in content, "Missing index creation"
    assert 'idx_coordizer_vocab_qfi_score' in content, "Missing QFI score index"
    assert 'idx_coordizer_vocab_generation' in content, "Missing generation index"
    
    # Check for helper functions
    assert 'CREATE OR REPLACE FUNCTION is_generation_eligible' in content, \
        "Missing is_generation_eligible function"
    assert 'CREATE OR REPLACE FUNCTION count_generation_safe_tokens' in content, \
        "Missing count_generation_safe_tokens function"
    assert 'CREATE OR REPLACE FUNCTION qfi_coverage_metrics' in content, \
        "Missing qfi_coverage_metrics function"
    
    print("✓ Migration 0016 has all required SQL statements")


def test_sql_syntax_basic():
    """Basic SQL syntax validation."""
    migrations = [
        'qig-backend/migrations/0015_special_symbols_qfi.sql',
        'qig-backend/migrations/0016_qfi_generation_view.sql',
    ]
    
    for migration_rel_path in migrations:
        migration_path = os.path.join(PROJECT_ROOT, migration_rel_path)
        
        with open(migration_path, 'r') as f:
            content = f.read()
        
        # Check for common SQL syntax issues
        assert content.count('BEGIN;') == content.count('COMMIT;'), \
            f"{migration_rel_path}: Mismatched BEGIN/COMMIT"
        
        # Check for balanced parentheses
        assert content.count('(') == content.count(')'), \
            f"{migration_rel_path}: Mismatched parentheses"
        
        # Check for balanced single quotes (roughly)
        single_quotes = [c for c in content if c == "'"]
        assert len(single_quotes) % 2 == 0, \
            f"{migration_rel_path}: Odd number of single quotes"
    
    print("✓ Basic SQL syntax validation passed")


def test_qfi_threshold_consistency():
    """Test QFI threshold is consistent across files."""
    files_to_check = [
        'qig-backend/qig_geometry/canonical_upsert.py',
        'qig-backend/scripts/quarantine_low_qfi_tokens.py',
        'qig-backend/migrations/0016_qfi_generation_view.sql',
    ]
    
    threshold_pattern = re.compile(r'0\.01\b')
    
    for file_rel_path in files_to_check:
        file_path = os.path.join(PROJECT_ROOT, file_rel_path)
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        matches = threshold_pattern.findall(content)
        assert len(matches) > 0, f"{file_rel_path}: Missing QFI threshold 0.01"
    
    print("✓ QFI threshold (0.01) is consistent across files")


def run_all_tests():
    """Run all integration tests."""
    print("Running QFI Integrity Gate Integration Tests...\n")
    
    tests = [
        test_quarantine_script_exists,
        test_quarantine_script_structure,
        test_migration_0015_exists,
        test_migration_0015_structure,
        test_migration_0016_exists,
        test_migration_0016_structure,
        test_sql_syntax_basic,
        test_qfi_threshold_consistency,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: ERROR - {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Integration Test Results: {passed}/{len(tests)} passed, {failed} failed")
    print(f"{'='*60}")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
