#!/usr/bin/env python3
"""
SQL Schema Naming Convention Validator

Validates SQL migrations and schema files against naming conventions:
- Tables: snake_case, plural nouns
- Columns: snake_case, singular nouns, proper prefixes (is_, has_, _at)
- Indexes: idx_table_column, uniq_table_column, fk_table_reftable
- No reserved keywords as column names

Usage:
    python scripts/validate-sql-naming.py [file1.sql file2.sql ...]
    python scripts/validate-sql-naming.py --check-all

Exit codes:
    0 - All validations passed
    1 - Violations found
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import argparse

# Reserved PostgreSQL keywords that should not be used as column names
RESERVED_KEYWORDS = {
    'user', 'order', 'group', 'table', 'select', 'from', 'where', 'join',
    'insert', 'update', 'delete', 'create', 'drop', 'alter', 'index',
    'primary', 'foreign', 'key', 'constraint', 'default', 'null', 'not',
    'and', 'or', 'in', 'between', 'like', 'limit', 'offset', 'having',
}

# Common singular -> plural patterns (for validation)
COMMON_PLURALS = {
    'session': 'sessions',
    'user': 'users',
    'checkpoint': 'checkpoints',
    'observation': 'observations',
    'vocabulary': 'vocabularies',
    'word': 'words',
    'kernel': 'kernels',
    'merge': 'merges',
    'rule': 'rules',
}

def extract_table_names(sql: str) -> List[str]:
    """Extract table names from CREATE TABLE statements."""
    pattern = r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"?([a-zA-Z_][a-zA-Z0-9_]*)"?'
    matches = re.findall(pattern, sql, re.IGNORECASE)
    return matches

def extract_column_names(sql: str, table_name: str) -> List[str]:
    """Extract column names from CREATE TABLE statement."""
    # Find the CREATE TABLE block for this specific table
    pattern = rf'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"?{re.escape(table_name)}"?\s*\((.*?)\);'
    match = re.search(pattern, sql, re.IGNORECASE | re.DOTALL)
    
    if not match:
        return []
    
    table_body = match.group(1)
    
    # Extract column definitions (before constraints)
    columns = []
    for line in table_body.split('\n'):
        line = line.strip()
        if not line or line.startswith('CONSTRAINT') or line.startswith('PRIMARY') or line.startswith('FOREIGN'):
            continue
        if line.startswith('--'):
            continue
            
        # Extract column name (first word in the line)
        col_match = re.match(r'"?([a-zA-Z_][a-zA-Z0-9_]*)"?', line)
        if col_match:
            columns.append(col_match.group(1))
    
    return columns

def extract_index_names(sql: str) -> List[str]:
    """Extract index names from CREATE INDEX statements."""
    pattern = r'CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?"?([a-zA-Z_][a-zA-Z0-9_]*)"?'
    matches = re.findall(pattern, sql, re.IGNORECASE)
    return matches

def is_snake_case(name: str) -> bool:
    """Check if name follows snake_case convention."""
    return bool(re.match(r'^[a-z][a-z0-9_]*$', name))

def is_likely_plural(name: str) -> bool:
    """Check if table name is likely plural."""
    # Skip temp_ prefixed tables
    if name.startswith('temp_'):
        name = name[5:]
    
    # Check common patterns
    if name.endswith('s') or name.endswith('es') or name.endswith('ies'):
        return True
    
    # Check against known singular forms
    for singular, plural in COMMON_PLURALS.items():
        if name == plural:
            return True
        if name == singular:
            return False
    
    return True  # Default to accepting

def validate_table_name(name: str) -> List[str]:
    """Validate table name against conventions."""
    violations = []
    
    if not is_snake_case(name):
        violations.append(f"Table '{name}' is not snake_case")
    
    if name.startswith('temp_'):
        # Temporary tables are OK
        pass
    elif not is_likely_plural(name):
        violations.append(f"Table '{name}' should be plural (e.g., '{name}s' or add to COMMON_PLURALS)")
    
    return violations

def validate_column_name(name: str, table: str) -> List[str]:
    """Validate column name against conventions."""
    violations = []
    
    if not is_snake_case(name):
        violations.append(f"Column '{table}.{name}' is not snake_case")
    
    # Check for reserved keywords
    if name.lower() in RESERVED_KEYWORDS:
        violations.append(f"Column '{table}.{name}' uses reserved keyword '{name}' - use '{name}_id' or similar")
    
    # Boolean columns should have is_/has_/can_ prefix (but only check if starts with these)
    if name.startswith('is') and not name.startswith('is_'):
        if len(name) > 2 and name[2].isupper():  # e.g., isActive
            violations.append(f"Column '{table}.{name}' should use is_ prefix with underscore")
    
    # Timestamp columns should end with _at (but only certain patterns)
    timestamp_patterns = ['created', 'updated', 'deleted', 'started', 'ended', 'finished']
    for pattern in timestamp_patterns:
        if name == pattern and not name.endswith('_at'):
            violations.append(f"Column '{table}.{name}' should end with '_at' (e.g., '{name}_at')")
    
    return violations

def validate_index_name(name: str) -> List[str]:
    """Validate index name against conventions."""
    violations = []
    
    if not is_snake_case(name):
        violations.append(f"Index '{name}' is not snake_case")
    
    # Check for proper prefix
    valid_prefixes = ['idx_', 'uniq_', 'fk_']
    if not any(name.startswith(prefix) for prefix in valid_prefixes):
        violations.append(f"Index '{name}' should start with idx_/uniq_/fk_")
    
    return violations

def validate_sql_file(filepath: Path) -> Tuple[bool, List[str]]:
    """Validate a SQL file against naming conventions."""
    try:
        sql = filepath.read_text()
    except Exception as e:
        return False, [f"Error reading {filepath}: {e}"]
    
    violations = []
    
    # Validate table names
    tables = extract_table_names(sql)
    for table in tables:
        table_violations = validate_table_name(table)
        violations.extend(table_violations)
        
        # Validate column names for this table
        columns = extract_column_names(sql, table)
        for column in columns:
            column_violations = validate_column_name(column, table)
            violations.extend(column_violations)
    
    # Validate index names
    indexes = extract_index_names(sql)
    for index in indexes:
        index_violations = validate_index_name(index)
        violations.extend(index_violations)
    
    return len(violations) == 0, violations

def main():
    parser = argparse.ArgumentParser(description='Validate SQL naming conventions')
    parser.add_argument('files', nargs='*', help='SQL files to validate')
    parser.add_argument('--check-all', action='store_true', help='Check all SQL files in migrations/')
    parser.add_argument('--verbose', action='store_true', help='Show all files checked')
    
    args = parser.parse_args()
    
    files_to_check = []
    
    if args.check_all:
        migrations_dir = Path('migrations')
        if migrations_dir.exists():
            files_to_check.extend(migrations_dir.glob('*.sql'))
        
        qig_migrations = Path('qig-backend/migrations')
        if qig_migrations.exists():
            files_to_check.extend(qig_migrations.glob('*.sql'))
    else:
        files_to_check = [Path(f) for f in args.files]
    
    if not files_to_check:
        print("No SQL files to check")
        return 0
    
    all_passed = True
    total_violations = 0
    
    for filepath in files_to_check:
        if not filepath.exists():
            print(f"❌ File not found: {filepath}")
            all_passed = False
            continue
        
        passed, violations = validate_sql_file(filepath)
        
        if passed:
            if args.verbose:
                print(f"✅ {filepath}: OK")
        else:
            all_passed = False
            print(f"❌ {filepath}:")
            for violation in violations:
                print(f"   {violation}")
                total_violations += 1
    
    print()
    if all_passed:
        print(f"✅ All {len(files_to_check)} SQL files follow naming conventions")
        return 0
    else:
        print(f"❌ Found {total_violations} naming violations in {len([f for f in files_to_check if not validate_sql_file(f)[0]])} files")
        print()
        print("Fix violations or update COMMON_PLURALS in scripts/validate-sql-naming.py")
        return 1

if __name__ == '__main__':
    sys.exit(main())
