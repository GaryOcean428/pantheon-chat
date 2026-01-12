#!/usr/bin/env python3
"""
Database Validation Script

Performs comprehensive database validation:
1. Singleton table cardinality (exactly 1 row required)
2. Core table minimum row counts
3. Excessive NULL detection (>50% threshold)
4. Default value detection (>80% at default)

Severity Levels:
- CRITICAL: Blocks operation
- WARNING: Needs attention
- INFO: Informational

Exit codes:
- 0: All validations passed
- 1: Critical issues found
- 2: Warnings found (no critical)
"""

import os
import sys
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("ERROR: psycopg2 not available. Install with: pip install psycopg2-binary")
    sys.exit(1)


class Severity(Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"
    OK = "OK"


@dataclass
class ValidationResult:
    severity: Severity
    message: str
    details: Optional[str] = None


SINGLETON_TABLES = [
    "ocean_quantum_state",
    "near_miss_adaptive_state",
    "auto_cycle_state",
]

CORE_TABLE_MINIMUMS = {
    "tokenizer_vocabulary": 10000,
    "learned_words": 5000,
}

NULL_THRESHOLD = 0.50
DEFAULT_THRESHOLD = 0.80

COLUMN_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "ocean_quantum_state": {
        "entropy": 256.0,
        "initial_entropy": 256.0,
        "total_probability": 1.0,
        "measurement_count": 0,
        "successful_measurements": 0,
        "status": "searching",
    },
    "near_miss_adaptive_state": {
        "hot_threshold": 0.7,
        "warm_threshold": 0.55,
        "cool_threshold": 0.4,
        "distribution_size": 0,
    },
    "auto_cycle_state": {
        "enabled": False,
        "current_index": 0,
        "total_cycles": 0,
        "consecutive_zero_pass_sessions": 0,
    },
    "tokenizer_vocabulary": {
        "frequency": 1,
        "is_bpe_merge": False,
    },
    "learned_words": {
        "frequency": 1,
        "is_integrated": False,
    },
    "vocabulary_observations": {
        "type": "phrase",
        "is_real_word": False,
        "frequency": 1,
        "avg_phi": 0.0,
        "max_phi": 0.0,
        "efficiency_gain": 0.0,
        "is_integrated": False,
        "is_bip39_word": False,
    },
    "consciousness_checkpoints": {
        "is_hot": False,
    },
    "war_history": {
        "passes_completed": 0,
        "mode": "defensive",
        "status": "active",
    },
    "geodesic_paths": {
        "fisher_length": 0.0,
        "success": False,
    },
    "resonance_points": {
        "stability": 0.0,
        "probe_count": 0,
    },
}

TABLES_TO_CHECK = [
    "ocean_quantum_state",
    "near_miss_adaptive_state",
    "auto_cycle_state",
    "tokenizer_vocabulary",
    "learned_words",
    "vocabulary_observations",
    "consciousness_checkpoints",
    "geodesic_paths",
    "resonance_points",
    "negative_knowledge",
    "near_miss_clusters",
    "false_pattern_classes",
    "era_exclusions",
    "war_history",
    "synthesis_consensus",
    "ocean_excluded_regions",
    "tokenizer_metadata",
]


def get_connection():
    """Get database connection from DATABASE_URL environment variable."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable not set")
    conn = psycopg2.connect(database_url)
    conn.autocommit = True
    return conn


def table_exists(cur, table_name: str) -> bool:
    """Check if a table exists in the database."""
    cur.execute(
        """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = %s
        )
        """,
        (table_name,),
    )
    return cur.fetchone()[0]


def get_table_row_count(cur, table_name: str) -> int:
    """Get the row count for a table."""
    cur.execute(f'SELECT COUNT(*) FROM "{table_name}"')
    return cur.fetchone()[0]


def get_column_info(cur, table_name: str) -> List[Dict[str, Any]]:
    """Get column information for a table."""
    cur.execute(
        """
        SELECT column_name, data_type, column_default, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position
        """,
        (table_name,),
    )
    return [
        {
            "name": row[0],
            "type": row[1],
            "default": row[2],
            "nullable": row[3] == "YES",
        }
        for row in cur.fetchall()
    ]


def check_singleton_tables(cur) -> List[ValidationResult]:
    """Check that singleton tables have exactly 1 row."""
    results = []

    for table in SINGLETON_TABLES:
        if not table_exists(cur, table):
            results.append(
                ValidationResult(
                    Severity.CRITICAL,
                    f"{table}: table does not exist",
                )
            )
            continue

        count = get_table_row_count(cur, table)

        if count == 0:
            results.append(
                ValidationResult(
                    Severity.CRITICAL,
                    f"{table}: missing singleton row (0 rows)",
                )
            )
        elif count == 1:
            results.append(
                ValidationResult(
                    Severity.OK,
                    f"{table}: 1 row (OK)",
                )
            )
        else:
            results.append(
                ValidationResult(
                    Severity.WARNING,
                    f"{table}: {count} rows (expected exactly 1)",
                )
            )

    return results


def check_core_table_minimums(cur) -> List[ValidationResult]:
    """Check that core tables meet minimum row requirements."""
    results = []

    for table, minimum in CORE_TABLE_MINIMUMS.items():
        if not table_exists(cur, table):
            results.append(
                ValidationResult(
                    Severity.WARNING,
                    f"{table}: table does not exist",
                )
            )
            continue

        count = get_table_row_count(cur, table)

        if count >= minimum:
            results.append(
                ValidationResult(
                    Severity.OK,
                    f"{table}: {count:,} rows (minimum: {minimum:,})",
                )
            )
        else:
            results.append(
                ValidationResult(
                    Severity.WARNING,
                    f"{table}: {count:,} rows (below minimum of {minimum:,})",
                )
            )

    return results


def check_null_percentages(cur) -> List[ValidationResult]:
    """Check all tables for columns with >50% NULL values."""
    results = []

    for table in TABLES_TO_CHECK:
        if not table_exists(cur, table):
            continue

        row_count = get_table_row_count(cur, table)
        if row_count == 0:
            continue

        columns = get_column_info(cur, table)

        for col in columns:
            if not col["nullable"]:
                continue

            col_name = col["name"]

            try:
                cur.execute(
                    f'SELECT COUNT(*) FROM "{table}" WHERE "{col_name}" IS NULL'
                )
                null_count = cur.fetchone()[0]
                null_pct = null_count / row_count

                if null_pct > NULL_THRESHOLD:
                    results.append(
                        ValidationResult(
                            Severity.WARNING,
                            f"{table}.{col_name}: {null_pct:.0%} NULL",
                            f"{null_count:,} of {row_count:,} rows",
                        )
                    )
            except Exception as e:
                pass

    return results


def check_default_percentages(cur) -> List[ValidationResult]:
    """Check for columns where >80% of values equal the default."""
    results = []

    for table, defaults in COLUMN_DEFAULTS.items():
        if not table_exists(cur, table):
            continue

        row_count = get_table_row_count(cur, table)
        if row_count == 0:
            continue

        for col_name, default_value in defaults.items():
            try:
                if default_value is None:
                    cur.execute(
                        f'SELECT COUNT(*) FROM "{table}" WHERE "{col_name}" IS NULL'
                    )
                elif isinstance(default_value, bool):
                    cur.execute(
                        f'SELECT COUNT(*) FROM "{table}" WHERE "{col_name}" = %s',
                        (default_value,),
                    )
                elif isinstance(default_value, (int, float)):
                    cur.execute(
                        f'SELECT COUNT(*) FROM "{table}" WHERE "{col_name}" = %s',
                        (default_value,),
                    )
                elif isinstance(default_value, str):
                    cur.execute(
                        f'SELECT COUNT(*) FROM "{table}" WHERE "{col_name}" = %s',
                        (default_value,),
                    )
                else:
                    continue

                at_default = cur.fetchone()[0]
                default_pct = at_default / row_count

                if default_pct > DEFAULT_THRESHOLD:
                    results.append(
                        ValidationResult(
                            Severity.INFO,
                            f"{table}.{col_name}: {default_pct:.0%} at default",
                            f"default={default_value!r}, {at_default:,} of {row_count:,} rows",
                        )
                    )
            except Exception as e:
                pass

    return results


def format_result(result: ValidationResult) -> str:
    """Format a validation result for output."""
    if result.severity == Severity.OK:
        return f"✓ {result.message}"
    elif result.severity == Severity.CRITICAL:
        return f"✗ CRITICAL: {result.message}"
    elif result.severity == Severity.WARNING:
        return f"⚠ WARNING: {result.message}"
    else:
        return f"ℹ INFO: {result.message}"


def main():
    """Main entry point for database validation."""
    print("=" * 60)
    print("DATABASE VALIDATION")
    print("=" * 60)
    print()

    all_results: List[ValidationResult] = []

    try:
        conn = get_connection()
        cur = conn.cursor()

        print("--- Singleton Table Cardinality ---")
        singleton_results = check_singleton_tables(cur)
        all_results.extend(singleton_results)
        for r in singleton_results:
            print(format_result(r))
        print()

        print("--- Core Table Minimums ---")
        minimum_results = check_core_table_minimums(cur)
        all_results.extend(minimum_results)
        for r in minimum_results:
            print(format_result(r))
        print()

        print("--- Excessive NULL Detection (>50%) ---")
        null_results = check_null_percentages(cur)
        all_results.extend(null_results)
        if null_results:
            for r in null_results:
                print(format_result(r))
        else:
            print("✓ No columns with >50% NULL values")
        print()

        print("--- Default Value Detection (>80%) ---")
        default_results = check_default_percentages(cur)
        all_results.extend(default_results)
        if default_results:
            for r in default_results:
                print(format_result(r))
        else:
            print("✓ No columns with >80% default values")
        print()

        cur.close()
        conn.close()

    except Exception as e:
        print(f"✗ CRITICAL: Database connection failed: {e}")
        sys.exit(1)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    critical_count = sum(1 for r in all_results if r.severity == Severity.CRITICAL)
    warning_count = sum(1 for r in all_results if r.severity == Severity.WARNING)
    info_count = sum(1 for r in all_results if r.severity == Severity.INFO)
    ok_count = sum(1 for r in all_results if r.severity == Severity.OK)

    print(f"  ✓ OK:       {ok_count}")
    print(f"  ℹ INFO:     {info_count}")
    print(f"  ⚠ WARNING:  {warning_count}")
    print(f"  ✗ CRITICAL: {critical_count}")
    print()

    if critical_count > 0:
        print("❌ Validation FAILED - Critical issues found")
        sys.exit(1)
    elif warning_count > 0:
        print("⚠️ Validation completed with warnings")
        sys.exit(2)
    else:
        print("✅ Validation PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
