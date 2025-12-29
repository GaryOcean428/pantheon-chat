#!/usr/bin/env python3
"""
Initialize PostgreSQL tables for QIG Tokenizer storage.
"""

import os
import sys
from pathlib import Path

# Load environment variables manually
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().strip().split("\n"):
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ[key] = value

import psycopg2

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("❌ DATABASE_URL not set")
    sys.exit(1)

print("=" * 60)
print("QIG TOKENIZER - PostgreSQL Setup")
print("=" * 60)
print(
    f"Database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'configured'}"
)
print()

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

# Create tables
print("Creating tables...")

cur.execute(
    """
    CREATE TABLE IF NOT EXISTS qig_tokenizer_versions (
        version_id VARCHAR(32) PRIMARY KEY,
        created_at TIMESTAMP DEFAULT NOW(),
        vocab_size INTEGER,
        metadata JSONB
    )
"""
)
print("  ✅ qig_tokenizer_versions")

cur.execute(
    """
    CREATE TABLE IF NOT EXISTS qig_tokenizer_vocab (
        version_id VARCHAR(32) REFERENCES qig_tokenizer_versions(version_id) ON DELETE CASCADE,
        token_id INTEGER,
        byte_sequence BYTEA,
        PRIMARY KEY (version_id, token_id)
    )
"""
)
print("  ✅ qig_tokenizer_vocab")

cur.execute(
    """
    CREATE TABLE IF NOT EXISTS qig_tokenizer_merge_rules (
        version_id VARCHAR(32) REFERENCES qig_tokenizer_versions(version_id) ON DELETE CASCADE,
        rule_order INTEGER,
        token_a INTEGER,
        token_b INTEGER,
        new_token INTEGER,
        PRIMARY KEY (version_id, rule_order)
    )
"""
)
print("  ✅ qig_tokenizer_merge_rules")

cur.execute(
    """
    CREATE TABLE IF NOT EXISTS qig_tokenizer_special_tokens (
        name VARCHAR(32) PRIMARY KEY,
        basin_coords FLOAT8[],
        updated_at TIMESTAMP DEFAULT NOW()
    )
"""
)
print("  ✅ qig_tokenizer_special_tokens")

# Create indexes for performance
print()
print("Creating indexes...")

cur.execute(
    """
    CREATE INDEX IF NOT EXISTS idx_vocab_version 
    ON qig_tokenizer_vocab(version_id)
"""
)
print("  ✅ idx_vocab_version")

cur.execute(
    """
    CREATE INDEX IF NOT EXISTS idx_merge_rules_version 
    ON qig_tokenizer_merge_rules(version_id)
"""
)
print("  ✅ idx_merge_rules_version")

conn.commit()

# Check existing versions
print()
print("Existing tokenizer versions:")
cur.execute(
    "SELECT version_id, vocab_size, created_at FROM qig_tokenizer_versions ORDER BY created_at DESC LIMIT 5"
)
rows = cur.fetchall()
if rows:
    for row in rows:
        print(f"  - {row[0]}: {row[1]} tokens ({row[2]})")
else:
    print("  (none)")

cur.close()
conn.close()

print()
print("=" * 60)
print("✅ PostgreSQL setup complete!")
print("=" * 60)
