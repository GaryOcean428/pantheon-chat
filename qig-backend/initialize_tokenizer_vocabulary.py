#!/usr/bin/env python3
"""
Initialize Tokenizer Vocabulary

Ensures the tokenizer_vocabulary table is populated with real English words.
This is a convenience wrapper around populate_tokenizer_vocabulary.py.

Usage:
    python initialize_tokenizer_vocabulary.py

The script will:
1. Check if tokenizer_vocabulary has enough real words
2. If not, populate it with BIP39 + common + domain words
3. Each word gets a deterministic 64D basin embedding
"""

import os
import sys
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

try:
    import psycopg2
except ImportError:
    print("Installing psycopg2-binary...")
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'psycopg2-binary', '-q'])
    import psycopg2

# Constants
MIN_VOCAB_SIZE = 500  # Minimum viable vocabulary
BASIN_DIM = 64


def check_vocabulary_status() -> tuple:
    """Check if vocabulary needs initialization.
    
    Returns (total_count, real_word_count, needs_init)
    """
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print("ERROR: DATABASE_URL not set")
        return 0, 0, True
    
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        # Check total
        cur.execute("SELECT COUNT(*) FROM tokenizer_vocabulary")
        total = cur.fetchone()[0]
        
        # Check real words (not BPE/byte tokens)
        cur.execute("""
            SELECT COUNT(*) FROM tokenizer_vocabulary
            WHERE source_type IN ('bip39', 'common', 'domain', 'base', 'learned')
              AND LENGTH(token) >= 2
              AND token NOT LIKE '<byte_%%>'
        """)
        real_words = cur.fetchone()[0]
        
        conn.close()
        
        needs_init = real_words < MIN_VOCAB_SIZE
        return total, real_words, needs_init
        
    except Exception as e:
        print(f"Database check failed: {e}")
        return 0, 0, True


def main():
    print("=" * 50)
    print("Tokenizer Vocabulary Initialization")
    print("=" * 50)
    
    total, real_words, needs_init = check_vocabulary_status()
    
    print(f"\nCurrent status:")
    print(f"  Total tokens: {total}")
    print(f"  Real words: {real_words}")
    print(f"  Minimum required: {MIN_VOCAB_SIZE}")
    
    if not needs_init:
        print(f"\n✅ Vocabulary is already initialized with {real_words} real words")
        return
    
    print(f"\n⚠️  Vocabulary needs initialization (only {real_words} real words)")
    print("Running populate_tokenizer_vocabulary.py...")
    print()
    
    # Run the population script
    import subprocess
    result = subprocess.run(
        [sys.executable, 'populate_tokenizer_vocabulary.py'],
        cwd=Path(__file__).parent
    )
    
    if result.returncode == 0:
        # Verify
        total, real_words, needs_init = check_vocabulary_status()
        if not needs_init:
            print(f"\n✅ Vocabulary initialized successfully with {real_words} real words")
        else:
            print(f"\n⚠️  Population completed but only {real_words} words. May need manual review.")
    else:
        print(f"\n❌ Population failed with exit code {result.returncode}")
        sys.exit(1)


if __name__ == '__main__':
    main()
