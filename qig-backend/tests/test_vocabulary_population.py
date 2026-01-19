#!/usr/bin/env python3
"""Test script to verify coordizer_vocabulary population.

Run after: python populate_coordizer_vocabulary.py
"""

import os
import sys

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path='../.env')
except ImportError:
    pass


def test_vocabulary():
    """Test that vocabulary is properly populated and coordizer works."""
    print("="*60)
    print("VOCABULARY POPULATION TEST")
    print("="*60)
    
    # Check database connection
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ DATABASE_URL not set")
        return False
    print("✅ DATABASE_URL is set")
    
    # Check table contents directly
    try:
        import psycopg2
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        # Count total tokens
        cur.execute("SELECT COUNT(*) FROM coordizer_vocabulary")
        total = cur.fetchone()[0]
        print(f"\nTotal tokens in coordizer_vocabulary: {total}")
        
        # Count by source_type
        cur.execute("""
            SELECT source_type, COUNT(*) 
            FROM coordizer_vocabulary 
            GROUP BY source_type
        """)
        for source_type, count in cur.fetchall():
            print(f"  - {source_type}: {count}")
        
        # Check BIP39 words specifically
        cur.execute("SELECT COUNT(*) FROM coordizer_vocabulary WHERE source_type = 'bip39'")
        bip39_count = cur.fetchone()[0]
        
        if bip39_count == 0:
            print("\n❌ No BIP39 words found!")
            print("   Run: python populate_coordizer_vocabulary.py")
            conn.close()
            return False
        
        print(f"\n✅ Found {bip39_count} BIP39 words")
        
        # Sample some words
        cur.execute("""
            SELECT token, phi_score 
            FROM coordizer_vocabulary 
            WHERE source_type = 'bip39'
            ORDER BY phi_score DESC
            LIMIT 10
        """)
        print("\nTop 10 BIP39 words by phi:")
        for word, phi in cur.fetchall():
            print(f"  {word}: phi={phi:.4f}")
        
        conn.close()
        
    except ImportError:
        print("❌ psycopg2 not installed")
        return False
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False
    
    # Test coordizer loading
    print("\n" + "-"*40)
    print("Testing PostgresCoordizer...")
    
    try:
        from coordizers import create_coordizer_from_pg
        coordizer = create_coordizer_from_pg()
        
        print(f"Vocabulary size: {coordizer.vocab_size}")
        print(f"BIP39 words: {len(coordizer.bip39_words)}")
        print(f"Word tokens: {len(coordizer.word_tokens)}")
        
        if coordizer.vocab_size == 0:
            print("\n❌ Coordizer failed to load vocabulary")
            return False
        
        # Test encoding
        test_phrase = "hello world"
        basin = coordizer.encode(test_phrase)
        print(f"\nEncoded '{test_phrase}' to basin (norm={basin.sum():.4f})")
        
        # Test decoding
        decoded = coordizer.decode(basin, top_k=5)
        print(f"Decoded top 5 tokens:")
        for token, similarity in decoded:
            print(f"  {token}: {similarity:.4f}")
        
        # Test similar words
        if 'hello' in coordizer.vocab:
            similar = coordizer.find_similar_tokens('hello', top_k=5)
            print(f"\nWords similar to 'hello':")
            for word, sim in similar:
                print(f"  {word}: {sim:.4f}")
        
        # Test random words
        random_words = coordizer.get_random_words(12)
        print(f"\nRandom 12-word phrase: {' '.join(random_words)}")
        
        print("\n✅ Coordizer working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Coordizer error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_vocabulary()
    print("\n" + "="*60)
    if success:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ TESTS FAILED")
        print("\nTo fix, run:")
        print("  cd qig-backend && python populate_coordizer_vocabulary.py")
    print("="*60)
    sys.exit(0 if success else 1)
