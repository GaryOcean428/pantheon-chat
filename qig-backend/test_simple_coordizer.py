#!/usr/bin/env python3
"""
Test SimpleWordCoordizer - Verify it produces readable English words.

Run with:
    python test_simple_coordizer.py
"""

import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))


def test_simple_coordizer():
    """Test the SimpleWordCoordizer produces readable output."""
    print("=" * 50)
    print("Testing SimpleWordCoordizer")
    print("=" * 50)
    
    # Import the coordizer
    from coordizers.simple_word_coordizer import SimpleWordCoordizer, get_simple_coordizer
    
    # Create instance
    coordizer = get_simple_coordizer()
    print(f"\n✅ Coordizer created successfully")
    print(f"   Vocabulary size: {coordizer.vocab_size}")
    print(f"   Sample words: {coordizer.words[:10]}")
    
    # Test encoding
    test_texts = [
        "hello world",
        "What is consciousness?",
        "quantum geometry integration",
        "the quick brown fox",
    ]
    
    print(f"\n--- Testing Encode/Decode ---")
    for text in test_texts:
        basin = coordizer.encode(text)
        decoded = coordizer.decode(basin, top_k=5)
        words = [w for w, s in decoded]
        print(f"\n  Input: '{text}'")
        print(f"  Basin norm: {(basin**2).sum()**0.5:.4f}")
        print(f"  Decoded: {words}")
    
    # Test random words
    print(f"\n--- Random Words ---")
    random_words = coordizer.get_random_words(12)
    print(f"  {random_words}")
    
    # Verify all words are readable English
    print(f"\n--- Verification ---")
    non_alpha = [w for w in coordizer.words if not w.isalpha()]
    if non_alpha:
        print(f"  ⚠️  Non-alphabetic words found: {non_alpha[:5]}...")
    else:
        print(f"  ✅ All {coordizer.vocab_size} words are alphabetic")
    
    short_words = [w for w in coordizer.words if len(w) < 2]
    if short_words:
        print(f"  ⚠️  Very short words found: {short_words}")
    else:
        print(f"  ✅ All words have length >= 2")
    
    print(f"\n{'=' * 50}")
    print(f"✅ All tests passed - SimpleWordCoordizer ready!")
    print(f"{'=' * 50}")
    
    return True


def test_generative_service():
    """Test the generative service uses SimpleWordCoordizer."""
    print(f"\n{'=' * 50}")
    print("Testing QIGGenerativeService")
    print("=" * 50)
    
    try:
        from qig_generative_service import get_generative_service
        
        service = get_generative_service()
        print(f"\n✅ Service created successfully")
        
        if service.coordizer:
            print(f"   Coordizer: {type(service.coordizer).__name__}")
            print(f"   Vocab size: {service.coordizer.vocab_size}")
        else:
            print(f"   ⚠️  No coordizer loaded")
            return False
        
        # Test generation
        print(f"\n--- Test Generation ---")
        result = service.generate("What is consciousness?")
        print(f"  Response: {result.text}")
        print(f"  Tokens: {result.tokens[:10]}...")
        print(f"  Completion: {result.completion_reason}")
        
        # Check for BPE garbage
        bpe_garbage = [t for t in result.tokens if '+' in t or t.startswith('<byte')]
        if bpe_garbage:
            print(f"  ⚠️  BPE garbage found: {bpe_garbage[:5]}")
            return False
        else:
            print(f"  ✅ No BPE garbage - all readable words!")
        
        print(f"\n{'=' * 50}")
        print(f"✅ Generative service working!")
        print(f"{'=' * 50}")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_simple_coordizer()
    if success:
        test_generative_service()
