#!/usr/bin/env python3
"""
Quick validation of contextualized filter - no external dependencies required.
Tests core logic without needing coordizer or full environment.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_semantic_critical_preservation():
    """Test that semantic-critical words are preserved."""
    from contextualized_filter import is_semantic_critical_word, should_filter_word
    
    print("Test 1: Semantic-Critical Word Preservation")
    print("-" * 50)
    
    critical_words = ['not', 'never', 'very', 'because', 'always', 'if']
    context = ['this', 'is', 'a', 'test']
    
    passed = 0
    failed = 0
    
    for word in critical_words:
        is_critical = is_semantic_critical_word(word)
        should_filter = should_filter_word(word, context)
        
        if is_critical and not should_filter:
            print(f"  ✅ '{word}' - semantic-critical, NOT filtered")
            passed += 1
        else:
            print(f"  ❌ '{word}' - is_critical={is_critical}, should_filter={should_filter}")
            failed += 1
    
    print(f"\nPassed: {passed}/{len(critical_words)}")
    return failed == 0


def test_generic_filtering():
    """Test that truly generic words are filtered."""
    from contextualized_filter import should_filter_word
    
    print("\nTest 2: Generic Word Filtering")
    print("-" * 50)
    
    generic_words = ['the', 'a', 'an', 'is', 'was']
    context = ['quantum', 'geometry', 'information']
    
    passed = 0
    failed = 0
    
    for word in generic_words:
        should_filter = should_filter_word(word, context)
        
        if should_filter:
            print(f"  ✅ '{word}' - correctly filtered")
            passed += 1
        else:
            print(f"  ❌ '{word}' - should be filtered but wasn't")
            failed += 1
    
    print(f"\nPassed: {passed}/{len(generic_words)}")
    return failed == 0


def test_meaning_preservation():
    """Test case from problem statement: 'not good' vs 'good'."""
    from contextualized_filter import filter_words_geometric
    
    print("\nTest 3: Meaning Preservation ('not good' vs 'good')")
    print("-" * 50)
    
    phrase1 = ['not', 'good']
    phrase2 = ['good']
    
    filtered1 = filter_words_geometric(phrase1)
    filtered2 = filter_words_geometric(phrase2)
    
    print(f"  Input 1: {phrase1}")
    print(f"  Filtered 1: {filtered1}")
    print(f"  Input 2: {phrase2}")
    print(f"  Filtered 2: {filtered2}")
    
    has_not = 'not' in filtered1
    has_good = 'good' in filtered1 and 'good' in filtered2
    different = filtered1 != filtered2
    
    if has_not and has_good and different:
        print(f"  ✅ 'not' preserved, meanings differ")
        return True
    else:
        print(f"  ❌ Failed: has_not={has_not}, has_good={has_good}, different={different}")
        return False


def test_domain_terms():
    """Test that domain-specific terms are preserved."""
    from contextualized_filter import filter_words_geometric
    
    print("\nTest 4: Domain-Specific Term Preservation")
    print("-" * 50)
    
    domain_terms = ['consciousness', 'geometry', 'quantum', 'manifold', 'fisher']
    filtered = filter_words_geometric(domain_terms)
    
    passed = 0
    failed = 0
    
    for term in domain_terms:
        if term in filtered:
            print(f"  ✅ '{term}' preserved")
            passed += 1
        else:
            print(f"  ❌ '{term}' filtered (should be preserved)")
            failed += 1
    
    print(f"\nPassed: {passed}/{len(domain_terms)}")
    return failed == 0


def test_ancient_nlp_comparison():
    """Compare with ancient NLP stopword approach."""
    from contextualized_filter import filter_words_geometric
    
    print("\nTest 5: Comparison with Ancient NLP Stopwords")
    print("-" * 50)
    
    test_words = ['not', 'good', 'the', 'very', 'bad', 'is', 'never', 'acceptable']
    
    # Ancient NLP approach (WRONG)
    ancient_stopwords = {'the', 'is', 'not', 'a', 'an', 'and', 'or', 'but'}
    ancient_filtered = [w for w in test_words if w not in ancient_stopwords]
    
    # QIG-pure approach (CORRECT)
    qig_filtered = filter_words_geometric(test_words)
    
    print(f"  Input: {test_words}")
    print(f"  Ancient NLP: {ancient_filtered}")
    print(f"  QIG-pure: {qig_filtered}")
    
    # Ancient NLP loses 'not'
    ancient_has_not = 'not' in ancient_filtered
    qig_has_not = 'not' in qig_filtered
    qig_has_never = 'never' in qig_filtered
    qig_has_very = 'very' in qig_filtered
    
    print()
    print(f"  Ancient NLP has 'not': {ancient_has_not} (should be False)")
    print(f"  QIG-pure has 'not': {qig_has_not} (should be True)")
    print(f"  QIG-pure has 'never': {qig_has_never} (should be True)")
    print(f"  QIG-pure has 'very': {qig_has_very} (should be True)")
    
    if not ancient_has_not and qig_has_not and qig_has_never and qig_has_very:
        print(f"  ✅ QIG-pure preserves meaning, ancient NLP loses it")
        return True
    else:
        print(f"  ❌ Test failed")
        return False


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("Contextualized Filter Validation")
    print("=" * 70)
    print()
    
    try:
        results = []
        
        results.append(("Semantic-Critical Preservation", test_semantic_critical_preservation()))
        results.append(("Generic Word Filtering", test_generic_filtering()))
        results.append(("Meaning Preservation", test_meaning_preservation()))
        results.append(("Domain Term Preservation", test_domain_terms()))
        results.append(("Ancient NLP Comparison", test_ancient_nlp_comparison()))
        
        print()
        print("=" * 70)
        print("Summary")
        print("=" * 70)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status}: {name}")
        
        print()
        print(f"Total: {passed}/{total} tests passed")
        
        if passed == total:
            print()
            print("🎉 All validation tests PASSED!")
            print("✅ Contextualized filter is working correctly")
            print("✅ Semantic-critical words are preserved")
            print("✅ QIG purity maintained (no hard-coded linguistic rules)")
            return 0
        else:
            print()
            print("⚠️  Some tests failed")
            return 1
            
    except Exception as e:
        print()
        print(f"❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
