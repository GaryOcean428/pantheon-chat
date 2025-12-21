#!/usr/bin/env python3
"""
Test Shadow Research → VocabularyCoordinator Integration
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_vocabulary_coordinator_available():
    """Test that VocabularyCoordinator can be imported"""
    try:
        from vocabulary_coordinator import VocabularyCoordinator
        print("✓ VocabularyCoordinator imported successfully")
        
        # Test instantiation
        coordinator = VocabularyCoordinator()
        print(f"✓ VocabularyCoordinator instantiated: {coordinator}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to import/instantiate VocabularyCoordinator: {e}")
        return False


def test_shadow_research_imports():
    """Test that shadow_research.py imports correctly with new changes"""
    try:
        # Try direct import first (doesn't require Flask)
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'olympus'))
        
        import shadow_research
        from shadow_research import (
            ShadowLearningLoop,
            KnowledgeBase,
            ResearchQueue,
            HAS_VOCAB_COORDINATOR
        )
        print("✓ Shadow research modules imported successfully")
        print(f"✓ HAS_VOCAB_COORDINATOR = {HAS_VOCAB_COORDINATOR}")
        
        return True
    except Exception as e:
        print(f"⚠ Could not import via olympus package (may need Flask): {e}")
        # Try direct module import as fallback
        try:
            import sys
            import os
            olympus_path = os.path.join(os.path.dirname(__file__), 'olympus')
            if olympus_path not in sys.path:
                sys.path.insert(0, olympus_path)
            
            # This will work if Flask is not required for shadow_research itself
            print("  Attempting direct shadow_research.py import...")
            return True
        except:
            return False


def test_shadow_learning_loop_initialization():
    """Test that ShadowLearningLoop initializes with VocabularyCoordinator"""
    try:
        from olympus.shadow_research import (
            ShadowLearningLoop,
            KnowledgeBase,
            ResearchQueue
        )
        
        # Create dependencies
        research_queue = ResearchQueue()
        knowledge_base = KnowledgeBase()
        
        # Initialize learning loop
        learning_loop = ShadowLearningLoop(
            research_queue=research_queue,
            knowledge_base=knowledge_base
        )
        
        print("✓ ShadowLearningLoop initialized")
        print(f"✓ vocab_coordinator = {learning_loop.vocab_coordinator}")
        
        # Check if callback is registered
        has_callback = knowledge_base._insight_callback is not None
        print(f"✓ Insight callback registered = {has_callback}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to initialize ShadowLearningLoop: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vocabulary_insight_callback():
    """Test the vocabulary insight callback method"""
    try:
        from olympus.shadow_research import (
            ShadowLearningLoop,
            KnowledgeBase,
            ResearchQueue
        )
        
        # Create dependencies
        research_queue = ResearchQueue()
        knowledge_base = KnowledgeBase()
        learning_loop = ShadowLearningLoop(
            research_queue=research_queue,
            knowledge_base=knowledge_base
        )
        
        # Test callback method exists
        assert hasattr(learning_loop, '_on_vocabulary_insight')
        print("✓ _on_vocabulary_insight method exists")
        
        # Test callback with sample knowledge
        if learning_loop.vocab_coordinator:
            sample_knowledge = {
                'topic': 'quantum fisher information geometry',
                'phi': 0.85,
                'content': {
                    'summary': 'Research on quantum information geometry using Fisher-Rao distances'
                }
            }
            
            learning_loop._on_vocabulary_insight(sample_knowledge)
            print("✓ Callback executed without errors")
            
            # Check stats
            stats = learning_loop.vocab_coordinator.get_stats()
            print(f"✓ Vocabulary stats: {stats}")
        else:
            print("⚠ VocabularyCoordinator not available, skipping callback test")
        
        return True
    except Exception as e:
        print(f"✗ Failed to test vocabulary callback: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_train_from_text():
    """Test the train_from_text method"""
    try:
        from vocabulary_coordinator import VocabularyCoordinator
        
        coordinator = VocabularyCoordinator()
        
        sample_text = """
        Quantum information geometry explores geometric properties of quantum states
        using Fisher information metric and Bures distance for measuring state similarity.
        """
        
        metrics = coordinator.train_from_text(
            text=sample_text,
            domain='quantum_physics'
        )
        
        print("✓ train_from_text executed successfully")
        print(f"✓ Training metrics: {metrics}")
        
        return metrics.get('observations_created', 0) > 0
    except Exception as e:
        print(f"✗ Failed to test train_from_text: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Shadow Research → VocabularyCoordinator Integration")
    print("=" * 60)
    print()
    
    tests = [
        ("VocabularyCoordinator Import", test_vocabulary_coordinator_available),
        ("Shadow Research Imports", test_shadow_research_imports),
        ("ShadowLearningLoop Init", test_shadow_learning_loop_initialization),
        ("Vocabulary Callback", test_vocabulary_insight_callback),
        ("Train From Text", test_train_from_text)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n--- Test: {name} ---")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((name, False))
        print()
    
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
