#!/usr/bin/env python3
"""
Test script to verify search capability integration with kernels.

Tests:
1. CapabilityType.SEARCH exists in capability mesh
2. BaseGod has request_search() method
3. BaseGod has search awareness in mission context
4. SearchOrchestrator can be wired to kernels
"""

import sys
import numpy as np
from typing import Dict, Optional

def test_capability_type_search():
    """Test that SEARCH capability type exists."""
    print("\n=== Test 1: CapabilityType.SEARCH exists ===")
    try:
        from olympus.capability_mesh import CapabilityType, EventType
        
        # Check SEARCH exists
        assert hasattr(CapabilityType, 'SEARCH'), "CapabilityType.SEARCH not found"
        assert CapabilityType.SEARCH.value == 'search', "SEARCH capability value incorrect"
        print("✅ CapabilityType.SEARCH exists")
        
        # Check search-related event types
        assert hasattr(EventType, 'SEARCH_REQUESTED'), "EventType.SEARCH_REQUESTED not found"
        assert hasattr(EventType, 'SEARCH_COMPLETE'), "EventType.SEARCH_COMPLETE not found"
        assert hasattr(EventType, 'SOURCE_DISCOVERED'), "EventType.SOURCE_DISCOVERED not found"
        print("✅ Search event types exist (SEARCH_REQUESTED, SEARCH_COMPLETE, SOURCE_DISCOVERED)")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_base_god_search_methods():
    """Test that BaseGod has search capability methods."""
    print("\n=== Test 2: BaseGod search methods exist ===")
    try:
        from olympus.base_god import BaseGod, SearchCapabilityMixin
        
        # Check mixin exists
        print("✅ SearchCapabilityMixin class exists")
        
        # Check mixin is in base classes
        assert SearchCapabilityMixin in BaseGod.__mro__, "SearchCapabilityMixin not in BaseGod inheritance"
        print("✅ SearchCapabilityMixin in BaseGod inheritance chain")
        
        # Check methods exist
        methods = ['request_search', 'get_available_search_providers', 
                   'discover_source', 'query_search_history', 
                   'get_search_capability_status']
        
        for method in methods:
            assert hasattr(SearchCapabilityMixin, method), f"Method {method} not found"
            print(f"✅ SearchCapabilityMixin.{method}() exists")
        
        # Check class methods
        assert hasattr(SearchCapabilityMixin, 'set_search_orchestrator'), "set_search_orchestrator not found"
        assert hasattr(SearchCapabilityMixin, 'get_search_orchestrator'), "get_search_orchestrator not found"
        print("✅ SearchCapabilityMixin class methods exist")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_base_god_mission_awareness():
    """Test that BaseGod includes search capability in mission context."""
    print("\n=== Test 3: BaseGod mission context includes search ===")
    try:
        from olympus.base_god import BaseGod
        
        # Create a mock god (we need to implement abstract methods)
        class MockGod(BaseGod):
            def assess_target(self, target: str, context: Optional[Dict] = None) -> Dict:
                return {'probability': 0.5, 'confidence': 0.5}
            
            def get_status(self) -> Dict:
                return {'name': self.name, 'domain': self.domain}
        
        god = MockGod(name="TestGod", domain="testing")
        
        # Check mission context has search capabilities
        assert 'search_capabilities' in god.mission, "search_capabilities not in mission context"
        print("✅ search_capabilities in mission context")
        
        search_caps = god.mission['search_capabilities']
        assert search_caps['can_request_search'] == True, "can_request_search should be True"
        assert 'how_to_request' in search_caps, "how_to_request not documented"
        assert 'available_providers' in search_caps, "available_providers not documented"
        assert 'providers' in search_caps, "providers list not in mission"
        print("✅ Search capability properly documented in mission")
        print(f"   Providers: {search_caps['providers']}")
        print(f"   How to use: {search_caps['how_to_request']}")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_search_orchestrator_wiring():
    """Test that SearchOrchestrator can be wired to BaseGod."""
    print("\n=== Test 4: SearchOrchestrator wiring ===")
    try:
        from olympus.base_god import BaseGod
        from geometric_search import SearchOrchestrator
        
        # Create orchestrator
        orchestrator = SearchOrchestrator()
        print("✅ SearchOrchestrator created")
        
        # Wire it to BaseGod
        BaseGod.set_search_orchestrator(orchestrator)
        print("✅ SearchOrchestrator wired to BaseGod")
        
        # Verify it's accessible
        retrieved = BaseGod.get_search_orchestrator()
        assert retrieved is orchestrator, "Retrieved orchestrator doesn't match"
        print("✅ SearchOrchestrator retrieval works")
        
        # Create a mock god and test access
        class MockGod(BaseGod):
            def assess_target(self, target: str, context: Optional[Dict] = None) -> Dict:
                return {'probability': 0.5, 'confidence': 0.5}
            
            def get_status(self) -> Dict:
                return {'name': self.name, 'domain': self.domain}
        
        god = MockGod(name="TestGod", domain="testing")
        
        # Test that god can access search methods
        status = god.get_search_capability_status()
        assert status['available'] == True, "Search capability should be available"
        print("✅ God can check search capability status")
        print(f"   Status: {status}")
        
        # Test get_available_search_providers
        providers = god.get_available_search_providers()
        print(f"✅ God can query available providers: {providers}")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_capability_event_creation():
    """Test that search capability events can be created."""
    print("\n=== Test 5: Search capability event creation ===")
    try:
        from olympus.capability_mesh import CapabilityEvent, EventType, CapabilityType
        
        # Create a search requested event
        event = CapabilityEvent(
            source=CapabilityType.SEARCH,
            event_type=EventType.SEARCH_REQUESTED,
            content={
                'query': 'test query',
                'requester': 'TestKernel'
            },
            phi=0.75,
            basin_coords=np.random.randn(64),
            priority=7
        )
        
        print("✅ SEARCH_REQUESTED event created")
        print(f"   Event ID: {event.event_id}")
        print(f"   Source: {event.source.value}")
        print(f"   Type: {event.event_type.value}")
        
        # Create a search complete event
        complete_event = CapabilityEvent(
            source=CapabilityType.SEARCH,
            event_type=EventType.SEARCH_COMPLETE,
            content={
                'query': 'test query',
                'results_count': 10
            },
            phi=0.8,
            priority=5
        )
        
        print("✅ SEARCH_COMPLETE event created")
        
        # Test to_dict serialization
        event_dict = event.to_dict()
        assert 'event_id' in event_dict, "event_id missing from serialization"
        assert 'source' in event_dict, "source missing from serialization"
        print("✅ Event serialization works")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("Search Capability Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Capability Type", test_capability_type_search),
        ("BaseGod Methods", test_base_god_search_methods),
        ("Mission Awareness", test_base_god_mission_awareness),
        ("Orchestrator Wiring", test_search_orchestrator_wiring),
        ("Event Creation", test_capability_event_creation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
