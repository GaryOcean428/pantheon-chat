#!/usr/bin/env python3
"""
Test script to verify the 4 spot fixes.

Spot Fixes:
1. Auto-Flush Pending Messages
2. Peer Discovery
3. Honest Shadow Research Status
4. Training Auto-Trigger
"""

import sys
from typing import Dict, Optional

def test_flush_pending_messages():
    """Test Spot Fix #1: Auto-Flush Pending Messages"""
    print("\n=== Test 1: Auto-Flush Pending Messages ===")
    try:
        from olympus.base_god import BaseGod
        
        # Create mock god
        class MockGod(BaseGod):
            def assess_target(self, target: str, context: Optional[Dict] = None) -> Dict:
                return {'probability': 0.5, 'confidence': 0.5}
            def get_status(self) -> Dict:
                return {'name': self.name, 'domain': self.domain}
        
        god = MockGod(name="TestGod", domain="testing")
        
        # Create some pending messages
        god.share_insight("Test insight", confidence=0.8)
        god.praise_peer("Zeus", "Great work!")
        
        # Test flush
        assert hasattr(god, 'flush_pending_messages'), "flush_pending_messages method missing"
        count = god.flush_pending_messages()
        
        print(f"✅ flush_pending_messages() exists")
        print(f"✅ Flushed {count} messages")
        assert count == 2, f"Expected 2 messages, got {count}"
        assert len(god.pending_messages) == 0, "Messages not cleared"
        print("✅ Messages cleared after flush")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_peer_discovery():
    """Test Spot Fix #2: Peer Discovery"""
    print("\n=== Test 2: Peer Discovery ===")
    try:
        from olympus.base_god import BaseGod
        
        # Create mock gods
        class MockGod(BaseGod):
            def assess_target(self, target: str, context: Optional[Dict] = None) -> Dict:
                return {'probability': 0.5, 'confidence': 0.5}
            def get_status(self) -> Dict:
                return {'name': self.name, 'domain': self.domain}
        
        # Clear registry
        BaseGod._god_registry.clear()
        
        # Create gods
        zeus = MockGod(name="Zeus", domain="leadership")
        athena = MockGod(name="Athena", domain="strategy")
        ares = MockGod(name="Ares", domain="war")
        
        # Test discovery
        assert hasattr(BaseGod, 'discover_peers'), "discover_peers method missing"
        peers = BaseGod.discover_peers()
        print(f"✅ discover_peers() exists")
        print(f"✅ Found {len(peers)} peers: {peers}")
        assert len(peers) == 3, f"Expected 3 peers, got {len(peers)}"
        assert "Zeus" in peers and "Athena" in peers and "Ares" in peers
        
        # Test get_peer_info
        assert hasattr(BaseGod, 'get_peer_info'), "get_peer_info method missing"
        info = BaseGod.get_peer_info("Athena")
        print(f"✅ get_peer_info() exists")
        print(f"✅ Athena info: domain={info['domain']}, reputation={info['reputation']}")
        assert info['domain'] == 'strategy'
        
        # Test find_expert_for_domain
        assert hasattr(BaseGod, 'find_expert_for_domain'), "find_expert_for_domain method missing"
        athena.skills['strategy'] = 1.5
        expert = BaseGod.find_expert_for_domain('strategy')
        print(f"✅ find_expert_for_domain() exists")
        print(f"✅ Found strategy expert: {expert}")
        assert expert == "Athena", f"Expected Athena, got {expert}"
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_shadow_research_status():
    """Test Spot Fix #3: Honest Shadow Research Status"""
    print("\n=== Test 3: Honest Shadow Research Status ===")
    try:
        from olympus.base_god import BaseGod
        
        # Create mock god
        class MockGod(BaseGod):
            def assess_target(self, target: str, context: Optional[Dict] = None) -> Dict:
                return {'probability': 0.5, 'confidence': 0.5}
            def get_status(self) -> Dict:
                return {'name': self.name, 'domain': self.domain}
        
        god = MockGod(name="TestGod", domain="testing")
        
        # Check shadow research capabilities
        shadow_caps = god.mission.get('shadow_research_capabilities', {})
        assert 'available' in shadow_caps, "Shadow research status not documented"
        available = shadow_caps.get('available', False)
        can_request = shadow_caps.get('can_request_research', False)
        
        print(f"✅ Shadow research status documented")
        print(f"✅ Available: {available}")
        print(f"✅ Can request: {can_request}")
        print(f"✅ Status matches availability: {available == can_request}")
        
        # Status should be honest (either both True or both False)
        assert available == can_request, "Status mismatch between available and can_request"
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_auto_trigger():
    """Test Spot Fix #4: Training Auto-Trigger"""
    print("\n=== Test 4: Training Auto-Trigger ===")
    try:
        from olympus.base_god import BaseGod
        
        # Create mock god
        class MockGod(BaseGod):
            def assess_target(self, target: str, context: Optional[Dict] = None) -> Dict:
                return {'probability': 0.5, 'confidence': 0.5}
            def get_status(self) -> Dict:
                return {'name': self.name, 'domain': self.domain}
        
        god = MockGod(name="TestGod", domain="testing")
        
        # Test learn_from_outcome (should not crash even if training not available)
        result = god.learn_from_outcome(
            target="test_target",
            assessment={'probability': 0.8, 'confidence': 0.7},
            actual_outcome={'domain': 'testing', 'phi': 0.75},
            success=True
        )
        
        print(f"✅ learn_from_outcome() executed without error")
        print(f"✅ Training auto-trigger integrated (may not be active)")
        print(f"✅ Result: {result}")
        assert result['learned'] == True
        assert 'reputation_change' in result
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_peer_discovery_in_mission():
    """Test that peer discovery is documented in mission context"""
    print("\n=== Test 5: Peer Discovery in Mission Context ===")
    try:
        from olympus.base_god import BaseGod
        
        class MockGod(BaseGod):
            def assess_target(self, target: str, context: Optional[Dict] = None) -> Dict:
                return {'probability': 0.5, 'confidence': 0.5}
            def get_status(self) -> Dict:
                return {'name': self.name, 'domain': self.domain}
        
        god = MockGod(name="TestGod", domain="testing")
        
        # Check peer discovery in mission
        assert 'peer_discovery_capabilities' in god.mission, "Peer discovery not in mission"
        peer_caps = god.mission['peer_discovery_capabilities']
        
        print(f"✅ Peer discovery documented in mission")
        assert peer_caps['can_discover_peers'] == True
        print(f"✅ can_discover_peers: {peer_caps['can_discover_peers']}")
        print(f"✅ How to discover: {peer_caps['how_to_discover']}")
        print(f"✅ How to get info: {peer_caps['how_to_get_info']}")
        print(f"✅ How to find expert: {peer_caps['how_to_find_expert']}")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all spot fix tests."""
    print("=" * 60)
    print("Spot Fixes Test Suite")
    print("=" * 60)
    
    tests = [
        ("Flush Pending Messages", test_flush_pending_messages),
        ("Peer Discovery", test_peer_discovery),
        ("Shadow Research Status", test_shadow_research_status),
        ("Training Auto-Trigger", test_training_auto_trigger),
        ("Peer Discovery in Mission", test_peer_discovery_in_mission),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
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
        print("\n🎉 All spot fix tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
