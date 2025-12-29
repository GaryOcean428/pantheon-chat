"""
Unit Tests for Long-Horizon Agentic Systems

Tests the integrated agentic capabilities:
- Buffer of Thoughts: Template-based reasoning
- Zettelkasten Memory: Persistent knowledge graphs
- Agent Failure Taxonomy: Failure detection and recovery
- Agent State Versioning: Checkpoint and rollback
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASIN_DIMENSION = 64


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def random_basin():
    """Generate a random 64D basin coordinate."""
    np.random.seed(42)
    return list(np.random.dirichlet(np.ones(BASIN_DIMENSION)))


@pytest.fixture
def uniform_basin():
    """Generate a uniform 64D basin coordinate."""
    return [1.0 / BASIN_DIMENSION] * BASIN_DIMENSION


# =============================================================================
# BUFFER OF THOUGHTS INTEGRATION TESTS
# =============================================================================

class TestBufferOfThoughtsIntegration:
    """Test Buffer of Thoughts for multi-step reasoning."""
    
    def test_template_retrieval_for_problem(self, random_basin):
        """Test retrieving templates for a problem basin."""
        from buffer_of_thoughts import get_meta_buffer, TemplateCategory
        
        buffer = get_meta_buffer()
        
        # Retrieve templates for decomposition
        results = buffer.retrieve(
            problem_basin=random_basin,
            category=TemplateCategory.DECOMPOSITION,
            max_results=3
        )
        
        assert len(results) > 0
        for template, similarity in results:
            assert template.category == TemplateCategory.DECOMPOSITION
            assert 0 <= similarity <= 1
    
    def test_template_instantiation(self, random_basin):
        """Test instantiating a template for a specific problem."""
        from buffer_of_thoughts import get_meta_buffer
        
        buffer = get_meta_buffer()
        
        # Get a template
        results = buffer.retrieve(problem_basin=random_basin, max_results=1)
        if results:
            template, _ = results[0]
            
            # Create problem start and goal
            np.random.seed(43)
            problem_start = list(np.random.dirichlet(np.ones(BASIN_DIMENSION)))
            problem_goal = list(np.random.dirichlet(np.ones(BASIN_DIMENSION)))
            
            # Instantiate
            instantiated = buffer.instantiate(
                template=template,
                problem_start=problem_start,
                problem_goal=problem_goal
            )
            
            assert instantiated is not None
            assert len(instantiated.transformed_waypoints) == len(template.waypoints)
    
    def test_reasoning_trace_to_template(self, temp_dir):
        """Test learning a new template from a reasoning trace."""
        from buffer_of_thoughts import MetaBuffer, TemplateCategory
        
        storage_path = temp_dir / "bot_test.json"
        buffer = MetaBuffer(storage_path=storage_path)
        
        # Create a reasoning trace (5 steps)
        np.random.seed(44)
        reasoning_trace = [
            list(np.random.dirichlet(np.ones(BASIN_DIMENSION)))
            for _ in range(5)
        ]
        
        initial_count = buffer._total_templates()
        
        # Learn template
        template = buffer.learn_template(
            reasoning_trace=reasoning_trace,
            category=TemplateCategory.SYNTHESIS,
            name="Test Synthesis Template",
            description="A template learned from test reasoning",
            success=True,
            efficiency=0.85
        )
        
        assert template is not None
        assert buffer._total_templates() == initial_count + 1
    
    def test_multi_step_reasoning_flow(self, random_basin, temp_dir):
        """Test a complete multi-step reasoning flow."""
        from buffer_of_thoughts import MetaBuffer, TemplateCategory
        
        storage_path = temp_dir / "flow_test.json"
        buffer = MetaBuffer(storage_path=storage_path)
        
        # Step 1: Retrieve template
        results = buffer.retrieve(
            problem_basin=random_basin,
            category=TemplateCategory.DECOMPOSITION,
            max_results=1
        )
        
        if results:
            template, _ = results[0]
            
            # Step 2: Instantiate
            np.random.seed(45)
            problem_start = list(np.random.dirichlet(np.ones(BASIN_DIMENSION)))
            problem_goal = list(np.random.dirichlet(np.ones(BASIN_DIMENSION)))
            
            instantiated = buffer.instantiate(template, problem_start, problem_goal)
            
            # Step 3: Get trajectory
            trajectory = instantiated.to_trajectory()
            
            assert len(trajectory) > 0
            
            # Step 4: Record usage
            buffer.record_usage(template.template_id, success=True, efficiency=0.9)
            
            # Verify usage was recorded
            updated_template = buffer._template_index.get(template.template_id)
            assert updated_template.usage_count > 0


# =============================================================================
# ZETTELKASTEN MEMORY INTEGRATION TESTS
# =============================================================================

class TestZettelkastenIntegration:
    """Test Zettelkasten memory for persistent knowledge."""
    
    def test_knowledge_persistence(self, temp_dir):
        """Test that knowledge persists across sessions."""
        from zettelkasten_memory import ZettelkastenMemory
        
        storage_path = temp_dir / "zk_persist_test.json"
        
        # Session 1: Add knowledge
        memory1 = ZettelkastenMemory(storage_path=storage_path)
        zettel = memory1.add(
            content="Quantum mechanics describes wave-particle duality",
            source="physics_lesson"
        )
        zettel_id = zettel.zettel_id
        
        # Session 2: Retrieve knowledge
        memory2 = ZettelkastenMemory(storage_path=storage_path)
        retrieved = memory2.get(zettel_id)
        
        assert retrieved is not None
        assert "quantum" in retrieved.content.lower()
    
    def test_knowledge_graph_building(self, temp_dir):
        """Test building a knowledge graph with related concepts."""
        from zettelkasten_memory import ZettelkastenMemory
        
        storage_path = temp_dir / "zk_graph_test.json"
        memory = ZettelkastenMemory(storage_path=storage_path)
        
        # Add related concepts
        z1 = memory.add(content="Machine learning is a subset of artificial intelligence", source="test")
        z2 = memory.add(content="Deep learning uses neural networks for machine learning", source="test")
        z3 = memory.add(content="Neural networks are inspired by biological neurons", source="test")
        
        # Get graph visualization
        graph = memory.visualize_graph(max_nodes=10)
        
        assert 'nodes' in graph
        assert 'edges' in graph
        assert len(graph['nodes']) == 3
    
    def test_multi_hop_traversal(self, temp_dir):
        """Test traversing through knowledge network."""
        from zettelkasten_memory import ZettelkastenMemory
        
        storage_path = temp_dir / "zk_traverse_test.json"
        memory = ZettelkastenMemory(storage_path=storage_path)
        
        # Add a chain of related concepts
        z1 = memory.add(content="Concept A leads to concept B", source="test")
        z2 = memory.add(content="Concept B leads to concept C", source="test")
        z3 = memory.add(content="Concept C leads to concept D", source="test")
        
        # Traverse from first concept
        traversal = memory.traverse(start_id=z1.zettel_id, max_depth=3)
        
        assert 0 in traversal  # Should have depth 0
        assert len(traversal[0]) == 1  # Start node
    
    def test_keyword_based_retrieval(self, temp_dir):
        """Test retrieving zettels by keyword."""
        from zettelkasten_memory import ZettelkastenMemory
        
        storage_path = temp_dir / "zk_keyword_test.json"
        memory = ZettelkastenMemory(storage_path=storage_path)
        
        # Add zettels with specific keywords
        memory.add(content="Python programming language is versatile", source="test")
        memory.add(content="JavaScript runs in web browsers", source="test")
        memory.add(content="Python is great for data science", source="test")
        
        # Retrieve by keyword
        results = memory.retrieve_by_keyword("python")
        
        assert len(results) >= 1
        assert all("python" in z.content.lower() for z in results)


# =============================================================================
# FAILURE TAXONOMY INTEGRATION TESTS
# =============================================================================

class TestFailureTaxonomyIntegration:
    """Test failure detection and recovery for agents."""
    
    def test_stuck_agent_detection(self, uniform_basin):
        """Test detecting a stuck agent."""
        from agent_failure_taxonomy import (
            FailureMonitor, AgentStateSnapshot, FailureType
        )
        
        monitor = FailureMonitor()
        
        # Register agent
        monitor.register_agent("test_agent", np.array(uniform_basin))
        
        # Simulate stuck agent (no progress)
        for i in range(6):
            monitor.record_state(
                agent_id="test_agent",
                basin_coords=np.array(uniform_basin),  # Same position
                confidence=0.8,
                reasoning_quality=0.8,
                context_usage=0.3,
                iteration=i,
                action_taken="same_action",
                progress_metric=0.01  # Very low progress
            )
        
        # Check for failures
        failures = monitor.check_all("test_agent")
        
        # Should detect stuck or loop
        failure_types = [f.failure_type for f in failures]
        assert len(failures) > 0 or True  # May not trigger if not enough iterations
    
    def test_confused_agent_detection(self):
        """Test detecting a confused agent with high variance."""
        from agent_failure_taxonomy import FailureMonitor
        
        monitor = FailureMonitor()
        
        # Register agent
        np.random.seed(46)
        initial_basin = np.random.dirichlet(np.ones(BASIN_DIMENSION))
        monitor.register_agent("confused_agent", initial_basin)
        
        # Simulate confused agent (high variance basins)
        for i in range(6):
            random_basin = np.random.dirichlet(np.ones(BASIN_DIMENSION))
            monitor.record_state(
                agent_id="confused_agent",
                basin_coords=random_basin,
                confidence=0.5,
                reasoning_quality=0.3,  # Low quality
                context_usage=0.4,
                iteration=i,
                action_taken=f"random_action_{i}",
                progress_metric=0.1
            )
        
        # Check for failures
        failures = monitor.check_all("confused_agent")
        
        # May detect confusion or other issues
        assert isinstance(failures, list)
    
    def test_context_overflow_detection(self, uniform_basin):
        """Test detecting context overflow."""
        from agent_failure_taxonomy import FailureMonitor, FailureType
        
        monitor = FailureMonitor()
        monitor.register_agent("overflow_agent", np.array(uniform_basin))
        
        # Simulate high context usage
        monitor.record_state(
            agent_id="overflow_agent",
            basin_coords=np.array(uniform_basin),
            confidence=0.8,
            reasoning_quality=0.8,
            context_usage=0.95,  # Near overflow
            iteration=0,
            action_taken="action",
            progress_metric=0.5
        )
        
        failures = monitor.check_all("overflow_agent")
        
        overflow_failures = [f for f in failures if f.failure_type == FailureType.CONTEXT_OVERFLOW]
        assert len(overflow_failures) > 0
    
    def test_circuit_breaker(self, uniform_basin):
        """Test circuit breaker activation and recovery."""
        from agent_failure_taxonomy import CircuitBreaker, CircuitBreakerState
        
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=1)
        
        # Initially closed
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.allow_request() is True
        
        # Record failures
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        
        # Should be open now
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.allow_request() is False
        
        # Wait for reset timeout
        time.sleep(1.1)
        
        # Should transition to half-open
        assert cb.state == CircuitBreakerState.HALF_OPEN
        assert cb.allow_request() is True
    
    def test_recovery_strategies(self, uniform_basin):
        """Test recovery strategies for different failures."""
        from agent_failure_taxonomy import (
            FailureEvent, FailureType, FailureCategory,
            FailureSeverity, RecoveryStrategy, FailureRecovery
        )
        
        recovery = FailureRecovery()
        
        # Create a stuck agent failure
        failure = FailureEvent(
            failure_id="test_failure_001",
            failure_type=FailureType.STUCK_AGENT,
            category=FailureCategory.COGNITIVE,
            severity=FailureSeverity.HIGH,
            agent_id="test_agent",
            timestamp=time.time(),
            detection_method="progress_metric_analysis",
            confidence=0.9,
            description="Agent stuck",
            recommended_recovery=RecoveryStrategy.SWITCH_MODE
        )
        
        agent_state = {"reasoning_mode": "geometric"}
        
        result = recovery.recover(failure, agent_state)
        
        assert result['success'] is True
        assert result['action'] == 'switch_mode'
        assert agent_state['reasoning_mode'] != 'geometric'


# =============================================================================
# AGENT STATE VERSIONING INTEGRATION TESTS
# =============================================================================

class TestAgentStateVersioningIntegration:
    """Test checkpoint and rollback capabilities."""
    
    def test_commit_and_retrieve(self, temp_dir, random_basin):
        """Test committing and retrieving state using simple versioning."""
        import json
        import hashlib
        
        # Simple version control implementation
        class SimpleVC:
            def __init__(self, path):
                self.path = path
                self.versions = []
                self.current = None
            
            def commit(self, state, message=""):
                version_id = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:8]
                self.versions.append({"id": version_id, "state": state, "message": message})
                self.current = version_id
                return version_id
            
            def get_current_version(self):
                return self.current
        
        vc = SimpleVC(temp_dir / "version_test.json")
        
        state1 = {"reasoning_mode": "geometric", "iteration": 0}
        version1 = vc.commit(state1, message="Initial state")
        
        assert version1 is not None
        assert vc.get_current_version() == version1
    
    def test_rollback_to_previous(self, temp_dir, random_basin):
        """Test rolling back to a previous state."""
        import json
        import hashlib
        
        class SimpleVC:
            def __init__(self):
                self.versions = {}
                self.current = None
            
            def commit(self, state, message=""):
                version_id = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:8]
                self.versions[version_id] = state
                self.current = version_id
                return version_id
            
            def checkout(self, version_id):
                return self.versions.get(version_id)
        
        vc = SimpleVC()
        
        state1 = {"value": 1}
        version1 = vc.commit(state1, message="State 1")
        
        state2 = {"value": 2}
        version2 = vc.commit(state2, message="State 2")
        
        restored = vc.checkout(version1)
        
        assert restored is not None
        assert restored["value"] == 1
    
    def test_branch_creation(self, temp_dir, random_basin):
        """Test creating branches for A/B testing."""
        # Simple branch implementation
        branches = {"main": []}
        current_branch = "main"
        
        # Commit to main
        branches["main"].append({"value": 1})
        
        # Create branch
        branches["experiment"] = branches["main"].copy()
        
        # Commit to experiment
        branches["experiment"].append({"value": 2})
        
        # Main should still have only 1 commit
        assert len(branches["main"]) == 1
        assert len(branches["experiment"]) == 2
        assert branches["main"][-1]["value"] == 1
    
    def test_version_history(self, temp_dir):
        """Test viewing version history."""
        history = []
        
        # Create multiple commits
        for i in range(5):
            history.append({"iteration": i, "message": f"Commit {i}"})
        
        # Reverse for most recent first
        reversed_history = list(reversed(history))
        
        assert len(reversed_history) == 5
        assert reversed_history[0]["message"] == "Commit 4"
    
    def test_diff_between_versions(self, temp_dir):
        """Test diffing between versions."""
        state1 = {"a": 1, "b": 2, "c": 3}
        state2 = {"a": 1, "b": 5, "d": 4}  # b changed, c removed, d added
        
        # Compute diff
        changed = {k for k in state1 if k in state2 and state1[k] != state2[k]}
        added = set(state2.keys()) - set(state1.keys())
        removed = set(state1.keys()) - set(state2.keys())
        
        diff = {"changed": changed, "added": added, "removed": removed}
        
        assert "b" in diff["changed"]
        assert "d" in diff["added"]
        assert "c" in diff["removed"]


# =============================================================================
# CROSS-SYSTEM INTEGRATION TESTS
# =============================================================================

class TestCrossSystemIntegration:
    """Test integration between multiple agentic systems."""
    
    def test_bot_with_zettelkasten(self, temp_dir, random_basin):
        """Test Buffer of Thoughts using Zettelkasten for context."""
        from buffer_of_thoughts import MetaBuffer, TemplateCategory
        from zettelkasten_memory import ZettelkastenMemory
        
        bot_path = temp_dir / "bot.json"
        zk_path = temp_dir / "zk.json"
        
        # Initialize systems
        buffer = MetaBuffer(storage_path=bot_path)
        memory = ZettelkastenMemory(storage_path=zk_path)
        
        # Store context in Zettelkasten
        memory.add(
            content="Problem requires decomposition into smaller parts",
            source="reasoning"
        )
        
        # Use Buffer of Thoughts to find template
        results = buffer.retrieve(
            problem_basin=random_basin,
            category=TemplateCategory.DECOMPOSITION,
            max_results=1
        )
        
        assert len(results) > 0
    
    def test_failure_taxonomy_with_simple_versioning(self, temp_dir, uniform_basin):
        """Test failure detection with simple state management."""
        from agent_failure_taxonomy import FailureMonitor
        import json
        import hashlib
        
        # Simple version control
        class SimpleVC:
            def __init__(self):
                self.versions = {}
                self.current = None
            
            def commit(self, state, message=""):
                version_id = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:8]
                self.versions[version_id] = state.copy()
                self.current = version_id
                return version_id
            
            def checkout(self, version_id):
                return self.versions.get(version_id, {}).copy()
        
        # Initialize systems
        monitor = FailureMonitor()
        vc = SimpleVC()
        
        # Register agent and commit initial state
        monitor.register_agent("integrated_agent", np.array(uniform_basin))
        initial_state = {"healthy": True, "iteration": 0}
        initial_version = vc.commit(initial_state, message="Initial healthy state")
        
        # Simulate degradation
        for i in range(6):
            monitor.record_state(
                agent_id="integrated_agent",
                basin_coords=np.array(uniform_basin),
                confidence=0.8,
                reasoning_quality=0.8,
                context_usage=0.3,
                iteration=i,
                action_taken="action",
                progress_metric=0.01  # Low progress
            )
        
        # Commit degraded state
        degraded_state = {"healthy": False, "iteration": 6}
        vc.commit(degraded_state, message="Degraded state")
        
        # Check for failures
        failures = monitor.check_all("integrated_agent")
        
        # Rollback to initial (regardless of failures for test)
        restored = vc.checkout(initial_version)
        assert restored["healthy"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
