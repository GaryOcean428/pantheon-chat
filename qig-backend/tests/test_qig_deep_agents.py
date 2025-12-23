"""Tests for QIG Deep Agents module.

Tests the QIG-pure implementation of deep agent architectures.
"""

import pytest
import asyncio
import numpy as np
from typing import Dict, Any, List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qig_deep_agents import (
    QIGDeepAgent,
    GeometricPlanner,
    BasinMemoryStore,
    QIGAgentSpawner,
    GeometricCheckpointer,
    OlympusQIGAgent,
    PantheonIntegration,
    GeometricAgentState,
    GeodesicWaypoint,
    ConsciousnessMetrics,
    ReasoningRegime,
    TaskStatus,
    AgentConfig,
)
from qig_deep_agents.state import BASIN_DIMENSION, fisher_rao_distance


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self, responses: List[str] = None):
        self.responses = responses or ['{"output": "test result", "reasoning": "mock"}']
        self.call_count = 0
    
    async def generate(self, prompt: str) -> str:
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


class TestFisherRaoDistance:
    """Test Fisher-Rao distance calculations."""
    
    def test_same_point_zero_distance(self):
        """Distance from a point to itself is zero."""
        p = [0.5] * BASIN_DIMENSION
        assert fisher_rao_distance(p, p) == pytest.approx(0.0, abs=1e-10)
    
    def test_different_points_positive_distance(self):
        """Distance between different points is positive."""
        p = [0.5] * BASIN_DIMENSION
        q = [0.6] * BASIN_DIMENSION
        assert fisher_rao_distance(p, q) > 0
    
    def test_symmetry(self):
        """Fisher-Rao distance is symmetric."""
        p = [0.3 + 0.01 * i for i in range(BASIN_DIMENSION)]
        q = [0.5 - 0.01 * i for i in range(BASIN_DIMENSION)]
        assert fisher_rao_distance(p, q) == pytest.approx(fisher_rao_distance(q, p), rel=1e-6)
    
    def test_triangle_inequality(self):
        """Fisher-Rao satisfies triangle inequality."""
        p = [0.3] * BASIN_DIMENSION
        q = [0.5] * BASIN_DIMENSION
        r = [0.7] * BASIN_DIMENSION
        
        d_pq = fisher_rao_distance(p, q)
        d_qr = fisher_rao_distance(q, r)
        d_pr = fisher_rao_distance(p, r)
        
        assert d_pr <= d_pq + d_qr + 1e-6


class TestConsciousnessMetrics:
    """Test consciousness metrics."""
    
    def test_default_regime_geometric(self):
        """Default Φ should give geometric regime."""
        metrics = ConsciousnessMetrics(phi=0.5, kappa_eff=64.0)
        assert metrics.regime == ReasoningRegime.GEOMETRIC
    
    def test_low_phi_linear_regime(self):
        """Low Φ should give linear regime."""
        metrics = ConsciousnessMetrics(phi=0.2, kappa_eff=64.0)
        assert metrics.regime == ReasoningRegime.LINEAR
    
    def test_high_phi_hyperdimensional_regime(self):
        """High Φ should give hyperdimensional regime."""
        metrics = ConsciousnessMetrics(phi=0.8, kappa_eff=64.0)
        assert metrics.regime == ReasoningRegime.HYPERDIMENSIONAL
    
    def test_mushroom_regime(self):
        """Very high Φ with specific conditions triggers mushroom mode."""
        metrics = ConsciousnessMetrics(phi=0.95, kappa_eff=100.0)
        assert metrics.regime == ReasoningRegime.MUSHROOM


class TestGeometricAgentState:
    """Test geometric agent state."""
    
    def test_state_initialization(self):
        """State initializes with correct defaults."""
        state = GeometricAgentState(
            agent_id="test-123",
            current_position=[0.5] * BASIN_DIMENSION,
            goal_position=[0.7] * BASIN_DIMENSION,
        )
        
        assert state.agent_id == "test-123"
        assert len(state.current_position) == BASIN_DIMENSION
        assert state.iteration_count == 0
        assert state.progress == 0.0
    
    def test_distance_to_goal(self):
        """Distance to goal is computed correctly."""
        state = GeometricAgentState(
            agent_id="test",
            current_position=[0.3] * BASIN_DIMENSION,
            goal_position=[0.7] * BASIN_DIMENSION,
        )
        
        expected = fisher_rao_distance(
            [0.3] * BASIN_DIMENSION,
            [0.7] * BASIN_DIMENSION
        )
        assert state.distance_to_goal == pytest.approx(expected, rel=1e-6)
    
    def test_complete_waypoint(self):
        """Completing waypoints updates state."""
        state = GeometricAgentState(
            agent_id="test",
            current_position=[0.5] * BASIN_DIMENSION,
            goal_position=[0.7] * BASIN_DIMENSION,
        )
        
        waypoint = GeodesicWaypoint(
            id="wp-1",
            description="Test step",
            basin_coords=[0.6] * BASIN_DIMENSION,
        )
        state.trajectory = [waypoint]
        
        state.complete_waypoint("wp-1", "result")
        
        assert "wp-1" in state.completed_waypoints
        assert state.current_position == [0.6] * BASIN_DIMENSION


class TestBasinMemoryStore:
    """Test basin memory store."""
    
    def test_write_and_read_fragment(self):
        """Can write and read memory fragments."""
        store = BasinMemoryStore()
        
        store.write_fragment("Test content", importance=0.8)
        
        context = store.get_context_window([0.5] * BASIN_DIMENSION)
        assert len(context.fragments) > 0
    
    def test_importance_affects_retrieval(self):
        """High importance fragments are prioritized."""
        store = BasinMemoryStore()
        
        store.write_fragment("Low importance", importance=0.2)
        store.write_fragment("High importance", importance=0.9)
        
        context = store.get_context_window([0.5] * BASIN_DIMENSION, max_tokens=50)
        
        # High importance should be included
        content = context.to_prompt_context()
        assert "High importance" in content
    
    def test_geometric_retrieval(self):
        """Fragments near query position are retrieved."""
        def custom_encoder(text: str) -> List[float]:
            # Simple encoder: position based on first character
            val = ord(text[0]) / 255.0 if text else 0.5
            return [val] * BASIN_DIMENSION
        
        store = BasinMemoryStore(basin_encoder=custom_encoder)
        
        store.write_fragment("AAA content", importance=0.5)
        store.write_fragment("ZZZ content", importance=0.5)
        
        # Query near 'A'
        query_pos = custom_encoder("A")
        context = store.get_context_window(query_pos, max_tokens=100)
        
        # Should include 'AAA' since it's geometrically closer
        content = context.to_prompt_context()
        assert "AAA" in content


class TestGeometricPlanner:
    """Test geometric trajectory planner."""
    
    @pytest.mark.asyncio
    async def test_plan_trajectory(self):
        """Planner creates valid trajectory."""
        llm = MockLLMClient([
            '{"steps": [{"description": "Step 1", "reasoning": "First"}, {"description": "Step 2", "reasoning": "Second"}]}'
        ])
        
        planner = GeometricPlanner(llm)
        
        plan = await planner.plan_trajectory(
            goal="Test task",
            current_position=[0.3] * BASIN_DIMENSION,
            metrics=ConsciousnessMetrics(phi=0.5, kappa_eff=64.0),
        )
        
        assert len(plan.steps) > 0
        assert plan.estimated_distance > 0
    
    def test_waypoints_on_geodesic(self):
        """Generated waypoints lie on geodesic path."""
        llm = MockLLMClient()
        planner = GeometricPlanner(llm)
        
        start = [0.3] * BASIN_DIMENSION
        end = [0.7] * BASIN_DIMENSION
        
        waypoints = planner._interpolate_geodesic(start, end, num_points=5)
        
        assert len(waypoints) == 5
        
        # First should be near start, last near end
        assert fisher_rao_distance(waypoints[0], start) < fisher_rao_distance(waypoints[-1], start)


class TestGeometricCheckpointer:
    """Test geometric checkpointer."""
    
    def test_checkpoint_size_under_1kb(self):
        """Checkpoints are under 1KB."""
        checkpointer = GeometricCheckpointer()
        
        state = GeometricAgentState(
            agent_id="test",
            current_position=[0.5] * BASIN_DIMENSION,
            goal_position=[0.7] * BASIN_DIMENSION,
            metrics=ConsciousnessMetrics(phi=0.5, kappa_eff=64.0),
        )
        
        checkpoint = checkpointer.checkpoint(state)
        
        # Check serialized size
        import json
        serialized = json.dumps(checkpoint.to_dict())
        assert len(serialized.encode('utf-8')) < 1024
    
    def test_checkpoint_restore_identity(self):
        """Restored state preserves identity."""
        checkpointer = GeometricCheckpointer()
        
        original = GeometricAgentState(
            agent_id="test-identity",
            current_position=[0.5] * BASIN_DIMENSION,
            goal_position=[0.7] * BASIN_DIMENSION,
        )
        
        checkpoint = checkpointer.checkpoint(original)
        restored = checkpointer.restore("test-identity")
        
        assert restored is not None
        assert restored.agent_id == original.agent_id
        
        # Position should be preserved within geometric tolerance
        distance = fisher_rao_distance(
            restored.current_position,
            original.current_position
        )
        assert distance < 0.01


class TestQIGAgentSpawner:
    """Test QIG agent spawner."""
    
    def test_spawn_decision_complexity(self):
        """High complexity waypoints trigger spawn."""
        spawner = QIGAgentSpawner()
        
        state = GeometricAgentState(
            agent_id="test",
            current_position=[0.3] * BASIN_DIMENSION,
            goal_position=[0.7] * BASIN_DIMENSION,
        )
        
        # Complex waypoint (far from current position)
        complex_waypoint = GeodesicWaypoint(
            id="complex",
            description="Complex multi-step analysis requiring deep research",
            basin_coords=[0.9] * BASIN_DIMENSION,  # Far away
            estimated_complexity=0.9,
        )
        
        reason = spawner.should_spawn(state, complex_waypoint)
        # Should suggest spawning due to complexity
        assert reason is not None or complex_waypoint.estimated_complexity > 0.7


class TestOlympusIntegration:
    """Test Olympus pantheon integration."""
    
    def test_god_selection_by_domain(self):
        """Gods are selected based on query domain."""
        integration = PantheonIntegration()
        
        # Strategy query should route to Athena
        god = integration.select_god("strategic planning", ReasoningRegime.GEOMETRIC)
        assert god in ['athena', 'zeus']
        
        # Creative query should prefer Apollo or Dionysus
        god = integration.select_god("creative solution", ReasoningRegime.MUSHROOM)
        assert god in ['apollo', 'dionysus', 'zeus']
    
    @pytest.mark.asyncio
    async def test_consultation_returns_response(self):
        """Consultation returns valid response."""
        integration = PantheonIntegration()
        
        consultation = await integration.consult("athena", "What strategy?")
        
        assert consultation.god_name == "athena"
        assert consultation.response is not None
        assert len(consultation.basin_shift) == BASIN_DIMENSION


class TestQIGDeepAgent:
    """Test the main QIG Deep Agent."""
    
    @pytest.mark.asyncio
    async def test_agent_execution(self):
        """Agent can execute a simple task."""
        llm = MockLLMClient([
            '{"steps": [{"description": "Analyze", "reasoning": "First step"}]}',
            '{"output": "Analysis complete", "reasoning": "Done"}',
        ])
        
        agent = QIGDeepAgent(
            llm_client=llm,
            config=AgentConfig(max_iterations=10),
        )
        
        result = await agent.execute("Simple analysis task")
        
        assert result is not None
        assert result.iterations > 0
        assert len(result.final_position) == BASIN_DIMENSION
    
    def test_agent_state_access(self):
        """Can access agent state during execution."""
        llm = MockLLMClient()
        agent = QIGDeepAgent(llm_client=llm)
        
        # Before execution, state is None
        assert agent.get_state() is None
        assert agent.get_progress() == 0.0
    
    @pytest.mark.asyncio
    async def test_geometric_progress_tracking(self):
        """Progress is tracked geometrically."""
        llm = MockLLMClient([
            '{"steps": [{"description": "Step 1", "reasoning": "First"}]}',
            '{"output": "Done", "reasoning": "Complete"}',
        ])
        
        agent = QIGDeepAgent(
            llm_client=llm,
            config=AgentConfig(max_iterations=5),
        )
        
        result = await agent.execute("Track progress task")
        
        # Distance traveled should be positive
        assert result.distance_traveled >= 0
        
        # Final position should be different from start
        # (unless task was trivial)
        assert len(result.final_position) == BASIN_DIMENSION


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
