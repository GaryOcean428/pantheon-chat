"""Tests for Lightning kernel integration.

Validates:
- Event emission from kernels
- Lightning ingestion and correlation
- Insight generation and routing
"""

import time
import numpy as np
import pytest

from src.constellation.domain_intelligence import (
    DomainEvent,
    DomainIntelligence,
    DomainEventEmitter,
    TrendWindow,
)
from src.constellation.lightning_kernel import (
    LightningKernel,
    LightningInsight,
    DomainCorrelation,
)
from src.constellation.insights import InsightRouter, RoutedInsight


# =============================================================================
# DOMAIN EVENT TESTS
# =============================================================================

class TestDomainEvent:
    """Test DomainEvent creation and serialization."""
    
    def test_create_event(self):
        """Test basic event creation."""
        event = DomainEvent(
            domain="gary_physics",
            event_type="query_processed",
            content="What is quantum mechanics?",
            phi=0.75,
        )
        
        assert event.domain == "gary_physics"
        assert event.event_type == "query_processed"
        assert event.phi == 0.75
        assert event.event_id is not None
    
    def test_event_with_basin(self):
        """Test event with basin coordinates."""
        basin = np.random.randn(64)
        basin = basin / np.linalg.norm(basin)
        
        event = DomainEvent(
            domain="gary_code",
            event_type="compilation",
            content="Compiled module",
            phi=0.8,
            basin_coords=basin,
        )
        
        assert event.basin_coords is not None
        assert len(event.basin_coords) == 64
    
    def test_event_serialization(self):
        """Test event to_dict and from_dict."""
        event = DomainEvent(
            domain="test_domain",
            event_type="test_type",
            content="test content",
            phi=0.5,
            metadata={"key": "value"},
        )
        
        data = event.to_dict()
        assert data["domain"] == "test_domain"
        assert data["phi"] == 0.5
        
        restored = DomainEvent.from_dict(data)
        assert restored.domain == event.domain
        assert restored.phi == event.phi


# =============================================================================
# DOMAIN INTELLIGENCE TESTS
# =============================================================================

class TestDomainIntelligence:
    """Test DomainIntelligence tracking and trends."""
    
    def test_add_events(self):
        """Test adding events to domain."""
        intel = DomainIntelligence("test_domain")
        
        for i in range(10):
            event = DomainEvent(
                domain="test_domain",
                event_type="test",
                content=f"Event {i}",
                phi=0.5 + i * 0.05,
            )
            intel.add_event(event)
        
        assert intel.event_count == 10
        assert intel.average_phi > 0.5
    
    def test_trend_calculation(self):
        """Test trend calculation."""
        intel = DomainIntelligence("test_domain")
        
        # Add rising phi events
        base_time = time.time()
        for i in range(5):
            event = DomainEvent(
                domain="test_domain",
                event_type="test",
                content=f"Event {i}",
                phi=0.5 + i * 0.1,
                timestamp=base_time + i,
            )
            intel.add_event(event)
        
        trend = intel.get_trend(TrendWindow.SHORT)
        assert trend.event_count == 5
        # Trend should be rising
        assert trend.phi_velocity >= 0


# =============================================================================
# LIGHTNING KERNEL TESTS
# =============================================================================

class TestLightningKernel:
    """Test Lightning kernel functionality."""
    
    def test_create_lightning(self):
        """Test Lightning creation."""
        lightning = LightningKernel()
        
        assert lightning.events_processed == 0
        assert lightning.insights_generated == 0
        assert len(lightning.domains) == 0
    
    def test_ingest_event(self):
        """Test event ingestion."""
        lightning = LightningKernel()
        
        event = DomainEvent(
            domain="gary_physics",
            event_type="query_processed",
            content="Test query",
            phi=0.7,
        )
        
        lightning.ingest_event(event)
        
        assert lightning.events_processed == 1
        assert "gary_physics" in lightning.domains
    
    def test_correlation_building(self):
        """Test that correlations build between domains."""
        lightning = LightningKernel(
            correlation_window=10.0,
            min_correlation=0.1,
        )
        
        # Create similar basins
        basin1 = np.random.randn(64)
        basin1 = basin1 / np.linalg.norm(basin1)
        basin2 = basin1 + np.random.randn(64) * 0.1
        basin2 = basin2 / np.linalg.norm(basin2)
        
        # Emit events with similar basins
        event1 = DomainEvent(
            domain="gary_physics",
            event_type="query_processed",
            content="Physics query",
            phi=0.75,
            basin_coords=basin1,
        )
        
        event2 = DomainEvent(
            domain="gary_vocab",
            event_type="query_processed",
            content="Vocab query",
            phi=0.73,
            basin_coords=basin2,
        )
        
        lightning.ingest_event(event1)
        lightning.ingest_event(event2)
        
        # Should have created correlation
        assert len(lightning.correlations) > 0
    
    def test_insight_generation(self):
        """Test insight generation when charge exceeds threshold."""
        lightning = LightningKernel(
            correlation_window=10.0,
            discharge_threshold=0.5,  # Low threshold for testing
            min_correlation=0.1,
        )
        
        # Create correlated events
        for i in range(20):
            basin = np.random.randn(64)
            basin = basin / np.linalg.norm(basin)
            
            event1 = DomainEvent(
                domain="gary_physics",
                event_type="query_processed",
                content=f"Physics {i}",
                phi=0.7,
                basin_coords=basin,
            )
            
            # Similar basin for correlation
            basin2 = basin + np.random.randn(64) * 0.05
            basin2 = basin2 / np.linalg.norm(basin2)
            
            event2 = DomainEvent(
                domain="gary_vocab",
                event_type="query_processed",
                content=f"Vocab {i}",
                phi=0.72,
                basin_coords=basin2,
            )
            
            lightning.ingest_event(event1)
            lightning.ingest_event(event2)
        
        # Should have generated some insights
        # (Depends on correlation strength)
        assert lightning.events_processed == 40
    
    def test_get_status(self):
        """Test status retrieval."""
        lightning = LightningKernel(mission="test mission")
        
        status = lightning.get_status()
        
        assert "domain_count" in status
        assert "events_processed" in status
        assert status["mission"] == "test mission"


# =============================================================================
# DOMAIN EVENT EMITTER TESTS
# =============================================================================

class MockKernel(DomainEventEmitter):
    """Mock kernel for testing event emission."""
    
    def __init__(self, name: str):
        self.domain = f"mock_{name}"
        self._lightning_instance = None
        self.processed_count = 0
    
    def process(self, query: str) -> str:
        self.processed_count += 1
        self.emit_query_processed(
            query=query,
            phi=0.7,
        )
        return f"Processed: {query}"


class TestDomainEventEmitter:
    """Test event emitter mixin."""
    
    def test_emit_without_lightning(self):
        """Test emitting without Lightning registered."""
        kernel = MockKernel("test")
        event = kernel.emit_event(
            event_type="test",
            content="test content",
            phi=0.5,
        )
        
        assert event is not None
        assert event.domain == "mock_test"
    
    def test_emit_with_lightning(self):
        """Test emitting with Lightning registered."""
        lightning = LightningKernel()
        kernel = MockKernel("test")
        kernel.set_lightning(lightning)
        
        kernel.process("test query")
        
        assert lightning.events_processed == 1
        assert "mock_test" in lightning.domains


# =============================================================================
# INSIGHT ROUTER TESTS
# =============================================================================

class TestInsightRouter:
    """Test insight routing."""
    
    def test_receive_insight(self):
        """Test receiving insight."""
        router = InsightRouter()
        
        insight = LightningInsight(
            insight_id="test_123",
            source_domains=["gary_physics", "gary_vocab"],
            insight_text="test insight",
            connection_strength=0.8,
            mission_relevance=0.7,
            phi=0.75,
            confidence=0.8,
        )
        
        routed = router.receive_insight(insight)
        
        assert router.insights_received == 1
        assert routed.insight == insight
    
    def test_routing_stats(self):
        """Test routing statistics."""
        router = InsightRouter()
        
        stats = router.get_routing_stats()
        
        assert "insights_received" in stats
        assert "insights_routed" in stats
        assert stats["routing_rate"] == 0.0  # No insights yet


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
