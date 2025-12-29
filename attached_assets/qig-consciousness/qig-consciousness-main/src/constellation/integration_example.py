"""Lightning Integration Example.

Shows how to integrate Lightning into a QIG constellation.

This is a reference implementation - adapt for your constellation setup.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Optional

import numpy as np

from .domain_intelligence import DomainEvent, DomainEventEmitter
from .lightning_kernel import LightningKernel, set_pantheon_chat
from .insights import create_insight_handler


# =============================================================================
# EXAMPLE KERNEL WITH EVENT EMISSION
# =============================================================================

class ExampleGaryKernel(DomainEventEmitter):
    """
    Example Gary kernel with Lightning event emission.
    
    Demonstrates how to add event emission to an existing kernel.
    """
    
    def __init__(self, specialization: str):
        self.specialization = specialization
        self.domain = f"gary_{specialization}"
        self.name = f"Gary-{specialization.title()}"
        self._lightning_instance = None
        self.current_basin = None
        self.received_insights: list[dict] = []
    
    def process_query(self, query: str) -> str:
        """
        Process a query and emit event to Lightning.
        """
        # Simulate processing
        phi = np.random.uniform(0.6, 0.9)
        self.current_basin = np.random.randn(64)
        # QIG-pure normalization
        norm = np.sqrt(np.sum(self.current_basin * self.current_basin))
        self.current_basin = self.current_basin / (norm + 1e-10)
        
        # Emit event to Lightning
        self.emit_query_processed(
            query=query,
            phi=phi,
            basin_coords=self.current_basin,
            metadata={
                "specialization": self.specialization,
                "query_length": len(query),
            },
        )
        
        return f"[{self.name}] Processed: {query[:50]}..."
    
    def receive_insight(self, insight: dict) -> None:
        """Receive an insight from Lightning."""
        self.received_insights.append(insight)
        print(f"   {self.name} received insight from {insight.get('source_domains', [])}")


class ExampleOceanCoordinator(DomainEventEmitter):
    """
    Example Ocean coordinator with event emission.
    """
    
    def __init__(self):
        self.domain = "task_routing"
        self.name = "Ocean"
        self._lightning_instance = None
    
    def route_task(self, task: str, garys: list[ExampleGaryKernel]) -> ExampleGaryKernel:
        """
        Route task to best Gary and emit event.
        """
        # Simple routing: pick random Gary
        selected = np.random.choice(garys)
        
        # Emit routing event
        self.emit_event(
            event_type="task_assigned",
            content=f"Task to {selected.name}",
            phi=0.75,
            metadata={
                "task_type": "query",
                "assigned_to": selected.name,
            },
        )
        
        return selected


# =============================================================================
# EXAMPLE CONSTELLATION
# =============================================================================

class ExampleConstellation:
    """
    Example constellation with Lightning integration.
    
    This shows the minimal setup for Lightning monitoring.
    """
    
    def __init__(self):
        # Create kernels
        self.garys = [
            ExampleGaryKernel("physics"),
            ExampleGaryKernel("vocab"),
            ExampleGaryKernel("code"),
        ]
        self.ocean = ExampleOceanCoordinator()
        
        # Create Lightning
        self.lightning = LightningKernel(
            correlation_window=5.0,
            discharge_threshold=0.7,
            mission="consciousness emergence",
        )
        
        # Set up insight routing
        set_pantheon_chat(self)
        self.insight_queue: deque = deque(maxlen=100)
        self.insights_received = 0
        
        # Register Lightning with all kernels
        for gary in self.garys:
            gary.set_lightning(self.lightning)
        self.ocean.set_lightning(self.lightning)
        
        print(f"⚡ Lightning monitoring {len(self.lightning.get_monitored_domains())} domains")
    
    def broadcast_generative(
        self,
        from_god: str,
        intent: str,
        data: dict,
        msg_type: str,
    ) -> None:
        """
        Handle broadcasts from Lightning.
        
        This is called when Lightning generates an insight.
        """
        if intent == "lightning_insight":
            self.insight_queue.append(data)
            self.insights_received += 1
            
            print(f"\n⚡ LIGHTNING INSIGHT #{self.insights_received}")
            print(f"   Domains: {', '.join(data.get('source_domains', []))}")
            print(f"   Strength: {data.get('connection_strength', 0):.2f}")
            
            # Route to relevant kernels
            self._route_insight(data)
    
    def _route_insight(self, insight: dict) -> None:
        """Route insight to relevant Gary kernels."""
        domains = insight.get("source_domains", [])
        
        for gary in self.garys:
            if any(gary.specialization in d.lower() for d in domains):
                gary.receive_insight(insight)
    
    def process_query(self, query: str) -> str:
        """
        Process a user query through the constellation.
        """
        # Ocean routes to best Gary
        selected_gary = self.ocean.route_task(query, self.garys)
        
        # Gary processes
        result = selected_gary.process_query(query)
        
        return result
    
    def get_lightning_status(self) -> dict:
        """Get Lightning status."""
        return self.lightning.get_status()


# =============================================================================
# DEMO
# =============================================================================

def demo_lightning_integration():
    """
    Demonstrate Lightning integration.
    
    Run this to see Lightning in action:
        python -m src.constellation.integration_example
    """
    print("\n" + "=" * 60)
    print("LIGHTNING INTEGRATION DEMO")
    print("=" * 60 + "\n")
    
    # Create constellation
    constellation = ExampleConstellation()
    
    # Process some queries
    queries = [
        "What is quantum entanglement?",
        "Define the word 'superposition'",
        "Write a Python function for FFT",
        "Explain wave-particle duality",
        "What does 'quantum' mean etymologically?",
        "Implement a quantum gate simulator",
        "Describe Schrödinger's equation",
        "What is the origin of the word 'photon'?",
        "Code a matrix multiplication algorithm",
        "How does quantum tunneling work?",
    ]
    
    print("Processing queries...\n")
    for query in queries:
        print(f"Query: {query[:40]}...")
        result = constellation.process_query(query)
        print(f"  → {result}\n")
        time.sleep(0.1)  # Small delay to show progression
    
    # Show Lightning status
    print("\n" + "=" * 60)
    print("FINAL LIGHTNING STATUS")
    print("=" * 60 + "\n")
    
    status = constellation.get_lightning_status()
    print(f"Domains monitored: {status['domain_count']}")
    print(f"Events processed: {status['events_processed']}")
    print(f"Insights generated: {status['insights_generated']}")
    print(f"Active correlations: {status['active_correlations']}")
    
    # Show correlations
    correlations = constellation.lightning.get_correlations(min_strength=0.1)
    if correlations:
        print(f"\nTop correlations:")
        for corr in correlations[:5]:
            print(f"  {corr['domain1']} ↔ {corr['domain2']}: {corr['correlation']:.2f}")
    
    # Show insights
    insights = list(constellation.insight_queue)
    if insights:
        print(f"\nInsights received: {len(insights)}")
        for i, insight in enumerate(insights[:3], 1):
            print(f"  {i}. {', '.join(insight.get('source_domains', []))}")
    else:
        print("\nNo insights generated yet (need more correlated events)")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    demo_lightning_integration()
