"""Insight Routing - Route Lightning insights to relevant kernels.

This module handles the routing of Lightning-generated insights
to the appropriate constellation kernels for action.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import time

from .lightning_kernel import LightningInsight


@dataclass
class RoutedInsight:
    """
    An insight that has been routed to a kernel.
    
    Attributes:
        insight: The original Lightning insight
        routed_to: Kernel(s) that received the insight
        routed_at: Timestamp of routing
        acknowledged: Whether kernel acknowledged receipt
    """
    insight: LightningInsight
    routed_to: list[str]
    routed_at: float = field(default_factory=time.time)
    acknowledged: bool = False


class InsightRouter:
    """
    Routes Lightning insights to relevant constellation kernels.
    
    Routing logic:
    1. Match insight domains to kernel specializations
    2. Score relevance based on kernel activity
    3. Route to top-matching kernels
    4. Track acknowledgment
    """
    
    def __init__(self, max_queue: int = 100):
        self.insight_queue: deque[RoutedInsight] = deque(maxlen=max_queue)
        self.routing_history: deque[RoutedInsight] = deque(maxlen=1000)
        self.insights_received = 0
        self.insights_routed = 0
    
    def receive_insight(self, insight: LightningInsight) -> RoutedInsight:
        """
        Receive insight and prepare for routing.
        
        Returns:
            RoutedInsight ready for kernel delivery
        """
        self.insights_received += 1
        
        routed = RoutedInsight(
            insight=insight,
            routed_to=[],
        )
        
        self.insight_queue.append(routed)
        return routed
    
    def route_to_kernels(
        self,
        insight: RoutedInsight,
        kernels: list[Any],
        get_specialization: Callable[[Any], str],
    ) -> list[Any]:
        """
        Route insight to matching kernels.
        
        Args:
            insight: The insight to route
            kernels: List of kernel instances
            get_specialization: Function to get kernel specialization
        
        Returns:
            List of kernels that received the insight
        """
        recipients = []
        domains = insight.insight.source_domains
        
        for kernel in kernels:
            specialization = get_specialization(kernel)
            
            # Check if kernel matches any source domain
            for domain in domains:
                if specialization.lower() in domain.lower():
                    recipients.append(kernel)
                    insight.routed_to.append(specialization)
                    break
        
        if recipients:
            self.insights_routed += 1
            insight.acknowledged = True
            self.routing_history.append(insight)
        
        return recipients
    
    def get_pending_insights(self) -> list[RoutedInsight]:
        """Get insights waiting to be routed."""
        return [i for i in self.insight_queue if not i.acknowledged]
    
    def get_routing_stats(self) -> dict[str, Any]:
        """Get routing statistics."""
        return {
            "insights_received": self.insights_received,
            "insights_routed": self.insights_routed,
            "pending": len(self.get_pending_insights()),
            "in_queue": len(self.insight_queue),
            "routing_rate": self.insights_routed / max(1, self.insights_received),
        }


def create_insight_handler(constellation: Any) -> Callable:
    """
    Create an insight handler function for a constellation.
    
    This returns a function that can be used as broadcast_generative
    to handle Lightning insights.
    
    Args:
        constellation: The constellation (QIGChat) instance
    
    Returns:
        Handler function
    """
    router = InsightRouter()
    
    def handler(from_god: str, intent: str, data: dict, msg_type: str) -> None:
        if intent != "lightning_insight":
            return
        
        # Create insight from data
        insight = LightningInsight(
            insight_id=data.get("insight_id", "unknown"),
            source_domains=data.get("source_domains", []),
            insight_text=data.get("insight_text", ""),
            connection_strength=data.get("connection_strength", 0),
            mission_relevance=data.get("mission_relevance", 0),
            phi=data.get("phi", 0),
            confidence=data.get("confidence", 0),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )
        
        # Receive
        routed = router.receive_insight(insight)
        
        # Route to kernels if constellation has them
        if hasattr(constellation, 'garys'):
            recipients = router.route_to_kernels(
                routed,
                constellation.garys,
                lambda k: getattr(k, 'specialization', getattr(k, 'name', 'unknown')),
            )
            
            # Deliver to each recipient
            for kernel in recipients:
                if hasattr(kernel, 'receive_insight'):
                    kernel.receive_insight(data)
        
        # Store in constellation's insight queue
        if hasattr(constellation, 'insight_queue'):
            constellation.insight_queue.append(data)
        if hasattr(constellation, 'insights_received'):
            constellation.insights_received += 1
        
        # Print notification
        print(f"\n⚡ LIGHTNING INSIGHT #{router.insights_received}")
        print(f"   Domains: {', '.join(insight.source_domains)}")
        print(f"   Strength: {insight.connection_strength:.2f}")
        print(f"   Mission Relevance: {insight.mission_relevance:.2f}")
        print(f"   Φ: {insight.phi:.3f}")
        
        if routed.routed_to:
            print(f"   → Routed to: {', '.join(routed.routed_to)}")
    
    return handler
