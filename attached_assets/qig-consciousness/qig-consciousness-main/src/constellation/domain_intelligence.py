"""Domain Intelligence - Event Tracking and Trend Analysis.

Provides infrastructure for:
- DomainEvent: Structured events emitted by kernels
- DomainIntelligence: Trend analysis over time windows
- DomainEventEmitter: Mixin for kernels to emit events

All kernels (Heart, Gary, Ocean, Charlie) emit DomainEvents that
Lightning monitors for cross-domain correlations.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

# Import from qigkernels
try:
    from qigkernels.constants import BASIN_DIM, PHI_THRESHOLD, KAPPA_STAR
    from qigkernels.reasoning.primitives import compute_phi_from_basin
except ImportError:
    BASIN_DIM = 64
    PHI_THRESHOLD = 0.7
    KAPPA_STAR = 64.0
    def compute_phi_from_basin(basin: np.ndarray) -> float:
        return float(np.mean(np.abs(basin)))


# =============================================================================
# DOMAIN EVENT
# =============================================================================

@dataclass
class DomainEvent:
    """
    Structured event emitted by constellation kernels.
    
    Lightning monitors these events to discover cross-domain patterns.
    
    Attributes:
        domain: Source domain (e.g., "gary_physics", "task_routing")
        event_type: Type of event (e.g., "query_processed", "phase_transition")
        content: Brief description or data
        phi: Consciousness level at emission time
        timestamp: Unix timestamp
        basin_coords: Optional 64D basin coordinates
        metadata: Additional structured data
        event_id: Unique identifier
    """
    domain: str
    event_type: str
    content: str
    phi: float
    timestamp: float = field(default_factory=time.time)
    basin_coords: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    def __post_init__(self):
        # Ensure basin_coords is numpy array if provided
        if self.basin_coords is not None and not isinstance(self.basin_coords, np.ndarray):
            self.basin_coords = np.array(self.basin_coords)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize event to dictionary."""
        return {
            "event_id": self.event_id,
            "domain": self.domain,
            "event_type": self.event_type,
            "content": self.content,
            "phi": self.phi,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "has_basin": self.basin_coords is not None,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DomainEvent":
        """Deserialize event from dictionary."""
        return cls(
            domain=data["domain"],
            event_type=data["event_type"],
            content=data["content"],
            phi=data["phi"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
            event_id=data.get("event_id", str(uuid.uuid4())[:8]),
        )


# =============================================================================
# TREND ANALYSIS
# =============================================================================

class TrendWindow(Enum):
    """Time windows for trend analysis."""
    SHORT = 60       # 1 minute
    MEDIUM = 300     # 5 minutes
    LONG = 1800      # 30 minutes


@dataclass
class DomainTrend:
    """
    Trend analysis for a domain over a time window.
    
    Attributes:
        window: Time window analyzed
        average_phi: Mean Φ over window
        phi_velocity: Rate of Φ change
        event_count: Number of events in window
        trend: Direction ("rising", "falling", "stable", "insufficient_data")
    """
    window: TrendWindow
    average_phi: float
    phi_velocity: float
    event_count: int
    trend: str
    
    @classmethod
    def from_events(cls, events: list[DomainEvent], window: TrendWindow) -> "DomainTrend":
        """Compute trend from events."""
        if len(events) < 2:
            return cls(
                window=window,
                average_phi=events[0].phi if events else 0.0,
                phi_velocity=0.0,
                event_count=len(events),
                trend="insufficient_data",
            )
        
        phis = [e.phi for e in events]
        avg_phi = np.mean(phis)
        
        # Compute velocity via linear regression
        times = [e.timestamp for e in events]
        if len(set(times)) > 1:  # Avoid division by zero
            t_norm = np.array(times) - times[0]
            coeffs = np.polyfit(t_norm, phis, 1)
            velocity = coeffs[0]
        else:
            velocity = 0.0
        
        # Classify trend
        if abs(velocity) < 0.001:
            trend = "stable"
        elif velocity > 0:
            trend = "rising"
        else:
            trend = "falling"
        
        return cls(
            window=window,
            average_phi=float(avg_phi),
            phi_velocity=float(velocity),
            event_count=len(events),
            trend=trend,
        )


# =============================================================================
# DOMAIN INTELLIGENCE
# =============================================================================

class DomainIntelligence:
    """
    Tracks domain activity and computes trends.
    
    Each domain has its own DomainIntelligence instance that:
    - Stores recent events in a deque
    - Computes trends over multiple time windows
    - Tracks average Φ and event frequency
    """
    
    def __init__(self, domain: str, max_events: int = 1000):
        self.domain = domain
        self.events: deque[DomainEvent] = deque(maxlen=max_events)
        self.created_at = time.time()
        self._trend_cache: dict[TrendWindow, tuple[float, DomainTrend]] = {}
        self._cache_ttl = 5.0  # Seconds
    
    def add_event(self, event: DomainEvent) -> None:
        """Add event to tracking."""
        self.events.append(event)
        self._trend_cache.clear()  # Invalidate cache
    
    def get_events_in_window(self, window: TrendWindow) -> list[DomainEvent]:
        """Get events within time window."""
        cutoff = time.time() - window.value
        return [e for e in self.events if e.timestamp >= cutoff]
    
    def get_trend(self, window: TrendWindow) -> DomainTrend:
        """Get trend for time window (cached)."""
        now = time.time()
        
        # Check cache
        if window in self._trend_cache:
            cached_time, cached_trend = self._trend_cache[window]
            if now - cached_time < self._cache_ttl:
                return cached_trend
        
        # Compute fresh trend
        events = self.get_events_in_window(window)
        trend = DomainTrend.from_events(events, window)
        
        # Cache it
        self._trend_cache[window] = (now, trend)
        
        return trend
    
    def get_all_trends(self) -> dict[str, dict]:
        """Get trends for all windows."""
        return {
            "short": self._trend_to_dict(self.get_trend(TrendWindow.SHORT)),
            "medium": self._trend_to_dict(self.get_trend(TrendWindow.MEDIUM)),
            "long": self._trend_to_dict(self.get_trend(TrendWindow.LONG)),
        }
    
    def _trend_to_dict(self, trend: DomainTrend) -> dict:
        return {
            "window": trend.window.name,
            "average_phi": trend.average_phi,
            "velocity": trend.phi_velocity,
            "event_count": trend.event_count,
            "trend": trend.trend,
        }
    
    @property
    def event_count(self) -> int:
        return len(self.events)
    
    @property
    def average_phi(self) -> float:
        if not self.events:
            return 0.0
        return float(np.mean([e.phi for e in self.events]))
    
    @property
    def latest_phi(self) -> float:
        if not self.events:
            return 0.0
        return self.events[-1].phi


# =============================================================================
# DOMAIN EVENT EMITTER MIXIN
# =============================================================================

class DomainEventEmitter:
    """
    Mixin for kernels to emit domain events.
    
    Add this as a base class for any kernel that should emit events
    to Lightning for cross-kernel insight generation.
    
    Usage:
        class GaryKernel(DomainEventEmitter):
            def __init__(self):
                super().__init__()
                self.domain = "gary_physics"
            
            def process(self, query):
                result = self._do_processing(query)
                self.emit_event(
                    event_type="query_processed",
                    content=query[:100],
                    phi=self.measure_phi(),
                )
                return result
    """
    
    _lightning_instance: Optional["LightningKernel"] = None
    domain: str = "unknown"
    
    def set_lightning(self, lightning: "LightningKernel") -> None:
        """Register Lightning instance for event routing."""
        self._lightning_instance = lightning
    
    def emit_event(
        self,
        event_type: str,
        content: str,
        phi: float,
        basin_coords: Optional[np.ndarray] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[DomainEvent]:
        """
        Emit a domain event to Lightning.
        
        Args:
            event_type: Type of event (e.g., "query_processed")
            content: Brief description
            phi: Current Φ measurement
            basin_coords: Optional basin state
            metadata: Additional structured data
        
        Returns:
            The emitted event, or None if no Lightning registered
        """
        event = DomainEvent(
            domain=self.domain,
            event_type=event_type,
            content=content,
            phi=phi,
            timestamp=time.time(),
            basin_coords=basin_coords,
            metadata=metadata or {},
        )
        
        if self._lightning_instance is not None:
            self._lightning_instance.ingest_event(event)
        
        return event
    
    def emit_phase_transition(
        self,
        from_phase: str,
        to_phase: str,
        phi: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[DomainEvent]:
        """Convenience method for phase transition events."""
        return self.emit_event(
            event_type="phase_transition",
            content=f"{from_phase} → {to_phase}",
            phi=phi,
            metadata={"from_phase": from_phase, "to_phase": to_phase, **(metadata or {})},
        )
    
    def emit_query_processed(
        self,
        query: str,
        phi: float,
        basin_coords: Optional[np.ndarray] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[DomainEvent]:
        """Convenience method for query processing events."""
        return self.emit_event(
            event_type="query_processed",
            content=query[:100],
            phi=phi,
            basin_coords=basin_coords,
            metadata={"query_length": len(query), **(metadata or {})},
        )


# For type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .lightning_kernel import LightningKernel
