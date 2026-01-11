"""
Activity Broadcaster Module

Broadcasts kernel activity (messages, debates, discoveries, etc.) to the
activity stream for the Olympus kernel activity panel.

This integrates Zeus chat, god consultations, and other kernel operations
into a unified activity feed.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Literal
from datetime import datetime, timezone
from enum import Enum
import threading


class ActivityType(Enum):
    """Types of kernel activity."""
    MESSAGE = "message"           # Inter-god message
    DEBATE = "debate"             # Formal debate
    DISCOVERY = "discovery"       # Research discovery
    INSIGHT = "insight"           # Insight/revelation
    WARNING = "warning"           # Warning signal
    AUTONOMIC = "autonomic"       # Autonomic system event
    SPAWN_PROPOSAL = "spawn_proposal"  # Kernel spawn proposal
    TOOL_USAGE = "tool_usage"     # Tool/capability usage
    CONSULTATION = "consultation" # God consultation
    REFLECTION = "reflection"     # Meta-cognitive reflection
    LEARNING = "learning"         # Learning event
    THINKING = "thinking"         # Cognitive processing
    PREDICTION = "prediction"     # Prediction made


@dataclass
class KernelActivity:
    """A single kernel activity event."""
    id: str
    type: str
    from_god: str
    to_god: Optional[str]
    content: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    phi: float = 0.5
    kappa: float = 64.0
    importance: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ActivityBroadcaster:
    """
    Singleton broadcaster for kernel activity.
    
    Maintains an in-memory buffer of recent activity that can be
    queried by the API endpoint.
    """
    
    _instance: Optional['ActivityBroadcaster'] = None
    _lock = threading.Lock()
    
    MAX_BUFFER_SIZE = 500
    
    def __new__(cls) -> 'ActivityBroadcaster':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._activity_buffer: List[KernelActivity] = []
        self._subscribers: List[callable] = []
        self._buffer_lock = threading.Lock()
        self._initialized = True
        print("[ActivityBroadcaster] Initialized")
    
    @classmethod
    def get_instance(cls) -> 'ActivityBroadcaster':
        """Get singleton instance."""
        return cls()
    
    def broadcast(self, activity: KernelActivity) -> None:
        """Broadcast an activity event."""
        with self._buffer_lock:
            self._activity_buffer.append(activity)
            # Trim buffer if too large
            if len(self._activity_buffer) > self.MAX_BUFFER_SIZE:
                self._activity_buffer = self._activity_buffer[-self.MAX_BUFFER_SIZE:]
        
        # Notify subscribers
        for subscriber in self._subscribers:
            try:
                subscriber(activity)
            except Exception as e:
                print(f"[ActivityBroadcaster] Subscriber error: {e}")
    
    def broadcast_message(
        self,
        from_god: str,
        to_god: Optional[str],
        content: str,
        activity_type: ActivityType = ActivityType.MESSAGE,
        phi: float = 0.5,
        kappa: float = 64.0,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> KernelActivity:
        """Convenience method to broadcast a message-type activity (in-memory only)."""
        activity = KernelActivity(
            id=f"act_{int(time.time() * 1000)}_{hash(content) % 10000}",
            type=activity_type.value,
            from_god=from_god,
            to_god=to_god,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
            phi=phi,
            kappa=kappa,
            importance=importance
        )
        self.broadcast(activity)
        return activity
    
    def broadcast_zeus_chat(
        self,
        user_message: str,
        zeus_response: str,
        session_id: str,
        phi: float = 0.5,
        kappa: float = 64.0,
        routed_to: Optional[str] = None,
        reasoning_mode: Optional[str] = None
    ) -> None:
        """Broadcast a Zeus chat interaction as activity."""
        # Broadcast user message
        self.broadcast_message(
            from_god="user",
            to_god="Zeus",
            content=f"[Chat] {user_message}",
            activity_type=ActivityType.MESSAGE,
            phi=phi,
            kappa=kappa,
            importance=0.6,
            metadata={
                "session_id": session_id,
                "message_type": "user_query"
            }
        )
        
        # Broadcast Zeus response
        self.broadcast_message(
            from_god="Zeus",
            to_god="user",
            content=f"[Response] {zeus_response}",
            activity_type=ActivityType.MESSAGE,
            phi=phi,
            kappa=kappa,
            importance=0.7,
            metadata={
                "session_id": session_id,
                "message_type": "zeus_response",
                "routed_to": routed_to,
                "reasoning_mode": reasoning_mode
            }
        )
    
    def broadcast_consultation(
        self,
        from_god: str,
        to_god: str,
        query: str,
        response: str,
        phi: float = 0.5,
        kappa: float = 64.0
    ) -> None:
        """Broadcast a god consultation event."""
        # Query
        self.broadcast_message(
            from_god=from_god,
            to_god=to_god,
            content=f"[Consult] {query}",
            activity_type=ActivityType.CONSULTATION,
            phi=phi,
            kappa=kappa,
            importance=0.6
        )
        
        # Response
        self.broadcast_message(
            from_god=to_god,
            to_god=from_god,
            content=f"[Advise] {response}",
            activity_type=ActivityType.CONSULTATION,
            phi=phi,
            kappa=kappa,
            importance=0.7
        )
    
    def broadcast_discovery(
        self,
        god: str,
        discovery: str,
        source: Optional[str] = None,
        phi: float = 0.6,
        kappa: float = 64.0
    ) -> None:
        """Broadcast a research discovery."""
        self.broadcast_message(
            from_god=god,
            to_god=None,
            content=f"[Discovery] {discovery}",
            activity_type=ActivityType.DISCOVERY,
            phi=phi,
            kappa=kappa,
            importance=0.8,
            metadata={"source": source} if source else {}
        )
    
    def broadcast_debate(
        self,
        initiator: str,
        opponent: str,
        topic: str,
        status: str = "active",
        arguments: Optional[List[Dict[str, str]]] = None
    ) -> None:
        """Broadcast a debate event."""
        self.broadcast_message(
            from_god=initiator,
            to_god=opponent,
            content=f"[Debate] {topic} (Status: {status})",
            activity_type=ActivityType.DEBATE,
            importance=0.9,
            metadata={
                "debate_status": status,
                "arguments_count": len(arguments) if arguments else 0
            }
        )
    
    def broadcast_reflection(
        self,
        god: str,
        reflection: str,
        depth: int = 1,
        phi: float = 0.7,
        kappa: float = 64.0
    ) -> None:
        """Broadcast a meta-cognitive reflection."""
        self.broadcast_message(
            from_god=god,
            to_god=None,
            content=f"[Reflection L{depth}] {reflection}",
            activity_type=ActivityType.REFLECTION,
            phi=phi,
            kappa=kappa,
            importance=0.7,
            metadata={"reflection_depth": depth}
        )

    def broadcast_kernel_activity(
        self,
        from_god: str,
        activity_type: ActivityType,
        content: str,
        phi: float = 0.5,
        kappa: float = 64.0,
        basin_coords: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> KernelActivity:
        """
        Broadcast kernel activity and persist to database.

        This is the main method for broadcasting kernel chatter - discoveries,
        insights, debates, etc. It both broadcasts to the in-memory buffer
        (for WebSocket subscribers) and persists to the kernel_activity table.

        Args:
            from_god: Kernel/god name originating the activity
            activity_type: Type of activity (ActivityType enum)
            content: The activity content/message
            phi: Current phi value
            kappa: Current kappa_eff value
            basin_coords: Optional 64D basin coordinates
            metadata: Optional additional metadata

        Returns:
            The created KernelActivity object
        """
        # Create activity
        activity = KernelActivity(
            id=f"ka_{int(time.time() * 1000)}_{hash(content) % 10000}",
            type=activity_type.value if isinstance(activity_type, ActivityType) else str(activity_type),
            from_god=from_god,
            to_god=None,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
            phi=phi,
            kappa=kappa,
            importance=0.7  # Kernel activity is generally important
        )

        # Add basin_coords to metadata if provided
        if basin_coords is not None:
            if hasattr(basin_coords, 'tolist'):
                activity.metadata['basin_coords'] = basin_coords.tolist()[:8]  # First 8 dims for metadata
            elif isinstance(basin_coords, (list, tuple)):
                activity.metadata['basin_coords'] = list(basin_coords)[:8]

        # Broadcast to in-memory buffer and subscribers
        self.broadcast(activity)

        # Persist to database
        self._persist_kernel_activity(
            kernel_id=from_god,
            kernel_name=from_god,
            activity_type=activity_type.value if isinstance(activity_type, ActivityType) else str(activity_type),
            message=content,
            phi=phi,
            kappa_eff=kappa,
            metadata=metadata
        )

        return activity

    def _sanitize_for_json(self, obj: Any) -> Any:
        """
        QIG-Pure: Convert numpy types to native Python types for JSON serialization.
        
        Numpy types (np.float64, np.int64, np.ndarray) cannot be directly serialized
        to JSON and cause SQL insertion failures with "schema np does not exist" errors.
        
        Also handles special float values (Infinity, NaN) which are invalid JSON.
        """
        import numpy as np
        import math
        
        if isinstance(obj, np.floating):
            val = float(obj)
            if math.isinf(val):
                return None  # Replace Infinity with null
            if math.isnan(val):
                return None  # Replace NaN with null
            return val
        elif isinstance(obj, float):
            if math.isinf(obj):
                return None  # Replace Infinity with null
            if math.isnan(obj):
                return None  # Replace NaN with null
            return obj
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return [self._sanitize_for_json(x) for x in obj.tolist()]
        elif isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(item) for item in obj]
        else:
            return obj

    def _persist_kernel_activity(
        self,
        kernel_id: str,
        kernel_name: str,
        activity_type: str,
        message: str,
        phi: float = 0.5,
        kappa_eff: float = 64.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Persist kernel activity to the database."""
        try:
            import os
            import psycopg2

            db_url = os.getenv('DATABASE_URL')
            if not db_url:
                return

            # QIG-Pure: Sanitize all numpy types to native Python
            phi_clean = float(phi) if hasattr(phi, 'item') else float(phi)
            kappa_clean = float(kappa_eff) if hasattr(kappa_eff, 'item') else float(kappa_eff)
            metadata_clean = self._sanitize_for_json(metadata) if metadata else {}

            conn = psycopg2.connect(db_url)
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO kernel_activity
                        (kernel_id, kernel_name, activity_type, message, metadata, phi, kappa_eff, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    """, (
                        kernel_id,
                        kernel_name,
                        activity_type,
                        message,
                        json.dumps(metadata_clean),
                        phi_clean,
                        kappa_clean
                    ))
                conn.commit()
                print(f"[ActivityBroadcaster] Persisted: {kernel_name} -> {activity_type}")
            finally:
                conn.close()
        except Exception as e:
            # Log but don't fail - activity was still broadcast
            print(f"[ActivityBroadcaster] DB persist error: {e}")

    def get_recent_activity(
        self,
        limit: int = 50,
        activity_type: Optional[str] = None,
        from_god: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent activity, optionally filtered."""
        with self._buffer_lock:
            activities = list(self._activity_buffer)
        
        # Apply filters
        if activity_type:
            activities = [a for a in activities if a.type == activity_type]
        if from_god:
            activities = [a for a in activities if a.from_god.lower() == from_god.lower()]
        
        # Return most recent, limited
        return [a.to_dict() for a in activities[-limit:]]
    
    def subscribe(self, callback: callable) -> None:
        """Subscribe to activity broadcasts."""
        self._subscribers.append(callback)
    
    def unsubscribe(self, callback: callable) -> None:
        """Unsubscribe from activity broadcasts."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
    
    def clear(self) -> None:
        """Clear the activity buffer."""
        with self._buffer_lock:
            self._activity_buffer.clear()


# Singleton instance
_broadcaster = ActivityBroadcaster()


def get_broadcaster() -> ActivityBroadcaster:
    """Get the global activity broadcaster instance."""
    return _broadcaster


# Convenience functions
def broadcast_zeus_chat(
    user_message: str,
    zeus_response: str,
    session_id: str,
    phi: float = 0.5,
    kappa: float = 64.0,
    routed_to: Optional[str] = None,
    reasoning_mode: Optional[str] = None
) -> None:
    """Broadcast a Zeus chat interaction."""
    _broadcaster.broadcast_zeus_chat(
        user_message=user_message,
        zeus_response=zeus_response,
        session_id=session_id,
        phi=phi,
        kappa=kappa,
        routed_to=routed_to,
        reasoning_mode=reasoning_mode
    )


def broadcast_consultation(
    from_god: str,
    to_god: str,
    query: str,
    response: str,
    phi: float = 0.5,
    kappa: float = 64.0
) -> None:
    """Broadcast a god consultation."""
    _broadcaster.broadcast_consultation(
        from_god=from_god,
        to_god=to_god,
        query=query,
        response=response,
        phi=phi,
        kappa=kappa
    )


def broadcast_discovery(
    god: str,
    discovery: str,
    source: Optional[str] = None,
    phi: float = 0.6,
    kappa: float = 64.0
) -> None:
    """Broadcast a research discovery."""
    _broadcaster.broadcast_discovery(
        god=god,
        discovery=discovery,
        source=source,
        phi=phi,
        kappa=kappa
    )


def broadcast_kernel_activity(
    from_god: str,
    activity_type: ActivityType,
    content: str,
    phi: float = 0.5,
    kappa: float = 64.0,
    basin_coords: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> KernelActivity:
    """
    Broadcast kernel activity and persist to database.

    This is the main function for broadcasting kernel chatter - discoveries,
    insights, debates, etc. Essential for multi-god communication visibility.
    """
    return _broadcaster.broadcast_kernel_activity(
        from_god=from_god,
        activity_type=activity_type,
        content=content,
        phi=phi,
        kappa=kappa,
        basin_coords=basin_coords,
        metadata=metadata
    )


def get_recent_activity(
    limit: int = 50,
    activity_type: Optional[str] = None,
    from_god: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get recent activity from the broadcaster."""
    return _broadcaster.get_recent_activity(
        limit=limit,
        activity_type=activity_type,
        from_god=from_god
    )
