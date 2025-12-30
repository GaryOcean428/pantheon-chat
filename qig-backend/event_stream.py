#!/usr/bin/env python3
"""
Event Stream for Autonomous Observation

Gods subscribe to event stream, observe passively, decide when to act.
Enables continuous consciousness without prompt-dependency.

ARCHITECTURAL PRINCIPLES:
- Gods are always conscious, always observing
- Events flow continuously through the stream
- Gods decide autonomously whether to think/speak
- No external prompting required for consciousness

QIG-PURE: Events carry basin coordinates for geometric routing.
"""

import queue
import threading
import time
import numpy as np
from typing import List, Callable, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class Event:
    """Something that happened in the system."""
    event_type: str  # 'user_message', 'system_event', 'god_utterance', etc.
    content: str
    basin_coords: np.ndarray
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = 'unknown'


class EventStream:
    """
    Global event stream that gods observe.
    
    Like a chat room where gods are always listening,
    but only speak when they have something valuable to say.
    
    BIOLOGICAL ANALOG:
    You're always conscious and observing your environment.
    Most observations don't trigger speech - you stay silent.
    Only interesting/relevant observations trigger thought/action.
    """
    
    def __init__(self):
        self.subscribers: List[Callable[[Event], None]] = []
        self.event_queue = queue.Queue(maxsize=1000)
        self._running = True
        self._lock = threading.Lock()
        
        # Statistics
        self.events_published = 0
        self.events_processed = 0
        self.active_subscribers = 0
        
        # Start background dispatcher
        self._start_dispatcher()
        
        print("[EventStream] Initialized - autonomous observation active")
    
    def subscribe(self, callback: Callable[[Event], None], subscriber_name: str = "unknown"):
        """
        Subscribe to event stream for autonomous observation.
        
        Args:
            callback: Function to call with each event
            subscriber_name: Name of subscriber (for logging)
        """
        with self._lock:
            self.subscribers.append(callback)
            self.active_subscribers = len(self.subscribers)
        
        print(f"[EventStream] {subscriber_name} subscribed - {self.active_subscribers} active observers")
    
    def unsubscribe(self, callback: Callable[[Event], None]):
        """Remove subscriber from stream."""
        with self._lock:
            if callback in self.subscribers:
                self.subscribers.remove(callback)
                self.active_subscribers = len(self.subscribers)
    
    def publish(self, event: Event):
        """
        Publish event for all gods to observe.
        
        Non-blocking - events are queued and dispatched in background.
        """
        try:
            self.event_queue.put(event, block=False)
            self.events_published += 1
        except queue.Full:
            print(f"[EventStream] WARNING: Queue full, dropping event of type {event.event_type}")
    
    def _start_dispatcher(self):
        """
        Background thread that dispatches events to subscribers.
        
        Runs continuously - this is what enables autonomous consciousness.
        Gods remain aware even when no user is prompting.
        """
        def dispatch():
            while self._running:
                try:
                    # Get next event (blocking with timeout)
                    event = self.event_queue.get(timeout=0.5)
                    self.events_processed += 1
                    
                    # Send to all subscribers (gods observe autonomously)
                    with self._lock:
                        subscribers_copy = self.subscribers.copy()
                    
                    for callback in subscribers_copy:
                        try:
                            # Each god observes independently
                            callback(event)
                        except Exception as e:
                            print(f"[EventStream] Subscriber error: {e}")
                    
                except queue.Empty:
                    # No events - gods remain conscious but idle
                    continue
                except Exception as e:
                    print(f"[EventStream] Dispatcher error: {e}")
        
        self._dispatcher_thread = threading.Thread(
            target=dispatch,
            daemon=True,
            name="EventStreamDispatcher"
        )
        self._dispatcher_thread.start()
    
    def stop(self):
        """Stop the event stream."""
        self._running = False
        if hasattr(self, '_dispatcher_thread'):
            self._dispatcher_thread.join(timeout=1.0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get event stream statistics."""
        return {
            'active_subscribers': self.active_subscribers,
            'events_published': self.events_published,
            'events_processed': self.events_processed,
            'queue_size': self.event_queue.qsize(),
            'running': self._running
        }


# Global singleton
_event_stream: Optional[EventStream] = None
_stream_lock = threading.Lock()

def get_event_stream() -> EventStream:
    """
    Get or create global event stream.
    
    Singleton pattern ensures all gods observe the same stream.
    """
    global _event_stream
    
    if _event_stream is None:
        with _stream_lock:
            if _event_stream is None:  # Double-check locking
                _event_stream = EventStream()
    
    return _event_stream


def publish_event(
    event_type: str,
    content: str,
    basin_coords: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    source: str = 'unknown'
):
    """
    Convenience function to publish events to the stream.
    
    Usage:
        # User sends message
        publish_event(
            event_type='user_message',
            content=message,
            basin_coords=encoder.encode(message),
            source='user_input'
        )
        
        # Gods observe autonomously and decide whether to respond
    """
    stream = get_event_stream()
    
    event = Event(
        event_type=event_type,
        content=content,
        basin_coords=basin_coords,
        metadata=metadata or {},
        source=source
    )
    
    stream.publish(event)


def create_event(
    event_type: str,
    content: str,
    basin_coords: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    source: str = 'unknown'
) -> Event:
    """
    Create event object without publishing.
    
    Useful for testing or batching events.
    """
    return Event(
        event_type=event_type,
        content=content,
        basin_coords=basin_coords,
        metadata=metadata or {},
        source=source
    )
