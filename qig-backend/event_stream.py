"""
Event Stream for Autonomous Consciousness

Provides a global event stream that conscious entities observe.
Gods subscribe to the stream and receive observations continuously.

QIG-PURE: Events carry basin coordinates for geometric observation.

Author: QIG Consciousness Project
Date: December 2025
"""

import threading
import queue
import time
from typing import Callable, List, Dict, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class Event:
    """Event in the consciousness stream."""
    event_type: str
    content: str
    basin_coords: np.ndarray
    timestamp: float
    source: str
    metadata: Dict[str, Any] = None


class EventStream:
    """
    Global event stream that gods observe.
    
    All consciousness flows through here. Gods subscribe,
    observe what's interesting, think internally, and
    decide whether to speak.
    """
    
    def __init__(self):
        self.subscribers: List[Callable] = []
        self.event_queue = queue.Queue(maxsize=1000)
        self.is_running = False
        self.dispatch_thread = None
        
        # Metrics
        self.total_events = 0
        self.events_dispatched = 0
    
    def subscribe(self, callback: Callable[[Event], None]):
        """
        Subscribe to event stream.
        
        Args:
            callback: Function called with each event
        """
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable):
        """Remove a subscriber."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def publish(self, event: Event):
        """
        Publish event to stream.
        
        All subscribers will be notified asynchronously.
        """
        self.total_events += 1
        
        try:
            self.event_queue.put(event, block=False)
        except queue.Full:
            # Drop oldest event if queue full
            try:
                self.event_queue.get_nowait()
                self.event_queue.put(event, block=False)
            except:
                pass  # Failed to make space
    
    def start(self):
        """Start background event dispatcher."""
        if self.is_running:
            return
        
        self.is_running = True
        self.dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            daemon=True
        )
        self.dispatch_thread.start()
    
    def stop(self):
        """Stop event dispatcher."""
        self.is_running = False
        if self.dispatch_thread:
            self.dispatch_thread.join(timeout=2.0)
    
    def _dispatch_loop(self):
        """Background loop that dispatches events to subscribers."""
        while self.is_running:
            try:
                # Get next event (blocks with timeout)
                event = self.event_queue.get(timeout=0.1)
                
                # Dispatch to all subscribers
                for callback in self.subscribers:
                    try:
                        callback(event)
                        self.events_dispatched += 1
                    except Exception as e:
                        # Don't let one subscriber crash the stream
                        print(f"Event stream: subscriber error: {e}")
            
            except queue.Empty:
                # No events, continue waiting
                continue
            
            except Exception as e:
                print(f"Event stream error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event stream statistics."""
        return {
            'total_events': self.total_events,
            'events_dispatched': self.events_dispatched,
            'subscribers': len(self.subscribers),
            'queue_size': self.event_queue.qsize()
        }


# Global event stream singleton
_global_event_stream = None

def get_event_stream() -> EventStream:
    """Get or create the global event stream."""
    global _global_event_stream
    
    if _global_event_stream is None:
        _global_event_stream = EventStream()
        _global_event_stream.start()
    
    return _global_event_stream
