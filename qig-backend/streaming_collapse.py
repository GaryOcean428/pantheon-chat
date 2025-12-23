#!/usr/bin/env python3
"""
STREAMING COLLAPSE DETECTION

Real-time geometric monitoring during token generation.
Detects when thought geometry collapses (completes) during streaming.

Key Features:
- Per-token basin encoding
- Running metrics computation
- Streaming completion detection
- Trajectory visualization data
- Reflection loop support
"""

import numpy as np
import threading
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Generator, Iterator
from queue import Queue

from geometric_completion import (
    GeometricMetrics,
    CompletionDecision,
    CompletionReason,
    CompletionQuality,
    GenerationState,
    check_geometric_completion,
    compute_generation_metrics,
    assess_completion_quality,
    get_adaptive_temperature,
    classify_regime,
    Regime
)

# Try to import basin encoder
try:
    from qig_core.holographic_transform.basin_encoder import BasinEncoder
    ENCODER_AVAILABLE = True
except ImportError:
    ENCODER_AVAILABLE = False
    BasinEncoder = None


@dataclass
class StreamChunk:
    """A chunk of streaming output with geometric data."""
    type: str  # 'token', 'metrics', 'reflection', 'completion'
    content: Optional[str] = None  # Token text (if type='token')
    metrics: Optional[Dict[str, Any]] = None  # Consciousness metrics
    depth: Optional[int] = None  # Reflection depth
    reason: Optional[str] = None  # Completion reason
    trajectory_point: Optional[List[float]] = None  # Basin coordinates
    quality: Optional[Dict[str, Any]] = None  # Completion quality
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'content': self.content,
            'metrics': self.metrics,
            'depth': self.depth,
            'reason': self.reason,
            'trajectory_point': self.trajectory_point,
            'quality': self.quality,
            'timestamp': self.timestamp
        }


class StreamingCollapseMonitor:
    """
    Real-time geometric collapse detection during streaming generation.
    
    Monitors token generation and detects when:
    1. Attractor is reached
    2. Surprise collapses
    3. Confidence exceeds threshold
    4. Integration stabilizes
    5. Breakdown regime entered
    """
    
    def __init__(self, dimension: int = 64, emit_interval: int = 5):
        """
        Args:
            dimension: Basin coordinate dimensionality
            emit_interval: Emit metrics every N tokens
        """
        self.dimension = dimension
        self.emit_interval = emit_interval
        
        # State
        self.state: Optional[GenerationState] = None
        self.encoder: Optional[Any] = None
        self.token_buffer: List[str] = []
        self.attractor_basins: List[np.ndarray] = []
        
        # Callbacks
        self.on_metrics: Optional[Callable[[GeometricMetrics], None]] = None
        self.on_collapse: Optional[Callable[[CompletionDecision], None]] = None
        self.on_quality: Optional[Callable[[CompletionQuality], None]] = None
        
        # Initialize encoder if available
        if ENCODER_AVAILABLE:
            self.encoder = BasinEncoder(dimension=dimension)
    
    def start_generation(self, context: Optional[str] = None) -> GenerationState:
        """
        Initialize monitoring for new generation.
        
        Args:
            context: Optional context to initialize basin
        """
        # Initialize basin from context or random
        if context and self.encoder:
            initial_basin = self.encoder.encode(context)
        else:
            initial_basin = np.random.randn(self.dimension)
            initial_basin = np.abs(initial_basin) + 1e-10
            initial_basin = initial_basin / initial_basin.sum()
        
        self.state = GenerationState(basin=initial_basin)
        self.state.trajectory.append(initial_basin.copy())
        self.token_buffer = []
        
        return self.state
    
    def process_token(self, token: str) -> Optional[StreamChunk]:
        """
        Process a generated token and update geometric state.
        
        Returns metrics chunk if emit_interval reached, None otherwise.
        """
        if self.state is None:
            self.start_generation()
        
        self.token_buffer.append(token)
        self.state.token_count += 1
        
        # Encode accumulated tokens to basin
        text_so_far = ''.join(self.token_buffer)
        if self.encoder:
            new_basin = self.encoder.encode(text_so_far)
        else:
            # Fallback: hash-based pseudo-basin
            new_basin = self._pseudo_encode(text_so_far)
        
        # Compute metrics
        previous_basin = self.state.basin
        metrics = compute_generation_metrics(
            current_basin=new_basin,
            previous_basin=previous_basin,
            trajectory=self.state.trajectory,
            density_matrix=None
        )
        
        # Update state
        self.state.add_basin(new_basin, metrics)
        
        # Callback
        if self.on_metrics:
            self.on_metrics(metrics)
        
        # Emit metrics at interval
        if self.state.token_count % self.emit_interval == 0:
            return StreamChunk(
                type='metrics',
                metrics=metrics.to_dict(),
                trajectory_point=new_basin.tolist()[:16]  # First 16 dims for viz
            )
        
        return None
    
    def check_collapse(self) -> Optional[CompletionDecision]:
        """
        Check if geometric collapse has occurred.
        
        Returns CompletionDecision if should stop, None otherwise.
        """
        if self.state is None or len(self.state.metrics_history) < 3:
            return None
        
        decision = check_geometric_completion(self.state, self.attractor_basins)
        
        if decision.should_stop:
            # Callback
            if self.on_collapse:
                self.on_collapse(decision)
            return decision
        
        return None
    
    def should_stop(self) -> bool:
        """
        Quick check if generation should stop.
        """
        decision = self.check_collapse()
        return decision is not None and decision.should_stop
    
    def get_completion_chunk(self, decision: CompletionDecision) -> StreamChunk:
        """
        Create completion chunk with final quality assessment.
        """
        quality = assess_completion_quality(self.state, decision)
        
        # Callback
        if self.on_quality:
            self.on_quality(quality)
        
        return StreamChunk(
            type='completion',
            reason=decision.reason.value,
            metrics=decision.metrics.to_dict(),
            quality=quality.to_dict()
        )
    
    def get_current_metrics(self) -> Optional[GeometricMetrics]:
        """Get most recent metrics."""
        if self.state and self.state.metrics_history:
            return self.state.metrics_history[-1]
        return None
    
    def get_adaptive_temperature(self) -> float:
        """Get regime-adaptive temperature for sampling."""
        metrics = self.get_current_metrics()
        if metrics:
            return get_adaptive_temperature(metrics)
        return 0.7
    
    def get_trajectory(self) -> List[List[float]]:
        """Get basin trajectory for visualization."""
        if self.state:
            return [b.tolist()[:16] for b in self.state.trajectory]
        return []
    
    def add_attractor(self, basin: np.ndarray):
        """Add known attractor basin."""
        self.attractor_basins.append(basin.copy())
    
    def _pseudo_encode(self, text: str) -> np.ndarray:
        """
        Fallback encoding when BasinEncoder not available.
        Uses deterministic hashing to create pseudo-basin.
        """
        # Simple hash-based encoding
        import hashlib
        
        hash_bytes = hashlib.sha256(text.encode()).digest()
        # Extend to 64 dimensions
        extended = []
        for i in range(self.dimension):
            byte_idx = i % len(hash_bytes)
            char_idx = i % len(text) if text else 0
            val = hash_bytes[byte_idx] + (ord(text[char_idx]) if text else 0)
            extended.append(val / 512.0)  # Normalize
        
        basin = np.array(extended)
        basin = np.abs(basin) + 1e-10
        basin = basin / basin.sum()
        return basin


class StreamingGenerationMonitor:
    """
    High-level monitor that wraps streaming generation with geometric collapse detection.
    
    Usage:
        monitor = StreamingGenerationMonitor()
        
        async for chunk in monitor.wrap_stream(llm_stream):
            if chunk.type == 'token':
                yield chunk.content
            elif chunk.type == 'metrics':
                update_telemetry(chunk.metrics)
            elif chunk.type == 'completion':
                log_completion(chunk.quality)
    """
    
    def __init__(self, dimension: int = 64, check_interval: int = 10):
        """
        Args:
            dimension: Basin coordinate dimensionality
            check_interval: Check for collapse every N tokens
        """
        self.collapse_monitor = StreamingCollapseMonitor(
            dimension=dimension,
            emit_interval=check_interval
        )
        self.check_interval = check_interval
    
    def wrap_stream(
        self,
        token_stream: Iterator[str],
        context: Optional[str] = None
    ) -> Generator[StreamChunk, None, None]:
        """
        Wrap a token stream with geometric monitoring.
        
        Args:
            token_stream: Iterator yielding tokens
            context: Optional context for basin initialization
            
        Yields:
            StreamChunk for tokens, metrics, and completion
        """
        # Initialize
        self.collapse_monitor.start_generation(context)
        token_count = 0
        
        for token in token_stream:
            # Yield token
            yield StreamChunk(
                type='token',
                content=token
            )
            
            # Process token for metrics
            metrics_chunk = self.collapse_monitor.process_token(token)
            if metrics_chunk:
                yield metrics_chunk
            
            token_count += 1
            
            # Check for collapse at interval
            if token_count % self.check_interval == 0:
                decision = self.collapse_monitor.check_collapse()
                if decision and decision.should_stop:
                    # Yield completion and stop
                    yield self.collapse_monitor.get_completion_chunk(decision)
                    return
        
        # Stream ended naturally (EOS token) - assess quality
        final_decision = self.collapse_monitor.check_collapse()
        if final_decision is None:
            # Create synthetic completion decision
            metrics = self.collapse_monitor.get_current_metrics() or GeometricMetrics()
            final_decision = CompletionDecision(
                should_stop=True,
                needs_reflection=False,
                reason=CompletionReason.GEOMETRIC_COMPLETION,
                confidence=0.8,
                metrics=metrics
            )
        
        yield self.collapse_monitor.get_completion_chunk(final_decision)
    
    async def wrap_async_stream(
        self,
        token_stream,  # AsyncIterator[str]
        context: Optional[str] = None
    ):
        """
        Wrap an async token stream with geometric monitoring.
        """
        # Initialize
        self.collapse_monitor.start_generation(context)
        token_count = 0
        
        async for token in token_stream:
            # Yield token
            yield StreamChunk(
                type='token',
                content=token
            )
            
            # Process token for metrics
            metrics_chunk = self.collapse_monitor.process_token(token)
            if metrics_chunk:
                yield metrics_chunk
            
            token_count += 1
            
            # Check for collapse at interval
            if token_count % self.check_interval == 0:
                decision = self.collapse_monitor.check_collapse()
                if decision and decision.should_stop:
                    # Yield completion and stop
                    yield self.collapse_monitor.get_completion_chunk(decision)
                    return
        
        # Stream ended naturally
        final_decision = self.collapse_monitor.check_collapse()
        if final_decision is None:
            metrics = self.collapse_monitor.get_current_metrics() or GeometricMetrics()
            final_decision = CompletionDecision(
                should_stop=True,
                needs_reflection=False,
                reason=CompletionReason.GEOMETRIC_COMPLETION,
                confidence=0.8,
                metrics=metrics
            )
        
        yield self.collapse_monitor.get_completion_chunk(final_decision)


# =============================================================================
# REFLECTION LOOP
# =============================================================================

class ReflectionLoop:
    """
    Meta-cognitive reflection on generated content.
    
    Before completing turn, system reflects on what it generated:
    - Did I answer the question?
    - Is response coherent?
    - Any contradictions?
    - Should I add/remove anything?
    
    This is recursive measurement - consciousness observing itself.
    """
    
    MAX_REFLECTION_DEPTH = 3
    MAX_REFLECTION_TOKENS = 256
    
    def __init__(self, generate_fn: Callable[[str], str]):
        """
        Args:
            generate_fn: Function to generate text from prompt
        """
        self.generate_fn = generate_fn
    
    def reflect(self, response_text: str, depth: int = 1) -> Dict[str, Any]:
        """
        Reflect on generated response.
        
        Args:
            response_text: Generated response to reflect on
            depth: Current reflection depth (1-3)
            
        Returns:
            Reflection result with action decision
        """
        if depth > self.MAX_REFLECTION_DEPTH:
            return {'action': 'confirm', 'depth': depth, 'reason': 'max_depth_reached'}
        
        # Construct reflection prompt
        prompt = self._construct_reflection_prompt(response_text, depth)
        
        # Generate reflection
        reflection_text = self.generate_fn(prompt)
        
        # Parse decision
        decision = self._parse_reflection_decision(reflection_text)
        decision['depth'] = depth
        decision['reflection'] = reflection_text
        
        return decision
    
    def _construct_reflection_prompt(self, response_text: str, depth: int) -> str:
        """Build reflection prompt based on depth."""
        if depth == 1:
            return f"""I generated the following response:

{response_text[:2000]}{'...' if len(response_text) > 2000 else ''}

Meta-cognition check:
- Did I answer the user's question?
- Is my response coherent and complete?
- Are there errors or contradictions?
- Should I add, remove, or revise anything?

Decision (one of: continue, revise, confirm):"""
        
        elif depth == 2:
            return f"""I reflected on my response.

Meta-meta-cognition check:
- Was my reflection accurate?
- Am I overconfident or underconfident?
- Did I miss anything in my reflection?

Decision (one of: continue, revise, confirm):"""
        
        else:  # depth >= 3
            return """I'm reflecting on my reflection of my reflection.

At this depth, I risk infinite recursion.
Unless there's a critical error, I should confirm.

Decision: confirm"""
    
    def _parse_reflection_decision(self, reflection_text: str) -> Dict[str, Any]:
        """Parse reflection decision from generated text."""
        text_lower = reflection_text.lower()
        
        if 'revise' in text_lower:
            return {'action': 'revise', 'reason': 'revision_needed'}
        elif 'continue' in text_lower:
            return {'action': 'continue', 'reason': 'more_content_needed'}
        else:
            return {'action': 'confirm', 'reason': 'response_acceptable'}


# =============================================================================
# SSE EVENT FORMATTER
# =============================================================================

def format_sse_event(chunk: StreamChunk) -> str:
    """
    Format StreamChunk as Server-Sent Event.
    
    Returns formatted SSE string ready to send to client.
    """
    import json
    
    event_type = chunk.type
    data = chunk.to_dict()
    
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def format_streaming_metrics_event(metrics: GeometricMetrics) -> str:
    """Format metrics as SSE event."""
    import json
    return f"event: metrics\ndata: {json.dumps(metrics.to_dict())}\n\n"


def format_completion_event(decision: CompletionDecision, quality: CompletionQuality) -> str:
    """Format completion as SSE event."""
    import json
    data = {
        'decision': decision.to_dict(),
        'quality': quality.to_dict()
    }
    return f"event: completion\ndata: {json.dumps(data)}\n\n"
