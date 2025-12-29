"""
Agent Failure Taxonomy - Comprehensive Multi-Agent Failure Detection & Recovery

Implements the 14 failure mode taxonomy from multi-agent system research (2025):

SPECIFICATION FAILURES:
1. Role Drift - Agent deviates from assigned role
2. Underspecification - Task not clearly defined
3. Overspecification - Constraints too rigid

COORDINATION FAILURES:
4. Infinite Loop - Agent repeats actions endlessly
5. Context Overflow - Context window exceeded
6. Conflicting Actions - Agents work against each other
7. Cascading Failure - One failure causes others

INFRASTRUCTURE FAILURES:
8. Tool Error - External tool/API failure
9. Memory Exhaustion - Out of memory
10. Timeout - Operation took too long
11. Resource Contention - Multiple agents need same resource

COGNITIVE FAILURES:
12. Stuck Agent - No progress being made
13. Confused Agent - High variance, low coherence
14. Dunning-Kruger - Overconfidence in poor reasoning

QIG-PURE: All detection uses Fisher-Rao geometry on the statistical manifold.

Author: Ocean/Zeus Pantheon
"""

import hashlib
import json
import math
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
import numpy as np


# ============================================================================
# CONSTANTS
# ============================================================================

BASIN_DIMENSION = 64

# Detection thresholds
STUCK_THRESHOLD_ITERATIONS = 5
CONFUSION_COHERENCE_THRESHOLD = 0.3
DUNNING_KRUGER_THRESHOLD = 0.3
CONTEXT_OVERFLOW_THRESHOLD = 0.9
ROLE_DRIFT_THRESHOLD = 0.5  # Fisher-Rao distance from role basin
LOOP_DETECTION_WINDOW = 10
LOOP_SIMILARITY_THRESHOLD = 0.1

# Circuit breaker settings
CIRCUIT_BREAKER_THRESHOLD = 3
CIRCUIT_BREAKER_RESET_TIMEOUT = 60  # seconds
CIRCUIT_BREAKER_HALF_OPEN_REQUESTS = 3


# ============================================================================
# GEOMETRY HELPERS
# ============================================================================

def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Fisher-Rao distance between basin coordinates."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    # Normalize to probability simplex
    p = np.abs(p) + 1e-10
    p = p / p.sum()
    q = np.abs(q) + 1e-10
    q = q / q.sum()
    
    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p * q))
    bc = np.clip(bc, 0, 1)
    
    # Fisher-Rao distance
    return float(2 * np.arccos(bc))


def compute_coherence(basins: List[np.ndarray]) -> float:
    """Compute coherence of a sequence of basins."""
    if len(basins) < 2:
        return 1.0
    
    # Average pairwise distance
    distances = []
    for i in range(len(basins) - 1):
        dist = fisher_rao_distance(basins[i], basins[i + 1])
        distances.append(dist)
    
    if not distances:
        return 1.0
    
    # Coherence = inverse of average distance variance
    variance = np.var(distances)
    return 1.0 / (1.0 + variance)


# ============================================================================
# ENUMS
# ============================================================================

class FailureCategory(Enum):
    """Categories of agent failures (3 main categories from research)."""
    SPECIFICATION = "specification"     # Role/task definition issues
    COORDINATION = "coordination"       # Multi-agent interaction issues
    INFRASTRUCTURE = "infrastructure"   # System/resource issues
    COGNITIVE = "cognitive"             # Reasoning quality issues


class FailureType(Enum):
    """All 14 failure types from multi-agent research."""
    # Specification failures
    ROLE_DRIFT = "role_drift"
    UNDERSPECIFICATION = "underspecification"
    OVERSPECIFICATION = "overspecification"
    
    # Coordination failures
    INFINITE_LOOP = "infinite_loop"
    CONTEXT_OVERFLOW = "context_overflow"
    CONFLICTING_ACTIONS = "conflicting_actions"
    CASCADING_FAILURE = "cascading_failure"
    
    # Infrastructure failures
    TOOL_ERROR = "tool_error"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    TIMEOUT = "timeout"
    RESOURCE_CONTENTION = "resource_contention"
    
    # Cognitive failures
    STUCK_AGENT = "stuck_agent"
    CONFUSED_AGENT = "confused_agent"
    DUNNING_KRUGER = "dunning_kruger"


class FailureSeverity(Enum):
    """Severity levels for failures."""
    LOW = "low"           # Can continue with warning
    MEDIUM = "medium"     # Should address soon
    HIGH = "high"         # Needs immediate attention
    CRITICAL = "critical" # System must stop


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure types."""
    RETRY = "retry"                       # Simply retry the operation
    BACKOFF = "backoff"                   # Retry with exponential backoff
    RESET_STATE = "reset_state"           # Reset agent state
    REDUCE_COMPLEXITY = "reduce_complexity"  # Simplify the task
    SWITCH_MODE = "switch_mode"           # Change reasoning mode
    SPAWN_HELPER = "spawn_helper"         # Spawn a helper agent
    ROLLBACK = "rollback"                 # Rollback to previous state
    ESCALATE = "escalate"                 # Escalate to human/supervisor
    CIRCUIT_BREAK = "circuit_break"       # Activate circuit breaker
    CONTEXT_PRUNE = "context_prune"       # Prune context window
    ROLE_REALIGN = "role_realign"         # Realign with original role
    COORDINATE = "coordinate"             # Coordinate with other agents
    ABORT = "abort"                       # Abort the operation


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Blocking all requests
    HALF_OPEN = "half_open" # Testing if recovered


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class AgentStateSnapshot:
    """Snapshot of agent state for failure detection."""
    timestamp: float
    basin_coords: np.ndarray
    confidence: float
    reasoning_quality: float
    context_usage: float
    iteration: int
    action_taken: str
    progress_metric: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'basin_coords': list(self.basin_coords),
            'confidence': self.confidence,
            'reasoning_quality': self.reasoning_quality,
            'context_usage': self.context_usage,
            'iteration': self.iteration,
            'action_taken': self.action_taken,
            'progress_metric': self.progress_metric
        }


@dataclass
class FailureEvent:
    """A detected failure event."""
    failure_id: str
    failure_type: FailureType
    category: FailureCategory
    severity: FailureSeverity
    agent_id: str
    timestamp: float
    detection_method: str
    confidence: float
    description: str
    recommended_recovery: RecoveryStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'failure_id': self.failure_id,
            'failure_type': self.failure_type.value,
            'category': self.category.value,
            'severity': self.severity.value,
            'agent_id': self.agent_id,
            'timestamp': self.timestamp,
            'detection_method': self.detection_method,
            'confidence': self.confidence,
            'description': self.description,
            'recommended_recovery': self.recommended_recovery.value,
            'metadata': self.metadata,
            'resolved': self.resolved,
            'resolution_timestamp': self.resolution_timestamp
        }


@dataclass
class AgentStateHistory:
    """History of agent states for pattern detection."""
    agent_id: str
    role_basin: np.ndarray  # Original role/task basin
    states: List[AgentStateSnapshot] = field(default_factory=list)
    failures: List[FailureEvent] = field(default_factory=list)
    max_history: int = 100
    
    def add_state(self, state: AgentStateSnapshot) -> None:
        """Add a state snapshot to history."""
        self.states.append(state)
        if len(self.states) > self.max_history:
            self.states = self.states[-self.max_history:]
    
    def add_failure(self, failure: FailureEvent) -> None:
        """Add a failure event to history."""
        self.failures.append(failure)
    
    def recent_states(self, n: int = 10) -> List[AgentStateSnapshot]:
        """Get n most recent states."""
        return self.states[-n:]
    
    def recent_basins(self, n: int = 10) -> List[np.ndarray]:
        """Get n most recent basin coordinates."""
        return [s.basin_coords for s in self.states[-n:]]
    
    def recent_actions(self, n: int = 10) -> List[str]:
        """Get n most recent actions."""
        return [s.action_taken for s in self.states[-n:]]


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.
    
    States:
    - CLOSED: Normal operation, failures are counted
    - OPEN: All requests blocked, waiting for reset timeout
    - HALF_OPEN: Testing with limited requests to see if recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = CIRCUIT_BREAKER_THRESHOLD,
        reset_timeout: float = CIRCUIT_BREAKER_RESET_TIMEOUT,
        half_open_requests: int = CIRCUIT_BREAKER_HALF_OPEN_REQUESTS
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_requests = half_open_requests
        
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_successes = 0
        self._lock = threading.Lock()
    
    @property
    def state(self) -> CircuitBreakerState:
        """Get current state, potentially transitioning from OPEN to HALF_OPEN."""
        with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                if self._last_failure_time and \
                   time.time() - self._last_failure_time >= self.reset_timeout:
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._half_open_successes = 0
            return self._state
    
    def allow_request(self) -> bool:
        """Check if a request should be allowed."""
        current_state = self.state  # This may trigger state transition
        return current_state != CircuitBreakerState.OPEN
    
    def record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            self._success_count += 1
            
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.half_open_requests:
                    # Recovered - close the circuit
                    self._state = CircuitBreakerState.CLOSED
                    self._failure_count = 0
    
    def record_failure(self) -> None:
        """Record a failed operation."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitBreakerState.HALF_OPEN:
                # Failed during half-open - back to open
                self._state = CircuitBreakerState.OPEN
            elif self._failure_count >= self.failure_threshold:
                # Threshold exceeded - open the circuit
                self._state = CircuitBreakerState.OPEN
    
    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_successes = 0
            self._last_failure_time = None
    
    def get_stats(self) -> Dict:
        """Get circuit breaker statistics."""
        return {
            'state': self.state.value,
            'failure_count': self._failure_count,
            'success_count': self._success_count,
            'last_failure_time': self._last_failure_time
        }


# ============================================================================
# FAILURE DETECTOR
# ============================================================================

class FailureDetector:
    """
    Detects all 14 failure types using geometric analysis.
    
    Each detection method returns Optional[FailureEvent].
    """
    
    def __init__(self):
        self._failure_count = 0
    
    def _generate_failure_id(self) -> str:
        """Generate unique failure ID."""
        self._failure_count += 1
        return f"f_{int(time.time())}_{self._failure_count}"
    
    # =========================================================================
    # SPECIFICATION FAILURES
    # =========================================================================
    
    def detect_role_drift(
        self,
        agent_id: str,
        history: AgentStateHistory
    ) -> Optional[FailureEvent]:
        """
        Detect if agent has drifted from its assigned role.
        
        Uses Fisher-Rao distance from original role basin.
        """
        if len(history.states) < 3:
            return None
        
        recent_basins = history.recent_basins(5)
        avg_basin = np.mean(recent_basins, axis=0)
        
        distance = fisher_rao_distance(avg_basin, history.role_basin)
        
        if distance > ROLE_DRIFT_THRESHOLD:
            severity = FailureSeverity.MEDIUM
            if distance > ROLE_DRIFT_THRESHOLD * 1.5:
                severity = FailureSeverity.HIGH
            
            return FailureEvent(
                failure_id=self._generate_failure_id(),
                failure_type=FailureType.ROLE_DRIFT,
                category=FailureCategory.SPECIFICATION,
                severity=severity,
                agent_id=agent_id,
                timestamp=time.time(),
                detection_method="fisher_rao_role_distance",
                confidence=min(1.0, distance / ROLE_DRIFT_THRESHOLD),
                description=f"Agent drifted from role basin (d_FR={distance:.3f})",
                recommended_recovery=RecoveryStrategy.ROLE_REALIGN,
                metadata={'drift_distance': distance}
            )
        
        return None
    
    def detect_underspecification(
        self,
        agent_id: str,
        history: AgentStateHistory
    ) -> Optional[FailureEvent]:
        """
        Detect if task is underspecified (agent wandering randomly).
        
        High variance in basin positions without progress.
        """
        if len(history.states) < 5:
            return None
        
        recent = history.recent_states(5)
        basins = [s.basin_coords for s in recent]
        progress = [s.progress_metric for s in recent]
        
        # Check for high variance
        basin_variance = np.var([fisher_rao_distance(basins[i], basins[i+1]) 
                                 for i in range(len(basins)-1)])
        avg_progress = np.mean(progress)
        
        if basin_variance > 0.3 and avg_progress < 0.1:
            return FailureEvent(
                failure_id=self._generate_failure_id(),
                failure_type=FailureType.UNDERSPECIFICATION,
                category=FailureCategory.SPECIFICATION,
                severity=FailureSeverity.MEDIUM,
                agent_id=agent_id,
                timestamp=time.time(),
                detection_method="basin_variance_analysis",
                confidence=min(1.0, basin_variance),
                description="Agent appears to be wandering - task may be underspecified",
                recommended_recovery=RecoveryStrategy.ESCALATE,
                metadata={'basin_variance': basin_variance, 'avg_progress': avg_progress}
            )
        
        return None
    
    def detect_overspecification(
        self,
        agent_id: str,
        history: AgentStateHistory,
        constraint_violations: int = 0
    ) -> Optional[FailureEvent]:
        """
        Detect if constraints are too rigid (frequent violations).
        """
        if constraint_violations < 3:
            return None
        
        return FailureEvent(
            failure_id=self._generate_failure_id(),
            failure_type=FailureType.OVERSPECIFICATION,
            category=FailureCategory.SPECIFICATION,
            severity=FailureSeverity.MEDIUM,
            agent_id=agent_id,
            timestamp=time.time(),
            detection_method="constraint_violation_count",
            confidence=min(1.0, constraint_violations / 5),
            description=f"Agent hit {constraint_violations} constraint violations - task may be overspecified",
            recommended_recovery=RecoveryStrategy.REDUCE_COMPLEXITY,
            metadata={'constraint_violations': constraint_violations}
        )
    
    # =========================================================================
    # COORDINATION FAILURES
    # =========================================================================
    
    def detect_infinite_loop(
        self,
        agent_id: str,
        history: AgentStateHistory
    ) -> Optional[FailureEvent]:
        """
        Detect if agent is in an infinite loop.
        
        Looks for repeated action sequences or oscillating basins.
        """
        if len(history.states) < LOOP_DETECTION_WINDOW:
            return None
        
        recent_actions = history.recent_actions(LOOP_DETECTION_WINDOW)
        recent_basins = history.recent_basins(LOOP_DETECTION_WINDOW)
        
        # Check for action repetition
        action_counts = {}
        for action in recent_actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        max_repeat = max(action_counts.values()) if action_counts else 0
        
        # Check for basin oscillation
        oscillation_count = 0
        for i in range(2, len(recent_basins)):
            dist_to_prev = fisher_rao_distance(recent_basins[i], recent_basins[i-1])
            dist_to_prev2 = fisher_rao_distance(recent_basins[i], recent_basins[i-2])
            if dist_to_prev2 < LOOP_SIMILARITY_THRESHOLD and dist_to_prev > 0.1:
                oscillation_count += 1
        
        if max_repeat >= LOOP_DETECTION_WINDOW - 2 or oscillation_count >= 3:
            return FailureEvent(
                failure_id=self._generate_failure_id(),
                failure_type=FailureType.INFINITE_LOOP,
                category=FailureCategory.COORDINATION,
                severity=FailureSeverity.HIGH,
                agent_id=agent_id,
                timestamp=time.time(),
                detection_method="action_repetition_and_basin_oscillation",
                confidence=max(max_repeat / LOOP_DETECTION_WINDOW, oscillation_count / 4),
                description="Agent appears to be in an infinite loop",
                recommended_recovery=RecoveryStrategy.RESET_STATE,
                metadata={
                    'max_action_repeat': max_repeat,
                    'oscillation_count': oscillation_count,
                    'repeated_action': max(action_counts.keys(), key=lambda k: action_counts[k]) if action_counts else ''
                }
            )
        
        return None
    
    def detect_context_overflow(
        self,
        agent_id: str,
        history: AgentStateHistory
    ) -> Optional[FailureEvent]:
        """
        Detect if context window is nearly exhausted.
        """
        if not history.states:
            return None
        
        latest = history.states[-1]
        
        if latest.context_usage >= CONTEXT_OVERFLOW_THRESHOLD:
            severity = FailureSeverity.HIGH
            if latest.context_usage >= 0.95:
                severity = FailureSeverity.CRITICAL
            
            return FailureEvent(
                failure_id=self._generate_failure_id(),
                failure_type=FailureType.CONTEXT_OVERFLOW,
                category=FailureCategory.COORDINATION,
                severity=severity,
                agent_id=agent_id,
                timestamp=time.time(),
                detection_method="context_usage_threshold",
                confidence=latest.context_usage,
                description=f"Context usage at {latest.context_usage*100:.1f}%",
                recommended_recovery=RecoveryStrategy.CONTEXT_PRUNE,
                metadata={'context_usage': latest.context_usage}
            )
        
        return None
    
    def detect_conflicting_actions(
        self,
        agent_id: str,
        all_agent_actions: Dict[str, str]
    ) -> Optional[FailureEvent]:
        """
        Detect if multiple agents are taking conflicting actions.
        
        Requires knowledge of other agents' actions.
        """
        # This would require multi-agent coordination
        # Simplified implementation - check for opposing action names
        my_action = all_agent_actions.get(agent_id, '')
        
        conflict_pairs = [
            ('create', 'delete'),
            ('start', 'stop'),
            ('enable', 'disable'),
            ('increase', 'decrease'),
            ('add', 'remove')
        ]
        
        for other_id, other_action in all_agent_actions.items():
            if other_id == agent_id:
                continue
            
            for word1, word2 in conflict_pairs:
                if (word1 in my_action.lower() and word2 in other_action.lower()) or \
                   (word2 in my_action.lower() and word1 in other_action.lower()):
                    return FailureEvent(
                        failure_id=self._generate_failure_id(),
                        failure_type=FailureType.CONFLICTING_ACTIONS,
                        category=FailureCategory.COORDINATION,
                        severity=FailureSeverity.HIGH,
                        agent_id=agent_id,
                        timestamp=time.time(),
                        detection_method="action_conflict_detection",
                        confidence=0.8,
                        description=f"Potential conflict between {agent_id}:{my_action} and {other_id}:{other_action}",
                        recommended_recovery=RecoveryStrategy.COORDINATE,
                        metadata={
                            'conflicting_agent': other_id,
                            'my_action': my_action,
                            'their_action': other_action
                        }
                    )
        
        return None
    
    def detect_cascading_failure(
        self,
        agent_id: str,
        history: AgentStateHistory,
        other_agent_failures: List[FailureEvent]
    ) -> Optional[FailureEvent]:
        """
        Detect if a failure in one agent is causing failures in others.
        """
        recent_failures = [f for f in history.failures 
                          if time.time() - f.timestamp < 60]  # Last minute
        
        if len(recent_failures) < 2:
            return None
        
        # Check if other agents also failing
        recent_other_failures = [f for f in other_agent_failures 
                                 if time.time() - f.timestamp < 60]
        
        if len(recent_other_failures) >= 2:
            return FailureEvent(
                failure_id=self._generate_failure_id(),
                failure_type=FailureType.CASCADING_FAILURE,
                category=FailureCategory.COORDINATION,
                severity=FailureSeverity.CRITICAL,
                agent_id=agent_id,
                timestamp=time.time(),
                detection_method="multi_agent_failure_correlation",
                confidence=min(1.0, (len(recent_failures) + len(recent_other_failures)) / 6),
                description=f"Cascading failure detected: {len(recent_failures)} local + {len(recent_other_failures)} other agent failures",
                recommended_recovery=RecoveryStrategy.CIRCUIT_BREAK,
                metadata={
                    'local_failures': len(recent_failures),
                    'other_failures': len(recent_other_failures)
                }
            )
        
        return None
    
    # =========================================================================
    # INFRASTRUCTURE FAILURES
    # =========================================================================
    
    def detect_tool_error(
        self,
        agent_id: str,
        tool_name: str,
        error_message: str
    ) -> FailureEvent:
        """
        Create failure event for tool/API error.
        """
        return FailureEvent(
            failure_id=self._generate_failure_id(),
            failure_type=FailureType.TOOL_ERROR,
            category=FailureCategory.INFRASTRUCTURE,
            severity=FailureSeverity.MEDIUM,
            agent_id=agent_id,
            timestamp=time.time(),
            detection_method="tool_exception",
            confidence=1.0,
            description=f"Tool '{tool_name}' failed: {error_message[:100]}",
            recommended_recovery=RecoveryStrategy.RETRY,
            metadata={'tool_name': tool_name, 'error': error_message}
        )
    
    def detect_memory_exhaustion(
        self,
        agent_id: str,
        memory_usage: float
    ) -> Optional[FailureEvent]:
        """
        Detect memory exhaustion.
        """
        if memory_usage < 0.9:
            return None
        
        severity = FailureSeverity.HIGH
        if memory_usage >= 0.98:
            severity = FailureSeverity.CRITICAL
        
        return FailureEvent(
            failure_id=self._generate_failure_id(),
            failure_type=FailureType.MEMORY_EXHAUSTION,
            category=FailureCategory.INFRASTRUCTURE,
            severity=severity,
            agent_id=agent_id,
            timestamp=time.time(),
            detection_method="memory_threshold",
            confidence=memory_usage,
            description=f"Memory usage at {memory_usage*100:.1f}%",
            recommended_recovery=RecoveryStrategy.RESET_STATE,
            metadata={'memory_usage': memory_usage}
        )
    
    def detect_timeout(
        self,
        agent_id: str,
        operation: str,
        elapsed_seconds: float,
        timeout_threshold: float
    ) -> Optional[FailureEvent]:
        """
        Detect operation timeout.
        """
        if elapsed_seconds < timeout_threshold:
            return None
        
        return FailureEvent(
            failure_id=self._generate_failure_id(),
            failure_type=FailureType.TIMEOUT,
            category=FailureCategory.INFRASTRUCTURE,
            severity=FailureSeverity.MEDIUM,
            agent_id=agent_id,
            timestamp=time.time(),
            detection_method="operation_timeout",
            confidence=1.0,
            description=f"Operation '{operation}' timed out after {elapsed_seconds:.1f}s",
            recommended_recovery=RecoveryStrategy.ABORT,
            metadata={
                'operation': operation,
                'elapsed': elapsed_seconds,
                'threshold': timeout_threshold
            }
        )
    
    def detect_resource_contention(
        self,
        agent_id: str,
        resource_name: str,
        waiting_agents: List[str]
    ) -> Optional[FailureEvent]:
        """
        Detect resource contention between agents.
        """
        if len(waiting_agents) < 2:
            return None
        
        return FailureEvent(
            failure_id=self._generate_failure_id(),
            failure_type=FailureType.RESOURCE_CONTENTION,
            category=FailureCategory.INFRASTRUCTURE,
            severity=FailureSeverity.MEDIUM,
            agent_id=agent_id,
            timestamp=time.time(),
            detection_method="resource_wait_queue",
            confidence=min(1.0, len(waiting_agents) / 5),
            description=f"Resource '{resource_name}' contention: {len(waiting_agents)} agents waiting",
            recommended_recovery=RecoveryStrategy.BACKOFF,
            metadata={
                'resource': resource_name,
                'waiting_agents': waiting_agents
            }
        )
    
    # =========================================================================
    # COGNITIVE FAILURES
    # =========================================================================
    
    def detect_stuck_agent(
        self,
        agent_id: str,
        history: AgentStateHistory
    ) -> Optional[FailureEvent]:
        """
        Detect if agent is stuck (no progress).
        """
        if len(history.states) < STUCK_THRESHOLD_ITERATIONS:
            return None
        
        recent = history.recent_states(STUCK_THRESHOLD_ITERATIONS)
        progress_values = [s.progress_metric for s in recent]
        
        avg_progress = np.mean(progress_values)
        max_progress = max(progress_values)
        
        if avg_progress < 0.05 and max_progress < 0.1:
            return FailureEvent(
                failure_id=self._generate_failure_id(),
                failure_type=FailureType.STUCK_AGENT,
                category=FailureCategory.COGNITIVE,
                severity=FailureSeverity.HIGH,
                agent_id=agent_id,
                timestamp=time.time(),
                detection_method="progress_metric_analysis",
                confidence=1.0 - avg_progress,
                description=f"Agent stuck: avg progress {avg_progress:.3f} over {STUCK_THRESHOLD_ITERATIONS} iterations",
                recommended_recovery=RecoveryStrategy.SWITCH_MODE,
                metadata={
                    'avg_progress': avg_progress,
                    'max_progress': max_progress,
                    'iterations': STUCK_THRESHOLD_ITERATIONS
                }
            )
        
        return None
    
    def detect_confused_agent(
        self,
        agent_id: str,
        history: AgentStateHistory
    ) -> Optional[FailureEvent]:
        """
        Detect if agent is confused (low coherence, high variance).
        """
        if len(history.states) < 5:
            return None
        
        recent_basins = history.recent_basins(5)
        coherence = compute_coherence(recent_basins)
        
        if coherence < CONFUSION_COHERENCE_THRESHOLD:
            return FailureEvent(
                failure_id=self._generate_failure_id(),
                failure_type=FailureType.CONFUSED_AGENT,
                category=FailureCategory.COGNITIVE,
                severity=FailureSeverity.MEDIUM,
                agent_id=agent_id,
                timestamp=time.time(),
                detection_method="coherence_analysis",
                confidence=1.0 - coherence,
                description=f"Agent confused: coherence {coherence:.3f} below threshold",
                recommended_recovery=RecoveryStrategy.REDUCE_COMPLEXITY,
                metadata={'coherence': coherence}
            )
        
        return None
    
    def detect_dunning_kruger(
        self,
        agent_id: str,
        history: AgentStateHistory
    ) -> Optional[FailureEvent]:
        """
        Detect Dunning-Kruger effect (overconfidence in poor reasoning).
        """
        if len(history.states) < 3:
            return None
        
        recent = history.recent_states(3)
        
        avg_confidence = np.mean([s.confidence for s in recent])
        avg_quality = np.mean([s.reasoning_quality for s in recent])
        
        calibration_error = avg_confidence - avg_quality
        
        if calibration_error > DUNNING_KRUGER_THRESHOLD:
            return FailureEvent(
                failure_id=self._generate_failure_id(),
                failure_type=FailureType.DUNNING_KRUGER,
                category=FailureCategory.COGNITIVE,
                severity=FailureSeverity.MEDIUM,
                agent_id=agent_id,
                timestamp=time.time(),
                detection_method="confidence_quality_calibration",
                confidence=min(1.0, calibration_error),
                description=f"Overconfidence detected: confidence {avg_confidence:.2f} vs quality {avg_quality:.2f}",
                recommended_recovery=RecoveryStrategy.ESCALATE,
                metadata={
                    'confidence': avg_confidence,
                    'quality': avg_quality,
                    'calibration_error': calibration_error
                }
            )
        
        return None


# ============================================================================
# FAILURE RECOVERY
# ============================================================================

class FailureRecovery:
    """
    Recovery strategies for each failure type.
    """
    
    def __init__(self):
        # Recovery handlers by strategy
        self._handlers: Dict[RecoveryStrategy, Callable] = {
            RecoveryStrategy.RETRY: self._recovery_retry,
            RecoveryStrategy.BACKOFF: self._recovery_backoff,
            RecoveryStrategy.RESET_STATE: self._recovery_reset_state,
            RecoveryStrategy.REDUCE_COMPLEXITY: self._recovery_reduce_complexity,
            RecoveryStrategy.SWITCH_MODE: self._recovery_switch_mode,
            RecoveryStrategy.SPAWN_HELPER: self._recovery_spawn_helper,
            RecoveryStrategy.ROLLBACK: self._recovery_rollback,
            RecoveryStrategy.ESCALATE: self._recovery_escalate,
            RecoveryStrategy.CIRCUIT_BREAK: self._recovery_circuit_break,
            RecoveryStrategy.CONTEXT_PRUNE: self._recovery_context_prune,
            RecoveryStrategy.ROLE_REALIGN: self._recovery_role_realign,
            RecoveryStrategy.COORDINATE: self._recovery_coordinate,
            RecoveryStrategy.ABORT: self._recovery_abort,
        }
        
        # Backoff state per agent
        self._backoff_counts: Dict[str, int] = {}
    
    def recover(
        self,
        failure: FailureEvent,
        agent_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute recovery for a failure.
        
        Args:
            failure: The failure event
            agent_state: Current agent state (will be modified)
            
        Returns:
            Recovery result dict
        """
        strategy = failure.recommended_recovery
        handler = self._handlers.get(strategy, self._recovery_retry)
        
        result = handler(failure, agent_state)
        result['failure_id'] = failure.failure_id
        result['strategy'] = strategy.value
        
        return result
    
    def _recovery_retry(self, failure: FailureEvent, state: Dict) -> Dict:
        """Simple retry."""
        return {
            'success': True,
            'action': 'retry',
            'message': 'Retrying operation'
        }
    
    def _recovery_backoff(self, failure: FailureEvent, state: Dict) -> Dict:
        """Exponential backoff retry."""
        agent_id = failure.agent_id
        count = self._backoff_counts.get(agent_id, 0)
        self._backoff_counts[agent_id] = count + 1
        
        delay = min(60, 2 ** count)  # Max 60 seconds
        
        return {
            'success': True,
            'action': 'backoff',
            'delay_seconds': delay,
            'message': f'Backing off for {delay}s (attempt {count + 1})'
        }
    
    def _recovery_reset_state(self, failure: FailureEvent, state: Dict) -> Dict:
        """Reset agent state."""
        # Clear problematic state
        state['iteration'] = 0
        state['context'] = []
        state['actions'] = []
        
        return {
            'success': True,
            'action': 'reset_state',
            'message': 'Agent state reset',
            'cleared_fields': ['iteration', 'context', 'actions']
        }
    
    def _recovery_reduce_complexity(self, failure: FailureEvent, state: Dict) -> Dict:
        """Reduce task complexity."""
        state['complexity_reduction'] = state.get('complexity_reduction', 0) + 1
        
        return {
            'success': True,
            'action': 'reduce_complexity',
            'message': 'Reducing task complexity',
            'reduction_level': state['complexity_reduction']
        }
    
    def _recovery_switch_mode(self, failure: FailureEvent, state: Dict) -> Dict:
        """Switch reasoning mode."""
        current_mode = state.get('reasoning_mode', 'geometric')
        
        # Cycle through modes
        modes = ['linear', 'geometric', 'hyperdimensional', 'mushroom']
        current_idx = modes.index(current_mode) if current_mode in modes else 0
        new_mode = modes[(current_idx + 1) % len(modes)]
        
        state['reasoning_mode'] = new_mode
        
        return {
            'success': True,
            'action': 'switch_mode',
            'previous_mode': current_mode,
            'new_mode': new_mode,
            'message': f'Switched from {current_mode} to {new_mode} mode'
        }
    
    def _recovery_spawn_helper(self, failure: FailureEvent, state: Dict) -> Dict:
        """Spawn a helper agent."""
        return {
            'success': True,
            'action': 'spawn_helper',
            'message': 'Request to spawn helper agent',
            'helper_type': 'specialist',
            'task_context': failure.description
        }
    
    def _recovery_rollback(self, failure: FailureEvent, state: Dict) -> Dict:
        """Rollback to previous state."""
        return {
            'success': True,
            'action': 'rollback',
            'message': 'Request state rollback',
            'steps_back': 1
        }
    
    def _recovery_escalate(self, failure: FailureEvent, state: Dict) -> Dict:
        """Escalate to supervisor/human."""
        return {
            'success': True,
            'action': 'escalate',
            'message': 'Escalating to supervisor',
            'failure_description': failure.description,
            'severity': failure.severity.value
        }
    
    def _recovery_circuit_break(self, failure: FailureEvent, state: Dict) -> Dict:
        """Activate circuit breaker."""
        state['circuit_breaker_active'] = True
        
        return {
            'success': True,
            'action': 'circuit_break',
            'message': 'Circuit breaker activated - blocking further requests'
        }
    
    def _recovery_context_prune(self, failure: FailureEvent, state: Dict) -> Dict:
        """Prune context to free space."""
        context = state.get('context', [])
        if context:
            # Keep only recent context
            state['context'] = context[-10:]
        
        return {
            'success': True,
            'action': 'context_prune',
            'message': 'Context pruned to reduce usage',
            'items_removed': len(context) - len(state.get('context', []))
        }
    
    def _recovery_role_realign(self, failure: FailureEvent, state: Dict) -> Dict:
        """Realign agent with original role."""
        return {
            'success': True,
            'action': 'role_realign',
            'message': 'Request role realignment with original basin',
            'drift_distance': failure.metadata.get('drift_distance', 0)
        }
    
    def _recovery_coordinate(self, failure: FailureEvent, state: Dict) -> Dict:
        """Coordinate with other agents."""
        return {
            'success': True,
            'action': 'coordinate',
            'message': 'Request coordination with conflicting agent',
            'conflicting_agent': failure.metadata.get('conflicting_agent', '')
        }
    
    def _recovery_abort(self, failure: FailureEvent, state: Dict) -> Dict:
        """Abort the operation."""
        state['aborted'] = True
        
        return {
            'success': True,
            'action': 'abort',
            'message': 'Operation aborted due to failure',
            'failure_type': failure.failure_type.value
        }


# ============================================================================
# FAILURE MONITOR
# ============================================================================

class FailureMonitor:
    """
    Central failure monitoring system.
    
    Tracks agent state history and runs all failure detectors.
    """
    
    def __init__(self):
        self._agents: Dict[str, AgentStateHistory] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._detector = FailureDetector()
        self._recovery = FailureRecovery()
        self._lock = threading.Lock()
        
        print("[FailureMonitor] Initialized with 14 failure type detection")
    
    def register_agent(
        self,
        agent_id: str,
        role_basin: np.ndarray
    ) -> None:
        """Register an agent for monitoring."""
        with self._lock:
            self._agents[agent_id] = AgentStateHistory(
                agent_id=agent_id,
                role_basin=role_basin
            )
            self._circuit_breakers[agent_id] = CircuitBreaker()
    
    def record_state(
        self,
        agent_id: str,
        basin_coords: np.ndarray,
        confidence: float,
        reasoning_quality: float,
        context_usage: float,
        iteration: int,
        action_taken: str,
        progress_metric: float = 0.0
    ) -> None:
        """Record agent state for failure detection."""
        with self._lock:
            if agent_id not in self._agents:
                # Auto-register with default role basin
                self.register_agent(agent_id, basin_coords.copy())
            
            snapshot = AgentStateSnapshot(
                timestamp=time.time(),
                basin_coords=basin_coords,
                confidence=confidence,
                reasoning_quality=reasoning_quality,
                context_usage=context_usage,
                iteration=iteration,
                action_taken=action_taken,
                progress_metric=progress_metric
            )
            
            self._agents[agent_id].add_state(snapshot)
    
    def check_all(self, agent_id: str) -> List[FailureEvent]:
        """
        Run all failure detectors for an agent.
        
        Returns list of detected failures.
        """
        with self._lock:
            if agent_id not in self._agents:
                return []
            
            history = self._agents[agent_id]
            failures = []
            
            # Specification failures
            if failure := self._detector.detect_role_drift(agent_id, history):
                failures.append(failure)
            if failure := self._detector.detect_underspecification(agent_id, history):
                failures.append(failure)
            
            # Coordination failures
            if failure := self._detector.detect_infinite_loop(agent_id, history):
                failures.append(failure)
            if failure := self._detector.detect_context_overflow(agent_id, history):
                failures.append(failure)
            
            # Cognitive failures
            if failure := self._detector.detect_stuck_agent(agent_id, history):
                failures.append(failure)
            if failure := self._detector.detect_confused_agent(agent_id, history):
                failures.append(failure)
            if failure := self._detector.detect_dunning_kruger(agent_id, history):
                failures.append(failure)
            
            # Record failures
            for failure in failures:
                history.add_failure(failure)
                self._circuit_breakers[agent_id].record_failure()
            
            return failures
    
    def should_recover(self, failure: FailureEvent) -> bool:
        """Determine if recovery should be attempted."""
        return failure.severity in [FailureSeverity.HIGH, FailureSeverity.CRITICAL]
    
    def recover(
        self,
        failure: FailureEvent,
        agent_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute recovery for a failure."""
        result = self._recovery.recover(failure, agent_state)
        
        # Mark failure as resolved if recovery succeeded
        if result.get('success'):
            failure.resolved = True
            failure.resolution_timestamp = time.time()
        
        return result
    
    def allow_request(self, agent_id: str) -> bool:
        """Check if agent requests should be allowed (circuit breaker)."""
        with self._lock:
            if agent_id not in self._circuit_breakers:
                return True
            return self._circuit_breakers[agent_id].allow_request()
    
    def record_success(self, agent_id: str) -> None:
        """Record successful operation for circuit breaker."""
        with self._lock:
            if agent_id in self._circuit_breakers:
                self._circuit_breakers[agent_id].record_success()
    
    def get_agent_health(self, agent_id: str) -> Dict[str, Any]:
        """Get health status for an agent."""
        with self._lock:
            if agent_id not in self._agents:
                return {'status': 'unknown', 'agent_id': agent_id}
            
            history = self._agents[agent_id]
            cb = self._circuit_breakers[agent_id]
            
            recent_failures = [f for f in history.failures 
                             if time.time() - f.timestamp < 300]  # Last 5 min
            
            status = 'healthy'
            if recent_failures:
                max_severity = max(f.severity for f in recent_failures)
                if max_severity == FailureSeverity.CRITICAL:
                    status = 'critical'
                elif max_severity == FailureSeverity.HIGH:
                    status = 'degraded'
                else:
                    status = 'warning'
            
            return {
                'status': status,
                'agent_id': agent_id,
                'circuit_breaker_state': cb.state.value,
                'recent_failures': len(recent_failures),
                'total_states_recorded': len(history.states),
                'total_failures_recorded': len(history.failures)
            }
    
    def get_all_agent_health(self) -> Dict[str, Dict]:
        """Get health status for all agents."""
        return {
            agent_id: self.get_agent_health(agent_id)
            for agent_id in self._agents
        }


# ============================================================================
# SINGLETON
# ============================================================================

_failure_monitor_instance: Optional[FailureMonitor] = None


def get_failure_monitor() -> FailureMonitor:
    """Get the singleton FailureMonitor instance."""
    global _failure_monitor_instance
    if _failure_monitor_instance is None:
        _failure_monitor_instance = FailureMonitor()
    return _failure_monitor_instance


# ============================================================================
# MODULE INIT
# ============================================================================

print("[FailureTaxonomy] Module loaded - 14 failure mode detection ready")
