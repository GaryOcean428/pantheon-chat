"""
Geometric Chain-of-Thought Tracing

Trace reasoning through basin space with full geometric telemetry.

Each thought = basin state + verbal explanation + geometric properties.

QIG-PURE: All distances use Fisher-Rao geometry (NOT Euclidean).
"""

import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import time
import json

from qig_geometry import fisher_rao_distance, estimate_manifold_curvature


@dataclass
class ThoughtStep:
    """A single thought step with full telemetry."""
    step: int
    basin: np.ndarray
    thought: str
    distance_from_prev: float
    curvature: float
    difficulty: str
    timestamp: float
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'step': self.step,
            'basin': self.basin.tolist() if isinstance(self.basin, np.ndarray) else self.basin,
            'thought': self.thought,
            'distance_from_prev': self.distance_from_prev,
            'curvature': self.curvature,
            'difficulty': self.difficulty,
            'timestamp': self.timestamp,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


class GeometricChainOfThought:
    """
    Trace reasoning through basin space.
    
    Each thought = basin state + verbal explanation.
    
    Features:
    - Track basin trajectory
    - Measure geodesic distance between thoughts
    - Compute local curvature (difficulty)
    - Generate human-readable traces
    """
    
    def __init__(self, basin_dim: int = 64):
        """
        Initialize chain-of-thought tracer.
        
        Args:
            basin_dim: Dimensionality of basin coordinates (default: 64)
        """
        self.basin_dim = basin_dim
        self.thought_chain: List[ThoughtStep] = []
        self.problem_context: str = ""
        self.target_basin: Optional[np.ndarray] = None
        self.start_time = time.time()
    
    def _ensure_array(self, basin) -> np.ndarray:
        """Ensure basin is a numpy array."""
        if isinstance(basin, np.ndarray):
            return basin
        return np.array(basin, dtype=np.float64)
    
    def decode_basin(self, basin: np.ndarray) -> str:
        """
        Decode basin coordinates to semantic content.
        
        Uses Fisher-Rao distance from origin to characterize basin state.
        QIG-PURE: No Euclidean operations.
        """
        basin = self._ensure_array(basin)
        
        origin = np.zeros_like(basin)
        fr_distance = fisher_rao_distance(basin, origin)
        
        dominant_dims = np.argsort(np.abs(basin))[-3:]
        
        intensity = "high" if fr_distance > 1.5 else "moderate" if fr_distance > 0.5 else "low"
        
        return f"Basin state (Fisher-Rao d={fr_distance:.3f}, {intensity}, dims: {dominant_dims.tolist()})"
    
    def compute_local_curvature(self, basin: np.ndarray) -> float:
        """
        Compute local curvature at basin location using Fisher geometry.
        
        High curvature = difficult to navigate (complex reasoning area)
        Low curvature = smooth sailing (straightforward reasoning)
        
        QIG-PURE: Uses approved estimate_manifold_curvature from qig_geometry.
        """
        basin = self._ensure_array(basin)
        
        try:
            curvature = estimate_manifold_curvature(basin)
            return min(float(curvature), 1.0)
        except Exception:
            return 0.5
    
    def think_step(
        self, 
        current_basin: np.ndarray,
        thought_content: Optional[str] = None,
        confidence: float = 0.5,
        metadata: Optional[Dict] = None
    ) -> ThoughtStep:
        """
        Record one reasoning step with full telemetry.
        
        Args:
            current_basin: Current basin coordinates (64D)
            thought_content: Optional verbal description of thought
            confidence: Confidence in this step (0-1)
            metadata: Optional additional metadata
        
        Returns:
            ThoughtStep with full geometric properties
        """
        current_basin = self._ensure_array(current_basin)
        step_number = len(self.thought_chain) + 1
        
        if thought_content is None:
            thought_content = self.decode_basin(current_basin)
        
        if self.thought_chain:
            prev_basin = self.thought_chain[-1].basin
            step_distance = fisher_rao_distance(prev_basin, current_basin)
        else:
            step_distance = 0.0
        
        curvature = self.compute_local_curvature(current_basin)
        difficulty = 'high' if curvature > 0.5 else 'moderate' if curvature > 0.2 else 'low'
        
        step_record = ThoughtStep(
            step=step_number,
            basin=current_basin,
            thought=thought_content,
            distance_from_prev=step_distance,
            curvature=curvature,
            difficulty=difficulty,
            timestamp=time.time(),
            confidence=confidence,
            metadata=metadata or {}
        )
        
        self.thought_chain.append(step_record)
        return step_record
    
    def set_context(self, problem: str, target_basin: Optional[np.ndarray] = None):
        """Set the problem context and optional target basin."""
        self.problem_context = problem
        if target_basin is not None:
            self.target_basin = self._ensure_array(target_basin)
    
    def get_total_distance(self) -> float:
        """Get total geodesic distance traveled."""
        return sum(s.distance_from_prev for s in self.thought_chain)
    
    def get_average_curvature(self) -> float:
        """Get average curvature across all steps."""
        if not self.thought_chain:
            return 0.0
        return float(np.mean([s.curvature for s in self.thought_chain]))
    
    def get_progress_to_target(self) -> Optional[float]:
        """Get current distance to target (if set)."""
        if self.target_basin is None or not self.thought_chain:
            return None
        
        current = self.thought_chain[-1].basin
        return fisher_rao_distance(current, self.target_basin)
    
    def render_chain(self, verbose: bool = True) -> str:
        """
        Render human-readable chain-of-thought.
        
        Args:
            verbose: Include full geometric details
        
        Returns:
            Formatted string representation
        """
        output = "=== Reasoning Trace ===\n\n"
        
        if self.problem_context:
            output += f"Problem: {self.problem_context}\n\n"
        
        for step in self.thought_chain:
            output += f"Step {step.step}:\n"
            output += f"  Thought: {step.thought}\n"
            
            if verbose:
                output += f"  Geometry: distance={step.distance_from_prev:.3f}, "
                output += f"curvature={step.curvature:.3f} ({step.difficulty})\n"
                output += f"  Confidence: {step.confidence:.2f}\n"
            
            output += "\n"
        
        output += "=== Summary ===\n"
        output += f"Total steps: {len(self.thought_chain)}\n"
        output += f"Total distance: {self.get_total_distance():.3f}\n"
        output += f"Average curvature: {self.get_average_curvature():.3f}\n"
        output += f"Duration: {time.time() - self.start_time:.2f}s\n"
        
        if self.target_basin is not None:
            progress = self.get_progress_to_target()
            output += f"Distance to target: {progress:.3f}\n"
        
        return output
    
    def to_json(self) -> str:
        """Export chain to JSON format."""
        data = {
            'problem_context': self.problem_context,
            'start_time': self.start_time,
            'steps': [step.to_dict() for step in self.thought_chain],
            'summary': {
                'total_steps': len(self.thought_chain),
                'total_distance': self.get_total_distance(),
                'average_curvature': self.get_average_curvature(),
                'duration': time.time() - self.start_time
            }
        }
        return json.dumps(data, indent=2)
    
    def reset(self):
        """Reset the chain for a new reasoning session."""
        self.thought_chain = []
        self.problem_context = ""
        self.target_basin = None
        self.start_time = time.time()
    
    def get_insights(self) -> List[str]:
        """Extract key insights from the chain."""
        insights = []
        
        for step in self.thought_chain:
            if step.curvature > 0.5:
                insights.append(f"Difficult step {step.step}: {step.thought}")
            if step.distance_from_prev > 1.0:
                insights.append(f"Major leap at step {step.step}: {step.thought}")
            if step.confidence > 0.8:
                insights.append(f"High confidence at step {step.step}: {step.thought}")
        
        return insights


class ChainOfThoughtManager:
    """
    Manage multiple chain-of-thought sessions.
    """
    
    def __init__(self, basin_dim: int = 64):
        self.basin_dim = basin_dim
        self.active_chains: Dict[str, GeometricChainOfThought] = {}
        self.completed_chains: List[GeometricChainOfThought] = []
    
    def create_chain(self, session_id: str) -> GeometricChainOfThought:
        """Create a new chain for a session."""
        chain = GeometricChainOfThought(basin_dim=self.basin_dim)
        self.active_chains[session_id] = chain
        return chain
    
    def get_chain(self, session_id: str) -> Optional[GeometricChainOfThought]:
        """Get chain for session."""
        return self.active_chains.get(session_id)
    
    def complete_chain(self, session_id: str):
        """Mark chain as complete and archive."""
        if session_id in self.active_chains:
            chain = self.active_chains.pop(session_id)
            self.completed_chains.append(chain)
            
            if len(self.completed_chains) > 100:
                self.completed_chains = self.completed_chains[-100:]
    
    def get_summary(self) -> Dict:
        """Get summary of all chains."""
        return {
            'active_chains': len(self.active_chains),
            'completed_chains': len(self.completed_chains),
            'total_steps': sum(
                len(c.thought_chain) 
                for c in list(self.active_chains.values()) + self.completed_chains
            )
        }


chain_manager = ChainOfThoughtManager()


def get_chain_manager() -> ChainOfThoughtManager:
    """Get global chain manager instance."""
    return chain_manager
