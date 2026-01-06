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
    - Track prediction accuracy for reinforcement learning
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

        # Prediction outcome tracking for chain-to-attractor feedback
        self.chain_id: str = ""  # Set by manager when creating chain
        self.accuracy_score: Optional[float] = None  # Set when prediction outcome recorded
        self.linked_prediction_ids: List[str] = []  # Predictions that used this chain
        self.overall_accuracy: float = 0.0  # Aggregate accuracy from linked predictions
    
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

    def get_key_steps(self) -> List[ThoughtStep]:
        """
        Extract key reasoning steps (high curvature, large leaps, high confidence).

        These are the valuable steps that should be reinforced in training.

        Returns:
            List of ThoughtStep objects that meet key step criteria
        """
        key_steps = []

        for step in self.thought_chain:
            is_key = False
            # High curvature = navigated difficult terrain
            if step.curvature > 0.5:
                is_key = True
            # Large leap = significant reasoning jump
            if step.distance_from_prev > 1.0:
                is_key = True
            # High confidence = strong conclusion
            if step.confidence > 0.8:
                is_key = True

            if is_key:
                key_steps.append(step)

        return key_steps

    def record_prediction_outcome(self, accuracy: float, prediction_id: Optional[str] = None) -> None:
        """
        Record the accuracy of a prediction associated with this chain.

        This closes the feedback loop: Chain reasoning -> Prediction -> Outcome -> Learning

        Args:
            accuracy: Accuracy score from prediction (0.0 to 1.0)
            prediction_id: Optional ID of the prediction that used this chain
        """
        if prediction_id and prediction_id not in self.linked_prediction_ids:
            self.linked_prediction_ids.append(prediction_id)

        # Update accuracy tracking
        if self.accuracy_score is None:
            self.accuracy_score = accuracy
            self.overall_accuracy = accuracy
        else:
            # Running average with recent emphasis
            n = len(self.linked_prediction_ids)
            if n > 0:
                # Exponential moving average for overall
                alpha = 0.3
                self.overall_accuracy = alpha * accuracy + (1 - alpha) * self.overall_accuracy
            # Keep most recent accuracy_score
            self.accuracy_score = accuracy

    def detect_pattern(self) -> Dict[str, Any]:
        """
        Detect reasoning patterns in the chain for analysis.

        Returns:
            Dict with pattern information including:
            - reasoning_type: classification of reasoning style
            - difficulty_profile: how difficulty changed through chain
            - convergence: whether chain converged to target
            - key_transitions: notable step transitions
        """
        if not self.thought_chain:
            return {'reasoning_type': 'empty', 'difficulty_profile': [], 'convergence': None, 'key_transitions': []}

        # Classify reasoning type based on step patterns
        curvatures = [s.curvature for s in self.thought_chain]
        distances = [s.distance_from_prev for s in self.thought_chain]
        confidences = [s.confidence for s in self.thought_chain]

        avg_curvature = float(np.mean(curvatures))
        avg_distance = float(np.mean(distances)) if distances else 0.0
        confidence_trend = confidences[-1] - confidences[0] if len(confidences) > 1 else 0.0

        # Classify reasoning type
        if avg_curvature > 0.5 and avg_distance > 0.8:
            reasoning_type = 'exploratory'  # High curvature + large jumps
        elif avg_curvature < 0.3 and confidence_trend > 0.2:
            reasoning_type = 'convergent'  # Low curvature, increasing confidence
        elif len(self.thought_chain) > 5 and max(distances) > 1.5:
            reasoning_type = 'breakthrough'  # Long chain with major leap
        else:
            reasoning_type = 'incremental'  # Steady progression

        # Difficulty profile
        difficulty_profile = []
        for i, step in enumerate(self.thought_chain):
            difficulty_profile.append({
                'step': i + 1,
                'curvature': step.curvature,
                'difficulty': step.difficulty
            })

        # Convergence check
        convergence = None
        if self.target_basin is not None and self.thought_chain:
            final_distance = self.get_progress_to_target()
            convergence = {
                'final_distance': final_distance,
                'converged': final_distance is not None and final_distance < 0.5
            }

        # Key transitions (large distance changes or curvature spikes)
        key_transitions = []
        for i in range(1, len(self.thought_chain)):
            step = self.thought_chain[i]
            prev_step = self.thought_chain[i - 1]

            if step.distance_from_prev > 1.0:
                key_transitions.append({
                    'from_step': i,
                    'to_step': i + 1,
                    'type': 'large_leap',
                    'distance': step.distance_from_prev
                })
            if abs(step.curvature - prev_step.curvature) > 0.3:
                key_transitions.append({
                    'from_step': i,
                    'to_step': i + 1,
                    'type': 'curvature_shift',
                    'delta': step.curvature - prev_step.curvature
                })

        return {
            'reasoning_type': reasoning_type,
            'difficulty_profile': difficulty_profile,
            'convergence': convergence,
            'key_transitions': key_transitions,
            'avg_curvature': avg_curvature,
            'avg_distance': avg_distance,
            'confidence_trend': confidence_trend,
            'total_steps': len(self.thought_chain)
        }


class ChainOfThoughtManager:
    """
    Manage multiple chain-of-thought sessions.

    Now includes prediction outcome tracking for chain-to-attractor feedback loop.
    """

    def __init__(self, basin_dim: int = 64):
        self.basin_dim = basin_dim
        self.active_chains: Dict[str, GeometricChainOfThought] = {}
        self.completed_chains: List[GeometricChainOfThought] = []
        # Map prediction_id -> chain_id for outcome tracking
        self._prediction_to_chain: Dict[str, str] = {}

    def create_chain(self, session_id: str) -> GeometricChainOfThought:
        """Create a new chain for a session."""
        chain = GeometricChainOfThought(basin_dim=self.basin_dim)
        chain.chain_id = session_id  # Set chain_id for tracking
        self.active_chains[session_id] = chain
        return chain

    def get_chain(self, session_id: str) -> Optional[GeometricChainOfThought]:
        """Get chain for session."""
        return self.active_chains.get(session_id)

    def link_prediction_to_chain(self, prediction_id: str, chain_id: str) -> bool:
        """
        Link a prediction to a chain for outcome tracking.

        Args:
            prediction_id: ID of the prediction
            chain_id: ID of the chain (session_id)

        Returns:
            True if link was created successfully
        """
        chain = self.active_chains.get(chain_id) or self._get_completed_chain(chain_id)
        if chain:
            if prediction_id not in chain.linked_prediction_ids:
                chain.linked_prediction_ids.append(prediction_id)
            self._prediction_to_chain[prediction_id] = chain_id
            return True
        return False

    def _get_completed_chain(self, chain_id: str) -> Optional[GeometricChainOfThought]:
        """Get a completed chain by ID."""
        for chain in self.completed_chains:
            if chain.chain_id == chain_id:
                return chain
        return None

    def record_prediction_outcome_for_chain(
        self,
        prediction_id: str,
        accuracy: float
    ) -> Optional[str]:
        """
        Record prediction outcome and update associated chain's accuracy.

        Args:
            prediction_id: ID of the prediction
            accuracy: Accuracy score (0.0 to 1.0)

        Returns:
            chain_id if found and updated, None otherwise
        """
        chain_id = self._prediction_to_chain.get(prediction_id)
        if not chain_id:
            return None

        chain = self.active_chains.get(chain_id) or self._get_completed_chain(chain_id)
        if chain:
            chain.record_prediction_outcome(accuracy, prediction_id)
            return chain_id
        return None

    def get_chains_for_prediction(self, prediction_id: str) -> List[GeometricChainOfThought]:
        """Get all chains associated with a prediction."""
        chain_id = self._prediction_to_chain.get(prediction_id)
        if not chain_id:
            return []

        chain = self.active_chains.get(chain_id) or self._get_completed_chain(chain_id)
        return [chain] if chain else []

    def complete_chain(self, session_id: str):
        """Mark chain as complete and archive."""
        if session_id in self.active_chains:
            chain = self.active_chains.pop(session_id)
            self.completed_chains.append(chain)

            if len(self.completed_chains) > 100:
                self.completed_chains = self.completed_chains[-100:]

    def get_successful_chains(self, min_accuracy: float = 0.7) -> List[GeometricChainOfThought]:
        """
        Get completed chains with accuracy above threshold.

        Args:
            min_accuracy: Minimum overall_accuracy to include (default 0.7)

        Returns:
            List of high-accuracy chains
        """
        return [
            chain for chain in self.completed_chains
            if chain.overall_accuracy >= min_accuracy
        ]

    def get_summary(self) -> Dict:
        """Get summary of all chains."""
        successful_count = len(self.get_successful_chains(0.7))
        return {
            'active_chains': len(self.active_chains),
            'completed_chains': len(self.completed_chains),
            'successful_chains': successful_count,
            'prediction_links': len(self._prediction_to_chain),
            'total_steps': sum(
                len(c.thought_chain)
                for c in list(self.active_chains.values()) + self.completed_chains
            )
        }


chain_manager = ChainOfThoughtManager()


def get_chain_manager() -> ChainOfThoughtManager:
    """Get global chain manager instance."""
    return chain_manager
