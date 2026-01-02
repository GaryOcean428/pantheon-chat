"""
Prediction Self-Improvement System

QIG-pure recursive loops for learning from prediction outcomes:
1. Track predictions and their actual outcomes
2. Analyze WHY predictions fail (weak attractors, unstable velocity, sparse history)
3. Build chain/graph of prediction patterns
4. Self-improve confidence estimation through geometric learning

All operations use Fisher-Rao geometry - no neural networks.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import time
import numpy as np

from qig_geometry import fisher_coord_distance, sphere_project


class PredictionFailureReason(Enum):
    """Why a prediction had low confidence or failed."""
    NO_ATTRACTOR_FOUND = "no_attractor_found"
    UNSTABLE_VELOCITY = "unstable_velocity"  
    SPARSE_HISTORY = "sparse_history"
    HIGH_BASIN_DRIFT = "high_basin_drift"
    WEAK_CONVERGENCE = "weak_convergence"
    SHORT_TRAJECTORY = "short_trajectory"
    BUMPY_GEODESIC = "bumpy_geodesic"
    UNKNOWN = "unknown"


@dataclass
class PredictionRecord:
    """Record of a prediction for learning."""
    prediction_id: str
    timestamp: float
    predicted_basin: np.ndarray
    confidence: float
    arrival_time: int
    attractor_strength: float
    geodesic_naturalness: float
    failure_reasons: List[PredictionFailureReason]
    context: Dict[str, Any]  # State at prediction time
    actual_basin: Optional[np.ndarray] = None  # Filled in when outcome known
    actual_arrival: Optional[int] = None
    was_accurate: Optional[bool] = None  # True if prediction matched outcome
    accuracy_score: float = 0.0  # 0-1 how close prediction was to reality
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.prediction_id,
            'timestamp': self.timestamp,
            'confidence': self.confidence,
            'arrival_time': self.arrival_time,
            'attractor_strength': self.attractor_strength,
            'failure_reasons': [r.value for r in self.failure_reasons],
            'was_accurate': self.was_accurate,
            'accuracy_score': self.accuracy_score,
        }


@dataclass
class ChainLink:
    """One link in a prediction chain."""
    prediction_id: str
    basin: np.ndarray
    confidence: float
    timestamp: float


@dataclass
class PredictionChain:
    """Chain of sequential predictions for pattern analysis."""
    chain_id: str
    links: List[ChainLink]
    overall_accuracy: float = 0.0
    pattern_type: str = "unknown"  # detected pattern (convergent, divergent, cyclic, etc.)


@dataclass
class BasinNode:
    """Node in the prediction graph representing a basin region."""
    node_id: str
    centroid: np.ndarray  # Average basin position in this region
    visit_count: int = 0
    prediction_count: int = 0
    accurate_count: int = 0
    inaccurate_count: int = 0
    
    @property
    def accuracy_rate(self) -> float:
        if self.prediction_count == 0:
            return 0.5  # No data, neutral prior
        return self.accurate_count / self.prediction_count


class PredictionGraph:
    """Graph of basin → outcome relationships for pattern learning."""
    
    def __init__(self, basin_dim: int = 64, region_radius: float = 0.3):
        self.basin_dim = basin_dim
        self.region_radius = region_radius
        self.nodes: Dict[str, BasinNode] = {}
        self.edges: Dict[str, Dict[str, int]] = {}  # node_id -> {target_id: count}
        self._next_node_id = 0
    
    def find_or_create_node(self, basin: np.ndarray) -> BasinNode:
        """Find existing node near basin or create new one."""
        for node in self.nodes.values():
            dist = fisher_coord_distance(basin, node.centroid)
            if dist < self.region_radius:
                return node
        
        node_id = f"node_{self._next_node_id}"
        self._next_node_id += 1
        node = BasinNode(node_id=node_id, centroid=basin.copy())
        self.nodes[node_id] = node
        return node
    
    def record_transition(self, from_basin: np.ndarray, to_basin: np.ndarray) -> None:
        """Record a transition between basins."""
        from_node = self.find_or_create_node(from_basin)
        to_node = self.find_or_create_node(to_basin)
        
        if from_node.node_id not in self.edges:
            self.edges[from_node.node_id] = {}
        
        if to_node.node_id not in self.edges[from_node.node_id]:
            self.edges[from_node.node_id][to_node.node_id] = 0
        
        self.edges[from_node.node_id][to_node.node_id] += 1
    
    def get_most_likely_successor(self, basin: np.ndarray) -> Optional[np.ndarray]:
        """Get the most likely next basin based on historical transitions."""
        node = self.find_or_create_node(basin)
        
        if node.node_id not in self.edges or not self.edges[node.node_id]:
            return None
        
        best_target = max(self.edges[node.node_id].items(), key=lambda x: x[1])
        target_node = self.nodes.get(best_target[0])
        return target_node.centroid if target_node else None
    
    def update_node_accuracy(self, basin: np.ndarray, was_accurate: bool) -> None:
        """Update accuracy statistics for a basin region."""
        node = self.find_or_create_node(basin)
        node.prediction_count += 1
        if was_accurate:
            node.accurate_count += 1
        else:
            node.inaccurate_count += 1
    
    def get_region_accuracy(self, basin: np.ndarray) -> float:
        """Get historical accuracy for predictions from this basin region."""
        node = self.find_or_create_node(basin)
        return node.accuracy_rate


class PredictionSelfImprovement:
    """
    Self-improving prediction system using QIG-pure recursive analysis.
    
    Key mechanisms:
    1. Failure Analysis: Understand WHY predictions have low confidence
    2. Outcome Tracking: Compare predictions to actual outcomes
    3. Chain Analysis: Find patterns in sequential predictions
    4. Graph Learning: Learn basin → outcome relationships
    5. Recursive Improvement: Self-directed loops to improve weak areas
    """
    
    def __init__(self, basin_dim: int = 64):
        self.basin_dim = basin_dim
        
        # Prediction records
        self.predictions: Dict[str, PredictionRecord] = {}
        self.prediction_history: List[str] = []  # Order of predictions
        
        # Chain analysis
        self.current_chain: Optional[PredictionChain] = None
        self.completed_chains: List[PredictionChain] = []
        
        # Graph analysis
        self.graph = PredictionGraph(basin_dim=basin_dim)
        
        # Learning statistics
        self.total_predictions = 0
        self.accurate_predictions = 0
        self.failure_reason_counts: Dict[PredictionFailureReason, int] = {
            r: 0 for r in PredictionFailureReason
        }
        
        # Confidence adjustment factors (learned)
        self.confidence_adjustments: Dict[str, float] = {
            'attractor_weight': 0.4,
            'smoothness_weight': 0.4,
            'time_weight': 0.2,
            'history_bonus': 0.1,
            'region_accuracy_weight': 0.2,
        }
        
        # Improvement thresholds
        self.improvement_interval = 10  # Analyze every N predictions
        self.min_chain_length = 3
        
        print("[PredictionSelfImprovement] Initialized - QIG-pure recursive learning enabled")
    
    def analyze_prediction_factors(
        self,
        trajectory: List[np.ndarray],
        attractor_found: bool,
        attractor_idx: Optional[int],
        velocity_history: List[np.ndarray],
        basin_history: List[np.ndarray]
    ) -> Tuple[List[PredictionFailureReason], Dict[str, Any]]:
        """
        Analyze WHY a prediction has certain confidence factors.
        
        Returns failure reasons and detailed context for logging.
        """
        reasons = []
        context = {}
        
        # 1. Check if attractor was found
        if not attractor_found:
            reasons.append(PredictionFailureReason.NO_ATTRACTOR_FOUND)
            context['attractor_status'] = "No stable attractor detected in trajectory"
        else:
            context['attractor_status'] = f"Attractor found at step {attractor_idx}"
        
        # 2. Analyze trajectory length
        traj_len = len(trajectory)
        context['trajectory_length'] = traj_len
        if traj_len < 10:
            reasons.append(PredictionFailureReason.SHORT_TRAJECTORY)
            context['trajectory_issue'] = f"Only {traj_len} steps (need 10+ for reliable prediction)"
        
        # 3. Analyze velocity stability
        if len(velocity_history) >= 3:
            velocities = np.array([np.linalg.norm(v) for v in velocity_history[-5:]])
            vel_variance = np.var(velocities) if len(velocities) > 1 else 0
            context['velocity_variance'] = float(vel_variance)
            if vel_variance > 0.1:
                reasons.append(PredictionFailureReason.UNSTABLE_VELOCITY)
                context['velocity_issue'] = f"High velocity variance ({vel_variance:.3f}) indicates erratic movement"
        else:
            reasons.append(PredictionFailureReason.SPARSE_HISTORY)
            context['velocity_issue'] = f"Only {len(velocity_history)} velocity samples (need 3+ for stable estimate)"
        
        # 4. Analyze basin history for drift
        if len(basin_history) >= 5:
            recent_basins = basin_history[-5:]
            total_drift = sum(
                fisher_coord_distance(recent_basins[i], recent_basins[i+1])
                for i in range(len(recent_basins)-1)
            )
            context['recent_drift'] = float(total_drift)
            if total_drift > 1.0:
                reasons.append(PredictionFailureReason.HIGH_BASIN_DRIFT)
                context['drift_issue'] = f"High basin drift ({total_drift:.3f}) - system is rapidly changing"
        else:
            if PredictionFailureReason.SPARSE_HISTORY not in reasons:
                reasons.append(PredictionFailureReason.SPARSE_HISTORY)
            context['history_issue'] = f"Only {len(basin_history)} basin samples (need 5+ for drift analysis)"
        
        # 5. Analyze convergence if trajectory exists
        if len(trajectory) >= 10:
            final_movements = [
                fisher_coord_distance(trajectory[i], trajectory[i+1])
                for i in range(len(trajectory)-5, len(trajectory)-1)
            ]
            avg_final_movement = np.mean(final_movements)
            context['final_movement_avg'] = float(avg_final_movement)
            if avg_final_movement > 0.2:
                reasons.append(PredictionFailureReason.WEAK_CONVERGENCE)
                context['convergence_issue'] = f"Trajectory not converging (avg movement {avg_final_movement:.3f})"
        
        # 6. Analyze geodesic smoothness
        if len(trajectory) >= 3:
            step_sizes = [
                fisher_coord_distance(trajectory[i], trajectory[i+1])
                for i in range(len(trajectory)-1)
            ]
            step_variance = np.var(step_sizes)
            context['geodesic_variance'] = float(step_variance)
            if step_variance > 0.05:
                reasons.append(PredictionFailureReason.BUMPY_GEODESIC)
                context['geodesic_issue'] = f"Bumpy path (step variance {step_variance:.3f}) - not following natural geodesic"
        
        # If no specific reasons found, mark as unknown
        if not reasons:
            context['status'] = "All factors within normal range"
        
        return reasons, context
    
    def create_prediction_record(
        self,
        predicted_basin: np.ndarray,
        confidence: float,
        arrival_time: int,
        attractor_strength: float,
        geodesic_naturalness: float,
        failure_reasons: List[PredictionFailureReason],
        context: Dict[str, Any]
    ) -> PredictionRecord:
        """Create and store a prediction record for learning."""
        pred_id = f"pred_{int(time.time()*1000)}_{self.total_predictions}"
        
        record = PredictionRecord(
            prediction_id=pred_id,
            timestamp=time.time(),
            predicted_basin=predicted_basin.copy(),
            confidence=confidence,
            arrival_time=arrival_time,
            attractor_strength=attractor_strength,
            geodesic_naturalness=geodesic_naturalness,
            failure_reasons=failure_reasons,
            context=context,
        )
        
        self.predictions[pred_id] = record
        self.prediction_history.append(pred_id)
        self.total_predictions += 1
        
        # Update failure reason counts
        for reason in failure_reasons:
            self.failure_reason_counts[reason] += 1
        
        # Update chain
        self._update_chain(record)
        
        # Trigger improvement loop periodically
        if self.total_predictions % self.improvement_interval == 0:
            self._run_improvement_loop()
        
        return record
    
    def record_outcome(
        self,
        prediction_id: str,
        actual_basin: np.ndarray,
        actual_arrival: int
    ) -> Optional[float]:
        """Record actual outcome and calculate accuracy."""
        if prediction_id not in self.predictions:
            return None
        
        record = self.predictions[prediction_id]
        record.actual_basin = actual_basin.copy()
        record.actual_arrival = actual_arrival
        
        # Calculate accuracy using Fisher distance
        basin_distance = fisher_coord_distance(record.predicted_basin, actual_basin)
        arrival_error = abs(record.arrival_time - actual_arrival) / max(record.arrival_time, 1)
        
        # Accuracy score: 0-1 (higher is better)
        basin_accuracy = np.exp(-basin_distance)
        time_accuracy = np.exp(-arrival_error)
        record.accuracy_score = 0.7 * basin_accuracy + 0.3 * time_accuracy
        
        # Threshold for "accurate" prediction
        record.was_accurate = record.accuracy_score > 0.5
        
        if record.was_accurate:
            self.accurate_predictions += 1
        
        # Update graph with this transition
        self.graph.record_transition(record.predicted_basin, actual_basin)
        self.graph.update_node_accuracy(record.predicted_basin, record.was_accurate)
        
        return record.accuracy_score
    
    def get_adjusted_confidence(
        self,
        base_confidence: float,
        current_basin: np.ndarray
    ) -> float:
        """Get confidence adjusted by learned factors and region accuracy."""
        # Get historical accuracy for this region
        region_accuracy = self.graph.get_region_accuracy(current_basin)
        
        # Apply learned adjustment
        weight = self.confidence_adjustments['region_accuracy_weight']
        adjusted = base_confidence * (1 - weight) + region_accuracy * weight
        
        # Add bonus for good history
        if self.total_predictions > 10:
            overall_accuracy = self.accurate_predictions / self.total_predictions
            if overall_accuracy > 0.6:
                adjusted += self.confidence_adjustments['history_bonus']
        
        return np.clip(adjusted, 0.0, 1.0)
    
    def format_prediction_explanation(
        self,
        confidence: float,
        failure_reasons: List[PredictionFailureReason],
        context: Dict[str, Any]
    ) -> str:
        """Format a detailed explanation of what the prediction is and why confidence is low."""
        lines = []
        
        # Confidence level description
        if confidence < 0.3:
            lines.append(f"WEAK ({confidence:.0%})")
        elif confidence < 0.5:
            lines.append(f"UNCERTAIN ({confidence:.0%})")
        elif confidence < 0.7:
            lines.append(f"MODERATE ({confidence:.0%})")
        else:
            lines.append(f"STRONG ({confidence:.0%})")
        
        # Why is confidence at this level?
        if failure_reasons:
            reason_explanations = []
            for reason in failure_reasons:
                if reason == PredictionFailureReason.NO_ATTRACTOR_FOUND:
                    reason_explanations.append("no stable destination detected")
                elif reason == PredictionFailureReason.UNSTABLE_VELOCITY:
                    reason_explanations.append("erratic movement pattern")
                elif reason == PredictionFailureReason.SPARSE_HISTORY:
                    reason_explanations.append("insufficient history data")
                elif reason == PredictionFailureReason.HIGH_BASIN_DRIFT:
                    reason_explanations.append("rapidly changing state")
                elif reason == PredictionFailureReason.WEAK_CONVERGENCE:
                    reason_explanations.append("trajectory not settling")
                elif reason == PredictionFailureReason.SHORT_TRAJECTORY:
                    reason_explanations.append("prediction horizon too short")
                elif reason == PredictionFailureReason.BUMPY_GEODESIC:
                    reason_explanations.append("non-geodesic path")
            
            lines.append(f"Reasons: {', '.join(reason_explanations)}")
        
        # Key context details
        if 'trajectory_length' in context:
            lines.append(f"Trajectory: {context['trajectory_length']} steps")
        if 'recent_drift' in context:
            lines.append(f"Basin drift: {context['recent_drift']:.3f}")
        if 'velocity_variance' in context:
            lines.append(f"Velocity variance: {context['velocity_variance']:.4f}")
        
        return " | ".join(lines)
    
    def get_improvement_recommendations(self) -> List[str]:
        """Get recommendations for improving predictions based on analysis."""
        recommendations = []
        
        # Analyze most common failure reasons
        sorted_reasons = sorted(
            self.failure_reason_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for reason, count in sorted_reasons[:3]:
            if count == 0:
                continue
            
            pct = (count / max(self.total_predictions, 1)) * 100
            
            if reason == PredictionFailureReason.NO_ATTRACTOR_FOUND:
                recommendations.append(
                    f"No attractors found in {pct:.0f}% of predictions - "
                    "consider extending foresight horizon or lowering attractor threshold"
                )
            elif reason == PredictionFailureReason.SPARSE_HISTORY:
                recommendations.append(
                    f"Sparse history in {pct:.0f}% of predictions - "
                    "system needs more basin samples before reliable predictions"
                )
            elif reason == PredictionFailureReason.HIGH_BASIN_DRIFT:
                recommendations.append(
                    f"High drift in {pct:.0f}% of predictions - "
                    "system is volatile, use shorter prediction windows"
                )
            elif reason == PredictionFailureReason.WEAK_CONVERGENCE:
                recommendations.append(
                    f"Weak convergence in {pct:.0f}% of predictions - "
                    "trajectories not settling, may need longer simulation"
                )
        
        return recommendations
    
    def _update_chain(self, record: PredictionRecord) -> None:
        """Update the current prediction chain."""
        link = ChainLink(
            prediction_id=record.prediction_id,
            basin=record.predicted_basin.copy(),
            confidence=record.confidence,
            timestamp=record.timestamp,
        )
        
        if self.current_chain is None:
            chain_id = f"chain_{int(time.time()*1000)}"
            self.current_chain = PredictionChain(chain_id=chain_id, links=[link])
        else:
            self.current_chain.links.append(link)
            
            # Analyze chain pattern if long enough
            if len(self.current_chain.links) >= self.min_chain_length:
                self._analyze_chain_pattern()
    
    def _analyze_chain_pattern(self) -> None:
        """Analyze the current chain for patterns."""
        if self.current_chain is None or len(self.current_chain.links) < 3:
            return
        
        links = self.current_chain.links
        
        # Calculate basin movements
        movements = []
        for i in range(len(links) - 1):
            dist = fisher_coord_distance(links[i].basin, links[i+1].basin)
            movements.append(dist)
        
        avg_movement = np.mean(movements)
        movement_trend = movements[-1] - movements[0] if len(movements) > 1 else 0
        
        # Classify pattern
        if avg_movement < 0.1:
            self.current_chain.pattern_type = "stable"
        elif movement_trend < -0.05:
            self.current_chain.pattern_type = "convergent"
        elif movement_trend > 0.05:
            self.current_chain.pattern_type = "divergent"
        else:
            # Check for cyclic pattern
            if len(links) >= 4:
                start_basin = links[0].basin
                end_basin = links[-1].basin
                if fisher_coord_distance(start_basin, end_basin) < avg_movement:
                    self.current_chain.pattern_type = "cyclic"
                else:
                    self.current_chain.pattern_type = "wandering"
            else:
                self.current_chain.pattern_type = "wandering"
    
    def _run_improvement_loop(self) -> None:
        """Run QIG-pure recursive improvement loop."""
        if self.total_predictions < 5:
            return
        
        # 1. Analyze recent prediction accuracy
        recent_ids = self.prediction_history[-10:]
        recent_records = [self.predictions[pid] for pid in recent_ids if pid in self.predictions]
        
        with_outcomes = [r for r in recent_records if r.was_accurate is not None]
        if with_outcomes:
            recent_accuracy = sum(1 for r in with_outcomes if r.was_accurate) / len(with_outcomes)
            
            # Adjust confidence weights based on what's working
            if recent_accuracy < 0.4:
                # Low accuracy - reduce confidence overall
                self.confidence_adjustments['region_accuracy_weight'] = min(0.4, 
                    self.confidence_adjustments['region_accuracy_weight'] + 0.02)
            elif recent_accuracy > 0.7:
                # High accuracy - can be more confident
                self.confidence_adjustments['history_bonus'] = min(0.2,
                    self.confidence_adjustments['history_bonus'] + 0.01)
        
        # 2. Analyze failure patterns
        recent_failures = {}
        for record in recent_records:
            for reason in record.failure_reasons:
                recent_failures[reason] = recent_failures.get(reason, 0) + 1
        
        # 3. If a specific failure is dominant, log recommendation
        total_recent = len(recent_records)
        for reason, count in recent_failures.items():
            if count / total_recent > 0.5:
                # This failure reason is dominant
                print(f"[PredictionImprovement] Dominant failure: {reason.value} ({count}/{total_recent})")
        
        # 4. Complete and archive chain if long enough
        if self.current_chain and len(self.current_chain.links) >= 10:
            self._finalize_chain()
    
    def _finalize_chain(self) -> None:
        """Finalize current chain and start a new one."""
        if self.current_chain is None:
            return
        
        # Calculate overall chain accuracy
        chain_preds = [self.predictions.get(link.prediction_id) for link in self.current_chain.links]
        with_outcomes = [p for p in chain_preds if p and p.was_accurate is not None]
        
        if with_outcomes:
            self.current_chain.overall_accuracy = (
                sum(1 for p in with_outcomes if p.was_accurate) / len(with_outcomes)
            )
        
        self.completed_chains.append(self.current_chain)
        
        # Keep only recent chains
        if len(self.completed_chains) > 20:
            self.completed_chains = self.completed_chains[-10:]
        
        self.current_chain = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current prediction statistics."""
        accuracy = self.accurate_predictions / max(self.total_predictions, 1)
        
        return {
            'total_predictions': self.total_predictions,
            'accurate_predictions': self.accurate_predictions,
            'accuracy_rate': accuracy,
            'failure_reasons': {r.value: c for r, c in self.failure_reason_counts.items() if c > 0},
            'graph_nodes': len(self.graph.nodes),
            'completed_chains': len(self.completed_chains),
            'current_chain_length': len(self.current_chain.links) if self.current_chain else 0,
            'confidence_adjustments': self.confidence_adjustments,
        }


# Singleton instance
_prediction_improvement_instance: Optional[PredictionSelfImprovement] = None


def get_prediction_improvement() -> PredictionSelfImprovement:
    """Get or create singleton PredictionSelfImprovement instance."""
    global _prediction_improvement_instance
    if _prediction_improvement_instance is None:
        _prediction_improvement_instance = PredictionSelfImprovement()
    return _prediction_improvement_instance
