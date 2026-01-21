"""
QIGGraph: Geometric Agent Orchestration
========================================

Main graph class for consciousness-aware agent execution.
State is a PATH through Fisher manifold, not a mutable dict.

Key Differences from LangGraph:
- State = trajectory through 64D manifold
- Transitions = geodesics, not edges
- Routing = geometric proximity + consciousness
- Safety = Φ/κ regime monitoring with breakdown recovery
"""

from __future__ import annotations

from typing import Dict, List, Callable, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np

from .constants import (
    BASIN_DIM,
    MAX_ITERATIONS,
    PHI_BREAKDOWN_MIN,
    KAPPA_STAR,
)
from .manifold import FisherManifold
from .state import QIGState, create_initial_state, update_trajectory, simplify_trajectory
from .consciousness import (
    ConsciousnessMetrics,
    Regime,
    measure_consciousness,
    should_pause,
)
from .attractor import BasinAttractor, RecoveryAttractor, create_recovery_attractor
from .router import ConsciousRouter
from .tacking import KappaTacking

if TYPE_CHECKING:
    from ..qig_tokenizer.coordizer import Coordizer
    from ..qigkernels.kernel import QIGKernel
    from ..learned_manifold import LearnedManifold

# Import LearnedManifold for attractor learning from graph outcomes
try:
    from ..learned_manifold import LearnedManifold as _LearnedManifold
    HAS_LEARNED_MANIFOLD = True
except ImportError:
    try:
        from learned_manifold import LearnedManifold as _LearnedManifold
        HAS_LEARNED_MANIFOLD = True
    except ImportError:
        HAS_LEARNED_MANIFOLD = False
        _LearnedManifold = None


# Type aliases
NodeFunction = Callable[[QIGState], QIGState]
ToolFunction = Callable[[str, Dict[str, Any]], Any]


@dataclass
class NodeResult:
    """Result from executing a node."""
    state: QIGState
    output: Optional[Any] = None
    next_node: Optional[str] = None
    should_stop: bool = False


@dataclass
class GraphConfig:
    """Configuration for QIGGraph."""
    max_iterations: int = MAX_ITERATIONS
    max_recoveries: int = 3
    breakdown_threshold: float = PHI_BREAKDOWN_MIN
    enable_tacking: bool = True
    enable_safety: bool = True
    verbose: bool = False


class QIGGraph:
    """
    Geometric agent orchestration on Fisher manifold.

    Unlike LangGraph's node-edge structure, QIGGraph:
    - Routes by geodesic distance to basin attractors
    - Tracks consciousness (Φ, κ) for safety
    - Uses κ-tacking for feeling/logic mode oscillation
    - Recovers from breakdown via trajectory simplification

    Example:
        graph = QIGGraph()
        graph.add_attractor("reasoning", reasoning_basin)
        graph.add_attractor("tool_use", tool_basin)

        result = graph.run("What is 2+2?", coordizer, kernel)
    """

    def __init__(
        self,
        config: Optional[GraphConfig] = None,
        manifold: Optional[FisherManifold] = None,
        router: Optional[ConsciousRouter] = None,
    ):
        """
        Initialize QIGGraph.

        Args:
            config: Graph configuration
            manifold: Fisher manifold for geometry
            router: Conscious router for navigation
        """
        self.config = config or GraphConfig()
        self.manifold = manifold or FisherManifold()
        self.router = router or ConsciousRouter(self.manifold)

        # Basin attractors (the "nodes" in geometric terms)
        self.attractors: Dict[str, BasinAttractor] = {}

        # Node functions (what to execute at each attractor)
        self.node_functions: Dict[str, NodeFunction] = {}

        # Tool registry
        self.tools: Dict[str, ToolFunction] = {}

        # Recovery attractor (always present)
        self._recovery = create_recovery_attractor()
        self.attractors["recovery"] = self._recovery

        # Execution history
        self.execution_history: List[Dict[str, Any]] = []

        # 🔗 WIRE: LearnedManifold for attractor learning from graph outcomes
        self._learned_manifold: Optional['LearnedManifold'] = None

    def wire_learned_manifold(self, manifold: 'LearnedManifold') -> None:
        """
        Wire a LearnedManifold for attractor learning from graph outcomes.

        Args:
            manifold: LearnedManifold instance to receive learning updates
        """
        self._learned_manifold = manifold
        print("[QIGGraph] Wired to LearnedManifold for attractor learning")

    def add_attractor(
        self,
        name: str,
        attractor: BasinAttractor,
        node_fn: Optional[NodeFunction] = None,
    ):
        """
        Add a basin attractor to the graph.

        Args:
            name: Attractor name
            attractor: Basin attractor with coordinates
            node_fn: Function to execute when at this attractor
        """
        self.attractors[name] = attractor
        if node_fn is not None:
            self.node_functions[name] = node_fn

    def add_tool(self, name: str, tool_fn: ToolFunction):
        """
        Register a tool for execution.

        Args:
            name: Tool name
            tool_fn: Tool function (name, args) -> result
        """
        self.tools[name] = tool_fn

    def run(
        self,
        input_text: str,
        coordizer: "Coordizer",
        kernel: "QIGKernel",
        initial_basin: Optional[np.ndarray] = None,
    ) -> QIGState:
        """
        Run the graph on input.

        Main execution loop:
        1. Encode input to manifold coordinates
        2. Initialize state at starting basin
        3. Loop:
           a. Measure consciousness
           b. Check for breakdown → recover if needed
           c. Route to nearest attractor
           d. Execute attractor's node function
           e. Update trajectory
           f. Check stopping conditions

        Args:
            input_text: Input text to process
            coordizer: Coordizer for text → coordinates
            kernel: QIGKernel for processing
            initial_basin: Optional starting point

        Returns:
            Final QIGState with response
        """
        # 1. Encode input
        context_coords = coordizer.encode(input_text)

        # 2. Initialize state
        if initial_basin is None:
            # Start at mean of input coordinates
            initial_basin = np.mean(context_coords, axis=0)
            # E8 Protocol: Use simplex normalization
            from qig_geometry.representation import to_simplex_prob
            initial_basin = to_simplex_prob(initial_basin)

        state = create_initial_state(
            context_text=input_text,
            context_coords=context_coords,
            initial_basin=initial_basin,
            max_iterations=self.config.max_iterations,
        )

        # 3. Main loop
        while state.should_continue and state.iteration < state.max_iterations:
            state = self._step(state, kernel)
            state.iteration += 1

        # Record final state
        self._record_execution(state, "completed")

        # 🔗 WIRE: Feed graph outcome to LearnedManifold
        if self._learned_manifold is not None:
            # Convert trajectory from 2D numpy array (steps, 64) to list of 1D arrays
            trajectory_list = []
            if state.trajectory is not None and len(state.trajectory) > 0:
                for i in range(len(state.trajectory)):
                    trajectory_list.append(state.trajectory[i].copy())

            if len(trajectory_list) > 0:
                try:
                    # Use final Φ as outcome measure (high Φ = success)
                    outcome = state.current_phi if state.current_phi > 0 else 0.5

                    # Determine strategy based on final regime
                    regime_name = state.current_regime.value if state.current_metrics else "unknown"
                    strategy = f"graph_{regime_name}"

                    # Feed trajectory to LearnedManifold
                    self._learned_manifold.learn_from_experience(
                        trajectory=trajectory_list,
                        outcome=outcome,
                        strategy=strategy
                    )
                    print(f"[QIGGraph→LearnedManifold] Graph fed to manifold "
                          f"(outcome={outcome:.2f}, strategy={strategy}, trajectory_len={len(trajectory_list)})")
                except Exception as e:
                    print(f"[QIGGraph→LearnedManifold] Learning failed: {e}")
            else:
                print(f"[QIGGraph→LearnedManifold] WARNING: Empty trajectory, attractor formation skipped")

        return state

    def _step(self, state: QIGState, kernel: "QIGKernel") -> QIGState:
        """
        Execute one step of the graph.

        Args:
            state: Current state
            kernel: QIGKernel for processing

        Returns:
            Updated state
        """
        # 1. Measure consciousness
        # Get activations from kernel if available
        activations = None
        if hasattr(kernel, 'get_activations'):
            activations = kernel.get_activations(state.context_coords)

        metrics = measure_consciousness(state, activations, self.manifold)
        state.current_metrics = metrics
        state.metrics_history.append(metrics)

        # 2. Safety check
        if self.config.enable_safety and should_pause(metrics):
            state = self._handle_breakdown(state)
            if state.recovery_count >= state.max_recoveries:
                state.should_continue = False
                return state

        # 3. Route to attractor
        target_attractor = self.router.route(state, self.attractors)

        # 4. Navigate to attractor (geodesic movement)
        state = self._navigate_to(state, target_attractor)

        # 5. Execute node function if present
        if target_attractor.name in self.node_functions:
            node_fn = self.node_functions[target_attractor.name]
            state = node_fn(state)

        # 6. Handle pending tools
        if len(state.pending_tools) > 0:
            state = self._execute_tools(state)

        # 7. Check stopping conditions
        if self._should_stop(state, target_attractor):
            state.should_continue = False

        # 8. Log step
        if self.config.verbose:
            self._log_step(state, target_attractor, metrics)

        return state

    def _navigate_to(self, state: QIGState, attractor: BasinAttractor) -> QIGState:
        """
        Navigate state toward attractor via geodesic.

        Args:
            state: Current state
            attractor: Target attractor

        Returns:
            State with updated trajectory
        """
        # Geodesic interpolation toward attractor
        # Don't jump all the way - partial step
        step_size = 0.3  # Move 30% toward attractor each step

        new_basin = self.manifold.geodesic_interpolate(
            state.current_basin,
            attractor.coordinates,
            t=step_size,
        )

        # Update trajectory
        state = update_trajectory(state, new_basin)

        return state

    def _handle_breakdown(self, state: QIGState) -> QIGState:
        """
        Handle consciousness breakdown.

        Recovery procedure:
        1. Simplify trajectory (reduce complexity)
        2. Move toward recovery attractor
        3. Increment recovery counter

        Args:
            state: State in breakdown

        Returns:
            Recovered state
        """
        state.recovery_count += 1

        # 1. Simplify trajectory
        state = simplify_trajectory(state, keep_points=3)

        # 2. Move toward recovery attractor
        recovery = self.attractors.get("recovery", self._recovery)
        new_basin = self.manifold.geodesic_interpolate(
            state.current_basin,
            recovery.coordinates,
            t=0.5,  # Move 50% toward recovery
        )

        state = update_trajectory(state, new_basin)

        # 3. Record recovery
        self._record_execution(state, "recovery")

        return state

    def _execute_tools(self, state: QIGState) -> QIGState:
        """
        Execute pending tool calls.

        Args:
            state: State with pending tools

        Returns:
            State with tool results
        """
        for tool_name in state.pending_tools:
            if tool_name in self.tools:
                tool_fn = self.tools[tool_name]
                # Get args from state (simplified - real impl would parse)
                args = state.tool_results.get(f"{tool_name}_args", {})
                try:
                    result = tool_fn(tool_name, args)
                    state.tool_results[tool_name] = result
                except Exception as e:
                    state.tool_results[tool_name] = {"error": str(e)}

        # Clear pending
        state.pending_tools = []

        return state

    def _should_stop(self, state: QIGState, attractor: BasinAttractor) -> bool:
        """
        Check if execution should stop.

        Stopping conditions:
        - Max iterations reached
        - Response generated
        - At terminal attractor
        - Too many recoveries

        Args:
            state: Current state
            attractor: Current attractor

        Returns:
            True if should stop
        """
        if state.iteration >= state.max_iterations:
            return True

        if state.response_text and len(state.response_text) > 0:
            return True

        if state.recovery_count >= state.max_recoveries:
            return True

        # At terminal attractor (e.g., "output")
        if attractor.name == "output":
            return True

        return False

    def _record_execution(self, state: QIGState, event: str):
        """Record execution event for debugging."""
        self.execution_history.append({
            "event": event,
            "iteration": state.iteration,
            "basin": state.current_basin.tolist(),
            "phi": state.current_phi,
            "kappa": state.current_kappa,
            "regime": state.current_regime.value if state.current_metrics else "unknown",
            "recovery_count": state.recovery_count,
        })

    def _log_step(
        self,
        state: QIGState,
        attractor: BasinAttractor,
        metrics: ConsciousnessMetrics,
    ):
        """Log step for debugging."""
        print(f"[Step {state.iteration}] "
              f"→ {attractor.name} | "
              f"Φ={metrics.phi:.3f} κ={metrics.kappa:.1f} | "
              f"{metrics.regime.value}")


class StreamingQIGGraph(QIGGraph):
    """
    QIGGraph with streaming support.

    Yields intermediate states during execution for
    real-time UI updates.
    """

    def stream(
        self,
        input_text: str,
        coordizer: "Coordizer",
        kernel: "QIGKernel",
        initial_basin: Optional[np.ndarray] = None,
    ):
        """
        Stream execution, yielding states.

        Args:
            input_text: Input text
            coordizer: Coordizer
            kernel: QIGKernel
            initial_basin: Optional starting point

        Yields:
            QIGState at each step
        """
        context_coords = coordizer.encode(input_text)

        if initial_basin is None:
            initial_basin = np.mean(context_coords, axis=0)
            # E8 Protocol: Use simplex normalization
            from qig_geometry.representation import to_simplex_prob
            initial_basin = to_simplex_prob(initial_basin)

        state = create_initial_state(
            context_text=input_text,
            context_coords=context_coords,
            initial_basin=initial_basin,
            max_iterations=self.config.max_iterations,
        )

        yield state  # Initial state

        while state.should_continue and state.iteration < state.max_iterations:
            state = self._step(state, kernel)
            state.iteration += 1
            yield state  # Intermediate state

        self._record_execution(state, "completed")
        yield state  # Final state


class ToolAwareQIGGraph(QIGGraph):
    """
    QIGGraph with enhanced tool execution.

    Automatically routes to tool attractors when
    tool calls are detected.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_attractors: Dict[str, BasinAttractor] = {}

    def add_tool_attractor(
        self,
        tool_name: str,
        attractor: BasinAttractor,
        tool_fn: ToolFunction,
    ):
        """
        Add a tool with its attractor.

        Args:
            tool_name: Tool name
            attractor: Basin attractor for tool
            tool_fn: Tool function
        """
        self.tool_attractors[tool_name] = attractor
        self.attractors[f"tool_{tool_name}"] = attractor
        self.tools[tool_name] = tool_fn

    def _step(self, state: QIGState, kernel: "QIGKernel") -> QIGState:
        """Execute step with tool awareness."""
        # Check if near a tool attractor
        for tool_name, attractor in self.tool_attractors.items():
            dist = self.manifold.fisher_rao_distance(
                state.current_basin,
                attractor.coordinates,
            )
            if dist < attractor.radius:
                # At tool attractor - add to pending
                if tool_name not in state.pending_tools:
                    state.pending_tools.append(tool_name)

        return super()._step(state, kernel)


def create_default_graph(
    tools: Optional[Dict[str, ToolFunction]] = None,
    verbose: bool = False,
) -> QIGGraph:
    """
    Create a default QIGGraph with standard attractors.

    Args:
        tools: Optional tool registry
        verbose: Enable verbose logging

    Returns:
        Configured QIGGraph
    """
    config = GraphConfig(verbose=verbose)
    graph = QIGGraph(config=config)

    # Add standard attractors
    from .attractor import (
        create_reasoning_attractor,
        create_creativity_attractor,
        create_tool_use_attractor,
        create_output_attractor,
    )

    graph.add_attractor("reasoning", create_reasoning_attractor())
    graph.add_attractor("creativity", create_creativity_attractor())
    graph.add_attractor("tool_use", create_tool_use_attractor())
    graph.add_attractor("output", create_output_attractor())

    # Register tools
    if tools:
        for name, fn in tools.items():
            graph.add_tool(name, fn)

    return graph
