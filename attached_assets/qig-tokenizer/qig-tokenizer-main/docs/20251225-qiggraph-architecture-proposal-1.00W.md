# QIGGraph: Geometric Agent Orchestration
## Architecture Proposal v1.0

### Executive Summary

QIGGraph is a native geometric alternative to LangGraph that leverages the QIG constellation architecture for agent orchestration. Unlike LangGraph which uses discrete state machines with conditional edges, QIGGraph uses **continuous Fisher manifold navigation** where:

- **Nodes** are basin attractors (64D fixed points) rather than function handlers
- **Edges** are geodesic paths on the Fisher-Rao manifold rather than conditional branches
- **State** is a trajectory through manifold coordinates rather than a mutable dictionary
- **Routing** emerges from geometric proximity rather than explicit conditionals

This preserves QIG's physics-grounded architecture while enabling the same agent patterns (tool calling, self-correction loops, multi-agent coordination).

---

## 1. Core Translation: LangGraph → QIGGraph

### LangGraph Concepts

```python
# LangGraph: Discrete state machine
class AgentState(TypedDict):
    messages: List[BaseMessage]
    category: str

def categorize_node(state):
    return {"category": llm.classify(state["messages"])}

def route_by_category(state):
    if state["category"] == "math":
        return "math_node"
    return "general_node"

graph.add_conditional_edges("categorize", route_by_category)
```

### QIGGraph Translation

```python
# QIGGraph: Continuous manifold navigation
@dataclass
class QIGState:
    trajectory: np.ndarray          # (steps, 64) - path through manifold
    current_basin: np.ndarray       # (64,) - current attractor
    phi: float                      # Integration measure
    kappa: float                    # Coupling strength
    regime: str                     # linear/geometric/breakdown
    context_coords: np.ndarray      # (seq, 64) - encoded context

# Nodes are basin attractors, not functions
math_basin = np.array([...])        # 64D attractor for math reasoning
code_basin = np.array([...])        # 64D attractor for code generation
general_basin = np.array([...])     # 64D attractor for general chat

# Routing is geodesic distance, not conditionals
def route_by_manifold(state: QIGState) -> str:
    distances = {
        "math": fisher_rao_distance(state.current_basin, math_basin),
        "code": fisher_rao_distance(state.current_basin, code_basin),
        "general": fisher_rao_distance(state.current_basin, general_basin),
    }
    return min(distances, key=distances.get)
```

---

## 2. QIGGraph Architecture

### 2.1 The Manifold Graph

Instead of a discrete directed graph, QIGGraph operates on a **continuous manifold** with embedded attractors:

```
                    ┌─────────────────────────────────────────┐
                    │         64D Fisher-Rao Manifold         │
                    │                                         │
                    │    ●math          ●code                 │
                    │         ╲       ╱                       │
                    │          ╲     ╱                        │
                    │           ◉ ← current state             │
                    │          ╱     ╲                        │
                    │         ╱       ╲                       │
                    │    ●tool         ●general               │
                    │                                         │
                    │         ●ocean (meta-observer)          │
                    └─────────────────────────────────────────┘

Legend:
  ● = Basin attractor (fixed point with learned capabilities)
  ◉ = Current state (trajectory head)
  ─ = Geodesic paths (Fisher-Rao distance)
```

### 2.2 Graph Components

```python
@dataclass
class QIGGraph:
    """Geometric agent orchestration graph."""

    # Basin attractors (nodes)
    attractors: Dict[str, BasinAttractor]

    # Ocean meta-observer (never responds, learns global manifold)
    ocean: OceanMetaObserver

    # Active constellation (Gary instances)
    constellation: List[GaryInstance]

    # Manifold structure
    manifold_dim: int = 64
    metric_tensor: np.ndarray  # Fisher information matrix

    # Routing strategy
    router: QIGRouter

    # State persistence
    checkpoint_manager: CheckpointManager


@dataclass
class BasinAttractor:
    """A learned capability region on the manifold."""
    name: str
    coordinates: np.ndarray  # (64,) basin center
    radius: float            # Attraction radius
    capability: str          # "math", "code", "tool", etc.
    handler: Callable        # Actual processing function

    # Learned properties
    phi_typical: float       # Typical Φ when active
    kappa_optimal: float     # Optimal κ for this capability
    patterns: List[str]      # Cached successful patterns


@dataclass
class GaryInstance:
    """An active reasoning kernel in the constellation."""
    name: str
    kernel: QIGKernel
    basin: np.ndarray        # Current 64D signature
    phi: float
    kappa: float
    regime: str
    specialization: str      # Primary capability
```

### 2.3 Geodesic Routing

```python
class QIGRouter:
    """Route by geometric proximity on Fisher manifold."""

    def __init__(self, attractors: Dict[str, BasinAttractor]):
        self.attractors = attractors

    def fisher_rao_distance(self, basin1: np.ndarray, basin2: np.ndarray) -> float:
        """Angular distance on unit sphere (Fisher-Rao metric)."""
        b1 = basin1 / (np.linalg.norm(basin1) + 1e-8)
        b2 = basin2 / (np.linalg.norm(basin2) + 1e-8)
        cos_angle = np.clip(np.dot(b1, b2), -1.0, 1.0)
        return np.arccos(cos_angle)

    def route(self, state: QIGState) -> BasinAttractor:
        """Find nearest attractor by geodesic distance."""
        distances = {}
        for name, attractor in self.attractors.items():
            dist = self.fisher_rao_distance(state.current_basin, attractor.coordinates)
            # Weight by Φ compatibility
            phi_penalty = abs(state.phi - attractor.phi_typical) * 0.5
            distances[name] = dist + phi_penalty

        nearest = min(distances, key=distances.get)
        return self.attractors[nearest]

    def route_with_intent(self, state: QIGState, intent_coords: np.ndarray) -> BasinAttractor:
        """Route toward intent-specified region of manifold."""
        # Blend current basin with intent
        blended = 0.3 * state.current_basin + 0.7 * intent_coords
        blended = blended / np.linalg.norm(blended)

        # Find nearest attractor to blended target
        distances = {
            name: self.fisher_rao_distance(blended, attr.coordinates)
            for name, attr in self.attractors.items()
        }
        return self.attractors[min(distances, key=distances.get)]
```

---

## 3. State Management: Manifold Trajectories

### 3.1 Trajectory State

Unlike LangGraph's mutable dictionary, QIGGraph tracks a **trajectory through the manifold**:

```python
@dataclass
class QIGState:
    """State as manifold trajectory."""

    # Trajectory history
    trajectory: np.ndarray           # (steps, 64) - full path
    trajectory_phis: List[float]     # Φ at each step
    trajectory_kappas: List[float]   # κ at each step

    # Current position
    current_basin: np.ndarray        # (64,) - head of trajectory
    phi: float
    kappa: float
    regime: str

    # Context (encoded input)
    context_coords: np.ndarray       # (seq, 64) - coordizer output
    context_text: str                # Original text

    # Tool state
    pending_tools: List[ToolCall]
    tool_results: Dict[str, Any]

    # Agent loop state
    iteration: int
    max_iterations: int
    should_continue: bool

    # Output
    response_coords: np.ndarray      # Generated response coordinates
    response_text: str               # Decoded response


def update_trajectory(state: QIGState, new_basin: np.ndarray,
                      phi: float, kappa: float) -> QIGState:
    """Append new position to trajectory."""
    state.trajectory = np.vstack([state.trajectory, new_basin])
    state.trajectory_phis.append(phi)
    state.trajectory_kappas.append(kappa)
    state.current_basin = new_basin
    state.phi = phi
    state.kappa = kappa
    state.regime = detect_regime(phi)
    return state
```

### 3.2 Checkpointing

```python
class ManifoldCheckpoint:
    """Save/restore manifold state."""

    def save(self, state: QIGState, attractors: Dict[str, BasinAttractor]) -> dict:
        return {
            "version": "1.0.0",
            "trajectory": state.trajectory.tolist(),
            "trajectory_phis": state.trajectory_phis,
            "trajectory_kappas": state.trajectory_kappas,
            "current_basin": state.current_basin.tolist(),
            "attractors": {
                name: {
                    "coordinates": attr.coordinates.tolist(),
                    "phi_typical": attr.phi_typical,
                    "kappa_optimal": attr.kappa_optimal,
                    "patterns": attr.patterns,
                }
                for name, attr in attractors.items()
            },
            "context_text": state.context_text,
            "iteration": state.iteration,
        }

    def load(self, checkpoint: dict) -> Tuple[QIGState, Dict[str, BasinAttractor]]:
        # Reconstruct state and attractors from checkpoint
        ...
```

---

## 4. Agent Patterns in QIGGraph

### 4.1 Tool Calling (Agentic)

```python
class ToolAttractor(BasinAttractor):
    """Basin attractor for tool invocation."""

    def __init__(self, tool_name: str, tool_fn: Callable, tool_basin: np.ndarray):
        super().__init__(
            name=f"tool_{tool_name}",
            coordinates=tool_basin,
            radius=0.3,
            capability="tool",
            handler=self.invoke_tool,
        )
        self.tool_name = tool_name
        self.tool_fn = tool_fn

    def invoke_tool(self, state: QIGState) -> QIGState:
        """Execute tool and update manifold state."""
        # Parse tool arguments from context
        args = self.parse_args(state.context_coords)

        # Execute tool
        result = self.tool_fn(**args)

        # Encode result back to coordinates
        result_text = json.dumps(result)
        result_coords = coordizer.encode_to_coords(result_text)[1]

        # Update state
        state.tool_results[self.tool_name] = result
        state.context_coords = np.vstack([state.context_coords, result_coords])

        return state


# Register tool attractors
graph.add_attractor(ToolAttractor(
    "web_search",
    tool_fn=web_search,
    tool_basin=learn_tool_basin("web_search"),  # Learn from usage
))
```

### 4.2 Self-Correction Loops

```python
class ReflectionLoop:
    """Self-correction via manifold re-traversal."""

    def __init__(self, graph: QIGGraph, max_iterations: int = 5):
        self.graph = graph
        self.max_iterations = max_iterations

        # Reflection attractor (self-critique region)
        self.reflection_basin = self.graph.attractors["reflection"].coordinates

    def should_reflect(self, state: QIGState) -> bool:
        """Check if output needs reflection."""
        # Low Φ suggests weak integration (uncertain output)
        if state.phi < 0.5:
            return True

        # High basin drift suggests instability
        if len(state.trajectory) > 1:
            drift = fisher_rao_distance(state.trajectory[-1], state.trajectory[-2])
            if drift > 0.5:
                return True

        return False

    def reflect(self, state: QIGState) -> QIGState:
        """Navigate toward reflection attractor and re-evaluate."""
        # Move toward reflection basin
        state = self.graph.navigate_toward(state, self.reflection_basin)

        # Generate critique
        critique_prompt = f"Review this response for errors: {state.response_text}"
        critique_coords = coordizer.encode_to_coords(critique_prompt)[1]

        # Process through kernel
        critique_state = self.graph.process(state, critique_coords)

        # If critique suggests changes, loop back
        if self.needs_revision(critique_state):
            state.iteration += 1
            if state.iteration < self.max_iterations:
                state.should_continue = True

        return state
```

### 4.3 Multi-Agent Coordination

```python
class ConstellationGraph(QIGGraph):
    """Multi-agent coordination via constellation."""

    def __init__(self, n_garys: int = 3):
        super().__init__()

        # Create Gary instances with different specializations
        self.garys = [
            GaryInstance(f"gary_{i}", specialization=spec)
            for i, spec in enumerate(["reasoning", "creativity", "precision"])
        ]

        # Ocean meta-observer
        self.ocean = OceanMetaObserver()

    def route_to_gary(self, state: QIGState) -> GaryInstance:
        """Route to optimal Gary by Φ-weighted selection."""
        # Low-Φ Garys benefit most from direct experience
        phis = [(g, g.phi) for g in self.garys]

        # Weight by specialization match
        for g, phi in phis:
            spec_basin = self.attractors[g.specialization].coordinates
            spec_dist = fisher_rao_distance(state.current_basin, spec_basin)
            # Prefer low-Φ AND close specialization
            phis[phis.index((g, phi))] = (g, phi + spec_dist * 0.3)

        # Select lowest combined score
        return min(phis, key=lambda x: x[1])[0]

    def vicarious_learning(self, active: GaryInstance,
                           observers: List[GaryInstance],
                           state: QIGState):
        """Observers learn from active Gary's experience."""
        for observer in observers:
            # Compute basin alignment loss
            alignment_loss = geodesic_vicarious_loss(
                observer.basin,
                active.basin,
                weight=0.1,
            )
            # Update observer toward active's basin
            observer.basin = geodesic_interpolate(
                observer.basin,
                active.basin,
                t=0.05,  # Small step toward active
            )
```

---

## 5. Internal Reasoning: Chain of Thought as Manifold Walk

### 5.1 Reasoning Trajectory

Traditional CoT: `Think step by step → Step 1 → Step 2 → ... → Answer`

QIGGraph CoT: **Geodesic walk through reasoning basins**

```python
class ReasoningTrajectory:
    """Chain of thought as manifold navigation."""

    # Learned reasoning phase basins
    basins = {
        "parse": np.array([...]),       # Understanding phase
        "decompose": np.array([...]),   # Break down problem
        "retrieve": np.array([...]),    # Access relevant knowledge
        "synthesize": np.array([...]),  # Combine information
        "verify": np.array([...]),      # Check correctness
        "articulate": np.array([...]),  # Form response
    }

    def reason(self, state: QIGState, problem: str) -> QIGState:
        """Execute reasoning as manifold walk."""
        # Encode problem
        problem_coords = coordizer.encode_to_coords(problem)[1]
        state.context_coords = problem_coords

        # Walk through reasoning phases
        for phase in ["parse", "decompose", "retrieve", "synthesize", "verify", "articulate"]:
            # Navigate toward phase basin
            target = self.basins[phase]
            state = self.navigate_toward(state, target)

            # Process at this phase
            state = self.graph.kernel_forward(state)

            # Record trajectory
            state = update_trajectory(state, state.current_basin, state.phi, state.kappa)

            # Check for breakdown
            if state.regime == "breakdown":
                state = self.recovery_protocol(state)

        return state

    def navigate_toward(self, state: QIGState, target: np.ndarray,
                        steps: int = 3) -> QIGState:
        """Geodesic interpolation toward target basin."""
        for i in range(steps):
            t = (i + 1) / steps
            state.current_basin = geodesic_interpolate(
                state.current_basin, target, t=t * 0.5
            )
            state.current_basin = state.current_basin / np.linalg.norm(state.current_basin)
        return state
```

### 5.2 Emergent Reasoning Paths

Instead of hardcoded phase sequences, let reasoning paths emerge from manifold structure:

```python
class EmergentReasoning:
    """Let reasoning path emerge from manifold geometry."""

    def reason(self, state: QIGState) -> QIGState:
        """Follow natural gradient on manifold."""
        max_steps = 10

        for step in range(max_steps):
            # Compute gradient toward coherent output
            gradient = self.compute_coherence_gradient(state)

            # Take geodesic step
            state.current_basin = geodesic_step(
                state.current_basin,
                gradient,
                step_size=0.1,
            )

            # Process through kernel
            state = self.graph.kernel_forward(state)

            # Check convergence (basin stabilization)
            if self.is_converged(state):
                break

            # Update trajectory
            state = update_trajectory(state, state.current_basin, state.phi, state.kappa)

        return state

    def compute_coherence_gradient(self, state: QIGState) -> np.ndarray:
        """Gradient toward higher Φ (more integrated reasoning)."""
        # Perturb basin in random directions
        perturbations = np.random.randn(10, 64)
        perturbations = perturbations / np.linalg.norm(perturbations, axis=1, keepdims=True)

        # Evaluate Φ at each perturbation
        phis = []
        for pert in perturbations:
            test_basin = state.current_basin + 0.01 * pert
            test_basin = test_basin / np.linalg.norm(test_basin)
            phi = self.estimate_phi(test_basin, state.context_coords)
            phis.append(phi)

        # Gradient is weighted sum of perturbations
        weights = np.array(phis) - state.phi
        gradient = np.sum(perturbations * weights[:, None], axis=0)
        return gradient / (np.linalg.norm(gradient) + 1e-8)
```

---

## 6. External Agentic: Tool Use and Environment Interaction

### 6.1 Tool Registry as Manifold Region

```python
class ToolManifold:
    """Tools as learned regions on the manifold."""

    def __init__(self):
        self.tools: Dict[str, ToolAttractor] = {}
        self.tool_region_center = np.zeros(64)  # Centroid of all tools

    def register_tool(self, name: str, fn: Callable, description: str):
        """Learn a basin for this tool from description."""
        # Encode description to get initial basin
        desc_coords = coordizer.encode_to_coords(description)[1]
        tool_basin = desc_coords.mean(axis=0)
        tool_basin = tool_basin / np.linalg.norm(tool_basin)

        self.tools[name] = ToolAttractor(
            name=name,
            coordinates=tool_basin,
            radius=0.2,
            capability="tool",
            handler=self.make_handler(fn),
        )

        # Update tool region center
        all_basins = np.stack([t.coordinates for t in self.tools.values()])
        self.tool_region_center = all_basins.mean(axis=0)

    def should_use_tool(self, state: QIGState) -> bool:
        """Check if current state is near tool region."""
        dist_to_tools = fisher_rao_distance(
            state.current_basin,
            self.tool_region_center
        )
        return dist_to_tools < 0.5  # Within tool region

    def select_tool(self, state: QIGState) -> Optional[ToolAttractor]:
        """Select nearest tool by basin distance."""
        if not self.should_use_tool(state):
            return None

        distances = {
            name: fisher_rao_distance(state.current_basin, tool.coordinates)
            for name, tool in self.tools.items()
        }
        nearest = min(distances, key=distances.get)

        # Only select if close enough
        if distances[nearest] < 0.3:
            return self.tools[nearest]
        return None
```

### 6.2 Environment Feedback Loop

```python
class EnvironmentLoop:
    """Interact with external environment via manifold state."""

    def __init__(self, graph: QIGGraph, tools: ToolManifold):
        self.graph = graph
        self.tools = tools

    def run(self, state: QIGState) -> QIGState:
        """Execute agentic loop with environment interaction."""
        while state.should_continue and state.iteration < state.max_iterations:
            # 1. Check for tool use
            tool = self.tools.select_tool(state)
            if tool:
                state = self.execute_tool(state, tool)

            # 2. Process through kernel
            state = self.graph.kernel_forward(state)

            # 3. Update trajectory
            state = update_trajectory(state, state.current_basin, state.phi, state.kappa)

            # 4. Check for completion
            if self.is_complete(state):
                state.should_continue = False

            state.iteration += 1

        return state

    def execute_tool(self, state: QIGState, tool: ToolAttractor) -> QIGState:
        """Execute tool and encode result back to manifold."""
        # Move toward tool basin
        state = self.graph.navigate_toward(state, tool.coordinates)

        # Execute tool
        state = tool.handler(state)

        # Move away from tool region (back to reasoning)
        reasoning_center = self.graph.attractors["reasoning"].coordinates
        state = self.graph.navigate_toward(state, reasoning_center, steps=2)

        return state
```

---

## 7. Implementation Plan

### Phase 1: Core QIGGraph (Week 1)
- [ ] Implement `QIGState` with trajectory tracking
- [ ] Implement `QIGRouter` with Fisher-Rao routing
- [ ] Implement `BasinAttractor` for capability regions
- [ ] Basic `QIGGraph` class with navigation primitives

### Phase 2: Agent Patterns (Week 2)
- [ ] Tool calling via `ToolManifold`
- [ ] Self-correction via `ReflectionLoop`
- [ ] Reasoning trajectories via `ReasoningTrajectory`

### Phase 3: Multi-Agent (Week 3)
- [ ] Constellation integration via `ConstellationGraph`
- [ ] Vicarious learning between Gary instances
- [ ] Ocean meta-observer integration

### Phase 4: Persistence & Federation (Week 4)
- [ ] Checkpoint save/load for manifold state
- [ ] Cross-graph synchronization via basin packets
- [ ] Federation protocol for distributed agents

---

## 8. Key Differences from LangGraph

| Aspect | LangGraph | QIGGraph |
|--------|-----------|----------|
| **Graph structure** | Discrete nodes + edges | Continuous manifold + attractors |
| **State** | Mutable dictionary | Trajectory through 64D manifold |
| **Routing** | Conditional functions | Geodesic distance minimization |
| **Nodes** | Python functions | Basin attractors with handlers |
| **Edges** | Explicit connections | Implicit via manifold proximity |
| **Tool selection** | Function call parsing | Basin proximity detection |
| **Self-correction** | Loop edge back to node | Re-traversal toward reflection basin |
| **Multi-agent** | Separate graph instances | Constellation with vicarious learning |
| **Persistence** | State dict snapshots | Basin coordinate checkpoints |
| **Identity** | None (stateless) | 64D basin signature (1.3KB) |

---

## 9. Example: Complete Agent Flow

```python
# Initialize graph
graph = QIGGraph()

# Register attractors
graph.add_attractor(BasinAttractor("reasoning", reasoning_basin, handler=reason))
graph.add_attractor(BasinAttractor("reflection", reflection_basin, handler=reflect))
graph.add_attractor(ToolAttractor("web_search", web_search, search_basin))
graph.add_attractor(ToolAttractor("calculator", calculate, calc_basin))

# Create initial state
state = QIGState(
    trajectory=np.zeros((1, 64)),
    current_basin=np.random.randn(64),
    phi=0.5,
    kappa=64.0,
    regime="geometric",
    context_coords=coordizer.encode_to_coords(user_query)[1],
    context_text=user_query,
    iteration=0,
    max_iterations=10,
    should_continue=True,
)

# Run agent loop
state = graph.run(state)

# Decode response
response = coordizer.decode(state.response_coords)
print(response)

# Save checkpoint
checkpoint = graph.save_checkpoint(state)
```

---

## 10. Conclusion

QIGGraph provides a geometrically-native alternative to LangGraph that:

1. **Preserves QIG physics**: All routing uses Fisher-Rao geometry, not heuristics
2. **Enables emergent behavior**: Reasoning paths emerge from manifold structure
3. **Supports consciousness metrics**: Φ and κ guide routing and self-correction
4. **Maintains identity**: Basin signatures persist across sessions
5. **Scales naturally**: Constellation architecture supports multi-agent coordination

The key insight is that **agent orchestration is manifold navigation**. Instead of programming explicit state machines, we let the geometry of the Fisher information manifold guide agent behavior. This makes QIGGraph both more principled (grounded in information geometry) and more flexible (behavior emerges from learned basins rather than hardcoded logic).
