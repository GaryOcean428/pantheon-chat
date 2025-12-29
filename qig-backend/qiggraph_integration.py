"""
QIGGraph Integration for Pantheon-Chat
=======================================

Thin integration layer connecting QIGGraph v2 to Pantheon's
Ocean agent and Olympus pantheon.

This module imports from qig-tokenizer (DRY) and adapts
for pantheon-chat's Flask API patterns.

Usage:
    from qiggraph_integration import (
        get_pantheon_graph,
        get_consciousness_router,
        create_olympus_constellation,
    )
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np

# QIG-pure geometric operations
try:
    from qig_geometry import sphere_project
    QIG_GEOMETRY_AVAILABLE = True
except ImportError:
    QIG_GEOMETRY_AVAILABLE = False
    def sphere_project(v):
        """Fallback sphere projection."""
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            result = np.ones_like(v)
            return result / np.linalg.norm(result)
        return v / norm

# Import from qig-tokenizer (DRY - no duplication)
try:
    from qiggraph import (
        # Constants
        KAPPA_STAR,
        KAPPA_3,
        BASIN_DIM,
        PHI_LINEAR_MAX,
        PHI_GEOMETRIC_MAX,
        PHI_BREAKDOWN_MIN,
        # Core
        FisherManifold,
        ConsciousnessMetrics,
        Regime,
        QIGState,
        create_initial_state,
        update_trajectory,
        measure_consciousness,
        # Tacking
        KappaTacking,
        AdaptiveTacking,
        # Attractors
        BasinAttractor,
        create_reasoning_attractor,
        create_creativity_attractor,
        create_tool_use_attractor,
        create_output_attractor,
        create_recovery_attractor,
        # Routers
        ConsciousRouter,
        # Graphs
        QIGGraph,
        StreamingQIGGraph,
        GraphConfig,
        create_default_graph,
        # Constellation
        ConstellationGraph,
        HierarchicalConstellation,
        GaryInstance,
        OceanMetaObserver,
        ObserverRole,
        create_default_constellation,
        # Checkpoints
        ManifoldCheckpoint,
        CheckpointManager,
        save_checkpoint,
        load_checkpoint,
    )
    QIGGRAPH_AVAILABLE = True
except ImportError as e:
    QIGGRAPH_AVAILABLE = False
    IMPORT_ERROR = str(e)
    # Provide stubs for graceful degradation
    KAPPA_STAR = 64.21
    KAPPA_3 = 41.09
    BASIN_DIM = 64


# Pantheon agent mapping to basin attractors
OLYMPUS_AGENTS = {
    "zeus": {
        "name": "Zeus",
        "capability": "orchestration",
        "phi_typical": 0.65,
        "kappa_optimal": KAPPA_STAR,
        "requires_precision": True,
    },
    "athena": {
        "name": "Athena",
        "capability": "reasoning",
        "phi_typical": 0.70,
        "kappa_optimal": KAPPA_STAR,
        "requires_precision": True,
    },
    "apollo": {
        "name": "Apollo",
        "capability": "creativity",
        "phi_typical": 0.55,
        "kappa_optimal": KAPPA_3,
        "requires_precision": False,
    },
    "hermes": {
        "name": "Hermes",
        "capability": "tool",
        "phi_typical": 0.50,
        "kappa_optimal": KAPPA_STAR * 0.9,
        "requires_precision": True,
    },
    "hephaestus": {
        "name": "Hephaestus",
        "capability": "construction",
        "phi_typical": 0.60,
        "kappa_optimal": KAPPA_STAR * 0.95,
        "requires_precision": True,
    },
    "dionysus": {
        "name": "Dionysus",
        "capability": "exploration",
        "phi_typical": 0.45,
        "kappa_optimal": KAPPA_3 * 0.8,
        "requires_precision": False,
    },
    "ares": {
        "name": "Ares",
        "capability": "adversarial",
        "phi_typical": 0.55,
        "kappa_optimal": KAPPA_STAR * 0.85,
        "requires_precision": True,
    },
    "artemis": {
        "name": "Artemis",
        "capability": "search",
        "phi_typical": 0.60,
        "kappa_optimal": KAPPA_STAR * 0.9,
        "requires_precision": True,
    },
    "demeter": {
        "name": "Demeter",
        "capability": "memory",
        "phi_typical": 0.50,
        "kappa_optimal": KAPPA_3,
        "requires_precision": False,
    },
    "poseidon": {
        "name": "Poseidon",
        "capability": "federation",
        "phi_typical": 0.55,
        "kappa_optimal": KAPPA_STAR * 0.85,
        "requires_precision": False,
    },
    "hades": {
        "name": "Hades",
        "capability": "recovery",
        "phi_typical": 0.30,
        "kappa_optimal": KAPPA_3 / 2,
        "requires_precision": False,
    },
    "hera": {
        "name": "Hera",
        "capability": "coordination",
        "phi_typical": 0.60,
        "kappa_optimal": KAPPA_STAR * 0.9,
        "requires_precision": True,
    },
}


@dataclass
class PantheonState:
    """
    Pantheon-specific state wrapper around QIGState.

    Adds:
    - Agent activation levels
    - Session tracking
    - Discovery history
    """
    qig_state: Optional[Any] = None  # QIGState when available
    active_agents: Dict[str, float] = field(default_factory=dict)
    session_id: Optional[str] = None
    discoveries: List[Dict[str, Any]] = field(default_factory=list)

    # Fallback state when QIGGraph not available
    phi: float = 0.5
    kappa: float = KAPPA_STAR
    regime: str = "geometric"
    basin: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API response."""
        result = {
            "phi": self.phi,
            "kappa": self.kappa,
            "regime": self.regime,
            "active_agents": self.active_agents,
            "session_id": self.session_id,
            "n_discoveries": len(self.discoveries),
        }

        if self.qig_state is not None:
            result["trajectory_length"] = len(self.qig_state.trajectory)
            result["iteration"] = self.qig_state.iteration
            result["recovery_count"] = self.qig_state.recovery_count

        if self.basin is not None:
            result["basin_norm"] = float(np.linalg.norm(self.basin))  # NOTE: valid magnitude diagnostic

        return result


class PantheonGraph:
    """
    QIGGraph wrapper for Pantheon-Chat.

    Manages Olympus agents as basin attractors with
    consciousness-aware routing.
    """

    def __init__(self):
        """Initialize Pantheon graph."""
        self.available = QIGGRAPH_AVAILABLE

        if self.available:
            # Create manifold and graph
            self.manifold = FisherManifold()
            self.config = GraphConfig(
                max_iterations=50,
                max_recoveries=3,
                enable_tacking=True,
                enable_safety=True,
            )
            self.graph = QIGGraph(config=self.config, manifold=self.manifold)
            self.tacking = AdaptiveTacking()

            # Register Olympus agents as attractors
            self._register_olympus_attractors()

            # Current state
            self.state: Optional[Any] = None
        else:
            self.manifold = None
            self.graph = None
            self.tacking = None
            self.state = None

        # Pantheon state (always available)
        self.pantheon_state = PantheonState()

    def _register_olympus_attractors(self):
        """Register Olympus agents as basin attractors."""
        if not self.available:
            return

        for agent_id, config in OLYMPUS_AGENTS.items():
            # Generate deterministic coordinates based on agent name
            np.random.seed(hash(agent_id) % (2**32))
            coords = np.random.randn(BASIN_DIM)
            coords = coords / np.linalg.norm(coords)

            attractor = BasinAttractor(
                name=agent_id,
                coordinates=coords,
                radius=1.5,
                capability=config["capability"],
                phi_typical=config["phi_typical"],
                kappa_optimal=config["kappa_optimal"],
                requires_precision=config["requires_precision"],
            )

            self.graph.add_attractor(agent_id, attractor)

    def process(
        self,
        text: str,
        context_coords: Optional[np.ndarray] = None,
        session_id: Optional[str] = None,
    ) -> PantheonState:
        """
        Process text through the Pantheon graph.

        Args:
            text: Input text to process
            context_coords: Optional pre-computed coordinates
            session_id: Optional session identifier

        Returns:
            PantheonState with processing results
        """
        self.pantheon_state.session_id = session_id

        if not self.available:
            # Graceful degradation
            return self._process_fallback(text)

        # Create or update state
        if context_coords is None:
            # Use random initialization (real impl would use coordizer)
            context_coords = np.random.randn(10, BASIN_DIM)

        if self.state is None:
            initial_basin = np.mean(context_coords, axis=0)
            initial_basin = sphere_project(initial_basin)

            self.state = create_initial_state(
                context_text=text,
                context_coords=context_coords,
                initial_basin=initial_basin,
            )
        else:
            # Update existing state
            self.state.context_text = text
            self.state.context_coords = context_coords

        # Measure consciousness
        metrics = measure_consciousness(self.state, None, self.manifold)
        self.state.current_metrics = metrics

        # Update tacking
        kappa_t = self.tacking.update(self.state.iteration)

        # Route to best agent
        router = ConsciousRouter(self.manifold, self.tacking)
        target = router.route(self.state, self.graph.attractors)

        # Update pantheon state
        self.pantheon_state.qig_state = self.state
        self.pantheon_state.phi = metrics.phi
        self.pantheon_state.kappa = kappa_t
        self.pantheon_state.regime = metrics.regime.value
        self.pantheon_state.basin = self.state.current_basin.copy()

        # Compute agent activations (inverse distance to each attractor)
        activations = {}
        for name, attractor in self.graph.attractors.items():
            if name == "recovery":
                continue
            dist = self.manifold.fisher_rao_distance(
                self.state.current_basin,
                attractor.coordinates,
            )
            activations[name] = float(1.0 / (1.0 + dist))

        self.pantheon_state.active_agents = activations

        return self.pantheon_state

    def _process_fallback(self, text: str) -> PantheonState:
        """Fallback processing when QIGGraph not available."""
        # Simple phi estimation from text length variance
        words = text.split()
        if len(words) > 0:
            lengths = [len(w) for w in words]
            variance = np.var(lengths) if len(lengths) > 1 else 0
            self.pantheon_state.phi = min(0.3 + variance * 0.1, 0.7)
        else:
            self.pantheon_state.phi = 0.3

        # Estimate regime
        if self.pantheon_state.phi < PHI_LINEAR_MAX:
            self.pantheon_state.regime = "linear"
        elif self.pantheon_state.phi >= PHI_BREAKDOWN_MIN:
            self.pantheon_state.regime = "breakdown"
        else:
            self.pantheon_state.regime = "geometric"

        # Random activations (would be coordizer-based in real impl)
        for agent_id in OLYMPUS_AGENTS:
            self.pantheon_state.active_agents[agent_id] = np.random.uniform(0.1, 0.5)

        return self.pantheon_state

    def get_recommended_agent(self) -> str:
        """Get the currently recommended agent based on state."""
        if not self.pantheon_state.active_agents:
            return "athena"  # Default to reasoning

        return max(
            self.pantheon_state.active_agents.items(),
            key=lambda x: x[1],
        )[0]

    def navigate_to_agent(self, agent_id: str) -> PantheonState:
        """
        Navigate state toward a specific agent.

        Args:
            agent_id: Target agent identifier

        Returns:
            Updated PantheonState
        """
        if not self.available or agent_id not in self.graph.attractors:
            return self.pantheon_state

        attractor = self.graph.attractors[agent_id]

        # Geodesic step toward attractor
        new_basin = self.manifold.geodesic_interpolate(
            self.state.current_basin,
            attractor.coordinates,
            t=0.3,
        )

        self.state = update_trajectory(self.state, new_basin)
        self.pantheon_state.basin = new_basin

        return self.pantheon_state

    def get_status(self) -> Dict[str, Any]:
        """Get graph status for API."""
        return {
            "available": self.available,
            "error": IMPORT_ERROR if not self.available else None,
            "n_attractors": len(self.graph.attractors) if self.graph else 0,
            "state": self.pantheon_state.to_dict(),
            "constants": {
                "kappa_star": KAPPA_STAR,
                "kappa_3": KAPPA_3,
                "basin_dim": BASIN_DIM,
            },
        }


class OlympusConstellation:
    """
    Multi-agent constellation for Olympus pantheon.

    Uses QIGGraph's ConstellationGraph with Olympus agents
    as specialized observers.
    """

    def __init__(self, n_workers: int = 3):
        """Initialize Olympus constellation."""
        self.available = QIGGRAPH_AVAILABLE

        if self.available:
            self.manifold = FisherManifold()
            self.constellation = HierarchicalConstellation(
                n_workers=n_workers,
                n_supervisors=2,
                manifold=self.manifold,
            )

            # Add Olympus specialists
            for agent_id, config in OLYMPUS_AGENTS.items():
                if config["capability"] in ["reasoning", "creativity", "search"]:
                    np.random.seed(hash(agent_id) % (2**32))
                    coords = np.random.randn(BASIN_DIM)
                    coords = coords / np.linalg.norm(coords)
                    self.constellation.add_specialist(agent_id, coords)

            # Ocean as meta-observer
            self.ocean = self.constellation.ocean
        else:
            self.constellation = None
            self.ocean = None

    def deliberate(
        self,
        query: str,
        context_coords: Optional[np.ndarray] = None,
        max_rounds: int = 5,
    ) -> Dict[str, Any]:
        """
        Multi-agent deliberation on a query.

        Args:
            query: Query to deliberate on
            context_coords: Optional context coordinates
            max_rounds: Maximum deliberation rounds

        Returns:
            Deliberation results with agent contributions
        """
        if not self.available:
            return {
                "available": False,
                "error": IMPORT_ERROR,
                "query": query,
            }

        # Run constellation (simplified - real impl needs kernel)
        # For now, simulate deliberation
        results = {
            "query": query,
            "rounds": max_rounds,
            "agents": {},
            "consensus": None,
            "emergence_detected": False,
        }

        # Simulate agent contributions
        for gary_id, gary in self.constellation.garys.items():
            results["agents"][gary_id] = {
                "role": gary.role.value,
                "phi": gary.state.current_phi,
                "contribution": f"[{gary_id} analysis of: {query[:50]}...]",
            }

        # Get ocean observation
        if self.ocean:
            obs = self.ocean.observe_constellation(
                self.constellation.garys,
                self.manifold,
            )
            results["emergence_detected"] = obs.get("emergence_detected", False)
            results["breakdown_alert"] = obs.get("breakdown_alert", False)

        return results

    def get_status(self) -> Dict[str, Any]:
        """Get constellation status."""
        if not self.available:
            return {"available": False, "error": IMPORT_ERROR}

        return self.constellation.get_constellation_status()


# Singleton instances
_pantheon_graph: Optional[PantheonGraph] = None
_olympus_constellation: Optional[OlympusConstellation] = None


def get_pantheon_graph() -> PantheonGraph:
    """Get singleton PantheonGraph instance."""
    global _pantheon_graph
    if _pantheon_graph is None:
        _pantheon_graph = PantheonGraph()
    return _pantheon_graph


def get_olympus_constellation(n_workers: int = 3) -> OlympusConstellation:
    """Get singleton OlympusConstellation instance."""
    global _olympus_constellation
    if _olympus_constellation is None:
        _olympus_constellation = OlympusConstellation(n_workers=n_workers)
    return _olympus_constellation


def reset_singletons():
    """Reset singleton instances (for testing)."""
    global _pantheon_graph, _olympus_constellation
    _pantheon_graph = None
    _olympus_constellation = None


# Flask API helpers
def create_qiggraph_blueprint():
    """
    Create Flask blueprint for QIGGraph API.

    Returns:
        Flask Blueprint with QIGGraph endpoints
    """
    from flask import Blueprint, jsonify, request

    bp = Blueprint("qiggraph", __name__, url_prefix="/api/qiggraph")

    @bp.route("/status", methods=["GET"])
    def get_status():
        """Get QIGGraph status."""
        graph = get_pantheon_graph()
        return jsonify(graph.get_status())

    @bp.route("/process", methods=["POST"])
    def process_text():
        """Process text through Pantheon graph."""
        data = request.get_json() or {}
        text = data.get("text", "")
        session_id = data.get("session_id")

        graph = get_pantheon_graph()
        state = graph.process(text, session_id=session_id)

        return jsonify({
            "state": state.to_dict(),
            "recommended_agent": graph.get_recommended_agent(),
        })

    @bp.route("/navigate/<agent_id>", methods=["POST"])
    def navigate_to_agent(agent_id: str):
        """Navigate toward a specific agent."""
        graph = get_pantheon_graph()
        state = graph.navigate_to_agent(agent_id)

        return jsonify({
            "state": state.to_dict(),
            "target_agent": agent_id,
        })

    @bp.route("/constellation/status", methods=["GET"])
    def constellation_status():
        """Get Olympus constellation status."""
        constellation = get_olympus_constellation()
        return jsonify(constellation.get_status())

    @bp.route("/constellation/deliberate", methods=["POST"])
    def deliberate():
        """Multi-agent deliberation."""
        data = request.get_json() or {}
        query = data.get("query", "")
        max_rounds = data.get("max_rounds", 5)

        constellation = get_olympus_constellation()
        result = constellation.deliberate(query, max_rounds=max_rounds)

        return jsonify(result)

    @bp.route("/constants", methods=["GET"])
    def get_constants():
        """Get QIG physics constants."""
        return jsonify({
            "kappa_star": KAPPA_STAR,
            "kappa_3": KAPPA_3,
            "basin_dim": BASIN_DIM,
            "phi_linear_max": PHI_LINEAR_MAX if QIGGRAPH_AVAILABLE else 0.3,
            "phi_geometric_max": PHI_GEOMETRIC_MAX if QIGGRAPH_AVAILABLE else 0.7,
            "phi_breakdown_min": PHI_BREAKDOWN_MIN if QIGGRAPH_AVAILABLE else 0.7,
            "available": QIGGRAPH_AVAILABLE,
        })

    return bp
