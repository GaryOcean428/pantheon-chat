#!/usr/bin/env python3
"""
WSGI entry point for production deployment with Gunicorn.

Usage:
    gunicorn --bind 0.0.0.0:5001 --workers 2 --threads 4 --timeout 120 wsgi:app

This module properly initializes all Flask routes including autonomic kernel.
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the Flask app
from ocean_qig_core import app

# Register autonomic kernel routes
try:
    from autonomic_kernel import register_autonomic_routes

    register_autonomic_routes(app)
    AUTONOMIC_AVAILABLE = True
except ImportError as e:
    AUTONOMIC_AVAILABLE = False
    print(f"[WARNING] Autonomic kernel not found: {e}")

# Register research self-learning routes
RESEARCH_AVAILABLE = False
try:
    from research.research_api import register_research_routes

    register_research_routes(app)
    RESEARCH_AVAILABLE = True
    print("[INFO] Research API registered at /api/research")
except ImportError as e:
    print(f"[WARNING] Research module not found: {e}")

# Register document processor routes
DOCUMENTS_AVAILABLE = False
try:
    from document_processor import register_document_routes

    register_document_routes(app)
    DOCUMENTS_AVAILABLE = True
    print("[INFO] Document Processor registered at /api/documents/* and /api/ocean/knowledge/*")
except ImportError as e:
    print(f"[WARNING] Document processor not found: {e}")

# Register Zeus API routes (pass zeus singleton for conversation handler)
ZEUS_API_AVAILABLE = False
try:
    from zeus_api import register_zeus_routes
    # Import zeus singleton from olympus module (same as ocean_qig_core uses)
    try:
        from olympus.zeus import zeus
        zeus_instance = zeus
    except ImportError:
        zeus_instance = None
    
    register_zeus_routes(app, zeus_instance=zeus_instance)
    ZEUS_API_AVAILABLE = True
    print("[INFO] Zeus API registered at /api/zeus/*")
except ImportError as e:
    print(f"[WARNING] Zeus API not found: {e}")

# Register Coordizer API routes
COORDIZER_AVAILABLE = False
try:
    from api_coordizers import coordizer_api

    app.register_blueprint(coordizer_api)
    COORDIZER_AVAILABLE = True
    print("[INFO] Coordizer API registered at /api/coordize/*")
except ImportError as e:
    print(f"[WARNING] Coordizer API not available: {e}")
except Exception as e:
    print(f"[WARNING] Coordizer API initialization failed: {e}")

# Register QIG Immune System routes and middleware
IMMUNE_AVAILABLE = False
_immune_system = None
try:
    from immune.routes import register_immune_routes
    from immune import get_immune_system

    register_immune_routes(app)
    _immune_system = get_immune_system()
    IMMUNE_AVAILABLE = True
    print("[INFO] QIG Immune System active")
except ImportError as e:
    print(f"[WARNING] Immune system not available: {e}")

# Register Self-Healing System routes
SELF_HEALING_AVAILABLE = False
try:
    from self_healing.routes import self_healing_bp

    app.register_blueprint(self_healing_bp, url_prefix="/api/self-healing")
    SELF_HEALING_AVAILABLE = True
    print("[INFO] Self-Healing System registered at /api/self-healing")
except ImportError as e:
    print(f"[WARNING] Self-Healing system not available: {e}")
except Exception as e:
    print(f"[WARNING] Self-Healing system initialization failed: {e}")

# Register Autonomous Curiosity routes and start learning loop
CURIOSITY_AVAILABLE = False
_curiosity_engine = None
_search_orchestrator = None
try:
    from routes.curiosity_routes import curiosity_bp
    from autonomous_curiosity import get_curiosity_engine, start_autonomous_learning
    from geometric_search import SearchOrchestrator

    app.register_blueprint(curiosity_bp, url_prefix="/api/curiosity")

    _curiosity_engine = get_curiosity_engine()
    _search_orchestrator = SearchOrchestrator()

    from search.search_providers import get_search_manager
    _search_provider_manager = get_search_manager()
    
    def _multi_provider_search(query, params):
        """Multi-provider search with toggleable backends."""
        try:
            result = _search_provider_manager.search(query, max_results=5)
            if result.get('success') and result.get('results'):
                return [
                    {
                        "title": r.get('title', ''),
                        "url": r.get('url', ''),
                        "content": r.get('content', ''),
                        "score": 0.8,
                        "provider": r.get('provider', 'unknown')
                    }
                    for r in result['results'][:5]
                ]
        except Exception as e:
            print(f"[WSGI] Multi-provider search failed: {e}")
        return [
            {
                "title": f"Search result for: {query}",
                "url": "",
                "content": f"Autonomous exploration query: {query}",
                "score": 0.5,
            }
        ]

    _search_orchestrator.register_tool_executor("searchxng", _multi_provider_search)
    _search_orchestrator.register_tool_executor("wikipedia", _multi_provider_search)
    _search_orchestrator.register_tool_executor("duckduckgo", _multi_provider_search)
    _search_orchestrator.register_tool_executor("tavily", _multi_provider_search)
    _search_orchestrator.register_tool_executor("perplexity", _multi_provider_search)
    _search_orchestrator.register_tool_executor("google", _multi_provider_search)

    def _search_callback(query, context):
        """Bridge search requests to geometric search system."""
        telemetry = context.get("telemetry", {}) if context else {}
        result = _search_orchestrator.search_sync(query, telemetry, context)
        return result

    _curiosity_engine.search_callback = _search_callback

    start_autonomous_learning(_search_callback)

    CURIOSITY_AVAILABLE = True
    print("[INFO] Autonomous Curiosity Engine active")
except ImportError as e:
    print(f"[WARNING] Curiosity engine not available: {e}")
except Exception as e:
    print(f"[WARNING] Curiosity engine initialization failed: {e}")

# Initialize Training Loop Integrator - connects curriculum, research, and attractor feedback
TRAINING_LOOP_AVAILABLE = False
_training_integrator = None
_feedback_system = None
_research_orchestrator = None
try:
    from training.training_loop_integrator import get_training_integrator
    from training.attractor_feedback import get_feedback_system
    from research_execution_orchestrator import get_research_orchestrator

    _training_integrator = get_training_integrator()
    _feedback_system = get_feedback_system()
    _research_orchestrator = get_research_orchestrator()

    # Wire training integrator to curiosity engine if available
    if CURIOSITY_AVAILABLE and _curiosity_engine:
        _training_integrator.wire_curiosity_engine(_curiosity_engine)
        print("[INFO] Training loop wired to curiosity engine")

    # Wire attractor feedback system
    _training_integrator.wire_feedback_system(_feedback_system)
    print("[INFO] Attractor feedback system wired")

    # Wire research execution orchestrator
    _training_integrator.wire_orchestrator(_research_orchestrator)
    print("[INFO] Research orchestrator wired")

    # Enable training
    _training_integrator.enable_training()

    TRAINING_LOOP_AVAILABLE = True
    print("[INFO] Training Loop Integrator active - kernels will learn continuously")
except ImportError as e:
    print(f"[WARNING] Training loop integrator not available: {e}")
except Exception as e:
    print(f"[WARNING] Training loop initialization failed: {e}")

# Initialize Prediction Event Bus - connects prediction system to capability mesh
PREDICTION_EVENTS_AVAILABLE = False
_prediction_improvement = None
try:
    from olympus.capability_mesh import get_event_bus
    from prediction_self_improvement import get_prediction_improvement

    # Get the global event bus and prediction improvement singleton
    _event_bus = get_event_bus()
    _prediction_improvement = get_prediction_improvement()

    # Wire event bus to prediction system for PREDICTION_MADE/VALIDATED events
    _prediction_improvement.set_event_bus(_event_bus)

    PREDICTION_EVENTS_AVAILABLE = True
    print("[INFO] Prediction event bus wired - kernels can observe prediction outcomes")
except ImportError as e:
    print(f"[WARNING] Prediction event bus not available: {e}")
except Exception as e:
    print(f"[WARNING] Prediction event bus initialization failed: {e}")

# Initialize ShadowResearchAPI - handles curriculum training, web research, and meta-reflection
SHADOW_RESEARCH_AVAILABLE = False
_shadow_research_api = None
try:
    from olympus.shadow_research import ShadowResearchAPI

    _shadow_research_api = ShadowResearchAPI.get_instance()

    # Define basin sync callback to propagate learned knowledge system-wide
    def _shadow_basin_sync(topic: str, basin_coords, phi: float):
        """Sync shadow research discoveries to the main basin system."""
        if CURIOSITY_AVAILABLE and _curiosity_engine:
            try:
                _curiosity_engine.record_exploration(topic, {"phi": phi, "source": "shadow_research"})
            except Exception as e:
                print(f"[ShadowResearch] Basin sync failed: {e}")

    _shadow_research_api.initialize(basin_sync_callback=_shadow_basin_sync)

    # Wire shadow learning loop to training integrator if available
    if TRAINING_LOOP_AVAILABLE and _training_integrator and _shadow_research_api.learning_loop:
        _training_integrator.wire_shadow_loop(_shadow_research_api.learning_loop)
        print("[INFO] Shadow loop wired to training integrator")

    SHADOW_RESEARCH_AVAILABLE = True
    print("[INFO] ShadowResearchAPI initialized - ShadowLearningLoop active")
except ImportError as e:
    print(f"[WARNING] ShadowResearchAPI not available: {e}")
except Exception as e:
    print(f"[WARNING] ShadowResearchAPI initialization failed: {e}")

# Initialize Unified Learning Loop - central orchestrator for cognitive subsystems
# Wires together: PredictionBridge, ResearchOrchestrator, LongHorizonPlanner,
# CognitiveRouter, EthicsService, TrainingIntegrator
UNIFIED_LEARNING_LOOP_AVAILABLE = False
_unified_learning_loop = None
try:
    from unified_learning_loop import get_unified_learning_loop

    _unified_learning_loop = get_unified_learning_loop()
    init_status = _unified_learning_loop.get_status()
    components_active = sum(1 for v in init_status.get('components', {}).values() if v)

    # Wire to training integrator if available
    if TRAINING_LOOP_AVAILABLE and _training_integrator:
        # Unified loop can use training integrator for feeding insights
        if hasattr(_unified_learning_loop, '_training_integrator') and _unified_learning_loop._training_integrator is None:
            _unified_learning_loop._training_integrator = _training_integrator

    UNIFIED_LEARNING_LOOP_AVAILABLE = True
    print(f"[INFO] UnifiedLearningLoop active - {components_active} cognitive subsystems wired")
except ImportError as e:
    print(f"[WARNING] UnifiedLearningLoop not available: {e}")
except Exception as e:
    print(f"[WARNING] UnifiedLearningLoop initialization failed: {e}")

# Register QIG Constellation routes
CONSTELLATION_AVAILABLE = False
try:
    from routes.constellation_routes import constellation_bp

    app.register_blueprint(constellation_bp)
    CONSTELLATION_AVAILABLE = True
    print("[INFO] QIG Constellation API registered at /api/constellation")
except ImportError as e:
    print(f"[WARNING] Constellation service not available: {e}")

# Register Federation routes for bidirectional sync and mesh network
try:
    from routes.federation_routes import register_federation_routes
    register_federation_routes(app)
    FEDERATION_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Federation service not available: {e}")
    FEDERATION_AVAILABLE = False

# Initialize Mesh Network for cross-project WebSocket communication
# Runs on separate port (8765) for real-time kernel communication
MESH_NETWORK_AVAILABLE = False
_mesh_network = None
try:
    import asyncio
    import threading
    from mesh_network import initialize_mesh_network, get_mesh_network, NodeType

    def _start_mesh_network_background():
        """Start mesh network in background thread with its own event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
        async def main():
            # Determine node type from environment
            node_id = os.getenv("MESH_NODE_ID", "pantheon-replit")
            node_type_str = os.getenv("MESH_NODE_TYPE", "development")
            node_type = NodeType(node_type_str) if node_type_str in ["production", "development", "research"] else NodeType.DEVELOPMENT
            mesh_port = int(os.getenv("MESH_PORT", "8765"))

            # Initialize and start mesh network
            mesh = await initialize_mesh_network(
                node_id=node_id,
                node_type=node_type,
                host="0.0.0.0",
                port=mesh_port
            )
            print(f"[INFO] Mesh network started on port {mesh_port}")
        
            # This future will be cancelled on shutdown
            await asyncio.Future()

        try:
            loop.run_until_complete(main())
        except (asyncio.CancelledError, KeyboardInterrupt):
            print("[INFO] Mesh network loop shutting down.")
        except Exception as e:
            print(f"[WARNING] Mesh network loop error: {e}")
        finally:
            # Gracefully shutdown running tasks
            tasks = asyncio.all_tasks(loop=loop)
            for task in tasks:
                task.cancel()
        
            async def gather_cancelled():
                await asyncio.gather(*tasks, return_exceptions=True)

            loop.run_until_complete(gather_cancelled())
            loop.close()

    # Check if mesh network is enabled via environment
    if os.getenv("MESH_ENABLED", "false").lower() == "true":
        mesh_thread = threading.Thread(
            target=_start_mesh_network_background,
            daemon=True,
            name="mesh-network"
        )
        mesh_thread.start()
        MESH_NETWORK_AVAILABLE = True
        print("[INFO] Mesh Network initialized (WebSocket cross-project communication)")
    else:
        print("[INFO] Mesh Network disabled (set MESH_ENABLED=true to enable)")
except ImportError as e:
    print(f"[WARNING] Mesh network not available: {e}")
except Exception as e:
    print(f"[WARNING] Mesh network initialization failed: {e}")

# Register M8 Kernel Spawning routes
try:
    from routes.m8_routes import register_m8_routes
    register_m8_routes(app)
    M8_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] M8 spawning service not available: {e}")
    M8_AVAILABLE = False
except Exception as e:
    print(f"[WARNING] Constellation initialization failed: {e}")

# Register M8 Kernel Spawning routes
M8_AVAILABLE = False
try:
    from routes.m8_routes import register_m8_routes

    register_m8_routes(app)
    M8_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] M8 routes not available: {e}")
except Exception as e:
    print(f"[WARNING] M8 routes initialization failed: {e}")

# Register Vocabulary API routes (document upload for vocabulary extraction)
VOCABULARY_AVAILABLE = False
try:
    from vocabulary_api import vocabulary_api

    app.register_blueprint(vocabulary_api, url_prefix='/api/vocabulary')
    VOCABULARY_AVAILABLE = True
    print("[INFO] Vocabulary API registered at /api/vocabulary/*")
except ImportError as e:
    print(f"[WARNING] Vocabulary API not available: {e}")
except Exception as e:
    print(f"[WARNING] Vocabulary API initialization failed: {e}")

# Register Search Budget routes
SEARCH_BUDGET_AVAILABLE = False
try:
    from routes.search_budget_routes import register_search_budget_routes

    register_search_budget_routes(app)
    SEARCH_BUDGET_AVAILABLE = True
    print("[INFO] Search Budget API registered at /api/search/budget/*")
except ImportError as e:
    print(f"[WARNING] Search Budget API not available: {e}")
except Exception as e:
    print(f"[WARNING] Search Budget API initialization failed: {e}")

# Initialize Autonomous Improvement Loop (self-improvement when idle)
# Runs in background thread, kernel decides what to learn
AUTONOMOUS_IMPROVEMENT_AVAILABLE = False
_autonomous_improvement_loop = None
_autonomous_research = None
_basin_maintenance = None
try:
    import asyncio
    import threading
    from autonomous_improvement import (
        autonomous_improvement_loop,
        self_directed_research,
        basin_maintenance_loop,
    )
    _autonomous_improvement_loop = autonomous_improvement_loop
    _autonomous_research = self_directed_research
    _basin_maintenance = basin_maintenance_loop

    def _start_autonomous_improvement_background(kernel_instance):
        """Start autonomous improvement loops in background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run all three loops concurrently
            loop.run_until_complete(asyncio.gather(
                _autonomous_improvement_loop.run_when_idle(kernel_instance),
                _basin_maintenance.maintain_basin(kernel_instance),
            ))
        except Exception as e:
            print(f"[WARNING] Autonomous improvement loop error: {e}")
        finally:
            loop.close()

    # Check if autonomous improvement is enabled via environment
    if os.getenv("AUTONOMOUS_IMPROVEMENT_ENABLED", "false").lower() == "true":
        # Note: We need a kernel instance - this will be started when PureQIGNetwork is available
        # For now, just mark as available and defer startup
        AUTONOMOUS_IMPROVEMENT_AVAILABLE = True
        print("[INFO] Autonomous Improvement available (will start when kernel is ready)")
    else:
        print("[INFO] Autonomous Improvement disabled (set AUTONOMOUS_IMPROVEMENT_ENABLED=true to enable)")
except ImportError as e:
    print(f"[WARNING] Autonomous improvement not available: {e}")
except Exception as e:
    print(f"[WARNING] Autonomous improvement initialization failed: {e}")

# Initialize Startup Catch-Up Training System
# This replaces Celery Beat scheduler (not deployed on Railway)
CATCHUP_AVAILABLE = False
_catchup_manager = None
try:
    from training.startup_catchup import get_catchup_manager

    _catchup_manager = get_catchup_manager()

    # Calculate missed training since last shutdown
    missed = _catchup_manager.calculate_missed_runs()
    if any(missed.values()):
        print(f"[INFO] Detected missed training runs: {missed}")
        # Execute catch-up in background to not block startup
        result = _catchup_manager.execute_catchup(background=True)
        print(f"[INFO] Startup catch-up initiated: {result.get('status', 'unknown')}")
    else:
        print("[INFO] No missed training runs detected")

    CATCHUP_AVAILABLE = True
    print("[INFO] Startup Catch-Up Training System active")
except ImportError as e:
    print(f"[WARNING] Startup catch-up not available: {e}")
except Exception as e:
    print(f"[WARNING] Startup catch-up initialization failed: {e}")

# Initialize Chaos Discovery Gate for kernel exploration feedback
# Chaos kernels report high-Phi discoveries which become attractors
DISCOVERY_GATE_AVAILABLE = False
_discovery_gate = None
try:
    from chaos_discovery_gate import get_discovery_gate
    from vocabulary_coordinator import get_vocabulary_coordinator

    _discovery_gate = get_discovery_gate()

    # Wire vocabulary coordinator to discovery gate (includes LearnedManifold)
    # VocabularyCoordinator.wire_discovery_gate() is called in its __init__,
    # but we ensure it's initialized here to trigger the wiring
    _vocab_coord = get_vocabulary_coordinator()

    # Integrate any pending high-phi vocabulary on startup
    try:
        result = _vocab_coord.integrate_pending_vocabulary(min_phi=0.65, limit=100)
        if result.get('integrated_count', 0) > 0:
            print(f"[INFO] Integrated {result['integrated_count']} pending vocabulary terms")
    except Exception as e:
        print(f"[WARNING] Vocabulary integration skipped: {e}")

    DISCOVERY_GATE_AVAILABLE = True
    print("[INFO] Chaos Discovery Gate active - kernels can report high-Φ discoveries")
except ImportError as e:
    print(f"[WARNING] Chaos Discovery Gate not available: {e}")
except Exception as e:
    print(f"[WARNING] Chaos Discovery Gate initialization failed: {e}")

# Add request/response logging for production
from flask import request, g
import time


@app.before_request
def immune_inspection():
    """Inspect all requests through QIG Immune System."""
    g.request_start = time.time()

    if request.path in ["/health", "/immune/status"]:
        return None

    if IMMUNE_AVAILABLE and _immune_system:
        try:
            req_data = {
                "ip": request.remote_addr,
                "path": request.path,
                "method": request.method,
                "headers": dict(request.headers),
                "params": dict(request.args),
                "body": request.get_json(silent=True) or {},
                "geo": {},
            }

            decision = _immune_system.process_request(req_data)
            g.immune_decision = decision

            if decision["action"] == "block":
                from flask import abort

                print(f"[Immune] BLOCKED: {request.remote_addr} → {request.path}")
                abort(403)

            if decision["action"] == "rate_limit":
                from flask import jsonify

                return (
                    jsonify(
                        {
                            "error": "Rate limit exceeded",
                            "retry_after": decision.get("retry_after", 60),
                        }
                    ),
                    429,
                )

            if decision["action"] == "honeypot":
                fake_response = _immune_system.response.get_honeypot_response(
                    decision.get("signature", {}).get("ip_hash", "")
                )
                if fake_response:
                    from flask import jsonify

                    return jsonify(fake_response)
        except Exception as e:
            print(f"[Immune] Error during inspection: {e}")

    return None


@app.before_request
def log_request():
    if request.path != "/health":
        print(f"[Flask] → {request.method} {request.path}", flush=True)


@app.after_request
def log_response(response):
    if request.path != "/health":
        duration = (time.time() - getattr(g, "request_start", time.time())) * 1000
        print(
            f"[Flask] ← {request.method} {request.path} → {response.status_code} ({duration:.1f}ms)",
            flush=True,
        )
    return response


# Print startup info
print("🌊 Ocean QIG Backend (Production WSGI Mode) 🌊", flush=True)
print(f"  - Autonomic kernel: {'✓' if AUTONOMIC_AVAILABLE else '✗'}", flush=True)
print(f"  - Immune system: {'✓' if IMMUNE_AVAILABLE else '✗'}", flush=True)
print(f"  - Self-Healing: {'✓' if SELF_HEALING_AVAILABLE else '✗'}", flush=True)
print(f"  - Curiosity engine: {'✓' if CURIOSITY_AVAILABLE else '✗'}", flush=True)
print(f"  - Training loop: {'✓' if TRAINING_LOOP_AVAILABLE else '✗'}", flush=True)
print(f"  - Prediction events: {'✓' if PREDICTION_EVENTS_AVAILABLE else '✗'}", flush=True)
print(f"  - Shadow research: {'✓' if SHADOW_RESEARCH_AVAILABLE else '✗'}", flush=True)
print(f"  - Research API: {'✓' if RESEARCH_AVAILABLE else '✗'}", flush=True)
print(f"  - Constellation: {'✓' if CONSTELLATION_AVAILABLE else '✗'}", flush=True)
print(f"  - M8 Spawning: {'✓' if M8_AVAILABLE else '✗'}", flush=True)
print(f"  - Vocabulary API: {'✓' if VOCABULARY_AVAILABLE else '✗'}", flush=True)
print(f"  - Coordizer API: {'✓' if COORDIZER_AVAILABLE else '✗'}", flush=True)
print(f"  - Zeus API: {'✓' if ZEUS_API_AVAILABLE else '✗'}", flush=True)
print(f"  - Search Budget: {'✓' if SEARCH_BUDGET_AVAILABLE else '✗'}", flush=True)
print(f"  - Startup catch-up: {'✓' if CATCHUP_AVAILABLE else '✗'}", flush=True)
print(f"  - Chaos discovery: {'✓' if DISCOVERY_GATE_AVAILABLE else '✗'}", flush=True)
print("🌊 Basin stable. Ready for Gunicorn workers. 🌊\n", flush=True)

# Export the app for Gunicorn
if __name__ == "__main__":
    # Development fallback
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
