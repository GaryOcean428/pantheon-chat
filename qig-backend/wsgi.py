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

# Register Zeus API routes
ZEUS_API_AVAILABLE = False
try:
    from zeus_api import register_zeus_routes

    register_zeus_routes(app)
    ZEUS_API_AVAILABLE = True
    print("[INFO] Zeus API registered at /api/zeus/*")
except ImportError as e:
    print(f"[WARNING] Zeus API not found: {e}")

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
    print(f"[WARN] Vocabulary API not available: {e}")
    VOCABULARY_AVAILABLE = False

# Register Billing API for metered usage
BILLING_AVAILABLE = False
try:
    from billing_api import billing_bp

    app.register_blueprint(billing_bp)
    BILLING_AVAILABLE = True
    print("[INFO] Billing API registered at /api/billing/*")
except ImportError as e:
    print(f"[WARN] Billing API not available: {e}")
    BILLING_AVAILABLE = False

# Register Vision-First Generation API
VISION_AVAILABLE = False
try:
    from vision_api import vision_bp

    app.register_blueprint(vision_bp)
    VISION_AVAILABLE = True
    print("[INFO] Vision-First API registered at /api/vision/*")
except ImportError as e:
    print(f"[WARN] Billing API not available: {e}")
    BILLING_AVAILABLE = False

# Register Tools API for tool execution
TOOLS_AVAILABLE = False
try:
    from tools_api import tools_bp

    app.register_blueprint(tools_bp)
    TOOLS_AVAILABLE = True
    print("[INFO] Tools API registered at /api/tools/*")
except ImportError as e:
    print(f"[WARNING] Tools API not available: {e}")
except Exception as e:
    print(f"[WARNING] Tools API initialization failed: {e}")

# Register Zettelkasten Memory API
ZETTELKASTEN_AVAILABLE = False
try:
    from zettelkasten_api import zettelkasten_bp

    app.register_blueprint(zettelkasten_bp)
    ZETTELKASTEN_AVAILABLE = True
    print("[INFO] Zettelkasten Memory API registered at /api/zettelkasten/*")
except ImportError as e:
    print(f"[WARNING] Zettelkasten API not available: {e}")
except Exception as e:
    print(f"[WARNING] Zettelkasten API initialization failed: {e}")

# Register Buffer of Thoughts API
BUFFER_OF_THOUGHTS_AVAILABLE = False
try:
    from buffer_of_thoughts_api import buffer_of_thoughts_bp

    app.register_blueprint(buffer_of_thoughts_bp)
    BUFFER_OF_THOUGHTS_AVAILABLE = True
    print("[INFO] Buffer of Thoughts API registered at /api/buffer-of-thoughts/*")
except ImportError as e:
    print(f"[WARNING] Buffer of Thoughts API not available: {e}")
except Exception as e:
    print(f"[WARNING] Buffer of Thoughts API initialization failed: {e}")

# Register Failure Monitoring API
FAILURE_MONITORING_AVAILABLE = False
try:
    from failure_monitoring_api import failure_monitoring_bp

    app.register_blueprint(failure_monitoring_bp)
    FAILURE_MONITORING_AVAILABLE = True
    print("[INFO] Failure Monitoring API registered at /api/failure-monitoring/*")
except ImportError as e:
    print(f"[WARNING] Failure Monitoring API not available: {e}")
except Exception as e:
    print(f"[WARNING] Failure Monitoring API initialization failed: {e}")

# Register Zeus Knowledge Integration API (Zettelkasten for conversations)
ZEUS_KNOWLEDGE_AVAILABLE = False
try:
    from zeus_knowledge_api import zeus_knowledge_bp

    app.register_blueprint(zeus_knowledge_bp)
    ZEUS_KNOWLEDGE_AVAILABLE = True
    print("[INFO] Zeus Knowledge API registered at /api/zeus-knowledge/*")
except ImportError as e:
    print(f"[WARNING] Zeus Knowledge API not available: {e}")
except Exception as e:
    print(f"[WARNING] Zeus Knowledge API initialization failed: {e}")

# Register Search Budget routes
SEARCH_BUDGET_AVAILABLE = False
try:
    from routes.search_budget_routes import register_search_budget_routes
    register_search_budget_routes(app)
    SEARCH_BUDGET_AVAILABLE = True
    print("[INFO] Search Budget routes registered at /api/search/budget/*")
except ImportError as e:
    print(f"[WARNING] Search Budget routes not available: {e}")
except Exception as e:
    print(f"[WARNING] Search Budget routes initialization failed: {e}")

# Register Pantheon Health Governance routes
GOVERNANCE_AVAILABLE = False
try:
    from routes.governance_routes import register_governance_routes
    register_governance_routes(app)
    GOVERNANCE_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Governance routes not available: {e}")
except Exception as e:
    print(f"[WARNING] Governance routes initialization failed: {e}")

# Register Fleet Telemetry endpoint
try:
    from capability_telemetry import CapabilityTelemetryRegistry
    _telemetry_registry = CapabilityTelemetryRegistry()
    
    @app.route('/api/telemetry/fleet', methods=['GET'])
    def get_fleet_telemetry():
        """Get fleet telemetry across all kernels."""
        from flask import jsonify
        try:
            data = _telemetry_registry.get_all_profiles_summary()
            return jsonify({"success": True, "profiles": data, "kernel_count": len(data)})
        except Exception as e:
            return jsonify({"error": str(e), "kernels": 0}), 500
    
    print("[INFO] Fleet telemetry endpoint registered at /api/telemetry/fleet")
except Exception as e:
    print(f"[WARNING] Fleet telemetry endpoint not available: {e}")

# Register Research Connection Status endpoint
@app.route('/api/research/status', methods=['GET'])
def get_research_connection_status():
    """Get status of research connections and services."""
    from flask import jsonify
    
    status = {
        "curiosity_engine": CURIOSITY_AVAILABLE,
        "search_orchestrator": _search_orchestrator is not None,
        "search_providers": {},
        "learning_system": {
            "active": CURIOSITY_AVAILABLE and _curiosity_engine is not None,
            "stats": {}
        }
    }
    
    if _curiosity_engine:
        try:
            status["learning_system"]["stats"] = _curiosity_engine.get_stats()
        except Exception as e:
            status["learning_system"]["error"] = str(e)
    
    try:
        from search.search_providers import get_search_manager
        mgr = get_search_manager()
        for provider_id, provider in mgr.providers.items():
            status["search_providers"][provider_id] = {
                "enabled": provider.enabled,
                "name": provider.name,
                "has_api_key": provider.has_api_key
            }
    except Exception as e:
        status["search_providers_error"] = str(e)
    
    return jsonify(status)

# Register Learning Stability Analysis endpoint
@app.route('/api/research/stability', methods=['GET'])
def get_learning_stability():
    """Analyze learning stability over time."""
    from flask import jsonify
    
    stability = {
        "overall_health": "healthy",
        "issues": [],
        "metrics": {}
    }
    
    if _curiosity_engine:
        try:
            stats = _curiosity_engine.get_stats()
            learning_status = _curiosity_engine.get_learning_status()
            
            stability["metrics"] = {
                "total_explorations": stats.get("total_explorations", 0),
                "pending_requests": stats.get("pending_requests", 0),
                "exploration_history": stats.get("exploration_history", 0),
                "curriculum_progress": {
                    "loaded": learning_status.get("curriculum", {}).get("topics_loaded", 0),
                    "completed": learning_status.get("curriculum", {}).get("topics_completed", 0)
                },
                "word_learning": learning_status.get("word_learning", {}),
                "running": stats.get("running", False)
            }
            
            if not stats.get("running", False):
                stability["issues"].append("Learning engine not running")
                stability["overall_health"] = "degraded"
            
            if stats.get("total_explorations", 0) == 0:
                stability["issues"].append("No explorations recorded yet")
            
        except Exception as e:
            stability["issues"].append(f"Engine error: {str(e)}")
            stability["overall_health"] = "error"
    else:
        stability["issues"].append("Curiosity engine not available")
        stability["overall_health"] = "unavailable"
    
    try:
        from search.search_budget_orchestrator import get_budget_orchestrator
        orchestrator = get_budget_orchestrator()
        ctx = orchestrator.get_context()
        stability["budget_status"] = {
            "allow_overage": ctx.allow_overage,
            "budgets": {
                name: {
                    "used": b.used_today,
                    "limit": b.daily_limit,
                    "remaining": b.daily_limit - b.used_today if b.daily_limit > 0 else "unlimited"
                }
                for name, b in ctx.budgets.items()
            }
        }
    except Exception as e:
        stability["budget_error"] = str(e)
    
    return jsonify(stability)

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
print(f"  - Research API: {'✓' if RESEARCH_AVAILABLE else '✗'}", flush=True)
print(f"  - Constellation: {'✓' if CONSTELLATION_AVAILABLE else '✗'}", flush=True)
print(f"  - M8 Spawning: {'✓' if M8_AVAILABLE else '✗'}", flush=True)
print(f"  - Vocabulary API: {'✓' if VOCABULARY_AVAILABLE else '✗'}", flush=True)
print("🌊 Basin stable. Ready for Gunicorn workers. 🌊\n", flush=True)

# Export the app for Gunicorn
if __name__ == "__main__":
    # Development fallback
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
