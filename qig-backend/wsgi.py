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

    def _ddg_search(query, params):
        """DuckDuckGo search for autonomous learning."""
        try:
            from search.duckduckgo_adapter import search_duckduckgo
            result = search_duckduckgo(query, max_results=5)
            if result.get('success') and result.get('results'):
                return [
                    {
                        "title": r.get('title', ''),
                        "url": r.get('url', ''),
                        "content": r.get('body', r.get('description', '')),
                        "score": 0.8,
                    }
                    for r in result['results'][:5]
                ]
        except Exception as e:
            print(f"[WSGI] DuckDuckGo search failed: {e}")
        return [
            {
                "title": f"Search result for: {query}",
                "url": "",
                "content": f"Autonomous exploration query: {query}",
                "score": 0.5,
            }
        ]

    _search_orchestrator.register_tool_executor("searchxng", _ddg_search)
    _search_orchestrator.register_tool_executor("wikipedia", _ddg_search)
    _search_orchestrator.register_tool_executor("duckduckgo", _ddg_search)

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
