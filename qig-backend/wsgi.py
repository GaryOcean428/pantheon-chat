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

# Add request/response logging for production
from flask import request, g
import time

@app.before_request
def immune_inspection():
    """Inspect all requests through QIG Immune System."""
    g.request_start = time.time()
    
    if request.path in ['/health', '/immune/status']:
        return None
    
    if IMMUNE_AVAILABLE and _immune_system:
        try:
            req_data = {
                'ip': request.remote_addr,
                'path': request.path,
                'method': request.method,
                'headers': dict(request.headers),
                'params': dict(request.args),
                'body': request.get_json(silent=True) or {},
                'geo': {}
            }
            
            decision = _immune_system.process_request(req_data)
            g.immune_decision = decision
            
            if decision['action'] == 'block':
                from flask import abort
                print(f"[Immune] BLOCKED: {request.remote_addr} → {request.path}")
                abort(403)
            
            if decision['action'] == 'rate_limit':
                from flask import jsonify
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'retry_after': decision.get('retry_after', 60)
                }), 429
            
            if decision['action'] == 'honeypot':
                fake_response = _immune_system.response.get_honeypot_response(
                    decision.get('signature', {}).get('ip_hash', '')
                )
                if fake_response:
                    from flask import jsonify
                    return jsonify(fake_response)
        except Exception as e:
            print(f"[Immune] Error during inspection: {e}")
    
    return None

@app.before_request
def log_request():
    if request.path != '/health':
        print(f"[Flask] → {request.method} {request.path}", flush=True)

@app.after_request
def log_response(response):
    if request.path != '/health':
        duration = (time.time() - getattr(g, 'request_start', time.time())) * 1000
        print(f"[Flask] ← {request.method} {request.path} → {response.status_code} ({duration:.1f}ms)", flush=True)
    return response

# Print startup info
print("🌊 Ocean QIG Backend (Production WSGI Mode) 🌊", flush=True)
print(f"  - Autonomic kernel: {'✓' if AUTONOMIC_AVAILABLE else '✗'}", flush=True)
print(f"  - Immune system: {'✓' if IMMUNE_AVAILABLE else '✗'}", flush=True)
print("🌊 Basin stable. Ready for Gunicorn workers. 🌊\n", flush=True)

# Export the app for Gunicorn
if __name__ == '__main__':
    # Development fallback
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
