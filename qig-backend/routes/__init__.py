"""
Routes Package

Flask route blueprints for the QIG backend.
"""

# Import blueprints for easy access
try:
    from .m8_routes import m8_bp, register_m8_routes
except ImportError:
    m8_bp = None
    register_m8_routes = None

try:
    from .upload_routes import upload_bp, register_upload_routes
except ImportError:
    upload_bp = None
    register_upload_routes = None

try:
    from .search_budget_routes import search_budget_bp, register_search_budget_routes
except ImportError:
    search_budget_bp = None
    register_search_budget_routes = None

try:
    from .governance_routes import governance_bp, register_governance_routes
except ImportError:
    governance_bp = None
    register_governance_routes = None


def register_all_routes(app):
    """Register all route blueprints with the Flask app."""
    count = 0
    
    if register_m8_routes:
        try:
            register_m8_routes(app)
            count += 1
        except Exception as e:
            print(f"[WARN] Failed to register m8_routes: {e}")
    
    if register_upload_routes:
        try:
            register_upload_routes(app)
            count += 1
        except Exception as e:
            print(f"[WARN] Failed to register upload_routes: {e}")
    
    if register_search_budget_routes:
        try:
            register_search_budget_routes(app)
            count += 1
        except Exception as e:
            print(f"[WARN] Failed to register search_budget_routes: {e}")
    
    if register_governance_routes:
        try:
            register_governance_routes(app)
            count += 1
        except Exception as e:
            print(f"[WARN] Failed to register governance_routes: {e}")
    
    # Register fleet telemetry endpoint
    try:
        from capability_telemetry import CapabilityTelemetryRegistry
        from flask import jsonify
        _telemetry_registry = CapabilityTelemetryRegistry()
        
        @app.route('/api/telemetry/fleet', methods=['GET'])
        def get_fleet_telemetry():
            """Get fleet telemetry across all kernels."""
            try:
                data = _telemetry_registry.get_fleet_telemetry()
                return jsonify(data)
            except Exception as e:
                return jsonify({"error": str(e), "kernels": 0}), 500
        
        count += 1
        print("[INFO] Fleet telemetry endpoint registered at /api/telemetry/fleet")
    except Exception as e:
        print(f"[WARN] Failed to register fleet telemetry: {e}")
    
    return count


__all__ = [
    'm8_bp', 'register_m8_routes', 
    'upload_bp', 'register_upload_routes', 
    'search_budget_bp', 'register_search_budget_routes',
    'register_all_routes'
]
