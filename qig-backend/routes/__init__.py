"""
Routes Package

Flask route blueprints for the QIG backend.
"""

# Import blueprints for easy access
try:
    from .e8_routes import e8_bp, register_e8_routes
except ImportError:
    e8_bp = None
    register_e8_routes = None

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
    from .training_monitor import training_monitor_bp
except ImportError:
    training_monitor_bp = None

try:
    from .cron_routes import cron_bp
except ImportError:
    cron_bp = None

try:
    from .confidence_routes import confidence_bp, register_confidence_routes
except ImportError:
    confidence_bp = None
    register_confidence_routes = None

try:
    from .basin_routes import basin_bp, register_basin_routes
except ImportError:
    basin_bp = None
    register_basin_routes = None

try:
    from .basin_memory_routes import basin_memory_bp, register_basin_memory_routes
except ImportError:
    basin_memory_bp = None
    register_basin_memory_routes = None

try:
    from .vocabulary_decision_routes import vocabulary_bp, register_vocabulary_routes
except ImportError:
    vocabulary_bp = None
    register_vocabulary_routes = None


def register_all_routes(app):
    """Register all route blueprints with the Flask app."""
    count = 0
    
    if register_e8_routes:
        try:
            register_e8_routes(app)
            count += 1
        except Exception as e:
            print(f"[WARN] Failed to register e8_routes: {e}")
    
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
    
    if training_monitor_bp:
        try:
            app.register_blueprint(training_monitor_bp)
            count += 1
            print("[INFO] Registered training_monitor_bp")
        except Exception as e:
            print(f"[WARN] Failed to register training_monitor: {e}")

    if cron_bp:
        try:
            app.register_blueprint(cron_bp, url_prefix='/api/cron')
            count += 1
            print("[INFO] Registered cron_bp at /api/cron")
        except Exception as e:
            print(f"[WARN] Failed to register cron_routes: {e}")

    if register_confidence_routes:
        try:
            register_confidence_routes(app)
            count += 1
        except Exception as e:
            print(f"[WARN] Failed to register confidence_routes: {e}")

    if register_basin_routes:
        try:
            register_basin_routes(app)
            count += 1
        except Exception as e:
            print(f"[WARN] Failed to register basin_routes: {e}")

    if register_basin_memory_routes:
        try:
            register_basin_memory_routes(app)
            count += 1
        except Exception as e:
            print(f"[WARN] Failed to register basin_memory_routes: {e}")

    if register_vocabulary_routes:
        try:
            register_vocabulary_routes(app)
            count += 1
        except Exception as e:
            print(f"[WARN] Failed to register vocabulary_routes: {e}")

    return count


__all__ = [
    'e8_bp', 'register_e8_routes',
    'upload_bp', 'register_upload_routes',
    'search_budget_bp', 'register_search_budget_routes',
    'training_monitor_bp',
    'cron_bp',
    'confidence_bp', 'register_confidence_routes',
    'basin_bp', 'register_basin_routes',
    'basin_memory_bp', 'register_basin_memory_routes',
    'vocabulary_bp', 'register_vocabulary_routes',
    'register_all_routes'
]
