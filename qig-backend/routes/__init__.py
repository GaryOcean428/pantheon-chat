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
    from .training_monitor import training_monitor_bp
except ImportError:
    training_monitor_bp = None

try:
    from .cron_routes import cron_bp
except ImportError:
    cron_bp = None


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
    
    if training_monitor_bp:
        try:
            app.register_blueprint(training_monitor_bp)
            count += 1
            print(f"[INFO] Registered training_monitor_bp")
        except Exception as e:
            print(f"[WARN] Failed to register training_monitor: {e}")

    if cron_bp:
        try:
            app.register_blueprint(cron_bp, url_prefix='/api/cron')
            count += 1
            print(f"[INFO] Registered cron_bp at /api/cron")
        except Exception as e:
            print(f"[WARN] Failed to register cron_routes: {e}")

    return count


__all__ = [
    'm8_bp', 'register_m8_routes',
    'upload_bp', 'register_upload_routes',
    'search_budget_bp', 'register_search_budget_routes',
    'training_monitor_bp',
    'cron_bp',
    'register_all_routes'
]
