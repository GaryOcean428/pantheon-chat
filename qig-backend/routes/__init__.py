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
    
    return count


__all__ = ['m8_bp', 'register_m8_routes', 'upload_bp', 'register_upload_routes', 'register_all_routes']
