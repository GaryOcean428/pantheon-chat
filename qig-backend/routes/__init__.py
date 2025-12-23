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

__all__ = ['m8_bp', 'register_m8_routes']
