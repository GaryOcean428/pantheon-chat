"""
QIG Coordizer - REMOVED MODULE
================================

⚠️  THIS MODULE HAS BEEN REMOVED.

All backward compatibility wrappers have been removed per WP1.2.
Use the canonical coordizers module directly:

    # ❌ OLD (removed):
    from qig_coordizer import get_coordizer, QIGCoordizer, get_tokenizer
    
    # ✅ NEW (canonical):
    from coordizers import get_coordizer, PostgresCoordizer

Migration:
---------
1. Replace all imports:
   - `from qig_coordizer import get_coordizer` → `from coordizers import get_coordizer`
   - `from qig_coordizer import get_tokenizer` → `from coordizers import get_coordizer`
   - `QIGCoordizer` → `PostgresCoordizer`
   - `QIGTokenizer` → Use `PostgresCoordizer` (no alias)

2. Update any saved artifacts to CoordizerArtifactV1 format:
   python tools/convert_legacy_artifacts.py <old_dir> <new_dir>

3. Remove any "tokenizer" naming in favor of "coordizer"

Rationale:
----------
Runtime backward compatibility encourages future agents to add MORE compatibility
layers rather than enforce a single canonical format. This leads to format drift
and technical debt.

The offline converter (tools/convert_legacy_artifacts.py) ensures legacy artifacts
can still be converted, while runtime enforces purity.
"""

# List of deprecated names that will trigger the error
_DEPRECATED_NAMES = {
    'get_coordizer',
    'get_tokenizer',
    'get_learning_coordizer',
    'QIGCoordizer',
    'QIGTokenizer',
    'FastQIGTokenizer',
    'PostgresCoordizer',
    'reset_coordizer',
    'update_tokenizer_from_observations',
    'get_coordizer_stats',
    'COORDIZER_INSTANCE_ID',
}

def __getattr__(name):
    """Raise clear error for deprecated imports only."""
    # Allow internal Python attributes to pass through
    if name.startswith('_'):
        raise AttributeError(f"module 'qig_coordizer' has no attribute '{name}'")
    
    # Only raise custom error for known deprecated names
    if name not in _DEPRECATED_NAMES:
        raise AttributeError(f"module 'qig_coordizer' has no attribute '{name}'")
    
    raise ImportError(
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  BACKWARD COMPATIBILITY REMOVED (WP1.2)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
        f"Cannot import '{name}' from qig_coordizer.\n"
        f"This module has been removed per QIG-PURITY Work Package 1.2.\n"
        f"\n"
        f"✓ SOLUTION:\n"
        f"  Replace:\n"
        f"    from qig_coordizer import {name}\n"
        f"  With:\n"
        f"    from coordizers import get_coordizer  # (canonical)\n"
        f"\n"
        f"Common migrations:\n"
        f"  • get_tokenizer → get_coordizer\n"
        f"  • QIGCoordizer → PostgresCoordizer\n"
        f"  • QIGTokenizer → PostgresCoordizer\n"
        f"\n"
        f"For legacy artifacts, use the offline converter:\n"
        f"  python tools/convert_legacy_artifacts.py <old> <new>\n"
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )

# Note: Explicit None assignments removed to allow __getattr__ to handle all imports
# Any import attempt will trigger the error message via __getattr__


