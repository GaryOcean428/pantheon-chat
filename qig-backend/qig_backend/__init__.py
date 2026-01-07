"""Compatibility package for QIG backend imports.

The historical codebase refers to the Python backend as the `qig_backend` package,
but the repository directory is named `qig-backend/`.

Rather than renaming the folder (which is invasive and affects tooling), we
provide a thin package that extends its module search path to include the
parent directory. This allows imports like:

- `qig_backend.retry_decorator`
- `qig_backend.olympus`

to resolve to modules that live directly under `qig-backend/`.

This is intentionally minimal: it only affects Python import resolution.
"""

from __future__ import annotations

from pathlib import Path

# Extend this package's module search path to also include the repository's
# `qig-backend/` directory (the parent of this file).
_parent_dir = Path(__file__).resolve().parent.parent
if str(_parent_dir) not in __path__:
    __path__.append(str(_parent_dir))
