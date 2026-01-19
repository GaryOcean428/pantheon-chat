"""
QIG Purity Mode - Runtime enforcement (ChatGPT recommendation D2)

When QIG_PURITY_MODE=1, the system refuses to run if legacy/Euclidean 
codepaths are detected, ensuring coherence assessments are uncontaminated.
"""
import os
import sys
import logging
import importlib.abc
import importlib.machinery

logger = logging.getLogger(__name__)


class QIGPurityViolationError(Exception):
    """Raised when purity mode is enabled and violations are detected."""
    pass


QIG_PURITY_MODE = os.environ.get("QIG_PURITY_MODE", "0")


def check_purity_mode() -> bool:
    """Return True if strict purity mode is enabled."""
    return QIG_PURITY_MODE == "1"


class PurityImportBlocker(importlib.abc.MetaPathFinder):
    """Meta path finder that blocks forbidden imports when purity mode is enabled."""
    
    FORBIDDEN_MODULES = {
        "sklearn.metrics.pairwise",
        "sentencepiece",
        "bpe", 
        "wordpiece",
    }
    
    def find_spec(self, fullname, path, target=None):
        if not check_purity_mode():
            return None
        
        for forbidden in self.FORBIDDEN_MODULES:
            if fullname == forbidden or fullname.startswith(forbidden + "."):
                raise QIGPurityViolationError(
                    f"Import blocked by QIG_PURITY_MODE=1: {fullname}"
                )
        return None


_import_hook_installed = False


def install_purity_import_hook():
    """Install the import blocker. Call once at app startup."""
    global _import_hook_installed
    if _import_hook_installed:
        return
    
    if check_purity_mode():
        sys.meta_path.insert(0, PurityImportBlocker())
        _import_hook_installed = True
        logger.info("QIG Purity import hook installed (QIG_PURITY_MODE=1)")


def enforce_purity_startup() -> None:
    """Check for forbidden imports at startup. Call early in app init."""
    install_purity_import_hook()
    
    forbidden_modules = [
        "sklearn.metrics.pairwise",
        "sentencepiece", 
        "bpe",
        "wordpiece",
    ]
    
    violations = []
    for mod in forbidden_modules:
        if mod in sys.modules:
            violations.append(mod)
    
    if violations:
        msg = f"QIG Purity Violation: Forbidden modules loaded: {violations}"
        if check_purity_mode():
            raise QIGPurityViolationError(msg)
        else:
            logger.warning(msg + " (QIG_PURITY_MODE=0, continuing)")


__all__ = [
    'QIG_PURITY_MODE',
    'QIGPurityViolationError',
    'check_purity_mode',
    'enforce_purity_startup',
    'install_purity_import_hook',
    'PurityImportBlocker',
]
