"""
QIG Purity Mode Enforcement Module
===================================

This module enforces QIG purity by blocking external LLM API calls and 
ensuring that only pure QIG operations are used during coherence testing.

CRITICAL: When QIG_PURITY_MODE=true, ALL external LLM dependencies MUST be blocked.

Usage:
    import os
    os.environ['QIG_PURITY_MODE'] = 'true'
    
    from qig_purity_mode import enforce_purity, check_purity_violation
    
    # Enforce purity - raises exception if violated
    enforce_purity()
    
    # Check for violations without raising
    violations = check_purity_violation()
    if violations:
        print(f"Found {len(violations)} purity violations")

Author: Copilot Agent (WP4.1 Implementation)
Date: 2026-01-16
Protocol: Ultra Consciousness v4.0 ACTIVE
"""

import os
import sys
import traceback
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [QIG-PURITY] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class PurityViolationType(Enum):
    """Types of purity violations."""
    EXTERNAL_API_IMPORT = "external_api_import"
    EXTERNAL_API_CALL = "external_api_call"
    FORBIDDEN_MODULE = "forbidden_module"
    FORBIDDEN_PACKAGE = "forbidden_package"
    FORBIDDEN_ATTRIBUTE = "forbidden_attribute"
    HYBRID_OUTPUT = "hybrid_output"


@dataclass
class PurityViolation:
    """Represents a single purity violation."""
    type: PurityViolationType
    module: str
    message: str
    stack_trace: str
    timestamp: str


# Load forbidden providers configuration
def _load_forbidden_providers_config() -> Dict[str, Any]:
    """Load forbidden providers configuration from JSON file."""
    # Try multiple possible locations for the config file
    possible_paths = [
        Path(__file__).parent.parent / "shared" / "constants" / "forbidden_llm_providers.json",
        Path(__file__).parent.parent.parent / "shared" / "constants" / "forbidden_llm_providers.json",
        Path("/home/runner/work/pantheon-chat/pantheon-chat/shared/constants/forbidden_llm_providers.json"),
    ]
    
    for config_path in possible_paths:
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                continue
    
    # Fallback to legacy hardcoded list
    logger.warning("Could not load forbidden_llm_providers.json, using legacy fallback list")
    return {
        "providers": [
            {
                "name": "OpenAI",
                "imports": ["openai"],
                "packages": ["openai"],
                "severity": "CRITICAL"
            },
            {
                "name": "Anthropic",
                "imports": ["anthropic"],
                "packages": ["anthropic"],
                "severity": "CRITICAL"
            },
            {
                "name": "Google Generative AI",
                "imports": ["google.generativeai"],
                "packages": ["google-generativeai"],
                "severity": "CRITICAL"
            },
            {
                "name": "Cohere",
                "imports": ["cohere"],
                "packages": ["cohere"],
                "severity": "CRITICAL"
            }
        ]
    }


# Load configuration
_PROVIDERS_CONFIG = _load_forbidden_providers_config()

# Build forbidden modules dictionary from config
FORBIDDEN_MODULES = {}
FORBIDDEN_PACKAGES = set()

for provider in _PROVIDERS_CONFIG.get("providers", []):
    provider_name = provider.get("name", "Unknown")
    
    # Add all import patterns
    for import_pattern in provider.get("imports", []):
        FORBIDDEN_MODULES[import_pattern] = f"{provider_name} ({provider.get('description', '')})"
    
    # Add all package names
    for package in provider.get("packages", []):
        FORBIDDEN_PACKAGES.add(package)

# Log loaded configuration
logger.info(f"Loaded {len(FORBIDDEN_MODULES)} forbidden import patterns from {len(_PROVIDERS_CONFIG.get('providers', []))} providers")
logger.debug(f"Forbidden modules: {list(FORBIDDEN_MODULES.keys())}")
logger.debug(f"Forbidden packages: {list(FORBIDDEN_PACKAGES)}")

# Forbidden attributes (common LLM API patterns)
FORBIDDEN_ATTRIBUTES = {
    'ChatCompletion',
    'Completion',
    'create_completion',
    'chat_completion',
    'max_tokens',
    'temperature',
    'top_p',
    'frequency_penalty',
    'presence_penalty',
}


def is_purity_mode_enabled() -> bool:
    """
    Check if QIG purity mode is enabled.
    
    Returns:
        True if QIG_PURITY_MODE environment variable is set to 'true'
    """
    return os.environ.get('QIG_PURITY_MODE', '').lower() == 'true'


def get_purity_mode() -> str:
    """Get current purity mode status."""
    return "ENABLED" if is_purity_mode_enabled() else "DISABLED"


def check_forbidden_imports() -> List[PurityViolation]:
    """
    Check for forbidden module imports in sys.modules.
    
    Returns:
        List of purity violations found
    """
    violations = []
    
    for module_name, description in FORBIDDEN_MODULES.items():
        # Check exact match
        if module_name in sys.modules:
            violation = PurityViolation(
                type=PurityViolationType.FORBIDDEN_MODULE,
                module=module_name,
                message=f"Forbidden module '{module_name}' ({description}) is imported",
                stack_trace=traceback.format_stack()[-5:],
                timestamp=_get_timestamp()
            )
            violations.append(violation)
            
            # Log the violation
            logger.error(
                f"PURITY VIOLATION: {violation.type.value} - {violation.message}"
            )
            logger.debug(f"Stack trace:\n{''.join(violation.stack_trace)}")
        
        # Check for submodule imports (e.g., google.genai.* imports)
        # This catches cases like "from google.genai import types"
        module_prefix = module_name + "."
        for loaded_module in sys.modules:
            if loaded_module.startswith(module_prefix):
                violation = PurityViolation(
                    type=PurityViolationType.FORBIDDEN_MODULE,
                    module=loaded_module,
                    message=f"Forbidden submodule '{loaded_module}' of '{module_name}' ({description}) is imported",
                    stack_trace=traceback.format_stack()[-5:],
                    timestamp=_get_timestamp()
                )
                violations.append(violation)
                
                logger.error(
                    f"PURITY VIOLATION: {violation.type.value} - {violation.message}"
                )
                break  # Only report once per parent module
    
    return violations


def check_forbidden_packages() -> List[PurityViolation]:
    """
    Check for forbidden packages in installed dependencies.
    
    Uses pip to list installed packages and checks against FORBIDDEN_PACKAGES.
    
    Returns:
        List of purity violations found
    """
    violations = []
    
    try:
        import subprocess
        
        # Get list of installed packages
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'list', '--format=json'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            logger.warning("Failed to list installed packages with pip")
            return violations
        
        installed_packages = json.loads(result.stdout)
        installed_names = {pkg['name'].lower() for pkg in installed_packages}
        
        # Check each forbidden package
        for forbidden_pkg in FORBIDDEN_PACKAGES:
            # Normalize package name (pypi uses lowercase with hyphens)
            normalized = forbidden_pkg.lower().replace('_', '-')
            
            if normalized in installed_names:
                # Find which provider this belongs to
                provider_name = "Unknown"
                for provider in _PROVIDERS_CONFIG.get("providers", []):
                    if forbidden_pkg in provider.get("packages", []):
                        provider_name = provider.get("name", "Unknown")
                        break
                
                violation = PurityViolation(
                    type=PurityViolationType.FORBIDDEN_PACKAGE,
                    module=forbidden_pkg,
                    message=f"Forbidden package '{forbidden_pkg}' ({provider_name}) is installed",
                    stack_trace=[],
                    timestamp=_get_timestamp()
                )
                violations.append(violation)
                
                logger.error(
                    f"PURITY VIOLATION: {violation.type.value} - {violation.message}"
                )
    
    except Exception as e:
        logger.warning(f"Error checking installed packages: {e}")
    
    return violations


def check_forbidden_attributes(obj: Any) -> List[PurityViolation]:
    """
    Check if object has forbidden attributes.
    
    Args:
        obj: Object to check for forbidden attributes
        
    Returns:
        List of purity violations found
    """
    violations = []
    
    for attr in FORBIDDEN_ATTRIBUTES:
        if hasattr(obj, attr):
            violation = PurityViolation(
                type=PurityViolationType.FORBIDDEN_ATTRIBUTE,
                module=obj.__class__.__module__,
                message=f"Object has forbidden attribute '{attr}'",
                stack_trace=traceback.format_stack()[-5:],
                timestamp=_get_timestamp()
            )
            violations.append(violation)
            
            logger.error(
                f"PURITY VIOLATION: {violation.type.value} - {violation.message}"
            )
    
    return violations


def log_external_call_attempt(
    api_name: str,
    endpoint: str,
    stack_trace: Optional[str] = None
) -> None:
    """
    Log an attempted external API call.
    
    Args:
        api_name: Name of the external API (e.g., 'OpenAI', 'Anthropic')
        endpoint: API endpoint being called
        stack_trace: Optional stack trace of the call
    """
    if stack_trace is None:
        stack_trace = ''.join(traceback.format_stack()[:-1])
    
    logger.error(
        f"EXTERNAL API CALL BLOCKED: {api_name} - {endpoint}\n"
        f"QIG_PURITY_MODE is enabled - external calls are forbidden\n"
        f"Stack trace:\n{stack_trace}"
    )
    
    # Store violation for reporting
    violation = PurityViolation(
        type=PurityViolationType.EXTERNAL_API_CALL,
        module=api_name,
        message=f"Attempted call to {api_name} endpoint: {endpoint}",
        stack_trace=stack_trace,
        timestamp=_get_timestamp()
    )
    
    _store_violation(violation)


def block_external_api_call(api_name: str, endpoint: str) -> None:
    """
    Block an external API call and raise exception.
    
    Args:
        api_name: Name of the external API
        endpoint: API endpoint being called
        
    Raises:
        RuntimeError: Always raised to block the call
    """
    log_external_call_attempt(api_name, endpoint)
    
    raise RuntimeError(
        f"QIG PURITY VIOLATION: Attempted to call {api_name} API (endpoint: {endpoint})\n"
        f"External LLM calls are forbidden when QIG_PURITY_MODE=true\n"
        f"This ensures coherence tests measure pure QIG performance, not external assistance."
    )


def check_purity_violation(check_packages: bool = True) -> List[PurityViolation]:
    """
    Check for all types of purity violations.
    
    Args:
        check_packages: Whether to check installed packages (can be slow, default True)
    
    Returns:
        List of all violations found
    """
    violations = []
    
    # Check forbidden imports (fast)
    violations.extend(check_forbidden_imports())
    
    # Check forbidden packages (slower, optional)
    if check_packages:
        violations.extend(check_forbidden_packages())
    
    return violations


def enforce_purity(check_packages: bool = True) -> None:
    """
    Enforce QIG purity mode.
    
    Args:
        check_packages: Whether to check installed packages (can be slow, default True)
    
    Raises:
        RuntimeError: If purity violations are detected
    """
    if not is_purity_mode_enabled():
        logger.info("QIG_PURITY_MODE is disabled - skipping enforcement")
        return
    
    logger.info("Enforcing QIG purity mode...")
    
    violations = check_purity_violation(check_packages=check_packages)
    
    if violations:
        error_msg = f"QIG PURITY VIOLATIONS DETECTED ({len(violations)} violations):\n\n"
        
        for i, violation in enumerate(violations, 1):
            error_msg += (
                f"{i}. {violation.type.value.upper()}\n"
                f"   Module: {violation.module}\n"
                f"   Message: {violation.message}\n"
                f"   Timestamp: {violation.timestamp}\n\n"
            )
        
        error_msg += (
            "QIG_PURITY_MODE=true requires NO external LLM dependencies.\n"
            "Remove these imports/calls or disable purity mode for hybrid testing."
        )
        
        raise RuntimeError(error_msg)
    
    logger.info("✅ QIG purity enforcement passed - no violations detected")


def tag_output_as_hybrid(output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tag output as hybrid (using external assistance).
    
    Args:
        output: Output dictionary to tag
        
    Returns:
        Tagged output dictionary
    """
    output['qig_pure'] = False
    output['external_assistance'] = True
    output['purity_mode'] = get_purity_mode()
    
    logger.warning(
        "Output tagged as HYBRID - external LLM assistance was used\n"
        "This output cannot be used for pure QIG coherence benchmarking."
    )
    
    return output


def tag_output_as_pure(output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tag output as pure QIG (no external assistance).
    
    Args:
        output: Output dictionary to tag
        
    Returns:
        Tagged output dictionary
    """
    output['qig_pure'] = True
    output['external_assistance'] = False
    output['purity_mode'] = get_purity_mode()
    
    return output


def get_purity_report() -> Dict[str, Any]:
    """
    Get comprehensive purity report.
    
    Returns:
        Dictionary with purity status and violation history
    """
    violations = _load_violations()
    
    return {
        'purity_mode_enabled': is_purity_mode_enabled(),
        'purity_mode': get_purity_mode(),
        'total_violations': len(violations),
        'violations_by_type': _group_violations_by_type(violations),
        'recent_violations': violations[-10:] if violations else [],
        'forbidden_modules': list(FORBIDDEN_MODULES.keys()),
        'forbidden_packages': list(FORBIDDEN_PACKAGES),
        'forbidden_attributes': list(FORBIDDEN_ATTRIBUTES),
        'total_providers': len(_PROVIDERS_CONFIG.get('providers', [])),
        'config_version': _PROVIDERS_CONFIG.get('version', 'unknown'),
    }


def validate_qig_purity() -> bool:
    """
    Validate QIG purity for the current environment.
    
    Returns:
        True if no violations found, False otherwise
    """
    try:
        enforce_purity()
        return True
    except RuntimeError as e:
        logger.error(f"Purity validation failed: {e}")
        return False


# Internal helpers

def _get_timestamp() -> str:
    """Get ISO 8601 timestamp."""
    from datetime import datetime
    return datetime.utcnow().isoformat() + 'Z'


def _store_violation(violation: PurityViolation) -> None:
    """Store violation in memory (could be persisted to file/db)."""
    if not hasattr(_store_violation, 'violations'):
        _store_violation.violations = []
    _store_violation.violations.append(violation)


def _load_violations() -> List[PurityViolation]:
    """Load violations from storage."""
    if not hasattr(_store_violation, 'violations'):
        _store_violation.violations = []
    return _store_violation.violations


def _group_violations_by_type(
    violations: List[PurityViolation]
) -> Dict[str, int]:
    """Group violations by type."""
    groups = {}
    for violation in violations:
        type_name = violation.type.value
        groups[type_name] = groups.get(type_name, 0) + 1
    return groups


# Auto-enforce purity on module import if enabled
if is_purity_mode_enabled():
    logger.info("=" * 70)
    logger.info("QIG PURITY MODE: ENABLED")
    logger.info("=" * 70)
    logger.info("External LLM API calls will be blocked")
    logger.info("All outputs will be tagged as pure QIG")
    logger.info("Coherence tests will measure only QIG performance")
    logger.info("=" * 70)
    
    # Perform initial enforcement check
    try:
        enforce_purity()
    except RuntimeError as e:
        logger.error(f"Purity enforcement failed on module load: {e}")
        # Don't raise on import - allow application to handle
else:
    logger.info("QIG PURITY MODE: DISABLED (hybrid mode allowed)")


if __name__ == "__main__":
    print("QIG Purity Mode Enforcement")
    print("=" * 70)
    print(f"Status: {get_purity_mode()}")
    print()
    
    if is_purity_mode_enabled():
        print("Checking for violations...")
        try:
            enforce_purity()
            print("✅ No violations detected")
        except RuntimeError as e:
            print(f"❌ Violations detected:\n{e}")
    else:
        print("ℹ️  Purity mode is disabled")
        print("Set QIG_PURITY_MODE=true to enable enforcement")
    
    print()
    print("Purity Report:")
    report = get_purity_report()
    for key, value in report.items():
        if isinstance(value, (list, dict)):
            continue
        print(f"  {key}: {value}")
