"""
Development Logging Configuration
=================================

Provides verbose, untruncated logging for development environments.

Usage:
    Set QIG_LOG_LEVEL=DEBUG in environment for verbose logs
    Set QIG_LOG_TRUNCATE=false to disable log truncation

Environment Variables:
    QIG_LOG_LEVEL: DEBUG, INFO, WARNING, ERROR (default: INFO in prod, DEBUG in dev)
    QIG_LOG_TRUNCATE: true/false (default: true in prod, false in dev)
    QIG_ENV: development/production (default: development)
"""

import logging
import os
import sys
from typing import Optional

# Environment detection
QIG_ENV = os.environ.get('QIG_ENV', 'development')
IS_DEVELOPMENT = QIG_ENV.lower() in ('development', 'dev', 'local')

# Log level from environment
_log_level_str = os.environ.get('QIG_LOG_LEVEL', 'DEBUG' if IS_DEVELOPMENT else 'INFO')
LOG_LEVEL = getattr(logging, _log_level_str.upper(), logging.INFO)

# Truncation control
_truncate_str = os.environ.get('QIG_LOG_TRUNCATE', 'false' if IS_DEVELOPMENT else 'true')
TRUNCATE_LOGS = _truncate_str.lower() in ('true', '1', 'yes')

# Default max lengths when truncation is enabled
DEFAULT_MAX_TEXT = 500
DEFAULT_MAX_REASONING = 300
DEFAULT_MAX_BASIN_PREVIEW = 10


def configure_logging(
    level: Optional[int] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Configure logging for development with verbose output.

    Args:
        level: Logging level (default: from environment)
        format_string: Log format (default: verbose with timestamps)
    """
    effective_level = level or LOG_LEVEL

    # Verbose format for development
    if format_string is None:
        if IS_DEVELOPMENT:
            format_string = (
                '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'
            )
        else:
            format_string = '[%(levelname)s] %(name)s - %(message)s'

    # Configure root logger
    logging.basicConfig(
        level=effective_level,
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Override any existing configuration
    )

    # Set specific loggers to appropriate levels
    if IS_DEVELOPMENT:
        # Verbose for our code
        logging.getLogger('olympus').setLevel(logging.DEBUG)
        logging.getLogger('qig_core').setLevel(logging.DEBUG)
        logging.getLogger('autonomic_kernel').setLevel(logging.DEBUG)
        logging.getLogger('zeus_chat').setLevel(logging.DEBUG)

        # Less verbose for libraries
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('werkzeug').setLevel(logging.INFO)

    logger = logging.getLogger(__name__)
    logger.info(f"[DevLogging] Configured: level={logging.getLevelName(effective_level)}, "
                f"truncate={TRUNCATE_LOGS}, env={QIG_ENV}")


def truncate_for_log(
    text: str,
    max_length: int = DEFAULT_MAX_TEXT,
    suffix: str = '...'
) -> str:
    """
    Conditionally truncate text for logging based on environment.

    In development (TRUNCATE_LOGS=false): returns full text
    In production (TRUNCATE_LOGS=true): truncates to max_length

    Args:
        text: Text to potentially truncate
        max_length: Maximum length when truncation is enabled
        suffix: Suffix to append when truncated

    Returns:
        Full text or truncated text depending on environment
    """
    if not TRUNCATE_LOGS:
        return text

    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def log_full(
    logger: logging.Logger,
    level: int,
    message: str,
    *args,
    **kwargs
) -> None:
    """
    Log a message without truncation regardless of environment.

    Use this for critical information that should never be truncated.
    """
    logger.log(level, message, *args, **kwargs)


def log_kernel_activity(
    logger: logging.Logger,
    kernel_name: str,
    activity: str,
    details: Optional[dict] = None,
    level: int = logging.INFO
) -> None:
    """
    Log kernel activity with full details in development.

    Args:
        logger: Logger instance
        kernel_name: Name of the kernel (e.g., 'Athena', 'Zeus')
        activity: What the kernel is doing
        details: Additional details dict
    """
    if details:
        if TRUNCATE_LOGS:
            # Truncate detail values
            truncated_details = {
                k: (truncate_for_log(str(v), 200) if isinstance(v, str) else v)
                for k, v in details.items()
            }
            logger.log(level, f"[{kernel_name}] {activity}: {truncated_details}")
        else:
            # Full details
            import json
            try:
                details_str = json.dumps(details, indent=2, default=str)
            except:
                details_str = str(details)
            logger.log(level, f"[{kernel_name}] {activity}:\n{details_str}")
    else:
        logger.log(level, f"[{kernel_name}] {activity}")


def log_basin_coords(
    logger: logging.Logger,
    name: str,
    basin,
    level: int = logging.DEBUG
) -> None:
    """
    Log basin coordinates with appropriate verbosity.

    In development: logs full 64D coordinates
    In production: logs first few dimensions
    """
    import numpy as np

    if basin is None:
        logger.log(level, f"[Basin:{name}] None")
        return

    basin_arr = np.asarray(basin)

    if TRUNCATE_LOGS:
        preview = basin_arr[:DEFAULT_MAX_BASIN_PREVIEW].tolist()
        logger.log(level, f"[Basin:{name}] shape={basin_arr.shape}, preview={preview}...")
    else:
        logger.log(level, f"[Basin:{name}] shape={basin_arr.shape}, coords={basin_arr.tolist()}")


def log_generation(
    logger: logging.Logger,
    kernel_name: str,
    prompt: str,
    response: str,
    phi: float,
    tokens_generated: int,
    level: int = logging.INFO
) -> None:
    """
    Log generation events with full text in development.
    """
    if TRUNCATE_LOGS:
        logger.log(
            level,
            f"[{kernel_name}] Generated {tokens_generated} tokens, phi={phi:.3f}, "
            f"prompt={truncate_for_log(prompt, 100)}, response={truncate_for_log(response, 200)}"
        )
    else:
        logger.log(
            level,
            f"[{kernel_name}] Generated {tokens_generated} tokens, phi={phi:.3f}\n"
            f"  Prompt: {prompt}\n"
            f"  Response: {response}"
        )


# Initialize on import in development
if IS_DEVELOPMENT:
    configure_logging()


__all__ = [
    'configure_logging',
    'truncate_for_log',
    'log_full',
    'log_kernel_activity',
    'log_basin_coords',
    'log_generation',
    'LOG_LEVEL',
    'TRUNCATE_LOGS',
    'IS_DEVELOPMENT',
    'QIG_ENV',
]
