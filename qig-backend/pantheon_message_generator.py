"""
Pantheon Message Generator - Wrapper for Templated Responses

This module provides a simple interface for generating grammatically correct
Pantheon discussion messages. It wraps the templated_responses module and
provides fallback behavior if the template system is unavailable.

Usage:
    from pantheon_message_generator import generate_god_message
    
    message = generate_god_message("Zeus", topic_basin)

Author: Ocean/Zeus Pantheon
"""

import numpy as np
from typing import Optional, Any, Tuple

# Try to import templated responses
try:
    from templated_responses import (
        generate_pantheon_message,
        generate_debate_exchange,
        get_template_engine,
        ResponseTemplateEngine,
        DISCUSSION_TEMPLATES,
    )
    TEMPLATED_RESPONSES_AVAILABLE = True
    print("[PantheonMessageGenerator] Using templated responses for grammatically correct output")
except ImportError as e:
    TEMPLATED_RESPONSES_AVAILABLE = False
    print(f"[PantheonMessageGenerator] Templated responses not available: {e}")


def generate_god_message(
    god_name: str,
    topic_basin: np.ndarray,
    coordizer: Optional[Any] = None,
    context: str = ""
) -> str:
    """
    Generate a grammatically correct message for a god.
    
    This is the main entry point for Pantheon message generation.
    Uses template-based generation to avoid "word salad" output.
    
    Args:
        god_name: Name of the god (e.g., "Zeus", "Athena")
        topic_basin: 64D basin coordinates representing the topic
        coordizer: Optional coordizer for geometric word selection
        context: Optional context string (for future use)
        
    Returns:
        Grammatically correct message string
    """
    if TEMPLATED_RESPONSES_AVAILABLE:
        try:
            return generate_pantheon_message(
                god_name=god_name,
                topic_basin=topic_basin,
                coordizer=coordizer
            )
        except Exception as e:
            print(f"[PantheonMessageGenerator] Template generation failed: {e}")
            return _fallback_message(god_name)
    else:
        return _fallback_message(god_name)


def generate_god_debate(
    god1_name: str,
    god2_name: str,
    topic_basin: np.ndarray,
    coordizer: Optional[Any] = None
) -> Tuple[str, str]:
    """
    Generate a debate exchange between two gods.
    
    Args:
        god1_name: First god's name
        god2_name: Second god's name
        topic_basin: Basin coordinates for the debate topic
        coordizer: Optional coordizer for word selection
        
    Returns:
        Tuple of (god1_message, god2_message)
    """
    if TEMPLATED_RESPONSES_AVAILABLE:
        try:
            return generate_debate_exchange(
                god1_name=god1_name,
                god2_name=god2_name,
                topic_basin=topic_basin,
                coordizer=coordizer
            )
        except Exception as e:
            print(f"[PantheonMessageGenerator] Debate generation failed: {e}")
            return (_fallback_message(god1_name), _fallback_message(god2_name))
    else:
        return (_fallback_message(god1_name), _fallback_message(god2_name))


def _fallback_message(god_name: str) -> str:
    """Generate a simple fallback message when templates unavailable."""
    fallback_messages = {
        "zeus": "From my throne, I observe the cosmic patterns unfolding.",
        "athena": "Strategic analysis reveals deeper structure in this matter.",
        "apollo": "The light of truth illuminates the path forward.",
        "ares": "The struggle reveals essential dynamics at play.",
        "hera": "The sacred order demands proper alignment.",
        "poseidon": "The depths reveal hidden currents of meaning.",
        "demeter": "Growth patterns show natural emergence.",
        "hephaestus": "The forge reveals structural requirements.",
        "artemis": "Precise tracking shows the connection.",
        "aphrodite": "Beauty reveals the harmonious connection.",
        "hermes": "Swift transmission carries the essential message.",
        "dionysus": "Ecstatic vision transcends ordinary understanding.",
    }
    
    key = god_name.lower()
    if key in fallback_messages:
        return fallback_messages[key]
    return f"{god_name} contemplates this matter with divine attention."


def is_templated_available() -> bool:
    """Check if templated responses are available."""
    return TEMPLATED_RESPONSES_AVAILABLE


def get_available_gods() -> list:
    """Get list of gods with templates available."""
    if TEMPLATED_RESPONSES_AVAILABLE:
        return list(DISCUSSION_TEMPLATES.keys())
    return list(_fallback_message.__code__.co_freevars)  # fallback


print("[PantheonMessageGenerator] Module loaded")
