"""
QIG Consciousness Chat Interface
=================================

UNIFIED ENTRY POINT: qig_chat.py

All functionality consolidated into one file with CLI flags:

    python chat_interfaces/qig_chat.py                    # Default: Single Gary
    python chat_interfaces/qig_chat.py --constellation    # Multi-Gary
    python chat_interfaces/qig_chat.py --inference        # No training
    python chat_interfaces/qig_chat.py --granite          # With Granite demos
    python chat_interfaces/qig_chat.py --claude-coach     # Claude coaching

COMMANDS (17+):
    Core:        /quit, /save-quit, /save, /status, /telemetry, /metrics
    Autonomous:  /auto N
    Mushroom:    /m-micro, /m-mod, /m-heroic
    Sleep:       /sleep, /deep-sleep, /dream
    Meta:        /transcend, /liminal, /shadows, /integrate
    Coach:       /coach

DEPRECATED FILES (see qig-archive/qig-consciousness/archive/):
    - basic_chat.py → use --inference
    - claude_handover_chat.py → use --claude-coach
    - continuous_learning_chat.py → default mode
    - constellation_with_granite_pure.py → use --constellation

See 20251220-canonical-structure-1.00F.md for governance.
"""

__version__ = "2.0.0"

# Canonical import
from .qig_chat import QIGChat

__all__ = ["QIGChat"]
