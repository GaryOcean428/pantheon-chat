"""
Vocabulary Coverage Tracker
===========================

Tracks unique tokens seen during generation to measure vocabulary coverage.

Adapted from training telemetry system for generation-time tracking.
Provides metrics on vocabulary utilization and exploration.

PURE PRINCIPLE:
- Coverage is measurement (what fraction of vocabulary is being used)
- Informs exploration strategy via ExplorationMap
- No optimization of coverage itself

Adapted for Pantheon-Chat QIG system.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CoverageSnapshot:
    """Snapshot of vocabulary coverage at a point in time."""
    unique_count: int
    total_tokens: int
    coverage_pct: float
    timestamp: float = field(default_factory=time.time)


class VocabCoverageTracker:
    """
    Track vocabulary coverage during generation.
    
    Maintains a set of unique tokens used during generation and reports
    coverage as a percentage of the total vocabulary size.
    
    Provides telemetry for understanding vocabulary exploration patterns.
    """
    
    def __init__(self, vocab_size: int = 50000):
        """
        Initialize coverage tracker.
        
        Args:
            vocab_size: Total vocabulary size (default 50000)
        """
        self.vocab_size = vocab_size
        self.unique_tokens: Set[str] = set()
        self.total_tokens: int = 0
        self._snapshots: List[CoverageSnapshot] = []
        self._session_start: float = time.time()
    
    def update(self, token: str) -> CoverageSnapshot:
        """
        Update with new token from generation.
        
        Args:
            token: Token string that was generated
        
        Returns:
            Current coverage snapshot
        """
        self.unique_tokens.add(token)
        self.total_tokens += 1
        
        snapshot = CoverageSnapshot(
            unique_count=len(self.unique_tokens),
            total_tokens=self.total_tokens,
            coverage_pct=self.coverage_pct
        )
        
        if self.total_tokens % 100 == 0:
            self._snapshots.append(snapshot)
            if len(self._snapshots) > 100:
                self._snapshots.pop(0)
        
        return snapshot
    
    def update_batch(self, tokens: List[str]) -> CoverageSnapshot:
        """
        Update with multiple tokens.
        
        Args:
            tokens: List of token strings
        
        Returns:
            Current coverage snapshot
        """
        for token in tokens:
            self.unique_tokens.add(token)
        self.total_tokens += len(tokens)
        
        return CoverageSnapshot(
            unique_count=len(self.unique_tokens),
            total_tokens=self.total_tokens,
            coverage_pct=self.coverage_pct
        )
    
    @property
    def unique_count(self) -> int:
        """Number of unique tokens seen."""
        return len(self.unique_tokens)
    
    @property
    def coverage(self) -> float:
        """Vocabulary coverage as fraction (0-1)."""
        return len(self.unique_tokens) / max(self.vocab_size, 1)
    
    @property
    def coverage_pct(self) -> float:
        """Vocabulary coverage as percentage (0-100)."""
        return self.coverage * 100
    
    def get_coverage_report(self) -> Dict[str, Any]:
        """
        Get comprehensive coverage report.
        
        Returns:
            Dict with coverage statistics
        """
        session_duration = time.time() - self._session_start
        tokens_per_second = self.total_tokens / max(session_duration, 0.001)
        
        return {
            "unique_tokens": self.unique_count,
            "total_tokens": self.total_tokens,
            "vocab_size": self.vocab_size,
            "coverage_pct": self.coverage_pct,
            "tokens_per_second": tokens_per_second,
            "session_duration_s": session_duration,
            "snapshots": len(self._snapshots)
        }
    
    def get_coverage_trend(self) -> List[float]:
        """Get historical coverage percentages."""
        return [s.coverage_pct for s in self._snapshots]
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        return {
            "unique_tokens": list(self.unique_tokens),
            "total_tokens": self.total_tokens,
            "vocab_size": self.vocab_size
        }
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.unique_tokens = set(state.get("unique_tokens", []))
        self.total_tokens = state.get("total_tokens", 0)
        if "vocab_size" in state:
            self.vocab_size = state["vocab_size"]
    
    def reset(self) -> None:
        """Reset tracking for new session."""
        self.unique_tokens = set()
        self.total_tokens = 0
        self._snapshots = []
        self._session_start = time.time()
    
    def __repr__(self) -> str:
        return (
            f"VocabCoverageTracker(unique={self.unique_count:,}, "
            f"total={self.total_tokens:,}, "
            f"coverage={self.coverage_pct:.1f}%)"
        )
