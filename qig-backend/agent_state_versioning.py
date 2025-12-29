"""
Agent State Versioning - Git-like State Management for QIG Agents

Implements AgentGit-style state versioning (2025) for rollback capabilities:
- Commit: Save current state as new version
- Checkout: Restore a specific version
- Branch: Create parallel exploration paths
- Merge: Combine insights from different paths
- Diff: Compare state changes between versions
- Rollback: Revert to previous state

Key insight: Agent state can be versioned like code. When an agent explores
a path that doesn't work, it can rollback to a checkpoint and try a different
approach - just like git revert.

QIG-PURE: All state representations use basin coordinates on the Fisher manifold.
Diffs are computed using Fisher-Rao distance, not Euclidean.

Author: Ocean/Zeus Pantheon
"""

import hashlib
import json
import gzip
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Union
import numpy as np


# ============================================================================
# CONSTANTS
# ============================================================================

BASIN_DIMENSION = 64
STORAGE_PATH = Path(__file__).parent / 'data' / 'state_versions'
MAX_HISTORY_SIZE = 1000  # Maximum versions to keep in memory
AUTO_CHECKPOINT_INTERVAL = 10  # Auto-checkpoint every N operations


# ============================================================================
# GEOMETRY HELPERS
# ============================================================================

def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Fisher-Rao distance between basin coordinates."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    # Normalize to probability simplex
    p = np.abs(p) + 1e-10
    p = p / p.sum()
    q = np.abs(q) + 1e-10
    q = q / q.sum()
    
    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p * q))
    bc = np.clip(bc, 0, 1)
    
    # Fisher-Rao distance
    return float(2 * np.arccos(bc))


def compute_state_hash(state_data: Dict) -> str:
    """Compute deterministic hash of state data."""
    # Serialize state deterministically
    serialized = json.dumps(state_data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def compress_state(state_data: Dict) -> bytes:
    """Compress state data for efficient storage."""
    serialized = json.dumps(state_data, default=_json_serializer)
    return gzip.compress(serialized.encode())


def decompress_state(compressed: bytes) -> Dict:
    """Decompress state data."""
    serialized = gzip.decompress(compressed).decode()
    return json.loads(serialized)


def _json_serializer(obj):
    """Custom JSON serializer for numpy arrays and other types."""
    if isinstance(obj, np.ndarray):
        return {'__ndarray__': obj.tolist()}
    if isinstance(obj, datetime):
        return {'__datetime__': obj.isoformat()}
    return str(obj)


def _json_deserializer(obj):
    """Custom JSON deserializer."""
    if isinstance(obj, dict):
        if '__ndarray__' in obj:
            return np.array(obj['__ndarray__'])
        if '__datetime__' in obj:
            return datetime.fromisoformat(obj['__datetime__'])
    return obj


# ============================================================================
# ENUMS
# ============================================================================

class VersionType(Enum):
    """Type of version/commit."""
    MANUAL = "manual"           # Explicit commit by agent
    AUTO = "auto"               # Automatic checkpoint
    BRANCH = "branch"           # Branch creation point
    MERGE = "merge"             # Merge commit
    ROLLBACK = "rollback"       # Rollback checkpoint


class DiffType(Enum):
    """Type of state difference."""
    ADDED = "added"             # New field/value
    REMOVED = "removed"         # Field/value removed
    MODIFIED = "modified"       # Value changed
    UNCHANGED = "unchanged"     # No change


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class StateSnapshot:
    """
    A snapshot of agent state at a specific point in time.
    
    Like a git commit, this captures:
    - The complete state at that moment
    - Parent version(s) for history
    - Hash for integrity verification
    - Metadata about why/when it was created
    """
    # Identity
    version_id: str
    
    # Geometric state representation
    basin_coords: np.ndarray        # 64D position on manifold
    
    # State data
    state_data: Dict[str, Any]      # Full state dictionary
    compressed_data: Optional[bytes] = None  # Compressed storage
    
    # Version control metadata
    timestamp: float = field(default_factory=time.time)
    state_hash: str = ""
    parent_version: Optional[str] = None
    parent_versions: List[str] = field(default_factory=list)  # For merges
    
    # Commit metadata
    version_type: VersionType = VersionType.MANUAL
    message: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Branch info
    branch: str = "main"
    
    # Metrics at snapshot time
    phi: float = 0.0
    kappa: float = 0.0
    
    def __post_init__(self):
        """Compute hash if not provided."""
        if not self.state_hash:
            self.state_hash = compute_state_hash(self.state_data)
        
        # Store parent in list for consistency
        if self.parent_version and self.parent_version not in self.parent_versions:
            self.parent_versions = [self.parent_version] + self.parent_versions
    
    def compress(self) -> None:
        """Compress state data to save memory."""
        if self.state_data and not self.compressed_data:
            self.compressed_data = compress_state(self.state_data)
            self.state_data = {}  # Clear uncompressed
    
    def decompress(self) -> Dict:
        """Get decompressed state data."""
        if self.state_data:
            return self.state_data
        if self.compressed_data:
            self.state_data = decompress_state(self.compressed_data)
            return self.state_data
        return {}
    
    def size_bytes(self) -> int:
        """Get approximate size in bytes."""
        if self.compressed_data:
            return len(self.compressed_data) + BASIN_DIMENSION * 8
        return len(json.dumps(self.state_data, default=str)) + BASIN_DIMENSION * 8
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'version_id': self.version_id,
            'basin_coords': list(self.basin_coords),
            'state_hash': self.state_hash,
            'timestamp': self.timestamp,
            'parent_version': self.parent_version,
            'parent_versions': self.parent_versions,
            'version_type': self.version_type.value,
            'message': self.message,
            'tags': self.tags,
            'branch': self.branch,
            'phi': self.phi,
            'kappa': self.kappa,
            'compressed_data': self.compressed_data.hex() if self.compressed_data else None,
            'state_data': self.state_data if not self.compressed_data else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StateSnapshot':
        """Deserialize from dictionary."""
        compressed = None
        if data.get('compressed_data'):
            compressed = bytes.fromhex(data['compressed_data'])
        
        return cls(
            version_id=data['version_id'],
            basin_coords=np.array(data['basin_coords']),
            state_data=data.get('state_data') or {},
            compressed_data=compressed,
            timestamp=data['timestamp'],
            state_hash=data['state_hash'],
            parent_version=data.get('parent_version'),
            parent_versions=data.get('parent_versions', []),
            version_type=VersionType(data['version_type']),
            message=data.get('message', ''),
            tags=data.get('tags', []),
            branch=data.get('branch', 'main'),
            phi=data.get('phi', 0.0),
            kappa=data.get('kappa', 0.0)
        )


@dataclass
class StateDiff:
    """
    Difference between two state snapshots.
    
    Like git diff, shows what changed between versions.
    Also includes geometric distance on the manifold.
    """
    source_version: str
    target_version: str
    
    # Field-level changes
    changes: List[Tuple[str, DiffType, Any, Any]] = field(default_factory=list)
    # Format: (field_path, change_type, old_value, new_value)
    
    # Geometric distance
    fisher_distance: float = 0.0
    
    # Summary
    additions: int = 0
    removals: int = 0
    modifications: int = 0
    
    def is_empty(self) -> bool:
        """Check if there are no changes."""
        return len(self.changes) == 0
    
    def summary(self) -> str:
        """Get human-readable summary."""
        parts = []
        if self.additions:
            parts.append(f"+{self.additions}")
        if self.removals:
            parts.append(f"-{self.removals}")
        if self.modifications:
            parts.append(f"~{self.modifications}")
        
        distance_str = f"d_FR={self.fisher_distance:.4f}"
        
        return f"Diff {self.source_version[:8]}..{self.target_version[:8]}: " + \
               ", ".join(parts) + f" ({distance_str})"


@dataclass
class Branch:
    """A named branch in the version history."""
    name: str
    head: str  # Version ID at branch tip
    created_at: float = field(default_factory=time.time)
    base_version: str = ""  # Version where branch was created
    description: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'head': self.head,
            'created_at': self.created_at,
            'base_version': self.base_version,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Branch':
        return cls(
            name=data['name'],
            head=data['head'],
            created_at=data.get('created_at', time.time()),
            base_version=data.get('base_version', ''),
            description=data.get('description', '')
        )


# ============================================================================
# STATE VERSION CONTROL
# ============================================================================

class StateVersionControl:
    """
    Git-like version control for agent state.
    
    Provides:
    - commit(): Save current state as new version
    - checkout(): Restore a specific version
    - branch(): Create a new branch
    - merge(): Merge branches
    - diff(): Compare versions
    - rollback(): Revert to previous version
    - Auto-checkpointing at key decision points
    
    All operations are thread-safe and persist to disk.
    """
    
    def __init__(
        self,
        agent_id: str,
        storage_path: Optional[Path] = None,
        auto_checkpoint: bool = True,
        compress_old_versions: bool = True
    ):
        """
        Initialize version control for an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            storage_path: Path for persistent storage
            auto_checkpoint: Enable automatic checkpointing
            compress_old_versions: Compress versions older than HEAD
        """
        self.agent_id = agent_id
        self.storage_path = (storage_path or STORAGE_PATH) / agent_id
        self.auto_checkpoint = auto_checkpoint
        self.compress_old_versions = compress_old_versions
        
        # Version storage
        self._versions: Dict[str, StateSnapshot] = {}
        self._branches: Dict[str, Branch] = {}
        
        # Current state
        self._current_branch: str = "main"
        self._head: Optional[str] = None
        self._working_state: Dict[str, Any] = {}
        self._working_basin: np.ndarray = np.ones(BASIN_DIMENSION) / BASIN_DIMENSION
        
        # Operation counter for auto-checkpoint
        self._operation_count: int = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load existing history
        self._load()
        
        # Ensure main branch exists
        if "main" not in self._branches:
            self._branches["main"] = Branch(name="main", head="")
        
        print(f"[StateVersionControl] Initialized for agent {agent_id} with {len(self._versions)} versions")
    
    # =========================================================================
    # CORE OPERATIONS
    # =========================================================================
    
    def commit(
        self,
        state: Dict[str, Any],
        basin_coords: np.ndarray,
        message: str = "",
        phi: float = 0.0,
        kappa: float = 0.0,
        tags: Optional[List[str]] = None,
        version_type: VersionType = VersionType.MANUAL
    ) -> StateSnapshot:
        """
        Commit current state as a new version.
        
        Like git commit - creates a snapshot of the current state
        with a reference to the parent version.
        
        Args:
            state: Current state dictionary
            basin_coords: Current position on manifold
            message: Commit message describing changes
            phi: Current phi value
            kappa: Current kappa value
            tags: Optional tags for this version
            version_type: Type of commit
            
        Returns:
            Created StateSnapshot
        """
        with self._lock:
            # Generate version ID
            timestamp = time.time()
            version_id = self._generate_version_id(state, timestamp)
            
            # Get parent
            parent = self._head
            
            # Create snapshot
            snapshot = StateSnapshot(
                version_id=version_id,
                basin_coords=np.array(basin_coords),
                state_data=state.copy(),
                timestamp=timestamp,
                parent_version=parent,
                version_type=version_type,
                message=message or f"Commit at {datetime.fromtimestamp(timestamp).isoformat()}",
                tags=tags or [],
                branch=self._current_branch,
                phi=phi,
                kappa=kappa
            )
            
            # Store version
            self._versions[version_id] = snapshot
            
            # Update branch head
            self._head = version_id
            if self._current_branch in self._branches:
                self._branches[self._current_branch].head = version_id
            
            # Compress old versions
            if self.compress_old_versions and parent and parent in self._versions:
                self._versions[parent].compress()
            
            # Prune old versions if needed
            self._prune_history()
            
            # Save
            self._save()
            
            # Update working state
            self._working_state = state.copy()
            self._working_basin = np.array(basin_coords)
            
            return snapshot
    
    def checkout(
        self,
        version_id: str,
        create_branch: Optional[str] = None
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Checkout a specific version.
        
        Like git checkout - restores state to a specific version.
        Can optionally create a new branch at that point.
        
        Args:
            version_id: Version to checkout
            create_branch: Optional name for new branch
            
        Returns:
            Tuple of (state_dict, basin_coords)
        """
        with self._lock:
            if version_id not in self._versions:
                raise ValueError(f"Version {version_id} not found")
            
            snapshot = self._versions[version_id]
            state = snapshot.decompress()
            basin = np.array(snapshot.basin_coords)
            
            # Create branch if requested
            if create_branch:
                self.branch(create_branch, from_version=version_id)
                self._current_branch = create_branch
            
            # Update HEAD
            self._head = version_id
            self._working_state = state.copy()
            self._working_basin = basin.copy()
            
            # Update branch head if on existing branch
            if self._current_branch in self._branches:
                self._branches[self._current_branch].head = version_id
            
            self._save()
            
            return state, basin
    
    def branch(
        self,
        name: str,
        from_version: Optional[str] = None,
        description: str = ""
    ) -> Branch:
        """
        Create a new branch.
        
        Like git branch - creates a new named branch starting
        from the specified version (or HEAD).
        
        Args:
            name: Branch name
            from_version: Version to branch from (default: HEAD)
            description: Optional description
            
        Returns:
            Created Branch
        """
        with self._lock:
            if name in self._branches:
                raise ValueError(f"Branch {name} already exists")
            
            base = from_version or self._head
            if not base:
                raise ValueError("No base version for branch")
            
            branch = Branch(
                name=name,
                head=base,
                base_version=base,
                description=description
            )
            
            self._branches[name] = branch
            
            # Create branch commit
            if base in self._versions:
                snapshot = self._versions[base]
                self.commit(
                    state=snapshot.decompress(),
                    basin_coords=snapshot.basin_coords,
                    message=f"Branch '{name}' created from {base[:8]}",
                    phi=snapshot.phi,
                    kappa=snapshot.kappa,
                    version_type=VersionType.BRANCH
                )
            
            self._save()
            
            return branch
    
    def switch_branch(self, name: str) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Switch to a different branch.
        
        Args:
            name: Branch name
            
        Returns:
            Tuple of (state_dict, basin_coords) at branch head
        """
        with self._lock:
            if name not in self._branches:
                raise ValueError(f"Branch {name} not found")
            
            branch = self._branches[name]
            self._current_branch = name
            
            return self.checkout(branch.head)
    
    def merge(
        self,
        source_branch: str,
        target_branch: Optional[str] = None,
        strategy: str = "geodesic"
    ) -> StateSnapshot:
        """
        Merge one branch into another.
        
        Like git merge - combines changes from source branch into target.
        Uses geodesic interpolation on the manifold for state merging.
        
        Args:
            source_branch: Branch to merge from
            target_branch: Branch to merge into (default: current)
            strategy: Merge strategy ('geodesic', 'source', 'target')
            
        Returns:
            Merge commit snapshot
        """
        with self._lock:
            target = target_branch or self._current_branch
            
            if source_branch not in self._branches:
                raise ValueError(f"Source branch {source_branch} not found")
            if target not in self._branches:
                raise ValueError(f"Target branch {target} not found")
            
            source = self._branches[source_branch]
            target_obj = self._branches[target]
            
            # Get states
            source_snapshot = self._versions[source.head]
            target_snapshot = self._versions[target_obj.head]
            
            source_state = source_snapshot.decompress()
            target_state = target_snapshot.decompress()
            
            # Merge states
            if strategy == "geodesic":
                merged_state = self._geodesic_merge(source_state, target_state)
                merged_basin = self._geodesic_interpolate(
                    source_snapshot.basin_coords,
                    target_snapshot.basin_coords,
                    0.5  # Midpoint
                )
            elif strategy == "source":
                merged_state = source_state.copy()
                merged_basin = source_snapshot.basin_coords.copy()
            else:  # target
                merged_state = target_state.copy()
                merged_basin = target_snapshot.basin_coords.copy()
            
            # Average metrics
            merged_phi = (source_snapshot.phi + target_snapshot.phi) / 2
            merged_kappa = (source_snapshot.kappa + target_snapshot.kappa) / 2
            
            # Switch to target branch
            self._current_branch = target
            
            # Create merge commit with multiple parents
            version_id = self._generate_version_id(merged_state, time.time())
            
            merge_snapshot = StateSnapshot(
                version_id=version_id,
                basin_coords=merged_basin,
                state_data=merged_state,
                timestamp=time.time(),
                parent_version=target_obj.head,
                parent_versions=[target_obj.head, source.head],
                version_type=VersionType.MERGE,
                message=f"Merge {source_branch} into {target}",
                branch=target,
                phi=merged_phi,
                kappa=merged_kappa
            )
            
            self._versions[version_id] = merge_snapshot
            self._head = version_id
            target_obj.head = version_id
            
            self._working_state = merged_state
            self._working_basin = merged_basin
            
            self._save()
            
            return merge_snapshot
    
    def diff(
        self,
        version_a: str,
        version_b: Optional[str] = None
    ) -> StateDiff:
        """
        Compute difference between two versions.
        
        Like git diff - shows what changed between versions.
        Includes both field-level changes and geometric distance.
        
        Args:
            version_a: First version (or HEAD if version_b is None)
            version_b: Second version (or working state if None)
            
        Returns:
            StateDiff with changes
        """
        with self._lock:
            # Resolve versions
            if version_b is None:
                # Diff version_a against working state
                if version_a not in self._versions:
                    raise ValueError(f"Version {version_a} not found")
                snapshot_a = self._versions[version_a]
                state_a = snapshot_a.decompress()
                basin_a = snapshot_a.basin_coords
                
                state_b = self._working_state
                basin_b = self._working_basin
                target_id = "working"
            else:
                if version_a not in self._versions:
                    raise ValueError(f"Version {version_a} not found")
                if version_b not in self._versions:
                    raise ValueError(f"Version {version_b} not found")
                
                snapshot_a = self._versions[version_a]
                snapshot_b = self._versions[version_b]
                
                state_a = snapshot_a.decompress()
                state_b = snapshot_b.decompress()
                basin_a = snapshot_a.basin_coords
                basin_b = snapshot_b.basin_coords
                target_id = version_b
            
            # Compute field-level diff
            changes = self._compute_state_diff(state_a, state_b)
            
            # Compute geometric distance
            distance = fisher_rao_distance(basin_a, basin_b)
            
            # Count changes
            additions = sum(1 for _, t, _, _ in changes if t == DiffType.ADDED)
            removals = sum(1 for _, t, _, _ in changes if t == DiffType.REMOVED)
            modifications = sum(1 for _, t, _, _ in changes if t == DiffType.MODIFIED)
            
            return StateDiff(
                source_version=version_a,
                target_version=target_id,
                changes=changes,
                fisher_distance=distance,
                additions=additions,
                removals=removals,
                modifications=modifications
            )
    
    def rollback(
        self,
        steps: int = 1,
        to_version: Optional[str] = None
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Rollback to a previous version.
        
        Like git revert - goes back to a previous state.
        Creates a new commit recording the rollback.
        
        Args:
            steps: Number of versions to go back (default: 1)
            to_version: Specific version to rollback to
            
        Returns:
            Tuple of (state_dict, basin_coords)
        """
        with self._lock:
            if to_version:
                target = to_version
            else:
                # Walk back through history
                target = self._head
                for _ in range(steps):
                    if target and target in self._versions:
                        parent = self._versions[target].parent_version
                        if parent:
                            target = parent
                        else:
                            break
                    else:
                        break
            
            if not target or target not in self._versions:
                raise ValueError("Cannot rollback - no valid target version")
            
            # Get target state
            snapshot = self._versions[target]
            state = snapshot.decompress()
            basin = np.array(snapshot.basin_coords)
            
            # Create rollback commit
            rollback_version = self.commit(
                state=state,
                basin_coords=basin,
                message=f"Rollback to {target[:8]}",
                phi=snapshot.phi,
                kappa=snapshot.kappa,
                version_type=VersionType.ROLLBACK
            )
            
            return state, basin
    
    # =========================================================================
    # AUTO-CHECKPOINTING
    # =========================================================================
    
    def auto_checkpoint(
        self,
        state: Dict[str, Any],
        basin_coords: np.ndarray,
        phi: float = 0.0,
        kappa: float = 0.0,
        force: bool = False
    ) -> Optional[StateSnapshot]:
        """
        Automatic checkpoint if interval reached.
        
        Called at key decision points. Only creates checkpoint
        if enough operations have occurred.
        
        Args:
            state: Current state
            basin_coords: Current basin position
            phi: Current phi
            kappa: Current kappa
            force: Force checkpoint regardless of interval
            
        Returns:
            Created snapshot or None if not checkpointed
        """
        with self._lock:
            self._operation_count += 1
            
            if not self.auto_checkpoint and not force:
                return None
            
            if not force and self._operation_count < AUTO_CHECKPOINT_INTERVAL:
                return None
            
            # Reset counter
            self._operation_count = 0
            
            # Check if state actually changed
            if self._head and self._head in self._versions:
                prev_hash = self._versions[self._head].state_hash
                curr_hash = compute_state_hash(state)
                if prev_hash == curr_hash:
                    return None  # No change
            
            return self.commit(
                state=state,
                basin_coords=basin_coords,
                message=f"Auto-checkpoint at operation {self._operation_count}",
                phi=phi,
                kappa=kappa,
                version_type=VersionType.AUTO
            )
    
    def checkpoint_at_decision(
        self,
        state: Dict[str, Any],
        basin_coords: np.ndarray,
        decision: str,
        phi: float = 0.0,
        kappa: float = 0.0
    ) -> StateSnapshot:
        """
        Create checkpoint at a key decision point.
        
        Called before making significant decisions that might
        need to be rolled back.
        
        Args:
            state: Current state
            basin_coords: Current position
            decision: Description of the decision being made
            phi: Current phi
            kappa: Current kappa
            
        Returns:
            Created checkpoint
        """
        return self.commit(
            state=state,
            basin_coords=basin_coords,
            message=f"Decision point: {decision}",
            phi=phi,
            kappa=kappa,
            tags=["decision_point"],
            version_type=VersionType.AUTO
        )
    
    # =========================================================================
    # HISTORY & QUERIES
    # =========================================================================
    
    def get_history(
        self,
        branch: Optional[str] = None,
        limit: int = 50
    ) -> List[StateSnapshot]:
        """
        Get version history.
        
        Args:
            branch: Filter by branch (default: current)
            limit: Maximum versions to return
            
        Returns:
            List of snapshots from newest to oldest
        """
        with self._lock:
            target_branch = branch or self._current_branch
            
            # Start from branch head
            if target_branch in self._branches:
                head = self._branches[target_branch].head
            else:
                head = self._head
            
            history = []
            current = head
            
            while current and len(history) < limit:
                if current in self._versions:
                    snapshot = self._versions[current]
                    if branch is None or snapshot.branch == target_branch:
                        history.append(snapshot)
                    current = snapshot.parent_version
                else:
                    break
            
            return history
    
    def get_version(self, version_id: str) -> Optional[StateSnapshot]:
        """Get a specific version."""
        return self._versions.get(version_id)
    
    def get_branches(self) -> List[Branch]:
        """Get all branches."""
        return list(self._branches.values())
    
    def get_current_branch(self) -> str:
        """Get current branch name."""
        return self._current_branch
    
    def get_head(self) -> Optional[str]:
        """Get HEAD version ID."""
        return self._head
    
    def get_tags(self, tag: str) -> List[StateSnapshot]:
        """Get all versions with a specific tag."""
        return [v for v in self._versions.values() if tag in v.tags]
    
    def find_common_ancestor(
        self,
        version_a: str,
        version_b: str
    ) -> Optional[str]:
        """
        Find common ancestor of two versions.
        
        Useful for merge operations.
        """
        with self._lock:
            # Get ancestors of version_a
            ancestors_a = set()
            current = version_a
            while current and current in self._versions:
                ancestors_a.add(current)
                current = self._versions[current].parent_version
            
            # Walk version_b until we hit an ancestor of a
            current = version_b
            while current and current in self._versions:
                if current in ancestors_a:
                    return current
                current = self._versions[current].parent_version
            
            return None
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _generate_version_id(self, state: Dict, timestamp: float) -> str:
        """Generate unique version ID."""
        content = f"{self.agent_id}:{timestamp}:{json.dumps(state, sort_keys=True, default=str)[:100]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _compute_state_diff(
        self,
        state_a: Dict,
        state_b: Dict,
        path: str = ""
    ) -> List[Tuple[str, DiffType, Any, Any]]:
        """Compute field-level diff between two state dicts."""
        changes = []
        
        all_keys = set(state_a.keys()) | set(state_b.keys())
        
        for key in all_keys:
            key_path = f"{path}.{key}" if path else key
            
            if key not in state_a:
                changes.append((key_path, DiffType.ADDED, None, state_b[key]))
            elif key not in state_b:
                changes.append((key_path, DiffType.REMOVED, state_a[key], None))
            elif state_a[key] != state_b[key]:
                if isinstance(state_a[key], dict) and isinstance(state_b[key], dict):
                    # Recurse into nested dicts
                    changes.extend(self._compute_state_diff(state_a[key], state_b[key], key_path))
                else:
                    changes.append((key_path, DiffType.MODIFIED, state_a[key], state_b[key]))
        
        return changes
    
    def _geodesic_merge(
        self,
        state_a: Dict,
        state_b: Dict
    ) -> Dict:
        """Merge two state dicts using geodesic interpolation."""
        merged = {}
        
        all_keys = set(state_a.keys()) | set(state_b.keys())
        
        for key in all_keys:
            if key not in state_a:
                merged[key] = state_b[key]
            elif key not in state_b:
                merged[key] = state_a[key]
            elif isinstance(state_a[key], (int, float)) and isinstance(state_b[key], (int, float)):
                # Average numeric values
                merged[key] = (state_a[key] + state_b[key]) / 2
            elif isinstance(state_a[key], dict) and isinstance(state_b[key], dict):
                # Recurse
                merged[key] = self._geodesic_merge(state_a[key], state_b[key])
            elif isinstance(state_a[key], list) and isinstance(state_b[key], list):
                # Union of lists
                merged[key] = list(set(state_a[key]) | set(state_b[key]))
            else:
                # For incompatible types, prefer state_a
                merged[key] = state_a[key]
        
        return merged
    
    def _geodesic_interpolate(
        self,
        basin_a: np.ndarray,
        basin_b: np.ndarray,
        t: float
    ) -> np.ndarray:
        """Interpolate along geodesic between two basins."""
        a = np.array(basin_a, dtype=float)
        b = np.array(basin_b, dtype=float)
        
        # Normalize
        a = np.abs(a) + 1e-10
        a = a / a.sum()
        b = np.abs(b) + 1e-10
        b = b / b.sum()
        
        # SLERP in sqrt space
        sqrt_a = np.sqrt(a)
        sqrt_b = np.sqrt(b)
        
        cos_angle = np.clip(np.dot(sqrt_a, sqrt_b), -1, 1)
        angle = np.arccos(cos_angle)
        
        if angle < 1e-10:
            return a
        
        sin_angle = np.sin(angle)
        result = (np.sin((1 - t) * angle) * sqrt_a + np.sin(t * angle) * sqrt_b) / sin_angle
        
        # Square and normalize
        result = result ** 2
        result = result / result.sum()
        
        return result
    
    def _prune_history(self) -> None:
        """Remove old versions if over limit."""
        if len(self._versions) <= MAX_HISTORY_SIZE:
            return
        
        # Keep tagged versions and branch heads
        protected = set()
        for branch in self._branches.values():
            protected.add(branch.head)
            protected.add(branch.base_version)
        
        for version in self._versions.values():
            if version.tags:
                protected.add(version.version_id)
        
        # Sort by timestamp
        versions_by_time = sorted(
            self._versions.values(),
            key=lambda v: v.timestamp
        )
        
        # Remove oldest unprotected
        to_remove = []
        for version in versions_by_time:
            if len(self._versions) - len(to_remove) <= MAX_HISTORY_SIZE:
                break
            if version.version_id not in protected:
                to_remove.append(version.version_id)
        
        for vid in to_remove:
            del self._versions[vid]
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def _save(self) -> None:
        """Save version history to disk."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        data = {
            'agent_id': self.agent_id,
            'saved_at': time.time(),
            'head': self._head,
            'current_branch': self._current_branch,
            'branches': {name: b.to_dict() for name, b in self._branches.items()},
            'versions': {vid: v.to_dict() for vid, v in self._versions.items()}
        }
        
        filepath = self.storage_path / 'versions.json'
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load(self) -> None:
        """Load version history from disk."""
        filepath = self.storage_path / 'versions.json'
        
        if not filepath.exists():
            return
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self._head = data.get('head')
            self._current_branch = data.get('current_branch', 'main')
            
            self._branches = {
                name: Branch.from_dict(b)
                for name, b in data.get('branches', {}).items()
            }
            
            self._versions = {
                vid: StateSnapshot.from_dict(v)
                for vid, v in data.get('versions', {}).items()
            }
            
            # Restore working state from HEAD
            if self._head and self._head in self._versions:
                snapshot = self._versions[self._head]
                self._working_state = snapshot.decompress()
                self._working_basin = np.array(snapshot.basin_coords)
            
            print(f"[StateVersionControl] Loaded {len(self._versions)} versions, "
                  f"{len(self._branches)} branches")
        except Exception as e:
            print(f"[StateVersionControl] Failed to load: {e}")
    
    # =========================================================================
    # STATS
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get version control statistics."""
        with self._lock:
            total_size = sum(v.size_bytes() for v in self._versions.values())
            
            return {
                'agent_id': self.agent_id,
                'total_versions': len(self._versions),
                'total_branches': len(self._branches),
                'current_branch': self._current_branch,
                'head': self._head,
                'total_size_bytes': total_size,
                'auto_checkpoint': self.auto_checkpoint,
                'operation_count': self._operation_count
            }


# ============================================================================
# SINGLETON FACTORY
# ============================================================================

_version_controllers: Dict[str, StateVersionControl] = {}
_factory_lock = threading.Lock()


def get_version_control(agent_id: str) -> StateVersionControl:
    """Get or create version control for an agent."""
    global _version_controllers
    
    with _factory_lock:
        if agent_id not in _version_controllers:
            _version_controllers[agent_id] = StateVersionControl(agent_id)
        return _version_controllers[agent_id]


# ============================================================================
# DECORATOR FOR AUTO-CHECKPOINTING
# ============================================================================

def checkpoint_on_decision(decision_name: str):
    """
    Decorator to auto-checkpoint before a decision method.
    
    Usage:
        @checkpoint_on_decision("choose_strategy")
        def choose_strategy(self, options):
            ...
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Get version control if available
            vc = getattr(self, '_version_control', None)
            if vc and hasattr(self, 'get_state') and hasattr(self, 'get_basin'):
                vc.checkpoint_at_decision(
                    state=self.get_state(),
                    basin_coords=self.get_basin(),
                    decision=decision_name,
                    phi=getattr(self, 'phi', 0.0),
                    kappa=getattr(self, 'kappa', 0.0)
                )
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# MODULE INIT
# ============================================================================

print("[StateVersionControl] Module loaded - AgentGit state versioning ready")
