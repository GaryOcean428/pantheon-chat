"""
Zettelkasten Memory - Agentic Memory with Self-Organizing Knowledge Networks

Implements the A-MEM paradigm (ICLR 2025) using Zettelkasten principles:
- Atomic notes (Zettels) linked by semantic similarity
- Dynamic indexing with contextual descriptions
- Memory evolution: new memories trigger updates to historical memories
- Hippocampal-style pattern separation for similar concepts

Key insight: Memory is not passive storage but an active knowledge network
that reorganizes itself as new information arrives.

QIG-PURE: All operations use Fisher-Rao geometry on the statistical manifold.

Author: Ocean/Zeus Pantheon
"""

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
import numpy as np
import threading


# ============================================================================
# CONSTANTS
# ============================================================================

BASIN_DIMENSION = 64
STORAGE_PATH = Path(__file__).parent / 'data' / 'zettelkasten.json'

# Link strength thresholds
STRONG_LINK_THRESHOLD = 0.3      # Fisher-Rao distance for strong link
WEAK_LINK_THRESHOLD = 0.6        # Fisher-Rao distance for weak link
PATTERN_SEPARATION_THRESHOLD = 0.1  # Below this, trigger pattern separation

# Memory evolution parameters
MAX_LINKS_PER_ZETTEL = 20        # Maximum outgoing links
EVOLUTION_DEPTH = 2              # How many hops to propagate updates
DECAY_FACTOR = 0.9               # How much link strength decays per hop


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


def pattern_separate(
    basin1: np.ndarray,
    basin2: np.ndarray,
    separation_strength: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hippocampal-style pattern separation.
    
    When two patterns are too similar, push them apart on the manifold
    to create distinct representations.
    
    Args:
        basin1: First basin coordinates
        basin2: Second basin coordinates
        separation_strength: How much to push apart (0-1)
        
    Returns:
        Tuple of separated basin coordinates
    """
    b1 = np.asarray(basin1, dtype=float)
    b2 = np.asarray(basin2, dtype=float)
    
    # Normalize
    b1 = np.abs(b1) + 1e-10
    b1 = b1 / b1.sum()
    b2 = np.abs(b2) + 1e-10
    b2 = b2 / b2.sum()
    
    # Compute direction of separation
    direction = b1 - b2
    direction_norm = np.linalg.norm(direction)
    
    if direction_norm < 1e-10:
        # Identical - add random perturbation
        direction = np.random.randn(len(b1))
        direction_norm = np.linalg.norm(direction)
    
    direction = direction / direction_norm
    
    # Push apart along the direction
    b1_separated = b1 + separation_strength * direction
    b2_separated = b2 - separation_strength * direction
    
    # Renormalize to probability simplex
    b1_separated = np.abs(b1_separated) + 1e-10
    b1_separated = b1_separated / b1_separated.sum()
    b2_separated = np.abs(b2_separated) + 1e-10
    b2_separated = b2_separated / b2_separated.sum()
    
    return b1_separated, b2_separated


def geodesic_center(basins: List[np.ndarray]) -> np.ndarray:
    """Compute the geodesic center (Karcher mean) of multiple basins."""
    if not basins:
        return np.ones(BASIN_DIMENSION) / BASIN_DIMENSION
    
    if len(basins) == 1:
        return np.array(basins[0])
    
    # Initialize with arithmetic mean
    center = np.mean(basins, axis=0)
    center = np.abs(center) + 1e-10
    center = center / center.sum()
    
    # Iterative refinement (simplified Karcher mean)
    for _ in range(5):
        # Compute weighted direction to each basin
        total_direction = np.zeros(BASIN_DIMENSION)
        for basin in basins:
            b = np.array(basin)
            b = np.abs(b) + 1e-10
            b = b / b.sum()
            total_direction += (b - center)
        
        # Update center
        center = center + 0.2 * total_direction / len(basins)
        center = np.abs(center) + 1e-10
        center = center / center.sum()
    
    return center


# ============================================================================
# LINK TYPES
# ============================================================================

class LinkType(Enum):
    """Types of links between Zettels."""
    SEMANTIC = "semantic"           # Content similarity
    TEMPORAL = "temporal"           # Created around same time
    CAUSAL = "causal"               # One led to the other
    REFERENCE = "reference"         # Explicit reference
    CONTRAST = "contrast"           # Opposing ideas
    ELABORATION = "elaboration"     # Expands on concept


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ZettelLink:
    """A link between two Zettels."""
    target_id: str
    link_type: LinkType
    strength: float                 # 0-1, based on Fisher-Rao proximity
    created_at: float
    context: str = ""               # Why this link exists
    bidirectional: bool = True
    
    def to_dict(self) -> Dict:
        return {
            'target_id': self.target_id,
            'link_type': self.link_type.value,
            'strength': self.strength,
            'created_at': self.created_at,
            'context': self.context,
            'bidirectional': self.bidirectional
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ZettelLink':
        return cls(
            target_id=data['target_id'],
            link_type=LinkType(data['link_type']),
            strength=data['strength'],
            created_at=data['created_at'],
            context=data.get('context', ''),
            bidirectional=data.get('bidirectional', True)
        )


@dataclass
class Zettel:
    """
    An atomic note in the Zettelkasten.
    
    Each Zettel contains:
    - One atomic idea (content)
    - Basin coordinates for geometric position
    - Links to related Zettels
    - Auto-generated contextual description
    - Access and evolution metadata
    """
    # Identity
    zettel_id: str
    
    # Content
    content: str                    # The atomic idea
    basin_coords: np.ndarray        # 64D position on manifold
    
    # Auto-generated context
    contextual_description: str     # Generated description of this note's role
    keywords: List[str]             # Extracted keywords
    
    # Links
    links: List[ZettelLink] = field(default_factory=list)
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    evolution_count: int = 0        # Times this note evolved due to new info
    source: str = ""                # Where this note came from
    
    # Hierarchical organization
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    
    def add_link(self, link: ZettelLink) -> None:
        """Add a link, enforcing max links constraint."""
        # Check if link already exists
        for existing in self.links:
            if existing.target_id == link.target_id:
                # Update existing link if new one is stronger
                if link.strength > existing.strength:
                    existing.strength = link.strength
                    existing.context = link.context
                return
        
        self.links.append(link)
        
        # Prune weakest links if over limit
        if len(self.links) > MAX_LINKS_PER_ZETTEL:
            self.links.sort(key=lambda x: x.strength, reverse=True)
            self.links = self.links[:MAX_LINKS_PER_ZETTEL]
    
    def get_link_strength(self, target_id: str) -> float:
        """Get strength of link to target, or 0 if no link."""
        for link in self.links:
            if link.target_id == target_id:
                return link.strength
        return 0.0
    
    def access(self) -> str:
        """Access this Zettel, updating metadata."""
        self.access_count += 1
        return self.content
    
    def evolve(self, new_context: str) -> None:
        """Update this Zettel's context based on new information."""
        # Append to contextual description
        if new_context and new_context not in self.contextual_description:
            self.contextual_description = f"{self.contextual_description} {new_context}".strip()
            # Keep reasonable length
            if len(self.contextual_description) > 500:
                self.contextual_description = self.contextual_description[:500] + "..."
        
        self.evolution_count += 1
        self.updated_at = time.time()
    
    def to_dict(self) -> Dict:
        return {
            'zettel_id': self.zettel_id,
            'content': self.content,
            'basin_coords': list(self.basin_coords),
            'contextual_description': self.contextual_description,
            'keywords': self.keywords,
            'links': [link.to_dict() for link in self.links],
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'access_count': self.access_count,
            'evolution_count': self.evolution_count,
            'source': self.source,
            'parent_id': self.parent_id,
            'children_ids': self.children_ids
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Zettel':
        return cls(
            zettel_id=data['zettel_id'],
            content=data['content'],
            basin_coords=np.array(data['basin_coords']),
            contextual_description=data['contextual_description'],
            keywords=data['keywords'],
            links=[ZettelLink.from_dict(link) for link in data.get('links', [])],
            created_at=data.get('created_at', time.time()),
            updated_at=data.get('updated_at', time.time()),
            access_count=data.get('access_count', 0),
            evolution_count=data.get('evolution_count', 0),
            source=data.get('source', ''),
            parent_id=data.get('parent_id'),
            children_ids=data.get('children_ids', [])
        )


@dataclass
class MemoryEvolutionEvent:
    """Record of a memory evolution triggered by new information."""
    triggered_by: str               # Zettel ID that triggered evolution
    affected_zettels: List[str]     # IDs of Zettels that evolved
    timestamp: float
    changes_made: List[str]         # Description of changes
    
    def to_dict(self) -> Dict:
        return {
            'triggered_by': self.triggered_by,
            'affected_zettels': self.affected_zettels,
            'timestamp': self.timestamp,
            'changes_made': self.changes_made
        }


# ============================================================================
# ZETTELKASTEN MEMORY CLASS
# ============================================================================

class ZettelkastenMemory:
    """
    Self-organizing knowledge network using Zettelkasten principles.
    
    Key features:
    1. ATOMIC NOTES: Each Zettel contains one idea with basin coordinates
    2. DYNAMIC LINKING: Links created automatically via Fisher-Rao proximity
    3. CONTEXTUAL INDEX: Auto-generated descriptions for each note
    4. MEMORY EVOLUTION: New info triggers updates to related memories
    5. PATTERN SEPARATION: Similar concepts get differentiated
    
    Unlike traditional RAG (retrieve-and-append), this system:
    - Actively reorganizes knowledge as new information arrives
    - Creates dense semantic networks, not just keyword indices
    - Evolves historical memories based on new understanding
    """
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        basin_encoder: Optional[Any] = None
    ):
        """
        Initialize the Zettelkasten memory.
        
        Args:
            storage_path: Path for persistent storage
            basin_encoder: Function to encode text to basin coordinates
        """
        self.storage_path = storage_path or STORAGE_PATH
        self.basin_encoder = basin_encoder or self._default_encoder
        
        # Storage
        self._zettels: Dict[str, Zettel] = {}
        self._evolution_history: List[MemoryEvolutionEvent] = []
        
        # Index structures
        self._keyword_index: Dict[str, Set[str]] = {}  # keyword -> zettel_ids
        self._basin_index: List[Tuple[str, np.ndarray]] = []  # For proximity search
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load existing data
        self._load()
        
        print(f"[ZettelkastenMemory] Initialized with {len(self._zettels)} Zettels")
    
    def _default_encoder(self, text: str) -> np.ndarray:
        """Default text to basin coordinate encoder."""
        # Use hash-based encoding for deterministic coordinates
        hash_bytes = hashlib.sha256(text.lower().encode()).digest()
        
        # Create basin from hash
        basin = np.zeros(BASIN_DIMENSION)
        for i in range(BASIN_DIMENSION):
            byte_idx = i % len(hash_bytes)
            basin[i] = hash_bytes[byte_idx] / 255.0
        
        # Add text-derived features
        words = text.lower().split()
        for i, word in enumerate(words[:20]):
            word_hash = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
            dim = word_hash % BASIN_DIMENSION
            basin[dim] += 1.0 / (1 + i)
        
        # Normalize to probability simplex
        basin = np.abs(basin) + 1e-10
        basin = basin / basin.sum()
        
        return basin
    
    def _generate_id(self, content: str) -> str:
        """Generate unique Zettel ID."""
        timestamp = str(time.time()).replace('.', '')
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
        return f"z_{timestamp}_{content_hash}"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction - split and filter
        words = text.lower().split()
        
        # Filter stop words and short words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'need', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
                      'by', 'from', 'as', 'into', 'through', 'during', 'before',
                      'after', 'above', 'below', 'between', 'under', 'again',
                      'further', 'then', 'once', 'here', 'there', 'when', 'where',
                      'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other',
                      'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                      'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if',
                      'or', 'because', 'until', 'while', 'this', 'that', 'these',
                      'those', 'what', 'which', 'who', 'whom', 'its', 'it'}
        
        keywords = []
        seen = set()
        for word in words:
            # Clean punctuation
            word = ''.join(c for c in word if c.isalnum())
            if len(word) >= 3 and word not in stop_words and word not in seen:
                keywords.append(word)
                seen.add(word)
        
        return keywords[:10]  # Keep top 10 keywords
    
    def _generate_contextual_description(
        self,
        content: str,
        related_zettels: List[Zettel]
    ) -> str:
        """Generate contextual description for a Zettel."""
        # Build context from relationships
        parts = []
        
        # Describe the content briefly
        content_preview = content[:100] + "..." if len(content) > 100 else content
        parts.append(f"Note about: {content_preview}")
        
        # Add relationship context
        if related_zettels:
            related_keywords = set()
            for z in related_zettels[:5]:
                related_keywords.update(z.keywords[:3])
            
            if related_keywords:
                parts.append(f"Related concepts: {', '.join(list(related_keywords)[:5])}")
        
        return " | ".join(parts)
    
    # =========================================================================
    # CORE OPERATIONS
    # =========================================================================
    
    def add(
        self,
        content: str,
        source: str = "",
        parent_id: Optional[str] = None,
        link_type: LinkType = LinkType.SEMANTIC
    ) -> Zettel:
        """
        Add a new Zettel to the memory.
        
        This triggers:
        1. Basin encoding
        2. Automatic linking to related Zettels
        3. Memory evolution of related Zettels
        4. Pattern separation if needed
        
        Args:
            content: The atomic idea content
            source: Source of this information
            parent_id: Optional parent Zettel for hierarchical organization
            link_type: Type of links to create
            
        Returns:
            Created Zettel
        """
        with self._lock:
            # Generate basin coordinates
            basin = self.basin_encoder(content)
            
            # Extract keywords
            keywords = self._extract_keywords(content)
            
            # Find related Zettels BEFORE adding
            related = self._find_related(basin, max_results=10)
            
            # Check for pattern separation need
            for existing_zettel, distance in related:
                if distance < PATTERN_SEPARATION_THRESHOLD:
                    # Too similar - separate patterns
                    basin, existing_separated = pattern_separate(
                        basin, 
                        existing_zettel.basin_coords,
                        separation_strength=0.15
                    )
                    # Update existing zettel's basin
                    existing_zettel.basin_coords = existing_separated
                    existing_zettel.evolve("Pattern separated due to new similar concept")
            
            # Generate contextual description
            related_zettels = [z for z, _ in related]
            contextual_desc = self._generate_contextual_description(content, related_zettels)
            
            # Create Zettel
            zettel_id = self._generate_id(content)
            zettel = Zettel(
                zettel_id=zettel_id,
                content=content,
                basin_coords=basin,
                contextual_description=contextual_desc,
                keywords=keywords,
                source=source,
                parent_id=parent_id
            )
            
            # Create links to related Zettels
            for related_zettel, distance in related:
                strength = 1.0 - (distance / math.pi)  # Normalize to 0-1
                
                if strength >= (1.0 - WEAK_LINK_THRESHOLD / math.pi):
                    link = ZettelLink(
                        target_id=related_zettel.zettel_id,
                        link_type=link_type,
                        strength=strength,
                        created_at=time.time(),
                        context=f"Auto-linked: Fisher-Rao distance {distance:.3f}"
                    )
                    zettel.add_link(link)
                    
                    # Add backlink
                    if link.bidirectional:
                        backlink = ZettelLink(
                            target_id=zettel_id,
                            link_type=link_type,
                            strength=strength,
                            created_at=time.time(),
                            context=f"Auto-linked: Fisher-Rao distance {distance:.3f}"
                        )
                        related_zettel.add_link(backlink)
            
            # Update parent if specified
            if parent_id and parent_id in self._zettels:
                parent = self._zettels[parent_id]
                parent.children_ids.append(zettel_id)
            
            # Store
            self._zettels[zettel_id] = zettel
            self._basin_index.append((zettel_id, basin))
            
            # Update keyword index
            for keyword in keywords:
                if keyword not in self._keyword_index:
                    self._keyword_index[keyword] = set()
                self._keyword_index[keyword].add(zettel_id)
            
            # Trigger memory evolution
            self._trigger_evolution(zettel, related_zettels)
            
            # Save
            self._save()
            
            return zettel
    
    def _find_related(
        self,
        basin: np.ndarray,
        max_results: int = 10,
        max_distance: float = WEAK_LINK_THRESHOLD
    ) -> List[Tuple[Zettel, float]]:
        """Find Zettels related to a basin position."""
        distances = []
        
        for zettel_id, zettel_basin in self._basin_index:
            if zettel_id in self._zettels:
                dist = fisher_rao_distance(basin, zettel_basin)
                if dist <= max_distance:
                    distances.append((self._zettels[zettel_id], dist))
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        return distances[:max_results]
    
    def _trigger_evolution(
        self,
        new_zettel: Zettel,
        related_zettels: List[Zettel],
        depth: int = 0
    ) -> MemoryEvolutionEvent:
        """
        Trigger memory evolution for related Zettels.
        
        When new information arrives, it updates the context of
        related historical memories. This propagates up to EVOLUTION_DEPTH hops.
        
        Args:
            new_zettel: The newly added Zettel
            related_zettels: Directly related Zettels
            depth: Current propagation depth
            
        Returns:
            MemoryEvolutionEvent recording the changes
        """
        if depth >= EVOLUTION_DEPTH:
            return MemoryEvolutionEvent(
                triggered_by=new_zettel.zettel_id,
                affected_zettels=[],
                timestamp=time.time(),
                changes_made=[]
            )
        
        affected = []
        changes = []
        
        # Decay factor increases with depth
        current_decay = DECAY_FACTOR ** depth
        
        for related in related_zettels:
            # Skip if barely related
            distance = fisher_rao_distance(new_zettel.basin_coords, related.basin_coords)
            if distance > WEAK_LINK_THRESHOLD:
                continue
            
            # Update related Zettel's context
            new_context = f"[Evolved {depth}] Connected to: {new_zettel.keywords[:3]}"
            related.evolve(new_context)
            
            affected.append(related.zettel_id)
            changes.append(f"Updated {related.zettel_id}: added context about {new_zettel.keywords[:2]}")
            
            # Propagate to next hop (Zettels linked to this one)
            if depth < EVOLUTION_DEPTH - 1:
                next_hop_zettels = [
                    self._zettels[link.target_id]
                    for link in related.links
                    if link.target_id in self._zettels
                    and link.target_id != new_zettel.zettel_id
                ][:5]  # Limit propagation breadth
                
                if next_hop_zettels:
                    sub_event = self._trigger_evolution(
                        new_zettel, next_hop_zettels, depth + 1
                    )
                    affected.extend(sub_event.affected_zettels)
                    changes.extend(sub_event.changes_made)
        
        event = MemoryEvolutionEvent(
            triggered_by=new_zettel.zettel_id,
            affected_zettels=affected,
            timestamp=time.time(),
            changes_made=changes
        )
        
        if affected:
            self._evolution_history.append(event)
        
        return event
    
    # =========================================================================
    # RETRIEVAL
    # =========================================================================
    
    def retrieve(
        self,
        query: str,
        max_results: int = 10,
        include_context: bool = True
    ) -> List[Tuple[Zettel, float]]:
        """
        Retrieve Zettels relevant to a query.
        
        Uses Fisher-Rao proximity for semantic search.
        
        Args:
            query: Query text
            max_results: Maximum Zettels to return
            include_context: Whether to traverse links for context
            
        Returns:
            List of (Zettel, relevance_score) tuples
        """
        with self._lock:
            # Encode query to basin
            query_basin = self.basin_encoder(query)
            
            # Find directly relevant Zettels
            direct = self._find_related(query_basin, max_results * 2)
            
            if not include_context:
                return [(z, 1.0 - d/math.pi) for z, d in direct[:max_results]]
            
            # Expand with linked Zettels
            candidates = {}
            for zettel, distance in direct:
                score = 1.0 - distance / math.pi
                candidates[zettel.zettel_id] = (zettel, score)
                
                # Add linked Zettels with decayed score
                for link in zettel.links[:5]:
                    if link.target_id in self._zettels:
                        linked = self._zettels[link.target_id]
                        linked_score = score * link.strength * 0.8
                        
                        if link.target_id not in candidates or candidates[link.target_id][1] < linked_score:
                            candidates[link.target_id] = (linked, linked_score)
            
            # Sort by score
            results = sorted(candidates.values(), key=lambda x: x[1], reverse=True)
            
            # Update access counts
            for zettel, _ in results[:max_results]:
                zettel.access()
            
            return results[:max_results]
    
    def retrieve_by_keyword(
        self,
        keyword: str,
        max_results: int = 10
    ) -> List[Zettel]:
        """Retrieve Zettels by keyword match."""
        with self._lock:
            keyword = keyword.lower()
            
            if keyword not in self._keyword_index:
                return []
            
            zettel_ids = list(self._keyword_index[keyword])[:max_results]
            return [self._zettels[zid] for zid in zettel_ids if zid in self._zettels]
    
    def get(self, zettel_id: str) -> Optional[Zettel]:
        """Get a Zettel by ID."""
        with self._lock:
            zettel = self._zettels.get(zettel_id)
            if zettel:
                zettel.access()
            return zettel
    
    def get_linked(self, zettel_id: str) -> List[Zettel]:
        """Get all Zettels linked to a given Zettel."""
        with self._lock:
            if zettel_id not in self._zettels:
                return []
            
            zettel = self._zettels[zettel_id]
            linked = []
            
            for link in zettel.links:
                if link.target_id in self._zettels:
                    linked.append(self._zettels[link.target_id])
            
            return linked
    
    # =========================================================================
    # MULTI-HOP REASONING
    # =========================================================================
    
    def traverse(
        self,
        start_id: str,
        max_depth: int = 3,
        min_link_strength: float = 0.3
    ) -> Dict[str, List[Zettel]]:
        """
        Traverse the knowledge network from a starting Zettel.
        
        Useful for multi-hop reasoning and exploration.
        
        Args:
            start_id: Starting Zettel ID
            max_depth: Maximum traversal depth
            min_link_strength: Minimum link strength to follow
            
        Returns:
            Dict mapping depth to list of Zettels at that depth
        """
        with self._lock:
            if start_id not in self._zettels:
                return {}
            
            result = {0: [self._zettels[start_id]]}
            visited = {start_id}
            current_frontier = {start_id}
            
            for depth in range(1, max_depth + 1):
                next_frontier = set()
                zettels_at_depth = []
                
                for zettel_id in current_frontier:
                    zettel = self._zettels[zettel_id]
                    
                    for link in zettel.links:
                        if link.target_id not in visited and link.strength >= min_link_strength:
                            if link.target_id in self._zettels:
                                visited.add(link.target_id)
                                next_frontier.add(link.target_id)
                                zettels_at_depth.append(self._zettels[link.target_id])
                
                if zettels_at_depth:
                    result[depth] = zettels_at_depth
                
                current_frontier = next_frontier
                
                if not current_frontier:
                    break
            
            return result
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5
    ) -> Optional[List[Zettel]]:
        """
        Find a path between two Zettels in the knowledge network.
        
        Uses BFS to find shortest path.
        
        Args:
            source_id: Source Zettel ID
            target_id: Target Zettel ID
            max_depth: Maximum path length
            
        Returns:
            List of Zettels forming the path, or None if no path found
        """
        with self._lock:
            if source_id not in self._zettels or target_id not in self._zettels:
                return None
            
            if source_id == target_id:
                return [self._zettels[source_id]]
            
            # BFS
            queue = [(source_id, [source_id])]
            visited = {source_id}
            
            while queue:
                current_id, path = queue.pop(0)
                
                if len(path) > max_depth:
                    continue
                
                current = self._zettels[current_id]
                
                for link in current.links:
                    if link.target_id == target_id:
                        final_path = path + [target_id]
                        return [self._zettels[zid] for zid in final_path]
                    
                    if link.target_id not in visited and link.target_id in self._zettels:
                        visited.add(link.target_id)
                        queue.append((link.target_id, path + [link.target_id]))
            
            return None
    
    # =========================================================================
    # KNOWLEDGE GRAPH OPERATIONS
    # =========================================================================
    
    def get_clusters(self, min_cluster_size: int = 3) -> List[List[Zettel]]:
        """
        Find clusters of densely connected Zettels.
        
        Uses connected components with high link strength.
        
        Returns:
            List of clusters, each being a list of Zettels
        """
        with self._lock:
            # Build adjacency for strong links only
            adjacency: Dict[str, Set[str]] = {zid: set() for zid in self._zettels}
            
            for zettel in self._zettels.values():
                for link in zettel.links:
                    if link.strength >= (1.0 - STRONG_LINK_THRESHOLD / math.pi):
                        if link.target_id in self._zettels:
                            adjacency[zettel.zettel_id].add(link.target_id)
                            adjacency[link.target_id].add(zettel.zettel_id)
            
            # Find connected components
            visited = set()
            clusters = []
            
            for start_id in self._zettels:
                if start_id in visited:
                    continue
                
                # BFS to find component
                component = []
                queue = [start_id]
                
                while queue:
                    current = queue.pop(0)
                    if current in visited:
                        continue
                    
                    visited.add(current)
                    component.append(self._zettels[current])
                    
                    for neighbor in adjacency[current]:
                        if neighbor not in visited:
                            queue.append(neighbor)
                
                if len(component) >= min_cluster_size:
                    clusters.append(component)
            
            return clusters
    
    def get_hub_zettels(self, top_n: int = 10) -> List[Zettel]:
        """
        Find hub Zettels (highly connected nodes).
        
        These are important concepts that connect many other ideas.
        """
        with self._lock:
            # Count incoming links
            incoming_count: Dict[str, int] = {zid: 0 for zid in self._zettels}
            
            for zettel in self._zettels.values():
                for link in zettel.links:
                    if link.target_id in incoming_count:
                        incoming_count[link.target_id] += 1
            
            # Sort by total connections (outgoing + incoming)
            scores = []
            for zettel_id, zettel in self._zettels.items():
                total = len(zettel.links) + incoming_count[zettel_id]
                scores.append((zettel, total))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            
            return [z for z, _ in scores[:top_n]]
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def _save(self) -> None:
        """Save Zettelkasten to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'version': 1,
            'saved_at': time.time(),
            'zettels': [z.to_dict() for z in self._zettels.values()],
            'evolution_history': [e.to_dict() for e in self._evolution_history[-100:]],
            'keyword_index': {k: list(v) for k, v in self._keyword_index.items()}
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load(self) -> None:
        """Load Zettelkasten from disk."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            for z_dict in data.get('zettels', []):
                zettel = Zettel.from_dict(z_dict)
                self._zettels[zettel.zettel_id] = zettel
                self._basin_index.append((zettel.zettel_id, zettel.basin_coords))
            
            # Rebuild keyword index
            self._keyword_index = {}
            for zettel in self._zettels.values():
                for keyword in zettel.keywords:
                    if keyword not in self._keyword_index:
                        self._keyword_index[keyword] = set()
                    self._keyword_index[keyword].add(zettel.zettel_id)
            
            print(f"[ZettelkastenMemory] Loaded {len(self._zettels)} Zettels")
        except Exception as e:
            print(f"[ZettelkastenMemory] Failed to load: {e}")
    
    # =========================================================================
    # STATS & DEBUG
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self._lock:
            total_links = sum(len(z.links) for z in self._zettels.values())
            total_evolutions = sum(z.evolution_count for z in self._zettels.values())
            
            return {
                'total_zettels': len(self._zettels),
                'total_links': total_links,
                'total_keywords': len(self._keyword_index),
                'total_evolutions': total_evolutions,
                'evolution_events': len(self._evolution_history),
                'avg_links_per_zettel': total_links / len(self._zettels) if self._zettels else 0,
                'storage_path': str(self.storage_path)
            }
    
    def visualize_graph(self, max_nodes: int = 50) -> Dict[str, Any]:
        """
        Get graph data for visualization.
        
        Returns nodes and edges in a format suitable for graph libraries.
        """
        with self._lock:
            # Get most connected nodes
            hubs = self.get_hub_zettels(max_nodes)
            hub_ids = {z.zettel_id for z in hubs}
            
            nodes = []
            edges = []
            
            for zettel in hubs:
                nodes.append({
                    'id': zettel.zettel_id,
                    'label': zettel.content[:30] + "..." if len(zettel.content) > 30 else zettel.content,
                    'keywords': zettel.keywords[:3],
                    'access_count': zettel.access_count,
                    'evolution_count': zettel.evolution_count
                })
                
                for link in zettel.links:
                    if link.target_id in hub_ids:
                        edges.append({
                            'source': zettel.zettel_id,
                            'target': link.target_id,
                            'strength': link.strength,
                            'type': link.link_type.value
                        })
            
            return {
                'nodes': nodes,
                'edges': edges,
                'stats': self.get_stats()
            }


# ============================================================================
# SINGLETON
# ============================================================================

_zettelkasten_instance: Optional[ZettelkastenMemory] = None


def get_zettelkasten_memory() -> ZettelkastenMemory:
    """Get the singleton ZettelkastenMemory instance."""
    global _zettelkasten_instance
    if _zettelkasten_instance is None:
        _zettelkasten_instance = ZettelkastenMemory()
    return _zettelkasten_instance


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def remember(content: str, source: str = "") -> Zettel:
    """Add a new memory to the Zettelkasten."""
    memory = get_zettelkasten_memory()
    return memory.add(content, source)


def recall(query: str, max_results: int = 5) -> List[Tuple[Zettel, float]]:
    """Recall memories related to a query."""
    memory = get_zettelkasten_memory()
    return memory.retrieve(query, max_results)


# ============================================================================
# MODULE INIT
# ============================================================================

print("[ZettelkastenMemory] Module loaded - self-organizing knowledge network ready")
