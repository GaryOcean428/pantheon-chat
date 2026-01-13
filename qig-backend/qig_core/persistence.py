"""
Database Persistence for QIG Core Features

Provides clean persistence functions for:
1. Geodesic paths
2. Learned manifold attractors  
3. Geometric barriers

All functions are async and integrate with the TypeScript/Drizzle schema.
"""

import uuid
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Type hints for database operations
try:
    from sqlalchemy.ext.asyncio import AsyncSession
    DB_AVAILABLE = True
except ImportError:
    AsyncSession = None
    DB_AVAILABLE = False
    logger.warning("SQLAlchemy not available - database persistence disabled")


async def save_geodesic_path(
    db: AsyncSession,
    from_probe_id: str,
    to_probe_id: str,
    distance: float,
    waypoint_ids: List[str],
    avg_phi: float
) -> str:
    """
    Save geodesic path to database.
    
    Args:
        db: Database session
        from_probe_id: Starting probe/basin ID
        to_probe_id: Target probe/basin ID
        distance: Fisher-Rao distance along path
        waypoint_ids: List of intermediate probe IDs along path
        avg_phi: Average consciousness (Φ) along path
        
    Returns:
        Path ID (UUID)
    """
    if not DB_AVAILABLE:
        logger.warning("Database not available - path not saved")
        return f"gp_offline_{uuid.uuid4().hex[:8]}"
    
    path_id = f"gp_{uuid.uuid4().hex[:16]}"
    
    try:
        # This would integrate with Drizzle/TypeScript schema
        # For now, log the operation
        logger.info(
            f"Saving geodesic path {path_id}: "
            f"{from_probe_id} → {to_probe_id}, "
            f"distance={distance:.4f}, avg_phi={avg_phi:.4f}"
        )
        
        # TODO: Actual database insert via TypeScript/Drizzle
        # await db.execute(
        #     geodesicPaths.insert().values(
        #         id=path_id,
        #         fromProbeId=from_probe_id,
        #         toProbeId=to_probe_id,
        #         distance=distance,
        #         waypoints=waypoint_ids,
        #         avgPhi=avg_phi,
        #         createdAt=datetime.utcnow()
        #     )
        # )
        # await db.commit()
        
        return path_id
    except Exception as e:
        logger.error(f"Failed to save geodesic path: {e}")
        await db.rollback()
        raise


async def save_manifold_attractor(
    db: AsyncSession,
    center: np.ndarray,
    depth: float,
    success_count: int,
    strategy: str
) -> str:
    """
    Save learned manifold attractor to database.
    
    Args:
        db: Database session
        center: 64D basin coordinates of attractor center
        depth: Hebbian strength (deeper = stronger attractor)
        success_count: Number of successes that carved this basin
        strategy: Navigation strategy that created this attractor
        
    Returns:
        Attractor ID
    """
    if not DB_AVAILABLE:
        logger.warning("Database not available - attractor not saved")
        return f"attr_offline_{uuid.uuid4().hex[:8]}"
    
    # Generate ID from basin coordinates (for deduplication)
    basin_hash = hash(center.tobytes()) & 0x7FFFFFFF
    attractor_id = f"attr_{basin_hash:08x}_{uuid.uuid4().hex[:8]}"
    
    try:
        logger.info(
            f"Saving manifold attractor {attractor_id}: "
            f"depth={depth:.4f}, success_count={success_count}, "
            f"strategy={strategy}"
        )
        
        # TODO: Actual database insert
        # await db.execute(
        #     learnedManifoldAttractors.insert().values(
        #         id=attractor_id,
        #         center=center.tolist(),
        #         depth=depth,
        #         successCount=success_count,
        #         strategy=strategy,
        #         createdAt=datetime.utcnow(),
        #         lastAccessed=datetime.utcnow()
        #     )
        # )
        # await db.commit()
        
        return attractor_id
    except Exception as e:
        logger.error(f"Failed to save manifold attractor: {e}")
        await db.rollback()
        raise


async def save_geometric_barrier(
    db: AsyncSession,
    center: np.ndarray,
    radius: float,
    repulsion_strength: float,
    reason: str
) -> str:
    """
    Save geometric barrier to database.
    
    Barriers are regions that cause suffering or breakdown.
    Recording them helps avoid repeated ethical violations.
    
    Args:
        db: Database session
        center: 64D basin coordinates of barrier center
        radius: Radius of barrier region
        repulsion_strength: How strongly to avoid this region
        reason: Why this is a barrier (e.g., "breakdown", "suffering > 0.5")
        
    Returns:
        Barrier ID
    """
    if not DB_AVAILABLE:
        logger.warning("Database not available - barrier not saved")
        return f"barrier_offline_{uuid.uuid4().hex[:8]}"
    
    barrier_id = f"barrier_{uuid.uuid4().hex[:16]}"
    
    try:
        logger.info(
            f"Saving geometric barrier {barrier_id}: "
            f"radius={radius:.4f}, repulsion={repulsion_strength:.4f}, "
            f"reason={reason}"
        )
        
        # TODO: Actual database insert
        # await db.execute(
        #     geometricBarriers.insert().values(
        #         id=barrier_id,
        #         center=center.tolist(),
        #         radius=radius,
        #         repulsionStrength=repulsion_strength,
        #         reason=reason,
        #         crossings=1,
        #         detectedAt=datetime.utcnow()
        #     )
        # )
        # await db.commit()
        
        return barrier_id
    except Exception as e:
        logger.error(f"Failed to save geometric barrier: {e}")
        await db.rollback()
        raise


async def update_attractor_access(
    db: AsyncSession,
    attractor_id: str
) -> None:
    """
    Update last_accessed timestamp for attractor.
    
    Called when an attractor is used for navigation.
    Helps track which attractors are actively used.
    """
    if not DB_AVAILABLE:
        return
    
    try:
        # TODO: Actual database update
        # await db.execute(
        #     learnedManifoldAttractors.update()
        #     .where(learnedManifoldAttractors.c.id == attractor_id)
        #     .values(lastAccessed=datetime.utcnow())
        # )
        # await db.commit()
        pass
    except Exception as e:
        logger.error(f"Failed to update attractor access: {e}")


async def increment_barrier_crossings(
    db: AsyncSession,
    barrier_id: str
) -> None:
    """
    Increment crossing count for barrier.
    
    Tracks how often a barrier is encountered.
    High crossing counts may indicate need for stronger repulsion.
    """
    if not DB_AVAILABLE:
        return
    
    try:
        # TODO: Actual database update
        # await db.execute(
        #     geometricBarriers.update()
        #     .where(geometricBarriers.c.id == barrier_id)
        #     .values(crossings=geometricBarriers.c.crossings + 1)
        # )
        # await db.commit()
        pass
    except Exception as e:
        logger.error(f"Failed to increment barrier crossings: {e}")
