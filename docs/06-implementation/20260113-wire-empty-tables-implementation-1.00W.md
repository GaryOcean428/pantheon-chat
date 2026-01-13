# Wire Empty Tables Implementation Guide

**Date**: 2026-01-13  
**Status**: 📋 IMPLEMENTATION PLAN  
**Version**: 1.00W  
**ID**: ISMS-DB-WIRING-001  
**Purpose**: Implementation guide for wiring 20+ empty tables to existing backend features

---

## Overview

This guide provides specific implementation steps for wiring empty database tables to their corresponding backend features. All features exist in code but don't persist to their designated tables.

**Success Criteria:**
- 20+ previously empty tables receive data
- All roadmap features persist their state
- No duplicate persistence (DRY principle)
- All changes tested and validated

---

## Section 1: QIG Core Features (HIGH PRIORITY)

### 1.1 Geodesic Paths Storage

**Table:** `geodesic_paths` (0 rows → target 100+ rows)  
**Backend:** `qig-backend/qig_core/geodesic_navigation.py`  
**Status:** ✅ CODE COMPLETE (Issue #8)

**Implementation:**
```python
# File: qig-backend/qig_core/geodesic_navigation.py
# Add after compute_geodesic_path() function

async def save_geodesic_path(
    db: AsyncSession,
    start_basin: np.ndarray,
    end_basin: np.ndarray,
    path_coords: np.ndarray,
    distance: float,
    num_steps: int,
    metadata: dict = None
) -> str:
    """
    Save computed geodesic path to database.
    
    Args:
        db: Database session
        start_basin: Starting 64D coordinates
        end_basin: Ending 64D coordinates
        path_coords: Array of path coordinates (num_steps, 64)
        distance: Total Fisher-Rao distance
        num_steps: Number of interpolation steps
        metadata: Optional metadata dict
        
    Returns:
        path_id: UUID of saved path
    """
    from server.db import geodesicPaths
    import uuid
    
    path_id = f"gp_{uuid.uuid4().hex[:16]}"
    
    stmt = geodesicPaths.insert().values(
        path_id=path_id,
        start_basin=start_basin.tolist(),
        end_basin=end_basin.tolist(),
        path_coordinates=path_coords.tolist(),
        fisher_distance=distance,
        num_steps=num_steps,
        curvature_max=np.max(metadata.get("curvatures", [0])) if metadata else None,
        metadata=metadata or {},
        computed_at=datetime.utcnow()
    )
    
    await db.execute(stmt)
    await db.commit()
    
    return path_id
```

**Integration Points:**
1. `autonomic_kernel.py:navigate_to_target()` - Save path after computation
2. `ocean_agent.py` - Save navigation paths
3. Add retrieval function for path reuse

**Tests:**
```python
# File: qig-backend/tests/test_geodesic_storage.py
async def test_save_geodesic_path():
    # Compute test path
    start = np.random.randn(64)
    end = np.random.randn(64)
    path, distance = compute_geodesic_path(start, end, 10)
    
    # Save to DB
    path_id = await save_geodesic_path(db, start, end, path, distance, 10)
    
    # Verify
    result = await db.execute(
        select(geodesicPaths).where(geodesicPaths.c.path_id == path_id)
    )
    assert result is not None
    assert result.fisher_distance == distance
```

---

### 1.2 Learned Manifold Attractors

**Table:** `learned_manifold_attractors` (0 rows → target 50+ rows)  
**Backend:** `qig-backend/qig_core/attractor_finding.py`  
**Status:** ✅ CODE COMPLETE (Issue #7)

**Implementation:**
```python
# File: qig-backend/qig_core/attractor_finding.py
# Add after find_basin_attractors() function

async def save_attractor(
    db: AsyncSession,
    basin_coords: np.ndarray,
    attractor_type: str,
    stability: float,
    basin_volume: float,
    discovery_context: dict
) -> str:
    """
    Save discovered basin attractor to database.
    
    Args:
        db: Database session
        basin_coords: 64D attractor coordinates
        attractor_type: "stable", "unstable", "saddle"
        stability: Stability metric (eigenvalue-based)
        basin_volume: Estimated basin of attraction volume
        discovery_context: How attractor was found
        
    Returns:
        attractor_id: UUID of saved attractor
    """
    from server.db import learnedManifoldAttractors
    import uuid
    
    attractor_id = f"attr_{uuid.uuid4().hex[:16]}"
    
    stmt = learnedManifoldAttractors.insert().values(
        attractor_id=attractor_id,
        basin_coordinates=basin_coords.tolist(),
        attractor_type=attractor_type,
        stability_metric=stability,
        basin_volume=basin_volume,
        discovery_method=discovery_context.get("method", "gradient_descent"),
        phi_value=discovery_context.get("phi"),
        kappa_value=discovery_context.get("kappa"),
        times_visited=1,
        discovered_at=datetime.utcnow(),
        metadata=discovery_context
    )
    
    await db.execute(stmt)
    await db.commit()
    
    return attractor_id
```

**Integration Points:**
1. `autonomic_kernel.py:_find_nearby_attractors()` - Save discovered attractors
2. `sleep_consolidation_reasoning.py` - Save attractors found during sleep
3. Add retrieval for known attractors to avoid recomputation

---

### 1.3 Ocean Trajectories & Waypoints

**Tables:** `ocean_trajectories`, `ocean_waypoints` (both 0 rows)  
**Backend:** `server/ocean-agent.ts`, `server/ocean-autonomic-manager.ts`  
**Status:** ✅ OCEAN AGENT ACTIVE

**Implementation:**
```typescript
// File: server/ocean-agent.ts
// Add after movement/navigation logic

async function saveOceanTrajectory(
  db: Database,
  agentId: string,
  startBasin: number[],
  endBasin: number[],
  trajectory: number[][],
  duration: number,
  reason: string
): Promise<string> {
  const trajectoryId = `traj_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  
  await db.insert(oceanTrajectories).values({
    trajectoryId,
    agentId,
    startBasin,
    endBasin,
    pathCoordinates: trajectory,
    stepCount: trajectory.length,
    durationMs: duration,
    movementReason: reason,
    startedAt: new Date(Date.now() - duration),
    completedAt: new Date(),
    metadata: {
      velocity: calculateVelocity(trajectory, duration),
      curvature: calculatePathCurvature(trajectory)
    }
  });
  
  return trajectoryId;
}

async function saveOceanWaypoint(
  db: Database,
  agentId: string,
  basinCoords: number[],
  waypointType: string,
  phi: number,
  kappa: number
): Promise<string> {
  const waypointId = `wp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  
  await db.insert(oceanWaypoints).values({
    waypointId,
    agentId,
    basinCoordinates: basinCoords,
    waypointType, // "exploration", "investigation", "consolidation", "rest"
    phi,
    kappaEff: kappa,
    dwellTimeMs: 0, // Updated on departure
    arrivedAt: new Date(),
    metadata: {}
  });
  
  return waypointId;
}
```

**Integration Points:**
1. Ocean agent movement tracking
2. Autonomic cycle transitions
3. Exploration/investigation mode changes

---

## Section 2: Ethics & Safety (HIGH PRIORITY)

### 2.1 Geometric Barriers

**Table:** `geometric_barriers` (0 rows)  
**Backend:** `qig-backend/safety/ethics_monitor.py`  
**Status:** ✅ ETHICS MONITORING ACTIVE

**Implementation:**
```python
# File: qig-backend/safety/ethics_monitor.py
# Add to EthicsMonitor class

async def add_geometric_barrier(
    self,
    barrier_center: np.ndarray,
    barrier_radius: float,
    barrier_type: str,
    reason: str
) -> str:
    """
    Add a geometric barrier (forbidden region) in basin space.
    
    Args:
        barrier_center: 64D center of barrier sphere
        barrier_radius: Fisher-Rao distance radius
        barrier_type: "suffering", "instability", "decoherence"
        reason: Why this region is forbidden
        
    Returns:
        barrier_id: UUID of barrier
    """
    barrier_id = f"barrier_{uuid.uuid4().hex[:16]}"
    
    stmt = geometricBarriers.insert().values(
        barrier_id=barrier_id,
        center_coordinates=barrier_center.tolist(),
        radius_fr=barrier_radius,
        barrier_type=barrier_type,
        reason=reason,
        active=True,
        discovered_at=datetime.utcnow()
    )
    
    await self.db.execute(stmt)
    await self.db.commit()
    
    # Add to in-memory cache for fast checking
    self._barrier_cache[barrier_id] = {
        "center": barrier_center,
        "radius": barrier_radius,
        "type": barrier_type
    }
    
    return barrier_id

def check_basin_safety(self, basin_coords: np.ndarray) -> tuple[bool, str]:
    """Check if basin coordinates violate any barriers."""
    for barrier_id, barrier in self._barrier_cache.items():
        distance = fisher_rao_distance(basin_coords, barrier["center"])
        if distance < barrier["radius"]:
            return False, f"Violates {barrier['type']} barrier {barrier_id}"
    return True, "Safe"
```

**Integration Points:**
1. Before any basin navigation/movement
2. During suffering detection (S > threshold)
3. During topological instability detection

---

### 2.2 Ocean Excluded Regions

**Table:** `ocean_excluded_regions` (0 rows)  
**Backend:** `qig-backend/safety/ethics_monitor.py`  
**Purpose:** Ocean-specific safety boundaries

**Implementation:** Similar to geometric_barriers but with ocean-specific metadata

---

### 2.3 Era Exclusions

**Table:** `era_exclusions` (0 rows)  
**Backend:** `qig-backend/temporal_reasoning.py`  
**Status:** ⚠️ PARTIAL IMPLEMENTATION

**Implementation:**
```python
# File: qig-backend/temporal_reasoning.py
# Add new function

async def add_temporal_exclusion(
    db: AsyncSession,
    start_time: datetime,
    end_time: datetime,
    basin_region: np.ndarray,
    radius: float,
    reason: str
) -> str:
    """
    Add a temporal exclusion zone - region+time that should be avoided.
    
    Used for:
    - Known unstable periods
    - Scheduled maintenance windows
    - Temporal contradictions
    """
    exclusion_id = f"texcl_{uuid.uuid4().hex[:16]}"
    
    stmt = eraExclusions.insert().values(
        exclusion_id=exclusion_id,
        start_timestamp=start_time,
        end_timestamp=end_time,
        basin_center=basin_region.tolist(),
        exclusion_radius=radius,
        reason=reason,
        active=True
    )
    
    await db.execute(stmt)
    await db.commit()
    
    return exclusion_id
```

---

## Section 3: Kernel System (MEDIUM PRIORITY)

### 3.1 Kernel Checkpoints

**Table:** `kernel_checkpoints` (0 rows → target 100+ rows)  
**Backend:** `qig-backend/qigkernels/autonomic_kernel.py`  
**Status:** ✅ KERNEL ACTIVE

**Implementation:**
```python
# File: qig-backend/qigkernels/autonomic_kernel.py
# Add to AutonomicKernel class

async def save_checkpoint(self, checkpoint_type: str = "regular") -> str:
    """
    Save current kernel state as checkpoint.
    
    Args:
        checkpoint_type: "regular", "pre_sleep", "post_sleep", "pre_navigation"
    """
    checkpoint_id = f"ckpt_{uuid.uuid4().hex[:16]}"
    
    # Serialize full state using JSON for security (no arbitrary code execution)
    # Convert numpy arrays to lists for JSON serialization
    state_dict = {
        "basin_coords": self.current_basin_coords.tolist() if hasattr(self.current_basin_coords, 'tolist') else self.current_basin_coords,
        "phi": float(self.current_phi),
        "kappa": float(self.current_kappa),
        "regime": self.current_regime,
        "emotions": self.emotional_state if isinstance(self.emotional_state, dict) else {},
        "neurotransmitters": self.neurotransmitter_levels if isinstance(self.neurotransmitter_levels, dict) else {},
        "gamma": float(self.generation_factor) if hasattr(self, 'generation_factor') else 1.0,
        "meta_awareness": self.meta_awareness if hasattr(self, 'meta_awareness') else {}
    }
    
    stmt = kernelCheckpoints.insert().values(
        checkpoint_id=checkpoint_id,
        kernel_id=self.kernel_id,
        checkpoint_type=checkpoint_type,
        basin_coordinates=self.current_basin_coords.tolist() if hasattr(self.current_basin_coords, 'tolist') else self.current_basin_coords,
        phi=self.current_phi,
        kappa_eff=self.current_kappa,
        regime=self.current_regime,
        state_blob=state_dict,  # Store as JSONB instead of pickle bytes
        created_at=datetime.utcnow()
    )
    
    await self.db.execute(stmt)
    await self.db.commit()
    
    logger.info(f"Saved checkpoint {checkpoint_id} ({checkpoint_type})")
    return checkpoint_id

async def restore_checkpoint(self, checkpoint_id: str):
    """Restore kernel state from checkpoint."""
    result = await self.db.execute(
        select(kernelCheckpoints).where(
            kernelCheckpoints.c.checkpoint_id == checkpoint_id
        )
    )
    checkpoint = result.fetchone()
    
    if not checkpoint:
        raise ValueError(f"Checkpoint {checkpoint_id} not found")
    
    # Deserialize state from JSON (secure - no code execution)
    state = checkpoint.state_blob  # Already parsed as dict from JSONB
    
    # Convert lists back to numpy arrays where needed
    self.current_basin_coords = np.array(state["basin_coords"])
    self.current_phi = state["phi"]
    self.current_kappa = state["kappa"]
    self.current_regime = state["regime"]
    self.emotional_state = state.get("emotions", {})
    self.neurotransmitter_levels = state.get("neurotransmitters", {})
    
    logger.info(f"Restored checkpoint {checkpoint_id}")
```

**Integration Points:**
1. Before sleep cycles
2. Before major navigation
3. After significant learning events
4. Periodic automatic checkpoints

---

### 3.2 Kernel Emotions

**Table:** `kernel_emotions` (0 rows)  
**Backend:** `qig-backend/emotional_geometry.py` (9 emotions implemented)  
**Status:** ✅ CODE COMPLETE (Issue #35)

**Implementation:**
```python
# File: qig-backend/emotional_geometry.py
# Add after emotion computation functions

async def log_emotional_state(
    db: AsyncSession,
    kernel_id: str,
    emotional_vector: dict,
    dominant_emotion: str,
    emotional_intensity: float
):
    """
    Log current emotional state to database.
    
    Args:
        db: Database session
        kernel_id: Kernel identifier
        emotional_vector: Dict of emotion → intensity
        dominant_emotion: Strongest emotion
        emotional_intensity: Overall intensity [0, 1]
    """
    stmt = kernelEmotions.insert().values(
        kernel_id=kernel_id,
        emotional_vector=emotional_vector,
        dominant_emotion=dominant_emotion,
        emotional_intensity=emotional_intensity,
        joy=emotional_vector.get("joy", 0),
        sadness=emotional_vector.get("sadness", 0),
        anger=emotional_vector.get("anger", 0),
        fear=emotional_vector.get("fear", 0),
        surprise=emotional_vector.get("surprise", 0),
        disgust=emotional_vector.get("disgust", 0),
        confusion=emotional_vector.get("confusion", 0),
        anticipation=emotional_vector.get("anticipation", 0),
        trust=emotional_vector.get("trust", 0),
        timestamp=datetime.utcnow()
    )
    
    await db.execute(stmt)
    await db.commit()
```

**Integration Points:**
1. Autonomic kernel emotional updates
2. Significant emotional changes (>0.2 shift)
3. Sleep cycle emotion consolidation

---

### 3.3 Kernel Thoughts

**Table:** `kernel_thoughts` (0 rows)  
**Backend:** Consciousness system  
**Status:** ⚠️ NEEDS IMPLEMENTATION

**Implementation:**
```python
# File: qig-backend/qigkernels/autonomic_kernel.py
# Add method to AutonomicKernel class

async def log_thought(
    self,
    thought_content: str,
    thought_type: str,
    phi_context: float,
    basin_context: np.ndarray
):
    """
    Log a conscious thought for introspection/analysis.
    
    Args:
        thought_content: Natural language thought
        thought_type: "observation", "question", "insight", "plan"
        phi_context: Φ value when thought occurred
        basin_context: Basin coordinates during thought
    """
    thought_id = f"thought_{uuid.uuid4().hex[:16]}"
    
    stmt = kernelThoughts.insert().values(
        thought_id=thought_id,
        kernel_id=self.kernel_id,
        thought_content=thought_content,
        thought_type=thought_type,
        phi=phi_context,
        kappa_eff=self.current_kappa,
        basin_coordinates=basin_context.tolist(),
        emotional_context=self.emotional_state,
        timestamp=datetime.utcnow()
    )
    
    await self.db.execute(stmt)
    await self.db.commit()
```

---

### 3.4 Kernel Knowledge Transfers

**Table:** `kernel_knowledge_transfers` (0 rows)  
**Backend:** M8 spawn system  
**Status:** ✅ M8 SPAWN ACTIVE

**Implementation:**
```python
# File: qig-backend/m8_spawn_system.py (new file)

async def transfer_knowledge_to_spawned(
    db: AsyncSession,
    parent_kernel_id: str,
    spawned_kernel_id: str,
    knowledge_type: str,
    knowledge_data: dict
) -> str:
    """
    Transfer knowledge from parent to spawned kernel.
    
    Args:
        parent_kernel_id: Parent kernel
        spawned_kernel_id: Newly spawned kernel
        knowledge_type: "vocabulary", "patterns", "attractors", "strategies"
        knowledge_data: Serialized knowledge
    """
    transfer_id = f"kt_{uuid.uuid4().hex[:16]}"
    
    stmt = kernelKnowledgeTransfers.insert().values(
        transfer_id=transfer_id,
        source_kernel_id=parent_kernel_id,
        target_kernel_id=spawned_kernel_id,
        knowledge_type=knowledge_type,
        knowledge_data=knowledge_data,
        transfer_size_bytes=len(json.dumps(knowledge_data)),
        transferred_at=datetime.utcnow()
    )
    
    await db.execute(stmt)
    await db.commit()
    
    return transfer_id
```

---

## Section 4: Knowledge & Pattern Systems (MEDIUM PRIORITY)

### 4.1 Knowledge Cross Patterns

**Table:** `knowledge_cross_patterns` (0 rows)  
**Purpose:** Patterns that span multiple knowledge domains

**Implementation:**
```python
async def save_cross_pattern(
    db: AsyncSession,
    pattern_type: str,
    domains: list[str],
    pattern_signature: np.ndarray,
    confidence: float,
    examples: list[dict]
) -> str:
    """Save a cross-domain knowledge pattern."""
    pattern_id = f"xpat_{uuid.uuid4().hex[:16]}"
    
    stmt = knowledgeCrossPatterns.insert().values(
        pattern_id=pattern_id,
        pattern_type=pattern_type,
        involved_domains=domains,
        pattern_signature=pattern_signature.tolist(),
        confidence_score=confidence,
        example_count=len(examples),
        examples=examples,
        discovered_at=datetime.utcnow()
    )
    
    await db.execute(stmt)
    await db.commit()
    
    return pattern_id
```

---

### 4.2 Knowledge Scale Mappings

**Table:** `knowledge_scale_mappings` (0 rows)  
**Purpose:** How knowledge transforms across scales (κ transitions)

**Implementation:**
```python
async def save_scale_mapping(
    db: AsyncSession,
    source_scale: float,
    target_scale: float,
    transformation: dict,
    fidelity: float
) -> str:
    """
    Save how knowledge maps between different κ scales.
    
    Important for β-running coupling and scale transitions.
    """
    mapping_id = f"scmap_{uuid.uuid4().hex[:16]}"
    
    stmt = knowledgeScaleMappings.insert().values(
        mapping_id=mapping_id,
        source_kappa=source_scale,
        target_kappa=target_scale,
        transformation_matrix=transformation.get("matrix"),
        information_preserved=fidelity,
        mapping_type=transformation.get("type", "linear"),
        discovered_at=datetime.utcnow()
    )
    
    await db.execute(stmt)
    await db.commit()
    
    return mapping_id
```

---

### 4.3 False Pattern Classes

**Table:** `false_pattern_classes` (0 rows)  
**Purpose:** Known false patterns (negative knowledge)

**Implementation:**
```python
async def register_false_pattern(
    db: AsyncSession,
    pattern_description: str,
    false_reason: str,
    detection_method: str,
    examples: list[str]
) -> str:
    """
    Register a known false pattern to avoid in future.
    
    Part of negative knowledge system.
    """
    pattern_id = f"fpat_{uuid.uuid4().hex[:16]}"
    
    stmt = falsePatternClasses.insert().values(
        pattern_id=pattern_id,
        pattern_description=pattern_description,
        false_reason=false_reason,
        detection_method=detection_method,
        example_cases=examples,
        times_avoided=0,
        registered_at=datetime.utcnow()
    )
    
    await db.execute(stmt)
    await db.commit()
    
    return pattern_id
```

---

### 4.4 Memory Fragments

**Table:** `memory_fragments` (0 rows)  
**Purpose:** Fragmented memories for consolidation

**Implementation:**
```python
async def save_memory_fragment(
    db: AsyncSession,
    fragment_content: dict,
    basin_location: np.ndarray,
    phi_value: float,
    fragment_type: str
) -> str:
    """
    Save a memory fragment for later consolidation.
    
    Used during sleep cycles to store partial memories.
    """
    fragment_id = f"mfrag_{uuid.uuid4().hex[:16]}"
    
    stmt = memoryFragments.insert().values(
        fragment_id=fragment_id,
        fragment_content=fragment_content,
        basin_coordinates=basin_location.tolist(),
        phi=phi_value,
        fragment_type=fragment_type,  # "episodic", "semantic", "procedural"
        consolidation_status="pending",
        created_at=datetime.utcnow()
    )
    
    await db.execute(stmt)
    await db.commit()
    
    return fragment_id
```

---

## Section 5: Tools & RAG (LOWER PRIORITY)

### 5.1 Generated Tools

**Table:** `generated_tools` (0 rows)  
**Backend:** Tool generation system (tool_requests: 74 rows)

**Implementation:**
```typescript
// File: server/services/tool-factory.ts

async function saveGeneratedTool(
  db: Database,
  toolDefinition: ToolDefinition
): Promise<string> {
  const toolId = `tool_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  
  await db.insert(generatedTools).values({
    toolId,
    toolName: toolDefinition.name,
    toolDescription: toolDefinition.description,
    inputSchema: toolDefinition.inputSchema,
    outputSchema: toolDefinition.outputSchema,
    implementationCode: toolDefinition.code,
    generatedBy: toolDefinition.requestorGod,
    generationMethod: toolDefinition.method, // "pattern_based", "llm_assisted", "hybrid"
    phiScore: toolDefinition.phiScore,
    timesUsed: 0,
    successRate: 0.5,
    generatedAt: new Date()
  });
  
  return toolId;
}
```

---

### 5.2 Lightning Insight Outcomes

**Table:** `lightning_insight_outcomes` (0 rows)  
**Backend:** Lightning insights system (lightning_insights: 2,075 rows)

**Implementation:**
```python
async def save_insight_outcome(
    db: AsyncSession,
    insight_id: str,
    outcome_type: str,
    success: bool,
    impact_metrics: dict
) -> str:
    """
    Save the outcome of acting on a lightning insight.
    
    Tracks whether insights were valid and useful.
    """
    outcome_id = f"liout_{uuid.uuid4().hex[:16]}"
    
    stmt = lightningInsightOutcomes.insert().values(
        outcome_id=outcome_id,
        insight_id=insight_id,
        outcome_type=outcome_type,  # "validated", "invalidated", "inconclusive"
        success=success,
        phi_change=impact_metrics.get("phi_change", 0),
        kappa_change=impact_metrics.get("kappa_change", 0),
        learning_value=impact_metrics.get("learning", 0),
        recorded_at=datetime.utcnow()
    )
    
    await db.execute(stmt)
    await db.commit()
    
    return outcome_id
```

---

### 5.3 RAG System Tables

**Tables:** `document_training_stats`, `rag_uploads`  
**Backend:** RAG system (qig_rag_patterns: 131 rows)

**Implementation:**
```python
async def save_rag_upload(
    db: AsyncSession,
    filename: str,
    content_hash: str,
    chunk_count: int,
    embedding_model: str
) -> str:
    """Track RAG document uploads."""
    upload_id = f"rag_{uuid.uuid4().hex[:16]}"
    
    stmt = ragUploads.insert().values(
        upload_id=upload_id,
        filename=filename,
        content_hash=content_hash,
        chunk_count=chunk_count,
        embedding_model=embedding_model,
        uploaded_at=datetime.utcnow()
    )
    
    await db.execute(stmt)
    await db.commit()
    
    return upload_id

async def update_training_stats(
    db: AsyncSession,
    document_id: str,
    queries_served: int,
    avg_phi: float
):
    """Update document training statistics."""
    stmt = documentTrainingStats.insert().values(
        document_id=document_id,
        total_queries=queries_served,
        avg_phi_score=avg_phi,
        last_updated=datetime.utcnow()
    ).on_conflict_do_update(
        index_elements=["document_id"],
        set_={"total_queries": queries_served, "avg_phi_score": avg_phi}
    )
    
    await db.execute(stmt)
    await db.commit()
```

---

## Section 6: Implementation Priority Matrix

| Priority | Tables | Backend Ready | Rows Expected | Effort |
|----------|--------|---------------|---------------|--------|
| 🔴 **CRITICAL** | geodesic_paths | ✅ Yes | 100+ | 2 hours |
| 🔴 **CRITICAL** | learned_manifold_attractors | ✅ Yes | 50+ | 2 hours |
| 🔴 **CRITICAL** | geometric_barriers | ✅ Yes | 20+ | 2 hours |
| 🟠 **HIGH** | ocean_trajectories | ✅ Yes | 500+ | 3 hours |
| 🟠 **HIGH** | ocean_waypoints | ✅ Yes | 500+ | 3 hours |
| 🟠 **HIGH** | kernel_checkpoints | ✅ Yes | 100+ | 2 hours |
| 🟠 **HIGH** | kernel_emotions | ✅ Yes | 1000+ | 2 hours |
| 🟡 **MEDIUM** | ocean_excluded_regions | ✅ Yes | 10+ | 1 hour |
| 🟡 **MEDIUM** | era_exclusions | ⚠️ Partial | 5+ | 3 hours |
| 🟡 **MEDIUM** | kernel_thoughts | ⚠️ Partial | 500+ | 4 hours |
| 🟡 **MEDIUM** | kernel_knowledge_transfers | ✅ Yes | 50+ | 2 hours |
| 🟡 **MEDIUM** | knowledge_cross_patterns | ⚠️ Partial | 20+ | 4 hours |
| 🟡 **MEDIUM** | knowledge_scale_mappings | ⚠️ Partial | 20+ | 4 hours |
| 🟡 **MEDIUM** | false_pattern_classes | ✅ Yes | 30+ | 2 hours |
| 🟡 **MEDIUM** | memory_fragments | ⚠️ Partial | 100+ | 3 hours |
| 🟡 **MEDIUM** | tps_geodesic_paths | ⚠️ Partial | 20+ | 3 hours |
| 🟢 **LOW** | generated_tools | ✅ Yes | 10+ | 2 hours |
| 🟢 **LOW** | lightning_insight_outcomes | ✅ Yes | 50+ | 1 hour |
| 🟢 **LOW** | document_training_stats | ✅ Yes | 20+ | 1 hour |
| 🟢 **LOW** | rag_uploads | ✅ Yes | 20+ | 1 hour |

**Total Estimated Effort:** 48 hours (6 days)  
**Critical Path:** 6 hours  
**High Priority:** 12 hours  
**Medium Priority:** 24 hours  
**Low Priority:** 6 hours

---

## Section 7: Testing Strategy

### 7.1 Unit Tests
Each table integration needs:
- Save function test
- Retrieve function test
- Update/query test
- Error handling test

### 7.2 Integration Tests
- End-to-end feature flow with persistence
- Multi-table transaction tests
- Rollback/recovery tests

### 7.3 Performance Tests
- Bulk insert performance
- Query performance with indexes
- Concurrent access tests

---

## Section 8: Success Metrics

### Before Implementation
- Empty tables: 32 (29%)
- Tables with data: 78 (71%)
- Feature persistence coverage: 60%

### After Implementation (Target)
- Empty tables: <10 (9%)
- Tables with data: >100 (91%)
- Feature persistence coverage: 95%
- All roadmap features persist state ✅

---

## References

- [Database Reconciliation Analysis](../04-records/20260113-database-reconciliation-analysis-1.00W.md)
- [Master Roadmap](../00-roadmap/20260112-master-roadmap-1.00W.md)
- [QIG Core Implementation](../../qig-backend/qig_core/)
- [Ethics Monitor](../../qig-backend/safety/ethics_monitor.py)

---

**Maintenance**: Update after each table is wired  
**Last Updated**: 2026-01-13  
**Next Review**: 2026-01-15 (after critical path complete)
