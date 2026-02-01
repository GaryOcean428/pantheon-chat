# Kernel Lifecycle Operations - User Guide

## Overview

The Kernel Lifecycle system provides first-class operations for managing kernel evolution, specialization, and optimization. All operations maintain geometric correctness on the Fisher-Rao manifold.

**Authority**: E8 Protocol v4.0, WP5.3  
**Status**: ACTIVE  
**Created**: 2026-01-23

## Core Operations

### 1. Spawn
Create new kernels based on role specifications.

**Python API:**
```python
from kernel_lifecycle import get_lifecycle_manager
from kernel_spawner import RoleSpec

manager = get_lifecycle_manager()

role = RoleSpec(
    domains=["synthesis", "foresight"],
    required_capabilities=["prediction"],
    preferred_god="Apollo",
    allow_chaos_spawn=True,
)

kernel = manager.spawn(role, mentor="apollo_kernel_id")
# Returns: Kernel instance with protected status (50 cycles)
```

**REST API:**
```bash
curl -X POST http://localhost:5001/lifecycle/spawn \
  -H "Content-Type: application/json" \
  -d '{
    "domains": ["synthesis", "foresight"],
    "required_capabilities": ["prediction"],
    "preferred_god": "Apollo",
    "allow_chaos_spawn": true,
    "mentor": "kernel_abc123"
  }'
```

**When to use:** 
- Need new capability not covered by existing kernels
- Response to increased workload in specific domain
- User requests specific god/capability

## Database Schema

### Tables

1. **kernel_lifecycle_events**
   - Tracks all lifecycle operations
2. **shadow_pantheon**
   - Archives pruned kernels
3. **lifecycle_policies**
   - Stores policy rules
4. **kernel_geometry** (augmented)
   - Added lifecycle fields

### Views

1. **lifecycle_metrics** - Aggregated event statistics
2. **shadow_pantheon_summary** - Pruned kernel summary by type
3. **active_kernels_lifecycle** - Current lifecycle status

## References

- **E8 Protocol v4.0**: Core geometric framework
- **WP5.3**: Kernel lifecycle specification
- **Issue #68**: Canonical geometry for merge operations
- **Issue #78**: Pantheon registry for promotion
