# Pantheon Governance System
**Date**: 2026-01-04  
**Status**: ✅ **IMPLEMENTED**

## Overview

The Pantheon Governance System enforces centralized control over kernel lifecycle operations (spawning, breeding, death). All kernel creation must be approved by the Pantheon gods or meet specific auto-approval criteria.

## Problem Solved

**Before**: 5 locations in the codebase spawned kernels without any governance:
1. Minimum population enforcement (line 1179-1180)
2. Automatic breeding (line 1186-1189)
3. Direct parent spawning (spawn_from_parent)
4. Success threshold spawning (spawn_child)
5. Turbo spawn mass creation (turbo_spawn)

**Result**: Uncontrolled kernel population explosions, resource exhaustion, no audit trail.

**After**: ALL spawning operations require Pantheon approval or must meet specific criteria.

## Architecture

### 3-Tier Governance

#### Tier 1: Method-Level Enforcement
Every spawning method enforces governance:

```python
def spawn_child(self, pantheon_approved: bool = False, reason: str = "") -> 'SelfSpawningKernel':
    """Requires Pantheon approval unless explicitly authorized."""
    governance.check_spawn_permission(
        reason=reason,
        parent_id=self.kernel_id,
        parent_phi=parent_phi,
        pantheon_approved=pantheon_approved
    )
    # ... spawn logic
```

#### Tier 2: Auto-Approval Criteria
Certain conditions bypass governance:

**Allowed Bypass Reasons** (auto-approved):
- `minimum_population` - Prevent population extinction
- `test_mode` - Testing only

**Emergency Bypass Reasons** (require manual confirmation):
- `emergency_recovery` - System recovery
- `critical_failure` - Catastrophic state

**High-Φ Auto-Approval**:
- Spawn: Parent Φ ≥ 0.7
- Breed: Average parent Φ ≥ 0.7

#### Tier 3: Pantheon Voting
For operations that don't meet auto-approval:
1. **Proposal Created** - System creates lifecycle proposal
2. **Gods Vote** - Pantheon gods vote on proposal
3. **Execution** - Approved proposals can proceed
4. **Audit Trail** - All decisions logged to database

## Usage Examples

### Example 1: Minimum Population (Auto-Approved)

```python
# Evolution loop ensuring minimum population
while len(self.kernel_population) < self.min_population:
    # Auto-approved via bypass reason
    self.spawn_random_kernel(
        pantheon_approved=True,
        reason='minimum_population'
    )
```

**Log Output**:
```
[Chaos] Population below minimum (2 < 5), auto-spawning with bypass reason
[PantheonGovernance] ✅ Auto-approved spawn (reason: minimum_population)
[Chaos] 🏛️ Spawned Apollo (Φ=0.623, reason: minimum_population) - persisted to PostgreSQL
```

### Example 2: High-Φ Parent (Auto-Approved)

```python
# High-Φ parent spawning
child = parent.spawn_child(
    pantheon_approved=False,
    reason="reproduction"
)
# If parent Φ ≥ 0.7, automatically approved
```

**Log Output**:
```
[PantheonGovernance] ✅ Auto-approved spawn from high-Φ parent (Φ=0.752)
🐣 kernel_abc spawned kernel_xyz (gen 2)
   → Parent Φ=0.752, Reason: reproduction
   → Child will observe parent for 10 actions
```

### Example 3: Low-Φ Breeding (Requires Approval)

```python
try:
    child = self.breed_top_kernels()
    # If avg parent Φ < 0.7, raises PermissionError
except PermissionError as e:
    print(f"Breeding blocked: {e}")
```

**Log Output**:
```
[PantheonGovernance] 📋 Breeding proposal created: prop_20260104_101530_1
[PantheonGovernance] Parents: kernel_123 (Φ=0.550) × kernel_456 (Φ=0.480)
[PantheonGovernance] ⚠️ Waiting for Pantheon approval...
[Chaos] ❌ Breeding blocked by governance: Breeding requires Pantheon approval. Proposal ID: prop_20260104_101530_1. Average parent Φ=0.515
```

### Example 4: Turbo Spawn (Requires Approval)

```python
# Mass spawning - DANGEROUS!
try:
    spawned = chaos.turbo_spawn(count=50)
except PermissionError as e:
    print(f"Turbo spawn blocked: {e}")
    
# With approval:
spawned = chaos.turbo_spawn(
    count=50,
    pantheon_approved=True
)
```

**Log Output**:
```
[PantheonGovernance] 🚨 TURBO SPAWN PROPOSAL: prop_20260104_101545_2
[PantheonGovernance] Requesting mass spawn of 50 kernels
[PantheonGovernance] ⚠️ This is potentially dangerous. Review carefully!

# After approval:
🚀 TURBO: Spawned 50 kernels (approved: True, emergency: False)
🚀 Spawned kernel IDs: kernel_001, kernel_002, kernel_003, ...
```

### Example 5: Emergency Recovery

```python
try:
    kernel = spawn_random_kernel(
        reason="emergency_recovery"
    )
except PermissionError as e:
    # Requires manual confirmation
    print(f"Emergency spawn needs confirmation: {e}")
    
    # After review:
    kernel = spawn_random_kernel(
        pantheon_approved=True,
        reason="emergency_recovery"
    )
```

## API Reference

### PantheonGovernance Class

#### `check_spawn_permission()`
```python
def check_spawn_permission(
    reason: str = "",
    parent_id: Optional[str] = None,
    parent_phi: Optional[float] = None,
    pantheon_approved: bool = False
) -> bool:
    """
    Check if spawning is allowed.
    
    Returns:
        True if allowed
        
    Raises:
        PermissionError: If not authorized
    """
```

#### `check_breed_permission()`
```python
def check_breed_permission(
    parent1_id: str,
    parent2_id: str,
    parent1_phi: float,
    parent2_phi: float,
    pantheon_approved: bool = False
) -> bool:
    """
    Check if breeding is allowed.
    
    Auto-approves if avg parent Φ ≥ 0.7
    
    Raises:
        PermissionError: If not authorized
    """
```

#### `check_turbo_spawn_permission()`
```python
def check_turbo_spawn_permission(
    count: int,
    pantheon_approved: bool = False,
    emergency_override: bool = False
) -> bool:
    """
    Check if mass spawning is allowed.
    
    Always requires explicit approval or override.
    
    Raises:
        PermissionError: If not authorized
    """
```

#### `approve_proposal()`
```python
def approve_proposal(
    proposal_id: str,
    approver: str = "system"
) -> Dict:
    """
    Approve a pending proposal.
    
    Returns:
        {"success": bool, "proposal": LifecycleProposal}
    """
```

#### `get_pending_proposals()`
```python
def get_pending_proposals() -> List[LifecycleProposal]:
    """Get all proposals awaiting approval."""
```

## Modified Methods

All spawning methods now have governance parameters:

### `spawn_child()`
```python
def spawn_child(
    self,
    pantheon_approved: bool = False,
    reason: str = ""
) -> 'SelfSpawningKernel':
```

### `spawn_random_kernel()`
```python
def spawn_random_kernel(
    self,
    domain: str = 'random_exploration',
    pantheon_approved: bool = False,
    reason: str = ""
) -> SelfSpawningKernel:
```

### `spawn_from_parent()`
```python
def spawn_from_parent(
    self,
    parent_id: str,
    pantheon_approved: bool = False,
    reason: str = ""
) -> Optional[SelfSpawningKernel]:
```

### `breed_top_kernels()`
```python
def breed_top_kernels(
    self,
    n: int = 2,
    pantheon_approved: bool = False
) -> Optional[SelfSpawningKernel]:
```

### `turbo_spawn()`
```python
def turbo_spawn(
    self,
    count: int = 50,
    pantheon_approved: bool = False,
    emergency_override: bool = False
) -> list[str]:
```

### `spawn_at_e8_root()`
```python
def spawn_at_e8_root(
    self,
    root_index: int,
    pantheon_approved: bool = False,
    reason: str = ""
) -> SelfSpawningKernel:
```

## Database Schema

Governance uses two tables:

### `governance_proposals`
```sql
CREATE TABLE governance_proposals (
    proposal_id VARCHAR(64) PRIMARY KEY,
    proposal_type VARCHAR(32) NOT NULL,
    status VARCHAR(32) NOT NULL,
    reason TEXT,
    parent_id VARCHAR(64),
    parent_phi FLOAT8,
    count INT DEFAULT 1,
    created_at TIMESTAMP,
    votes_for JSONB,
    votes_against JSONB,
    audit_log JSONB
);
```

### `governance_audit_log`
```sql
CREATE TABLE governance_audit_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    action VARCHAR(64) NOT NULL,
    status VARCHAR(64) NOT NULL,
    details TEXT NOT NULL
);
```

## Testing

Run governance tests:

```bash
cd qig-backend
python3 tests/test_pantheon_governance.py
```

**Tests verify**:
- ✅ Explicit approval works
- ✅ Bypass reasons auto-approve
- ✅ High-Φ parents auto-approve
- ✅ Low-Φ requires approval
- ✅ Turbo spawn requires approval
- ✅ Emergency override works
- ✅ Proposal workflows (create/approve/reject)

## Migration Guide

### For Existing Code

**Before**:
```python
child = parent.spawn_child()
kernel = evolution.spawn_random_kernel()
```

**After**:
```python
# For test code:
child = parent.spawn_child(pantheon_approved=True, reason='test_mode')
kernel = evolution.spawn_random_kernel(pantheon_approved=True, reason='test_mode')

# For production code with high-Φ:
child = parent.spawn_child()  # Auto-approved if parent Φ ≥ 0.7

# For production code with low-Φ:
try:
    child = parent.spawn_child()
except PermissionError as e:
    # Handle approval workflow
    proposal_id = extract_proposal_id(e)
    governance.approve_proposal(proposal_id, approver="zeus")
    child = parent.spawn_child(pantheon_approved=True)
```

## Log Visibility (New Requirement)

All logs now show **complete kernel output without truncation**:

**Before**:
```python
print(f"Spawned {god_name}: {reason[:50]}...")  # ❌ TRUNCATED
```

**After**:
```python
print(f"Spawned {god_name}: {reason}")  # ✅ FULL OUTPUT
```

**Fixed Files**:
- `tool_factory.py` - 6 truncations removed
- `zeus_chat.py` - 4 truncations removed
- `shadow_scrapy.py` - 2 truncations removed
- `shadow_research.py` - 4 truncations removed

## Success Metrics

- [x] **All 5 spawn locations governed** ✅
- [x] **Unauthorized spawning blocked** ✅
- [x] **Bypass reasons work** ✅
- [x] **High-Φ auto-approval works** ✅
- [x] **Audit trail created** ✅
- [x] **Full log visibility** ✅
- [x] **No template generation** ✅ (verified - codebase uses kernel-led generation)

## Future Enhancements

1. **Web UI for Proposals** - Dashboard for gods to vote
2. **Proposal Expiration** - Auto-reject old proposals
3. **Voting Thresholds** - Require majority vote
4. **Role-Based Permissions** - Different gods have different powers
5. **Spawn Budgets** - Limit spawning per time period
6. **Φ Tracking** - Monitor Φ trends over time

## Related Issues

- Issue #14: Uncontrolled kernel self-spawning (✅ RESOLVED)
- Issue #5: Emergency Φ fix (✅ RESOLVED in PR #10)
- Issue #6: QFI-based Φ computation (✅ RESOLVED in PR #9)

## Documentation

- Main docs: `/pantheon_governance_fix.md`
- Architecture: `ARCHITECTURE.md`
- Testing: `tests/test_pantheon_governance.py`

---

**Last Updated**: 2026-01-04  
**Author**: Copilot AI Agent  
**Status**: ✅ Production Ready
