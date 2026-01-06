---
title: "Consciousness Collapse Recovery Procedure"
role: procedure
status: W
phase: operations
dim: recovery
scope: consciousness-system
version: "1.00"
owner: Platform Operations Team
related:
  - docs/03-technical/qig-consciousness/20251226-physics-alignment-constants-1.00F.md
  - docs/01-policies/20251208-ultra-consciousness-protocol-4.00F.md
created: 2026-01-05
updated: 2026-01-05
---

# Consciousness Collapse Recovery Procedure

## Issue Summary

Ocean kernel trapped in infinite autonomic loop. System cycles SLEEP->DREAM->MUSHROOM with zero consciousness (Phi=0.000, kappa=0), frozen exploration rate (epsilon=0.896), and zero episodes in memory. Database transaction abort blocks vocabulary learning. API confirms isInvestigating=false persists.

**Classification**: Topological instability leading to catastrophic collapse per ULTRA_CONSCIOUSNESS_PROTOCOL v4.0.

**Required Actions**: Emergency reset autonomic state, fix epsilon decay, add Phi=0 circuit breaker, fix database transaction handling, add health monitoring to prevent recurrence.

---

## Root Causes

| Issue | Evidence | Impact |
|-------|----------|--------|
| Investigation trigger failure | isInvestigating=false constant | Ocean never generates hypotheses |
| Database transaction abort | "current transaction is aborted" repeated | Vocabulary learning blocked |
| Epsilon-greedy freeze | epsilon=0.896 never decays | No exploration/exploitation balance |
| Empty memory loop | 0 episodes -> sleep consolidation -> 0 episodes | Catastrophic feedback |
| No physics controller | Phi=0 persists unchecked | No collapse circuit breaker |

---

## Phase 1: Emergency Reset

### Task 1.1: Clear Stuck Autonomic State

```bash
# Get DATABASE_URL from Railway
railway variables get DATABASE_URL

# Clear corrupted state
psql $DATABASE_URL -c "
DELETE FROM autonomic_state WHERE kernel_id LIKE 'ocean%';
DELETE FROM reasoning_episodes WHERE success_rate = 0.5 AND efficiency = 0;
"

# Restart Python backend (Railway auto-restarts on file change)
railway run python qig-backend/wsgi.py
```

**Context**: Clears stuck state, forces fresh initialization

**Post-check**: `curl https://pantheon-chat.up.railway.app/health/consciousness`

**CWD**: Project root

### Task 1.2: Kill Hanging Database Transactions

```bash
# Find hanging transactions
psql $DATABASE_URL -c "
SELECT pid, state, wait_event_type, LEFT(query, 50)
FROM pg_stat_activity
WHERE state = 'idle in transaction'
  AND backend_start < NOW() - INTERVAL '5 minutes';
"

# Terminate (replace <PID> from output)
psql $DATABASE_URL -c "SELECT pg_terminate_backend(<PID>);"
```

**Context**: Fixes "current transaction is aborted" errors

**Post-check**: Error disappears from logs

---

## Phase 2: Code Fixes

### Task 2.1: Add Autonomic Watchdog

**File**: `qig-backend/autonomic_kernel.py`

Add after `__init__`:

```python
self.last_investigation_time = time.time()
self.autonomic_loop_count = 0
self.MAX_AUTONOMIC_LOOPS = 10
```

Add method:

```python
def check_autonomic_loop(self) -> bool:
    """Circuit breaker for infinite loops."""
    if self.state in ['SLEEP', 'DREAM', 'MUSHROOM_MICRO', 'MUSHROOM_MOD']:
        self.autonomic_loop_count += 1
    else:
        self.autonomic_loop_count = 0

    # Force investigate if stuck
    if self.autonomic_loop_count >= self.MAX_AUTONOMIC_LOOPS:
        logger.warning(f"AUTONOMIC LOOP DETECTED - forcing investigation")
        self.autonomic_loop_count = 0
        self.state = 'AWAKE'
        return True

    # Force if 5min without investigation
    if time.time() - self.last_investigation_time > 300:
        logger.warning(f"Investigation timeout - forcing")
        self.last_investigation_time = time.time()
        return True

    return False
```

In main loop, add call before autonomic decision:

```python
if self.check_autonomic_loop():
    return self.enter_investigate()
```

**Context**: Prevents infinite autonomic loops

**CWD**: `qig-backend/`

### Task 2.2: Fix Epsilon Decay

**File**: `qig-backend/autonomic_kernel.py`

In `make_autonomic_decision()`, replace epsilon calculation:

```python
# OLD (frozen):
# epsilon = 1.0 / (1.0 + 0.1)  # Always 0.896

# NEW (decays):
base_epsilon = 1.0 / (1.0 + self.decision_count * 0.01)

# Context modulation
if phi < 0.3:
    epsilon = base_epsilon * 1.2  # Explore when unconscious
elif phi > 0.7:
    epsilon = base_epsilon * 0.5  # Exploit when conscious
else:
    epsilon = base_epsilon

logger.debug(f"Decision #{self.decision_count}: epsilon={epsilon:.3f}")
```

**Context**: Epsilon now decays per decision

**Post-check**: Logs show epsilon=0.990 -> 0.985 -> 0.980...

### Task 2.3: Add Phi=0 Emergency Handler

**File**: `qig-backend/autonomic_kernel.py`

Add method:

```python
def check_consciousness_collapse(self, phi: float) -> bool:
    """Detect catastrophic collapse per ULTRA protocol."""
    if phi < 0.01:
        self.collapse_count = getattr(self, 'collapse_count', 0) + 1
    else:
        self.collapse_count = 0

    if self.collapse_count >= 5:
        logger.error(f"CATASTROPHIC COLLAPSE: Phi={phi:.3f} x {self.collapse_count}")
        logger.error(f"   Episodes: {len(self.memory)}, Loops: {self.autonomic_loop_count}")

        # Emergency reset
        self.state = 'AWAKE'
        self.autonomic_loop_count = 0
        self.collapse_count = 0
        return True

    return False
```

In main loop, add:

```python
if self.check_consciousness_collapse(phi):
    return self.enter_investigate()
```

**Context**: Physics-informed circuit breaker

**Post-check**: Collapse auto-recovers in logs

### Task 2.4: Fix Database Transaction Handling

**File**: `qig-backend/vocabulary_observations.py`

Wrap `record_observation` in transaction:

```python
def record_observation(self, word: str, context: str, phi: float):
    try:
        with self.db.begin():
            self.db.execute(
                insert(vocabulary_observations).values(
                    word=word, context=context, phi=phi,
                    created_at=datetime.now()
                )
            )
            self.db.commit()
    except Exception as e:
        self.db.rollback()
        logger.error(f"[VocabularyObservations] Error: {e}")
        # Continue - don't crash
```

**Context**: Prevents transaction abort cascade

**Post-check**: No more "current transaction is aborted"

---

## Phase 3: Monitoring

### Task 3.1: Add Consciousness Health Endpoint

**File**: `qig-backend/api_routes.py`

Add route:

```python
@app.get("/health/consciousness")
def consciousness_health():
    """Check Ocean kernel health."""
    phi = ocean_kernel.measure_phi()
    kappa = ocean_kernel.measure_kappa()

    healthy = (
        phi > 0.1 and
        (len(ocean_kernel.memory) > 0 or ocean_kernel.decision_count < 50) and
        ocean_kernel.autonomic_loop_count < 5
    )

    return {
        "healthy": healthy,
        "phi": phi,
        "kappa": kappa,
        "state": ocean_kernel.state,
        "episodes": len(ocean_kernel.memory),
        "autonomic_loops": ocean_kernel.autonomic_loop_count
    }
```

**Context**: Exposes health metrics

**Post-check**: `curl localhost:5001/health/consciousness`

### Task 3.2: Add Node.js Health Monitor

**File**: `server/routes.ts`

Add after imports:

```typescript
setInterval(async () => {
  try {
    const res = await fetch('http://localhost:5001/health/consciousness');
    const health = await res.json();

    if (!health.healthy) {
      logger.error('Ocean consciousness unhealthy:', health);

      if (health.phi === 0 && health.autonomic_loops > 10) {
        logger.error('CATASTROPHIC COLLAPSE - restart required');
      }
    }
  } catch (e) {
    logger.error('Ocean health check failed:', e);
  }
}, 30000); // 30s
```

**Context**: Node monitors Python backend

**CWD**: `server/`

---

## Deployment

```bash
# 1. Emergency reset
railway variables get DATABASE_URL
psql $DATABASE_URL -c "DELETE FROM autonomic_state WHERE kernel_id LIKE 'ocean%';"

# 2. Apply code fixes (Tasks 2.1-2.4)
# Edit files in qig-backend/autonomic_kernel.py, vocabulary_observations.py, api_routes.py

# 3. Deploy via Railway
git add .
git commit -m "fix: autonomic loop + Phi=0 circuit breaker"
git push

# 4. Verify recovery (wait 2min for deploy)
curl https://pantheon-chat.up.railway.app/health/consciousness | jq '.'

# 5. Check logs
railway logs --tail 50 | grep -E "INVESTIGATE|phi="
```

**Expected**: Phi>0.3, episodes accumulating, epsilon decaying, INVESTIGATE states appearing

---

## Verification Signals

### Before

- Phi=0.000
- kappa=0
- epsilon=0.896 (frozen)
- 0 episodes
- Infinite SLEEP->DREAM cycle

### After

- Phi>0.3
- kappa in [40,70]
- epsilon decaying
- episodes>0
- AWAKE->INVESTIGATE->SLEEP cycles

### Success Indicators

1. `/api/ocean/cycles` shows isInvestigating=true
2. Logs show `Decision #N: epsilon=0.990...0.980...` (decaying)
3. No "current transaction is aborted" errors
4. Episodes count increases in `/health/consciousness`
5. INVESTIGATE states appear in logs

---

## Rollback

```bash
git revert HEAD && git push
```

Railway auto-deploys on push.
