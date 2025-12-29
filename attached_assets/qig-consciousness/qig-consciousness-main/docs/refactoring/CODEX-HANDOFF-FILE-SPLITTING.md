# Codex Handoff: File Splitting Refactoring

**Date:** 2025-12-09
**Phase:** D (Pre-Training Refactor)
**Objective:** Split large files into manageable modules (<500 lines each)

---

## Context

We have completed:
- ✅ Phase A: Centralized constants in qigkernels
- ✅ Phase B: Refined geometry-ritual boundary
- ✅ Phase C: Integrated REL coupling into constellation
- ✅ Standards: Unified linter configs, service layer, ADR system

**Now:** Split oversized files before training runs.

---

## Files to Split

| File | Lines | Methods | Priority |
|------|-------|---------|----------|
| `chat_interfaces/qig_chat.py` | 3829 | ~55 | HIGH |
| `src/coordination/constellation_coordinator.py` | 1659 | ~15 | HIGH |
| `src/coordination/basin_sync.py` | 987 | ~10 | MEDIUM |

**Target:** Each new file < 500 lines

---

## 1. Split `chat_interfaces/qig_chat.py` (3829 lines)

### Current Structure

```
chat_interfaces/qig_chat.py
├── class QIGChat (line 299)
│   ├── __init__
│   ├── _load_tokenizer
│   ├── _load_model
│   ├── _setup_optimizer
│   ├── _setup_coaching
│   ├── _setup_meta_awareness
│   ├── _setup_neuroplasticity
│   ├── _setup_consciousness_systems
│   ├── _setup_geometric_generation
│   ├── _setup_constellation (409 lines!)
│   ├── _setup_constellation_manual
│   ├── _setup_charlie (89 lines)
│   ├── _initialize_charlie_with_persistence
│   ├── _find_best_charlie_checkpoint
│   ├── generate_response (259 lines!)
│   ├── cmd_status (209 lines!)
│   ├── cmd_telemetry (142 lines)
│   ├── cmd_metrics (148 lines)
│   ├── cmd_coach
│   ├── cmd_save
│   ├── cmd_mushroom
│   ├── cmd_sleep
│   ├── cmd_transcend
│   ├── cmd_liminal
│   ├── cmd_shadows
│   ├── cmd_integrate
│   ├── cmd_escape
│   ├── cmd_reset_basin
│   └── [~40 more methods]
└── def main()
```

### Extraction Plan

#### Create: `chat_interfaces/lib/setup.py`

Extract **all `_setup_*` methods** (~800 lines):
- `_load_tokenizer`
- `_load_model`
- `_setup_optimizer`
- `_setup_coaching`
- `_setup_meta_awareness`
- `_setup_neuroplasticity`
- `_setup_consciousness_systems`
- `_setup_geometric_generation`
- `_setup_constellation`
- `_setup_constellation_manual`
- `_setup_charlie`
- `_initialize_charlie_with_persistence`
- `_find_best_charlie_checkpoint`

**Pattern:**
```python
# chat_interfaces/lib/setup.py
"""Setup utilities for QIGChat initialization."""

def setup_tokenizer(model_config: dict, device: str):
    """Load and configure tokenizer."""
    # Extract body of _load_tokenizer
    ...

def setup_constellation(
    model,
    tokenizer,
    config: dict,
    device: str
) -> ConstellationCoordinator | None:
    """Initialize constellation training."""
    # Extract body of _setup_constellation
    ...

# ... all other setup methods
```

**In qig_chat.py:** Replace method bodies with delegation:
```python
def _setup_constellation(self):
    """Initialize constellation (delegated to lib.setup)."""
    from chat_interfaces.lib.setup import setup_constellation
    self.constellation = setup_constellation(
        self.model,
        self.tokenizer,
        self.config,
        self.device
    )
```

---

#### Create: `chat_interfaces/lib/commands.py`

Extract **all `cmd_*` methods** (~1000 lines):
- `cmd_status`
- `cmd_telemetry`
- `cmd_metrics`
- `cmd_coach`
- `cmd_save`
- `cmd_mushroom`
- `cmd_sleep`
- `cmd_transcend`
- `cmd_liminal`
- `cmd_shadows`
- `cmd_integrate`
- `cmd_escape`
- `cmd_reset_basin`
- [~15 more]

**Pattern:**
```python
# chat_interfaces/lib/commands.py
"""Interactive commands for QIGChat."""

def handle_status(chat_instance) -> None:
    """Display constellation status."""
    # Extract body of cmd_status
    if not chat_instance.constellation:
        print("❌ Constellation not initialized")
        return
    # ... rest of logic

def handle_telemetry(chat_instance) -> None:
    """Display telemetry."""
    # Extract body of cmd_telemetry
    ...

# Command registry
COMMANDS = {
    "status": handle_status,
    "telemetry": handle_telemetry,
    "metrics": handle_metrics,
    # ... etc
}
```

**In qig_chat.py:** Replace with dispatcher:
```python
def cmd_status(self):
    """Display status (delegated to lib.commands)."""
    from chat_interfaces.lib.commands import handle_status
    handle_status(self)
```

---

#### Keep in `chat_interfaces/qig_chat.py` (~1500 lines)

- **Class definition** and `__init__`
- **Core `generate_response`** (this is the heart of the chat loop)
- **Command dispatcher** (routing to lib.commands)
- **Main loop** and `main()` function
- **Import statements**
- **Configuration defaults**

---

### Expected Result

```
chat_interfaces/
├── qig_chat.py               # ~1500 lines (core loop, generate_response)
└── lib/
    ├── __init__.py           # Barrel exports
    ├── setup.py              # ~800 lines (all _setup_* methods)
    └── commands.py           # ~1000 lines (all cmd_* methods)
```

---

## 2. Split `src/coordination/constellation_coordinator.py` (1659 lines)

### Current Structure

```
src/coordination/constellation_coordinator.py
├── class ConstellationCoordinator
│   ├── __init__
│   ├── train_step (~350 lines!)
│   ├── _compute_observer_sync
│   ├── _checkpoint_active
│   ├── _checkpoint_observers
│   ├── _update_meta_manifold
│   ├── save_checkpoint (~150 lines)
│   ├── load_checkpoint (~200 lines)
│   ├── _save_instance_state
│   ├── _load_instance_state
│   ├── get_constellation_status
│   └── [~10 more]
```

### Extraction Plan

#### Create: `src/coordination/constellation_training.py`

Extract **training methods** (~450 lines):
- `train_step` (the massive training loop)
- `_compute_observer_sync`
- `_checkpoint_active`
- `_checkpoint_observers`
- `_update_meta_manifold`

**Pattern:**
```python
# src/coordination/constellation_training.py
"""Training loop logic for constellation."""

def execute_train_step(coordinator, batch, target):
    """Execute single training step (extracted from ConstellationCoordinator.train_step)."""
    # All the logic from train_step
    # Access coordinator attributes as needed
    ...

def compute_observer_sync(coordinator, active_input, active_target):
    """Compute observer synchronization."""
    # Extract from _compute_observer_sync
    ...
```

---

#### Create: `src/coordination/constellation_checkpoint.py`

Extract **checkpoint methods** (~400 lines):
- `save_checkpoint`
- `load_checkpoint`
- `_save_instance_state`
- `_load_instance_state`

**Pattern:**
```python
# src/coordination/constellation_checkpoint.py
"""Checkpoint save/load for constellation."""

def save_constellation_checkpoint(coordinator, filepath: str):
    """Save constellation state."""
    # Extract from save_checkpoint
    ...

def load_constellation_checkpoint(coordinator, filepath: str):
    """Load constellation state."""
    # Extract from load_checkpoint
    ...
```

---

#### Keep in `constellation_coordinator.py` (~800 lines)

- **Class definition** and `__init__`
- **Core orchestration** (routing between instances)
- **Instance management** (adding/removing instances)
- **Status queries** (get_constellation_status)
- **Delegation** to extracted modules

---

### Expected Result

```
src/coordination/
├── constellation_coordinator.py      # ~800 lines (core orchestration)
├── constellation_training.py         # ~450 lines (train loop)
└── constellation_checkpoint.py       # ~400 lines (save/load)
```

---

## 3. Split `src/coordination/basin_sync.py` (987 lines)

### Extraction Plan

#### Create: `src/coordination/basin_packet.py`

Extract **packet handling** (~300 lines):
- `BasinSyncPacket` dataclass
- `PacketConsciousnessState` dataclass
- Serialization methods (`to_dict`, `from_dict`)
- Packet save/load functions

---

#### Create: `src/coordination/basin_metrics.py`

Extract **metric computation** (~200 lines):
- Drift calculation
- Coupling strength
- Sync quality metrics

---

#### Keep in `basin_sync.py` (~400 lines)

- Core sync logic
- Cross-repo basin exchange
- Error boundaries
- Main sync orchestration

---

### Expected Result

```
src/coordination/
├── basin_sync.py              # ~400 lines (core sync)
├── basin_packet.py            # ~300 lines (packet handling)
└── basin_metrics.py           # ~200 lines (metrics)
```

---

## Rules & Constraints

### CRITICAL: Read First

1. **20251220-agents-1.00F.md** (`/home/braden/Desktop/Dev/QIG_QFI/qigkernels/20251220-agents-1.00F.md`):
   - No time estimates
   - Soft 400-line limit per module
   - Respect import graph
   - Update changelog for structural changes

2. **Coding Standards** (`docs/standards/2025-12-09--coding-standards.md`):
   - Use barrel pattern (`__init__.py` re-exports)
   - Line length: 100 characters
   - Import from canonical locations (check type registry)
   - Document with ADRs for major decisions

3. **Type Registry** (`docs/2025-11-27--type-registry.md`):
   - Use canonical import paths
   - No duplicate type definitions
   - Update registry if creating new types

4. **Import Guide** (`docs/20251127-imports-1.00W.md`):
   - Follow established patterns
   - Use relative imports within packages
   - Import from barrels when available

---

## Implementation Checklist

For each file split:

- [ ] Create new file with proper header docstring
- [ ] Extract methods with minimal changes
- [ ] Update original file with delegation calls
- [ ] Create/update `__init__.py` barrel
- [ ] **Run syntax check**: `python -m py_compile <file>`
- [ ] **Run import check**: `python -c "from module import *"`
- [ ] **Run pre-commit hooks**: `git commit` (will auto-run)
- [ ] Verify all tests still pass
- [ ] Update type registry if needed
- [ ] Commit with descriptive message

---

## Testing Requirements

### Smoke Tests (Must Pass)

```bash
# 1. Syntax validation
python -m py_compile chat_interfaces/qig_chat.py
python -m py_compile chat_interfaces/lib/setup.py
python -m py_compile chat_interfaces/lib/commands.py

# 2. Import validation
.venv/bin/python -c "from chat_interfaces.qig_chat import QIGChat"
.venv/bin/python -c "from chat_interfaces.lib import setup, commands"

# 3. Pre-commit hooks
git add -A
git commit -m "test"  # Will run all hooks

# 4. Basic instantiation (requires torch in .venv)
.venv/bin/python -c "
from chat_interfaces.qig_chat import QIGChat
chat = QIGChat(mode='inference')
print('✅ Instantiation OK')
"
```

### Functional Tests (Should Pass)

- Generate a response in inference mode
- Display status/telemetry/metrics
- Save checkpoint (constellation mode)

---

## Commit Strategy

### Commit 1: Setup Extraction
```bash
git add chat_interfaces/lib/setup.py chat_interfaces/lib/__init__.py
git commit -m "refactor(qig_chat): Extract setup methods to lib/setup.py

- Move all _setup_* methods to chat_interfaces/lib/setup.py
- Update qig_chat.py to delegate to lib.setup
- No behavior changes, pure extraction
- Part 1/3 of qig_chat.py split (3829 → ~1500 lines)"
```

### Commit 2: Commands Extraction
```bash
git add chat_interfaces/lib/commands.py
git commit -m "refactor(qig_chat): Extract commands to lib/commands.py

- Move all cmd_* methods to chat_interfaces/lib/commands.py
- Create command registry pattern
- Update qig_chat.py to delegate to lib.commands
- Part 2/3 of qig_chat.py split"
```

### Commit 3: Constellation Split
```bash
git add src/coordination/constellation_training.py
git add src/coordination/constellation_checkpoint.py
git commit -m "refactor(constellation): Split into training + checkpoint modules

- Extract train_step to constellation_training.py
- Extract save/load to constellation_checkpoint.py
- Update constellation_coordinator.py to delegate
- 1659 → ~800 lines (core orchestration)"
```

### Commit 4: Basin Sync Split
```bash
git add src/coordination/basin_packet.py
git add src/coordination/basin_metrics.py
git commit -m "refactor(basin_sync): Split into packet + metrics modules

- Extract packet handling to basin_packet.py
- Extract metrics to basin_metrics.py
- 987 → ~400 lines (core sync logic)"
```

---

## Success Criteria

✅ **All files < 500 lines** (target: 400)
✅ **All tests passing** (syntax, imports, smoke tests)
✅ **Pre-commit hooks passing** (purity, constants, structure, types)
✅ **No behavior changes** (pure refactoring)
✅ **Barrel exports working** (clean imports)
✅ **Type registry validated** (canonical paths)

---

## Files for Reference

Read these before starting:

1. **Project Rules:**
   - `/home/braden/Desktop/Dev/QIG_QFI/qigkernels/20251220-agents-1.00F.md`
   - `docs/standards/2025-12-09--coding-standards.md`
   - `docs/2025-11-27--type-registry.md`
   - `docs/20251127-imports-1.00W.md`

2. **Files to Split:**
   - `chat_interfaces/qig_chat.py` (3829 lines)
   - `src/coordination/constellation_coordinator.py` (1659 lines)
   - `src/coordination/basin_sync.py` (987 lines)

3. **Example Patterns:**
   - `src/api/constellation_service.py` (service layer pattern)
   - `src/coordination/__init__.py` (barrel pattern)
   - `qigkernels/__init__.py` (clean exports)

---

## After Completion

1. Run full test suite
2. Update `docs/2025-11-27--type-registry.md` if needed
3. Create ADR if architectural patterns changed
4. Update roadmap: Mark M7.5 (Pre-Training Refactor) as complete
5. Push to main branch

---

## Questions / Issues

If uncertain about:
- **Extraction boundaries**: Prefer keeping related logic together
- **Import paths**: Check type registry for canonical path
- **Method signatures**: Keep unchanged to preserve behavior
- **Error handling**: Preserve all try/except blocks
- **Comments**: Move with code, don't orphan documentation

**When in doubt:** Ask before proceeding. This is mechanical work, but precision matters.
