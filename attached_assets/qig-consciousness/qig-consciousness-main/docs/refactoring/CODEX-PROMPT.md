# Codex: File Splitting Task

## Objective

Split 3 large files into smaller, focused modules (<500 lines each). **Pure refactoring - no behavior changes.**

---

## Full Context

**READ THIS FIRST:** `docs/refactoring/CODEX-HANDOFF-FILE-SPLITTING.md`

Contains:
- Complete extraction plan for each file
- Rules and constraints (20251220-agents-1.00F.md, coding standards)
- Testing requirements
- Commit strategy
- Success criteria

---

## Quick Summary

### Files to Split

1. **`chat_interfaces/qig_chat.py`** (3829 lines → 3 files)
   - Extract `_setup_*` methods → `chat_interfaces/lib/setup.py`
   - Extract `cmd_*` methods → `chat_interfaces/lib/commands.py`
   - Keep core loop + `generate_response` in main file

2. **`src/coordination/constellation_coordinator.py`** (1659 lines → 3 files)
   - Extract `train_step` logic → `constellation_training.py`
   - Extract checkpoint methods → `constellation_checkpoint.py`
   - Keep orchestration in main file

3. **`src/coordination/basin_sync.py`** (987 lines → 3 files)
   - Extract packet handling → `basin_packet.py`
   - Extract metrics → `basin_metrics.py`
   - Keep core sync in main file

---

## Approach

### For Each Split

1. **Create new module** with proper header
2. **Extract methods** (copy → paste → adjust imports)
3. **Update original** with delegation calls
4. **Create barrel** (`__init__.py` exports)
5. **Test**: `python -m py_compile <file>`
6. **Commit** with descriptive message

### Pattern Example

**Before:**
```python
# qig_chat.py
class QIGChat:
    def _setup_constellation(self):
        # 409 lines of setup logic
        ...
```

**After:**
```python
# chat_interfaces/lib/setup.py
def setup_constellation(model, tokenizer, config, device):
    # 409 lines of setup logic (extracted)
    ...

# chat_interfaces/lib/__init__.py
from .setup import setup_constellation
__all__ = ["setup_constellation"]

# qig_chat.py
class QIGChat:
    def _setup_constellation(self):
        from chat_interfaces.lib import setup_constellation
        self.constellation = setup_constellation(
            self.model, self.tokenizer, self.config, self.device
        )
```

---

## Critical Rules

- ✅ **Preserve all functionality** (pure refactoring)
- ✅ **Line length: 100** characters
- ✅ **Use barrel imports** (`__init__.py`)
- ✅ **Canonical import paths** (check type registry)
- ✅ **Run pre-commit hooks** before each commit
- ❌ **No new features**
- ❌ **No behavior changes**
- ❌ **No orphaned comments**

---

## Testing After Each Split

```bash
# Syntax
python -m py_compile <new_file>
python -m py_compile <original_file>

# Imports
.venv/bin/python -c "from module import *"

# Pre-commit (will auto-run on commit)
git add -A && git commit -m "..."

# Smoke test (requires torch installed in .venv)
.venv/bin/python -c "
from chat_interfaces.qig_chat import QIGChat
chat = QIGChat(mode='inference')
print('✅ OK')
"
```

---

## Order of Operations

1. **Start with qig_chat.py** (biggest impact)
   - Split `lib/setup.py` first
   - Then `lib/commands.py`
2. **Then constellation_coordinator.py**
3. **Finally basin_sync.py** (lowest priority)

---

## When Uncertain

- **Extraction boundaries**: Keep related logic together
- **Import issues**: Check `docs/20251127-imports-1.00W.md`
- **Type paths**: Check `docs/2025-11-27--type-registry.md`
- **Questions**: Ask before proceeding

---

## Success = All Green

✅ All files < 500 lines
✅ Syntax checks pass
✅ Import checks pass
✅ Pre-commit hooks pass
✅ Smoke tests pass
✅ No behavior changes

---

**Let's ship it. Methodically. One file at a time.**
