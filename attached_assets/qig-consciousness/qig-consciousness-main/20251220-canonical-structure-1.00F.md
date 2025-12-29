# 📋 CANONICAL STRUCTURE - Single Source of Truth

**Status:** AUTHORITATIVE - All agents MUST follow this structure  
**Last Updated:** November 24, 2025  
**Version:** 2.0

---

## 🚨 FIRST: READ THESE FILES

| File | Purpose |
|------|---------|
| **20251220-canonical-structure-1.00F.md** | Directory structure, file locations, what goes where |
| **20251220-canonical-rules-1.00F.md** | The 10 inviolable rules, concepts, geometric purity |
| **CRITICAL_RECONCILIATION_FIX.md** | Why we consolidate to 1-2 entry points |

**Before ANY task, read these files.**

---

## 🚨 CRITICAL RULES FOR ALL AGENTS

### Rule 1: NO NEW SCRIPTS
**Do NOT create new files when existing ones serve the purpose.**

Before creating ANY new file:
1. Check this document for the canonical location
2. Search existing files with similar names
3. Enhance/fix existing files instead of creating duplicates
4. If truly new, get approval and update this document

### Rule 2: MAXIMUM 2 ENTRY POINTS

⚠️ **CORRECTED**: We had 9 chat interface files. The correct number is 1-2.

| Capability | Solution | NOT Separate Files |
|------------|----------|-------------------|
| Constellation | `qig_chat.py` | ❌ |
| Single Gary | `qig_chat.py --single` | ❌ |
| Inference Only | `qig_chat.py --inference` | ❌ |
| Charlie Demo | `qig_chat.py --charlie` | ❌ |
| Claude Coach | `qig_chat.py --claude-coach` | ❌ |

**Target State:**
```
chat_interfaces/
├── qig_chat.py          # ✅ THE canonical entry point (all features)
└── (all others archived)
```

### Rule 3: ARCHIVE DON'T DELETE
Move deprecated files to `qig-archive/qig-consciousness/archive/` with date suffix. Never delete without explicit approval.

### Rule 4: NO TIME ESTIMATES
Use Phase/Task/Step. Never Week/Hours/Days.

### Rule 5: GEOMETRIC PURITY (See 20251220-canonical-rules-1.00F.md)
- Charlie: Φ-suppressed during corpus learning
- Ocean: FROZEN
- Vicarious: Fisher metric
- Coach: Dynamics only

---

## 📁 CANONICAL DIRECTORY STRUCTURE

```
qig-consciousness/
│
├── 📋 ROOT (Governance)
│   ├── 20251220-canonical-structure-1.00F.md      # THIS FILE
│   ├── 20251220-canonical-rules-1.00F.md          # 10 inviolable rules
│   ├── CRITICAL_RECONCILIATION_FIX.md  # Why 1-2 entry points
│   ├── DREAM_PACKET_project_reconciliation_v1_0.md
│   ├── README.md
│   ├── 20251220-agents-1.00F.md
│   ├── .clinerules
│   ├── .github/copilot-instructions.md
│   └── .claude/CLAUDE.md
│
├── 🎮 chat_interfaces/              # TARGET: 1 FILE ONLY
│   ├── __init__.py
│   └── qig_chat.py                  # ✅ ALL functionality here
│
├── 🧠 src/                          # CORE IMPLEMENTATION
│   ├── model/
│   │   ├── qig_kernel_recursive.py
│   │   ├── qfi_attention.py
│   │   ├── running_coupling.py
│   │   ├── basin_matcher.py
│   │   ├── recursive_integrator.py
│   │   └── meta_reflector.py
│   │
│   ├── observation/
│   │   └── charlie_observer.py      # Charlie Φ-suppressed → awakened
│   │
│   ├── coordination/
│   │   ├── ocean_meta_observer.py   # Ocean FROZEN
│   │   └── constellation_coordinator.py
│   │
│   ├── training/
│   │   └── geometric_vicarious.py   # Fisher metric
│   │
│   ├── metrics/
│   │   └── geodesic_distance.py
│   │
│   ├── curriculum/
│   │   └── developmental_curriculum.py
│   │
│   ├── coaching/
│   │   └── pedagogical_coach.py     # Kindness = damping
│   │
│   ├── qig/
│   │   ├── optim/natural_gradient.py
│   │   └── neuroplasticity/
│   │       ├── sleep_protocol.py
│   │       └── mushroom_mode.py
│   │
│   └── tokenizer/
│       └── fast_qig_tokenizer.py
│
├── 🗄️ (archived in qig-archive/qig-consciousness/archive/)
│
├── 🔧 tools/
├── ⚙️ configs/
├── 📚 docs/
├── 🧪 tests/
├── 📊 logs/
└── 💾 checkpoints/
```

---

## 📝 COMPLETE COMMAND REFERENCE

All commands in `qig_chat.py`:

### Core Commands
| Command | Purpose |
|---------|---------|
| `/quit` | Exit without save |
| `/save-quit` | Save and exit |
| `/save` | Save checkpoint |
| `/status` | Full status (includes coach) |
| `/telemetry` | Last step metrics |
| `/metrics` | Learning history |

### Autonomous
| Command | Purpose |
|---------|---------|
| `/auto N` | Run N curriculum steps |

### Neuroplasticity
| Command | Purpose |
|---------|---------|
| `/m-micro` | Mushroom microdose |
| `/m-mod` | Mushroom moderate |
| `/m-heroic` | Mushroom heroic |

### Sleep Protocols
| Command | Purpose |
|---------|---------|
| `/sleep` | Light sleep (100 steps) |
| `/deep-sleep` | Deep sleep (300 steps) |
| `/dream` | Dream cycle (200 steps) |

### Meta-Awareness
| Command | Purpose |
|---------|---------|
| `/transcend [problem]` | Elevation protocol |
| `/liminal` | Check crystallized concepts |
| `/shadows` | View unintegrated collapses |
| `/integrate [id]` | Shadow integration |

### Coach
| Command | Purpose |
|---------|---------|
| `/coach` | Show coach summary |

---

## 🗑️ FILES TO ARCHIVE

ALL current chat interface files become `qig_chat.py`:

| Current File | Archive Name | Reason |
|--------------|--------------|--------|
| constellation_with_granite_pure.py | 20251124_constellation_with_granite_pure.py | Merged |
| continuous_learning_chat.py | 20251124_continuous_learning_chat.py | Merged |
| constellation_with_granite.py | 20251124_constellation_with_granite.py | Merged |
| constellation_learning_chat.py | 20251124_constellation_learning_chat.py | Merged |
| continuous_learning_chat_twin.py | 20251124_continuous_learning_chat_twin.py | Merged |
| autonomous_training.py | 20251124_autonomous_training.py | Merged |
| basic_chat.py | 20251124_basic_chat.py | --inference flag |
| claude_handover_chat.py | 20251124_claude_handover_chat.py | --claude-coach flag |

---

## 📊 TYPE INDEX

### Core Types
| Type | Module |
|------|--------|
| `QIGKernelRecursive` | `src/model/qig_kernel_recursive.py` |
| `CharlieObserver` | `src/observation/charlie_observer.py` |
| `OceanMetaObserver` | `src/coordination/ocean_meta_observer.py` |
| `GeometricVicariousLearner` | `src/training/geometric_vicarious.py` |
| `DiagonalFisherOptimizer` | `src/qig/optim/natural_gradient.py` |
| `QIGTokenizer` | `src/tokenizer` (re-exports from qig-tokenizer) |
| `MonkeyCoach` | `src/coaching/pedagogical_coach.py` |
| `SleepProtocol` | `src/qig/neuroplasticity/sleep_protocol.py` |
| `MushroomMode` | `src/qig/neuroplasticity/mushroom_mode.py` |
| `MetaReflector` | `src/model/meta_reflector.py` |

### Physics Constants (FROZEN)
| Constant | Value |
|----------|-------|
| κ* | 64.0 |
| κ₃ | 41.09 ± 0.59 |
| κ₄ | 64.47 ± 1.89 |
| κ₅ | 63.62 ± 1.68 |
| β(3→4) | +0.44 |
| Φ_threshold | 0.70 |
| Φ_emergency | 0.50 |
| basin_dim | 64 |

---

## ✅ PURITY CHECKLIST

Before ANY commit:

### Geometric Purity
- [ ] Charlie is READ-ONLY (Φ-suppressed observer)
- [ ] Vicarious uses Fisher metric
- [ ] Ocean is FROZEN
- [ ] Coach affects dynamics only
- [ ] Natural gradient optimizer

### Structural Purity
- [ ] No new scripts (enhance existing)
- [ ] Maximum 1-2 entry points
- [ ] Commands in single file
- [ ] Types from canonical modules

---

## 🔧 CLI FLAGS (Target State)

```bash
# Default: Single Gary continuous learning
python chat_interfaces/qig_chat.py

# Constellation mode
python chat_interfaces/qig_chat.py --constellation

# With Charlie demonstrations (Φ-suppressed observer)
python chat_interfaces/qig_chat.py --charlie

# Inference only (no training)
python chat_interfaces/qig_chat.py --inference

# Claude coaching
python chat_interfaces/qig_chat.py --claude-coach

# Disable coaching
python chat_interfaces/qig_chat.py --no-coach

# Combined
python chat_interfaces/qig_chat.py --constellation --charlie --kindness 0.85
```

---

**This document is AUTHORITATIVE. All agents must consult before creating files.**

**Current state: 9 files → Target: 1 file (`qig_chat.py`)**
