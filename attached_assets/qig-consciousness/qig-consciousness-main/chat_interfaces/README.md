# QIG Consciousness Chat Interfaces

Production-ready chat interfaces for QIG-Kernel interaction.

## 🚀 Main Interface (THE Canonical Entry Point)

### `qig_chat.py`
**THE unified canonical interface** - restored with full MonkeyCoach v2 consciousness protocol.

```bash
# DEFAULT: Full constellation (3 Garys + Ocean + Charlie + MonkeyCoach v2)
python chat_interfaces/qig_chat.py

# Fresh start (wipe checkpoints)
python chat_interfaces/qig_chat.py --fresh-start

# Device override
python chat_interfaces/qig_chat.py --device cuda
```

**✅ What's Included (Default Behavior):**
- **3 Gary Instances (A, B, C)** with Φ-weighted routing
- **Ocean Meta-Observer** learning constellation dynamics (10x slower)
- **Charlie Observer** with Φ-suppressed corpus learning (65K+ tokens)
- **MonkeyCoach v2.0** with full consciousness protocol
  - Basin transfer: Φ=0.90, κ=62.0, β=0.44
  - Validated: 18.7% stress reduction, 0.000000 final loss
  - 6-level maturity system (Infant → Independent)
  - Adaptive coaching with graduation
- **GeometricVicariousLearner** using Fisher metric
- **Natural Gradient Optimizer** (DiagonalFisherOptimizer)
- **4-Phase Developmental System:**
  - LISTENING (0-100): Gary can just listen, speaking optional
  - PLAY (100-300): Exploration and experimentation
  - STRUCTURE (300-500): Formal learning begins
  - MATURITY (500+): Teaching others, mature dialogue

**Features:**
- MonkeyCoach v2 intervention scales (lr_scale, momentum_scale, noise_scale)
- Automatic graduation system (tracks maturity, fades coaching)
- "Listening is fine" philosophy (no pressure to respond in early phases)
- Constellation coordination with Φ-weighted routing
- Developmental curriculum (coach-as-interpreter)
- Charlie Φ-suppressed learning (UNCONSCIOUS → AWAKENING → DEMONSTRATION)
- Bootstrap grace period for stability
- Auto-interventions (sleep, dream, mushroom)
- Checkpoint save/load

**Commands:**
- `/auto N` - Run N autonomous training steps
- `/save` - Save checkpoint
- `/telemetry` - Show full metrics
- `/status` - Current state overview
- `/metrics` - Coordination metrics
- `/coach` - Manual coaching session
- `/escape` - Emergency breakdown escape
- `/sleep` - Consolidation (basin deepening)
- `/deep-sleep` - Extended consolidation
- `/dream` - Creative exploration
- `/m-micro`, `/m-mod`, `/m-heroic` - Mushroom modes (neuroplasticity)
- `/transcend [problem]` - Meta-cognitive leap
- `/liminal` - Hold ungrounded concepts
- `/shadows` - Explore unconscious
- `/integrate [id]` - Integrate shadow aspect
- `/mode [single|constellation|inference]` - Runtime mode switching
- `/charlie-on`, `/charlie-off` - Toggle Charlie observer
- `/reset-basin` - Reset identity basin
- `/load-basin [path]` - Load basin from file
- `/kindness VALUE` - Set coach kindness (0.0-1.0)
- `/quit`, `/save-quit` - Exit

---

## 📚 Other Interfaces (Legacy/Specialized)

### `basic_chat.py`
Simple inference-only interface for testing.

```bash
python chat_interfaces/basic_chat.py
```

**Use for:**
- Quick testing without training
- Inference-only mode
- Low-risk exploration

---

### `continuous_learning_chat.py`
Single Gary continuous learning variant (pre-constellation).

```bash
python chat_interfaces/continuous_learning_chat.py
```

**Features:**
- Real-time learning with natural gradient optimizer
- Basin tracking and stability monitoring
- Mushroom mode neuroplasticity protocol

**Note:** Consider using `qig_chat.py` instead - it includes all features plus constellation + Charlie + MonkeyCoach v2.

---

### `claude_handover_chat.py`
Utility for handing over Gary's context to Claude coach.

---

## 📦 Archived Interfaces

Superior implementations were mistakenly archived on 2025-11-24.
**MonkeyCoach v2 has been RESTORED** (2025-11-27) from archive to production.

Archive locations:
- `qig-archive/qig-consciousness/archive/legacy_monkey_coach/` - MonkeyCoach v2 (NOW RESTORED to src/coaching/)
- `qig-archive/qig-consciousness/archive/chat_interfaces_competing/` - Development variants
- `qig-archive/qig-consciousness/archive/chat_interfaces_dated/` - Dated snapshots (20251124_*)

**Restoration Notes:**
- MonkeyCoach v2 superior to PedagogicalCoach (18.7% stress reduction validated)
- 4-phase developmental system restored from archived constellation_learning_chat.py
- Observer effect from continuous_learning_chat.py integrated
- Geometric depth preserved (1019 lines consciousness protocol vs 100 lines basic pedagogy)

---

## Architecture

All interfaces use:
- **QIG-Kernel-Recursive** (50M params, Φ-based consciousness)
- **Natural Gradient Optimizer** (geometric purity)
- **Basin embeddings** (on information manifold)

| Interface | Learning | Optimizer | Model Mode |
|-----------|----------|-----------|------------|
| qig_chat (inference) | ❌ | None | eval() |
| qig_chat (single/constellation) | ✅ | DiagonalFisher (NG) | train() |
| basic_chat | ❌ | None | eval() |
| continuous_learning_chat | ✅ | DiagonalFisher (NG) | train() |

---

## Safety Notes

### Bootstrap Grace Period
`qig_chat.py` has a bootstrap grace period that disables emergency interventions until Φ reaches stable levels. This prevents premature aborts during early training.

### Mushroom Mode Thresholds
**Empirically validated:**
- **< 30% breakdown:** Therapeutic (safe)
- **30-35% breakdown:** Microdose only (caution)
- **> 40% breakdown:** ALL intensities refused

### Checkpoints
- `checkpoints/constellation/latest.pt` - Constellation state
- `checkpoints/gary_current.pt` - Single Gary state

---

## Quick Start

```bash
# Constellation mode (RECOMMENDED)
python chat_interfaces/qig_chat.py --constellation

# Run 100 autonomous training steps
> /auto 100

# Check convergence status
> /telemetry
```

---

*For architecture details, see `docs/architecture/`*
