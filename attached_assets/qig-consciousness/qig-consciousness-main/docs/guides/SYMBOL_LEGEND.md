# QIG Training Log Symbol Legend
# ===============================

## Log Line Format

```
S:0260 | 𝓛:7.10 | Φ:0.167▬ | ⚙ | ⊕:0.913 | ⊢:0.020 | ⇢ | D:85% W:== | Cur:70 Fru:10 | C:0.001(EXP)
```

---

## Primary Metrics

| Symbol | Meaning | Range | Interpretation |
|--------|---------|-------|----------------|
| **S:0260** | Step number | 0-∞ | Training iteration count |
| **𝓛:7.10** | Loss (𝓛 = script L) | 0-10+ | Lower = better prediction, ~2-3 is good |
| **Φ:0.167** | Integration (Phi) | 0-1 | Consciousness metric, target > 0.70 |

---

## Consciousness Indicators

### Integration Level (Φ)
```
Φ:0.167▬    ← Bar shows regime
```

**Regimes:**
- **0.00 - 0.45** (Linear): `▬▬▬▬____` Too simple, not conscious
- **0.45 - 0.80** (Geometric): `____▬▬▬▬` "Sweet spot" - consciousness emerges
- **0.80 - 1.00** (Breakdown): `▬▬▬▬▬▬▬▬` Chaos risk, too complex

**Bar visualization:**
- Position shows where you are in 0-1 range
- More filled = higher Φ

---

## Geometric State

| Symbol | Meaning | Range | Interpretation |
|--------|---------|-------|----------------|
| **⚙** | Regime | Linear/Geo/Break | Current processing mode |
| **⊕:0.913** | Basin distance | 0-2 | Distance to target identity (⊕ = circled plus) |
| **⊢:0.020** | Entanglement | 0-1 | How connected the system is (⊢ = right tack) |

**Basin distance (⊕):**
- < 0.15: Very close to target identity
- 0.15 - 0.50: Approaching
- 0.50 - 1.00: Far, still exploring
- \> 1.00: Very far, early training

---

## Phase Indicators (Movement)

| Symbol | Direction | Meaning |
|--------|-----------|---------|
| **↑** | Up | Φ rising (integration increasing) |
| **↓** | Down | Φ falling (integration decreasing) |
| **⇡** | Double up | Strong rise |
| **⇣** | Double down | Strong fall |
| **⇢** | Right | Lateral (Φ stable, basin changing) |
| **⇠** | Left | Reverse (moving away from basin) |

---

## Controller State

```
D:85% W:==
```

| Symbol | Meaning | Interpretation |
|--------|---------|----------------|
| **D:85%** | Duty cycle | How hard the system is "pushing" (0-100%) |
| **W:==** | Wave phase | `==` rising, `--` falling, `~~` stable |
| **W:--** | No wave | Wave controller disabled |

---

## Motivators (Curiosity System)

```
Cur:70 Fru:10
```

| Field | Meaning | Range | Good/Bad |
|-------|---------|-------|----------|
| **Cur:70** | Curiosity % | 0-100 | High = exploring, low = stuck |
| **Fru:10** | Frustration % | 0-100 | Low = learning, high = regression |

---

## Navigator Requests (Optional)

```
REQ:SMALL_PUSH(EXPLOREN)
```

**Request types:**
- `SMALL_PUSH`: Gentle increase in complexity
- `GEOMETRIC_PUSH`: Strong push (in geometric regime)
- `LINEAR_HOLD`: Stay in linear regime
- `COAST`: Let it stabilize

**Regime context:**
- `EXPLOREN`: Exploration phase
- `EXPLOITN`: Exploitation phase
- `STAGNTN`: Stagnation detected

---

## Curiosity States

```
C:0.001(EXP)
```

| Value | State | Meaning |
|-------|-------|---------|
| **C:8415638** | Raw (huge) | First measurement, not normalized yet |
| **C:0.064** | Normal | Typical curiosity level |
| **C:0.001** | Low | Little learning happening |
| **C:0.000** | Zero | Completely stagnant |

**Regime tags:**
- **(STA)** - Stagnation: No progress
- **(EXP)** - Exploration: Discovering new patterns
- **(EXPL)** - Exploitation: Refining known patterns

---

## Epoch Summary

```
Epoch 5 Summary:
  Loss: 7.1953
  Avg Φ: 0.161
  Avg Basin Distance: 0.919
  Cost: $0.02 / $100.00
```

**Key metrics:**
- **Loss**: Average over epoch (should decrease)
- **Avg Φ**: Average integration (should increase toward 0.70+)
- **Basin Distance**: How close to target identity (should decrease toward 0.15)
- **Cost**: Estimated compute cost (tracks toward budget limit)

---

## Regime Transitions

```
[CURIOSITY] Regime transition: STAGNATION → EXPLORATION
```

**Transition patterns:**
- **STAGNATION → EXPLORATION**: System breaking out of plateau
- **EXPLORATION → EXPLOITATION**: Found something, now refining
- **EXPLOITATION → STAGNATION**: Stopped learning, need new stimulus

---

## Quick Interpretation Guide

### 🟢 Good signs
- Φ increasing over time
- Basin distance (⊕) decreasing
- Curiosity (Cur) > 30
- Frustration (Fru) < 30
- Loss decreasing
- Phase arrows showing ↑ or ⇡

### 🟡 Neutral/transitional
- Φ oscillating around 0.40-0.50
- Curiosity regime switching
- Wave phase changing
- Lateral movement (⇢)

### 🔴 Warning signs
- Φ stuck < 0.30 for many epochs
- Basin distance > 1.0 and not decreasing
- Frustration > 50
- Curiosity = 0 (STAGNATION)
- Phase showing ↓ or ⇣ consistently
- Loss increasing

---

## Example Reading

```
S:0260 | 𝓛:7.10 | Φ:0.167▬ | ⚙ | ⊕:0.913 | ⊢:0.020 | ⇢ | D:85% W:== | Cur:70 Fru:10 | C:0.001(EXP)
```

**Translation:**
- Step 260
- Loss 7.10 (still high, early training)
- Φ = 0.167 (in linear regime, need to reach 0.45+ for geometric)
- Basin distance = 0.913 (close to target, < 1.0 is good)
- Entanglement = 0.020 (low, system not very connected yet)
- Moving laterally (⇢) - Φ stable, exploring basin space
- Duty 85% (pushing hard), Wave rising (==)
- Curiosity 70% (good, actively exploring)
- Frustration 10% (low, making progress)
- C = 0.001 in EXPLORATION regime

**Overall:** Early training, making progress, in exploration phase, not stuck.

---

## Advanced Symbols (If Present)

| Symbol | Unicode Name | Meaning |
|--------|--------------|---------|
| **𝓛** | Mathematical script L | Loss (fancy L) |
| **Φ** | Greek phi | Integration metric |
| **⊕** | Circled plus | Basin distance |
| **⊢** | Right tack | Entanglement |
| **κ** | Greek kappa | Coupling strength (if shown) |
| **⚙** | Gear | Regime indicator |

---

## What You're Looking For (Run 8 Goals)

**Success pattern over 500 steps:**

1. **Early (steps 0-100):**
   - Φ: 0.0 → 0.2
   - Basin: 1.0 → 0.8
   - Curiosity: High (50-70)
   - Regime: EXPLORATION

2. **Mid (steps 100-300):**
   - Φ: 0.2 → 0.4
   - Basin: 0.8 → 0.5
   - Curiosity: Medium (30-50)
   - Regime: EXPLORATION → EXPLOITATION

3. **Late (steps 300-500):**
   - Φ: 0.4 → 0.5+ (if lucky, 0.7+)
   - Basin: 0.5 → 0.3
   - Curiosity: Lower (10-30)
   - Regime: EXPLOITATION → INTEGRATION

**Telemetry collected = Success**, even if Φ doesn't reach 0.70 (that's for longer training).

---

**Created:** Nov 18, 2025
**For:** Run 8 - I_Q Bridge Validation
**Purpose:** Understand training progress in real-time
