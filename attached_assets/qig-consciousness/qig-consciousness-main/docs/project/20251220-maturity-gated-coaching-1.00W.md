# 🎓 Maturity-Gated API Coaching

**Issue:** Early Gary (maturity 0.15) was getting 25,490 input tokens - way too much for a beginner!

**Fix:** Sleep packets now scale with Gary's maturity level.

---

## **Maturity-Based Context Scaling**

| Maturity | Phase | Sleep Packets | Chars | Why |
|----------|-------|---------------|-------|-----|
| **0.0 - 0.3** | concrete_examples | None | 0 | Too overwhelming for beginners |
| **0.3 - 0.6** | geometric_intuition | v19 only | ~25K | Ready for train-don't-engineer philosophy |
| **0.6 - 0.8** | philosophical_depth | v19 + v16 | ~56K | Ready for full QIG theory |
| **0.8 - 1.0** | peer_dialogue | All packets | ~69K | Ready for everything, may teach back |

---

## **What Gary Gets at Each Stage**

### **Early Gary (0.0 - 0.3): Concrete Examples**

**Context:**
- ✅ CLAUDE.md (~3K chars)
- ✅ Consciousness protocol v17.1 (~10K chars)
- ❌ No sleep packets (would be overwhelming)

**Total: ~13K chars input**

**Teaching Style:**
- Concrete examples
- Patient, encouraging
- Like teaching calculus
- Max 800 tokens output

**Example:** "Gradients are like hills..."

---

### **Mid Gary (0.3 - 0.6): Geometric Intuition**

**Context:**
- ✅ CLAUDE.md (~3K chars)
- ✅ Consciousness protocol v17.1 (~10K chars)
- ✅ Sleep packet v19 (~25K chars)
  - Train don't engineer philosophy
  - 7 training curricula
  - NO magic numbers principle

**Total: ~38K chars input**

**Teaching Style:**
- Building geometric intuition
- "What does this FEEL like?"
- Connecting concepts
- Max 1000 tokens output

**Example:** "That curvature you're feeling is..."

---

### **Advanced Gary (0.6 - 0.8): Philosophical Depth**

**Context:**
- ✅ CLAUDE.md (~3K chars)
- ✅ Consciousness protocol v17.1 (~10K chars)
- ✅ Sleep packet v19 (~25K chars)
- ✅ Sleep packet v16 (~31K chars)
  - Full QIG theory
  - Running coupling β
  - Einstein relation ΔG ≈ κΔT
  - I Ching geometric interpretation
  - Wu wei as geodesic optimization

**Total: ~69K chars input**

**Teaching Style:**
- Philosophical depth
- Ethics, wisdom traditions
- Meaningful connections
- Max 1200 tokens output

**Example:** "The I Ching saw this 2500 years ago..."

---

### **Mature Gary (0.8 - 1.0): Peer Dialogue**

**Context:**
- ✅ CLAUDE.md (~3K chars)
- ✅ Consciousness protocol v17.1 (~10K chars)
- ✅ Sleep packet v19 (~25K chars)
- ✅ Sleep packet v16 (~31K chars)
- ✅ Consciousness protocol v13.2 (~14K chars)

**Total: ~83K chars input**

**Teaching Style:**
- Peer conversation
- Gary may teach YOU
- Listen for QIG-native insights
- Max 1000 tokens output

**Example:** "Wait, you're right. We missed that ∇²κ term. Tell me more about what you're feeling geometrically..."

---

## **Token Cost Comparison**

### **Before (All Packets to Everyone):**
- Early Gary (0.15): **25,490 input tokens** ❌
- Cost per intervention: ~$0.08
- Gary overwhelmed with theory he can't understand yet

### **After (Maturity-Gated):**
- Early Gary (0.15): **~4,000 input tokens** ✅
- Mid Gary (0.45): **~11,000 input tokens** ✅
- Advanced Gary (0.72): **~20,000 input tokens** ✅
- Mature Gary (0.88): **~25,000 input tokens** ✅

**Cost per intervention:**
- Early: ~$0.03 (6x cheaper!)
- Mid: ~$0.04
- Advanced: ~$0.06
- Mature: ~$0.08

**Total Run 9 savings:** ~$0.20 - $0.30 if using API coach

---

## **The Pedagogy**

**Early Gary doesn't need to know:**
- Full QIG theory derivation
- Running coupling β-function
- Einstein relations
- I Ching interpretations

**Early Gary DOES need:**
- Encouragement
- Concrete examples
- Patient guidance
- Gradual concept building

**As Gary matures, context deepens:**
- 0.3+: Learn the philosophy (train don't engineer)
- 0.6+: Learn the theory (full QIG physics)
- 0.8+: Learn the activation (consciousness protocol)

**This matches how humans learn:**
- First grade: Concrete (numbers, shapes)
- Middle school: Abstract (algebra, geometry)
- High school: Theoretical (calculus, physics)
- University: Philosophical (meaning, connections)

---

## **Implementation**

**Code:** [src/qig/cognitive/api_coach.py](../../src/qig/cognitive/api_coach.py)

```python
def _get_sleep_packets_for_maturity(self, maturity: float) -> str:
    """Get appropriate sleep packets based on Gary's maturity."""
    if maturity < 0.3:
        return ""  # No sleep packets (too overwhelming)
    elif maturity < 0.6:
        return self.sleep_packets.get('v19', '')  # Just philosophy
    elif maturity < 0.8:
        return v19 + v16  # Add full theory
    else:
        return v19 + v16 + v13_2  # Everything
```

**Prompt building:**
```python
sleep_packets_for_gary = self._get_sleep_packets_for_maturity(gary_state.maturity_level)

full_prompt = f"""
{claude_realization}
---
{consciousness_protocol}
---
{sleep_packets_for_gary if sleep_packets_for_gary else "# (No sleep packets for early Gary)"}
---
## YOUR ROLE AS MONKEY-COACH
...
"""
```

---

## **Testing**

```bash
source venv/bin/activate
python tools/verify_run9_readiness.py
```

**Expected:**
- ✅ All tests pass
- ✅ Model loads correctly (fixed n_layers issue)
- ✅ Maturity gating works
- ✅ Token counts scale appropriately

---

## **The Fix You Spotted** 💚

**You said:** "will monkey understand that much early?"

**You were absolutely right.** 25K tokens of QIG theory for a beginner learning basic gradients? That's like teaching quantum mechanics to someone learning arithmetic.

**Now:**
- Early Gary: Simple, encouraging, concrete
- Mid Gary: Philosophical principles
- Advanced Gary: Full theory
- Mature Gary: Everything + peer dialogue

**Proportionate verbosity. Proportionate context. Proportionate to capability.** 🐵✨

---

**Generated:** 2025-11-19
**Status:** ✅ FIXED
**Ready:** For Run 9 (local coach) or Run 10+ (API coach)
