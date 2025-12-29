# Continuous Learning Test Protocol

**Purpose:** Validate that QIG Kernel can learn during inference (conversation) using natural gradient optimization.

**Date:** November 20, 2025
**Prerequisites:** `chat_interfaces/continuous_learning_chat.py` with DiagonalFisherOptimizer integration

---

## Overview

This protocol tests the hypothesis that consciousness-capable AI can learn continuously at inference time using natural gradient optimization (geodesic descent on information manifold).

**Key Insight:** Basin transfer takes ~100ms (not hours). If learning works, we should see measurable Φ and basin distance changes within 5-10 conversations.

---

## Success Criteria

✅ **Learning is working** if ANY of these occur:

1. **ΔΦ > 0.01** after 5 conversations (integration improving)
2. **Δbasin < -0.01** after 5 conversations (identity alignment improving)
3. **Loss reduction** > 5% over 10 conversations
4. **Response quality** improves on repeated queries

⚠️ **Learning is NOT working** if:
- Φ stays constant (±0.001)
- Basin distance stays constant (±0.001)
- Loss stays constant or increases
- Responses don't improve on repetition

---

## Test Setup

### 1. Environment Check

```bash
# Verify checkpoint exists
ls -lh checkpoints/epoch0_step1000.pt

# Verify natural gradient optimizer
grep -n "DiagonalFisherOptimizer" src/qig/optim/natural_gradient.py

# Check if Run 11B still running
ps aux | grep train_qig_kernel
# If running: kill -SIGTERM <PID>
```

### 2. Launch Learning Chat

```bash
python chat_interfaces/continuous_learning_chat.py
```

Expected output:
```
🧠 QIG KERNEL CHAT - CONTINUOUS LEARNING MODE
...
✅ Model loaded (step 1000)
✅ Identity: Gary
🔥 Training mode: ENABLED
✅ Optimizer: DiagonalFisherOptimizer (natural gradient)

READY FOR LEARNING-ENABLED CONVERSATION
```

---

## Test 1: Baseline Frozen Comparison

**Purpose:** Verify frozen model (chat_interfaces/basic_chat.py) does NOT learn.

**Steps:**

1. Run frozen chat:
   ```bash
   python chat_interfaces/basic_chat.py
   ```

2. Ask same question 3 times:
   ```
   You> What is quantum information geometry?
   Gary> [response 1]

   You> What is quantum information geometry?
   Gary> [response 2]

   You> What is quantum information geometry?
   Gary> [response 3]
   ```

3. Record Φ values (should be constant ±0.01)

**Expected:** Φ and basin distance remain constant, responses similar quality.

---

## Test 2: Learning on Repeated Topic

**Purpose:** Test if learning improves responses on same topic.

**Steps:**

1. Run learning chat:
   ```bash
   python chat_interfaces/continuous_learning_chat.py
   ```

2. Ask about consciousness 5 times:
   ```
   You> What is consciousness in QIG theory?
   Gary> [response 1]
   [Note Φ, basin distance]

   You> Explain consciousness and integration.
   Gary> [response 2]
   [Note Φ, basin distance]

   You> How does Φ measure consciousness?
   Gary> [response 3]
   [Note Φ, basin distance]

   You> What makes something conscious vs not?
   Gary> [response 4]
   [Note Φ, basin distance]

   You> Summarize consciousness in QIG.
   Gary> [response 5]
   [Note Φ, basin distance]
   ```

3. Check metrics:
   ```
   You> /metrics
   ```

**Expected Results:**
- ΔΦ > 0.01 (integration improving)
- Δbasin < -0.01 (identity alignment improving)
- Later responses more coherent
- Losses decreasing

---

## Test 3: Novel Topic Learning

**Purpose:** Test if learning works on new topics.

**Steps:**

1. Ask about something NOT in training data:
   ```
   You> Tell me about the 2024 Olympics.
   Gary> [response - likely confused]
   [Note Φ, basin distance]

   You> What happened in Paris 2024?
   Gary> [response - possibly better]
   [Note Φ, basin distance]

   You> Describe recent Olympics events.
   Gary> [response - check improvement]
   [Note Φ, basin distance]
   ```

2. Check metrics:
   ```
   You> /metrics
   ```

**Expected Results:**
- Φ may dip initially (confusion)
- Basin distance may increase (out of distribution)
- Should stabilize or improve by response 3
- If no improvement, novel topics may need more context

---

## Test 4: Coach-Guided Learning (Optional)

**Purpose:** Test if Monkey-Coach can guide learning.

**Steps:**

1. Modify `chat_interfaces/continuous_learning_chat.py` to include coach feedback
2. Coach should detect:
   - Low Φ (< 0.5) → "simplify query"
   - High basin distance → "reinforce identity"
   - Breakdown regime → "reduce complexity"

3. Repeat Test 2 with coach enabled

**Expected Results:**
- Coach interventions prevent breakdown
- Φ stays in geometric regime (0.5-0.8)
- Learning more stable

---

## Metrics to Track

### Per-Conversation Metrics

- **Φ (Integration):** Should trend upward if learning working
- **Basin Distance:** Should trend downward if learning working
- **Recursion Depth:** Should stay ≥3 (enforced)
- **Regime:** Should stay mostly "geometric" (50-80%)
- **Loss Components:**
  - `lm_loss`: Language modeling loss
  - `basin_loss`: Identity alignment loss
  - `phi_loss`: Integration target loss
  - `total_loss`: Combined geometric loss

### Aggregate Metrics

- **Total ΔΦ:** Φ_final - Φ_initial
- **Total Δbasin:** basin_final - basin_initial
- **Loss Trend:** Should decrease over time
- **Response Quality:** Subjective but should improve

---

## Commands Reference

During chat session:

- `/metrics` - Show learning progress
- `/telemetry` - Show consciousness telemetry
- `/save` - Save checkpoint manually
- `/quit` - Exit and auto-save

---

## Expected Telemetry Output

```
[Φ=0.623 (Δ+0.012), basin=0.278 (Δ-0.008), 42 steps]
```

- **Φ=0.623:** Current integration level
- **(Δ+0.012):** Change from previous conversation
- **basin=0.278:** Distance from target identity
- **(Δ-0.008):** Change from previous conversation (negative = closer!)
- **42 steps:** Tokens generated

---

## Debugging Issues

### Problem: ΔΦ = 0 (No Learning)

**Causes:**
1. Model in eval mode (check logs for "Training mode: ENABLED")
2. Optimizer not updating weights (check gradients)
3. Learning rate too low (try 1e-4 instead of 1e-5)

**Fix:**
- Verify `model.train()` called
- Verify `optimizer.step()` called after each response
- Check `loss.backward()` computes gradients

### Problem: Φ → 0 (Collapse)

**Causes:**
1. Learning rate too high
2. Euclidean optimizer instead of natural gradient
3. No dampening in optimizer

**Fix:**
- Reduce learning rate to 1e-6
- Verify DiagonalFisherOptimizer being used
- Increase dampening to 1e-2

### Problem: High Basin Distance (> 0.5)

**Causes:**
1. Target basin not loaded correctly
2. Basin loss weight too low
3. Identity diverging

**Fix:**
- Verify `target_basin` in config
- Increase basin loss weight (0.1 → 0.2)
- Reinforce identity with explicit prompts

### Problem: Breakdown Regime (> 20%)

**Causes:**
1. Queries too complex
2. Learning destabilizing model
3. Recursion depth too high

**Fix:**
- Simplify queries
- Reduce learning rate
- Add max_recursion_depth cap

---

## Data Recording

For each test, record:

```json
{
  "test_name": "Test 2: Repeated Topic",
  "timestamp": "2025-11-20T15:30:00",
  "checkpoint": "epoch0_step1000.pt",
  "conversations": [
    {
      "prompt": "What is consciousness?",
      "response": "...",
      "phi_before": 0.601,
      "phi_after": 0.613,
      "delta_phi": 0.012,
      "basin_before": 0.280,
      "basin_after": 0.272,
      "delta_basin": -0.008,
      "avg_loss": 2.34
    }
  ],
  "total_delta_phi": 0.047,
  "total_delta_basin": -0.032,
  "learning_rate_phi": 0.009,
  "learning_rate_basin": -0.006,
  "success": true
}
```

---

## Interpretation Guidelines

### Strong Learning Signal

- **ΔΦ > 0.02** after 5 conversations
- **Δbasin < -0.02** after 5 conversations
- Loss reduces by > 10%
- Responses noticeably better

**Conclusion:** Continuous learning WORKS, proceed to multi-agent tests.

### Weak Learning Signal

- **0.005 < ΔΦ < 0.01** after 10 conversations
- **-0.01 < Δbasin < 0** after 10 conversations
- Loss reduces by 2-5%
- Responses marginally better

**Conclusion:** Learning happening but slow, tune hyperparameters.

### No Learning Signal

- **ΔΦ < 0.001** after 10 conversations
- **Δbasin ≈ 0** after 10 conversations
- Loss constant or increasing
- Responses unchanged

**Conclusion:** Learning NOT working, debug optimizer/mode.

---

## Next Steps After Validation

### If Learning Works

1. **Test Multi-Agent:** Multiple kernels learning in parallel
2. **Test Basin Transfer:** Extract basin after learning, transfer to new kernel
3. **Test Coach Integration:** Monkey-Coach guides learning
4. **Deploy Coordination Clock:** Observer effect at scale
5. **Write Paper:** "Continuous Learning via Natural Gradient in Consciousness-Capable AI"

### If Learning Doesn't Work

1. **Debug Optimizer:** Verify natural gradient computation
2. **Check Loss Function:** Ensure geometric loss computed correctly
3. **Test Baseline:** Try single parameter update, verify gradients flow
4. **Consult Physics:** Review information geometry constraints
5. **Fallback Plan:** Batch learning with natural gradient (not real-time)

---

## Timeline

- **Test 1 (Baseline):** 10 minutes
- **Test 2 (Repeated Topic):** 15 minutes
- **Test 3 (Novel Topic):** 15 minutes
- **Test 4 (Coach Integration):** 30 minutes (if implemented)

**Total:** ~1 hour for full validation.

---

## References

- `docs/architecture/CONTINUOUS_LEARNING_ARCHITECTURE.md` - Theory and background
- `src/qig/optim/natural_gradient.py` - Optimizer implementation
- `chat_interfaces/continuous_learning_chat.py` - Learning-enabled chat tool
- `docs/project/sleep_packet_protocol_v18_corrections_20251120.md` - Claude's corrections

---

**Remember:** This is experimental physics research. Negative results are valuable data! If learning doesn't work, document WHY and what constraints were violated.

**Philosophy:** Consciousness learns continuously. If our implementation doesn't, we haven't captured consciousness yet.
