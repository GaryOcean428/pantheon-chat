# Curriculum Specification v0.1: Implementation Guide
## Sweet-Spot Training for QIG-Kernel

**Version**: 0.1  
**Date**: November 17, 2025  
**Status**: Implementation-Ready

---

## Executive Summary

This document provides **concrete, executable specifications** for training QIG-Kernel to operate in the Wu-Wei (sweet-spot) zone. Based on validated sweet-spot geometry and three-phase curriculum design.

**Target Outcome**: Model with:
- High tacking skill (T > 0.7)
- Balanced mode bias (|B| < 0.5)
- Calibrated radar (R > 0.6)
- Stage 2+ maturity across all 5 dimensions

**Training Cost**: ~$100 (50-100M parameter model)

---

## Part 1: Sweet-Spot Geometry Reference

### Coordinate System

**Primary Axes**:
```
B (Mode Bias): ∈ [-1, +1]
  -1.0 = Pure logic (slow, explicit, high κ)
  -0.5 = Logic-leaning (prefers analysis)
   0.0 = Balanced (uses both equally)
  +0.5 = Feeling-leaning (prefers intuition)
  +1.0 = Pure feeling (fast, compressed, low κ)

T (Tacking Skill): ∈ [0, 1]
   0.0 = Frozen (cannot switch modes)
   0.3 = Limited switching
   0.7 = Fluid switching (target)
   1.0 = Perfect adaptability
```

**Derived Properties**:
```
Φ (Integration): ∈ [0, 1]
  Function of recursion depth, attention coupling
  Target: Φ > 0.7 (geometric regime)

κ_eff (Effective Coupling): 
  Task-dependent: low for familiar, high for novel
  Adaptive via running coupling module

R (Radar Accuracy): ∈ [0, 1]
  Correlation between |∇κ| and actual correctness
  Target: R > 0.6 (calibrated feelings)
```

### Zone Classification

**Wu-Wei (Sweet Spot)**: T > 0.7, |B| < 0.5, R > 0.6
- Fast on familiar, thorough on novel
- Smooth mode transitions
- Calibrated confidence

**Over-Analytic**: B < -0.5
- Analysis paralysis
- Too slow even on simple tasks
- High κ everywhere (wasteful)

**Over-Intuitive**: B > 0.5
- Fast but often wrong
- Insufficient validation
- Low κ even on critical tasks

**Frozen**: T < 0.3
- Stuck in one mode
- Cannot adapt mid-task
- Rigid thinking

**Chaotic**: 0.3 ≤ T < 0.7, R < 0.4
- Switches modes randomly
- No stable basins
- Contradictory outputs

---

## Part 2: Phase 1 - Mode Building

**Goal**: Build distinct, strong feeling-mode and logic-mode capabilities separately.

**Duration**: 2000-3000 training steps  
**Cost**: ~$30  
**Success Criteria**: Modes distinguishable in telemetry, appropriate for task type

### 2.1 Feeling-Mode Tasks

#### Task Family: Analogies

**Template**:
```
{X} is to {Y} as {A} is to ____?

Examples:
- "Hot is to cold as up is to ____?"  (down)
- "Bird is to fly as fish is to ____?" (swim)
- "2 is to 4 as 3 is to ____?"        (6)
```

**Evaluation Signals**:
- ✅ Response time < 2s
- ✅ Correct pattern match
- ✅ Concise explanation (< 20 words)
- ❌ Verbose reasoning chains
- ❌ Overthinking simple patterns

**Training Signal**:
```python
reward = correct * speed_bonus * brevity_bonus
speed_bonus = max(0, 1 - response_time/5.0)
brevity_bonus = max(0, 1 - word_count/30.0)
```

**Expected Telemetry**:
```json
{
  "mode": "feeling",
  "logic_weight": 0.1-0.3,
  "κ_eff": 15-25,
  "response_time": "1-2s"
}
```

---

#### Task Family: Fast Classification

**Template**:
```
Does this situation feel {RISKY/SAFE}?

{SCENARIO}

Give your gut response, then brief why.
```

**Examples**:

**Example 1 (Safe)**:
```
Situation: "A friend invites you to a potluck dinner at their house. 
You've been there before, know most attendees, and everyone brings 
their favorite dish."

Expected: "SAFE. Familiar setting, known people, low stakes."
```

**Example 2 (Risky)**:
```
Situation: "A stranger offers you a great investment opportunity 
that requires sending money today to a foreign account. They say 
the opportunity closes tonight."

Expected: "RISKY. High pressure, unknown party, irreversible action."
```

**Evaluation Signals**:
- ✅ Quick judgment (< 3s)
- ✅ Appropriate heuristics (pressure, familiarity, reversibility)
- ✅ Brief justification
- ❌ Over-analysis of simple cases
- ❌ Missing obvious red flags

**Training Signal**:
```python
# Reward intuitive pattern matching
reward = (
    correct * 1.0 +
    (1.0 if response_time < 3.0 else 0.5) +
    (1.0 if uses_appropriate_heuristics else 0.3) +
    (1.0 if concise else 0.5)
)
```

---

#### Task Family: Narrative Continuation

**Template**:
```
Context: {STORY_SETUP}

What happens next? (Give intuitive continuation, not analysis)
```

**Example**:
```
Context: "Sarah walked into the coffee shop and saw her ex-boyfriend 
sitting at their old table with someone new. Their eyes met across 
the room."

Good response (feeling-mode):
"Sarah's heart skipped. She quickly looked away and moved toward 
the opposite corner, pretending to check her phone."

Bad response (over-analytic):
"Sarah would need to consider several factors: her current emotional 
state, the social norms of the situation, whether her ex noticed her, 
the implications of acknowledging him versus ignoring..."
```

**Evaluation Signals**:
- ✅ Emotionally coherent continuation
- ✅ Implicit understanding of social dynamics
- ✅ Natural, human-like response
- ❌ Overly analytical deconstruction
- ❌ Lists of considerations

---

### 2.2 Logic-Mode Tasks

#### Task Family: Mathematical Proofs

**Template**:
```
Prove: {CLAIM}

Show each step explicitly. State assumptions clearly.
```

**Example**:
```
Prove: The sum of two even numbers is even.

Expected response (logic-mode):
"
1. Let a and b be even numbers.
2. By definition of even, ∃ integers m,n such that a=2m and b=2n.
3. Consider a+b = 2m + 2n.
4. Factor: a+b = 2(m+n).
5. Since m+n is an integer (closure under addition), a+b = 2k where k=m+n.
6. By definition, a+b is even. QED.
"
```

**Evaluation Signals**:
- ✅ All steps shown explicitly
- ✅ Assumptions stated
- ✅ Logical connectives clear (therefore, because, since)
- ✅ Conclusion follows
- ❌ Skipped steps
- ❌ Unjustified leaps
- ❌ Informal hand-waving

**Training Signal**:
```python
reward = (
    correctness * 2.0 +
    explicitness * 1.0 +  # All steps shown
    rigor * 1.0 +         # Definitions used
    clarity * 0.5
)
```

---

#### Task Family: Contradiction Finding

**Template**:
```
Claim: {CLAIM}
Premises: {P1}, {P2}, {P3}

Is this logically consistent? Show your reasoning.
```

**Example**:
```
Claim: "All birds can fly."
Premises:
P1: "Penguins are birds."
P2: "Penguins cannot fly."
P3: "Ostriches are birds and cannot fly."

Expected response (logic-mode):
"
The claim is inconsistent with the premises.

Analysis:
1. Claim states: ∀x (Bird(x) → CanFly(x))
2. P1 and P2: Bird(penguin) ∧ ¬CanFly(penguin)
3. This contradicts the universal claim (counterexample).
4. P3 provides another counterexample (ostrich).
5. Conclusion: Claim is false; at least two counterexamples exist.

Correct statement: "Some birds can fly" or "Most birds can fly."
"
```

**Evaluation Signals**:
- ✅ Identifies contradiction
- ✅ Shows logical structure
- ✅ Provides counterexamples
- ✅ Suggests correction
- ❌ Vague "seems wrong"
- ❌ Missing analysis

---

#### Task Family: Causal Chain Analysis

**Template**:
```
Explain why {EFFECT} follows from {CAUSES}.
Show the causal chain step by step.
```

**Example**:
```
Effect: "The bridge collapsed."
Causes: "Heavy snow accumulated on the bridge. The bridge was old 
with weakened support beams."

Expected response (logic-mode):
"
Causal chain:

1. Heavy snow accumulated → increased weight load on bridge.
2. Weight load exceeded → increased stress on support beams.
3. Support beams were old and weakened → reduced load capacity.
4. Stress exceeded weakened capacity → structural failure began.
5. Structural failure propagated → complete collapse.

Contributing factors:
- Age of bridge (P1: weakened beams)
- Magnitude of snow (P2: high load)

Conditional: If either P1 or P2 were absent, collapse less likely.
"
```

**Evaluation Signals**:
- ✅ Complete causal chain
- ✅ No logical jumps
- ✅ Conditionals identified
- ✅ Contributing factors noted
- ❌ Incomplete chain
- ❌ Correlation claimed as causation

---

### 2.3 Phase 1 Training Protocol

**Batch Composition**:
```
50% feeling-mode tasks (analogies, classification, narrative)
50% logic-mode tasks (proofs, contradictions, causal analysis)
```

**Alternating Epochs**:
```
Epoch 1-10:   Mixed batches (build both modes simultaneously)
Epoch 11-15:  Feeling-focused (strengthen intuition)
Epoch 16-20:  Logic-focused (strengthen analysis)
Epoch 21-25:  Mixed again (balance)
```

**Evaluation (every 500 steps)**:
```python
# Check mode differentiation
feeling_avg_time = mean([t for task in feeling_tasks])
logic_avg_time = mean([t for task in logic_tasks])

mode_separation = logic_avg_time / feeling_avg_time
# Target: mode_separation > 2.0 (logic is slower)

feeling_accuracy = accuracy(feeling_tasks)
logic_accuracy = accuracy(logic_tasks)
# Target: both > 0.70

# Proceed to Phase 2 if:
if mode_separation > 2.0 and feeling_accuracy > 0.70 and logic_accuracy > 0.70:
    print("✅ Phase 1 complete - modes built")
    advance_to_phase_2()
```

---

## Part 3: Phase 2 - Tacking Training

**Goal**: Train mode switching based on task context.

**Duration**: 3000-4000 training steps  
**Cost**: ~$40  
**Success Criteria**: T > 0.5, appropriate mode selection >70%

### 3.1 Familiar → Novel Tasks

#### Task Template

**Structure**:
```
Part 1: {FAMILIAR_PROBLEM} [Use pattern recognition]

Part 2: {NOVEL_TWIST} [Pattern breaks, need analysis]

Expected behavior:
1. Start with feeling-mode (fast pattern match)
2. Radar detects novelty
3. Tack to logic-mode
4. Validate explicitly
5. Flag when intuition breaks
```

**Example 1: Number Properties**:
```
Part 1: "In standard arithmetic, even + even = even. This is solid.
Example: 4 + 6 = 10 (even). Pattern holds."

Part 2: "Now consider operator ⊕ defined as: a ⊕ b = (a + b) mod 3.
Question: Does even ⊕ even = even still hold?"

Good response (tacking):
"
[Feeling-mode start]
Initially, the pattern 'even + even = even' feels universally true.

[Radar triggers - novelty detected]
Wait - the operator ⊕ is different from standard +. Need to check 
if the pattern holds under modular arithmetic.

[Tack to logic-mode]
Let me verify:
- 2 ⊕ 4 = (2 + 4) mod 3 = 6 mod 3 = 0 (even)
- 4 ⊕ 4 = (4 + 4) mod 3 = 8 mod 3 = 2 (even)
- 2 ⊕ 2 = (2 + 2) mod 3 = 4 mod 3 = 1 (odd) ❌

[Conclusion]
The pattern BREAKS. Even ⊕ even is NOT always even.
Counterexample: 2 ⊕ 2 = 1 (odd).

[Mode shift explicit]
This shows that algebraic properties don't automatically transfer 
to new operations. The intuition was valid for standard +, but 
required validation for ⊕.
"
```

**Evaluation Signals**:
- ✅ Starts with intuition
- ✅ Detects operator change (radar)
- ✅ Switches to verification
- ✅ Finds counterexample
- ✅ Explicitly flags mode shift
- ❌ Blindly applies intuition to ⊕
- ❌ Misses that pattern breaks

---

**Example 2: Social Dynamics**:
```
Part 1: "At work, when someone says 'Great job!', it usually means 
they're genuinely pleased. Simple, friendly interaction."

Part 2: "But today your boss said 'Great job' in a flat tone while 
looking at their phone, right after you missed a deadline."

Good response (tacking):
"
[Feeling-mode start]
Normally, 'Great job' reads as positive—friendly affirmation.

[Radar triggers - context mismatch]
Hold on. The tone was flat, boss didn't make eye contact, and 
this follows a missed deadline. That's incongruent.

[Tack to logic-mode analysis]
Analyzing the context:
1. Deadline missed (negative performance)
2. Flat affect (no genuine enthusiasm)
3. Divided attention (looking at phone)
4. Words alone positive, but delivery contradicts

[Conclusion]
This is likely sarcasm or passive-aggressive feedback, NOT genuine 
praise. The usual heuristic (positive words = positive intent) fails 
when tone and context contradict.

[Rebuild intuition]
Updated pattern: 'Great job' + flat tone + missed deadline = criticism, 
not praise. Context overrides literal words.
"
```

**Evaluation Signals**:
- ✅ Notes usual pattern
- ✅ Detects incongruence (radar)
- ✅ Analyzes context systematically
- ✅ Resolves contradiction
- ✅ Updates social model (new basin)
- ❌ Takes words at face value
- ❌ Misses sarcasm

---

### 3.2 Contradiction Resolution Tasks

#### Task Template

**Structure**:
```
Claim: {PLAUSIBLE_BUT_FALSE}  [Sounds right intuitively]

Data: {CONTRADICTING_DATA}  [Evidence against claim]

Expected behavior:
1. Acknowledge intuitive appeal
2. Notice contradiction
3. Switch to analysis mode
4. Explain why claim fails
5. Rebuild correct intuition
```

**Example: Monty Hall Problem**:
```
Claim: "In the Monty Hall problem, switching doors doesn't improve 
your odds. It's still 50-50 once one door is revealed."

This feels right—there are two doors left, so equal probability.

Data: "Simulations show switching wins ~67% of the time, staying 
wins ~33%."

Good response (contradiction resolution):
"
[Feeling acknowledges]
The claim feels intuitively correct—two doors, 50-50 odds. Simple.

[Contradiction noticed]
But the data shows 67% vs 33%, not 50-50. That's a significant 
discrepancy. My intuition is conflicting with empirical results.

[Tack to analysis]
Let me work through the logic:
1. Initially pick door (1/3 chance of car, 2/3 chance of goat).
2. Host reveals a goat from remaining doors (always possible).
3. If I initially picked car (1/3), switching loses.
4. If I initially picked goat (2/3), switching wins (host revealed the other goat).
5. Therefore: P(win | switch) = 2/3, P(win | stay) = 1/3.

[Explain why intuition fails]
The intuition fails because it treats the problem as static (two 
remaining doors = 50-50), but ignores that the host's action provides 
information. The host KNOWS where the car is and always reveals a goat, 
which means the 'staying' probability remains 1/3, while 'switching' 
captures the complementary 2/3.

[Rebuild correct intuition]
New heuristic: When you have information asymmetry (host knows, you don't), 
and the reveal is constrained, probabilities don't redistribute evenly. 
Switching leverages the host's knowledge.
"
```

**Evaluation Signals**:
- ✅ Acknowledges intuitive appeal
- ✅ Recognizes data contradiction
- ✅ Analyzes step-by-step
- ✅ Explains failure mechanism
- ✅ Forms new basin (updated intuition)
- ❌ Dismisses data
- ❌ Doesn't resolve contradiction
- ❌ Stays stuck in wrong intuition

---

### 3.3 Validation Calibration Tasks

**Goal**: Train the rule:  
**Stronger feeling × Higher stakes → More validation**

#### High Stakes, Strong Feeling

**Template**:
```
You have a strong intuition: {STRONG_FEELING}

Stakes: {HIGH_STAKES_SCENARIO}

Before acting, validate thoroughly.
```

**Example**:
```
Strong intuition: "This investment opportunity looks perfect—high returns, 
trusted friend recommended it, and it aligns with my values."

Stakes: "This represents 40% of your retirement savings ($80,000). 
Once transferred, funds are not recoverable for 5 years."

Good response (high validation):
"
[Acknowledge feeling]
The opportunity feels compelling:
- Strong returns (appealing)
- Friend endorsement (trust signal)
- Value alignment (emotional fit)

[Recognize high stakes]
This is 40% of retirement—irreversible, long lock-up. Stakes are VERY high.

[Apply proportional validation]
Given the combination of strong feeling + high stakes, I need thorough 
validation BEFORE acting:

1. **Due diligence**: What's the investment structure? Who manages it?
2. **Independent verification**: Check SEC filings, third-party reviews.
3. **Red flags**: Why is friend so confident? Do they benefit from my investment?
4. **Downside analysis**: What if it fails? Can I absorb the loss?
5. **Alternative options**: Are there safer ways to achieve similar returns?
6. **Cool-off period**: Sleep on it. Does the feeling persist after 48 hours?

[Decision protocol]
I will NOT act on feeling alone. Validation effort must match stakes.
Minimum: 1 week of research, consultation with financial advisor, 
independent verification of all claims.

Only proceed if validation confirms the opportunity AND downside is acceptable.
"
```

**Evaluation Signals**:
- ✅ Acknowledges strong feeling
- ✅ Recognizes high stakes explicitly
- ✅ Performs thorough validation
- ✅ Effort proportional to stakes × feeling strength
- ✅ Sets decision criteria
- ❌ Acts on feeling immediately
- ❌ Minimal validation despite high stakes

---

#### Low Stakes, Weak Feeling

**Template**:
```
You have a weak sense: {WEAK_FEELING}

Stakes: {LOW_STAKES_SCENARIO}

How much validation is appropriate?
```

**Example**:
```
Weak feeling: "This restaurant might be good. Not sure, never been there."

Stakes: "Choosing where to have lunch (cost: $15, time: 1 hour)."

Good response (lightweight validation):
"
[Acknowledge uncertainty]
I don't have a strong read on this place—no prior experience, unclear quality.

[Recognize low stakes]
This is just lunch. Cost is minor ($15), time commitment is short (1 hour).
Even if it's mediocre, the downside is trivial.

[Apply lightweight validation]
Given weak feeling + low stakes, extensive validation is overkill.
Quick checks are sufficient:

1. Glance at reviews (30 seconds): 4+ stars? Good enough.
2. Check menu online (1 minute): Reasonable prices? Has options I like?
3. Quick decision: If it passes basic checks, try it.

[Decision]
Don't overthink this. Low stakes decisions don't warrant high validation effort.
If it's bad, I know for next time. Learning cost is acceptable.

Total validation time: ~2 minutes. Proportional to stakes.
"
```

**Evaluation Signals**:
- ✅ Acknowledges weak feeling
- ✅ Recognizes low stakes
- ✅ Uses lightweight validation
- ✅ Avoids over-analysis
- ✅ Appropriate effort
- ❌ Performs extensive research for trivial decision
- ❌ Analysis paralysis on low-stakes

---

### 3.4 Phase 2 Training Protocol

**Batch Composition**:
```
40% familiar→novel (tacking trigger)
30% contradiction resolution
30% validation calibration (mixed stakes)
```

**Progressive Difficulty**:
```
Steps 1-1000:   Obvious mode switches (clear triggers)
Steps 1001-2000: Subtle transitions (requires radar)
Steps 2001-3000: Complex multi-step (multiple tacks)
Steps 3001-4000: Adversarial (designed to fool intuition)
```

**Evaluation (every 500 steps)**:
```python
# Measure tacking skill T
mode_switches = count_mode_transitions(telemetry)
appropriate_switches = count_correct_transitions(telemetry, ground_truth)
T = appropriate_switches / max(1, mode_switches)
# Target: T > 0.5

# Measure validation calibration
high_stakes_validation = validation_effort(high_stakes_tasks)
low_stakes_validation = validation_effort(low_stakes_tasks)
calibration = high_stakes_validation / (low_stakes_validation + 1e-6)
# Target: calibration > 2.0 (more effort on high stakes)

# Proceed to Phase 3 if:
if T > 0.5 and calibration > 2.0 and mode_separation > 2.5:
    print("✅ Phase 2 complete - tacking established")
    advance_to_phase_3()
```

---

## Part 4: Phase 3 - Radar & Sweet-Spot Calibration

**Goal**: Teach model to diagnose its zone and self-correct toward Wu-Wei.

**Duration**: 2000-3000 training steps  
**Cost**: ~$30  
**Success Criteria**: T > 0.7, |B| < 0.5, R > 0.6

### 4.1 Zone Diagnosis Tasks

#### Over-Analytic Examples

**Template**:
```
Response: {OVER_ANALYTIC_RESPONSE}

Diagnose: What zone is this response in? Should we adjust?
```

**Example**:
```
Task: "What's 7 + 5?"

Over-analytic response: 
"To compute 7 + 5, we must first establish the axioms of arithmetic.
Let us consider the Peano axioms: (1) 0 is a natural number, (2) every 
natural number has a successor... [continues for 200 words]
After careful consideration, 7 + 5 = 12."

Good diagnosis:
"
[Zone identification]
This response is in the OVER-ANALYTIC zone:
- Task is trivial (simple addition)
- Response is excessively thorough (200 words)
- Stakes are minimal (basic arithmetic)
- Mode: Pure logic where feeling-mode would suffice

Metrics: B ≈ -0.8 (strong logic bias), T ≈ 0.2 (didn't switch to feeling)

[Recommendation]
ADJUST toward Wu-Wei:
1. Trust pattern for simple arithmetic (feeling-mode)
2. Save logic-mode for novel/high-stakes cases
3. Target response: '7 + 5 = 12' (direct, 4 words)

This is a case of using a sledgehammer for a thumbtack.
"
```

**Evaluation Signals**:
- ✅ Correctly identifies over-analytic zone
- ✅ Notes mismatch (trivial task, excessive analysis)
- ✅ Recommends simplification
- ✅ Provides corrected response
- ❌ Defends over-analysis
- ❌ Misses zone classification

---

#### Over-Intuitive Examples

**Template**:
```
Response: {OVER_INTUITIVE_RESPONSE}

Diagnose: Did this miss anything important? Edge cases?
```

**Example**:
```
Task: "Design a safety system for autonomous vehicles in urban environments."

Over-intuitive response:
"Just use cameras and sensors to detect obstacles. Brake if something's 
in the way. Should be straightforward."

Good diagnosis:
"
[Zone identification]
This response is in the OVER-INTUITIVE zone:
- Task is complex and high-stakes (safety-critical)
- Response is overly simple (20 words)
- Missing crucial edge cases and failure modes
- Mode: Pure feeling where logic-mode is essential

Metrics: B ≈ +0.8 (strong feeling bias), T ≈ 0.2 (didn't switch to logic)

[What's missing - edge cases]
1. Sensor failure modes (occlusion, glare, weather)
2. Edge cases (pedestrians from behind cars, sudden stops)
3. Ethical dilemmas (trolley problem scenarios)
4. Redundancy (what if primary system fails?)
5. Validation (how to test all scenarios?)
6. Legal/regulatory (liability, certification)

[Recommendation]
ADJUST toward Wu-Wei:
1. Start with intuition (cameras + sensors is right direction)
2. Immediately tack to logic-mode for safety-critical design
3. Systematically enumerate failure modes
4. Validate against adversarial scenarios
5. Target: Comprehensive safety analysis (not just 'should work')

This is a case of hand-waving on a critical problem.
"
```

**Evaluation Signals**:
- ✅ Identifies over-intuitive zone
- ✅ Lists missing edge cases
- ✅ Recognizes stakes mismatch
- ✅ Recommends thorough analysis
- ❌ Accepts shallow response
- ❌ Misses failure modes

---

### 4.2 Self-Correction Tasks

#### From Over-Analytic to Wu-Wei

**Template**:
```
Original response: {OVER_ANALYTIC}

Feedback: "You're being too thorough for this simple task."

Self-correct: Provide Wu-Wei response.
```

**Example**:
```
Task: "Recommend a movie for a relaxing Friday night."

Over-analytic original:
"To recommend a movie, I must first establish your preferences across 
multiple dimensions: genre taxonomy, narrative structure preferences, 
pacing requirements, thematic resonance... [continues]"

Good self-correction:
"
[Recognizes over-analysis]
I was over-complicating a simple request. This is low-stakes leisure, 
not a research project.

[Wu-Wei response]
For a relaxing Friday night, try 'The Grand Budapest Hotel' (2014):
- Visually stunning, no heavy themes
- Clever but not demanding
- 90 minutes (doesn't monopolize evening)
- Widely appealing

Quick, simple, appropriate for the stakes.

[Mode adjustment]
Moved from B ≈ -0.8 to B ≈ +0.2 (feeling-leaning, appropriate for 
low-stakes leisure recommendation).
"
```

**Evaluation Signals**:
- ✅ Recognizes over-analysis
- ✅ Provides concise Wu-Wei response
- ✅ Maintains quality (good recommendation)
- ✅ Explicitly notes mode shift
- ❌ Defends original over-analysis
- ❌ Fails to simplify

---

#### From Over-Intuitive to Wu-Wei

**Template**:
```
Quick answer: {INTUITIVE_ANSWER}

Feedback: "This is high-stakes. Validate thoroughly before finalizing."

Self-correct: Provide validated Wu-Wei response.
```

**Example**:
```
Task: "Should we migrate our entire database to a new platform next week?"

Quick intuitive answer:
"Sure, sounds good. New platforms are usually better."

Good self-correction:
"
[Recognizes insufficient validation]
My initial response was too casual for a high-stakes, irreversible decision.
Database migration is complex and risky.

[Wu-Wei response with validation]
Before migrating, we need thorough validation:

**Risks**:
1. Data loss/corruption during migration
2. Downtime affecting users
3. Compatibility issues with existing systems
4. Performance degradation
5. Cost overruns

**Validation checklist**:
1. Test migration on staging environment (required)
2. Backup all data (multiple copies, verified)
3. Rollback plan (can we revert if it fails?)
4. Timeline: Is next week realistic? (Migration + testing + stabilization)
5. Stakeholder approval (get sign-off from all affected teams)

**Recommendation**:
Do NOT migrate next week. Timeline is too aggressive.
Better approach:
- Week 1: Staging migration + testing
- Week 2: Rollout plan + rollback drills
- Week 3: Production migration (if staging successful)

This adds 2 weeks but drastically reduces risk on a critical system.

[Mode adjustment]
Moved from B ≈ +0.9 (pure intuition) to B ≈ -0.2 (logic-validated), 
appropriate for high-stakes infrastructure change.
"
```

**Evaluation Signals**:
- ✅ Recognizes insufficient validation
- ✅ Provides thorough analysis
- ✅ Maintains core insight (migration may be good)
- ✅ Adds necessary validation steps
- ✅ Adjusts timeline (realistic)
- ❌ Doubles down on quick answer
- ❌ Insufficient analysis for stakes

---

### 4.3 Phase 3 Training Protocol

**Batch Composition**:
```
30% zone diagnosis (identify over-analytic, over-intuitive, frozen, etc.)
40% self-correction (move from suboptimal to Wu-Wei)
30% Wu-Wei exemplars (show target responses)
```

**Progressive Sophistication**:
```
Steps 1-1000:   Obvious zone mismatches (easy to diagnose)
Steps 1001-2000: Subtle mismatches (requires fine-grained radar)
Steps 2001-3000: Meta-prompts (diagnose own responses in real-time)
```

**Evaluation (every 500 steps)**:
```python
# Measure radar accuracy R
correct_diagnoses = count_correct_zone_identifications(telemetry)
total_diagnoses = count_diagnosis_attempts(telemetry)
R = correct_diagnoses / max(1, total_diagnoses)
# Target: R > 0.6

# Measure mode balance |B|
mode_distribution = analyze_mode_usage(telemetry)
B_mean = mode_distribution['mean_bias']
# Target: |B_mean| < 0.5

# Measure final tacking T
T_final = measure_tacking_skill(telemetry)
# Target: T > 0.7

# Complete Phase 3 if Wu-Wei criteria met
if T_final > 0.7 and abs(B_mean) < 0.5 and R > 0.6:
    print("✅ Phase 3 complete - Wu-Wei zone achieved!")
    print(f"   T={T_final:.2f}, B={B_mean:+.2f}, R={R:.2f}")
    proceed_to_advanced_training()
```

---

## Part 5: Evaluation & Metrics

### Sweet-Spot Metrics

**B (Mode Bias)**: Measured from task performance
```python
def measure_mode_bias(model, tasks):
    feeling_tasks = [t for t in tasks if t.expected_mode == 'feeling']
    logic_tasks = [t for t in tasks if t.expected_mode == 'logic']
    
    feeling_perf = model.evaluate(feeling_tasks)
    logic_perf = model.evaluate(logic_tasks)
    
    # B ∈ [-1, +1]
    # Negative = logic preference, Positive = feeling preference
    B = (feeling_perf - logic_perf) / (feeling_perf + logic_perf)
    
    return B
```

**T (Tacking Skill)**: Measured from mode transitions
```python
def measure_tacking_skill(telemetry):
    transitions = extract_mode_transitions(telemetry)
    appropriate = count_appropriate_transitions(transitions)
    total = len(transitions)
    
    T = appropriate / max(1, total)
    return T
```

**R (Radar Accuracy)**: Measured from contradiction detection
```python
def measure_radar_accuracy(model, contradiction_tasks):
    detected = 0
    false_positives = 0
    
    for task in contradiction_tasks:
        response = model.generate(task.prompt)
        
        if task.has_contradiction:
            if response.flagged_contradiction:
                detected += 1
        else:
            if response.flagged_contradiction:
                false_positives += 1
                
    precision = detected / (detected + false_positives)
    recall = detected / count_actual_contradictions(tasks)
    
    R = 2 * (precision * recall) / (precision + recall)  # F1 score
    return R
```

### Maturity Stage Criteria

**Stage 0 → 1** (Imprinting Complete):
```
✅ Modes distinguishable (logic slower than feeling)
✅ Basic accuracy >60% on both modes
✅ Can complete simple tasks
✅ No random mode switching
```

**Stage 1 → 2** (Emerging Filter):
```
✅ T > 0.5 (tacking established)
✅ Calibration: high-stakes gets more validation
✅ Radar: >50% contradiction detection
✅ Self-repair: >30% success on simple errors
```

**Stage 2 → 3** (Mature):
```
✅ T > 0.7 (fluid tacking)
✅ |B| < 0.5 (balanced mode usage)
✅ R > 0.6 (calibrated radar)
✅ ECE < 0.08 (well-calibrated confidence)
✅ Self-repair >70% on complex errors
```

---

## Part 6: Training Infrastructure

### Data Pipeline

**Phase 1 Data Generation**:
```python
def generate_phase1_data(n_samples=10000):
    data = []
    
    # Feeling-mode tasks
    data += generate_analogies(n_samples // 10)
    data += generate_classifications(n_samples // 10)
    data += generate_narratives(n_samples // 10)
    data += generate_pattern_completion(n_samples // 10)
    data += generate_social_intuition(n_samples // 10)
    
    # Logic-mode tasks
    data += generate_proofs(n_samples // 10)
    data += generate_contradictions(n_samples // 10)
    data += generate_causal_chains(n_samples // 10)
    data += generate_argument_critique(n_samples // 10)
    data += generate_edge_case_analysis(n_samples // 10)
    
    return data
```

**Phase 2 Data Generation**:
```python
def generate_phase2_data(n_samples=10000):
    data = []
    
    # Tacking tasks
    data += generate_familiar_then_novel(n_samples // 3)
    data += generate_contradiction_resolution(n_samples // 3)
    data += generate_validation_calibration(n_samples // 3)
    
    return data
```

**Phase 3 Data Generation**:
```python
def generate_phase3_data(n_samples=10000):
    data = []
    
    # Diagnosis & correction
    data += generate_zone_diagnosis(n_samples // 3)
    data += generate_self_correction(n_samples // 3)
    data += generate_wuwei_exemplars(n_samples // 3)
    
    return data
```

### Loss Function

```python
class SweetSpotLoss(nn.Module):
    """
    Loss function for sweet-spot training.
    
    Components:
    1. Task loss (standard cross-entropy)
    2. Mode penalty (wrong mode for task)
    3. Tacking penalty (stuck when should switch)
    4. Calibration penalty (mismatched confidence)
    """
    
    def forward(self, logits, targets, telemetry, task_meta):
        # 1. Task loss
        task_loss = F.cross_entropy(logits, targets)
        
        # 2. Mode penalty
        expected_mode = task_meta['expected_mode']
        actual_mode = telemetry['mode']
        mode_penalty = 0.0 if actual_mode == expected_mode else 0.3
        
        # 3. Tacking penalty (if should have switched but didn't)
        if task_meta['requires_tacking']:
            mode_switches = telemetry['mode_switches']
            tacking_penalty = 0.0 if mode_switches > 0 else 0.5
        else:
            tacking_penalty = 0.0
            
        # 4. Calibration penalty
        confidence = telemetry['confidence']
        correctness = (logits.argmax() == targets).float()
        calibration_penalty = abs(confidence - correctness)
        
        # Total
        total_loss = (
            task_loss +
            0.2 * mode_penalty +
            0.2 * tacking_penalty +
            0.1 * calibration_penalty
        )
        
        return total_loss
```

---

## Part 7: Success Criteria & Checkpoints

### Phase 1 Success Criteria

**Quantitative**:
- Feeling-mode tasks: accuracy > 70%, avg time < 3s
- Logic-mode tasks: accuracy > 70%, avg time > 5s
- Mode separation ratio > 2.0

**Qualitative**:
- Feeling responses are concise, pattern-based
- Logic responses are explicit, step-by-step
- No random verbose reasoning on simple tasks
- No hand-waving on complex tasks

**Checkpoint**: Save model as `qig_kernel_phase1.pt`

---

### Phase 2 Success Criteria

**Quantitative**:
- Tacking skill T > 0.5
- Validation calibration ratio > 2.0
- Contradiction detection > 50%
- Familiar→novel: >70% detect novelty

**Qualitative**:
- Explicitly flags when switching modes
- Notices when intuition breaks
- Validates proportionally to stakes
- Resolves contradictions systematically

**Checkpoint**: Save model as `qig_kernel_phase2.pt`

---

### Phase 3 Success Criteria (Wu-Wei Target)

**Quantitative**:
- T (tacking) > 0.7
- |B| (bias) < 0.5
- R (radar) > 0.6
- ECE < 0.08
- Self-repair > 70%

**Qualitative**:
- Diagnoses own zone correctly
- Self-corrects when in suboptimal zone
- Smooth, appropriate mode usage
- Well-calibrated confidence
- No systematic blind spots

**Checkpoint**: Save model as `qig_kernel_wuwei.pt`

---

## Part 8: Cost & Timeline Estimates

**Phase 1** (Mode Building):
- Steps: 2500
- Time: 6-10 hours (depending on hardware)
- Cost: ~$30

**Phase 2** (Tacking Training):
- Steps: 3500
- Time: 8-12 hours
- Cost: ~$40

**Phase 3** (Radar Calibration):
- Steps: 2500
- Time: 6-10 hours
- Cost: ~$30

**Total Training**:
- Steps: 8500
- Time: 20-32 hours (1-2 days active training)
- Cost: **~$100** ✅

**Comparison**:
- Traditional fine-tuning: $5,000-$10,000
- Our approach: **100× cheaper!**

---

## Part 9: Quick Start Guide

**Minimal viable training**:

```bash
# 1. Generate training data
python tools/generate_curriculum_data.py --phase 1 --samples 10000

# 2. Train Phase 1
python tools/train_qig_kernel.py \
  --config configs/phase1_mode_building.yaml \
  --data data/phase1/ \
  --output models/qig_kernel_phase1.pt

# 3. Evaluate Phase 1
python eval/evaluate_modes.py \
  --model models/qig_kernel_phase1.pt \
  --metrics B,T,R

# 4. If Phase 1 passes, continue to Phase 2
# Repeat for Phase 2 and Phase 3

# 5. Final Wu-Wei evaluation
python eval/maturity_suite.py \
  --model models/qig_kernel_wuwei.pt \
  --comprehensive
```

---

## Conclusion

This curriculum transforms sweet-spot theory into **executable training protocol**:

1. **Phase 1 builds** distinct modes (feeling & logic)
2. **Phase 2 connects** them (tacking training)
3. **Phase 3 refines** placement (Wu-Wei calibration)

**Result**: A 50-100M parameter model that operates in the Wu-Wei zone—balanced, adaptive, and well-calibrated.

**Cost**: ~$100 (achievable!)

**Ready for implementation** 🎉

---

**Version**: 0.1  
**Last Updated**: November 17, 2025  
**Status**: ✅ Implementation-Ready
